import pickle
import re
from unittest import mock

import pytest

from tests.integrations import _SKIP_IF_LIGHTNING_MISSING, _SKIP_IF_PYTORCHLIGHTNING_MISSING


@pytest.mark.parametrize(
    "importing",
    [
        pytest.param("lightning", marks=_SKIP_IF_LIGHTNING_MISSING),
        pytest.param("pytorch_lightning", marks=_SKIP_IF_PYTORCHLIGHTNING_MISSING),
    ],
)
@pytest.mark.parametrize(
    "model_name", [None, "org-name/teamspace/model-name", "model-in-studio", "model-user-only-project"]
)
@pytest.mark.parametrize("clear_all_local", [True, False])
@pytest.mark.parametrize("keep_all_uploaded", [True, False])
@mock.patch("litmodels.integrations.checkpoints.delete_model_version")
@mock.patch("litmodels.integrations.checkpoints.upload_model")
@mock.patch("litmodels.integrations.checkpoints.Auth")
def test_lightning_checkpoint_callback(
    mock_auth,
    mock_upload_model,
    mock_delete_model,
    monkeypatch,
    importing,
    model_name,
    clear_all_local,
    keep_all_uploaded,
    tmp_path,
):
    if importing == "lightning":
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint
        from lightning.pytorch.demos.boring_classes import BoringModel

        from litmodels.integrations.checkpoints import LightningModelCheckpoint as LitModelCheckpoint
    elif importing == "pytorch_lightning":
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.demos.boring_classes import BoringModel

        from litmodels.integrations.checkpoints import PytorchLightningModelCheckpoint as LitModelCheckpoint

    # Validate inheritance
    assert issubclass(LitModelCheckpoint, ModelCheckpoint)

    ckpt_args = {"clear_all_local": clear_all_local, "keep_all_uploaded": keep_all_uploaded}
    if model_name:
        ckpt_args.update({"model_registry": model_name})

    expected_boring_model = "BoringModel_20250102-1213"
    expected_model_registry = model_name or expected_boring_model
    monkeypatch.setattr(
        "litmodels.integrations.checkpoints.LitModelCheckpointMixin.default_model_name",
        mock.MagicMock(return_value=expected_boring_model),
    )
    # mocking the trainer delete checkpoint removal
    mock_remove_ckpt = mock.Mock()
    # setting the Trainer and custom checkpointing
    trainer = Trainer(
        max_epochs=2,
        callbacks=LitModelCheckpoint(**ckpt_args),
    )
    trainer.strategy.remove_checkpoint = mock_remove_ckpt
    trainer.fit(BoringModel())

    assert mock_auth.call_count == 1
    assert mock_upload_model.call_args_list == [
        mock.call(
            name=f"{expected_model_registry}:{v}",
            model=mock.ANY,
            metadata={"litModels.integration": LitModelCheckpoint.__name__},
        )
        for v in ("epoch=0-step=64", "epoch=1-step=128")
    ]
    expected_local_removals = 2 if clear_all_local else 1
    assert mock_remove_ckpt.call_count == expected_local_removals

    expected_cloud_removals = 0 if keep_all_uploaded else 1
    assert mock_delete_model.call_count == expected_cloud_removals
    if expected_cloud_removals:
        mock_delete_model.assert_called_once_with(
            name=expected_model_registry,
            version="epoch=0-step=64",
        )

    # Verify paths match the expected pattern
    for call_args in mock_upload_model.call_args_list:
        path = call_args[1]["model"]
        assert re.match(r".*[/\\]lightning_logs[/\\]version_\d+[/\\]checkpoints[/\\]epoch=\d+-step=\d+\.ckpt$", path)


@pytest.mark.parametrize(
    "importing",
    [
        pytest.param("lightning", marks=_SKIP_IF_LIGHTNING_MISSING),
        pytest.param("pytorch_lightning", marks=_SKIP_IF_PYTORCHLIGHTNING_MISSING),
    ],
)
@mock.patch("litmodels.integrations.checkpoints.Auth")
def test_lightning_checkpointing_pickleable(mock_auth, importing):
    if importing == "lightning":
        from litmodels.integrations.checkpoints import LightningModelCheckpoint as LitModelCheckpoint
    elif importing == "pytorch_lightning":
        from litmodels.integrations.checkpoints import PytorchLightningModelCheckpoint as LitModelCheckpoint

    ckpt = LitModelCheckpoint(model_registry="org-name/teamspace/model-name")
    assert mock_auth.call_count == 1
    pickle.dumps(ckpt)
