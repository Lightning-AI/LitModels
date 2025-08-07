import os
import platform
from contextlib import redirect_stdout
from io import StringIO
from typing import Optional

import pytest
import torch
from lightning_sdk import Teamspace
from lightning_sdk.lightning_cloud.rest_client import GridRestClient
from lightning_sdk.utils.resolve import _resolve_teamspace

from litmodels import download_model, load_model, save_model, upload_model
from litmodels.integrations.duplicate import duplicate_hf_model
from litmodels.integrations.mixins import PickleRegistryMixin, PyTorchRegistryMixin
from litmodels.io.cloud import _list_available_teamspaces
from litmodels.io.utils import _KERAS_AVAILABLE
from tests.integrations import (
    _SKIP_IF_LIGHTNING_BELLOW_2_5_1,
    _SKIP_IF_PYTORCHLIGHTNING_BELLOW_2_5_1,
    LIT_ORG,
    LIT_TEAMSPACE,
)


def _prepare_variables(test_name: str) -> tuple[Teamspace, str, str]:
    model_name = f"ci-test_integrations_{test_name}+{os.urandom(8).hex()}"
    teamspace = _resolve_teamspace(org=LIT_ORG, teamspace=LIT_TEAMSPACE, user=None)
    org_team = f"{teamspace.owner.name}/{teamspace.name}"
    return teamspace, org_team, model_name


def _cleanup_model(teamspace: Teamspace, model_name: str, expected_num_versions: Optional[int] = None) -> None:
    """Cleanup model from the teamspace."""
    client = GridRestClient()
    # cleaning created models as each test run shall have unique model name
    model = client.models_store_get_model_by_name(
        project_owner_name=teamspace.owner.name,
        project_name=teamspace.name,
        model_name=model_name,
    )
    if expected_num_versions is not None:
        versions = client.models_store_list_model_versions(project_id=model.project_id, model_id=model.id)
        assert expected_num_versions == len(versions.versions)
    client.models_store_delete_model(project_id=model.project_id, model_id=model.id)


@pytest.mark.cloud
@pytest.mark.parametrize(
    "in_studio",
    [False, pytest.param(True, marks=pytest.mark.skipif(platform.system() != "Linux", reason="Studio is just Linux"))],
)
def test_upload_download_model(in_studio, monkeypatch, tmp_path):
    """Verify that the model is uploaded to the teamspace"""
    if in_studio:
        # mock env variables as it would run in studio
        monkeypatch.setenv("LIGHTNING_ORG", LIT_ORG)
        monkeypatch.setenv("LIGHTNING_TEAMSPACE", LIT_TEAMSPACE)

    # create a dummy file
    file_path = tmp_path / "dummy.txt"
    with open(file_path, "w") as f:
        f.write("dummy")

    # model name with random hash
    teamspace, org_team, model_name = _prepare_variables("upload_download")
    model_registry = f"{org_team}/{model_name}" if not in_studio else model_name

    out = StringIO()
    with redirect_stdout(out):
        upload_model(name=model_registry, model=file_path)

    # validate the output
    assert (
        f"Model uploaded successfully. Link to the model: 'https://lightning.ai/{org_team}/models/{model_name}'"
    ) in out.getvalue()

    os.remove(file_path)
    assert not os.path.isfile(file_path)

    model_files = download_model(name=model_registry, download_dir=tmp_path)
    assert model_files == ["dummy.txt"]
    for file in model_files:
        assert os.path.isfile(os.path.join(tmp_path, file))

    # CLEANING
    _cleanup_model(teamspace, model_name, expected_num_versions=1)


@pytest.mark.parametrize(
    "importing",
    [
        pytest.param("lightning", marks=_SKIP_IF_LIGHTNING_BELLOW_2_5_1),
        pytest.param("pytorch_lightning", marks=_SKIP_IF_PYTORCHLIGHTNING_BELLOW_2_5_1),
    ],
)
@pytest.mark.parametrize(
    "in_studio",
    [
        False,
        pytest.param(True, marks=pytest.mark.skipif(platform.system() == "Windows", reason="studio is not Windows")),
    ],
)
@pytest.mark.cloud
def test_lightning_default_checkpointing(importing, in_studio, monkeypatch, tmp_path):
    if in_studio:
        # mock env variables as it would run in studio
        monkeypatch.setenv("LIGHTNING_ORG", LIT_ORG)
        monkeypatch.setenv("LIGHTNING_TEAMSPACE", LIT_TEAMSPACE)

    if importing == "lightning":
        from lightning import Trainer
        from lightning.pytorch.demos.boring_classes import BoringModel
    elif importing == "pytorch_lightning":
        from pytorch_lightning import Trainer
        from pytorch_lightning.demos.boring_classes import BoringModel

    # model name with random hash
    teamspace, org_team, model_name = _prepare_variables("default_checkpoint")
    model_registry = f"{org_team}/{model_name}" if not in_studio else model_name

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmp_path,
        model_registry=model_registry,
    )
    trainer.fit(BoringModel())

    # CLEANING
    _cleanup_model(teamspace, model_name, expected_num_versions=2)


@pytest.mark.parametrize("trainer_method", ["fit", "validate", "test", "predict"])
@pytest.mark.parametrize(
    "registry", ["registry", "registry:version:v1", "registry:<model>", "registry:<model>:version:v1"]
)
@pytest.mark.parametrize(
    "importing",
    [
        pytest.param("lightning", marks=_SKIP_IF_LIGHTNING_BELLOW_2_5_1),
        pytest.param("pytorch_lightning", marks=_SKIP_IF_PYTORCHLIGHTNING_BELLOW_2_5_1),
    ],
)
@pytest.mark.parametrize(
    "in_studio",
    [
        False,
        pytest.param(True, marks=pytest.mark.skipif(platform.system() == "Windows", reason="studio is not Windows")),
    ],
)
@pytest.mark.cloud
def test_lightning_plain_resume(trainer_method, registry, importing, in_studio, tmp_path, monkeypatch):
    if importing == "lightning":
        from lightning import Trainer
        from lightning.pytorch.demos.boring_classes import BoringModel
    elif importing == "pytorch_lightning":
        from pytorch_lightning import Trainer
        from pytorch_lightning.demos.boring_classes import BoringModel

    if in_studio:
        # mock env variables as it would run in studio
        monkeypatch.setenv("LIGHTNING_ORG", LIT_ORG)
        monkeypatch.setenv("LIGHTNING_TEAMSPACE", LIT_TEAMSPACE)

    trainer = Trainer(max_epochs=1, limit_train_batches=50, limit_val_batches=20, default_root_dir=tmp_path)
    trainer.fit(BoringModel())
    checkpoint_path = getattr(trainer.checkpoint_callback, "best_model_path")

    # model name with random hash
    teamspace, org_team, model_name = _prepare_variables(f"resume_{trainer_method}")
    model_registry = f"{org_team}/{model_name}" if not in_studio else model_name
    upload_model(model=checkpoint_path, name=model_registry)
    expected_num_versions = 1

    trainer_kwargs = {"model_registry": model_registry} if "<model>" not in registry else {}
    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmp_path,
        limit_train_batches=10,
        limit_val_batches=10,
        limit_test_batches=10,
        limit_predict_batches=10,
        **trainer_kwargs,
    )
    registry = registry.replace("<model>", model_registry)
    if trainer_method == "fit":
        trainer.fit(BoringModel(), ckpt_path=registry)
        if trainer_kwargs:
            expected_num_versions += 1
    elif trainer_method == "validate":
        trainer.validate(BoringModel(), ckpt_path=registry)
    elif trainer_method == "test":
        trainer.test(BoringModel(), ckpt_path=registry)
    elif trainer_method == "predict":
        trainer.predict(BoringModel(), ckpt_path=registry)
    else:
        raise ValueError(f"Unknown trainer method: {trainer_method}")

    # CLEANING
    _cleanup_model(teamspace, model_name, expected_num_versions=expected_num_versions)


@pytest.mark.parametrize(
    "importing",
    [
        pytest.param("lightning", marks=_SKIP_IF_LIGHTNING_BELLOW_2_5_1),
        pytest.param("pytorch_lightning", marks=_SKIP_IF_PYTORCHLIGHTNING_BELLOW_2_5_1),
    ],
)
@pytest.mark.cloud
@pytest.mark.skipif(platform.system() != "Linux", reason="Only Linux is supported for DDP")
def test_lightning_checkpoint_ddp(importing, tmp_path):
    if importing == "lightning":
        from lightning import Trainer
        from lightning.pytorch.demos.boring_classes import BoringModel
    elif importing == "pytorch_lightning":
        from pytorch_lightning import Trainer
        from pytorch_lightning.demos.boring_classes import BoringModel

    # model name with random hash
    teamspace, org_team, model_name = _prepare_variables("checkpoint_resume")
    trainer_args = {
        "default_root_dir": tmp_path,
        "accelerator": "cpu",
        "strategy": "ddp_spawn",
        "devices": 4,
        "model_registry": f"{org_team}/{model_name}",
    }

    trainer = Trainer(max_epochs=2, **trainer_args)
    trainer.fit(BoringModel())

    # FIXME: seems like barrier is not respected in the test, but in real life it correctly waits for all GPUs
    # trainer = Trainer(max_epochs=5, **trainer_args)
    # trainer.fit(BoringModel(), ckpt_path="registry")

    # CLEANING
    _cleanup_model(teamspace, model_name, expected_num_versions=2)


class DummyModel(PickleRegistryMixin):
    def __init__(self, value):
        self.value = value


@pytest.mark.cloud
def test_pickle_mixin_push_and_pull():
    # model name with random hash
    teamspace, org_team, model_name = _prepare_variables("pickle_mixin")
    model_registry = f"{org_team}/{model_name}"

    # Create an instance of DummyModel and call push_to_registry.
    dummy = DummyModel(42)
    dummy.upload_model(model_registry)

    # Call pull_from_registry and load the DummyModel instance.
    loaded_dummy = DummyModel.download_model(model_registry)
    # Verify that the unpickled instance has the expected value.
    assert isinstance(loaded_dummy, DummyModel)
    assert loaded_dummy.value == 42

    # CLEANING
    _cleanup_model(teamspace, model_name, expected_num_versions=1)


# This is a dummy model for PyTorch that uses the PyTorchRegistryMixin.
# This mixin has to be first in the inheritance order.
# Otherwise, `PyTorchRegistryMixin.__init__` need to be called explicitly.
class DummyTorchModel(PyTorchRegistryMixin, torch.nn.Module):
    def __init__(self, input_size: int, output_size: int = 10):
        # PyTorchRegistryMixin.__init__ will capture these arguments
        super().__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.mark.cloud
def test_pytorch_mixin_push_and_pull():
    # model name with random hash
    teamspace, org_team, model_name = _prepare_variables("torch_mixin")
    model_registry = f"{org_team}/{model_name}"

    # Create an instance, push the model and record its forward output.
    dummy = DummyTorchModel(784)
    dummy.eval()
    input_tensor = torch.randn(1, 784)
    output_before = dummy(input_tensor)

    dummy.upload_model(model_registry)

    loaded_dummy = DummyTorchModel.download_model(model_registry)
    loaded_dummy.eval()
    output_after = loaded_dummy(input_tensor)

    assert isinstance(loaded_dummy, DummyTorchModel)
    # Compare the outputs as a verification.
    assert torch.allclose(output_before, output_after), "Loaded model output differs from original."

    # CLEANING
    _cleanup_model(teamspace, model_name, expected_num_versions=1)


@pytest.mark.cloud
def test_duplicate_real_hf_model(tmp_path):
    """Verify that the HF model can be duplicated to the teamspace"""

    # model name with random hash
    model_name = f"litmodels_hf_model+{os.urandom(8).hex()}"
    teamspace = _resolve_teamspace(org=LIT_ORG, teamspace=LIT_TEAMSPACE, user=None)
    org_team = f"{teamspace.owner.name}/{teamspace.name}"

    duplicate_hf_model(hf_model="google/t5-efficient-tiny", lit_model=f"{org_team}/{model_name}")

    client = GridRestClient()
    model = client.models_store_get_model_by_name(
        project_owner_name=teamspace.owner.name,
        project_name=teamspace.name,
        model_name=model_name,
    )
    client.models_store_delete_model(project_id=teamspace.id, model_id=model.id)


@pytest.mark.cloud
def test_list_available_teamspaces():
    teams = _list_available_teamspaces()
    assert len(teams) > 0
    # using sanitized teamspace name
    assert f"{LIT_ORG}/oss-litmodels" in teams


@pytest.mark.cloud
@pytest.mark.skipif(
    not _KERAS_AVAILABLE,
    reason="TensorFlow Keras is not supported on Windows for now.",
)
def test_save_load_tensorflow_keras(tmp_path):
    from tensorflow import keras

    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(784,), name="dense_1"),
        keras.layers.Dense(10, name="dense_2"),
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    # model name with random hash
    teamspace, org_team, model_name = _prepare_variables("tf-keras")
    save_model(f"{org_team}/{model_name}", model=model)

    # Load the model
    model_ = load_model(f"{org_team}/{model_name}", download_dir=str(tmp_path))

    # validate the model
    assert isinstance(model_, type(model))
    _cleanup_model(teamspace, model_name, expected_num_versions=1)
