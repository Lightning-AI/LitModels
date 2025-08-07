import os
from contextlib import nullcontext
from unittest import mock

import joblib
import pytest
import torch
import torch.jit as torch_jit
from sklearn import svm
from torch.nn import Module

import litmodels
from litmodels import download_model, load_model, save_model
from litmodels.io import upload_model_files
from litmodels.io.utils import _KERAS_AVAILABLE
from tests.integrations import LIT_ORG, LIT_TEAMSPACE


@pytest.mark.parametrize("name", ["/too/many/slashes", "org/model", "model-name"])
@pytest.mark.parametrize("in_studio", [True, False])
def test_upload_wrong_model_name(name, in_studio, monkeypatch):
    if in_studio:
        # mock env variables as it would run in studio
        monkeypatch.setenv("LIGHTNING_ORG", LIT_ORG)
        monkeypatch.setenv("LIGHTNING_TEAMSPACE", LIT_TEAMSPACE)
        monkeypatch.setattr("lightning_sdk.organization.Organization", mock.MagicMock)
        monkeypatch.setattr("lightning_sdk.teamspace.Teamspace", mock.MagicMock)
        monkeypatch.setattr("lightning_sdk.teamspace.TeamspaceApi", mock.MagicMock)
        monkeypatch.setattr("lightning_sdk.models._get_teamspace", mock.MagicMock)

    in_studio_only_name = in_studio and name == "model-name"
    with (
        pytest.raises(ValueError, match=r".*organization/teamspace/model.*")
        if not in_studio_only_name
        else nullcontext()
    ):
        upload_model_files(path="path/to/checkpoint", name=name)


@pytest.mark.parametrize("name", ["/too/many/slashes", "org/model", "model-name"])
@pytest.mark.parametrize("in_studio", [True, False])
def test_download_wrong_model_name(name, in_studio, monkeypatch):
    if in_studio:
        # mock env variables as it would run in studio
        monkeypatch.setenv("LIGHTNING_ORG", LIT_ORG)
        monkeypatch.setenv("LIGHTNING_TEAMSPACE", LIT_TEAMSPACE)
        monkeypatch.setattr("lightning_sdk.organization.Organization", mock.MagicMock)
        monkeypatch.setattr("lightning_sdk.teamspace.Teamspace", mock.MagicMock)
        monkeypatch.setattr("lightning_sdk.models.TeamspaceApi", mock.MagicMock)
    in_studio_only_name = in_studio and name == "model-name"
    with (
        pytest.raises(ValueError, match=r".*organization/teamspace/model.*")
        if not in_studio_only_name
        else nullcontext()
    ):
        download_model(name=name)


@pytest.mark.parametrize(
    ("model", "model_path", "verbose"),
    [
        # ("path/to/checkpoint", "path/to/checkpoint", False),
        # (BoringModel(), "%s/BoringModel.ckpt"),
        (torch_jit.script(Module()), f"%s{os.path.sep}RecursiveScriptModule.ts", True),
        (Module(), f"%s{os.path.sep}Module.pth", True),
        (svm.SVC(), f"%s{os.path.sep}SVC.pkl", 1),
    ],
)
@mock.patch("litmodels.io.cloud.sdk_upload_model")
def test_upload_model(mock_upload_model, tmp_path, model, model_path, verbose):
    mock_upload_model.return_value.name = "org-name/teamspace/model-name"

    # The lit-logger function is just a wrapper around the SDK function
    save_model(
        model=model,
        name="org-name/teamspace/model-name",
        cloud_account="cluster_id",
        staging_dir=str(tmp_path),
        verbose=verbose,
    )
    expected_path = model_path % str(tmp_path) if "%" in model_path else model_path
    mock_upload_model.assert_called_once_with(
        path=expected_path,
        name="org-name/teamspace/model-name",
        cloud_account="cluster_id",
        progress_bar=True,
        metadata={"litModels": litmodels.__version__, "litModels.integration": "save_model"},
    )


@mock.patch("litmodels.io.cloud.sdk_download_model")
def test_download_model(mock_download_model):
    # The lit-logger function is just a wrapper around the SDK function
    download_model(
        name="org-name/teamspace/model-name",
        download_dir="where/to/download",
    )
    mock_download_model.assert_called_once_with(
        name="org-name/teamspace/model-name", download_dir="where/to/download", progress_bar=True
    )


@mock.patch("litmodels.io.cloud.sdk_download_model")
def test_load_model_pickle(mock_download_model, tmp_path):
    # create a dummy model file
    model_file = tmp_path / "dummy_model.pkl"
    test_data = svm.SVC()
    joblib.dump(test_data, model_file)
    mock_download_model.return_value = [str(model_file.name)]

    # The lit-logger function is just a wrapper around the SDK function
    model = load_model(
        name="org-name/teamspace/model-name",
        download_dir=str(tmp_path),
    )
    mock_download_model.assert_called_once_with(
        name="org-name/teamspace/model-name", download_dir=str(tmp_path), progress_bar=True
    )
    assert isinstance(model, svm.SVC)


@mock.patch("litmodels.io.cloud.sdk_download_model")
def test_load_model_torch_jit(mock_download_model, tmp_path):
    # create a dummy model file
    model_file = tmp_path / "dummy_model.ts"
    test_data = torch_jit.script(Module())
    test_data.save(model_file)
    mock_download_model.return_value = [str(model_file.name)]

    # The lit-logger function is just a wrapper around the SDK function
    model = load_model(
        name="org-name/teamspace/model-name",
        download_dir=str(tmp_path),
    )
    mock_download_model.assert_called_once_with(
        name="org-name/teamspace/model-name", download_dir=str(tmp_path), progress_bar=True
    )
    assert isinstance(model, torch.jit.ScriptModule)


@mock.patch("litmodels.io.cloud.sdk_download_model")
def test_load_model_torch(mock_download_model, tmp_path):
    # craete a dummy file
    model_file = tmp_path / "dummy_model.pth"
    test_data = torch.nn.Module()
    torch.save(test_data, model_file)
    mock_download_model.return_value = [str(model_file.name)]

    # The lit-logger function is just a wrapper around the SDK function
    model = load_model(
        name="org-name/teamspace/model-name", download_dir=str(tmp_path), model_instance=torch.nn.Module()
    )
    mock_download_model.assert_called_once_with(
        name="org-name/teamspace/model-name", download_dir=str(tmp_path), progress_bar=True
    )
    assert isinstance(model, torch.nn.Module)


@pytest.mark.skipif(not _KERAS_AVAILABLE, reason="TensorFlow/Keras is not available")
@mock.patch("litmodels.io.cloud.sdk_download_model")
def test_load_model_tf_keras(mock_download_model, tmp_path):
    from tensorflow import keras

    # create a dummy model file
    model_file = tmp_path / "dummy_model.keras"
    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(784,), name="dense_1"),
        keras.layers.Dense(10, name="dense_2"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.save(model_file)
    # prepare mocked SDK download function
    mock_download_model.return_value = [str(model_file.name)]

    # The lit-logger function is just a wrapper around the SDK function
    model = load_model(
        name="org-name/teamspace/model-name",
        download_dir=str(tmp_path),
    )
    mock_download_model.assert_called_once_with(
        name="org-name/teamspace/model-name", download_dir=str(tmp_path), progress_bar=True
    )
    assert isinstance(model, keras.models.Model)
