import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from litmodels.io.cloud import download_model_files, upload_model_files
from litmodels.io.utils import _KERAS_AVAILABLE, _PYTORCH_AVAILABLE, dump_pickle, load_pickle

if _PYTORCH_AVAILABLE:
    import torch

if _KERAS_AVAILABLE:
    from tensorflow import keras

if TYPE_CHECKING:
    from lightning_sdk.models import UploadedModelInfo


def upload_model(
    name: str,
    model: Union[str, Path],
    progress_bar: bool = True,
    cloud_account: Optional[str] = None,
    verbose: Union[bool, int] = 1,
    metadata: Optional[dict[str, str]] = None,
) -> "UploadedModelInfo":
    """Upload a local artifact (file or directory) to Lightning Cloud Models.

    Args:
        name: Model registry name in the form 'organization/teamspace/modelname[:version]'.
            If the version is omitted, one may be assigned automatically by the service.
        model: Path to a checkpoint file or a directory containing model artifacts.
        progress_bar: Whether to show a progress bar during the upload.
        cloud_account: Optional cloud account to store the model in, when it cannot be auto-resolved.
        verbose: Verbosity of informational output (0 = silent, 1 = print link once, 2 = print link always).
        metadata: Optional metadata key/value pairs to attach to the uploaded model/version.

    Returns:
        UploadedModelInfo describing the created or updated model version.

    Raises:
        ValueError: If `model` is not a filesystem path. For in-memory objects, use `save_model()` instead.
    """
    if not isinstance(model, (str, Path)):
        raise ValueError(
            "The `model` argument should be a path to a file or folder, not an python object."
            " For smooth integrations with PyTorch model, Lightning model and many more, use `save_model` instead."
        )

    return upload_model_files(
        path=model,
        name=name,
        progress_bar=progress_bar,
        cloud_account=cloud_account,
        verbose=verbose,
        metadata=metadata,
    )


def save_model(
    name: str,
    model: Union["torch.nn.Module", Any],
    progress_bar: bool = True,
    cloud_account: Optional[str] = None,
    staging_dir: Optional[str] = None,
    verbose: Union[bool, int] = 1,
    metadata: Optional[dict[str, str]] = None,
) -> "UploadedModelInfo":
    """Serialize an in-memory model and upload it to Lightning Cloud Models.

    Supported models:
        - TorchScript (torch.jit.ScriptModule) → saved as .ts via model.save()
        - PyTorch nn.Module → saved as .pth (state_dict via torch.save)
        - Keras (tf.keras.Model) → saved as .keras via model.save()
        - Any other Python object → saved as .pkl via pickle or joblib

    Args:
        name: Model registry name in the form 'organization/teamspace/modelname[:version]'.
        model: The in-memory model instance to serialize and upload.
        progress_bar: Whether to show a progress bar during the upload.
        cloud_account: Optional cloud account to store the model in, when it cannot be auto-resolved.
        staging_dir: Optional temporary directory used for serialization. A new temp directory is created if omitted.
        verbose: Verbosity of informational output (0 = silent, 1 = print link once, 2 = print link always).
        metadata: Optional metadata key/value pairs to attach to the uploaded model/version. Integration markers are
            added automatically.

    Returns:
        UploadedModelInfo describing the created or updated model version.

    Raises:
        ValueError: If `model` is a path. For file/folder uploads use `upload_model()` instead.
    """
    if isinstance(model, (str, Path)):
        raise ValueError(
            "The `model` argument should be a PyTorch model or a Lightning model, not a path to a file."
            " With file or folder path use `upload_model` instead."
        )

    if not staging_dir:
        staging_dir = tempfile.mkdtemp()
    # if LightningModule and isinstance(model, LightningModule):
    #     path = os.path.join(staging_dir, f"{model.__class__.__name__}.ckpt")
    #     model.save_checkpoint(path)
    if _PYTORCH_AVAILABLE and isinstance(model, torch.jit.ScriptModule):
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.ts")
        model.save(path)
    elif _PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.pth")
        torch.save(model.state_dict(), path)
    elif _KERAS_AVAILABLE and isinstance(model, keras.models.Model):
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.keras")
        model.save(path)
    else:
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.pkl")
        dump_pickle(model=model, path=path)

    if not metadata:
        metadata = {}
    metadata.update({"litModels.integration": "save_model"})

    return upload_model(
        model=path,
        name=name,
        progress_bar=progress_bar,
        cloud_account=cloud_account,
        verbose=verbose,
        metadata=metadata,
    )


def download_model(
    name: str,
    download_dir: Union[str, Path] = ".",
    progress_bar: bool = True,
) -> Union[str, list[str]]:
    """Download a model version from Lightning Cloud Models to a local directory.

    Args:
        name: Model registry name in the form 'organization/teamspace/modelname[:version]'.
        download_dir: Directory where the artifact(s) will be stored. Defaults to the current working directory.
        progress_bar: Whether to show a progress bar during the download.

    Returns:
        str | list[str]: Absolute path(s) to the downloaded file(s) or directory content.
    """
    return download_model_files(
        name=name,
        download_dir=download_dir,
        progress_bar=progress_bar,
    )


def load_model(name: str, download_dir: str = ".") -> Any:
    """Download a model and load it into memory based on its file extension.

    Supported formats:
        - .ts → torch.jit.load
        - .keras → keras.models.load_model
        - .pkl → pickle/joblib via load_pickle

    Args:
        name: Model registry name in the form 'organization/teamspace/modelname[:version]'.
        download_dir: Directory to store the downloaded artifact(s) before loading. Defaults to the current directory.

    Returns:
        Any: The loaded model object.

    Raises:
        NotImplementedError: If multiple files are downloaded or the file extension is not supported.
    """
    download_paths = download_model(name=name, download_dir=download_dir)
    # filter out all Markdown, TXT and RST files
    download_paths = [p for p in download_paths if Path(p).suffix.lower() not in {".md", ".txt", ".rst"}]
    if len(download_paths) > 1:
        raise NotImplementedError("Downloaded model with multiple files is not supported yet.")
    model_path = Path(download_dir) / download_paths[0]
    if model_path.suffix.lower() == ".ts":
        return torch.jit.load(model_path)
    if model_path.suffix.lower() == ".keras":
        return keras.models.load_model(model_path)
    if model_path.suffix.lower() == ".pkl":
        return load_pickle(path=model_path)
    raise NotImplementedError(f"Loading model from {model_path.suffix} is not supported yet.")
