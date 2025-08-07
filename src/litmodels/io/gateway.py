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
    """Upload a checkpoint to the model store.

    Args:
        name: Name of the model to upload. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        model: The model to upload. Can be a path to a checkpoint file or a folder.
        progress_bar: Whether to show a progress bar for the upload.
        cloud_account: The name of the cloud account to store the Model in. Only required if it can't be determined
            automatically.
        verbose: Whether to print some additional information about the uploaded model.
        metadata: Optional metadata to attach to the model. If not provided, a default metadata will be used.

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
    """Upload a checkpoint to the model store.

    Args:
        name: Name of the model to upload. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        model: The model to upload. Can be a PyTorch model, or a Lightning model a.
        progress_bar: Whether to show a progress bar for the upload.
        cloud_account: The name of the cloud account to store the Model in. Only required if it can't be determined
            automatically.
        staging_dir: A directory where the model can be saved temporarily. If not provided, a temporary directory will
            be created and used.
        verbose: Whether to print some additional information about the uploaded model.
        metadata: Optional metadata to attach to the model. If not provided, a default metadata will be used.

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
    """Download a checkpoint from the model store.

    Args:
        name: Name of the model to download. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        download_dir: A path to directory where the model should be downloaded. Defaults
            to the current working directory.
        progress_bar: Whether to show a progress bar for the download.

    Returns:
        The absolute path to the downloaded model file or folder.
    """
    return download_model_files(
        name=name,
        download_dir=download_dir,
        progress_bar=progress_bar,
    )


def load_model(
    name: str, 
    download_dir: str = ".",
    model_instance: Optional[object] = None
) -> Any:
    """Download a model from the model store and load it into memory.

    Args:
        name: Name of the model to download. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        download_dir: A path to directory where the model should be downloaded. Defaults
            to the current working directory.
        model_instance: Optional argument needed if loading a pure Pytorch model. Pass in a initialized
            instance of the model to load the model weights into.

    Returns:
        The loaded model.
    """
    download_paths = download_model(name=name, download_dir=download_dir)
    # filter out all Markdown, TXT and RST files
    download_paths = [p for p in download_paths if Path(p).suffix.lower() not in {".md", ".txt", ".rst"}]
    if len(download_paths) > 1:
        raise NotImplementedError("Downloaded model with multiple files is not supported yet.")
    model_path = Path(download_dir) / download_paths[0]
    if model_path.suffix.lower() == ".ts":
        return torch.jit.load(model_path)
    if model_path.suffix.lower() == ".pth":
        if model_instance is not None and isinstance(model_instance, torch.nn.Module):
            return model_instance.load_state_dict(torch.load(model_path))
        else:
            raise ValueError(
                "Trying to load a Pure Pytorch model. Expected the optional `model_instance`"
                "to be provided with a instance of the saved model to load the model weights into."
            )
    if model_path.suffix.lower() == ".keras":
        return keras.models.load_model(model_path)
    if model_path.suffix.lower() == ".pkl":
        return load_pickle(path=model_path)
    raise NotImplementedError(f"Loading model from {model_path.suffix} is not supported yet.")
