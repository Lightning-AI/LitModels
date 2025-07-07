# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from lightning_sdk.lightning_cloud.env import LIGHTNING_CLOUD_URL
from lightning_sdk.models import _extend_model_name_with_teamspace, _parse_org_teamspace_model_version
from lightning_sdk.models import delete_model as sdk_delete_model
from lightning_sdk.models import download_model as sdk_download_model
from lightning_sdk.models import upload_model as sdk_upload_model

import litmodels

if TYPE_CHECKING:
    from lightning_sdk.models import UploadedModelInfo


_SHOWED_MODEL_LINKS = []


def _print_model_link(name: str, verbose: Union[bool, int]) -> None:
    """Print a link to the uploaded model.

    Args:
        name: Name of the model.
        verbose: Whether to print the link:

            - If set to 0, no link will be printed.
            - If set to 1, the link will be printed only once.
            - If set to 2, the link will be printed every time.
    """
    name = _extend_model_name_with_teamspace(name)
    org_name, teamspace_name, model_name, _ = _parse_org_teamspace_model_version(name)

    url = f"{LIGHTNING_CLOUD_URL}/{org_name}/{teamspace_name}/models/{model_name}"
    msg = f"Model uploaded successfully. Link to the model: '{url}'"
    if int(verbose) > 1:
        print(msg)
    elif url not in _SHOWED_MODEL_LINKS:
        print(msg)
        _SHOWED_MODEL_LINKS.append(url)


def upload_model_files(
    name: str,
    path: Union[str, Path, list[Union[str, Path]]],
    progress_bar: bool = True,
    cloud_account: Optional[str] = None,
    verbose: Union[bool, int] = 1,
    metadata: Optional[dict[str, str]] = None,
) -> "UploadedModelInfo":
    """Upload a local checkpoint file to the model store.

    Args:
        name: Name of the model to upload. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        path: Path to the model file to upload.
        progress_bar: Whether to show a progress bar for the upload.
        cloud_account: The name of the cloud account to store the Model in. Only required if it can't be determined
            automatically.
        verbose: Whether to print a link to the uploaded model. If set to 0, no link will be printed.
        metadata: Optional metadata to attach to the model. If not provided, a default metadata will be used.

    """
    if not metadata:
        metadata = {}
    metadata.update({"litModels": litmodels.__version__})
    info = sdk_upload_model(
        name=name,
        path=path,
        progress_bar=progress_bar,
        cloud_account=cloud_account,
        metadata=metadata,
    )
    if verbose:
        _print_model_link(name, verbose)
    return info


def download_model_files(
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
    return sdk_download_model(
        name=name,
        download_dir=download_dir,
        progress_bar=progress_bar,
    )


def _list_available_teamspaces() -> dict[str, dict]:
    """List available teamspaces for the authenticated user.

    Returns:
        Dict with teamspace names as keys and their details as values.
    """
    from lightning_sdk.api import OrgApi, UserApi
    from lightning_sdk.utils import resolve as sdk_resolvers

    org_api = OrgApi()
    user = sdk_resolvers._get_authed_user()
    teamspaces = {}
    for ts in UserApi()._get_all_teamspace_memberships(""):
        if ts.owner_type == "organization":
            org = org_api._get_org_by_id(ts.owner_id)
            teamspaces[f"{org.name}/{ts.name}"] = {"name": ts.name, "org": org.name}
        elif ts.owner_type == "user":  # todo: check also the name
            teamspaces[f"{user.name}/{ts.name}"] = {"name": ts.name, "user": user}
        else:
            raise RuntimeError(f"Unknown organization type {ts.organization_type}")
    return teamspaces


def delete_model_version(
    name: str,
    version: Optional[str] = None,
) -> None:
    """Delete a model version from the model store.

    Args:
        name: Name of the model to delete. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        version: Version of the model to delete. If not provided, all versions will be deleted.
    """
    sdk_delete_model(name=f"{name}:{version}")
