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
    """Print a stable URL to the uploaded model.

    Args:
        name: Model registry name. Teamspace defaults may be applied before URL construction.
        verbose: Controls printing behavior:
            - 0: do not print
            - 1: print the link only once for a given model
            - 2: always print the link
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
    """Upload local artifact(s) to Lightning Cloud using the SDK.

    Args:
        name: Model registry name in the form 'organization/teamspace/modelname[:version]'.
        path: File path, directory path, or list of paths to upload.
        progress_bar: Whether to show a progress bar during upload.
        cloud_account: Optional cloud account to store the model in, when it cannot be auto-resolved.
        verbose: Verbosity for printing the model link (0 = no output, 1 = print once, 2 = print always).
        metadata: Optional metadata to attach to the model/version. The package version is added automatically.

    Returns:
        UploadedModelInfo describing the created or updated model version.
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
    """Download artifact(s) for a model version using the SDK.

    Args:
        name: Model registry name in the form 'organization/teamspace/modelname[:version]'.
        download_dir: Directory where downloaded artifact(s) will be stored. Defaults to the current directory.
        progress_bar: Whether to show a progress bar during download.

    Returns:
        str | list[str]: Absolute path(s) to the downloaded artifact(s).
    """
    return sdk_download_model(
        name=name,
        download_dir=download_dir,
        progress_bar=progress_bar,
    )


def _list_available_teamspaces() -> dict[str, dict]:
    """List teamspaces available to the authenticated user.

    Returns:
        dict[str, dict]: Mapping of 'org/teamspace' to a metadata dictionary with details.
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
    version: str,
) -> None:
    """Delete a specific model version from the model store.

    Args:
        name: Base model registry name in the form 'organization/teamspace/modelname'.
        version: Identifier of the version to delete. This argument is required.
    """
    sdk_delete_model(name=f"{name}:{version}")
