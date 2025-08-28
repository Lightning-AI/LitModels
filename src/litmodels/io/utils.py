import os
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union

from lightning_utilities import module_available
from lightning_utilities.core.imports import RequirementCache


@contextmanager
def _suppress_os_stderr() -> Iterator[None]:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)  # redirect stderr (fd 2) to /dev/null
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(old_stderr_fd, 2)  # restore stderr
        os.close(old_stderr_fd)


_JOBLIB_AVAILABLE = module_available("joblib")
_PYTORCH_AVAILABLE = module_available("torch")
with _suppress_os_stderr():
    _TENSORFLOW_AVAILABLE = module_available("tensorflow")
    _KERAS_AVAILABLE = RequirementCache("tensorflow >=2.0.0")

if _JOBLIB_AVAILABLE:
    import joblib


def dump_pickle(model: Any, path: Union[str, Path]) -> None:
    """Serialize a Python object to disk using joblib (if available) or pickle.

    Args:
        model: The object to serialize.
        path: Destination file path.

    Notes:
        - Uses joblib with compression (level 7) when available for smaller artifacts.
        - Falls back to pickle with the highest protocol otherwise.
    """
    if _JOBLIB_AVAILABLE:
        joblib.dump(model, filename=path, compress=7)
    else:
        with open(path, "wb") as fp:
            pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load a Python object from a joblib/pickle file.

    Args:
        path: Path to the serialized artifact.

    Returns:
        Any: The deserialized object.

    Warning:
        Loading pickle/joblib files can execute arbitrary code. Only open files from trusted sources.
    """
    if _JOBLIB_AVAILABLE:
        return joblib.load(path)
    with open(path, "rb") as fp:
        return pickle.load(fp)
