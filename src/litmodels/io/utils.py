import os
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union

from lightning_utilities import module_available
from lightning_utilities.core.imports import RequirementCache


@contextmanager
def _suppress_os_stderr() -> None:
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
    """Dump a model to a pickle file.

    Args:
        model: The model to be pickled.
        path: The path where the model will be saved.
    """
    if _JOBLIB_AVAILABLE:
        joblib.dump(model, filename=path, compress=7)
    else:
        with open(path, "wb") as fp:
            pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load a model from a pickle file.

    Args:
        path: The path to the pickle file.

    Returns:
        The unpickled model.
    """
    if _JOBLIB_AVAILABLE:
        return joblib.load(path)
    with open(path, "rb") as fp:
        return pickle.load(fp)
