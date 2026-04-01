import pytest

from litmodels.integrations.imports import (
    _LIGHTNING_AVAILABLE,
    _LIGHTNING_GREATER_EQUAL_2_5_1,
    _PYTORCHLIGHTNING_AVAILABLE,
    _PYTORCHLIGHTNING_GREATER_EQUAL_2_5_1,
)

_SKIP_IF_LIGHTNING_MISSING = pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="Lightning not available")
_SKIP_IF_LIGHTNING_BELLOW_2_5_1 = pytest.mark.skipif(
    not _LIGHTNING_GREATER_EQUAL_2_5_1, reason="Lightning without integration introduced in 2.5.1"
)
_SKIP_IF_PYTORCHLIGHTNING_MISSING = pytest.mark.skipif(
    not _PYTORCHLIGHTNING_AVAILABLE, reason="PyTorch Lightning not available"
)
_SKIP_IF_PYTORCHLIGHTNING_BELLOW_2_5_1 = pytest.mark.skipif(
    not _PYTORCHLIGHTNING_GREATER_EQUAL_2_5_1, reason="PyTorch Lightning without integration introduced in 2.5.1"
)

LIT_USER = "i-am-ci"
LIT_TEAMSPACE = "oss-litlogger"
