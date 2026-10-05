"""Remote execution abstractions and runners for cloud GPU environments."""

from newsbias.remote.base import (
    RemoteJobConfig,
    RemoteJobStatus,
    RemoteRunner,
    RemoteRunResult,
)
from newsbias.remote.kaggle import KaggleRunner

__all__ = [
    "KaggleRunner",
    "RemoteJobConfig",
    "RemoteJobStatus",
    "RemoteRunResult",
    "RemoteRunner",
]
