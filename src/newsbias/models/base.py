"""Minimal model contract; inheritance is optional for adapters."""

from pathlib import Path
from typing import NotRequired, TypedDict

from datasets import Dataset


class Prediction(TypedDict):
    """Serializable result for one input row from any model adapter.

    `probabilities` may be absent in practice (or explicitly `None`) when a model
    does not expose scores or inference fails. Optional usage/timing/error metadata
    lets API adapters report operational details without changing the common fields.
    """

    label: str
    probabilities: dict[str, float] | None
    probability_source: NotRequired[str]
    usage: NotRequired[dict[str, int]]
    latency_seconds: NotRequired[float]
    error: NotRequired[str]
    text_truncated: NotRequired[bool]
    chunk_count: NotRequired[int]


class Model:
    """Optional typed contract shared by trainable and inference-only model adapters.

    Implementations may subclass this class or provide the same methods by duck
    typing. Unsupported operations should raise `NotImplementedError`, making model
    capabilities explicit without introducing a separate class hierarchy.
    """

    def fit(self, train: Dataset, validation: Dataset | None = None) -> None:
        """Learn model/task parameters from training examples when supported.

        Validation data is provided for model-specific selection or early stopping;
        it must not be used as training data. Inference-only adapters should leave
        this unsupported and raise `NotImplementedError`.
        """
        raise NotImplementedError("This model does not support training")

    def predict(self, dataset: Dataset, *, batch_size: int = 128) -> list[Prediction]:
        """Return one prediction per row in input order using the normalized `text` column.

        The benchmark runner supplies no gold labels or source metadata at inference.
        Adapters that cannot predict from their current state should raise clearly.
        """
        raise NotImplementedError("This model does not support inference")

    def save(self, path: Path) -> None:
        """Write a reloadable model bundle beneath `path` when the adapter supports it.

        The operation returns no object; implementations should raise
        `NotImplementedError` when there is no local state to persist.
        """
        raise NotImplementedError("This model does not support saving")
