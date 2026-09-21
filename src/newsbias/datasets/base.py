"""Small duck-typed base interface for research dataset sources."""

from datasets import DatasetDict


class DatasetSource:
    """Optional base contract for repository dataset adapters.

    Subclasses implement `load()` and return the shared `id`, `text`, and `label`
    fields plus train/validation/test splits. The class provides no ABC machinery,
    so third-party sources can implement the same method by duck typing.
    """

    def load(self) -> DatasetDict:
        """Load, normalize, and split source data in a concrete adapter.

        The base raises to reveal an incomplete subclass immediately rather than
        returning a silently empty dataset.
        """
        raise NotImplementedError("Dataset source does not implement loading")
