"""Shared stratified splitting for datasets with one label per example."""

from datasets import DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split


def stratified_splits(
    dataset: DatasetDict,
    *,
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> DatasetDict:
    """Merge source partitions and create deterministic, label-stratified splits.

    Validation and test sizes are rounded from the requested fractions; train gets
    the remaining rows. `seed` and stable source row order determine membership.
    Every label must have enough examples to appear in all three outputs.
    """
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("Each split fraction must be between zero and one")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1")
    if not dataset:
        raise ValueError("No source splits supplied")
    combined = concatenate_datasets(list(dataset.values()))
    labels = list(combined["label"])
    indexes = list(range(len(combined)))
    validation_count = round(len(indexes) * validation_fraction)
    test_count = round(len(indexes) * test_fraction)
    remainder_count = validation_count + test_count
    if min(validation_count, test_count, len(indexes) - remainder_count) < 1:
        raise ValueError("Requested split fractions produce an empty split")
    try:
        train_indexes, remainder_indexes = train_test_split(
            indexes,
            test_size=remainder_count,
            random_state=seed,
            stratify=labels,
        )
        remainder_labels = [labels[index] for index in remainder_indexes]
        validation_indexes, test_indexes = train_test_split(
            remainder_indexes,
            train_size=validation_count,
            random_state=seed,
            stratify=remainder_labels,
        )
    except ValueError as error:
        raise ValueError(
            "Cannot stratify requested splits; every label needs enough examples "
            "for train, validation, and test"
        ) from error
    return DatasetDict({
        "train": combined.select(train_indexes),
        "validation": combined.select(validation_indexes),
        "test": combined.select(test_indexes),
    })
