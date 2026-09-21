"""Shared small Hugging Face dataset fixtures for offline test cases."""

import pytest
from datasets import Dataset, DatasetDict

from newsbias.datasets.all_sides import normalize_allsides


@pytest.fixture
def dataset() -> DatasetDict:
    """Build a normalized, exact-text-disjoint fixture for model and runner tests.

    Four examples per class go to training and one per class to each evaluation
    split, allowing metrics, label coverage, and leakage checks to run offline.
    """
    texts = {
        "left": "progressive workers equality public welfare",
        "center": "committee meeting report schedule announcement",
        "right": "conservative markets liberty private enterprise",
    }
    splits = {}
    for split, count in (("train", 4), ("validation", 1), ("test", 1)):
        rows = [
            {"ID": f"{split}-{label}-{i}", "bias": number, "bias_text": label,
             "content": f"{text} {split} unique{i}", "source": "example"}
            for number, (label, text) in enumerate(texts.items())
            for i in range(count)
        ]
        splits[split] = Dataset.from_list(rows)
    return normalize_allsides(DatasetDict(splits))
