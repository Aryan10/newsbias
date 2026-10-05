"""Verify local/Kaggle split membership parity without remote API access."""

import pytest
from datasets import Dataset, DatasetDict

from newsbias.datasets.splitting import stratified_splits
from newsbias.remote.scripts.run_transformer_kaggle import load_project_splits


def test_kaggle_split_matches_local_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Kaggle reconstructs exactly the local seed-42 membership."""
    rows = [
        {"id": f"{label}-{index}", "label": label, "category": label,
         "text": f"Example for {label}, unique id {index}", "source_split": "source"}
        for label in ("economy", "health", "science")
        for index in range(30)
    ]
    original = DatasetDict({
        "train": Dataset.from_list(rows[:54]),
        "validation": Dataset.from_list(rows[54:72]),
        "test": Dataset.from_list(rows[72:]),
    })
    expected = stratified_splits(original, seed=42)

    def fake_load_dataset(name: str, *, revision: str) -> DatasetDict:
        """Return fixture splits and verify pinned source arguments."""
        assert name == "fixture/allsides"
        assert revision == "fixture-revision"
        return original

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    actual = load_project_splits(
        "fixture/allsides", "fixture-revision", label_column="category", seed=42,
    )
    assert {
        split: list(data["id"]) for split, data in actual.items()
    } == {
        split: list(data["id"]) for split, data in expected.items()
    }
