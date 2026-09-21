"""Regression checks for source normalization, integrity audits, and stratification."""

from collections import Counter
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from newsbias.data import (
    audit_dataset,
    text_hash,
    validate_splits,
)
from newsbias.datasets.all_sides import (
    ALLSIDES_NAME,
    ALLSIDES_REVISION,
    AllSidesDataset,
    normalize_allsides,
)
from newsbias.datasets.splitting import stratified_splits


def test_normalization_and_audit(dataset: DatasetDict) -> None:
    """Verify AllSides rows expose the common schema and retain audit provenance."""
    validate_splits(dataset)
    audit = audit_dataset(dataset)
    assert audit["splits"]["train"]["label_counts"] == {"left": 4, "center": 4, "right": 4}
    assert audit["numeric_label_mapping"] == {"0": "left", "1": "center", "2": "right"}
    assert audit["overlaps"]["train/test"] == {"ids": 0, "texts": 0, "sources": 1}
    assert dataset["train"][0]["id"] == "allsides:train-left-0"
    assert dataset["train"][0]["text"] == dataset["train"][0]["content"]
    assert text_hash("a\n b") == text_hash("a b")


def test_leakage_is_rejected(dataset: DatasetDict) -> None:
    """Fail validation when the same article appears in training and evaluation."""
    dataset["test"] = dataset["train"].select([0])
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(dataset)


@pytest.mark.parametrize("change, message", [
    ({"bias_text": "unknown"}, "unknown label"),
    ({"ID": None}, "missing article ID"),
])
def test_bad_source_rows(change: dict[str, str | None], message: str) -> None:
    """Reject malformed native rows before creating normalized model examples."""
    row = {"ID": "1", "bias": 0, "bias_text": "left", "content": "text", "source": "x"}
    row.update(change)
    with pytest.raises(ValueError, match=message):
        normalize_allsides(DatasetDict(train=Dataset.from_list([row])))


def test_mapping_consistency() -> None:
    """Keep the source's integer labels and readable class names one-to-one."""
    rows = [{"ID": str(i), "bias": 0, "bias_text": label, "content": label, "source": "x"}
            for i, label in enumerate(["left", "right"])]
    with pytest.raises(ValueError, match="Inconsistent"):
        normalize_allsides(DatasetDict(train=Dataset.from_list(rows)))


def test_empty_text_is_audited_and_rejected(dataset: DatasetDict) -> None:
    """Make empty examples visible in audit output and invalid for model evaluation."""
    empty = dataset["test"].remove_columns(["text"]).add_column("text", ["", "a", "b"])
    dataset["test"] = empty
    assert audit_dataset(dataset)["splits"]["test"]["empty_texts"] == 1
    with pytest.raises(ValueError, match="empty or duplicate"):
        validate_splits(dataset)


def test_loader_pins_revision_and_reuses_processed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin source data and verify persistent cache reuse and seed-based invalidation.

    The fake Hub loader counts fetches: identical settings reuse disk artifacts,
    while changing the random seed selects a new processed cache entry.
    """
    rows = [
        {"ID": f"id-{label}-{i}", "bias": number, "bias_text": label,
         "content": f"article {label} {i}", "source": f"source-{i % 4}"}
        for number, label in enumerate(("left", "center", "right"))
        for i in range(30)
    ]
    raw = DatasetDict({
        "train": Dataset.from_list(rows[:54]),
        "validation": Dataset.from_list(rows[54:72]),
        "test": Dataset.from_list(rows[72:]),
    })

    calls = 0

    def fake_load(name: str, *, revision: str, cache_dir: Path | None) -> DatasetDict:
        """Count Hub loads and assert the adapter requests its configured revision."""
        nonlocal calls
        calls += 1
        assert name == ALLSIDES_NAME
        assert revision == ALLSIDES_REVISION
        assert cache_dir is None
        return raw

    monkeypatch.setattr("newsbias.datasets.all_sides.load_dataset", fake_load)
    source = AllSidesDataset(processed_cache_dir=tmp_path, download_cache_dir=None)
    loaded = source.load()
    reloaded_source = AllSidesDataset(
        processed_cache_dir=tmp_path,
        download_cache_dir=None,
    )
    reloaded = reloaded_source.load()
    assert calls == 1
    assert list(reloaded["test"]["id"]) == list(loaded["test"]["id"])
    changed_split = AllSidesDataset(
        processed_cache_dir=tmp_path,
        download_cache_dir=None,
        split_seed=17,
    ).load()
    assert calls == 2
    assert set(changed_split["test"]["id"]) != set(loaded["test"]["id"])
    assert {name: len(split) for name, split in loaded.items()} == {
        "train": 63, "validation": 9, "test": 18,
    }
    assert set(loaded["train"]["source_split"]) | set(loaded["validation"]["source_split"]) | set(
        loaded["test"]["source_split"]
    ) == {"train", "validation", "test"}
    validate_splits(loaded)


def test_stratified_70_10_20_split_is_reproducible() -> None:
    """Check exact small-sample ratios, per-label balance, stable IDs, and no overlap."""
    rows = [
        {"ID": f"{label}-{i}", "bias": number, "bias_text": label,
         "content": f"article {label} {i}", "source": f"source-{i % 4}"}
        for number, label in enumerate(("left", "center", "right"))
        for i in range(10)
    ]
    normalized = normalize_allsides(DatasetDict(train=Dataset.from_list(rows)))
    split_a = stratified_splits(normalized)
    split_b = stratified_splits(normalized)
    assert {name: len(split) for name, split in split_a.items()} == {
        "train": 21, "validation": 3, "test": 6,
    }
    assert {name: dict(Counter(split["label"])) for name, split in split_a.items()} == {
        "train": {"left": 7, "center": 7, "right": 7},
        "validation": {"left": 1, "center": 1, "right": 1},
        "test": {"left": 2, "center": 2, "right": 2},
    }
    assert {name: list(split["id"]) for name, split in split_a.items()} == {
        name: list(split["id"]) for name, split in split_b.items()
    }
    for split in split_a.values():
        assert set(split["label"]) == {"left", "center", "right"}
    assert set(split_a["train"]["id"]).isdisjoint(split_a["test"]["id"])


@pytest.mark.parametrize("fractions", [(0.7, 0.2, 0.2), (0.0, 0.2, 0.8)])
def test_invalid_split_fractions(
    dataset: DatasetDict, fractions: tuple[float, float, float],
) -> None:
    """Reject split ratios that cannot define three positive partitions summing to one."""
    with pytest.raises(ValueError):
        stratified_splits(
            dataset,
            train_fraction=fractions[0],
            validation_fraction=fractions[1],
            test_fraction=fractions[2],
        )
