"""Tests for concrete normalized dataset-source adapters and their caches."""

import csv
from pathlib import Path

import pytest
from datasets import Dataset

from newsbias.datasets.babe import BABEDataset, normalize_babe


def test_babe_source_loads_normalizes_splits_and_caches(tmp_path: Path) -> None:
    """Read the authors' semicolon CSV, normalize both labels, and reuse its cache."""
    csv_path = tmp_path / "final_labels_SG2.csv"
    fields = ["text", "news_link", "outlet", "topic", "type", "label_bias",
              "label_opinion", "biased_words"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for label in ("Biased", "Non-biased"):
            for row in range(15):
                writer.writerow({
                    "text": f"{label} unique example sentence {row}",
                    "news_link": f"https://example.test/{label}/{row}",
                    "outlet": f"outlet-{row % 3}",
                    "topic": "politics",
                    "type": "left",
                    "label_bias": label,
                    "label_opinion": "Entirely factual",
                    "biased_words": "[]",
                })
    source = BABEDataset(
        csv_path,
        processed_cache_dir=tmp_path / "processed",
        download_cache_dir=tmp_path / "downloads",
    )
    first = source.load()
    second = source.load()
    assert {name: len(split) for name, split in first.items()} == {
        "train": 21, "validation": 3, "test": 6,
    }
    assert {name: set(split["label"]) for name, split in first.items()} == {
        name: {"biased", "non-biased"} for name in first
    }
    assert list(first["test"]["id"]) == list(second["test"]["id"])
    assert first["train"][0]["source"] == f"outlet-{int(first['train'][0]['text'].split()[-1]) % 3}"


def test_babe_normalization_rejects_unknown_labels() -> None:
    """Do not silently coerce an annotation outside BABE's binary target vocabulary."""
    raw = Dataset.from_dict({"text": ["A complete sentence."], "label_bias": ["unknown"]})
    with pytest.raises(ValueError, match="unknown bias label"):
        normalize_babe(raw)


def test_babe_requires_expected_columns() -> None:
    """Require both sentence text and expert bias annotation before loading examples."""
    raw = Dataset.from_dict({"sentence": ["A sentence"]})
    with pytest.raises(ValueError, match="missing columns"):
        normalize_babe(raw)


def test_merged_bias_dataset_loading(tmp_path: Path) -> None:
    """MergedBiasDataset parses partitioned JSON files into Hugging Face DatasetDict."""
    from newsbias.datasets.merged_bias import MergedBiasDataset

    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()

    for split in ("train", "validation", "test"):
        records = [
            {"id": f"{split}-1", "text": f"Sentence {split}", "label": "biased", "label_binary": 1}
        ]
        import json

        (proc_dir / f"merged_bias_{split}.json").write_text(
            json.dumps(records), encoding="utf-8"
        )

    ds_source = MergedBiasDataset(processed_dir=str(proc_dir))
    dataset_dict = ds_source.load()

    assert set(dataset_dict.keys()) == {"train", "validation", "test"}
    assert dataset_dict["train"][0]["text"] == "Sentence train"
    assert dataset_dict["train"][0]["label_id"] == 1
    assert dataset_dict["test"][0]["label"] == "biased"
