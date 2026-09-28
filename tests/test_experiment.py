"""Offline integration checks for configured experiment execution and artifacts."""

import json
from pathlib import Path

import pytest
from datasets import DatasetDict

from newsbias.experiment import read_config, run_experiment

CONFIG = Path(__file__).resolve().parents[1] / "configs/tfidf_allsides.toml"


def test_offline_end_to_end(dataset: DatasetDict, tmp_path: Path) -> None:
    """Exercise configured adapter construction, fit, save, inference, and artifacts offline."""
    config = read_config(CONFIG)
    config["model"]["parameters"]["min_df"] = 1
    config["evaluation"]["splits"] = ["validation", "test"]
    output = run_experiment(config, output_root=tmp_path, dataset=dataset)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["results"]["test"]["model"]["accuracy"] == 1
    assert summary["results"]["test"]["majority_baseline"]["accuracy"] == pytest.approx(1 / 3)
    lines = (output / "test_predictions.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [row["id"] for row in records] == list(dataset["test"]["id"])
    assert len((output / "manifest.jsonl").read_text().splitlines()) == 18
    assert (output / "model/pipeline.joblib").exists()
    assert len(summary["code"]["sha256"]) == 64


def test_config_rejects_wrong_types(tmp_path: Path) -> None:
    """Reject malformed adapter import paths before attempting dynamic construction."""
    path = tmp_path / "config.toml"
    path.write_text(CONFIG.read_text().replace(
        'adapter = "newsbias.datasets.all_sides:AllSidesDataset"',
        'adapter = "not-a-valid-adapter"',
    ))
    with pytest.raises(ValueError, match="module:ClassName"):
        read_config(path)
