"""Tests for standalone merge_bias_datasets script and split generation."""

import csv
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path for scripts import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.merge_bias_datasets import build_parser, main, run_merge, save_records


def test_build_parser() -> None:
    """Parser configures all expected arguments with defaults."""
    parser = build_parser()
    args = parser.parse_args(["--split-strategy", "stratified", "--seed", "123"])
    assert args.split_strategy == "stratified"
    assert args.seed == 123
    assert args.test_dataset == "mbic"


def test_save_records_json_and_csv(tmp_path: Path) -> None:
    """Records persist accurately in either JSON or CSV format."""
    records = [
        {"id": "1", "text": "First sentence", "label": "biased", "label_binary": 1},
        {"id": "2", "text": "Second sentence", "label": "non-biased", "label_binary": 0},
    ]

    # Test JSON
    json_path = tmp_path / "out.json"
    save_records(records, json_path)
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(loaded_json) == 2
    assert loaded_json[0]["text"] == "First sentence"

    # Test CSV
    csv_path = tmp_path / "out.csv"
    save_records(records, csv_path)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 2
    assert reader[1]["label"] == "non-biased"


def test_run_merge_with_csv_inputs(tmp_path: Path) -> None:
    """Merge script harmonizes real CSV files, deduplicates, and saves output."""
    babe_csv = tmp_path / "babe.csv"
    babe_csv.write_text(
        "text,label\nControversial sentence,1\nBabe unique sentence,0\n",
        encoding="utf-8",
    )

    mbic_csv = tmp_path / "mbic.csv"
    mbic_csv.write_text(
        "text,label\nControversial sentence,0\nMbic unique statement,1\n",
        encoding="utf-8",
    )

    basil_csv = tmp_path / "basil.csv"
    basil_csv.write_text(
        "text,label,bias_type\nBasil lexical sentence,1,lexical\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "merged.json"
    parser = build_parser()
    args = parser.parse_args([
        "--babe", str(babe_csv),
        "--mbic", str(mbic_csv),
        "--basil", str(basil_csv),
        "--output", str(out_file),
    ])

    merged = run_merge(args)
    # Total raw was 5, duplicate "Controversial sentence" resolved to 1 -> 4 unique
    assert len(merged) == 4
    assert out_file.exists()

    # BABE priority must win for the duplicate
    dup = [r for r in merged if r["text"] == "Controversial sentence"][0]
    assert dup["dataset"] == "babe"
    assert dup["label"] == "biased"


def test_run_merge_stratified_splits(tmp_path: Path) -> None:
    """Merge script generates stratified partition files when requested."""
    babe_csv = tmp_path / "babe.csv"
    rows = ["text,label"]
    for i in range(20):
        rows.append(f"Sentence {i},{1 if i % 2 == 0 else 0}")
    babe_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("text,label\n", encoding="utf-8")

    out_file = tmp_path / "dataset.json"
    parser = build_parser()
    args = parser.parse_args([
        "--babe", str(babe_csv),
        "--mbic", str(empty_csv),
        "--basil", str(empty_csv),
        "--output", str(out_file),
        "--split-strategy", "stratified",
    ])

    run_merge(args)
    assert (tmp_path / "dataset_train.json").exists()
    assert (tmp_path / "dataset_validation.json").exists()
    assert (tmp_path / "dataset_test.json").exists()


def test_main_cli_execution(tmp_path: Path) -> None:
    """Script main entrypoint executes cleanly via argv parameters."""
    sample_csv = tmp_path / "sample.csv"
    sample_csv.write_text("text,label\nSimple sentence,1\n", encoding="utf-8")
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("text,label\n", encoding="utf-8")
    out_file = tmp_path / "final.json"

    exit_code = main([
        "--babe", str(sample_csv),
        "--mbic", str(empty_csv),
        "--basil", str(empty_csv),
        "--output", str(out_file),
    ])
    assert exit_code == 0
    assert out_file.exists()
