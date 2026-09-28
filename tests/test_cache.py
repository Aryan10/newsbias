"""Tests for run cache indexing and lookup."""

import json
from pathlib import Path

from newsbias.cache import find_cached_benchmark, list_benchmark_runs


def test_find_cached_benchmark_and_listing(tmp_path: Path) -> None:
    """Reuse only results with matching run identity, while listing newest summaries first.

    The fixture includes a second model and sample count to ensure lookup does not
    accidentally treat any result on the same split as a cache hit.
    """
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir(parents=True)

    # Run 1: test-model, split=test, limit=50
    run1 = benchmarks_dir / "20261029T100000Z-11111111"
    run1.mkdir()
    (run1 / "summary.json").write_text(
        json.dumps({
            "model_identifier": "test-model",
            "split": "test",
            "sample_count": 50,
            "metrics": {"accuracy": 0.8, "macro_f1": 0.78, "count": 50},
        }),
        encoding="utf-8",
    )

    # Run 2: other-model, split=test
    run2 = benchmarks_dir / "20261029T110000Z-22222222"
    run2.mkdir()
    (run2 / "summary.json").write_text(
        json.dumps({
            "model_identifier": "other-model",
            "split": "test",
            "sample_count": 100,
            "metrics": {"accuracy": 0.9, "macro_f1": 0.89, "count": 100},
        }),
        encoding="utf-8",
    )

    # Match exact
    hit = find_cached_benchmark(tmp_path, model_identifier="test-model", split="test", limit=50)
    assert hit is not None
    assert hit[0] == run1
    assert hit[1]["metrics"]["accuracy"] == 0.8

    # Limit mismatch should return None
    assert (
        find_cached_benchmark(
            tmp_path, model_identifier="test-model", split="test", limit=100
        )
        is None
    )

    # Split mismatch should return None
    assert (
        find_cached_benchmark(
            tmp_path, model_identifier="test-model", split="validation", limit=50
        )
        is None
    )

    # List all runs
    all_runs = list_benchmark_runs(tmp_path)
    assert len(all_runs) == 2
    assert all_runs[0][0] == run2  # newest first
