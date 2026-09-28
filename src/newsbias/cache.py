"""Read benchmark-run artifacts so command-line comparisons can reuse completed work."""

import json
from pathlib import Path
from typing import Any


def find_cached_benchmark(
    output_root: Path,
    *,
    model_identifier: str,
    split: str,
    limit: int | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    """Return the newest readable run whose recorded model, split, and size match.

    Searches `output_root/benchmarks` when it exists and otherwise supports older
    runs written directly beneath `output_root`. Incomplete or malformed summaries
    are skipped. A missing limit matches any run size; a supplied limit must match
    the recorded sample count exactly.
    """
    benchmark_dir = output_root / "benchmarks"
    search_dirs = [benchmark_dir, output_root] if benchmark_dir.is_dir() else [output_root]

    matching_candidates: list[tuple[Path, dict[str, Any]]] = []

    for base_dir in search_dirs:
        if not base_dir.is_dir():
            continue
        for run_path in sorted(base_dir.iterdir(), reverse=True):
            if not run_path.is_dir():
                continue
            summary_file = run_path / "summary.json"
            if not summary_file.is_file():
                continue
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            if summary.get("split") != split:
                continue

            recorded_model = (
                summary.get("model_metadata", {}).get("parameters", {}).get("model_name")
                or summary.get("model", {}).get("parameters", {}).get("model_name")
                or summary.get("model_identifier")
                or ""
            )
            # Match either exact identifier, adapter path, or model name stem
            if (
                recorded_model != model_identifier
                and recorded_model.lower() != model_identifier.lower()
                and Path(recorded_model).stem != Path(model_identifier).stem
            ):
                continue

            recorded_count = summary.get("sample_count") or summary.get("metrics", {}).get(
                "count"
            )
            if limit is not None and recorded_count != limit:
                continue

            matching_candidates.append((run_path, summary))

    return matching_candidates[0] if matching_candidates else None


def list_benchmark_runs(output_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load readable run summaries from the active benchmark directory, newest first.

    This is a shallow listing: only immediate child directories with a valid
    `summary.json` are returned. Malformed or partially written summaries are ignored.
    """
    benchmark_dir = output_root / "benchmarks"
    search_dir = benchmark_dir if benchmark_dir.is_dir() else output_root
    if not search_dir.is_dir():
        return []

    runs: list[tuple[Path, dict[str, Any]]] = []
    for run_path in sorted(search_dir.iterdir(), reverse=True):
        summary_file = run_path / "summary.json"
        if not summary_file.is_file():
            continue
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            runs.append((run_path, summary))
        except (json.JSONDecodeError, OSError):
            continue
    return runs
