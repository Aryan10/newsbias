"""Model- and dataset-agnostic inference benchmarking and result persistence."""

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

from datasets import DatasetDict

from newsbias.evaluation import Metrics, evaluate
from newsbias.experiment import code_identity, write_json
from newsbias.models.base import Model


class ModelBenchmarkResult(TypedDict):
    """Serializable fields shared by benchmark commands and artifact readers.

    `usage` collects optional adapter counters without assuming a particular provider
    or model-specific token accounting schema.
    """

    run_id: str
    model_identifier: str
    split: str
    metrics: Metrics
    inference_seconds: float
    usage: dict[str, int]


def run_model_benchmark(
    model: Model,
    dataset: DatasetDict,
    *,
    model_identifier: str,
    split: str,
    output_root: Path = Path("runs/benchmarks"),
    limit: int | None = None,
    batch_size: int = 32,
    dataset_metadata: dict[str, object] | None = None,
    model_metadata: dict[str, object] | None = None,
) -> Path:
    """Run any configured inference adapter over any normalized dataset split.

    Class labels are collected from all splits and sorted to give every adapter the
    same metric order. The model receives only IDs and text; gold labels and source
    metadata are joined back after inference. Per-sample predictions, failures,
    usage, adapter settings, and code identity are written beneath a unique run ID.

    `limit` selects the first N rows for smoke tests; `batch_size` is forwarded to
    the model adapter. The returned path identifies all artifacts for this run.
    """
    if split not in dataset:
        raise ValueError(f"Dataset has no {split!r} split")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    examples = dataset[split]
    if limit is not None:
        examples = examples.select(range(min(limit, len(examples))))
    inputs = examples.select_columns(["id", "text"])
    started = perf_counter()
    predictions = model.predict(inputs, batch_size=batch_size)
    inference_seconds = perf_counter() - started
    class_labels = sorted({
        label for split_dataset in dataset.values() for label in split_dataset["label"]
    })
    metrics = evaluate(list(examples["label"]), predictions, labels=class_labels)
    usage_keys = {
        key for prediction in predictions for key in prediction.get("usage", {})
    }
    usage_totals = {
        key: sum(prediction.get("usage", {}).get(key, 0) for prediction in predictions)
        for key in sorted(usage_keys)
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, prediction in zip(examples, predictions, strict=True):
            record = {"id": row["id"], "gold": row["label"], **prediction}
            if "source_split" in row:
                record["source_split"] = row["source_split"]
            handle.write(json.dumps(record) + "\n")
    summary: ModelBenchmarkResult = {
        "run_id": run_id,
        "model_identifier": model_identifier,
        "split": split,
        "metrics": metrics,
        "inference_seconds": inference_seconds,
        "usage": usage_totals,
    }
    write_json(output / "summary.json", {
        **summary,
        "dataset": dataset_metadata or {},
        "model": model_metadata or {},
        "code": code_identity(),
        "sample_count": len(examples),
        "probability_sources": sorted({
            prediction.get("probability_source", "unspecified")
            for prediction in predictions
        }),
    })
    return output
