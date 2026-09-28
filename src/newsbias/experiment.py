"""Configuration-driven adapter execution and reproducible run artifacts."""

import json
import platform
import subprocess
import tomllib
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import TypedDict, cast
from uuid import uuid4

from datasets import DatasetDict

from newsbias.adapters import AdapterConfig, load_adapter
from newsbias.data import audit_dataset, validate_splits
from newsbias.evaluation import evaluate
from newsbias.models.base import Model, Prediction


class TrainingConfig(TypedDict):
    """Choose fit/save behavior without assuming every model is trainable.

    Inference-only adapters set `enabled` false; saving can only be requested after fit.
    """

    enabled: bool
    save_model: bool


class EvaluationConfig(TypedDict):
    """Select returned split names and a positive batch size for prediction."""

    splits: list[str]
    batch_size: int


class ExperimentConfig(TypedDict):
    """Adapter-neutral experiment schema used by execution and CLI parsing."""

    dataset: AdapterConfig
    model: AdapterConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def _validate_adapter_config(section: object, name: str) -> AdapterConfig:
    """Check a plugin table before import while leaving its constructor schema open.

    Shared config validates only the `module:ClassName` path and parameter table;
    each concrete adapter owns validation for model- or dataset-specific options.
    """
    if not isinstance(section, dict) or set(section) != {"adapter", "parameters"}:
        raise ValueError(f"{name} must contain exactly 'adapter' and 'parameters'")
    if not isinstance(section["adapter"], str) or not section["adapter"]:
        raise ValueError(f"{name}.adapter must be a nonempty import path")
    if ":" not in section["adapter"]:
        raise ValueError(f"{name}.adapter must use 'module:ClassName' syntax")
    if not isinstance(section["parameters"], dict):
        raise ValueError(f"{name}.parameters must be a TOML table")
    return cast(AdapterConfig, section)


def read_config(path: Path) -> ExperimentConfig:
    """Read a run file and validate common fields without naming plugins.

    The TOML file chooses dataset and model adapters by import path. This function
    validates the shared structure, train/save policy, split list, and batch size,
    leaving adapter-specific constructor parameters to their implementations.
    """
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    expected_sections = {"dataset", "model", "training", "evaluation"}
    if set(config) != expected_sections:
        raise ValueError(f"Config sections must be exactly {sorted(expected_sections)}")
    config["dataset"] = _validate_adapter_config(config["dataset"], "dataset")
    config["model"] = _validate_adapter_config(config["model"], "model")

    training = config["training"]
    if not isinstance(training, dict) or set(training) != {"enabled", "save_model"}:
        raise ValueError("training must contain exactly 'enabled' and 'save_model'")
    if any(type(value) is not bool for value in training.values()):
        raise ValueError("training.enabled and training.save_model must be booleans")
    if training["save_model"] and not training["enabled"]:
        raise ValueError("A model artifact can only be saved when training is enabled")

    evaluation = config["evaluation"]
    if not isinstance(evaluation, dict) or set(evaluation) != {"splits", "batch_size"}:
        raise ValueError("evaluation must contain exactly 'splits' and 'batch_size'")
    splits = evaluation["splits"]
    if not isinstance(splits, list) or not splits or any(
        not isinstance(split, str) or not split for split in splits
    ):
        raise ValueError("evaluation.splits must be a nonempty list of split names")
    if len(set(splits)) != len(splits):
        raise ValueError("evaluation.splits must not contain duplicates")
    if type(evaluation["batch_size"]) is not int or evaluation["batch_size"] < 1:
        raise ValueError("evaluation.batch_size must be a positive integer")
    return cast(ExperimentConfig, config)


def write_json(path: Path, value: object) -> None:
    """Write an indented UTF-8 artifact, rejecting non-finite metric values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code_identity() -> dict[str, str | bool | None]:
    """Record source/dependency content hash plus available Git revision and dirty state.

    The content digest remains useful before the repository has its first commit and
    changes whenever Python source, project metadata, or the lockfile changes.
    """
    root = Path(__file__).resolve().parents[2]
    digest = sha256()
    for source_path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(str(source_path.relative_to(root)).encode())
        digest.update(source_path.read_bytes())
    for filename in ("pyproject.toml", "uv.lock"):
        lock_path = root / filename
        if lock_path.exists():
            digest.update(filename.encode())
            digest.update(lock_path.read_bytes())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False,
    )
    return {
        "sha256": digest.hexdigest(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout) if status.returncode == 0 else None,
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    output_root: Path = Path("runs"),
    dataset: DatasetDict | None = None,
) -> Path:
    """Execute configured dataset/model adapters and persist aligned predictions.

    The dataset adapter supplies normalized splits. Training and model saving follow
    the explicit config flags, so inference-only adapters need not train or export.
    The optional `dataset` argument injects a prepared DatasetDict for tests or
    programmatic callers. The returned directory contains config, audits, stable
    sample manifests, predictions, metrics, and any requested model artifact.
    """
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    started = perf_counter()

    if dataset is None:
        source = load_adapter(config["dataset"])
        if not callable(getattr(source, "load", None)):
            raise TypeError("Configured dataset adapter must implement load()")
        dataset = source.load()
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Dataset adapter load() must return a Hugging Face DatasetDict")
    audit = audit_dataset(dataset)
    validate_splits(dataset, audit)
    write_json(output / "audit.json", audit)
    _write_manifest(output / "manifest.jsonl", dataset)

    model_adapter = load_adapter(config["model"])
    for method in ("fit", "predict"):
        if not callable(getattr(model_adapter, method, None)):
            raise TypeError(f"Configured model adapter must implement {method}()")
    model = cast(Model, model_adapter)

    fit_seconds = 0.0
    if config["training"]["enabled"]:
        fit_started = perf_counter()
        model.fit(dataset["train"], dataset["validation"])
        fit_seconds = perf_counter() - fit_started
        if config["training"]["save_model"]:
            model.save(output / "model")

    class_labels = sorted(set(dataset["train"]["label"]))
    majority_label = Counter(dataset["train"]["label"]).most_common(1)[0][0]
    results: dict[str, object] = {}
    for split_name in config["evaluation"]["splits"]:
        if split_name not in dataset:
            raise ValueError(f"Dataset adapter returned no {split_name!r} split")
        examples = dataset[split_name]
        inputs = examples.select_columns(["id", "text"])
        inference_started = perf_counter()
        predictions = model.predict(inputs, batch_size=config["evaluation"]["batch_size"])
        inference_seconds = perf_counter() - inference_started
        metrics = evaluate(list(examples["label"]), predictions, labels=class_labels)
        baseline_predictions: list[Prediction] = [
            {"label": majority_label, "probabilities": None} for _ in examples
        ]
        _write_predictions(output / f"{split_name}_predictions.jsonl", examples, predictions)
        results[split_name] = {
            "model": metrics,
            "majority_baseline": evaluate(
                list(examples["label"]), baseline_predictions, labels=class_labels,
            ),
            "majority_label": majority_label,
            "inference_seconds": inference_seconds,
            "batch_amortized_ms_per_sample": inference_seconds * 1000 / len(examples),
        }

    write_json(output / "summary.json", {
        "run_id": run_id,
        "dataset_adapter": config["dataset"],
        "model_adapter": config["model"],
        "training": config["training"],
        "evaluation": config["evaluation"],
        "code": code_identity(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: version(package)
                for package in ("datasets", "scikit-learn", "numpy", "joblib", "pyarrow")
            },
        },
        "fit_seconds": fit_seconds,
        "total_seconds": perf_counter() - started,
        "results": results,
    })
    return output


def _write_manifest(path: Path, dataset: DatasetDict) -> None:
    """Write sample identity and provenance without copying article text to artifacts.

    Stable IDs and labels come from the normalized dataset contract. Source-split
    names and text hashes are included when available for later alignment checks.
    """
    with path.open("w", encoding="utf-8") as handle:
        for split_name, split in dataset.items():
            for row in split:
                record = {"split": split_name, "id": row["id"], "label": row["label"]}
                if "source_split" in row:
                    record["source_split"] = row["source_split"]
                if "text_hash" in row:
                    record["text_hash"] = row["text_hash"]
                handle.write(json.dumps(record) + "\n")


def _write_predictions(path: Path, examples: object, predictions: list[Prediction]) -> None:
    """Join predictions to gold labels in original row order using strict alignment.

    A count mismatch raises instead of silently writing predictions under the wrong
    sample IDs. Optional source provenance is retained when the adapter provides it.
    """
    with path.open("w", encoding="utf-8") as handle:
        for row, prediction in zip(examples, predictions, strict=True):
            record = {"id": row["id"], "gold": row["label"], **prediction}
            if "source_split" in row:
                record["source_split"] = row["source_split"]
            handle.write(json.dumps(record) + "\n")
