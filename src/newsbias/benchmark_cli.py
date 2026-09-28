"""Configuration-driven CLI for benchmarking any inference-capable adapter pair."""

import argparse
import json
from pathlib import Path
from typing import cast

from datasets import DatasetDict

from newsbias.adapters import load_adapter, load_environment_file
from newsbias.benchmark import run_model_benchmark
from newsbias.experiment import read_config
from newsbias.models.base import Model


def main() -> None:
    """Execute an inference-only model/dataset pair selected by a TOML file.

    Dataset and model implementations are loaded from their configured adapter
    paths. `--split`, `--limit`, and `--batch-size` override evaluation settings;
    `--env-file` optionally loads local credentials for adapters that need them.
    Configurations requesting training are rejected rather than silently skipping it.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark configured dataset and model adapters",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="TOML file selecting dataset and model adapters")
    parser.add_argument(
        "--split",
        help="Dataset split to evaluate (defaults to first configured split)",
    )
    parser.add_argument("--limit", type=int, help="Optional maximum number of examples")
    parser.add_argument("--batch-size", type=int,
                        help="Override the configured model batch size")
    parser.add_argument("--env-file", type=Path,
                        help="Optional local env file for adapters requiring credentials")
    parser.add_argument("--output-root", type=Path, default=Path("runs/benchmarks"))
    args = parser.parse_args()

    config = read_config(args.config)
    if args.env_file is not None:
        load_environment_file(args.env_file)
    if config["training"]["enabled"]:
        parser.error("Inference benchmark configuration must set training.enabled = false")
    source = load_adapter(config["dataset"])
    if not callable(getattr(source, "load", None)):
        parser.error("Configured dataset adapter must implement load()")
    dataset = source.load()
    if not isinstance(dataset, DatasetDict):
        parser.error("Configured dataset adapter load() must return DatasetDict")
    model_adapter = load_adapter(config["model"])
    if not callable(getattr(model_adapter, "predict", None)):
        parser.error("Configured model adapter must implement predict()")
    model = cast(Model, model_adapter)
    split = args.split or config["evaluation"]["splits"][0]
    batch_size = (
        args.batch_size if args.batch_size is not None else config["evaluation"]["batch_size"]
    )
    output = run_model_benchmark(
        model,
        dataset,
        model_identifier=config["model"]["adapter"],
        split=split,
        output_root=args.output_root,
        limit=args.limit,
        batch_size=batch_size,
        dataset_metadata=dict(config["dataset"]),
        model_metadata=dict(config["model"]),
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    print(f"Run artifacts: {output}")
    print(f"{split}: accuracy={metrics['accuracy']:.4f}, "
          f"macro-F1={metrics['macro_f1']:.4f}, "
          f"failures={metrics['failed_predictions']}, n={metrics['count']}")


if __name__ == "__main__":
    main()
