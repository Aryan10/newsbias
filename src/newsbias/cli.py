"""NewsBias CLI for training baselines, evaluating models, running benchmarks, and serving the web app.

Supports:
- Simple ML training and evaluation (train, eval)
- Interactive web server and tribunal UI (serve)
- Unified model benchmarking and remote Kaggle dispatch (run, compare, status, download)
- Article analysis and dataset consolidation (analyze, merge-bias)
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import sys
from typing import Any, cast
from uuid import uuid4

from datasets import Dataset, DatasetDict

from newsbias.adapters import load_adapter, load_environment_file
from newsbias.benchmark import run_model_benchmark
from newsbias.cache import find_cached_benchmark, list_benchmark_runs
from newsbias.evaluation import evaluate
from newsbias.experiment import ExperimentConfig, read_config, run_experiment
from newsbias.models.base import Model, Prediction
from newsbias.models.tfidf import TfidfModel
from newsbias.report import print_benchmark_report

MODEL_REGISTRY: dict[str, str] = {
    "tfidf": "tfidf",
    "gpt-5-mini": "gpt-5-mini",
    "gpt5mini": "gpt-5-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "deberta-large": "matous-volf/political-leaning-deberta-large",
    "political-leaning-deberta-large": "matous-volf/political-leaning-deberta-large",
    "politics": "matous-volf/political-leaning-politics",
    "political-leaning-politics": "matous-volf/political-leaning-politics",
    "bert": "bucketresearch/politicalBiasBERT",
    "politicalbiasbert": "bucketresearch/politicalBiasBERT",
    "roberta-babe": "mediabiasgroup/da-roberta-babe-ft",
    "da-roberta-babe": "mediabiasgroup/da-roberta-babe-ft",
    "babe-roberta": "mediabiasgroup/da-roberta-babe-ft",
}


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_task_dataset(task: str, dataset_name: str | None = None) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Load train/validation/test splits and canonical labels for a given task."""
    if task == "political":
        from newsbias.datasets import create_political_splits, load_allsides
        records = load_allsides()
        splits = create_political_splits(records, seed=42)
        splits["val"] = splits.get("validation", [])
        labels = ["left", "center", "right"]
        return splits, labels
    else:  # sentence-level linguistic bias
        from newsbias.datasets import create_bias_splits, load_babe, load_basil, load_mbic
        ds = (dataset_name or "babe").lower()
        if ds == "mbic":
            records = load_mbic()
        elif ds == "basil":
            records = load_basil()
        else:
            records = load_babe()
        splits = create_bias_splits(records, seed=42)
        splits["val"] = splits.get("validation", [])
        labels = ["non-biased", "biased"]
        return splits, labels


# ── Subcommand Handlers ────────────────────────────────────────────────────────
def handle_train(args: argparse.Namespace) -> None:
    """Train a machine learning baseline (e.g. TF-IDF) on training data."""
    task = "political" if args.political else ("bias" if args.bias else "political")
    model_name = args.model.lower()

    if model_name != "tfidf":
        sys.exit(f"Training is currently supported for 'tfidf' baseline (got: {args.model}). Pretrained deep learning models can be evaluated via 'cli.py eval'.")

    print("=" * 80)
    print(f"TRAINING BASELINE MODEL: {model_name.upper()} (Task: {task})")
    print("=" * 80)

    splits, labels = load_task_dataset(task, dataset_name=args.dataset)
    train_records = splits["train"]
    val_records = splits["val"]

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    print(f"Loaded {len(train_records):,} training samples, {len(val_records):,} validation samples.")
    print("Fitting TF-IDF vectorizer and Logistic Regression classifier...")

    model = TfidfModel(min_df=2, max_features=100_000, ngram_max=2)
    model.fit(train_ds)

    # Evaluate on validation split
    val_preds = model.predict(val_ds)
    val_gold = [r["label"] for r in val_records]
    metrics = evaluate(val_gold, val_preds, labels=labels)

    # Save artifact
    output_dir = Path(args.output_dir) if args.output_dir != "runs/models" else (_REPO_ROOT / "runs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"tfidf_{task}.joblib"
    model.save(model_path)

    print("-" * 80)
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation Macro-F1: {metrics['macro_f1']:.4f}")
    print("Per-class performance:")
    for lbl, m in metrics["per_class"].items():
        print(f"  {lbl:<12} precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f} (support: {m['support']})")
    print(f"Model saved to:      {model_path}")
    print("=" * 80)


def handle_eval(args: argparse.Namespace) -> None:
    """Evaluate a model on benchmark splits."""
    model_alias = args.model.lower()
    canonical_model = MODEL_REGISTRY.get(model_alias, args.model)

    # Infer task from flags or model name
    if args.bias or "babe" in model_alias or "roberta" in model_alias:
        task = "bias"
    else:
        task = "political"

    split_name = args.split
    splits, labels = load_task_dataset(task, dataset_name=args.dataset)
    split_key = "val" if split_name in ("val", "validation") else split_name
    records = splits.get(split_key) or splits.get("test") or []

    if args.limit and args.limit > 0:
        records = records[:args.limit]

    if not records:
        sys.exit(f"Error: Split '{split_name}' contains no data.")

    print("=" * 80)
    print(f"EVALUATING MODEL: {canonical_model} (Task: {task}, Split: {split_name})")
    print("=" * 80)
    print(f"Evaluating {len(records):,} samples...")

    gold_labels = [r["label"] for r in records]
    predictions: list[Prediction] = []

    if canonical_model == "tfidf":
        candidates = [
            Path(args.model_path) if args.model_path else None,
            _REPO_ROOT / f"runs/models/tfidf_{task}.joblib",
            Path(f"runs/models/tfidf_{task}.joblib"),
        ]
        model_path = next((p for p in candidates if p and p.exists()), None)
        if model_path:
            print(f"Loading trained weights from: {model_path}")
            model = TfidfModel.load(model_path)
        else:
            print("No saved checkpoint found. Fitting on training split on the fly...")
            train_ds = Dataset.from_list(splits["train"])
            model = TfidfModel(min_df=2, max_features=100_000)
            model.fit(train_ds)
        predictions = model.predict(Dataset.from_list(records))

    elif canonical_model.startswith("gpt-"):
        from newsbias.models.llm import get_openai_client
        client = get_openai_client(model=canonical_model)
        if task == "political":
            from newsbias.prompts import JinjaPrompt
            prompt = JinjaPrompt.load("zero_shot_bias")
            for i, r in enumerate(records):
                rendered = prompt.render(title=r.get("outlet", ""), content=r["text"])
                res = rendered.complete_and_parse_json(client=client)
                pred_label = (res.get("political_leaning") or {}).get("label", "center").lower()
                predictions.append(Prediction(label=pred_label, probabilities={}))
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(records)} samples...")
        else:
            from newsbias.models.llm import classify_sentences_llm
            batch_size = 20
            for i in range(0, len(records), batch_size):
                chunk = records[i:i + batch_size]
                res = classify_sentences_llm([c["text"] for c in chunk], model=canonical_model, client=client)
                for item in res:
                    predictions.append(Prediction(label=item["label"], probabilities={}))
                print(f"  Processed {min(i + batch_size, len(records))}/{len(records)} samples...")

    elif task == "political":
        from newsbias.models.political import predict_political_leaning
        raw_preds = predict_political_leaning(records, model_name=canonical_model)
        for p in raw_preds:
            predictions.append(Prediction(label=p["label"].lower(), probabilities=p.get("probabilities", {})))

    else:  # sentence bias
        from newsbias.models.bias import predict_sentence_bias
        raw_preds = predict_sentence_bias([r["text"] for r in records], model_name=canonical_model)
        for p in raw_preds:
            predictions.append(Prediction(label=p["label"].lower(), probabilities={}))

    # Compute standardized classification metrics
    metrics = evaluate(gold_labels, predictions, labels=labels)

    print("-" * 80)
    print(f"Evaluated Samples: {metrics['count']:,}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Macro-F1:          {metrics['macro_f1']:.4f}")
    print("\nClass Breakdown:")
    for lbl, m in metrics["per_class"].items():
        print(f"  {lbl:<12} precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f} (support: {m['support']})")
    print("\nConfusion Matrix (rows: gold, cols: pred):")
    print(" " * 12 + "  ".join(f"{lbl[:6]:>6}" for lbl in labels))
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"  {labels[i]:<10}" + "  ".join(f"{val:>6}" for val in row))
    print("=" * 80)


def handle_serve(args: argparse.Namespace) -> None:
    """Launch the FastAPI server and Vite React standalone interface."""
    import uvicorn
    print("=" * 80)
    print(f"Launching NewsBias Web App on http://{args.host}:{args.port}")
    print("=" * 80)
    uvicorn.run("newsbias.web.app:app", host=args.host, port=args.port, reload=args.reload)


# ── Parser Definition ──────────────────────────────────────────────────────────
DEBERTA_LARGE_CONFIG: dict[str, Any] = {
    "adapter": "newsbias.models.huggingface:HuggingFaceSequenceClassifier",
    "parameters": {
        "model_name": "matous-volf/political-leaning-deberta-large",
        "id2label": {0: "left", 1: "center", 2: "right"},
        "max_length": 512,
        "stride": 64,
        "max_chunks": 16,
    },
    "canonical_name": "matous-volf/political-leaning-deberta-large",
    "label_map": "0:left,1:center,2:right",
}

POLITICS_CONFIG: dict[str, Any] = {
    "adapter": "newsbias.models.huggingface:HuggingFaceSequenceClassifier",
    "parameters": {
        "model_name": "matous-volf/political-leaning-politics",
        "id2label": {0: "left", 1: "center", 2: "right"},
        "max_length": 512,
        "stride": 64,
        "max_chunks": 16,
    },
    "canonical_name": "matous-volf/political-leaning-politics",
    "label_map": "0:left,1:center,2:right",
}

POLITICAL_BIAS_BERT_CONFIG: dict[str, Any] = {
    "adapter": "newsbias.models.huggingface:HuggingFaceSequenceClassifier",
    "parameters": {
        "model_name": "bucketresearch/politicalBiasBERT",
        "revision": "5e5aa509b3f0fe76a6cf102f56490bd82ca04e2f",
        "max_length": 512,
        "stride": 64,
        "max_chunks": 16,
    },
    "canonical_name": "bucketresearch/politicalBiasBERT",
    "label_map": None,
}

MODEL_ALIASES: dict[str, dict[str, Any]] = {
    "political-leaning-deberta-large": DEBERTA_LARGE_CONFIG,
    "political-learning-deberta-large": DEBERTA_LARGE_CONFIG,
    "deberta-large": DEBERTA_LARGE_CONFIG,
    "political-leaning-politics": POLITICS_CONFIG,
    "political-learning-politics": POLITICS_CONFIG,
    "politics": POLITICS_CONFIG,
    "political-bias-bert": POLITICAL_BIAS_BERT_CONFIG,
    "politicalbiasbert": POLITICAL_BIAS_BERT_CONFIG,
    "bert": POLITICAL_BIAS_BERT_CONFIG,
    "gpt-5-mini": {
        "adapter": "newsbias.models.openai:OpenAIChatClassifier",
        "parameters": {
            "model_name": "gpt-5-mini",
            "labels": ["left", "center", "right"],
            "system_prompt": (
                "Classify the political bias of the news text into one of: "
                "left, center, right. Return JSON with predicted_label and scores."
            ),
        },
        "canonical_name": "gpt-5-mini",
        "label_map": None,
    },
    "tfidf": {
        "adapter": "newsbias.models.tfidf:TfidfModel",
        "parameters": {
            "min_df": 2,
        },
        "canonical_name": "tfidf",
        "label_map": None,
    },
}

DATASET_ALIASES: dict[str, dict[str, Any]] = {
    "allsides": {
        "adapter": "newsbias.datasets.all_sides:AllSidesDataset",
        "parameters": {
            "name": "upasanachatterjee/AllSides-random-split",
            "revision": "4d33d17c5591ad5d4e37ffd6238d7ead8590a70e",
            "text_column": "content",
            "processed_cache_dir": "data/processed",
            "download_cache_dir": "data/cache",
            "split_seed": 42,
            "train_fraction": 0.7,
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
        },
        "canonical_name": "AllSides",
        "dataset_name": "upasanachatterjee/AllSides-random-split",
        "dataset_revision": "4d33d17c5591ad5d4e37ffd6238d7ead8590a70e",
        "text_column": "content",
        "label_column": "label",
    },
    "babe": {
        "adapter": "newsbias.datasets.babe:BABEDataset",
        "parameters": {
            "csv_path": "data/raw/babe/final_labels_SG2.csv",
            "processed_cache_dir": "data/processed",
            "download_cache_dir": "data/downloads",
            "split_seed": 42,
            "train_fraction": 0.7,
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
        },
        "canonical_name": "BABE-SG2",
        "dataset_name": "babe",
        "dataset_revision": None,
        "text_column": "text",
        "label_column": "label",
    },
}


def resolve_model(
    name: str | None, config_model: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any], str | None]:
    """Resolve model alias or Hugging Face ID into adapter config and label map."""
    if config_model:
        model_name = (
            config_model.get("parameters", {}).get("model_name")
            or config_model.get("adapter", "")
        )
        return model_name, config_model, None

    if not name:
        raise ValueError("Must specify --model or provide a --config file")

    key = name.strip().lower()
    if key in MODEL_ALIASES:
        entry = MODEL_ALIASES[key]
        return (
            entry["canonical_name"],
            {
                "adapter": entry["adapter"],
                "parameters": dict(entry["parameters"]),
            },
            entry["label_map"],
        )

    if "/" in name:
        return (
            name,
            {
                "adapter": "newsbias.models.huggingface:HuggingFaceSequenceClassifier",
                "parameters": {
                    "model_name": name,
                    "max_length": 512,
                    "stride": 64,
                    "max_chunks": 16,
                },
            },
            None,
        )

    raise ValueError(
        f"Unknown model alias {name!r}. Supported: {sorted(MODEL_ALIASES.keys())}"
    )


def resolve_dataset(
    name: str | None, config_dataset: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Resolve dataset alias or configuration into adapter config and metadata."""
    if config_dataset:
        dataset_name = (
            config_dataset.get("parameters", {}).get("name")
            or config_dataset.get("adapter", "")
        )
        return dataset_name, config_dataset, {}

    target = (name or "allsides").strip().lower()
    if target in DATASET_ALIASES:
        entry = DATASET_ALIASES[target]
        return (
            entry["canonical_name"],
            {
                "adapter": entry["adapter"],
                "parameters": dict(entry["parameters"]),
            },
            {
                "dataset_name": entry["dataset_name"],
                "dataset_revision": entry["dataset_revision"],
                "text_column": entry["text_column"],
                "label_column": entry["label_column"],
            },
        )

    raise ValueError(
        f"Unknown dataset alias {name!r}. Supported: {sorted(DATASET_ALIASES.keys())}"
    )


def execute_local_run(
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    *,
    canonical_model_name: str,
    split: str,
    output_root: Path,
    limit: int | None = None,
    batch_size: int = 32,
    dataset_override: DatasetDict | None = None,
) -> Path:
    """Execute evaluation locally, saving artifacts and returning the output directory."""
    if dataset_override is not None:
        dataset = dataset_override
    else:
        dataset_source = load_adapter(cast(Any, dataset_config))
        if not callable(getattr(dataset_source, "load", None)):
            raise TypeError("Dataset adapter must implement load()")
        dataset = cast(DatasetDict, dataset_source.load())

    model_adapter = load_adapter(cast(Any, model_config))
    if not callable(getattr(model_adapter, "predict", None)):
        raise TypeError("Model adapter must implement predict()")
    model = cast(Model, model_adapter)

    # Train model if it requires fitting (e.g. TF-IDF baseline)
    if (
        hasattr(model, "fit")
        and not getattr(model, "_pipeline", None)
        and "train" in dataset
    ):
        model.fit(dataset["train"])

    out_dir = run_model_benchmark(
        model,
        dataset,
        model_identifier=canonical_model_name,
        split=split,
        output_root=output_root / "benchmarks",
        limit=limit,
        batch_size=batch_size,
        dataset_metadata=dataset_config,
        model_metadata=model_config,
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    print_benchmark_report(
        summary, cached=False, runner="local", artifacts_path=str(out_dir)
    )
    return out_dir


def execute_kaggle_run(
    canonical_model_name: str,
    dataset_meta: dict[str, Any],
    *,
    split: str,
    label_map: str | None,
    output_root: Path,
    limit: int | None = None,
    no_wait: bool = False,
    timeout: int = 3600,
) -> Path:
    """Stage and dispatch a remote Kaggle GPU evaluation job."""
    from newsbias.remote.base import RemoteJobConfig, RemoteJobStatus
    from newsbias.remote.kaggle import KaggleRunner

    runner = KaggleRunner()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output_dir = output_root / "benchmarks" / run_id

    script_path = (
        Path(__file__).resolve().parent / "remote" / "scripts" / "run_transformer_kaggle.py"
    )
    if not script_path.is_file():
        raise FileNotFoundError(f"Kaggle entrypoint script not found: {script_path}")

    args_list = [
        "--task", "eval",
        "--model", canonical_model_name,
        "--dataset", str(dataset_meta.get("dataset_name", "upasanachatterjee/AllSides-random-split")),
        "--text-column", str(dataset_meta.get("text_column", "content")),
        "--label-column", str(dataset_meta.get("label_column", "label")),
        "--split", split,
    ]
    if dataset_meta.get("dataset_revision"):
        args_list.extend(["--dataset-revision", str(dataset_meta["dataset_revision"])])
    if label_map:
        args_list.extend(["--label-map", label_map])
    if limit is not None:
        args_list.extend(["--limit", str(limit)])

    job_args = " ".join(shlex.quote(a) for a in args_list)
    model_slug = Path(canonical_model_name).stem.replace("_", "-")
    job_name = f"eval-{model_slug}-{split}"

    job_config = RemoteJobConfig(
        entrypoint=script_path,
        name=job_name,
        args=job_args,
        pip_packages=["transformers", "datasets", "scikit-learn", "torch", "accelerate"],
        gpu=True,
    )

    if no_wait:
        if hasattr(runner, "submit"):
            kernel_id = runner.submit(job_config)
        else:
            bundle = runner.stage_bundle(job_config)
            kernel_id = runner._push_kernel(bundle)
        print(f"Submitted Kaggle kernel: {kernel_id}")
        print(f"Check status with: newsbias status --id {kernel_id}")
        print(f"Download results with: newsbias download --id {kernel_id} --output-dir {output_dir}")
        return output_dir

    try:
        result = runner.run(job_config, output_dir=output_dir, timeout=timeout)
    except TypeError:
        result = runner.run(job_config, wait=True, output_dir=output_dir)

    is_complete = result.status in (RemoteJobStatus.COMPLETE, RemoteJobStatus.COMPLETE)
    if is_complete and (output_dir / "summary.json").is_file():
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        print_benchmark_report(
            summary, cached=False, runner="kaggle", artifacts_path=str(output_dir)
        )
    else:
        print(f"Kaggle run completed with status: {result.status.value}")
        if result.error_message:
            print(f"Error: {result.error_message}")
    return output_dir


def handle_run(args: argparse.Namespace) -> Path:
    """Handle the unified 'newsbias run' command with cache lookup and runner dispatch."""
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.env_file is not None:
        load_environment_file(args.env_file)

    config: dict[str, Any] | None = None
    if args.config:
        config = read_config(args.config)

    canonical_model, model_config, label_map = resolve_model(
        args.model, config.get("model") if config else None
    )
    dataset_name, dataset_config, dataset_meta = resolve_dataset(
        args.dataset, config.get("dataset") if config else None
    )
    split = args.split

    # Check cache first unless --force-rerun is passed
    if not args.force_rerun:
        cached = find_cached_benchmark(
            output_root,
            model_identifier=canonical_model,
            split=split,
            limit=args.limit,
        )
        if cached is not None:
            run_dir, summary = cached
            print(f"[CACHE HIT] Loaded existing run from: {run_dir}")
            print_benchmark_report(
                summary,
                cached=True,
                runner=args.runner,
                artifacts_path=str(run_dir),
            )
            return run_dir

    # Runner execution
    if args.runner == "kaggle":
        return execute_kaggle_run(
            canonical_model,
            dataset_meta,
            split=split,
            label_map=label_map,
            output_root=output_root,
            limit=args.limit,
            no_wait=args.no_wait,
        )
    return execute_local_run(
        model_config,
        dataset_config,
        canonical_model_name=canonical_model,
        split=split,
        output_root=output_root,
        limit=args.limit,
        batch_size=args.batch_size,
    )


def handle_compare(args: argparse.Namespace) -> None:
    """Print comparative table of all benchmark summaries for a given split."""
    runs = list_benchmark_runs(args.output_root)
    split_runs: dict[str, tuple[Path, dict[str, Any]]] = {}

    for run_dir, summary in runs:
        if summary.get("split") != args.split:
            continue
        model_name = (
            summary.get("model_name")
            or summary.get("model_metadata", {}).get("parameters", {}).get("model_name")
            or summary.get("model", {}).get("parameters", {}).get("model_name")
            or summary.get("model_identifier")
            or "Unknown"
        )
        if model_name not in split_runs:
            split_runs[model_name] = (run_dir, summary)

    if not split_runs:
        print(f"No benchmark runs found for split: {args.split}")
        return

    print("=" * 86)
    print(f"NEWSBIAS BENCHMARK COMPARISON (Split: {args.split})")
    print("=" * 86)
    print(f"{'Model':<38} {'Accuracy':<10} {'Macro-F1':<10} {'Samples':<10} {'Run ID'}")
    print("-" * 86)

    for model_name, (run_dir, summary) in sorted(split_runs.items()):
        metrics = summary.get("metrics") or {}
        acc = metrics.get("accuracy", summary.get("accuracy", 0.0))
        f1 = metrics.get("macro_f1", summary.get("macro_f1", 0.0))
        n = metrics.get("count", summary.get("total_evaluated", 0))
        print(f"{model_name:<38} {acc:<10.4f} {f1:<10.4f} {n:<10} {run_dir.name}")
    print("=" * 86)


def run_legacy_experiment(args: argparse.Namespace) -> None:
    """Execute configured legacy training and evaluation experiment."""
    config = read_config(args.config)
    if args.env_file is not None:
        load_environment_file(args.env_file)
    if args.split:
        if len(set(args.split)) != len(args.split):
            sys.exit("--split must not contain duplicates")
        config["evaluation"]["splits"] = args.split
    if args.batch_size is not None:
        config["evaluation"]["batch_size"] = args.batch_size
    output = run_experiment(cast(ExperimentConfig, config), output_root=args.output_root)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print(f"Run artifacts: {output}")
    for split, result in summary["results"].items():
        metrics = result["model"]
        print(
            f"{split}: accuracy={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}, n={metrics['count']}"
        )



def handle_analyze(args: argparse.Namespace) -> None:
    """Analyze an article and print interactive breakdown."""
    from newsbias.pipeline.analyzer import analyze_article

    text = args.text
    if args.file and args.file.is_file():
        text = args.file.read_text(encoding="utf-8")
    if not text:
        sys.exit("Error: Provide article text or --file")

    result = analyze_article(text, use_mock=args.mock)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    pol = result["political_leaning"]
    bias = result["bias_summary"]
    sents = result["sentences"]

    print("=" * 80)
    print("NEWSBIAS ARTICLE ANALYSIS")
    print("=" * 80)
    probs = pol.get("probabilities", {})
    l_p = probs.get("left", 0.0)
    c_p = probs.get("center", 0.0)
    r_p = probs.get("right", 0.0)
    print(
        f"Political Leaning: {pol['label'].upper()} "
        f"(Left: {l_p:.2f}, Center: {c_p:.2f}, Right: {r_p:.2f})"
    )
    print(
        f"Bias Density:      {bias['level'].upper()} "
        f"({bias['biased_sentences_count']} of {bias['total_sentences']} sentences biased, "
        f"ratio: {bias['biased_ratio']:.1%})"
    )
    print("-" * 80)
    print("Sentence Breakdown:")
    for s in sents:
        tag = "[BIASED]" if s["is_biased"] else "[NEUTRAL]"
        print(f"  {tag:<9} (score: {s['score']:.2f}) {s['text']}")
    print("=" * 80)


def handle_merge_bias(args: argparse.Namespace) -> None:
    """Merge sentence bias datasets and save to JSON."""
    from newsbias.datasets.bias import (
        load_babe,
        load_basil,
        load_mbic,
        merge_bias_datasets,
    )

    babe_data = load_babe()
    mbic_data = load_mbic()
    basil_data = load_basil()
    merged = merge_bias_datasets([babe_data, mbic_data, basil_data])

    print(f"BABE records:   {len(babe_data)}")
    print(f"MBIC records:   {len(mbic_data)}")
    print(f"BASIL records:  {len(basil_data)}")
    print(f"Merged & Dedup: {len(merged)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Saved merged dataset to: {args.output}")

def build_parser() -> argparse.ArgumentParser:
    """Build unified CLI parser supporting both streamlined and research workflows."""
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="NewsBias: Model training, benchmark evaluation, and web app runner",
    )
    # Legacy compatibility flags
    parser.add_argument("--config", type=Path, help="Legacy TOML experiment config")
    parser.add_argument("--split", nargs="+", help="Override configured evaluation splits")
    parser.add_argument("--batch-size", type=int, help="Override configured inference batch size")
    parser.add_argument("--env-file", type=Path, help="Local env file for credentials")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))

    subparsers = parser.add_subparsers(dest="subcommand")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train baseline model (e.g. TF-IDF)")
    train_parser.add_argument("model", nargs="?", default="tfidf", help="Model to train (default: tfidf)")
    train_parser.add_argument("--political", action="store_true", help="Train for macro political stance (AllSides)")
    train_parser.add_argument("--bias", action="store_true", help="Train for sentence-level bias (BABE)")
    train_parser.add_argument("--dataset", type=str, default=None, help="Explicit dataset (allsides, babe, mbic, basil)")
    train_parser.add_argument("--output-dir", type=str, default="runs/models", help="Directory to save trained model")

    # Command: eval / evaluate
    eval_parser = subparsers.add_parser("eval", aliases=["evaluate"], help="Evaluate model on benchmark splits")
    eval_parser.add_argument(
        "model",
        nargs="?",
        default="tfidf",
        help="Model alias (tfidf, gpt-5-mini, deberta-large, roberta-babe) or HuggingFace ID",
    )
    eval_parser.add_argument("--political", action="store_true", help="Evaluate political stance (AllSides)")
    eval_parser.add_argument("--bias", action="store_true", help="Evaluate sentence bias (BABE)")
    eval_parser.add_argument("--dataset", type=str, default=None, help="Explicit dataset name")
    eval_parser.add_argument("--split", type=str, default="test", choices=["test", "val", "validation", "train"])
    eval_parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for fast smoke testing")
    eval_parser.add_argument("--model-path", type=str, default=None, help="Path to saved .joblib model file")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI server and React web UI")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Command: run (benchmarking runner)
    run_parser = subparsers.add_parser("run", help="Run model evaluation locally or on Kaggle")
    run_parser.add_argument(
        "--runner", choices=["local", "kaggle"], default="local", help="Execution environment"
    )
    run_parser.add_argument(
        "--model", type=str,
        help="Model alias (deberta-large, politics, tfidf, gpt-5-mini) or Hugging Face ID",
    )
    run_parser.add_argument(
        "--dataset", type=str, default="allsides", help="Dataset alias (allsides, babe)"
    )
    run_parser.add_argument(
        "--split", type=str, default="test", help="Dataset split (test, validation, train)"
    )
    run_parser.add_argument(
        "--limit", type=int, default=None, help="Optional sample limit for quick smoke test"
    )
    run_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for local inference"
    )
    run_parser.add_argument(
        "--force-rerun", action="store_true", help="Bypass cached results and re-run evaluation"
    )
    run_parser.add_argument(
        "--no-wait", action="store_true", help="Submit Kaggle job without waiting for completion"
    )
    run_parser.add_argument(
        "--env-file", type=Path, help="Local .env file for credentials"
    )
    run_parser.add_argument(
        "--config", type=Path, help="Optional TOML configuration file"
    )
    run_parser.add_argument(
        "--output-root", type=Path, default=Path("runs"), help="Root directory for run artifacts"
    )

    # Command: compare
    compare_parser = subparsers.add_parser(
        "compare", help="Compare benchmark results across models"
    )
    compare_parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to compare"
    )
    compare_parser.add_argument(
        "--output-root", type=Path, default=Path("runs"), help="Root directory for run artifacts"
    )

    # Command: status
    status_parser = subparsers.add_parser("status", help="Check status of a remote Kaggle kernel")
    status_parser.add_argument("--id", type=str, required=True, help="Kernel ID (username/slug)")

    # Command: download
    dl_parser = subparsers.add_parser(
        "download", help="Download output artifacts of a completed Kaggle kernel"
    )
    dl_parser.add_argument("--id", type=str, required=True, help="Kernel ID (username/slug)")
    dl_parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to save artifacts"
    )

    # Command: analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze article text for sentence bias and political leaning"
    )
    analyze_parser.add_argument("text", nargs="?", default="", help="Raw article text to analyze")
    analyze_parser.add_argument("--file", type=Path, help="File containing article text")
    analyze_parser.add_argument("--json", action="store_true", help="Output raw JSON")
    analyze_parser.add_argument(
        "--mock", action="store_true", help="Use offline deterministic mock"
    )

    # Command: merge-bias
    merge_parser = subparsers.add_parser(
        "merge-bias", help="Merge and deduplicate sentence bias datasets"
    )
    merge_parser.add_argument(
        "--output", type=Path, default=Path("data/processed/merged_bias.json")
    )

    return parser


def main() -> None:
    """CLI entrypoint dispatching train, eval, serve, and benchmark commands."""
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "train":
        handle_train(args)
    elif args.subcommand in ("eval", "evaluate"):
        handle_eval(args)
    elif args.subcommand == "serve":
        handle_serve(args)
    elif args.subcommand == "run":
        handle_run(args)
    elif args.subcommand == "compare":
        handle_compare(args)
    elif args.subcommand == "status":
        from newsbias.remote.kaggle import KaggleRunner
        status = KaggleRunner().status(args.id)
        print(f"Kernel {args.id} status: {status.value}")
    elif args.subcommand == "download":
        from newsbias.remote.kaggle import KaggleRunner
        KaggleRunner().download(args.id, args.output_dir)
        print(f"Downloaded artifacts from {args.id} to {args.output_dir}")
    elif args.subcommand == "analyze":
        handle_analyze(args)
    elif args.subcommand == "merge-bias":
        handle_merge_bias(args)
    elif args.config is not None:
        run_legacy_experiment(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
