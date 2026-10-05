"""Self-contained sequence-classification evaluation and training runner for Kaggle.

Can be dispatched directly via:
newsbias-remote kaggle run --script src/newsbias/remote/scripts/run_transformer_kaggle.py \
    --name eval-sequence-classifier \
    --pip transformers datasets scikit-learn \
    --args "--task eval --model <checkpoint> --dataset <dataset-id> \
        --text-column text --label-column label"
"""

import argparse
import json
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("newsbias.kaggle_transformer")

def load_project_splits(
    dataset_name: str,
    revision: str | None,
    *,
    label_column: str,
    source_splits: Sequence[str] = ("train", "validation", "test"),
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    """Build deterministic project splits from caller-selected source partitions.

    `label_column` identifies the target vocabulary; `source_splits` controls which
    Hugging Face partitions are merged and in what order. The returned selections
    retain all source columns and use the same two-stage stratification as local runs.
    """
    from datasets import concatenate_datasets, load_dataset
    from sklearn.model_selection import train_test_split

    if not source_splits:
        raise ValueError("At least one source split must be selected")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value <= 0 or value >= 1 for value in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("Split fractions must be positive and sum to one")
    source = load_dataset(dataset_name, revision=revision)
    usable_splits = [s for s in source_splits if s in set(source)]
    if not usable_splits:
        usable_splits = list(source.keys())
    combined = concatenate_datasets([source[name] for name in usable_splits])
    if label_column not in combined.column_names:
        raise ValueError(f"Label column {label_column!r} is missing")
    labels = [str(value).strip().lower() for value in combined[label_column]]
    indexes = list(range(len(combined)))
    validation_count = round(len(indexes) * validation_fraction)
    test_count = round(len(indexes) * test_fraction)
    train_indexes, remaining_indexes = train_test_split(
        indexes,
        test_size=validation_count + test_count,
        random_state=seed,
        stratify=labels,
    )
    remaining_labels = [labels[index] for index in remaining_indexes]
    validation_indexes, test_indexes = train_test_split(
        remaining_indexes,
        train_size=validation_count,
        random_state=seed,
        stratify=remaining_labels,
    )
    return {
        "train": combined.select(train_indexes),
        "validation": combined.select(validation_indexes),
        "test": combined.select(test_indexes),
    }


def get_working_dir() -> Path:
    """Choose Kaggle's writable artifact directory, with a local-run fallback."""
    kw = Path("/kaggle/working")
    if kw.is_dir() and os.access(kw, os.W_OK):
        return kw
    return Path.cwd()


def run_evaluation(
    model_name: str,
    output_dir: Path,
    dataset_name: str,
    text_column: str,
    label_column: str,
    split: str = "test",
    dataset_revision: str | None = None,
    model_revision: str | None = None,
    split_seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    source_splits: Sequence[str] = ("train", "validation", "test"),
    id_column: str | None = None,
    source_column: str | None = None,
    max_length: int = 512,
    stride: int = 64,
    chunk_batch_size: int = 16,
    limit: int | None = None,
    label_map: str | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on a caller-configured task and deterministic split.

    Dataset/model revisions, text/label columns, source partitions, split seed, and
    ratios are explicit inputs. Class names come from checkpoint `id2label` or the
    optional mapping override and must match the selected data labels. Sliding text
    windows are forwarded in bounded batches to limit GPU memory; aligned prediction
    records and aggregate metrics are written to `output_dir`.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_desc = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("Using compute device: %s (%s)", device, device_desc)

    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=model_revision,
    ).to(device)
    model.eval()

    if label_map:
        parsed_map = {}
        for pair in label_map.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                parsed_map[int(k.strip())] = v.strip().lower()
        active_label_map = parsed_map
    else:
        active_label_map = {
            int(k): str(v).strip().lower() for k, v in model.config.id2label.items()
        }
    logger.info("Model label mapping: %s", active_label_map)

    logger.info(
        "Loading pinned dataset %s@%s and creating stratified seed-%d splits",
        dataset_name,
        dataset_revision,
        split_seed,
    )
    splits = load_project_splits(
        dataset_name,
        dataset_revision or "main",
        label_column=label_column,
        source_splits=source_splits,
        seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}; choose train, validation, or test")
    ds = splits[split]
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    logger.info("Evaluating %d articles", len(ds))

    def _normalize_row_label(raw_val: Any) -> str:
        text_val = str(raw_val).strip().lower()
        try:
            int_val = int(text_val)
            if int_val in active_label_map:
                return active_label_map[int_val]
        except ValueError:
            pass
        return text_val

    dataset_labels = {_normalize_row_label(row[label_column]) for row in ds}
    target_labels = set(active_label_map.values())
    if target_labels != dataset_labels:
        raise ValueError(
            f"Checkpoint labels {sorted(target_labels)} do not match "
            f"dataset labels {sorted(dataset_labels)}; provide a checkpoint with a compatible head"
        )
    labels = sorted(target_labels)
    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    predictions_out: list[dict[str, Any]] = []

    for required_column in (text_column, label_column):
        if required_column not in ds.column_names:
            raise ValueError(f"Configured column {required_column!r} is missing from the dataset")

    started = time.perf_counter()
    for idx, row in enumerate(ds):
        text = str(row.get(text_column) or "")
        gt_label = _normalize_row_label(row.get(label_column))
        if gt_label not in target_labels:
            continue

        t0 = time.perf_counter()
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            stride=stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        chunk_probability_sum = None
        chunk_count = 0
        # Keep GPU memory bounded for unusually long full-article input.
        for chunk_start in range(0, input_ids.shape[0], chunk_batch_size):
            chunk_end = chunk_start + chunk_batch_size
            chunk_ids = input_ids[chunk_start:chunk_end].to(device)
            chunk_mask = (
                attention_mask[chunk_start:chunk_end].to(device)
                if attention_mask is not None else None
            )
            with torch.inference_mode():
                out = model(input_ids=chunk_ids, attention_mask=chunk_mask)
                probabilities = torch.softmax(out.logits, dim=-1).sum(dim=0)
            chunk_probability_sum = (
                probabilities if chunk_probability_sum is None
                else chunk_probability_sum + probabilities
            )
            chunk_count += out.logits.shape[0]
        mean_probs = (chunk_probability_sum / chunk_count).cpu().numpy()

        pred_idx = int(np.argmax(mean_probs))
        pred_label = active_label_map.get(pred_idx, "unknown")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        y_true.append(gt_label)
        y_pred.append(pred_label)
        latencies.append(latency_ms)

        pred_record = {
            "index": idx,
            "id": row.get(id_column, f"sample-{idx}") if id_column else f"sample-{idx}",
            "ground_truth": gt_label,
            "predicted": pred_label,
            "scores": {active_label_map[i]: float(mean_probs[i]) for i in range(len(mean_probs))},
            "latency_ms": round(latency_ms, 2),
        }
        if source_column and source_column in row:
            pred_record["source"] = row[source_column]
        predictions_out.append(pred_record)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(ds):
            pct = ((idx + 1) / len(ds)) * 100
            logger.info("Processed %d / %d articles (%.1f%%)", idx + 1, len(ds), pct)

    duration = time.perf_counter() - started
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    accuracy = correct / max(1, total)

    cm = {true_l: {pred_l: 0 for pred_l in labels} for true_l in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    per_class = {}
    f1_list = []
    for label in labels:
        tp = cm[label][label]
        fp = sum(cm[other][label] for other in labels if other != label)
        fn = sum(cm[label][other] for other in labels if other != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(cm[label].values()),
        }

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    summary = {
        "model_name": model_name,
        "model_revision": model_revision,
        "dataset": dataset_name,
        "dataset_revision": dataset_revision,
        "split": split,
        "split_seed": split_seed,
        "split_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "text_column": text_column,
        "label_column": label_column,
        "labels": labels,
        "total_evaluated": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "duration_seconds": round(duration, 2),
        "throughput_articles_per_sec": round(total / max(0.001, duration), 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
    }

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in predictions_out:
            f.write(json.dumps(p) + "\n")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluation Complete! Accuracy: %.2f%%, Macro F1: %.4f", accuracy * 100, macro_f1)
    return summary


def run_training(
    base_model: str,
    output_dir: Path,
    dataset_name: str,
    text_column: str,
    label_column: str,
    dataset_revision: str | None = None,
    model_revision: str | None = None,
    split_seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    source_splits: Sequence[str] = ("train", "validation", "test"),
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 512,
) -> dict[str, Any]:
    """Fine-tune a sequence-classification head for caller-selected text and labels.

    The label vocabulary is derived from generated training/validation partitions.
    Tokenization, optimizer steps, checkpoint selection, and training history all use
    the supplied dataset columns and model checkpoint.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    splits = load_project_splits(
        dataset_name,
        dataset_revision or "main",
        label_column=label_column,
        source_splits=source_splits,
        seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    train_raw = splits["train"]
    val_raw = splits["validation"]
    observed_labels = {
        str(label).strip().lower()
        for label in list(train_raw[label_column]) + list(val_raw[label_column])
    }
    label_to_id = {label: index for index, label in enumerate(sorted(observed_labels))}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        revision=model_revision,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    ).to(device)

    logger.info("Loaded %d training and %d validation examples", len(train_raw), len(val_raw))
    for column in (text_column, label_column):
        if column not in train_raw.column_names:
            raise ValueError(f"Configured column {column!r} is missing from the dataset")

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        """Turn a text/label batch into padded model inputs and target class IDs."""
        enc = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        labels = [label_to_id[str(label).strip().lower()] for label in batch[label_column]]
        enc["labels"] = labels
        return enc

    train_ds = train_raw.map(tokenize_batch, batched=True, remove_columns=train_raw.column_names)
    val_ds = val_raw.map(tokenize_batch, batched=True, remove_columns=val_raw.column_names)

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            if (step + 1) % 50 == 0:
                logger.info(
                    "Epoch %d/%d | Step %d/%d | Loss: %.4f",
                    epoch,
                    epochs,
                    step + 1,
                    len(train_loader),
                    loss.item(),
                )

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.inference_mode():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                out = model(input_ids=ids, attention_mask=mask)
                preds = torch.argmax(out.logits, dim=-1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(batch["labels"].tolist())

        correct = sum(1 for p, t in zip(val_preds, val_targets, strict=True) if p == t)
        val_acc = correct / max(1, len(val_targets))

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_accuracy": round(val_acc, 4),
        }
        history.append(epoch_record)
        logger.info("Epoch %d Finished — Val Accuracy: %.4f", epoch, val_acc)

        if val_acc > best_val_f1:
            best_val_f1 = val_acc
            save_path = output_dir / "best_model"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info("Saved best model checkpoint to %s", save_path)

    summary = {
        "base_model": base_model,
        "model_revision": model_revision,
        "dataset": dataset_name,
        "dataset_revision": dataset_revision,
        "split_seed": split_seed,
        "split_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "text_column": text_column,
        "label_column": label_column,
        "labels": sorted(label_to_id),
        "epochs": epochs,
        "learning_rate": lr,
        "best_val_accuracy": best_val_f1,
        "history": history,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    """Parse generic model, dataset, column, split, and training options for Kaggle."""
    parser = argparse.ArgumentParser(
        description="Evaluate or fine-tune a configured Hugging Face sequence classifier",
    )
    parser.add_argument(
        "--task", choices=("eval", "train"), default="eval", help="Task to execute"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face sequence-classification checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hugging Face dataset repository",
    )
    parser.add_argument(
        "--text-column", required=True, help="Dataset column containing input text",
    )
    parser.add_argument(
        "--label-column", required=True, help="Dataset column containing target labels",
    )
    parser.add_argument("--id-column", help="Optional stable sample ID column")
    parser.add_argument("--source-column", help="Optional source metadata column")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate")
    parser.add_argument(
        "--dataset-revision", type=str,
        help="Pinned dataset commit or branch/revision",
    )
    parser.add_argument(
        "--model-revision", type=str,
        help="Pinned model commit or branch/revision",
    )
    parser.add_argument("--source-splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional sample limit for quick test"
    )
    parser.add_argument(
        "--label-map", type=str, default=None,
        help="Mapping of integer head output to class names (e.g. 0:left,1:center,2:right)",
    )
    args = parser.parse_args()

    output_dir = get_working_dir()
    logger.info("Output directory set to: %s", output_dir)

    if args.task == "eval":
        run_evaluation(
            model_name=args.model,
            output_dir=output_dir,
            dataset_name=args.dataset,
            text_column=args.text_column,
            label_column=args.label_column,
            split=args.split,
            dataset_revision=args.dataset_revision,
            model_revision=args.model_revision,
            split_seed=args.split_seed,
            source_splits=args.source_splits,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            id_column=args.id_column,
            source_column=args.source_column,
            limit=args.limit,
            label_map=args.label_map,
        )
    else:
        run_training(
            base_model=args.model,
            output_dir=output_dir,
            dataset_name=args.dataset,
            text_column=args.text_column,
            label_column=args.label_column,
            dataset_revision=args.dataset_revision,
            split_seed=args.split_seed,
            source_splits=args.source_splits,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
