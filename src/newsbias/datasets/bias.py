"""Sentence-level bias dataset loaders, harmonization, deduplication, and splits.

Handles BABE (gold), MBIC (gold), BASIL (gold lexical), and Anno-lexical (weak pretrain).
Flat functional design returning standard Python dictionaries.
"""

import random
import re
from hashlib import sha256
from pathlib import Path

DATASET_PRIORITY = {"babe": 1, "mbic": 2, "basil": 3, "anno_lexical": 4}


def hash_text(text):
    """Normalize whitespace and punctuation to compute a deterministic text hash."""
    clean = re.sub(r"[^\w\s]", "", str(text or "").lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return sha256(clean.encode("utf-8")).hexdigest()


def normalize_bias_label(raw_label, threshold=0.5):
    """Harmonize string/integer/float labels into standard binary ('biased' / 'non-biased')."""
    text = str(raw_label).strip().lower()
    try:
        score = float(text)
        if score >= threshold:
            return "biased", 1
        return "non-biased", 0
    except ValueError:
        pass
    if text in ("1", "biased", "bias", "true", "yes"):
        return "biased", 1
    return "non-biased", 0


def _extract_text(row):
    """Extract raw sentence text across diverse dataset schema fieldnames."""
    return (row.get("text") or row.get("\ufefftext") or row.get("sentence") or "").strip()


def load_babe(source=None):
    """Load BABE SG2 expert sentence bias annotations.

    Accepts a file path, an iterable of records, or falls back to standard locations.
    """
    records = []
    if source is None:
        candidates = [
            Path("data/raw/babe/final_labels_SG2.csv"),
            Path("/tmp/final_labels_SG2.csv"),
        ]
        for c in candidates:
            if c.is_file():
                source = c
                break

    if isinstance(source, (str, Path)) and Path(source).is_file():
        import csv
        path = Path(source)
        sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:500]
        delimiter = ";" if ";" in sample else ","
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                text = _extract_text(row)
                if not text:
                    continue
                label_str, label_int = normalize_bias_label(
                    row.get("label_bias") or row.get("label") or row.get("bias")
                )
                records.append({
                    "id": f"babe-{i}-{hash_text(text)[:10]}",
                    "text": text,
                    "label": label_str,
                    "label_binary": label_int,
                    "dataset": "babe",
                    "outlet": row.get("outlet", ""),
                    "role": "gold",
                })
    elif isinstance(source, list):
        for i, row in enumerate(source):
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            label_str, label_int = normalize_bias_label(row.get("label"))
            records.append({
                "id": f"babe-{i}-{hash_text(text)[:10]}",
                "text": text,
                "label": label_str,
                "label_binary": label_int,
                "dataset": "babe",
                "outlet": row.get("outlet", ""),
                "role": "gold",
            })
    return records


def load_mbic(source=None, threshold=0.5):
    """Load MBIC crowd-annotated statements, thresholding consensus score to binary."""
    records = []
    if source is None:
        candidates = [
            Path("data/raw/mbic/final_labels_MBIC.csv"),
            Path("data/raw/mbic/mbic.csv"),
            Path("/tmp/final_labels_MBIC.csv"),
        ]
        for c in candidates:
            if c.is_file():
                source = c
                break

    if isinstance(source, (str, Path)) and Path(source).is_file():
        import csv
        path = Path(source)
        snippet = path.read_text(encoding="utf-8-sig", errors="ignore")[:500]
        delimiter = ";" if ";" in snippet else ","
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                text = _extract_text(row)
                if not text:
                    continue
                raw_score = (
                    row.get("label_bias")
                    or row.get("bias_score")
                    or row.get("label")
                    or "0"
                )
                try:
                    score = float(raw_score)
                    label_str = "biased" if score >= threshold else "non-biased"
                    label_int = 1 if score >= threshold else 0
                except ValueError:
                    label_str, label_int = normalize_bias_label(raw_score)

                records.append({
                    "id": f"mbic-{i}-{hash_text(text)[:10]}",
                    "text": text,
                    "label": label_str,
                    "label_binary": label_int,
                    "dataset": "mbic",
                    "outlet": row.get("outlet", ""),
                    "role": "gold",
                })
    elif isinstance(source, list):
        for i, row in enumerate(source):
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            label_str, label_int = normalize_bias_label(row.get("label"))
            records.append({
                "id": f"mbic-{i}-{hash_text(text)[:10]}",
                "text": text,
                "label": label_str,
                "label_binary": label_int,
                "dataset": "mbic",
                "outlet": row.get("outlet", ""),
                "role": "gold",
            })
    return records


def load_basil(source=None, lexical_only=True):
    """Load BASIL sentence dataset, filtering for lexical/wording bias."""
    records = []
    if source is None:
        candidates = [
            Path("data/raw/basil/basil.csv"),
            Path("/tmp/basil.csv"),
        ]
        for c in candidates:
            if c.is_file():
                source = c
                break

    if isinstance(source, (str, Path)) and Path(source).is_file():
        import csv
        with Path(source).open("r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = _extract_text(row)
                if not text:
                    continue
                bias_type = str(row.get("bias_type") or "").lower()
                is_non_lexical = bias_type and "lexical" not in bias_type and bias_type != "none"
                if lexical_only and is_non_lexical:
                    continue
                label_str, label_int = normalize_bias_label(
                    row.get("label") or row.get("has_bias")
                )
                records.append({
                    "id": f"basil-{i}-{hash_text(text)[:10]}",
                    "text": text,
                    "label": label_str,
                    "label_binary": label_int,
                    "dataset": "basil",
                    "outlet": row.get("outlet", ""),
                    "role": "gold",
                })
    elif isinstance(source, list):
        for i, row in enumerate(source):
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            label_str, label_int = normalize_bias_label(row.get("label"))
            records.append({
                "id": f"basil-{i}-{hash_text(text)[:10]}",
                "text": text,
                "label": label_str,
                "label_binary": label_int,
                "dataset": "basil",
                "outlet": row.get("outlet", ""),
                "role": "gold",
            })
    return records


def load_anno_lexical(source=None):
    """Load Anno-lexical (48k LLM-annotated sentences), strictly tagged as weak_pretrain."""
    records = []
    if source is None:
        p = Path("data/raw/anno_lexical/anno_lexical.csv")
        if p.is_file():
            source = p

    if isinstance(source, (str, Path)) and Path(source).is_file():
        import csv
        with Path(source).open("r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = _extract_text(row)
                if not text:
                    continue
                label_str, label_int = normalize_bias_label(
                    row.get("label") or row.get("is_biased")
                )
                records.append({
                    "id": f"anno-{i}-{hash_text(text)[:10]}",
                    "text": text,
                    "label": label_str,
                    "label_binary": label_int,
                    "dataset": "anno_lexical",
                    "outlet": row.get("outlet", ""),
                    "role": "weak_pretrain",
                })
    elif isinstance(source, list):
        for i, row in enumerate(source):
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            label_str, label_int = normalize_bias_label(row.get("label"))
            records.append({
                "id": f"anno-{i}-{hash_text(text)[:10]}",
                "text": text,
                "label": label_str,
                "label_binary": label_int,
                "dataset": "anno_lexical",
                "outlet": row.get("outlet", ""),
                "role": "weak_pretrain",
            })
    return records


def merge_bias_datasets(dataset_lists, include_weak=False):
    """Combine sentence bias datasets, deduplicating by normalized text hash.

    Cross-dataset collisions are resolved using priority: babe > mbic > basil > anno_lexical.
    Tracks all originating dataset sources in the `sources` list for leakage prevention.
    """
    by_hash = {}
    for ds_records in dataset_lists:
        for r in ds_records:
            if not include_weak and r.get("role") == "weak_pretrain":
                continue
            th = hash_text(r["text"])
            if th not in by_hash:
                r_copy = dict(r)
                r_copy["sources"] = [r.get("dataset")]
                by_hash[th] = r_copy
            else:
                existing = by_hash[th]
                ds_name = r.get("dataset")
                if ds_name and ds_name not in existing.get("sources", []):
                    existing.setdefault("sources", []).append(ds_name)
                # If collision, keep the one with higher priority
                p_existing = DATASET_PRIORITY.get(existing.get("dataset"), 99)
                p_new = DATASET_PRIORITY.get(r.get("dataset"), 99)
                if p_new < p_existing:
                    accumulated_sources = existing.get("sources", [])
                    existing.update(r)
                    existing["sources"] = accumulated_sources

    return list(by_hash.values())


def create_leave_one_out_splits(records, test_dataset="mbic", val_ratio=0.1, seed=42):
    """Split merged records so one gold dataset is completely held out for testing.

    Checks originating `sources` to guarantee no cross-dataset duplicates leak into train/val.
    The remaining in-domain data is partitioned into training and validation splits.
    """
    train_records = []
    test_records = []
    for r in records:
        origin_sources = r.get("sources") or [r.get("dataset")]
        if test_dataset in origin_sources or r.get("dataset") == test_dataset:
            test_records.append(r)
        else:
            train_records.append(r)

    rng = random.Random(seed)
    rng.shuffle(train_records)
    n_val = round(len(train_records) * val_ratio) if val_ratio else 0
    val_records = train_records[:n_val]
    train_split = train_records[n_val:]

    return {
        "train": train_split,
        "validation": val_records,
        "test": test_records,
    }


def create_stratified_splits(records, seed=42, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """Partition records into train, validation, and test splits stratified by label."""
    rng = random.Random(seed)
    by_label = {}
    for r in records:
        lbl = r.get("label_binary", 0)
        by_label.setdefault(lbl, []).append(r)

    train, val, test = [], [], []
    for _lbl, group in by_label.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = round(n * val_ratio)
        n_test = round(n * test_ratio)
        n_train = n - n_val - n_test

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {
        "train": train,
        "validation": val,
        "test": test,
    }
