"""AllSides political leaning dataset loader, label mapping, and split strategies.

Supports both seen-outlets random split (standard benchmark comparability)
and outlet-held-out split (true out-of-distribution evaluation).
Flat functional design returning standard Python dictionaries.
"""

import random
from pathlib import Path

POLITICAL_LABEL_MAP = {
    "left": ("left", 0),
    "lean left": ("left", 0),
    "lean_left": ("left", 0),
    "0": ("left", 0),
    0: ("left", 0),
    "center": ("center", 1),
    "1": ("center", 1),
    1: ("center", 1),
    "right": ("right", 2),
    "lean right": ("right", 2),
    "lean_right": ("right", 2),
    "2": ("right", 2),
    2: ("right", 2),
}


def normalize_political_label(raw_label):
    """Map raw leaning string/integer into canonical 'left', 'center', or 'right'."""
    key = str(raw_label).strip().lower()
    if key in POLITICAL_LABEL_MAP:
        return POLITICAL_LABEL_MAP[key]
    if "left" in key:
        return "left", 0
    if "right" in key:
        return "right", 2
    return "center", 1


def load_allsides(source=None, name="upasanachatterjee/AllSides-random-split", revision=None):
    """Load AllSides article political bias dataset.

    Accepts an in-memory list, a JSON/CSV file path, or defaults to Hugging Face.
    """
    records = []
    if isinstance(source, list):
        for i, row in enumerate(source):
            text = (row.get("text") or row.get("content") or "").strip()
            if not text:
                continue
            lbl_str, lbl_id = normalize_political_label(row.get("label") or row.get("bias"))
            records.append({
                "id": str(row.get("id") or f"allsides-{i}"),
                "text": text,
                "title": row.get("title", ""),
                "outlet": str(row.get("outlet") or row.get("publication") or "unknown").strip(),
                "label": lbl_str,
                "label_id": lbl_id,
            })
        return records

    if isinstance(source, (str, Path)) and Path(source).is_file():
        import json
        raw = json.loads(Path(source).read_text(encoding="utf-8"))
        return load_allsides(raw)

    # Load via datasets library
    from datasets import concatenate_datasets, load_dataset
    ds_dict = load_dataset(name, revision=revision)
    splits = [ds_dict[s] for s in ds_dict]
    combined = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

    for i, row in enumerate(combined):
        text = (row.get("content") or row.get("text") or "").strip()
        if not text:
            continue
        lbl_str, lbl_id = normalize_political_label(row.get("bias") or row.get("label"))
        records.append({
            "id": str(row.get("id") or f"allsides-{i}"),
            "text": text,
            "title": row.get("title", ""),
            "outlet": str(row.get("publication") or row.get("outlet") or "unknown").strip(),
            "label": lbl_str,
            "label_id": lbl_id,
        })
    return records


def create_random_splits(records, seed=42, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """Create random stratified train/validation/test splits (seen outlets)."""
    rng = random.Random(seed)
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

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
    return {"train": train, "validation": val, "test": test}


def create_outlet_held_out_splits(records, test_outlets=None, seed=42, test_outlet_ratio=0.2):
    """Partition dataset so specific outlets are strictly held out in the test set.

    If test_outlets is None, randomly samples test_outlet_ratio of all outlets.
    """
    all_outlets = sorted({r["outlet"] for r in records if r["outlet"] and r["outlet"] != "unknown"})
    rng = random.Random(seed)

    if test_outlets is None:
        shuffled_outlets = list(all_outlets)
        rng.shuffle(shuffled_outlets)
        n_test_outlets = max(1, round(len(shuffled_outlets) * test_outlet_ratio))
        test_outlets = set(shuffled_outlets[:n_test_outlets])
    else:
        test_outlets = set(test_outlets)

    train = []
    test = []
    for r in records:
        if r["outlet"] in test_outlets:
            test.append(r)
        else:
            train.append(r)

    rng.shuffle(train)
    rng.shuffle(test)
    return {
        "train": train,
        "test": test,
        "test_outlets": sorted(test_outlets),
    }
