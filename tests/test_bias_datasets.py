"""Tests for sentence bias loaders, merging, deduplication, and splits."""

from newsbias.datasets.bias import (
    create_leave_one_out_splits,
    create_stratified_splits,
    hash_text,
    load_anno_lexical,
    load_babe,
    load_basil,
    load_mbic,
    merge_bias_datasets,
    normalize_bias_label,
)


def test_normalize_bias_label() -> None:
    """Harmonize diverse label inputs to biased/non-biased."""
    for val in ("1", 1, "biased", "bias", "true", "yes"):
        lbl_str, lbl_int = normalize_bias_label(val)
        assert lbl_str == "biased"
        assert lbl_int == 1

    for val in ("0", 0, "non-biased", "non_biased", "false", "no"):
        lbl_str, lbl_int = normalize_bias_label(val)
        assert lbl_str == "non-biased"
        assert lbl_int == 0


def test_hash_text_deterministic_and_clean() -> None:
    """Text hashing ignores case, outer spacing, and punctuation variations."""
    h1 = hash_text("The president made a statement.")
    h2 = hash_text("  the president made a statement!  ")
    assert h1 == h2


def test_loaders_from_list() -> None:
    """Loaders normalize in-memory list records with appropriate metadata."""
    babe_raw = [{"text": "Babe sentence one", "label": "biased", "outlet": "NYT"}]
    babe_recs = load_babe(babe_raw)
    assert len(babe_recs) == 1
    assert babe_recs[0]["dataset"] == "babe"
    assert babe_recs[0]["role"] == "gold"
    assert babe_recs[0]["label"] == "biased"

    mbic_raw = [{"text": "Mbic statement", "label": "0.8"}]
    mbic_recs = load_mbic(mbic_raw)
    assert len(mbic_recs) == 1
    assert mbic_recs[0]["dataset"] == "mbic"
    assert mbic_recs[0]["role"] == "gold"
    assert mbic_recs[0]["label"] == "biased"

    basil_raw = [{"text": "Basil sentence", "label": "non-biased", "bias_type": "lexical"}]
    basil_recs = load_basil(basil_raw)
    assert len(basil_recs) == 1
    assert basil_recs[0]["dataset"] == "basil"

    anno_raw = [{"text": "Anno sentence", "label": "1"}]
    anno_recs = load_anno_lexical(anno_raw)
    assert len(anno_recs) == 1
    assert anno_recs[0]["dataset"] == "anno_lexical"
    assert anno_recs[0]["role"] == "weak_pretrain"


def test_merge_bias_datasets_dedup_and_priority() -> None:
    """Cross-dataset duplicate sentences resolve with BABE > MBIC > BASIL priority."""
    shared_text = "This is a duplicated controversial sentence."

    babe_recs = [{"text": shared_text, "label": "biased", "dataset": "babe", "role": "gold"}]
    mbic_recs = [{"text": shared_text, "label": "non-biased", "dataset": "mbic", "role": "gold"}]
    basil_recs = [{"text": shared_text, "label": "biased", "dataset": "basil", "role": "gold"}]
    unique_rec = [
        {"text": "Unique sentence", "label": "non-biased", "dataset": "mbic", "role": "gold"}
    ]

    merged = merge_bias_datasets([babe_recs, mbic_recs, basil_recs, unique_rec])
    assert len(merged) == 2

    # Verify BABE priority won for the duplicate
    dup_entry = [r for r in merged if r["text"] == shared_text][0]
    assert dup_entry["dataset"] == "babe"
    assert dup_entry["label"] == "biased"


def test_leave_one_dataset_out_splits() -> None:
    """LODO split isolates the selected gold benchmark exclusively for testing."""
    records = [
        {"id": "1", "text": "t1", "dataset": "babe"},
        {"id": "2", "text": "t2", "dataset": "babe"},
        {"id": "3", "text": "t3", "dataset": "basil"},
        {"id": "4", "text": "t4", "dataset": "mbic"},
    ]
    splits = create_leave_one_out_splits(records, test_dataset="mbic")
    assert len(splits["test"]) == 1
    assert splits["test"][0]["dataset"] == "mbic"
    assert len(splits["train"]) == 3
    assert all(r["dataset"] != "mbic" for r in splits["train"])


def test_create_stratified_splits() -> None:
    """Stratified splitting preserves binary class distribution across partitions."""
    records = [
        {"id": f"b-{i}", "text": f"text {i}", "label_binary": 1}
        for i in range(50)
    ] + [
        {"id": f"nb-{i}", "text": f"text {i}", "label_binary": 0}
        for i in range(50)
    ]
    splits = create_stratified_splits(
        records, seed=42, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2
    )
    assert len(splits["train"]) == 70
    assert len(splits["validation"]) == 10
    assert len(splits["test"]) == 20
