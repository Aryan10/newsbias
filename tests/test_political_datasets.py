"""Tests for AllSides political dataset loading, label mapping, and splits."""

from newsbias.datasets.political import (
    create_outlet_held_out_splits,
    create_random_splits,
    load_allsides,
    normalize_political_label,
)


def test_normalize_political_label() -> None:
    """Harmonize string variants into canonical left/center/right."""
    assert normalize_political_label("Left") == ("left", 0)
    assert normalize_political_label("lean left") == ("left", 0)
    assert normalize_political_label("0") == ("left", 0)

    assert normalize_political_label("Center") == ("center", 1)
    assert normalize_political_label("1") == ("center", 1)

    assert normalize_political_label("Right") == ("right", 2)
    assert normalize_political_label("lean right") == ("right", 2)
    assert normalize_political_label("2") == ("right", 2)


def test_load_allsides_from_list() -> None:
    """Load AllSides in-memory records and normalize fields."""
    raw = [
        {"text": "Article from Fox", "outlet": "Fox News", "bias": "Right"},
        {"text": "Article from MSNBC", "outlet": "MSNBC", "bias": "Left"},
        {"text": "Article from Reuters", "outlet": "Reuters", "bias": "Center"},
    ]
    records = load_allsides(raw)
    assert len(records) == 3
    assert records[0]["label"] == "right"
    assert records[0]["outlet"] == "Fox News"
    assert records[1]["label"] == "left"
    assert records[2]["label"] == "center"


def test_create_random_splits_seen_outlets() -> None:
    """Random splitting stratifies across political labels."""
    records = [
        {"id": f"l-{i}", "text": f"text {i}", "label": "left", "outlet": "Out1"}
        for i in range(30)
    ] + [
        {"id": f"c-{i}", "text": f"text {i}", "label": "center", "outlet": "Out2"}
        for i in range(30)
    ] + [
        {"id": f"r-{i}", "text": f"text {i}", "label": "right", "outlet": "Out3"}
        for i in range(30)
    ]
    splits = create_random_splits(records, seed=42)
    assert len(splits["train"]) == 63
    assert len(splits["validation"]) == 9
    assert len(splits["test"]) == 18


def test_create_outlet_held_out_splits() -> None:
    """Held-out outlet split strictly isolates test publishers from training."""
    records = [
        {"id": "1", "text": "t1", "label": "right", "outlet": "Fox News"},
        {"id": "2", "text": "t2", "label": "right", "outlet": "Fox News"},
        {"id": "3", "text": "t3", "label": "left", "outlet": "MSNBC"},
        {"id": "4", "text": "t4", "label": "center", "outlet": "Reuters"},
    ]
    splits = create_outlet_held_out_splits(records, test_outlets=["Fox News"])
    assert len(splits["test"]) == 2
    assert all(r["outlet"] == "Fox News" for r in splits["test"])
    assert len(splits["train"]) == 2
    assert all(r["outlet"] != "Fox News" for r in splits["train"])
