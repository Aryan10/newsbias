"""Shared data-integrity checks for normalized Hugging Face dataset splits."""

from collections import Counter
from hashlib import sha256
from itertools import combinations
from typing import TypedDict

from datasets import DatasetDict


class SplitAudit(TypedDict):
    """Per-split counts used to identify data-quality and class-balance problems.

    Duplicate fields count repeated values within a split; `sources` counts distinct
    nonempty publisher identifiers.
    """

    rows: int
    label_counts: dict[str, int]
    empty_texts: int
    duplicate_ids: int
    duplicate_texts: int
    sources: int


class OverlapAudit(TypedDict):
    """Pairwise intersections distinguishing article leakage from publisher reuse.

    ID or exact-text overlap is invalid; shared publishers are reported separately
    because whether they are allowed depends on the evaluation protocol.
    """

    ids: int
    texts: int
    sources: int


class DataAudit(TypedDict):
    """Complete JSON-ready integrity report for a normalized DatasetDict."""

    splits: dict[str, SplitAudit]
    overlaps: dict[str, OverlapAudit]
    numeric_label_mapping: dict[str, str]


def text_hash(text: str) -> str:
    """Return a stable SHA-256 key after collapsing whitespace in `text`.

    This detects exact content duplicates despite layout differences; it does not
    identify paraphrases or other near-duplicates.
    """
    return sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def audit_dataset(dataset: DatasetDict) -> DataAudit:
    """Measure class counts, blank text, duplicate samples, publishers, and leakage.

    The adapter contract supplies `id`, `text`, `label`, `text_hash`, and `source`
    columns. A numeric-to-readable mapping is also reported when `label_id` exists.
    The function does not mutate or repair its input.
    """
    splits: dict[str, SplitAudit] = {}
    keys: dict[str, dict[str, set[str]]] = {}
    mapping: dict[str, str] = {}
    for name, split in dataset.items():
        ids, hashes = list(split["id"]), list(split["text_hash"])
        sources = {source for source in split["source"] if source}
        keys[name] = {"ids": set(ids), "texts": set(hashes), "sources": sources}
        splits[name] = {
            "rows": len(split),
            "label_counts": dict(Counter(split["label"])),
            "empty_texts": sum(not text.strip() for text in split["text"]),
            "duplicate_ids": len(ids) - len(set(ids)),
            "duplicate_texts": len(hashes) - len(set(hashes)),
            "sources": len(sources),
        }
        if "label_id" in split.column_names:
            for numeric, label in zip(split["label_id"], split["label"], strict=True):
                mapping[str(numeric)] = label
    overlaps: dict[str, OverlapAudit] = {}
    for first, second in combinations(dataset, 2):
        overlaps[f"{first}/{second}"] = {
            "ids": len(keys[first]["ids"] & keys[second]["ids"]),
            "texts": len(keys[first]["texts"] & keys[second]["texts"]),
            "sources": len(keys[first]["sources"] & keys[second]["sources"]),
        }
    return {"splits": splits, "overlaps": overlaps, "numeric_label_mapping": mapping}


def validate_splits(dataset: DatasetDict, audit: DataAudit | None = None) -> None:
    """Reject missing/empty standard splits, blank text, duplicate samples, or leakage.

    Pass an existing audit to avoid scanning the data twice; otherwise this function
    computes one. Publisher overlap is reported but allowed, since article-random
    protocols can share publishers across splits.
    """
    if missing := {"train", "validation", "test"} - set(dataset):
        raise ValueError(f"Missing splits: {sorted(missing)}")
    if audit is None:
        audit = audit_dataset(dataset)
    for name, split in audit["splits"].items():
        if not split["rows"] or any(split[key] for key in (
            "empty_texts", "duplicate_ids", "duplicate_texts",
        )):
            raise ValueError(f"{name}: empty or duplicate samples; inspect audit.json")
    for pair, overlap in audit["overlaps"].items():
        if overlap["ids"] or overlap["texts"]:
            raise ValueError(f"{pair}: article leakage; inspect audit.json")
