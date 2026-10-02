#!/usr/bin/env python3
"""Standalone pipeline script to harmonize, deduplicate, and merge sentence-level bias datasets.

Combines gold-standard annotations (BABE, MBIC, BASIL lexical) and optional weak-pretraining
corpora (Anno-lexical) into a unified dataset for training and cross-benchmark evaluation.
Resolves overlapping sentences using priority order (BABE > MBIC > BASIL > Anno-lexical)
and calculates class distribution and deduplication statistics.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from newsbias.datasets.bias import (
    create_leave_one_out_splits,
    create_stratified_splits,
    load_anno_lexical,
    load_babe,
    load_basil,
    load_mbic,
    merge_bias_datasets,
)


def build_parser():
    """Build CLI parser for dataset merge and splitting options."""
    parser = argparse.ArgumentParser(
        description="Harmonize and deduplicate sentence-level bias datasets (BABE, MBIC, BASIL)."
    )
    parser.add_argument(
        "--babe",
        type=Path,
        default=None,
        help="Path or HuggingFace identifier for BABE dataset (default: auto/hf)",
    )
    parser.add_argument(
        "--mbic",
        type=Path,
        default=None,
        help="Path or HuggingFace identifier for MBIC dataset (default: auto/hf)",
    )
    parser.add_argument(
        "--basil",
        type=Path,
        default=None,
        help="Path to BASIL dataset CSV (default: auto/hf)",
    )
    parser.add_argument(
        "--anno-lexical",
        type=Path,
        default=None,
        help="Optional path to Anno-lexical weak-pretraining CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/merged_bias.json"),
        help="Output path for merged dataset (.json or .csv)",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["none", "stratified", "lodo"],
        default="none",
        help="Data partitioning strategy: none, stratified, or lodo (leave-one-dataset-out)",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="mbic",
        help="Held-out benchmark name when --split-strategy is lodo (e.g., 'mbic' or 'babe')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split partitioning",
    )
    return parser


def save_records(records, path):
    """Write list of record dicts to JSON or CSV based on file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        if not records:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(records[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    else:
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def run_merge(args):
    """Execute dataset merging, print diagnostics, and persist output files."""
    print("=" * 70)
    print("NEWSBIAS GENERAL SENTENCE BIAS DATASET MERGER")
    print("=" * 70)

    # 1. Load each dataset source
    sources = []
    source_counts = {}

    print("Loading BABE (gold standard)...")
    babe_recs = load_babe(source=str(args.babe) if args.babe else None)
    sources.append(babe_recs)
    source_counts["BABE"] = len(babe_recs)

    print("Loading MBIC (gold standard)...")
    mbic_recs = load_mbic(source=str(args.mbic) if args.mbic else None)
    sources.append(mbic_recs)
    source_counts["MBIC"] = len(mbic_recs)

    print("Loading BASIL (gold lexical subset)...")
    basil_recs = load_basil(source=str(args.basil) if args.basil else None)
    sources.append(basil_recs)
    source_counts["BASIL"] = len(basil_recs)

    if args.anno_lexical:
        print("Loading Anno-lexical (weak pretrain)...")
        anno_recs = load_anno_lexical(source=str(args.anno_lexical))
        sources.append(anno_recs)
        source_counts["Anno-lexical"] = len(anno_recs)

    total_raw = sum(source_counts.values())

    # 2. Harmonize and deduplicate with priority order
    merged = merge_bias_datasets(sources)
    total_dedup = len(merged)
    duplicates_removed = total_raw - total_dedup

    # 3. Compute statistics
    biased_count = sum(1 for r in merged if r.get("label_binary") == 1)
    non_biased_count = total_dedup - biased_count
    bias_pct = (biased_count / total_dedup * 100) if total_dedup > 0 else 0.0

    print("-" * 70)
    print("Source Breakdown (Raw):")
    for name, cnt in source_counts.items():
        print(f"  - {name:<15}: {cnt:>6} sentences")
    print(f"  Total Raw:       {total_raw:>6} sentences")
    print(f"  Duplicates Cut:  {duplicates_removed:>6} sentences")
    print(f"  Final Merged:    {total_dedup:>6} unique sentences")
    print("-" * 70)
    print(f"Class Balance:     {biased_count} biased ({bias_pct:.1f}%), {non_biased_count} neutral")
    print("=" * 70)

    # 4. Handle split strategies and persistence
    output_path = Path(args.output)
    if args.split_strategy == "none":
        save_records(merged, output_path)
        print(f"Saved merged dataset to: {output_path}")
    elif args.split_strategy == "stratified":
        splits = create_stratified_splits(merged, seed=args.seed)
        stem = output_path.stem
        ext = output_path.suffix or ".json"
        parent = output_path.parent
        for part, split_records in splits.items():
            part_path = parent / f"{stem}_{part}{ext}"
            save_records(split_records, part_path)
            print(f"Saved {part:<10} split ({len(split_records)} records): {part_path}")
    elif args.split_strategy == "lodo":
        splits = create_leave_one_out_splits(
            merged, test_dataset=args.test_dataset, seed=args.seed
        )
        stem = output_path.stem
        ext = output_path.suffix or ".json"
        parent = output_path.parent
        for part, split_records in splits.items():
            part_path = parent / f"{stem}_lodo_{args.test_dataset}_{part}{ext}"
            save_records(split_records, part_path)
            print(f"Saved {part:<10} split ({len(split_records)} records): {part_path}")

    return merged


def main(argv=None):
    """Main CLI entrypoint for standalone dataset merge script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_merge(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
