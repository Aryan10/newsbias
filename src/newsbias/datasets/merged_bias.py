"""Hugging Face DatasetDict adapter for the merged sentence bias dataset."""

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict

from newsbias.datasets.base import DatasetSource


@dataclass
class MergedBiasDataset(DatasetSource):
    """Dataset source for the harmonized, deduplicated sentence bias benchmark.

    Loads the stratified partitions (train, validation, test) from `data/processed/`.
    Guarantees standard columns: `id`, `text`, `label` ("biased"/"non-biased"),
    and `label_id` (1/0).
    """

    processed_dir: str = "data/processed"
    name_prefix: str = "merged_bias"

    def load(self) -> DatasetDict:
        """Load stratified partitions into a standard Hugging Face DatasetDict."""
        base_dir = Path(self.processed_dir)
        splits = {}

        for split_name in ("train", "validation", "test"):
            file_path = base_dir / f"{self.name_prefix}_{split_name}.json"
            if not file_path.is_file():
                # Check for single combined file
                combined_path = base_dir / f"{self.name_prefix}.json"
                if combined_path.is_file():
                    from newsbias.datasets.bias import create_stratified_splits

                    data = json.loads(combined_path.read_text(encoding="utf-8"))
                    stratified = create_stratified_splits(data)
                    return DatasetDict({
                        k: Dataset.from_list([
                            {
                                "id": r["id"],
                                "text": r["text"],
                                "label": r["label"],
                                "label_id": r.get(
                                    "label_binary", 1 if r["label"] == "biased" else 0
                                ),
                            }
                            for r in records
                        ])
                        for k, records in stratified.items()
                    })
                err_msg = (
                    f"Dataset split not found: {file_path}. "
                    "Run scripts/merge_bias_datasets.py first."
                )
                raise FileNotFoundError(err_msg)

            records = json.loads(file_path.read_text(encoding="utf-8"))
            splits[split_name] = Dataset.from_list([
                {
                    "id": r.get("id", f"{split_name}-{i}"),
                    "text": r["text"],
                    "label": r["label"],
                    "label_id": r.get(
                        "label_binary", 1 if r["label"] == "biased" else 0
                    ),
                }
                for i, r in enumerate(records)
            ])

        return DatasetDict(splits)
