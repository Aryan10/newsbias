"""Dataset modules for sentence bias (merged BABE/MBIC/BASIL) and AllSides political leaning."""

from newsbias.datasets.bias import (
    create_leave_one_out_splits,
    hash_text,
    load_anno_lexical,
    load_babe,
    load_basil,
    load_mbic,
    merge_bias_datasets,
    normalize_bias_label,
)
from newsbias.datasets.bias import (
    create_stratified_splits as create_bias_splits,
)
from newsbias.datasets.political import (
    create_outlet_held_out_splits,
    load_allsides,
    normalize_political_label,
)
from newsbias.datasets.political import (
    create_random_splits as create_political_splits,
)

__all__ = [
    "create_bias_splits",
    "create_leave_one_out_splits",
    "create_outlet_held_out_splits",
    "create_political_splits",
    "hash_text",
    "load_allsides",
    "load_anno_lexical",
    "load_babe",
    "load_basil",
    "load_mbic",
    "merge_bias_datasets",
    "normalize_bias_label",
    "normalize_political_label",
]
