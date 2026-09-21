"""Pinned AllSides article-leaning source and normalized dataset cache."""

from importlib.metadata import version
from pathlib import Path

from datasets import DatasetDict, load_dataset

from newsbias.data import text_hash
from newsbias.datasets.base import DatasetSource
from newsbias.datasets.cache import (
    CacheMetadata,
    load_or_create_cache,
    processor_fingerprint,
)
from newsbias.datasets.splitting import stratified_splits

ALLSIDES_NAME = "upasanachatterjee/AllSides-random-split"
ALLSIDES_REVISION = "4d33d17c5591ad5d4e37ffd6238d7ead8590a70e"
ALLSIDES_CACHE_FORMAT = 1
ALLSIDES_LABELS = ("left", "center", "right")


def normalize_allsides(raw: DatasetDict, *, text_column: str = "content") -> DatasetDict:
    """Map the pinned AllSides schema to common fields while retaining source columns.

    `bias_text` becomes the lowercase class label, `bias` is retained as `label_id`,
    and native `ID` values receive a dataset prefix. The chosen body column becomes
    `text`; a normalized exact-text hash and original `source_split` are also added.
    Numeric/readable label consistency is checked across all source partitions.
    """
    result = DatasetDict()
    numeric_mapping: dict[int, str] = {}
    for split, original in raw.items():
        required = {"ID", "bias", "bias_text", text_column, "source"}
        if missing := required - set(original.column_names):
            raise ValueError(f"{split}: missing columns {sorted(missing)}")
        ids: list[str] = []
        label_ids: list[int] = []
        texts: list[str] = []
        labels: list[str] = []
        hashes: list[str] = []
        for row in original:
            article_id = row["ID"]
            if not isinstance(article_id, str) or not article_id.strip():
                raise ValueError(f"{split}: missing article ID")
            label = str(row["bias_text"]).strip().lower()
            if label not in ALLSIDES_LABELS:
                raise ValueError(f"{split}: unknown label {label!r}")
            numeric = row["bias"]
            if not isinstance(numeric, int):
                raise ValueError(f"{split}: invalid numeric bias {numeric!r}")
            if numeric_mapping.setdefault(numeric, label) != label:
                raise ValueError(f"Inconsistent numeric label mapping for {numeric}")
            text = row[text_column]
            if text is None:
                text = ""
            if not isinstance(text, str):
                raise ValueError(f"{split}: non-string article text")
            ids.append(f"allsides:{article_id}")
            label_ids.append(numeric)
            texts.append(text)
            labels.append(label)
            hashes.append(text_hash(text))
        dataset = original
        for column, values in (
            ("id", ids),
            ("label_id", label_ids),
            ("text", texts),
            ("label", labels),
            ("text_hash", hashes),
            ("source_split", [split] * len(original)),
        ):
            dataset = dataset.add_column(column, values)
        result[split] = dataset
    if len(set(numeric_mapping.values())) != len(numeric_mapping):
        raise ValueError("Multiple numeric labels map to the same readable label")
    return result


class AllSidesDataset(DatasetSource):
    """Load the pinned article-leaning corpus through the shared dataset contract.

    The adapter combines the Hub snapshot's source splits, then performs the
    configured deterministic label-stratified partition. The normalized Arrow
    DatasetDict is cached locally so later experiments reuse IDs, hashes, and splits.
    """

    def __init__(
        self,
        *,
        name: str = ALLSIDES_NAME,
        revision: str = ALLSIDES_REVISION,
        text_column: str = "content",
        processed_cache_dir: Path | str = Path("data/processed"),
        download_cache_dir: Path | str | None = Path("data/cache"),
        split_seed: int = 42,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.2,
    ) -> None:
        """Set source identity, selected text field, split ratios, and cache paths.

        Defaults pin the reviewed Hugging Face revision and use a 70/10/20 split
        with seed 42. Raw downloads and processed cache data use separate directories.
        Paths may be supplied as strings by TOML or as `Path` instances by Python.
        """
        self.name = name
        self.revision = revision
        self.text_column = text_column
        self.processed_cache_dir = Path(processed_cache_dir)
        self.download_cache_dir = Path(download_cache_dir) if download_cache_dir else None
        self.split_seed = split_seed
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction

    def _cache_metadata(self) -> CacheMetadata:
        """Build the cache key inputs for source data and every transformation setting.

        Dependency versions and processor source are included so a changed loader,
        splitter, or incompatible dataset library produces a fresh cache entry.
        """
        return {
            "format": ALLSIDES_CACHE_FORMAT,
            "dataset": self.name,
            "revision": self.revision,
            "text_column": self.text_column,
            "split_seed": self.split_seed,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "datasets_version": version("datasets"),
            "scikit_learn_version": version("scikit-learn"),
            "processor_fingerprint": processor_fingerprint(
                normalize_allsides, stratified_splits, self._create_processed,
            ),
        }

    def _create_processed(self) -> DatasetDict:
        """Fetch the pinned Hub snapshot and materialize normalized project splits.

        This is only called on a cache miss; `load()` stores the result before
        returning it for reuse by later runs.
        """
        raw = load_dataset(
            self.name,
            revision=self.revision,
            cache_dir=self.download_cache_dir,
        )
        if not isinstance(raw, DatasetDict):
            raise TypeError("Expected named AllSides dataset splits")
        normalized = normalize_allsides(raw, text_column=self.text_column)
        return stratified_splits(
            normalized,
            seed=self.split_seed,
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
        )

    def load(self) -> DatasetDict:
        """Load this source's processed DatasetDict from cache or build it once.

        A matching cache avoids re-iterating article bodies, recalculating hashes,
        and running stratification. Changed revisions/settings use a different key.
        """
        return load_or_create_cache(
            self.processed_cache_dir,
            "allsides",
            self._cache_metadata(),
            self._create_processed,
        )


def load_allsides(
    *,
    name: str = ALLSIDES_NAME,
    revision: str = ALLSIDES_REVISION,
    text_column: str = "content",
    processed_cache_dir: Path | str = Path("data/processed"),
    download_cache_dir: Path | str | None = Path("data/cache"),
    split_seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> DatasetDict:
    """Load cached AllSides splits without constructing the adapter explicitly.

    All arguments mirror `AllSidesDataset`; repeated calls with the same settings
    share its persistent processed cache.
    """
    return AllSidesDataset(
        name=name,
        revision=revision,
        text_column=text_column,
        processed_cache_dir=processed_cache_dir,
        download_cache_dir=download_cache_dir,
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    ).load()
