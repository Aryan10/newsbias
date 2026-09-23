"""Local CSV adapter for the expert-labeled BABE SG2 sentence-bias dataset."""

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from newsbias.data import text_hash
from newsbias.datasets.base import DatasetSource
from newsbias.datasets.cache import (
    CacheMetadata,
    load_or_create_cache,
    processor_fingerprint,
)
from newsbias.datasets.splitting import stratified_splits

BABE_CACHE_FORMAT = 1
BABE_LABELS = {"biased": "biased", "non-biased": "non-biased"}


def normalize_babe(raw: Dataset, *, source_name: str = "babe-sg2") -> DatasetDict:
    """Normalize the authors' expert-labeled sentence CSV for binary classification.

    `text` and `label_bias` become the common text/label fields; stable row-based IDs
    include a text hash, `outlet` becomes optional source metadata, and all original
    annotation columns remain available. The source is returned as one input split
    for subsequent shared stratification.
    """
    required = {"text", "label_bias"}
    if missing := required - set(raw.column_names):
        raise ValueError(f"BABE source is missing columns {sorted(missing)}")
    ids: list[str] = []
    texts: list[str] = []
    labels: list[str] = []
    hashes: list[str] = []
    sources: list[str] = []
    for row_index, row in enumerate(raw):
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"BABE row {row_index}: sentence text is empty or invalid")
        label_text = str(row["label_bias"]).strip().casefold()
        if label_text not in BABE_LABELS:
            raise ValueError(f"BABE row {row_index}: unknown bias label {label_text!r}")
        digest = text_hash(text)
        ids.append(f"{source_name}:{row_index}:{digest[:16]}")
        texts.append(text)
        labels.append(BABE_LABELS[label_text])
        hashes.append(digest)
        outlet = row.get("outlet")
        sources.append(str(outlet).strip() if outlet else "")
    result = raw
    for column, values in (
        ("id", ids),
        ("text", texts),
        ("label", labels),
        ("text_hash", hashes),
        ("source", sources),
        ("source_split", [source_name] * len(raw)),
    ):
        if column == "text":
            continue
        result = result.add_column(column, values)
    return DatasetDict({"source": result})


class BABEDataset(DatasetSource):
    """Prepare the locally obtained BABE SG2 sentence-bias annotations.

    This adapter does not bundle or download the authors' data. It reads the
    semicolon-delimited CSV supplied by the caller, stratifies sentence labels,
    and caches the normalized splits by the source file's content hash.
    """

    def __init__(
        self,
        csv_path: Path,
        *,
        processed_cache_dir: Path | str = Path("data/processed"),
        download_cache_dir: Path | str | None = Path("data/downloads"),
        split_seed: int = 42,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.2,
    ) -> None:
        """Set the author-provided CSV, cache locations, seed, and split proportions.

        The default proportions are 70/10/20 with seed 42. The processed cache is
        invalidated if the CSV bytes, split settings, processor code, or library
        versions change.
        """
        self.csv_path = Path(csv_path)
        self.processed_cache_dir = Path(processed_cache_dir)
        self.download_cache_dir = Path(download_cache_dir) if download_cache_dir else None
        self.split_seed = split_seed
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction

    def _cache_metadata(self) -> CacheMetadata:
        """Fingerprint the source CSV and every setting that changes prepared rows.

        Reading the file digest detects replacement or edits even when the path and
        split seed remain unchanged.
        """
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"BABE CSV not found: {self.csv_path}")
        digest = sha256(self.csv_path.read_bytes()).hexdigest()
        return {
            "format": BABE_CACHE_FORMAT,
            "dataset": "BABE-SG2",
            "source_sha256": digest,
            "datasets_version": version("datasets"),
            "scikit_learn_version": version("scikit-learn"),
            "processor_fingerprint": processor_fingerprint(
                normalize_babe, stratified_splits, self._create_processed,
            ),
            "split_seed": self.split_seed,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
        }

    def _create_processed(self) -> DatasetDict:
        """Read the author CSV through Hugging Face datasets and normalize its rows.

        The generic splitter then assigns sentences to train, validation, and test;
        original annotation fields stay attached for analysis but are not model inputs.
        """
        raw = load_dataset(
            "csv",
            data_files={"source": str(self.csv_path)},
            delimiter=";",
            cache_dir=self.download_cache_dir,
        )
        source = normalize_babe(raw["source"])
        return stratified_splits(
            source,
            seed=self.split_seed,
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
        )

    def load(self) -> DatasetDict:
        """Return normalized splits, creating and persisting them on a cache miss.

        Matching calls skip CSV parsing, sentence hashing, and stratification while
        source bytes, code, settings, and dependency versions remain unchanged.
        """
        metadata = self._cache_metadata()
        return load_or_create_cache(
            self.processed_cache_dir,
            "babe",
            metadata,
            self._create_processed,
        )
