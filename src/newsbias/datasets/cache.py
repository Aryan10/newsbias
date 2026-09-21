"""Persistent, atomic cache helpers for normalized dataset splits."""

import inspect
import json
import logging
import shutil
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import TypeAlias
from uuid import uuid4

from datasets import DatasetDict, load_from_disk
from filelock import FileLock

CacheMetadata: TypeAlias = dict[str, str | int | float]
logger = logging.getLogger(__name__)


def cache_fingerprint(metadata: CacheMetadata) -> str:
    """Produce a stable SHA-256 cache key from sorted JSON metadata.

    Sorting keys and using compact separators makes equivalent settings map to the
    same directory regardless of dictionary insertion order.
    """
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def processor_fingerprint(*functions: Callable[..., object]) -> str:
    """Fingerprint transformation source so code edits cannot silently reuse old data.

    Call this with every normalization/splitting function that affects cached rows.
    The resulting value belongs in the cache metadata alongside data and dependency
    revisions.
    """
    source = "\n".join(inspect.getsource(function) for function in functions)
    return sha256(source.encode("utf-8")).hexdigest()


def remove_path(path: Path) -> None:
    """Delete a file or recursively clear a directory during cache recovery.

    Missing paths are harmless, which makes cleanup safe after failed atomic writes.
    """
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def load_or_create_cache(
    root: Path,
    namespace: str,
    metadata: CacheMetadata,
    create: Callable[[], DatasetDict],
) -> DatasetDict:
    """Return a matching processed cache, creating it under a per-key lock if absent.

    Cache data and metadata are written to a temporary sibling directory before
    rename, preventing concurrent workers from observing partially saved Arrow files.
    Stale or corrupt entries are discarded and rebuilt through `create`.
    """
    fingerprint = cache_fingerprint(metadata)
    namespace_path = root / namespace
    cache_path = namespace_path / fingerprint
    metadata_path = cache_path / "newsbias-cache.json"
    lock_path = namespace_path / f"{fingerprint}.lock"
    namespace_path.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path)):
        if cache_path.exists():
            if cache_path.is_dir() and metadata_path.is_file():
                try:
                    cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if cached_metadata == metadata:
                        cached = load_from_disk(str(cache_path))
                        if isinstance(cached, DatasetDict):
                            logger.info("Loaded processed dataset cache %s", cache_path)
                            return cached
                except Exception as error:
                    logger.warning("Ignoring invalid dataset cache %s: %s", cache_path, error)
            remove_path(cache_path)

        temporary_path = namespace_path / f".{fingerprint}.{uuid4().hex}.tmp"
        try:
            processed = create()
            processed.save_to_disk(str(temporary_path))
            (temporary_path / "newsbias-cache.json").write_text(
                json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8",
            )
            temporary_path.rename(cache_path)
        except Exception:
            remove_path(temporary_path)
            raise
        logger.info("Created processed dataset cache %s", cache_path)
        return processed
