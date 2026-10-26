"""Upload reader dispatch. Add new file formats here as support is introduced."""

from pathlib import PurePath

from newsbias.ingestion.readers.text import read_text
from newsbias.ingestion.schemas import MAX_FILE_BYTES, ExtractionError

READERS = {".txt": read_text}


def read_upload(data: bytes, filename: str) -> str:
    if len(data) > MAX_FILE_BYTES:
        raise ExtractionError("Uploads must be 1 MB or smaller.", 413)
    reader = READERS.get(PurePath(filename).suffix.lower())
    if reader is None:
        raise ExtractionError(
            "Upload currently supports .txt files. More formats will be added.", 415,
        )
    return reader(data)
