"""Read UTF-8 and BOM-marked UTF-16 text without silently losing characters."""

from newsbias.ingestion.schemas import ExtractionError


def read_text(data: bytes) -> str:
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    try:
        text = data.decode(encoding)
    except UnicodeError as error:
        raise ExtractionError(
            "Save the text file as UTF-8 or UTF-16 and try again.", 415,
        ) from error
    if "\x00" in text:
        raise ExtractionError("This file contains binary data, rather than plain text.", 415)
    return text
