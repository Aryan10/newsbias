"""Select source spans without changing the wording or paragraph spacing."""

import re
from dataclasses import dataclass

from newsbias.ingestion.schemas import MAX_INPUT_CHARS, BlockSelection, ExtractionError


@dataclass(frozen=True)
class TextBlock:
    id: int
    text: str
    start: int
    end: int


def prepare_blocks(text: str) -> tuple[str, list[TextBlock]]:
    if len(text) > MAX_INPUT_CHARS:
        raise ExtractionError("The input is too long. Import an article-sized excerpt.", 413)
    original = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = [
        TextBlock(index, match.group(), match.start(), match.end())
        for index, match in enumerate(re.finditer(r"[^\n]+", original), 1)
        if match.group().strip()
    ]
    if not blocks:
        raise ExtractionError("Paste or upload some article text first.")
    if len(blocks) > 2000:
        raise ExtractionError("Too many text blocks. Import an article-sized excerpt.", 413)
    return original, blocks


def assemble_article(
    original: str, blocks: list[TextBlock], selection: BlockSelection, title_hint: str = "",
) -> tuple[str, str]:
    if not selection.is_article:
        raise ExtractionError(
            "No single news article was found. Paste a specific article or excerpt.",
        )
    by_id = {block.id: block for block in blocks}
    ids = selection.body_block_ids
    if not ids or len(ids) != len(set(ids)) or any(index not in by_id for index in ids):
        raise ExtractionError(
            "The extraction agent selected invalid article blocks. Try again.", 502,
        )
    title_id = selection.title_block_id
    if title_id is not None and (title_id not in by_id or title_id in ids):
        raise ExtractionError("The extraction agent selected an invalid headline. Try again.", 502)
    chosen = sorted(ids, key=lambda index: by_id[index].start)
    groups: list[tuple[int, int]] = []
    for index in chosen:
        block = by_id[index]
        if groups and not original[groups[-1][1]:block.start].strip():
            groups[-1] = (groups[-1][0], block.end)
        else:
            groups.append((block.start, block.end))
    body = "\n\n".join(original[start:end] for start, end in groups).strip()
    title = by_id[title_id].text.strip() if title_id is not None else title_hint.strip()
    return title, body
