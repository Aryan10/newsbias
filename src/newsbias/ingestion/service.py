"""All imported sources share block selection, validation, and editable output."""

import hashlib
import json

from newsbias.ingestion.agent import select_article_blocks
from newsbias.ingestion.blocks import assemble_article, prepare_blocks
from newsbias.ingestion.cache import ExpiringCache
from newsbias.ingestion.fetch import FetchedPage, fetch_public_html
from newsbias.ingestion.parsers import html_candidates
from newsbias.ingestion.readers import read_upload
from newsbias.ingestion.schemas import (
    ArticleSource,
    BlockSelection,
    ExtractedArticle,
    ExtractionDetails,
    ExtractionError,
)
from newsbias.models.llm import get_openai_client
from newsbias.prompts import JinjaPrompt

EXTRACTOR_VERSION = 2
EXTRACTION_CACHE = ExpiringCache[BlockSelection](ttl=3600, max_entries=128)
URL_CACHE = ExpiringCache[FetchedPage](ttl=60, max_entries=32)


def extract_document(
    text: str, source: ArticleSource, *, method: str, title_hint: str = "",
) -> ExtractedArticle:
    original, blocks = prepare_blocks(text)
    try:
        client = get_openai_client()
        prompt = JinjaPrompt.load("article_extraction")
    except Exception as error:
        raise ExtractionError(
            "Configure OPENAI_* settings to use the extraction agent.", 503,
        ) from error
    # Cache only block selection; attribution always comes from the current import.
    payload = json.dumps([
        EXTRACTOR_VERSION, prompt.template_content, client.default_model,
        str(client.base_url), original, title_hint,
    ], ensure_ascii=False)
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    selection = EXTRACTION_CACHE.get(key)
    if selection is None:
        selection = select_article_blocks(
            blocks, title_hint=title_hint, client=client, prompt=prompt,
        )
        if not selection.is_article:
            # Repeated URL imports need not resubmit a rejected candidate before
            # reaching the successful fallback. Model/prompt changes invalidate this.
            EXTRACTION_CACHE.put(key, selection)
        title, body = assemble_article(original, blocks, selection, title_hint)
        EXTRACTION_CACHE.put(key, selection)
    else:
        title, body = assemble_article(original, blocks, selection, title_hint)
    return ExtractedArticle(
        title=title, text=body, source=source,
        extraction=ExtractionDetails(method=method, notices=selection.notices),
    )


def import_raw(text: str) -> ExtractedArticle:
    return extract_document(text, ArticleSource(type="raw"), method="raw+agent")


def import_url(url: str) -> ExtractedArticle:
    url = url.strip()
    page = URL_CACHE.get(url)
    if page is None:
        page = fetch_public_html(url)
        URL_CACHE.put(url, page)
    last_error = None
    for parsed in html_candidates(page.html, page.url):
        source = ArticleSource(
            type="url", url=page.url, author=parsed.author,
            publication_date=parsed.publication_date,
        )
        try:
            return extract_document(
                parsed.text, source, method=f"{parsed.method}+agent", title_hint=parsed.title,
            )
        except ExtractionError as error:
            # Empty, noisy, or oversized candidates can improve with another parser.
            # Provider/configuration failures cannot; do not repeat failed API calls.
            if error.status_code not in (422, 413):
                raise
            last_error = error
    if last_error is not None:
        raise last_error
    raise ExtractionError("No readable article was found on this page. Try the Raw tab.")


def import_file(data: bytes, filename: str) -> ExtractedArticle:
    # Treat the client filename as display metadata, never as a filesystem path.
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    text = read_upload(data, filename)
    return extract_document(
        text, ArticleSource(type="upload", filename=filename), method="txt+agent",
    )
