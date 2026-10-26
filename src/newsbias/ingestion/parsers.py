"""Try specialized article parsers before the generic BeautifulSoup source blocks."""

import json
import logging
from collections.abc import Iterator
from dataclasses import replace

from newsbias.ingestion.html import ParsedPage, parse_html
from newsbias.ingestion.schemas import MAX_INPUT_CHARS, ExtractionError

logger = logging.getLogger(__name__)


def parse_trafilatura(html: str, url: str) -> ParsedPage | None:
    import trafilatura

    output = trafilatura.extract(
        html, url=url, output_format="json", with_metadata=True,
        include_comments=False, include_tables=True, favor_recall=True,
    )
    if not output:
        return None
    data = json.loads(output)
    text = data.get("text") or ""
    if not text.strip():
        return None
    return ParsedPage(
        text, data.get("title") or "", data.get("author"), data.get("date"), "trafilatura",
    )


def parse_newspaper(html: str, url: str) -> ParsedPage | None:
    from newspaper import Article, Config

    config = Config()
    config.fetch_images = False
    config.follow_meta_refresh = False
    # Never silently truncate a long article before the shared input limit checks it.
    config.max_text = MAX_INPUT_CHARS + 1
    article = Article(url, config=config)
    # Only parse the HTML fetched by our validated, size-limited fetcher. No NLP/image
    # downloads, redirects, or additional article requests are made by Newspaper.
    article.download(input_html=html, ignore_read_more=True)
    article.parse()
    if not article.text.strip():
        return None
    return ParsedPage(
        article.text, article.title or "", ", ".join(article.authors) or None,
        article.publish_date.isoformat() if article.publish_date else None, "newspaper4k",
    )


def html_candidates(html: str, url: str) -> Iterator[ParsedPage]:
    """Lazily continue on empty/failed parsers; keep generic blocks as the last option."""
    try:
        generic = parse_html(html)
    except ExtractionError:
        generic = None
    seen = set()
    for parser in (parse_trafilatura, parse_newspaper):
        try:
            candidate = parser(html, url)
        except Exception as error:
            logger.debug(
                "Article parser %s failed: %s",
                getattr(parser, "__name__", type(parser).__name__), type(error).__name__,
            )
            continue
        if candidate is None or not candidate.text.strip():
            continue
        if generic is not None:
            preferred_title = generic.title if generic.title_from_heading else candidate.title
            candidate = replace(
                candidate,
                title=preferred_title or candidate.title or generic.title,
                author=candidate.author or generic.author,
                publication_date=candidate.publication_date or generic.publication_date,
            )
        signature = (candidate.text, candidate.title)
        if signature not in seen:
            seen.add(signature)
            yield candidate
    if generic is not None and (generic.text, generic.title) not in seen:
        yield generic
