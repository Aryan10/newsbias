"""News page metadata and paragraph extraction utilities."""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from newsbias.ingestion.schemas import ExtractionError


@dataclass(frozen=True)
class ParsedPage:
    text: str
    title: str
    author: str | None
    publication_date: str | None
    method: str = "beautifulsoup"
    title_from_heading: bool = False


def parse_html(html: str) -> ParsedPage:
    soup = BeautifulSoup(html, "html.parser")

    def meta(name: str) -> str | None:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        value = tag.get("content") if tag else None
        return str(value).strip() if value else None

    author = meta("author")
    published = meta("article:published_time") or meta("datePublished")
    title_hint = meta("og:title") or (soup.title.get_text(" ", strip=True) if soup.title else "")
    # News sites also use <article> for author bios and recommendation cards.
    # Prefer the main region; otherwise choose the largest article container.
    articles = soup.find_all("article")
    root = soup.find("main") or (
        max(articles, key=lambda node: len(node.get_text())) if articles else None
    ) or soup.body or soup
    for node in root.find_all(["script", "style", "nav", "footer", "aside", "form", "dialog"]):
        node.decompose()
    headline = root.find("h1")
    title = headline.get_text(" ", strip=True) if headline else title_hint
    tags = {"h1", "h2", "h3", "p", "blockquote", "li"}
    paragraphs = []
    for node in root.find_all(list(tags)):
        # A blockquote/list item containing paragraphs should not be copied twice.
        parent = node.parent
        nested = False
        while parent is not None and parent is not root:
            if parent.name in tags:
                nested = True
                break
            parent = parent.parent
        if nested:
            continue
        # Keep original spaces around inline tags, rather than adding spaces before punctuation.
        text = re.sub(r"\s+", " ", node.get_text()).strip()
        if text:
            paragraphs.append(text)
    text = "\n\n".join(paragraphs) if paragraphs else root.get_text("\n", strip=True)
    if not text.strip():
        raise ExtractionError("No readable text was found on this page. Try the Raw tab.")
    return ParsedPage(text, title, author, published, title_from_heading=headline is not None)
