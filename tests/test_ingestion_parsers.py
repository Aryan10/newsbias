"""Independent parser fallbacks and agent-driven retry behavior for URL imports."""

from unittest.mock import Mock

import pytest
import requests
import trafilatura

from newsbias.ingestion import parsers, service
from newsbias.ingestion.cache import ExpiringCache
from newsbias.ingestion.fetch import FetchedPage
from newsbias.ingestion.html import ParsedPage
from newsbias.ingestion.schemas import ExtractedArticle, ExtractionDetails, ExtractionError

HTML = """<html><head><title>Council approves new transport plan</title>
<meta name="author" content="Jane Smith">
<meta property="article:published_time" content="2026-10-30"></head>
<body><main><h1>Council approves new transport plan</h1>
<p>City councillors voted eight to three to approve a new transport plan on Tuesday.
The decision followed a public meeting attended by local residents and business owners.
The council said work would begin next month and continue through the following year.</p>
<p>Opponents called the proposal a reckless waste of public money and demanded an
independent review. Supporters said the investment would improve access to schools,
reduce congestion and provide more frequent connections between the city centre
and neighbourhoods currently served by limited bus routes.</p>
<p>The proposal allocates funding for new bus shelters, revised timetables and road
improvements. Officials said the details would be published before construction
begins. Residents will be able to submit comments through the council website.</p>
</main></body></html>"""


def test_trafilatura_extracts_body_and_metadata_without_rewriting():
    page = parsers.parse_trafilatura(HTML, "https://news.example/story")
    assert page is not None
    assert "reckless waste of public money" in page.text
    assert "\n" in page.text
    assert page.title == "Council approves new transport plan"
    assert page.author == "Jane Smith"
    assert page.method == "trafilatura"


def test_newspaper_uses_supplied_html_without_extra_network_requests(monkeypatch):
    def disallow_network(*args, **kwargs):
        raise AssertionError("Article parsers must not fetch additional content")

    monkeypatch.setattr(requests.sessions.Session, "request", disallow_network)
    page = parsers.parse_newspaper(HTML, "https://news.example/story")
    assert page is not None
    assert "reckless waste of public money" in page.text
    assert "\n" in page.text
    assert page.title == "Council approves new transport plan"
    assert page.method == "newspaper4k"


@pytest.mark.parametrize("first", [None, RuntimeError("parse failure")])
def test_empty_or_failed_primary_parser_advances_to_newspaper(monkeypatch, first):
    primary = Mock(side_effect=first) if isinstance(first, Exception) else Mock(return_value=None)
    secondary = Mock(return_value=ParsedPage("Article body.", "", None, None, "newspaper4k"))
    monkeypatch.setattr(parsers, "parse_trafilatura", primary)
    monkeypatch.setattr(parsers, "parse_newspaper", secondary)
    page = next(parsers.html_candidates(HTML, "https://news.example/story"))
    assert page.method == "newspaper4k"
    assert page.title == "Council approves new transport plan"
    assert page.author == "Jane Smith"
    secondary.assert_called_once()


def test_beautifulsoup_remains_the_last_fallback(monkeypatch):
    monkeypatch.setattr(parsers, "parse_trafilatura", lambda *args: None)
    monkeypatch.setattr(parsers, "parse_newspaper", lambda *args: None)
    pages = list(parsers.html_candidates(HTML, "https://news.example/story"))
    assert len(pages) == 1
    assert pages[0].method == "beautifulsoup"
    assert "reckless waste" in pages[0].text


def test_successful_primary_does_not_run_secondary(monkeypatch):
    primary = ParsedPage("Article body.", "Headline", None, None, "trafilatura")
    secondary = Mock(side_effect=AssertionError("unnecessary fallback"))
    monkeypatch.setattr(parsers, "parse_trafilatura", lambda *args: primary)
    monkeypatch.setattr(parsers, "parse_newspaper", secondary)
    result = next(parsers.html_candidates(HTML, "https://news.example/story"))
    assert result.text == primary.text
    assert result.method == "trafilatura"
    assert result.title == "Council approves new transport plan"
    secondary.assert_not_called()


def test_duplicate_candidate_text_is_not_submitted_twice(monkeypatch):
    page = ParsedPage("Article body.", "Headline", None, None, "trafilatura")
    monkeypatch.setattr(parsers, "parse_trafilatura", lambda *args: page)
    monkeypatch.setattr(parsers, "parse_newspaper", lambda *args: page)
    monkeypatch.setattr(parsers, "parse_html", lambda *args: page)
    assert list(parsers.html_candidates(HTML, "https://news.example/story")) == [page]


@pytest.fixture
def url_candidates(monkeypatch):
    candidates = [
        ParsedPage("Website navigation", "", None, None, "trafilatura"),
        ParsedPage("Original article body.", "Original headline", "Jane", None, "newspaper4k"),
    ]
    monkeypatch.setattr(service, "URL_CACHE", ExpiringCache(ttl=60, max_entries=32))
    monkeypatch.setattr(service, "fetch_public_html", lambda url: FetchedPage(HTML, url))
    monkeypatch.setattr(service, "html_candidates", lambda *args: iter(candidates))
    return candidates


def test_agent_rejection_tries_another_parser(url_candidates, monkeypatch):
    calls = []

    def extract(text, source, *, method, title_hint):
        calls.append(text)
        if text == "Website navigation":
            raise ExtractionError("No article", 422)
        return ExtractedArticle(title=title_hint, text=text, source=source,
                                extraction=ExtractionDetails(method=method))

    monkeypatch.setattr(service, "extract_document", extract)
    result = service.import_url("https://news.example/story")
    assert result.text == "Original article body."
    assert result.source.author == "Jane"
    assert result.extraction.method == "newspaper4k+agent"
    assert len(calls) == 2


@pytest.mark.parametrize("status", [502, 503])
def test_provider_failure_does_not_repeat_api_calls(url_candidates, monkeypatch, status):
    extract = Mock(side_effect=ExtractionError("Provider unavailable", status))
    monkeypatch.setattr(service, "extract_document", extract)
    with pytest.raises(ExtractionError, match="Provider unavailable"):
        service.import_url("https://news.example/story")
    extract.assert_called_once()


def test_parser_failure_cannot_change_or_fetch_source_content(monkeypatch):
    def extract(html, **kwargs):
        assert html == HTML
        assert kwargs["include_comments"] is False
        assert kwargs["favor_recall"] is True
        return None

    monkeypatch.setattr(trafilatura, "extract", extract)
    assert parsers.parse_trafilatura(HTML, "https://news.example/story") is None
