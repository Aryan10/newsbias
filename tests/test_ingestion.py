"""Imports preserve source wording, reject invalid selections, and cache complete inputs."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from newsbias.ingestion import cache, service
from newsbias.ingestion.blocks import assemble_article, prepare_blocks
from newsbias.ingestion.cache import ExpiringCache
from newsbias.ingestion.fetch import FetchedPage
from newsbias.ingestion.html import parse_html
from newsbias.ingestion.readers import read_upload
from newsbias.ingestion.schemas import BlockSelection, ExtractionError
from newsbias.web.app import app

RAW = (
    "Home | Politics\n\nA controversial plan\n\n"
    "Officials called the plan “reckless”.\nThe vote is tomorrow.\n\nRelated stories"
)
BODY = "Officials called the plan “reckless”.\nThe vote is tomorrow."
HTML = """<html><head><meta name="author" content="Jane Smith">
<meta property="article:published_time" content="2026-10-30"></head><body>
<nav>Home | Politics</nav><article><header><h1>A controversial plan</h1></header>
<p>Officials called the plan <strong>“reckless”</strong>.</p>
<p>The vote is tomorrow.</p><aside>Related stories</aside></article></body></html>"""


@pytest.fixture
def imported_client(monkeypatch):
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        user = kwargs["messages"][1]["content"]
        blocks, _ = json.JSONDecoder().raw_decode(user.split("Source blocks:\n", 1)[1])
        titles = [block["id"] for block in blocks if block["text"] == "A controversial plan"]
        ids = [block["id"] for block in blocks
               if block["text"].startswith(("Officials called", "The vote"))]
        content = json.dumps({
            "is_article": bool(ids), "title_block_id": titles[0] if titles else None,
            "body_block_ids": ids, "notices": [],
        })
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    agent = SimpleNamespace(
        default_model="gpt-6-luna", base_url="https://provider.example/openai/v1/",
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete)),
    )
    monkeypatch.setattr(service, "get_openai_client", lambda: agent)
    monkeypatch.setattr(service, "EXTRACTION_CACHE", ExpiringCache(ttl=3600, max_entries=128))
    monkeypatch.setattr(service, "URL_CACHE", ExpiringCache(ttl=60, max_entries=32))
    # These tests exercise import/cache behavior, independently of third-party heuristics.
    monkeypatch.setattr(service, "html_candidates", lambda html, url: iter([parse_html(html)]))
    with TestClient(app) as client:
        yield client, requests, agent


def test_raw_import_preserves_original_words_and_line_breaks(imported_client):
    client, requests, _ = imported_client
    response = client.post("/bias/import/raw", json={"text": RAW})
    assert response.status_code == 200
    article = response.json()
    assert article["title"] == "A controversial plan"
    assert article["text"] == BODY
    assert article["source"]["type"] == "raw"
    assert article["extraction"]["method"] == "raw+agent"
    assert requests[0]["model"] == "gpt-6-luna"
    assert client.post("/bias/import/raw", json={"text": RAW}).json() == article
    assert len(requests) == 1
    changed = client.post("/bias/import/raw", json={"text": RAW.replace("reckless", "disastrous")})
    assert "disastrous" in changed.json()["text"]
    assert len(requests) == 2


def test_model_changes_invalidate_extraction_cache(imported_client):
    client, requests, agent = imported_client
    client.post("/bias/import/raw", json={"text": RAW})
    agent.default_model = "another-model"
    response = client.post("/bias/import/raw", json={"text": RAW})
    assert response.status_code == 200
    assert len(requests) == 2


def test_prompt_changes_invalidate_extraction_cache(imported_client, monkeypatch):
    client, requests, _ = imported_client
    client.post("/bias/import/raw", json={"text": RAW})
    original_load = service.JinjaPrompt.load

    def updated_prompt(name):
        prompt = original_load(name)
        return service.JinjaPrompt(prompt.template_content + "\nPreserve all article quotations.\n")

    monkeypatch.setattr(service.JinjaPrompt, "load", updated_prompt)
    client.post("/bias/import/raw", json={"text": RAW})
    assert len(requests) == 2


def test_url_import_keeps_metadata_and_refreshes_changed_pages(imported_client, monkeypatch):
    client, requests, _ = imported_client
    now = [0.0]
    monkeypatch.setattr(cache, "monotonic", lambda: now[0])
    fetches = []

    def fetch(url):
        fetches.append(url)
        html = HTML if len(fetches) == 1 else HTML.replace("reckless", "disastrous")
        return FetchedPage(html, "https://news.example/final-article")

    monkeypatch.setattr(service, "fetch_public_html", fetch)
    first = client.post("/bias/import/url", json={"url": "https://news.example/article"})
    assert first.status_code == 200
    assert first.json()["source"]["url"] == "https://news.example/final-article"
    assert first.json()["source"]["author"] == "Jane Smith"
    assert first.json()["source"]["publication_date"] == "2026-10-30"
    assert "Related stories" not in first.json()["text"]
    repeated = client.post("/bias/import/url", json={"url": "https://news.example/article"})
    assert repeated.json() == first.json()
    assert len(fetches) == len(requests) == 1
    now[0] = 61
    changed = client.post("/bias/import/url", json={"url": "https://news.example/article"})
    assert "disastrous" in changed.json()["text"]
    assert len(fetches) == len(requests) == 2


def test_txt_import_uses_content_not_filename_and_keeps_attribution(imported_client):
    client, requests, _ = imported_client
    for filename in ("first.txt", "../../second.TXT"):
        response = client.post("/bias/import/file", files={
            "file": (filename, RAW.encode(), "text/plain"),
        })
        assert response.status_code == 200
        assert response.json()["text"] == BODY
        assert response.json()["source"]["filename"] == filename.rsplit("/", 1)[-1]
    assert len(requests) == 1
    changed = client.post("/bias/import/file", files={
        "file": ("first.txt", RAW.replace("reckless", "disastrous").encode(), "text/plain"),
    })
    assert "disastrous" in changed.json()["text"]
    assert len(requests) == 2


@pytest.mark.parametrize(("filename", "data", "status"), [
    ("article.pdf", b"%PDF", 415),
    ("article.txt", b"\x00binary", 415),
    ("article.txt", b"\xffbad encoding", 415),
    ("article.txt", b"x" * 1_000_001, 413),
    ("article.txt", b"   ", 422),
])
def test_invalid_uploads_do_not_call_agent(imported_client, filename, data, status):
    client, requests, _ = imported_client
    response = client.post("/bias/import/file", files={"file": (filename, data)})
    assert response.status_code == status
    assert requests == []


def test_oversized_streamed_requests_are_rejected_before_parsing(imported_client):
    client, requests, _ = imported_client
    response = client.post("/bias/import/raw", content=iter([b"x" * 600_000, b"x" * 600_000]),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert requests == []


def test_plain_text_encodings():
    for encoding in ("utf-8-sig", "utf-16"):
        assert read_upload(RAW.encode(encoding), "article.txt") == RAW


def test_missing_article_is_explicit_not_a_generated_fallback(imported_client):
    client, requests, _ = imported_client
    response = client.post("/bias/import/raw", json={"text": "Home\nLogin\nSubscribe"})
    assert response.status_code == 422
    assert "No single news article" in response.json()["detail"]
    repeated = client.post("/bias/import/raw", json={"text": "Home\nLogin\nSubscribe"})
    assert repeated.status_code == 422
    assert len(requests) == 1


def test_provider_failure_does_not_expose_credentials(imported_client, monkeypatch):
    client, _, agent = imported_client

    def fail(**kwargs):
        raise RuntimeError("provider error with secret-key")

    monkeypatch.setattr(agent.chat.completions, "create", fail)
    response = client.post("/bias/import/raw", json={"text": RAW})
    assert response.status_code == 502
    assert "secret-key" not in response.text


def test_assembly_rejects_unknown_or_duplicate_blocks_and_preserves_source_order():
    original, blocks = prepare_blocks(RAW)
    title, body = assemble_article(original, blocks, BlockSelection(
        is_article=True, title_block_id=2, body_block_ids=[4, 3],
    ))
    assert title == "A controversial plan"
    assert body == BODY
    for ids in ([999], [3, 3], []):
        with pytest.raises(ExtractionError):
            assemble_article(original, blocks, BlockSelection(
                is_article=True, title_block_id=2, body_block_ids=ids,
            ))


def test_html_preserves_article_headline_and_does_not_duplicate_nested_quotes():
    parsed = parse_html(HTML)
    assert parsed.title == "A controversial plan"
    assert "Officials called the plan “reckless”." in parsed.text
    quoted = parse_html("<article><blockquote><p>Original quote.</p></blockquote></article>")
    assert quoted.text == "Original quote."


def test_main_article_is_not_replaced_by_an_author_bio_or_related_card():
    html = """<main><h1>Actual news headline</h1><div><p>The full article body.</p></div>
    <article class="author-bio"><p>Author biography.</p></article></main>"""
    parsed = parse_html(html)
    assert parsed.title == "Actual news headline"
    assert "The full article body." in parsed.text
    cards = """<article><p>Related card.</p></article>
    <article><h1>Actual article</h1><p>A substantially longer news article body.</p></article>"""
    assert parse_html(cards).title == "Actual article"


def test_extraction_cache_is_bounded_and_does_not_share_mutable_results(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(cache, "monotonic", lambda: now[0])
    entries = ExpiringCache(ttl=60, max_entries=1)
    entries.put("one", {"notices": []})
    returned = entries.get("one")
    returned["notices"].append("Changed")
    assert entries.get("one") == {"notices": []}
    entries.put("two", {})
    assert entries.get("one") is None
    now[0] = 61
    assert entries.get("two") is None
