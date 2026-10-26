"""Unit tests for NewsBias live newsfeed adapters, normalization, caching, and API endpoints."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from newsbias.ingestion.live_news import (
    GLOBAL_LIVE_NEWS_CACHE,
    fetch_gnews,
    fetch_mediastack,
    fetch_newsapi,
    fetch_rapidapi,
    get_live_news_feed,
    is_valid_http_url,
    make_story_id,
)
from newsbias.web.app import BENCHMARK_SAMPLES, app


@pytest.fixture(autouse=True)
def clear_live_cache():
    """Ensure in-memory cache is clean for every test."""
    GLOBAL_LIVE_NEWS_CACHE.clear()
    yield
    GLOBAL_LIVE_NEWS_CACHE.clear()


# ── 1. ID Generation & URL Validation ─────────────────────────────────────────

def test_make_story_id_stable_and_provider_qualified():
    url = "https://example.com/world/news-item"
    id_gnews = make_story_id("gnews", url)
    id_newsapi = make_story_id("newsapi", url)

    assert id_gnews.startswith("gnews:")
    assert id_newsapi.startswith("newsapi:")
    # Stable: same input produces same ID
    assert make_story_id("gnews", url) == id_gnews
    # Distinct across providers
    assert id_gnews != id_newsapi
    # Distinct from curated sample IDs
    curated_ids = {s["id"] for s in BENCHMARK_SAMPLES}
    assert id_gnews not in curated_ids


def test_is_valid_http_url():
    assert is_valid_http_url("https://example.com/article") is True
    assert is_valid_http_url("http://insecure.example.com") is True
    assert is_valid_http_url("javascript:alert(1)") is False
    assert is_valid_http_url("data:text/html;base64,...") is False
    assert is_valid_http_url("/relative/path") is False
    assert is_valid_http_url("") is False
    assert is_valid_http_url(None) is False
    assert is_valid_http_url(12345) is False


# ── 2. Provider Fetching & Normalization ───────────────────────────────────────

def test_fetch_gnews_capping_and_normalization():
    mock_articles = [
        {
            "id": f"g-{i}",
            "title": f"GNews Headline {i}",
            "description": f"GNews summary {i}",
            "content": f"Truncated GNews preview text {i} ... [500 chars]",
            "url": f"https://publisher.example/gnews-{i}",
            "image": f"https://publisher.example/img-{i}.jpg",
            "publishedAt": "2026-10-30T10:00:00Z",
            "source": {"name": "Example Wire"},
        }
        for i in range(10)
    ]
    # Include an invalid article to ensure filtering
    mock_articles.append({"title": "No URL", "url": ""})

    fake_resp = io.BytesIO(json.dumps({"articles": mock_articles}).encode("utf-8"))

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = fake_resp

        results = fetch_gnews(api_key="test-key", limit=3)

        assert len(results) == 3
        # Verify request parameters
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "max=3" in req.full_url
        assert "page=1" in req.full_url
        assert "apikey=test-key" in req.full_url

        # Check normalization structure
        story = results[0]
        assert story["id"].startswith("gnews:")
        assert story["title"] == "GNews Headline 0"
        assert story["source"] == "Example Wire"
        assert story["section"] == "General"
        assert story["description"] == "GNews summary 0"
        assert story["content"] is None  # Never store truncated provider preview as content!
        assert story["url"] == "https://publisher.example/gnews-0"
        assert story["imageUrl"] == "https://publisher.example/img-0.jpg"
        assert story["publishedAt"] == "2026-10-30T10:00:00Z"
        assert story["collection"] == "live"


def test_fetch_newsapi_capping_and_normalization():
    mock_articles = [
        {
            "title": f"NewsAPI Headline {i}",
            "description": f"NewsAPI summary {i}",
            "url": f"https://newsapi.example/{i}",
            "urlToImage": f"https://newsapi.example/{i}.png",
            "publishedAt": "2026-10-30T11:00:00Z",
            "source": {"id": "reuters", "name": "Reuters"},
        }
        for i in range(8)
    ]
    fake_resp = io.BytesIO(json.dumps({"articles": mock_articles}).encode("utf-8"))

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = fake_resp

        results = fetch_newsapi(api_key="news-key", limit=4)

        assert len(results) == 4
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "pageSize=4" in req.full_url
        assert "page=1" in req.full_url
        assert req.headers.get("X-api-key") == "news-key"

        story = results[0]
        assert story["id"].startswith("newsapi:")
        assert story["title"] == "NewsAPI Headline 0"
        assert story["source"] == "Reuters"
        assert story["section"] == "General"
        assert story["content"] is None
        assert story["url"] == "https://newsapi.example/0"
        assert story["imageUrl"] == "https://newsapi.example/0.png"
        assert story["collection"] == "live"


def test_fetch_mediastack_capping_and_normalization():
    mock_data = [
        {
            "title": f"MediaStack Headline {i}",
            "description": f"MediaStack summary {i}",
            "url": f"https://mediastack.example/{i}",
            "source": "Bloomberg",
            "image": f"https://mediastack.example/img/{i}.jpg",
            "category": "business",
            "published_at": "2026-10-30T09:30:00+00:00",
        }
        for i in range(7)
    ]
    fake_resp = io.BytesIO(json.dumps({"data": mock_data}).encode("utf-8"))

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = fake_resp

        results = fetch_mediastack(api_key="media-key", limit=2)

        assert len(results) == 2
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "limit=2" in req.full_url
        assert "offset=0" in req.full_url

        story = results[0]
        assert story["id"].startswith("mediastack:")
        assert story["title"] == "MediaStack Headline 0"
        assert story["source"] == "Bloomberg"
        assert story["section"] == "Business"  # Category capitalized
        assert story["content"] is None
        assert story["url"] == "https://mediastack.example/0"
        assert story["publishedAt"] == "2026-10-30T09:30:00+00:00"
        assert story["collection"] == "live"


def test_fetch_rapidapi_capping_and_normalization():
    mock_data = [
        {
            "title": f"RapidAPI Headline {i}",
            "snippet": f"RapidAPI snippet {i}",
            "link": f"https://thehindu.com/article-{i}",
            "source_name": "The Hindu",
            "photo_url": f"https://thehindu.com/img-{i}.jpg",
            "published_datetime_utc": "2026-10-30T12:00:00.000Z",
        }
        for i in range(6)
    ]
    fake_resp = io.BytesIO(json.dumps({"data": mock_data}).encode("utf-8"))

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = fake_resp

        results = fetch_rapidapi(api_key="rapid-key", limit=3)

        assert len(results) == 3
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "limit=3" in req.full_url
        assert req.headers.get("X-rapidapi-key") == "rapid-key"
        assert req.headers.get("X-rapidapi-host") == "real-time-news-data.p.rapidapi.com"

        story = results[0]
        assert story["id"].startswith("rapidapi:")
        assert story["title"] == "RapidAPI Headline 0"
        assert story["source"] == "The Hindu"
        assert story["section"] == "National"
        assert story["content"] is None
        assert story["url"] == "https://thehindu.com/article-0"
        assert story["imageUrl"] == "https://thehindu.com/img-0.jpg"
        assert story["publishedAt"] == "2026-10-30T12:00:00.000Z"
        assert story["collection"] == "live"


# ── 3. Error Resilience & Partial Results ─────────────────────────────────────

def test_missing_credentials_resilience():
    with patch("newsbias.ingestion.live_news.get_provider_keys", return_value={}):
        feed = get_live_news_feed(providers=["gnews", "newsapi"])
        assert feed["stories"] == []
        assert feed["providers"]["gnews"]["status"] == "unconfigured"
        assert feed["providers"]["newsapi"]["status"] == "unconfigured"


def test_partial_provider_failure_returns_working_provider_stories():
    """If one provider errors (e.g. HTTP 401 or timeout), working providers still succeed."""
    working_story = {
        "id": "mediastack:abc123",
        "title": "Working Story",
        "source": "AP",
        "section": "General",
        "description": "...",
        "content": None,
        "url": "https://apnews.com/article-1",
        "imageUrl": None,
        "publishedAt": "2026-10-30T10:00:00Z",
        "collection": "live",
    }

    def fake_gnews_error(key, limit, timeout):
        raise urllib.error.HTTPError("https://gnews.io", 401, "Unauthorized", {}, None)

    def fake_mediastack_ok(key, limit, timeout):
        return [working_story]

    mock_fetchers = {
        "gnews": fake_gnews_error,
        "mediastack": fake_mediastack_ok,
    }

    provider_keys = {"gnews": "k1", "mediastack": "k2"}
    with (
        patch("newsbias.ingestion.live_news.get_provider_keys", return_value=provider_keys),
        patch.dict("newsbias.ingestion.live_news.PROVIDER_FETCHERS", mock_fetchers, clear=True),
    ):
        feed = get_live_news_feed(providers=["gnews", "mediastack"])

        assert len(feed["stories"]) == 1
        assert feed["stories"][0]["title"] == "Working Story"
        assert feed["providers"]["mediastack"]["status"] == "ok"
        assert feed["providers"]["gnews"]["status"] == "error"
        # Verify credential is not present anywhere in the output
        assert "k1" not in json.dumps(feed)
        assert "k2" not in json.dumps(feed)


# ── 4. Caching & Invalidation ──────────────────────────────────────────────────

def test_live_news_caching_and_force_refresh():
    call_count = 0

    def mock_fetch(key, limit, timeout):
        nonlocal call_count
        call_count += 1
        return [{
            "id": f"gnews:{call_count}",
            "title": f"Title {call_count}",
            "source": "Source",
            "section": "General",
            "description": "Desc",
            "content": None,
            "url": f"https://example.com/{call_count}",
            "imageUrl": None,
            "publishedAt": "2026-10-30T10:00:00Z",
            "collection": "live",
        }]

    with (
        patch("newsbias.ingestion.live_news.get_provider_keys", return_value={"gnews": "mock-key"}),
        patch.dict(
            "newsbias.ingestion.live_news.PROVIDER_FETCHERS", {"gnews": mock_fetch}, clear=True
        ),
    ):
        # Initial request: fetches from provider
        feed1 = get_live_news_feed(providers=["gnews"], force_refresh=False)
        assert call_count == 1
        assert feed1["cached"] is False
        assert feed1["stories"][0]["title"] == "Title 1"

        # Subsequent request without refresh: served from cache
        feed2 = get_live_news_feed(providers=["gnews"], force_refresh=False)
        assert call_count == 1
        assert feed2["cached"] is True
        assert feed2["stories"][0]["title"] == "Title 1"

        # Force refresh: bypasses cache and re-fetches
        feed3 = get_live_news_feed(providers=["gnews"], force_refresh=True)
        assert call_count == 2
        assert feed3["cached"] is False
        assert feed3["stories"][0]["title"] == "Title 2"


# ── 5. Web Endpoint Integration ────────────────────────────────────────────────

def test_live_news_endpoint_and_samples_preservation():
    client = TestClient(app)

    # 1. Verify GET /bias/samples remains completely preserved and unaffected
    samples_resp = client.get("/bias/samples")
    assert samples_resp.status_code == 200
    samples_data = samples_resp.json()
    assert len(samples_data) == len(BENCHMARK_SAMPLES)
    assert samples_data[0]["id"] == BENCHMARK_SAMPLES[0]["id"]

    # 2. Verify GET /bias/live-news endpoint responds with expected contract
    mock_story = {
        "id": "rapidapi:fedcba987654",
        "title": "Live Breaking News",
        "source": "The Hindu",
        "section": "National",
        "description": "Live report description",
        "content": None,
        "url": "https://thehindu.com/news/123",
        "imageUrl": "https://thehindu.com/img.jpg",
        "publishedAt": "2026-10-30T12:00:00Z",
        "collection": "live",
    }

    mock_payload = {
        "stories": [mock_story],
        "providers": {"rapidapi": {"status": "ok", "count": 1}},
        "cached": False,
    }
    with patch("newsbias.web.app.get_live_news_feed", return_value=mock_payload) as mock_get_feed:
        resp = client.get("/bias/live-news?refresh=true&limit=5")
        assert resp.status_code == 200
        payload = resp.json()
        assert "stories" in payload
        assert len(payload["stories"]) == 1
        assert payload["stories"][0]["id"] == "rapidapi:fedcba987654"
        mock_get_feed.assert_called_once_with(limit_per_provider=5, force_refresh=True)
