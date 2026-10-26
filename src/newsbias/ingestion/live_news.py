"""Live newsfeed provider adapters, normalization, and caching for NewsBias.

Connects to configured external news APIs (GNews, NewsAPI, MediaStack, RapidAPI)
to retrieve a bounded live news feed. Ensures stories are capped to small per-provider
limits, queries page 1 only, normalizes entries to FeedStory format, and caches responses
with a short TTL. Never prints or leaks credentials to logs, frontend, or responses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger("newsbias.live_news")

# ── Provider Keys & Defaults ──────────────────────────────────────────────────
PROVIDER_GNEWS = "gnews"
PROVIDER_NEWSAPI = "newsapi"
PROVIDER_MEDIASTACK = "mediastack"
PROVIDER_RAPIDAPI = "rapidapi"

ALL_PROVIDERS = [PROVIDER_GNEWS, PROVIDER_NEWSAPI, PROVIDER_MEDIASTACK, PROVIDER_RAPIDAPI]

DEFAULT_PER_PROVIDER_LIMIT = 5
MAX_PER_PROVIDER_LIMIT = 10
DEFAULT_CACHE_TTL = 300.0  # 5 minutes


def load_live_news_env() -> None:
    """Ensure provider API keys are loaded from environment or local .env files.

    Checks repository root .env first, then parent directory .env if available.
    Does not log or expose any key values.
    """
    keys = ["GNEWS_API_KEY", "NEWS_API_KEY", "MEDIASTACK_API_KEY", "RAPIDAPI_KEY"]
    if all(os.getenv(k) for k in keys):
        return

    current_dir = Path(__file__).resolve().parent.parent.parent.parent
    for env_path in [current_dir / ".env", current_dir.parent / ".env"]:
        if env_path.is_file():
            load_dotenv(env_path, override=False)


def get_provider_keys() -> dict[str, str | None]:
    """Retrieve configured provider API keys from environment."""
    load_live_news_env()
    return {
        PROVIDER_GNEWS: os.getenv("GNEWS_API_KEY") or None,
        PROVIDER_NEWSAPI: os.getenv("NEWS_API_KEY") or None,
        PROVIDER_MEDIASTACK: os.getenv("MEDIASTACK_API_KEY") or None,
        PROVIDER_RAPIDAPI: os.getenv("RAPIDAPI_KEY") or None,
    }


def is_valid_http_url(url: Any) -> bool:
    """Validate that a URL uses http or https scheme and is non-empty."""
    if not url or not isinstance(url, str):
        return False
    trimmed = url.strip()
    return trimmed.startswith("http://") or trimmed.startswith("https://")


def make_story_id(provider: str, url: str) -> str:
    """Generate a stable, collision-free provider-qualified story ID.

    Format: {provider}:{sha256(url)[:12]}
    Ensures live IDs cannot collide with curated benchmark sample IDs.
    """
    normalized_url = url.strip()
    digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:12]
    return f"{provider}:{digest}"


# ── Provider Fetchers ─────────────────────────────────────────────────────────

def fetch_gnews(
    api_key: str,
    limit: int = DEFAULT_PER_PROVIDER_LIMIT,
    timeout: float = 8.0,
) -> list[dict[str, Any]]:
    """Fetch live top headlines from GNews API."""
    cap = max(1, min(limit, MAX_PER_PROVIDER_LIMIT))
    params = {
        "category": "general",
        "lang": "en",
        "max": str(cap),
        "page": "1",
        "apikey": api_key,
    }
    url = f"https://gnews.io/api/v4/top-headlines?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBias/1.0)"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw_articles = data.get("articles", []) if isinstance(data, dict) else []
    stories: list[dict[str, Any]] = []

    for item in raw_articles:
        if not isinstance(item, dict):
            continue
        article_url = item.get("url")
        title = (item.get("title") or "").strip()
        if not title or not is_valid_http_url(article_url):
            continue

        source_info = item.get("source")
        source_name = source_info.get("name") if isinstance(source_info, dict) else None
        image_url = item.get("image")
        valid_image = image_url if is_valid_http_url(image_url) else None

        stories.append({
            "id": make_story_id(PROVIDER_GNEWS, article_url),
            "title": title,
            "source": source_name or "GNews",
            "section": "General",
            "description": (item.get("description") or "").strip(),
            "content": None,  # Full article extracted on open; not preview snippet.
            "url": article_url,
            "imageUrl": valid_image,
            "publishedAt": item.get("publishedAt"),
            "collection": "live",
        })
        if len(stories) >= cap:
            break

    return stories


def fetch_newsapi(
    api_key: str,
    limit: int = DEFAULT_PER_PROVIDER_LIMIT,
    timeout: float = 8.0,
) -> list[dict[str, Any]]:
    """Fetch live top headlines from NewsAPI."""
    cap = max(1, min(limit, MAX_PER_PROVIDER_LIMIT))
    params = {
        "category": "general",
        "language": "en",
        "pageSize": str(cap),
        "page": "1",
    }
    url = f"https://newsapi.org/v2/top-headlines?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; NewsBias/1.0)",
            "X-Api-Key": api_key,
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw_articles = data.get("articles", []) if isinstance(data, dict) else []
    stories: list[dict[str, Any]] = []

    for item in raw_articles:
        if not isinstance(item, dict):
            continue
        article_url = item.get("url")
        title = (item.get("title") or "").strip()
        if not title or not is_valid_http_url(article_url):
            continue

        source_info = item.get("source")
        source_name = source_info.get("name") if isinstance(source_info, dict) else None
        image_url = item.get("urlToImage")
        valid_image = image_url if is_valid_http_url(image_url) else None

        stories.append({
            "id": make_story_id(PROVIDER_NEWSAPI, article_url),
            "title": title,
            "source": source_name or "NewsAPI",
            "section": "General",
            "description": (item.get("description") or "").strip(),
            "content": None,
            "url": article_url,
            "imageUrl": valid_image,
            "publishedAt": item.get("publishedAt"),
            "collection": "live",
        })
        if len(stories) >= cap:
            break

    return stories


def fetch_mediastack(
    api_key: str,
    limit: int = DEFAULT_PER_PROVIDER_LIMIT,
    timeout: float = 8.0,
) -> list[dict[str, Any]]:
    """Fetch live news from MediaStack API."""
    cap = max(1, min(limit, MAX_PER_PROVIDER_LIMIT))
    params = {
        "access_key": api_key,
        "languages": "en",
        "limit": str(cap),
        "offset": "0",
    }
    url = f"https://api.mediastack.com/v1/news?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBias/1.0)"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw_articles = data.get("data", []) if isinstance(data, dict) else []
    stories: list[dict[str, Any]] = []

    for item in raw_articles:
        if not isinstance(item, dict):
            continue
        article_url = item.get("url")
        title = (item.get("title") or "").strip()
        if not title or not is_valid_http_url(article_url):
            continue

        source_name = item.get("source")
        category = item.get("category")
        section = category.title() if isinstance(category, str) and category else "General"
        image_url = item.get("image")
        valid_image = image_url if is_valid_http_url(image_url) else None

        stories.append({
            "id": make_story_id(PROVIDER_MEDIASTACK, article_url),
            "title": title,
            "source": source_name or "MediaStack",
            "section": section,
            "description": (item.get("description") or "").strip(),
            "content": None,
            "url": article_url,
            "imageUrl": valid_image,
            "publishedAt": item.get("published_at"),
            "collection": "live",
        })
        if len(stories) >= cap:
            break

    return stories


def fetch_rapidapi(
    api_key: str,
    limit: int = DEFAULT_PER_PROVIDER_LIMIT,
    timeout: float = 8.0,
) -> list[dict[str, Any]]:
    """Fetch live news from RapidAPI Real-Time News Data."""
    cap = max(1, min(limit, MAX_PER_PROVIDER_LIMIT))
    params = {
        "topic": "HEADLINES",
        "limit": str(cap),
        "country": "IN",
        "lang": "en",
    }
    url = (
        "https://real-time-news-data.p.rapidapi.com/topic-news-by-section"
        f"?{urllib.parse.urlencode(params)}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; NewsBias/1.0)",
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "real-time-news-data.p.rapidapi.com",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw_articles = data.get("data", []) if isinstance(data, dict) else []
    stories: list[dict[str, Any]] = []

    for item in raw_articles:
        if not isinstance(item, dict):
            continue
        article_url = item.get("link") or item.get("url")
        title = (item.get("title") or "").strip()
        if not title or not is_valid_http_url(article_url):
            continue

        source_name = item.get("source_name") or item.get("source_title") or item.get("source")
        image_url = item.get("photo_url") or item.get("thumbnail_url")
        valid_image = image_url if is_valid_http_url(image_url) else None

        stories.append({
            "id": make_story_id(PROVIDER_RAPIDAPI, article_url),
            "title": title,
            "source": source_name or "RapidAPI",
            "section": "National",
            "description": (item.get("snippet") or item.get("description") or "").strip(),
            "content": None,
            "url": article_url,
            "imageUrl": valid_image,
            "publishedAt": item.get("published_datetime_utc") or item.get("published_at"),
            "collection": "live",
        })
        if len(stories) >= cap:
            break

    return stories


PROVIDER_FETCHERS: dict[str, Callable[[str, int, float], list[dict[str, Any]]]] = {
    PROVIDER_GNEWS: fetch_gnews,
    PROVIDER_NEWSAPI: fetch_newsapi,
    PROVIDER_MEDIASTACK: fetch_mediastack,
    PROVIDER_RAPIDAPI: fetch_rapidapi,
}


# ── Server-Side Cache & Aggregation ───────────────────────────────────────────

class LiveNewsCache:
    """Thread-safe in-memory cache for live news feed responses."""

    def __init__(self, ttl_seconds: float = DEFAULT_CACHE_TTL):
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            cached_time, data = entry
            if time.time() - cached_time > self.ttl:
                del self._cache[key]
                return None
            return data

    def set(self, key: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = (time.time(), data)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


GLOBAL_LIVE_NEWS_CACHE = LiveNewsCache()


def get_live_news_feed(
    limit_per_provider: int = DEFAULT_PER_PROVIDER_LIMIT,
    force_refresh: bool = False,
    providers: list[str] | None = None,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Retrieve normalized live news from configured news providers.

    Guarantees:
    - Bounded result count per provider (capped at limit_per_provider).
    - Page 1 only.
    - Caches responses server-side with a short TTL unless force_refresh=True.
    - Resilient to individual provider failures (returns partial results).
    - Never leaks or logs provider credentials.
    """
    cap = max(1, min(limit_per_provider, MAX_PER_PROVIDER_LIMIT))
    selected_providers = providers or ALL_PROVIDERS
    cache_key = f"{','.join(sorted(selected_providers))}:{cap}"

    if not force_refresh:
        cached = GLOBAL_LIVE_NEWS_CACHE.get(cache_key)
        if cached is not None:
            result = dict(cached)
            result["cached"] = True
            return result

    keys = get_provider_keys()
    provider_status: dict[str, Any] = {}
    all_stories: list[dict[str, Any]] = []

    for provider in selected_providers:
        fetcher = PROVIDER_FETCHERS.get(provider)
        if not fetcher:
            continue

        api_key = keys.get(provider)
        if not api_key:
            provider_status[provider] = {"status": "unconfigured", "count": 0}
            continue

        try:
            stories = fetcher(api_key, limit=cap, timeout=timeout)
            provider_status[provider] = {"status": "ok", "count": len(stories)}
            all_stories.extend(stories)
        except Exception as exc:
            # Safe logging: log exception type name only, never any URL with query params
            err_name = type(exc).__name__
            logger.warning("Live news provider '%s' failed: %s", provider, err_name)
            provider_status[provider] = {
                "status": "error",
                "error": f"Provider temporarily unavailable ({err_name})",
                "count": 0,
            }

    # Interleave or sort stories so one provider does not completely drown out others
    # Stories with valid publication timestamps are ordered newest first; others follow.
    def sort_key(s: dict[str, Any]) -> str:
        return str(s.get("publishedAt") or "")

    sorted_stories = sorted(all_stories, key=sort_key, reverse=True)

    result_payload = {
        "stories": sorted_stories,
        "providers": provider_status,
        "cached": False,
        "total": len(sorted_stories),
        "updated_at": datetime.now(UTC).isoformat(),
    }

    GLOBAL_LIVE_NEWS_CACHE.set(cache_key, result_payload)
    return result_payload
