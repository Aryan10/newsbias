"""Tests for article-level political leaning model inference."""

from pathlib import Path

from newsbias.models.political import (
    DEFAULT_ARTICLE_MODEL,
    DEFAULT_FALLBACK_MODEL,
    _is_article_hierarchy,
    predict_political_leaning,
    predict_political_leaning_mock,
)


def test_predict_political_leaning_mock() -> None:
    """Mock predictor assigns partisan probabilities based on lexical signals."""
    articles = [
        "Conservative Republicans voted for tax cuts.",
        "Liberal Democrats opposed the bill.",
        "The committee met to review annual reports.",
    ]
    results = predict_political_leaning_mock(articles)
    assert len(results) == 3
    assert results[0]["label"] == "right"
    assert results[1]["label"] == "left"
    assert results[2]["label"] == "center"
    for r in results:
        assert set(r["probabilities"].keys()) == {"left", "center", "right"}
        assert r["method"] == "mock"


def test_is_article_hierarchy_detection(tmp_path: Path) -> None:
    """Verifies that an in-house model directory is correctly detected."""
    cfg = tmp_path / "config.json"
    cfg.write_text('{"model_type": "newsbias_article"}')
    assert _is_article_hierarchy(str(tmp_path)) is True
    assert _is_article_hierarchy("/nonexistent/path") is False


def test_predict_political_leaning_empty() -> None:
    """Empty list returns empty list."""
    assert predict_political_leaning([]) == []
