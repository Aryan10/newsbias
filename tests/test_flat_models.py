"""Tests for flat model inference functions and LLM adapter."""

import json
from types import SimpleNamespace

import pytest

from newsbias.models.bias import predict_sentence_bias_mock
from newsbias.models.llm import (
    analyze_article_llm,
    classify_political_llm,
    classify_sentences_llm,
    get_openai_client,
)
from newsbias.models.political import predict_political_leaning_mock


class FakeChatClient:
    """Mock OpenAI client returning fixed structured JSON response."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=self.response_text))]
                )
            )
        )


def test_predict_sentence_bias_mock() -> None:
    """Sentence bias mock correctly flags loaded keywords."""
    res = predict_sentence_bias_mock([
        "The president made a statement.",
        "Critics attacked the reckless plan.",
    ])
    assert len(res) == 2
    assert res[0]["label"] == "non-biased"
    assert res[0]["is_biased"] is False
    assert res[1]["label"] == "biased"
    assert res[1]["is_biased"] is True
    assert res[1]["score"] > 0.5


def test_predict_political_leaning_mock() -> None:
    """Political leaning mock outputs valid probabilities across 3 classes."""
    res = predict_political_leaning_mock([
        "The Democratic party announced healthcare reform.",
        "Republican leaders opposed new spending.",
    ])
    assert len(res) == 2
    assert res[0]["label"] == "left"
    assert "left" in res[0]["probabilities"]
    assert res[1]["label"] == "right"


def test_classify_sentences_llm_mocked() -> None:
    """LLM sentence classifier parses structured JSON results correctly."""
    fake_json = json.dumps({
        "results": [
            {"is_biased": False, "score": 0.05, "label": "non-biased"},
            {"is_biased": True, "score": 0.88, "label": "biased"},
        ]
    })
    client = FakeChatClient(fake_json)
    res = classify_sentences_llm(["Neutral sentence", "Biased sentence"], client=client)
    assert len(res) == 2
    assert res[0]["label"] == "non-biased"
    assert res[1]["label"] == "biased"
    assert res[1]["score"] == 0.88


def test_classify_political_llm_mocked() -> None:
    """LLM political leaning classifier parses 3-way label and scores."""
    fake_json = json.dumps({
        "label": "center",
        "scores": {"left": 0.2, "center": 0.6, "right": 0.2},
    })
    client = FakeChatClient(fake_json)
    res = classify_political_llm("Article text", client=client)
    assert res["label"] == "center"
    assert res["probabilities"]["center"] == 0.6


def test_analyze_article_llm_mocked() -> None:
    """LLM joint single-pass analysis extracts political leaning and biased quotes."""
    fake_json = json.dumps({
        "political_leaning": {"label": "center", "confidence": 0.85},
        "biased_sentences": [{"quote": "reckless move", "reason": "emotive wording", "score": 0.9}],
    })
    client = FakeChatClient(fake_json)
    res = analyze_article_llm("Article text", client=client)
    assert res["political_leaning"]["label"] == "center"
    assert len(res["biased_sentences"]) == 1
    assert res["biased_sentences"][0]["quote"] == "reckless move"


def test_get_openai_client_requires_api_key(monkeypatch) -> None:
    """An API key must be supplied explicitly or through OPENAI_API_KEY."""
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_openai_client()
