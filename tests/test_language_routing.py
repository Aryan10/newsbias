"""Unsupported article languages use the LLM without fictional transformer evidence."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from newsbias.language import detect_article_language

web = importlib.import_module("newsbias.web.app")
ENGLISH = (
    "The city council announced a new transport policy after a public meeting. "
    "Officials said residents would receive full details before the next vote."
)
TAMIL = (
    "தமிழ்நாடு அரசு புதிய போக்குவரத்து திட்டத்தை அறிவித்துள்ளது. "
    "இந்த திட்டத்தின் விவரங்கள் அடுத்த வாரம் வெளியிடப்படும் என்று அதிகாரிகள் தெரிவித்தனர்."
)
HINDI = (
    "सरकार ने नई परिवहन योजना की घोषणा की है। "
    "अधिकारियों ने बताया कि अगले सप्ताह योजना का पूरा विवरण प्रकाशित किया जाएगा।"
)
SPANISH = (
    "El gobierno anunció una nueva política de transporte después de una reunión pública. "
    "Los funcionarios dijeron que publicarían los detalles completos la próxima semana."
)


@pytest.mark.parametrize(("text", "code"), [(TAMIL, "ta"), (HINDI, "hi"), (SPANISH, "es")])
def test_nonenglish_detection(text, code):
    language = detect_article_language(text)
    assert language.code == code
    assert not language.transformers_supported


def test_english_and_uncertain_detection():
    assert detect_article_language(ENGLISH).transformers_supported
    assert detect_article_language("1234").status == "uncertain"
    assert not detect_article_language("1234").transformers_supported


def test_substantial_mixed_language_text_skips_english_checkpoints():
    language = detect_article_language(ENGLISH + "\n\n" + TAMIL)
    assert language.status == "mixed"
    assert not language.transformers_supported


@pytest.fixture
def language_client(monkeypatch):
    monkeypatch.setattr(web, "ANALYSIS_CACHE", {})
    pipeline = Mock(return_value={
        "political_leaning": {"label": "center", "probabilities": {"center": 0.8}},
        "sentences": [{"text": ENGLISH, "is_biased": False, "score": 0.1}],
    })
    monkeypatch.setattr(web, "analyze_article", pipeline)
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        if "Supreme Arbiter" in kwargs["messages"][0]["content"]:
            result = {
                "final_verdict": {
                    "is_biased": True, "overall_leaning": "center", "confidence": 0.9,
                },
                "why_biased_explanation": "LLM-only reading with original-language evidence.",
                "key_findings": ["Original-language loaded phrasing."],
            }
        else:
            result = {
                "political_leaning": {"label": "center", "confidence": 0.91},
                "linguistic_bias": {"is_biased": True, "label": "biased", "bias_score": 0.8},
                "loaded_phrases": ["Original-language phrase"],
                "reasoning": "Original-language LLM evidence.",
            }
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(result),
        ))])

    completions = SimpleNamespace(create=complete)
    client = SimpleNamespace(
        default_model="gpt-6-luna", chat=SimpleNamespace(completions=completions),
    )
    monkeypatch.setattr(web, "get_openai_client", lambda: client)
    with TestClient(web.app) as api:
        yield api, pipeline, calls, completions


@pytest.mark.parametrize("text", [TAMIL, HINDI, SPANISH, ENGLISH + "\n\n" + TAMIL])
def test_unsupported_languages_do_not_run_transformers(language_client, text):
    api, pipeline, calls, _ = language_client
    response = api.post("/bias/analyze", json={"title": "English title", "text": text})
    assert response.status_code == 200
    data = response.json()
    assert data["analysis_method"] == "llm_only"
    assert data["specialist_sentence"]["status"] == "skipped"
    assert data["specialist_sentence"]["bias_ratio"] is None
    assert data["specialist_sentence"]["biased_sentence_count"] is None
    assert data["specialist_political"]["status"] == "skipped"
    assert data["specialist_political"]["confidence"] is None
    assert data["specialist_political"]["scores"] == {}
    assert all(sentence["is_biased"] is None and sentence["score"] is None
               for sentence in data["sentences"])
    pipeline.assert_not_called()
    assert len(calls) == 2
    evidence = calls[1]["messages"][1]["content"]
    assert "NOT EVALUATED" in evidence
    assert "Biased Sentences Detected:" not in evidence
    assert api.post("/bias/analyze", json={"title": "English title", "text": text}).json() == data
    assert len(calls) == 2


def test_supported_english_keeps_specialists(language_client):
    api, pipeline, _, _ = language_client
    data = api.post("/bias/analyze", json={"text": ENGLISH}).json()
    assert data["analysis_method"] == "specialists+llm"
    assert data["specialist_sentence"]["status"] == "available"
    assert data["specialist_political"]["confidence"] == 0.8
    pipeline.assert_called_once_with(ENGLISH, use_mock=False)


def test_llm_failure_is_not_reported_as_a_neutral_article(language_client, monkeypatch):
    api, pipeline, _, _ = language_client
    monkeypatch.setattr(web, "get_openai_client", lambda: None)
    response = api.post("/bias/analyze", json={"text": TAMIL})
    assert response.status_code == 503
    assert "LLM analysis is required" in response.json()["detail"]
    pipeline.assert_not_called()
    assert not web.ANALYSIS_CACHE


def test_arbiter_failure_uses_llm_evidence_not_skipped_specialists(language_client):
    api, pipeline, _, completions = language_client
    original = completions.create

    def complete(**kwargs):
        if "Supreme Arbiter" in kwargs["messages"][0]["content"]:
            raise RuntimeError("Arbiter unavailable")
        return original(**kwargs)

    completions.create = complete
    data = api.post("/bias/analyze", json={"text": TAMIL}).json()
    assert data["arbiter"]["final_verdict"]["is_biased"] is True
    assert data["arbiter"]["final_verdict"]["confidence"] == 0.91
    assert data["arbiter"]["why_biased_explanation"] == "Original-language LLM evidence."
    pipeline.assert_not_called()


def test_old_cache_profile_cannot_reuse_unsupported_predictions(language_client, monkeypatch):
    api, pipeline, calls, _ = language_client
    version = web.ANALYSIS_VERSION
    with monkeypatch.context() as old:
        old.setattr(web, "ANALYSIS_VERSION", version - 1)
        old_key = web.get_cache_key("", TAMIL)
    web.ANALYSIS_CACHE[old_key] = {"input": {"text": TAMIL}, "arbiter": {"fake_old_result": True}}
    data = api.post("/bias/analyze", json={"text": TAMIL}).json()
    assert data["analysis_version"] == version
    assert data["analysis_method"] == "llm_only"
    assert len(calls) == 2
    pipeline.assert_not_called()
