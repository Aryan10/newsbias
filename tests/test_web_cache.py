"""Regression coverage for caching editable articles independently of sample selection."""

import importlib

import pytest
from fastapi.testclient import TestClient

web = importlib.import_module("newsbias.web.app")


@pytest.fixture
def cached_client(monkeypatch):
    # Seed a known result independently of the user's disposable web cache.
    sample = web.BENCHMARK_SAMPLES[0]
    seeded = {
        "input": {"title": sample["title"], "text": sample["text"]},
        "sentences": [{"text": sample["text"], "is_biased": True, "score": 0.9}],
    }
    monkeypatch.setattr(web, "ANALYSIS_CACHE", {
        web.get_cache_key(sample["title"], sample["text"]): seeded,
    })
    monkeypatch.setattr(web, "get_openai_client", lambda: None)
    calls = []

    def analyze(text, *, use_mock=False):
        calls.append((text, use_mock))
        return {
            "political_leaning": {
                "label": "center",
                "probabilities": {"left": 0.2, "center": 0.6, "right": 0.2},
            },
            "sentences": [{"text": text, "is_biased": False, "score": 0.1}],
        }

    monkeypatch.setattr(web, "analyze_article", analyze)
    with TestClient(web.app) as client:
        yield client, calls


@pytest.mark.parametrize("field", ["title", "text"])
def test_editing_selected_benchmark_reanalyzes_current_input(cached_client, field):
    client, calls = cached_client
    sample = web.BENCHMARK_SAMPLES[0]
    payload = {"title": sample["title"], "text": sample["text"], "sample_id": sample["id"]}

    original = client.post("/bias/analyze", json=payload)
    assert original.status_code == 200
    assert calls == []  # Unedited samples still use precomputed analyses.

    payload[field] = "An edited article with different context."
    edited = client.post("/bias/analyze", json=payload)
    assert edited.status_code == 200
    assert calls == [(payload["text"], False)]
    assert edited.json()["input"] == {"title": payload["title"], "text": payload["text"]}
    assert edited.json()["sentences"][0]["text"] == payload["text"]

    # Selection is UI metadata: changing or removing it cannot affect cache identity.
    for sample_id in ("another-sample", None):
        repeated = client.post("/bias/analyze", json={**payload, "sample_id": sample_id})
        assert repeated.json() == edited.json()
    assert len(calls) == 1

    restored = client.post("/bias/analyze", json={
        "title": sample["title"], "text": sample["text"], "sample_id": sample["id"],
    })
    assert restored.json() == original.json()
    assert len(calls) == 1


@pytest.mark.parametrize("second", [
    {"title": "Different title", "text": "Gamma"},
    {"title": "alpha\nBeta", "text": "Gamma"},
    {"title": "Alpha", "text": "Beta\nGamma"},
    {"title": "Alpha\nBeta", "text": "Different body"},
])
def test_runtime_cache_requires_both_title_and_text(cached_client, second):
    client, calls = cached_client
    first = {"title": "Alpha\nBeta", "text": "Gamma", "sample_id": "selected"}
    assert client.post("/bias/analyze", json=first).status_code == 200
    response = client.post("/bias/analyze", json={**second, "sample_id": "selected"})
    assert response.status_code == 200
    assert response.json()["input"] == second
    assert len(calls) == 2

    # Repeated content hits the cache even with a different selected sample.
    repeated = client.post("/bias/analyze", json={**second, "sample_id": "elsewhere"})
    assert repeated.json() == response.json()
    assert len(calls) == 2


def test_precomputed_cache_does_not_override_mock_mode(cached_client):
    client, calls = cached_client
    sample = web.BENCHMARK_SAMPLES[0]
    payload = {"title": sample["title"], "text": sample["text"]}
    assert client.post("/bias/analyze", json=payload).status_code == 200
    assert calls == []

    response = client.post("/bias/analyze", json={**payload, "use_mock": True})
    assert response.status_code == 200
    assert calls == [(sample["text"], True)]
    repeated = client.post("/bias/analyze", json={**payload, "use_mock": True})
    assert repeated.json() == response.json()
    assert len(calls) == 1
