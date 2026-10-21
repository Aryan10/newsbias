"""Rewrite caching must reuse complete requests without mixing edited inputs."""

import importlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

web = importlib.import_module("newsbias.web.app")


@pytest.fixture
def rewrite_client(monkeypatch):
    monkeypatch.setattr(web, "REWRITE_CACHE", {})
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        content = json.dumps({
            "rewritten_text": f"Revision {len(calls)}.\n\nSecond paragraph.",
            "editorial_summary": "Removed loaded phrasing.",
            "changes": [{"original": "reckless", "rewritten": "proposed", "reason": "Neutral"}],
        })
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete)),
    )
    monkeypatch.setattr(web, "get_openai_client", lambda: fake_client)
    with TestClient(web.app) as client:
        yield client, calls


def test_identical_rewrite_requests_reuse_result(rewrite_client):
    client, calls = rewrite_client
    payload = {"text": "A reckless plan.\n\nAnother paragraph."}
    first = client.post("/bias/rewrite", json=payload)
    assert first.status_code == 200
    assert len(calls) == 1

    # Omitted fields and explicitly supplied defaults describe the same request.
    repeated = client.post("/bias/rewrite", json={
        **payload, "why_biased_explanation": "", "detected_leaning": "neutral",
        "key_findings": None,
    })
    assert repeated.status_code == 200
    assert repeated.json() == first.json()
    assert "\n\n" in repeated.json()["rewritten_text"]
    assert len(calls) == 1


@pytest.mark.parametrize(("field", "value"), [
    ("text", "A different article."),
    ("text", "A reckless plan.\n\nAnother paragraph."),
    ("why_biased_explanation", "Different explanation."),
    ("detected_leaning", "left"),
    ("key_findings", ["Changed finding"]),
    ("key_findings", ["Second finding", "First finding"]),
])
def test_rewrite_cache_includes_all_context_fields(rewrite_client, field, value):
    client, calls = rewrite_client
    original = {
        "text": "A reckless plan.\nAnother paragraph.",
        "why_biased_explanation": "Loaded language.",
        "detected_leaning": "right",
        "key_findings": ["First finding", "Second finding"],
    }
    first = client.post("/bias/rewrite", json=original)
    edited_payload = {**original, field: value}
    edited = client.post("/bias/rewrite", json=edited_payload)
    assert first.status_code == edited.status_code == 200
    assert len(calls) == 2
    assert edited.json() != first.json()

    assert client.post("/bias/rewrite", json=edited_payload).json() == edited.json()
    assert client.post("/bias/rewrite", json=original).json() == first.json()
    assert len(calls) == 2


def test_fallback_rewrites_are_cached(rewrite_client, monkeypatch):
    client, _ = rewrite_client
    attempts = []

    def unavailable_client():
        attempts.append(True)
        raise ValueError("No provider configured")

    monkeypatch.setattr(web, "get_openai_client", unavailable_client)
    payload = {"text": "The death knell.\n\nAnother paragraph."}
    first = client.post("/bias/rewrite", json=payload)
    repeated = client.post("/bias/rewrite", json=payload)
    assert first.status_code == repeated.status_code == 200
    assert first.json() == repeated.json()
    assert first.json()["rewritten_text"] == "The major policy shift.\n\nAnother paragraph."
    assert len(attempts) == 1
