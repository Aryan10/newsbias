"""Unit and integration tests for NewsBias standalone FastAPI web server."""

import importlib

import pytest
from fastapi.testclient import TestClient
from newsbias.web.app import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(importlib.import_module("newsbias.web.app"), "get_openai_client", lambda: None)
    return TestClient(app)


def test_samples_endpoint(client):
    """Ensure benchmark samples are properly exposed."""
    response = client.get("/bias/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) >= 3
    for s in samples:
        assert "id" in s
        assert "dataset" in s
        assert "title" in s
        assert "text" in s


def test_analyze_endpoint_mock(client):
    """Ensure multi-specialist analysis works with mock predictors."""
    payload = {
        "text": "The radical extremists completely captured party leadership.",
        "title": "Partisan Strife",
        "use_mock": True,
    }
    response = client.post("/bias/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "sentences" in data
    assert "specialist_sentence" in data
    assert "specialist_political" in data
    assert "arbiter" in data

    arbiter = data["arbiter"]
    assert "why_biased_heading" in arbiter
    # Dynamic heading requirement: strictly "Why is it biased?" or "Why is it not biased?"
    heading = arbiter["why_biased_heading"]
    assert "(not)" not in heading
    assert heading in ["Why is it biased?", "Why is it not biased?"]


def test_analyze_stream_endpoint_mock(client):
    """Ensure SSE streaming returns progressive chunks (sentence, political, isolated_llm, complete)."""
    import json
    payload = {
        "text": "Unique fresh text for streaming evaluation.",
        "title": "Stream Fresh",
        "use_mock": True,
    }
    response = client.post("/bias/analyze/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    chunks = []
    for line in response.text.split("\n\n"):
        if line.startswith("data: "):
            chunks.append(json.loads(line[6:]))

    assert len(chunks) == 4
    stages = [c["stage"] for c in chunks]
    assert set(stages[:3]) == {"sentence_specialist", "political_specialist", "isolated_llm"}
    assert stages[3] == "complete"
    chunk_by_stage = {c["stage"]: c for c in chunks[:3]}
    assert "sentences" in chunk_by_stage["sentence_specialist"]
    assert "specialist_political" in chunk_by_stage["political_specialist"]
    assert "isolated_llm" in chunk_by_stage["isolated_llm"]
    assert "arbiter" in chunks[3]["data"]


def test_serve_index(client):
    """Ensure root path serves the built React frontend application."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "NewsBias" in response.text or "vite" in response.text or "root" in response.text
