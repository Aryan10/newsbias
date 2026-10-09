"""Unit and integration tests for NewsBias standalone FastAPI web server."""

import pytest
from fastapi.testclient import TestClient
from newsbias.web.app import app


@pytest.fixture
def client():
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


def test_serve_index(client):
    """Ensure root path serves the built React frontend application."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "NewsBias" in response.text or "vite" in response.text or "root" in response.text
