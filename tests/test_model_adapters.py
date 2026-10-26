"""Offline tests for generic pretrained, hosted, and benchmark model adapters."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from newsbias.benchmark import run_model_benchmark
from newsbias.models import HuggingFaceSequenceClassifier, OpenAIChatClassifier, TfidfModel


class FakeCompletions:
    """Capture chat-completion requests and return a deterministic fake response."""

    def __init__(self, response: object) -> None:
        """Store the response to return and initialize the request capture list."""
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        """Record request data and return the configured fake response."""
        self.calls.append(kwargs)
        return self.response


def fake_response(content: str) -> SimpleNamespace:
    """Build the response fields used by the generic OpenAI chat adapter."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def openai_model(client: object, *, max_retries: int = 3) -> OpenAIChatClassifier:
    """Build a small three-label task adapter without making network requests."""
    return OpenAIChatClassifier(
        labels=("left", "center", "right"),
        system_prompt="Classify political leaning. Return JSON labels and scores.",
        model_name="gpt-5-mini",
        client=client,
        max_retries=max_retries,
    )


def test_openai_classifier_prediction_and_usage() -> None:
    """Normalize case-insensitive class scores and preserve provider usage counts."""
    completions = FakeCompletions(fake_response(json.dumps({
        "predicted_label": "LEFT",
        "scores": {"LEFT": 2, "CENTER": 1, "RIGHT": 1},
    })))
    model = openai_model(SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    ))
    dataset = Dataset.from_dict({"text": ["Article text"]})
    prediction = model.predict(dataset)[0]
    assert prediction["label"] == "left"
    assert prediction["probabilities"] == {"left": 0.5, "center": 0.25, "right": 0.25}
    assert prediction["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert prediction["probability_source"] == "llm_reported_uncalibrated"
    assert completions.calls[0]["model"] == "gpt-5-mini"
    assert "Classify political leaning" in completions.calls[0]["messages"][0]["content"]


def test_openai_classifier_retries_then_marks_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry malformed responses and return an explicit unknown prediction on exhaustion."""
    completions = FakeCompletions(fake_response("not-json"))
    model = openai_model(
        SimpleNamespace(chat=SimpleNamespace(completions=completions)), max_retries=2,
    )
    monkeypatch.setattr("newsbias.models.openai.time.sleep", lambda _: None)
    prediction = model.predict(Dataset.from_dict({"text": ["Article"]}))[0]
    assert prediction["label"] == "unknown"
    assert prediction["probabilities"] is None
    assert prediction["error"] == "JSONDecodeError"
    assert len(completions.calls) == 2


def test_openai_failure_does_not_persist_sdk_error_text() -> None:
    """Sanitize provider exceptions before a benchmark can persist failure metadata."""

    class RaisingCompletions:
        """Test double whose request error deliberately includes a secret-like string."""

        def create(self, **kwargs: object) -> object:
            """Raise an error that the model adapter must sanitize."""
            raise RuntimeError("request failed with key=do-not-persist")

    model = openai_model(
        SimpleNamespace(chat=SimpleNamespace(completions=RaisingCompletions())),
        max_retries=1,
    )
    prediction = model.predict(Dataset.from_dict({"text": ["Article"]}))[0]
    assert prediction["error"] == "RuntimeError"
    assert "do-not-persist" not in str(prediction)


def test_openai_classifier_parses_fenced_json() -> None:
    """Accept a valid JSON object surrounded by a markdown code fence."""
    parsed = OpenAIChatClassifier._parse_response('```json\n{"predicted_label":"left"}\n```')
    assert parsed == {"predicted_label": "left"}


def test_huggingface_adapter_custom_id2label() -> None:
    """Allow checkpoint label metadata to be overridden for nonstandard configs."""
    adapter = HuggingFaceSequenceClassifier(
        model_name="mock-model",
        id2label={0: "LEFT", 1: "CENTER", 2: "RIGHT"},
    )
    assert adapter.id2label == {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
    assert adapter.model_name == "mock-model"


def test_inference_only_adapters_reject_training() -> None:
    """Ensure both hosted and pretrained adapters reject unsupported fitting."""
    data = Dataset.from_dict({"text": ["Text"], "label": ["left"]})
    with pytest.raises(NotImplementedError, match="does not support training"):
        HuggingFaceSequenceClassifier(model_name="local-checkpoint").fit(data)
    with pytest.raises(NotImplementedError, match="does not support training"):
        openai_model(object()).fit(data)


def test_model_benchmark_persists_predictions_and_failures(
    dataset: DatasetDict, tmp_path: Path,
) -> None:
    """Persist generic adapter predictions aligned to source IDs and dataset labels."""
    model = TfidfModel(min_df=1)
    model.fit(dataset["train"])
    output = run_model_benchmark(
        model, dataset, model_identifier="test-tfidf", split="test", output_root=tmp_path,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["accuracy"] == 1
    assert summary["metrics"]["failed_predictions"] == 0
    rows = (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(dataset["test"])
    assert Path(output / "predictions.jsonl").exists()
