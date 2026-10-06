"""Configured text classification through the OpenAI chat-completions API."""

import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset

from newsbias.models.base import Model, Prediction

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIChatClassifier(Model):
    """Use a chat-completions endpoint as a configurable zero-shot classifier.

    Task labels and instructions are supplied by adapter configuration, keeping the
    implementation reusable across datasets and classification taxonomies.
    """

    def __init__(
        self,
        *,
        labels: Sequence[str],
        system_prompt: str,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_output_tokens: int = 512,
        max_retries: int = 3,
        concurrency: int = 4,
        max_chars: int = 24_000,
        client: "OpenAI | None" = None,
    ) -> None:
        """Configure task instructions, endpoint settings, and request behavior.

        `labels` and `system_prompt` define the classification task. The model name
        comes from `model_name`, then `OPENAI_MODEL`, then defaults to `gpt-5-mini`.
        The endpoint comes from `base_url`, then `OPENAI_BASE_URL`, then the SDK
        default. The API key comes from `api_key` or `OPENAI_API_KEY`. Requests run
        concurrently up to `concurrency` (default 4), retry up to `max_retries`
        times, allow `max_output_tokens`, and truncate inputs at `max_chars`.
        Supplying `client` bypasses SDK construction for tests/custom transports.
        """
        self.labels = tuple(str(label).strip().lower() for label in labels)
        if (
            not self.labels
            or any(not label for label in self.labels)
            or len(set(self.labels)) != len(self.labels)
        ):
            raise ValueError("labels must contain unique, nonempty values")
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        self.system_prompt = system_prompt
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.max_chars = max_chars
        if max_output_tokens < 1 or max_retries < 1 or concurrency < 1 or max_chars < 1:
            raise ValueError("Token limit, retries, concurrency, and max_chars must be positive")
        self._client = client or self._create_client(api_key)

    def _create_client(self, api_key: str | None) -> "OpenAI":
        """Construct an OpenAI SDK client without logging or persisting its credential.

        An explicit `api_key` wins over `OPENAI_API_KEY`; `base_url` was resolved
        during initialization from its argument or `OPENAI_BASE_URL`.
        """
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("Set OPENAI_API_KEY in the environment")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install the 'llm' extra to use the OpenAI adapter") from error
        return OpenAI(
            api_key=resolved_key,
            base_url=self.base_url or None,
            timeout=30.0,
            max_retries=0,
        )

    def fit(self, train: Dataset, validation: Dataset | None = None) -> None:
        """Reject training because this adapter only issues hosted inference requests."""
        raise NotImplementedError("OpenAI chat classifier does not support training")

    def _predict_one(self, text: str) -> Prediction:
        """Classify one text, normalize reported scores, and retain safe failure metadata.

        The configured prompt requests JSON containing a label and class scores.
        Scores are normalized to sum to one and marked as uncalibrated. Request or
        parsing failures are retried; after the final failure the result uses label
        `unknown` and stores only the exception type, never raw provider error text.
        """
        truncated = text[:self.max_chars]
        user_content = f"Article text:\n{truncated}"
        last_error = "No response returned"
        started = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    max_completion_tokens=self.max_output_tokens,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                content = response.choices[0].message.content or ""
                decoded = self._parse_response(content)
                requested_label = str(decoded.get("predicted_label", "")).strip().lower()
                if requested_label not in self.labels:
                    raise ValueError(f"Unknown predicted_label: {requested_label!r}")
                raw_scores = decoded.get("scores", {})
                if not isinstance(raw_scores, dict):
                    raw_scores = {}
                normalized_raw_scores = {
                    str(label).strip().lower(): value for label, value in raw_scores.items()
                }
                scores: dict[str, float] = {}
                for label in self.labels:
                    scores[label] = max(0.0, float(normalized_raw_scores.get(label, 0.0)))
                total = sum(scores.values())
                if total <= 0:
                    scores = {label: float(label == requested_label) for label in self.labels}
                else:
                    scores = {label: score / total for label, score in scores.items()}
                usage = getattr(response, "usage", None)
                prediction: Prediction = {
                    "label": requested_label,
                    "probabilities": scores,
                    "probability_source": "llm_reported_uncalibrated",
                    "latency_seconds": time.perf_counter() - started,
                    "text_truncated": len(text) > self.max_chars,
                }
                if usage is not None:
                    prediction["usage"] = {
                        "prompt_tokens": int(usage.prompt_tokens or 0),
                        "completion_tokens": int(usage.completion_tokens or 0),
                    }
                return prediction
            except Exception as error:
                # Avoid persisting SDK error text, which can echo credential prefixes.
                last_error = type(error).__name__
                if attempt + 1 < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        return {
            "label": "unknown",
            "probabilities": None,
            "error": last_error,
            "probability_source": "unavailable",
            "text_truncated": len(text) > self.max_chars,
            "latency_seconds": time.perf_counter() - started,
        }

    @staticmethod
    def _parse_response(content: str) -> dict[str, object]:
        """Extract a JSON object from direct, fenced, or prose-prefaced model output.

        Invalid JSON and non-object results raise so `_predict_one` can retry and
        ultimately record an explicit failed prediction.
        """
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                raise
            decoded = json.loads(content[start:end + 1])
        if not isinstance(decoded, dict):
            raise ValueError("Expected a JSON object")
        return decoded

    def predict(self, dataset: Dataset, *, batch_size: int = 128) -> list[Prediction]:
        """Classify every dataset text in input order and return one prediction per row.

        `batch_size` is accepted for compatibility with other Model adapters. API
        concurrency is controlled by the constructor's `concurrency` parameter.
        Failed API calls are represented as explicit `unknown` predictions.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        texts = list(dataset["text"])
        if self.concurrency == 1:
            return [self._predict_one(text) for text in texts]
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(self._predict_one, texts))

    def save(self, path: Path) -> None:
        """Reject local export because this adapter stores request settings, not weights."""
        raise NotImplementedError("OpenAI chat adapter has no local model bundle")
