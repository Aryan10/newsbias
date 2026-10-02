"""Inference adapter for Hugging Face sequence-classification checkpoints."""

from pathlib import Path
from typing import Any

from datasets import Dataset

from newsbias.models.base import Model, Prediction


class HuggingFaceSequenceClassifier(Model):
    """Run a configured Transformers sequence-classification checkpoint without fitting.

    Checkpoint labels are read from `id2label` (or supplied explicitly). Long inputs
    are tokenized into overlapping windows and their softmax outputs are averaged
    before selecting a class.
    """

    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None = None,
        id2label: dict[int, str] | dict[str, str] | None = None,
        device: str | None = None,
        max_length: int = 512,
        stride: int = 64,
        max_chunks: int = 16,
    ) -> None:
        """Choose the checkpoint, optional revision, device, and long-text window limits.

        `device=None` selects CUDA when available and otherwise CPU. The defaults use
        512-token windows, overlap 64 tokens, and process at most 16 windows per text.
        """
        self.model_name = model_name
        self.revision = revision
        self.id2label = id2label
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.max_chunks = max_chunks
        if max_length < 2 or stride < 0 or stride >= max_length or max_chunks < 1:
            raise ValueError(
                "Require max_length >= 2, 0 <= stride < max_length, and max_chunks >= 1"
            )
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: Any | None = None
        self._labels: dict[int, str] = {}

    def _load(self) -> None:
        """Load optional torch/Transformers components once, on first inference or save.

        Deferring imports keeps the base package lightweight. `id2label` values are
        normalized to lowercase; missing or duplicate class names reject the bundle.
        """
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Install the 'transformers' extra to use Hugging Face classifiers"
            ) from error
        device_name = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, revision=self.revision,
        ).to(self._device)
        self._model.eval()
        if self.id2label is not None:
            self._labels = {
                int(index): str(label).strip().lower()
                for index, label in self.id2label.items()
            }
        else:
            raw_mapping = self._model.config.id2label
            self._labels = {
                int(index): str(label).strip().lower()
                for index, label in raw_mapping.items()
            }
        if not self._labels or len(set(self._labels.values())) != len(self._labels):
            raise ValueError(f"Checkpoint has an invalid label mapping: {self._labels}")

    def fit(self, train: Dataset, validation: Dataset | None = None) -> None:
        """Reject fitting; use a trainable adapter to fine-tune a checkpoint."""
        raise NotImplementedError("Hugging Face checkpoint adapter does not support training")

    def predict(self, dataset: Dataset, *, batch_size: int = 128) -> list[Prediction]:
        """Run each input through bounded sliding windows and average class probabilities.

        Predictions preserve dataset row order. `batch_size` is validated for common
        interface compatibility; window batching is controlled per text by
        `max_chunks`. Each result reports the number of windows used and truncation.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._load()
        import torch

        predictions: list[Prediction] = []
        for text in dataset["text"]:
            encoded = self._tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                stride=self.stride,
                return_overflowing_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            total_chunks = int(encoded["input_ids"].shape[0])
            input_ids = encoded["input_ids"][:self.max_chunks].to(self._device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask[:self.max_chunks].to(self._device)
            with torch.inference_mode():
                output = self._model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(output.logits, dim=-1).mean(dim=0).cpu().tolist()
            scores = {
                self._labels[index]: float(probability)
                for index, probability in enumerate(probabilities)
            }
            predictions.append({
                "label": max(scores, key=scores.__getitem__),
                "probabilities": scores,
                "probability_source": "softmax_mean_over_chunks_uncalibrated",
                "chunk_count": int(input_ids.shape[0]),
                "text_truncated": total_chunks > self.max_chunks,
            })
        return predictions

    def save(self, path: Path) -> None:
        """Export the loaded checkpoint and tokenizer for offline reuse beneath `path`.

        Saving triggers lazy model loading if inference has not run and may write
        several large weight/tokenizer files.
        """
        self._load()
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
