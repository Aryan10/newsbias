"""Sentence-level linguistic bias model inference.

Evaluates sentences with fine-tuned sentence bias classifiers.
Supports configured models via SENTENCE_MODEL_PATH and cached pipeline inference.
"""

import functools
import os
from pathlib import Path

DEFAULT_IN_HOUSE_MODEL = "/mnt/OS/newsbias-training/models/newsbias_sentence_full"
DEFAULT_FALLBACK_MODEL = "mediabiasgroup/da-roberta-babe-ft"

DEFAULT_SENTENCE_MODEL = os.getenv(
    "SENTENCE_MODEL_PATH",
    DEFAULT_IN_HOUSE_MODEL if Path(DEFAULT_IN_HOUSE_MODEL).is_dir() else DEFAULT_FALLBACK_MODEL,
)


@functools.lru_cache(maxsize=4)
def _get_sentence_pipeline(model_name: str, device_id: int):
    """Cached pipeline instance so model weights stay warm in memory."""
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=model_name,
        device=device_id,
        truncation=True,
        max_length=512,
        top_k=None,
    )


def predict_sentence_bias(
    texts,
    model_name=None,
    device=None,
    batch_size=32,
):
    """Predict whether each sentence is biased or non-biased using a fine-tuned classifier.

    Returns a list of dicts:
    [
        {"text": "...", "label": "biased", "is_biased": True, "score": 0.87},
        ...
    ]
    """
    if not texts:
        return []

    try:
        import torch
    except ImportError as e:
        raise RuntimeError("Install torch and transformers to run sentence bias prediction") from e

    active_model = model_name or os.getenv("SENTENCE_MODEL_PATH") or DEFAULT_SENTENCE_MODEL

    device_id = -1
    if device is None and torch.cuda.is_available():
        device_id = 0
    elif isinstance(device, str) and "cuda" in device:
        device_id = 0

    pipe = _get_sentence_pipeline(active_model, device_id)
    raw_outputs = pipe(texts, batch_size=batch_size)
    results = []

    for text, out_row in zip(texts, raw_outputs, strict=True):
        # Map label names (e.g. 'LABEL_1', 'BIASED', 'biased', etc.)
        score_dict = {d["label"].lower(): float(d["score"]) for d in out_row}
        bias_score = (
            score_dict.get("biased")
            or score_dict.get("bias")
            or score_dict.get("label_1")
            or 0.0
        )
        non_bias_score = (
            score_dict.get("non-biased")
            or score_dict.get("non_biased")
            or score_dict.get("label_0")
            or (1.0 - bias_score)
        )

        is_biased = bias_score > non_bias_score
        results.append({
            "text": text,
            "label": "biased" if is_biased else "non-biased",
            "is_biased": is_biased,
            "score": round(bias_score, 4),
            "probabilities": {
                "biased": round(bias_score, 4),
                "non-biased": round(non_bias_score, 4),
            },
        })
    return results


def predict_sentence_bias_mock(texts):
    """Deterministic mock for offline test verification without downloading weights."""
    keywords = {"reckless", "attacked", "slammed", "blasted", "corrupt", "disastrous", "radical"}
    results = []
    for text in texts:
        has_kw = any(w in text.lower() for w in keywords)
        score = 0.85 if has_kw else 0.10
        results.append({
            "text": text,
            "label": "biased" if has_kw else "non-biased",
            "is_biased": has_kw,
            "score": score,
            "probabilities": {
                "biased": score,
                "non-biased": round(1.0 - score, 2),
            },
        })
    return results
