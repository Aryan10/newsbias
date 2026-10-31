"""Article-level political leaning model inference.

Supports the in-house 2-level hierarchical ModernBERT model (`jhu-clsp/ettin-encoder-400m`)
with cross-passage TransformerEncoder attention and gated pooling, with graceful fallback
to reference DeBERTa chunk pooling.
Supports configured models via ARTICLE_MODEL_PATH and cached model inference.
"""

import functools
import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_IN_HOUSE_MODEL = "/mnt/OS/newsbias-training/models/newsbias_article"
DEFAULT_FALLBACK_MODEL = "premsa/political-bias-prediction-allsides-DeBERTa"

DEFAULT_ARTICLE_MODEL = os.getenv(
    "ARTICLE_MODEL_PATH",
    DEFAULT_IN_HOUSE_MODEL if Path(DEFAULT_IN_HOUSE_MODEL).is_dir() else DEFAULT_FALLBACK_MODEL,
)


def _is_article_hierarchy(model_path: str) -> bool:
    """Check if the path contains a NewsBias hierarchical ArticleModel."""
    cfg_file = Path(model_path) / "config.json"
    if not cfg_file.is_file():
        return False
    try:
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        return (
            data.get("model_type") == "newsbias_article"
            or "ArticleModel" in data.get("architectures", [])
        )
    except Exception:
        return False


@functools.lru_cache(maxsize=2)
def _get_article_hierarchy_model(model_path: str, device_str: str):
    """Cached loader for the in-house hierarchical ArticleModel."""
    import torch
    from transformers import AutoTokenizer

    from newsbias.models.article_hierarchy import ArticleModel

    dev = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=Path(model_path).is_dir())
    model = ArticleModel.from_pretrained(model_path, local_files_only=Path(model_path).is_dir()).to(dev)
    model.eval()
    return model, tokenizer


@functools.lru_cache(maxsize=2)
def _get_flat_political_model(model_name: str, device_str: str):
    """Cached loader for flat sequence classification fallback models."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev)
    model.eval()
    return model, tokenizer


def predict_political_leaning(
    articles,
    model_name=None,
    device=None,
    max_length=512,
    stride=64,
    max_chunks=16,
    id2label=None,
):
    """Predict Left/Center/Right leaning for articles.

    Uses the in-house hierarchical ArticleModel when available, or falls back to
    sliding-window mean-pooling on flat sequence classifiers.

    Returns a list of dicts:
    [
        {
            "label": "center",
            "probabilities": {"left": 0.15, "center": 0.70, "right": 0.15},
            "chunk_count": 2,
            "text_truncated": False,
            "model_name": "newsbias_article",
            "method": "in_house",
        },
        ...
    ]
    """
    if not articles:
        return []

    try:
        import torch
    except ImportError as e:
        msg = "Install torch and transformers to run political leaning prediction"
        raise RuntimeError(msg) from e

    active_model = model_name or os.getenv("ARTICLE_MODEL_PATH") or DEFAULT_ARTICLE_MODEL
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)

    if _is_article_hierarchy(active_model):
        return _predict_article_hierarchy(
            articles=articles,
            model_path=active_model,
            device=dev,
            max_length=max_length,
            overlap=stride,
            max_passages=min(6, max_chunks),
            id2label_override=id2label,
        )

    return _predict_flat_political(
        articles=articles,
        model_name=active_model,
        device=dev,
        max_length=max_length,
        stride=stride,
        max_chunks=max_chunks,
        id2label_override=id2label,
    )


def _predict_article_hierarchy(
    articles,
    model_path,
    device,
    max_length=512,
    overlap=64,
    max_passages=6,
    id2label_override=None,
):
    """Execute prediction using the in-house hierarchical ArticleModel."""
    import torch

    model, tokenizer = _get_article_hierarchy_model(str(model_path), str(device))

    # Resolve labels from model config
    if id2label_override:
        label_map = {int(k): str(v).lower() for k, v in id2label_override.items()}
    elif hasattr(model.config, "id2label") and model.config.id2label:
        label_map = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    else:
        label_map = {0: "left", 1: "center", 2: "right"}

    results = []
    for text in articles:
        clean_text = str(text or "").strip()
        if not clean_text:
            results.append({
                "label": "center",
                "probabilities": {"left": 0.3333, "center": 0.3334, "right": 0.3333},
                "chunk_count": 0,
                "text_truncated": False,
                "model_name": Path(model_path).name,
                "method": "in_house",
            })
            continue

        tokenized = tokenizer(
            clean_text,
            truncation=True,
            max_length=max_length,
            stride=overlap,
            return_overflowing_tokens=True,
            return_token_type_ids=False,
        )["input_ids"]

        total_windows = len(tokenized)
        if total_windows > max_passages:
            positions = np.linspace(0, total_windows - 1, max_passages, dtype=int).tolist()
            windows = [tokenized[i] for i in positions]
        else:
            positions = list(range(total_windows))
            windows = tokenized

        passage_count = len(windows)
        length = max(len(w) for w in windows)
        ids = torch.full((1, passage_count, length), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        pos = torch.zeros((1, passage_count), dtype=torch.long)

        for j, tokens in enumerate(windows):
            ids[0, j, :len(tokens)] = torch.tensor(tokens)
            mask[0, j, :len(tokens)] = 1
            pos[0, j] = positions[j]

        batch = {
            "input_ids": ids.to(device),
            "attention_mask": mask.to(device),
            "passage_positions": pos.to(device),
        }

        with torch.inference_mode():
            logits = model(**batch).logits
            probs = torch.softmax(logits.float(), dim=-1).squeeze(0).cpu().tolist()

        scores = {label_map[i]: round(probs[i], 4) for i in range(len(probs))}
        pred_label = max(scores, key=scores.get)

        results.append({
            "label": pred_label,
            "probabilities": scores,
            "chunk_count": passage_count,
            "text_truncated": total_windows > max_passages,
            "model_name": Path(model_path).name,
            "method": "in_house",
        })

    return results


def _predict_flat_political(
    articles,
    model_name,
    device,
    max_length=512,
    stride=64,
    max_chunks=16,
    id2label_override=None,
):
    """Fallback: Execute prediction using flat sequence classifier with chunk pooling."""
    import torch

    model, tokenizer = _get_flat_political_model(str(model_name), str(device))

    if id2label_override:
        label_map = {int(k): str(v).lower() for k, v in id2label_override.items()}
    else:
        label_map = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        if set(label_map.values()) == {"label_0", "label_1", "label_2"}:
            label_map = {0: "left", 1: "center", 2: "right"}

    results = []
    for text in articles:
        encoded = tokenizer(
            str(text or ""),
            max_length=max_length,
            truncation=True,
            stride=stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        total_chunks = int(encoded["input_ids"].shape[0])
        input_ids = encoded["input_ids"][:max_chunks].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask[:max_chunks].to(device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).mean(dim=0).cpu().tolist()

        scores = {label_map[i]: round(probs[i], 4) for i in range(len(probs))}
        pred_label = max(scores, key=scores.get)

        results.append({
            "label": pred_label,
            "probabilities": scores,
            "chunk_count": int(input_ids.shape[0]),
            "text_truncated": total_chunks > max_chunks,
            "model_name": str(model_name),
            "method": "huggingface",
        })

    return results


def predict_political_leaning_mock(articles):
    """Deterministic mock for offline test verification without downloading weights."""
    results = []
    for text in articles:
        lower = str(text or "").lower()
        if "republican" in lower or "conservative" in lower:
            scores = {"left": 0.1, "center": 0.2, "right": 0.7}
        elif "democrat" in lower or "liberal" in lower or "progressive" in lower:
            scores = {"left": 0.7, "center": 0.2, "right": 0.1}
        else:
            scores = {"left": 0.2, "center": 0.6, "right": 0.2}

        pred_label = max(scores, key=scores.get)
        results.append({
            "label": pred_label,
            "probabilities": scores,
            "chunk_count": 1,
            "text_truncated": False,
            "model_name": "mock",
            "method": "mock",
        })
    return results
