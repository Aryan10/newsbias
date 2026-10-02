"""Article-level political leaning model inference with sliding-window chunk pooling.

Evaluates articles with `premsa/political-bias-prediction-allsides-DeBERTa` (or POLITICS).
Flat functional design returning standard Python dictionaries.
"""


def predict_political_leaning(
    articles,
    model_name="premsa/political-bias-prediction-allsides-DeBERTa",
    device=None,
    max_length=512,
    stride=64,
    max_chunks=16,
    id2label=None,
):
    """Predict Left/Center/Right leaning for articles via sliding-window mean-pooling.

    Returns a list of dicts:
    [
        {
            "label": "center",
            "probabilities": {"left": 0.15, "center": 0.70, "right": 0.15},
            "chunk_count": 2,
        },
        ...
    ]
    """
    if not articles:
        return []

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        msg = "Install torch and transformers to run political leaning prediction"
        raise RuntimeError(msg) from e

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev)
    model.eval()

    # Determine label mapping
    if id2label:
        label_map = {int(k): str(v).lower() for k, v in id2label.items()}
    else:
        label_map = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        # Normalize generic LABEL_0/1/2 to standard AllSides classes
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
        input_ids = encoded["input_ids"][:max_chunks].to(dev)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask[:max_chunks].to(dev)

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
        })
    return results
