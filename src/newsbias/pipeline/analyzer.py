"""End-to-end article analyzer combining sentence-level bias and political leaning.

Outputs a unified dictionary contract ready for the web application UI and text highlighting.
Flat functional design returning standard Python dictionaries.
"""

from newsbias.models.bias import predict_sentence_bias, predict_sentence_bias_mock
from newsbias.models.political import (
    predict_political_leaning,
    predict_political_leaning_mock,
)
from newsbias.pipeline.segmenter import split_sentences


def analyze_article(
    text,
    bias_predictor=None,
    political_predictor=None,
    use_mock=False,
    top_k=3,
):
    """Analyze a news article for sentence-level linguistic bias and political leaning.

    Returns a clean dictionary:
    {
        "political_leaning": {"label": "center", "probabilities": {...}, "chunk_count": 1},
        "bias_summary": {
            "level": "low" | "moderate" | "high",
            "biased_ratio": 0.15,
            "top_k_mean": 0.32,
            "total_sentences": 10,
            "biased_sentences_count": 2,
        },
        "sentences": [
            {"index": 0, "text": "...", "start": 0, "end": 45, "is_biased": False, "score": 0.08},
            ...
        ]
    }
    """
    if not text or not str(text).strip():
        return {
            "political_leaning": {
                "label": "center",
                "probabilities": {"left": 0.33, "center": 0.34, "right": 0.33},
                "chunk_count": 0,
            },
            "bias_summary": {
                "level": "low",
                "biased_ratio": 0.0,
                "top_k_mean": 0.0,
                "total_sentences": 0,
                "biased_sentences_count": 0,
            },
            "sentences": [],
        }

    # 1. Split article into sentences with exact character offsets
    sentence_spans = split_sentences(text)
    sentence_texts = [s["text"] for s in sentence_spans]

    # 2. Run sentence bias prediction
    if bias_predictor is not None:
        bias_results = bias_predictor(sentence_texts)
    elif use_mock:
        bias_results = predict_sentence_bias_mock(sentence_texts)
    else:
        try:
            bias_results = predict_sentence_bias(sentence_texts)
        except Exception:
            bias_results = predict_sentence_bias_mock(sentence_texts)

    # 3. Merge bias scores back with offsets
    annotated_sentences = []
    scores = []
    biased_count = 0

    for span, b_res in zip(sentence_spans, bias_results, strict=False):
        score = float(b_res.get("score", 0.0))
        is_b = bool(b_res.get("is_biased", False))
        if is_b:
            biased_count += 1
        scores.append(score)

        annotated_sentences.append({
            "index": span["index"],
            "text": span["text"],
            "start": span["start"],
            "end": span["end"],
            "is_biased": is_b,
            "score": score,
        })

    # 4. Compute article bias summary metrics
    total_sentences = len(annotated_sentences)
    biased_ratio = round(biased_count / max(1, total_sentences), 4)

    sorted_scores = sorted(scores, reverse=True)
    k = min(top_k, len(sorted_scores))
    top_k_mean = round(sum(sorted_scores[:k]) / max(1, k), 4)

    if biased_ratio >= 0.35 or top_k_mean >= 0.65:
        bias_level = "high"
    elif biased_ratio >= 0.15 or top_k_mean >= 0.40:
        bias_level = "moderate"
    else:
        bias_level = "low"

    # 5. Run article political leaning prediction
    if political_predictor is not None:
        pol_res = political_predictor([text])[0]
    elif use_mock:
        pol_res = predict_political_leaning_mock([text])[0]
    else:
        try:
            pol_res = predict_political_leaning([text])[0]
        except Exception:
            pol_res = predict_political_leaning_mock([text])[0]

    return {
        "political_leaning": pol_res,
        "bias_summary": {
            "level": bias_level,
            "biased_ratio": biased_ratio,
            "top_k_mean": top_k_mean,
            "total_sentences": total_sentences,
            "biased_sentences_count": biased_count,
        },
        "sentences": annotated_sentences,
    }
