"""Unified OpenAI/LLM adapter for sentence bias and political leaning.

Can be used in place of the dedicated bias model, political model, or both.
Flat functional design returning standard Python dictionaries.
"""

import json
import os
import re


def get_openai_client(api_key=None, base_url=None, model=None):
    """Initialize OpenAI client from explicit args or environment variables."""
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        for p in [Path(".env"), Path("../.env"), Path(__file__).resolve().parent.parent.parent.parent / ".env"]:
            if p.is_file():
                load_dotenv(p)
                break
    except Exception:
        pass

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY in the environment or pass api_key")
    from openai import OpenAI
    url = base_url or os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=key, base_url=url) if url else OpenAI(api_key=key)
    client.default_model = model or os.getenv("OPENAI_MODEL") or "gpt-5-mini"
    return client


def _extract_json(text):
    """Parse JSON dictionary from raw model text or fenced markdown blocks."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(text.strip())


def classify_sentences_llm(texts, model="gpt-5-mini", client=None):
    """Use an LLM to predict linguistic bias for a list of sentences.

    Returns:
    [
        {"text": "...", "label": "biased", "is_biased": True, "score": 0.9},
        ...
    ]
    """
    if not texts:
        return []

    c = client or get_openai_client()
    system_prompt = (
        "You are an expert linguist analyzing media bias. For each sentence provided, "
        "determine whether it contains biased, subjective, or loaded language. "
        "Respond ONLY with a JSON object: "
        '{"results": [{"is_biased": bool, "score": float, "label": "biased"|"non-biased"}]}'
    )
    user_prompt = "Sentences:\n" + "\n".join(f"{i}: {t}" for i, t in enumerate(texts))

    resp = c.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    parsed = _extract_json(content)
    raw_results = parsed.get("results", [])

    output = []
    fallback = {"is_biased": False, "score": 0.1, "label": "non-biased"}
    for i, t in enumerate(texts):
        item = raw_results[i] if i < len(raw_results) else fallback
        is_b = bool(item.get("is_biased", False))
        score = float(item.get("score", 0.9 if is_b else 0.1))
        output.append({
            "text": t,
            "label": "biased" if is_b else "non-biased",
            "is_biased": is_b,
            "score": round(score, 4),
            "probabilities": {
                "biased": round(score, 4),
                "non-biased": round(1.0 - score, 4),
            },
        })
    return output


def classify_political_llm(article_text, model="gpt-5-mini", client=None):
    """Use an LLM to classify article political leaning into Left, Center, or Right.

    Returns:
    {
        "label": "center",
        "probabilities": {"left": 0.2, "center": 0.6, "right": 0.2},
        "chunk_count": 1,
    }
    """
    c = client or get_openai_client()
    system_prompt = (
        "Classify the overall political leaning of this news article into: left, center, right. "
        "Respond ONLY with a JSON object: "
        '{"label": "left"|"center"|"right", '
        '"scores": {"left": float, "center": float, "right": float}}'
    )
    # Truncate to reasonable token length for context window if needed
    sample_text = str(article_text)[:12000]

    resp = c.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Article Text:\n{sample_text}"},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    parsed = _extract_json(content)
    label = str(parsed.get("label", "center")).strip().lower()
    raw_scores = parsed.get("scores", {"left": 0.33, "center": 0.34, "right": 0.33})
    total = sum(float(v) for v in raw_scores.values()) or 1.0
    probs = {k: round(float(v) / total, 4) for k, v in raw_scores.items()}

    return {
        "label": label if label in ("left", "center", "right") else "center",
        "probabilities": probs,
        "chunk_count": 1,
    }


def analyze_article_llm(article_text, model="gpt-5-mini", client=None):
    """Single-pass joint analysis extracting both political leaning AND biased sentences."""
    c = client or get_openai_client()
    system_prompt = (
        "Analyze the news article for political leaning (left/center/right) and identify any "
        "specific biased or emotionally loaded sentences. "
        "Respond ONLY with JSON: {"
        "\"political_leaning\": {\"label\": \"left\"|\"center\"|\"right\", \"confidence\": float},"
        "\"biased_sentences\": [{\"quote\": string, \"reason\": string, \"score\": float}]"
        "}"
    )
    sample_text = str(article_text)[:12000]
    resp = c.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Article:\n{sample_text}"},
        ],
        temperature=0.0,
    )
    return _extract_json(resp.choices[0].message.content)
