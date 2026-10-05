"""Unit tests for standalone Jinja prompt engine in newsbias."""

import pytest
from newsbias.prompts import JinjaPrompt, extract_json_payload


def test_extract_json_payload():
    raw_markdown = """```json
    {
      "rewritten_text": "Sample text.",
      "changes": [{"original": "bad", "rewritten": "good", "reason": "neutralize"}]
    }
    ```"""
    parsed = extract_json_payload(raw_markdown)
    assert parsed["rewritten_text"] == "Sample text."
    assert len(parsed["changes"]) == 1


def test_load_and_render_zero_shot():
    p = JinjaPrompt.load("zero_shot_bias")
    assert p.metadata.get("name") == "zero_shot_bias"
    rendered = p.render(title="Test", content="Some news text.")
    assert "Test" in rendered.user
    assert "Some news text." in rendered.user
    assert "System" not in rendered.user
    assert len(rendered.system) > 20


def test_load_and_render_arbiter():
    p = JinjaPrompt.load("bias_detection")
    rendered = p.render(
        title="Test Heading",
        content="Testing content",
        sentence_specialist={"label": "BIASED", "biased_sentence_count": 1, "total_sentences": 2, "bias_ratio": 0.5},
        political_specialist={"label": "RIGHT", "confidence": 0.88},
    )
    assert "Testing content" in rendered.user
    assert "da-roberta-babe-ft" in rendered.user
    assert "AllSides-DeBERTa" in rendered.user


def test_load_and_render_bias_removal():
    p = JinjaPrompt.load("bias_removal")
    rendered = p.render(
        original_text="The radical extremists completely captured party leadership.",
        why_biased_explanation="Sensational wording.",
        detected_leaning="right",
        key_findings=["radical extremists"],
    )
    assert "The radical extremists" in rendered.user
    assert "Sensational wording." in rendered.user
    assert "Target Stance: Neutral" in rendered.user
