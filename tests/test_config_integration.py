"""Integration tests making live API calls to models configured in APP_CONFIG.

Run explicitly with:
    pytest -m integration
"""

import os
import pytest
from dotenv import load_dotenv

# Load credentials from .env for local testing
load_dotenv()

from newsbias.config import (
    AppConfig,
    FallbackChain,
    Gpt4oMini,
    Gpt5Mini,
    Gpt6Luna,
    ModelConfig,
    get_config,
)
from newsbias.prompts import JinjaPrompt, extract_json_payload

HAS_API_KEY = bool(os.getenv("OPENAI_API_KEY"))
HAS_ALT_KEY = bool(os.getenv("ALT_OPENAI_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not HAS_API_KEY, reason="OPENAI_API_KEY not found in environment.")
def test_gpt4o_mini_direct_completion():
    """Verify Gpt4oMini executes a deterministic live completion."""
    model = Gpt4oMini()
    res = model.complete([{"role": "user", "content": "Respond strictly with the single word: VERIFIED"}])
    assert "VERIFIED" in res.upper()


@pytest.mark.integration
@pytest.mark.skipif(not HAS_API_KEY, reason="OPENAI_API_KEY not found in environment.")
def test_gpt5_mini_reasoning_completion():
    """Verify Gpt5Mini handles reasoning tokens and completion without temperature errors."""
    model = Gpt5Mini()
    res = model.complete([{"role": "user", "content": "Respond strictly with the single word: CONFIRMED"}])
    assert "CONFIRMED" in res.upper()


@pytest.mark.integration
@pytest.mark.skipif(not (HAS_ALT_KEY or HAS_API_KEY), reason="API key not found in environment.")
def test_gpt6_luna_reasoning_completion():
    """Verify Gpt6Luna handles reasoning tokens and completion on alt key."""
    model = Gpt6Luna()
    res = model.complete([{"role": "user", "content": "Respond strictly with the single word: LUNA"}])
    assert "LUNA" in res.upper()


@pytest.mark.integration
@pytest.mark.skipif(not HAS_API_KEY, reason="OPENAI_API_KEY not found in environment.")
def test_app_config_arbiter_task_live():
    """Verify APP_CONFIG.arbiter executes live analysis via configured model or fallback chain."""
    config = get_config()
    target = config.get_model_target("arbiter")
    assert target is not None

    prompt = JinjaPrompt.load("bias_detection")
    rendered = prompt.render(
        title="Bipartisan infrastructure agreement reached",
        content="Lawmakers from both parties reached an agreement on the highway funding bill.",
        sentence_specialist={"status": "available", "label": "NON-BIASED", "bias_ratio": 0.0, "flagged_sentences": []},
        political_specialist={"status": "available", "predicted_label": "CENTER", "confidence": 0.85, "scores": {"LEFT": 0.1, "CENTER": 0.85, "RIGHT": 0.05}},
        isolated_llm_specialist={"political_leaning": {"label": "center"}, "linguistic_bias": {"is_biased": False}},
        language={"code": "en", "name": "English", "transformers_supported": True},
    )
    result = rendered.complete_and_parse_json(config=target)
    assert isinstance(result, dict)
    assert "final_verdict" in result
    assert result["final_verdict"].get("overall_leaning") in ("center", "left", "right")


@pytest.mark.integration
@pytest.mark.skipif(not HAS_API_KEY, reason="OPENAI_API_KEY not found in environment.")
def test_app_config_neutral_rewrite_task_live():
    """Verify APP_CONFIG.neutral_rewrite executes live de-biasing rewrite with diff log."""
    config = get_config()
    target = config.get_model_target("neutral_rewrite")
    assert target is not None

    prompt = JinjaPrompt.load("bias_removal")
    rendered = prompt.render(
        original_text="Greedy politicians recklessly devastated local industries with socialist regulations.",
        why_biased_explanation="Loaded phrasing used to describe lawmakers and regulatory policy.",
        detected_leaning="right",
        key_findings=["Greedy politicians", "recklessly devastated"],
    )
    result = rendered.complete_and_parse_json(config=target)
    assert isinstance(result, dict)
    assert "rewritten_text" in result
    assert len(result["rewritten_text"]) > 10


@pytest.mark.integration
@pytest.mark.skipif(not HAS_API_KEY, reason="OPENAI_API_KEY not found in environment.")
def test_fallback_chain_recovers_from_bad_key():
    """Verify FallbackChain catches authentication error on primary and falls back to secondary."""
    broken_primary = ModelConfig(
        model="gpt-4o-mini",
        api_key="sk-proj-invalid-dead-key-for-testing-1234567890",
        timeout=10.0,
    )
    working_secondary = Gpt4oMini()

    chain = FallbackChain(broken_primary, working_secondary)
    res = chain.complete([{"role": "user", "content": "Respond strictly with the single word: RECOVERED"}])
    assert "RECOVERED" in res.upper()
