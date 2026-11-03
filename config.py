"""NewsBias Application Configuration.

You can modify this file directly to set models, API keys, endpoints,
and fallback chains across tasks.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

# Automatically load environment variables if present in .env
load_dotenv()

from newsbias.config import (
    AppConfig,
    FallbackChain,
    Gpt4oMini,
    Gpt5Mini,
    Gpt6Luna,
    ModelConfig,
)

# ── 1. Primary Models (using OPENAI_API_KEY) ──────────────────────────────────
primary_gpt5 = Gpt5Mini(api_key=os.getenv("OPENAI_API_KEY"))
primary_gpt4o = Gpt4oMini(api_key=os.getenv("OPENAI_API_KEY"))

# ── 2. Fallback Models (using ALT_OPENAI_API_KEY) ─────────────────────────────
# ALT_OPENAI_API_KEY verified supporting gpt-6-luna, gpt-5-mini, gpt-4o-mini
alt_key = os.getenv("ALT_OPENAI_API_KEY")
alt_gpt6_luna = Gpt6Luna(api_key=alt_key)
alt_gpt5_mini = Gpt5Mini(api_key=alt_key)
alt_gpt4o_mini = Gpt4oMini(api_key=alt_key)

# ── 3. Task Fallback Chains ───────────────────────────────────────────────────
config = AppConfig(
    # Neutral AP wire rewriter: Try gpt-5-mini, fallback to gpt-6-luna, alt gpt-5, then fast gpt-4o
    neutral_rewrite=FallbackChain(
        primary_gpt5,
        alt_gpt6_luna,
        alt_gpt5_mini,
        primary_gpt4o,
    ),
    # Supreme Arbiter multi-agent synthesis: gpt-5-mini, fallback to gpt-6-luna / alt key
    arbiter=FallbackChain(
        primary_gpt5,
        alt_gpt6_luna,
        alt_gpt5_mini,
        primary_gpt4o,
    ),
    # Zero-shot isolated LLM specialist: fast classifier with resilient fallbacks
    isolated_llm=FallbackChain(
        primary_gpt4o,
        alt_gpt4o_mini,
        primary_gpt5,
    ),
    # Web ingestion article body extractor
    article_extraction=FallbackChain(
        primary_gpt4o,
        alt_gpt4o_mini,
    ),
    # Web server settings
    server_host="127.0.0.1",
    server_port=8000,
)

# Export for both config and APP_CONFIG conventions
APP_CONFIG = config
