"""NewsBias dynamic model and application configuration engine.

Provides class-based model specifications (ModelConfig, Gpt4oMini, Gpt5Mini),
resilient multi-tier fallback chains (FallbackChain), and task-to-model
routing (AppConfig) that can be defined in Python code or loaded from
a user-editable `config.py` at the project root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import logging
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Optional, Union

from dotenv import load_dotenv

# Ensure local .env is loaded if present
load_dotenv()

logger = logging.getLogger("newsbias.config")


@dataclass
class ModelConfig:
    """Class-based LLM configuration with explicit credentials and execution parameters.

    Subclasses and instances allow customizing model name, API key (or env var),
    base URL, temperature, token ceilings, reasoning effort, and timeout.
    """
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 4096
    reasoning_effort: Optional[str] = None  # "low", "medium", "high" for reasoning models
    timeout: float = 60.0
    max_retries: int = 2
    extra_params: dict[str, Any] = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """Resolve API key from explicit value or default environment variable."""
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                f"Missing API key for model '{self.model}'. Set OPENAI_API_KEY or specify api_key."
            )
        return key

    def resolve_base_url(self) -> Optional[str]:
        """Resolve base URL from explicit value or default environment variable."""
        return self.base_url or os.getenv("OPENAI_BASE_URL")

    def is_reasoning_model(self) -> bool:
        """Return True if model uses reasoning tokens and strict parameter constraints."""
        m = self.model.lower()
        return m.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))

    def create_client(self):
        """Instantiate an official OpenAI client tailored to this model's endpoint and credentials."""
        from openai import OpenAI
        url = self.resolve_base_url()
        client = OpenAI(
            api_key=self.resolve_api_key(),
            base_url=url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        client.default_model = self.model
        return client

    def build_completion_kwargs(self, messages: list[dict[str, str]], **overrides) -> dict[str, Any]:
        """Build API kwargs adhering to model-specific parameter constraints."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "timeout": overrides.pop("timeout", self.timeout),
        }
        if self.is_reasoning_model():
            # Reasoning models require max_completion_tokens and disallow temperature=0.0
            max_comp = (
                overrides.pop("max_completion_tokens", None)
                or overrides.pop("max_tokens", None)
                or self.max_tokens
                or 4096
            )
            kwargs["max_completion_tokens"] = max_comp
            effort = overrides.pop("reasoning_effort", self.reasoning_effort)
            if effort:
                kwargs["reasoning_effort"] = effort
            temp = overrides.pop("temperature", self.temperature)
            # Only include temperature if explicitly set and not 0.0 (reasoning models reject 0.0)
            if temp is not None and temp != 0.0:
                kwargs["temperature"] = temp
        else:
            # Standard conversational models
            max_tok = overrides.pop("max_tokens", self.max_tokens)
            if max_tok is not None:
                kwargs["max_tokens"] = max_tok
            temp = overrides.pop("temperature", self.temperature)
            kwargs["temperature"] = 0.0 if temp is None else temp

        # Include extra_params & remaining overrides
        kwargs.update(self.extra_params)
        kwargs.update(overrides)
        return kwargs

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Execute chat completion and return the assistant message content."""
        client = self.create_client()
        call_kwargs = self.build_completion_kwargs(messages, **kwargs)
        resp = client.chat.completions.create(**call_kwargs)
        return resp.choices[0].message.content or ""


class Gpt4oMini(ModelConfig):
    """Fast, deterministic classification preset using gpt-4o-mini."""
    def __init__(self, **kwargs):
        kwargs.setdefault("model", "gpt-4o-mini")
        kwargs.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        kwargs.setdefault("base_url", os.getenv("OPENAI_BASE_URL"))
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("max_tokens", 4096)
        super().__init__(**kwargs)


class Gpt5Mini(ModelConfig):
    """Reasoning and synthesis model preset using gpt-5-mini."""
    def __init__(self, **kwargs):
        kwargs.setdefault("model", "gpt-5-mini")
        kwargs.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        kwargs.setdefault("base_url", os.getenv("OPENAI_BASE_URL"))
        kwargs.setdefault("temperature", None)  # Defaults to provider default (1.0)
        kwargs.setdefault("max_tokens", 4096)   # Handled as max_completion_tokens
        kwargs.setdefault("reasoning_effort", "low")
        super().__init__(**kwargs)


class Gpt6Luna(ModelConfig):
    """Advanced reasoning model preset using gpt-6-luna."""
    def __init__(self, **kwargs):
        kwargs.setdefault("model", "gpt-6-luna")
        kwargs.setdefault("api_key", os.getenv("ALT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
        kwargs.setdefault("base_url", os.getenv("OPENAI_BASE_URL"))
        kwargs.setdefault("temperature", None)
        kwargs.setdefault("max_tokens", 4096)
        kwargs.setdefault("reasoning_effort", "low")
        super().__init__(**kwargs)


class FallbackChain:
    """Sequential chain of models tried in order upon API or quota failure."""
    def __init__(self, *models: Union[ModelConfig, Iterable[ModelConfig]]):
        flat_models: list[ModelConfig] = []
        for m in models:
            if isinstance(m, (list, tuple)):
                flat_models.extend(m)
            elif isinstance(m, ModelConfig):
                flat_models.append(m)
            elif isinstance(m, FallbackChain):
                flat_models.extend(m.models)
        if not flat_models:
            raise ValueError("FallbackChain requires at least one ModelConfig.")
        self.models = flat_models

    @property
    def primary_model(self) -> ModelConfig:
        return self.models[0]

    def __iter__(self):
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Attempt completion across candidate models until one succeeds."""
        last_error: Optional[Exception] = None
        for idx, model in enumerate(self.models):
            try:
                # Fast check that an API key is available before hitting network
                model.resolve_api_key()
                logger.debug(
                    "Attempting completion with model %s (attempt %d/%d)...",
                    model.model, idx + 1, len(self.models),
                )
                return model.complete(messages, **kwargs)
            except Exception as e:
                logger.warning(
                    "Model %s (attempt %d/%d) failed with %s: %s. Trying next fallback in chain...",
                    model.model, idx + 1, len(self.models), type(e).__name__, e,
                )
                last_error = e
                continue

        raise RuntimeError(
            f"All {len(self.models)} model(s) in FallbackChain failed. Last error: {last_error}"
        ) from last_error


@dataclass
class AppConfig:
    """Master application configuration mapping pipeline tasks to models or fallback chains."""
    neutral_rewrite: Union[ModelConfig, FallbackChain] = field(default_factory=Gpt5Mini)
    arbiter: Union[ModelConfig, FallbackChain] = field(default_factory=Gpt5Mini)
    isolated_llm: Union[ModelConfig, FallbackChain] = field(default_factory=Gpt4oMini)
    article_extraction: Union[ModelConfig, FallbackChain] = field(default_factory=Gpt4oMini)
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    def get_model_target(self, task: str) -> Union[ModelConfig, FallbackChain]:
        """Return the configured ModelConfig or FallbackChain for a given task name."""
        task_clean = task.lower().strip()
        if task_clean in ("neutral_rewrite", "rewrite", "bias_removal"):
            return self.neutral_rewrite
        elif task_clean in ("arbiter", "bias_detection", "bias_analysis"):
            return self.arbiter
        elif task_clean in ("isolated_llm", "zero_shot_bias", "zero_shot"):
            return self.isolated_llm
        elif task_clean in ("article_extraction", "extraction", "ingestion"):
            return self.article_extraction
        # Default fallback
        return self.arbiter or self.neutral_rewrite


_CONFIG_CACHE: Optional[AppConfig] = None


def get_config(force_reload: bool = False) -> AppConfig:
    """Retrieve the singleton AppConfig instance, discovering config.py if present."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    loaded = _discover_and_load_config()
    _CONFIG_CACHE = loaded
    return _CONFIG_CACHE


def reload_config() -> AppConfig:
    """Force reloading of the application configuration."""
    return get_config(force_reload=True)


def _discover_and_load_config() -> AppConfig:
    """Look for root config.py in current working directory or repository root."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        Path.cwd() / "config.py",
        repo_root / "config.py",
    ]
    for p in candidates:
        if p.is_file():
            try:
                spec = importlib.util.spec_from_file_location("newsbias_user_config", p)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    cfg = getattr(mod, "APP_CONFIG", None) or getattr(mod, "config", None)
                    if isinstance(cfg, AppConfig):
                        logger.info("Loaded custom AppConfig from %s", p)
                        return cfg
            except Exception as e:
                logger.warning("Error loading config.py from %s: %s. Using default AppConfig.", p, e)

    # Return default AppConfig if no user config.py found
    return AppConfig(
        neutral_rewrite=FallbackChain(Gpt5Mini(), Gpt4oMini()),
        arbiter=FallbackChain(Gpt5Mini(), Gpt4oMini()),
        isolated_llm=FallbackChain(Gpt4oMini(), Gpt5Mini()),
        article_extraction=Gpt4oMini(),
    )
