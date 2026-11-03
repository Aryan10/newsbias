"""Jinja2-based Prompt template engine for newsbias standalone tribunal.

Loads and renders templates with YAML frontmatter, markdown # System / # User
sections, and powerful Jinja2 logic (conditionals, loops, filters).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Optional

import jinja2
import yaml

from newsbias.models.llm import get_openai_client


def extract_json_payload(text: str) -> dict[str, Any]:
    """Parse JSON dictionary from raw model text or fenced markdown blocks."""
    if not text:
        return {}
    # Check for ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # Try finding outer braces (greedy to match nested structures)
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except Exception:
            pass
    try:
        return json.loads(text.strip())
    except Exception:
        return {}


class RenderedPrompt:
    """Represents a rendered prompt ready for LLM consumption."""

    def __init__(
        self,
        system: str,
        user: str,
        metadata: Optional[dict[str, Any]] = None,
        raw_rendered: str = "",
    ):
        self.system = system.strip()
        self.user = user.strip()
        self.metadata = metadata or {}
        self.raw_rendered = raw_rendered

    def complete(
        self,
        client=None,
        model: Optional[str] = None,
        config: Optional[Any] = None,
        task: Optional[str] = None,
    ) -> str:
        """Call the OpenAI chat completions endpoint using ModelConfig, FallbackChain, or legacy client."""
        # 1. If an explicit/mock client was passed, preserve direct client execution
        if client is not None:
            c = client
            target_model = model or getattr(c, "default_model", None) or "gpt-5-mini"

            kwargs: dict[str, Any] = {}
            if target_model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
                kwargs["max_completion_tokens"] = 4096
                kwargs["reasoning_effort"] = "low"
            else:
                kwargs["max_tokens"] = 4096

            resp = c.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": self.user},
                ],
                **kwargs,
            )
            return resp.choices[0].message.content or ""

        # 2. Config / Task based execution with FallbackChain support
        from newsbias.config import FallbackChain, ModelConfig, get_config

        if config is not None:
            target = config
        elif model is not None:
            target = ModelConfig(model=model)
        else:
            target = get_config().get_model_target(task or "arbiter")

        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        if self.user:
            messages.append({"role": "user", "content": self.user})

        return target.complete(messages)

    def complete_and_parse_json(
        self,
        client=None,
        model: Optional[str] = None,
        config: Optional[Any] = None,
        task: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute completion and parse JSON from the response."""
        content = self.complete(client=client, model=model, config=config, task=task)
        return extract_json_payload(content)

    def __str__(self) -> str:
        if self.system and self.user:
            return f"# System\n{self.system}\n\n# User\n{self.user}"
        return self.user or self.system


_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent.parent


class JinjaPrompt:
    """Prompt template engine powered by Jinja2 and frontmatter parsing."""

    DEFAULT_SEARCH_PATHS = [
        Path("prompts"),
        _PROJECT_ROOT / "prompts",
        _PACKAGE_ROOT / "prompts",
        Path("../prompts"),
    ]

    def __init__(
        self,
        template_content: str,
        metadata: Optional[dict[str, Any]] = None,
        source_path: Optional[Path] = None,
    ):
        self.template_content = template_content
        self.metadata = metadata or {}
        self.source_path = source_path
        self._env = jinja2.Environment(
            undefined=jinja2.Undefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        self._template = self._env.from_string(self.template_content)

    @classmethod
    def load(
        cls,
        name_or_path: str | Path,
        search_paths: Optional[list[Path]] = None,
    ) -> JinjaPrompt:
        """Load a prompt template by relative name or path."""
        candidate = Path(name_or_path)

        # Allow omitting .prompt.j2 extension
        candidates = [candidate]
        if not candidate.suffix:
            candidates.extend([
                candidate.with_suffix(".prompt.j2"),
                candidate.with_suffix(".j2"),
            ])

        resolved_path: Optional[Path] = None
        for cand in candidates:
            if cand.is_file():
                resolved_path = cand
                break

        if not resolved_path:
            dirs = search_paths or cls.DEFAULT_SEARCH_PATHS
            for d in dirs:
                for cand in candidates:
                    target = d / cand
                    if target.is_file():
                        resolved_path = target
                        break
                if resolved_path:
                    break

        if not resolved_path:
            raise FileNotFoundError(f"Could not locate prompt template for: {name_or_path}")

        raw = resolved_path.read_text(encoding="utf-8")
        metadata, body = cls._parse_frontmatter(raw)
        return cls(template_content=body, metadata=metadata, source_path=resolved_path)

    @staticmethod
    def _parse_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
        """Split leading YAML frontmatter (--- ... ---) from markdown body."""
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, raw_text, re.DOTALL)
        if match:
            fm_text, body = match.group(1), match.group(2)
            try:
                metadata = yaml.safe_load(fm_text) or {}
            except Exception:
                metadata = {}
            return metadata, body
        return {}, raw_text

    def render(self, **kwargs: Any) -> RenderedPrompt:
        """Render the Jinja template and split into # System and # User sections."""
        rendered = self._template.render(**kwargs)
        system_part, user_part = self._split_sections(rendered)
        return RenderedPrompt(
            system=system_part,
            user=user_part,
            metadata=dict(self.metadata),
            raw_rendered=rendered,
        )

    @staticmethod
    def _split_sections(text: str) -> tuple[str, str]:
        """Split text by # System and # User markdown headers."""
        system_match = re.search(r"#\s*System\s*\n(.*?)(?=#\s*User|$)", text, re.DOTALL | re.IGNORECASE)
        user_match = re.search(r"#\s*User\s*\n(.*)$", text, re.DOTALL | re.IGNORECASE)

        system_text = system_match.group(1).strip() if system_match else ""
        user_text = user_match.group(1).strip() if user_match else ""

        if not system_text and not user_text:
            user_text = text.strip()

        return system_text, user_text
