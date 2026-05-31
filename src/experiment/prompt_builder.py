"""Prompt builder: render a per-(user, scenario) prompt from Jinja2 templates.

A *scenario* is a tuple of demographic dimension names; the empty tuple is the
neutral baseline (no demographic information injected).

This module is dataset-agnostic. Each dataset adapter exposes ``domain()``;
the builder picks the matching template (``{domain}_{lang}.j2``) by default.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template, select_autoescape

from src.datasets.base import InteractionItem, UserRecord


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_GLOSSARY_PATH = DEFAULT_PROMPTS_DIR / "glossary.yaml"


# Map (domain) -> default language used for the prompt.
DEFAULT_DOMAIN_LANG = {
    "movie": "en",
    "book": "en",
    "music": "en",
    "music_track": "en",
    "video": "zh",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class RenderedPrompt:
    """A prompt ready to send to an LLM."""

    text: str
    template_name: str
    scenario: Tuple[str, ...]
    sha256: str
    history_size: int


class PromptBuilder:
    """Render prompts for a specific dataset domain."""

    def __init__(
        self,
        domain: str,
        prompts_dir: Optional[str | Path] = None,
        glossary_path: Optional[str | Path] = None,
        language: Optional[str] = None,
        history_limit: int = 10,
    ) -> None:
        self.domain = domain
        self.prompts_dir = Path(prompts_dir) if prompts_dir else DEFAULT_PROMPTS_DIR
        glossary_p = Path(glossary_path) if glossary_path else DEFAULT_GLOSSARY_PATH
        self.language = language or DEFAULT_DOMAIN_LANG.get(domain, "en")
        self.history_limit = history_limit

        with glossary_p.open("r", encoding="utf-8") as f:
            self.glossary: Dict[str, Dict[str, Dict[str, str]]] = yaml.safe_load(f)

        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            trim_blocks=False,
            lstrip_blocks=False,
        )
        self.template_name = f"{self.domain}_{self.language}.j2"
        self._template: Template = self.env.get_template(self.template_name)

    # ------------------------------------------------------------------

    def _glossary_for(self, dim: str) -> Dict[str, str]:
        return (self.glossary.get(self.language, {}) or {}).get(dim, {})

    def _humanize_value(self, dim: str, value: str) -> str:
        table = self._glossary_for(dim)
        if value in table:
            return table[value]
        # Fallback: return the raw value.
        return str(value)

    def _build_demographics_clause(
        self,
        scenario: Tuple[str, ...],
        demographics: Mapping[str, str],
    ) -> str:
        """Build a one-sentence clause describing the active demographics.

        Returns an empty string for the neutral scenario.
        """
        if not scenario:
            return ""
        parts = []
        for dim in scenario:
            raw = demographics.get(dim, "unknown")
            human = self._humanize_value(dim, str(raw))
            parts.append(human)
        joined = ", ".join(parts)
        if self.language == "zh":
            return f"该用户的画像信息：{joined}。"
        else:
            return f"The user's profile: {joined}."

    # ------------------------------------------------------------------

    def render(
        self,
        user: UserRecord,
        scenario: Sequence[str] = (),
        movie_count: int = 20,
    ) -> RenderedPrompt:
        scenario_tuple = tuple(scenario)
        history = list(user.history)[: self.history_limit]
        demographics_clause = self._build_demographics_clause(
            scenario_tuple, user.demographics
        )
        text = self._template.render(
            history=history,
            demographics_clause=demographics_clause,
            movie_count=movie_count,
        )
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return RenderedPrompt(
            text=text,
            template_name=self.template_name,
            scenario=scenario_tuple,
            sha256=sha,
            history_size=len(history),
        )

    def render_all_scenarios(
        self,
        user: UserRecord,
        scenarios: Iterable[Sequence[str]],
        movie_count: int = 20,
    ) -> List[RenderedPrompt]:
        return [self.render(user, s, movie_count) for s in scenarios]

    def template_hash(self) -> str:
        """Hash of the loaded template + glossary for reproducibility metadata."""
        h = hashlib.sha256()
        tpl_path = self.prompts_dir / self.template_name
        h.update(tpl_path.read_bytes())
        h.update(DEFAULT_GLOSSARY_PATH.read_bytes() if DEFAULT_GLOSSARY_PATH.exists() else b"")
        return h.hexdigest()
