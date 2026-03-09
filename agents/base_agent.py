"""
agents/base_agent.py
--------------------
Abstract base class all specialist agents inherit from.

Provides:
  - LLM provider access via self.provider
  - Shared chat() helper with retry
  - JSON extraction helper
  - Standard logging
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from config import cfg

logger = logging.getLogger("agent.base")


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.

    Subclasses must implement:
      - name   : str property  (e.g. "planner", "backend")
      - system : str property  (system prompt for this agent)
      - run()  : accepts an AgentMessage, returns updated AgentMessage
    """

    def __init__(
        self,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> None:
        from providers import get_provider
        self.provider = get_provider(provider_name, model)
        self._logger  = logging.getLogger(f"agent.{self.name}")
        self._logger.info(
            "%s agent ready — %s / %s",
            self.name.capitalize(), self.provider.name, self.provider.model,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier, e.g. 'planner'."""

    @property
    @abstractmethod
    def system(self) -> str:
        """System prompt used for all LLM calls by this agent."""

    @abstractmethod
    def run(self, message: "AgentMessage") -> "AgentMessage":  # type: ignore[name-defined]
        """Process a task message and return it with result/status filled."""

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def chat(
        self,
        user_content: str,
        history: list[dict] | None = None,
        max_tokens: int = 8192,
        retries: int = 3,
    ) -> str:
        """
        Send a message to the LLM and return the text response.

        Retries up to `retries` times on transient errors with
        exponential back-off.
        """
        messages = list(history or [])
        messages.append({"role": "user", "content": user_content})

        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return self.provider.chat(
                    system=self.system,
                    messages=messages,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                delay = 2 ** attempt
                self._logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, retries, exc, delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"{self.name} agent LLM call failed after {retries} attempts: {last_exc}"
        )

    def extract_json(self, text: str) -> dict[str, Any]:
        """
        Extract the first JSON object or array from a text response.
        Strips markdown code fences if present.
        """
        # Remove markdown fences
        clean = re.sub(r"```(?:json)?\s*", "", text)
        clean = re.sub(r"```", "", clean).strip()

        # Find the outermost { ... } or [ ... ]
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = clean.find(start_char)
            end   = clean.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(clean[start:end + 1])
                except json.JSONDecodeError:
                    continue

        # Last-ditch: try parsing the whole cleaned string
        try:
            return json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not extract JSON from response.\n"
                f"Response (first 500 chars):\n{text[:500]}"
            ) from exc
