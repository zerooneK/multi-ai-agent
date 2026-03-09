"""
agents/base_agent.py
--------------------
Abstract base class all specialist agents inherit from.

Provides:
  - LLM provider access via self.provider
  - Shared chat() helper with retry
  - JSON extraction helper with truncation recovery
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

        def _is_context_error(exc: Exception) -> bool:
            msg = str(exc).lower()
            return (
                "context window" in msg
                or "context_length" in msg
                or "maximum context" in msg
                or "prompt too long" in msg
                or "prompt is too long" in msg
            )

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
                # Context-window errors won't be fixed by retrying — raise immediately
                # so the caller (e.g. QA agent) can retry with a smaller prompt.
                if _is_context_error(exc):
                    self._logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt, retries, exc, 0,
                    )
                    raise
                delay = 2 ** attempt
                self._logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, retries, exc, delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"{self.name} agent LLM call failed after {retries} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # JSON extraction — robust against truncated / wrapped responses
    # ------------------------------------------------------------------

    def extract_json(self, text: str) -> dict[str, Any]:
        """
        Extract and parse a JSON object or array from an LLM response.

        Handles these common LLM output problems:
          1. Markdown code fences  (```json ... ```)
          2. Prose before/after the JSON block
          3. Truncated JSON (model hit max_tokens mid-object)
             → attempts to auto-close unclosed braces/brackets
          4. Ollama <think>...</think> reasoning tags before the JSON

        Returns the parsed dict/list, or raises ValueError with context.
        """
        # ── Step 0: strip <think>...</think> blocks (Ollama reasoning models)
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # ── Step 1: strip markdown code fences
        clean = re.sub(r"```(?:json)?\s*", "", clean)
        clean = re.sub(r"```", "", clean).strip()

        # ── Step 2: find the outermost { ... } or [ ... ]
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = clean.find(start_char)
            if start == -1:
                continue

            # Try the full slice first (normal case)
            end = clean.rfind(end_char)
            if end != -1 and end > start:
                try:
                    return json.loads(clean[start:end + 1])
                except json.JSONDecodeError:
                    pass  # fall through to truncation recovery

            # ── Step 3: truncation recovery
            # The model stopped mid-JSON (hit max_tokens).
            # Try to close all unclosed brackets so json.loads can parse it.
            partial = clean[start:]
            recovered = self._close_truncated_json(partial)
            if recovered:
                try:
                    result = json.loads(recovered)
                    self._logger.warning(
                        "JSON was truncated — recovered partial response "
                        "(%d chars original, %d chars recovered)",
                        len(partial), len(recovered),
                    )
                    return result
                except json.JSONDecodeError:
                    pass

        # ── Step 4: last-ditch — try the whole cleaned string
        try:
            return json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not extract JSON from response.\n"
                f"Response (first 500 chars):\n{text[:500]}"
            ) from exc

    @staticmethod
    def _close_truncated_json(partial: str) -> str | None:
        """
        Attempt to auto-close a truncated JSON string by tracking
        open braces/brackets and appending the missing closing chars.

        Also handles the common case where the last value was cut mid-string
        (e.g. "title": "some long tex  — closes the string first).

        Returns the repaired string, or None if repair is not possible.
        """
        stack        = []   # tracks '{' and '['
        in_string    = False
        escape_next  = False
        last_colon   = -1   # position of most recent ':' outside strings

        for i, ch in enumerate(partial):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                stack.append(ch)
            elif ch in ('}', ']'):
                if stack:
                    stack.pop()
            elif ch == ':':
                last_colon = i

        if not stack:
            return None  # JSON looks balanced already — not a truncation issue

        # Build closing suffix
        # If we're inside a string when truncation happened, close it first
        suffix = ""
        if in_string:
            suffix += '"'
            # If the cut happened in the middle of a value (after ':'),
            # we need a placeholder so the key:value pair is valid
            suffix += "... (truncated)"
            suffix += '"'

        # Close all open containers in reverse order
        close_map = {'{': '}', '[': ']'}
        for opener in reversed(stack):
            # For an open object, append null value if needed
            if opener == '{' and in_string:
                suffix += '}'
            else:
                suffix += close_map[opener]

        return partial + suffix