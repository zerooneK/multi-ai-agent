"""
config.py
---------
Configuration for the Multi-Agent Team framework.
Loads .env automatically before reading os.environ.

Per-Agent Model Configuration
------------------------------
Each agent can use a different provider and model independently.
Set in .env using the pattern:

  <AGENT>_PROVIDER = provider name
  <AGENT>_MODEL    = model name

Where <AGENT> is one of: PLANNER, BACKEND, FRONTEND, QA

If not set, each agent falls back to the global AGENT_PROVIDER / AGENT_MODEL.

Example .env:
  AGENT_PROVIDER=anthropic                          # global fallback
  AGENT_MODEL=claude-haiku-4-5-20251001             # global fallback model

  # Generation limits (tune per model)
  MAX_MODELS=5              # 0 = unlimited
  MAX_OUTPUT_TOKENS=16384   # 8192 for 7B, 12288 for 14B, 16384 for cloud
  MAX_FILES_PER_AGENT=20    # 0 = no limit hint in prompt

  PLANNER_PROVIDER=anthropic
  PLANNER_MODEL=claude-sonnet-4-20250514            # best model for planning

  BACKEND_PROVIDER=openai
  BACKEND_MODEL=gpt-4o                              # strong at code gen

  FRONTEND_PROVIDER=openai
  FRONTEND_MODEL=gpt-4o-mini                        # cheaper for HTML/JS

  QA_PROVIDER=anthropic
  QA_MODEL=claude-haiku-4-5-20251001                # fast + cheap for review
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    candidates = [Path(__file__).parent / ".env", Path.cwd() / ".env"]
    env_path   = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass
    with open(env_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip(); val = val.strip()
            if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()


@dataclass(frozen=True)
class AgentConfig:
    """Provider + model config for one specific agent."""
    provider: str
    model:    str


@dataclass(frozen=True)
class Config:
    # ── Global fallback provider ─────────────────────────────────────────
    PROVIDER: str = "anthropic"

    # ── Default models per provider ──────────────────────────────────────
    ANTHROPIC_MODEL:  str = "claude-sonnet-4-20250514"
    OPENAI_MODEL:     str = "gpt-4o"
    OPENROUTER_MODEL: str = "openai/gpt-4o"
    GEMINI_MODEL:     str = "gemini-2.0-flash"
    OLLAMA_MODEL:     str = "llama3.3"

    # ── Per-agent config (provider + model per agent) ────────────────────
    PLANNER_PROVIDER:  str = ""   # empty = use global PROVIDER
    PLANNER_MODEL:     str = ""   # empty = use provider default
    BACKEND_PROVIDER:  str = ""
    BACKEND_MODEL:     str = ""
    FRONTEND_PROVIDER: str = ""
    FRONTEND_MODEL:    str = ""
    QA_PROVIDER:       str = ""
    QA_MODEL:          str = ""

    # ── Endpoints ────────────────────────────────────────────────────────
    OLLAMA_BASE_URL:     str = "http://localhost:11434/v1"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # ── Output ───────────────────────────────────────────────────────────
    OUTPUT_DIR: str = "output"

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── Logging to file ─────────────────────────────────────────────────
    # LOG_DIR: directory to store run logs (relative to project root or absolute)
    #           set to "" to disable file logging
    # LOG_LEVEL: DEBUG | INFO | WARNING | ERROR
    LOG_DIR: str = "logs"

    # ── Parallel generation ──────────────────────────────────────────────
    # Set PARALLEL_GENERATION=false in .env to disable (e.g. when using Ollama)
    # Set PARALLEL_GENERATION=true for cloud APIs (Gemini, OpenRouter, Anthropic)
    PARALLEL_GENERATION: bool = True

    # ── Generation limits ────────────────────────────────────────────────
    # MAX_MODELS           : max database models planner is allowed to design
    #                        0 = unlimited (planner decides freely)
    #                        recommended: 3 for 7B, 5 for 14B, 0 for cloud
    # MAX_OUTPUT_TOKENS    : max tokens backend/frontend agents may generate per request
    #                        tune to model: 8192 for 7B, 12288 for 14B, 16384 for cloud
    # MAX_FILES_PER_AGENT  : hint to LLM — max files to generate in one request
    #                        0 = unlimited (no hint injected into prompt)
    MAX_MODELS:          int = 0
    MAX_OUTPUT_TOKENS:   int = 16384
    MAX_FILES_PER_AGENT: int = 0

    # ------------------------------------------------------------------
    # Helper: resolve provider+model for a specific agent
    # ------------------------------------------------------------------
    def for_agent(self, agent: str) -> AgentConfig:
        """
        Return the resolved (provider, model) for a named agent.

        Resolution order:
          1. <AGENT>_PROVIDER / <AGENT>_MODEL  (per-agent override)
          2. AGENT_PROVIDER   / AGENT_MODEL    (global override)
          3. Built-in defaults per provider

        Parameters
        ----------
        agent : "planner" | "backend" | "frontend" | "qa"
        """
        prefix   = agent.upper()
        provider = (
            getattr(self, f"{prefix}_PROVIDER", "") or self.PROVIDER
        ).lower()
        model = getattr(self, f"{prefix}_MODEL", "") or self._default_model(provider)
        return AgentConfig(provider=provider, model=model)

    def _default_model(self, provider: str) -> str:
        """Return the default model string for a given provider name."""
        return {
            "anthropic":  self.ANTHROPIC_MODEL,
            "openai":     self.OPENAI_MODEL,
            "openrouter": self.OPENROUTER_MODEL,
            "gemini":     self.GEMINI_MODEL,
            "ollama":     self.OLLAMA_MODEL,
        }.get(provider, self.ANTHROPIC_MODEL)


def _build() -> Config:
    """Build Config from defaults + ENV overrides."""
    global_provider = os.environ.get("AGENT_PROVIDER", Config.PROVIDER).lower()
    global_model    = os.environ.get("AGENT_MODEL", "")

    kwargs: dict = dict(
        PROVIDER        = global_provider,
        OUTPUT_DIR           = os.environ.get("OUTPUT_DIR",           Config.OUTPUT_DIR),
        LOG_DIR              = os.environ.get("LOG_DIR",              Config.LOG_DIR),
        LOG_LEVEL            = os.environ.get("AGENT_LOG_LEVEL",     Config.LOG_LEVEL),
        OLLAMA_BASE_URL      = os.environ.get("OLLAMA_BASE_URL",     Config.OLLAMA_BASE_URL),
        PARALLEL_GENERATION  = os.environ.get("PARALLEL_GENERATION", "true").strip().lower() != "false",
        MAX_MODELS           = int(os.environ.get("MAX_MODELS",         str(Config.MAX_MODELS))),
        MAX_OUTPUT_TOKENS    = int(os.environ.get("MAX_OUTPUT_TOKENS",  str(Config.MAX_OUTPUT_TOKENS))),
        MAX_FILES_PER_AGENT  = int(os.environ.get("MAX_FILES_PER_AGENT",str(Config.MAX_FILES_PER_AGENT))),
    )

    # Apply global model override to the active provider's default
    if global_model:
        model_key = {
            "anthropic":  "ANTHROPIC_MODEL",
            "openai":     "OPENAI_MODEL",
            "openrouter": "OPENROUTER_MODEL",
            "gemini":     "GEMINI_MODEL",
            "ollama":     "OLLAMA_MODEL",
        }.get(global_provider)
        if model_key:
            kwargs[model_key] = global_model

    # Per-agent overrides — read PLANNER_PROVIDER, BACKEND_MODEL, etc.
    for agent in ("PLANNER", "BACKEND", "FRONTEND", "QA"):
        p = os.environ.get(f"{agent}_PROVIDER", "").strip().lower()
        m = os.environ.get(f"{agent}_MODEL",    "").strip()
        if p:
            kwargs[f"{agent}_PROVIDER"] = p
        if m:
            kwargs[f"{agent}_MODEL"] = m

    return Config(**kwargs)


cfg = _build()