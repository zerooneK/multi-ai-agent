# =============================================================================
# providers/__init__.py
# -----------------------------------------------------------------------------
# LLM provider factory + lightweight adapter classes.
#
# Unlike other __init__.py files in this project (which only re-export),
# this file contains the actual implementation because providers are a
# self-contained module — no other file in providers/ needs to exist.
#
# Supported providers:
#   anthropic   — Claude (claude-sonnet-4-20250514, opus, haiku, ...)
#   openai      — GPT    (gpt-4o, gpt-4o-mini, ...)
#   openrouter  — 100+ models via OpenRouter (uses openai SDK)
#   gemini      — Google Gemini (gemini-2.0-flash, gemini-1.5-pro, ...)
#   ollama      — Local models  (llama3.3, mistral, qwen2.5, ... — no key needed)
#
# Usage:
#   from providers import get_provider
#   provider = get_provider("anthropic", "claude-sonnet-4-20250514")
#   response = provider.chat(system="...", messages=[...])
#
# All adapters expose the same interface:
#   provider.name   : str   — provider identifier
#   provider.model  : str   — model name
#   provider.chat() : str   — send messages, return text response
# =============================================================================

from __future__ import annotations
import os
import logging

logger = logging.getLogger("providers")


# =============================================================================
# Factory function — call this to get a provider instance
# =============================================================================

def get_provider(provider: str | None = None, model: str | None = None):
    """
    Return an LLM provider adapter for the given provider name and model.

    Parameters
    ----------
    provider : One of: anthropic | openai | openrouter | gemini | ollama
               Falls back to cfg.PROVIDER (from .env) if not given.
    model    : Model name override.
               Falls back to the provider's default in cfg if not given.

    Returns
    -------
    One of: _AnthropicAdapter | _OpenAIAdapter | _GeminiAdapter
    All have the same .chat(system, messages, max_tokens) → str interface.
    """
    from config import cfg

    name = (provider or cfg.PROVIDER).lower()
    logger.info("Initialising provider: %s  model: %s", name, model or "(default)")

    # ── Anthropic ─────────────────────────────────────────────────────────────
    if name == "anthropic":
        import anthropic as _anthropic
        client = _anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        return _AnthropicAdapter(client, model or cfg.ANTHROPIC_MODEL)

    # ── OpenAI / OpenRouter / Ollama (all use the openai SDK) ────────────────
    if name in ("openai", "openrouter", "ollama"):
        import openai as _openai
        base_url = None
        api_key  = None
        if name == "openrouter":
            base_url = cfg.OPENROUTER_BASE_URL          # https://openrouter.ai/api/v1
            api_key  = os.environ.get("OPENROUTER_API_KEY", "")
        elif name == "ollama":
            base_url = cfg.OLLAMA_BASE_URL               # http://localhost:11434/v1
            api_key  = "ollama"                          # ollama ignores the key
        client = _openai.OpenAI(
            **({ "base_url": base_url } if base_url else {}),
            **({ "api_key":  api_key  } if api_key  else {}),
        )
        m = model or (
            cfg.OPENAI_MODEL     if name == "openai"     else
            cfg.OPENROUTER_MODEL if name == "openrouter" else
            cfg.OLLAMA_MODEL
        )
        return _OpenAIAdapter(client, m)

    # ── Google Gemini ─────────────────────────────────────────────────────────
    if name == "gemini":
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
        return _GeminiAdapter(genai, model or cfg.GEMINI_MODEL)

    raise ValueError(
        f"Unknown provider: {name!r}. "
        f"Choose one of: anthropic | openai | openrouter | gemini | ollama"
    )


# =============================================================================
# Adapter classes — normalise each SDK to the same .chat() interface
# =============================================================================

class _AnthropicAdapter:
    """
    Adapter for Anthropic Claude.
    SDK   : anthropic
    Key   : ANTHROPIC_API_KEY
    Docs  : https://docs.anthropic.com
    """
    def __init__(self, client, model: str):
        self.client = client
        self.model  = model
        self.name   = "anthropic"

    def chat(self, system: str, messages: list[dict], max_tokens: int = 8192) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,        # Anthropic takes system as a top-level param
            messages=messages,
        )
        return "".join(b.text for b in response.content if hasattr(b, "text")).strip()


class _OpenAIAdapter:
    """
    Adapter for OpenAI, OpenRouter, and Ollama.
    All three use the openai Python SDK with different base_url / api_key.

    SDK   : openai
    Keys  : OPENAI_API_KEY | OPENROUTER_API_KEY | (none for Ollama)
    Docs  : https://platform.openai.com/docs
            https://openrouter.ai/docs
            https://ollama.com/blog/openai-compatibility
    """
    def __init__(self, client, model: str):
        self.client = client
        self.model  = model
        self.name   = "openai"

    def chat(self, system: str, messages: list[dict], max_tokens: int = 8192) -> str:
        # OpenAI-style: system message is the first message with role="system"
        full     = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=full,
        )
        return (response.choices[0].message.content or "").strip()


class _GeminiAdapter:
    """
    Adapter for Google Gemini.
    SDK   : google-generativeai
    Key   : GEMINI_API_KEY
    Docs  : https://ai.google.dev/gemini-api/docs
    """
    def __init__(self, genai, model: str):
        self.genai  = genai
        self.model  = model
        self.name   = "gemini"

    def chat(self, system: str, messages: list[dict], max_tokens: int = 8192) -> str:
        m = self.genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,   # Gemini takes system as constructor param
        )
        # Convert messages to Gemini's Content format
        history = [
            self.genai.protos.Content(
                role="model" if msg["role"] == "assistant" else "user",
                parts=[self.genai.protos.Part(text=msg["content"])],
            )
            for msg in messages
        ]
        response = m.generate_content(
            history,
            generation_config=self.genai.types.GenerationConfig(
                max_output_tokens=max_tokens
            ),
        )
        return response.text.strip()