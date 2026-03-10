"""
agents/fixer_agent.py
---------------------
Interactive Bug Fixer Agent.

Workflow
--------
1. User runs the generated project (uvicorn / npm run dev)
2. User encounters a runtime error and pastes the error message
3. FixerAgent reads the relevant source files from the output directory
4. FixerAgent asks the LLM to diagnose and fix the error
5. Fixed files are written back to disk
6. Repeat up to MAX_FIX_ROUNDS if the user reports another error

This agent is invoked AFTER the main pipeline via `python main.py --fix`
or interactively after a pipeline run.

It does NOT re-run QA — it trusts the user's runtime error as ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agents.base_agent import BaseAgent
from config import cfg
from models.messages import AgentMessage
from tools.file_tools import create_file, list_files, read_file

logger = logging.getLogger("agent.fixer")

MAX_FIX_ROUNDS = 3


class FixerAgent(BaseAgent):
    """
    Runtime bug fixer that reads error messages from the user and
    patches the generated source files accordingly.
    """

    @property
    def name(self) -> str:
        return "fixer"

    @property
    def system(self) -> str:
        return """You are an expert full-stack debugger specialising in
FastAPI (Python) backends and Next.js 14 (TypeScript) frontends.

You receive:
  1. A runtime error message from the user (uvicorn traceback, browser console, etc.)
  2. The current contents of relevant source files

Your job:
  - Diagnose the root cause of the error
  - Output a JSON array of FIXED files

Output format — JSON array only, no prose, no markdown:
[
  {"path": "backend/main.py",   "content": "full corrected file content"},
  {"path": "backend/auth.py",   "content": "full corrected file content"}
]

Rules:
- Output ONLY the JSON array. Start with [ and end with ]
- Only include files that actually need changes
- Every file must be 100% complete — no placeholders, no # TODO
- Keep all existing correct logic — only fix what the error points to
- For backend: preserve FastAPI structure, SQLAlchemy models, JWT httpOnly cookie
- For frontend: preserve Next.js App Router structure, TypeScript types
- If the error is in a file you were not shown, signal by returning:
  [{"path": "__need_more_context__", "content": "<filename that is needed>"}]
"""

    def run(self, message: AgentMessage) -> AgentMessage:
        message.mark_running()

        error_message = message.payload.get("error_message", "")
        output_dir    = message.payload.get("output_dir", "")
        extra_files   = message.payload.get("extra_files", [])

        if not error_message:
            message.mark_failed("No error message provided")
            return message

        if not output_dir or not Path(output_dir).exists():
            message.mark_failed(f"Output directory not found: {output_dir}")
            return message

        logger.info("FixerAgent diagnosing error in: %s", output_dir)

        context = self._build_context(error_message, output_dir, extra_files)
        prompt  = self._build_prompt(error_message, context)

        # Attempt 1: full prompt
        written: list[str] = []
        try:
            raw   = self.chat(prompt, max_tokens=cfg.MAX_OUTPUT_TOKENS)
            files = self.extract_json(raw)
            written = self._write_files(files, output_dir)
            message.mark_done(self._summary(written, error_message))
            return message
        except Exception as exc:
            logger.warning("Fixer attempt 1 failed (%s) — retrying with minimal prompt", exc)

        # Attempt 2: minimal prompt
        try:
            raw   = self.chat(self._minimal_prompt(error_message, context),
                              max_tokens=cfg.MAX_OUTPUT_TOKENS)
            files = self.extract_json(raw)
            written = self._write_files(files, output_dir)
            message.mark_done(self._summary(written, error_message))
        except Exception as exc2:
            logger.error("Fixer agent failed: %s", exc2)
            message.mark_failed(str(exc2))

        return message

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_prompt(self, error_message: str, context: str) -> str:
        return (
            f"The user encountered this runtime error:\n\n"
            f"```\n{error_message}\n```\n\n"
            f"Here are the relevant source files:\n\n"
            f"{context}\n\n"
            f"Diagnose the error and output a JSON array of fixed files.\n"
            f"Start with [ and end with ] — no prose before or after.\n\n["
        )

    def _minimal_prompt(self, error_message: str, context: str) -> str:
        return (
            f"Fix this runtime error:\n{error_message[:500]}\n\n"
            f"Files:\n{context[:3000]}\n\n"
            f"Output JSON array only. Start with [. No text before or after.\n["
        )

    # ------------------------------------------------------------------
    # Context builder — smart file selection based on error content
    # ------------------------------------------------------------------

    def _build_context(
        self,
        error_message: str,
        output_dir: str,
        extra_files: list[dict],
        max_files: int = 10,
        max_chars: int = 3000,
    ) -> str:
        all_files   = list_files(output_dir)
        error_lower = error_message.lower()

        # Detect which domain the error belongs to
        is_backend = any(k in error_lower for k in [
            "traceback", "fastapi", "uvicorn", "sqlalchemy", "pydantic",
            "importerror", "attributeerror", "typeerror", "backend",
            ".py", "router", "model", "schema",
        ])
        is_frontend = any(k in error_lower for k in [
            "typescript", "next", "react", "tsx", "npm",
            "cannot find module", "is not a function",
            "frontend", "component", "undefined",
        ])
        if not is_backend and not is_frontend:
            is_backend = is_frontend = True

        # Files explicitly mentioned in the traceback
        mentioned: list[str] = []
        for line in error_message.splitlines():
            for f in all_files:
                if Path(f).name in line and f not in mentioned:
                    mentioned.append(f)

        # Key files per domain
        key_backend = [
            "backend/main.py", "backend/auth.py", "backend/database.py",
            "backend/config.py", "backend/models/models.py",
            "backend/schemas/schemas.py", "backend/requirements.txt",
        ]
        key_frontend = [
            "frontend/lib/api.ts", "frontend/lib/auth.ts",
            "frontend/types/index.ts", "frontend/app/layout.tsx",
            "frontend/next.config.js", "frontend/package.json",
        ]

        priority: list[str] = []
        if is_backend:
            priority += key_backend
        if is_frontend:
            priority += key_frontend

        # Build ordered unique list: mentioned > priority > rest
        ordered: list[str] = []
        for f in mentioned + priority + all_files:
            norm = f.replace("\\", "/")
            if norm not in ordered:
                ordered.append(norm)

        sections: list[str] = []

        # User-provided extra file contents
        for ef in extra_files:
            path    = ef.get("path", "unknown")
            content = ef.get("content", "")
            sections.append(f"=== {path} (provided by user) ===\n{content}")

        # Files from disk
        remaining = max_files - len(sections)
        for rel_path in ordered[:remaining]:
            full_path = str(Path(output_dir) / rel_path)
            raw = read_file(full_path)
            if raw.startswith("[Error]") or raw.startswith("[Skipped]"):
                continue
            truncated = raw[:max_chars]
            if len(raw) > max_chars:
                truncated += f"\n... [truncated — {len(raw) - max_chars} more chars]"
            sections.append(f"=== {rel_path} ===\n{truncated}")

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # File writer
    # ------------------------------------------------------------------

    def _write_files(self, files: list | dict, output_dir: str) -> list[str]:
        if not isinstance(files, list):
            files = [files]
        written = []
        for f in files:
            if not isinstance(f, dict):
                continue
            rel_path = f.get("path", "")
            content  = f.get("content", "")
            if rel_path == "__need_more_context__":
                logger.info("Fixer needs more context — missing file: %s", content)
                continue
            if not rel_path or content is None:
                continue
            full_path = str(Path(output_dir) / rel_path)
            create_file(full_path, content)
            written.append(full_path)
            logger.info("Fixer wrote: %s", full_path)
        return written

    @staticmethod
    def _summary(written: list[str], error: str) -> str:
        short_error = error.splitlines()[0][:100] if error else "unknown error"
        return (
            f"Fixed {len(written)} file(s) for: {short_error}\n"
            + "\n".join(f"  - {p}" for p in written)
        )