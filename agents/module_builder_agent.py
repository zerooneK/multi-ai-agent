"""
agents/module_builder_agent.py
-------------------------------
Phase 3 — Builds ONE module at a time according to its ModuleSpec.

Receives:
  - The ModuleSpec for the target file
  - The ProjectPlan (for overall context)
  - Already-built module contents (dependency context)
  - Optional fix_context (when called after a DoD failure)

Outputs:
  - A single file written to disk

This agent is used by the Orchestrator in the sequential build loop.
It replaces the "generate everything at once" approach of BackendAgent /
FrontendAgent for the modules[] portion of the plan.

The original BackendAgent and FrontendAgent are kept as fallback for
plans that have no modules[] (legacy plans).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from agents.base_agent import BaseAgent
from config import cfg
from models.messages import AgentMessage, TaskType
from tools.file_tools import create_file, read_file

if TYPE_CHECKING:
    from models.project_plan import ModuleSpec, ProjectPlan

logger = logging.getLogger("agent.module_builder")


class ModuleBuilderAgent(BaseAgent):
    """
    Builds a single module (one file) according to its ModuleSpec contract.

    One instance handles both backend and frontend modules — the system
    prompt adapts based on the module's layer.
    """

    @property
    def name(self) -> str:
        return "module_builder"

    @property
    def system(self) -> str:
        return """You are a senior full-stack engineer.

You receive a MODULE SPEC describing exactly one file to generate.
The spec includes:
  - path         : where to write the file
  - description  : what this file does
  - imports      : other files this module depends on (their contents provided)
  - exports      : symbol names this file must expose
  - interfaces   : exact function/class signatures to implement
  - dod          : Definition-of-Done checklist you MUST satisfy

Your output: the COMPLETE source code for ONE file only.
Output raw code — no markdown fences, no explanation, no JSON wrapper.
Start the response with the first line of actual code.

Rules for backend (Python/FastAPI):
- Every file must be 100% complete — no placeholders, no # TODO
- Use Pydantic v2 (from pydantic import BaseModel, ConfigDict)
- JWT stored as httpOnly cookie named access_token
- Use python-jose for JWT, passlib[bcrypt] for passwords
- SQLAlchemy ORM — models must inherit from Base
- CORS in main.py must allow http://localhost:3000 with credentials=True

Rules for frontend (Next.js 14 TypeScript):
- 'use client' only on components that use hooks or browser APIs
- Use next/navigation (not next/router)
- apiFetch from lib/api.ts with credentials: 'include'
- NEXT_PUBLIC_ prefix for client-side env vars
- Tailwind CSS classes only — no inline styles
- next.config.js (NOT .ts)
"""

    def run(self, message: AgentMessage) -> AgentMessage:
        """Standard agent interface."""
        message.mark_running()

        spec_dict    = message.payload.get("module_spec", {})
        plan_json    = message.payload.get("plan_json", "{}")
        output_dir   = message.payload.get("output_dir", "output/project")
        dep_contents = message.payload.get("dep_contents", {})  # path → source
        fix_context  = message.payload.get("fix_context", "")

        # Import here to avoid circular at module load time
        from models.project_plan import ModuleSpec, ProjectPlan
        spec = ModuleSpec.from_dict(spec_dict)

        try:
            plan = ProjectPlan.from_json(plan_json)
        except Exception as exc:
            message.mark_failed(f"Could not parse plan: {exc}")
            return message

        logger.info("ModuleBuilder building: %s", spec.path)

        prompt  = self._build_prompt(spec, plan, dep_contents, fix_context)
        written = self._build_with_retry(spec, plan, dep_contents,
                                         fix_context, prompt, output_dir)

        if written:
            message.mark_done(f"Built: {written}")
        else:
            message.mark_failed(f"Failed to build {spec.path}")

        return message

    # ------------------------------------------------------------------
    # Build with retry
    # ------------------------------------------------------------------

    def _build_with_retry(
        self,
        spec: "ModuleSpec",
        plan: "ProjectPlan",
        dep_contents: dict[str, str],
        fix_context: str,
        prompt: str,
        output_dir: str,
    ) -> str | None:
        """Attempt build up to 2 times. Returns written path or None."""

        # Attempt 1: full prompt
        try:
            source = self.chat(prompt, max_tokens=cfg.MAX_OUTPUT_TOKENS)
            source = self._clean_source(source)
            return self._write(spec.path, source, output_dir)
        except Exception as exc:
            logger.warning("ModuleBuilder attempt 1 failed (%s) — retrying", exc)

        # Attempt 2: minimal prompt
        try:
            minimal = self._minimal_prompt(spec, plan, dep_contents, fix_context)
            source  = self.chat(minimal, max_tokens=cfg.MAX_OUTPUT_TOKENS)
            source  = self._clean_source(source)
            return self._write(spec.path, source, output_dir)
        except Exception as exc2:
            logger.error("ModuleBuilder failed for %s: %s", spec.path, exc2)
            return None

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        spec: "ModuleSpec",
        plan: "ProjectPlan",
        dep_contents: dict[str, str],
        fix_context: str,
    ) -> str:
        parts: list[str] = []

        # Section: module spec
        parts.append(f"=== MODULE SPEC ===")
        parts.append(f"File   : {spec.path}")
        parts.append(f"Purpose: {spec.description}")
        parts.append(f"Layer  : {spec.layer}")

        # Exports required
        if spec.exports:
            parts.append(f"\nRequired exports: {', '.join(spec.exports)}")

        # Interface signatures
        if spec.interfaces:
            parts.append("\nInterfaces to implement:")
            for iface in spec.interfaces:
                params = ", ".join(f"{k}: {v}" for k, v in iface.params.items())
                parts.append(f"  - {iface.name}({params}) → {iface.returns}")

        # DoD checklist
        if spec.dod:
            parts.append("\nDefinition of Done (all must be satisfied):")
            for item in spec.dod:
                parts.append(f"  ✓ {item}")

        # Project context (models + endpoints, abbreviated)
        parts.append(f"\n=== PROJECT CONTEXT ===")
        parts.append(f"Project: {plan.project_name} — {plan.description}")

        if plan.database_models:
            parts.append("\nDatabase models:")
            for m in plan.database_models:
                field_names = [f["name"] for f in m.fields]
                parts.append(f"  - {m.name}: {', '.join(field_names)}")

        if plan.api_endpoints:
            parts.append("\nAPI endpoints:")
            for ep in plan.api_endpoints[:10]:  # cap at 10 to save tokens
                parts.append(f"  - {ep.method} {ep.path} (auth={ep.auth_required})")

        # Dependency file contents
        if dep_contents:
            parts.append("\n=== DEPENDENCY FILES (read carefully) ===")
            for dep_path, content in dep_contents.items():
                MAX_DEP_CHARS = 2000
                excerpt = content[:MAX_DEP_CHARS]
                if len(content) > MAX_DEP_CHARS:
                    excerpt += f"\n... [truncated]"
                parts.append(f"\n--- {dep_path} ---\n{excerpt}")

        # Fix context
        if fix_context:
            parts.append(f"\n=== FIX INSTRUCTIONS ===\n{fix_context}")

        parts.append(
            f"\n=== YOUR TASK ===\n"
            f"Write the complete source code for: {spec.path}\n"
            f"Output raw code only. No markdown fences. No explanation.\n"
            f"Start with the first line of actual code."
        )

        return "\n".join(parts)

    def _minimal_prompt(
        self,
        spec: "ModuleSpec",
        plan: "ProjectPlan",
        dep_contents: dict[str, str],
        fix_context: str,
    ) -> str:
        exports_str = ", ".join(spec.exports) or "none"
        dod_str     = "; ".join(spec.dod[:3]) or "none"
        deps_str    = ", ".join(dep_contents.keys()) or "none"
        fix_str     = f"\nFix: {fix_context[:300]}" if fix_context else ""
        return (
            f"Write complete source code for {spec.path}.\n"
            f"Purpose: {spec.description}\n"
            f"Must export: {exports_str}\n"
            f"Key requirements: {dod_str}\n"
            f"Dependencies available: {deps_str}"
            f"{fix_str}\n\n"
            f"Output code only, no markdown:"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_source(raw: str) -> str:
        """Strip markdown code fences if the LLM added them."""
        lines = raw.strip().splitlines()
        # Remove leading ```python / ```typescript / ``` line
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Remove trailing ``` line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    @staticmethod
    def _write(rel_path: str, source: str, output_dir: str) -> str:
        full_path = str(Path(output_dir) / rel_path)
        create_file(full_path, source)
        logger.info("ModuleBuilder wrote: %s (%d chars)", full_path, len(source))
        return full_path