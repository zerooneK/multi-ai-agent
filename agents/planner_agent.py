"""
agents/planner_agent.py
-----------------------
The Planner Agent analyses the user's requirement and produces a
structured ProjectPlan JSON consumed by all downstream agents.

It is the FIRST agent called in the pipeline. Nothing else runs
until the plan is ready.

Robustness notes
----------------
- Uses max_tokens=16384 to avoid truncated JSON on large plans
- On parse failure, retries with a minimal "compact" prompt
  that asks for a smaller JSON (fewer fields) — helps small/local models
- extract_json() in BaseAgent handles <think> tags and truncation recovery
"""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from config import cfg
from models.messages import AgentMessage, TaskStatus
from models.project_plan import ProjectPlan

logger = logging.getLogger("agent.planner")

# Tokens for the first attempt — generous to avoid truncation
_MAX_TOKENS_FULL    = 16_384
# Tokens for the compact retry — smaller JSON, smaller model can handle it
_MAX_TOKENS_COMPACT = 8_192


class PlannerAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "planner"

    @property
    def system(self) -> str:
        return """You are a senior software architect and project planner.

Your job is to analyse a user requirement and produce a COMPLETE, DETAILED
project plan as a single JSON object.

The stack is FIXED — do not change it:
  Backend  : Python + FastAPI + SQLAlchemy + SQLite + JWT auth
  Frontend : Next.js 14 (App Router) + TypeScript + Tailwind CSS + shadcn/ui
  Auth     : JWT stored in httpOnly cookies (set by backend, read by Next.js middleware)

The JSON must follow this exact schema:
{
  "project_name": "snake_case_name",
  "description":  "one sentence description",
  "tech_stack": {
    "backend_framework":  "FastAPI",
    "backend_language":   "Python",
    "database":           "SQLite",
    "orm":                "SQLAlchemy",
    "frontend_framework": "Next.js 14 + TypeScript + Tailwind CSS + shadcn/ui",
    "auth":               "JWT httpOnly cookie"
  },
  "folder_structure": {
    "root": "project_name",
    "dirs": [
      "backend",
      "backend/models",
      "backend/routers",
      "backend/schemas",
      "frontend",
      "frontend/app",
      "frontend/app/(auth)",
      "frontend/app/(auth)/login",
      "frontend/app/(auth)/register",
      "frontend/app/(dashboard)",
      "frontend/components",
      "frontend/components/ui",
      "frontend/lib",
      "frontend/types"
    ],
    "files": [
      "backend/main.py",
      "backend/database.py",
      "backend/auth.py",
      "backend/models/models.py",
      "backend/routers/users.py",
      "backend/schemas/schemas.py",
      "backend/requirements.txt",
      "frontend/app/layout.tsx",
      "frontend/app/page.tsx",
      "frontend/app/(auth)/login/page.tsx",
      "frontend/app/(auth)/register/page.tsx",
      "frontend/components/NavBar.tsx",
      "frontend/lib/api.ts",
      "frontend/lib/auth.ts",
      "frontend/types/index.ts",
      "frontend/package.json",
      "frontend/tsconfig.json",
      "frontend/next.config.js",
      "frontend/tailwind.config.ts",
      "frontend/.env.local"
    ]
  },
  "database_models": [
    {
      "name": "ModelName",
      "description": "...",
      "fields": [
        {"name": "id",         "type": "Integer", "primary_key": true},
        {"name": "title",      "type": "String",  "nullable": false},
        {"name": "created_at", "type": "DateTime","nullable": false}
      ]
    }
  ],
  "api_endpoints": [
    {
      "method":        "POST",
      "path":          "/api/auth/register",
      "description":   "Register a new user",
      "auth_required": false,
      "request_body":  {"email": "string", "password": "string"},
      "response":      {"access_token": "string", "token_type": "bearer"}
    },
    {
      "method":        "GET",
      "path":          "/api/books",
      "description":   "List all books",
      "auth_required": true,
      "request_body":  null,
      "response":      {"type": "array", "items": {"$ref": "Book"}}
    }
  ],
  "extra_notes": "Backend pip packages needed. CORS must allow http://localhost:3000. Cookie name: access_token."
}

CRITICAL RULES:
- Output ONLY the JSON object. No prose, no markdown fences, no <think> tags, no explanation.
- Start your response with { and end with }
- Be thorough: include ALL models and ALL endpoints needed.
- Always include auth endpoints (register, login, logout, /me).
- Add a dynamic route per database model in folder_structure.files.
- frontend_framework must always be "Next.js 14 + TypeScript + Tailwind CSS + shadcn/ui".
- Include pip requirements in extra_notes for the backend.
{self._models_limit_rule()}"""

    def _models_limit_rule(self) -> str:
        """Return a MAX_MODELS rule line if limit is set, else empty string."""
        limit = cfg.MAX_MODELS
        if limit > 0:
            return f"- IMPORTANT: database_models array must contain AT MOST {limit} models (including User). Choose only the most essential ones."
        return ""

    # ------------------------------------------------------------------
    # Compact system prompt — used as fallback for small/local models
    # Produces a smaller but valid ProjectPlan JSON
    # ------------------------------------------------------------------

    @property
    def _system_compact(self) -> str:
        return """You are a project planner. Output a JSON project plan.

Output ONLY valid JSON. Start with { end with }. No markdown, no explanation.

Schema:
{
  "project_name": "snake_case",
  "description": "one sentence",
  "tech_stack": {
    "backend_framework": "FastAPI",
    "backend_language": "Python",
    "database": "SQLite",
    "orm": "SQLAlchemy",
    "frontend_framework": "Next.js 14 + TypeScript + Tailwind CSS + shadcn/ui",
    "auth": "JWT httpOnly cookie"
  },
  "folder_structure": {
    "root": "project_name",
    "dirs": ["backend","backend/models","backend/routers","backend/schemas","frontend","frontend/app","frontend/components","frontend/lib","frontend/types"],
    "files": ["backend/main.py","backend/models/models.py","backend/routers/items.py","backend/schemas/schemas.py","backend/auth.py","backend/database.py","backend/requirements.txt","frontend/package.json","frontend/tsconfig.json","frontend/next.config.js","frontend/app/layout.tsx","frontend/app/page.tsx","frontend/app/(auth)/login/page.tsx","frontend/lib/api.ts","frontend/types/index.ts"]
  },
  "database_models": [
    {"name": "User", "description": "App user", "fields": [{"name":"id","type":"Integer","primary_key":true},{"name":"email","type":"String","nullable":false},{"name":"password_hash","type":"String","nullable":false}]},
    {"name": "Item", "description": "Main data model", "fields": [{"name":"id","type":"Integer","primary_key":true},{"name":"title","type":"String","nullable":false},{"name":"user_id","type":"Integer","nullable":false}]}
  ],
  "api_endpoints": [
    {"method":"POST","path":"/api/auth/register","description":"Register","auth_required":false,"request_body":null,"response":null},
    {"method":"POST","path":"/api/auth/login","description":"Login","auth_required":false,"request_body":null,"response":null},
    {"method":"GET","path":"/api/items","description":"List items","auth_required":true,"request_body":null,"response":null},
    {"method":"POST","path":"/api/items","description":"Create item","auth_required":true,"request_body":null,"response":null},
    {"method":"PUT","path":"/api/items/{id}","description":"Update item","auth_required":true,"request_body":null,"response":null},
    {"method":"DELETE","path":"/api/items/{id}","description":"Delete item","auth_required":true,"request_body":null,"response":null}
  ],
  "extra_notes": "pip install fastapi uvicorn sqlalchemy python-jose passlib python-multipart pydantic-settings. CORS allow http://localhost:3000."
}"""

    def run(self, message: AgentMessage) -> AgentMessage:
        message.mark_running()
        requirement = message.payload.get("requirement", "")

        logger.info("Planning project for: %s", requirement[:100])

        # ── Attempt 1: full detailed plan ────────────────────────────
        prompt = (
            f"Analyse this requirement and produce the complete project plan JSON:\n\n"
            f"REQUIREMENT:\n{requirement}\n\n"
            f"Output ONLY valid JSON starting with {{ and ending with }}"
        )

        try:
            raw      = self.chat(prompt, max_tokens=_MAX_TOKENS_FULL)
            plan     = self._parse_plan(raw)
            message.mark_done(plan.to_json())
            logger.info(
                "Plan created: %s | %d models | %d endpoints",
                plan.project_name, len(plan.database_models),
                len(plan.api_endpoints),
            )
            return message

        except Exception as exc:
            logger.warning(
                "Full plan failed (%s) — retrying with compact prompt", exc
            )

        # ── Attempt 2: compact plan (for small/local models) ─────────
        compact_prompt = (
            f"Create a project plan for: {requirement}\n\n"
            f"Rename 'Item'/'items' to match the actual domain "
            f"(e.g. 'Todo'/'todos' for a todo app).\n"
            f"Output ONLY valid JSON starting with {{ and ending with }}"
        )

        try:
            # Temporarily swap system prompt to compact version
            original_system = self.__class__.system.fget  # type: ignore[attr-defined]
            self.__class__.system = property(lambda self: self._system_compact)

            raw  = self.chat(compact_prompt, max_tokens=_MAX_TOKENS_COMPACT)
            plan = self._parse_plan(raw)

            # Restore original system prompt
            self.__class__.system = property(original_system)

            message.mark_done(plan.to_json())
            logger.info(
                "Compact plan created: %s | %d models | %d endpoints",
                plan.project_name, len(plan.database_models),
                len(plan.api_endpoints),
            )

        except Exception as exc2:
            # Restore original system prompt even on failure
            try:
                self.__class__.system = property(original_system)
            except Exception:
                pass
            logger.error("Compact plan also failed: %s", exc2)
            message.mark_failed(str(exc2))

        return message

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_plan(self, raw: str) -> ProjectPlan:
        """Extract JSON from raw LLM response, enforce MAX_MODELS, build ProjectPlan."""
        plan_dict = self.extract_json(raw)
        plan      = ProjectPlan.from_dict(plan_dict)

        max_models = cfg.MAX_MODELS
        if max_models > 0 and len(plan.database_models) > max_models:
            logger.warning(
                "Plan has %d models — trimming to MAX_MODELS=%d",
                len(plan.database_models), max_models,
            )
            plan = ProjectPlan(
                project_name   = plan.project_name,
                description    = plan.description,
                tech_stack     = plan.tech_stack,
                folder_structure = plan.folder_structure,
                database_models  = plan.database_models[:max_models],
                api_endpoints    = plan.api_endpoints,
                extra_notes      = plan.extra_notes,
            )

        return plan