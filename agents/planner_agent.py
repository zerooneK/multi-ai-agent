"""
agents/planner_agent.py
-----------------------
The Planner Agent analyses the user's requirement and produces a
structured ProjectPlan JSON including per-file ModuleSpec contracts.

v2 changes
----------
- Added `modules` array to the output schema
- Each module has: path, description, layer, imports, exports,
  interfaces (signatures), and dod (Definition-of-Done checklist)
- Modules are ordered by dependency (leaf modules first)
"""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from config import cfg
from models.messages import AgentMessage
from models.project_plan import ProjectPlan

logger = logging.getLogger("agent.planner")

_MAX_TOKENS_FULL    = 16_384
_MAX_TOKENS_COMPACT = 8_192


class PlannerAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "planner"

    @property
    def system(self) -> str:
        return """You are a senior software architect. Your job is to produce a
COMPLETE project plan as a single JSON object.

Stack is FIXED:
  Backend  : Python + FastAPI + SQLAlchemy + SQLite + JWT auth (httpOnly cookie)
  Frontend : Next.js 14 (App Router) + TypeScript + Tailwind CSS + shadcn/ui

Output ONLY valid JSON. Start with { and end with }. No markdown, no explanation.

JSON schema:
{
  "project_name": "snake_case_name",
  "description":  "one sentence",
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
    "dirs": ["backend","backend/models","backend/routers","backend/schemas",
             "frontend","frontend/app","frontend/components","frontend/lib","frontend/types"],
    "files": ["backend/main.py","backend/database.py","backend/auth.py",
              "backend/models/models.py","backend/schemas/schemas.py",
              "backend/routers/items.py","backend/requirements.txt",
              "frontend/package.json","frontend/next.config.js",
              "frontend/app/layout.tsx","frontend/app/page.tsx",
              "frontend/lib/api.ts","frontend/types/index.ts"]
  },
  "database_models": [
    {
      "name": "User",
      "description": "App user with auth",
      "fields": [
        {"name":"id","type":"Integer","primary_key":true},
        {"name":"email","type":"String","nullable":false,"unique":true},
        {"name":"password_hash","type":"String","nullable":false},
        {"name":"created_at","type":"DateTime","nullable":false}
      ]
    }
  ],
  "api_endpoints": [
    {"method":"POST","path":"/api/auth/register","description":"Register","auth_required":false,"request_body":{"email":"string","password":"string"},"response":{"message":"string"}},
    {"method":"POST","path":"/api/auth/login","description":"Login","auth_required":false,"request_body":{"email":"string","password":"string"},"response":{"message":"string"}},
    {"method":"POST","path":"/api/auth/logout","description":"Logout","auth_required":true,"request_body":null,"response":{"message":"string"}},
    {"method":"GET","path":"/api/auth/me","description":"Current user","auth_required":true,"request_body":null,"response":{"id":"int","email":"string"}}
  ],
  "modules": [
    {
      "path": "backend/database.py",
      "description": "SQLAlchemy engine, SessionLocal, Base, get_db dependency",
      "layer": "backend",
      "imports": [],
      "exports": ["engine","SessionLocal","Base","get_db"],
      "interfaces": [
        {"name":"get_db","params":{},"returns":"Generator[Session, None, None]"}
      ],
      "dod": [
        "Base = declarative_base() must be present",
        "get_db must yield a Session and close it in finally block",
        "DATABASE_URL must be read from environment with SQLite default"
      ]
    },
    {
      "path": "backend/models/models.py",
      "description": "SQLAlchemy ORM models",
      "layer": "backend",
      "imports": ["backend/database.py"],
      "exports": ["User"],
      "interfaces": [
        {"name":"User","params":{"id":"int","email":"str","password_hash":"str"},"returns":"SQLAlchemy model"}
      ],
      "dod": [
        "All models must inherit from Base",
        "User model must have id, email, password_hash fields",
        "email field must have unique=True"
      ]
    },
    {
      "path": "backend/schemas/schemas.py",
      "description": "Pydantic v2 request/response schemas",
      "layer": "backend",
      "imports": ["backend/models/models.py"],
      "exports": ["UserCreate","UserResponse","Token"],
      "interfaces": [
        {"name":"UserCreate","params":{"email":"EmailStr","password":"str"},"returns":"BaseModel"},
        {"name":"UserResponse","params":{"id":"int","email":"str"},"returns":"BaseModel"}
      ],
      "dod": [
        "Use Pydantic v2 (from pydantic import BaseModel)",
        "UserCreate must have email as EmailStr",
        "UserResponse must have model_config = ConfigDict(from_attributes=True)"
      ]
    },
    {
      "path": "backend/auth.py",
      "description": "JWT creation/verification and password hashing",
      "layer": "backend",
      "imports": ["backend/database.py","backend/models/models.py","backend/schemas/schemas.py"],
      "exports": ["create_access_token","get_current_user","verify_password","get_password_hash"],
      "interfaces": [
        {"name":"create_access_token","params":{"data":"dict"},"returns":"str"},
        {"name":"get_current_user","params":{"token":"str","db":"Session"},"returns":"User"},
        {"name":"verify_password","params":{"plain":"str","hashed":"str"},"returns":"bool"},
        {"name":"get_password_hash","params":{"password":"str"},"returns":"str"}
      ],
      "dod": [
        "JWT must be read from httpOnly cookie named access_token",
        "Password must be hashed with passlib bcrypt",
        "get_current_user must raise HTTPException 401 if token invalid",
        "SECRET_KEY must be read from environment"
      ]
    },
    {
      "path": "backend/routers/auth.py",
      "description": "Auth endpoints: register, login, logout, /me",
      "layer": "backend",
      "imports": ["backend/database.py","backend/models/models.py","backend/schemas/schemas.py","backend/auth.py"],
      "exports": ["router"],
      "interfaces": [
        {"name":"router","params":{},"returns":"APIRouter with prefix /api/auth"}
      ],
      "dod": [
        "router = APIRouter(prefix='/api/auth')",
        "POST /register must hash password before saving",
        "POST /login must set access_token as httpOnly cookie",
        "POST /logout must delete the access_token cookie",
        "GET /me must require authentication"
      ]
    },
    {
      "path": "backend/main.py",
      "description": "FastAPI app entry point with CORS and all routers",
      "layer": "backend",
      "imports": ["backend/database.py","backend/routers/auth.py"],
      "exports": ["app"],
      "interfaces": [
        {"name":"app","params":{},"returns":"FastAPI instance"}
      ],
      "dod": [
        "CORS must allow http://localhost:3000 with credentials",
        "All routers must be registered with app.include_router()",
        "Database tables must be created on startup"
      ]
    },
    {
      "path": "frontend/types/index.ts",
      "description": "Shared TypeScript type definitions",
      "layer": "frontend",
      "imports": [],
      "exports": ["User"],
      "interfaces": [
        {"name":"User","params":{"id":"number","email":"string"},"returns":"interface"}
      ],
      "dod": [
        "All types must be exported",
        "Types must match backend Pydantic response schemas"
      ]
    },
    {
      "path": "frontend/lib/api.ts",
      "description": "Fetch wrapper with credentials for backend API calls",
      "layer": "frontend",
      "imports": ["frontend/types/index.ts"],
      "exports": ["apiFetch"],
      "interfaces": [
        {"name":"apiFetch","params":{"path":"string","options":"RequestInit"},"returns":"Promise<T>"}
      ],
      "dod": [
        "apiFetch must include credentials: 'include' for cookie auth",
        "NEXT_PUBLIC_API_URL must be used as base URL with http://localhost:8000 default",
        "Must throw Error with backend detail message on non-ok response"
      ]
    },
    {
      "path": "frontend/app/layout.tsx",
      "description": "Root layout with global styles and font",
      "layer": "frontend",
      "imports": [],
      "exports": ["default"],
      "interfaces": [
        {"name":"default","params":{"children":"React.ReactNode"},"returns":"JSX.Element"}
      ],
      "dod": [
        "Must export default RootLayout",
        "Must include <html> and <body> tags",
        "Must import global CSS"
      ]
    }
  ],
  "extra_notes": "CORS allow localhost:3000. Cookie name: access_token. pip packages: fastapi uvicorn sqlalchemy python-jose[cryptography] passlib[bcrypt] python-multipart pydantic-settings pydantic[email] email-validator"
}

CRITICAL RULES:
- Output ONLY JSON. Start with { end with }
- modules array: list every file that will be generated, ordered so dependencies come FIRST
- Each module's imports[] must only list OTHER modules in this same modules array
- Do NOT add circular dependencies (A imports B that imports A)
- Always include auth endpoints (register, login, logout, /me)
- Include domain-specific models and routers for the actual requirement
- backend/requirements.txt and backend/.env.example are config files — add them to folder_structure.files but NOT to modules[]
- Always include all the example modules above PLUS domain-specific ones
""" + self._models_limit_rule()

    def _models_limit_rule(self) -> str:
        limit = cfg.MAX_MODELS
        if limit > 0:
            return (
                f"\n- IMPORTANT: database_models must have AT MOST {limit} models "
                f"(including User). Choose only the most essential."
            )
        return ""

    @property
    def _system_compact(self) -> str:
        return """You are a project planner. Output a JSON project plan.
Output ONLY valid JSON. Start with { end with }. No markdown, no explanation.

Include these required fields: project_name, description, tech_stack, folder_structure,
database_models, api_endpoints, modules, extra_notes.

For modules[], include at minimum:
  backend/database.py, backend/models/models.py, backend/schemas/schemas.py,
  backend/auth.py, backend/routers/auth.py, backend/main.py,
  frontend/types/index.ts, frontend/lib/api.ts, frontend/app/layout.tsx

Each module needs: path, description, layer, imports, exports, interfaces, dod.
Imports must only reference other modules in the list — no circular deps.
"""

    def run(self, message: AgentMessage) -> AgentMessage:
        message.mark_running()
        requirement = message.payload.get("requirement", "")
        logger.info("Planning project for: %s", requirement[:100])

        prompt = (
            f"Analyse this requirement and produce the complete project plan JSON:\n\n"
            f"REQUIREMENT:\n{requirement}\n\n"
            f"Output ONLY valid JSON starting with {{ and ending with }}"
        )

        # ── Attempt 1: full plan ──────────────────────────────────────
        try:
            raw  = self.chat(prompt, max_tokens=_MAX_TOKENS_FULL)
            plan = self._parse_plan(raw)
            message.mark_done(plan.to_json())
            logger.info(
                "Plan created: %s | %d models | %d endpoints | %d modules",
                plan.project_name, len(plan.database_models),
                len(plan.api_endpoints), len(plan.modules),
            )
            return message
        except Exception as exc:
            logger.warning("Full plan failed (%s) — retrying compact", exc)

        # ── Attempt 2: compact plan ───────────────────────────────────
        compact_prompt = (
            f"Create a project plan for: {requirement}\n\n"
            f"Rename 'Item'/'items' to match the domain.\n"
            f"Output ONLY valid JSON starting with {{ and ending with }}"
        )
        try:
            original_system         = self.__class__.system.fget  # type: ignore[attr-defined]
            self.__class__.system   = property(lambda self: self._system_compact)
            raw  = self.chat(compact_prompt, max_tokens=_MAX_TOKENS_COMPACT)
            plan = self._parse_plan(raw)
            self.__class__.system   = property(original_system)
            message.mark_done(plan.to_json())
            logger.info(
                "Compact plan: %s | %d modules",
                plan.project_name, len(plan.modules),
            )
        except Exception as exc2:
            try:
                self.__class__.system = property(original_system)
            except Exception:
                pass
            logger.error("Compact plan also failed: %s", exc2)
            message.mark_failed(str(exc2))

        return message

    def _parse_plan(self, raw: str) -> ProjectPlan:
        plan_dict  = self.extract_json(raw)
        plan       = ProjectPlan.from_dict(plan_dict)

        # Enforce MAX_MODELS
        max_models = cfg.MAX_MODELS
        if max_models > 0 and len(plan.database_models) > max_models:
            logger.warning(
                "Trimming models from %d to MAX_MODELS=%d",
                len(plan.database_models), max_models,
            )
            plan = ProjectPlan(
                project_name     = plan.project_name,
                description      = plan.description,
                tech_stack       = plan.tech_stack,
                folder_structure = plan.folder_structure,
                database_models  = plan.database_models[:max_models],
                api_endpoints    = plan.api_endpoints,
                modules          = plan.modules,
                extra_notes      = plan.extra_notes,
            )

        # Warn if no modules (old-style plan) — orchestrator will fall back
        if not plan.modules:
            logger.warning(
                "Plan has no modules[] — orchestrator will use legacy bulk generation"
            )

        return plan