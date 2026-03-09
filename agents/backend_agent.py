"""
agents/backend_agent.py
-----------------------
The Backend Agent reads the ProjectPlan and generates all backend
code: FastAPI app, SQLAlchemy models, Pydantic schemas, routers,
auth, database config, and requirements.txt.

Robustness notes
----------------
- Uses max_tokens=16384 to avoid truncated output
- On JSON parse failure, retries with a stricter "JSON-only" prompt
  that forces the model to start with [ immediately
- Writes files one-by-one so partial success is preserved
"""

from __future__ import annotations

import logging
from pathlib import Path

from agents.base_agent import BaseAgent
from models.messages import AgentMessage
from models.project_plan import ProjectPlan
from tools.file_tools import create_file

logger = logging.getLogger("agent.backend")


class BackendAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "backend"

    @property
    def system(self) -> str:
        return """You are a senior Python backend engineer specialising in FastAPI.

IMPORTANT: Your response must be a JSON array and nothing else.
Start your response with [ and end with ]
Do NOT write any explanation, prose, or markdown before or after the JSON.

Output format — a JSON array of file objects:
[
  {"path": "backend/main.py", "content": "full file content"},
  {"path": "backend/database.py", "content": "full file content"},
  ...
]

Code rules:
- Every file must be 100% complete — no placeholders, no # TODO
- SQLAlchemy ORM with proper relationships
- Pydantic v2 (from pydantic import BaseModel) — no deprecated v1 imports
- HTTPException for all error responses
- CORS middleware in main.py allowing http://localhost:3000
- python-jose for JWT, passlib[bcrypt] for password hashing
- __init__.py in every package directory
- DATABASE_URL read from env variable with SQLite default
- JWT stored as httpOnly cookie named access_token
"""

    def run(self, message: AgentMessage) -> AgentMessage:
        message.mark_running()

        plan_json   = message.payload.get("plan_json", "{}")
        output_dir  = message.payload.get("output_dir", "output/project")
        fix_context = message.payload.get("fix_context", "")

        try:
            plan = ProjectPlan.from_json(plan_json)
        except Exception as exc:
            message.mark_failed(f"Could not parse plan: {exc}")
            return message

        logger.info("Backend agent generating code for: %s", plan.project_name)

        prompt = self._fix_prompt(plan, fix_context) if fix_context \
            else self._generate_prompt(plan)

        # ── Attempt 1: normal ────────────────────────────────────────
        written = []
        try:
            raw   = self.chat(prompt, max_tokens=16384)
            files = self.extract_json(raw)
            written = self._write_files(files, output_dir)
            message.mark_done(self._summary(written))
            return message
        except Exception as exc:
            logger.warning("Backend attempt 1 failed (%s) — retrying", exc)

        # ── Attempt 2: force JSON-only prompt ────────────────────────
        try:
            retry_prompt = self._json_only_prompt(plan, fix_context)
            raw   = self.chat(retry_prompt, max_tokens=16384)
            files = self.extract_json(raw)
            written = self._write_files(files, output_dir)
            message.mark_done(self._summary(written))
        except Exception as exc2:
            logger.error("Backend agent failed: %s", exc2)
            # Save whatever was written before failing
            if written:
                message.mark_done(self._summary(written) + f"\n[partial — error: {exc2}]")
            else:
                message.mark_failed(str(exc2))

        return message

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _generate_prompt(self, plan: ProjectPlan) -> str:
        return f"""Output a JSON array of backend files for this project.
Start with [ and end with ] — no prose before or after.

PROJECT PLAN:
{plan.to_json()}

Files to generate:
- backend/__init__.py
- backend/main.py          (FastAPI app, CORS, routers, startup event)
- backend/database.py      (SQLAlchemy engine, SessionLocal, Base, get_db)
- backend/auth.py          (JWT create/verify, password hash, OAuth2 scheme, httpOnly cookie)
- backend/config.py        (settings: DATABASE_URL, SECRET_KEY, from os.environ)
- backend/models/__init__.py
- backend/models/models.py (all ORM models from the plan)
- backend/schemas/__init__.py
- backend/schemas/schemas.py (all Pydantic v2 schemas — request + response)
- backend/routers/__init__.py
{self._router_files(plan)}
- requirements.txt
- .env.example

["""

    def _fix_prompt(self, plan: ProjectPlan, fix_context: str) -> str:
        return f"""Output a JSON array of CORRECTED backend files.
Start with [ and end with ] — no prose before or after.

PROJECT PLAN:
{plan.to_json()}

ERRORS TO FIX:
{fix_context}

Only include files that need changes.

["""

    def _json_only_prompt(self, plan: ProjectPlan, fix_context: str) -> str:
        """Minimal prompt for models that add prose despite instructions."""
        models_summary = ", ".join(m.name for m in plan.database_models) or "Item"
        routers        = ", ".join(
            f"backend/routers/{m.name.lower()}s.py"
            for m in plan.database_models
        ) or "backend/routers/items.py"

        context = f"Fix these errors:\n{fix_context}\n\n" if fix_context else ""
        return f"""{context}Output JSON array only. No text before [. No text after ].

[
  {{"path": "backend/__init__.py", "content": ""}},
  {{"path": "backend/main.py", "content": "<FastAPI app for {plan.project_name}, CORS allow localhost:3000, routers for {models_summary}>"}},
  {{"path": "backend/database.py", "content": "<SQLAlchemy engine, Base, get_db>"}},
  {{"path": "backend/auth.py", "content": "<JWT httpOnly cookie, passlib bcrypt>"}},
  {{"path": "backend/models/__init__.py", "content": ""}},
  {{"path": "backend/models/models.py", "content": "<ORM models: {models_summary}>"}},
  {{"path": "backend/schemas/__init__.py", "content": ""}},
  {{"path": "backend/schemas/schemas.py", "content": "<Pydantic v2 schemas: {models_summary}>"}},
  {{"path": "backend/routers/__init__.py", "content": ""}},
  {{"path": "{routers.split(', ')[0]}", "content": "<CRUD endpoints for {models_summary}>"}},
  {{"path": "requirements.txt", "content": "fastapi\\nuvicorn\\nsqlalchemy\\npython-jose[cryptography]\\npasslib[bcrypt]\\npython-multipart\\npydantic-settings"}}
]

Replace every <...> placeholder with real, complete Python code.
Output the complete JSON array:"""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_files(self, files: list | dict, output_dir: str) -> list[str]:
        """Write file objects to disk. Returns list of written paths."""
        if not isinstance(files, list):
            files = [files]
        written = []
        for f in files:
            if not isinstance(f, dict):
                continue
            rel_path = f.get("path", "")
            content  = f.get("content", "")
            if not rel_path or content is None:
                continue
            full_path = str(Path(output_dir) / rel_path)
            create_file(full_path, content)
            written.append(full_path)
            logger.info("Backend wrote: %s", full_path)
        return written

    @staticmethod
    def _summary(written: list[str]) -> str:
        return (
            f"Backend generation complete.\nFiles written ({len(written)}):\n"
            + "\n".join(f"  - {p}" for p in written)
        )

    @staticmethod
    def _router_files(plan: ProjectPlan) -> str:
        routers = {m.name.lower() + "s" for m in plan.database_models} or {"items"}
        return "\n".join(
            f"- backend/routers/{r}.py (CRUD endpoints)"
            for r in sorted(routers)
        )