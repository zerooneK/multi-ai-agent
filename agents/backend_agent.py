"""
agents/backend_agent.py
-----------------------
The Backend Agent reads the ProjectPlan and generates all backend
code: FastAPI app, SQLAlchemy models, Pydantic schemas, routers,
auth, database config, and requirements.txt.

It writes every file directly to the output directory using file_tools.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.base_agent import BaseAgent
from models.messages import AgentMessage, TaskStatus
from models.project_plan import ProjectPlan
from tools.file_tools import create_file, create_directory

logger = logging.getLogger("agent.backend")


class BackendAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "backend"

    @property
    def system(self) -> str:
        return """You are a senior Python backend engineer specialising in FastAPI.

You will receive a project plan and must generate COMPLETE, PRODUCTION-READY
Python backend code.

For each file requested, output a JSON array of file objects:
[
  {
    "path": "relative/path/to/file.py",
    "content": "full file content here"
  },
  ...
]

Rules:
- Output ONLY the JSON array. No prose, no markdown fences.
- Every file must be complete — no placeholders, no "# TODO".
- Use SQLAlchemy ORM with proper relationships.
- Use Pydantic v2 schemas (BaseModel).
- Include proper error handling (HTTPException).
- Include CORS middleware in main.py.
- Use python-jose for JWT auth.
- Use passlib for password hashing.
- Always include an __init__.py in every package directory.
- The database URL must read from an env variable with a SQLite default.
- Write clean, well-commented, production-ready code.
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

        # Build the prompt
        if fix_context:
            prompt = self._fix_prompt(plan, fix_context)
        else:
            prompt = self._generate_prompt(plan)

        try:
            raw  = self.chat(prompt, max_tokens=8192)
            files = self.extract_json(raw)

            if not isinstance(files, list):
                files = [files]

            written: list[str] = []
            for f in files:
                if not isinstance(f, dict):
                    continue
                rel_path = f.get("path", "")
                content  = f.get("content", "")
                if not rel_path or not content:
                    continue

                full_path = str(Path(output_dir) / rel_path)
                result    = create_file(full_path, content)
                written.append(full_path)
                logger.info("Backend wrote: %s", full_path)

            summary = (
                f"Backend generation complete.\n"
                f"Files written ({len(written)}):\n"
                + "\n".join(f"  - {p}" for p in written)
            )
            message.mark_done(summary)

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Backend agent failed: %s", exc)
            message.mark_failed(str(exc))

        return message

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _generate_prompt(self, plan: ProjectPlan) -> str:
        return f"""Generate the complete backend codebase for this project.

PROJECT PLAN:
{plan.to_json()}

Generate ALL of these backend files:
- backend/__init__.py
- backend/main.py          (FastAPI app, CORS, routers, startup)
- backend/database.py      (SQLAlchemy engine, session, Base)
- backend/models/__init__.py
- backend/models/models.py (all ORM models from the plan)
- backend/schemas/__init__.py
- backend/schemas/schemas.py (all Pydantic schemas)
- backend/routers/__init__.py
{self._router_files(plan)}
- backend/auth.py          (JWT creation, password hashing, OAuth2)
- backend/config.py        (settings via pydantic-settings or os.environ)
- requirements.txt         (all pip dependencies)
- .env.example             (example env file with DATABASE_URL, SECRET_KEY)

Output the JSON array of file objects now:"""

    def _fix_prompt(self, plan: ProjectPlan, fix_context: str) -> str:
        return f"""The QA agent found errors in the backend code. Fix them.

PROJECT PLAN:
{plan.to_json()}

QA ERRORS FOUND:
{fix_context}

Generate the CORRECTED files as a JSON array.
Only include files that need to be changed.
Output the JSON array now:"""

    @staticmethod
    def _router_files(plan: ProjectPlan) -> str:
        """List router files based on database models."""
        routers = set()
        for model in plan.database_models:
            routers.add(model.name.lower() + "s")
        if not routers:
            routers = {"items"}
        return "\n".join(
            f"- backend/routers/{r}.py (CRUD endpoints)"
            for r in sorted(routers)
        )
