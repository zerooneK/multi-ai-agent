"""
models/project_plan.py
----------------------
Structured output produced by the Planner Agent.

The ProjectPlan is the single source of truth shared by all downstream
agents — Backend, Frontend, and QA all read from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApiEndpoint:
    """Describes one REST API endpoint."""
    method:      str          # GET | POST | PUT | DELETE | PATCH
    path:        str          # e.g. /api/books
    description: str
    request_body: dict | None = None   # JSON Schema of request body
    response:    dict | None  = None   # JSON Schema of response
    auth_required: bool       = False


@dataclass
class DatabaseModel:
    """Describes one database table / ORM model."""
    name:        str           # e.g. Book, User, Rental
    description: str
    fields:      list[dict]    # [{name, type, nullable, unique, ...}]


@dataclass
class TechStack:
    backend_framework:  str = "FastAPI"
    backend_language:   str = "Python"
    database:           str = "SQLite"
    orm:                str = "SQLAlchemy"
    frontend_framework: str = "Vanilla JS + TailwindCSS"
    auth:               str = "JWT"
    package_manager:    str = "pip"


@dataclass
class FolderStructure:
    """Describes the target directory layout."""
    root:     str               # project root directory name
    dirs:     list[str]         # directories to create
    files:    list[str]         # files that will be generated


@dataclass
class ProjectPlan:
    """
    Complete plan produced by the Planner Agent.

    All other agents receive this plan as their primary input.

    Attributes
    ----------
    project_name   : Snake-case project identifier.
    description    : One-sentence description of the project.
    tech_stack     : Technology choices.
    folder_structure : Directories and files to create.
    database_models  : ORM models the backend must implement.
    api_endpoints    : REST API contract.
    extra_notes      : Any additional context for agents.
    """
    project_name:     str
    description:      str
    tech_stack:       TechStack            = field(default_factory=TechStack)
    folder_structure: FolderStructure | None = None
    database_models:  list[DatabaseModel]  = field(default_factory=list)
    api_endpoints:    list[ApiEndpoint]    = field(default_factory=list)
    extra_notes:      str                  = ""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (JSON-serialisable)."""
        return {
            "project_name": self.project_name,
            "description":  self.description,
            "tech_stack": {
                "backend_framework":  self.tech_stack.backend_framework,
                "backend_language":   self.tech_stack.backend_language,
                "database":           self.tech_stack.database,
                "orm":                self.tech_stack.orm,
                "frontend_framework": self.tech_stack.frontend_framework,
                "auth":               self.tech_stack.auth,
            },
            "folder_structure": {
                "root":  self.folder_structure.root  if self.folder_structure else "",
                "dirs":  self.folder_structure.dirs  if self.folder_structure else [],
                "files": self.folder_structure.files if self.folder_structure else [],
            },
            "database_models": [
                {
                    "name":        m.name,
                    "description": m.description,
                    "fields":      m.fields,
                }
                for m in self.database_models
            ],
            "api_endpoints": [
                {
                    "method":       e.method,
                    "path":         e.path,
                    "description":  e.description,
                    "request_body": e.request_body,
                    "response":     e.response,
                    "auth_required": e.auth_required,
                }
                for e in self.api_endpoints
            ],
            "extra_notes": self.extra_notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectPlan":
        """Reconstruct a ProjectPlan from a plain dict."""
        ts_data = data.get("tech_stack", {})
        tech_stack = TechStack(
            backend_framework  = ts_data.get("backend_framework",  "FastAPI"),
            backend_language   = ts_data.get("backend_language",   "Python"),
            database           = ts_data.get("database",           "SQLite"),
            orm                = ts_data.get("orm",                 "SQLAlchemy"),
            frontend_framework = ts_data.get("frontend_framework", "Vanilla JS + TailwindCSS"),
            auth               = ts_data.get("auth",               "JWT"),
        )

        fs_data = data.get("folder_structure", {})
        folder_structure = FolderStructure(
            root  = fs_data.get("root",  "project"),
            dirs  = fs_data.get("dirs",  []),
            files = fs_data.get("files", []),
        )

        database_models = [
            DatabaseModel(
                name        = m["name"],
                description = m.get("description", ""),
                fields      = m.get("fields", []),
            )
            for m in data.get("database_models", [])
        ]

        api_endpoints = [
            ApiEndpoint(
                method        = e["method"],
                path          = e["path"],
                description   = e.get("description", ""),
                request_body  = e.get("request_body"),
                response      = e.get("response"),
                auth_required = e.get("auth_required", False),
            )
            for e in data.get("api_endpoints", [])
        ]

        return cls(
            project_name     = data.get("project_name", "project"),
            description      = data.get("description",  ""),
            tech_stack       = tech_stack,
            folder_structure = folder_structure,
            database_models  = database_models,
            api_endpoints    = api_endpoints,
            extra_notes      = data.get("extra_notes", ""),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ProjectPlan":
        """Parse a JSON string (strips markdown fences if present)."""
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()
        return cls.from_dict(json.loads(clean))