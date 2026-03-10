"""
models/project_plan.py
----------------------
Structured output produced by the Planner Agent.

The ProjectPlan is the single source of truth shared by all downstream
agents — Backend, Frontend, and QA all read from it.

v2 additions
------------
- ModuleSpec : per-file interface contract (imports, exports, signatures, DoD)
- ProjectPlan.modules : list[ModuleSpec] — ordered by build dependency
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApiEndpoint:
    """Describes one REST API endpoint."""
    method:       str           # GET | POST | PUT | DELETE | PATCH
    path:         str           # e.g. /api/books
    description:  str
    request_body: dict | None = None
    response:     dict | None = None
    auth_required: bool       = False


@dataclass
class DatabaseModel:
    """Describes one database table / ORM model."""
    name:        str
    description: str
    fields:      list[dict]    # [{name, type, nullable, unique, ...}]


@dataclass
class TechStack:
    backend_framework:  str = "FastAPI"
    backend_language:   str = "Python"
    database:           str = "SQLite"
    orm:                str = "SQLAlchemy"
    frontend_framework: str = "Next.js 14 + TypeScript + Tailwind CSS + shadcn/ui"
    auth:               str = "JWT httpOnly cookie"
    package_manager:    str = "pip"


@dataclass
class FolderStructure:
    """Describes the target directory layout."""
    root:  str
    dirs:  list[str]
    files: list[str]


@dataclass
class InterfaceSignature:
    """
    One exported symbol from a module — function, class, or constant.

    Examples
    --------
    Function : name="create_todo", params={"title":"str","user_id":"int"}, returns="TodoResponse"
    Class    : name="Todo",        params={},                               returns="SQLAlchemy model"
    Router   : name="router",      params={},                               returns="APIRouter"
    """
    name:    str
    params:  dict[str, str] = field(default_factory=dict)   # param_name → type string
    returns: str            = ""                             # return type string


@dataclass
class ModuleSpec:
    """
    Per-file build contract produced by the Planner.

    Every module the backend or frontend agent will create gets one
    ModuleSpec.  The build order is determined by topological sort of
    the `imports` dependency graph (Phase 2).

    Attributes
    ----------
    path        : Relative file path, e.g. "backend/routers/todos.py"
    description : One-line purpose of this file
    layer       : "backend" | "frontend"
    imports     : Other module paths this file depends on (must be in plan)
    exports     : Symbol names this file makes available to other modules
    interfaces  : Detailed signatures of exported symbols
    dod         : Definition-of-Done checklist items (verified after build)
    """
    path:        str
    description: str
    layer:       str                         # "backend" | "frontend"
    imports:     list[str]  = field(default_factory=list)   # relative paths
    exports:     list[str]  = field(default_factory=list)   # symbol names
    interfaces:  list[InterfaceSignature] = field(default_factory=list)
    dod:         list[str]  = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path":        self.path,
            "description": self.description,
            "layer":       self.layer,
            "imports":     self.imports,
            "exports":     self.exports,
            "interfaces": [
                {"name": i.name, "params": i.params, "returns": i.returns}
                for i in self.interfaces
            ],
            "dod":         self.dod,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModuleSpec":
        interfaces = [
            InterfaceSignature(
                name    = i.get("name", ""),
                params  = i.get("params", {}),
                returns = i.get("returns", ""),
            )
            for i in data.get("interfaces", [])
        ]
        return cls(
            path        = data.get("path", ""),
            description = data.get("description", ""),
            layer       = data.get("layer", "backend"),
            imports     = data.get("imports", []),
            exports     = data.get("exports", []),
            interfaces  = interfaces,
            dod         = data.get("dod", []),
        )


@dataclass
class ProjectPlan:
    """
    Complete plan produced by the Planner Agent.

    All other agents receive this plan as their primary input.

    Attributes
    ----------
    project_name     : Snake-case project identifier.
    description      : One-sentence description of the project.
    tech_stack       : Technology choices.
    folder_structure : Directories and files to create.
    database_models  : ORM models the backend must implement.
    api_endpoints    : REST API contract.
    modules          : Ordered module specs for sequential build (v2).
    extra_notes      : Any additional context for agents.
    """
    project_name:     str
    description:      str
    tech_stack:       TechStack              = field(default_factory=TechStack)
    folder_structure: FolderStructure | None = None
    database_models:  list[DatabaseModel]   = field(default_factory=list)
    api_endpoints:    list[ApiEndpoint]     = field(default_factory=list)
    modules:          list[ModuleSpec]      = field(default_factory=list)
    extra_notes:      str                   = ""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def backend_modules(self) -> list[ModuleSpec]:
        """Return only backend-layer modules."""
        return [m for m in self.modules if m.layer == "backend"]

    def frontend_modules(self) -> list[ModuleSpec]:
        """Return only frontend-layer modules."""
        return [m for m in self.modules if m.layer == "frontend"]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
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
                {"name": m.name, "description": m.description, "fields": m.fields}
                for m in self.database_models
            ],
            "api_endpoints": [
                {
                    "method":        e.method,
                    "path":          e.path,
                    "description":   e.description,
                    "request_body":  e.request_body,
                    "response":      e.response,
                    "auth_required": e.auth_required,
                }
                for e in self.api_endpoints
            ],
            "modules":     [m.to_dict() for m in self.modules],
            "extra_notes": self.extra_notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectPlan":
        ts_data    = data.get("tech_stack", {})
        tech_stack = TechStack(
            backend_framework  = ts_data.get("backend_framework",  "FastAPI"),
            backend_language   = ts_data.get("backend_language",   "Python"),
            database           = ts_data.get("database",           "SQLite"),
            orm                = ts_data.get("orm",                 "SQLAlchemy"),
            frontend_framework = ts_data.get("frontend_framework",
                                             "Next.js 14 + TypeScript + Tailwind CSS + shadcn/ui"),
            auth               = ts_data.get("auth",               "JWT httpOnly cookie"),
        )

        fs_data          = data.get("folder_structure", {})
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

        modules = [
            ModuleSpec.from_dict(m)
            for m in data.get("modules", [])
        ]

        return cls(
            project_name     = data.get("project_name", "project"),
            description      = data.get("description",  ""),
            tech_stack       = tech_stack,
            folder_structure = folder_structure,
            database_models  = database_models,
            api_endpoints    = api_endpoints,
            modules          = modules,
            extra_notes      = data.get("extra_notes", ""),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ProjectPlan":
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()
        return cls.from_dict(json.loads(clean))