# =============================================================================
# models/__init__.py
# -----------------------------------------------------------------------------
# Re-exports all data-structure classes used across the entire framework.
#
# Two source files:
#   models/messages.py      — inter-agent communication envelope
#   models/project_plan.py  — structured project blueprint (plan output)
#
# Import anywhere with:
#   from models import AgentMessage, ProjectPlan, ApiEndpoint, ...
# =============================================================================

# ── Inter-agent communication ─────────────────────────────────────────────────
from models.messages import (
    AgentMessage,   # envelope passed between agents (sender, receiver, payload, status)
    TaskStatus,     # enum: pending | running | done | failed
    TaskType,       # enum: plan | backend | frontend | qa | fix
)

# ── Project blueprint ─────────────────────────────────────────────────────────
from models.project_plan import (
    ProjectPlan,      # top-level plan produced by PlannerAgent
    ApiEndpoint,      # describes one REST endpoint (method, path, auth, schema)
    DatabaseModel,    # describes one ORM model (name, fields)
    TechStack,        # technology choices (framework, db, orm, auth, ...)
    FolderStructure,  # directory + file list the project must have
)

__all__ = [
    # messages.py
    "AgentMessage", "TaskStatus", "TaskType",
    # project_plan.py
    "ProjectPlan", "ApiEndpoint",
    "DatabaseModel", "TechStack", "FolderStructure",
]