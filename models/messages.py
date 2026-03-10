"""
models/messages.py
------------------
Data structures for inter-agent communication.

Every message passed between agents uses AgentMessage so the
Orchestrator can track status, log history, and route results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"


class TaskType(str, Enum):
    PLAN     = "plan"       # Planner → ProjectPlan
    BACKEND  = "backend"    # Backend → code files
    FRONTEND = "frontend"   # Frontend → code files
    QA       = "qa"         # QA → test report
    FIX      = "fix"        # Backend/Frontend → fix after QA failure


@dataclass
class AgentMessage:
    """
    Envelope passed between agents.

    Attributes
    ----------
    sender      : Who sent this message.
    receiver    : Who should process it.
    task_type   : What kind of work is requested.
    payload     : Task-specific input data (dict).
    status      : Current lifecycle status.
    result      : Output produced by the receiving agent (filled on completion).
    error       : Error detail if status == FAILED.
    created_at  : ISO timestamp when the message was created.
    updated_at  : ISO timestamp of last status change.
    attempt     : How many times this task has been attempted (for retry tracking).
    """
    sender:     str
    receiver:   str
    task_type:  TaskType
    payload:    dict[str, Any]
    status:     TaskStatus          = TaskStatus.PENDING
    result:     str | None          = None
    error:      str | None          = None
    created_at: str                 = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str                 = field(default_factory=lambda: datetime.now().isoformat())
    attempt:    int                 = 0

    def mark_running(self) -> None:
        self.status     = TaskStatus.RUNNING
        self.updated_at = datetime.now().isoformat()
        self.attempt   += 1

    def mark_done(self, result: str) -> None:
        self.status     = TaskStatus.DONE
        self.result     = result
        self.updated_at = datetime.now().isoformat()

    def mark_failed(self, error: str) -> None:
        self.status     = TaskStatus.FAILED
        self.error      = error
        self.updated_at = datetime.now().isoformat()

    def summary(self) -> str:
        icon = {"pending": "⏳", "running": "🔄", "done": "✅",
                "failed": "❌"}.get(self.status, "•")
        return (
            f"{icon} [{self.task_type.value.upper()}] "
            f"{self.sender} → {self.receiver} | "
            f"status={self.status} attempt={self.attempt}"
        )