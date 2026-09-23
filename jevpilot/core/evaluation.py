"""Evaluation results. The core provides the shape; domains define "good"."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from jevpilot.core._base import FrozenModel, new_id, utcnow
from jevpilot.core.control import ControlAction


class ConstraintStatus(FrozenModel):
    """Whether a named constraint/requirement is satisfied (``None`` = unknown)."""

    constraint_id: str
    satisfied: bool | None = None
    detail: str | None = None


class EvaluationResult(FrozenModel):
    """Assessment of a workflow state produced by an evaluator."""

    id: str = Field(default_factory=lambda: new_id("eval"))
    evaluator_id: str
    step: int | None = None
    goal_progress: float | None = Field(default=None, ge=0.0, le=1.0)
    constraint_status: tuple[ConstraintStatus, ...] = ()
    quality: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    remaining_gaps: tuple[str, ...] = ()
    recommendation: ControlAction = ControlAction.CONTINUE
    rationale: str = ""
    timestamp: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
