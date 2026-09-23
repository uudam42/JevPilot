"""Plans: a planner's decomposition of a goal into (optional) steps."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from jevpilot.core._base import FrozenModel, new_id, utcnow


class PlanStep(FrozenModel):
    id: str = Field(default_factory=lambda: new_id("pstep"))
    description: str
    capability_hint: str | None = None
    inputs_hint: dict[str, Any] = Field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class Plan(FrozenModel):
    """Advisory structure a router may consult. Routers are free to ignore it."""

    id: str = Field(default_factory=lambda: new_id("plan"))
    steps: tuple[PlanStep, ...] = ()
    rationale: str = ""
    planner_id: str | None = None
    revision: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
