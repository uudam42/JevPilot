"""Observations: the structured result of executing a capability."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from jevpilot.core._base import FrozenModel, new_id, utcnow
from jevpilot.core.artifact import Artifact
from jevpilot.core.candidate import Candidate
from jevpilot.core.provenance import Provenance
from jevpilot.core.uncertainty import Uncertainty


class ErrorInfo(FrozenModel):
    """Structured description of a failed execution."""

    type: str
    message: str
    retryable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class StateEffects(FrozenModel):
    """Declarative state changes a capability proposes.

    The generic :class:`~jevpilot.orchestration.state_manager.StateManager`
    applies these; domain :class:`~jevpilot.interfaces.reducer.StateReducer`
    implementations can apply anything more specific.
    """

    context_updates: dict[str, Any] = Field(default_factory=dict)
    candidates: tuple[Candidate, ...] = ()
    uncertainty_updates: dict[str, Uncertainty] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.context_updates or self.candidates or self.uncertainty_updates)


class Observation(FrozenModel):
    """O_t — what happened when capability A_t was executed."""

    id: str = Field(default_factory=lambda: new_id("obs"))
    workflow_id: str
    step: int
    capability_id: str | None
    decision_id: str | None = None
    success: bool
    result: Any = None
    error: ErrorInfo | None = None
    uncertainty: Uncertainty | None = None
    artifacts: tuple[Artifact, ...] = ()
    effects: StateEffects = Field(default_factory=StateEffects)
    provenance: Provenance
    execution_time: float = Field(default=0.0, ge=0.0)  # seconds
    timestamp: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def confidence(self) -> float | None:
        return self.uncertainty.confidence if self.uncertainty else None
