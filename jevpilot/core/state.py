"""WorkflowState — S_t, the single structured source of truth for a workflow.

The state is immutable. Transitions ``S_{t+1} = Update(S_t, O_t)`` are
performed by :class:`~jevpilot.orchestration.state_manager.StateManager`,
which always returns a new instance.

Domains extend the state by subclassing::

    class MyDomainState(WorkflowState):
        my_field: int = 0

Subclass instances flow through the core unchanged (``model_copy`` preserves
the concrete type).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import Field

from jevpilot.core._base import FrozenModel, new_id, utcnow
from jevpilot.core.artifact import Artifact
from jevpilot.core.candidate import Candidate
from jevpilot.core.control import ControlDecision
from jevpilot.core.decision import RoutingDecision
from jevpilot.core.evaluation import EvaluationResult
from jevpilot.core.observation import ErrorInfo, Observation
from jevpilot.core.plan import Plan
from jevpilot.core.provenance import Provenance
from jevpilot.core.uncertainty import Uncertainty


class WorkflowStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    AWAITING_HUMAN = "awaiting_human"
    CANCELLED = "cancelled"

    @property
    def is_final(self) -> bool:
        """True when the control loop must stop (``AWAITING_HUMAN`` included)."""
        return self is not WorkflowStatus.PENDING and self is not WorkflowStatus.RUNNING


class Goal(FrozenModel):
    """What the workflow is trying to achieve.

    ``parameters`` and ``success_criteria`` are opaque to the core; only the
    domain (its evaluators, capabilities and routing rules) interprets them.
    """

    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Requirement(FrozenModel):
    id: str = Field(default_factory=lambda: new_id("req"))
    description: str
    spec: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Constraint(FrozenModel):
    id: str = Field(default_factory=lambda: new_id("con"))
    description: str
    spec: dict[str, Any] = Field(default_factory=dict)
    hard: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoryEntry(FrozenModel):
    """One iteration of the loop: decision → observation → control decision.

    When routing failed, ``decision`` is a no-action placeholder and
    ``routing_error`` describes the failure.
    """

    step: int
    decision: RoutingDecision
    observation_id: str | None = None
    control: ControlDecision | None = None
    routing_error: ErrorInfo | None = None
    timestamp: datetime = Field(default_factory=utcnow)


class WorkflowState(FrozenModel):
    workflow_id: str = Field(default_factory=lambda: new_id("wf"))
    goal: Goal
    context: dict[str, Any] = Field(default_factory=dict)

    requirements: tuple[Requirement, ...] = ()
    constraints: tuple[Constraint, ...] = ()

    observations: tuple[Observation, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    candidate_solutions: tuple[Candidate, ...] = ()

    evaluations: tuple[EvaluationResult, ...] = ()
    uncertainty: dict[str, Uncertainty] = Field(default_factory=dict)
    provenance: tuple[Provenance, ...] = ()

    plan: Plan | None = None
    history: tuple[HistoryEntry, ...] = ()
    step: int = 0
    status: WorkflowStatus = WorkflowStatus.PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- controlled updates -------------------------------------------------

    def evolve(self, **changes: Any) -> Self:
        """Return a copy with ``changes`` applied, preserving the concrete subclass."""
        return self.model_copy(update=changes)

    # -- read helpers -------------------------------------------------------

    @property
    def last_observation(self) -> Observation | None:
        return self.observations[-1] if self.observations else None

    @property
    def last_evaluation(self) -> EvaluationResult | None:
        return self.evaluations[-1] if self.evaluations else None

    def artifacts_of_kind(self, kind: str) -> tuple[Artifact, ...]:
        return tuple(a for a in self.artifacts if a.kind == kind)

    def latest_artifact(self, kind: str) -> Artifact | None:
        matches = self.artifacts_of_kind(kind)
        return matches[-1] if matches else None
