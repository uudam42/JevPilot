"""The canonical, serialisable routing problem: ``(S_t, C_t)`` as plain data.

:class:`RoutingRequest` is what model-backed routers send to their adapters,
what benchmark fixtures and offline replays store, and what future learned
routers will train on. It is built from a :class:`WorkflowState` and the
available :class:`CapabilitySpec` descriptors, and contains JSON data only:
no executable objects, no Python classes, no provider-specific formatting.

It deliberately does not serialise the whole state object graph. Provenance
records, timestamps and full artifact contents are left out. Long values are
truncated, so request size stays bounded as a workflow grows.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from jevpilot.core import (
    EvaluationResult,
    FrozenModel,
    HistoryEntry,
    Observation,
    WorkflowState,
    stable_digest,
    to_jsonable,
)
from jevpilot.interfaces.capability import CapabilitySpec

REQUEST_SCHEMA_VERSION = "jevpilot.routing_request/1"


class CapabilityDescription(FrozenModel):
    """Router-visible description of one capability (JSON Schemas, no classes).

    Mirrors :meth:`CapabilitySpec.describe`. ``cost_estimate`` and
    ``latency_estimate`` are whatever the capability declared, or ``None``;
    they are never invented.
    """

    id: str
    name: str
    description: str
    version: str
    domain: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    tags: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    side_effects: bool = False
    cost_estimate: float | None = None
    latency_estimate: float | None = None
    requirements: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: CapabilitySpec) -> CapabilityDescription:
        return cls.model_validate(spec.describe())


class ActionRecord(FrozenModel):
    """One previous loop iteration, as a router needs to see it."""

    step: int
    capability_id: str | None
    intent: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    success: bool | None = None  # None: nothing was executed
    result: Any = None
    error: dict[str, Any] | None = None
    control: str | None = None
    retry: bool = False


class EvaluationSummary(FrozenModel):
    evaluator_id: str
    step: int | None = None
    recommendation: str
    goal_progress: float | None = None
    remaining_gaps: tuple[str, ...] = ()
    unsatisfied_constraints: tuple[str, ...] = ()
    rationale: str = ""


class StateSummary(FrozenModel):
    """The parts of ``S_t`` a router reasons about."""

    status: str
    context: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)  # fields added by a state subclass
    artifacts: tuple[dict[str, Any], ...] = ()  # id, name, kind, media_type (no content)
    candidates: tuple[dict[str, Any], ...] = ()
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    plan: tuple[dict[str, Any], ...] = ()


class RoutingRequest(FrozenModel):
    """``(S_t, C_t)`` as a stable, JSON-serialisable routing problem."""

    schema_version: str = REQUEST_SCHEMA_VERSION
    goal: dict[str, Any]
    state_summary: StateSummary
    requirements: tuple[dict[str, Any], ...] = ()
    constraints: tuple[dict[str, Any], ...] = ()
    available_capabilities: tuple[CapabilityDescription, ...]
    previous_actions: tuple[ActionRecord, ...] = ()
    previous_evaluations: tuple[EvaluationSummary, ...] = ()
    step: int = 0
    retry_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_state(
        cls,
        state: WorkflowState,
        capabilities: Sequence[CapabilitySpec],
        *,
        max_actions: int = 20,
        max_evaluations: int = 3,
        max_value_chars: int = 2000,
        metadata: dict[str, Any] | None = None,
    ) -> RoutingRequest:
        clip = _Clipper(max_value_chars)
        observations = {o.id: o for o in state.observations}
        actions = [_action(h, observations, clip) for h in state.history]
        base_fields = set(WorkflowState.model_fields)
        extensions = {
            name: clip(getattr(state, name))
            for name in type(state).model_fields
            if name not in base_fields
        }
        summary = StateSummary(
            status=str(state.status),
            context={k: clip(v) for k, v in state.context.items()},
            extensions=extensions,
            artifacts=tuple(
                {"id": a.id, "name": a.name, "kind": a.kind, "media_type": a.media_type}
                for a in state.artifacts
            ),
            candidates=tuple(
                {"content": clip(c.content), "score": c.score} for c in state.candidate_solutions
            ),
            uncertainty={
                subject: u.model_dump(mode="json", exclude_defaults=True)
                for subject, u in state.uncertainty.items()
            },
            plan=tuple(
                {
                    "description": s.description,
                    "capability_hint": s.capability_hint,
                    "inputs_hint": clip(s.inputs_hint),
                }
                for s in (state.plan.steps if state.plan else ())
            ),
        )
        return cls(
            goal={
                "description": state.goal.description,
                "parameters": clip(state.goal.parameters),
                "success_criteria": clip(state.goal.success_criteria),
            },
            state_summary=summary,
            requirements=tuple(
                r.model_dump(mode="json", exclude={"metadata"}) for r in state.requirements
            ),
            constraints=tuple(
                c.model_dump(mode="json", exclude={"metadata"}) for c in state.constraints
            ),
            available_capabilities=tuple(CapabilityDescription.from_spec(c) for c in capabilities),
            previous_actions=tuple(actions[-max_actions:]) if max_actions else (),
            previous_evaluations=tuple(_evaluation(e) for e in state.evaluations[-max_evaluations:])
            if max_evaluations
            else (),
            step=state.step,
            retry_count=_trailing_retries(state),
            metadata=dict(metadata or {}),
        )

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = None) -> str:
        """Canonical JSON (sorted keys), suitable for prompts, fixtures and hashing."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent, ensure_ascii=False)

    def fingerprint(self) -> str:
        """Deterministic digest of the request, e.g. for replay caches."""
        return stable_digest(self.to_dict())

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.available_capabilities)


class _Clipper:
    """Converts values to JSON data and truncates oversized ones deterministically."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    def __call__(self, value: Any) -> Any:
        data = to_jsonable(value)
        if self.max_chars <= 0:
            return data
        text = json.dumps(data, sort_keys=True, ensure_ascii=False)
        if len(text) <= self.max_chars:
            return data
        return {"_truncated": True, "_chars": len(text), "preview": text[: self.max_chars]}


def _action(
    entry: HistoryEntry, observations: dict[str, Observation], clip: _Clipper
) -> ActionRecord:
    obs = observations.get(entry.observation_id) if entry.observation_id else None
    error: dict[str, Any] | None = None
    if obs is not None and obs.error is not None:
        error = {
            "type": obs.error.type,
            "message": obs.error.message,
            "retryable": obs.error.retryable,
        }
    elif entry.routing_error is not None:
        error = {
            "type": entry.routing_error.type,
            "message": entry.routing_error.message,
            "routing": True,
        }
    return ActionRecord(
        step=entry.step,
        capability_id=entry.decision.capability_id,
        intent=str(entry.decision.intent),
        inputs=clip(entry.decision.inputs),
        success=obs.success if obs is not None else None,
        result=clip(obs.result) if obs is not None and obs.success else None,
        error=error,
        control=str(entry.control.action) if entry.control else None,
        retry="retry_of" in entry.decision.metadata,
    )


def _evaluation(e: EvaluationResult) -> EvaluationSummary:
    return EvaluationSummary(
        evaluator_id=e.evaluator_id,
        step=e.step,
        recommendation=str(e.recommendation),
        goal_progress=e.goal_progress,
        remaining_gaps=e.remaining_gaps,
        unsatisfied_constraints=tuple(
            c.constraint_id for c in e.constraint_status if c.satisfied is False
        ),
        rationale=e.rationale,
    )


def _trailing_retries(state: WorkflowState) -> int:
    n = 0
    for entry in reversed(state.history):
        if "retry_of" not in entry.decision.metadata:
            break
        n += 1
    return n
