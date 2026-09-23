"""Explicit, pure state transitions: S_{t+1} = Update(S_t, O_t)."""

from __future__ import annotations

from collections.abc import Sequence

from jevpilot.core import (
    ControlDecision,
    ErrorInfo,
    EvaluationResult,
    HistoryEntry,
    Observation,
    Plan,
    Provenance,
    RoutingDecision,
    WorkflowState,
    WorkflowStatus,
)
from jevpilot.exceptions import ProvenanceError
from jevpilot.interfaces.reducer import StateReducer


class StateManager:
    """All state transitions go through here. Every method is pure."""

    def __init__(self, reducers: Sequence[StateReducer] = ()) -> None:
        self._reducers = tuple(reducers)

    def start(self, state: WorkflowState) -> WorkflowState:
        return state.evolve(status=WorkflowStatus.RUNNING)

    def update(
        self, state: WorkflowState, decision: RoutingDecision, observation: Observation
    ) -> WorkflowState:
        """Incorporate an observation produced for ``decision``."""
        if observation.workflow_id != state.workflow_id:
            raise ValueError("observation belongs to a different workflow")
        _require_provenance(observation)

        effects = observation.effects
        known = {p.id for p in state.provenance}
        new_prov: list[Provenance] = []
        for p in (
            observation.provenance,
            *(a.provenance for a in observation.artifacts),
            *(c.provenance for c in effects.candidates),
        ):
            if p is not None and p.id not in known:
                known.add(p.id)
                new_prov.append(p)

        new = state.evolve(
            observations=(*state.observations, observation),
            artifacts=(*state.artifacts, *observation.artifacts),
            candidate_solutions=(*state.candidate_solutions, *effects.candidates),
            context={**state.context, **effects.context_updates},
            uncertainty={**state.uncertainty, **effects.uncertainty_updates},
            provenance=(*state.provenance, *new_prov),
            history=(
                *state.history,
                HistoryEntry(step=state.step, decision=decision, observation_id=observation.id),
            ),
            step=state.step + 1,
        )
        for reducer in self._reducers:
            new = reducer.reduce(new, observation)
        return new

    def record_no_action(self, state: WorkflowState, decision: RoutingDecision) -> WorkflowState:
        entry = HistoryEntry(step=state.step, decision=decision)
        return state.evolve(history=(*state.history, entry))

    def record_routing_failure(
        self, state: WorkflowState, decision: RoutingDecision, error: ErrorInfo
    ) -> WorkflowState:
        """Record an iteration in which the router failed to produce a valid decision."""
        entry = HistoryEntry(step=state.step, decision=decision, routing_error=error)
        return state.evolve(history=(*state.history, entry))

    def apply_evaluation(self, state: WorkflowState, evaluation: EvaluationResult) -> WorkflowState:
        return state.evolve(evaluations=(*state.evaluations, evaluation))

    def record_control(self, state: WorkflowState, control: ControlDecision) -> WorkflowState:
        """Attach the control decision to the most recent history entry."""
        if not state.history:
            return state
        last = state.history[-1].model_copy(update={"control": control})
        return state.evolve(history=(*state.history[:-1], last))

    def set_plan(self, state: WorkflowState, plan: Plan) -> WorkflowState:
        revision = state.plan.revision + 1 if state.plan else 0
        return state.evolve(plan=plan.model_copy(update={"revision": revision}))

    def set_status(self, state: WorkflowState, status: WorkflowStatus) -> WorkflowState:
        return state.evolve(status=status)


def _require_provenance(observation: Observation) -> None:
    missing = [a.id for a in observation.artifacts if a.provenance is None]
    missing += [c.id for c in observation.effects.candidates if c.provenance is None]
    if missing:
        raise ProvenanceError(f"items without provenance cannot enter state: {missing}")
