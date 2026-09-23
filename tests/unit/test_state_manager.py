import pytest

from jevpilot import (
    Artifact,
    ControlAction,
    ControlDecision,
    Observation,
    Plan,
    Provenance,
    RoutingDecision,
    StateEffects,
    StateManager,
    StateReducer,
    Uncertainty,
    WorkflowState,
)
from jevpilot.exceptions import ProvenanceError


def _obs(state: WorkflowState, **kw: object) -> Observation:
    return Observation(
        workflow_id=state.workflow_id,
        step=state.step,
        capability_id="c",
        success=True,
        provenance=Provenance(capability_id="c"),
        **kw,  # type: ignore[arg-type]
    )


DEC = RoutingDecision(capability_id="c")


def test_update_is_pure_and_complete(state: WorkflowState) -> None:
    prov = Provenance(capability_id="c")
    obs = _obs(
        state,
        artifacts=(Artifact(name="a", provenance=prov),),
        effects=StateEffects(
            context_updates={"k": 1}, uncertainty_updates={"x": Uncertainty(confidence=0.5)}
        ),
    )
    new = StateManager().update(state, DEC, obs)
    assert state.step == 0 and state.observations == ()  # input untouched
    assert new.step == 1
    assert new.observations == (obs,)
    assert new.artifacts == obs.artifacts
    assert new.context == {"k": 1}
    assert new.uncertainty["x"].confidence == 0.5
    assert new.history[-1].observation_id == obs.id and new.history[-1].decision == DEC
    assert {p.id for p in new.provenance} == {obs.provenance.id, prov.id}


def test_artifact_without_provenance_rejected(state: WorkflowState) -> None:
    obs = _obs(state, artifacts=(Artifact(name="orphan"),))
    with pytest.raises(ProvenanceError):
        StateManager().update(state, DEC, obs)


def test_foreign_observation_rejected(state: WorkflowState) -> None:
    other = state.evolve(workflow_id="wf_other")
    with pytest.raises(ValueError):
        StateManager().update(state, DEC, _obs(other))


def test_reducers_applied_in_order(state: WorkflowState) -> None:
    class Tag(StateReducer):
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def reduce(self, s: WorkflowState, o: Observation) -> WorkflowState:
            return s.evolve(context={**s.context, "tags": [*s.context.get("tags", []), self.tag]})

    new = StateManager([Tag("a"), Tag("b")]).update(state, DEC, _obs(state))
    assert new.context["tags"] == ["a", "b"]


def test_control_plan_and_status(state: WorkflowState) -> None:
    sm = StateManager()
    s = sm.update(state, DEC, _obs(state))
    s = sm.record_control(s, ControlDecision(action=ControlAction.RETRY))
    assert s.history[-1].control and s.history[-1].control.action is ControlAction.RETRY
    s = sm.set_plan(s, Plan())
    s = sm.set_plan(s, Plan())
    assert s.plan and s.plan.revision == 1
