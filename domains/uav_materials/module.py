"""The UAV materials domain module: capabilities, reducer, evaluator, state type."""

from __future__ import annotations

from collections.abc import Sequence

from domains.uav_materials.capabilities import all_capabilities
from domains.uav_materials.profile import TargetMaterialProfile
from domains.uav_materials.schema import MaterialCandidate
from domains.uav_materials.state import UAVMaterialsState
from jevpilot import (
    Capability,
    ControlAction,
    DomainModule,
    EvaluationResult,
    Evaluator,
    Goal,
    Observation,
    StateReducer,
    WorkflowState,
)


class MaterialsReducer(StateReducer):
    """Moves successful capability outputs into the typed domain fields."""

    def reduce(self, state: WorkflowState, observation: Observation) -> WorkflowState:
        if not isinstance(state, UAVMaterialsState) or not observation.success:
            return state
        result = observation.result
        if isinstance(result, TargetMaterialProfile):
            return state.evolve(target_profile=result)
        if isinstance(result, tuple) and all(isinstance(c, MaterialCandidate) for c in result):
            known = {c.candidate_id for c in state.candidate_materials}
            new = tuple(c for c in result if c.candidate_id not in known)
            return state.evolve(candidate_materials=(*state.candidate_materials, *new))
        return state


class SetupEvaluator(Evaluator):
    """Success once the target profile exists (and candidates, if the goal asks for them)."""

    evaluator_id = "uav_materials.setup"

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        if not isinstance(state, UAVMaterialsState):
            return EvaluationResult(evaluator_id=self.evaluator_id, rationale="not this domain")
        need_candidates = bool(state.goal.success_criteria.get("candidates_registered"))
        gaps = []
        if state.target_profile is None:
            gaps.append("no target profile")
        if need_candidates and not state.candidate_materials:
            gaps.append("no candidate materials")
        done = not gaps
        return EvaluationResult(
            evaluator_id=self.evaluator_id,
            step=state.step,
            goal_progress=1.0 if done else 0.5 if len(gaps) == 1 and need_candidates else 0.0,
            remaining_gaps=tuple(gaps),
            recommendation=ControlAction.TERMINATE_SUCCESS if done else ControlAction.CONTINUE,
            rationale="setup complete" if done else "; ".join(gaps),
        )


class UAVMaterialsDomain(DomainModule):
    name = "uav_materials"
    version = "0.1.0"
    description = (
        "UAV material selection: target property profiles and material records. "
        "Schema iteration: no retrieval, ranking, design or simulation yet."
    )

    def capabilities(self) -> Sequence[Capability]:
        return all_capabilities()

    def evaluators(self) -> Sequence[Evaluator]:
        return [SetupEvaluator()]

    def reducers(self) -> Sequence[StateReducer]:
        return [MaterialsReducer()]

    def state_type(self) -> type[WorkflowState]:
        return UAVMaterialsState

    def new_workflow(
        self, description: str, *, candidates_registered: bool = False
    ) -> UAVMaterialsState:
        goal = Goal(
            description=description,
            success_criteria={"candidates_registered": candidates_registered},
        )
        state = self.create_state(goal)
        assert isinstance(state, UAVMaterialsState)
        return state
