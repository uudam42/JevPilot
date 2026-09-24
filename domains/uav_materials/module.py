"""The UAV materials domain module: capabilities, reducers, evaluator, state type."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from domains.uav_materials.capabilities import all_capabilities
from domains.uav_materials.decision import ExistingMaterialAcceptancePolicy
from domains.uav_materials.interpretation import RequirementInterpreter
from domains.uav_materials.inverse_design import DesignSearchConfig
from domains.uav_materials.profile import TargetMaterialProfile
from domains.uav_materials.schema import MaterialCandidate
from domains.uav_materials.state import UAVMaterialsState
from domains.uav_materials.workflow import (
    ProgressReducer,
    evaluate_workflow,
    workflow_capabilities,
)
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

REQUEST_GOAL = (
    "Answer the user's UAV material request: interpret it into engineering requirements, "
    "search the existing materials, recommend one if it meets the acceptance policy, "
    "otherwise design a composite candidate within the validated design space, and write "
    "an evidence-aware engineering report."
)


class MaterialsReducer(StateReducer):
    """Moves successful set-up capability outputs into the typed domain fields."""

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
    """Set-up workflows: success once the target profile exists (and candidates, if asked).

    Request-driven workflows: success once the final report is written.
    """

    evaluator_id = "uav_materials.setup"

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        if not isinstance(state, UAVMaterialsState):
            return EvaluationResult(evaluator_id=self.evaluator_id, rationale="not this domain")
        if state.request_text is not None:
            return evaluate_workflow(state, self.evaluator_id)
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
    version = "0.2.0"
    description = (
        "UAV material selection and composite design: requirement interpretation, "
        "evidence-aware search of real materials, a configurable decision policy, bounded "
        "inverse design with NASA-sourced micromechanics and lamination theory, and reports."
    )

    def __init__(
        self,
        *,
        interpreter: RequirementInterpreter | None = None,
        policy: ExistingMaterialAcceptancePolicy | None = None,
        design_config: DesignSearchConfig | None = None,
    ) -> None:
        self.interpreter = interpreter
        self.policy = policy
        self.design_config = design_config

    def capabilities(self) -> Sequence[Capability]:
        return [
            *all_capabilities(),
            *workflow_capabilities(self.interpreter, self.policy, self.design_config),
        ]

    def evaluators(self) -> Sequence[Evaluator]:
        return [SetupEvaluator()]

    def reducers(self) -> Sequence[StateReducer]:
        return [MaterialsReducer(), ProgressReducer()]

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

    def new_request_workflow(
        self, request_text: str, *, run_info: dict[str, Any] | None = None
    ) -> UAVMaterialsState:
        """A request-driven workflow: natural-language request → engineering report."""
        goal = Goal(
            description=REQUEST_GOAL,
            parameters={"request": request_text},
            success_criteria={"report_written": True},
        )
        state = self.create_state(goal, metadata={"run": dict(run_info or {})})
        assert isinstance(state, UAVMaterialsState)
        return state.evolve(request_text=request_text)
