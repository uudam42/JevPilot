"""Arithmetic demo domain: transform a number into a target number."""

from __future__ import annotations

from collections.abc import Sequence

from domains.demo.capabilities import CompareOutput, ValueOutput, all_capabilities
from domains.demo.state import NumberState
from jevpilot import (
    Capability,
    ControlAction,
    DomainModule,
    EvaluationResult,
    Evaluator,
    Goal,
    Observation,
    Rule,
    StateReducer,
    WorkflowState,
)


class ValueReducer(StateReducer):
    """Moves a successful transform's output into ``NumberState.value``."""

    def reduce(self, state: WorkflowState, observation: Observation) -> WorkflowState:
        if (
            isinstance(state, NumberState)
            and observation.success
            and isinstance(observation.result, ValueOutput)
        ):
            return state.evolve(value=observation.result.value)
        return state


class TargetReachedEvaluator(Evaluator):
    """Success = value equals target *and* a comparison has confirmed it."""

    evaluator_id = "arithmetic.target_reached"

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        if not isinstance(state, NumberState):
            return EvaluationResult(
                evaluator_id=self.evaluator_id, rationale="not an arithmetic workflow"
            )
        start = float(state.goal.parameters.get("start", 0.0))
        span = abs(state.target - start) or 1.0
        progress = max(0.0, 1.0 - abs(state.target - state.value) / span)
        confirmed = any(
            o.success
            and isinstance(o.result, CompareOutput)
            and o.result.equal
            and o.result.value == state.value
            for o in state.observations
        )
        if state.value == state.target and confirmed:
            return EvaluationResult(
                evaluator_id=self.evaluator_id,
                step=state.step,
                goal_progress=1.0,
                confidence=1.0,
                recommendation=ControlAction.TERMINATE_SUCCESS,
                rationale=f"value {state.value} equals target (confirmed)",
            )
        gaps = (
            ("comparison not yet run",)
            if state.value == state.target
            else (f"difference {state.target - state.value}",)
        )
        return EvaluationResult(
            evaluator_id=self.evaluator_id,
            step=state.step,
            goal_progress=min(progress, 0.99),
            remaining_gaps=gaps,
            rationale="target not confirmed",
        )


class ArithmeticDomain(DomainModule):
    name = "arithmetic"
    version = "0.1.0"
    description = "Toy domain: reach a target number with add/multiply/compare."

    def capabilities(self) -> Sequence[Capability]:
        return all_capabilities()

    def evaluators(self) -> Sequence[Evaluator]:
        return [TargetReachedEvaluator()]

    def reducers(self) -> Sequence[StateReducer]:
        return [ValueReducer()]

    def state_type(self) -> type[WorkflowState]:
        return NumberState

    def new_workflow(self, start: float, target: float) -> NumberState:
        goal = Goal(
            description=f"transform {start} into {target}",
            parameters={"start": start, "target": target},
            success_criteria={"equals": target},
        )
        state = self.create_state(goal, value=start, target=target)
        assert isinstance(state, NumberState)
        return state


def routing_rules() -> list[Rule]:
    """Domain-supplied heuristics for the generic RuleRouter."""

    def num(state: WorkflowState) -> NumberState:
        assert isinstance(state, NumberState)
        return state

    def confirmed(state: WorkflowState) -> bool:
        last = state.last_observation
        return bool(last and isinstance(last.result, CompareOutput) and last.result.equal)

    return [
        Rule(
            "arith.compare",
            when=lambda s: num(s).value == num(s).target and not confirmed(s),
            reason="value matches target; confirm",
        ),
        Rule(
            "arith.multiply",
            when=lambda s: 0 < num(s).value and num(s).value * 2 <= num(s).target,
            inputs={"factor": 2},
            reason="doubling does not overshoot",
        ),
        Rule(
            "arith.add",
            when=lambda s: num(s).value != num(s).target,
            inputs=lambda s: {"amount": num(s).target - num(s).value},
            reason="close the remaining gap",
        ),
    ]
