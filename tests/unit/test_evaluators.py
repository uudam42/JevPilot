import pytest

from jevpilot import (
    CompositeEvaluator,
    ControlAction,
    EvaluationResult,
    Evaluator,
    NullEvaluator,
    WorkflowState,
)


class Fixed(Evaluator):
    def __init__(self, rec: ControlAction, progress: float | None = None) -> None:
        self.rec, self.progress = rec, progress
        self.evaluator_id = f"fixed_{rec}"

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        return EvaluationResult(
            evaluator_id=self.evaluator_id, recommendation=self.rec, goal_progress=self.progress
        )


A = ControlAction


def test_null_evaluator_continues(state: WorkflowState) -> None:
    assert NullEvaluator().evaluate(state).recommendation is A.CONTINUE


def test_composite_success_requires_all(state: WorkflowState) -> None:
    both = CompositeEvaluator([Fixed(A.TERMINATE_SUCCESS), Fixed(A.TERMINATE_SUCCESS)])
    assert both.evaluate(state).recommendation is A.TERMINATE_SUCCESS
    mixed = CompositeEvaluator([Fixed(A.TERMINATE_SUCCESS, 1.0), Fixed(A.CONTINUE, 0.4)])
    r = mixed.evaluate(state)
    assert r.recommendation is A.CONTINUE and r.goal_progress == 0.4


def test_composite_most_severe_wins(state: WorkflowState) -> None:
    ev = CompositeEvaluator([Fixed(A.REPLAN), Fixed(A.TERMINATE_FAILURE), Fixed(A.CONTINUE)])
    assert ev.evaluate(state).recommendation is A.TERMINATE_FAILURE


def test_composite_requires_members() -> None:
    with pytest.raises(ValueError):
        CompositeEvaluator([])
