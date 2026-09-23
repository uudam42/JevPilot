"""Generic evaluator combinators (no domain semantics)."""

from __future__ import annotations

from collections.abc import Sequence

from jevpilot.core import ControlAction, EvaluationResult, WorkflowState
from jevpilot.interfaces.evaluator import Evaluator

# Higher = more severe; the most severe recommendation wins.
_SEVERITY = {
    ControlAction.CONTINUE: 0,
    ControlAction.RETRY: 1,
    ControlAction.REPLAN: 2,
    ControlAction.TERMINATE_SUCCESS: 3,
    ControlAction.HUMAN_INTERVENTION: 4,
    ControlAction.TERMINATE_FAILURE: 5,
}


class NullEvaluator(Evaluator):
    """Always recommends CONTINUE; termination then comes from the policy's budgets."""

    evaluator_id = "null_evaluator"

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        return EvaluationResult(
            evaluator_id=self.evaluator_id,
            step=state.step,
            rationale="no evaluator configured",
        )


class CompositeEvaluator(Evaluator):
    """Combines several evaluators.

    Success requires *all* members to recommend success; otherwise the most
    severe non-success recommendation wins. Progress is the minimum reported.
    """

    evaluator_id = "composite_evaluator"

    def __init__(self, evaluators: Sequence[Evaluator]) -> None:
        if not evaluators:
            raise ValueError("CompositeEvaluator needs at least one evaluator")
        self._evaluators = tuple(evaluators)

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        results = [e.evaluate(state) for e in self._evaluators]
        recs = [r.recommendation for r in results]
        if all(r is ControlAction.TERMINATE_SUCCESS for r in recs):
            rec = ControlAction.TERMINATE_SUCCESS
        else:
            rec = max(
                (r for r in recs if r is not ControlAction.TERMINATE_SUCCESS),
                key=_SEVERITY.__getitem__,
            )
        progress = [r.goal_progress for r in results if r.goal_progress is not None]
        confidence = [r.confidence for r in results if r.confidence is not None]
        return EvaluationResult(
            evaluator_id=self.evaluator_id,
            step=state.step,
            goal_progress=min(progress) if progress else None,
            confidence=min(confidence) if confidence else None,
            constraint_status=tuple(c for r in results for c in r.constraint_status),
            remaining_gaps=tuple(g for r in results for g in r.remaining_gaps),
            recommendation=rec,
            rationale="; ".join(f"{r.evaluator_id}: {r.rationale}" for r in results),
            metadata={"members": [r.id for r in results]},
        )
