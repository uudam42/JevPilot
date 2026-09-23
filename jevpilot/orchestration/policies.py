"""Default, domain-neutral control policy."""

from __future__ import annotations

from dataclasses import dataclass

from jevpilot.core import (
    ControlAction,
    ControlDecision,
    ErrorInfo,
    EvaluationResult,
    Observation,
    WorkflowState,
)
from jevpilot.interfaces.policy import ControlPolicy


@dataclass
class DefaultControlPolicy(ControlPolicy):
    """Follows the evaluator, bounded by generic budgets.

    Precedence:
    1. terminal evaluator recommendations (success / failure / human)
    2. step budget
    3. failed observation → retry (if retryable and budget) or give up after
       too many consecutive failures
    4. evaluator replan / retry recommendations (bounded)
    5. continue

    Routing failures (:meth:`on_routing_failure`) stop the workflow unless
    ``max_routing_failures`` allows the router to be asked again.
    """

    max_steps: int = 50
    max_retries: int = 2
    max_consecutive_failures: int = 5
    max_replans: int = 3
    max_routing_failures: int = 0  # consecutive routing failures tolerated (re-route)
    policy_id: str = "default_policy"

    def decide(
        self,
        state: WorkflowState,
        evaluation: EvaluationResult | None,
        observation: Observation | None,
    ) -> ControlDecision:
        rec = evaluation.recommendation if evaluation else ControlAction.CONTINUE

        if rec.is_terminal:
            return self._d(rec, f"evaluator: {evaluation.rationale if evaluation else ''}")
        if state.step >= self.max_steps:
            return self._d(ControlAction.TERMINATE_FAILURE, f"step budget {self.max_steps} used")

        if observation is not None and not observation.success:
            if _consecutive_failures(state) >= self.max_consecutive_failures:
                return self._d(ControlAction.TERMINATE_FAILURE, "too many consecutive failures")
            retryable = observation.error is not None and observation.error.retryable
            if retryable and _trailing_retries(state) < self.max_retries:
                return self._d(ControlAction.RETRY, "retryable failure")
            return self._d(ControlAction.CONTINUE, "failure recorded; router may choose again")

        if rec is ControlAction.REPLAN:
            if _count(state, ControlAction.REPLAN) < self.max_replans:
                return self._d(ControlAction.REPLAN, "evaluator requested replan")
            return self._d(ControlAction.HUMAN_INTERVENTION, "replan budget exhausted")
        if rec is ControlAction.RETRY and _trailing_retries(state) < self.max_retries:
            return self._d(ControlAction.RETRY, "evaluator requested retry")
        return self._d(ControlAction.CONTINUE, "no stop condition met")

    def on_routing_failure(self, state: WorkflowState, error: ErrorInfo) -> ControlDecision:
        if _consecutive_routing_failures(state) <= self.max_routing_failures:
            return ControlDecision(
                action=ControlAction.CONTINUE,
                reason=f"routing failed ({error.type}); re-routing within budget "
                f"{self.max_routing_failures}",
                source=self.policy_id,
                metadata={"routing_error": error.model_dump(mode="json")},
            )
        return super().on_routing_failure(state, error)

    def _d(self, action: ControlAction, reason: str) -> ControlDecision:
        return ControlDecision(action=action, reason=reason, source=self.policy_id)


def _consecutive_failures(state: WorkflowState) -> int:
    n = 0
    for obs in reversed(state.observations):
        if obs.success:
            break
        n += 1
    return n


def _trailing_retries(state: WorkflowState) -> int:
    n = 0
    for entry in reversed(state.history):
        if entry.control is None:
            continue  # the entry currently being decided
        if entry.control.action is not ControlAction.RETRY:
            break
        n += 1
    return n


def _consecutive_routing_failures(state: WorkflowState) -> int:
    n = 0
    for entry in reversed(state.history):
        if entry.routing_error is None:
            break
        n += 1
    return n


def _count(state: WorkflowState, action: ControlAction) -> int:
    return sum(1 for e in state.history if e.control and e.control.action is action)
