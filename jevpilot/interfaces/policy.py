"""Control policy: turns evaluation + observation into the loop's next action."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jevpilot.core import (
    ControlAction,
    ControlDecision,
    ErrorInfo,
    EvaluationResult,
    Observation,
    WorkflowState,
)


class ControlPolicy(ABC):
    """Decides continue / retry / replan / terminate / escalate.

    Evaluators *recommend*; the policy *decides*, applying domain-neutral
    budgets (steps, retries, replans). Swapping the policy changes termination
    behaviour without touching the controller.
    """

    policy_id: str = "policy"

    @abstractmethod
    def decide(
        self,
        state: WorkflowState,
        evaluation: EvaluationResult | None,
        observation: Observation | None,
    ) -> ControlDecision: ...

    def on_routing_failure(self, state: WorkflowState, error: ErrorInfo) -> ControlDecision:
        """Decide what happens when the router failed to produce a valid decision.

        The default stops the workflow: a routing failure never silently turns
        into continued execution. Policies may override this to re-route
        (``CONTINUE``) within a budget or to escalate to a human.
        """
        return ControlDecision(
            action=ControlAction.TERMINATE_FAILURE,
            reason=f"routing failed ({error.type}): {error.message}",
            source=self.policy_id,
            metadata={"routing_error": error.model_dump(mode="json")},
        )
