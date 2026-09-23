"""Explicit, traceable fallback between routers, by composition.

    FallbackRouter(JevRouter(...), RuleRouter(...))            Jev → Rule
    FallbackRouter(JevRouter(...), LLMRouter(...), RuleRouter(...))   Jev → LLM → Rule

Routers never fall back on their own. A fallback happens only here, only for
the :class:`RoutingError` types listed in ``fallback_on``, and every attempt
is kept: the returned :class:`RoutingOutcome` lists the failed attempts
(router, error, latency, usage) before the one that succeeded. The decision's
``metadata["fallback"]`` records the primary router, the failure types and
which router finally decided. Other exceptions (router bugs) propagate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jevpilot.core import RoutingAttempt, RoutingDecision, RoutingOutcome, WorkflowState
from jevpilot.exceptions import RoutingError
from jevpilot.interfaces.capability import CapabilitySpec
from jevpilot.interfaces.router import Router, error_info, validate_decision


class FallbackRouter(Router):
    router_id = "fallback_router"

    def __init__(
        self,
        primary: Router,
        *fallbacks: Router,
        fallback_on: tuple[type[RoutingError], ...] = (RoutingError,),
        router_id: str | None = None,
    ) -> None:
        if not fallbacks:
            raise ValueError("FallbackRouter needs at least one fallback router")
        self.chain: tuple[Router, ...] = (primary, *fallbacks)
        self.fallback_on = fallback_on
        if router_id is not None:
            self.router_id = router_id

    def select_next(
        self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]
    ) -> RoutingDecision:
        return self.route(state, capabilities).decision

    def route(self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]) -> RoutingOutcome:
        attempts: list[RoutingAttempt] = []
        failures: list[dict[str, str]] = []
        for position, router in enumerate(self.chain):
            try:
                outcome = router.route(state, capabilities)
                validate_decision(outcome.decision, capabilities)
            except RoutingError as exc:
                attempts.extend(_attempts_of(exc, router))
                failures.append({"router_id": router.router_id, "error": type(exc).__name__})
                last = position == len(self.chain) - 1
                if last or not isinstance(exc, self.fallback_on):
                    exc.attempts = tuple(attempts)
                    exc.details = {**exc.details, "fallback_failures": failures}
                    raise
                continue
            attempts.extend(outcome.attempts)
            decision = outcome.decision
            if position > 0:
                decision = decision.model_copy(
                    update={
                        "metadata": {
                            **decision.metadata,
                            "fallback": {
                                "primary_router": self.chain[0].router_id,
                                "decided_by": router.router_id,
                                "failures": failures,
                            },
                        }
                    }
                )
            return RoutingOutcome(decision=decision, attempts=tuple(attempts))
        raise AssertionError("unreachable: the chain is never empty")

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "chain": [r.config() for r in self.chain],
            "fallback_on": [t.__name__ for t in self.fallback_on],
        }


def _attempts_of(exc: RoutingError, router: Router) -> list[RoutingAttempt]:
    """The attempts a failing router reported, or a synthesised one if it reported none."""
    recorded = [a for a in exc.attempts if isinstance(a, RoutingAttempt)]
    if recorded:
        return recorded
    return [router.attempt(success=False, error=error_info(exc))]
