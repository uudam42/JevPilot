"""Generic rule-based router: the deterministic, model-free routing baseline.

Rules are data supplied from outside (by a domain or a benchmark suite), never
knowledge built into the core. The router is deterministic, makes no model
calls, and its rules can be listed with :meth:`RuleRouter.config`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevpilot.core import RoutingDecision, WorkflowState
from jevpilot.interfaces.capability import CapabilitySpec
from jevpilot.interfaces.router import Router

Predicate = Callable[[WorkflowState], bool]
InputsFn = Callable[[WorkflowState], dict[str, Any]]


@dataclass(frozen=True)
class Rule:
    """If ``when(state)`` holds and ``capability_id`` is available, invoke it."""

    capability_id: str
    when: Predicate = lambda _state: True
    inputs: InputsFn | dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class RuleRouter(Router):
    """First matching rule wins; no match → no-action."""

    router_id = "rule_router"

    def __init__(self, rules: Sequence[Rule]) -> None:
        self._rules = tuple(rules)

    def select_next(
        self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]
    ) -> RoutingDecision:
        available = {c.id for c in capabilities}
        for rule in self._rules:
            if rule.capability_id in available and rule.when(state):
                inputs = rule.inputs(state) if callable(rule.inputs) else dict(rule.inputs)
                others = tuple(sorted(available - {rule.capability_id}))
                return RoutingDecision(
                    capability_id=rule.capability_id,
                    inputs=inputs,
                    reason=rule.reason or f"rule for {rule.capability_id} matched",
                    confidence=1.0,
                    alternatives=others,
                    router_id=self.router_id,
                )
        return RoutingDecision.no_action("no rule matched", router_id=self.router_id)

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "rules": [{"capability_id": r.capability_id, "reason": r.reason} for r in self._rules],
        }
