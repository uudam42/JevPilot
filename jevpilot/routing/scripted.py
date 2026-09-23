"""Deterministic router that replays a fixed script (testing/benchmark baseline)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jevpilot.core import RoutingDecision, WorkflowState
from jevpilot.interfaces.capability import CapabilitySpec
from jevpilot.interfaces.router import Router


class ScriptedRouter(Router):
    """Returns ``(capability_id, inputs)`` pairs in order, then no-action.

    Stateful: create a new instance (or call :meth:`reset`) per workflow run.
    """

    router_id = "scripted_router"

    def __init__(self, script: Sequence[tuple[str, dict[str, Any]]]) -> None:
        self._script = list(script)
        self._pos = 0

    def reset(self) -> None:
        self._pos = 0

    def select_next(
        self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]
    ) -> RoutingDecision:
        if self._pos >= len(self._script):
            return RoutingDecision.no_action("script exhausted", router_id=self.router_id)
        capability_id, inputs = self._script[self._pos]
        self._pos += 1
        return RoutingDecision(
            capability_id=capability_id,
            inputs=dict(inputs),
            reason=f"script step {self._pos}",
            confidence=1.0,
            router_id=self.router_id,
        )
