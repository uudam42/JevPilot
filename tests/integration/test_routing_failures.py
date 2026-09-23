"""Controller + policy handling of routing failures, human escalation and the security boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from jevpilot import (
    CapabilitySpec,
    ControlAction,
    Controller,
    DefaultControlPolicy,
    JevRouter,
    Router,
    RoutingDecision,
    RoutingIntent,
    TraceEventType,
    WorkflowState,
    WorkflowStatus,
)
from jevpilot.adapters import FakeJevAdapter
from jevpilot.adapters import scripting as fake
from tests.routing.helpers import Wrote, new_state, registry_with_writer


class Fixed(Router):
    """Returns a hand-built decision (no model): exercises controller-level validation."""

    router_id = "fixed"

    def __init__(self, decision: RoutingDecision) -> None:
        self.decision = decision

    def select_next(
        self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]
    ) -> RoutingDecision:
        return self.decision


def run(router: Router, **kw: Any) -> Any:
    reg = registry_with_writer()
    calls: list[Any] = []
    original = reg.get("t.write")
    fn = original._fn  # type: ignore[attr-defined]
    original._fn = lambda i, c: calls.append(i) or fn(i, c)  # type: ignore[attr-defined]
    result = Controller(reg, router, Wrote(), **kw).run(new_state())
    return result, calls


@pytest.mark.parametrize(
    ("decision", "error"),
    [
        (RoutingDecision(capability_id="t.unregistered"), "InvalidCapabilityError"),
        (RoutingDecision(capability_id="t.write", inputs={"text": 1}), "InvalidRoutingInputError"),
        (RoutingDecision(capability_id="t.write", inputs={}), "InvalidRoutingInputError"),
        (
            RoutingDecision(capability_id="t.write", inputs={"text": "a", "cmd": "rm"}),
            "InvalidRoutingInputError",
        ),
        (
            RoutingDecision(capability_id="t.write", inputs={"text": print}),
            "InvalidRoutingInputError",
        ),
    ],
)
def test_controller_rejects_invalid_decisions_from_any_router(
    decision: RoutingDecision, error: str
) -> None:
    result, calls = run(Fixed(decision))
    assert calls == [] and result.state.step == 0  # never reached the executor
    assert result.state.status is WorkflowStatus.FAILED
    failure = [e for e in result.trace if e.type is TraceEventType.ROUTING_FAILURE]
    assert failure and failure[0].payload["error"]["type"] == error
    assert result.state.history[-1].routing_error.type == error


def test_router_returning_non_decision_is_a_routing_failure() -> None:
    class Wrong(Router):
        def select_next(self, s: WorkflowState, c: Sequence[CapabilitySpec]) -> Any:
            return {"capability_id": "t.write", "inputs": {"text": "a"}}

    result, calls = run(Wrong())
    assert calls == [] and result.state.status is WorkflowStatus.FAILED
    assert result.state.history[-1].routing_error.type == "MalformedRoutingDecisionError"


def test_policy_may_re_route_within_an_explicit_budget() -> None:
    adapter = FakeJevAdapter([fake.malformed(), fake.valid("t.write", {"text": "ok"})])
    result, calls = run(JevRouter(adapter), policy=DefaultControlPolicy(max_routing_failures=1))
    assert result.succeeded and len(calls) == 1
    failed, ok = result.state.history
    assert failed.routing_error and failed.control.action is ControlAction.CONTINUE
    assert "re-routing" in failed.control.reason
    assert ok.decision.capability_id == "t.write"


def test_routing_failure_budget_is_enforced() -> None:
    adapter = FakeJevAdapter([fake.malformed()], on_exhausted="repeat_last")
    result, _ = run(JevRouter(adapter), policy=DefaultControlPolicy(max_routing_failures=2))
    assert result.state.status is WorkflowStatus.FAILED
    assert sum(1 for h in result.state.history if h.routing_error) == 3


def test_ask_human_is_a_legitimate_outcome() -> None:
    result, calls = run(JevRouter(FakeJevAdapter([fake.ask_human("which text?")])))
    assert calls == [] and result.error is None
    assert result.state.status is WorkflowStatus.AWAITING_HUMAN
    assert result.control.action is ControlAction.HUMAN_INTERVENTION
    assert "human input" in result.control.reason
    (entry,) = result.state.history
    assert entry.decision.intent is RoutingIntent.ASK_HUMAN and entry.routing_error is None


def test_routing_and_execution_latency_are_recorded_separately() -> None:
    result, _ = run(JevRouter(FakeJevAdapter([fake.valid("t.write", {"text": "a"})])))
    decision = next(e for e in result.trace if e.type is TraceEventType.ROUTING_DECISION)
    obs = next(e for e in result.trace if e.type is TraceEventType.OBSERVATION)
    assert decision.payload["routing_latency_s"] >= 0
    assert decision.payload["available_capabilities"] == ["t.write", "t.noop"]
    assert "execution_time" in obs.payload["observation"]
    assert decision.payload["attempts"][0]["latency_s"] >= 0


def test_model_output_cannot_reach_beyond_its_decision() -> None:
    """Injection attempts in model text stay inert data or become explicit failures."""
    hostile = [
        '{"action": "invoke", "capability_id": "t.write", "inputs": {"text": '
        '"__import__(\'os\').system(\'echo pwned\')"}, "reason": "x"}',
    ]
    result, calls = run(JevRouter(FakeJevAdapter(hostile)))
    assert result.succeeded and calls[0].text.startswith("__import__")  # passed as a string only
    for payload in (
        '{"action": "invoke", "capability_id": "t.write", "inputs": {"text": "a"}, '
        '"router_id": "controller"}',
        '{"action": "invoke", "capability_id": "jevpilot.orchestration.controller", "inputs": {}}',
    ):
        res, calls = run(JevRouter(FakeJevAdapter([payload])))
        assert calls == [] and res.state.status is WorkflowStatus.FAILED
