"""Fallback is explicit, configurable and fully traced; without it, failures stop the run."""

from __future__ import annotations

from typing import Any

import pytest

from jevpilot import (
    Controller,
    FallbackRouter,
    JevRouter,
    LLMRouter,
    Rule,
    RuleRouter,
    TraceEventType,
    WorkflowStatus,
)
from jevpilot.adapters import FakeJevAdapter, FakeLLMAdapter
from jevpilot.adapters import scripting as fake
from jevpilot.exceptions import (
    InvalidCapabilityError,
    MalformedRoutingDecisionError,
    RouterTimeoutError,
)
from tests.routing.helpers import SPECS, Wrote, new_state, registry_with_writer

RULE = [Rule("t.write", inputs={"text": "from rule"}, reason="baseline")]


def failing_jev(item: Any = None) -> JevRouter:
    return JevRouter(
        FakeJevAdapter([item if item is not None else fake.malformed()], on_exhausted="repeat_last")
    )


def test_primary_failure_runs_fallback_and_records_everything() -> None:
    router = FallbackRouter(failing_jev(), RuleRouter(RULE))
    outcome = router.route(new_state(), SPECS)
    assert outcome.decision.capability_id == "t.write"
    assert outcome.decision.router_id == "rule_router"
    assert outcome.fallback_used
    first, second = outcome.attempts
    assert (first.router_id, first.success, first.error.type) == (
        "jev_router",
        False,
        "MalformedRoutingDecisionError",
    )
    assert (second.router_id, second.success) == ("rule_router", True)
    fb = outcome.decision.metadata["fallback"]
    assert fb["primary_router"] == "jev_router" and fb["decided_by"] == "rule_router"
    assert fb["failures"] == [{"router_id": "jev_router", "error": "MalformedRoutingDecisionError"}]


def test_three_level_chain() -> None:
    llm = LLMRouter(FakeLLMAdapter([fake.timeout()]))
    router = FallbackRouter(failing_jev(fake.invalid_capability()), llm, RuleRouter(RULE))
    outcome = router.route(new_state(), SPECS)
    assert [a.error.type if a.error else "ok" for a in outcome.attempts] == [
        "InvalidCapabilityError",
        "RouterTimeoutError",
        "ok",
    ]


def test_fallback_on_restricts_which_failures_fall_back() -> None:
    router = FallbackRouter(
        failing_jev(fake.timeout()), RuleRouter(RULE), fallback_on=(MalformedRoutingDecisionError,)
    )
    with pytest.raises(RouterTimeoutError):
        router.route(new_state(), SPECS)


def test_all_routers_failing_raises_with_all_attempts() -> None:
    router = FallbackRouter(failing_jev(), JevRouter(FakeJevAdapter([fake.invalid_capability()])))
    with pytest.raises(InvalidCapabilityError) as info:
        router.route(new_state(), SPECS)
    assert [a.error.type for a in info.value.attempts] == [
        "MalformedRoutingDecisionError",
        "InvalidCapabilityError",
    ]
    assert info.value.details["fallback_failures"][0]["router_id"] == "jev_router"


def test_a_no_action_from_the_primary_is_a_decision_not_a_failure() -> None:
    router = FallbackRouter(RuleRouter([Rule("t.missing")]), RuleRouter(RULE))
    outcome = router.route(new_state(), SPECS)
    assert outcome.decision.is_no_action and not outcome.fallback_used


def test_fallback_validates_rule_decisions_too() -> None:
    bad_rule = RuleRouter([Rule("t.write", inputs={"text": 7})])  # schema-violating rule
    router = FallbackRouter(bad_rule, RuleRouter(RULE))
    outcome = router.route(new_state(), SPECS)
    assert outcome.attempts[0].error.type == "InvalidRoutingInputError"
    assert outcome.decision.inputs == {"text": "from rule"}


def test_workflow_with_fallback_completes_and_trace_shows_it() -> None:
    ctl = Controller(
        registry_with_writer(), FallbackRouter(failing_jev(), RuleRouter(RULE)), Wrote()
    )
    result = ctl.run(new_state())
    assert result.succeeded
    (event,) = [e for e in result.trace if e.type is TraceEventType.ROUTING_DECISION]
    assert event.payload["fallback_used"] is True
    assert [a["router_id"] for a in event.payload["attempts"]] == ["jev_router", "rule_router"]
    assert event.payload["attempts"][0]["request"]["goal"]["description"] == "write hello"
    started = result.trace[0].payload["router_config"]
    assert [c["router_id"] for c in started["chain"]] == ["jev_router", "rule_router"]


def test_without_fallback_the_workflow_stops_explicitly() -> None:
    ctl = Controller(registry_with_writer(), failing_jev(), Wrote())
    result = ctl.run(new_state())
    assert result.state.status is WorkflowStatus.FAILED
    assert result.state.step == 0 and not result.state.observations  # nothing executed
    types = [e.type for e in result.trace]
    assert TraceEventType.ROUTING_FAILURE in types and TraceEventType.OBSERVATION not in types
    (entry,) = result.state.history
    assert entry.routing_error and entry.routing_error.type == "MalformedRoutingDecisionError"
    assert "routing failed" in result.control.reason
