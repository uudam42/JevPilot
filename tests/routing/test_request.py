import json

from domains.demo import ArithmeticDomain, routing_rules
from jevpilot import (
    Capability,
    CapabilitySpec,
    Plan,
    PlanStep,
    RoutingRequest,
    RuleRouter,
    Runtime,
    WorkflowState,
)
from tests.routing.helpers import SPECS, new_state


def _arith_request() -> tuple[WorkflowState, RoutingRequest]:
    rt = Runtime()
    dom = rt.load(ArithmeticDomain())
    assert isinstance(dom, ArithmeticDomain)
    result = rt.controller(RuleRouter(routing_rules())).run(dom.new_workflow(start=3, target=20))
    state = result.state
    return state, RoutingRequest.from_state(state, rt.capabilities.list())


def test_request_is_pure_json_data() -> None:
    _, req = _arith_request()
    text = req.to_json()
    assert RoutingRequest.model_validate(json.loads(text)) == req  # round trip
    assert json.loads(text) == req.to_dict()


def test_request_contains_descriptions_not_executables() -> None:
    _, req = _arith_request()
    for cap in req.available_capabilities:
        assert isinstance(cap.input_schema, dict | type(None))
    flat = req.to_json()
    assert "execute" not in flat and "<class" not in flat and "object at 0x" not in flat
    for value in req.to_dict().values():
        assert not isinstance(value, Capability | CapabilitySpec | type)


def test_request_exposes_history_evaluations_and_state_extensions() -> None:
    state, req = _arith_request()
    assert req.step == state.step == 4
    assert [a.capability_id for a in req.previous_actions] == [
        "arith.multiply",
        "arith.multiply",
        "arith.add",
        "arith.compare",
    ]
    assert req.previous_actions[-1].success is True
    assert req.previous_actions[-1].result["equal"] is True
    assert req.state_summary.extensions == {"value": 20.0, "target": 20.0}
    assert req.previous_evaluations and req.previous_evaluations[-1].recommendation
    assert req.goal["parameters"] == {"start": 3, "target": 20}


def test_request_is_deterministic_for_same_state() -> None:
    state = new_state().evolve(plan=Plan(steps=(PlanStep(description="write"),)))
    a = RoutingRequest.from_state(state, SPECS)
    b = RoutingRequest.from_state(state, SPECS)
    assert a.to_json() == b.to_json() and a.fingerprint() == b.fingerprint()
    assert a.state_summary.plan[0]["description"] == "write"


def test_request_truncates_large_values_and_limits_history() -> None:
    state = new_state().evolve(context={"blob": "x" * 10_000})
    req = RoutingRequest.from_state(state, SPECS, max_value_chars=100)
    blob = req.state_summary.context["blob"]
    assert blob["_truncated"] is True and len(blob["preview"]) == 100
    _, full = _arith_request()
    state2, _ = _arith_request()
    assert len(RoutingRequest.from_state(state2, SPECS, max_actions=2).previous_actions) == 2
    assert full.retry_count == 0


def test_capability_costs_are_passed_through_not_invented() -> None:
    req = RoutingRequest.from_state(new_state(), SPECS)
    assert all(
        c.cost_estimate is None and c.latency_estimate is None for c in req.available_capabilities
    )
