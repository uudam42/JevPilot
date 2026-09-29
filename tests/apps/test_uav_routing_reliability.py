"""The UAV workflow under misbehaving routers: always bounded, never a silent fallback.

Each router here is the real ``JevRouter`` pipeline over an offline adapter whose
policy misbehaves the way a live model could (repeat a finished step, choose a step
whose preconditions do not hold, emit malformed output, time out, stop early).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from apps.uav_materials import DEMO_REQUEST, RunMode, UAVWorkflowResult, execute_workflow
from domains.uav_materials.interpretation import RuleBasedInterpreter
from domains.uav_materials.workflow import reference_routing_payload
from experiments.routing.real_jev import UAV_CASES
from jevpilot import JevRouter
from jevpilot.adapters import FakeJevAdapter

EXISTING = UAV_CASES["A_existing_material"]["request"]


def invoke(cid: str) -> dict[str, Any]:
    return {"action": "invoke", "capability_id": cid, "inputs": {}, "reason": "", "confidence": 0.9}


def run(policy: Callable[[Any], Any], request: str = DEMO_REQUEST, **kw: Any) -> UAVWorkflowResult:
    router = JevRouter(FakeJevAdapter(policy=policy), router_id="misbehaving_jev")
    return execute_workflow(
        request,
        interpreter=RuleBasedInterpreter(),
        router=router,
        mode=RunMode.OFFLINE_DEMO,
        max_steps=kw.get("max_steps", 30),
    )


def executed(result: UAVWorkflowResult) -> list[str]:
    return [str(o.capability_id) for o in result.state.observations]


def test_repeating_a_finished_step_is_rejected_not_executed_twice() -> None:
    result = run(lambda request: invoke("uavm.interpret_requirements"))
    assert executed(result) == ["uavm.interpret_requirements"]
    log = result.routing_log()
    assert log[-1]["status"] == "error:InvalidCapabilityError" and len(log) == 2
    assert result.report.status != "complete" and not result.complete


def test_design_cannot_run_when_its_preconditions_do_not_hold() -> None:
    def policy(request: Any) -> Any:
        offered = {c.id for c in request.available_capabilities}
        if "uavm.generate_material_report" in offered:
            return invoke("uavm.design_composite_candidate")  # existing branch: not offered
        return reference_routing_payload(request)

    result = run(policy, EXISTING)
    assert "uavm.design_composite_candidate" not in executed(result)
    assert result.decision == "use_existing"
    assert result.routing_log()[-1]["status"] == "error:InvalidCapabilityError"


def test_step_budget_stops_the_loop_with_an_incomplete_report() -> None:
    result = run(reference_routing_payload, max_steps=2)
    assert len(executed(result)) == 2
    assert result.report.status != "complete"
    assert "step budget" in result.run.control.reason


def test_malformed_output_stops_without_fallback() -> None:
    result = run(lambda request: {"action": "dance"})
    assert executed(result) == [] and result.report.status != "complete"
    assert result.routing_log()[-1]["status"] == "error:MalformedRoutingDecisionError"
    assert all(r["status"] != "fallback" for r in result.routing_log())


def test_model_timeout_stops_the_workflow() -> None:
    def slow(request: Any) -> Any:
        raise TimeoutError("no answer")

    result = run(slow)
    assert result.routing_log()[-1]["status"] == "error:RouterTimeoutError"
    assert result.report.status != "complete"


def test_premature_finish_is_reported_as_incomplete() -> None:
    result = run(lambda request: {"action": "finish", "reason": "", "confidence": 0.5})
    assert executed(result) == [] and result.report.status != "complete"
    assert result.routing_log()[-1]["intent"] == "finish"


def test_routing_log_holds_only_the_sanitized_fields() -> None:
    result = run(reference_routing_payload)
    allowed = {
        "timestamp",
        "step",
        "router",
        "model",
        "intent",
        "capability",
        "confidence",
        "latency_ms",
        "status",
        "retries",
    }
    assert result.complete and all(set(r) == allowed for r in result.routing_log())
