"""Controller behaviours (retry, replan, escalation, safety) with throwaway capabilities."""

from typing import Any

from jevpilot import (
    CapabilityRegistry,
    ControlAction,
    ControlDecision,
    Controller,
    ControlPolicy,
    EvaluationResult,
    Evaluator,
    NullEvaluator,
    Plan,
    Planner,
    Router,
    RoutingDecision,
    ScriptedRouter,
    WorkflowState,
    WorkflowStatus,
)
from tests.conftest import make_capability


class DoneAfter(Evaluator):
    """Success once ``n`` successful observations exist."""

    def __init__(self, n: int) -> None:
        self.n = n

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        ok = sum(o.success for o in state.observations)
        rec = ControlAction.TERMINATE_SUCCESS if ok >= self.n else ControlAction.CONTINUE
        return EvaluationResult(evaluator_id="done_after", recommendation=rec)


def test_retry_reexecutes_same_decision(registry: CapabilityRegistry, state: WorkflowState) -> None:
    calls = {"n": 0}

    def flaky(i: Any, c: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("transient")
        return "ok"

    registry.register(make_capability("t.flaky", flaky))
    result = Controller(registry, ScriptedRouter([("t.flaky", {})]), DoneAfter(1)).run(state)
    assert result.succeeded and calls["n"] == 2
    first, second = result.state.history
    assert first.control and first.control.action is ControlAction.RETRY
    assert second.decision.metadata["retry_of"] == first.decision.id


def test_router_idle_escalates_to_human(registry: CapabilityRegistry, state: WorkflowState) -> None:
    registry.register(make_capability("t.noop"))
    result = Controller(registry, ScriptedRouter([]), NullEvaluator()).run(state)
    assert result.state.status is WorkflowStatus.AWAITING_HUMAN
    assert result.control.action is ControlAction.HUMAN_INTERVENTION


def test_no_capabilities_escalates(registry: CapabilityRegistry, state: WorkflowState) -> None:
    result = Controller(registry, ScriptedRouter([]), NullEvaluator()).run(state)
    assert result.state.status is WorkflowStatus.AWAITING_HUMAN


def test_replan_invokes_planner_again(registry: CapabilityRegistry, state: WorkflowState) -> None:
    class ReplanOnce(Evaluator):
        def evaluate(self, s: WorkflowState) -> EvaluationResult:
            rec = {1: ControlAction.REPLAN, 2: ControlAction.TERMINATE_SUCCESS}.get(
                s.step, ControlAction.CONTINUE
            )
            return EvaluationResult(evaluator_id="r", recommendation=rec)

    class CountingPlanner(Planner):
        calls = 0

        def plan(self, s: WorkflowState) -> Plan:
            CountingPlanner.calls += 1
            return Plan(rationale=f"call {self.calls}")

    registry.register(make_capability("t.step"))
    ctl = Controller(
        registry, ScriptedRouter([("t.step", {})] * 2), ReplanOnce(), planner=CountingPlanner()
    )
    result = ctl.run(state)
    assert result.succeeded and CountingPlanner.calls == 2
    assert result.state.plan and result.state.plan.revision == 1


def test_custom_policy_and_safety_cap(registry: CapabilityRegistry, state: WorkflowState) -> None:
    class Forever(ControlPolicy):
        def decide(self, *a: Any) -> ControlDecision:
            return ControlDecision(action=ControlAction.CONTINUE)

    class Loop(Router):
        def select_next(self, s: WorkflowState, caps: Any) -> RoutingDecision:
            return RoutingDecision(capability_id="t.step")

    registry.register(make_capability("t.step"))
    result = Controller(registry, Loop(), NullEvaluator(), policy=Forever(), max_iterations=5).run(
        state
    )
    assert result.state.status is WorkflowStatus.FAILED and result.state.step == 5
    assert "safety cap" in result.control.reason


def test_router_crash_fails_workflow_with_trace(
    registry: CapabilityRegistry, state: WorkflowState
) -> None:
    class Broken(Router):
        def select_next(self, s: WorkflowState, caps: Any) -> RoutingDecision:
            raise RuntimeError("router bug")

    registry.register(make_capability("t.step"))
    result = Controller(registry, Broken(), NullEvaluator()).run(state)
    assert result.state.status is WorkflowStatus.FAILED
    assert result.error and "router bug" in result.error.message
    assert result.trace[-2].type == "error"
