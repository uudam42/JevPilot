"""Acceptance: fundamentally different domains share one unmodified core."""

from __future__ import annotations

from collections.abc import Sequence

import domains.demo as arithmetic
import domains.stats_demo as stats
from jevpilot import (
    Capability,
    ControlAction,
    Controller,
    DomainModule,
    EvaluationResult,
    Evaluator,
    Executor,
    Rule,
    RuleRouter,
    Runtime,
    StateManager,
    WorkflowState,
    WorkflowStatus,
)
from tests.conftest import make_capability


def test_fresh_runtime_knows_no_domain() -> None:
    rt = Runtime()
    assert len(rt.capabilities) == 0 and rt.domains.list() == []


def test_two_unrelated_domains_share_the_same_core() -> None:
    rt = Runtime()
    a = rt.load("domains.demo:ArithmeticDomain")
    b = rt.load("domains.stats_demo:StatsDomain")
    assert {s.domain for s in rt.capabilities.list()} == {"arithmetic", "stats"}

    ctl_a = rt.controller(RuleRouter(arithmetic.routing_rules()), domains=["arithmetic"])
    ctl_b = rt.controller(RuleRouter(stats.routing_rules()), domains=["stats"])
    res_a = ctl_a.run(a.new_workflow(start=2, target=9))  # type: ignore[attr-defined]
    res_b = ctl_b.run(  # type: ignore[attr-defined]
        b.new_workflow(true_mean=1.0, noise=1.0, ci_half_width=0.3)  # type: ignore[attr-defined]
    )

    assert res_a.state.status is res_b.state.status is WorkflowStatus.SUCCEEDED
    # Identical core machinery on both sides.
    for ctl in (ctl_a, ctl_b):
        assert type(ctl) is Controller
        assert type(ctl.executor) is Executor
        assert type(ctl.state_manager) is StateManager
        assert ctl.registry is rt.capabilities
    # Domains stay in their lane: each workflow only used its own capabilities.
    assert {h.decision.capability_id.split(".")[0] for h in res_a.state.history} == {"arith"}
    assert {h.decision.capability_id.split(".")[0] for h in res_b.state.history} == {"stats"}


def test_swapping_domains_in_and_out() -> None:
    rt = Runtime()
    rt.load("arithmetic")  # via entry point
    rt.domains.unload("arithmetic")
    rt.load("stats")
    assert {s.domain for s in rt.capabilities.list()} == {"stats"}


def test_ad_hoc_third_domain_needs_no_core_change() -> None:
    """A domain defined right here, unknown to anything else, runs on the core."""

    class Greeter(Evaluator):
        def evaluate(self, state: WorkflowState) -> EvaluationResult:
            done = state.context.get("greeting") is not None
            return EvaluationResult(
                evaluator_id="greet",
                recommendation=ControlAction.TERMINATE_SUCCESS if done else ControlAction.CONTINUE,
            )

    class GreetingDomain(DomainModule):
        name = "greeting"

        def capabilities(self) -> Sequence[Capability]:
            from jevpilot import CapabilityResult, StateEffects

            return [
                make_capability(
                    "greet.say",
                    lambda i, ctx: CapabilityResult(
                        output="hello",
                        effects=StateEffects(context_updates={"greeting": "hello"}),
                    ),
                    domain="greeting",
                )
            ]

        def evaluators(self) -> Sequence[Evaluator]:
            return [Greeter()]

    rt = Runtime()
    d = rt.load(GreetingDomain())
    result = rt.controller(RuleRouter([Rule("greet.say")])).run(d.create_state("say hello"))
    assert result.succeeded and result.state.context["greeting"] == "hello"
