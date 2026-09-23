"""Every router satisfies one contract, and swapping routers needs no controller change."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from domains.demo import ArithmeticDomain, NumberState, routing_rules
from experiments.routing.suites import arithmetic_policy
from jevpilot import (
    CapabilitySpec,
    Controller,
    Executor,
    FallbackRouter,
    JevRouter,
    LLMRouter,
    Router,
    RoutingDecision,
    RoutingOutcome,
    RuleRouter,
    Runtime,
    ScriptedRouter,
    StateManager,
    WorkflowState,
    WorkflowStatus,
)
from jevpilot.adapters import FakeJevAdapter, FakeLLMAdapter
from jevpilot.adapters import scripting as fake

RouterFactory = Callable[[], Router]

ROUTERS: dict[str, RouterFactory] = {
    "rule": lambda: RuleRouter(routing_rules()),
    "scripted": lambda: ScriptedRouter(
        [
            ("arith.multiply", {"factor": 2}),
            ("arith.multiply", {"factor": 2}),
            ("arith.add", {"amount": 8}),
            ("arith.compare", {}),
        ]
    ),
    "llm": lambda: LLMRouter(FakeLLMAdapter(policy=arithmetic_policy)),
    "jev": lambda: JevRouter(FakeJevAdapter(policy=arithmetic_policy)),
    "jev+rule": lambda: FallbackRouter(
        JevRouter(FakeJevAdapter([fake.malformed()], on_exhausted="repeat_last")),
        RuleRouter(routing_rules()),
    ),
}


def _run(router: Router) -> tuple[Controller, Any]:
    rt = Runtime()
    dom = rt.load(ArithmeticDomain())
    assert isinstance(dom, ArithmeticDomain)
    ctl = rt.controller(router)
    return ctl, ctl.run(dom.new_workflow(start=3, target=20))


@pytest.mark.parametrize("name", ROUTERS)
def test_every_router_returns_the_same_decision_abstraction(name: str) -> None:
    rt = Runtime()
    dom = rt.load(ArithmeticDomain())
    assert isinstance(dom, ArithmeticDomain)
    state = dom.new_workflow(start=3, target=20)
    specs = rt.capabilities.available(state)
    router = ROUTERS[name]()
    outcome = router.route(state, specs)
    assert isinstance(outcome, RoutingOutcome) and isinstance(outcome.decision, RoutingDecision)
    assert outcome.decision.capability_id == "arith.multiply"
    assert isinstance(router.config(), dict) and router.config()["router_id"] == router.router_id


@pytest.mark.parametrize("name", ROUTERS)
def test_router_replacement_needs_no_controller_change(name: str) -> None:
    ctl, result = _run(ROUTERS[name]())
    assert type(ctl) is Controller and type(ctl.executor) is Executor
    assert type(ctl.state_manager) is StateManager
    state = result.state
    assert isinstance(state, NumberState)
    assert result.succeeded and state.value == 20
    assert [h.decision.capability_id for h in state.history] == [
        "arith.multiply",
        "arith.multiply",
        "arith.add",
        "arith.compare",
    ]


def test_jev_and_llm_are_interchangeable() -> None:
    a = _run(ROUTERS["jev"]())[1].state
    b = _run(ROUTERS["llm"]())[1].state
    assert [h.decision.capability_id for h in a.history] == [
        h.decision.capability_id for h in b.history
    ]
    assert a.status is b.status is WorkflowStatus.SUCCEEDED


def test_routers_only_ever_see_descriptors() -> None:
    seen: list[Any] = []

    class Spy(Router):
        def select_next(
            self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]
        ) -> RoutingDecision:
            seen.extend(capabilities)
            return RoutingDecision.finish("done")

    _run(Spy())
    assert seen and all(type(c) is CapabilitySpec for c in seen)
    assert not any(hasattr(c, "execute") for c in seen)
