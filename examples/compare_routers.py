"""Phase 1 demo: same workflow, same initial state, same registry, different routers.

python examples/compare_routers.py

Rule, LLM and Jev routers drive the unmodified Controller. The LLM and Jev
routers here use OFFLINE FAKE adapters answering from a reference policy.
They show the plumbing, not the quality of any real model.
"""

from __future__ import annotations

from domains.demo import ArithmeticDomain, routing_rules
from experiments.routing.suites import arithmetic_policy
from jevpilot import (
    FallbackRouter,
    JevRouter,
    LLMRouter,
    Router,
    RuleRouter,
    Runtime,
    TraceEventType,
)
from jevpilot.adapters import FakeJevAdapter, FakeLLMAdapter
from jevpilot.adapters import scripting as fake


def main() -> None:
    routers: dict[str, Router] = {
        "rule": RuleRouter(routing_rules()),
        "llm (fake)": LLMRouter(FakeLLMAdapter(policy=arithmetic_policy)),
        "jev (fake)": JevRouter(FakeJevAdapter(policy=arithmetic_policy)),
        "jev→rule, jev broken": FallbackRouter(
            JevRouter(FakeJevAdapter([fake.malformed()], on_exhausted="repeat_last")),
            RuleRouter(routing_rules()),
        ),
    }
    runtime = Runtime()
    domain = runtime.load(ArithmeticDomain())
    assert isinstance(domain, ArithmeticDomain)
    initial = domain.new_workflow(start=3, target=20)
    for name, router in routers.items():
        result = runtime.controller(router).run(initial)
        decisions = [e for e in result.trace if e.type is TraceEventType.ROUTING_DECISION]
        route_ms = sum(e.payload["routing_latency_s"] for e in decisions) * 1000
        fallbacks = sum(bool(e.payload.get("fallback_used")) for e in decisions)
        path = " → ".join(str(h.decision.capability_id) for h in result.state.history)
        print(
            f"{name:<22} {result.state.status!s:<10} steps={result.state.step} "
            f"routing={route_ms:.2f}ms fallbacks={fallbacks}  {path}"
        )


if __name__ == "__main__":
    main()
