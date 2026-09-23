"""Router factories for benchmarks.

Every router is built fresh per workflow or decision case from a name, so
runs are independent and reproducible:

=================  ==========================================================
``rule``           RuleRouter with the suite's rules (deterministic baseline)
``jev``            JevRouter + FakeJevAdapter  (SIMULATED, offline)
``llm``            LLMRouter + FakeLLMAdapter  (SIMULATED, offline)
``jev+rule``       FallbackRouter(jev, rule)
``llm+rule``       FallbackRouter(llm, rule)
``jev+llm+rule``   FallbackRouter(jev, llm, rule)
``llm-anthropic``  LLMRouter + AnthropicLLMAdapter (REAL; needs credentials, costs money)
=================  ==========================================================

The simulated adapters answer with the suite's reference policy, wrapped with
seeded fault injection. Their numbers measure the harness and the
failure-handling machinery. They say nothing about how Jev or any LLM routes.
"""

from __future__ import annotations

import importlib
from typing import Any

from experiments.routing.suites import Suite
from jevpilot import FallbackRouter, JevRouter, LLMRouter, Router, RuleRouter
from jevpilot.adapters import FakeJevAdapter, FakeLLMAdapter, with_faults

SIMULATED_ROUTERS = ("rule", "jev", "llm", "jev+rule", "llm+rule", "jev+llm+rule")
REAL_ROUTERS = ("llm-anthropic",)
ROUTER_NAMES = SIMULATED_ROUTERS + REAL_ROUTERS

# Arbitrary simulation parameters chosen to exercise every failure path.
# They are not estimates of any real model's error rates.
JEV_FAULTS = {
    "malformed": 0.04,
    "invalid_capability": 0.04,
    "invalid_inputs": 0.04,
    "timeout": 0.03,
}
LLM_FAULTS = {
    "malformed": 0.06,
    "invalid_capability": 0.04,
    "invalid_inputs": 0.04,
    "adapter_error": 0.02,
}
SIMULATION_LABEL = "simulated: suite reference policy + seeded fault injection"


def make_router(name: str, suite: Suite, *, seed: int, faults: bool = True) -> Router:
    if name == "rule":
        return RuleRouter(suite.rules())
    if name == "jev":
        policy = with_faults(suite.policy, JEV_FAULTS if faults else {}, seed=seed)
        return JevRouter(FakeJevAdapter(policy=policy, label=SIMULATION_LABEL))
    if name == "llm":
        policy = with_faults(suite.policy, LLM_FAULTS if faults else {}, seed=seed + 1)
        return LLMRouter(FakeLLMAdapter(policy=policy, label=SIMULATION_LABEL))
    if name == "llm-anthropic":
        module: Any = importlib.import_module("integrations.anthropic_llm")
        return LLMRouter(module.AnthropicLLMAdapter(), timeout_s=120)
    if "+" in name:
        parts = name.split("+")
        routers = [make_router(p, suite, seed=seed, faults=faults) for p in parts]
        return FallbackRouter(routers[0], *routers[1:], router_id=f"fallback({name})")
    raise ValueError(f"unknown router {name!r}; choose from {', '.join(ROUTER_NAMES)}")


def fault_config(name: str, faults: bool) -> dict[str, Any]:
    parts = name.split("+")
    return {
        p: (JEV_FAULTS if p == "jev" else LLM_FAULTS) if faults else {}
        for p in parts
        if p in ("jev", "llm")
    }
