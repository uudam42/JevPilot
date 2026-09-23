"""Benchmark suites: an executable domain plus the routing heuristics that go with it.

Each suite supplies:

- ``domain()``: a fresh :class:`DomainModule` (the Phase 0 demo domains are reused unchanged);
- ``new_state(domain, params)``: the initial workflow state for a workflow case;
- ``rules()``: rules for the :class:`RuleRouter` baseline;
- ``policy``: a request-level policy that drives the *simulated* (fake) model
  routers. It sees only the JSON :class:`RoutingRequest`, like a real model.

All domain knowledge used by routing lives here, never in the core.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import domains.demo as arithmetic
import domains.stats_demo as stats
from experiments.routing.suites import pipeline
from jevpilot import DomainModule, Rule, WorkflowState
from jevpilot.routing import RoutingRequest


@dataclass(frozen=True)
class Suite:
    name: str
    domain: Callable[[], DomainModule]
    new_state: Callable[[DomainModule, dict[str, Any]], WorkflowState]
    rules: Callable[[], list[Rule]]
    policy: Callable[[RoutingRequest], dict[str, Any]]


def _invoke(cid: str, inputs: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "action": "invoke",
        "capability_id": cid,
        "inputs": inputs,
        "reason": reason,
        "confidence": 0.9,
    }


def _finish(reason: str) -> dict[str, Any]:
    return {
        "action": "finish",
        "capability_id": None,
        "inputs": {},
        "reason": reason,
        "confidence": 0.9,
    }


def arithmetic_policy(request: RoutingRequest) -> dict[str, Any]:
    ext = request.state_summary.extensions
    value, target = float(ext["value"]), float(ext["target"])
    last = request.previous_actions[-1] if request.previous_actions else None
    confirmed = (
        last is not None
        and last.capability_id == "arith.compare"
        and bool(last.success)
        and isinstance(last.result, dict)
        and bool(last.result.get("equal"))
    )
    if value == target:
        return _finish("target confirmed") if confirmed else _invoke("arith.compare", {}, "confirm")
    if 0 < value and value * 2 <= target and "arith.multiply" in request.capability_ids:
        return _invoke("arith.multiply", {"factor": 2}, "double")
    return _invoke("arith.add", {"amount": target - value}, "close the gap")


def stats_policy(request: RoutingRequest) -> dict[str, Any]:
    ctx = request.state_summary.context
    datasets = [a["id"] for a in request.state_summary.artifacts if a["kind"] == "dataset"]
    latest = datasets[-1] if datasets else None
    summarized = latest is not None and ctx.get("summary_of") == latest
    width = ctx.get("ci_half_width")
    precise = width is not None and width <= request.goal["success_criteria"]["ci_half_width"]
    reported = summarized and ctx.get("report_for") == ctx.get("summary_of")
    if latest is None:
        return _invoke("stats.sample", {"n": 8, "seed": 1}, "no data yet")
    if not summarized:
        return _invoke("stats.summarize", {}, "summarize new data")
    if not precise:
        n = int(ctx["sample_size"]) * 4
        return _invoke("stats.sample", {"n": n, "seed": request.step}, "interval too wide")
    if not reported:
        return _invoke("stats.report", {}, "write report")
    return _finish("precise estimate reported")


def _arith_state(domain: DomainModule, params: dict[str, Any]) -> WorkflowState:
    assert isinstance(domain, arithmetic.ArithmeticDomain)
    return domain.new_workflow(start=params["start"], target=params["target"])


def _stats_state(domain: DomainModule, params: dict[str, Any]) -> WorkflowState:
    assert isinstance(domain, stats.StatsDomain)
    return domain.new_workflow(**params)


def _pipeline_state(domain: DomainModule, params: dict[str, Any]) -> WorkflowState:
    assert isinstance(domain, pipeline.PipelineDomain)
    return domain.new_workflow(**params)


SUITES: dict[str, Suite] = {
    "arithmetic": Suite(
        "arithmetic",
        arithmetic.ArithmeticDomain,
        _arith_state,
        arithmetic.routing_rules,
        arithmetic_policy,
    ),
    "stats": Suite("stats", stats.StatsDomain, _stats_state, stats.routing_rules, stats_policy),
    "pipeline": Suite(
        "pipeline",
        pipeline.PipelineDomain,
        _pipeline_state,
        pipeline.routing_rules,
        pipeline.request_policy,
    ),
}
