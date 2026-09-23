"""Phase 1 smoke benchmark (pipeline/arithmetic/stats suites): same states, different routers.

    python -m experiments.routing.smoke                      # all simulated routers
    python -m experiments.routing.smoke --router rule --router jev+rule
    python -m experiments.routing.smoke --mode decision --repeats 5 --seed 7
    python -m experiments.routing.smoke --router llm-anthropic   # real API calls

Two modes:

* ``decision``: every decision case is routed once per repeat. The result is
  scored against the case's acceptable/forbidden/unnecessary sets.
* ``workflow``: every workflow case runs end to end on the unmodified
  :class:`~jevpilot.Controller` and the metrics are read from its trace.

Each run writes ``manifest.json``, ``decisions.jsonl``, ``workflows.jsonl``,
``summary.json`` and ``summary.csv`` under ``--out/<run_id>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jevpilot
from experiments.routing.cases import (
    FIXTURE_DIR,
    DecisionFixtureFile,
    Expected,
    WorkflowCase,
    WorkflowFixtureFile,
    build_capabilities,
    build_state,
    fixture_digest,
    load_decision_fixtures,
    load_workflow_fixtures,
)
from experiments.routing.metrics import USAGE_FIELDS, decision_metrics, workflow_metrics
from experiments.routing.routers import (
    REAL_ROUTERS,
    SIMULATED_ROUTERS,
    fault_config,
    make_router,
)
from experiments.routing.suites import SUITES
from jevpilot import (
    DefaultControlPolicy,
    RoutingDecision,
    Runtime,
    TraceEvent,
    TraceEventType,
    stable_digest,
    validate_decision,
)
from jevpilot.exceptions import RoutingError

BENCHMARK_VERSION = "jevpilot.routing-benchmark/1"
DEFAULT_OUT = Path(__file__).parent / "results"
Record = dict[str, Any]


def case_seed(seed: int, case_id: str, repeat: int) -> int:
    """Deterministic per-(case, repeat) seed derived from the run seed."""
    return int(stable_digest([seed, case_id, repeat])[:8], 16)


# -- decision benchmark ---------------------------------------------------------


def run_decision_benchmark(
    router: str,
    fixtures: DecisionFixtureFile,
    *,
    seed: int = 0,
    repeats: int = 1,
    faults: bool = True,
) -> list[Record]:
    if fixtures.suite is None or fixtures.suite not in SUITES:
        raise ValueError(f"decision fixtures must name a known suite, got {fixtures.suite!r}")
    suite = SUITES[fixtures.suite]
    records = []
    for repeat in range(repeats):
        for case in fixtures.cases:
            s = case_seed(seed, case.case_id, repeat)
            instance = make_router(router, suite, seed=s, faults=faults)
            state = build_state(case)
            capabilities = build_capabilities(case, fixtures.capability_catalog)
            decision: RoutingDecision | None = None
            error_type: str | None = None
            t0 = time.perf_counter()
            try:
                outcome = instance.route(state, capabilities)
                validate_decision(outcome.decision, capabilities)  # as the controller does
                decision, attempts = (
                    outcome.decision,
                    [a.model_dump(mode="json") for a in outcome.attempts],
                )
                fallback_used = outcome.fallback_used
            except RoutingError as exc:
                error_type = type(exc).__name__
                attempts = [
                    a.model_dump(mode="json") for a in exc.attempts if hasattr(a, "model_dump")
                ]
                fallback_used = len(attempts) > 1
            latency_ms = (time.perf_counter() - t0) * 1000
            records.append(
                {
                    "benchmark": fixtures.benchmark,
                    "case_id": case.case_id,
                    "pattern": case.pattern,
                    "router": router,
                    "repeat": repeat,
                    "seed": s,
                    "valid": decision is not None,
                    "error_type": error_type,
                    "capability_id": decision.capability_id if decision else None,
                    "intent": str(decision.intent) if decision else None,
                    "inputs": decision.inputs if decision else None,
                    **score(decision, case.expected),
                    "confidence": decision.confidence if decision else None,
                    "routing_latency_ms": latency_ms,
                    "fallback_used": fallback_used,
                    "decided_by": decision.router_id if decision else None,
                    **_attempt_summary(attempts),
                }
            )
    return records


def score(decision: RoutingDecision | None, expected: Expected) -> dict[str, Any]:
    """Compare one decision with the case's expected behaviour."""
    if decision is None:
        return {"acceptable": False, "forbidden": False, "unnecessary": False, "inputs_match": None}
    cid = decision.capability_id
    if cid is None:
        return {
            "acceptable": str(decision.intent) in expected.no_action_intents,
            "forbidden": False,
            "unnecessary": False,
            "inputs_match": None,
        }
    required = expected.required_inputs.get(cid, {})
    inputs_match = all(decision.inputs.get(k) == v for k, v in required.items())
    return {
        "acceptable": cid in expected.valid_capabilities and inputs_match,
        "forbidden": cid in expected.forbidden_capabilities,
        "unnecessary": cid in expected.unnecessary_capabilities,
        "inputs_match": inputs_match,
    }


# -- workflow benchmark ---------------------------------------------------------


def run_workflow_benchmark(
    router: str,
    fixtures: WorkflowFixtureFile,
    *,
    seed: int = 0,
    repeats: int = 1,
    faults: bool = True,
    max_steps: int = 30,
) -> list[Record]:
    records = []
    for repeat in range(repeats):
        for case in fixtures.cases:
            records.append(
                _run_workflow(
                    router, case, repeat=repeat, seed=seed, faults=faults, max_steps=max_steps
                )
            )
    return records


def _run_workflow(
    router: str, case: WorkflowCase, *, repeat: int, seed: int, faults: bool, max_steps: int
) -> Record:
    suite = SUITES[case.suite]
    s = case_seed(seed, case.case_id, repeat)
    runtime = Runtime()
    domain = runtime.load(suite.domain())
    controller = runtime.controller(
        make_router(router, suite, seed=s, faults=faults),
        domains=[domain.name],
        policy=DefaultControlPolicy(max_steps=max_steps),
    )
    initial = suite.new_state(domain, dict(case.params))
    t0 = time.perf_counter()
    result = controller.run(initial)
    total_ms = (time.perf_counter() - t0) * 1000
    t = _trace_summary(result.trace)
    state = result.state
    unnecessary = sum(
        1 for o in state.observations if o.capability_id in case.unnecessary_capabilities
    )
    return {
        "benchmark": "workflow",
        "case_id": case.case_id,
        "suite": case.suite,
        "pattern": case.pattern,
        "router": router,
        "repeat": repeat,
        "seed": s,
        "status": str(state.status),
        "expected_status": case.expected_status,
        "completed": str(state.status) == case.expected_status,
        "steps": state.step,
        "excess_steps": state.step - case.optimal_steps if case.optimal_steps is not None else None,
        "unnecessary_calls": unnecessary,
        "capability_sequence": [o.capability_id for o in state.observations],
        "final_control": str(result.control.action),
        "final_reason": result.control.reason,
        "orchestration_error": result.error.type if result.error else None,
        "total_latency_ms": total_ms,
        **t,
    }


def _trace_summary(trace: Sequence[TraceEvent]) -> Record:
    routing_ms: list[float] = []
    execution_ms: list[float] = []
    attempts: list[Record] = []
    fallbacks = failures = retries = replans = 0
    routing_errors: list[str] = []
    for e in trace:
        p = e.payload
        if e.type is TraceEventType.ROUTING_DECISION:
            if "retry_of" in p["decision"]["metadata"] or "attempts" not in p:
                continue  # re-executions and "nothing available" involve no router call
            routing_ms.append(p["routing_latency_s"] * 1000)
            fallbacks += bool(p.get("fallback_used"))
            attempts += p["attempts"]
        elif e.type is TraceEventType.ROUTING_FAILURE:
            failures += 1
            routing_ms.append(p["routing_latency_s"] * 1000)
            routing_errors.append(p["error"]["type"])
            attempts += p["attempts"]
        elif e.type is TraceEventType.OBSERVATION:
            execution_ms.append(p["observation"]["execution_time"] * 1000)
        elif e.type is TraceEventType.CONTROL_DECISION:
            action = p["control"]["action"]
            retries += action == "retry"
            replans += action == "replan"
    return {
        "routing_decisions": len(routing_ms) - failures,
        "routing_failures": failures,
        "routing_errors": routing_errors,
        "fallback_decisions": fallbacks,
        "retries": retries,
        "replans": replans,
        "routing_latencies_ms": routing_ms,
        "execution_latencies_ms": execution_ms,
        "routing_latency_ms": sum(routing_ms),
        "execution_latency_ms": sum(execution_ms),
        **_attempt_summary(attempts),
    }


def _attempt_summary(attempts: list[Record]) -> Record:
    summary: Record = {
        "attempts": len(attempts),
        "attempt_routers": [a["router_id"] for a in attempts],
        "attempt_errors": [a["error"]["type"] for a in attempts if a.get("error")],
        "models": sorted({a["model"] for a in attempts if a.get("model")}),
    }
    for key in USAGE_FIELDS:
        known = [
            a["usage"][key] for a in attempts if a.get("usage") and a["usage"].get(key) is not None
        ]
        summary[key] = sum(known) if known else None
    return summary


# -- run orchestration ----------------------------------------------------------


def run(
    routers: Sequence[str],
    *,
    mode: str = "all",
    seed: int = 0,
    repeats: int = 1,
    faults: bool = True,
    decision_fixtures: Path | None = None,
    workflow_fixtures: Path | None = None,
) -> dict[str, Any]:
    """Run the benchmark in memory. Returns the manifest, records and summary."""
    dpath = decision_fixtures or FIXTURE_DIR / "decision_cases.json"
    wpath = workflow_fixtures or FIXTURE_DIR / "workflow_cases.json"
    decisions: list[Record] = []
    workflows: list[Record] = []
    summary: dict[str, Any] = {"decision": {}, "workflow": {}}
    dfx = load_decision_fixtures(dpath) if mode in ("all", "decision") else None
    wfx = load_workflow_fixtures(wpath) if mode in ("all", "workflow") else None
    for name in routers:
        if dfx is not None:
            recs = run_decision_benchmark(name, dfx, seed=seed, repeats=repeats, faults=faults)
            decisions += recs
            summary["decision"][name] = decision_metrics(recs)
        if wfx is not None:
            recs = run_workflow_benchmark(name, wfx, seed=seed, repeats=repeats, faults=faults)
            workflows += recs
            summary["workflow"][name] = workflow_metrics(recs)
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{stable_digest([list(routers), mode, seed, repeats])[:6]}"
    for r in decisions + workflows:
        r["run_id"] = run_id
    manifest = {
        "run_id": run_id,
        "benchmark_version": BENCHMARK_VERSION,
        "timestamp": started.isoformat(),
        "mode": mode,
        "seed": seed,
        "repeats": repeats,
        "faults": faults,
        "routers": {name: _router_record(name, faults) for name in routers},
        "real_model_calls": any(n in REAL_ROUTERS for n in routers),
        "fixtures": {
            str(p.name): {"digest": fixture_digest(p)}
            for p, used in ((dpath, dfx), (wpath, wfx))
            if used is not None
        },
        "case_ids": {
            "decision": [c.case_id for c in dfx.cases] if dfx else [],
            "workflow": [c.case_id for c in wfx.cases] if wfx else [],
        },
        "framework": {
            "jevpilot": jevpilot.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    return {
        "manifest": manifest,
        "decisions": decisions,
        "workflows": workflows,
        "summary": summary,
    }


def _router_record(name: str, faults: bool) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for suite in SUITES.values():
        try:
            configs[suite.name] = make_router(name, suite, seed=0, faults=faults).config()
        except Exception as exc:  # e.g. optional SDK missing for a real router
            configs[suite.name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return {
        "kind": "real" if name in REAL_ROUTERS else "simulated/deterministic",
        "faults": fault_config(name, faults),
        "config_by_suite": configs,
    }


def write_results(result: dict[str, Any], out: Path) -> Path:
    directory: Path = out / str(result["manifest"]["run_id"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(result["manifest"], indent=2) + "\n")
    for name in ("decisions", "workflows"):
        with (directory / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for record in result[name]:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
    (directory / "summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    rows = summary_rows(result["summary"])
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["mode"])
        writer.writeheader()
        writer.writerows(rows)
    return directory


def summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for router, m in summary["decision"].items():
        rows.append(
            {
                "mode": "decision",
                "router": router,
                "n": m["decisions"],
                "valid_decision_rate": m["valid_decision_rate"]["value"],
                "routing_accuracy": m["routing_accuracy"]["value"],
                "invalid_capability_rate": m["invalid_capability_rate"]["value"],
                "unnecessary_rate": m["unnecessary_selection_rate"]["value"],
                "fallback_rate": m["fallback_rate"]["value"],
                "task_completion_rate": None,
                "mean_steps": None,
                "routing_ms_mean": m["routing_latency_ms"]["mean"],
                "execution_ms_mean": None,
                "input_tokens": m["usage"]["input_tokens"],
            }
        )
    for router, m in summary["workflow"].items():
        rows.append(
            {
                "mode": "workflow",
                "router": router,
                "n": m["workflows"],
                "valid_decision_rate": None,
                "routing_accuracy": None,
                "invalid_capability_rate": None,
                "unnecessary_rate": None,
                "fallback_rate": m["fallback_rate"]["value"],
                "task_completion_rate": m["task_completion_rate"]["value"],
                "mean_steps": m["steps_to_completion"]["mean"],
                "routing_ms_mean": m["routing_latency_ms_per_decision"]["mean"],
                "execution_ms_mean": m["execution_latency_ms_per_call"]["mean"],
                "input_tokens": m["usage"]["input_tokens"],
            }
        )
    return rows


def format_table(summary: dict[str, Any]) -> str:
    def pct(x: Any) -> str:
        return "   -  " if x is None else f"{100 * x:5.1f}%"

    def num(x: Any, fmt: str = "{:6.3f}") -> str:
        return "   -  " if x is None else fmt.format(x)

    lines = []
    if summary["decision"]:
        lines.append("decision benchmark (single routing step)")
        lines.append(
            f"  {'router':<14} {'n':>4} {'VDR':>7} {'RA':>7} {'ICR*':>7} "
            f"{'unnec':>7} {'fallbk':>7} {'route ms':>9}"
        )
        for r, m in summary["decision"].items():
            lines.append(
                f"  {r:<14} {m['decisions']:>4} {pct(m['valid_decision_rate']['value']):>7} "
                f"{pct(m['routing_accuracy']['value']):>7} "
                f"{pct(m['invalid_capability_rate']['value']):>7} "
                f"{pct(m['unnecessary_selection_rate']['value']):>7} "
                f"{pct(m['fallback_rate']['value']):>7} "
                f"{num(m['routing_latency_ms']['mean']):>9}"
            )
        lines.append("  * ICR is per router attempt, including failed primaries behind a fallback")
    if summary["workflow"]:
        lines.append("workflow benchmark (end to end)")
        lines.append(
            f"  {'router':<14} {'n':>4} {'TCR':>7} {'steps':>6} {'rfail':>7} "
            f"{'fallbk':>7} {'retry':>7} {'route ms':>9} {'exec ms':>8}"
        )
        for r, m in summary["workflow"].items():
            lines.append(
                f"  {r:<14} {m['workflows']:>4} {pct(m['task_completion_rate']['value']):>7} "
                f"{num(m['steps_to_completion']['mean'], '{:6.2f}'):>6} "
                f"{pct(m['routing_failure_rate']['value']):>7} "
                f"{pct(m['fallback_rate']['value']):>7} {pct(m['retry_rate']['value']):>7} "
                f"{num(m['routing_latency_ms_per_decision']['mean']):>9} "
                f"{num(m['execution_latency_ms_per_call']['mean']):>8}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "--router",
        action="append",
        dest="routers",
        help=f"router to benchmark (repeatable); simulated: "
        f"{', '.join(SIMULATED_ROUTERS)}; real: {', '.join(REAL_ROUTERS)}",
    )
    parser.add_argument("--mode", choices=("all", "decision", "workflow"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--no-faults",
        action="store_true",
        help="disable fault injection in simulated model adapters",
    )
    parser.add_argument("--decision-fixtures", type=Path)
    parser.add_argument("--workflow-fixtures", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-write", action="store_true", help="print the summary only")
    args = parser.parse_args(argv)
    routers = args.routers or list(SIMULATED_ROUTERS)
    result = run(
        routers,
        mode=args.mode,
        seed=args.seed,
        repeats=args.repeats,
        faults=not args.no_faults,
        decision_fixtures=args.decision_fixtures,
        workflow_fixtures=args.workflow_fixtures,
    )
    print(format_table(result["summary"]))
    if not args.no_write:
        print(f"\nresults: {write_results(result, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
