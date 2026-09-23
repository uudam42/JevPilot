"""Routing benchmark: fixtures, scoring, metrics, determinism and CLI output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from experiments.routing import benchmark
from experiments.routing.cases import (
    Expected,
    build_capabilities,
    build_state,
    load_decision_fixtures,
    load_workflow_fixtures,
    model_from_json_schema,
)
from experiments.routing.metrics import distribution, rate, usage_totals
from experiments.routing.routers import SIMULATED_ROUTERS
from jevpilot import RoutingDecision, RoutingIntent

VOLATILE = {
    "routing_latency_ms",
    "run_id",
    "total_latency_ms",
    "execution_latency_ms",
    "routing_latencies_ms",
    "execution_latencies_ms",
}


def _stable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in VOLATILE} for r in records]


def test_decision_fixtures_are_well_formed() -> None:
    fx = load_decision_fixtures()
    patterns = {c.pattern for c in fx.cases}
    assert {
        "sequential_dependency",
        "missing_information",
        "capability_preconditions",
        "redundant_capability",
        "invalid_capability",
        "multiple_valid_capabilities",
        "retry_after_failure",
        "replan_after_observation",
        "termination",
    } <= patterns
    for case in fx.cases:
        offered = {s.id for s in build_capabilities(case, fx.capability_catalog)}
        assert set(case.expected.valid_capabilities) <= offered, case.case_id
        assert case.expected.valid_capabilities or case.expected.no_action_intents, case.case_id
        state = build_state(case)
        assert state.step == sum(1 for h in case.state.history if h.success is not None)


def test_workflow_fixtures_cover_all_suites() -> None:
    assert {c.suite for c in load_workflow_fixtures().cases} == {"arithmetic", "stats", "pipeline"}


def test_json_schema_capabilities_validate_like_code_defined_ones() -> None:
    model = model_from_json_schema(
        "X",
        {
            "type": "object",
            "properties": {
                "style": {"type": "string", "enum": ["a", "b"]},
                "n": {"type": "integer", "minimum": 1},
            },
            "required": ["style"],
            "additionalProperties": False,
        },
    )
    model.model_validate({"style": "a", "n": 2})
    for bad in ({"style": "c"}, {"n": 2}, {"style": "a", "n": 0}, {"style": "a", "z": 1}):
        with pytest.raises(ValidationError):
            model.model_validate(bad)


def test_scoring_supports_sets_inputs_and_termination() -> None:
    exp = Expected(
        valid_capabilities=("a", "b"),
        forbidden_capabilities=("x",),
        unnecessary_capabilities=("u",),
        required_inputs={"a": {"k": 1}},
        no_action_intents=("finish",),
    )
    s = benchmark.score
    assert s(RoutingDecision(capability_id="b"), exp)["acceptable"]
    assert s(RoutingDecision(capability_id="a", inputs={"k": 1, "z": 2}), exp)["acceptable"]
    assert not s(RoutingDecision(capability_id="a", inputs={"k": 2}), exp)["acceptable"]
    assert s(RoutingDecision(capability_id="x"), exp)["forbidden"]
    assert s(RoutingDecision(capability_id="u"), exp)["unnecessary"]
    assert s(RoutingDecision.finish("done"), exp)["acceptable"]
    assert not s(RoutingDecision.ask_human("?"), exp)["acceptable"]
    assert not s(None, exp)["acceptable"]
    assert RoutingDecision.no_action("x").intent is RoutingIntent.IDLE


def test_metric_helpers() -> None:
    assert rate(1, 4) == {"value": 0.25, "numerator": 1, "denominator": 4}
    assert rate(0, 0)["value"] is None
    d = distribution([1, 2, 3, 10])
    assert (d["mean"], d["median"], d["min"], d["max"]) == (4.0, 2.5, 1, 10)
    assert distribution([])["mean"] is None
    assert usage_totals([{"input_tokens": None}, {}])["input_tokens"] is None  # unknown ≠ 0
    assert usage_totals([{"input_tokens": 5}, {"input_tokens": None}])["input_tokens"] == 5


def test_rule_baseline_on_decision_fixtures() -> None:
    fx = load_decision_fixtures()
    recs = benchmark.run_decision_benchmark("rule", fx)
    assert all(r["valid"] for r in recs)
    assert all(r["acceptable"] for r in recs), [r["case_id"] for r in recs if not r["acceptable"]]
    assert all(r["input_tokens"] is None for r in recs)


def test_simulated_routers_without_faults_match_the_baseline() -> None:
    fx = load_decision_fixtures()
    for name in ("jev", "llm"):
        recs = benchmark.run_decision_benchmark(name, fx, faults=False)
        assert all(r["acceptable"] for r in recs), name


def test_same_seed_reproduces_results_exactly() -> None:
    a = benchmark.run(SIMULATED_ROUTERS, seed=11, repeats=2)
    b = benchmark.run(SIMULATED_ROUTERS, seed=11, repeats=2)
    assert _stable(a["decisions"]) == _stable(b["decisions"])
    assert _stable(a["workflows"]) == _stable(b["workflows"])
    assert any(r["error_type"] for r in a["decisions"])  # faults actually injected


def test_fallback_chains_recover_every_workflow() -> None:
    wfx = load_workflow_fixtures()
    for name in ("rule", "jev+rule", "llm+rule", "jev+llm+rule"):
        recs = benchmark.run_workflow_benchmark(name, wfx, seed=3, repeats=2)
        assert all(r["completed"] for r in recs), (
            name,
            [r["case_id"] for r in recs if not r["completed"]],
        )


def test_workflow_records_split_latency_and_control_flow() -> None:
    recs = benchmark.run_workflow_benchmark("rule", load_workflow_fixtures())
    by_id = {r["case_id"]: r for r in recs}
    flaky = by_id["pipe_03_flaky_source"]
    assert flaky["retries"] == 1 and flaky["steps"] == 6 and flaky["excess_steps"] == 0
    dirty = by_id["pipe_04_dirty_data"]
    assert dirty["replans"] == 1 and "pipe.clean" in dirty["capability_sequence"]
    missing = by_id["pipe_05_missing_source"]
    assert missing["status"] == "awaiting_human" and missing["completed"]
    for r in recs:
        assert len(r["execution_latencies_ms"]) == r["steps"]
        assert r["routing_decisions"] == len(r["routing_latencies_ms"])


def test_cli_writes_machine_readable_results(tmp_path: Path, capsys: Any) -> None:
    code = benchmark.main(
        ["--router", "rule", "--router", "jev+rule", "--seed", "5", "--out", str(tmp_path)]
    )
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["seed"] == 5 and manifest["real_model_calls"] is False
    assert manifest["routers"]["jev+rule"]["kind"] == "simulated/deterministic"
    assert manifest["framework"]["jevpilot"] and manifest["fixtures"]
    decisions = [json.loads(line) for line in (run_dir / "decisions.jsonl").open()]
    assert {d["router"] for d in decisions} == {"rule", "jev+rule"}
    summary = json.loads((run_dir / "summary.json").read_text())
    assert set(summary["workflow"]) == {"rule", "jev+rule"}
    rows = list(csv.DictReader((run_dir / "summary.csv").open()))
    assert len(rows) == 4
    assert "decision benchmark" in capsys.readouterr().out
