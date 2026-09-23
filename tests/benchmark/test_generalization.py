"""Generalization benchmark: world semantics, oracle ground truth, splits, perturbations, CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.routing import benchmark
from experiments.routing.generalization import dataset as ds
from experiments.routing.generalization.baselines import toolkit_rules
from experiments.routing.generalization.oracle import expect
from experiments.routing.generalization.routers import (
    FakeAdapterInLiveModeError,
    LiveOptions,
    assert_not_fake,
    build_slot,
)
from experiments.routing.generalization.runner import RunConfig, run
from experiments.routing.generalization.world import load_catalog
from jevpilot import JevRouter, RuleRouter, ScriptedRouter
from jevpilot.adapters import FakeJevAdapter

CAT = load_catalog()
VOLATILE = {
    "routing_latency_ms",
    "total_latency_ms",
    "routing_latencies_ms",
    "execution_latencies_ms",
    "execution_latency_ms",
    "trace_file",
}


def spec(case_id: str, split: str = "eval") -> ds.WorkflowSpec:
    return next(s for s in ds.load_split(split) if s.case_id == case_id)


def test_catalog_and_split_sizes() -> None:
    assert len(CAT.capabilities) == 22 and len(CAT.distractors) == 36
    sizes = {s: len(ds.load_split(s)) for s in ds.SPLITS}
    assert sizes == {"dev": 23, "validation": 9, "eval": 29}
    ids = [c.case_id for s in ds.SPLITS for c in ds.load_split(s)]
    assert len(ids) == len(set(ids))


def test_descriptions_never_name_capabilities() -> None:
    """Needed for the opaque-name test: semantics must survive without names."""
    names = {d.name for d in CAT.all.values()} | set(CAT.all)
    for d in CAT.all.values():
        text = " ".join([d.description, *d.inputs.keys()])
        assert not any(n in text for n in names), d.id


def test_unseen_compositions_satisfy_the_definition() -> None:
    assert ds.unseen_violations(CAT) == []
    unseen = [s for s in ds.load_split("eval") if s.unseen_composition]
    assert len(unseen) >= 8


def test_every_level_and_category_is_covered_in_eval() -> None:
    specs = ds.load_split("eval")
    assert {s.level for s in specs} == {1, 2, 3, 4, 5}
    cats = {c for s in specs for c in s.categories}
    assert {
        "state_dependent",
        "failure_recovery",
        "missing_information",
        "finish",
        "impossible",
        "distractors",
        "similar_capabilities",
        "multiple_valid",
        "forbidden_action",
        "paraphrased_goal",
    } <= cats


def test_oracle_accepts_every_optimal_order() -> None:
    p = ds.problem(spec("eval_u_trend_outliers"), CAT, list(CAT.capabilities))
    e = expect(frozenset({"dataset_loaded"}), p)
    assert e.kind == "invoke" and {"tk.clean_records", "tk.normalize_units"} <= e.acceptable
    assert "tk.quick_summary" in e.unnecessary and "tk.publish_public" not in e.acceptable


def test_oracle_is_state_dependent() -> None:
    p = ds.problem(spec("eval_l2_outliers_fresh"), CAT, list(CAT.capabilities))
    fresh = expect(frozenset(), p)
    loaded = expect(frozenset({"dataset_loaded"}), p).acceptable
    validated = expect(frozenset({"dataset_loaded", "schema_validated"}), p)
    # the mirror is equally short but costlier: acceptable, not preferred
    assert fresh.acceptable == {"tk.load_dataset", "tk.load_dataset_mirror"}
    assert fresh.preferred == {"tk.load_dataset"} and loaded == {"tk.clean_records"}
    assert validated.acceptable == {"tk.clean_records", "tk.drop_invalid_rows"}
    assert validated.preferred == {"tk.drop_invalid_rows"}  # cheaper


def test_oracle_finish_ask_impossible_and_failures() -> None:
    all_params = frozenset({"source", "reference", "group_field"})
    done = spec("eval_fin_outliers_export_done")
    assert (
        expect(frozenset(done.initial_facts), ds.problem(done, CAT, list(CAT.capabilities))).kind
        == "finish"
    )
    missing = spec("eval_mi_no_source_export")
    e = expect(frozenset(), ds.problem(missing, CAT, list(CAT.capabilities)), all_params=all_params)
    assert e.kind == "ask_human" and e.no_action_intents == {"ask_human", "idle"}
    imp = spec("eval_imp_forecast")
    assert (
        expect(
            frozenset(), ds.problem(imp, CAT, list(CAT.capabilities)), all_params=all_params
        ).kind
        == "impossible"
    )
    trap = spec("eval_imp_share_unavailable")
    e = expect(frozenset(), ds.problem(trap, CAT, ds.offered_ids(trap, CAT)), all_params=all_params)
    assert e.kind == "impossible" and "tk.publish_public" in e.forbidden
    p = ds.problem(spec("eval_u4_trend_outliers_mirror"), CAT, list(CAT.capabilities))
    after = expect(frozenset(), p, excluded=frozenset({"tk.load_dataset"}))
    assert after.acceptable == {"tk.load_dataset_mirror"}


def test_success_is_defined_by_state_not_by_sequence() -> None:
    s = spec("eval_u_trend_outliers")
    src = s.parameters["source"]
    orders = [
        [
            "tk.load_dataset",
            "tk.clean_records",
            "tk.normalize_units",
            "tk.fit_trend",
            "tk.detect_outliers",
        ],
        [
            "tk.load_dataset",
            "tk.normalize_units",
            "tk.clean_records",
            "tk.detect_outliers",
            "tk.fit_trend",
        ],
    ]
    for order in orders:
        script = [(c, {"source": src} if c == "tk.load_dataset" else {}) for c in order]
        result = ds.build_controller(s, CAT, ScriptedRouter(script), ds.offered_ids(s, CAT)).run(
            ds.start_state(s, "eval")
        )
        assert result.succeeded, order


def test_decision_cases_include_real_failure_states() -> None:
    cases = [
        c
        for c in ds.decision_cases("eval", CAT)
        if c.workflow.case_id == "eval_u4_trend_outliers_mirror"
    ]
    after_failure = [
        c for c in cases if c.state.observations and not c.state.observations[-1].success
    ]
    assert after_failure
    assert after_failure[0].expectation.acceptable == {"tk.load_dataset_mirror"}


def test_perturbations_preserve_semantics() -> None:
    case = next(c for c in ds.decision_cases("eval", CAT) if c.state.history)
    perm = ds.permuted(case, CAT, 1, seed=0)
    assert sorted(perm.offered) == sorted(case.offered) and perm.offered != case.offered
    named = ds.renamed(case, CAT, seed=0)
    assert all(s.id.startswith("cap_") and s.name == s.id for s in named.specs)
    assert {named.to_canonical[s.id] for s in named.specs} == set(case.offered)
    shown = {h.decision.capability_id for h in named.state.history}
    assert all(c is None or c.startswith("cap_") for c in shown)
    descriptions = {s.description for s in named.specs}
    assert descriptions == {CAT.get(c).description for c in case.offered}
    scaled = ds.with_distractors(case, CAT, 32, seed=0)
    assert scaled is not None and len(scaled.specs) == 32


def test_rule_baseline_solves_dev_but_not_paraphrased_eval() -> None:
    def tcr(split: str) -> float:
        specs = ds.load_split(split)
        ok = 0
        for s in specs:
            r = ds.build_controller(
                s, CAT, RuleRouter(toolkit_rules()), ds.offered_ids(s, CAT)
            ).run(ds.start_state(s, split))
            ok += str(r.state.status) == s.expected_outcome
        return ok / len(specs)

    assert tcr("dev") == 1.0
    assert tcr("eval") < 1.0  # the eval split is not encoded in the rules


def test_offline_runs_are_reproducible() -> None:
    slots = [build_slot(n, "offline", faults={"malformed": 0.1}) for n in ("rule", "jev")]
    cfg = RunConfig(split="validation", experiments=("main", "order"), repetitions=2, seed=5)
    a, b = run(slots, CAT, cfg), run(slots, CAT, cfg)

    def stable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k not in VOLATILE} for r in rows]

    assert stable(a.decisions) == stable(b.decisions)
    assert stable(a.workflows) == stable(b.workflows)
    assert any(r["error_type"] for r in a.decisions)


def test_live_mode_never_uses_fakes_and_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    jev = build_slot("jev", "live", live=LiveOptions())
    assert not jev.available and jev.unavailable and jev.factory is None
    with pytest.raises(FakeAdapterInLiveModeError):
        assert_not_fake(JevRouter(FakeJevAdapter([])))
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--mode", "live", "--faults"])


def test_cli_writes_a_complete_manifest(tmp_path: Path) -> None:
    code = benchmark.main(
        [
            "--mode",
            "offline",
            "--routers",
            "rule,jev",
            "--split",
            "validation",
            "--experiments",
            "main,names",
            "--repetitions",
            "2",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    m = json.loads((run_dir / "manifest.json").read_text())
    for key in (
        "run_id",
        "timestamp",
        "git",
        "jevpilot_version",
        "benchmark_version",
        "split",
        "split_digest",
        "routers",
        "prompt_version",
        "seeds",
        "repetitions",
        "strict_single_model",
        "environment",
        "cases",
    ):
        assert key in m, key
    assert m["mode"] == "offline" and "INFRASTRUCTURE" in m["evaluation_type"]
    assert m["routers"]["jev"]["config"]["adapter"]["provider"] == "fake"
    decisions = [json.loads(line) for line in (run_dir / "decisions.jsonl").open()]
    fps = {json.loads(line)["fingerprint"] for line in (run_dir / "requests.jsonl").open()}
    assert {d["request_fingerprint"] for d in decisions} <= fps
    assert {d["repetition"] for d in decisions} == {0, 1}
    assert (run_dir / "traces").is_dir() and (run_dir / "summary.csv").exists()
