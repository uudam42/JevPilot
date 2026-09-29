"""The real-Jev validation harness, checked offline (no network, no key)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.uav_materials import DEMO_REQUEST, run_uav_material_workflow
from experiments.routing.generalization import dataset as ds
from experiments.routing.generalization.world import load_catalog
from experiments.routing.real_jev import UAV_CASES, _refuse_secrets, publish


@pytest.mark.parametrize("name", sorted(UAV_CASES))
def test_expected_uav_branches_are_the_offline_reference(name: str) -> None:
    case = UAV_CASES[name]
    result = run_uav_material_workflow(
        case["request"] or DEMO_REQUEST, llm_backend=case.get("llm_backend", "native")
    )
    assert result.decision == case["expected_decision"]
    invoked = {r["capability"] for r in result.routing_log()}
    assert ("uavm.design_composite_candidate" in invoked) is case["design_must_run"]


def test_frozen_live_sample_exists_in_the_eval_split() -> None:
    sample = ds.load_sample("jev_live_v1")
    ids = {c.case_id for c in ds.decision_cases("eval", load_catalog())}
    assert sample["split"] == "eval" and 20 <= len(sample["decision_cases"]) <= 40
    assert set(sample["decision_cases"]) <= ids


def test_publisher_refuses_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "tsk-" + "q" * 30
    monkeypatch.setenv("TYPESAFE_API_KEY", secret)
    with pytest.raises(SystemExit):
        _refuse_secrets(json.dumps({"x": f"value {secret}"}))
    with pytest.raises(SystemExit):
        _refuse_secrets(json.dumps({"Authorization": "Bearer abcdefghijklmnop"}))
    _refuse_secrets(json.dumps({"metric": 0.5}))


def test_publish_from_an_offline_run_labels_the_fake(tmp_path: Path) -> None:
    from experiments.routing.benchmark import main

    args = "--mode offline --routers rule,jev --sample tiny_live_v1 --experiments main"
    main([*args.split(), "--out", str(tmp_path / "runs"), "--quiet", "--no-traces"])
    (run,) = (tmp_path / "runs").iterdir()
    artifact = publish({"fake": run}, None, tmp_path / "out.json")
    assert set(artifact["stages"]["fake"]["routers"]) == {"FAKE_JEV", "RULE_ROUTER"}
    assert "REAL_JEV" not in artifact["stages"]["fake"]["routers"]


def test_fully_live_checks_on_an_offline_result() -> None:
    """The FULLY_LIVE record builder, exercised offline (no model is called)."""
    from experiments.routing.real_jev import _fully_live_record

    case = UAV_CASES["B_inverse_design"]
    result = run_uav_material_workflow(DEMO_REQUEST)
    record = _fully_live_record("B_inverse_design", DEMO_REQUEST, result, case["expected_decision"])
    assert record["label"] == "FULLY_LIVE" and record["decision_matches_offline_reference"]
    assert record["preconditions_respected"] and record["design_ran_only_after_design_decision"]
    assert record["untraced_report_numbers"] == []
    assert record["science_equals_offline_replay_of_same_interpretation"] is True
    assert record["interpretation"]["all_numbers_quoted_from_request"] is True
    json.dumps(record)
