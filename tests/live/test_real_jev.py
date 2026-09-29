"""Opt-in tests against the REAL TypeSafe Jev API (``pytest -m live_jev``).

Deselected by default (``addopts``) and skipped without ``TYPESAFE_API_KEY``. The key
is read by the SDK from the environment and never printed. Each test makes a few
real calls; answers are checked for validity, not for being the "right" choice,
except where the workflow's preconditions leave a single sensible option.
"""

from __future__ import annotations

import json
import os

import pytest

from jevpilot import JevRouter, validate_decision
from tests.integrations.live_problem import OFFERED, smoke_problem

pytestmark = [
    pytest.mark.live_jev,
    pytest.mark.skipif(
        not os.environ.get("TYPESAFE_API_KEY", "").strip(), reason="TYPESAFE_API_KEY: missing"
    ),
]


def test_authentication_and_model_discovery() -> None:
    from integrations.typesafe_jev import TypeSafeJevAdapter

    checked = TypeSafeJevAdapter().verify()
    assert checked["authentication"] == "ok"
    assert checked["requested_model"] in checked["listed_models"]


def test_one_real_routing_decision_is_valid_and_attributed() -> None:
    from integrations.typesafe_jev import TypeSafeJevAdapter

    state, specs = smoke_problem()
    adapter = TypeSafeJevAdapter()
    outcome = JevRouter(adapter, timeout_s=60).route(state, specs)
    validate_decision(outcome.decision, specs)
    d = outcome.decision
    assert d.capability_id is None or d.capability_id in OFFERED
    (attempt,) = outcome.attempts
    assert attempt.success and attempt.model and attempt.metadata["provider"] == "typesafe"
    assert d.confidence is None or 0.0 <= d.confidence <= 1.0
    probabilities = attempt.metadata["next_action_probabilities"]
    assert probabilities and all(0.0 <= p <= 1.0 for p in probabilities.values())
    assert set(probabilities) <= {*OFFERED, "__finish__", "__ask_human__"}
    assert abs(sum(probabilities.values()) - 1.0) < 0.05  # a distribution over the labels
    assert os.environ["TYPESAFE_API_KEY"] not in json.dumps(attempt.model_dump(mode="json"))


def test_uav_workflow_routed_by_real_jev_is_labelled_and_bounded() -> None:
    from apps.uav_materials import DEMO_REQUEST, RunMode, run_uav_material_workflow

    result = run_uav_material_workflow(DEMO_REQUEST, mode=RunMode.LIVE_ROUTING, max_steps=12)
    assert result.mode is RunMode.LIVE_ROUTING
    assert result.report.run.mode == "LIVE_ROUTING" and "not fully live" in result.report.run.note
    log = result.routing_log()
    assert 1 <= len(log) <= 12
    assert all(r["router"] == "jev_router[live]" for r in log if r["router"] != "controller")
    assert all(r["status"] != "fallback" for r in log)
    assert os.environ["TYPESAFE_API_KEY"] not in result.markdown + json.dumps(log)
