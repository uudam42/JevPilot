"""Optional LIVE end-to-end run: Claude interprets the request, Jev routes the workflow.

Skipped unless ``JEVPILOT_LIVE_TESTS=1`` and both keys are set:

    export JEVPILOT_LIVE_TESTS=1 ANTHROPIC_API_KEY=... TYPESAFE_API_KEY=...
    pytest tests/apps/test_uav_live.py -s

It makes real, billed model calls. What it checks is that the product works
end to end with live services and that the report stays grounded; the
routing choices themselves are the live models' and are recorded, not graded.
"""

from __future__ import annotations

import os

import pytest

from apps.uav_materials import RunMode, run_uav_material_workflow
from apps.uav_materials.workflow import live_requirements
from tests.apps.test_uav_end_to_end import untraced_numbers

LIVE = os.environ.get("JEVPILOT_LIVE_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not LIVE or bool(live_requirements()),
    reason="live tests disabled (set JEVPILOT_LIVE_TESTS=1 and both API keys)",
)


def test_live_request_to_report() -> None:
    result = run_uav_material_workflow(mode=RunMode.LIVE)
    assert result.mode is RunMode.LIVE and result.report.run.mode == "LIVE"
    assert result.report.run.interpreter.get("kind") == "llm"
    assert result.report.run.router.get("adapter", {}).get("provider") == "typesafe"
    assert untraced_numbers(result) == []  # no number the state does not contain
    print(result.markdown)
