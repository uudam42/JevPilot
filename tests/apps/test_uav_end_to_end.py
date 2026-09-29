"""End-to-end UAV materials workflow: one request in, one report out (offline, deterministic)."""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path
from typing import Any

import pytest

from apps.cli import main as jevpilot_main
from apps.uav_materials import (
    DEMO_REQUEST,
    LiveModeUnavailable,
    RunMode,
    UAVWorkflowResult,
    execute_workflow,
    run_uav_material_workflow,
)
from apps.uav_materials.cli import main as uav_main
from apps.uav_materials.llm_interpreter import LLMRequirementInterpreter
from apps.uav_materials.workflow import offline_components
from domains.uav_materials.datasets import load_real_materials
from domains.uav_materials.profile import PropertyRequirement
from domains.uav_materials.reporting import MaterialWorkflowReport, numbers_in_markdown
from domains.uav_materials.selection import select
from domains.uav_materials.workflow import (
    DESIGN_ASSESSMENT,
    DESIGN_SEARCH,
    EXISTING_SEARCH,
    WORKFLOW_ORDER,
)
from jevpilot import TraceEventType
from jevpilot.routing import LLMCompletion, LLMPrompt

EXISTING_FITS = (
    "I need a structural material for a UAV spar. The density must not exceed 3000 kg/m^3. "
    "It needs a stiffness of at least 65 GPa in both the x and y directions and an in-plane "
    "shear stiffness of at least 25 GPa. Prefer low density and high stiffness in the x and y "
    "directions. If no existing material is suitable, design a composite candidate."
)
VAGUE = (
    "I need a lightweight UAV structural material with high stiffness, low density, and good "
    "load-carrying capability. Find the closest existing material. If no existing material is "
    "sufficiently suitable, design a composite candidate and explain the result."
)
TEMPERATURE_ONLY = (
    "The maximum service temperature must be at least 80 °C. If no existing material is "
    "suitable, design a composite candidate."
)
TOO_STIFF = (
    "I need a panel with a stiffness of at least 200 GPa in both the x and y directions and "
    "the density must not exceed 1800 kg/m^3. Design a composite candidate if nothing "
    "existing fits."
)


@pytest.fixture(scope="module")
def demo() -> UAVWorkflowResult:
    return run_uav_material_workflow()


def steps(result: UAVWorkflowResult) -> list[str]:
    return [s.capability_id for s in result.report.run.steps if s.capability_id]


# -- the design branch, end to end ----------------------------------------------------------------


def test_request_to_report_through_the_design_branch(demo: UAVWorkflowResult) -> None:
    assert demo.mode is RunMode.OFFLINE_DEMO and demo.complete and demo.run.succeeded
    # router → capabilities in the order their preconditions allow
    assert steps(demo) == list(WORKFLOW_ORDER)
    report = demo.report
    assert report.interpretation is not None and len(report.interpretation.requirements) == 8
    assert report.existing is not None and report.existing.evaluated == 46
    assert report.decision is not None and report.decision.outcome == "design"
    designed = report.designed
    assert designed is not None and designed.status == "designed"
    assert (designed.layup, designed.fiber_volume_fraction, designed.plies) == (
        "[0/45/-45/90]s", 0.65, 8,
    )  # fmt: skip
    assert designed.design_target_feasibility == "feasible"
    assert report.comparison is not None and report.comparison.existing_name
    for section in ("## User Goal", "## Interpreted Engineering Requirements",
                    "## Existing Material Search", "## Decision", "## Designed Composite Candidate",
                    "## Existing vs Designed", "## Evidence / Confidence", "## Limitations",
                    "## Sources"):  # fmt: skip
        assert section in demo.markdown, section
    assert (
        "Mode: OFFLINE_DEMO" in demo.markdown and "not an experimentally validated" in demo.markdown
    )


def test_routing_went_through_the_jev_router_pipeline(demo: UAVWorkflowResult) -> None:
    decisions = [e for e in demo.run.trace if e.type is TraceEventType.ROUTING_DECISION]
    assert len(decisions) == len(WORKFLOW_ORDER)
    for event in decisions:
        attempt = event.payload["attempts"][0]
        assert attempt["router_type"] == "JevRouter" and attempt["success"]
        assert attempt["model"] == "offline-demo-reference-policy"  # labelled: not a model
        assert attempt["request"] is not None  # the canonical routing request was built
    router = demo.report.run.router
    assert router["router_id"] == "jev_router[offline_demo]"
    assert router["adapter"]["provider"] == "fake"


def test_designed_versus_existing_keeps_evidence_types_apart(demo: UAVWorkflowResult) -> None:
    rows = {r.requirement: r for r in demo.report.comparison.rows}  # type: ignore[union-attr]
    density = rows["density <= 1800 kg/m^3"]
    assert density.existing and density.existing.evidence_type == "datasheet"
    assert density.existing.check == "violated"
    assert density.designed and density.designed.evidence_type == "predicted"
    assert density.designed.check == "satisfied"
    strength = rows["tensile_ultimate_strength maximize"]
    assert strength.designed and strength.designed.gap == "unsupported"
    counts = {e.subject.split(":")[0]: e.counts for e in demo.report.evidence}
    assert counts["designed composite candidate"]["predicted"] == 7
    assert counts["designed composite candidate"]["unsupported"] == 1
    assert counts["existing"]["datasheet"] == 8 and counts["existing"]["predicted"] == 0


# -- report integrity -----------------------------------------------------------------------------

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")


def report_numbers(report: MaterialWorkflowReport) -> list[float]:
    out: list[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, int | float):
            out.append(float(x))
        elif isinstance(x, str):
            out.extend(float(t) for t in _NUM.findall(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list | tuple):
            for v in x:
                walk(v)

    walk(report.model_dump(mode="json"))
    return out


def untraced_numbers(result: UAVWorkflowResult) -> list[str]:
    known = report_numbers(result.report)
    missing = []
    for token in numbers_in_markdown(result.markdown):
        pct = token.endswith("%")
        text = token.rstrip("%")
        value = float(text) / (100 if pct else 1)
        decimals = len(text.split(".")[1]) if "." in text else 0
        tol = 0.5 * 10**-decimals / (100 if pct else 1) + 1e-12
        if not any(abs(n - value) <= tol for n in known):
            missing.append(token)
    return missing


@pytest.mark.parametrize("request_text", [DEMO_REQUEST, EXISTING_FITS, VAGUE, TOO_STIFF])
def test_every_number_in_the_markdown_is_in_the_structured_report(request_text: str) -> None:
    result = run_uav_material_workflow(request_text)
    assert untraced_numbers(result) == []


def test_structured_report_values_come_from_state(demo: UAVWorkflowResult) -> None:
    state = demo.state
    design = state.latest_artifact(DESIGN_SEARCH)
    assessment = state.latest_artifact(DESIGN_ASSESSMENT)
    search = state.latest_artifact(EXISTING_SEARCH)
    assert design is not None and assessment is not None and search is not None
    best = design.content.best
    predicted = {p.property: p.value for p in demo.report.designed.predictions}  # type: ignore[union-attr]
    for prop, value in predicted.items():
        assert value == best.material.measurements(prop)[0].value  # PredictedPropertyProfile
    assert [r.model_dump() for r in demo.report.designed.rows] == [  # type: ignore[union-attr]
        r.model_dump() for r in design.content.rows
    ]
    reference = assessment.content.reference_existing
    designed = assessment.content.designed
    for row, ref_ev, des_ev in zip(
        demo.report.comparison.rows,  # type: ignore[union-attr]
        reference.evidence.requirements,
        designed.evidence.requirements,
        strict=True,
    ):
        assert row.existing is not None and row.designed is not None
        assert row.existing.value == ref_ev.numeric_value  # CandidateEvaluation
        assert row.designed.value == des_ev.numeric_value
    # the density shown for the reference existing material is its MaterialRecord value
    record = next(m for m in load_real_materials() if m.material_id == reference.candidate_id)
    density = select(
        record,
        PropertyRequirement(
            property="density", operator="<=", value=1800, unit="kg/m^3", priority="hard"
        ),  # fmt: skip
    ).chosen
    assert density is not None
    shown = next(
        r.existing.value
        for r in demo.report.comparison.rows  # type: ignore[union-attr]
        if r.requirement == "density <= 1800 kg/m^3" and r.existing
    )
    assert shown == pytest.approx(density.value_in("kg/m^3"))
    near = {n.requirement: n for n in demo.report.existing.near_misses}  # type: ignore[union-attr]
    beryllium = next(m for m in load_real_materials() if m.material_id == "mil5j-7.2.1.0-c")
    be_density = select(beryllium, density_req := PropertyRequirement(
        property="density", operator="<=", value=1800, unit="kg/m^3", priority="hard"
    )).chosen  # fmt: skip
    assert density_req and be_density is not None
    assert near["density <= 1800 kg/m^3"].value == pytest.approx(be_density.value_in("kg/m^3"))
    evaluations = {e.candidate_id: e for e in search.content.evaluations}
    for row in demo.report.existing.top:  # type: ignore[union-attr]
        e = evaluations[row.candidate_id]
        assert (row.distance, row.pessimistic_distance, row.coverage) == (
            e.distance, e.pessimistic_distance, e.coverage,
        )  # fmt: skip


# -- branches -------------------------------------------------------------------------------------


def test_existing_material_found_skips_inverse_design() -> None:
    result = run_uav_material_workflow(EXISTING_FITS)
    assert result.decision == "use_existing" and result.complete
    assert "uavm.design_composite_candidate" not in steps(result)
    assert result.report.designed is None and result.report.decision is not None
    assert result.report.decision.selected
    assert result.state.latest_artifact(DESIGN_SEARCH) is None


def test_missing_requirement_asks_for_information() -> None:
    result = run_uav_material_workflow(VAGUE)
    assert result.decision == "needs_information" and result.complete
    assert "uavm.design_composite_candidate" not in steps(result)
    interp = result.report.interpretation
    assert interp is not None and any(
        "No numeric acceptance limit" in m for m in interp.missing_information
    )
    assert "More information is needed" in result.report.recommendation


def test_insufficient_existing_evidence_and_unsupported_target_property() -> None:
    result = run_uav_material_workflow(TEMPERATURE_ONLY)
    decision = result.report.decision
    assert decision is not None and decision.outcome == "no_suitable_existing"
    assert decision.counts["evidence_gap"] == 46 and decision.counts["engineering_failure"] == 0
    assert any("evidence gaps, not failures" in r for r in decision.reasons)
    assert decision.design_unsupported and "unsupported" in decision.design_unsupported[0]
    assert result.report.designed is None and result.complete


def test_no_valid_designed_candidate_is_reported_not_recommended() -> None:
    result = run_uav_material_workflow(TOO_STIFF)
    designed = result.report.designed
    assert designed is not None and designed.status == "no_feasible_design"
    assert designed.design_target_feasibility == "infeasible"
    assert "does not satisfy every hard requirement" in result.report.recommendation
    assert result.complete


def test_uninterpretable_request_still_gets_a_report() -> None:
    result = run_uav_material_workflow("Hello, what can you do?")
    assert result.complete and steps(result) == [WORKFLOW_ORDER[0], WORKFLOW_ORDER[-1]]
    assert "No engineering requirement could be interpreted" in result.report.recommendation


class _Scripted:
    """A text model that returns a fixed answer (no network)."""

    def __init__(self, text: str | Exception) -> None:
        self.text = text
        self.prompts: list[LLMPrompt] = []

    def complete(self, prompt: LLMPrompt, *, timeout_s: float | None = None) -> LLMCompletion:
        self.prompts.append(prompt)
        if isinstance(self.text, Exception):
            raise self.text
        return LLMCompletion(text=self.text, model="scripted-test-model")

    def describe(self) -> dict[str, Any]:
        return {"provider": "test", "model": "scripted-test-model"}


def _run_with_llm(text: str | Exception) -> UAVWorkflowResult:
    _, router = offline_components()
    return execute_workflow(
        DEMO_REQUEST,
        interpreter=LLMRequirementInterpreter(_Scripted(text)),
        router=router,
        mode=RunMode.OFFLINE_DEMO,
        note="test: scripted language-model output; no network",
    )


def test_llm_interpretation_is_validated_before_use() -> None:
    answer = {
        "requirements": [
            {"aspect": "density", "operator": "<=", "value": 1800, "unit": "kg/m^3",
             "priority": "hard", "source_text": "The density must not exceed 1800 kg/m^3."},
            {"aspect": "normal_stiffness", "direction": "laminate_x", "operator": ">=",
             "value": 45, "unit": "GPa", "priority": "hard",
             "source_text": "stiffness of at least 40 GPa in both the x and y directions"},
        ],
        "design_allowed": True,
    }  # fmt: skip
    result = _run_with_llm("```json\n" + json.dumps(answer) + "\n```")
    interp = result.report.interpretation
    assert interp is not None and [r.label for r in interp.requirements] == [
        "density <= 1800 kg/m^3"
    ]
    assert interp.integrity_findings and "45 is not written" in interp.integrity_findings[0]
    assert "Integrity findings" in result.markdown and result.complete


def test_interpreter_failure_is_explained_not_raised() -> None:
    result = _run_with_llm(TimeoutError("model did not answer"))
    interp = result.report.interpretation
    assert (
        interp is not None and interp.status == "failed" and "TimeoutError" in (interp.error or "")
    )
    assert result.complete and "No engineering requirement could be interpreted" in (
        result.report.recommendation
    )


# -- live mode, CLI, determinism, offline ---------------------------------------------------------


def test_live_mode_without_credentials_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(LiveModeUnavailable, match="ANTHROPIC_API_KEY"):
        run_uav_material_workflow(mode=RunMode.LIVE)
    assert uav_main(["--live"]) == 2
    captured = capsys.readouterr()
    assert "LIVE mode needs" in captured.err and captured.out == ""  # no demo fallback


def test_cli_writes_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    md, js = tmp_path / "report.md", tmp_path / "report.json"
    assert uav_main(["--demo", "--output", str(md), "--json", str(js), "--quiet"]) == 0
    assert md.read_text(encoding="utf-8").startswith("# JevPilot UAV Materials Report")
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["decision"]["outcome"] == "design" and data["run"]["mode"] == "OFFLINE_DEMO"
    assert "[OFFLINE_DEMO] decision: design" in capsys.readouterr().err


def test_top_level_command_dispatches(capsys: pytest.CaptureFixture[str]) -> None:
    assert jevpilot_main(["uav-materials", "--demo", "--quiet"]) == 0
    assert jevpilot_main(["--help"]) == 0 and "uav-materials" in capsys.readouterr().out
    assert jevpilot_main(["no-such-command"]) == 2


def test_runs_are_deterministic(demo: UAVWorkflowResult) -> None:
    assert run_uav_material_workflow().markdown == demo.markdown


def test_the_demo_needs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    assert run_uav_material_workflow().complete


def test_live_components_are_wired_to_the_real_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """With keys present, LIVE uses Claude for interpretation and Jev for routing.

    Only the wiring is checked: the adapters are constructed with dummy keys and no
    network access; nothing is sent.
    """
    import importlib.util

    from apps.uav_materials.workflow import live_components

    if (
        importlib.util.find_spec("anthropic") is None
        or importlib.util.find_spec("typesafe_sdk") is None
    ):
        pytest.skip("provider SDKs not installed")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-wiring-test")
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-key-for-wiring-test")

    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    from integrations.typesafe_jev import TypeSafeJevAdapter

    # The Jev preflight (authentication + model discovery) is a network call: stubbed here.
    monkeypatch.setattr(TypeSafeJevAdapter, "verify", lambda self: {"authentication": "ok"})
    interpreter, router = live_components()
    assert isinstance(interpreter, LLMRequirementInterpreter)
    assert interpreter.describe()["adapter"]["provider"] == "anthropic"
    config = router.config()
    assert config["type"] == "JevRouter" and config["adapter"]["provider"] == "typesafe"


def test_failed_jev_preflight_stops_before_anything_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from apps.uav_materials.workflow import live_components
    from jevpilot.exceptions import RouterAdapterError

    if importlib.util.find_spec("typesafe_sdk") is None:
        pytest.skip("TypeSafe SDK not installed")
    from integrations.typesafe_jev import TypeSafeJevAdapter

    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-key-for-wiring-test")

    def reject(self: Any) -> Any:
        raise RouterAdapterError("Jev models.list failed (authentication, HTTP 401)")

    monkeypatch.setattr(TypeSafeJevAdapter, "verify", reject)
    with pytest.raises(LiveModeUnavailable, match="preflight failed.*Nothing was run"):
        live_components(routing_only=True)
