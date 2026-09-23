"""The domain runs on the unmodified JevPilot loop; provenance survives state updates."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from domains.uav_materials import UAVMaterialsDomain, UAVMaterialsState
from domains.uav_materials.fixtures import synthetic_candidates
from jevpilot import Controller, Runtime, ScriptedRouter, TraceEventType, WorkflowStatus
from tests.domains.uav_materials.helpers import spar_profile

ROOT = Path(__file__).resolve().parents[3]


def _run(script: list[tuple[str, dict[str, object]]]) -> object:
    rt = Runtime()
    domain = rt.load("uav_materials")  # entry point, like any other domain
    assert isinstance(domain, UAVMaterialsDomain)
    ctl = rt.controller(ScriptedRouter(script), domains=["uav_materials"])
    assert type(ctl) is Controller
    return ctl.run(domain.new_workflow("set up material selection", candidates_registered=True))


def test_structured_requirements_become_the_target_profile() -> None:
    profile = spar_profile()
    cands = synthetic_candidates()
    result = _run(
        [
            ("uavm.construct_target_profile", {"profile": profile.model_dump(mode="json")}),
            (
                "uavm.register_candidate_materials",
                {
                    "candidates": [c.model_dump(mode="json") for c in cands],
                    "source": "synthetic fixtures",
                },
            ),
        ]
    )
    state = result.state  # type: ignore[attr-defined]
    assert result.succeeded and isinstance(state, UAVMaterialsState)  # type: ignore[attr-defined]
    assert state.target_profile == profile
    assert state.target_profile.requirements[-1].weight_source
    # measurement provenance survives JSON → executor → observation → reducer → state
    before = {
        (c.candidate_id, m.property, m.value): m.provenance
        for c in cands
        for m in c.material.all_measurements()
    }
    after = {
        (c.candidate_id, m.property, m.value): m.provenance
        for c in state.candidate_materials
        for m in c.material.all_measurements()
    }
    assert after == before
    assert state.candidate_materials[5].material.provenance_status == "unsourced"
    # the generic provenance ledger also recorded both steps
    assert {p.capability_id for p in state.provenance} == {
        "uavm.construct_target_profile",
        "uavm.register_candidate_materials",
    }
    art = state.latest_artifact("target_material_profile")
    assert art is not None and art.provenance is not None


def test_invalid_structured_requirements_never_execute() -> None:
    bad = spar_profile().model_dump(mode="json")
    bad["requirements"][0]["unit"] = "MPa"  # density in stress units
    result = _run([("uavm.construct_target_profile", {"profile": bad})])
    state = result.state  # type: ignore[attr-defined]
    assert state.status is WorkflowStatus.FAILED and state.target_profile is None
    assert not state.observations
    assert any(e.type is TraceEventType.ROUTING_FAILURE for e in result.trace)  # type: ignore[attr-defined]


def _imports(path: Path) -> list[str]:
    out: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def test_dependency_direction_uav_to_core_only() -> None:
    for path in (ROOT / "jevpilot").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "uav_materials" not in text, path
        assert not any(n.startswith("domains") for n in _imports(path)), path
    allowed = ("jevpilot", "domains.uav_materials", "pydantic", "__future__")
    stdlib = set(sys.stdlib_module_names)
    for path in (ROOT / "domains" / "uav_materials").rglob("*.py"):
        for name in _imports(path):
            root = name.split(".")[0]
            assert name.startswith(allowed) or root in stdlib, (path, name)
            assert not name.startswith(
                (
                    "jevpilot.orchestration",
                    "jevpilot.routing",
                    "jevpilot.adapters",
                    "experiments",
                    "integrations",
                )
            ), name
