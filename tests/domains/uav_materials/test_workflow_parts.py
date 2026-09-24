"""Workflow building blocks: interpretation integrity, decision policy, bounded design search."""

from __future__ import annotations

import json
from typing import Any

import pytest

from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.decision import (
    DecisionOutcome,
    ExistingMaterialAcceptancePolicy,
    decide,
)
from domains.uav_materials.evaluation import evaluate_candidates
from domains.uav_materials.interpretation import (
    InterpretationPayload,
    RuleBasedInterpreter,
    interpretation_system_prompt,
    parse_interpretation,
)
from domains.uav_materials.inverse_design import DesignSearchConfig, design_target, search_designs
from domains.uav_materials.requirements import Aspect, Direction
from domains.uav_materials.workflow import WORKFLOW_ORDER, reference_routing_payload

DEMO = (
    "I need a lightweight structural material for a small UAV wing-skin panel with low "
    "density, high in-plane stiffness in the x and y directions, and good load-carrying "
    "capability. The density must not exceed 1800 kg/m^3. The panel needs an in-plane "
    "stiffness of at least 40 GPa in both the x and y directions and an in-plane shear "
    "stiffness of at least 15 GPa. It will operate near the sea. Find the closest existing "
    "material. If no existing material is sufficiently suitable, design a composite candidate "
    "and explain the result."
)
INTERPRETER = {"name": "test"}


def item(**kw: Any) -> dict[str, Any]:
    base = {
        "aspect": "density",
        "operator": "<=",
        "value": 1800,
        "unit": "kg/m^3",
        "priority": "hard",
        "source_text": "The density must not exceed 1800 kg/m^3.",
    }
    return {**base, **kw}


def parse(*items: dict[str, Any], **payload: Any) -> Any:
    return parse_interpretation({"requirements": list(items), **payload}, DEMO, INTERPRETER)


# -- interpretation -------------------------------------------------------------------------------


def test_rule_based_interpreter_reads_the_demo_request() -> None:
    o = RuleBasedInterpreter().interpret(DEMO)
    assert o.status == "ok" and o.design_allowed and not o.findings
    assert [r.label for r in o.hard] == [
        "density <= 1800 kg/m^3",
        "normal_stiffness[laminate_x] >= 40 GPa",
        "normal_stiffness[laminate_y] >= 40 GPa",
        "shear_stiffness[laminate_xy] >= 15 GPa",
    ]
    assert [r.label for r in o.soft] == [
        "density minimize",
        "normal_stiffness[laminate_x] maximize",
        "normal_stiffness[laminate_y] maximize",
        "tensile_ultimate_strength maximize",
    ]
    assert all(r.weight == 1.0 and r.weight_source for r in o.soft)
    assert [(c.aspect, c.environment) for c in o.concerns] == [
        (Aspect.CORROSION_PENETRATION, "marine_atmosphere")
    ]
    quotes = [r.provenance.sources[0].metadata["quote"] for r in o.requirements if r.provenance]
    assert all(q in DEMO for q in quotes)


def test_a_valid_item_becomes_a_requirement() -> None:
    o = parse(item())
    assert o.status == "ok" and o.hard[0].label == "density <= 1800 kg/m^3"


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"value": 2000}, "not written in the quoted text"),  # invented number
        ({"unit": "g/cm^3"}, "unit 'g/cm^3' is not written"),  # unit changed
        ({"source_text": "density below 1800 kg/m^3"}, "not a verbatim quote"),
        ({"operator": "minimize", "value": 1800, "priority": "soft"}, "carries no number"),
        ({"priority": "soft"}, "a stated limit is a hard requirement"),
        ({"unit": "MPa"}, "not written in the quoted text"),
    ],
)
def test_integrity_rules_reject_items(changes: dict[str, Any], problem: str) -> None:
    o = parse(item(**changes))
    assert not o.requirements
    assert o.findings and problem in o.findings[0].problem


def test_the_schema_has_no_room_for_material_properties() -> None:
    schema = InterpretationPayload.model_json_schema()
    assert set(schema["properties"]) == {
        "requirements", "concerns", "application", "design_allowed", "missing_information",
    }  # fmt: skip
    smuggled = parse_interpretation(
        {"requirements": [item()], "predicted_density_kg_m3": 1550}, DEMO, INTERPRETER
    )
    assert smuggled.status == "failed" and "not a valid interpretation" in (smuggled.error or "")
    extra_field = parse_interpretation(
        {"requirements": [item(material="AS4/3501-6")]}, DEMO, INTERPRETER
    )
    assert extra_field.status == "failed"
    prompt = interpretation_system_prompt()
    assert "never state, estimate or recall any material property" in prompt
    assert json.dumps(schema, sort_keys=True) in prompt


def test_malformed_output_is_a_failed_interpretation_not_a_crash() -> None:
    o = parse_interpretation("I think you want carbon fibre.", DEMO, INTERPRETER)
    assert o.status == "failed" and o.raw_output and not o.requirements


def test_duplicates_are_merged_and_missing_information_is_reported() -> None:
    soft = {"aspect": "density", "operator": "minimize", "priority": "soft",
            "source_text": "low density"}  # fmt: skip
    o = parse(soft, soft)
    assert len(o.requirements) == 1 and o.status == "partial"
    assert any("No numeric acceptance limit" in m for m in o.missing_information)


def test_directionless_stiffness_is_flagged() -> None:
    o = RuleBasedInterpreter().interpret("I need high stiffness and the density must not "
                                         "exceed 2000 kg/m^3.")  # fmt: skip
    stiff = [r for r in o.soft if r.aspect is Aspect.NORMAL_STIFFNESS]
    assert stiff and stiff[0].direction is Direction.UNSPECIFIED
    assert any("No direction was stated" in m for m in o.missing_information)


# -- decision -------------------------------------------------------------------------------------


def evaluations(request: str) -> tuple[Any, Any]:
    target = RuleBasedInterpreter().interpret(request).target()
    assert target is not None
    return evaluate_candidates(target, real_candidates()), target


def test_evidence_gaps_are_never_accepted_or_called_failures() -> None:
    evals, target = evaluations(DEMO)
    d = decide(evals, target, design_allowed=True)
    assert d.outcome is DecisionOutcome.DESIGN and d.counts["accepted"] == 0
    assert d.counts["engineering_failure"] == 42 and d.counts["evidence_gap"] == 4
    gap_verdicts = [v for v in d.verdicts if v.evidence_classification == "evidence_gap"]
    assert gap_verdicts and all(not v.accepted for v in gap_verdicts)
    assert all(not any(r.startswith("violates") for r in v.reasons) for v in gap_verdicts)
    assert any("might qualify if the missing evidence were obtained" in r for r in d.reasons)
    assert d.policy.label.startswith("DEMONSTRATION POLICY")


def test_policy_is_configurable() -> None:
    request = (
        "The density must not exceed 3000 kg/m^3. It needs a stiffness of at least 65 GPa in "
        "both the x and y directions and an in-plane shear stiffness of at least 25 GPa. "
        "Prefer low density and high stiffness in the x and y directions."
    )
    evals, target = evaluations(request)
    accepted = decide(evals, target, design_allowed=True)
    assert accepted.outcome is DecisionOutcome.USE_EXISTING and accepted.selected_candidate_id
    strict = ExistingMaterialAcceptancePolicy(maximum_preference_distance=0.0)
    d = decide(evals, target, design_allowed=True, policy=strict)
    assert d.outcome is DecisionOutcome.USE_EXISTING or d.counts["rejected_by_preference"] > 0
    demanding = ExistingMaterialAcceptancePolicy(minimum_hard_constraints=5)
    assert decide(evals, target, design_allowed=True, policy=demanding).outcome is (
        DecisionOutcome.NEEDS_INFORMATION
    )
    assert decide(evals, target, design_allowed=False, policy=strict).outcome in (
        DecisionOutcome.USE_EXISTING,
        DecisionOutcome.NO_SUITABLE_EXISTING,
    )


# -- bounded inverse design -----------------------------------------------------------------------


def test_design_target_keeps_only_what_the_laminate_model_answers() -> None:
    target = RuleBasedInterpreter().interpret(DEMO).target()
    assert target is not None
    goal, excluded = design_target(target)
    assert goal is not None and len(goal.requirements) == 7
    assert [x.requirement for x in excluded] == ["tensile_ultimate_strength maximize"]
    assert excluded[0].status == "unsupported"


def test_grid_search_is_bounded_deterministic_and_uses_the_models() -> None:
    target = RuleBasedInterpreter().interpret(DEMO).target()
    assert target is not None
    first, again = search_designs(target), search_designs(target)
    assert [r.model_dump() for r in first.rows] == [r.model_dump() for r in again.rows]
    assert len(first.rows) == 28 and first.status == "designed"
    best = first.rows[0]
    assert (best.layup, best.fiber_volume_fraction, best.feasibility) == (
        "[0/45/-45/90]s", 0.65, "feasible",
    )  # fmt: skip
    # [0/60/-60]s has identical in-plane stiffness; the configured layup order breaks the tie
    assert first.rows[1].layup == "[0/60/-60]s"
    assert first.rows[0].properties == pytest.approx(first.rows[1].properties, rel=1e-12)
    assert first.best is not None and first.best.origin == "composite"
    assert first.best.material.measurements("laminate_ex")[0].basis == "predicted"
    small = search_designs(
        target, DesignSearchConfig(layups=("0/90",), fiber_volume_fractions=(0.6,))
    )
    assert small.status == "no_feasible_design" and len(small.rows) == 1


def test_no_supported_requirement_means_no_design() -> None:
    target = (
        RuleBasedInterpreter()
        .interpret("The maximum service temperature must be at least 80 °C.")
        .target()
    )
    assert target is not None
    result = search_designs(target)
    assert result.status == "no_supported_requirements" and result.best is None


# -- reference routing policy ---------------------------------------------------------------------


class _Request:
    def __init__(self, ids: tuple[str, ...]) -> None:
        self.capability_ids = ids


def test_reference_routing_policy_follows_preconditions() -> None:
    assert (
        reference_routing_payload(_Request(WORKFLOW_ORDER[2:4]))["capability_id"]
        == (WORKFLOW_ORDER[2])
    )
    assert reference_routing_payload(_Request(("unrelated.capability",)))["action"] == "finish"
