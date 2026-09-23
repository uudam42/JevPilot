"""Engineering requirement → material-system observable resolution (offline)."""

from __future__ import annotations

from typing import Any

import pytest

from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.demo import directional_demo_target, example_ply
from domains.uav_materials.evaluation import EvaluationContext, evaluate_candidates
from domains.uav_materials.evidence import GapReason
from domains.uav_materials.properties import PROPERTIES
from domains.uav_materials.requirements import (
    FAMILY,
    LIFT,
    NOMINAL,
    Aspect,
    Direction,
    EngineeringRequirement,
    EngineeringTarget,
    MaterialSystemSemantics,
    RequirementResolver,
    ResolutionStatus,
    Rule,
    TargetBinding,
    material_system_of,
)
from domains.uav_materials.search import CheckStatus, search_materials
from examples.uav_material_search import target

from .helpers import req

RESOLVER = RequirementResolver()
PLY, BULK = "continuous_fiber_ud_ply", "isotropic_bulk"
REAL = real_candidates()
BAR = next(c for c in REAL if c.candidate_id == "mil5j-3.7.6.0-d")
COMPOSITE = example_ply(0.60)


UNIT = {"strength": "MPa", "deformation": "GPa", "density": "kg/m^3", "corrosion": "mm/year",
        "water_absorption": "%", "temperature_effects": "degC"}  # fmt: skip
STRAINS = {Aspect.DUCTILITY, Aspect.TENSILE_FAILURE_STRAIN, Aspect.YIELD_STRAIN}
NEEDS = {
    Aspect.CORROSION_PENETRATION: {"environment": "marine_atmosphere"},
    Aspect.WATER_UPTAKE: {
        "environment": "lab_water_immersion",
        "exposure_duration": {"value": 24, "unit": "h"},
    },
}


def er(
    aspect: Aspect, direction: Direction = Direction.UNSPECIFIED, **kw: Any
) -> EngineeringRequirement:
    """A valid hard '>= 1' requirement in the family's unit, with required conditions."""
    kw.setdefault("operator", ">=")
    kw.setdefault("priority", "hard")
    kw.setdefault("value", 1)
    kw.setdefault("unit", "%" if aspect in STRAINS else UNIT[FAMILY[aspect].value])
    kw.setdefault("conditions", NEEDS.get(aspect, {}))
    return EngineeringRequirement(aspect=aspect, direction=direction, **kw)


def resolve(aspect: Aspect, direction: Direction = Direction.UNSPECIFIED, system: str = PLY) -> Any:
    return RESOLVER.resolve(er(aspect, direction), system)


# -- tables are complete and consistent --------------------------------------------------------


def test_every_registered_property_lifts_and_every_aspect_has_a_family_and_carrier() -> None:
    assert set(LIFT) == set(PROPERTIES)
    assert set(FAMILY) == set(Aspect) == set(NOMINAL)
    for prop, (aspect, _) in LIFT.items():
        assert FAMILY[aspect] is PROPERTIES[prop].category, prop


def test_a_rule_crossing_families_is_refused() -> None:
    bad = MaterialSystemSemantics(
        "bad", "", {}, {Aspect.DENSITY: Rule(ResolutionStatus.RESOLVED, ("youngs_modulus",))}
    )
    with pytest.raises(ValueError, match="another family"):
        RequirementResolver({"bad": bad}).resolve(er(Aspect.DENSITY, operator="<="), "bad")


# -- metals: backward compatibility -------------------------------------------------------------


def test_legacy_requirements_resolve_to_themselves_for_bulk_records() -> None:
    profile = target()
    binding = TargetBinding(profile)
    for r in profile.requirements:
        res = binding.resolve(BAR.material, r)
        assert res.status is ResolutionStatus.RESOLVED and res.property_requirement is r
    assert material_system_of(BAR.material) == BULK


def test_metal_results_unchanged_through_the_resolver() -> None:
    search = search_materials(target(), REAL)
    unified = evaluate_candidates(target(), REAL)
    assert search.counts == {"considered": 46, "feasible": 0, "infeasible": 34, "undetermined": 12}
    assert [(a.candidate_id, a.feasibility, a.distance, a.pessimistic_distance, a.score_coverage)
            for a in search.assessments] == [
        (e.candidate_id, e.feasibility, e.distance, e.pessimistic_distance, e.coverage)
        for e in unified
    ]  # fmt: skip
    for a, e in zip(search.assessments, unified, strict=True):
        assert [c.reason for c in a.checks] == [c.reason for c in e.checks]
        assert all(o.resolution is ResolutionStatus.RESOLVED for o in e.outcomes)


# -- positive resolutions -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("aspect", "direction", "prop"),
    [
        (Aspect.NORMAL_STIFFNESS, Direction.MATERIAL_1, "ply_longitudinal_modulus"),
        (Aspect.NORMAL_STIFFNESS, Direction.MATERIAL_2, "ply_transverse_modulus"),
        (Aspect.SHEAR_STIFFNESS, Direction.MATERIAL_12, "ply_inplane_shear_modulus"),
        (Aspect.TENSILE_ULTIMATE, Direction.MATERIAL_1, "ply_longitudinal_tensile_strength"),
        (Aspect.TENSILE_FAILURE_STRAIN, Direction.MATERIAL_1,
         "ply_longitudinal_tensile_failure_strain"),
    ],
)  # fmt: skip
def test_directional_ply_requirements_resolve(
    aspect: Aspect, direction: Direction, prop: str
) -> None:
    res = resolve(aspect, direction)
    assert res.status is ResolutionStatus.RESOLVED and res.properties == (prop,)
    assert res.property_requirement is not None and res.property_requirement.property == prop
    assert res.direction is direction


def test_density_resolves_for_metals_and_composites() -> None:
    for system in (BULK, PLY):
        for direction in (Direction.UNSPECIFIED, Direction.MATERIAL_1):
            res = RESOLVER.resolve(er(Aspect.DENSITY, direction, operator="<="), system)
            assert res.properties == ("density",) and res.resolved


# -- no silent equivalence (regressions) -------------------------------------------------------


def test_e11_does_not_answer_a_directionless_youngs_modulus() -> None:
    res = RESOLVER.resolve(EngineeringRequirement.lift(req("youngs_modulus", ">=", 70, "GPa")), PLY)
    assert res.status is ResolutionStatus.AMBIGUOUS and res.property_requirement is None
    assert res.properties == ("ply_longitudinal_modulus", "ply_transverse_modulus")


def test_s11t_does_not_answer_a_directionless_tensile_strength() -> None:
    res = RESOLVER.resolve(
        EngineeringRequirement.lift(req("tensile_strength", ">=", 350, "MPa")), PLY
    )
    assert res.status is ResolutionStatus.AMBIGUOUS and res.property_requirement is None


def test_failure_strain_does_not_answer_metallic_elongation() -> None:
    res = RESOLVER.resolve(
        EngineeringRequirement.lift(req("elongation_at_break", ">=", 5, "%")), PLY
    )
    assert res.status is ResolutionStatus.UNSUPPORTED and res.property_requirement is None
    assert "failure strain is a different quantity" in res.reason
    assert resolve(Aspect.TENSILE_FAILURE_STRAIN).status is ResolutionStatus.AMBIGUOUS


# -- ambiguous / unsupported --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("aspect", "direction"),
    [
        (Aspect.CORROSION_PENETRATION, Direction.UNSPECIFIED),
        (Aspect.WATER_UPTAKE, Direction.UNSPECIFIED),
        (Aspect.MAX_SERVICE_TEMPERATURE, Direction.UNSPECIFIED),
        (Aspect.TENSILE_ULTIMATE, Direction.MATERIAL_2),  # S22T not implemented
        (Aspect.NORMAL_STIFFNESS, Direction.LAMINATE_X),  # needs CLT
        (Aspect.TENSILE_YIELD, Direction.MATERIAL_1),
    ],
)
def test_unsupported_for_the_ply_model(aspect: Aspect, direction: Direction) -> None:
    res = RESOLVER.resolve(er(aspect, direction), PLY)
    assert res.status is ResolutionStatus.UNSUPPORTED and res.reason
    assert res.property_requirement is None


def test_unknown_material_system_is_unsupported_not_guessed() -> None:
    res = RESOLVER.resolve(er(Aspect.DENSITY, operator="<="), "graphene_aerogel_v9")
    assert res.status is ResolutionStatus.UNSUPPORTED and "no requirement semantics" in res.reason


def test_resolution_is_deterministic() -> None:
    cases = [(a, sys) for a in (Aspect.NORMAL_STIFFNESS, Aspect.DENSITY) for sys in (BULK, PLY)]
    first = [RESOLVER.resolve(er(a), sys).model_dump() for a, sys in cases]
    again = [RequirementResolver().resolve(er(a), sys).model_dump() for a, sys in cases]
    assert first == again


def test_duplicate_soft_keys_are_rejected() -> None:
    soft = {"operator": "maximize", "value": None, "unit": None, "priority": "soft", "weight": 1.0,
            "weight_source": "test"}  # fmt: skip
    with pytest.raises(ValueError, match="distinct keys"):
        EngineeringTarget(
            profile_id="x",
            name="x",
            requirements=(er(Aspect.TENSILE_ULTIMATE, Direction.MATERIAL_1, **soft),
                          er(Aspect.TENSILE_ULTIMATE, Direction.MATERIAL_2, **soft)),
        )  # fmt: skip


# -- unified evaluator integration ------------------------------------------------------------


@pytest.fixture(scope="module")
def uav() -> EvaluationContext:
    return EvaluationContext.fit(target(), REAL)


def test_composite_under_the_unchanged_uav_target(uav: EvaluationContext) -> None:
    e = uav.evaluate(COMPOSITE)
    assert e.material_system == PLY
    gaps = {(o.requirement, o.hard): o.gap for o in e.outcomes}
    assert gaps == {
        ("density <= 3000 kg/m^3", True): None,
        ("tensile_ultimate_strength >= 350 MPa", True): "ambiguous",
        ("ductility >= 5 %", True): "unsupported",
        ("corrosion_penetration_rate <= 0.025 mm/year", True): "unsupported",
        ("max_service_temperature >= 80 degC", True): "unsupported",
        ("tensile_ultimate_strength maximize", False): "ambiguous",
        ("density minimize", False): None,
        ("normal_stiffness in 65-120 GPa", False): "ambiguous",
        ("tensile_yield_strength maximize", False): "unsupported",
        ("water_uptake minimize", False): "unsupported",
    }
    status = {c.requirement.property: c.status for c in e.checks}
    assert status["density"] is CheckStatus.SATISFIED
    assert [s for p, s in status.items() if p != "density"] == [CheckStatus.UNDETERMINED] * 4
    tensile = next(c for c in e.checks if c.requirement.property == "tensile_strength")
    assert tensile.reason.startswith("ambiguous") and not tensile.used
    ev = {r.property: r.reasons for r in e.evidence.requirements if r.hard}
    assert ev["tensile_strength"] == (GapReason.AMBIGUOUS_REQUIREMENT,)
    assert ev["elongation_at_break"] == (GapReason.UNSUPPORTED_OBSERVABLE,)
    youngs = next(c for c in e.contributions if c.property == "youngs_modulus")
    assert youngs.value is None and youngs.note.startswith("ambiguous")
    assert e.feasibility == "undetermined" and e.evidence.classification == "evidence_gap"


def test_missing_is_distinct_from_ambiguous_and_unsupported(uav: EvaluationContext) -> None:
    e = uav.evaluate(BAR)
    gaps = {o.requirement: o.gap for o in e.outcomes if o.hard}
    assert gaps["corrosion_penetration_rate <= 0.025 mm/year"] == "missing"
    assert gaps["max_service_temperature >= 80 degC"] == "missing"
    assert {o.resolution for o in e.outcomes} == {ResolutionStatus.RESOLVED}


def test_directional_demo_target_uses_e11_and_s11t_legitimately() -> None:
    demo = EvaluationContext.fit(directional_demo_target(), REAL)
    e = demo.evaluate(COMPOSITE)
    used = {c.requirement.property: (c.status, c.value) for c in e.checks}
    assert used["ply_longitudinal_tensile_strength"][0] is CheckStatus.SATISFIED
    assert used["ply_longitudinal_modulus"][0] is CheckStatus.SATISFIED
    assert used["ply_longitudinal_tensile_strength"][1] == pytest.approx(1672.5, abs=0.5)
    assert used["ply_longitudinal_modulus"][1] == pytest.approx(129.6, abs=0.05)
    assert e.feasibility == "feasible"
    assert e.evidence.classification == "unclassified"  # no acceptable region: not ADEQUATE
    assert e.evidence.relies_on_predictions  # still reported as predictions
    metal = demo.evaluate(BAR)
    assert all(
        o.properties[0] in {"density", "tensile_strength", "youngs_modulus"} for o in metal.outcomes
    )
    assert any("isotropic bulk material" in a for r in metal.resolutions for a in r.assumptions)


def test_demo_target_is_not_the_uav_target() -> None:
    demo = directional_demo_target()
    assert demo.profile_id != target().profile_id and "not the UAV target" in demo.name
