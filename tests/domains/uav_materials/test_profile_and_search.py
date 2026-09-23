"""Target profiles, the search interface, buoyancy helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from domains.uav_materials import Operator, Priority, TargetMaterialProfile, search_materials
from domains.uav_materials.buoyancy import density_ratio, net_buoyant_force_per_volume
from domains.uav_materials.fixtures import synthetic_candidates
from domains.uav_materials.schema import Quantity
from domains.uav_materials.search import CheckStatus, Feasibility
from domains.uav_materials.units import UnitError
from tests.domains.uav_materials.helpers import WA24, req, spar_profile


def test_profile_separates_hard_soft_ranges_and_weights() -> None:
    p = spar_profile()
    assert [r.property for r in p.hard_constraints] == [
        "density",
        "tensile_strength",
        "water_absorption",
        "max_service_temperature",
    ]
    assert {(r.property, r.operator) for r in p.soft_preferences} == {
        ("youngs_modulus", Operator.BETWEEN),
        ("tensile_strength", Operator.MAXIMIZE),
        ("density", Operator.MINIMIZE),
    }
    (rng,) = p.ranges
    assert rng.bounds("GPa") == (40, 70) and rng.priority is Priority.SOFT
    w = p.weights()
    assert w["tensile_strength:maximize"]["weight"] == 0.5 and w["density:minimize"]["source"]
    assert sum(p.normalized_weights().values()) == pytest.approx(1.0)
    assert p.environment.fluid_density and p.environment.fluid_density.to("g/cm^3") == 1.025


@pytest.mark.parametrize(
    "bad",
    [
        lambda: req("tensile_strength", "maximize", priority="hard"),  # direction cannot be hard
        lambda: req(
            "density", "<=", 1600, "kg/m^3", weight=0.5, weight_source="x"
        ),  # hard + weight
        lambda: req("density", "minimize", weight=0.5),  # weight without its source
        lambda: req("density", "<=", 1600, "MPa"),  # wrong dimension
        lambda: req("density", "<=", None, None),  # bound without value
        lambda: req("youngs_modulus", "between", 70, "GPa", upper=40),  # inverted range
        lambda: req("youngs_modulus", "target", 55, "GPa", priority="hard"),  # exact hard target
        lambda: req("water_absorption", "<=", 1, "%"),  # condition-dependent without conditions
        lambda: req("corrosion_rate", "<=", 0.1, "mm/year"),  # no medium
    ],
)
def test_malformed_requirements_rejected(bad: object) -> None:
    with pytest.raises((ValidationError, UnitError)):
        bad()  # type: ignore[operator]


def test_weights_are_never_invented() -> None:
    p = TargetMaterialProfile(
        profile_id="p",
        name="n",
        requirements=(
            req("density", "minimize", importance="high"),  # qualitative only
            req("tensile_strength", "maximize", weight=1.0, weight_source="stated by test"),
        ),
    )
    assert p.weights() == {"tensile_strength:maximize": {"weight": 1.0, "source": "stated by test"}}
    with pytest.raises(ValueError, match="no numerical weight"):
        p.normalized_weights()


def test_physical_profile_is_not_a_feature_vector() -> None:
    fields = set(TargetMaterialProfile.model_fields)
    assert not {"features", "vector", "scores"} & fields
    assert "strength" not in fields  # categories are not collapsed into scalars


def _by_name(result: object) -> dict[str, object]:
    return {a.candidate_id.removeprefix("cand-"): a for a in result.assessments}  # type: ignore[attr-defined]


def test_hard_constraint_screen_distinguishes_bad_from_unknown() -> None:
    result = search_materials(spar_profile(), synthetic_candidates())
    a = _by_name(result)
    assert a["syn-composite-c"].feasibility is Feasibility.FEASIBLE  # type: ignore[attr-defined]
    sparse = a["syn-composite-f"]
    assert sparse.feasibility is Feasibility.UNDETERMINED  # type: ignore[attr-defined]
    assert {"water_absorption", "max_service_temperature"} <= set(sparse.missing)  # type: ignore[attr-defined]
    for heavy in ("syn-metal-a", "syn-metal-b", "syn-ceramic-g"):
        assert a[heavy].feasibility is Feasibility.INFEASIBLE  # type: ignore[attr-defined]
        assert "density" in {v.requirement.property for v in a[heavy].violations}  # type: ignore[attr-defined]
    order = [x.feasibility for x in result.assessments]
    assert order == sorted(order, key=["feasible", "undetermined", "infeasible"].index)
    assert all(x.distance is None for x in result.assessments)  # no ranking yet
    assert a["syn-composite-c"].provenance  # type: ignore[attr-defined]


def test_units_are_converted_before_comparing() -> None:
    kg = TargetMaterialProfile(
        profile_id="a", name="a", requirements=(req("density", "<=", 1600, "kg/m^3"),)
    )
    g = TargetMaterialProfile(
        profile_id="b", name="b", requirements=(req("density", "<=", 1.6, "g/cm^3"),)
    )
    cands = synthetic_candidates()
    assert [x.feasibility for x in search_materials(kg, cands).assessments] == [
        x.feasibility for x in search_materials(g, cands).assessments
    ]


def test_measurements_from_other_conditions_are_not_compared() -> None:
    salt = {"medium": "salt water (3.5% NaCl, synthetic spec)"}
    fresh = {"medium": "fresh water"}
    cands = synthetic_candidates()

    def status(conditions: dict[str, object], material: str) -> CheckStatus:
        p = TargetMaterialProfile(
            profile_id="c",
            name="c",
            requirements=(req("corrosion_rate", "<=", 0.1, "mm/year", conditions=conditions),),
        )
        return _by_name(search_materials(p, cands))[material].checks[0].status  # type: ignore[attr-defined]

    assert status(salt, "syn-metal-b") is CheckStatus.VIOLATED  # 0.3 mm/year in salt water
    assert status(fresh, "syn-metal-b") is CheckStatus.SATISFIED  # 0.02 mm/year in fresh water
    assert status(salt, "syn-foam-e") is CheckStatus.UNDETERMINED  # never measured
    hot = TargetMaterialProfile(
        profile_id="d",
        name="d",
        requirements=(
            req(
                "water_absorption",
                "<=",
                1.0,
                "%",
                conditions={**WA24, "exposure_duration": {"value": 168, "unit": "h"}},
            ),
        ),
    )
    assert (
        _by_name(search_materials(hot, cands))["syn-composite-c"].checks[0].status
        is CheckStatus.UNDETERMINED
    )  # 24 h data does not answer a 168 h question


def test_buoyancy_uses_material_and_fluid_density() -> None:
    sea = Quantity(value=1025, unit="kg/m^3")
    foam, metal = Quantity(value=200, unit="kg/m^3"), Quantity(value=4.4, unit="g/cm^3")
    assert density_ratio(foam, sea) > 1 > density_ratio(metal, sea)
    assert net_buoyant_force_per_volume(foam, sea) > 0 > net_buoyant_force_per_volume(metal, sea)
