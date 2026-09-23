"""Harmonization, compatibility, selection, not_applicable, ranking and explanations."""

from __future__ import annotations

import pytest

from domains.uav_materials import MaterialCandidate, MaterialRecord, TargetMaterialProfile
from domains.uav_materials import search_materials as search
from domains.uav_materials.compatibility import Compat, compatibility
from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.harmonize import harmonize
from domains.uav_materials.ranking import TargetRelativeExtractor
from domains.uav_materials.schema import TestConditions
from domains.uav_materials.search import CheckStatus, Feasibility, check
from domains.uav_materials.selection import select
from examples.uav_material_search import target
from jevpilot import Provenance, SourceRef
from tests.domains.uav_materials.helpers import SYN, meas, req


def rec(*ms: object, mid: str = "m", name: str = "SyntheticM") -> MaterialRecord:
    groups: dict[str, list[object]] = {}
    for m in ms:
        groups.setdefault(str(m.category), []).append(m)  # type: ignore[attr-defined]
    return MaterialRecord(
        material_id=mid,
        name=name,
        family="metal",
        provenance=SYN,
        provenance_status="synthetic",
        **{k: tuple(v) for k, v in groups.items()},
    )  # type: ignore[arg-type]


def cand(r: MaterialRecord) -> MaterialCandidate:
    return MaterialCandidate(candidate_id=r.material_id, origin="existing", material=r)


def test_harmonized_values_keep_the_original() -> None:
    h = harmonize(meas("youngs_modulus", 10.3, "Msi"))
    assert h is not None and h.unit == "GPa" and h.value == pytest.approx(71.016, rel=1e-4)
    assert (h.original_value, h.original_unit) == (10.3, "Msi")
    d = harmonize(meas("density", 0.101, "lb/in^3"))
    assert d is not None and d.value == pytest.approx(2795.67, rel=1e-4)


@pytest.mark.parametrize(
    ("required", "observed", "level"),
    [
        (
            {
                "exposure_duration": {"value": 168, "unit": "h"},
                "environment": "lab_water_immersion",
            },
            {"exposure_duration": {"value": 24, "unit": "h"}, "environment": "lab_water_immersion"},
            Compat.INCOMPATIBLE,
        ),
        (
            {"environment": "seawater_immersion"},
            {"environment": "freshwater_immersion"},
            Compat.INCOMPATIBLE,
        ),
        (
            {"environment": "marine_atmosphere"},
            {"environment": "marine_atmosphere", "medium": "Cristobal"},
            Compat.COMPATIBLE,
        ),
        (
            {"environment": "marine_atmosphere", "medium": "Cristobal"},
            {"environment": "marine_atmosphere", "medium": "Cristobal"},
            Compat.EXACT,
        ),
        (
            {"temperature": {"value": 23, "unit": "degC"}},
            {"temperature_regime": "room"},
            Compat.COMPATIBLE,
        ),
        (
            {"temperature": {"value": 200, "unit": "degC"}},
            {"temperature_regime": "room"},
            Compat.INCOMPATIBLE,
        ),
        (
            {"temperature": {"value": 200, "unit": "degC"}},
            {"temperature": {"value": 473.15, "unit": "K"}},
            Compat.EXACT,
        ),
        ({"temperature": {"value": 200, "unit": "degC"}}, {}, Compat.INCOMPATIBLE),
        ({}, {"temperature": {"value": 300, "unit": "degC"}}, Compat.INCOMPATIBLE),
        ({}, {"temperature_regime": "room"}, Compat.UNCONDITIONED),
        ({}, {}, Compat.UNCONDITIONED),
    ],
)
def test_condition_compatibility(
    required: dict[str, object], observed: dict[str, object], level: Compat
) -> None:
    got, reason = compatibility(
        TestConditions.model_validate(required), TestConditions.model_validate(observed)
    )
    assert got is level and reason


def _src(table: str, match: str = "exact") -> Provenance:
    return Provenance(
        sources=(SourceRef(kind="handbook", identifier="H", metadata={"table": table}),),
        metadata={"identity_match": match},
    )


def test_selection_policy_is_deterministic_and_keeps_alternatives() -> None:
    room = {"temperature_regime": "room"}
    hot = {"temperature": {"value": 300, "unit": "degC"}}
    ms = [
        meas(
            "tensile_strength",
            500,
            "MPa",
            conditions=room,
            statistical_basis="S",
            provenance=_src("s"),
            provenance_status="sourced",
        ),
        meas(
            "tensile_strength",
            480,
            "MPa",
            conditions=room,
            statistical_basis="A",
            provenance=_src("a1"),
            provenance_status="sourced",
        ),
        meas(
            "tensile_strength",
            470,
            "MPa",
            conditions=room,
            statistical_basis="A",
            provenance=_src("a2"),
            provenance_status="sourced",
        ),
        meas(
            "tensile_strength",
            400,
            "MPa",
            conditions=room,
            statistical_basis="A",
            provenance=_src("alloy", "alloy"),
            provenance_status="sourced",
        ),
        meas("tensile_strength", 300, "MPa", conditions=hot),
    ]
    r = rec(*ms)
    sel = select(r, req("tensile_strength", ">=", 350, "MPa"))
    # A-basis beats S, exact identity beats alloy-level, then the conservative (lowest) value
    assert sel.chosen is ms[2] and len(sel.alternatives) == 3
    assert [m for m, _ in sel.excluded] == [ms[4]]  # 300 °C data answers a different question
    assert select(r, req("tensile_strength", ">=", 350, "MPa")).chosen is ms[2]  # stable
    at_300 = req("tensile_strength", ">=", 350, "MPa", conditions=hot)
    assert select(r, at_300).chosen is ms[4]


def test_feasible_undetermined_infeasible_are_never_collapsed() -> None:
    p = TargetMaterialProfile(
        profile_id="p",
        name="p",
        requirements=(
            req("density", "<=", 3000, "kg/m^3"),
            req("tensile_strength", ">=", 400, "MPa"),
        ),
    )
    good = rec(meas("density", 2700, "kg/m^3"), meas("tensile_strength", 500, "MPa"), mid="g")
    bad = rec(meas("density", 2700, "kg/m^3"), meas("tensile_strength", 300, "MPa"), mid="b")
    unknown = rec(meas("density", 2700, "kg/m^3"), mid="u")
    result = search(p, [cand(unknown), cand(bad), cand(good)])
    assert [(a.candidate_id, a.feasibility) for a in result.assessments] == [
        ("g", Feasibility.FEASIBLE),
        ("u", Feasibility.UNDETERMINED),
        ("b", Feasibility.INFEASIBLE),
    ]
    assert result.counts == {"considered": 3, "feasible": 1, "infeasible": 1, "undetermined": 1}
    (u,) = [a for a in result.assessments if a.candidate_id == "u"]
    assert u.undetermined[0].reason == "no measurement recorded"


def test_not_applicable_policy() -> None:
    wa = {"environment": "lab_water_immersion", "exposure_duration": {"value": 24, "unit": "h"}}
    metal = cand(
        rec(
            meas("water_absorption", missing="not_applicable"),
            meas("glass_transition_temperature", missing="not_applicable"),
            meas("corrosion_rate", missing="not_applicable"),
        )
    )
    assert (
        check(req("water_absorption", "<=", 1, "%", conditions=wa), metal).status
        is CheckStatus.SATISFIED
    )
    assert (
        check(req("water_absorption", ">=", 1, "%", conditions=wa), metal).status
        is CheckStatus.UNDETERMINED
    )
    assert (
        check(req("glass_transition_temperature", ">=", 100, "degC"), metal).status
        is CheckStatus.UNDETERMINED
    )
    assert (
        check(
            req(
                "corrosion_rate",
                "<=",
                0.1,
                "mm/year",
                conditions={"environment": "marine_atmosphere"},
            ),
            metal,
        ).status
        is CheckStatus.UNDETERMINED
    )
    unknown = cand(rec(meas("water_absorption", missing="unknown")))
    assert (
        check(req("water_absorption", "<=", 1, "%", conditions=wa), unknown).status
        is CheckStatus.UNDETERMINED
    )


def test_upper_bounds_answer_only_upper_limits() -> None:
    env = {"environment": "marine_atmosphere"}
    bound = rec(meas("corrosion_rate", 0.01, "mm/year", qualifier="<=", conditions=env))
    ok = check(req("corrosion_rate", "<=", 0.025, "mm/year", conditions=env), cand(bound))
    assert ok.status is CheckStatus.SATISFIED and "upper bound" in ok.reason
    loose = rec(meas("corrosion_rate", 0.05, "mm/year", qualifier="<=", conditions=env))
    assert (
        check(req("corrosion_rate", "<=", 0.025, "mm/year", conditions=env), cand(loose)).status
        is CheckStatus.UNDETERMINED
    )  # true value may still pass
    assert (
        check(req("corrosion_rate", ">=", 0.001, "mm/year", conditions=env), cand(bound)).status
        is CheckStatus.UNDETERMINED
    )


def _profile(*reqs: object) -> TargetMaterialProfile:
    return TargetMaterialProfile(profile_id="rank", name="rank", requirements=tuple(reqs))  # type: ignore[arg-type]


W = {"weight_source": "test"}


def test_direction_aware_penalties() -> None:
    mats = [
        rec(
            meas("tensile_strength", v, "MPa"),
            meas("density", d, "kg/m^3"),
            meas("youngs_modulus", e, "GPa"),
            mid=f"m{v}",
        )
        for v, d, e in ((500, 2800, 71), (400, 2700, 110), (250, 4400, 30))
    ]
    p = _profile(
        req("tensile_strength", "maximize", weight=1.0, **W),
        req("density", "minimize", weight=1.0, **W),
        req("youngs_modulus", "between", 40, "GPa", upper=70, priority="soft", weight=1.0, **W),
        req("youngs_modulus", "target", 60, "GPa", weight=1.0, **W),
    )
    x = TargetRelativeExtractor.fit(p, mats)
    mx, mn, rng, tgt = p.soft_preferences
    assert x.penalty(mx, 500) == 0 and x.penalty(mx, 400) == pytest.approx(0.2)
    assert x.penalty(mx, 600) == 0  # above the best is not penalised
    assert x.penalty(mn, 2700) == 0 and x.penalty(mn, 2970) == pytest.approx(0.1)
    assert x.penalty(rng, 55) == 0 and x.penalty(rng, 75) == pytest.approx(5 / 30)
    assert x.penalty(rng, 25) == pytest.approx(15 / 30)
    assert x.penalty(tgt, 66) == pytest.approx(0.1)


def test_missing_soft_values_lower_coverage_and_never_help() -> None:
    full = rec(meas("tensile_strength", 450, "MPa"), meas("density", 2800, "kg/m^3"), mid="full")
    partial = rec(meas("tensile_strength", 500, "MPa"), mid="partial")
    p = _profile(
        req("tensile_strength", "maximize", weight=0.5, **W),
        req("density", "minimize", weight=0.5, **W),
        req(
            "water_absorption",
            "minimize",
            importance="high",
            conditions={
                "environment": "lab_water_immersion",
                "exposure_duration": {"value": 24, "unit": "h"},
            },
        ),
    )
    result = search(p, [cand(partial), cand(full)])
    a = {x.candidate_id: x for x in result.assessments}
    assert a["partial"].score_coverage == 0.5 and a["full"].score_coverage == 1.0
    assert a["partial"].distance == 0.0  # perfect on what it covers...
    assert (
        a["partial"].pessimistic_distance > a["full"].pessimistic_distance
    )  # ...but not ranked first
    assert [x.candidate_id for x in result.assessments] == ["full", "partial"]
    assert result.unscored_preferences == ("water_absorption:minimize",)
    missing = [k for k in a["partial"].contributions if k.value is None]
    assert missing and missing[0].property == "density" and "missing" in missing[0].note


def test_weighted_distance_formula() -> None:
    best = rec(meas("tensile_strength", 500, "MPa"), meas("density", 2000, "kg/m^3"), mid="b")
    other = rec(meas("tensile_strength", 400, "MPa"), meas("density", 3000, "kg/m^3"), mid="o")
    p = _profile(
        req("tensile_strength", "maximize", weight=3.0, **W),
        req("density", "minimize", weight=1.0, **W),
    )
    a = {x.candidate_id: x for x in search(p, [cand(best), cand(other)]).assessments}
    # penalties 0.2 and 0.5, weights 3 and 1 → sqrt((3*0.04 + 1*0.25) / 4)
    assert a["o"].distance == pytest.approx(((3 * 0.04 + 0.25) / 4) ** 0.5)
    assert sum(k.share or 0 for k in a["o"].contributions) == pytest.approx(a["o"].distance ** 2)
    assert a["b"].distance == 0 and a["b"].rank == 1


def test_real_example_search_is_deterministic_and_explained() -> None:
    first = search(target(), real_candidates())
    second = search(target(), real_candidates())
    assert [a.candidate_id for a in first.assessments] == [
        a.candidate_id for a in second.assessments
    ]
    assert first.counts["considered"] == len(real_candidates())
    assert first.normalization["normalization_pool"].startswith("non-infeasible")
    for a in first.assessments:
        assert a.rank is not None
        for c in a.checks:
            assert c.reason
            if c.used:
                assert c.used[0].provenance and "MIL-HDBK-5J" in c.reason or "NRL" in c.reason
        for k in a.contributions:
            assert k.value is None or k.source  # every scored value names its source
    ranks = [a.feasibility for a in first.assessments]
    assert ranks == sorted(ranks, key=["feasible", "undetermined", "infeasible"].index)
    # coverage matters: a partially covered candidate never outranks a fully covered one in
    # its group when its pessimistic distance is worse
    und = [a for a in first.assessments if a.feasibility is Feasibility.UNDETERMINED]
    pess = [a.pessimistic_distance for a in und]
    assert pess == sorted(pess)  # type: ignore[type-var]
