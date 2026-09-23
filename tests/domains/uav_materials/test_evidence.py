"""Evidence sufficiency, gap classification and acquisition priority (offline)."""

from __future__ import annotations

from typing import Any

from domains.uav_materials import MaterialCandidate, MaterialRecord, TargetMaterialProfile
from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.evidence import (
    EvidencePolicy,
    GapClass,
    GapReason,
    assess,
    design_gap_preconditions,
    evidence_report,
    format_report,
)
from domains.uav_materials.search import Feasibility, search_materials
from examples.uav_material_search import target
from jevpilot import Provenance, SourceRef

from .helpers import meas, req

LENIENT = EvidencePolicy(accepted_provenance=("sourced", "synthetic"))
MARINE = {"environment": "marine_atmosphere"}
WATER_24H = {"environment": "lab_water_immersion", "exposure_duration": {"value": 24, "unit": "h"}}


def profile() -> TargetMaterialProfile:
    return TargetMaterialProfile(
        profile_id="syn-evidence",
        name="synthetic evidence profile",
        requirements=(
            req("density", "<=", 3000, "kg/m^3"),
            req("tensile_strength", ">=", 350, "MPa"),
            req("corrosion_rate", "<=", 0.025, "mm/year", conditions=MARINE),
            req("max_service_temperature", ">=", 80, "degC"),
            req("tensile_strength", "maximize", weight=0.6, weight_source="test"),
            req("density", "minimize", weight=0.4, weight_source="test"),
            req("water_absorption", "minimize", conditions=WATER_24H, importance="high"),
        ),
    )


def sourced(identity: str | None = None) -> dict[str, Any]:
    meta = {"identity_match": identity} if identity else {}
    return {
        "provenance_status": "sourced",
        "provenance": Provenance(
            sources=(SourceRef(kind="report", identifier="SRC"),), metadata=meta
        ),
    }


def cand(**groups: Any) -> MaterialCandidate:
    rec = MaterialRecord(material_id="m", name="m", family="metal", provenance_status="synthetic",
                         **groups)  # fmt: skip
    return MaterialCandidate(candidate_id="m", origin="existing", material=rec)


FULL = {
    "density": (meas("density", 2800, "kg/m^3", **sourced()),),
    "strength": (meas("tensile_strength", 500, "MPa", **sourced()),),
    "corrosion": (meas("corrosion_rate", 0.01, "mm/year", conditions=MARINE, **sourced()),),
    "temperature_effects": (meas("max_service_temperature", 120, "degC", **sourced()),),
}


def by_prop(a: Any, prop: str, hard: bool = True) -> Any:
    return next(e for e in a.requirements if e.property == prop and e.hard is hard)


# -- coverage ------------------------------------------------------------------------------


def test_hard_evidence_coverage_counts_sufficient_hard_constraints() -> None:
    a = assess(profile(), cand(**{**FULL, "corrosion": (), "temperature_effects": ()}))
    assert a.hard_coverage == (2, 4) and a.hard_coverage_ratio == 0.5
    assert a.missing_critical == (
        "corrosion_rate [environment=marine_atmosphere]",
        "max_service_temperature",
    )
    assert a.classification is GapClass.EVIDENCE_GAP


def test_soft_evidence_coverage_counted_and_weighted_by_explicit_weights_only() -> None:
    a = assess(profile(), cand(density=FULL["density"]))
    assert a.soft_coverage == (1, 3)  # density minimize known; tensile, water missing
    assert a.soft_weighted_coverage == 0.4  # water_absorption has no weight: not in it


# -- gap reasons -----------------------------------------------------------------------------


def test_missing_measurement() -> None:
    e = by_prop(assess(profile(), cand(**{**FULL, "temperature_effects": ()})),
                "max_service_temperature")  # fmt: skip
    assert (e.measurement_available, e.sufficient, e.reasons) == (
        False,
        False,
        (GapReason.MISSING_PROPERTY,),
    )


def test_condition_mismatch() -> None:
    seawater = meas("corrosion_rate", 0.01, "mm/year", conditions={"environment": "seawater_tidal"},
                    **sourced())  # fmt: skip
    a = assess(profile(), cand(**{**FULL, "corrosion": (seawater,)}))
    e = by_prop(a, "corrosion_rate")
    assert e.measurement_available and not e.condition_compatible
    assert e.reasons == (GapReason.CONDITION_MISMATCH,) and e.tier == 2
    assert "seawater_tidal" in a.incompatible_evidence[0]
    assert a.classification is GapClass.EVIDENCE_GAP


def test_weak_identity_match() -> None:
    alloy_level = meas("max_service_temperature", 120, "degC", **sourced("alloy"))
    c = cand(**{**FULL, "temperature_effects": (alloy_level,)})
    assert assess(profile(), c).classification is GapClass.UNCLASSIFIED  # default: alloy ok
    strict = EvidencePolicy(min_identity="exact")
    a = assess(profile(), c, strict)
    e = by_prop(a, "max_service_temperature")
    assert e.identity_match == "alloy" and e.reasons == (GapReason.IDENTITY_MATCH_TOO_WEAK,)
    assert a.classification is GapClass.EVIDENCE_GAP


def test_insufficient_provenance_and_uncertainty() -> None:
    synthetic = meas("max_service_temperature", 120, "degC")
    a = assess(profile(), cand(**{**FULL, "temperature_effects": (synthetic,)}))
    assert by_prop(a, "max_service_temperature").reasons == (GapReason.INSUFFICIENT_PROVENANCE,)
    assert assess(profile(), cand(**{**FULL, "temperature_effects": (synthetic,)}), LENIENT
                  ).classification is GapClass.UNCLASSIFIED  # fmt: skip
    noisy = meas("max_service_temperature", 120, "degC",
                 uncertainty={"dispersion": 30.0}, **sourced())  # fmt: skip
    policy = EvidencePolicy(max_relative_uncertainty=0.1)
    e = by_prop(assess(profile(), cand(**{**FULL, "temperature_effects": (noisy,)}), policy),
                "max_service_temperature")  # fmt: skip
    assert e.uncertainty_known and e.reasons == (GapReason.UNCERTAINTY_TOO_HIGH,)


def test_provenance_is_preserved_in_the_assessment() -> None:
    a = assess(profile(), cand(**FULL))
    e = by_prop(a, "tensile_strength")
    assert (e.provenance, e.source, e.identity_match, e.value) == (
        "sourced",
        "SRC",
        "native",
        "500 MPa",
    )
    assert a.provenance_quality["sourced"] == a.provenance_quality["selected_measurements"] == 6


# -- classification --------------------------------------------------------------------------


def test_infeasible_needs_a_sufficiently_supported_violation() -> None:
    heavy = {**FULL, "density": (meas("density", 4500, "kg/m^3", **sourced()),)}
    a = assess(profile(), cand(**heavy))
    assert a.classification is GapClass.INFEASIBLE and a.violated == ("density",)
    # the same violation shown only by synthetic data is an evidence gap, not a failure
    weak = {**FULL, "density": (meas("density", 4500, "kg/m^3"),)}
    b = assess(profile(), cand(**weak))
    assert b.feasibility is Feasibility.INFEASIBLE  # search semantics unchanged
    assert b.classification is GapClass.EVIDENCE_GAP


def test_violation_wins_over_missing_evidence() -> None:
    heavy = {"density": (meas("density", 4500, "kg/m^3", **sourced()),)}
    a = assess(profile(), cand(**heavy))
    assert a.classification is GapClass.INFEASIBLE and len(a.missing_critical) == 3


def test_adequate_only_with_an_explicit_acceptable_region() -> None:
    c = cand(**FULL)
    assert assess(profile(), c).classification is GapClass.UNCLASSIFIED
    inside = (req("tensile_strength", ">=", 450, "MPa"),)
    assert assess(profile(), c, acceptable_region=inside).classification is GapClass.ADEQUATE
    outside = (req("tensile_strength", ">=", 600, "MPa"),)
    a = assess(profile(), c, acceptable_region=outside)
    assert a.classification is GapClass.UNCLASSIFIED  # never auto-DESIGN_GAP


def test_missing_evidence_never_becomes_design_gap() -> None:
    for groups in ({}, {"density": FULL["density"]}, {**FULL, "corrosion": ()}, FULL):
        for region in (None, (req("tensile_strength", ">=", 900, "MPa"),)):
            a = assess(profile(), cand(**groups), acceptable_region=region)
            assert a.classification is not GapClass.DESIGN_GAP
            assert design_gap_preconditions(a)
    a = assess(profile(), cand(density=FULL["density"]))
    assert "evidence is not sufficient for every requirement" in design_gap_preconditions(a)


# -- acquisition priority --------------------------------------------------------------------


def test_next_evidence_puts_hard_missing_before_mismatch_before_soft() -> None:
    seawater = meas("corrosion_rate", 0.01, "mm/year", conditions={"environment": "seawater_tidal"},
                    **sourced())  # fmt: skip
    a = assess(profile(), cand(density=FULL["density"], corrosion=(seawater,)))
    assert a.next_evidence == (
        "tensile_strength",  # hard, missing (tier 1)
        "max_service_temperature",  # hard, missing (tier 1)
        "corrosion_rate [environment=marine_atmosphere]",  # hard, mismatch (tier 2)
        # soft, unweighted (tier 5); the tensile maximize gap is already listed above
        "water_absorption [environment=lab_water_immersion, exposure_duration=24 h]",
    )


def test_bottlenecks_count_only_potentially_viable_candidates() -> None:
    heavy = cand(density=(meas("density", 9000, "kg/m^3", **sourced()),))
    gap = MaterialCandidate(
        candidate_id="g", origin="existing", material=cand(**{**FULL, "temperature_effects": ()})
        .material.model_copy(update={"material_id": "g", "name": "g"}),
    )  # fmt: skip
    r = evidence_report(profile(), [heavy, gap])
    assert r.counts["infeasible"] == 1 and r.counts["evidence_gap"] == 1
    first = r.evidence_bottlenecks[0]
    assert first.evidence == "max_service_temperature" and first.candidates == ("g",)
    assert first.sole_blocker_for == ("g",)
    assert r.engineering_failures[0].requirement == "density"


# -- the real dataset and unchanged target ---------------------------------------------------

ORIGINAL_UNDETERMINED = {
    "mil5j-3.7.7.0-c", "mil5j-3.7.6.0-d", "mil5j-3.7.10.0-b", "mil5j-3.2.5.0-b",
    "mil5j-3.7.6.0-g", "mil5j-3.2.2.0-b", "mil5j-3.2.3.0-i", "mil5j-3.2.12.0-b",
    "mil5j-6.3.10.0-b", "nrl-1100", "nrl-az31x", "nrl-alclad-2024-t",
}  # fmt: skip


def test_real_search_separates_failures_from_evidence_gaps() -> None:
    cands = real_candidates()
    r = evidence_report(target(), cands)
    assert r.counts == {
        "considered": 46,
        "infeasible": 34,
        "evidence_gap": 12,
        "design_gap": 0,
        "adequate": 0,
        "unclassified": 0,
    }
    gaps = {a.candidate_id for a in r.assessments if a.classification is GapClass.EVIDENCE_GAP}
    assert gaps == ORIGINAL_UNDETERMINED
    # consistent with (and not replacing) the search result
    search = {a.candidate_id: a.feasibility for a in search_materials(target(), cands).assessments}
    assert all(search[a.candidate_id] is a.feasibility for a in r.assessments)
    top = r.evidence_bottlenecks[:2]
    assert [(b.evidence, len(b.candidates)) for b in top] == [
        ("max_service_temperature", 12),
        ("corrosion_rate [environment=marine_atmosphere]", 9),
    ]
    assert r.decision_sets[0].evidence == (
        "corrosion_rate [environment=marine_atmosphere]",
        "max_service_temperature",
    )


def test_real_7075_bar_explanation() -> None:
    r = evidence_report(target(), real_candidates())
    a = next(a for a in r.assessments if a.candidate_id == "mil5j-3.7.6.0-d")
    assert a.classification is GapClass.EVIDENCE_GAP and a.hard_coverage == (3, 5)
    assert set(a.known_properties) == {
        "density", "tensile_strength", "elongation_at_break", "youngs_modulus", "yield_strength",
    }  # fmt: skip
    assert a.next_evidence[:2] == a.missing_critical


def test_report_is_deterministic() -> None:
    one = evidence_report(target(), real_candidates())
    two = evidence_report(target(), list(reversed(real_candidates())))
    assert one.model_dump() == two.model_dump()
    ids = [a.candidate_id for a in one.assessments]
    assert format_report(one, ids) == format_report(two, ids)
