"""Classical Lamination Theory: published validation, physics sanity, semantics, provenance."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from domains.uav_materials import clt
from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.demo import example_laminate, laminate_demo_target
from domains.uav_materials.design import DesignInvalid
from domains.uav_materials.evaluation import EvaluationContext
from domains.uav_materials.micromechanics import (
    ContinuousFiberCompositeDesignSpace,
    ContinuousFiberMicromechanicsPredictor,
)
from domains.uav_materials.requirements import (
    Aspect,
    Direction,
    EngineeringRequirement,
    RequirementResolver,
    ResolutionStatus,
)
from domains.uav_materials.search import CheckStatus, search_materials
from examples.uav_material_search import target

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "uav_materials"
RP = {
    json.loads(line)["page"]: " ".join(json.loads(line)["text"].split())
    for line in (DATA / "raw/nasa_rp_1351/pages.jsonl").open(encoding="utf-8")
}
PREDICTOR = clt.LaminatePredictor()


def printed(page: int, quote: str) -> float:
    """A number printed in NASA RP-1351, read from a quote verified against the raw text."""
    assert quote in RP[page], f"p.{page}: {quote!r} not in the committed text"
    m = re.search(r"=\s*(-?[\d,. ]+?)\s*(?:lb|$)", quote)
    token = m.group(1) if m else quote
    return float(token.replace(",", "").replace(" ", ""))


def props(half: str, **kw: Any) -> dict[str, float]:
    return PREDICTOR.stiffness(clt.layup_design(half, **kw)).constants


# -- published validation: NASA RP-1351 (text layer) ------------------------------------------

E1 = printed(82, "El = 20,010,000")
E2 = printed(82, "E2 = 1,30 1,000")  # OCR splits the number; digits verified in the quote
G12 = printed(82, "G12 = 1,001,000")
NU12 = printed(82, "v12 = 0.3")
NU21 = printed(82, "v21 = 0.02")  # printed rounded; the source's Q values use this rounding
T = printed(82, "Ply thickness = 0.005")
Q_RP = clt.reduced_stiffness(E1, E2, G12, NU12, nu21=NU21)
Q45 = clt.transformed_stiffness(Q_RP, 45)


def test_rp1351_inputs_are_read_from_the_text() -> None:
    assert (E1, E2, G12, NU12, NU21, T) == (20_010_000, 1_301_000, 1_001_000, 0.3, 0.02, 0.005)


def test_reduced_stiffness_reproduces_rp1351() -> None:
    assert "20,130,785" in RP[84] and "392,656" in RP[46] and "1,308,853" in RP[46]
    assert Q_RP[0][0] == pytest.approx(20_130_785, abs=1)
    assert Q_RP[0][1] == pytest.approx(392_656, abs=1)
    assert Q_RP[1][1] == pytest.approx(1_308_853, abs=1)
    assert Q_RP[2][2] == 1_001_000
    # with the exact reciprocal nu21 = nu12*E2/E1 (our default) Q11 differs by < 0.02 %
    exact = clt.reduced_stiffness(E1, E2, G12, NU12)
    assert exact[0][0] == pytest.approx(20_130_785, rel=2e-4) and exact[0][0] != Q_RP[0][0]


def test_transformation_reproduces_rp1351_45_degree_ply() -> None:
    for quote, (i, j) in (("6,557,237", (0, 0)), ("4,555,238", (0, 1)),
                          ("4,705,483", (0, 2)), ("5,163,582", (2, 2))):  # fmt: skip
        assert quote in RP[46]
        assert Q45[i][j] == pytest.approx(float(quote.replace(",", "")), abs=1), quote
    assert Q45[1][1] == pytest.approx(Q45[0][0]) and Q45[1][2] == pytest.approx(Q45[0][2])


def test_a_and_b_reproduce_rp1351_example_9() -> None:
    """3 plies of 0.005 in; the printed B factors place 45 above, 0 at, 90 below the midplane."""
    a, b, _, h = clt.abd([(clt.transformed_stiffness(Q_RP, 90), T), (Q_RP, T), (Q45, T)])
    assert h == pytest.approx(0.015)
    assert a[0][0] == pytest.approx(printed(84, "= 139,984 lbin"), abs=1)
    assert "23,527" in RP[84] and a[0][2] == pytest.approx(23_527, abs=1)
    assert "=35828" in RP[84] and a[2][2] == pytest.approx(35_828, abs=1)
    b_printed = (
        ("= 131.2 lbin", (0, 0), 0.05),
        ("= -339.4 lb", (1, 1), 0.1),
        ("= 117.6 lb", (0, 2), 0.05),
        ("= 104 lb", (0, 1), 0.5),  # printed without decimals
    )
    for quote, (i, j), tol in b_printed:
        assert b[i][j] == pytest.approx(printed(84, quote), abs=tol), quote


def test_b_reproduces_rp1351_example_2() -> None:
    """[0/+45]T, 2 plies: the printed factors put the 0-degree ply above the midplane."""
    _, b, _, _ = clt.abd([(Q45, T), (Q_RP, T)])
    for quote, (i, j) in (("= 170 lb", (0, 0)), ("= -52 lb", (0, 1)), ("= -66 lb", (1, 1)),
                          ("= -59 lb", (0, 2))):  # fmt: skip
        assert b[i][j] == pytest.approx(printed(46, quote), abs=0.5), quote


def test_source_manifest_checksums() -> None:
    manifest = json.loads((DATA / "sources/nasa_rp_1351.json").read_text())
    for f in manifest["files"]:
        if f["committed"]:
            assert hashlib.sha256((DATA / f["path"]).read_bytes()).hexdigest() == f["sha256"]
    assert "Strength of laminated composites is not covered" in RP[106]


# -- Q-bar special cases and invariants -------------------------------------------------------


def test_qbar_special_angles_and_invariants() -> None:
    q = clt.reduced_stiffness(130e9, 8e9, 4e9, 0.28)
    q0, q90 = clt.transformed_stiffness(q, 0), clt.transformed_stiffness(q, 90)
    assert all(q0[i][j] == pytest.approx(q[i][j]) for i in range(3) for j in range(3))
    assert q90[0][0] == pytest.approx(q[1][1]) and q90[1][1] == pytest.approx(q[0][0])
    assert q90[0][1] == pytest.approx(q[0][1]) and q90[2][2] == pytest.approx(q[2][2])
    for m in (q0, q90):
        assert abs(m[0][2]) < 1e-3 and abs(m[1][2]) < 1e-3
    for theta in (15, 30, 45, 60, 75):
        qb = clt.transformed_stiffness(q, theta)
        assert qb[0][0] + qb[1][1] + 2 * qb[0][1] == pytest.approx(q[0][0] + q[1][1] + 2 * q[0][1])
        assert qb[2][2] - qb[0][1] == pytest.approx(q[2][2] - q[0][1])
        minus = clt.transformed_stiffness(q, -theta)
        assert minus[0][2] == pytest.approx(-qb[0][2]) and minus[0][0] == pytest.approx(qb[0][0])


# -- laminate physics ------------------------------------------------------------------------------


def ply() -> dict[str, float]:
    design = ContinuousFiberCompositeDesignSpace().design(
        {"fiber_id": "AS--", "matrix_id": "IMLS", "fiber_volume_fraction": 0.6},
        generation_method="test",
    )
    out = {}
    for m in ContinuousFiberMicromechanicsPredictor().predict(design).predictions:
        out[m.property] = m.value_in("Pa") if m.unit == "GPa" else m.value
    return out  # type: ignore[return-value]


def test_unidirectional_laminates_return_the_ply_constants() -> None:
    p = ply()
    zero, ninety = props("0"), props("90")
    assert zero["laminate_ex"] == pytest.approx(p["ply_longitudinal_modulus"])
    assert zero["laminate_ey"] == pytest.approx(p["ply_transverse_modulus"])
    assert zero["laminate_gxy"] == pytest.approx(p["ply_inplane_shear_modulus"])
    assert zero["laminate_nuxy"] == pytest.approx(p["ply_major_poisson_ratio"])
    assert ninety["laminate_ex"] == pytest.approx(zero["laminate_ey"])
    assert ninety["laminate_ey"] == pytest.approx(zero["laminate_ex"])
    nu21 = (
        p["ply_major_poisson_ratio"] * p["ply_transverse_modulus"] / p["ply_longitudinal_modulus"]
    )
    assert ninety["laminate_nuxy"] == pytest.approx(nu21)


def test_zero_and_ninety_dominated_laminates() -> None:
    """More 0-degree plies: Ex rises and Ey falls; the 90-dominated layup mirrors it exactly."""
    series = [props(h) for h in ("0/90", "0/0/0/90", "0")]  # 50 %, 75 %, 100 % 0-degree
    ex = [c["laminate_ex"] for c in series]
    ey = [c["laminate_ey"] for c in series]
    assert ex == sorted(ex) and ey == sorted(ey, reverse=True)
    zero, ninety = series[1], props("90/90/90/0")
    assert zero["laminate_ex"] > zero["laminate_ey"]
    assert ninety["laminate_ey"] == pytest.approx(zero["laminate_ex"])
    assert ninety["laminate_ex"] == pytest.approx(zero["laminate_ey"])


def test_cross_ply_moves_ex_and_ey_together() -> None:
    ud, cross = props("0"), props("0/90")
    assert cross["laminate_ex"] == pytest.approx(cross["laminate_ey"])
    assert (
        ud["laminate_ey"] < cross["laminate_ey"] < cross["laminate_ex"] * 1.0001 < ud["laminate_ex"]
    )
    assert cross["laminate_gxy"] == pytest.approx(ud["laminate_gxy"])  # 0/90 plies add no shear


def test_angle_plies_carry_in_plane_shear() -> None:
    sweep = {h: props(h)["laminate_gxy"] for h in ("0", "0/90", "0/45/-45/90", "45/-45")}
    assert sweep["45/-45"] == max(sweep.values()) and sweep["45/-45"] > 5 * sweep["0"]


def test_quasi_isotropic_layups_have_isotropic_membrane_stiffness() -> None:
    for half in ("0/45/-45/90", "0/60/-60"):
        c = props(half)
        assert c["laminate_ex"] == pytest.approx(c["laminate_ey"])
        assert c["laminate_gxy"] == pytest.approx(c["laminate_ex"] / (2 * (1 + c["laminate_nuxy"])))


def test_symmetric_laminates_have_zero_b_and_unsymmetric_do_not() -> None:
    for half in ("0", "0/90", "45/-45", "0/45/-45/90", "30/-60/15"):
        assert PREDICTOR.stiffness(clt.layup_design(half)).b_is_zero, half
    q = clt.reduced_stiffness(130e9, 8e9, 4e9, 0.28)
    a, b, _, h = clt.abd([(q, 1e-4), (clt.transformed_stiffness(q, 90), 1e-4)])
    assert not clt.is_symmetric_b(b, a, h)


def test_thickness_is_explicit_and_scales_a_and_d() -> None:
    thin = PREDICTOR.stiffness(clt.layup_design("0/45/-45/90", ply_thickness_mm=0.1))
    thick = PREDICTOR.stiffness(clt.layup_design("0/45/-45/90", ply_thickness_mm=0.2))
    assert thin.thickness_m == pytest.approx(8 * 0.1e-3) and thick.thickness_m == pytest.approx(
        1.6e-3
    )
    assert [p.z_bottom_m for p in thin.plies][0] == pytest.approx(-thin.thickness_m / 2)
    assert all(
        a.z_top_m == pytest.approx(b.z_bottom_m)
        for a, b in zip(thin.plies, thin.plies[1:], strict=False)
    )
    assert thick.a[0][0] == pytest.approx(2 * thin.a[0][0])
    assert thick.d[0][0] == pytest.approx(8 * thin.d[0][0])
    assert thick.constants == pytest.approx(thin.constants)  # in-plane constants: intensive
    assert thin.units == {"A": "N/m", "B": "N", "D": "N*m", "Q": "Pa", "thickness": "m"}


# -- design space ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"half": "0/abc"}, "sequence of numbers"),
        ({"half": "0/95"}, "upper bound"),
        ({"half": "/".join(["0"] * 9)}, "items"),
        ({"ply_thickness_mm": 1.0}, "upper bound"),
        ({"vf": 0.8}, "upper bound"),
    ],
)
def test_invalid_laminate_designs_are_rejected(changes: dict[str, Any], problem: str) -> None:
    kw: dict[str, Any] = {"half": "0/90", "vf": 0.6, "ply_thickness_mm": 0.127, **changes}
    with pytest.raises(DesignInvalid, match=problem):
        clt.layup_design(**kw)


def test_design_records_every_variable() -> None:
    d = clt.layup_design("0/45/-45/90", ply_thickness_mm=0.127)
    assert d.variables == {
        "fiber_id": "AS--", "matrix_id": "IMLS", "fiber_volume_fraction": 0.6,
        "half_stack_deg": "0/45/-45/90", "ply_thickness_mm": 0.127,
    }  # fmt: skip
    assert d.design_space == "continuous_fiber_laminate" and d.design_space_version == "1"
    lam = PREDICTOR.stiffness(d)
    assert [p.angle_deg for p in lam.plies] == [0, 45, -45, 90, 90, -45, 45, 0]


# -- provenance chain, micromechanics unchanged -------------------------------------------------


def test_provenance_chain_reaches_the_constituent_documents() -> None:
    cand = example_laminate()
    m = cand.material.measurements("laminate_ex")[0]
    assert m.basis == "predicted" and m.provenance is not None
    levels = [s.metadata["chain_level"][0] for s in m.provenance.sources]
    assert levels[:2] == ["1", "2"] and set(levels) == {"1", "2", "3", "4"}
    kinds = {s.kind for s in m.provenance.sources}
    assert kinds == {"model", "ply_stack", "ply_prediction", "constituent"}
    direct = {
        p.property: p
        for p in ContinuousFiberMicromechanicsPredictor()
        .predict(PREDICTOR.ply_design(clt.layup_design("0/45/-45/90")))
        .predictions
    }
    for s in m.provenance.sources:
        if s.kind == "ply_prediction":
            prop = s.identifier.split(":")[1]
            assert s.metadata["value"] == direct[prop].value  # micromechanics unchanged
            assert (
                direct[prop].provenance
                and s.metadata["provenance_id"] == direct[prop].provenance.id
            )
        if s.kind == "constituent":
            assert s.metadata["documents"] and s.metadata["provenance_id"]
    lam = cand.metadata["laminate"]
    assert lam["layup"] == "[0/45/-45/90]s" and len(lam["plies"]) == 8 and lam["b_is_zero"]


def test_micromechanics_v1_outputs_unchanged() -> None:
    design = ContinuousFiberCompositeDesignSpace().design(
        {"fiber_id": "AS--", "matrix_id": "IMLS", "fiber_volume_fraction": 0.6},
        generation_method="test",
    )
    p = {
        m.property: m.value
        for m in ContinuousFiberMicromechanicsPredictor().predict(design).predictions
    }
    assert p["ply_longitudinal_modulus"] == pytest.approx(129.6214, abs=1e-4)
    assert p["density"] == pytest.approx(1555.6106, abs=1e-4)


# -- requirement semantics and unified evaluation ---------------------------------------------

LAM = "continuous_fiber_laminate"


def resolve(aspect: Aspect, direction: Direction, unit: str = "GPa") -> Any:
    er = EngineeringRequirement(aspect=aspect, direction=direction, operator=">=", value=1,
                                unit=unit, priority="hard")  # fmt: skip
    return RequirementResolver().resolve(er, LAM)


@pytest.mark.parametrize(
    ("aspect", "direction", "prop"),
    [
        (Aspect.NORMAL_STIFFNESS, Direction.LAMINATE_X, "laminate_ex"),
        (Aspect.NORMAL_STIFFNESS, Direction.LAMINATE_Y, "laminate_ey"),
        (Aspect.SHEAR_STIFFNESS, Direction.LAMINATE_XY, "laminate_gxy"),
    ],
)
def test_laminate_directions_resolve(aspect: Aspect, direction: Direction, prop: str) -> None:
    res = resolve(aspect, direction)
    assert res.status is ResolutionStatus.RESOLVED and res.properties == (prop,)


def test_directionless_or_ply_axis_or_strength_requirements_do_not_resolve() -> None:
    assert (
        resolve(Aspect.NORMAL_STIFFNESS, Direction.UNSPECIFIED).status is ResolutionStatus.AMBIGUOUS
    )
    assert (
        resolve(Aspect.NORMAL_STIFFNESS, Direction.MATERIAL_1).status
        is ResolutionStatus.UNSUPPORTED
    )
    strength = resolve(Aspect.TENSILE_ULTIMATE, Direction.LAMINATE_X, unit="MPa")
    assert strength.status is ResolutionStatus.UNSUPPORTED
    assert "ply S11T is not a laminate strength" in strength.reason


def test_quasi_isotropic_laminate_is_not_assumed_isotropic() -> None:
    uav = EvaluationContext.fit(target(), real_candidates())
    e = uav.evaluate(example_laminate("0/45/-45/90"))
    modulus = next(o for o in e.outcomes if o.requirement.startswith("normal_stiffness"))
    assert modulus.resolution is ResolutionStatus.AMBIGUOUS
    gaps = {o.requirement: o.gap for o in e.outcomes if o.hard}
    assert gaps == {
        "density <= 3000 kg/m^3": None,
        "tensile_ultimate_strength >= 350 MPa": "unsupported",
        "ductility >= 5 %": "unsupported",
        "corrosion_penetration_rate <= 0.025 mm/year": "unsupported",
        "max_service_temperature >= 80 degC": "unsupported",
    }
    assert e.feasibility == "undetermined" and e.material_system == LAM


def test_direction_aware_laminate_target() -> None:
    demo = EvaluationContext.fit(laminate_demo_target(), real_candidates())
    qi = demo.evaluate(example_laminate("0/45/-45/90"))
    used = {c.requirement.property: (c.status, c.value) for c in qi.checks}
    assert used["laminate_ex"][0] is CheckStatus.SATISFIED and used["laminate_ex"][
        1
    ] == pytest.approx(49.33, abs=0.01)
    assert used["laminate_ey"][0] is CheckStatus.SATISFIED
    assert used["laminate_gxy"][0] is CheckStatus.SATISFIED
    assert qi.feasibility == "feasible" and qi.evidence.classification == "unclassified"
    cross = demo.evaluate(example_laminate("0/90"))
    assert cross.feasibility == "infeasible" and cross.evidence.violated == ("laminate_gxy",)


def test_existing_results_and_determinism() -> None:
    r = search_materials(target(), real_candidates())
    assert r.counts == {"considered": 46, "feasible": 0, "infeasible": 34, "undetermined": 12}
    a = [m.model_dump() for m in PREDICTOR.predict(clt.layup_design("0/45/-45/90")).predictions]
    b = [
        m.model_dump()
        for m in clt.LaminatePredictor().predict(clt.layup_design("0/45/-45/90")).predictions
    ]
    for x, y in zip(a, b, strict=True):
        x["provenance"].pop("created_at"), y["provenance"].pop("created_at")
        assert x == y
    assert math.isfinite(a[0]["value"])
