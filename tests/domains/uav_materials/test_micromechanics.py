"""Continuous-fibre ply micromechanics: sources, equations, physics sanity, evaluation (offline)."""

from __future__ import annotations

import ast
import math
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from domains.uav_materials import micromechanics as mm
from domains.uav_materials.constituents import (
    AS_SPECS,
    TP3290,
    RenderingNotFound,
    ValueNotSupported,
    ValueSpec,
    accept,
    build_constituents,
    constituent_library,
    load_pages,
    read_number,
)
from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.design import CandidateDesign, DesignInvalid, PropertyPredictor
from domains.uav_materials.evaluation import EvaluationContext
from domains.uav_materials.schema import EvidenceType
from domains.uav_materials.search import CheckStatus, search_materials
from examples.uav_material_search import target

ROOT = Path(__file__).resolve().parents[3]
LIB = constituent_library()
AS, IMLS = LIB["AS--"], LIB["IMLS"]
SPACE = mm.ContinuousFiberCompositeDesignSpace()
PREDICTOR = mm.ContinuousFiberMicromechanicsPredictor()


def psi(c: Any, name: str) -> float:
    return float(c.get(name).value_in("psi"))


def design(vf: float, fiber: str = "AS--", matrix: str = "IMLS") -> CandidateDesign:
    return SPACE.design(
        {"fiber_id": fiber, "matrix_id": matrix, "fiber_volume_fraction": vf},
        generation_method="test",
    )


def predicted(vf: float) -> dict[str, Any]:
    return {m.property: m for m in PREDICTOR.predict(design(vf)).predictions}


# -- constituent sources -----------------------------------------------------------------------


def test_constituent_values_are_the_agreed_renderings() -> None:
    assert (AS.get("density").value, AS.get("density").unit) == (0.063, "lb/in^3")
    assert psi(AS, "axial_modulus") == 31e6 and psi(AS, "transverse_modulus") == 2e6
    assert psi(AS, "inplane_shear_modulus") == 2e6 and AS.get("major_poisson_ratio").value == 0.2
    assert psi(AS, "tensile_strength") == 400e3
    assert (IMLS.get("density").value, psi(IMLS, "modulus")) == (0.046, 5e5)
    assert IMLS.get("poisson_ratio").value == 0.41
    assert psi(IMLS, "shear_modulus") == pytest.approx(0.1773e6, rel=1e-3)  # ICAN echo GMP


@pytest.mark.parametrize(
    ("token", "value"),
    [("0.3100E 08", 3.1e7), ("O.2000E 07", 2e6), ("0.200OE 00", 0.2), ("0.4100E+00", 0.41),
     ("0.046", 0.046), ("0.310E+OS", None), ("0,2000E 07", None), ("0 ._600E-01", None)],
)  # fmt: skip
def test_ocr_numbers_are_read_with_unambiguous_repairs_only(token: str, value: float) -> None:
    assert read_number(token) == value


def test_disagreements_and_unreadable_renderings_are_kept() -> None:
    sft = AS.get("tensile_strength").provenance.metadata
    assert sft["conflicting"] == ["NASA-TP-2515 p.61: '0.4000E 05'"]  # OCR read 40 ksi
    assert len(sft["agreeing"]) == 3
    ef11 = AS.get("axial_modulus").provenance.metadata
    assert ef11["unreadable"] == ["NASA-TP-3290 p.27: '0.310E+OS'"]  # 'S' may be 5 or 8


def test_a_value_needs_two_documents() -> None:
    pages = load_pages()
    spec = AS_SPECS[0]
    single = ValueSpec(spec.name, spec.symbol, spec.unit,
                       tuple(r for r in spec.renderings if r.document == TP3290))  # fmt: skip
    with pytest.raises(ValueNotSupported):
        accept(pages, single)


def test_a_quote_outside_its_block_or_page_is_rejected() -> None:
    pages = load_pages()
    r = AS_SPECS[1].renderings[1]  # AS echo on TP-3290 p.40
    with pytest.raises(RenderingNotFound):
        accept(pages, replace(AS_SPECS[1], renderings=(replace(r, page=41),)))
    moved = replace(AS_SPECS[1].renderings[0], block=("SGLA S- GLASS FIBER.", "OVER"))
    with pytest.raises(RenderingNotFound):
        accept(pages, replace(AS_SPECS[1], renderings=(moved,)))
    tampered = dict(pages)
    key = (TP3290, 40)
    tampered[key] = tampered[key].replace("EFP1 0.3100E+08", "EFP1 0.3200E+08")
    with pytest.raises(RenderingNotFound):
        build_constituents(tampered)


def test_every_constituent_value_is_sourced_by_two_documents() -> None:
    for c in LIB.values():
        for p in c.properties.values():
            assert "indicative" in p.data_quality and p.conditions["state"] == "dry"
            if p.evidence_type is EvidenceType.DERIVED:
                assert p.equation and p.provenance.metadata["derived_from"]
                continue
            assert p.evidence_type is EvidenceType.DATASHEET
            docs = {s.identifier for s in p.provenance.sources}
            assert len(docs) >= 2 and all(s.metadata["quote"] for s in p.provenance.sources)


# -- equations reproduce the sources' worked numbers -------------------------------------------


def test_equations_reproduce_tm83320_example_4_1_and_4_2() -> None:
    """AS/IMLS, k_f = 0.63, zero voids: E11 19.7, E22 1.24, G12 0.64 (10^6 psi); nu12 0.278."""
    kf = 0.63
    assert mm.longitudinal_modulus(kf, psi(AS, "axial_modulus"), psi(IMLS, "modulus")) / 1e6 == (
        pytest.approx(19.7, abs=0.05)
    )
    e22 = mm.transverse_modulus(kf, psi(AS, "transverse_modulus"), psi(IMLS, "modulus"))
    assert e22 / 1e6 == pytest.approx(1.24, abs=0.005)
    g12 = mm.inplane_shear_modulus(kf, psi(AS, "inplane_shear_modulus"), psi(IMLS, "shear_modulus"))
    assert g12 / 1e6 == pytest.approx(0.64, abs=0.005)
    nu = mm.major_poisson_ratio(
        kf, AS.get("major_poisson_ratio").value, IMLS.get("poisson_ratio").value
    )
    assert nu == pytest.approx(0.278, abs=0.0005)


def test_equations_reproduce_tm83320_examples_7_2_and_8_2() -> None:
    rho = mm.density(0.58, AS.get("density").value, IMLS.get("density").value)
    assert rho == pytest.approx(0.056, abs=0.0005)  # lb/in^3, example 7.2 (k_m = 0.42)
    e11 = mm.longitudinal_modulus(0.6, psi(AS, "axial_modulus"), psi(IMLS, "modulus"))
    assert e11 / 1e6 == pytest.approx(18.8, abs=0.05)  # example 8.2, dry


def test_equations_reproduce_ican_sample_output() -> None:
    """TP-2515 sample: AS/IMLS, k_f 0.55, k_m 0.43 (2 % voids): E11, G12, S11T."""
    kf, km = 0.55, 0.43
    ef11, em = psi(AS, "axial_modulus"), psi(IMLS, "modulus")
    assert mm.longitudinal_modulus(kf, ef11, em, km) == pytest.approx(0.1726e8, rel=5e-4)
    g12 = mm.inplane_shear_modulus(kf, psi(AS, "inplane_shear_modulus"), psi(IMLS, "shear_modulus"))
    assert g12 == pytest.approx(0.5470e6, rel=5e-4)
    s11t = mm.longitudinal_tensile_strength(kf, psi(AS, "tensile_strength"), em, ef11, km)
    assert s11t == pytest.approx(0.2228e6, rel=5e-4)


# -- physics sanity ------------------------------------------------------------------------------

SWEEP = [0.50, 0.53, 0.56, 0.59, 0.62, 0.65]


def test_density_lies_between_constituent_densities() -> None:
    rho_f, rho_m = AS.get("density").value_in("kg/m^3"), IMLS.get("density").value_in("kg/m^3")
    for vf in SWEEP:
        assert rho_m < predicted(vf)["density"].value < rho_f


def test_stiffness_and_strength_increase_monotonically_with_vf() -> None:
    series = [predicted(vf) for vf in SWEEP]
    for prop in ("density", "ply_longitudinal_modulus", "ply_transverse_modulus",
                 "ply_inplane_shear_modulus", "ply_longitudinal_tensile_strength"):  # fmt: skip
        values = [s[prop].value for s in series]
        assert values == sorted(values) and len(set(values)) == len(values), prop
    nus = [s["ply_major_poisson_ratio"].value for s in series]  # nu_m > nu_f: decreases
    assert nus == sorted(nus, reverse=True)


def test_boundary_behaviour_of_the_equations() -> None:
    ef11, ef22, em = psi(AS, "axial_modulus"), psi(AS, "transverse_modulus"), psi(IMLS, "modulus")
    gf12, gm = psi(AS, "inplane_shear_modulus"), psi(IMLS, "shear_modulus")
    assert (
        mm.longitudinal_modulus(0, ef11, em) == em and mm.longitudinal_modulus(1, ef11, em) == ef11
    )
    assert mm.transverse_modulus(0, ef22, em) == em
    assert mm.transverse_modulus(1, ef22, em) == pytest.approx(ef22)
    assert mm.inplane_shear_modulus(0, gf12, gm) == gm
    assert mm.inplane_shear_modulus(1, gf12, gm) == pytest.approx(gf12)
    e11 = mm.longitudinal_modulus(0.6, ef11, em)
    assert e11 > mm.transverse_modulus(0.6, ef22, em)  # strongly anisotropic ply


def test_failure_strain_is_the_fibre_failure_strain() -> None:
    expected = psi(AS, "tensile_strength") / psi(AS, "axial_modulus") * 100
    for vf in SWEEP:
        assert predicted(vf)["ply_longitudinal_tensile_failure_strain"].value == pytest.approx(
            expected
        )


@pytest.mark.parametrize("vf", [0.45, 0.66, 0.0, 1.0, 1.2, -0.1, float("nan")])
def test_invalid_volume_fraction_is_rejected(vf: float) -> None:
    with pytest.raises(DesignInvalid):
        design(vf)


def test_invalid_constituent_choice_is_rejected() -> None:
    with pytest.raises(DesignInvalid, match="not one of"):
        design(0.6, fiber="IMLS")
    with pytest.raises(DesignInvalid, match="not one of"):
        design(0.6, matrix="EPOXY-X")


def test_design_records_identity_fraction_version_and_method() -> None:
    d = design(0.6)
    assert d.variables == {"fiber_id": "AS--", "matrix_id": "IMLS", "fiber_volume_fraction": 0.6}
    assert (d.design_space, d.design_space_version, d.generation_method) == (
        "continuous_fiber_ud_ply", "1", "test",
    )  # fmt: skip
    vf_spec = next(v for v in d.variable_specs if v.name == "fiber_volume_fraction")
    assert (
        vf_spec.lower,
        vf_spec.upper,
    ) == mm.VF_RANGE and "modeling assumption" in vf_spec.description


# -- units, directionality, provenance, unsupported ------------------------------------------


def test_units() -> None:
    p = predicted(0.6)
    assert {k: m.unit for k, m in p.items()} == {
        "density": "kg/m^3", "ply_longitudinal_modulus": "GPa", "ply_transverse_modulus": "GPa",
        "ply_inplane_shear_modulus": "GPa", "ply_major_poisson_ratio": "1",
        "ply_longitudinal_tensile_strength": "MPa", "ply_longitudinal_tensile_failure_strain": "%",
    }  # fmt: skip
    assert p["density"].value_in("lb/in^3") == pytest.approx(0.6 * 0.063 + 0.4 * 0.046)
    assert p["ply_longitudinal_modulus"].value_in("psi") == pytest.approx(0.6 * 31e6 + 0.4 * 5e5)
    assert p["ply_longitudinal_tensile_strength"].value_in("ksi") == pytest.approx(
        400 * (0.6 + 0.4 * 0.5 / 31)
    )


def test_directionality_is_kept() -> None:
    p = predicted(0.6)
    assert not {"youngs_modulus", "tensile_strength", "elongation_at_break"} & set(p)
    assert p["ply_longitudinal_modulus"].conditions.other["direction"].startswith("1 (fibre axis)")
    assert p["ply_transverse_modulus"].conditions.other["direction"].startswith("2")
    assert p["ply_inplane_shear_modulus"].conditions.other["direction"].startswith("1-2")


def test_provenance_propagates_from_constituents_to_candidate() -> None:
    c = SPACE.candidate_from_design(design(0.6))
    assert c.origin == "composite" and c.parents == ("constituent:AS--", "constituent:IMLS")
    m = c.material.measurements("ply_longitudinal_modulus")[0]
    assert m.basis == "predicted" and m.provenance_status == "sourced"
    assert m.provenance is not None
    model, *inputs = m.provenance.sources
    assert (model.kind, model.identifier) == ("model", "continuous-fiber-micromechanics/1")
    assert model.metadata["equation"] == "E11 = kf*Ef11 + km*Em"
    assert "TM-83320" in model.metadata["equation_source"] and model.metadata["limitations"]
    assert {s.identifier for s in inputs} == {"AS--:Ef11", "IMLS:Em"}
    assert all(s.kind == "constituent" and len(s.metadata["documents"]) >= 2 for s in inputs)
    assert c.metadata["model"]["validation"]["status"].startswith("implementation_checked")
    assert c.metadata["model"]["unsupported"]


def test_unsupported_properties_stay_unsupported() -> None:
    profile = PREDICTOR.predict(design(0.6))
    assert set(profile.not_predicted) >= {
        "corrosion_rate", "water_absorption", "max_service_temperature", "elongation_at_break",
    }  # fmt: skip
    produced = {m.property for m in profile.predictions}
    assert not produced & set(profile.not_predicted)


def test_predictor_is_deterministic_and_satisfies_the_contract() -> None:
    assert isinstance(PREDICTOR, PropertyPredictor)
    a, b = (
        PREDICTOR.predict(design(0.6)),
        mm.ContinuousFiberMicromechanicsPredictor().predict(design(0.6)),
    )
    assert [(m.property, m.value, m.provenance and m.provenance.id) for m in a.predictions] == [
        (m.property, m.value, m.provenance and m.provenance.id) for m in b.predictions
    ]


# -- unified evaluation, regression ------------------------------------------------------------


def test_unified_evaluation_against_the_unchanged_target() -> None:
    context = EvaluationContext.fit(target(), real_candidates())
    e = context.evaluate(SPACE.candidate_from_design(design(0.6)))
    status = {c.requirement.property: c.status for c in e.checks}
    assert status == {
        "density": CheckStatus.SATISFIED,
        "tensile_strength": CheckStatus.UNDETERMINED,  # a ply strength is directional
        "elongation_at_break": CheckStatus.UNDETERMINED,  # failure strain is not elongation
        "corrosion_rate": CheckStatus.UNDETERMINED,
        "max_service_temperature": CheckStatus.UNDETERMINED,
    }
    assert e.feasibility == "undetermined" and e.evidence.classification == "evidence_gap"
    assert e.prediction_status == "all" and e.engineering_valid is True
    assert e.models == ("continuous-fiber-micromechanics/1",)
    density = next(r for r in e.evidence.requirements if r.property == "density" and r.hard)
    assert density.sufficient and density.evidence_type == "predicted"
    youngs = next(c for c in e.contributions if c.property == "youngs_modulus")
    assert youngs.value is None  # E11 is not an isotropic Young's modulus


def test_existing_material_results_unchanged() -> None:
    r = search_materials(target(), real_candidates())
    assert r.counts == {"considered": 46, "feasible": 0, "infeasible": 34, "undetermined": 12}


def test_offline_and_no_llm() -> None:
    for name in ("micromechanics.py", "constituents.py"):
        tree = ast.parse((ROOT / "domains/uav_materials" / name).read_text())
        imported = {
            (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(tree)
            if isinstance(n, ast.Import | ast.ImportFrom)
            for a in n.names
        }
        assert not {i for i in imported if "anthropic" in i or "typesafe" in i or "llm" in i}


def test_constituents_build_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    lib = build_constituents(load_pages())
    assert set(lib) == {"AS--", "IMLS"}
    assert math.isfinite(
        mm.ContinuousFiberMicromechanicsPredictor(lib).predict(design(0.6)).predictions[0].value
        or 0
    )


def test_committed_raw_text_matches_manifest() -> None:
    import hashlib
    import json

    data = ROOT / "data" / "uav_materials"
    manifest = json.loads((data / "sources" / "nasa_micromechanics.json").read_text())
    committed = [f for f in manifest["files"] if f["committed"]]
    assert committed
    for f in committed:
        assert hashlib.sha256((data / f["path"]).read_bytes()).hexdigest() == f["sha256"]
    assert manifest["reuse"]["redistribution_of_extracts"] == "permitted"
