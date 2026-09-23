"""Temperature evidence: statement parsing, units, semantics, identity, selection (offline)."""

from __future__ import annotations

import json
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from domains.uav_materials import MaterialCandidate, MaterialRecord
from domains.uav_materials.compatibility import Compat, compatibility
from domains.uav_materials.datasets import load_real_materials, real_candidates
from domains.uav_materials.identity import IdentityIndex, MatchLevel
from domains.uav_materials.ingest import build as build_mod
from domains.uav_materials.ingest import temperature_statements as ts
from domains.uav_materials.schema import TemperatureRegime, TestConditions
from domains.uav_materials.search import CheckStatus, Feasibility, check, search_materials
from domains.uav_materials.selection import select
from domains.uav_materials.units import UnitError, convert
from examples.uav_material_search import target

from .helpers import meas, req

DATA = Path(__file__).resolve().parents[3] / "data" / "uav_materials"
PAGES = build_mod.load_statement_pages()
MATS = {m.material_id: m for m in load_real_materials()}
STATEMENT = {s.key: s for s in ts.STATEMENTS}
# The 12 candidates that were UNDETERMINED before this iteration.
ORIGINAL_UNDETERMINED = (
    "mil5j-3.7.7.0-c",
    "mil5j-3.7.6.0-d",
    "mil5j-3.7.10.0-b",
    "mil5j-3.2.5.0-b",
    "mil5j-3.7.6.0-g",
    "mil5j-3.2.2.0-b",
    "mil5j-3.2.3.0-i",
    "mil5j-3.2.12.0-b",
    "mil5j-6.3.10.0-b",
    "nrl-1100",
    "nrl-az31x",
    "nrl-alclad-2024-t",
)


def record(mid: str, *temps: Any) -> MaterialRecord:
    return MaterialRecord(
        material_id=mid, name=mid, family="metal", provenance_status="synthetic",
        temperature_effects=temps,
    )  # fmt: skip


def at(value: float, unit: str = "degC") -> dict[str, Any]:
    return {"temperature": {"value": value, "unit": unit}}


def temp_meas(prop: str, value: float, unit: str = "degC", **kw: Any) -> Any:
    return meas(prop, value, unit, **kw)


# -- units --------------------------------------------------------------------------------


def test_fahrenheit_conversion_is_affine() -> None:
    assert convert(500, "degF", "degC") == pytest.approx(260.0)
    assert convert(32, "degF", "degC") == pytest.approx(0.0, abs=1e-9)
    assert convert(-40, "degF", "degC") == pytest.approx(-40.0)
    assert convert(80, "degC", "degF") == pytest.approx(176.0)
    assert convert(477, "degC", "K") == pytest.approx(750.15)


def test_temperature_does_not_convert_to_other_dimensions() -> None:
    with pytest.raises(UnitError):
        convert(500, "degF", "MPa")
    with pytest.raises(ValueError, match="is a temperature"):  # UnitError inside validation
        meas("max_service_temperature", 500, "ksi")


# -- statement parsing -----------------------------------------------------------------


def test_parse_temperature_reads_number_and_unit_from_the_quote() -> None:
    assert ts.parse_temperature("is 500(F for up to 100 hours.", "500") == (500.0, None, "degF")
    assert ts.parse_temperature("(up to 900 (F or to 1100 (F", "1100") == (1100.0, None, "degF")
    assert ts.parse_temperature("approximately 477 to 638°C.", "477", "638") == (
        477.0,
        638.0,
        "degC",
    )
    assert ts.parse_temperature("approximately 477 to 638C.", "477", "638")[2] == "degC"


@pytest.mark.parametrize(
    ("quote", "number"),
    [
        ("strength of 500 ksi", "500"),  # not a temperature
        ("is 1500(F", "500"),  # only as part of a larger number
        ("up to 900 hours", "900"),  # no unit
    ],
)
def test_parse_temperature_rejects_non_temperatures(quote: str, number: str) -> None:
    with pytest.raises(ts.QuoteNotFound):
        ts.parse_temperature(quote, number)


def test_every_statement_verifies_against_the_committed_raw_text() -> None:
    for s in ts.STATEMENTS:
        ts.verify(PAGES, s)
    for n in ts.NOT_USED:
        ts.verify_not_used(PAGES, n)


def test_a_quote_that_is_not_on_the_cited_page_is_rejected() -> None:
    s = STATEMENT["c17200-max-service"]
    moved = replace(s, citations=(replace(s.citations[0], page=1191),))
    with pytest.raises(ts.QuoteNotFound):
        ts.verify(PAGES, moved)
    edited = replace(s, citations=(replace(s.citations[0], quote=s.citations[0].quote + "x"),))
    with pytest.raises(ts.QuoteNotFound):
        ts.verify(PAGES, edited)


def test_ocr_value_requires_both_editions_to_agree() -> None:
    s = STATEMENT["7075-melting-range"]
    assert {c.document for c in s.citations} == {ts.NASA_1966, ts.NASA_1972}
    assert ts.verify(PAGES, s) == (477.0, 638.0, "degC")
    key = (ts.NASA_1966, 19)
    for old, new in (("477 to 638C", "471 to 638C"), ("477 to 638C", "477 to 638F")):
        pages = {**PAGES, key: PAGES[key].replace(old, new)}
        quote = s.citations[1].quote.replace(old, new)
        edited = replace(s, citations=(s.citations[0], replace(s.citations[1], quote=quote)))
        with pytest.raises((ts.QuoteNotFound, ts.OcrDisagreement)) as err:
            ts.verify(pages, edited)
        # a different number is not found where expected; a different unit disagrees
        assert err.type is (ts.QuoteNotFound if "471" in new else ts.OcrDisagreement)


def test_committed_raw_extracts_match_their_manifest_checksums() -> None:
    import hashlib

    for manifest in ("mil_hdbk_5j.json", "nasa_mdh_7075.json"):
        doc = json.loads((DATA / "sources" / manifest).read_text())
        for f in doc["files"]:
            if f.get("committed") and "sha256" in f:
                digest = hashlib.sha256((DATA / f["path"]).read_bytes()).hexdigest()
                assert digest == f["sha256"], f["path"]


# -- provenance ----------------------------------------------------------------------------


def test_statement_measurements_carry_quote_page_and_identity_match() -> None:
    m = next(
        m
        for m in MATS["mil5j-3.7.6.0-d"].temperature_effects
        if m.property == "melting_temperature"
    )
    assert m.provenance is not None and m.provenance_status == "sourced"
    docs = [(s.identifier, s.metadata["page"]) for s in m.provenance.sources]
    assert docs == [(ts.NASA_1972, 18), (ts.NASA_1966, 19)]
    assert all("477" in s.metadata["quote"] for s in m.provenance.sources)
    assert m.provenance.metadata["identity_match"] == "alloy"
    assert m.provenance.metadata["cross_validated_by"] == [ts.NASA_1966]
    assert (m.value, m.unit, m.conditions.other["range_upper"]) == (477, "degC", "638 degC")


def test_one_statement_on_several_records_gets_distinct_provenance_ids() -> None:
    ids = {
        m.provenance.id
        for mid in ("mil5j-7.3.2.0-d", "mil5j-7.3.2.0-e", "mil5j-7.3.2.0-g", "mil5j-7.3.4.0-f")
        for m in MATS[mid].temperature_effects
        if m.provenance
    }
    assert len(ids) == 4


def test_existing_measurements_are_not_overwritten() -> None:
    rec = MATS["mil5j-3.7.6.0-d"]
    assert rec.status("tensile_strength") == "measured"
    assert all(m.provenance and m.provenance.sources[0].identifier == "MIL-HDBK-5J"
               for m in rec.strength)  # fmt: skip


# -- identity -----------------------------------------------------------------------------


def test_laminate_statements_do_not_attach_to_monolithic_alloys() -> None:
    """'The maximum service temperature is 200 F' belongs to aramid/aluminium laminates."""
    index = IdentityIndex([m.identity for m in MATS.values() if m.identity])
    for key in ("2024-t3-laminate-max-service", "7475-t761-laminate-max-service"):
        assert index.resolve(STATEMENT[key].material) == []
    # 7475 sheet lists temper T761 and 2024 extrusion lists T3: still nothing attached.
    for mid in ("mil5j-3.7.10.0-b", "mil5j-3.2.3.0-j"):
        assert MATS[mid].measurements("max_service_temperature") == ()


def test_az31b_statement_does_not_attach_to_az31x() -> None:
    assert MATS["nrl-az31x"].temperature_effects == ()
    report = build_mod.build()[1]["temperature_statements"]
    assert {u["statement"] for u in report["unresolved"]} == {
        "2024-t3-laminate-max-service",
        "7475-t761-laminate-max-service",
        "az31b-application",
    }


def test_designation_fallback_matches_whole_designations_at_alloy_level_only() -> None:
    index = IdentityIndex([m.identity for m in MATS.values() if m.identity])
    hits = index.resolve("C17200 copper beryllium")
    assert len(hits) == 4 and {level for _, level in hits} == {MatchLevel.ALLOY}
    assert [i.designation for i, _ in index.resolve("inconel  x-750")] == ["Inconel X-750"]
    assert index.resolve("Inconel") == [] and index.resolve("C172") == []


def test_melting_range_links_to_bare_7075_but_not_clad_7075() -> None:
    linked = {mid for mid, m in MATS.items() if m.measurements("melting_temperature")}
    assert linked == {"mil5j-3.7.6.0-b", "mil5j-3.7.6.0-d", "mil5j-3.7.6.0-f", "mil5j-3.7.6.0-g"}


# -- conditions and selection -------------------------------------------------------------


def test_room_and_elevated_temperature_values_are_kept_apart() -> None:
    room = TestConditions(temperature_regime=TemperatureRegime.ROOM)
    hot = TestConditions(temperature={"value": 300, "unit": "degF"})
    assert compatibility(TestConditions(), room)[0] is Compat.UNCONDITIONED
    assert compatibility(TestConditions(), hot)[0] is Compat.INCOMPATIBLE
    at_149c = TestConditions(temperature={"value": 149, "unit": "degC"})
    assert compatibility(at_149c, hot)[0] is Compat.EXACT  # 300 F = 148.9 C
    assert compatibility(at_149c, room)[0] is Compat.INCOMPATIBLE
    at_23c = TestConditions(temperature={"value": 23, "unit": "degC"})
    assert compatibility(at_23c, room)[0] is Compat.COMPATIBLE


def test_temperature_dependent_strength_selects_the_matching_temperature() -> None:
    rec = MaterialRecord(
        material_id="syn",
        name="syn",
        family="metal",
        provenance_status="synthetic",
        strength=(
            meas("tensile_strength", 500, "MPa", conditions={"temperature_regime": "room"}),
            meas("tensile_strength", 300, "MPa", conditions=at(300, "degF")),
            meas("tensile_strength", 400, "MPa", conditions=at(100)),
        ),
    )
    general = select(rec, req("tensile_strength", ">=", 350, "MPa"))
    assert general.chosen is not None and general.chosen.value == 500
    hot = select(
        rec,
        req("tensile_strength", ">=", 350, "MPa",
            conditions={"temperature": {"value": 150, "unit": "degC"}}),
    )  # fmt: skip
    assert hot.chosen is not None and hot.chosen.value == 300  # 300 F = 148.9 C, within 5 K
    none = select(
        rec,
        req("tensile_strength", ">=", 350, "MPa",
            conditions={"temperature": {"value": 250, "unit": "degC"}}),
    )  # fmt: skip
    assert none.chosen is None


def test_modulus_retention_needs_the_stated_temperature() -> None:
    rec = record(
        "syn",
        temp_meas("modulus_retention", 90, "%",
                  conditions={"temperature": {"value": 100, "unit": "degC"}}),
        temp_meas("modulus_retention", 70, "%",
                  conditions={"temperature": {"value": 200, "unit": "degC"}}),
    )  # fmt: skip
    at_200 = req("modulus_retention", ">=", 75, "%",
                 conditions={"temperature": {"value": 200, "unit": "degC"}})  # fmt: skip
    cand = MaterialCandidate(candidate_id="syn", origin="existing", material=rec)
    k = check(at_200, cand)
    assert (k.status, k.value) == (CheckStatus.VIOLATED, 70)


# -- max-service-temperature semantics ---------------------------------------------------


def _msr(rec: MaterialRecord) -> CheckStatus:
    cand = MaterialCandidate(candidate_id=rec.material_id, origin="existing", material=rec)
    return check(req("max_service_temperature", ">=", 80, "degC"), cand).status


def test_melting_point_cannot_satisfy_max_service_temperature() -> None:
    """Regression: a melting point (even 477 C) never answers a service-temperature limit."""
    assert _msr(record("syn", temp_meas("melting_temperature", 477))) is CheckStatus.UNDETERMINED
    real = MATS["mil5j-3.7.6.0-d"]  # 7075 bar: has the NASA melting range
    assert real.status("melting_temperature") == "measured"
    assert _msr(real) is CheckStatus.UNDETERMINED


def test_other_temperature_kinds_cannot_satisfy_max_service_temperature() -> None:
    for prop in ("application_temperature_limit", "exposure_stability_temperature",
                 "glass_transition_temperature"):  # fmt: skip
        assert _msr(record("syn", temp_meas(prop, 400))) is CheckStatus.UNDETERMINED
    assert _msr(MATS["mil5j-6.3.6.0-b"]) is CheckStatus.UNDETERMINED  # Inconel X-750
    assert _msr(MATS["mil5j-5.4.1.0-b"]) is CheckStatus.UNDETERMINED  # Ti-6Al-4V


def test_max_service_temperature_is_compared_across_units() -> None:
    assert _msr(record("syn", temp_meas("max_service_temperature", 200, "degF"))) is (
        CheckStatus.SATISFIED
    )  # 93.3 C
    assert _msr(record("syn", temp_meas("max_service_temperature", 170, "degF"))) is (
        CheckStatus.VIOLATED
    )  # 76.7 C


def test_time_limited_service_temperature_does_not_answer_a_continuous_requirement() -> None:
    limited = temp_meas(
        "max_service_temperature", 500, "degF",
        conditions={"exposure_duration": {"value": 100, "unit": "h"}},
    )  # fmt: skip
    assert _msr(record("syn", limited)) is CheckStatus.UNDETERMINED
    c17200 = MATS["mil5j-7.3.2.0-d"]
    assert c17200.measurements("max_service_temperature")[0].value_in("degC") == pytest.approx(260)
    assert _msr(c17200) is CheckStatus.UNDETERMINED
    at_100h = req("max_service_temperature", ">=", 80, "degC",
                  conditions={"exposure_duration": {"value": 100, "unit": "h"}})  # fmt: skip
    cand = MaterialCandidate(candidate_id="c", origin="existing", material=c17200)
    assert check(at_100h, cand).status is CheckStatus.SATISFIED


# -- search status transition --------------------------------------------------------------


def test_search_status_transitions_for_the_original_undetermined_candidates() -> None:
    result = search_materials(target(), real_candidates())
    assert result.counts == {"considered": 46, "feasible": 0, "infeasible": 34, "undetermined": 12}
    by_id = {a.candidate_id: a for a in result.assessments}
    for mid in ORIGINAL_UNDETERMINED:
        a = by_id[mid]
        assert a.feasibility is Feasibility.UNDETERMINED
        undetermined = {k.requirement.property for k in a.undetermined}
        assert {"max_service_temperature", "corrosion_rate"} <= undetermined or mid.startswith(
            "nrl-"
        )


def test_a_service_temperature_would_move_a_candidate_only_as_far_as_its_other_gaps() -> None:
    """Synthetic: adding a satisfied max service temperature leaves corrosion undetermined."""
    base = MATS["mil5j-3.7.6.0-d"]
    with_t = base.model_copy(
        update={"temperature_effects": (temp_meas("max_service_temperature", 120),)}
    )
    result = search_materials(
        target(), [MaterialCandidate(candidate_id="x", origin="existing", material=with_t)]
    )
    a = result.assessments[0]
    assert a.feasibility is Feasibility.UNDETERMINED
    assert [k.requirement.property for k in a.undetermined] == ["corrosion_rate"]


# -- offline -------------------------------------------------------------------------------


def test_temperature_build_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    materials, report = build_mod.build(from_pdf=False)
    assert len(report["temperature_statements"]["linked"]) == 18
    assert len(materials) == 46
