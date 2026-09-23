"""Real-data ingestion: raw extracts → strict parsers → validated MaterialRecords (offline)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from domains.uav_materials.datasets import load_real_materials
from domains.uav_materials.identity import IdentityIndex, MatchLevel, parse_designation
from domains.uav_materials.ingest import build as build_mod
from domains.uav_materials.ingest import mil_hdbk_5j as mil
from domains.uav_materials.ingest import nrl_tropical as nrl
from domains.uav_materials.ingest.mil_hdbk_5j import TableRejected, number, parse_page

DATA = Path(__file__).resolve().parents[3] / "data" / "uav_materials"
PAGES = {
    json.loads(line)["page"]: json.loads(line)["text"]
    for line in (DATA / "raw/mil_hdbk_5j/table_pages.jsonl").open(encoding="utf-8")
}
MATS = {m.material_id: m for m in load_real_materials()}


def test_parse_7075_table_cell_by_cell() -> None:
    t = parse_page(PAGES[673], 673)
    assert t.table_id == "3.7.6.0(b1)" and len(t.basis) == 21 and len(t.rows) == 15
    assert t.basis[:3] == ["S", "A", "B"]
    assert t.values["Ftu:LT"][0] == "74" and t.values["Ftu:L"][0] is None
    assert t.values["Fty:L"][1] == "69" and t.values["Fty:LT"][1] == "67"
    assert number(t.values["e:LT"][0]) == 5
    assert t.physical["density_lb_in3"] == ["0.101"] and t.physical["E"][0] == "10.3"


def test_unknown_row_groups_never_overwrite_known_properties() -> None:
    """Regression: 'RA, percent' rows used to be parsed as a second 'e:L' and overwrite it."""
    t = parse_page(PAGES[1202], 1202)  # MP35N bar
    assert t.values["e:L"] == ["8", None, "8", "8"]  # elongation
    assert t.values["RA:L"] == ["35", None, "35", "35"]  # reduction of area, kept apart
    mp35n = MATS["mil5j-7.4.1.0-b"]
    assert {m.value for m in mp35n.measurements("elongation_at_break")} == {8.0}


def test_ambiguous_layouts_are_rejected_not_guessed() -> None:
    with pytest.raises(TableRejected, match="letter-spaced"):
        parse_page(PAGES[841], 841)  # AZ31B: "3 2 3 2 ..." digit grouping is ambiguous
    with pytest.raises(TableRejected, match="multi-alloy"):
        parse_page(PAGES[76], 76)  # one steel alloy per column
    tampered = PAGES[673].replace("\n74\n", "\n74\n80\n", 1)
    with pytest.raises(TableRejected):
        parse_page(tampered, 673)


def test_build_is_reproducible_and_leaves_raw_untouched() -> None:
    raw = DATA / "raw/mil_hdbk_5j/table_pages.jsonl"
    before = hashlib.sha256(raw.read_bytes()).hexdigest()
    a, report = build_mod.build()
    b, _ = build_mod.build()
    assert [m.model_dump(mode="json") for m in a] == [m.model_dump(mode="json") for m in b]
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == before
    assert 20 <= len(a) <= 100
    assert report["mil_hdbk_5j"]["accepted"] == 53 and report["mil_hdbk_5j"]["rejected"]
    assert all(r["reason"] for r in report["mil_hdbk_5j"]["rejected"])


def test_committed_raw_files_match_their_manifests() -> None:
    manifest = json.loads((DATA / "sources/nrl_ad0609618.json").read_text())
    entry = next(f for f in manifest["files"] if f["committed"] and "sha256" in f)
    assert hashlib.sha256((DATA / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    mil_manifest = json.loads((DATA / "sources/mil_hdbk_5j.json").read_text())
    assert "Approved for public release" in mil_manifest["reuse"]["statement_in_document"]


def test_every_real_measurement_traces_to_a_table_cell_or_a_quote() -> None:
    for record in MATS.values():
        assert record.provenance_status == "sourced"
        for m in record.all_measurements():
            assert m.provenance_status == "sourced" and m.provenance and m.provenance.sources
            src = m.provenance.sources[0]
            if "statement" in src.metadata:  # prose temperature statement (MIL or NASA)
                assert m.category == "temperature_effects" and src.metadata["quote"]
                assert m.provenance.metadata["identity_match"] in ("exact", "alloy")
            elif src.identifier == mil.SOURCE_ID:
                assert src.metadata["page"] in PAGES
                assert number(src.metadata["raw_token"]) == m.value  # original value preserved
                assert m.conditions.temperature_regime == "room"
                assert m.conditions.temperature is None  # no number was invented
            else:
                assert src.identifier == nrl.SOURCE_ID and src.metadata["quote"]


def test_nrl_values_come_only_from_verified_quotes() -> None:
    raw = (DATA / "raw/nrl_ad0609618/DTIC_AD0609618_djvu.txt").read_text(encoding="utf-8")
    assert nrl.verify(raw, nrl.EXTRACTIONS[0]) == 0.15
    with pytest.raises(nrl.QuoteNotFound):
        nrl.verify(raw.replace("0.09  mil", "0.08  mil"), nrl.EXTRACTIONS[1])
    rates = nrl.build_measurements(raw, "2026-09-23")
    bound = rates["6061-T"][0]
    assert bound.qualifier == "<=" and bound.basis == "derived"
    assert bound.value == pytest.approx(0.15 / 16) and bound.unit == "mil/year"


def test_measurement_level_provenance_merges_two_sources() -> None:
    forging = MATS["mil5j-3.6.2.0-f"]
    sources = {
        m.provenance.sources[0].identifier for m in forging.all_measurements() if m.provenance
    }
    assert sources == {mil.SOURCE_ID, nrl.SOURCE_ID}
    linked = [m for m in forging.corrosion]
    assert linked and all(
        m.provenance and m.provenance.metadata["identity_match"] == "alloy" for m in linked
    )
    assert forging.identity and forging.identity.source_names[nrl.SOURCE_ID] == "6061-T"


def test_aliases_resolve_by_designation_not_display_name() -> None:
    keys = {
        parse_designation(n).key
        for n in (
            "Al 7075-T6",
            "7075-T6",
            "AA7075-T6",  # type: ignore[union-attr]
            "AA 7075-T6",
            "7075 aluminum alloy T6",
        )
    }
    assert keys == {("aluminum", "7075", False)}
    assert parse_designation("AA7075-T6").temper == "T6"  # type: ignore[union-attr]
    assert parse_designation("Clad 7075").clad  # type: ignore[union-attr]
    index = IdentityIndex(
        [m.identity for m in MATS.values() if m.identity and m.material_id.startswith("mil5j")]
    )
    exact = {i.canonical_id: lvl for i, lvl in index.resolve("AA7075-T6")}
    assert exact and set(exact.values()) == {MatchLevel.EXACT}
    assert all("Clad" not in i for i in exact)  # bare and clad are different materials
    assert {lvl for _, lvl in index.resolve("7075")} == {MatchLevel.ALLOY}
    assert index.resolve("AZ31X") == []  # not merged into any handbook material
    assert MATS["nrl-az31x"].identity and MATS["nrl-az31x"].identity.designation == "AZ31X"
    assert parse_designation("AZ31X").key != parse_designation("AZ31B").key  # type: ignore[union-attr]
    assert index.resolve("7075-T7351")  # temper listed in the bar/rod table
    assert not index.resolve("7075-T9")  # unknown temper: no silent match


def test_coverage_is_reported_honestly() -> None:
    _, report = build_mod.build()
    cov = report["coverage"]["materials_with_category"]
    assert cov["strength"] >= 40 and cov["density"] >= 40
    assert cov["water_absorption"] == 0
    # prose temperature statements only: 11 C17200/Ti/Ni records + 4 bare 7075 (melting)
    assert cov["temperature_effects"] == 15
    by_property = report["coverage"]["materials_with_property"]
    assert by_property["max_service_temperature"] == 4  # C17200 only (time-limited)
    assert 0 < cov["corrosion"] < 10
