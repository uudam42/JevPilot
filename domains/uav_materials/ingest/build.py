"""Build the real-material dataset: raw sources → parsers → validated MaterialRecords.

    python -m domains.uav_materials.ingest.build             # from committed raw extracts
    python -m domains.uav_materials.ingest.build --from-pdf  # re-extract MIL-HDBK-5J (needs pypdf)

Raw files are never modified. The build writes
``data/uav_materials/processed/materials_v1.json`` (the dataset) and
``build_report.json`` (accepted and rejected tables with reasons, identity
links, coverage).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from domains.uav_materials.identity import IdentityIndex, MatchLevel, MaterialIdentity
from domains.uav_materials.ingest import mil_hdbk_5j, nrl_tropical
from domains.uav_materials.properties import PROPERTIES, Category
from domains.uav_materials.schema import (
    MaterialFamily,
    MaterialRecord,
    Measurement,
    ProvenanceStatus,
)
from jevpilot import Provenance, SourceRef, stable_digest

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "uav_materials"
RAW_MIL_PDF = DATA / "raw" / "mil_hdbk_5j" / "MILHDBK5J.pdf"
RAW_MIL_PAGES = DATA / "raw" / "mil_hdbk_5j" / "table_pages.jsonl"
RAW_NRL = DATA / "raw" / "nrl_ad0609618" / "DTIC_AD0609618_djvu.txt"
PROCESSED = DATA / "processed" / "materials_v1.json"
REPORT = DATA / "processed" / "build_report.json"
DATASET_VERSION = "uav-materials-real/1"
RETRIEVED = "2026-09-23"

NRL_ONLY = {  # source name → (id, name, system, designation, clad)
    "1100": ("nrl-1100", "1100 aluminum (temper not stated)", "aluminum", "1100", False),
    "Alclad 2024-T": (
        "nrl-alclad-2024-t",
        "Alclad 2024 aluminum alloy (T temper not stated)",
        "aluminum",
        "2024",
        True,
    ),
    "AZ31X": (
        "nrl-az31x",
        "AZ31X magnesium alloy (controlled purity; not AZ31B)",
        "magnesium",
        "AZ31X",
        False,
    ),
}


def load_mil_pages(from_pdf: bool) -> list[dict[str, Any]]:
    if from_pdf:
        pages = mil_hdbk_5j.extract_pages(str(RAW_MIL_PDF))
        keep = [p for p in pages if mil_hdbk_5j.find_table(p["text"])]
        with RAW_MIL_PAGES.open("w", encoding="utf-8") as fh:
            for p in keep:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        return keep
    return [json.loads(line) for line in RAW_MIL_PAGES.open(encoding="utf-8")]


def build(from_pdf: bool = False) -> tuple[list[MaterialRecord], dict[str, Any]]:
    tables, rejected = mil_hdbk_5j.parse_all(load_mil_pages(from_pdf))
    records = mil_hdbk_5j.build_records(tables, RETRIEVED)
    index = IdentityIndex([r.identity for r in records if r.identity])
    by_id = {r.material_id: r for r in records}

    nrl = nrl_tropical.build_measurements(RAW_NRL.read_text(encoding="utf-8"), RETRIEVED)
    links: list[dict[str, Any]] = []
    for source_name, measurements in nrl.items():
        matches = index.resolve(source_name)
        if matches:
            for identity, level in matches:
                material_id = next(r.material_id for r in records if r.identity is identity)
                tagged = [_tag(m, level, source_name) for m in measurements]
                rec = by_id[material_id]
                by_id[material_id] = rec.model_copy(
                    update={
                        "corrosion": rec.corrosion + tuple(tagged),
                        "identity": _with_source(rec.identity, source_name),
                    }
                )
                links.append(
                    {
                        "source": nrl_tropical.SOURCE_ID,
                        "source_name": source_name,
                        "material_id": material_id,
                        "match": level.value,
                        "measurements": len(tagged),
                    }
                )
        else:
            mid, name, system, designation, clad = NRL_ONLY[source_name]
            by_id[mid] = MaterialRecord(
                material_id=mid,
                name=name,
                family=MaterialFamily.METAL,
                corrosion=tuple(measurements),
                identity=MaterialIdentity(
                    canonical_id=f"nrl:{source_name}",
                    canonical_name=name,
                    system=system,
                    designation=designation,
                    clad=clad,
                    source_names={nrl_tropical.SOURCE_ID: source_name},
                    source_ids={nrl_tropical.SOURCE_ID: [nrl_tropical.SOURCE_ID]},
                ),
                provenance=Provenance(
                    id=mil_hdbk_5j.deterministic_id(nrl_tropical.SOURCE_ID, mid),
                    created_at=mil_hdbk_5j.retrieved_at(RETRIEVED),
                    sources=(
                        SourceRef(
                            kind="report",
                            identifier=nrl_tropical.SOURCE_ID,
                            uri=nrl_tropical.SOURCE_URI,
                        ),
                    ),
                ),
                provenance_status=ProvenanceStatus.SOURCED,
                metadata={"system": system, "sources_only": [nrl_tropical.SOURCE_ID]},
            )
            links.append(
                {
                    "source": nrl_tropical.SOURCE_ID,
                    "source_name": source_name,
                    "material_id": mid,
                    "match": "new record (no match)",
                    "measurements": len(measurements),
                }
            )
    materials = sorted(by_id.values(), key=lambda r: r.material_id)
    report = {
        "dataset_version": DATASET_VERSION,
        "mil_hdbk_5j": {
            "tables_found": len(tables) + len(rejected),
            "accepted": len(tables),
            "rejected": rejected,
            "rejection_reasons": dict(Counter(r["reason"].split(" ≠")[0][:60] for r in rejected)),
        },
        "identity_links": links,
        "coverage": coverage(materials),
    }
    return materials, report


def _tag(m: Measurement, level: MatchLevel, source_name: str) -> Measurement:
    assert m.provenance is not None
    meta = {
        **m.provenance.metadata,
        "identity_match": level.value,
        "source_material_name": source_name,
    }
    return m.model_copy(update={"provenance": m.provenance.model_copy(update={"metadata": meta})})


def _with_source(identity: MaterialIdentity | None, name: str) -> MaterialIdentity | None:
    if identity is None:
        return None
    names = {**identity.source_names, nrl_tropical.SOURCE_ID: name}
    return identity.model_copy(update={"source_names": names})


def coverage(materials: list[MaterialRecord]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    for cat in Category:
        by_category[cat.value] = sum(
            1
            for r in materials
            if any(m.category is cat and not m.is_missing for m in r.all_measurements())
        )
    by_property = {
        p: sum(1 for r in materials if r.status(p) == "measured") for p in sorted(PROPERTIES)
    }
    cells = len(materials) * len(PROPERTIES)
    return {
        "materials": len(materials),
        "measurements": sum(1 for r in materials for _ in r.all_measurements()),
        "materials_with_category": by_category,
        "materials_with_property": by_property,
        "missing_rate_material_property": 1 - sum(by_property.values()) / cells,
    }


def write(materials: list[MaterialRecord], report: dict[str, Any]) -> None:
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "dataset": DATASET_VERSION,
        "note": "Real material data. Values keep their original units, conditions and "
        "measurement-level provenance; see data/uav_materials/sources/.",
        "transformations": {
            "mil_hdbk_5j": mil_hdbk_5j.PARSER_VERSION,
            "nrl_ad0609618": nrl_tropical.EXTRACTOR_VERSION,
        },
        "sources": sorted(p.name for p in (DATA / "sources").glob("*.json")),
        "materials": [m.model_dump(mode="json") for m in materials],
    }
    doc["digest"] = stable_digest(doc["materials"])
    PROCESSED.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--from-pdf", action="store_true")
    args = parser.parse_args(argv)
    materials, report = build(from_pdf=args.from_pdf)
    write(materials, report)
    cov = report["coverage"]
    print(
        f"{cov['materials']} materials, {cov['measurements']} measurements; "
        f"{report['mil_hdbk_5j']['accepted']}/{report['mil_hdbk_5j']['tables_found']} "
        "MIL-HDBK-5J tables accepted"
    )
    print("materials with data per category:", cov["materials_with_category"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
