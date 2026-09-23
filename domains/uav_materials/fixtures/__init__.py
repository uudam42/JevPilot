"""Synthetic, fictional materials for software tests. NOT engineering data."""

from __future__ import annotations

import json
from pathlib import Path

from domains.uav_materials.schema import CandidateOrigin, MaterialCandidate, MaterialRecord

FIXTURE = Path(__file__).with_name("synthetic_materials.json")


def load_synthetic_materials() -> list[MaterialRecord]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [MaterialRecord.model_validate(m) for m in data["materials"]]


def synthetic_candidates() -> list[MaterialCandidate]:
    return [
        MaterialCandidate(
            candidate_id=f"cand-{m.material_id}",
            origin=CandidateOrigin.EXISTING,
            material=m,
            metadata={"synthetic": True},
        )
        for m in load_synthetic_materials()
    ]
