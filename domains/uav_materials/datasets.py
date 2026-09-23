"""Access to the material datasets: real (built from raw sources) and synthetic (tests)."""

from __future__ import annotations

from functools import lru_cache

from domains.uav_materials.schema import CandidateOrigin, MaterialCandidate, MaterialRecord


@lru_cache(maxsize=1)
def load_real_materials() -> tuple[MaterialRecord, ...]:
    """Real records, rebuilt in memory from the committed raw extracts (no network)."""
    from domains.uav_materials.ingest.build import build

    materials, _ = build(from_pdf=False)
    return tuple(materials)


def real_candidates() -> list[MaterialCandidate]:
    return [
        MaterialCandidate(candidate_id=m.material_id, origin=CandidateOrigin.EXISTING, material=m)
        for m in load_real_materials()
    ]
