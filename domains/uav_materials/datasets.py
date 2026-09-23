"""Access to the material datasets: real (built from raw sources) and synthetic (tests)."""

from __future__ import annotations

from functools import lru_cache

from domains.uav_materials.schema import MaterialCandidate, MaterialRecord


@lru_cache(maxsize=1)
def load_real_materials() -> tuple[MaterialRecord, ...]:
    """Real records, rebuilt in memory from the committed raw extracts (no network)."""
    from domains.uav_materials.ingest.build import build

    materials, _ = build(from_pdf=False)
    return tuple(materials)


def real_candidates() -> list[MaterialCandidate]:
    from domains.uav_materials.evaluation import candidate_from_record

    return [candidate_from_record(m) for m in load_real_materials()]
