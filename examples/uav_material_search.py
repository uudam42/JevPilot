"""UAV materials iteration 2: rank REAL materials (MIL-HDBK-5J, NRL) against a target profile.

python examples/uav_material_search.py

The target profile is a synthetic example, fixed before looking at any
ranking. Weights are illustrative, not from a design study.
"""

from __future__ import annotations

from domains.uav_materials import (
    OperatingEnvironment,
    PropertyRequirement,
    Quantity,
    TargetMaterialProfile,
    search_materials,
)
from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.search import CheckStatus, MaterialSearchResult

W_SRC = "illustrative weights for this example; not from a design study"
MARINE = {"environment": "marine_atmosphere"}
WATER_24H = {"environment": "lab_water_immersion", "exposure_duration": {"value": 24, "unit": "h"}}


def target() -> TargetMaterialProfile:
    def req(
        prop: str, op: str, value: float | None = None, unit: str | None = None, **kw: object
    ) -> PropertyRequirement:
        kw.setdefault("priority", "hard" if op in ("<=", ">=") else "soft")
        return PropertyRequirement(property=prop, operator=op, value=value, unit=unit, **kw)

    return TargetMaterialProfile(
        profile_id="example-uav-airframe-member",
        name="Lightweight UAV airframe member (synthetic example target)",
        application="primary structure of a small fixed-wing UAV operating near the sea",
        environment=OperatingEnvironment(
            fluid="sea water", fluid_density=Quantity(value=1025, unit="kg/m^3")
        ),
        requirements=(
            req("density", "<=", 3000, "kg/m^3"),
            req("tensile_strength", ">=", 350, "MPa"),
            req("elongation_at_break", ">=", 5, "%"),
            req("corrosion_rate", "<=", 0.025, "mm/year", conditions=MARINE),
            req("max_service_temperature", ">=", 80, "degC"),
            req("tensile_strength", "maximize", weight=0.35, weight_source=W_SRC),
            req("density", "minimize", weight=0.35, weight_source=W_SRC),
            req(
                "youngs_modulus", "between", 65, "GPa", upper=120, weight=0.15, weight_source=W_SRC
            ),
            req("yield_strength", "maximize", weight=0.15, weight_source=W_SRC),
            req("water_absorption", "minimize", conditions=WATER_24H, importance="high"),
        ),
    )


def report(result: MaterialSearchResult, top: int = 8) -> None:
    print(f"method: {result.method}")
    print(f"counts: {result.counts}")
    print(f"unscored (no numeric weight): {list(result.unscored_preferences)}")
    for key, how in result.normalization["preferences"].items():
        print(f"  normalization {key}: {how}")
    print()
    for a in result.assessments[:top]:
        print(
            f"#{a.rank} {a.name}  [{a.feasibility}]  distance={a.distance:.3f}  "
            f"pessimistic={a.pessimistic_distance:.3f}  coverage={a.score_coverage:.0%}"
        )
        for c in a.checks:
            mark = {
                CheckStatus.SATISFIED: "PASS",
                CheckStatus.VIOLATED: "FAIL",
                CheckStatus.UNDETERMINED: "UNDETERMINED",
            }[c.status]
            print(f"    {c.requirement.property:24s} {mark:12s} {c.reason[:110]}")
        for k in a.contributions:
            value = "missing" if k.value is None else f"{k.value:.4g} {k.unit}"
            penalty = "-" if k.penalty is None else f"{k.penalty:.3f}"
            print(f"    · {k.property}:{k.operator:9s} value={value:16s} penalty={penalty}")


def main() -> None:
    report(search_materials(target(), real_candidates()))


if __name__ == "__main__":
    main()
