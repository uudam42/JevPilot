"""UAV materials (iteration 1): a target property profile screened against SYNTHETIC materials.

python examples/uav_target_profile.py

The seven materials are fictional test fixtures, not engineering data. Only
hard constraints are checked; similarity ranking is the next iteration.
"""

from __future__ import annotations

from domains.uav_materials import (
    OperatingEnvironment,
    PropertyRequirement,
    Quantity,
    TargetMaterialProfile,
    search_materials,
)
from domains.uav_materials.fixtures import synthetic_candidates

SRC = "illustrative weights for this example only"


def main() -> None:
    wa_conditions = {
        "environment": "lab_water_immersion",
        "medium": "fresh water",
        "exposure_duration": {"value": 24, "unit": "h"},
        "temperature": {"value": 23, "unit": "degC"},
    }
    profile = TargetMaterialProfile(
        profile_id="example-spar",
        name="Example UAV spar target (synthetic)",
        environment=OperatingEnvironment(
            fluid="sea water", fluid_density=Quantity(value=1025, unit="kg/m^3")
        ),
        requirements=(
            PropertyRequirement(
                property="density", operator="<=", value=1600, unit="kg/m^3", priority="hard"
            ),
            PropertyRequirement(
                property="tensile_strength", operator=">=", value=400, unit="MPa", priority="hard"
            ),
            PropertyRequirement(
                property="water_absorption",
                operator="<=",
                value=1.0,
                unit="%",
                conditions=wa_conditions,
                priority="hard",
            ),
            PropertyRequirement(
                property="max_service_temperature",
                operator=">=",
                value=100,
                unit="degC",
                priority="hard",
            ),
            PropertyRequirement(
                property="youngs_modulus",
                operator="between",
                value=40,
                upper=70,
                unit="GPa",
                priority="soft",
                weight=0.2,
                weight_source=SRC,
            ),
            PropertyRequirement(
                property="tensile_strength",
                operator="maximize",
                priority="soft",
                weight=0.5,
                weight_source=SRC,
            ),
            PropertyRequirement(
                property="density",
                operator="minimize",
                priority="soft",
                weight=0.3,
                weight_source=SRC,
            ),
        ),
    )
    result = search_materials(profile, synthetic_candidates())
    print(f"{result.method}: {result.notes}")
    for a in result.assessments:
        why = "; ".join(
            f"{c.requirement.property}: {c.reason}" for c in a.checks if c.status != "satisfied"
        )
        print(f"  {a.candidate_id:22s} {a.feasibility:12s} {why}")


if __name__ == "__main__":
    main()
