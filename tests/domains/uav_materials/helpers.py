"""Shared builders for UAV materials tests (synthetic data only)."""

from __future__ import annotations

from typing import Any

from domains.uav_materials import (
    Measurement,
    OperatingEnvironment,
    PropertyRequirement,
    Quantity,
    TargetMaterialProfile,
)
from jevpilot import Provenance, SourceRef

SYN = Provenance(sources=(SourceRef(kind="synthetic", identifier="unit-test"),))
WA24 = {
    "medium": "fresh water",
    "exposure_duration": {"value": 24, "unit": "h"},
    "temperature": {"value": 23, "unit": "degC"},
}


def meas(prop: str, value: float | None = None, unit: str | None = None, **kw: Any) -> Measurement:
    kw.setdefault("provenance_status", "synthetic")
    kw.setdefault("provenance", SYN)
    return Measurement(property=prop, value=value, unit=unit, **kw)


def req(
    prop: str, op: str, value: float | None = None, unit: str | None = None, **kw: Any
) -> PropertyRequirement:
    kw.setdefault("priority", "hard" if op in ("<=", ">=", "between") else "soft")
    return PropertyRequirement(property=prop, operator=op, value=value, unit=unit, **kw)


def spar_profile() -> TargetMaterialProfile:
    """Synthetic example: a light, stiff-enough, water-tolerant structural member."""
    src = "synthetic example: weights chosen for the unit test, not from a design study"
    return TargetMaterialProfile(
        profile_id="syn-spar-1",
        name="Synthetic UAV spar target",
        application="fixed-wing UAV wing spar (synthetic example)",
        environment=OperatingEnvironment(
            description="coastal operation",
            fluid="sea water",
            fluid_density=Quantity(value=1025, unit="kg/m^3"),
            temperature_min=Quantity(value=-20, unit="degC"),
            temperature_max=Quantity(value=60, unit="degC"),
        ),
        requirements=(
            req("density", "<=", 1600, "kg/m^3"),
            req("tensile_strength", ">=", 400, "MPa"),
            req("water_absorption", "<=", 1.0, "%", conditions=WA24),
            req("max_service_temperature", ">=", 100, "degC"),
            req(
                "youngs_modulus",
                "between",
                40,
                "GPa",
                upper=70,
                priority="soft",
                weight=0.2,
                weight_source=src,
                importance="medium",
            ),
            req("tensile_strength", "maximize", weight=0.5, weight_source=src, importance="high"),
            req("density", "minimize", weight=0.3, weight_source=src, importance="high"),
        ),
    )
