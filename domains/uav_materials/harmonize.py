"""Canonical comparison units. Originals are never modified.

A :class:`HarmonizedValue` is a *derived view* of a measurement in the
canonical unit of its property. The measurement's own value, unit,
conditions and provenance stay as recorded.
"""

from __future__ import annotations

from domains.uav_materials.properties import property_def
from domains.uav_materials.schema import Measurement
from domains.uav_materials.units import Dimension
from jevpilot import FrozenModel

CANONICAL_BY_DIMENSION: dict[Dimension, str] = {
    Dimension.STRESS: "MPa",
    Dimension.DENSITY: "kg/m^3",
    Dimension.FRACTION: "%",
    Dimension.TEMPERATURE: "degC",
    Dimension.RATE: "mm/year",
    Dimension.TIME: "h",
}
CANONICAL_BY_PROPERTY: dict[str, str] = {
    "youngs_modulus": "GPa",
    "shear_modulus": "GPa",
    "ply_longitudinal_modulus": "GPa",
    "ply_transverse_modulus": "GPa",
    "ply_inplane_shear_modulus": "GPa",
    "ply_major_poisson_ratio": "1",
}


def canonical_unit(prop: str) -> str:
    return CANONICAL_BY_PROPERTY.get(prop) or CANONICAL_BY_DIMENSION[property_def(prop).dimension]


class HarmonizedValue(FrozenModel):
    value: float
    unit: str
    original_value: float
    original_unit: str


def harmonize(m: Measurement) -> HarmonizedValue | None:
    if m.value is None or m.unit is None:
        return None
    unit = canonical_unit(m.property)
    converted = m.value_in(unit)
    assert converted is not None
    return HarmonizedValue(value=converted, unit=unit, original_value=m.value, original_unit=m.unit)
