"""Explicit physical units for material data.

This is a deliberately small table of the units this domain needs, each with
its physical dimension and an exact conversion to the SI reference unit of
that dimension (including the affine offset for degrees Celsius).

``pint`` was evaluated first. It is the established choice, but it would be
this domain's first third-party runtime dependency (with four transitive
packages and a global registry) for about twenty units. If the domain later
needs arbitrary compound units, replace :func:`convert` and
:func:`dimension_of` with ``pint``. Their contracts stay the same.

Rules enforced here:

* every stored value names a unit from :data:`UNITS` (unknown units are errors);
* values are only converted, and therefore only compared, within one dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Dimension(StrEnum):
    STRESS = "stress"  # strength, modulus
    DENSITY = "density"
    FRACTION = "fraction"  # strain, elongation, water uptake, retention
    TEMPERATURE = "temperature"
    RATE = "length_per_time"  # corrosion penetration rate
    TIME = "time"  # exposure duration


@dataclass(frozen=True)
class UnitDef:
    symbol: str
    dimension: Dimension
    scale: float  # SI value = value * scale + offset
    offset: float = 0.0


_DEFS = [
    UnitDef("Pa", Dimension.STRESS, 1.0),
    UnitDef("kPa", Dimension.STRESS, 1e3),
    UnitDef("MPa", Dimension.STRESS, 1e6),
    UnitDef("GPa", Dimension.STRESS, 1e9),
    UnitDef("psi", Dimension.STRESS, 6894.757293168361),
    UnitDef("ksi", Dimension.STRESS, 6894757.293168361),
    UnitDef("Msi", Dimension.STRESS, 6894757293.168361),  # 10^3 ksi (MIL-HDBK-5 moduli)
    UnitDef("kg/m^3", Dimension.DENSITY, 1.0),
    UnitDef("g/cm^3", Dimension.DENSITY, 1000.0),
    UnitDef("lb/in^3", Dimension.DENSITY, 27679.904710203125),  # 0.45359237 kg / 0.0254^3 m^3
    UnitDef("1", Dimension.FRACTION, 1.0),
    UnitDef("%", Dimension.FRACTION, 0.01),
    UnitDef("K", Dimension.TEMPERATURE, 1.0),
    UnitDef("degC", Dimension.TEMPERATURE, 1.0, 273.15),
    UnitDef("m/s", Dimension.RATE, 1.0),
    UnitDef("mm/year", Dimension.RATE, 1e-3 / 31_557_600),  # Julian year
    UnitDef("um/year", Dimension.RATE, 1e-6 / 31_557_600),
    UnitDef("mil/year", Dimension.RATE, 25.4e-6 / 31_557_600),  # 1 mil = 0.001 in
    UnitDef("s", Dimension.TIME, 1.0),
    UnitDef("h", Dimension.TIME, 3600.0),
    UnitDef("day", Dimension.TIME, 86_400.0),
    UnitDef("year", Dimension.TIME, 31_557_600.0),  # Julian year, as for the rate units
]
UNITS: dict[str, UnitDef] = {u.symbol: u for u in _DEFS}
SI: dict[Dimension, str] = {
    Dimension.STRESS: "Pa",
    Dimension.DENSITY: "kg/m^3",
    Dimension.FRACTION: "1",
    Dimension.TEMPERATURE: "K",
    Dimension.RATE: "m/s",
    Dimension.TIME: "s",
}


class UnitError(ValueError):
    """Unknown unit, or an attempt to mix incompatible dimensions."""


def unit_def(unit: str) -> UnitDef:
    try:
        return UNITS[unit]
    except KeyError:
        raise UnitError(f"unknown unit {unit!r}; known units: {sorted(UNITS)}") from None


def dimension_of(unit: str) -> Dimension:
    return unit_def(unit).dimension


def compatible(a: str, b: str) -> bool:
    return dimension_of(a) is dimension_of(b)


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between units of the same dimension; raises :class:`UnitError` otherwise."""
    src, dst = unit_def(from_unit), unit_def(to_unit)
    if src.dimension is not dst.dimension:
        raise UnitError(
            f"cannot convert {from_unit!r} ({src.dimension}) to {to_unit!r} ({dst.dimension})"
        )
    si = value * src.scale + src.offset
    return (si - dst.offset) / dst.scale


def to_si(value: float, unit: str) -> float:
    return convert(value, unit, SI[dimension_of(unit)])
