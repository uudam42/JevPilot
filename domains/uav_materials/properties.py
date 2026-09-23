"""Registry of material properties: name → category, physical dimension, meaning.

The six requirement families of the UAV materials domain are *categories*,
not scores. Each category holds several physically distinct properties, and
every measurement names exactly one of them. New properties are added here
(or with :func:`register_property`) without touching JevPilot core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from domains.uav_materials.units import Dimension


class Category(StrEnum):
    STRENGTH = "strength"
    DEFORMATION = "deformation"
    CORROSION = "corrosion"
    DENSITY = "density"  # the material side of buoyancy
    WATER_ABSORPTION = "water_absorption"
    TEMPERATURE = "temperature_effects"


class Direction(StrEnum):
    """What 'better' usually means. Only informative: targets always decide."""

    HIGHER = "higher"
    LOWER = "lower"
    CONTEXT = "context"  # depends on the application (e.g. stiffness)


@dataclass(frozen=True)
class PropertyDef:
    name: str
    category: Category
    dimension: Dimension
    description: str
    direction: Direction = Direction.CONTEXT
    # Condition fields without which values of this property are not comparable.
    required_conditions: tuple[str, ...] = field(default=())


PROPERTIES: dict[str, PropertyDef] = {}


def register_property(p: PropertyDef) -> PropertyDef:
    if p.name in PROPERTIES and PROPERTIES[p.name] != p:
        raise ValueError(f"property {p.name!r} is already registered differently")
    PROPERTIES[p.name] = p
    return p


def property_def(name: str) -> PropertyDef:
    try:
        return PROPERTIES[name]
    except KeyError:
        raise ValueError(
            f"unknown material property {name!r}; known: {sorted(PROPERTIES)}"
        ) from None


S, D, C, R, W, T = (
    Category.STRENGTH,
    Category.DEFORMATION,
    Category.CORROSION,
    Category.DENSITY,
    Category.WATER_ABSORPTION,
    Category.TEMPERATURE,
)
for _p in (
    # strength: stress at failure or yield, per loading mode
    PropertyDef(
        "tensile_strength", S, Dimension.STRESS, "ultimate tensile strength", Direction.HIGHER
    ),
    PropertyDef(
        "yield_strength", S, Dimension.STRESS, "stress at onset of plastic yield", Direction.HIGHER
    ),
    PropertyDef(
        "compressive_strength",
        S,
        Dimension.STRESS,
        "ultimate compressive strength",
        Direction.HIGHER,
    ),
    PropertyDef("shear_strength", S, Dimension.STRESS, "ultimate shear strength", Direction.HIGHER),
    # deformation: stiffness and strain behaviour, never strength
    PropertyDef("youngs_modulus", D, Dimension.STRESS, "elastic (Young's) modulus"),
    PropertyDef("shear_modulus", D, Dimension.STRESS, "elastic shear modulus"),
    PropertyDef("elongation_at_break", D, Dimension.FRACTION, "strain at fracture in tension"),
    PropertyDef("strain_at_yield", D, Dimension.FRACTION, "strain at onset of yield"),
    # corrosion: always tied to an environment
    PropertyDef(
        "corrosion_rate",
        C,
        Dimension.RATE,
        "uniform corrosion penetration rate",
        Direction.LOWER,
        required_conditions=("medium",),
    ),
    # density: the material input to buoyancy (fluid density belongs to the target environment)
    PropertyDef("density", R, Dimension.DENSITY, "bulk material density", Direction.LOWER),
    # water absorption: always tied to exposure
    PropertyDef(
        "water_absorption",
        W,
        Dimension.FRACTION,
        "mass gain from water uptake",
        Direction.LOWER,
        required_conditions=("medium", "exposure_duration"),
    ),
    # temperature effects: limits, transitions, and retention under temperature
    PropertyDef(
        "max_service_temperature",
        T,
        Dimension.TEMPERATURE,
        "upper continuous service temperature",
        Direction.HIGHER,
    ),
    PropertyDef(
        "min_service_temperature",
        T,
        Dimension.TEMPERATURE,
        "lower continuous service temperature",
        Direction.LOWER,
    ),
    PropertyDef(
        "glass_transition_temperature",
        T,
        Dimension.TEMPERATURE,
        "glass transition (polymers, polymer matrices)",
    ),
    PropertyDef("melting_temperature", T, Dimension.TEMPERATURE, "melting point or range onset"),
    PropertyDef(
        "tensile_strength_retention",
        T,
        Dimension.FRACTION,
        "tensile strength at temperature relative to room temperature",
        Direction.HIGHER,
        required_conditions=("temperature",),
    ),
    PropertyDef(
        "modulus_retention",
        T,
        Dimension.FRACTION,
        "Young's modulus at temperature relative to room temperature",
        Direction.HIGHER,
        required_conditions=("temperature",),
    ),
):
    register_property(_p)
del _p
