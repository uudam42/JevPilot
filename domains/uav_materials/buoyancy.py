"""Buoyancy-related quantities derived from density. Physics, not scores.

Buoyancy depends on the *material* density and the *fluid* density, which
belongs to the operating environment of the target profile. These helpers
compute derived quantities for a solid, fully submerged, non-absorbing body.
They are not a buoyancy simulator: geometry, trapped air and water uptake
are ignored.
"""

from __future__ import annotations

from domains.uav_materials.schema import Quantity

STANDARD_GRAVITY = 9.80665  # m/s^2


def density_ratio(material: Quantity, fluid: Quantity) -> float:
    """ρ_fluid / ρ_material: above 1 means a solid piece of the material floats."""
    return fluid.to("kg/m^3") / material.to("kg/m^3")


def net_buoyant_force_per_volume(material: Quantity, fluid: Quantity) -> float:
    """(ρ_fluid − ρ_material) · g in N/m^3 for a fully submerged solid; positive means it rises."""
    return (fluid.to("kg/m^3") - material.to("kg/m^3")) * STANDARD_GRAVITY
