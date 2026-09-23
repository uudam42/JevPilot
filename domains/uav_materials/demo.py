"""Demonstration fixtures for requirement semantics (architecture tests only).

``directional_demo_target`` is NOT a UAV design target and does not replace the
example UAV target: it only shows that properly directed engineering
requirements can legitimately be answered by E11 and S11T of a unidirectional
ply. ``example_ply`` is the AS/IMLS ply of the micromechanics example.
"""

from __future__ import annotations

from domains.uav_materials.micromechanics import ContinuousFiberCompositeDesignSpace
from domains.uav_materials.requirements import (
    Aspect,
    Direction,
    EngineeringRequirement,
    EngineeringTarget,
)
from domains.uav_materials.schema import MaterialCandidate

ALONG_MEMBER = "uniaxial tension along the member axis; fibres aligned with that axis"


def directional_demo_target() -> EngineeringTarget:
    """Demonstration only: shows that properly directed requirements can use E11 and S11T."""
    return EngineeringTarget(
        profile_id="demo-directional-ply",
        name="Direction-aware demonstration target (architecture test; not the UAV target)",
        requirements=(
            EngineeringRequirement(
                aspect=Aspect.DENSITY, operator="<=", value=3000, unit="kg/m^3", priority="hard"
            ),
            EngineeringRequirement(
                aspect=Aspect.TENSILE_ULTIMATE,
                direction=Direction.MATERIAL_1,
                loading=ALONG_MEMBER,
                operator=">=",
                value=350,
                unit="MPa",
                priority="hard",
            ),
            EngineeringRequirement(
                aspect=Aspect.NORMAL_STIFFNESS,
                direction=Direction.MATERIAL_1,
                loading=ALONG_MEMBER,
                operator=">=",
                value=70,
                unit="GPa",
                priority="hard",
            ),
        ),
    )


def example_ply(vf: float = 0.60) -> MaterialCandidate:
    space = ContinuousFiberCompositeDesignSpace()
    design = space.design(
        {"fiber_id": "AS--", "matrix_id": "IMLS", "fiber_volume_fraction": vf},
        generation_method="manual (example)",
    )
    return space.candidate_from_design(design)


def laminate_demo_target() -> EngineeringTarget:
    """Demonstration only: in-plane laminate requirements stated in laminate axes x, y, xy."""
    in_plane = "in-plane (membrane) loading of a flat symmetric panel, laminate axes x, y"

    def stiff(aspect: Aspect, direction: Direction, value: float) -> EngineeringRequirement:
        return EngineeringRequirement(
            aspect=aspect,
            direction=direction,
            loading=in_plane,
            operator=">=",
            value=value,
            unit="GPa",
            priority="hard",
        )

    return EngineeringTarget(
        profile_id="demo-directional-laminate",
        name="Direction-aware laminate demonstration target (architecture test; not the UAV "
        "target)",
        requirements=(
            EngineeringRequirement(
                aspect=Aspect.DENSITY, operator="<=", value=3000, unit="kg/m^3", priority="hard"
            ),
            stiff(Aspect.NORMAL_STIFFNESS, Direction.LAMINATE_X, 40),
            stiff(Aspect.NORMAL_STIFFNESS, Direction.LAMINATE_Y, 40),
            stiff(Aspect.SHEAR_STIFFNESS, Direction.LAMINATE_XY, 15),
        ),
    )


def example_laminate(half: str = "0/45/-45/90", vf: float = 0.60) -> MaterialCandidate:
    """AS/IMLS symmetric laminate, 0.127 mm plies (the RP-1351 example ply thickness)."""
    from domains.uav_materials.clt import ContinuousFiberLaminateDesignSpace, layup_design

    design = layup_design(half, vf=vf, ply_thickness_mm=0.127, generation_method="manual (demo)")
    return ContinuousFiberLaminateDesignSpace().candidate_from_design(design)
