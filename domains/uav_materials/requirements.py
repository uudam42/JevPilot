"""Engineering requirements and their resolution to material-system observables.

A target states *engineering intent* ("tensile load resistance along the
member axis ≥ 350 MPa"). Which measured or predicted property may answer it
depends on the material system: for an isotropic bulk metal it is
``tensile_strength``; for a unidirectional ply it is ``S11T``, but only when
the requirement says the load acts along the fibres.

    EngineeringRequirement ──RequirementResolver(material system)──▶ ObservableResolution
        family · aspect · direction                                   RESOLVED   → property
        limit · unit · hard/soft · conditions                         AMBIGUOUS  → underspecified
                                                                      UNSUPPORTED→ no observable

Families are the six UAV property families (:class:`~domains.uav_materials.properties.Category`).
An *aspect* is the physical quantity within a family (tensile ultimate
strength, normal stiffness, ductility, ...). A *direction* is where it acts:
``unspecified``, material axes ``1`` / ``2`` / ``12`` (ply: fibre,
transverse, in-plane shear), or laminate axes ``x`` / ``y`` / ``xy``
(reserved for classical lamination theory; no system resolves them yet).

Resolution is a fixed lookup table per material system; there is no
free-form or model-generated semantics. Rules:

* ``isotropic_bulk`` (every existing record): each aspect resolves to its one
  bulk property, in any direction, under the stated isotropy assumption.
  Existing ``PropertyRequirement`` targets are *lifted* losslessly and resolve
  to exactly the property they name, so metal results are unchanged;
* ``continuous_fiber_ud_ply``: directional aspects resolve only with a
  material direction. ``unspecified`` is AMBIGUOUS (E11 or E22? S11T or
  S22T?). E11 never answers a directionless Young's modulus, S11T never a
  directionless tensile strength, and the brittle ply failure strain never a
  ductility (elongation-at-break) requirement;
* anything not in a system's table is UNSUPPORTED, with the reason.

"Missing" is not a resolution status: it is a RESOLVED requirement whose
property has no value for the candidate (reported by the evaluator).

* ``continuous_fiber_laminate``: laminate directions ``x`` / ``y`` / ``xy``
  resolve to Ex / Ey / Gxy / νxy; ``unspecified`` stays AMBIGUOUS (no
  quasi-isotropy is inferred from a layup); strength and failure are
  UNSUPPORTED (ply S11T is not a laminate strength).

New observables are added by registering a system table; requirement
semantics do not change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from domains.uav_materials.profile import (
    Importance,
    Operator,
    Priority,
    PropertyRequirement,
    TargetMaterialProfile,
)
from domains.uav_materials.properties import Category, property_def
from domains.uav_materials.schema import MaterialRecord, TestConditions
from jevpilot import FrozenModel, Provenance

VERSION = "requirement_resolver/1"


class Aspect(StrEnum):
    # strength
    TENSILE_ULTIMATE = "tensile_ultimate_strength"
    TENSILE_YIELD = "tensile_yield_strength"
    COMPRESSIVE_ULTIMATE = "compressive_ultimate_strength"
    COMPRESSIVE_YIELD = "compressive_yield_strength"
    SHEAR_ULTIMATE = "shear_ultimate_strength"
    # deformation
    NORMAL_STIFFNESS = "normal_stiffness"
    SHEAR_STIFFNESS = "shear_stiffness"
    POISSON_RATIO = "poisson_ratio"
    DUCTILITY = "ductility"  # plastic strain capacity: metallic elongation at break
    TENSILE_FAILURE_STRAIN = "tensile_failure_strain"  # strain at tensile fracture
    YIELD_STRAIN = "yield_strain"
    # other families
    DENSITY = "density"
    CORROSION_PENETRATION = "corrosion_penetration_rate"
    WATER_UPTAKE = "water_uptake"
    MAX_SERVICE_TEMPERATURE = "max_service_temperature"
    MIN_SERVICE_TEMPERATURE = "min_service_temperature"
    GLASS_TRANSITION = "glass_transition_temperature"
    MELTING = "melting_temperature"
    STRENGTH_RETENTION = "tensile_strength_retention"
    MODULUS_RETENTION = "modulus_retention"
    APPLICATION_TEMPERATURE_LIMIT = "application_temperature_limit"
    EXPOSURE_STABILITY_TEMPERATURE = "exposure_stability_temperature"


FAMILY: dict[Aspect, Category] = {
    **{
        a: Category.STRENGTH
        for a in (
            Aspect.TENSILE_ULTIMATE,
            Aspect.TENSILE_YIELD,
            Aspect.COMPRESSIVE_ULTIMATE,
            Aspect.COMPRESSIVE_YIELD,
            Aspect.SHEAR_ULTIMATE,
        )
    },
    **{
        a: Category.DEFORMATION
        for a in (
            Aspect.NORMAL_STIFFNESS,
            Aspect.SHEAR_STIFFNESS,
            Aspect.POISSON_RATIO,
            Aspect.DUCTILITY,
            Aspect.TENSILE_FAILURE_STRAIN,
            Aspect.YIELD_STRAIN,
        )
    },
    Aspect.DENSITY: Category.DENSITY,
    Aspect.CORROSION_PENETRATION: Category.CORROSION,
    Aspect.WATER_UPTAKE: Category.WATER_ABSORPTION,
    **{
        a: Category.TEMPERATURE
        for a in (
            Aspect.MAX_SERVICE_TEMPERATURE,
            Aspect.MIN_SERVICE_TEMPERATURE,
            Aspect.GLASS_TRANSITION,
            Aspect.MELTING,
            Aspect.STRENGTH_RETENTION,
            Aspect.MODULUS_RETENTION,
            Aspect.APPLICATION_TEMPERATURE_LIMIT,
            Aspect.EXPOSURE_STABILITY_TEMPERATURE,
        )
    },
}


class Direction(StrEnum):
    UNSPECIFIED = "unspecified"
    MATERIAL_1 = "material_1"  # ply: fibre (longitudinal) axis
    MATERIAL_2 = "material_2"  # ply: in-plane transverse axis
    MATERIAL_12 = "material_12"  # ply: in-plane shear (1-2 plane)
    LAMINATE_X = "laminate_x"  # reserved: classical lamination theory
    LAMINATE_Y = "laminate_y"
    LAMINATE_XY = "laminate_xy"


MATERIAL_AXES = (Direction.MATERIAL_1, Direction.MATERIAL_2, Direction.MATERIAL_12)
LAMINATE_AXES = (Direction.LAMINATE_X, Direction.LAMINATE_Y, Direction.LAMINATE_XY)


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"  # the requirement is underspecified for this material system
    UNSUPPORTED = "unsupported"  # the system/model has no observable that answers it


# -- engineering requirements ----------------------------------------------------------------


class EngineeringRequirement(FrozenModel):
    """Physical intent, independent of any material system."""

    aspect: Aspect
    direction: Direction = Direction.UNSPECIFIED
    loading: str = ""  # free description of the load case, informative only
    operator: Operator
    value: float | None = None
    upper: float | None = None
    unit: str | None = None
    conditions: TestConditions = Field(default_factory=TestConditions)
    priority: Priority
    weight: float | None = Field(default=None, ge=0.0)
    weight_source: str | None = None
    importance: Importance | None = None
    rationale: str = ""
    provenance: Provenance | None = None
    legacy: PropertyRequirement | None = None  # the property requirement it was lifted from

    @model_validator(mode="after")
    def _checked_through_a_property(self) -> Self:
        self.carrier()  # validates value/unit/conditions like any PropertyRequirement
        return self

    @property
    def family(self) -> Category:
        return FAMILY[self.aspect]

    @property
    def label(self) -> str:
        where = "" if self.direction is Direction.UNSPECIFIED else f"[{self.direction}]"
        if self.operator is Operator.BETWEEN:
            limit = f" in {self.value:g}-{self.upper:g} {self.unit}"
        elif self.value is not None:
            limit = f" {self.operator} {self.value:g} {self.unit}"
        else:
            limit = f" {self.operator}"
        return f"{self.aspect}{where}{limit}"

    @classmethod
    def lift(cls, r: PropertyRequirement) -> EngineeringRequirement:
        """A legacy property requirement, read as the engineering intent it names."""
        aspect, direction = LIFT[r.property]
        return cls(
            aspect=aspect,
            direction=direction,
            operator=r.operator,
            value=r.value,
            upper=r.upper,
            unit=r.unit,
            conditions=r.conditions,
            priority=r.priority,
            weight=r.weight,
            weight_source=r.weight_source,
            importance=r.importance,
            rationale=r.rationale,
            provenance=r.provenance,
            legacy=r,
        )

    def as_property_requirement(self, prop: str) -> PropertyRequirement:
        """The constraint the resolved property must meet (the legacy object if unchanged)."""
        if self.legacy is not None and self.legacy.property == prop:
            return self.legacy
        return PropertyRequirement(
            property=prop,
            operator=self.operator,
            value=self.value,
            upper=self.upper,
            unit=self.unit,
            conditions=self.conditions,
            priority=self.priority,
            weight=self.weight,
            weight_source=self.weight_source,
            importance=self.importance,
            rationale=self.rationale,
            provenance=self.provenance,
        )

    def carrier(self) -> PropertyRequirement:
        """A nominal property requirement used as this requirement's key in ranking/evidence.

        It is never checked directly: every check goes through the resolver.
        """
        if self.legacy is not None:
            return self.legacy
        prop = NOMINAL_DIRECTIONAL.get((self.aspect, self.direction), NOMINAL[self.aspect])
        return self.as_property_requirement(prop)


# -- material-system semantics ---------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    status: ResolutionStatus
    properties: tuple[str, ...] = ()
    reason: str = ""
    assumptions: tuple[str, ...] = ()


def _resolved(prop: str, reason: str, *assumptions: str) -> Rule:
    return Rule(ResolutionStatus.RESOLVED, (prop,), reason, assumptions)


def _ambiguous(reason: str, *options: str) -> Rule:
    return Rule(ResolutionStatus.AMBIGUOUS, options, reason)


def _unsupported(reason: str) -> Rule:
    return Rule(ResolutionStatus.UNSUPPORTED, (), reason)


@dataclass(frozen=True)
class MaterialSystemSemantics:
    system: str
    description: str
    rules: Mapping[tuple[Aspect, Direction], Rule]
    any_direction: Mapping[Aspect, Rule] = field(default_factory=dict)  # direction-free rules

    def rule(self, aspect: Aspect, direction: Direction) -> Rule:
        if (aspect, direction) in self.rules:
            return self.rules[(aspect, direction)]
        if aspect in self.any_direction:
            return self.any_direction[aspect]
        return _unsupported(f"{self.system} has no observable for {aspect} in {direction}")


BULK_PROPERTY: dict[Aspect, str] = {
    Aspect.TENSILE_ULTIMATE: "tensile_strength",
    Aspect.TENSILE_YIELD: "yield_strength",
    Aspect.COMPRESSIVE_ULTIMATE: "compressive_strength",
    Aspect.COMPRESSIVE_YIELD: "compressive_yield_strength",
    Aspect.SHEAR_ULTIMATE: "shear_strength",
    Aspect.NORMAL_STIFFNESS: "youngs_modulus",
    Aspect.SHEAR_STIFFNESS: "shear_modulus",
    Aspect.DUCTILITY: "elongation_at_break",
    Aspect.YIELD_STRAIN: "strain_at_yield",
    Aspect.DENSITY: "density",
    Aspect.CORROSION_PENETRATION: "corrosion_rate",
    Aspect.WATER_UPTAKE: "water_absorption",
    Aspect.MAX_SERVICE_TEMPERATURE: "max_service_temperature",
    Aspect.MIN_SERVICE_TEMPERATURE: "min_service_temperature",
    Aspect.GLASS_TRANSITION: "glass_transition_temperature",
    Aspect.MELTING: "melting_temperature",
    Aspect.STRENGTH_RETENTION: "tensile_strength_retention",
    Aspect.MODULUS_RETENTION: "modulus_retention",
    Aspect.APPLICATION_TEMPERATURE_LIMIT: "application_temperature_limit",
    Aspect.EXPOSURE_STABILITY_TEMPERATURE: "exposure_stability_temperature",
}
ISOTROPY = (
    "isotropic bulk material: one value per property for every direction (for MIL-HDBK-5 "
    "records the selection takes the conservative envelope over tested directions)"
)
SCALAR = "scalar property: direction does not apply"
SCALAR_ASPECTS = {
    Aspect.DENSITY,
    Aspect.CORROSION_PENETRATION,
    Aspect.WATER_UPTAKE,
    *(a for a, fam in FAMILY.items() if fam is Category.TEMPERATURE),
}


def _bulk_rule(aspect: Aspect, prop: str) -> Rule:
    if aspect in SCALAR_ASPECTS:
        return _resolved(prop, f"{aspect} is measured as {prop}", SCALAR)
    return _resolved(prop, f"{aspect} of an isotropic material is {prop}", ISOTROPY)


ISOTROPIC_BULK = MaterialSystemSemantics(
    system="isotropic_bulk",
    description="bulk (quasi-)isotropic materials: all existing handbook and report records",
    rules={},
    any_direction={
        **{a: _bulk_rule(a, p) for a, p in BULK_PROPERTY.items()},
        Aspect.TENSILE_FAILURE_STRAIN: _resolved(
            "elongation_at_break",
            "tensile strain at fracture of a bulk material is its elongation at break",
            ISOTROPY,
            "elongation at break includes plastic strain",
        ),
        Aspect.POISSON_RATIO: _unsupported("no Poisson ratio is recorded for bulk materials"),
    },
)

_PLY = "continuous_fiber_ud_ply"
_NEED_DIR = "a unidirectional ply is anisotropic; state the material direction"
_NO_LAMINATE = "laminate-axis response needs classical lamination theory (not implemented)"
_PLY_MODEL = "not predicted by continuous-fiber-micromechanics/1"
_ply_rules: dict[tuple[Aspect, Direction], Rule] = {
    (Aspect.NORMAL_STIFFNESS, Direction.MATERIAL_1): _resolved(
        "ply_longitudinal_modulus", "normal stiffness along the fibres is E11"
    ),
    (Aspect.NORMAL_STIFFNESS, Direction.MATERIAL_2): _resolved(
        "ply_transverse_modulus", "normal stiffness transverse to the fibres is E22"
    ),
    (Aspect.NORMAL_STIFFNESS, Direction.UNSPECIFIED): _ambiguous(
        f"{_NEED_DIR}: E11 and E22 differ by an order of magnitude",
        "ply_longitudinal_modulus",
        "ply_transverse_modulus",
    ),
    (Aspect.SHEAR_STIFFNESS, Direction.MATERIAL_12): _resolved(
        "ply_inplane_shear_modulus", "in-plane shear stiffness is G12"
    ),
    (Aspect.SHEAR_STIFFNESS, Direction.UNSPECIFIED): _ambiguous(
        f"{_NEED_DIR}: in-plane (G12) and transverse (G23) shear differ",
        "ply_inplane_shear_modulus",
    ),
    (Aspect.POISSON_RATIO, Direction.MATERIAL_12): _resolved(
        "ply_major_poisson_ratio", "major Poisson ratio nu12"
    ),
    (Aspect.POISSON_RATIO, Direction.UNSPECIFIED): _ambiguous(
        f"{_NEED_DIR}: nu12 and nu21 differ", "ply_major_poisson_ratio"
    ),
    (Aspect.TENSILE_ULTIMATE, Direction.MATERIAL_1): _resolved(
        "ply_longitudinal_tensile_strength",
        "tensile strength along the fibres is S11T",
        "first-order fibre-breakage estimate; see its prediction limitations",
    ),
    (Aspect.TENSILE_ULTIMATE, Direction.MATERIAL_2): _unsupported(
        f"transverse tensile strength S22T is {_PLY_MODEL}"
    ),
    (Aspect.TENSILE_ULTIMATE, Direction.UNSPECIFIED): _ambiguous(
        f"{_NEED_DIR}: S11T and S22T differ by more than an order of magnitude",
        "ply_longitudinal_tensile_strength",
    ),
    (Aspect.TENSILE_FAILURE_STRAIN, Direction.MATERIAL_1): _resolved(
        "ply_longitudinal_tensile_failure_strain",
        "tensile strain at fracture along the fibres",
        "brittle failure: no plastic strain",
    ),
    (Aspect.TENSILE_FAILURE_STRAIN, Direction.UNSPECIFIED): _ambiguous(
        f"{_NEED_DIR}: failure strain depends on the load direction",
        "ply_longitudinal_tensile_failure_strain",
    ),
    **{
        (a, d): _unsupported(_NO_LAMINATE)
        for a in (
            Aspect.NORMAL_STIFFNESS,
            Aspect.SHEAR_STIFFNESS,
            Aspect.POISSON_RATIO,
            Aspect.TENSILE_ULTIMATE,
            Aspect.TENSILE_FAILURE_STRAIN,
        )
        for d in LAMINATE_AXES
    },
}
UD_PLY = MaterialSystemSemantics(
    system=_PLY,
    description="one unidirectional continuous-fibre polymer-matrix ply (material axes 1, 2)",
    rules=_ply_rules,
    any_direction={
        Aspect.DENSITY: _resolved("density", "ply density", SCALAR),
        Aspect.DUCTILITY: _unsupported(
            "ductility (metallic elongation at break) has no ply observable; the brittle "
            "ply failure strain is a different quantity (use tensile_failure_strain with a "
            "material direction)"
        ),
        **{
            a: _unsupported(f"{a} is {_PLY_MODEL}")
            for a in (
                Aspect.CORROSION_PENETRATION,
                Aspect.WATER_UPTAKE,
                Aspect.TENSILE_YIELD,
                Aspect.COMPRESSIVE_ULTIMATE,
                Aspect.COMPRESSIVE_YIELD,
                Aspect.SHEAR_ULTIMATE,
                Aspect.YIELD_STRAIN,
                *(a for a, fam in FAMILY.items() if fam is Category.TEMPERATURE),
            )
        },
    },
)

_LAM = "continuous_fiber_laminate"
_LAM_MODEL = "not predicted by the laminate model (classical lamination theory, stiffness only)"
_NEED_LAM_DIR = "a laminate is in general anisotropic; state the laminate direction (x, y, xy)"
_lam_rules: dict[tuple[Aspect, Direction], Rule] = {
    (Aspect.NORMAL_STIFFNESS, Direction.LAMINATE_X): _resolved(
        "laminate_ex",
        "in-plane normal stiffness along laminate x is Ex",
        "membrane (in-plane) response of a symmetric laminate; bending stiffness is D",
    ),
    (Aspect.NORMAL_STIFFNESS, Direction.LAMINATE_Y): _resolved(
        "laminate_ey",
        "in-plane normal stiffness along laminate y is Ey",
        "membrane (in-plane) response of a symmetric laminate; bending stiffness is D",
    ),
    (Aspect.NORMAL_STIFFNESS, Direction.UNSPECIFIED): _ambiguous(
        f"{_NEED_LAM_DIR}; no quasi-isotropy is assumed from the layup",
        "laminate_ex",
        "laminate_ey",
    ),
    (Aspect.SHEAR_STIFFNESS, Direction.LAMINATE_XY): _resolved(
        "laminate_gxy",
        "in-plane shear stiffness in the x-y plane is Gxy",
        "membrane (in-plane) response of a symmetric laminate",
    ),
    (Aspect.SHEAR_STIFFNESS, Direction.UNSPECIFIED): _ambiguous(_NEED_LAM_DIR, "laminate_gxy"),
    (Aspect.POISSON_RATIO, Direction.LAMINATE_XY): _resolved(
        "laminate_nuxy", "in-plane Poisson ratio nuxy (-ey/ex under load along x)"
    ),
    (Aspect.POISSON_RATIO, Direction.UNSPECIFIED): _ambiguous(_NEED_LAM_DIR, "laminate_nuxy"),
    **{
        (a, d): _unsupported(
            "material axes differ from ply to ply in a laminate; state a laminate direction"
        )
        for a in (Aspect.NORMAL_STIFFNESS, Aspect.SHEAR_STIFFNESS, Aspect.POISSON_RATIO)
        for d in MATERIAL_AXES
    },
}
LAMINATE = MaterialSystemSemantics(
    system=_LAM,
    description="symmetric laminate of identical unidirectional plies (laminate axes x, y)",
    rules=_lam_rules,
    any_direction={
        Aspect.DENSITY: _resolved("density", "laminate density", SCALAR),
        **{
            a: _unsupported(
                "laminate strength and failure are not implemented; ply S11T is not a "
                "laminate strength"
            )
            for a in (
                Aspect.TENSILE_ULTIMATE,
                Aspect.TENSILE_YIELD,
                Aspect.COMPRESSIVE_ULTIMATE,
                Aspect.COMPRESSIVE_YIELD,
                Aspect.SHEAR_ULTIMATE,
                Aspect.TENSILE_FAILURE_STRAIN,
                Aspect.YIELD_STRAIN,
            )
        },
        Aspect.DUCTILITY: _unsupported(
            "ductility (metallic elongation at break) has no laminate observable"
        ),
        **{
            a: _unsupported(f"{a} is {_LAM_MODEL}")
            for a in (
                Aspect.CORROSION_PENETRATION,
                Aspect.WATER_UPTAKE,
                *(a for a, fam in FAMILY.items() if fam is Category.TEMPERATURE),
            )
        },
    },
)

SYSTEMS: dict[str, MaterialSystemSemantics] = {
    ISOTROPIC_BULK.system: ISOTROPIC_BULK,
    UD_PLY.system: UD_PLY,
    LAMINATE.system: LAMINATE,
    # the synthetic architecture-test space predicts generic isotropic properties
    "synthetic_two_fraction_v0": ISOTROPIC_BULK,
}
DEFAULT_SYSTEM = ISOTROPIC_BULK.system

# legacy property → (aspect, direction) it expresses
LIFT: dict[str, tuple[Aspect, Direction]] = {
    **{p: (a, Direction.UNSPECIFIED) for a, p in BULK_PROPERTY.items()},
    "ply_longitudinal_modulus": (Aspect.NORMAL_STIFFNESS, Direction.MATERIAL_1),
    "ply_transverse_modulus": (Aspect.NORMAL_STIFFNESS, Direction.MATERIAL_2),
    "ply_inplane_shear_modulus": (Aspect.SHEAR_STIFFNESS, Direction.MATERIAL_12),
    "ply_major_poisson_ratio": (Aspect.POISSON_RATIO, Direction.MATERIAL_12),
    "ply_longitudinal_tensile_strength": (Aspect.TENSILE_ULTIMATE, Direction.MATERIAL_1),
    "ply_longitudinal_tensile_failure_strain": (
        Aspect.TENSILE_FAILURE_STRAIN,
        Direction.MATERIAL_1,
    ),
    "laminate_ex": (Aspect.NORMAL_STIFFNESS, Direction.LAMINATE_X),
    "laminate_ey": (Aspect.NORMAL_STIFFNESS, Direction.LAMINATE_Y),
    "laminate_gxy": (Aspect.SHEAR_STIFFNESS, Direction.LAMINATE_XY),
    "laminate_nuxy": (Aspect.POISSON_RATIO, Direction.LAMINATE_XY),
}
# nominal property per aspect: only a key/unit carrier for native engineering requirements
NOMINAL: dict[Aspect, str] = {
    **BULK_PROPERTY,
    Aspect.POISSON_RATIO: "ply_major_poisson_ratio",
    Aspect.TENSILE_FAILURE_STRAIN: "ply_longitudinal_tensile_failure_strain",
}
# Directional requirements get a direction-specific key, so that e.g. "maximize stiffness
# along x" and "... along y" are distinct preferences. Still only a key/unit carrier.
NOMINAL_DIRECTIONAL: dict[tuple[Aspect, Direction], str] = {
    (a, d): p for p, (a, d) in LIFT.items() if d is not Direction.UNSPECIFIED
}


class ObservableResolution(FrozenModel):
    requirement: EngineeringRequirement
    material_system: str
    status: ResolutionStatus
    properties: tuple[str, ...]  # resolved property; for AMBIGUOUS, the candidate observables
    direction: Direction
    reason: str
    assumptions: tuple[str, ...] = ()
    property_requirement: PropertyRequirement | None = None  # what the constraint check uses
    version: str = VERSION

    @property
    def resolved(self) -> bool:
        return self.status is ResolutionStatus.RESOLVED


def material_system_of(record: MaterialRecord) -> str:
    """Records without a declared system are legacy bulk records."""
    return str(record.metadata.get("material_system") or DEFAULT_SYSTEM)


class RequirementResolver:
    """Deterministic table lookup: (material system, aspect, direction) → observable."""

    def __init__(self, systems: Mapping[str, MaterialSystemSemantics] | None = None) -> None:
        self.systems = dict(SYSTEMS if systems is None else systems)

    def resolve(self, requirement: EngineeringRequirement, system: str) -> ObservableResolution:
        semantics = self.systems.get(system)
        rule = (
            semantics.rule(requirement.aspect, requirement.direction)
            if semantics
            else _unsupported(f"no requirement semantics registered for system {system!r}")
        )
        prop_req = None
        if rule.status is ResolutionStatus.RESOLVED:
            (prop,) = rule.properties
            if property_def(prop).category is not requirement.family:
                raise ValueError(f"rule maps {requirement.aspect} to {prop} of another family")
            prop_req = requirement.as_property_requirement(prop)
        return ObservableResolution(
            requirement=requirement,
            material_system=system,
            status=rule.status,
            properties=rule.properties,
            direction=requirement.direction,
            reason=rule.reason,
            assumptions=rule.assumptions,
            property_requirement=prop_req,
        )


# -- targets ----------------------------------------------------------------------------------


class EngineeringTarget(FrozenModel):
    """Engineering requirements, plus the carrier profile the existing machinery keys on."""

    profile_id: str
    name: str
    application: str = ""
    requirements: tuple[EngineeringRequirement, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_profile(cls, profile: TargetMaterialProfile) -> EngineeringTarget:
        return cls(
            profile_id=profile.profile_id,
            name=profile.name,
            application=profile.application,
            requirements=tuple(EngineeringRequirement.lift(r) for r in profile.requirements),
            metadata={"lifted_from": "TargetMaterialProfile"},
        )

    @model_validator(mode="after")
    def _unique_soft_keys(self) -> Self:
        keys = [
            f"{r.carrier().property}:{r.operator}"
            for r in self.requirements
            if r.priority is Priority.SOFT
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(f"soft preferences must have distinct keys: {keys}")
        return self

    def carrier_profile(self) -> TargetMaterialProfile:
        return TargetMaterialProfile(
            profile_id=self.profile_id,
            name=self.name,
            application=self.application,
            requirements=tuple(r.carrier() for r in self.requirements),
        )


class TargetBinding:
    """A target bound to the resolver: maps each carrier requirement to its resolution."""

    def __init__(
        self,
        target: EngineeringTarget | TargetMaterialProfile,
        resolver: RequirementResolver | None = None,
    ) -> None:
        self.target = (
            target
            if isinstance(target, EngineeringTarget)
            else EngineeringTarget.from_profile(target)
        )
        self.resolver = resolver or RequirementResolver()
        self.profile = (
            target if isinstance(target, TargetMaterialProfile) else self.target.carrier_profile()
        )
        self._by_carrier = dict(
            zip(map(id, self.profile.requirements), self.target.requirements, strict=True)
        )
        self._cache: dict[tuple[str, int], ObservableResolution] = {}

    def engineering(self, carrier: PropertyRequirement) -> EngineeringRequirement:
        return self._by_carrier[id(carrier)]

    def resolve(self, record: MaterialRecord, carrier: PropertyRequirement) -> ObservableResolution:
        system = material_system_of(record)
        key = (system, id(carrier))
        if key not in self._cache:
            self._cache[key] = self.resolver.resolve(self.engineering(carrier), system)
        return self._cache[key]

    def resolutions(self, record: MaterialRecord) -> tuple[ObservableResolution, ...]:
        return tuple(self.resolve(record, r) for r in self.profile.requirements)
