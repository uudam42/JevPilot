"""Material measurements, material records and candidates.

A :class:`Measurement` is one physical value of one registered property,
with its unit, test conditions, test method, basis (measured, specified,
derived or predicted), provenance and uncertainty. It reuses JevPilot's
:class:`~jevpilot.Provenance` and :class:`~jevpilot.Uncertainty`. A value
that is absent is stated as absent (:class:`MissingReason`), never as ``0``.

A :class:`MaterialRecord` groups measurements by requirement family
(strength, deformation, corrosion, density, water absorption, temperature
effects). Each group only accepts properties of its own category, so strength
can never be confused with stiffness. A :class:`MaterialCandidate` wraps a record
with its origin, so existing and future proposed/composite/virtual materials
can be evaluated against the same target profile.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterator
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from domains.uav_materials.properties import Category, property_def
from domains.uav_materials.units import UnitError, convert, dimension_of
from jevpilot import FrozenModel, Provenance, Uncertainty


class MissingReason(StrEnum):
    UNKNOWN = "unknown"  # may exist, but we do not know it
    NOT_MEASURED = "not_measured"  # known not to have been measured
    NOT_APPLICABLE = "not_applicable"  # meaningless for this material (e.g. Tg of a metal)


class MeasurementBasis(StrEnum):
    MEASURED = "measured"  # test result
    SPECIFIED = "specified"  # datasheet / specification minimum or typical
    DERIVED = "derived"  # computed from other measurements
    PREDICTED = "predicted"  # model / simulation output (future candidates)


class ProvenanceStatus(StrEnum):
    SOURCED = "sourced"  # provenance with at least one source
    UNSOURCED = "unsourced"  # allowed during development, never silently
    SYNTHETIC = "synthetic"  # invented test fixture data


class Quantity(FrozenModel):
    value: float
    unit: str

    @model_validator(mode="after")
    def _known_unit(self) -> Self:
        dimension_of(self.unit)
        return self

    def to(self, unit: str) -> float:
        return convert(self.value, self.unit, unit)


class TestConditions(FrozenModel):
    """Conditions a value was obtained under. ``None`` means *not reported*.

    Unreported conditions are never filled in with assumed defaults.
    """

    __test__ = False  # not a pytest test class

    temperature: Quantity | None = None
    relative_humidity: Quantity | None = None
    medium: str | None = None  # e.g. "salt water (3.5% NaCl)", "air", "fresh water"
    exposure_duration: Quantity | None = None
    specimen: str | None = None  # orientation, layup, thickness, ...
    other: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _dimensions(self) -> Self:
        for name, expected in (
            ("temperature", "temperature"),
            ("relative_humidity", "fraction"),
            ("exposure_duration", "time"),
        ):
            q = getattr(self, name)
            if q is not None and dimension_of(q.unit) != expected:
                raise UnitError(f"{name} must be a {expected}, got unit {q.unit!r}")
        return self

    def reported(self) -> set[str]:
        return {
            k
            for k in ("temperature", "relative_humidity", "medium", "exposure_duration", "specimen")
            if getattr(self, k) is not None
        }


class Measurement(FrozenModel):
    property: str
    value: float | None = None
    unit: str | None = None
    missing: MissingReason | None = None
    basis: MeasurementBasis = MeasurementBasis.MEASURED
    conditions: TestConditions = Field(default_factory=TestConditions)
    test_method: str | None = None  # e.g. a standard designation; None = not reported
    provenance: Provenance | None = None
    provenance_status: ProvenanceStatus
    uncertainty: Uncertainty | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        definition = property_def(self.property)
        if (self.value is None) == (self.missing is None):
            raise ValueError(f"{self.property}: give exactly one of value or missing")
        if self.value is not None:
            if self.unit is None:
                raise ValueError(f"{self.property}: a value needs a unit")
            if dimension_of(self.unit) != definition.dimension:
                raise UnitError(
                    f"{self.property} is a {definition.dimension}; unit {self.unit!r} is a "
                    f"{dimension_of(self.unit)}"
                )
        sources = self.provenance.sources if self.provenance else ()
        if self.provenance_status is ProvenanceStatus.SOURCED and not sources:
            raise ValueError(f"{self.property}: 'sourced' requires provenance with a source")
        if self.provenance_status is ProvenanceStatus.UNSOURCED and sources:
            raise ValueError(f"{self.property}: has sources but is marked 'unsourced'")
        return self

    @builtins.property  # the field named 'property' shadows the builtin here
    def category(self) -> Category:
        return property_def(self.property).category

    @builtins.property  # the field named 'property' shadows the builtin here
    def is_missing(self) -> bool:
        return self.value is None

    @builtins.property  # the field named 'property' shadows the builtin here
    def comparable_conditions(self) -> bool:
        """Whether the conditions this property needs for comparison were reported."""
        return set(property_def(self.property).required_conditions) <= self.conditions.reported()

    def value_in(self, unit: str) -> float | None:
        if self.value is None or self.unit is None:
            return None
        return convert(self.value, self.unit, unit)


class MaterialFamily(StrEnum):
    METAL = "metal"
    POLYMER = "polymer"
    COMPOSITE = "composite"
    CERAMIC = "ceramic"
    OTHER = "other"


_GROUPS: dict[str, Category] = {
    "density": Category.DENSITY,
    "strength": Category.STRENGTH,
    "deformation": Category.DEFORMATION,
    "corrosion": Category.CORROSION,
    "water_absorption": Category.WATER_ABSORPTION,
    "temperature_effects": Category.TEMPERATURE,
}


class MaterialRecord(FrozenModel):
    material_id: str
    name: str
    family: MaterialFamily
    density: tuple[Measurement, ...] = ()
    strength: tuple[Measurement, ...] = ()
    deformation: tuple[Measurement, ...] = ()
    corrosion: tuple[Measurement, ...] = ()
    water_absorption: tuple[Measurement, ...] = ()
    temperature_effects: tuple[Measurement, ...] = ()
    provenance: Provenance | None = None
    provenance_status: ProvenanceStatus
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _groups_hold_their_category(self) -> Self:
        for group, category in _GROUPS.items():
            for m in getattr(self, group):
                if m.category is not category:
                    raise ValueError(
                        f"{self.material_id}: {m.property} ({m.category}) cannot be stored "
                        f"under {group}"
                    )
        return self

    def all_measurements(self) -> Iterator[Measurement]:
        for group in _GROUPS:
            yield from getattr(self, group)

    def measurements(self, prop: str) -> tuple[Measurement, ...]:
        property_def(prop)  # reject typos
        return tuple(m for m in self.all_measurements() if m.property == prop)

    def status(self, prop: str) -> str:
        """``measured``, a :class:`MissingReason`, or ``unknown`` when not recorded at all."""
        found = self.measurements(prop)
        if any(not m.is_missing for m in found):
            return "measured"
        if found:
            return str(found[0].missing)
        return str(MissingReason.UNKNOWN)


class CandidateOrigin(StrEnum):
    EXISTING = "existing"  # a real material from a database
    PROPOSED = "proposed"  # a new material concept (future design step)
    COMPOSITE = "composite"  # a combination of existing materials (future)
    VIRTUAL = "virtual"  # a purely computational candidate (future optimizer)


class MaterialCandidate(FrozenModel):
    """Anything that can be judged against a :class:`TargetMaterialProfile`.

    Proposed, composite and virtual candidates use the same record type. Their
    measurements are expected to carry ``basis=predicted`` or ``derived``.
    """

    candidate_id: str
    origin: CandidateOrigin
    material: MaterialRecord
    parents: tuple[str, ...] = ()  # material_ids a composite/proposal derives from
    design: dict[str, Any] = Field(default_factory=dict)  # future: recipe, layup, fractions
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _origin_rules(self) -> Self:
        if self.origin is CandidateOrigin.EXISTING:
            if self.parents or self.design:
                raise ValueError("an existing material has no parents or design parameters")
            if any(m.basis is MeasurementBasis.PREDICTED for m in self.material.all_measurements()):
                raise ValueError("an existing material's properties cannot be predictions")
        if self.origin is CandidateOrigin.COMPOSITE and len(self.parents) < 2:
            raise ValueError("a composite candidate needs at least two parent materials")
        return self
