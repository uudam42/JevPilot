"""What the UAV design wants from a material: the target property profile.

A :class:`TargetMaterialProfile` is a set of :class:`PropertyRequirement`\\ s
over registered physical properties, with units and (optionally) the
conditions they apply under:

* **hard constraints** (``priority=hard``, operator ``<=``, ``>=`` or
  ``between``). Violating one makes a candidate infeasible;
* **soft preferences** (``priority=soft``): ``maximize`` or ``minimize``;
* **target ranges** (``between``, hard or soft) and **target values**
  (``target``, soft only);
* **relative importance**: a qualitative ``importance`` and/or a numerical
  ``weight``. A weight must say where it came from (``weight_source``).
  Nothing here invents numbers from qualitative importance.

The physical profile is deliberately *not* a feature vector. Turning it into
a search vector (normalisation, feature extraction) is a separate, later step,
typed here as :class:`FeatureExtractor` → :class:`SearchFeatureVector`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, Self

from pydantic import Field, model_validator

from domains.uav_materials.properties import property_def
from domains.uav_materials.schema import Quantity, TestConditions
from domains.uav_materials.units import UnitError, dimension_of
from jevpilot import FrozenModel, Provenance


class Priority(StrEnum):
    HARD = "hard"
    SOFT = "soft"


class Operator(StrEnum):
    LE = "<="
    GE = ">="
    BETWEEN = "between"  # value ≤ x ≤ upper
    TARGET = "target"  # as close as possible to value
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Importance(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


BOUNDED = {Operator.LE, Operator.GE, Operator.BETWEEN, Operator.TARGET}
DIRECTIONAL = {Operator.MAXIMIZE, Operator.MINIMIZE}


class PropertyRequirement(FrozenModel):
    property: str
    operator: Operator
    value: float | None = None
    upper: float | None = None  # only for "between"
    unit: str | None = None
    conditions: TestConditions = Field(default_factory=TestConditions)
    priority: Priority
    weight: float | None = Field(default=None, ge=0.0)
    weight_source: str | None = None
    importance: Importance | None = None
    rationale: str = ""
    provenance: Provenance | None = None  # where the requirement came from

    @model_validator(mode="after")
    def _well_formed(self) -> Self:
        definition = property_def(self.property)
        op = self.operator
        if op in BOUNDED:
            if self.value is None or self.unit is None:
                raise ValueError(f"{self.property} {op}: needs a value and a unit")
            if dimension_of(self.unit) != definition.dimension:
                raise UnitError(
                    f"{self.property} is a {definition.dimension}; "
                    f"unit {self.unit!r} is a {dimension_of(self.unit)}"
                )
        else:
            if self.value is not None or self.upper is not None:
                raise ValueError(f"{self.property} {op}: takes no value")
            if self.priority is Priority.HARD:
                raise ValueError(f"{self.property}: '{op}' can only be a soft preference")
        if op is Operator.TARGET and self.priority is Priority.HARD:
            raise ValueError(f"{self.property}: an exact 'target' can only be soft; use 'between'")
        unstated = set(definition.required_conditions) - self.conditions.reported()
        if unstated:
            raise ValueError(
                f"{self.property} depends on {sorted(unstated)}; state them in "
                "'conditions' so only comparable measurements are used"
            )
        if op is Operator.BETWEEN:
            if self.upper is None or self.value is None or self.upper < self.value:
                raise ValueError(f"{self.property} between: needs value <= upper")
        elif self.upper is not None:
            raise ValueError(f"{self.property} {op}: 'upper' is only valid for 'between'")
        if self.priority is Priority.HARD and (self.weight is not None or self.importance):
            raise ValueError(f"{self.property}: hard constraints carry no weight or importance")
        if self.weight is not None and not self.weight_source:
            raise ValueError(f"{self.property}: a numerical weight must state weight_source")
        return self

    def bounds(self, unit: str | None = None) -> tuple[float | None, float | None]:
        """(lower, upper) in ``unit`` (default: the requirement's unit); None = unbounded."""
        if self.value is None or self.unit is None:
            return None, None
        to = unit or self.unit
        v = Quantity(value=self.value, unit=self.unit).to(to)
        if self.operator is Operator.LE:
            return None, v
        if self.operator is Operator.GE:
            return v, None
        if self.operator is Operator.BETWEEN:
            assert self.upper is not None
            return v, Quantity(value=self.upper, unit=self.unit).to(to)
        return v, v  # target


class OperatingEnvironment(FrozenModel):
    """Where the aircraft operates. Carries the *fluid* side of buoyancy."""

    description: str = ""
    fluid: str | None = None  # e.g. "sea water"
    fluid_density: Quantity | None = None
    temperature_min: Quantity | None = None
    temperature_max: Quantity | None = None
    relative_humidity: Quantity | None = None

    @model_validator(mode="after")
    def _dimensions(self) -> Self:
        expected = {
            "fluid_density": "density",
            "temperature_min": "temperature",
            "temperature_max": "temperature",
            "relative_humidity": "fraction",
        }
        for name, dim in expected.items():
            q = getattr(self, name)
            if q is not None and dimension_of(q.unit) != dim:
                raise UnitError(f"{name} must be a {dim}, got unit {q.unit!r}")
        return self


class TargetMaterialProfile(FrozenModel):
    profile_id: str
    name: str
    application: str = ""  # e.g. "fixed-wing UAV wing spar (synthetic example)"
    environment: OperatingEnvironment = Field(default_factory=OperatingEnvironment)
    requirements: tuple[PropertyRequirement, ...]
    provenance: Provenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not self.requirements:
            raise ValueError("a target profile needs at least one requirement")
        return self

    @property
    def hard_constraints(self) -> tuple[PropertyRequirement, ...]:
        return tuple(r for r in self.requirements if r.priority is Priority.HARD)

    @property
    def soft_preferences(self) -> tuple[PropertyRequirement, ...]:
        return tuple(r for r in self.requirements if r.priority is Priority.SOFT)

    @property
    def ranges(self) -> tuple[PropertyRequirement, ...]:
        return tuple(r for r in self.requirements if r.operator is Operator.BETWEEN)

    def weights(self) -> dict[str, dict[str, Any]]:
        """Numerical soft weights exactly as given, with their sources (no rescaling)."""
        return {
            f"{r.property}:{r.operator}": {"weight": r.weight, "source": r.weight_source}
            for r in self.soft_preferences
            if r.weight is not None
        }

    def normalized_weights(self) -> dict[str, float]:
        """Weights rescaled to sum to 1. Refuses when any soft preference lacks a weight."""
        soft = self.soft_preferences
        missing = [r.property for r in soft if r.weight is None]
        if missing:
            raise ValueError(f"no numerical weight given for soft preferences {missing}")
        total = sum(r.weight or 0.0 for r in soft)
        if total <= 0:
            raise ValueError("soft preference weights sum to zero")
        return {f"{r.property}:{r.operator}": (r.weight or 0.0) / total for r in soft}


# -- the separate, later search representation --------------------------------------


class SearchFeatureVector(FrozenModel):
    """A numeric vector derived from a profile or a material, for similarity search.

    ``features`` values may be ``None`` (unknown) and are never imputed here.
    ``derivation`` records how each feature was computed (normaliser, units,
    conditions), so the vector can be traced back to physical quantities.
    """

    features: dict[str, float | None]
    derivation: dict[str, str]
    extractor: str


class FeatureExtractor(Protocol):
    """Physical profile / material → search vector. Implemented in a later iteration."""

    name: str

    def target_vector(self, profile: TargetMaterialProfile) -> SearchFeatureVector: ...

    def material_vector(
        self, material: Any, profile: TargetMaterialProfile
    ) -> SearchFeatureVector: ...
