"""Inverse-design contracts: design spaces, candidate designs, property predictors.

The inverse-design problem, in this domain's terms::

    z  ∈ Z                       a CandidateDesign in a MaterialDesignSpace
    x̂ = f(z)                     a PropertyPredictor's PredictedPropertyProfile
    E(f(z), x*)                  the same evaluation as for an existing material x_i
    z* = argmin_z D(f(z), x*)    subject to g_j(f(z)) ≤ 0  (future optimiser; not here)

Nothing here fixes a universal design schema. A design space declares its own
variables (composition, constituents, volume fractions, microstructure,
processing, architecture, ...) and constraints; a design is only meaningful
inside the space that validates it.

A predictor must be a real model: analytical, empirical, surrogate or an
external simulator, with its identity, version and assumptions recorded.
Values are never produced by an LLM. Every predicted value is a
:class:`~domains.uav_materials.schema.Measurement` with ``basis=predicted``
and a ``kind="model"`` source, so a prediction can be compared with a
measurement but can never pass for one.

The only predictor implemented is :class:`SyntheticLinearPredictor`: fake
numbers for architecture tests. It is not engineering-valid.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from domains.uav_materials.profile import OperatingEnvironment
from domains.uav_materials.schema import (
    CandidateOrigin,
    MaterialCandidate,
    MaterialFamily,
    MaterialRecord,
    Measurement,
    MeasurementBasis,
    ProvenanceStatus,
)
from jevpilot import FrozenModel, Provenance, SourceRef, Uncertainty, stable_digest

DesignValue = float | int | str


# -- design variables and constraints ------------------------------------------------------


class VariableKind(StrEnum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"


class DesignVariable(FrozenModel):
    """One variable a design space lets the designer change."""

    name: str
    kind: VariableKind
    unit: str | None = None  # None: dimensionless or categorical
    lower: float | None = None
    upper: float | None = None
    choices: tuple[str, ...] = ()  # categorical only
    description: str = ""

    def violations(self, value: Any) -> list[str]:
        if self.kind is VariableKind.CATEGORICAL:
            ok = isinstance(value, str) and value in self.choices
            return [] if ok else [f"{self.name}={value!r} is not one of {list(self.choices)}"]
        if isinstance(value, bool) or not isinstance(value, int | float):
            return [f"{self.name}={value!r} is not a number"]
        if self.kind is VariableKind.INTEGER and not float(value).is_integer():
            return [f"{self.name}={value!r} is not an integer"]
        if not math.isfinite(value):
            return [f"{self.name}={value!r} is not finite"]
        out = []
        if self.lower is not None and value < self.lower:
            out.append(f"{self.name}={value} < lower bound {self.lower}")
        if self.upper is not None and value > self.upper:
            out.append(f"{self.name}={value} > upper bound {self.upper}")
        return out


class LinearConstraint(FrozenModel):
    """Σ coefficient·variable  (<=, ==, >=)  bound, e.g. volume fractions summing to 1."""

    name: str
    coefficients: dict[str, float]
    operator: str = Field(pattern=r"^(<=|==|>=)$")
    bound: float
    tolerance: float = 1e-9

    def violations(self, values: Mapping[str, Any]) -> list[str]:
        missing = [k for k in self.coefficients if not isinstance(values.get(k), int | float)]
        if missing:
            return [f"{self.name}: numeric values needed for {missing}"]
        total = sum(c * float(values[k]) for k, c in self.coefficients.items())
        ok = {
            "<=": total <= self.bound + self.tolerance,
            ">=": total >= self.bound - self.tolerance,
            "==": abs(total - self.bound) <= self.tolerance,
        }[self.operator]
        return [] if ok else [f"{self.name}: {total:g} {self.operator} {self.bound:g} violated"]


# -- the design z --------------------------------------------------------------------------


class CandidateDesign(FrozenModel):
    """A point z in a design space, with the constraints it was generated under."""

    design_id: str
    design_space: str  # MaterialDesignSpace.space_id
    design_space_version: str
    variables: dict[str, DesignValue]
    variable_specs: tuple[DesignVariable, ...]  # snapshot of the space's variables
    constraints: tuple[LinearConstraint, ...] = ()  # snapshot of the space's constraints
    generation_method: str  # e.g. "manual", "grid", "optimizer:<name>/<version>"
    provenance: Provenance | None = None
    notes: str = ""


# -- forward models ------------------------------------------------------------------------


class ModelValidation(FrozenModel):
    """Reserved: how a forward model was validated. No validation infrastructure yet."""

    status: str = "not_validated"
    dataset: str | None = None
    error_metrics: dict[str, float] = Field(default_factory=dict)
    applicable_domain: str | None = None
    uncertainty: str | None = None
    version: str | None = None


class ModelIdentity(FrozenModel):
    name: str
    version: str
    kind: str  # analytical / empirical / surrogate / simulator / synthetic
    assumptions: tuple[str, ...] = ()
    # True: a real engineering model (not a test stub). This is NOT a validation claim;
    # what has been checked, and where, is in `validation` and per prediction.
    engineering_valid: bool  # False for test models: outputs must not inform decisions
    validation: ModelValidation = Field(default_factory=ModelValidation)
    equations: dict[str, str] = Field(default_factory=dict)  # property → equation
    sources: tuple[str, ...] = ()  # citations for the equations
    applicable_domain: str = ""
    unsupported: tuple[str, ...] = ()  # target properties the model does not predict

    @property
    def label(self) -> str:
        return f"{self.name}/{self.version}"


class PredictedPropertyProfile(FrozenModel):
    """x̂ = f(z): predicted properties, each a Measurement with basis=predicted."""

    design_id: str
    model: ModelIdentity
    predictions: tuple[Measurement, ...]
    not_predicted: tuple[str, ...] = ()  # properties the model does not cover (stay missing)

    @model_validator(mode="after")
    def _predictions_are_marked(self) -> Self:
        for m in self.predictions:
            if m.basis != MeasurementBasis.PREDICTED:
                raise ValueError(f"{m.property}: a prediction must have basis=predicted")
            src = m.provenance.sources if m.provenance else ()
            if not any(s.kind == "model" and s.identifier == self.model.label for s in src):
                raise ValueError(f"{m.property}: prediction must cite model {self.model.label}")
            if not self.model.engineering_valid and m.provenance_status != "synthetic":
                raise ValueError(f"{m.property}: a test model's output must be 'synthetic'")
        return self


@runtime_checkable
class PropertyPredictor(Protocol):
    """f(z) → x̂. Implementations: analytical, empirical, surrogate, simulator. Never an LLM."""

    model: ModelIdentity

    def predict(
        self, design: CandidateDesign, environment: OperatingEnvironment | None = None
    ) -> PredictedPropertyProfile: ...


# -- design spaces -------------------------------------------------------------------------


class DesignInvalid(ValueError):
    """The design is not a valid point of the design space."""


class MaterialDesignSpace(Protocol):
    space_id: str
    version: str
    family: MaterialFamily

    def validate_design(self, design: CandidateDesign) -> list[str]:
        """Violations (empty = valid)."""
        ...

    def candidate_from_design(
        self,
        design: CandidateDesign,
        predictor: PropertyPredictor,
        environment: OperatingEnvironment | None = None,
    ) -> MaterialCandidate: ...


class BoundedDesignSpace:
    """A design space defined by typed, bounded variables and linear constraints.

    A generic mechanism, not a universal schema: each material system declares
    its own variables and constraints.
    """

    def __init__(
        self,
        space_id: str,
        version: str,
        family: MaterialFamily,
        variables: tuple[DesignVariable, ...],
        constraints: tuple[LinearConstraint, ...] = (),
        origin: CandidateOrigin = CandidateOrigin.VIRTUAL,
    ) -> None:
        self.space_id = space_id
        self.version = version
        self.family = family
        self.variables = variables
        self.constraints = constraints
        self.origin = origin

    def design(
        self, values: dict[str, DesignValue], generation_method: str, notes: str = ""
    ) -> CandidateDesign:
        """Create a design (deterministic id from space + values) and validate it."""
        d = CandidateDesign(
            design_id=f"{self.space_id}:{stable_digest([self.version, values])[:12]}",
            design_space=self.space_id,
            design_space_version=self.version,
            variables=values,
            variable_specs=self.variables,
            constraints=self.constraints,
            generation_method=generation_method,
            notes=notes,
        )
        problems = self.validate_design(d)
        if problems:
            raise DesignInvalid("; ".join(problems))
        return d

    def validate_design(self, design: CandidateDesign) -> list[str]:
        out: list[str] = []
        if (design.design_space, design.design_space_version) != (self.space_id, self.version):
            out.append(
                f"design belongs to {design.design_space}/{design.design_space_version}, "
                f"not {self.space_id}/{self.version}"
            )
        names = {v.name for v in self.variables}
        out += [f"unknown variable {k!r}" for k in sorted(set(design.variables) - names)]
        for v in self.variables:
            if v.name not in design.variables:
                out.append(f"variable {v.name!r} missing")
            else:
                out += v.violations(design.variables[v.name])
        for c in self.constraints:
            out += c.violations(design.variables)
        return out

    def candidate_from_design(
        self,
        design: CandidateDesign,
        predictor: PropertyPredictor,
        environment: OperatingEnvironment | None = None,
    ) -> MaterialCandidate:
        problems = self.validate_design(design)
        if problems:
            raise DesignInvalid("; ".join(problems))
        return candidate_from_prediction(
            design, predictor.predict(design, environment), self.family, self.origin
        )


def candidate_from_prediction(
    design: CandidateDesign,
    predicted: PredictedPropertyProfile,
    family: MaterialFamily,
    origin: CandidateOrigin = CandidateOrigin.VIRTUAL,
    parents: tuple[str, ...] = (),
) -> MaterialCandidate:
    """Wrap x̂ in the same MaterialRecord/MaterialCandidate every evaluator consumes."""
    if predicted.design_id != design.design_id:
        raise ValueError("prediction belongs to a different design")
    if CandidateOrigin(origin) is CandidateOrigin.EXISTING:
        raise ValueError("a designed candidate cannot be an existing material")
    groups: dict[str, list[Measurement]] = {}
    for m in predicted.predictions:
        groups.setdefault(m.category.value, []).append(m)
    synthetic = not predicted.model.engineering_valid
    record = MaterialRecord(
        material_id=design.design_id,
        name=f"{design.design_id} (predicted by {predicted.model.label})",
        family=family,
        provenance=Provenance(
            sources=(
                SourceRef(kind="model", identifier=predicted.model.label),
                SourceRef(kind="design", identifier=design.design_id),
            ),
            metadata={"generation_method": design.generation_method},
        ),
        provenance_status=ProvenanceStatus.SYNTHETIC if synthetic else ProvenanceStatus.SOURCED,
        metadata={"predicted_by": predicted.model.label, "not_predicted": predicted.not_predicted},
        **{k: tuple(v) for k, v in groups.items()},
    )
    return MaterialCandidate(
        candidate_id=design.design_id,
        origin=origin,
        material=record,
        parents=parents,
        design=design.model_dump(mode="json"),
        metadata={
            "model": predicted.model.model_dump(mode="json"),
            "engineering_valid": predicted.model.engineering_valid,
        },
    )


# -- the one synthetic predictor (architecture tests only) --------------------------------

SYNTHETIC_SPACE = BoundedDesignSpace(
    space_id="synthetic_two_fraction_v0",
    version="0",
    family=MaterialFamily.OTHER,
    variables=(
        DesignVariable(name="fraction_a", kind=VariableKind.CONTINUOUS, lower=0, upper=1),
        DesignVariable(name="fraction_b", kind=VariableKind.CONTINUOUS, lower=0, upper=1),
    ),
    constraints=(
        LinearConstraint(
            name="fractions_at_most_one",
            coefficients={"fraction_a": 1, "fraction_b": 1},
            operator="<=",
            bound=1,
        ),
    ),
)


class SyntheticLinearPredictor:
    """FAKE forward model: fixed linear maps from two fractions to three properties.

    SYNTHETIC — PREDICTED — NOT ENGINEERING-VALID. The coefficients are
    arbitrary round numbers chosen for tests; they describe no material.
    It predicts density, tensile strength and Young's modulus only, so every
    other property stays missing.
    """

    model = ModelIdentity(
        name="synthetic-linear",
        version="0",
        kind="synthetic",
        assumptions=(
            "arbitrary linear coefficients chosen for architecture tests",
            "no physical basis; outputs are not engineering-valid",
            "no uncertainty model",
        ),
        engineering_valid=False,
    )
    # property → (unit, intercept, {variable: slope}); arbitrary test numbers
    MAPS: dict[str, tuple[str, float, dict[str, float]]] = {
        "density": ("kg/m^3", 1500.0, {"fraction_a": 1500.0}),
        "tensile_strength": ("MPa", 200.0, {"fraction_b": 400.0}),
        "youngs_modulus": ("GPa", 20.0, {"fraction_a": 80.0}),
    }

    def predict(
        self, design: CandidateDesign, environment: OperatingEnvironment | None = None
    ) -> PredictedPropertyProfile:
        if design.design_space != SYNTHETIC_SPACE.space_id:
            raise DesignInvalid(f"{self.model.label} only covers {SYNTHETIC_SPACE.space_id}")
        predictions = []
        for prop, (unit, intercept, slopes) in self.MAPS.items():
            value = intercept + sum(
                slope * float(design.variables[k]) for k, slope in slopes.items()
            )
            predictions.append(
                Measurement(
                    property=prop,
                    value=value,
                    unit=unit,
                    basis=MeasurementBasis.PREDICTED,
                    test_method=None,
                    provenance=Provenance(
                        id="prov_" + stable_digest([self.model.label, design.design_id, prop])[:16],
                        sources=(
                            SourceRef(
                                kind="model",
                                identifier=self.model.label,
                                version=self.model.version,
                                metadata={
                                    "design_id": design.design_id,
                                    "design_space": design.design_space,
                                    "assumptions": list(self.model.assumptions),
                                    "engineering_valid": False,
                                },
                            ),
                        ),
                        metadata={"model_kind": self.model.kind},
                    ),
                    provenance_status=ProvenanceStatus.SYNTHETIC,
                    uncertainty=Uncertainty.unknown("synthetic model; uncertainty not modelled"),
                    notes="SYNTHETIC PREDICTION - NOT ENGINEERING-VALID",
                )
            )
        return PredictedPropertyProfile(
            design_id=design.design_id,
            model=self.model,
            predictions=tuple(predictions),
            not_predicted=tuple(
                sorted(
                    {
                        "corrosion_rate",
                        "elongation_at_break",
                        "max_service_temperature",
                        "yield_strength",
                        "water_absorption",
                    }
                )
            ),
        )
