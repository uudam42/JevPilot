"""The existing-material search interface, with a minimal hard-constraint screen.

``search_materials(profile, candidates)`` returns a :class:`MaterialSearchResult`:
per candidate, which requirements are satisfied, violated or undetermined,
which measurements were missing, which measurements (and provenance) were
used, and a ``distance`` slot for similarity ranking.

Only a :class:`HardConstraintScreen` exists in this iteration, just enough to
exercise the interface. It:

* compares only after converting to the requirement's unit (same dimension);
* uses only measurements whose test conditions match the requirement's
  conditions (medium equal; temperature and exposure equal after conversion);
* is conservative: with several matching values, every one must satisfy the
  constraint;
* reports missing or non-comparable data as **undetermined**, never as a pass
  or a violation;
* does **not** score soft preferences (``distance`` stays ``None``).
Normalisation, distance and ranking are the next iteration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from domains.uav_materials.profile import Operator, PropertyRequirement, TargetMaterialProfile
from domains.uav_materials.schema import (
    MaterialCandidate,
    Measurement,
    Quantity,
    TestConditions,
)
from domains.uav_materials.units import UnitError
from jevpilot import FrozenModel, Provenance


class CheckStatus(StrEnum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNDETERMINED = "undetermined"


class Feasibility(StrEnum):
    FEASIBLE = "feasible"  # every hard constraint satisfied
    INFEASIBLE = "infeasible"  # at least one hard constraint violated
    UNDETERMINED = "undetermined"  # none violated, some could not be checked


class RequirementCheck(FrozenModel):
    requirement: PropertyRequirement
    status: CheckStatus
    reason: str
    used: tuple[Measurement, ...] = ()


class CandidateAssessment(FrozenModel):
    candidate_id: str
    origin: str
    feasibility: Feasibility
    checks: tuple[RequirementCheck, ...]
    missing: tuple[str, ...]  # properties with no usable measurement
    distance: float | None = None  # similarity to soft preferences: next iteration
    provenance: tuple[Provenance, ...] = ()  # of the measurements used

    @property
    def violations(self) -> tuple[RequirementCheck, ...]:
        return tuple(c for c in self.checks if c.status is CheckStatus.VIOLATED)


class MaterialSearchResult(FrozenModel):
    profile_id: str
    method: str
    assessments: tuple[CandidateAssessment, ...]
    notes: str = ""


class MaterialSearch(Protocol):
    name: str

    def search(
        self, profile: TargetMaterialProfile, candidates: Sequence[MaterialCandidate]
    ) -> MaterialSearchResult: ...


def conditions_match(required: TestConditions, observed: TestConditions) -> bool:
    """Every condition the requirement states must be reported identically by the measurement."""
    if required.medium is not None and required.medium != observed.medium:
        return False
    for name in ("temperature", "relative_humidity", "exposure_duration"):
        want: Quantity | None = getattr(required, name)
        if want is None:
            continue
        got: Quantity | None = getattr(observed, name)
        if got is None:
            return False
        try:
            if not math.isclose(got.to(want.unit), want.value, rel_tol=1e-6, abs_tol=1e-6):
                return False
        except UnitError:
            return False
    return True


def check(requirement: PropertyRequirement, candidate: MaterialCandidate) -> RequirementCheck:
    found = candidate.material.measurements(requirement.property)
    usable = [
        m
        for m in found
        if not m.is_missing
        and m.comparable_conditions
        and conditions_match(requirement.conditions, m.conditions)
    ]
    if not usable:
        why = (
            "no measurement"
            if not found
            else "only missing values: "
            + ", ".join(sorted({str(m.missing) for m in found if m.missing}))
            if all(m.is_missing for m in found)
            else "no measurement under the required conditions"
        )
        return RequirementCheck(
            requirement=requirement, status=CheckStatus.UNDETERMINED, reason=why
        )
    assert requirement.unit is not None
    lower, upper = requirement.bounds()
    values = [m.value_in(requirement.unit) for m in usable]
    ok = all(
        v is not None and (lower is None or v >= lower) and (upper is None or v <= upper)
        for v in values
    )
    status = CheckStatus.SATISFIED if ok else CheckStatus.VIOLATED
    shown = ", ".join(f"{v:.4g}" for v in values if v is not None)
    return RequirementCheck(
        requirement=requirement,
        status=status,
        used=tuple(usable),
        reason=f"{shown} {requirement.unit} vs {requirement.operator} "
        f"{requirement.value}{'' if requirement.upper is None else '–' + str(requirement.upper)}",
    )


class HardConstraintScreen:
    """Minimal engine: hard constraints only. Order: feasible, undetermined, infeasible."""

    name = "hard_constraint_screen/1"

    def search(
        self, profile: TargetMaterialProfile, candidates: Sequence[MaterialCandidate]
    ) -> MaterialSearchResult:
        order = {Feasibility.FEASIBLE: 0, Feasibility.UNDETERMINED: 1, Feasibility.INFEASIBLE: 2}
        assessments = sorted(
            (self.assess(profile, c) for c in candidates), key=lambda a: order[a.feasibility]
        )
        return MaterialSearchResult(
            profile_id=profile.profile_id,
            method=self.name,
            assessments=tuple(assessments),
            notes="hard constraints only; soft preferences not scored (distance=None)",
        )

    def assess(self, profile: TargetMaterialProfile, c: MaterialCandidate) -> CandidateAssessment:
        checks = tuple(
            check(r, c)
            for r in profile.hard_constraints
            if r.operator in (Operator.LE, Operator.GE, Operator.BETWEEN)
        )
        statuses = {k.status for k in checks}
        feasibility = (
            Feasibility.INFEASIBLE
            if CheckStatus.VIOLATED in statuses
            else Feasibility.UNDETERMINED
            if CheckStatus.UNDETERMINED in statuses
            else Feasibility.FEASIBLE
        )
        wanted = {r.property for r in profile.requirements}
        missing = tuple(sorted(p for p in wanted if c.material.status(p) != "measured"))
        provenance = tuple(m.provenance for k in checks for m in k.used if m.provenance)
        return CandidateAssessment(
            candidate_id=c.candidate_id,
            origin=str(c.origin),
            feasibility=feasibility,
            checks=checks,
            missing=missing,
            provenance=provenance,
        )


def search_materials(
    profile: TargetMaterialProfile,
    candidates: Sequence[MaterialCandidate],
    engine: MaterialSearch | None = None,
) -> MaterialSearchResult:
    return (engine or HardConstraintScreen()).search(profile, candidates)
