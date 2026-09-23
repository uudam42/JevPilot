"""Unified evaluation of existing and designed candidates against one target.

Mathematical model
------------------

* target profile                     x*  (TargetMaterialProfile: hard constraints g_j,
                                          weighted soft preferences)
* existing candidate i               x_i (measured / datasheet / derived properties)
* designed candidate i               z_i (CandidateDesign in a MaterialDesignSpace)
* forward prediction                 x̂_i = f(z_i)  (PropertyPredictor)
* unified evaluation                 E(x, x*)
    - existing path                  E(x_i, x*)
    - inverse-design path            E(f(z_i), x*)
* future inverse-design objective    z* = argmin_z D(f(z), x*)  s.t.  g_j(f(z)) ≤ 0
  (no optimiser is implemented)

E is one function for both paths, because both kinds of candidate are the same
:class:`~domains.uav_materials.schema.MaterialCandidate` holding a
:class:`~domains.uav_materials.schema.MaterialRecord` of Measurements. That
record *is* the candidate property profile (:class:`CandidatePropertyProfile`
is a read-only view over it). E reuses, unchanged:

* hard constraints g_j and condition compatibility   search.screen / selection.select
* distance D, pessimistic distance, coverage         ranking.TargetRelativeExtractor,
                                                     fitted by search.fit_normalization
* evidence sufficiency and gap classification        evidence.assess

What E does **not** do is merge evidence quality into D. A predicted value and
a measured value give the same D if they are equal; the evaluation reports
separately which properties were predicted, by which model, whether that model
is engineering-valid, and what is known about uncertainty.

Distances are relative to a normalisation pool (the best viable value per
preference), so they are only comparable within one :class:`EvaluationContext`.
To place a designed candidate on the scale of the existing materials without
changing their results, fit the context on the existing candidates and
evaluate the designed one in it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from domains.uav_materials.evidence import (
    EvidenceAssessment,
    EvidencePolicy,
    RequirementEvidence,
    assess,
)
from domains.uav_materials.profile import TargetMaterialProfile
from domains.uav_materials.schema import (
    CandidateOrigin,
    EvidenceType,
    MaterialCandidate,
    MaterialRecord,
    Measurement,
)
from domains.uav_materials.search import (
    GROUP_ORDER,
    Contribution,
    Feasibility,
    RequirementCheck,
    fit_normalization,
    screen,
)
from jevpilot import FrozenModel


@dataclass(frozen=True)
class CandidatePropertyProfile:
    """Read-only view of a candidate's properties. It holds references, never copies."""

    candidate_id: str
    origin: CandidateOrigin
    record: MaterialRecord

    @classmethod
    def of(cls, candidate: MaterialCandidate) -> CandidatePropertyProfile:
        return cls(candidate.candidate_id, candidate.origin, candidate.material)

    def properties(self) -> tuple[str, ...]:
        return tuple(sorted({m.property for m in self.record.all_measurements()}))

    def measurements(self, prop: str) -> tuple[Measurement, ...]:
        return self.record.measurements(prop)

    def evidence_types(self, prop: str) -> tuple[EvidenceType, ...]:
        found = (m.evidence_type for m in self.measurements(prop) if not m.is_missing)
        return tuple(sorted(set(found)))


def candidate_from_record(record: MaterialRecord) -> MaterialCandidate:
    """Existing-material adapter: the record is wrapped, not copied."""
    return MaterialCandidate(
        candidate_id=record.material_id, origin=CandidateOrigin.EXISTING, material=record
    )


class CandidateEvaluation(FrozenModel):
    candidate_id: str
    name: str
    origin: CandidateOrigin
    rank: int | None = None  # set by evaluate_candidates
    # hard constraints (unchanged search semantics)
    feasibility: Feasibility
    checks: tuple[RequirementCheck, ...]
    # property-space distance to the target (unchanged ranking formulas)
    distance: float | None
    pessimistic_distance: float | None
    coverage: float | None
    contributions: tuple[Contribution, ...]
    # what the numbers rest on, reported separately from the distance
    properties_used: tuple[RequirementEvidence, ...]
    evidence_types: dict[str, int]  # over properties_used
    prediction_status: str  # none / partial / all (share of properties_used predicted)
    models: tuple[str, ...]  # forward models behind predicted values
    engineering_valid: bool | None  # False: a test model; None: no model involved
    uncertainty_known: tuple[int, int]  # (values with known uncertainty, values used)
    evidence: EvidenceAssessment  # classification, coverage, gaps, next evidence
    warnings: tuple[str, ...]


class EvaluationContext:
    """One normalisation scale plus one evidence policy, for comparable evaluations."""

    def __init__(
        self,
        target: TargetMaterialProfile,
        extractor: Any,
        pool_label: str,
        policy: EvidencePolicy | None = None,
    ) -> None:
        self.target = target
        self.extractor = extractor
        self.pool_label = pool_label
        self.policy = policy or EvidencePolicy()

    @classmethod
    def fit(
        cls,
        target: TargetMaterialProfile,
        reference: Sequence[MaterialCandidate],
        policy: EvidencePolicy | None = None,
    ) -> EvaluationContext:
        screened = [(c, screen(target, c)) for c in reference]
        extractor, label = fit_normalization(target, screened)
        return cls(target, extractor, label, policy)

    def evaluate(self, candidate: MaterialCandidate) -> CandidateEvaluation:
        base = screen(self.target, candidate)
        scored = self.extractor.score(candidate.material)
        ev = assess(self.target, candidate, self.policy)
        used = tuple(e for e in ev.requirements if e.evidence_type is not None)
        types: dict[str, int] = {}
        for e in used:
            assert e.evidence_type is not None
            types[e.evidence_type] = types.get(e.evidence_type, 0) + 1
        predicted = types.get(EvidenceType.PREDICTED.value, 0)
        status = "none" if not predicted else "all" if predicted == len(used) else "partial"
        models = tuple(sorted({e.model for e in used if e.model}))
        valid = candidate.metadata.get("engineering_valid")
        warnings: list[str] = []
        if predicted:
            warnings.append(
                f"{predicted} of {len(used)} values used are predictions ({', '.join(models)}), "
                "not experimental evidence; no model-validation policy exists yet"
            )
        if valid is False:
            warnings.append("SYNTHETIC TEST MODEL - results are not engineering-valid")
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            name=candidate.material.name,
            origin=candidate.origin,
            feasibility=base.feasibility,
            checks=base.checks,
            distance=scored.distance,
            pessimistic_distance=scored.pessimistic_distance,
            coverage=scored.coverage,
            contributions=scored.contributions,
            properties_used=used,
            evidence_types=dict(sorted(types.items())),
            prediction_status=status,
            models=models,
            engineering_valid=None if valid is None else bool(valid),
            uncertainty_known=(sum(e.uncertainty_known for e in used), len(used)),
            evidence=ev,
            warnings=tuple(warnings),
        )


def evaluate_candidate(
    target: TargetMaterialProfile,
    candidate: MaterialCandidate,
    context: EvaluationContext | None = None,
) -> CandidateEvaluation:
    """E(x, x*) for any candidate origin. Without a context the pool is the candidate alone."""
    return (context or EvaluationContext.fit(target, [candidate])).evaluate(candidate)


def evaluate_candidates(
    target: TargetMaterialProfile,
    candidates: Sequence[MaterialCandidate],
    policy: EvidencePolicy | None = None,
) -> tuple[CandidateEvaluation, ...]:
    """Evaluate and rank like the similarity search: feasibility group, pessimistic
    distance, distance, name."""
    context = EvaluationContext.fit(target, candidates, policy)
    evaluated = [context.evaluate(c) for c in candidates]

    def key(e: CandidateEvaluation) -> tuple[int, float, float, str]:
        inf = math.inf
        return (
            GROUP_ORDER[e.feasibility],
            e.pessimistic_distance if e.pessimistic_distance is not None else inf,
            e.distance if e.distance is not None else inf,
            e.name or e.candidate_id,
        )

    evaluated.sort(key=key)
    return tuple(e.model_copy(update={"rank": i + 1}) for i, e in enumerate(evaluated))
