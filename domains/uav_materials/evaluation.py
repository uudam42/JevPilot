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
* requirement semantics                               requirements.RequirementResolver

Every requirement is first *resolved* for the candidate's material system
(:mod:`~domains.uav_materials.requirements`): EngineeringRequirement →
observable property → measurement/prediction → constraint check. For bulk
(isotropic) records the resolution is the identity, so existing results are
unchanged; a unidirectional ply answers only directionally stated
requirements. Each requirement's outcome distinguishes *missing* (resolved,
no value), *ambiguous* (underspecified for this system) and *unsupported*
(no observable), instead of collapsing them into missing data.

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
from domains.uav_materials.profile import PropertyRequirement, TargetMaterialProfile
from domains.uav_materials.requirements import (
    EngineeringTarget,
    ObservableResolution,
    ResolutionStatus,
    TargetBinding,
    material_system_of,
)
from domains.uav_materials.schema import (
    CandidateOrigin,
    EvidenceType,
    MaterialCandidate,
    MaterialRecord,
    Measurement,
)
from domains.uav_materials.search import (
    GROUP_ORDER,
    CheckStatus,
    Contribution,
    Feasibility,
    RankedCandidate,
    RequirementCheck,
    _feasibility,
    check,
    fit_normalization,
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


class RequirementOutcome(FrozenModel):
    """One requirement for one candidate: how it resolved and what the check found."""

    requirement: str  # engineering label, e.g. "tensile_ultimate_strength >= 350 MPa"
    hard: bool
    resolution: ResolutionStatus
    properties: tuple[str, ...]  # resolved property, or the candidate observables if ambiguous
    check: str | None  # hard constraints: satisfied / violated / undetermined
    gap: str | None  # missing / ambiguous / unsupported / insufficient_evidence / None
    reason: str


Target = TargetMaterialProfile | EngineeringTarget


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
    material_system: str = "isotropic_bulk"
    resolutions: tuple[ObservableResolution, ...] = ()
    outcomes: tuple[RequirementOutcome, ...] = ()


class EvaluationContext:
    """One normalisation scale plus one evidence policy, for comparable evaluations."""

    def __init__(
        self,
        binding: TargetBinding,
        extractor: Any,
        pool_label: str,
        policy: EvidencePolicy | None = None,
    ) -> None:
        self.binding = binding
        self.target = binding.profile  # carrier profile (the original one for legacy targets)
        self.extractor = extractor
        self.pool_label = pool_label
        self.policy = policy or EvidencePolicy()

    @classmethod
    def fit(
        cls,
        target: Target,
        reference: Sequence[MaterialCandidate],
        policy: EvidencePolicy | None = None,
    ) -> EvaluationContext:
        binding = TargetBinding(target)
        screened = [(c, _screen(binding, c)) for c in reference]
        extractor, label = fit_normalization(binding.profile, screened, _ranking_resolve(binding))
        return cls(binding, extractor, label, policy)

    def evaluate(self, candidate: MaterialCandidate) -> CandidateEvaluation:
        base = _screen(self.binding, candidate)
        scored = self.extractor.score(candidate.material)
        ev = assess(self.target, candidate, self.policy, binding=self.binding)
        resolutions = self.binding.resolutions(candidate.material)
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
            material_system=material_system_of(candidate.material),
            resolutions=resolutions,
            outcomes=_outcomes(resolutions, ev.requirements, base.checks),
        )


def _check(
    binding: TargetBinding, r: PropertyRequirement, c: MaterialCandidate
) -> RequirementCheck:
    res = binding.resolve(c.material, r)
    if res.property_requirement is not None:
        return check(res.property_requirement, c)
    return RequirementCheck(
        requirement=r,
        status=CheckStatus.UNDETERMINED,
        reason=f"{res.status.value}: {res.requirement.label} for {res.material_system}: "
        f"{res.reason}",
    )


def _screen(binding: TargetBinding, c: MaterialCandidate) -> RankedCandidate:
    """search.screen with resolution in front of every check (identical for bulk records)."""
    checks = tuple(_check(binding, r, c) for r in binding.profile.hard_constraints)
    wanted = {
        res.properties[0] if res.resolved else r.property
        for r in binding.profile.requirements
        for res in (binding.resolve(c.material, r),)
    }
    missing = tuple(sorted(p for p in wanted if c.material.status(p) != "measured"))
    return RankedCandidate(
        candidate_id=c.candidate_id,
        name=c.material.name,
        origin=str(c.origin),
        feasibility=_feasibility(checks),
        checks=checks,
        missing=missing,
        provenance=tuple(m.provenance for k in checks for m in k.used if m.provenance),
    )


def _ranking_resolve(binding: TargetBinding) -> Any:
    def resolve(
        record: MaterialRecord, r: PropertyRequirement
    ) -> tuple[PropertyRequirement | None, str]:
        res = binding.resolve(record, r)
        return res.property_requirement, f"{res.status.value}: {res.reason}"

    return resolve


def _outcomes(
    resolutions: Sequence[ObservableResolution],
    evidence: Sequence[RequirementEvidence],
    checks: Sequence[RequirementCheck],
) -> tuple[RequirementOutcome, ...]:
    hard_checks = iter(checks)
    out = []
    for res, ev in zip(resolutions, evidence, strict=True):
        hard = ev.hard
        status = next(hard_checks).status.value if hard else None
        if not res.resolved:
            gap: str | None = res.status.value
        elif not ev.measurement_available:
            gap = "missing"
        elif not ev.condition_compatible:
            gap = "condition_mismatch"
        elif not ev.sufficient:
            gap = "insufficient_evidence"
        else:
            gap = None
        out.append(
            RequirementOutcome(
                requirement=res.requirement.label,
                hard=hard,
                resolution=res.status,
                properties=res.properties,
                check=status,
                gap=gap,
                reason=res.reason if not res.resolved else ev.detail or res.reason,
            )
        )
    return tuple(out)


def evaluate_candidate(
    target: Target,
    candidate: MaterialCandidate,
    context: EvaluationContext | None = None,
) -> CandidateEvaluation:
    """E(x, x*) for any candidate origin. Without a context the pool is the candidate alone."""
    return (context or EvaluationContext.fit(target, [candidate])).evaluate(candidate)


def evaluate_candidates(
    target: Target,
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
