"""Existing material or design? An explicit, configurable decision policy.

The decision is made by rules on already-computed evaluations, never by an
LLM. For each existing candidate (in ranking order) the policy checks:

1. every hard constraint is satisfied (``require_all_hard_constraints``);
2. the hard constraints are backed by sufficient evidence
   (``minimum_hard_evidence_coverage``; an EVIDENCE_GAP is never a pass);
3. the supported preference distance is small enough
   (pessimistic distance ≤ ``maximum_preference_distance``);
4. enough preference weight is covered by values (``minimum_soft_coverage``).

Outcomes:

``use_existing``          the best-ranked candidate that passes all checks
``design``                none passes, the user allows a design, and the composite
                          design space can address at least one hard requirement
``needs_information``     fewer hard requirements than the policy needs to judge
``no_suitable_existing``  none passes and no design is possible or requested

Candidates that fail only for lack of evidence are reported as evidence gaps,
not as material failures.

The default values are a DEMONSTRATION POLICY, not a UAV design standard.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from domains.uav_materials.evaluation import CandidateEvaluation
from domains.uav_materials.evidence import GapClass
from domains.uav_materials.profile import Priority
from domains.uav_materials.requirements import EngineeringTarget, RequirementResolver
from domains.uav_materials.search import Feasibility
from jevpilot import FrozenModel

DESIGN_SYSTEM = "continuous_fiber_laminate"


class DecisionOutcome(StrEnum):
    USE_EXISTING = "use_existing"
    DESIGN = "design"
    NEEDS_INFORMATION = "needs_information"
    NO_SUITABLE_EXISTING = "no_suitable_existing"


class ExistingMaterialAcceptancePolicy(FrozenModel):
    policy_id: str = "demo-acceptance/1"
    label: str = "DEMONSTRATION POLICY - not a universal UAV design standard"
    require_all_hard_constraints: bool = True
    minimum_hard_constraints: int = 1
    minimum_hard_evidence_coverage: float = (
        1.0  # share of hard constraints with sufficient evidence
    )
    maximum_preference_distance: float | None = 0.5  # pessimistic distance; None = not checked
    minimum_soft_coverage: float | None = 0.5  # weighted preference coverage; None = not checked
    rationale: str = (
        "illustrative thresholds chosen for the demonstration; replace them with the "
        "acceptance rules of the actual application"
    )


class CandidateVerdict(FrozenModel):
    candidate_id: str
    name: str
    rank: int | None
    accepted: bool
    feasibility: str
    evidence_classification: str
    hard_coverage: tuple[int, int]
    pessimistic_distance: float | None
    coverage: float | None
    reasons: tuple[str, ...]  # why it was not accepted (empty when accepted)


class MaterialDecision(FrozenModel):
    outcome: DecisionOutcome
    selected_candidate_id: str | None = None
    policy: ExistingMaterialAcceptancePolicy
    reasons: tuple[str, ...]
    counts: dict[str, int]  # accepted / engineering_failure / evidence_gap / rejected_by_preference
    verdicts: tuple[CandidateVerdict, ...]  # every candidate, ranking order
    design_supported: tuple[str, ...] = ()  # hard requirements the design space can address
    design_unsupported: tuple[str, ...] = ()  # ... and those it cannot (with reasons)


def verdict(e: CandidateEvaluation, policy: ExistingMaterialAcceptancePolicy) -> CandidateVerdict:
    reasons: list[str] = []
    ev = e.evidence
    if policy.require_all_hard_constraints and e.feasibility is not Feasibility.FEASIBLE:
        if ev.violated:
            reasons.append("violates " + ", ".join(ev.violated))
        if ev.missing_critical:
            gaps = ", ".join(dict.fromkeys(ev.missing_critical))
            reasons.append(f"cannot be judged (evidence gap): {gaps}")
    known, total = ev.hard_coverage
    if total and known / total < policy.minimum_hard_evidence_coverage:
        reasons.append(
            f"hard-constraint evidence {known}/{total} below the policy's "
            f"{policy.minimum_hard_evidence_coverage:.0%}"
        )
    if (
        policy.maximum_preference_distance is not None
        and e.pessimistic_distance is not None
        and e.pessimistic_distance > policy.maximum_preference_distance
    ):
        reasons.append(
            f"preference distance {e.pessimistic_distance:.3f} above the policy's "
            f"{policy.maximum_preference_distance}"
        )
    if (
        policy.minimum_soft_coverage is not None
        and e.coverage is not None
        and e.coverage < policy.minimum_soft_coverage
    ):
        reasons.append(
            f"preference coverage {e.coverage:.0%} below the policy's "
            f"{policy.minimum_soft_coverage:.0%}"
        )
    return CandidateVerdict(
        candidate_id=e.candidate_id,
        name=e.name,
        rank=e.rank,
        accepted=not reasons,
        feasibility=e.feasibility.value,
        evidence_classification=ev.classification.value,
        hard_coverage=ev.hard_coverage,
        pessimistic_distance=e.pessimistic_distance,
        coverage=e.coverage,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def design_space_support(
    target: EngineeringTarget, system: str = DESIGN_SYSTEM
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(hard requirements the design space resolves, those it cannot, with reasons)."""
    resolver = RequirementResolver()
    supported, unsupported = [], []
    for r in target.requirements:
        if r.priority is not Priority.HARD:
            continue
        res = resolver.resolve(r, system)
        if res.resolved:
            supported.append(r.label)
        else:
            unsupported.append(f"{r.label} ({res.status.value}: {res.reason})")
    return tuple(supported), tuple(unsupported)


def decide(
    evaluations: Sequence[CandidateEvaluation],
    target: EngineeringTarget,
    *,
    design_allowed: bool,
    policy: ExistingMaterialAcceptancePolicy | None = None,
) -> MaterialDecision:
    policy = policy or ExistingMaterialAcceptancePolicy()
    ranked = sorted(evaluations, key=lambda e: (e.rank is None, e.rank or 0))
    verdicts = tuple(verdict(e, policy) for e in ranked)
    by_class = {e.candidate_id: e.evidence.classification for e in ranked}
    accepted = [v for v in verdicts if v.accepted]
    counts = {
        "candidates": len(verdicts),
        "accepted": len(accepted),
        "engineering_failure": sum(1 for c in by_class.values() if c is GapClass.INFEASIBLE),
        "evidence_gap": sum(1 for c in by_class.values() if c is GapClass.EVIDENCE_GAP),
        "rejected_by_preference": sum(
            1
            for v in verdicts
            if not v.accepted
            and by_class[v.candidate_id] not in (GapClass.INFEASIBLE, GapClass.EVIDENCE_GAP)
        ),
    }
    supported, unsupported = design_space_support(target)
    hard = [r for r in target.requirements if r.priority is Priority.HARD]
    summary = (
        f"{counts['candidates']} existing materials evaluated: {counts['accepted']} meet the "
        f"policy, {counts['engineering_failure']} violate a hard constraint on sufficient "
        f"evidence, {counts['evidence_gap']} cannot be judged for lack of evidence (not "
        f"counted as failures), {counts['rejected_by_preference']} pass the hard constraints "
        "but miss the preference thresholds"
    )

    def result(
        outcome: DecisionOutcome, *reasons: str, selected: str | None = None
    ) -> MaterialDecision:
        return MaterialDecision(
            outcome=outcome,
            selected_candidate_id=selected,
            policy=policy,
            reasons=tuple(reasons),
            counts=counts,
            verdicts=verdicts,
            design_supported=supported,
            design_unsupported=unsupported,
        )

    if len(hard) < policy.minimum_hard_constraints:
        return result(
            DecisionOutcome.NEEDS_INFORMATION,
            f"the request states {len(hard)} hard requirement(s); the policy needs at least "
            f"{policy.minimum_hard_constraints} to judge suitability",
            summary,
        )
    if accepted:
        best = accepted[0]
        return result(
            DecisionOutcome.USE_EXISTING,
            f"'{best.name}' (rank {best.rank}) meets every check of {policy.policy_id}",
            summary,
            selected=best.candidate_id,
        )
    gaps = [v.name for v in verdicts if by_class[v.candidate_id] is GapClass.EVIDENCE_GAP]
    evidence_note: tuple[str, ...] = ()
    if gaps:
        shown = "; ".join(gaps[:3]) + ("; ..." if len(gaps) > 3 else "")
        evidence_note = (
            f"{len(gaps)} existing material(s) cannot be judged and might qualify if the "
            f"missing evidence were obtained (evidence gaps, not failures): {shown}",
        )
    if design_allowed and supported:
        return result(
            DecisionOutcome.DESIGN,
            "no existing material meets the policy",
            summary,
            "the user allows a design and the composite design space can address "
            f"{len(supported)} of {len(hard)} hard requirement(s)",
            *evidence_note,
        )
    why = (
        "the user did not ask for a design"
        if not design_allowed
        else "the composite design space cannot address any hard requirement"
    )
    return result(
        DecisionOutcome.NO_SUITABLE_EXISTING,
        "no existing material meets the policy",
        summary,
        why,
        *evidence_note,
    )
