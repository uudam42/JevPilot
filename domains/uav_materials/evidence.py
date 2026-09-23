"""Evidence sufficiency and gap classification (an analysis layer over search).

Search answers "does the candidate pass?" with FEASIBLE / UNDETERMINED /
INFEASIBLE. This module answers "*why* can we not decide, and what evidence
would decide it?", without treating missing data as poor performance.
Search results and their semantics are unchanged.

Per requirement (hard constraint or soft preference) of a profile, the
existing selection policy (:mod:`~domains.uav_materials.selection`) picks the
measurement that answers it. The selected evidence is then **sufficient** when
all of these hold under an explicit :class:`EvidencePolicy`:

1. a measurement with a value exists                     else ``missing_property``
2. its test conditions are compatible                    else ``condition_mismatch``
3. its identity match is at least ``policy.min_identity`` else ``identity_match_too_weak``
4. its provenance status is in ``policy.accepted_provenance``
                                                         else ``insufficient_provenance``
5. its stated relative uncertainty ≤ ``policy.max_relative_uncertainty``
   (only checked when set; unknown uncertainty fails only when
   ``policy.require_uncertainty``)                       else ``uncertainty_too_high``
6. for hard constraints: the value decides the check (a bound such as
   "≤ x" cannot decide every requirement)                else ``inconclusive_bound``

Classification of a candidate (first rule that applies):

=====================  ===================================================================
``infeasible``         some hard constraint is VIOLATED by sufficient evidence
``evidence_gap``       some hard constraint lacks sufficient evidence (including a
                       violation shown only by insufficient evidence)
``adequate``           every hard constraint is SATISFIED by sufficient evidence, AND an
                       acceptable preferred-property region was supplied, AND the
                       candidate lies in it with sufficient evidence
``unclassified``       hard constraints pass with sufficient evidence, but no acceptable
                       region is defined (or the candidate is outside it / lacks the
                       evidence): ADEQUATE vs DESIGN_GAP cannot be decided yet
=====================  ===================================================================

``design_gap`` exists as a type but is **never assigned** by :func:`classify`.
:func:`design_gap_preconditions` lists which of its preconditions are unmet;
missing evidence always leaves at least one unmet.

Coverage is counted, not weighted: hard coverage = hard constraints with
sufficient evidence / hard constraints; soft coverage likewise over soft
preferences. A weighted soft coverage is reported too, using only the
profile's explicit numerical weights (unweighted preferences are excluded
from it and listed).

Acquisition priority, per gap (lower tier first):

====  ============================================================================
1     hard constraint, ``missing_property``
2     hard constraint, ``condition_mismatch``
3     hard constraint, any other reason (identity, provenance, uncertainty, bound)
4     soft preference with a numerical weight (larger weight first)
5     soft preference without a numerical weight
====  ============================================================================

Across candidates, gaps are grouped by (property, stated conditions) and
counted over *potentially viable* candidates (not ``infeasible``): evidence
for an already-failed candidate cannot make it selectable. Ranking: best tier,
then number of candidates for which the gap is the **only** hard gap
(acquiring it would decide them), then number of affected candidates, then
name. Because one missing property rarely decides a candidate alone, the
report also groups candidates by their complete set of hard gaps ("decision
sets"): acquiring every item of a set would decide those candidates.
Everything is deterministic; no model is involved.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any

from pydantic import Field

from domains.uav_materials.identity import MatchLevel
from domains.uav_materials.profile import Priority, PropertyRequirement, TargetMaterialProfile
from domains.uav_materials.schema import (
    MaterialCandidate,
    Measurement,
    ProvenanceStatus,
    StatisticalBasis,
)
from domains.uav_materials.search import CheckStatus, Feasibility, check, screen, source_label
from domains.uav_materials.selection import NOT_APPLICABLE_SATISFIES, select
from jevpilot import FrozenModel

VERSION = "evidence_assessment/1"


class GapClass(StrEnum):
    INFEASIBLE = "infeasible"  # an adequately supported hard constraint is violated
    EVIDENCE_GAP = "evidence_gap"  # hard constraints cannot be decided on current evidence
    DESIGN_GAP = "design_gap"  # reserved: never assigned automatically (see module doc)
    ADEQUATE = "adequate"  # hard constraints pass and inside the acceptable region
    UNCLASSIFIED = "unclassified"  # hard constraints pass; no acceptable region to judge by


class GapReason(StrEnum):
    MISSING_PROPERTY = "missing_property"
    CONDITION_MISMATCH = "condition_mismatch"
    IDENTITY_MATCH_TOO_WEAK = "identity_match_too_weak"
    INSUFFICIENT_PROVENANCE = "insufficient_provenance"
    UNCERTAINTY_TOO_HIGH = "uncertainty_too_high"
    INCONCLUSIVE_BOUND = "inconclusive_bound"


IDENTITY_ORDER = {"none": 0, MatchLevel.ALLOY.value: 1, MatchLevel.EXACT.value: 2, "native": 3}


class EvidencePolicy(FrozenModel):
    """What counts as sufficient evidence. Defaults are documented choices, not tuned."""

    # Alloy-level links (a source naming the alloy without the temper) are accepted by
    # default and reported as such; "exact" demands a temper-level or native match.
    min_identity: str = MatchLevel.ALLOY.value
    accepted_provenance: tuple[ProvenanceStatus, ...] = (ProvenanceStatus.SOURCED,)
    require_uncertainty: bool = False  # no current source states measurement uncertainty
    max_relative_uncertainty: float | None = None
    policy_note: str = (
        "defaults: sourced provenance required; alloy-level identity accepted; "
        "uncertainty not required (A/B design allowables count as quantified)"
    )


class RequirementEvidence(FrozenModel):
    """Evidence for one requirement of the profile, with every sufficiency factor exposed."""

    property: str
    operator: str
    hard: bool
    conditions: dict[str, Any] = Field(default_factory=dict)  # the requirement's stated ones
    weight: float | None = None
    measurement_available: bool  # any measurement with a value for this property
    condition_compatible: bool  # one of them has compatible conditions
    provenance: str | None = None  # provenance status of the selected measurement
    source: str | None = None
    identity_match: str | None = None  # native / exact / alloy
    statistical_basis: str | None = None
    uncertainty_known: bool = False
    check: str | None = None  # hard constraints: satisfied / violated / undetermined
    value: str | None = None
    sufficient: bool
    reasons: tuple[GapReason, ...] = ()
    detail: str = ""

    @builtins.property  # the field named 'property' shadows the builtin here
    def label(self) -> str:
        return requirement_label(self.property, self.conditions)

    @builtins.property  # the field named 'property' shadows the builtin here
    def tier(self) -> int:
        """Acquisition priority tier of this gap (see module doc); 0 when sufficient."""
        if self.sufficient:
            return 0
        if self.hard:
            if GapReason.MISSING_PROPERTY in self.reasons:
                return 1
            return 2 if GapReason.CONDITION_MISMATCH in self.reasons else 3
        return 4 if self.weight is not None else 5


class EvidenceAssessment(FrozenModel):
    candidate_id: str
    name: str
    feasibility: Feasibility  # unchanged search semantics
    classification: GapClass
    classification_reason: str
    hard_coverage: tuple[int, int]  # (sufficient, total)
    soft_coverage: tuple[int, int]
    soft_weighted_coverage: float | None  # over numerically weighted preferences only
    requirements: tuple[RequirementEvidence, ...]
    known_properties: tuple[str, ...]
    missing_critical: tuple[str, ...]  # hard-constraint gaps, as labels
    violated: tuple[str, ...]  # hard constraints violated by sufficient evidence
    incompatible_evidence: tuple[str, ...]  # measurements present but not usable, and why
    provenance_quality: dict[str, Any]
    next_evidence: tuple[str, ...]  # gaps in acquisition-priority order

    @property
    def hard_coverage_ratio(self) -> float:
        known, total = self.hard_coverage
        return known / total if total else 1.0


class Bottleneck(FrozenModel):
    evidence: str  # property [conditions]
    tier: int
    reasons: tuple[str, ...]
    candidates: tuple[str, ...]  # potentially viable candidates it blocks
    sole_blocker_for: tuple[str, ...]  # ... for which it is the only hard gap


class DecisionSet(FrozenModel):
    evidence: tuple[str, ...]  # all hard gaps these candidates share, and only those
    candidates: tuple[str, ...]


class Failure(FrozenModel):
    requirement: str
    candidates: tuple[str, ...]


class EvidenceReport(FrozenModel):
    profile_id: str
    version: str = VERSION
    policy: EvidencePolicy
    assessments: tuple[EvidenceAssessment, ...]
    counts: dict[str, int]
    engineering_failures: tuple[Failure, ...]  # what actually fails, by hard constraint
    evidence_bottlenecks: tuple[Bottleneck, ...]  # what is unknown, by priority
    decision_sets: tuple[DecisionSet, ...]  # evidence that would decide candidates, jointly
    acceptable_region_defined: bool


# -- helpers ---------------------------------------------------------------------------------


def requirement_label(prop: str, conditions: dict[str, Any]) -> str:
    if not conditions:
        return prop
    parts = []
    for k, v in sorted(conditions.items()):
        if isinstance(v, dict) and "value" in v:
            v = f"{v['value']:g} {v.get('unit', '')}".strip()
        parts.append(f"{k}={v}")
    return f"{prop} [{', '.join(parts)}]"


def _conditions(r: PropertyRequirement) -> dict[str, Any]:
    return r.conditions.model_dump(mode="json", exclude_none=True, exclude_defaults=True)


def identity_level(m: Measurement) -> str:
    """native: the record's own source; otherwise the recorded identity match level."""
    level = m.provenance.metadata.get("identity_match") if m.provenance else None
    return str(level) if level else "native"


def _uncertainty(m: Measurement) -> tuple[bool, float | None]:
    """(known, relative size). A/B allowables are one-sided tolerance bounds: quantified."""
    u = m.uncertainty
    rel: float | None = None
    if u is not None and m.value:
        if u.dispersion is not None:
            rel = u.dispersion / abs(m.value)
        elif u.interval is not None:
            rel = (u.interval[1] - u.interval[0]) / 2 / abs(m.value)
    known = rel is not None or m.statistical_basis in (StatisticalBasis.A, StatisticalBasis.B)
    return known, rel


def _assess_requirement(
    c: MaterialCandidate, r: PropertyRequirement, policy: EvidencePolicy
) -> RequirementEvidence:
    hard = r.priority is Priority.HARD
    sel = select(c.material, r)
    found = c.material.measurements(r.property)
    available = any(not m.is_missing for m in found)
    base: dict[str, Any] = {
        "property": r.property,
        "operator": str(r.operator),
        "hard": hard,
        "conditions": _conditions(r),
        "weight": r.weight,
        "measurement_available": available,
    }
    status = check(r, c).status if hard else None
    if sel.chosen is None:
        if hard and status is CheckStatus.SATISFIED:  # not-applicable policy (explicit)
            assert (r.property, r.operator) in NOT_APPLICABLE_SATISFIES
            return RequirementEvidence(
                **base,
                condition_compatible=True,
                check=status.value,
                sufficient=True,
                detail="recorded not applicable; satisfies an upper bound by policy",
            )
        reason = GapReason.CONDITION_MISMATCH if available else GapReason.MISSING_PROPERTY
        excluded = "; ".join(f"{source_label(m) or 'unsourced'}: {w}" for m, w in sel.excluded)
        return RequirementEvidence(
            **base,
            condition_compatible=False,
            check=status.value if status else None,
            sufficient=False,
            reasons=(reason,),
            detail=excluded or sel.reason,
        )
    m = sel.chosen
    identity = identity_level(m)
    known, rel = _uncertainty(m)
    reasons: list[GapReason] = []
    if IDENTITY_ORDER.get(identity, 0) < IDENTITY_ORDER[policy.min_identity]:
        reasons.append(GapReason.IDENTITY_MATCH_TOO_WEAK)
    if m.provenance_status not in policy.accepted_provenance:
        reasons.append(GapReason.INSUFFICIENT_PROVENANCE)
    if (policy.require_uncertainty and not known) or (
        policy.max_relative_uncertainty is not None
        and rel is not None
        and rel > policy.max_relative_uncertainty
    ):
        reasons.append(GapReason.UNCERTAINTY_TOO_HIGH)
    if hard and status is CheckStatus.UNDETERMINED:
        reasons.append(GapReason.INCONCLUSIVE_BOUND)
    unit = r.unit or m.unit
    shown = m.value_in(unit) if unit else m.value
    return RequirementEvidence(
        **base,
        condition_compatible=True,
        provenance=m.provenance_status.value,
        source=source_label(m),
        identity_match=identity,
        statistical_basis=m.statistical_basis.value if m.statistical_basis else None,
        uncertainty_known=known,
        check=status.value if status else None,
        value=f"{m.qualifier.value if m.qualifier.value != '=' else ''}{shown:.4g} {unit}".strip(),
        sufficient=not reasons,
        reasons=tuple(reasons),
        detail=sel.reason,
    )


# -- classification ----------------------------------------------------------------------


def classify(
    requirements: Sequence[RequirementEvidence],
    region: Sequence[RequirementEvidence] | None = None,
) -> tuple[GapClass, str]:
    hard = [e for e in requirements if e.hard]
    failed = [e for e in hard if e.sufficient and e.check == CheckStatus.VIOLATED.value]
    if failed:
        return GapClass.INFEASIBLE, "violated with sufficient evidence: " + ", ".join(
            e.label for e in failed
        )
    gaps = [e for e in hard if not e.sufficient]
    if gaps:
        return GapClass.EVIDENCE_GAP, "hard constraints without sufficient evidence: " + ", ".join(
            f"{e.label} ({', '.join(r.value for r in e.reasons)})" for e in gaps
        )
    if region is None:
        return (
            GapClass.UNCLASSIFIED,
            "all hard constraints pass; no acceptable preferred-property region is defined",
        )
    outside = [e for e in region if not (e.sufficient and e.check == CheckStatus.SATISFIED.value)]
    if outside:
        return GapClass.UNCLASSIFIED, "hard constraints pass; not shown inside the acceptable "
        "region: " + ", ".join(e.label for e in outside)
    return GapClass.ADEQUATE, "all hard constraints pass and inside the acceptable region"


def design_gap_preconditions(a: EvidenceAssessment) -> tuple[str, ...]:
    """Unmet preconditions for DESIGN_GAP. Empty tuple = a human may consider it.

    DESIGN_GAP needs: sufficient evidence for every requirement, hard feasibility
    understood (not infeasible, no hard gap), and a defined distance criterion from
    the preferred target. The last does not exist yet, so this is never empty today.
    """
    unmet: list[str] = []
    if any(not e.sufficient for e in a.requirements):
        unmet.append("evidence is not sufficient for every requirement")
    if a.classification in (GapClass.INFEASIBLE, GapClass.EVIDENCE_GAP):
        unmet.append("hard feasibility is not established")
    unmet.append("no criterion for 'meaningfully far from the preferred target' is defined")
    return tuple(unmet)


# -- per candidate -------------------------------------------------------------------------


def assess(
    profile: TargetMaterialProfile,
    candidate: MaterialCandidate,
    policy: EvidencePolicy | None = None,
    acceptable_region: Sequence[PropertyRequirement] | None = None,
) -> EvidenceAssessment:
    policy = policy or EvidencePolicy()
    reqs = tuple(_assess_requirement(candidate, r, policy) for r in profile.requirements)
    region = (
        None
        if acceptable_region is None
        else tuple(
            _assess_requirement(candidate, r.model_copy(update={"priority": Priority.HARD}), policy)
            for r in acceptable_region
        )
    )
    cls, why = classify(reqs, region)
    hard = [e for e in reqs if e.hard]
    soft = [e for e in reqs if not e.hard]
    weighted = [e for e in soft if e.weight is not None]
    total_w = sum(e.weight or 0.0 for e in weighted)
    gaps = sorted(
        (e for e in reqs if not e.sufficient),
        key=lambda e: (e.tier, -(e.weight or 0.0), reqs.index(e)),
    )
    used = [e for e in reqs if e.provenance is not None]
    material = candidate.material
    incompatible = tuple(
        f"{e.label}: {e.detail}"
        for e in reqs
        if e.measurement_available and GapReason.CONDITION_MISMATCH in e.reasons
    )
    return EvidenceAssessment(
        candidate_id=candidate.candidate_id,
        name=material.name,
        feasibility=screen(profile, candidate).feasibility,
        classification=cls,
        classification_reason=why,
        hard_coverage=(sum(e.sufficient for e in hard), len(hard)),
        soft_coverage=(sum(e.sufficient for e in soft), len(soft)),
        soft_weighted_coverage=(
            sum(e.weight or 0.0 for e in weighted if e.sufficient) / total_w if total_w else None
        ),
        requirements=reqs,
        known_properties=tuple(dict.fromkeys(e.property for e in reqs if e.sufficient)),
        missing_critical=tuple(e.label for e in hard if not e.sufficient),
        violated=tuple(
            e.label for e in hard if e.sufficient and e.check == CheckStatus.VIOLATED.value
        ),
        incompatible_evidence=incompatible,
        provenance_quality={
            "selected_measurements": len(used),
            "sourced": sum(e.provenance == ProvenanceStatus.SOURCED.value for e in used),
            "identity": _count(e.identity_match for e in used),
            "statistical_basis": _count(e.statistical_basis or "unstated" for e in used),
            "uncertainty_known": sum(e.uncertainty_known for e in used),
        },
        next_evidence=tuple(dict.fromkeys(e.label for e in gaps)),
    )


def _count(items: Iterable[str | None]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        key = i or "unstated"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


# -- dataset level -------------------------------------------------------------------------


def evidence_report(
    profile: TargetMaterialProfile,
    candidates: Sequence[MaterialCandidate],
    policy: EvidencePolicy | None = None,
    acceptable_region: Sequence[PropertyRequirement] | None = None,
) -> EvidenceReport:
    policy = policy or EvidencePolicy()
    assessed = sorted(
        (assess(profile, c, policy, acceptable_region) for c in candidates),
        key=lambda a: (list(GapClass).index(a.classification), a.name, a.candidate_id),
    )
    counts = {g.value: 0 for g in GapClass}
    for a in assessed:
        counts[a.classification.value] += 1

    failures: dict[str, list[str]] = {}
    for a in assessed:
        for label in a.violated:
            failures.setdefault(label, []).append(a.candidate_id)

    viable = [a for a in assessed if a.classification is not GapClass.INFEASIBLE]
    groups: dict[str, dict[str, Any]] = {}
    for a in viable:
        hard_gaps = [e for e in a.requirements if e.hard and not e.sufficient]
        for e in a.requirements:
            if e.sufficient:
                continue
            g = groups.setdefault(
                e.label, {"tier": e.tier, "reasons": set(), "cands": [], "sole": []}
            )
            g["tier"] = min(g["tier"], e.tier)
            g["reasons"].update(r.value for r in e.reasons)
            if a.candidate_id not in g["cands"]:
                g["cands"].append(a.candidate_id)
            if e.hard and len(hard_gaps) == 1:
                g["sole"].append(a.candidate_id)
    bottlenecks = sorted(
        (
            Bottleneck(
                evidence=label,
                tier=g["tier"],
                reasons=tuple(sorted(g["reasons"])),
                candidates=tuple(g["cands"]),
                sole_blocker_for=tuple(g["sole"]),
            )
            for label, g in groups.items()
        ),
        key=lambda b: (b.tier, -len(b.sole_blocker_for), -len(b.candidates), b.evidence),
    )
    sets: dict[tuple[str, ...], list[str]] = {}
    for a in viable:
        if a.missing_critical:
            sets.setdefault(tuple(sorted(a.missing_critical)), []).append(a.candidate_id)
    return EvidenceReport(
        profile_id=profile.profile_id,
        policy=policy,
        assessments=tuple(assessed),
        counts={"considered": len(assessed), **counts},
        engineering_failures=tuple(
            Failure(requirement=k, candidates=tuple(v))
            for k, v in sorted(failures.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ),
        evidence_bottlenecks=tuple(bottlenecks),
        decision_sets=tuple(
            DecisionSet(evidence=k, candidates=tuple(v))
            for k, v in sorted(sets.items(), key=lambda kv: (-len(kv[1]), len(kv[0]), kv[0]))
        ),
        acceptable_region_defined=acceptable_region is not None,
    )


TIER_NAMES = {
    1: "hard constraint, measurement missing",
    2: "hard constraint, condition mismatch",
    3: "hard constraint, evidence too weak",
    4: "weighted soft preference missing",
    5: "unweighted soft preference missing",
}


def format_report(report: EvidenceReport, detail_ids: Sequence[str] = ()) -> str:
    """Plain-text rendering (deterministic)."""
    c = report.counts
    lines = [
        f"evidence report for {report.profile_id} ({report.version})",
        f"policy: {report.policy.policy_note}",
        f"{c['considered']} materials: {c['infeasible']} engineering failures (infeasible), "
        f"{c['evidence_gap']} evidence gaps, {c['adequate']} adequate, "
        f"{c['unclassified']} unclassified (hard pass, no acceptable region), "
        f"{c['design_gap']} design gaps (never assigned automatically)",
        "",
        "What actually fails (hard constraint violated by sufficient evidence):",
    ]
    for f in report.engineering_failures:
        lines.append(f"  {f.requirement:28s} {len(f.candidates):3d} candidates")
    lines += ["", "What is unknown (gaps of potentially viable candidates, by priority):"]
    for i, b in enumerate(report.evidence_bottlenecks, 1):
        lines.append(
            f"  {i}. {b.evidence}: blocks {len(b.candidates)}, only hard gap for "
            f"{len(b.sole_blocker_for)}  [{TIER_NAMES[b.tier]}; {', '.join(b.reasons)}]"
        )
    lines += ["", "Evidence that would jointly decide candidates (all hard gaps of each group):"]
    for d in report.decision_sets:
        lines.append(f"  {len(d.candidates):3d} candidates need: {' + '.join(d.evidence)}")
    by_id = {a.candidate_id: a for a in report.assessments}
    for cid in detail_ids:
        a = by_id[cid]
        k, n = a.hard_coverage
        sk, sn = a.soft_coverage
        lines += [
            "",
            f"{a.name}  ({a.candidate_id})",
            f"  feasibility: {a.feasibility.value}   classification: {a.classification.value}",
            f"  hard evidence coverage: {k}/{n}   soft evidence coverage: {sk}/{sn}"
            + (
                f" (weighted {a.soft_weighted_coverage:.0%})"
                if a.soft_weighted_coverage is not None
                else ""
            ),
            f"  known: {', '.join(a.known_properties) or '-'}",
            f"  missing critical: {', '.join(a.missing_critical) or '-'}",
            f"  incompatible evidence: {'; '.join(a.incompatible_evidence) or '-'}",
            f"  provenance: {a.provenance_quality}",
            f"  next evidence: {', '.join(a.next_evidence) or '-'}",
        ]
    return "\n".join(lines)


__all__ = [
    "Bottleneck",
    "DecisionSet",
    "EvidenceAssessment",
    "EvidencePolicy",
    "EvidenceReport",
    "GapClass",
    "GapReason",
    "RequirementEvidence",
    "assess",
    "classify",
    "design_gap_preconditions",
    "evidence_report",
    "format_report",
]
