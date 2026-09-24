"""Existing-material search: hard-constraint screening and transparent similarity ranking.

    TargetMaterialProfile + candidates
      → per requirement: compatibility (compatibility.py) → selection (selection.py)
      → hard constraints: SATISFIED / VIOLATED / UNDETERMINED per constraint
      → feasibility: FEASIBLE / UNDETERMINED / INFEASIBLE (never collapsed)
      → soft preferences: direction-aware penalties (ranking.py) → distance + coverage
      → deterministic ranking: feasibility group, pessimistic distance, distance, name

Every result carries the measurement used for each check and preference,
with its source, plus the reasons for anything undetermined. No step imputes
a missing value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field

from domains.uav_materials.profile import Operator, PropertyRequirement, TargetMaterialProfile
from domains.uav_materials.schema import MaterialCandidate, Measurement, Qualifier
from domains.uav_materials.selection import NOT_APPLICABLE_SATISFIES, select
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
    value: float | None = None  # selected value, in the requirement's unit
    used: tuple[Measurement, ...] = ()  # the selected measurement (0 or 1)
    alternatives: int = 0  # other usable measurements not selected
    excluded: tuple[str, ...] = ()  # why other measurements could not be used


class Contribution(FrozenModel):
    """One soft preference's part of a candidate's distance."""

    property: str
    operator: str
    weight: float
    value: float | None  # canonical unit; None = missing
    unit: str
    penalty: float | None  # dimensionless; None = missing
    share: float | None  # weight * penalty^2 / total weight
    source: str | None = None  # e.g. "MIL-HDBK-5J Table 3.7.6.0(b1) p.673 col 5"
    note: str = ""


class RankedCandidate(FrozenModel):
    candidate_id: str
    name: str = ""
    origin: str
    feasibility: Feasibility
    checks: tuple[RequirementCheck, ...]
    missing: tuple[str, ...]  # requirement properties with no usable measurement
    rank: int | None = None
    distance: float | None = None  # weighted RMS penalty over covered preferences
    pessimistic_distance: float | None = None  # missing preferences at worst observed penalty
    score_coverage: float | None = None  # share of preference weight with a measurement
    matched_properties: tuple[str, ...] = ()
    contributions: tuple[Contribution, ...] = ()
    provenance: tuple[Provenance, ...] = ()

    @property
    def violations(self) -> tuple[RequirementCheck, ...]:
        return tuple(c for c in self.checks if c.status is CheckStatus.VIOLATED)

    @property
    def undetermined(self) -> tuple[RequirementCheck, ...]:
        return tuple(c for c in self.checks if c.status is CheckStatus.UNDETERMINED)


CandidateAssessment = RankedCandidate  # iteration-1 name


class MaterialSearchResult(FrozenModel):
    profile_id: str
    method: str
    assessments: tuple[RankedCandidate, ...]
    counts: dict[str, int] = Field(default_factory=dict)
    normalization: dict[str, Any] = Field(default_factory=dict)
    unscored_preferences: tuple[str, ...] = ()
    notes: str = ""


class MaterialSearch(Protocol):
    name: str

    def search(
        self, profile: TargetMaterialProfile, candidates: Sequence[MaterialCandidate]
    ) -> MaterialSearchResult: ...


def source_label(m: Measurement) -> str | None:
    if m.provenance is None or not m.provenance.sources:
        return None
    src = m.provenance.sources[0]
    meta = src.metadata
    parts = [src.identifier]
    for key, fmt in (
        ("table", "Table {}"),
        ("page", "p.{}"),
        ("column", "col {}"),
        ("line", "line {}"),
    ):
        if meta.get(key) is not None:
            parts.append(fmt.format(meta[key]))
    return " ".join(parts)


def check(req: PropertyRequirement, candidate: MaterialCandidate) -> RequirementCheck:
    sel = select(candidate.material, req)
    excluded = tuple(f"{source_label(m) or 'unsourced'}: {why}" for m, why in sel.excluded)
    if sel.chosen is None:
        if sel.not_applicable and (req.property, req.operator) in NOT_APPLICABLE_SATISFIES:
            return RequirementCheck(
                requirement=req,
                status=CheckStatus.SATISFIED,
                reason="recorded as not applicable (no uptake mechanism); "
                "satisfies an upper bound by policy",
                excluded=excluded,
            )
        return RequirementCheck(
            requirement=req, status=CheckStatus.UNDETERMINED, reason=sel.reason, excluded=excluded
        )
    m = sel.chosen
    assert req.unit is not None
    v = m.value_in(req.unit)
    assert v is not None
    lower, upper = req.bounds()
    ok_low = lower is None or v >= lower
    ok_high = upper is None or v <= upper
    shown = f"{v:.4g} {req.unit}"
    if m.qualifier is Qualifier.AT_MOST:
        status = CheckStatus.SATISFIED if ok_high else CheckStatus.UNDETERMINED
        shown = f"≤ {shown} (upper bound)"
    elif m.qualifier is Qualifier.AT_LEAST:
        status = CheckStatus.SATISFIED if ok_low else CheckStatus.UNDETERMINED
        shown = f"≥ {shown} (lower bound)"
    else:
        status = CheckStatus.SATISFIED if ok_low and ok_high else CheckStatus.VIOLATED
    target = {
        Operator.LE: f"≤ {req.value}",
        Operator.GE: f"≥ {req.value}",
        Operator.BETWEEN: f"{req.value}–{req.upper}",
    }.get(req.operator, str(req.value))
    return RequirementCheck(
        requirement=req,
        status=status,
        value=v,
        used=(m,),
        alternatives=len(sel.alternatives),
        excluded=excluded,
        reason=f"{shown} vs {target} {req.unit} ({sel.reason}; {source_label(m) or 'unsourced'})",
    )


def _feasibility(checks: Sequence[RequirementCheck]) -> Feasibility:
    statuses = {k.status for k in checks}
    if CheckStatus.VIOLATED in statuses:
        return Feasibility.INFEASIBLE
    if CheckStatus.UNDETERMINED in statuses:
        return Feasibility.UNDETERMINED
    return Feasibility.FEASIBLE


def screen(profile: TargetMaterialProfile, c: MaterialCandidate) -> RankedCandidate:
    checks = tuple(check(r, c) for r in profile.hard_constraints)
    wanted = {r.property for r in profile.requirements}
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


GROUP_ORDER = {Feasibility.FEASIBLE: 0, Feasibility.UNDETERMINED: 1, Feasibility.INFEASIBLE: 2}


class HardConstraintScreen:
    """Hard constraints only (no ranking). Order: feasible, undetermined, infeasible."""

    name = "hard_constraint_screen/2"

    def search(
        self, profile: TargetMaterialProfile, candidates: Sequence[MaterialCandidate]
    ) -> MaterialSearchResult:
        assessed = sorted(
            (screen(profile, c) for c in candidates), key=lambda a: GROUP_ORDER[a.feasibility]
        )
        return MaterialSearchResult(
            profile_id=profile.profile_id,
            method=self.name,
            assessments=tuple(assessed),
            counts=_counts(assessed),
            notes="hard constraints only; soft preferences not scored (distance=None)",
        )


class SimilarityRanker:
    """Screening plus target-relative, direction-aware weighted distance (see ranking.py)."""

    name = "screen+target_relative_rank/1"

    def search(
        self, profile: TargetMaterialProfile, candidates: Sequence[MaterialCandidate]
    ) -> MaterialSearchResult:
        screened = [(c, screen(profile, c)) for c in candidates]
        extractor, pool_label = fit_normalization(profile, screened)
        ranked: list[RankedCandidate] = []
        for c, base in screened:
            scored = extractor.score(c.material)
            ranked.append(
                base.model_copy(
                    update={
                        "distance": scored.distance,
                        "pessimistic_distance": scored.pessimistic_distance,
                        "score_coverage": scored.coverage,
                        "matched_properties": scored.matched,
                        "contributions": scored.contributions,
                        "provenance": base.provenance + scored.provenance,
                    }
                )
            )

        def key(a: RankedCandidate) -> tuple[int, float, float, str]:
            inf = math.inf
            return (
                GROUP_ORDER[a.feasibility],
                a.pessimistic_distance if a.pessimistic_distance is not None else inf,
                a.distance if a.distance is not None else inf,
                a.name or a.candidate_id,
            )

        ranked.sort(key=key)
        ranked = [a.model_copy(update={"rank": i + 1}) for i, a in enumerate(ranked)]
        return MaterialSearchResult(
            profile_id=profile.profile_id,
            method=self.name,
            assessments=tuple(ranked),
            counts=_counts(ranked),
            normalization={**extractor.parameters(), "normalization_pool": pool_label},
            unscored_preferences=extractor.unscored,
            notes=(
                "ordering: feasibility group, then pessimistic distance (missing preferences "
                "count at the worst observed penalty), then distance, then name; distances "
                "are only comparable within one search result"
            ),
        )


def fit_normalization(
    profile: TargetMaterialProfile,
    screened: Sequence[tuple[MaterialCandidate, RankedCandidate]],
    resolve: Any = None,
) -> tuple[Any, str]:
    """Fit the ranking scale on the normalisation pool; returns (extractor, pool label).

    Infeasible candidates must not set the scale for "how far from the best viable":
    the pool is the non-infeasible candidates, or all of them if fewer than two remain.
    A preference with no value in the pool takes its scale from all candidates.
    """
    from domains.uav_materials.ranking import TargetRelativeExtractor

    viable = [c.material for c, a in screened if a.feasibility is not Feasibility.INFEASIBLE]
    pool = viable if len(viable) >= 2 else [c.material for c, _ in screened]
    label = (
        f"non-infeasible candidates ({len(pool)})"
        if pool is viable
        else f"all candidates ({len(pool)}); fewer than 2 were non-infeasible"
    )
    everyone = [c.material for c, _ in screened]
    return TargetRelativeExtractor.fit(profile, pool, resolve, fallback=everyone), label


def _counts(assessed: Sequence[RankedCandidate]) -> dict[str, int]:
    out = {f.value: 0 for f in Feasibility}
    for a in assessed:
        out[a.feasibility.value] += 1
    return {"considered": len(assessed), **out}


def search_materials(
    profile: TargetMaterialProfile,
    candidates: Sequence[MaterialCandidate],
    engine: MaterialSearch | None = None,
) -> MaterialSearchResult:
    return (engine or SimilarityRanker()).search(profile, candidates)
