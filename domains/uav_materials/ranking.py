"""Target-relative, direction-aware similarity for soft preferences.

Only soft preferences with a **numerical weight** are scored. Preferences with
only a qualitative ``importance`` are listed as unscored; they are never
converted into hidden numbers. Weights are used as given, divided by their
sum over the scored preferences.

For each scored preference i, the candidate's value vᵢ is chosen by the
selection policy (conservative, condition-compatible, canonical unit) and
turned into a dimensionless penalty dᵢ ≥ 0:

=========  ==================================================================
maximize   dᵢ = max(0, (bestᵢ − vᵢ) / |bestᵢ|)   bestᵢ = largest value in the pool
minimize   dᵢ = max(0, (vᵢ − bestᵢ) / |bestᵢ|)   bestᵢ = smallest value in the pool
between    dᵢ = 0 inside [lo, hi], else distance to nearest bound / (hi − lo)
target     dᵢ = |vᵢ − t| / |t|
<= / >=    dᵢ = 0 if satisfied, else violation / |bound|
=========  ==================================================================

``maximize`` and ``minimize`` use the *relative gap to the best viable value*:
"this candidate is x·100 % worse than the best option in the normalisation
pool". The pool is the candidates that are not infeasible (all candidates if
fewer than two remain). A relative gap is scale-free and does not exaggerate
tiny spreads the way min–max range normalisation does: a 2 % density
difference stays 0.02. The best values used are reported in
:meth:`TargetRelativeExtractor.parameters`. ``between`` and ``target`` are
measured against the target itself.

With covered preferences C, all scored preferences S and weights wᵢ:

* distance     D  = sqrt( Σ_C wᵢ dᵢ² / Σ_C wᵢ )
* coverage        = Σ_C wᵢ / Σ_S wᵢ
* pessimistic  Dₚ = sqrt( (Σ_C wᵢ dᵢ² + Σ_{S∖C} wᵢ worstᵢ²) / Σ_S wᵢ ), where worstᵢ is
  max(1.0, the largest penalty any candidate has on preference i), so a missing value is
  never cheaper than the worst observed one.

Ranking uses Dₚ, so a missing measurement can never make a candidate look
better than it could be. D and coverage are reported alongside.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from domains.uav_materials.harmonize import canonical_unit
from domains.uav_materials.profile import (
    Operator,
    PropertyRequirement,
    SearchFeatureVector,
    TargetMaterialProfile,
)
from domains.uav_materials.schema import MaterialRecord
from domains.uav_materials.selection import select
from jevpilot import Provenance

NAME = "target_relative/1"

# Optional requirement resolution (see requirements.py): maps a target requirement to the
# property requirement that answers it for this material, or None (+ why) if none does.
Resolve = Callable[[MaterialRecord, PropertyRequirement], tuple[PropertyRequirement | None, str]]


def _identity(_: MaterialRecord, r: PropertyRequirement) -> tuple[PropertyRequirement | None, str]:
    return r, ""


def _key(r: PropertyRequirement) -> str:
    return f"{r.property}:{r.operator}"


@dataclass(frozen=True)
class Scored:
    distance: float | None
    pessimistic_distance: float | None
    coverage: float | None
    matched: tuple[str, ...]
    contributions: tuple[Any, ...]
    provenance: tuple[Provenance, ...]


class TargetRelativeExtractor:
    """Implements :class:`~domains.uav_materials.profile.FeatureExtractor`."""

    name = NAME

    def __init__(
        self,
        profile: TargetMaterialProfile,
        ranges: dict[str, tuple[float, float]],
        worst: dict[str, float],
        resolve: Resolve | None = None,
    ) -> None:
        self.resolve = resolve or _identity
        self.profile = profile
        self.scored = [r for r in profile.soft_preferences if r.weight is not None]
        self.unscored = tuple(_key(r) for r in profile.soft_preferences if r.weight is None)
        self.ranges = ranges
        self.worst = worst
        self.total_weight = sum(r.weight or 0.0 for r in self.scored)

    # -- fitting on the candidate set ------------------------------------------------

    @classmethod
    def fit(
        cls,
        profile: TargetMaterialProfile,
        materials: Sequence[MaterialRecord],
        resolve: Resolve | None = None,
    ) -> TargetRelativeExtractor:
        ranges: dict[str, tuple[float, float]] = {}
        values: dict[str, list[float]] = {}
        for r in profile.soft_preferences:
            if r.weight is None:
                continue
            vals = [v for m in materials if (v := _value(m, r, resolve)) is not None]
            values[_key(r)] = vals
            if vals:
                ranges[_key(r)] = (min(vals), max(vals))
        draft = cls(profile, ranges, {}, resolve)
        worst = {
            _key(r): max([draft.penalty(r, v) for v in values[_key(r)]] + [1.0])
            for r in draft.scored
        }
        return cls(profile, ranges, worst, resolve)

    # -- penalties -------------------------------------------------------------------

    def penalty(self, r: PropertyRequirement, v: float) -> float:
        unit = canonical_unit(r.property)
        if r.operator in (Operator.MAXIMIZE, Operator.MINIMIZE):
            lo, hi = self.ranges.get(_key(r), (v, v))
            best = hi if r.operator is Operator.MAXIMIZE else lo
            gap = best - v if r.operator is Operator.MAXIMIZE else v - best
            return max(gap, 0.0) / (abs(best) or 1.0)  # better than best viable: no penalty
        lower, upper = r.bounds(unit)
        if r.operator is Operator.BETWEEN:
            assert lower is not None and upper is not None
            width = (upper - lower) or abs(lower) or 1.0
            return 0.0 if lower <= v <= upper else min(abs(v - lower), abs(v - upper)) / width
        if r.operator is Operator.TARGET:
            assert lower is not None
            return abs(v - lower) / (abs(lower) or 1.0)
        bound = lower if lower is not None else upper
        assert bound is not None
        ok = (lower is None or v >= lower) and (upper is None or v <= upper)
        return 0.0 if ok else abs(v - bound) / (abs(bound) or 1.0)

    # -- scoring -----------------------------------------------------------------------

    def score(self, material: MaterialRecord) -> Scored:
        from domains.uav_materials.search import Contribution, source_label

        if not self.scored:
            return Scored(None, None, None, (), (), ())
        covered_w = sq = pess = 0.0
        contribs: list[Contribution] = []
        matched: list[str] = []
        provenance: list[Provenance] = []
        for r in self.scored:
            w = r.weight or 0.0
            answer, why = self.resolve(material, r)
            if answer is None:
                pess += w * self.worst.get(_key(r), 1.0) ** 2
                contribs.append(
                    Contribution(
                        property=r.property,
                        operator=str(r.operator),
                        weight=w,
                        value=None,
                        unit=canonical_unit(r.property),
                        penalty=None,
                        share=None,
                        note=why,
                    )
                )
                continue
            sel = select(material, answer)
            unit = canonical_unit(r.property)  # the carrier's scale (same dimension)
            if sel.chosen is None:
                worst = self.worst.get(_key(r), 1.0)
                pess += w * worst**2
                contribs.append(
                    Contribution(
                        property=answer.property,
                        operator=str(r.operator),
                        weight=w,
                        value=None,
                        unit=unit,
                        penalty=None,
                        share=None,
                        note=f"missing: {sel.reason}",
                    )
                )
                continue
            v = sel.chosen.value_in(unit)
            assert v is not None
            d = self.penalty(r, v)
            covered_w += w
            sq += w * d * d
            pess += w * d * d
            matched.append(r.property)
            if sel.chosen.provenance:
                provenance.append(sel.chosen.provenance)
            contribs.append(
                Contribution(
                    property=answer.property,
                    operator=str(r.operator),
                    weight=w,
                    value=v,
                    unit=unit,
                    penalty=d,
                    share=w * d * d / self.total_weight,
                    source=source_label(sel.chosen),
                    note=sel.reason,
                )
            )
        distance = math.sqrt(sq / covered_w) if covered_w else None
        return Scored(
            distance=distance,
            pessimistic_distance=math.sqrt(pess / self.total_weight),
            coverage=covered_w / self.total_weight,
            matched=tuple(dict.fromkeys(matched)),
            contributions=tuple(contribs),
            provenance=tuple(provenance),
        )

    # -- FeatureExtractor interface ---------------------------------------------------

    def target_vector(self, profile: TargetMaterialProfile) -> SearchFeatureVector:
        return SearchFeatureVector(
            features={_key(r): 0.0 for r in self.scored},
            derivation={_key(r): "ideal point: zero penalty" for r in self.scored},
            extractor=NAME,
        )

    def material_vector(self, material: Any, profile: TargetMaterialProfile) -> SearchFeatureVector:
        features: dict[str, float | None] = {}
        for r in self.scored:
            v = _value(material, r, self.resolve)
            features[_key(r)] = None if v is None else self.penalty(r, v)
        return SearchFeatureVector(
            features=features, derivation=self.parameters()["preferences"], extractor=NAME
        )

    def parameters(self) -> dict[str, Any]:
        prefs: dict[str, Any] = {}
        for r in self.scored:
            unit = canonical_unit(r.property)
            k = _key(r)
            if r.operator in (Operator.MAXIMIZE, Operator.MINIMIZE):
                lo_hi = self.ranges.get(k)
                best = (
                    None
                    if lo_hi is None
                    else (lo_hi[1] if r.operator is Operator.MAXIMIZE else lo_hi[0])
                )
                method = (
                    f"relative gap to best viable value: best={best:.6g} {unit} "
                    f"(pool range {lo_hi[0]:.6g}–{lo_hi[1]:.6g})"
                    if lo_hi and best is not None
                    else "no candidate has a value"
                )
            else:
                lower, upper = r.bounds(unit)
                method = f"target-relative: bounds=({lower}, {upper}) {unit}"
            prefs[k] = method
        return {
            "extractor": NAME,
            "weights": {_key(r): r.weight for r in self.scored},
            "weight_sources": {_key(r): r.weight_source for r in self.scored},
            "weight_sum": self.total_weight,
            "preferences": prefs,
            "worst_penalty_for_missing": self.worst,
            "unscored_preferences": list(self.unscored),
        }


def _value(
    material: MaterialRecord, r: PropertyRequirement, resolve: Resolve | None = None
) -> float | None:
    answer, _ = (resolve or _identity)(material, r)
    if answer is None:
        return None
    sel = select(material, answer)
    return None if sel.chosen is None else sel.chosen.value_in(canonical_unit(r.property))
