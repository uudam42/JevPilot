"""Which measurement answers a requirement, and what ``not_applicable`` means.

Selection policy (deterministic; nothing is averaged):

1. usable = has a value, and conditions are not ``INCOMPATIBLE``
   (:mod:`~domains.uav_materials.compatibility`), and its qualifier can speak
   to the requirement's direction (an upper bound "≤ x" says nothing about
   "≥" requirements or maximisation);
2. keep the best compatibility tier: exact > compatible > unconditioned;
3. then prefer exact material identity over alloy-level identity (a source
   naming the alloy without the temper);
4. then prefer the strongest statistical basis: A > B > S > typical > unstated;
5. then take the most **conservative** value for the requirement's direction:
   lowest for lower bounds and maximisation, highest for upper bounds and
   minimisation, the worst case for ranges and targets. Across directions
   and thicknesses of a design-allowable table, this is the envelope.

Everything not chosen is returned as an alternative, so nothing is lost.

``not_applicable`` policy (explicit, per property):

* ``water_absorption``: a source records "not applicable" only for materials
  with no water-uptake mechanism (e.g. fully dense metals), so an upper-bound
  requirement (``<=``) is **satisfied**. Any other requirement type stays
  undetermined;
* every other property: **undetermined**. "No glass transition" does not
  satisfy a glass-transition requirement, and "not applicable" corrosion
  (e.g. a polymer) does not prove resistance to the requirement's
  environment.

In soft ranking, ``not_applicable`` is a missing dimension, never a zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domains.uav_materials.compatibility import Compat, compatibility
from domains.uav_materials.harmonize import canonical_unit
from domains.uav_materials.profile import Operator, PropertyRequirement
from domains.uav_materials.schema import (
    MaterialRecord,
    Measurement,
    MissingReason,
    Qualifier,
    StatisticalBasis,
)

BASIS_ORDER = {
    StatisticalBasis.A: 4,
    StatisticalBasis.B: 3,
    StatisticalBasis.S: 2,
    StatisticalBasis.TYPICAL: 1,
    None: 0,
}
LOWER_SIDE = {Operator.GE, Operator.MAXIMIZE}  # a low value is the risk
UPPER_SIDE = {Operator.LE, Operator.MINIMIZE}  # a high value is the risk
NOT_APPLICABLE_SATISFIES = {("water_absorption", Operator.LE)}


@dataclass
class Selection:
    chosen: Measurement | None
    compat: Compat | None
    reason: str
    alternatives: list[Measurement] = field(default_factory=list)
    excluded: list[tuple[Measurement, str]] = field(default_factory=list)
    not_applicable: bool = False


def identity_rank(m: Measurement) -> int:
    level = (m.provenance.metadata.get("identity_match") if m.provenance else None) or "exact"
    return 1 if level == "exact" else 0


def qualifier_usable(m: Measurement, op: Operator) -> bool:
    if m.qualifier is Qualifier.EQUAL:
        return True
    if m.qualifier is Qualifier.AT_MOST:
        return op in UPPER_SIDE
    return op in LOWER_SIDE


def _penalty(v: float, req: PropertyRequirement, unit: str) -> float:
    lower, upper = req.bounds(unit)
    if lower is not None and v < lower:
        return lower - v
    if upper is not None and v > upper:
        return v - upper
    return 0.0


def select(material: MaterialRecord, req: PropertyRequirement) -> Selection:
    found = material.measurements(req.property)
    if not found:
        return Selection(None, None, "no measurement recorded")
    valued = [m for m in found if not m.is_missing]
    if not valued:
        reasons = sorted({str(m.missing) for m in found if m.missing})
        na = MissingReason.NOT_APPLICABLE.value in reasons
        return Selection(
            None, None, "only missing values: " + ", ".join(reasons), not_applicable=na
        )
    usable: list[tuple[Measurement, Compat]] = []
    excluded: list[tuple[Measurement, str]] = []
    for m in valued:
        if not m.comparable_conditions:
            excluded.append((m, "conditions needed for comparison not reported"))
            continue
        if not qualifier_usable(m, req.operator):
            excluded.append((m, f"a '{m.qualifier}' bound cannot answer '{req.operator}'"))
            continue
        level, why = compatibility(req.conditions, m.conditions)
        if level is Compat.INCOMPATIBLE:
            excluded.append((m, why))
        else:
            usable.append((m, level))
    if not usable:
        return Selection(None, None, "no measurement with compatible conditions", excluded=excluded)
    best = max(c for _, c in usable)
    tier = [m for m, c in usable if c == best]
    top_id = max(identity_rank(m) for m in tier)
    tier = [m for m in tier if identity_rank(m) == top_id]
    top_basis = max(BASIS_ORDER[m.statistical_basis] for m in tier)
    tier = [m for m in tier if BASIS_ORDER[m.statistical_basis] == top_basis]
    unit = canonical_unit(req.property)

    def value(m: Measurement) -> float:
        v = m.value_in(unit)
        assert v is not None
        return v

    if req.operator in LOWER_SIDE:
        chosen = min(tier, key=value)
    elif req.operator in UPPER_SIDE:
        chosen = max(tier, key=value)
    else:
        chosen = max(tier, key=lambda m: _penalty(value(m), req, unit))
    alternatives = [m for m, _ in usable if m is not chosen]
    return Selection(
        chosen,
        best,
        f"{best.name.lower()} conditions; conservative choice of "
        f"{len(tier)} equally ranked value(s)",
        alternatives,
        excluded,
    )
