"""Minimal material identity: canonical names, source names, aliases, designations.

Real sources name the same material differently ("Al 7075-T6", "7075-T6",
"AA7075-T6"). Identity is decided on *parsed designation parts* (alloy
system, designation, cladding, temper), never on display-name equality.

What this layer will not do:

* merge different designations (AZ31X is an experimental alloy, not AZ31B);
* merge different tempers. A query with an unspecified temper ("6061-T")
  matches at the *alloy level* only, and the match says so;
* invent aliases. Besides mechanical variants of the designation, it only
  uses the small curated table below, with each entry's rationale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field

from jevpilot import FrozenModel


class MatchLevel(StrEnum):
    EXACT = "exact"  # same system, designation, cladding and temper
    ALLOY = "alloy"  # same system, designation and cladding; temper unspecified in the query
    NONE = "none"


# name → (system, designation). Each entry is a well-known naming equivalence.
CURATED_ALIASES: dict[str, tuple[str, str]] = {
    "ti 6-4": ("titanium", "Ti-6Al-4V"),  # common shorthand
    "ti-6-4": ("titanium", "Ti-6Al-4V"),
    "cp titanium": ("titanium", "commercially pure"),
    "commercially pure titanium": ("titanium", "commercially pure"),
}

_ALU = re.compile(
    r"^(?:(?P<clad>alclad|clad)\s*)?(?:aa|al|aluminum|aluminium)?\s*"
    r"(?P<des>[1-8]\d{3})(?:\s*(?:aluminum|aluminium))?(?:\s*alloy)?"
    r"(?:[\s-]*(?P<temper>[OFHWT]\d*[A-Z0-9]*))?$",
    re.IGNORECASE,
)
_MAG = re.compile(
    r"^(?:magnesium\s*(?:alloy)?\s*)?(?P<des>[A-Z]{2}\d{2}[A-Z])(?:[\s-]*(?P<temper>[OFHT]\d*))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Designation:
    system: str  # aluminum, magnesium, titanium, nickel, ...
    designation: str  # "7075", "AZ31B", "Ti-6Al-4V"
    clad: bool = False
    temper: str | None = None  # None: not stated

    @property
    def key(self) -> tuple[str, str, bool]:
        return (self.system, self.designation.upper(), self.clad)


def parse_designation(name: str) -> Designation | None:
    """Parse a source's material name into designation parts (None if unrecognised)."""
    text = " ".join(name.replace("—", "-").split())
    lowered = text.lower()
    if lowered in CURATED_ALIASES:
        system, des = CURATED_ALIASES[lowered]
        return Designation(system, des)
    m = _ALU.match(text)
    if m:
        temper = m.group("temper")
        return Designation(
            "aluminum",
            m.group("des"),
            bool(m.group("clad")),
            temper.upper() if temper and temper.upper() != "T" else None,
        )
    m = _MAG.match(text)
    if m:
        return Designation("magnesium", m.group("des").upper(), False, m.group("temper"))
    ti = re.match(r"^(Ti-[0-9A-Za-z.-]+?)(?:\s+(?:sheet|bar|plate|extrusion|forging).*)?$", text)
    if ti:
        return Designation("titanium", ti.group(1))
    return None


class MaterialIdentity(FrozenModel):
    canonical_id: str
    canonical_name: str
    system: str
    designation: str
    clad: bool = False
    tempers: tuple[str, ...] = ()  # tempers / conditions covered by the record
    product_form: str = ""
    aliases: tuple[str, ...] = ()
    source_names: dict[str, str] = Field(default_factory=dict)  # source → name as printed
    source_ids: dict[str, list[str]] = Field(default_factory=dict)  # source → identifiers

    def match(self, query: Designation) -> MatchLevel:
        if (self.system, self.designation.upper(), self.clad) != query.key:
            return MatchLevel.NONE
        if query.temper is None:
            return MatchLevel.ALLOY
        tempers = {t.upper() for t in self.tempers}
        return MatchLevel.EXACT if query.temper.upper() in tempers else MatchLevel.NONE


def generated_aliases(
    system: str, designation: str, clad: bool, tempers: tuple[str, ...]
) -> tuple[str, ...]:
    """Mechanical spelling variants only (no knowledge-based aliases)."""
    if system != "aluminum":
        return tuple(dict.fromkeys([designation, *(f"{designation}-{t}" for t in tempers)]))
    prefixes = ["Clad ", "Alclad "] if clad else ["", "AA", "Al ", "AA "]
    names = [f"{p}{designation}" for p in prefixes]
    names += [f"{n}-{t}" for n in names for t in tempers]
    return tuple(dict.fromkeys(names))


class IdentityIndex:
    """Resolves source names to canonical identities by designation parts."""

    def __init__(self, identities: list[MaterialIdentity]) -> None:
        self.identities = identities

    def resolve(self, name: str) -> list[tuple[MaterialIdentity, MatchLevel]]:
        """Identities the name refers to, with the match level.

        Names the designation parser does not know (nickel, copper, cobalt
        alloys) match only when the whole name equals a record's designation,
        ignoring case and spacing, and then only at alloy level: no temper or
        condition is read from such a name.
        """
        query = parse_designation(name)
        if query is None:
            wanted = " ".join(name.split()).casefold()
            return [
                (i, MatchLevel.ALLOY)
                for i in self.identities
                if " ".join(i.designation.split()).casefold() == wanted
            ]
        hits = [(i, i.match(query)) for i in self.identities]
        return [(i, level) for i, level in hits if level is not MatchLevel.NONE]
