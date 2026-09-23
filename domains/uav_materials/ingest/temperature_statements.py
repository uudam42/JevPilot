"""Temperature statements quoted from handbook prose (MIL-HDBK-5J, NASA 7075 handbooks).

Elevated-temperature *curves* in these handbooks are figures and are not
used: values are never read off plots. What remains is a small number of
explicit sentences in machine-readable text. Each :class:`Statement` quotes
one verbatim. The parser checks that the quote (and a context quote naming
the material) occurs on the cited page and reads the number and unit from
the quote, so no value is typed in independently of the source.

Statement kinds keep their meanings apart:

=====================  ==================================================================
``max_service``        "The maximum service temperature for X is ..." → max_service_temperature
``application``        "used for parts requiring <use> up to ..." → application_temperature_limit
                       (valid for the stated use only; not a max service temperature)
``exposure_stability`` "can withstand prolonged exposure up to ... without loss of <property>"
                       → exposure_stability_temperature
``melting_range``      "Melting range is approximately a to b" → melting_temperature (onset a)
=====================  ==================================================================

Melting point never implies a service temperature, and none of these kinds
is converted into another. Statements that give no usable limit (comparative
strength claims, oxidation-resistance ranges, family-wide remarks, sentences
whose scope is ambiguous) are kept in :data:`NOT_USED` with the reason, so the
audit trail shows what was read and rejected.

The NASA 7075 value comes from two editions whose text layers are OCR of
scans. It is accepted only because both editions state the same numbers,
which is checked at build time (:class:`OcrDisagreement` otherwise).

MIL-HDBK-5J text renders the degree sign as "(": "500(F" is 500 °F.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

EXTRACTOR_VERSION = "temperature-statements/1"

MIL = "MIL-HDBK-5J"
MIL_URI = "https://archive.org/details/milhdbk-5-j"
NASA_1966 = "NASA-CR-80764"
NASA_1972 = "NASA-CR-123773"
DOCUMENTS: dict[str, dict[str, str]] = {
    MIL: {"uri": MIL_URI, "kind": "handbook", "extraction": "pypdf 6.19.0 digital text layer"},
    NASA_1966: {
        "uri": "https://ntrs.nasa.gov/citations/19670004550",
        "kind": "handbook",
        "extraction": "pypdf 6.19.0 of the NTRS PDF's embedded OCR text",
    },
    NASA_1972: {
        "uri": "https://ntrs.nasa.gov/citations/19720022809",
        "kind": "handbook",
        "extraction": "pypdf 6.19.0 of the NTRS PDF's embedded OCR text",
    },
}


class Kind(StrEnum):
    MAX_SERVICE = "max_service"
    APPLICATION = "application"
    EXPOSURE_STABILITY = "exposure_stability"
    MELTING_RANGE = "melting_range"


PROPERTY_FOR: dict[Kind, str] = {
    Kind.MAX_SERVICE: "max_service_temperature",
    Kind.APPLICATION: "application_temperature_limit",
    Kind.EXPOSURE_STABILITY: "exposure_stability_temperature",
    Kind.MELTING_RANGE: "melting_temperature",
}


@dataclass(frozen=True)
class Citation:
    document: str
    page: int  # PDF page number (1-based)
    quote: str  # verbatim, whitespace-normalised
    context: str = ""  # a second verbatim quote on the same page naming the material


@dataclass(frozen=True)
class Statement:
    key: str
    kind: Kind
    material: str  # material name as the source states it (for identity resolution)
    number_text: str  # the number as written in the quote
    citations: tuple[Citation, ...]  # every one must contain the same number and unit
    section: str = ""
    upper_text: str | None = None  # range end ("a to b"), melting ranges only
    stated_use: str = ""  # application: the use the limit belongs to
    criterion: str = ""  # exposure stability: the property that is not degraded
    duration_hours: float | None = None  # a stated exposure-time limit
    duration_text: str = ""  # the duration as stated, numeric or not
    note: str = ""


@dataclass(frozen=True)
class NotUsed:
    key: str
    material: str
    citation: Citation
    reason: str


def _mil(page: int, quote: str, context: str = "") -> tuple[Citation, ...]:
    return (Citation(MIL, page, quote, context),)


STATEMENTS: tuple[Statement, ...] = (
    Statement(
        key="c17200-max-service",
        kind=Kind.MAX_SERVICE,
        material="C17200 copper beryllium",
        number_text="500",
        section="7.3.2.0",
        citations=_mil(
            1192,
            "The maximum service temperature for C17200 copper beryllium products is 500(F "
            "for up to 100 hours.",
        ),
        duration_hours=100,
        duration_text="for up to 100 hours",
        note="applies to C17200 products in general (alloy level)",
    ),
    # The next two sentences look like max service temperatures of 2024-T3 and
    # 7475-T761. They belong to aramid-fibre/aluminium sheet LAMINATES (sections
    # 7.5.1, 7.5.2), which are not in the dataset, and must stay unresolved.
    Statement(
        key="2024-t3-laminate-max-service",
        kind=Kind.MAX_SERVICE,
        material="2024-T3 aramid fiber reinforced sheet laminate",
        number_text="200",
        section="7.5.1.0",
        citations=_mil(
            1212,
            "This product has good corrosion resistance. The maximum service temperature is 200(F.",
            context="7.5.1 2024-T3 ARAMID FIBER REINFORCED SHEET LAMINATE",
        ),
    ),
    Statement(
        key="7475-t761-laminate-max-service",
        kind=Kind.MAX_SERVICE,
        material="7475-T761 aramid fiber reinforced sheet laminate",
        number_text="200",
        section="7.5.2.0",
        citations=_mil(
            1221,
            "This product has good corrosion resistance. The maximum service temperature is 200(F.",
            context="7.5.2 7475-T761 ARAMID FIBER REINFORCED SHEET LAMINATE",
        ),
    ),
    Statement(
        key="az31b-application",
        kind=Kind.APPLICATION,
        material="AZ31B",
        number_text="300",
        section="4.2.1.0",
        citations=_mil(
            840,
            "is used primarily for applications where the temperature does not exceed 300(F.",
            context="AZ31B has good room-temperature strength and ductility",
        ),
        stated_use="primary applications",
    ),
    Statement(
        key="cp-titanium-usage-limit",
        kind=Kind.APPLICATION,
        material="commercially pure titanium",
        number_text="1050",
        section="5.2.1.0",
        citations=_mil(
            897,
            "Titanium has an unusually high affinity for oxygen, nitrogen, and hydrogen at "
            "temperatures above 1050(F. This results in embrittlement of the material, thus "
            "usage should be limited to temperatures below that indicated.",
            context="5.2.1 COMMERCIALLY PURE TITANIUM 5.2.1.0 Comments and Properties",
        ),
        stated_use="any use: embrittlement by oxygen, nitrogen and hydrogen uptake above it",
        note="an upper bound on use, not a rated continuous service temperature",
    ),
    Statement(
        key="ti-5al-2.5sn-application",
        kind=Kind.APPLICATION,
        material="Ti-5Al-2.5Sn",
        number_text="900",
        section="5.3.1.0",
        citations=_mil(
            907,
            "it is primarily suitable for room to elevated temperature applications (up to "
            "900 (F or to 1100 (F for short times)",
            context="The normal purity grade also may be used at low temperatures",
        ),
        stated_use="room to elevated temperature applications (normal purity grade)",
    ),
    Statement(
        key="ti-5al-2.5sn-application-short",
        kind=Kind.APPLICATION,
        material="Ti-5Al-2.5Sn",
        number_text="1100",
        section="5.3.1.0",
        citations=_mil(
            907,
            "it is primarily suitable for room to elevated temperature applications (up to "
            "900 (F or to 1100 (F for short times)",
            context="The normal purity grade also may be used at low temperatures",
        ),
        stated_use="room to elevated temperature applications (normal purity grade)",
        duration_text="short times (not quantified)",
    ),
    Statement(
        key="ti-6al-4v-exposure-stability",
        kind=Kind.EXPOSURE_STABILITY,
        material="Ti-6Al-4V",
        number_text="750",
        section="5.4.1.0",
        citations=_mil(
            943,
            "Ti-6Al-4V can withstand prolonged exposure to temperatures up to 750(F without "
            "loss of ductility.",
        ),
        criterion="no loss of ductility after prolonged exposure",
        duration_text="prolonged (not quantified)",
    ),
    *(
        Statement(
            key=f"inconel-x-750-application-{number}",
            kind=Kind.APPLICATION,
            material="Inconel X-750",
            number_text=number,
            section="6.3.6.0",
            citations=_mil(
                1117,
                "It is used for parts requiring high strength up to 1000 (F or high creep "
                "strength up to 1500(F and for low-stressed parts operating up to 1900 (F.",
                context="Inconel X-750 is a high-strength oxidation-resistant nickel-base alloy.",
            ),
            stated_use=use,
        )
        for number, use in (
            ("1000", "parts requiring high strength"),
            ("1500", "parts requiring high creep strength"),
            ("1900", "low-stressed parts"),
        )
    ),
    Statement(
        key="inconel-718-application",
        kind=Kind.APPLICATION,
        material="Inconel 718",
        number_text="1300",
        section="6.3.5.0",
        citations=_mil(
            1091,
            "this alloy finds applications requiring either (1) high resistance to creep and "
            "stress rupture to 1300(F",
            context="Inconel 718 is available in all wrought forms",
        ),
        stated_use="applications requiring high resistance to creep and stress rupture "
        "(depending on heat treatment)",
    ),
    Statement(
        key="7075-melting-range",
        kind=Kind.MELTING_RANGE,
        material="7075",
        number_text="477",
        upper_text="638",
        section="Critical Temperatures (1966 ed.) / 3.3 (1972 ed.)",
        citations=(
            Citation(
                NASA_1972,
                18,
                "Critical Temperatures. Melting range is approximately 477 to 638°C.",
            ),
            Citation(
                NASA_1966,
                19,
                "Critical Temperatures. Melting range is approximately 477 to 638C.",
            ),
        ),
        note="approximate melting range 477–638 °C; the value is the range onset "
        "(property definition). A melting range is not a service temperature.",
    ),
)

NOT_USED: tuple[NotUsed, ...] = (
    NotUsed(
        "hr-120-comparative",
        "HAYNES HR-120",
        Citation(
            MIL,
            1149,
            "Oxidation resistance is comparable to other Fe-Ni-Cr materials such as alloys 330 "
            "and 800H, yet with a greater strength at temperatures up to 2000(F.",
        ),
        "comparative strength claim; states no limit and no strength value",
    ),
    NotUsed(
        "ti-5al-2.5sn-oxidation",
        "Ti-5Al-2.5Sn",
        Citation(MIL, 907, "The alloy has good oxidation resistance up to 1050 (F."),
        "oxidation-resistance range, not a service or strength limit",
    ),
    NotUsed(
        "copper-alloys-general",
        "copper alloys (family)",
        Citation(MIL, 1188, "Copper alloys frequently are used at temperatures up to 480(F."),
        "family-wide remark about copper alloys; not attached to a specific alloy",
    ),
    NotUsed(
        "mp35n-scope",
        "MP35N",
        Citation(
            MIL,
            1201,
            "This alloy is suitable for parts requiring ultrahigh strength, good ductility and "
            "excellent corrosion and oxidation resistance up to 700 (F.",
        ),
        "ambiguous scope: 'up to 700 F' may qualify only the corrosion and oxidation resistance",
    ),
    NotUsed(
        "mp159-scope",
        "MP159",
        Citation(
            MIL,
            1207,
            "The alloy maintains its ultrahigh strength very well at temperatures up to 1100(F.",
        ),
        "qualitative strength-retention claim without a retention value; the companion "
        "sentence has the same scope ambiguity as MP35N",
    ),
)


class QuoteNotFound(ValueError):
    """A quote (or context) is not on the cited page of the raw text."""


class OcrDisagreement(ValueError):
    """Citations of one statement disagree on the number or unit."""


def normalise(text: str) -> str:
    text = re.sub(r"¬\s*", "", text)
    return " ".join(text.split())


_UNIT = {"F": "degF", "C": "degC"}


def parse_temperature(
    quote: str, number_text: str, upper_text: str | None = None
) -> tuple[float, float | None, str]:
    """Read ``number_text`` [to ``upper_text``] and its temperature unit from ``quote``.

    The unit letter must follow the (last) number directly, optionally after a
    space and a degree sign ("°", or "(" as MIL-HDBK-5J's text layer renders it).
    """
    number = re.escape(number_text)
    if upper_text is not None:
        number += r"\s*to\s*" + re.escape(upper_text)
    m = re.search(rf"(?<![\d.]){number}\s*[(°]?\s*(?P<unit>[FC])\b", quote)
    if not m:
        raise QuoteNotFound(f"no temperature '{number_text}' with a unit in {quote!r}")
    upper = float(upper_text) if upper_text is not None else None
    return float(number_text), upper, _UNIT[m.group("unit")]


def verify(pages: dict[tuple[str, int], str], s: Statement) -> tuple[float, float | None, str]:
    """Check every citation against the raw pages; return (value, upper, unit)."""
    readings = set()
    for c in s.citations:
        text = normalise(pages.get((c.document, c.page), ""))
        for q in (c.quote, c.context):
            if q and normalise(q) not in text:
                raise QuoteNotFound(f"{s.key}: not on {c.document} p.{c.page}: {q!r}")
        readings.add(parse_temperature(c.quote, s.number_text, s.upper_text))
    if len(readings) != 1:
        raise OcrDisagreement(f"{s.key}: citations disagree: {sorted(readings)}")
    return readings.pop()


def verify_not_used(pages: dict[tuple[str, int], str], n: NotUsed) -> None:
    text = normalise(pages.get((n.citation.document, n.citation.page), ""))
    if normalise(n.citation.quote) not in text:
        raise QuoteNotFound(f"{n.key}: not on {n.citation.document} p.{n.citation.page}")


@dataclass
class Extracted:
    statement: Statement
    measurement: Any  # schema.Measurement
    value: float
    unit: str
    extra: dict[str, Any] = field(default_factory=dict)


def build_measurements(pages: dict[tuple[str, int], str], retrieved: str) -> list[Extracted]:
    """One Measurement per statement, verified against ``pages``."""
    from domains.uav_materials.ingest.mil_hdbk_5j import deterministic_id, retrieved_at
    from domains.uav_materials.schema import (
        Measurement,
        MeasurementBasis,
        ProvenanceStatus,
        Quantity,
        StatisticalBasis,
        TestConditions,
    )
    from jevpilot import Provenance, SourceRef

    out: list[Extracted] = []
    for s in STATEMENTS:
        value, upper, unit = verify(pages, s)
        other: dict[str, Any] = {"statement_kind": s.kind.value}
        if s.stated_use:
            other["stated_use"] = s.stated_use
        if s.criterion:
            other["criterion"] = s.criterion
        if s.duration_text:
            other["duration"] = s.duration_text
        if upper is not None:
            other["range_upper"] = f"{upper:g} {unit}"
        sources = tuple(
            SourceRef(
                kind=DOCUMENTS[c.document]["kind"],
                identifier=c.document,
                uri=DOCUMENTS[c.document]["uri"],
                metadata={
                    "page": c.page,
                    "section": s.section,
                    "quote": c.quote,
                    "extraction": DOCUMENTS[c.document]["extraction"],
                    "statement": s.key,
                    "source_material_name": s.material,
                    "retrieved": retrieved,
                },
            )
            for c in s.citations
        )
        measurement = Measurement(
            property=PROPERTY_FOR[s.kind],
            value=value,
            unit=unit,
            basis=MeasurementBasis.SPECIFIED,
            statistical_basis=(StatisticalBasis.TYPICAL if s.kind is Kind.MELTING_RANGE else None),
            conditions=TestConditions(
                exposure_duration=(
                    Quantity(value=s.duration_hours, unit="h") if s.duration_hours else None
                ),
                other=other,
            ),
            test_method="handbook statement (prose)",
            provenance=Provenance(
                id=deterministic_id("temperature-statement", s.key),
                created_at=retrieved_at(retrieved),
                sources=sources,
                metadata={
                    "extractor": EXTRACTOR_VERSION,
                    "cross_validated_by": [c.document for c in s.citations[1:]],
                },
            ),
            provenance_status=ProvenanceStatus.SOURCED,
            notes=s.note,
        )
        out.append(Extracted(s, measurement, value, unit))
    return out
