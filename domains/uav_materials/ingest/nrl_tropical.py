"""Corrosion values from NRL Report "Corrosion of Metals in Tropical Environments, Part 6".

Source: Naval Research Laboratory, DTIC AD0609618 (Internet Archive item
``DTIC_AD0609618``). It is a US Government work, and the report states
"Unlimited availability".

The report's results table (Table 3) is a scanned image without a text layer
and is **not** used: its values would have to be transcribed by hand, which
this pipeline does not do. Only statements in the machine-readable text are
used. Each :class:`Extraction` quotes the text verbatim (OCR spelling
included). The parser checks that the quote occurs in the raw text and reads
the number from the quote, so no value is typed in independently of the
source.

Rates derived from average penetration divided by exposure time assume linear
progression. They are marked ``basis=derived`` with that note, because
corrosion is often non-linear in time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SOURCE_ID = "NRL-AD0609618"
SOURCE_URI = "https://archive.org/details/DTIC_AD0609618"
SOURCE_TITLE = (
    "Corrosion of Metals in Tropical Environments, Part 6: Aluminum and Magnesium "
    "(Naval Research Laboratory; DTIC AD0609618)"
)
EXTRACTOR_VERSION = "nrl-ad0609618-quotes/1"

SITES = {
    "marine_atmosphere": "tropical marine atmosphere: roof of the Washington Hotel, Cristobal, "
    "Canal Zone, 300 ft from the Caribbean shore",
    "inland_atmosphere": "tropical inland atmosphere: near Miraflores Locks, Canal Zone",
    "seawater_tidal": "tropical Pacific sea water, mean tide: Fort Amador, Canal Zone",
}


@dataclass(frozen=True)
class Extraction:
    key: str
    quote: str  # verbatim (whitespace-normalised) text from the report
    number_text: str  # the number as written inside the quote (OCR spacing included)
    materials: tuple[str, ...]  # source material names
    environment: str
    kind: str  # "rate_mil_per_year" or "penetration_mil"
    exposure_years: float | None
    qualifier: str  # "=" or "<="
    context: str = ""  # a second verbatim quote establishing which materials are meant


EXTRACTIONS = (
    Extraction(
        key="al-marine-atmosphere-16y",
        quote="it still caused only a maximum 0. 15-mil average penetration after 16 wears",
        number_text="0. 15",
        materials=("1100", "6061-T", "Alclad 2024-T"),
        environment="marine_atmosphere",
        kind="penetration_mil",
        exposure_years=16,
        qualifier="<=",
        context="Three aluminum alloys were exposed in the two atmospheric environments",
    ),
    Extraction(
        key="al-inland-atmosphere-16y",
        quote="the greatest average penetration of any of the alloys at the inland site was "
        "only 0.09 mil",
        number_text="0.09",
        materials=("1100", "6061-T", "Alclad 2024-T"),
        environment="inland_atmosphere",
        kind="penetration_mil",
        exposure_years=16,
        qualifier="<=",
        context="Three aluminum alloys were exposed in the two atmospheric environments",
    ),
    Extraction(
        key="al1100-mean-tide-16y",
        quote="the loss of 1100 amounted to only 0.5 mil after 16 years’ exposure",
        number_text="0.5",
        materials=("1100",),
        environment="seawater_tidal",
        kind="penetration_mil",
        exposure_years=16,
        qualifier="=",
    ),
    Extraction(
        key="az31x-marine-atmosphere-rate",
        quote="A steady corrosion rate of 0.94 mil per year was found for the former at the "
        "Caribbean coastal site",
        number_text="0.94",
        materials=("AZ31X",),
        environment="marine_atmosphere",
        kind="rate_mil_per_year",
        exposure_years=None,
        qualifier="=",
        context="magnesium alloys AZ31X and A261X",
    ),
    Extraction(
        key="az31x-mean-tide-8y",
        quote="The filial average penetration at mean tide was 45 mils",
        number_text="45",
        materials=("AZ31X",),
        environment="seawater_tidal",
        kind="penetration_mil",
        exposure_years=8,
        qualifier="=",
        context="Figure 8 presents penetration with time for AZ31X in the five environments "
        "over an ei^t-year period",
    ),
)


class QuoteNotFound(ValueError):
    """An extraction's quote (or context) is not in the raw text."""


def normalise(text: str) -> str:
    text = re.sub(r"¬\s*", "", text)  # OCR line-break hyphenation
    return " ".join(text.split())


def verify(raw_text: str, e: Extraction) -> float:
    """Check the quote (and context) exist verbatim and return the number read from the quote."""
    norm = normalise(raw_text)
    for q in (e.quote, e.context):
        if q and normalise(q) not in norm:
            raise QuoteNotFound(f"{e.key}: quote not found in raw text: {q!r}")
    if e.number_text not in e.quote:
        raise QuoteNotFound(f"{e.key}: {e.number_text!r} is not part of the quote")
    return float(e.number_text.replace(" ", ""))


def build_measurements(raw_text: str, retrieved: str) -> dict[str, list[Any]]:
    """Source material name → corrosion Measurements (verified against ``raw_text``)."""
    from domains.uav_materials.ingest.mil_hdbk_5j import deterministic_id, retrieved_at
    from domains.uav_materials.schema import (
        EnvironmentClass,
        Measurement,
        MeasurementBasis,
        ProvenanceStatus,
        Qualifier,
        Quantity,
        TestConditions,
    )
    from jevpilot import Provenance, SourceRef

    out: dict[str, list[Any]] = {}
    for e in EXTRACTIONS:
        number = verify(raw_text, e)
        if e.kind == "rate_mil_per_year":
            value, basis, note = number, MeasurementBasis.MEASURED, "rate as reported"
        else:
            assert e.exposure_years
            value = number / e.exposure_years
            basis = MeasurementBasis.DERIVED
            note = (
                f"average rate = {number} mil average penetration / {e.exposure_years:g} "
                "years; assumes linear progression"
            )
        for name in e.materials:
            out.setdefault(name, []).append(
                Measurement(
                    property="corrosion_rate",
                    value=value,
                    unit="mil/year",
                    qualifier=Qualifier(e.qualifier),
                    basis=basis,
                    conditions=TestConditions(
                        environment=EnvironmentClass(e.environment),
                        medium=SITES[e.environment],
                        exposure_duration=(
                            Quantity(value=e.exposure_years, unit="year")
                            if e.exposure_years
                            else None
                        ),
                    ),
                    test_method="natural exposure, duplicate panels; weight loss → penetration",
                    provenance=Provenance(
                        id=deterministic_id(SOURCE_ID, e.key, name),
                        created_at=retrieved_at(retrieved),
                        sources=(
                            SourceRef(
                                kind="report",
                                identifier=SOURCE_ID,
                                uri=SOURCE_URI,
                                metadata={
                                    "quote": e.quote,
                                    "extraction": e.key,
                                    "source_material_name": name,
                                    "retrieved": retrieved,
                                },
                            ),
                        ),
                        metadata={"extractor": EXTRACTOR_VERSION},
                    ),
                    provenance_status=ProvenanceStatus.SOURCED,
                    notes=note,
                )
            )
    return out
