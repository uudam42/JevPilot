"""Constituent (fibre / matrix) properties for micromechanics, with verified provenance.

Source: the NASA constituent data bank used with the Chamis micromechanics
equations. It is printed in three NASA documents (all NTRS "public", U.S.
Government works; see ``data/uav_materials/sources/nasa_micromechanics.json``):

* NASA TM-83320 (Chamis, 1983), Table 2 (matrix properties)
* NASA TP-2515 (Murthy & Chamis, 1986), ICAN manual: resident data bank
  (appendix C) and the sample run's data-bank echo
* NASA TP-3290 (Murthy, Ginty & Sanfeliz, 1993), ICAN-2 manual: dedicated
  data bank (appendix A) and the sample run's data-bank echo

**Data quality.** TM-83320 says of these tables: "The data in these tables
were compiled from many sources and many values are estimates which were
inferred from predicted results and curve fits." They are indicative
handbook values (evidence type ``datasheet``, quality ``indicative``), not
test results for a specific product lot.

**Extraction.** Every document is a scan with an OCR text layer, so no single
rendering is trusted. Each value lists its renderings: a verbatim quote from
the committed page text plus the number as printed. A rendering is read only
through *unambiguous* OCR repairs (O→0, l/I→1, "E 06"→"E+06"); a token with
any other character (for example "S", which these scans use for both 5 and 8)
is unreadable and ignored. A value is accepted only when renderings from at
least two different documents agree and agreeing renderings outnumber
disagreeing ones. Disagreements are kept in the provenance, never dropped.
Nothing is typed in independently of the quoted text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field

from domains.uav_materials.schema import EvidenceType
from domains.uav_materials.units import convert
from jevpilot import FrozenModel, Provenance, SourceRef, stable_digest

EXTRACTOR_VERSION = "nasa-constituent-renderings/1"
RAW = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "uav_materials"
    / "raw"
    / "nasa_micromechanics"
    / "pages.jsonl"
)
RETRIEVED = "2026-09-23"
TM83320, TP2515, TP3290 = "NASA-TM-83320", "NASA-TP-2515", "NASA-TP-3290"
DOCUMENTS = {
    TM83320: "https://ntrs.nasa.gov/citations/19830011546",
    TP2515: "https://ntrs.nasa.gov/citations/19860012143",
    TP3290: "https://ntrs.nasa.gov/citations/19930008950",
}
DATA_QUALITY = (
    "indicative: NASA TM-83320 notes its constituent tables were 'compiled from many "
    "sources and many values are estimates which were inferred from predicted results "
    "and curve fits'"
)
UNIT_BASIS_ICAN = "ICAN units table, NASA TP-3290 p.35 (E, G, S: psi; RHO: lb/in**3)"


# -- renderings ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rendering:
    """One printed occurrence of a value."""

    document: str
    page: int
    quote: str  # verbatim, whitespace-normalised page text
    token: str  # the number as printed inside the quote
    unit: str  # unit of this rendering (units.py symbol)
    unit_basis: str  # where the unit comes from
    block: tuple[str, str] | None = None  # quote must lie between these markers
    column: int | None = None  # table row: token is the n-th number after `label`
    label: str = ""


class RenderingNotFound(ValueError):
    """A quote is not on its page (or not inside its block)."""


class ValueNotSupported(ValueError):
    """Renderings do not agree across at least two documents."""


_NUMBER = re.compile(r"-?[0-9OoIl.,_?]+(?:E\s?[+-]?[0-9A-Za-z]+)?")


def normalise(text: str) -> str:
    return " ".join(text.split())


def read_number(token: str) -> float | None:
    """Parse an OCR number token with unambiguous repairs only; None if unreadable."""
    t = token.strip()
    t = re.sub(r"E\s+(?=[0-9Oo])", "E+", t)
    t = t.translate(str.maketrans("OolI", "0011"))
    if not re.fullmatch(r"-?(\d+\.?\d*|\.\d+)(E[+-]\d+)?", t):
        return None
    return float(t)


def _check(pages: dict[tuple[str, int], str], r: Rendering) -> None:
    text = normalise(pages.get((r.document, r.page), ""))
    if r.block is not None:
        start = text.find(r.block[0])
        end = text.find(r.block[1], start + 1) if start >= 0 else -1
        if start < 0 or end < 0:
            raise RenderingNotFound(f"{r.document} p.{r.page}: block {r.block} not found")
        text = text[start:end]
    if normalise(r.quote) not in text:
        raise RenderingNotFound(f"{r.document} p.{r.page}: quote not found: {r.quote!r}")
    if r.column is None:
        if not r.quote.endswith(r.token) and f"{r.token} " not in r.quote:
            raise RenderingNotFound(f"{r.document} p.{r.page}: {r.token!r} not in quote")
        return
    if not r.quote.startswith(r.label):
        raise RenderingNotFound(f"{r.document} p.{r.page}: row label {r.label!r} not at start")
    cells = _NUMBER.findall(r.quote[len(r.label) :])
    if r.column >= len(cells) or cells[r.column] != r.token:
        raise RenderingNotFound(
            f"{r.document} p.{r.page}: column {r.column} of {r.label!r} is not {r.token!r}"
        )


# -- constituent schema ----------------------------------------------------------------------


class ConstituentProperty(FrozenModel):
    name: str  # e.g. "axial_modulus"
    symbol: str  # as in the source, e.g. "Ef11"
    value: float
    unit: str
    evidence_type: EvidenceType
    data_quality: str
    conditions: dict[str, Any] = Field(default_factory=dict)
    equation: str | None = None  # for derived values
    provenance: Provenance

    def value_in(self, unit: str) -> float:
        return convert(self.value, self.unit, unit)


class Constituent(FrozenModel):
    constituent_id: str  # the data-bank code, e.g. "AS--"
    name: str
    role: str  # "fiber" / "matrix"
    description: str
    properties: dict[str, ConstituentProperty]

    def get(self, name: str) -> ConstituentProperty:
        try:
            return self.properties[name]
        except KeyError:
            raise KeyError(f"{self.constituent_id} has no sourced {name!r}") from None


@dataclass(frozen=True)
class ValueSpec:
    name: str
    symbol: str
    unit: str  # unit the accepted value is stored in
    renderings: tuple[Rendering, ...]


def accept(pages: dict[tuple[str, int], str], spec: ValueSpec) -> tuple[float, dict[str, Any]]:
    """Verify every rendering, then apply the agreement rule. Returns (value, audit)."""
    readings: list[tuple[Rendering, float | None]] = []
    for r in spec.renderings:
        _check(pages, r)
        raw = read_number(r.token)
        readings.append((r, None if raw is None else convert(raw, r.unit, spec.unit)))
    groups: dict[float, list[Rendering]] = {}
    for r, v in readings:
        if v is not None:
            key = next((k for k in groups if abs(k - v) <= 1e-9 * max(abs(k), 1e-300)), v)
            groups.setdefault(key, []).append(r)
    if not groups:
        raise ValueNotSupported(f"{spec.symbol}: no readable rendering")
    value, agreeing = max(groups.items(), key=lambda kv: len(kv[1]))
    conflicting = [r for k, rs in groups.items() if k != value for r in rs]
    documents = {r.document for r in agreeing}
    if len(documents) < 2 or len(agreeing) <= len(conflicting):
        raise ValueNotSupported(
            f"{spec.symbol}: {len(agreeing)} agreeing renderings from {sorted(documents)}, "
            f"{len(conflicting)} conflicting"
        )

    def cite(r: Rendering) -> str:
        return f"{r.document} p.{r.page}: {r.token!r}"

    audit = {
        "agreeing": [cite(r) for r in agreeing],
        "conflicting": [cite(r) for r in conflicting],
        "unreadable": [cite(r) for r, v in readings if v is None],
        "rule": ">=2 documents agree and agreeing > conflicting renderings",
    }
    return value, audit


def _prop(
    pages: dict[tuple[str, int], str], spec: ValueSpec, conditions: dict[str, Any]
) -> ConstituentProperty:
    value, audit = accept(pages, spec)
    agreeing = [
        r for r in spec.renderings if f"{r.document} p.{r.page}: {r.token!r}" in audit["agreeing"]
    ]
    return ConstituentProperty(
        name=spec.name,
        symbol=spec.symbol,
        value=value,
        unit=spec.unit,
        evidence_type=EvidenceType.DATASHEET,
        data_quality=DATA_QUALITY,
        conditions=conditions,
        provenance=Provenance(
            id="prov_" + stable_digest(["constituent", spec.symbol, spec.name, value])[:16],
            sources=tuple(
                SourceRef(
                    kind="report",
                    identifier=r.document,
                    uri=DOCUMENTS[r.document],
                    metadata={
                        "page": r.page,
                        "quote": r.quote,
                        "token": r.token,
                        "unit": r.unit,
                        "unit_basis": r.unit_basis,
                        "retrieved": RETRIEVED,
                    },
                )
                for r in agreeing
            ),
            metadata={"extractor": EXTRACTOR_VERSION, **audit},
        ),
    )


# -- the one example system: AS graphite fibre / IMLS epoxy ---------------------------------

_B3290_AS = ("AS-- GRAPHITE FIBER.", "SGLA S- GLASS FIBER.")
_E3290_AS = ('PRI"ARY FIBER PROPERTIESjAS-- FIBER', 'PRI"ARY "ATRIX PROPERTIES')
_E3290_IMLS = ('PRI"ARY "ATRIX PROPERTIESjI"LS', "DIFFUSIVITY DIF")
_E2515_AS = ("PRIMARY FIBER PROPERTIES; AS-- FIBER", "PRIMARY MATRIX PROPERTIES")
_E2515_IMLS = ("PRIMARY MATRIX PROPERTIES; IMLS MATRIX", "Item 5(b)")
_B2515_AS = ("FP 10000 0.300E-03 0.630E-01", "SGLA")
_B2515_IMLS = ("IMLS INTERMEDIATE MODULUS LOH STRENGTH MATRIX.", "IMHS")
_FE_ROW = "FE 0.310E 08 0.200E 07 0.200E O0 0.250E O0 O.200E 07 0.100E 07"
_DB = "explicit in quote"
_LAYOUT = "TP-2515 p.56 card layout 'FE; Ef11, Ef22, nu12, nu23, Gf12, Gf23'; " + UNIT_BASIS_ICAN


def _r(doc: str, page: int, quote: str, token: str, unit: str, basis: str, **kw: Any) -> Rendering:
    return Rendering(doc, page, quote, token, unit, basis, **kw)


AS_SPECS = (
    ValueSpec("density", "Rhof", "lb/in^3", (
        _r(TP3290, 27, "Weight density Rhof 0.630E-01 lb/in**3", "0.630E-01", "lb/in^3", _DB,
           block=_B3290_AS),
        _r(TP3290, 40, "DENSITY RHOFP 0.6300E-01", "0.6300E-01", "lb/in^3", UNIT_BASIS_ICAN,
           block=_E3290_AS),
        _r(TP2515, 61, "DENSITY RHOFP 0.6300E-01", "0.6300E-01", "lb/in^3", UNIT_BASIS_ICAN,
           block=_E2515_AS),
        _r(TP2515, 76, "FP 10000 0.300E-03 0.630E-01", "0.630E-01", "lb/in^3", UNIT_BASIS_ICAN,
           label="FP", column=2),
    )),
    ValueSpec("axial_modulus", "Ef11", "psi", (
        _r(TP3290, 27, "Normal moduli (11) Ef11 0.310E+OS psi", "0.310E+OS", "psi", _DB,
           block=_B3290_AS),  # unreadable: 'S' may be 5 or 8
        _r(TP3290, 40, "EFP1 0.3100E+08", "0.3100E+08", "psi", UNIT_BASIS_ICAN, block=_E3290_AS),
        _r(TP2515, 61, "EFPI 0.3100E 08", "0.3100E 08", "psi", UNIT_BASIS_ICAN, block=_E2515_AS),
        _r(TP2515, 76, _FE_ROW, "0.310E 08", "psi", _LAYOUT, label="FE", column=0,
           block=_B2515_AS),
    )),
    ValueSpec("transverse_modulus", "Ef22", "psi", (
        _r(TP3290, 27, "Normal moduli (22) Ef22 0.200E+07 psi", "0.200E+07", "psi", _DB,
           block=_B3290_AS),
        _r(TP3290, 40, "EFP2 0.2000E+07", "0.2000E+07", "psi", UNIT_BASIS_ICAN, block=_E3290_AS),
        _r(TP2515, 61, "EFP2 O.2000E 07", "O.2000E 07", "psi", UNIT_BASIS_ICAN, block=_E2515_AS),
        _r(TP2515, 76, _FE_ROW, "0.200E 07", "psi", _LAYOUT, label="FE", column=1,
           block=_B2515_AS),
    )),
    ValueSpec("inplane_shear_modulus", "Gf12", "psi", (
        _r(TP3290, 27, "Shear moduli (12) Gfl2 0.200E+07 psi", "0.200E+07", "psi", _DB,
           block=_B3290_AS),
        _r(TP3290, 40, "GFP12 0.2000E+07", "0.2000E+07", "psi", UNIT_BASIS_ICAN, block=_E3290_AS),
        _r(TP2515, 61, "GFPI2 0,2000E 07", "0,2000E 07", "psi", UNIT_BASIS_ICAN,
           block=_E2515_AS),  # unreadable: comma
        _r(TP2515, 76, _FE_ROW, "O.200E 07", "psi", _LAYOUT, label="FE", column=4,
           block=_B2515_AS),
    )),
    ValueSpec("major_poisson_ratio", "nuf12", "1", (
        _r(TP3290, 27, 'Poisson"s ratio (12) Nufl2 0.200E+00 non-dim', "0.200E+00", "1", _DB,
           block=_B3290_AS),
        _r(TP3290, 40, "NUFP12 0.2000E+00", "0.2000E+00", "1", UNIT_BASIS_ICAN, block=_E3290_AS),
        _r(TP2515, 61, "NUFPI2 0.200OE 00", "0.200OE 00", "1", UNIT_BASIS_ICAN, block=_E2515_AS),
        _r(TP2515, 76, _FE_ROW, "0.200E O0", "1", _LAYOUT, label="FE", column=2,
           block=_B2515_AS),
    )),
    ValueSpec("tensile_strength", "SfT", "psi", (
        _r(TP3290, 27, "Fiber tensile strength SfT 0.400E+06 psi", "0.400E+06", "psi", _DB,
           block=_B3290_AS),
        _r(TP3290, 40, "SFPT 0.4000E+06", "0.4000E+06", "psi", UNIT_BASIS_ICAN, block=_E3290_AS),
        _r(TP2515, 61, "SFPT 0.4000E 05", "0.4000E 05", "psi", UNIT_BASIS_ICAN,
           block=_E2515_AS),  # conflicts (reads 40 ksi); kept in the audit
        _r(TP2515, 76, "FS 0.400E 06 0.400E 06", "0.400E 06", "psi", _LAYOUT, label="FS",
           column=0, block=_B2515_AS),
    )),
)  # fmt: skip

_T2 = "TM-83320 Table 2 header 'LM IMLS IMHS HM Poly-imide PMR': IMLS is column 1"

IMLS_SPECS = (
    ValueSpec("density", "Rhom", "lb/in^3", (
        _r(TP3290, 40, 'DENSITY RHO"P 0.4600E-01', "0.4600E-01", "lb/in^3", UNIT_BASIS_ICAN,
           block=_E3290_IMLS),
        _r(TP2515, 61, "DE.USITY RHO_]P 0 ._600E-01", "0 ._600E-01", "lb/in^3",
           UNIT_BASIS_ICAN, block=_E2515_IMLS),  # unreadable
        _r(TP2515, 76, "lIP 0.460E-01", "0.460E-01", "lb/in^3", UNIT_BASIS_ICAN,
           block=_B2515_IMLS),
        _r(TM83320, 14, "Uensity Pm lb/_n3 0.046 0.046 0.044 0.045 0.044 0.044", "0.046",
           "lb/in^3", "row unit 'lb/_n3' (lb/in3); " + _T2, label="Uensity Pm lb/_n3",
           column=1),
    )),
    ValueSpec("modulus", "Em", "psi", (
        _r(TP3290, 40, 'ELASTIC "ODULUS E"P 0.5000E+06', "0.5000E+06", "psi", UNIT_BASIS_ICAN,
           block=_E3290_IMLS),
        _r(TP2515, 61, "ELASTIC MODULUS EMP 0.5000E 06", "0.5000E 06", "psi", UNIT_BASIS_ICAN,
           block=_E2515_IMLS),
        _r(TP2515, 76, "ME O.500E 06", "O.500E 06", "psi", UNIT_BASIS_ICAN,
           block=_B2515_IMLS),
        # TM-83320 Table 2 prints this row as "0.3 _ 0.50 0.50 0.75 0.50 0.47": whether "_"
        # is part of the LM value or a cell is ambiguous, so the row is not used.
    )),
    ValueSpec("poisson_ratio", "num", "1", (
        _r(TP3290, 40, 'NU"P 0.4100E+00', "0.4100E+00", "1", UNIT_BASIS_ICAN,
           block=_E3290_IMLS),
        _r(TP2515, 61, "HU:]P 0.4100E 00", "0.4100E 00", "1", UNIT_BASIS_ICAN,
           block=_E2515_IMLS),
        _r(TM83320, 14, "Poissons's ratio v - 0.43 0.41 0.35 0.35 0.35 0.36", "0.41", "1", _T2,
           label="Poissons's ratio v -", column=1),
    )),
)  # fmt: skip

DRY_RT = {"state": "dry", "temperature": "room temperature (ICAN 'DRY RT. PROPERTIES')"}


def load_pages(path: Path = RAW) -> dict[tuple[str, int], str]:
    pages: dict[tuple[str, int], str] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        pages[(row["document"], row["page"])] = row["text"]
    return pages


def build_constituents(pages: dict[tuple[str, int], str]) -> dict[str, Constituent]:
    fiber = Constituent(
        constituent_id="AS--",
        name="AS graphite (carbon) fibre",
        role="fiber",
        description="NASA ICAN data-bank fibre 'AS-- GRAPHITE FIBER' (transversely isotropic)",
        properties={s.name: _prop(pages, s, DRY_RT) for s in AS_SPECS},
    )
    matrix_props = {s.name: _prop(pages, s, DRY_RT) for s in IMLS_SPECS}
    em, nu = matrix_props["modulus"], matrix_props["poisson_ratio"]
    gm = em.value / (2 * (1 + nu.value))
    matrix_props["shear_modulus"] = ConstituentProperty(
        name="shear_modulus",
        symbol="Gm",
        value=gm,
        unit=em.unit,
        evidence_type=EvidenceType.DERIVED,
        data_quality=DATA_QUALITY,
        conditions=DRY_RT,
        equation="Gm = Em / (2 (1 + num))  (isotropic matrix; NASA TM-83320 section 4.0)",
        provenance=Provenance(
            id="prov_" + stable_digest(["constituent", "Gm", gm])[:16],
            sources=(
                SourceRef(
                    kind="report",
                    identifier=TM83320,
                    uri=DOCUMENTS[TM83320],
                    metadata={
                        "page": 5,
                        "quote": "Gm = Em/2 (1 + _m) since the matrix is assumed to be isotropic",
                    },
                ),
            ),
            metadata={"derived_from": [em.provenance.id, nu.provenance.id]},
        ),
    )
    matrix = Constituent(
        constituent_id="IMLS",
        name="IMLS epoxy matrix (intermediate modulus, low strength)",
        role="matrix",
        description="NASA ICAN data-bank matrix 'IMLS', an epoxy-type resin (TP-2515 p.56)",
        properties=matrix_props,
    )
    return {fiber.constituent_id: fiber, matrix.constituent_id: matrix}


@lru_cache(maxsize=1)
def constituent_library() -> dict[str, Constituent]:
    """AS fibre and IMLS matrix, verified against the committed raw text (offline)."""
    return build_constituents(load_pages())
