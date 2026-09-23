"""Parser for MIL-HDBK-5J "Design Mechanical and Physical Properties" tables.

Input is the text layer of one handbook page, extracted by
:func:`extract_pages` (pypdf) and stored under ``data/raw/mil_hdbk_5j/``. The
parser is deliberately **strict**. It understands only the column-major
layout that pypdf produces for most tables (row labels first, then every
column's values in row order), and it rejects a table instead of guessing
whenever:

* the number of value tokens is not ``rows × columns`` (``columns`` comes from
  the ``Basis`` header row);
* a value token is not a number, ``...`` or a number with footnote letters;
* the page uses the letter-spaced row-major layout (e.g. ``3 2 3 2``), whose
  digit grouping is ambiguous.

Rejected tables are reported with their reason; they are never partially
ingested.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

PARSER_VERSION = "mil-hdbk-5j-parser/1"

TABLE_RE = re.compile(
    r"Table (\d+\.\d+\.\d+\.\d+\([a-z]\d*\))\.\s+Design Mechanical and Physical Properties of"
    r"\s+(?P<title>.+?)(?:\n(?:S p e c|Specification))",
    re.DOTALL,
)
VALUE_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)(?P<notes>[a-z]*)$")
GROUPS = {
    "Ftu": "Ftu",
    "Fty": "Fty",
    "Fcy": "Fcy",
    "Fsu": "Fsu",
    "Fbru": "Fbru",
    "Fbry": "Fbry",
    "e": "e",
}
DIRECTIONS = {"L", "LT", "ST"}


class TableRejected(ValueError):
    """The page does not contain a table this parser can read unambiguously."""


@dataclass
class ParsedTable:
    table_id: str
    page: int
    title: str
    header: dict[str, str]  # specification, form, temper/condition (as printed)
    basis: list[str]  # one entry per value column
    rows: list[str]  # e.g. "Ftu:L", "Fsu", "e:LT"
    values: dict[str, list[str | None]]  # row → one raw token (or None) per column
    physical: dict[str, list[str]]  # "E", "Ec", "G", "mu" → groups; "density_lb_in3" → [..]
    footnotes: list[str] = field(default_factory=list)


def _squash(s: str) -> str:
    """Undo letter spacing ('S p e c' → 'Spec') and drop dot leaders."""
    s = re.sub(r"\.{2,}", " ", s)
    return re.sub(r"\s+", "", s)


def find_table(text: str) -> tuple[str, str] | None:
    m = TABLE_RE.search(text)
    if not m:
        return None
    return m.group(1), " ".join(m.group("title").split())


def parse_page(text: str, page: int) -> ParsedTable:
    found = find_table(text)
    if found is None:
        raise TableRejected("no design-properties table on this page")
    table_id, title = found
    lines = [ln.rstrip() for ln in text.splitlines()]
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("Mechanical Properties"))
    except StopIteration:
        raise TableRejected("no 'Mechanical Properties' block") from None

    if any(_squash(ln).startswith("Alloy") for ln in lines[:start]):
        raise TableRejected("multi-alloy table (one alloy per column); not mapped per column")
    header = _header(lines[:start])
    basis = _basis(lines[:start])
    # Row labels run until the first value token.
    rows: list[str] = []
    group: str | None = None
    i = start + 1
    while i < len(lines):
        raw = lines[i]
        squashed = _squash(raw)
        if _is_value(raw.strip()):
            break
        if re.search(r"\d\s\d", raw) and ".." in raw:
            raise TableRejected("letter-spaced row-major layout (ambiguous digit grouping)")
        head = re.match(r"^(Ftu|Fty|Fcy|Fsu|Fbru|Fbry|e)\b", raw.strip())
        other = re.match(r"^([A-Z][A-Za-z]*)\s*,.*:\s*$", raw.strip())
        if head:
            group = GROUPS[head.group(1)]
            if group == "Fsu" and ".." in raw:
                rows.append("Fsu")
        elif other:
            # an unrecognised row group (e.g. "RA, percent (S basis):"): keep its rows
            # under their own name so they can never be mistaken for a known property
            group = other.group(1)
        elif re.match(r"^(L|LT|ST)$", squashed.rstrip(":")) and group:
            rows.append(f"{group}:{squashed.rstrip(':')}")
        elif squashed.startswith("(e/D=") and group:
            rows.append(f"{group}:{squashed.split(')')[0] + ')'}")
        i += 1
    if not rows:
        raise TableRejected("no row labels found")
    duplicates = sorted({r for r in rows if rows.count(r) > 1})
    if duplicates:
        raise TableRejected(f"ambiguous duplicate row labels {duplicates}")

    tokens: list[str] = []
    while i < len(lines) and not lines[i].startswith("E, 10"):
        for tok in lines[i].split():
            # a lone footnote letter printed on its own line belongs to the preceding number
            if re.fullmatch(r"[a-h]", tok) and tokens and VALUE_RE.match(tokens[-1]):
                tokens[-1] += tok
            else:
                tokens.append(tok)
        i += 1
    if i >= len(lines):
        raise TableRejected("no modulus block after the values")
    n_rows, n_cols = len(rows), len(basis)
    if len(tokens) != n_rows * n_cols:
        raise TableRejected(f"{len(tokens)} value tokens ≠ {n_rows} rows × {n_cols} columns")
    for t in tokens:
        if t != "..." and not VALUE_RE.match(t):
            raise TableRejected(f"unreadable value token {t!r}")
    values = {
        row: [
            None if tokens[c * n_rows + r] == "..." else tokens[c * n_rows + r]
            for c in range(n_cols)
        ]
        for r, row in enumerate(rows)
    }
    _check_plausible(table_id, rows, values, len(basis))
    physical = _physical(lines[i:])
    footnotes = [ln for ln in lines if re.match(r"^[a-h] [A-Z]", ln)]
    return ParsedTable(table_id, page, title, header, basis, rows, values, physical, footnotes)


def number(token: str | None) -> float | None:
    """The numeric part of a value token ('70b' → 70.0); None for an empty cell."""
    if token is None:
        return None
    m = VALUE_RE.match(token)
    return float(m.group("num")) if m else None


def _check_plausible(
    table_id: str, rows: list[str], values: dict[str, list[str | None]], n_cols: int
) -> None:
    """Reject tables whose parsed columns violate basic physics (a sign of misalignment)."""
    for c in range(n_cols):
        for d in ("L", "LT", "ST"):
            tu, ty = (number(values.get(f"{g}:{d}", [None] * n_cols)[c]) for g in ("Ftu", "Fty"))
            if tu is not None and ty is not None and ty > tu:
                raise TableRejected(f"column {c}: Fty({d}) {ty} > Ftu({d}) {tu}")
        su = number(values.get("Fsu", [None] * n_cols)[c])
        tus = [number(values[r][c]) for r in rows if r.startswith("Ftu:")]
        tus_known = [t for t in tus if t is not None]
        if su is not None and tus_known and su >= max(tus_known):
            raise TableRejected(f"column {c}: Fsu {su} >= Ftu {max(tus_known)}")
        for r in rows:
            if r.startswith("e:"):
                e = number(values[r][c])
                if e is not None and not 0 < e < 100:
                    raise TableRejected(f"column {c}: elongation {e}%")


def _is_value(s: str) -> bool:
    return s == "..." or bool(VALUE_RE.match(s))


def unspace(text: str) -> str:
    """Undo pypdf letter spacing: 'S o l u t i o n  t r e a t e d' → 'Solution treated'."""
    words = re.split(r"\s{2,}", text.strip())
    out = []
    for w in words:
        parts = w.split(" ")
        out.append("".join(parts) if parts and all(len(x) <= 1 for x in parts) else w)
    return " ".join(x for x in out if x)


def _header(lines: Iterable[str]) -> dict[str, str]:
    """Specification / form / temper / condition, as printed (after the dot leader)."""
    out: dict[str, str] = {}
    for ln in lines:
        m = re.match(r"^(?P<label>[^.]+?)\s*\.{2,}\s*(?P<value>.*)$", ln)
        if not m:
            continue
        label = _squash(m.group("label")).rstrip(",").lower()
        for name in ("specification", "form", "temper", "condition"):
            if label.startswith(name):
                out[name] = unspace(m.group("value"))
    return out


def _basis(lines: Iterable[str]) -> list[str]:
    """Statistical basis per value column (A, B or S), which also gives the column count."""
    for ln in lines:
        if _squash(ln).startswith("Basis"):
            # footnote marks ("S a Sa Sa", "Sc,d") are lowercase letters/commas; the
            # rows × columns count and the plausibility checks still guard the result
            tail = re.sub(r"[a-z,]", "", _squash(ln)[len("Basis") :])
            if tail and set(tail) <= {"A", "B", "S"}:
                return list(tail)
            raise TableRejected(f"unreadable Basis row {ln!r}")
    raise TableRejected("no Basis header row")


def _physical(lines: list[str]) -> dict[str, list[str]]:
    labels: list[str] = []
    nums: list[str] = []
    density: list[str] = []
    in_density = False
    for ln in lines:
        s = ln.strip()
        sq = _squash(s)
        if sq.startswith("E,10"):
            labels.append("E")
        elif sq.startswith("Ec,10"):
            labels.append("Ec")
        elif sq.startswith("G,10"):
            labels.append("G")
        elif s.startswith("µ"):
            labels.append("mu")
        elif "lb/in" in s:
            in_density = True
        elif in_density and re.fullmatch(r"\d?\.\d+", s):
            density.append(s)
            in_density = False
        elif labels and not in_density and re.fullmatch(r"\d+(\.\d+)?", s) and not density:
            nums.append(s)
    out: dict[str, list[str]] = {}
    if labels and nums and len(nums) % len(labels) == 0:
        groups = len(nums) // len(labels)
        for k, lab in enumerate(labels):
            out[lab] = [nums[g * len(labels) + k] for g in range(groups)]
    if density:
        out["density_lb_in3"] = density
    return out


def summarize(table: ParsedTable) -> dict[str, Any]:
    return {
        "table": table.table_id,
        "page": table.page,
        "title": table.title,
        "rows": table.rows,
        "columns": len(table.basis),
        "physical": table.physical,
    }


# -- extraction and record building ---------------------------------------------------

SOURCE_ID = "MIL-HDBK-5J"
SOURCE_URI = "https://archive.org/details/milhdbk-5-j"
SOURCE_VERSION = "31 January 2003"
ROW_PROPERTY = {
    "Ftu": "tensile_strength",
    "Fty": "yield_strength",
    "Fcy": "compressive_yield_strength",
    "Fsu": "shear_strength",
    "e": "elongation_at_break",
}
SYSTEM_BY_CHAPTER = {
    "2": "steel",
    "3": "aluminum",
    "4": "magnesium",
    "5": "titanium",
    "6": "heat-resistant alloy",
    "7": "other metal",
}


def deterministic_id(*parts: Any) -> str:
    """Provenance id derived from the source cell, so rebuilds are bit-for-bit identical."""
    from jevpilot import stable_digest

    return "prov_" + stable_digest(list(parts))[:16]


def retrieved_at(date: str) -> Any:
    from datetime import UTC, datetime

    return datetime.fromisoformat(date).replace(tzinfo=UTC)


def extract_pages(pdf_path: str) -> list[dict[str, Any]]:
    """Text layer of every page, via pypdf (optional dependency: ``pip install '.[data]'``)."""
    import importlib

    pypdf: Any = importlib.import_module("pypdf")
    reader = pypdf.PdfReader(pdf_path)
    return [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(reader.pages)]


def parse_all(pages: list[dict[str, Any]]) -> tuple[list[ParsedTable], list[dict[str, Any]]]:
    accepted: list[ParsedTable] = []
    rejected: list[dict[str, Any]] = []
    for p in pages:
        found = find_table(p["text"])
        if found is None:
            continue
        try:
            accepted.append(parse_page(p["text"], p["page"]))
        except TableRejected as exc:
            rejected.append(
                {"table": found[0], "page": p["page"], "title": found[1], "reason": str(exc)}
            )
    return accepted, rejected


def group_key(table_id: str) -> str:
    """'3.7.6.0(b2)' → '3.7.6.0(b)': the handbook's own table grouping across pages."""
    return re.sub(r"\(([a-z])\d*\)", r"(\1)", table_id)


_FORM_WORDS = (
    r"(Sheet|Plate|Bar|Rod|Extrusion|Extruded|Die|Hand|Forging|Investment|Mechanical|Strip"
    r"|Tubing|Alloy|Titanium)"
)


def describe_title(title: str, chapter: str) -> tuple[str, bool, str]:
    """(designation, clad, product form) from a table title."""
    t = re.sub(r"[—-]\s*(Continued|Concluded)\s*$", "", title).strip()
    t = t.replace("Alu- minum", "Aluminum").replace("Alumi- num", "Aluminum")
    clad = bool(re.match(r"^(Clad|Alclad)\b", t))
    t = re.sub(r"^(Clad|Alclad)\s+", "", t)
    if chapter == "3":
        m = re.match(r"^(\d{4}(?:/\d{4})?)(?:-(T\w+))?\s+(?:Aluminum\s+)?(?:Alloy\s+)?(.*)$", t)
        if m:
            return m.group(1), clad, m.group(3).strip(" ,;")
    if t.startswith("Commercially Pure"):
        return "commercially pure", clad, ""
    m = re.match(r"^(.+?)\s+" + _FORM_WORDS + r"\b(.*)$", t)
    if m:
        return m.group(1).strip(), clad, (m.group(2) + m.group(3)).strip(" ,;")
    return t, clad, ""


def _tempers(header: dict[str, str], chapter: str) -> tuple[str, ...]:
    """Aluminium tempers (T6, T651, ...). Other alloys keep their condition text in metadata."""
    if chapter != "3":
        return ()
    text = (header.get("temper") or "").replace(" ", "")
    return tuple(dict.fromkeys(re.findall(r"T\d+|(?<![A-Za-z])[OF](?![A-Za-z])", text)))


def build_records(tables: list[ParsedTable], retrieved: str) -> list[Any]:
    """One MaterialRecord per handbook table group; one Measurement per table cell."""
    from domains.uav_materials.identity import MaterialIdentity, generated_aliases
    from domains.uav_materials.schema import (
        MaterialFamily,
        MaterialRecord,
        Measurement,
        MeasurementBasis,
        ProvenanceStatus,
        StatisticalBasis,
        TemperatureRegime,
        TestConditions,
    )
    from jevpilot import Provenance, SourceRef

    groups: dict[str, list[ParsedTable]] = {}
    for t in tables:
        groups.setdefault(group_key(t.table_id), []).append(t)

    records: list[Any] = []
    for key, members in sorted(groups.items()):
        first = members[0]
        chapter = key.split(".")[0]
        designation, clad, form = describe_title(first.title, chapter)
        tempers = _tempers(first.header, chapter)
        in_title = re.search(r"\d{4}-(T\w+)", first.title)
        if chapter == "3" and in_title and in_title.group(1) not in tempers:
            tempers = (in_title.group(1), *tempers)
        system = SYSTEM_BY_CHAPTER.get(chapter, "metal")
        measurements: dict[str, list[Measurement]] = {
            g: [] for g in ("density", "strength", "deformation")
        }

        def add(
            group: str,
            prop: str,
            value: float,
            unit: str,
            table: ParsedTable,
            column: int | None,
            row: str,
            raw: str,
            basis: StatisticalBasis | None,
            specimen: str,
            store: dict[str, list[Measurement]] = measurements,
        ) -> None:
            ref = SourceRef(
                kind="handbook",
                identifier=SOURCE_ID,
                version=SOURCE_VERSION,
                uri=SOURCE_URI,
                metadata={
                    "table": table.table_id,
                    "page": table.page,
                    "column": None if column is None else column + 1,
                    "columns": len(table.basis),
                    "row": row,
                    "raw_token": raw,
                    "retrieved": retrieved,
                },
            )
            store[group].append(
                Measurement(
                    property=prop,
                    value=value,
                    unit=unit,
                    basis=MeasurementBasis.SPECIFIED,
                    statistical_basis=basis,
                    conditions=TestConditions(
                        temperature_regime=TemperatureRegime.ROOM, specimen=specimen
                    ),
                    test_method="MIL-HDBK-5J design allowable (see handbook Section 1)",
                    provenance=Provenance(
                        id=deterministic_id(SOURCE_ID, table.table_id, column, row, raw, prop),
                        created_at=retrieved_at(retrieved),
                        sources=(ref,),
                        metadata={"parser": PARSER_VERSION, "identity_match": "exact"},
                    ),
                    provenance_status=ProvenanceStatus.SOURCED,
                    notes="room-temperature design allowable; handbook states no numeric test "
                    "temperature for the table",
                )
            )

        for t in members:
            page_says_e_s_basis = True  # MIL-HDBK-5J lists elongation as S-basis
            for row in t.rows:
                group, _, direction = row.partition(":")
                prop = ROW_PROPERTY.get(group)
                if prop is None:
                    continue  # bearing allowables are not part of the property registry
                for c, token in enumerate(t.values[row]):
                    v = number(token)
                    if v is None or token is None:
                        continue
                    if prop == "elongation_at_break":
                        basis = StatisticalBasis.S if page_says_e_s_basis else None
                        unit = "%"
                    else:
                        basis = StatisticalBasis(t.basis[c])
                        unit = "ksi"
                    where = f"{direction} direction; " if direction else ""
                    add(
                        "deformation" if prop == "elongation_at_break" else "strength",
                        prop,
                        v,
                        unit,
                        t,
                        c,
                        row,
                        token,
                        basis,
                        f"{where}{t.table_id} column {c + 1}/{len(t.basis)}",
                    )
            for label, prop in (("E", "youngs_modulus"), ("G", "shear_modulus")):
                vals = t.physical.get(label, [])
                distinct = list(dict.fromkeys(vals))
                for k, raw in enumerate(distinct):
                    spec = (
                        ""
                        if len(distinct) == 1
                        else f"value {k + 1} of {len(distinct)}; the extracted table text does "
                        "not state which columns it applies to"
                    )
                    add(
                        "deformation",
                        prop,
                        float(raw),
                        "Msi",
                        t,
                        None,
                        label,
                        raw,
                        StatisticalBasis.TYPICAL,
                        spec or f"{t.table_id}",
                    )
            for raw in dict.fromkeys(t.physical.get("density_lb_in3", [])):
                add(
                    "density",
                    "density",
                    float(raw),
                    "lb/in^3",
                    t,
                    None,
                    "density",
                    raw,
                    StatisticalBasis.TYPICAL,
                    t.table_id,
                )

        noun = {"3": "aluminum alloy", "5": "titanium"}.get(chapter, "")
        if chapter == "5" and designation.startswith("Ti-"):
            noun = ""
        name = " ".join(
            p for p in (("Clad " if clad else "") + designation, noun, form.lower()) if p
        ).strip()
        condition = first.header.get("temper") or first.header.get("condition", "")
        if tempers:
            name += f" ({', '.join(tempers)})"
        elif condition and len(condition) <= 40 and " " not in condition.strip():
            name += f" ({condition.strip()})"  # compact codes such as TH04
        identity = MaterialIdentity(
            canonical_id=f"mil5j:{key}",
            canonical_name=name,
            system=system,
            designation=designation,
            clad=clad,
            tempers=tempers,
            product_form=form,
            aliases=generated_aliases(system, designation, clad, tempers),
            source_names={SOURCE_ID: first.title},
            source_ids={SOURCE_ID: [t.table_id for t in members]},
        )
        records.append(
            MaterialRecord(
                material_id=f"mil5j-{key.replace('(', '-').replace(')', '')}",
                name=name,
                family=MaterialFamily.METAL,
                identity=identity,
                density=tuple(measurements["density"]),
                strength=tuple(measurements["strength"]),
                deformation=tuple(measurements["deformation"]),
                provenance=Provenance(
                    id=deterministic_id(SOURCE_ID, key),
                    created_at=retrieved_at(retrieved),
                    sources=(
                        SourceRef(
                            kind="handbook",
                            identifier=SOURCE_ID,
                            version=SOURCE_VERSION,
                            uri=SOURCE_URI,
                            metadata={
                                "tables": [t.table_id for t in members],
                                "pages": [t.page for t in members],
                            },
                        ),
                    ),
                ),
                provenance_status=ProvenanceStatus.SOURCED,
                metadata={
                    "system": system,
                    "product_form": form,
                    "specification": first.header.get("specification", ""),
                    "temper_or_condition_as_printed": first.header.get("temper")
                    or first.header.get("condition", ""),
                    "footnotes": {t.table_id: t.footnotes for t in members if t.footnotes},
                },
            )
        )
    names = [r.name for r in records]
    return [
        r.model_copy(update={"name": f"{r.name} [{r.identity.source_ids[SOURCE_ID][0]}]"})
        if names.count(r.name) > 1
        else r
        for r in records
    ]
