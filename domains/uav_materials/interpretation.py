"""Natural-language request → validated engineering requirements.

An interpreter (an LLM behind an adapter, or the deterministic offline
parser below) only *translates the user's words* into the structured schema
:class:`InterpretationPayload`. It never supplies material data: the schema
has no field for a material or a material property, and every item is checked
before it becomes an :class:`~domains.uav_materials.requirements.EngineeringRequirement`:

* ``source_text`` must be a verbatim quote from the request;
* every number (limit, exposure time) must be written in that quote;
* the unit must be written in that quote (so "40 GPa" cannot become 40 MPa);
* aspect, direction, operator and unit must pass the requirement validation
  (dimension check, required test conditions).

Items that fail are dropped and reported as integrity findings; nothing is
repaired silently. Soft preferences get equal numerical weights by a stated
convention (the interpreter never assigns weights).

Directions: ``laminate_x`` / ``laminate_y`` / ``laminate_xy`` are the in-plane
axes x, y and in-plane shear of the part or panel (laminate axes for a
laminate; for an isotropic metal every direction resolves to the same bulk
property). ``material_1`` / ``material_2`` are the fibre and transverse axes of a
single ply. Use ``unspecified`` when the user does not state a direction.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import Field, ValidationError

from domains.uav_materials.profile import Operator, Priority
from domains.uav_materials.requirements import (
    Aspect,
    Direction,
    EngineeringRequirement,
    EngineeringTarget,
)
from domains.uav_materials.schema import EnvironmentClass, TestConditions
from jevpilot import FrozenModel, Provenance, SourceRef

INTERPRETATION_PROMPT_VERSION = "uav-requirements-prompt/1"
EQUAL_WEIGHT_SOURCE = (
    "equal weights for the qualitative preferences stated in the request "
    "(interpretation convention, not from a design study)"
)

# canonical unit symbol → spellings accepted in the user's text (lower case)
UNIT_SPELLINGS: dict[str, tuple[str, ...]] = {
    "kg/m^3": ("kg/m^3", "kg/m3", "kg/m³", "kg m-3", "kg/m 3"),
    "g/cm^3": ("g/cm^3", "g/cm3", "g/cm³", "g/cc"),
    "lb/in^3": ("lb/in^3", "lb/in3", "lb/in³"),
    "GPa": ("gpa",),
    "MPa": ("mpa",),
    "ksi": ("ksi",),
    "Msi": ("msi",),
    "psi": ("psi",),
    "%": ("%", "percent"),
    "degC": ("°c", "degc", "deg c", "celsius", "º c", "°  c", "° c"),
    "degF": ("°f", "degf", "deg f", "fahrenheit"),
    "mm/year": ("mm/year", "mm/yr", "mm per year", "mm/a"),
    "h": ("h", "hours", "hour", "hrs"),
}


# -- the schema interpreters must produce ------------------------------------------------------


class RequirementItem(FrozenModel):
    """One requirement exactly as the user stated it."""

    aspect: Aspect
    direction: Direction = Direction.UNSPECIFIED
    operator: Operator
    value: float | None = None  # only for <=, >=, between: the number the user wrote
    upper: float | None = None  # only for between
    unit: str | None = None  # canonical unit symbol for the unit the user wrote
    priority: Priority
    environment: EnvironmentClass | None = None  # corrosion / water uptake exposure
    exposure_hours: float | None = None  # water uptake: the exposure time the user wrote
    source_text: str = Field(description="verbatim quote from the request")


class ConcernItem(FrozenModel):
    """A stated engineering concern without a limit (reported, never scored)."""

    aspect: Aspect
    environment: EnvironmentClass | None = None
    source_text: str = Field(description="verbatim quote from the request")
    note: str = ""


class InterpretationPayload(FrozenModel):
    """What an interpreter returns. Contains no material data by construction."""

    requirements: tuple[RequirementItem, ...] = ()
    concerns: tuple[ConcernItem, ...] = ()
    application: str = ""  # short description of the part/use, in the user's words
    design_allowed: bool = False  # the user asked for a design if nothing existing fits
    missing_information: tuple[str, ...] = ()


class IntegrityFinding(FrozenModel):
    item: str
    problem: str


class Concern(FrozenModel):
    aspect: Aspect
    environment: EnvironmentClass | None = None
    source_text: str
    note: str = ""


class InterpretationOutcome(FrozenModel):
    """Validated interpretation of one request (what the workflow uses)."""

    request_text: str
    interpreter: dict[str, Any]
    status: str  # ok | partial | empty | failed
    requirements: tuple[EngineeringRequirement, ...] = ()
    concerns: tuple[Concern, ...] = ()
    application: str = ""
    design_allowed: bool = False
    missing_information: tuple[str, ...] = ()
    findings: tuple[IntegrityFinding, ...] = ()
    error: str | None = None
    raw_output: str | None = None  # the interpreter's raw answer (audit), truncated

    @property
    def hard(self) -> tuple[EngineeringRequirement, ...]:
        return tuple(r for r in self.requirements if r.priority is Priority.HARD)

    @property
    def soft(self) -> tuple[EngineeringRequirement, ...]:
        return tuple(r for r in self.requirements if r.priority is Priority.SOFT)

    def target(self) -> EngineeringTarget | None:
        if not self.requirements:
            return None
        return EngineeringTarget(
            profile_id="interpreted-request",
            name="Engineering requirements interpreted from the user's request",
            application=self.application,
            requirements=self.requirements,
            metadata={"interpreter": self.interpreter},
        )


class RequirementInterpreter(Protocol):
    def interpret(self, request_text: str) -> InterpretationOutcome: ...

    def describe(self) -> dict[str, Any]: ...


# -- validation (shared by every interpreter) --------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


_NUMBER = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    return [float(m.group(0).replace(",", "")) for m in _NUMBER.finditer(text)]


def _stated(number: float, quote: str) -> bool:
    return any(abs(n - number) <= 1e-9 * max(1.0, abs(number)) for n in numbers_in(quote))


def _unit_stated(unit: str, quote: str) -> bool:
    q = _norm(quote)
    spellings = UNIT_SPELLINGS.get(unit, (unit.casefold(),))
    for s in spellings:
        if len(s) <= 2 and s.isalpha():  # short word units (h): whole-word match
            if re.search(rf"\d\s*{re.escape(s)}\b", q):
                return True
        elif s in q:
            return True
    return False


def parse_interpretation(
    raw: str | Mapping[str, Any],
    request_text: str,
    interpreter: Mapping[str, Any],
) -> InterpretationOutcome:
    """Validate an interpreter's answer against the request. Never raises for bad content."""
    raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    audit = raw_text[:4000]
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
        payload = InterpretationPayload.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        return InterpretationOutcome(
            request_text=request_text,
            interpreter=dict(interpreter),
            status="failed",
            error=f"interpreter output is not a valid interpretation: {type(exc).__name__}: "
            f"{str(exc)[:500]}",
            raw_output=audit,
        )

    request_norm = _norm(request_text)
    findings: list[IntegrityFinding] = []
    requirements: list[EngineeringRequirement] = []
    seen: set[tuple[Any, ...]] = set()
    for i, item in enumerate(payload.requirements):
        label = f"requirement {i + 1} ({item.aspect}, {item.operator})"
        problem = _check_item(item, request_norm)
        if problem is None:
            try:
                requirement = _to_requirement(item, interpreter)
            except (ValueError, ValidationError) as exc:
                problem = f"not a valid requirement: {str(exc).splitlines()[0][:200]}"
        if problem is not None:
            findings.append(IntegrityFinding(item=label, problem=problem))
            continue
        key = (item.aspect, item.direction, item.operator, item.priority, item.value, item.unit)
        if key in seen:
            continue  # the same requirement stated twice
        seen.add(key)
        requirements.append(requirement)

    concerns: list[Concern] = []
    for i, c in enumerate(payload.concerns):
        if _norm(c.source_text) not in request_norm or not c.source_text.strip():
            findings.append(
                IntegrityFinding(item=f"concern {i + 1}", problem="source_text is not a quote")
            )
            continue
        concerns.append(Concern(**c.model_dump()))

    missing = list(payload.missing_information)
    if not any(r.priority is Priority.HARD for r in requirements):
        missing.append(
            "No numeric acceptance limit was stated (for example a maximum density or a "
            "minimum stiffness), so suitability cannot be judged against a requirement."
        )
    for r in requirements:
        if r.direction is Direction.UNSPECIFIED and r.aspect in (
            Aspect.NORMAL_STIFFNESS,
            Aspect.SHEAR_STIFFNESS,
            Aspect.TENSILE_ULTIMATE,
        ):
            missing.append(
                f"No direction was stated for '{r.label}': directional materials such as "
                "composites cannot be judged against it."
            )
    status = (
        "empty"
        if not requirements
        else "ok"
        if not findings and any(r.priority is Priority.HARD for r in requirements)
        else "partial"
    )
    return InterpretationOutcome(
        request_text=request_text,
        interpreter=dict(interpreter),
        status=status,
        requirements=tuple(requirements),
        concerns=tuple(concerns),
        application=payload.application if _norm(payload.application) in request_norm else "",
        design_allowed=payload.design_allowed,
        missing_information=tuple(dict.fromkeys(missing)),
        findings=tuple(findings),
        raw_output=audit,
    )


def _check_item(item: RequirementItem, request_norm: str) -> str | None:
    quote = item.source_text
    if not quote.strip() or _norm(quote) not in request_norm:
        return "source_text is not a verbatim quote from the request"
    bounded = item.operator in (Operator.LE, Operator.GE, Operator.BETWEEN, Operator.TARGET)
    if bounded:
        if item.value is None or item.unit is None:
            return "a limit needs the value and unit the user wrote"
        numbers = [item.value, *([item.upper] if item.upper is not None else [])]
        for n in numbers:
            if not _stated(n, quote):
                return f"the number {n:g} is not written in the quoted text (invented limit)"
        if not _unit_stated(item.unit, quote):
            return f"the unit {item.unit!r} is not written in the quoted text"
        if item.priority is not Priority.HARD and item.operator is not Operator.TARGET:
            return "a stated limit is a hard requirement"
    elif item.value is not None or item.upper is not None or item.unit is not None:
        return "a qualitative preference (maximize/minimize) carries no number"
    if item.exposure_hours is not None and not _stated(item.exposure_hours, quote):
        return f"the exposure time {item.exposure_hours:g} h is not written in the quoted text"
    return None


def _to_requirement(
    item: RequirementItem, interpreter: Mapping[str, Any]
) -> EngineeringRequirement:
    conditions: dict[str, Any] = {}
    if item.environment is not None:
        conditions["environment"] = item.environment
    if item.exposure_hours is not None:
        conditions["exposure_duration"] = {"value": item.exposure_hours, "unit": "h"}
    soft = item.priority is Priority.SOFT
    return EngineeringRequirement(
        aspect=item.aspect,
        direction=item.direction,
        operator=item.operator,
        value=item.value,
        upper=item.upper,
        unit=item.unit,
        conditions=TestConditions.model_validate(conditions),
        priority=item.priority,
        weight=1.0 if soft and item.operator in (Operator.MAXIMIZE, Operator.MINIMIZE) else None,
        weight_source=EQUAL_WEIGHT_SOURCE if soft else None,
        rationale=f'user request: "{item.source_text.strip()}"',
        provenance=Provenance(
            sources=(
                SourceRef(
                    kind="user_request",
                    identifier="request",
                    metadata={"quote": item.source_text.strip(), "interpreter": dict(interpreter)},
                ),
            ),
        ),
    )


# -- the LLM prompt (used by LLM-backed interpreters outside this package) ---------------------

INTERPRETATION_SYSTEM_PROMPT = """\
You translate a user's natural-language request for a UAV structural material into \
structured engineering requirements. You only translate what the user wrote. You never \
state, estimate or recall any material property (no densities, strengths, moduli, \
temperatures or corrosion rates of any material), and you never name materials.

Rules:
1. Every requirement and concern quotes the exact words it comes from in "source_text" \
(a verbatim, contiguous substring of the request).
2. A numeric limit ("value", "upper") must be a number written in that quote, and "unit" \
must be the canonical symbol of the unit written there. Never convert units, never invent \
or round a number. If the user gives no number, it is not a limit.
3. Limits ("<=", ">=", "between") are hard requirements. Qualitative wishes ("lightweight", \
"high stiffness", "strong") are soft preferences with operator "maximize" or "minimize" \
and no value or unit.
4. Use direction "laminate_x", "laminate_y" or "laminate_xy" only when the user names the \
in-plane x axis, y axis or in-plane shear of the part; "material_1"/"material_2" only for \
the fibre/transverse axes of a single ply; otherwise "unspecified".
5. A concern without a limit (for example "it operates near the sea") goes into \
"concerns", not "requirements".
6. Set "design_allowed" to true only if the user asks for a new design when no existing \
material is suitable.
7. List in "missing_information" what an engineer would need but the user did not state.

Allowed aspects: {aspects}
Allowed directions: {directions}
Allowed environments: {environments}
Canonical units: {units}

Respond with exactly one JSON object and nothing else, matching this JSON Schema:
{schema}"""


def interpretation_system_prompt() -> str:
    schema = json.dumps(InterpretationPayload.model_json_schema(), sort_keys=True)
    return INTERPRETATION_SYSTEM_PROMPT.format(
        aspects=", ".join(a.value for a in Aspect),
        directions=", ".join(d.value for d in Direction),
        environments=", ".join(e.value for e in EnvironmentClass),
        units=", ".join(UNIT_SPELLINGS),
        schema=schema,
    )


def interpretation_user_prompt(request_text: str) -> str:
    return "User request:\n" + request_text


# -- deterministic offline interpreter ---------------------------------------------------------

_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_UNITS = {
    "density": r"kg/m\^?3|kg/m³|kg/m3|g/cm\^?3|g/cm³|g/cm3|lb/in\^?3|lb/in³",
    "stress": r"GPa|MPa|Msi|ksi|psi",
    "temperature": r"°\s?C|deg\s?C|degC|°\s?F|degF",
    "rate": r"mm/year|mm/yr",
}
_SHEAR = r"in-plane shear (?:stiffness|modulus)|shear (?:stiffness|modulus)"
_STIFFNESS = r"in-plane stiffness|stiffness|young's modulus|elastic modulus|modulus"
_TEMPERATURE = r"(?:maximum |max\.? )?(?:service|operating) temperature|temperature"
_KEYWORDS = [
    (_SHEAR, Aspect.SHEAR_STIFFNESS),
    (r"yield strength", Aspect.TENSILE_YIELD),
    (r"tensile strength|strength", Aspect.TENSILE_ULTIMATE),
    (_STIFFNESS, Aspect.NORMAL_STIFFNESS),
    (r"density", Aspect.DENSITY),
    (_TEMPERATURE, Aspect.MAX_SERVICE_TEMPERATURE),
    (r"corrosion rate", Aspect.CORROSION_PENETRATION),
]
_GE = r"(?:of at least|at least|no less than|not less than|minimum of|>=|≥|up to)"
_LE = r"(?:must not exceed|may not exceed|not exceed|of at most|at most|no more than|maximum of|<=|≤|below|under|less than)"  # noqa: E501
_LIGHT = (
    r"lightweight|light-weight|light weight|low density|low weight"
    r"|minimi[sz]e (?:the )?(?:weight|mass|density)"
)
_UNIT_KIND = {
    Aspect.DENSITY: "density",
    Aspect.NORMAL_STIFFNESS: "stress",
    Aspect.SHEAR_STIFFNESS: "stress",
    Aspect.TENSILE_ULTIMATE: "stress",
    Aspect.TENSILE_YIELD: "stress",
    Aspect.MAX_SERVICE_TEMPERATURE: "temperature",
    Aspect.CORROSION_PENETRATION: "rate",
}


def _canonical_unit(text: str) -> str:
    t = text.replace(" ", "").casefold()
    for symbol, spellings in UNIT_SPELLINGS.items():
        if t == symbol.casefold() or t in (s.replace(" ", "") for s in spellings):
            return symbol
    if t.startswith("°c") or t.startswith("degc"):
        return "degC"
    if t.startswith("°f") or t.startswith("degf"):
        return "degF"
    return text


def _directions(text: str, aspect: Aspect) -> list[Direction]:
    t = text.casefold()
    if aspect is Aspect.SHEAR_STIFFNESS:
        return [Direction.LAMINATE_XY] if "in-plane" in t else [Direction.UNSPECIFIED]
    xy = re.search(r"\bx\b[\w\s-]{0,12}\band\b[\w\s-]{0,12}\by\b", t)
    if xy:
        return [Direction.LAMINATE_X, Direction.LAMINATE_Y]
    if re.search(r"\b(?:along|in) (?:the )?x\b|\bx[- ]direction", t):
        return [Direction.LAMINATE_X]
    if re.search(r"\b(?:along|in) (?:the )?y\b|\by[- ]direction", t):
        return [Direction.LAMINATE_Y]
    if re.search(r"fib(?:re|er) direction|longitudinal", t):
        return [Direction.MATERIAL_1]
    if "transverse" in t:
        return [Direction.MATERIAL_2]
    return [Direction.UNSPECIFIED]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


class RuleBasedInterpreter:
    """Deterministic, offline interpreter for common phrasings (no model, no network).

    It recognises limits such as "density must not exceed 1800 kg/m^3", "stiffness of at
    least 40 GPa in both the x and y directions", "in-plane shear stiffness of at least
    15 GPa", qualitative wishes ("lightweight", "low density", "high stiffness", "high
    strength", "load-carrying capability") and marine exposure ("near the sea"). Anything
    else is ignored and the report says what is missing. Its output goes through exactly
    the same validation as an LLM's.
    """

    name = "rule_based_interpreter"
    version = "1"

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "rule_based",
            "name": self.name,
            "version": self.version,
            "mode": "OFFLINE_DEMO",
            "note": "deterministic phrase patterns; no language model",
        }

    def interpret(self, request_text: str) -> InterpretationOutcome:
        return parse_interpretation(self.payload(request_text), request_text, self.describe())

    def payload(self, request_text: str) -> dict[str, Any]:
        requirements: list[dict[str, Any]] = []
        concerns: list[dict[str, Any]] = []
        for sentence in _sentences(request_text):
            requirements += self._limits(sentence)
            requirements += self._preferences(sentence)
            concerns += self._concerns(sentence, requirements)
        design = bool(re.search(r"\bdesign\b", request_text, re.IGNORECASE))
        application = ""
        m = re.search(r"\bfor (?:an? )?([^.,;]*?(?:panel|skin|spar|rib|frame|structure|member))",
                      request_text, re.IGNORECASE)  # fmt: skip
        if m:
            application = m.group(1)
        return {
            "requirements": requirements,
            "concerns": concerns,
            "application": application,
            "design_allowed": design,
            "missing_information": [],
        }

    def _limits(self, sentence: str) -> list[dict[str, Any]]:
        """Numeric limits: keyword ... comparator number unit (within one clause)."""
        pattern = "|".join(f"(?P<k{i}>{kw})" for i, (kw, _) in enumerate(_KEYWORDS))
        hits = list(re.finditer(pattern, sentence, re.IGNORECASE))
        out: list[dict[str, Any]] = []
        for j, hit in enumerate(hits):
            index = next(i for i in range(len(_KEYWORDS)) if hit.group(f"k{i}"))
            aspect = _KEYWORDS[index][1]
            end = hits[j + 1].start() if j + 1 < len(hits) else len(sentence)
            segment = sentence[hit.start() : end]
            units = _UNITS[_UNIT_KIND[aspect]]
            for comparator, op in ((_LE, "<="), (_GE, ">=")):
                m = re.search(rf"{comparator}\s*{_NUM}\s*({units})", segment, re.IGNORECASE)
                if not m:
                    continue
                if aspect is Aspect.MAX_SERVICE_TEMPERATURE and op == "<=":
                    continue  # "temperature below X" is not a service-temperature minimum
                value = float(m.group(1).replace(",", ""))
                for direction in _directions(segment, aspect):
                    out.append(
                        {
                            "aspect": aspect.value,
                            "direction": direction.value,
                            "operator": op,
                            "value": value,
                            "unit": _canonical_unit(m.group(2)),
                            "priority": "hard",
                            "environment": "marine_atmosphere"
                            if aspect is Aspect.CORROSION_PENETRATION
                            else None,
                            "source_text": sentence,
                        }
                    )
                break
        return out

    def _preferences(self, sentence: str) -> list[dict[str, Any]]:
        t = sentence.casefold()
        out: list[dict[str, Any]] = []

        def pref(aspect: Aspect, op: str, direction: Direction = Direction.UNSPECIFIED) -> None:
            out.append(
                {
                    "aspect": aspect.value,
                    "direction": direction.value,
                    "operator": op,
                    "priority": "soft",
                    "source_text": sentence,
                }
            )

        if re.search(_LIGHT, t):
            pref(Aspect.DENSITY, "minimize")
        if re.search(r"high (?:in-plane )?stiffness|stiffer|maximi[sz]e (?:the )?stiffness", t):
            m = re.search(r"high (?:in-plane )?stiffness|stiffer|maximi[sz]e (?:the )?stiffness", t)
            assert m
            for direction in _directions(t[m.start() :], Aspect.NORMAL_STIFFNESS):
                pref(Aspect.NORMAL_STIFFNESS, "maximize", direction)
        if re.search(r"high strength|load[- ]carrying|load[- ]bearing|\bstrong\b", t):
            pref(Aspect.TENSILE_ULTIMATE, "maximize")
        return out

    def _concerns(self, sentence: str, requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        t = sentence.casefold()
        limited = any(r["aspect"] == Aspect.CORROSION_PENETRATION.value for r in requirements)
        if not limited and re.search(
            r"near the sea|marine|seawater|sea water|salt spray|coastal|corrosi", t
        ):  # noqa: E501
            return [
                {
                    "aspect": Aspect.CORROSION_PENETRATION.value,
                    "environment": EnvironmentClass.MARINE_ATMOSPHERE.value,
                    "source_text": sentence,
                    "note": "marine exposure stated without a corrosion limit",
                }
            ]
        return []
