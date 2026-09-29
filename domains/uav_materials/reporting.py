"""Final engineering report: structured result + Markdown, built only from workflow state.

``build_report`` reads the artifacts the workflow capabilities produced
(interpretation, existing-material search, decision, design search, design
assessment) and copies values out of them: measurements from
:class:`~domains.uav_materials.schema.MaterialRecord`, predictions from the
physics models, checks and distances from
:class:`~domains.uav_materials.evaluation.CandidateEvaluation`.
``render_markdown`` formats that structured report and nothing else, so every
number in the Markdown appears in the structured report. No language model
writes any part of it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from domains.uav_materials.clt import MODEL as LAMINATE_MODEL
from domains.uav_materials.constituents import constituent_library
from domains.uav_materials.decision import MaterialDecision
from domains.uav_materials.evaluation import CandidateEvaluation
from domains.uav_materials.evidence import RequirementEvidence
from domains.uav_materials.interpretation import InterpretationOutcome
from domains.uav_materials.inverse_design import DesignRow, DesignSearchResult, ExcludedRequirement
from domains.uav_materials.micromechanics import MODEL as PLY_MODEL
from domains.uav_materials.profile import Priority
from domains.uav_materials.search import CheckStatus
from domains.uav_materials.state import UAVMaterialsState
from domains.uav_materials.workflow import (
    DECISION,
    DESIGN_ASSESSMENT,
    DESIGN_SEARCH,
    EXISTING_SEARCH,
    INTERPRETATION,
    DesignAssessment,
    ExistingSearchResult,
)
from jevpilot import FrozenModel, WorkflowState

REPORT_VERSION = "uav-material-report/1"
TOP_EXISTING = 5

SOURCE_TITLES = {
    "MIL-HDBK-5J": "MIL-HDBK-5J, Metallic Materials and Elements for Aerospace Vehicle Structures "
    "(U.S. Department of Defense, 2003; Distribution Statement A)",
    "NRL-AD0609618": "Corrosion of Metals in Tropical Environments, Part 6: Aluminum and "
    "Magnesium (U.S. Naval Research Laboratory, DTIC AD0609618)",
    "NASA-CR-80764": "Materials Data Handbook: Aluminum Alloy 7075 (NASA CR-80764, 1966)",
    "NASA-CR-123773": "Materials Data Handbook: Aluminum Alloy 7075, 2nd ed. "
    "(NASA CR-123773, 1972)",
    "NASA-TM-83320": "Chamis: Simplified Composite Micromechanics Equations for Hygral, Thermal "
    "and Mechanical Properties (NASA TM-83320, 1983)",
    "NASA-TP-2515": "Murthy & Chamis: Integrated Composite Analyzer (ICAN) Users and Programmers "
    "Manual (NASA TP-2515, 1986)",
    "NASA-TP-3290": "Murthy, Ginty & Sanfeliz: Second Generation Integrated Composite Analyzer "
    "(ICAN) Computer Code (NASA TP-3290, 1993)",
    "NASA-RP-1351": "Nettles: Basic Mechanics of Laminated Composite Plates (NASA RP-1351, 1994)",
}
STATIC_LIMITATIONS = (
    "Laminate strength and failure are not modelled (no failure criterion is implemented); ply "
    "tensile strength is not a laminate strength, so load-carrying capability of a designed "
    "laminate is unverified.",
    "Corrosion, water absorption and temperature or environmental degradation of the designed "
    "composite are not predicted.",
    "Designed-candidate properties are model predictions (NASA-sourced micromechanics and "
    "classical lamination theory, checked against the sources' worked examples); they are not "
    "test results for this laminate, and no experimental validation has been done.",
    "Constituent properties are indicative NASA data-bank values (dry, room temperature); ply "
    "thickness and the fibre-volume-fraction range are modelling assumptions.",
    "The existing-material dataset has 46 records from public handbooks and reports; many lack "
    "corrosion, temperature or water-absorption data (reported as evidence gaps, not failures).",
    "The acceptance policy is a demonstration policy, not a UAV design standard.",
)


# -- structured report --------------------------------------------------------------------------


class StepRecord(FrozenModel):
    step: int
    capability_id: str | None
    intent: str
    router_id: str | None
    success: bool | None


class RunInfo(FrozenModel):
    mode: str
    note: str
    router: dict[str, Any]
    interpreter: dict[str, Any]
    steps: tuple[StepRecord, ...]


class RequirementRow(FrozenModel):
    label: str
    priority: str
    weight: float | None
    quote: str


class InterpretationSection(FrozenModel):
    status: str
    interpreter: str
    application: str
    design_allowed: bool
    requirements: tuple[RequirementRow, ...]
    concerns: tuple[str, ...]
    missing_information: tuple[str, ...]
    integrity_findings: tuple[str, ...]
    error: str | None = None


class Value(FrozenModel):
    """One requirement answered for one candidate, with where the value came from."""

    requirement: str
    property: str
    value: float | None
    unit: str | None
    evidence_type: str | None  # measured / datasheet / derived / predicted
    source: str | None
    resolution: str  # resolved / ambiguous / unsupported
    gap: str | None  # missing / ambiguous / unsupported / condition_mismatch / ...
    check: str | None  # satisfied / violated / undetermined (hard requirements)


class CandidateRow(FrozenModel):
    rank: int | None
    candidate_id: str
    name: str
    feasibility: str
    evidence_classification: str
    distance: float | None
    pessimistic_distance: float | None
    coverage: float | None
    hard_evidence: tuple[int, int]
    violated: tuple[str, ...]
    evidence_gaps: tuple[str, ...]
    values: tuple[Value, ...]  # the hard requirements


class NearMiss(FrozenModel):
    requirement: str
    candidate: str
    value: float
    limit: float
    unit: str
    relative_violation: float
    source: str | None


class ExistingSection(FrozenModel):
    dataset: str
    evaluated: int
    counts: dict[str, int]
    top: tuple[CandidateRow, ...]
    near_misses: tuple[NearMiss, ...]


class DecisionSection(FrozenModel):
    outcome: str
    selected: str | None
    reasons: tuple[str, ...]
    policy_label: str
    policy: dict[str, Any]
    counts: dict[str, int]
    design_supported: tuple[str, ...]
    design_unsupported: tuple[str, ...]


class Prediction(FrozenModel):
    property: str
    value: float
    unit: str
    direction: str
    equation: str
    model: str


class DesignedSection(FrozenModel):
    status: str
    message: str
    label: str
    design_id: str | None
    fiber: str | None
    matrix: str | None
    fiber_volume_fraction: float | None
    layup: str | None
    plies: int | None
    ply_thickness_mm: float | None
    laminate_thickness_mm: float | None
    predictions: tuple[Prediction, ...]
    ply_predictions: tuple[Prediction, ...]
    unsupported: tuple[str, ...]
    model: str
    model_validation: tuple[str, ...]
    excluded: tuple[ExcludedRequirement, ...]
    design_target: tuple[str, ...]
    selection_rule: str
    grid: str
    rows: tuple[DesignRow, ...]
    design_target_feasibility: str | None


class ComparisonRow(FrozenModel):
    requirement: str
    hard: bool
    existing: Value | None
    designed: Value | None


class ComparisonSection(FrozenModel):
    existing_name: str | None
    existing_reason: str
    designed_name: str | None
    rows: tuple[ComparisonRow, ...]


class EvidenceRow(FrozenModel):
    subject: str
    counts: dict[str, int]


class SourceRow(FrozenModel):
    identifier: str
    description: str
    used_for: str


class MaterialWorkflowReport(FrozenModel):
    version: str = REPORT_VERSION
    title: str = "JevPilot UAV Materials Report"
    status: str  # complete / incomplete
    status_note: str
    user_goal: str
    recommendation: str
    run: RunInfo
    interpretation: InterpretationSection | None
    existing: ExistingSection | None
    decision: DecisionSection | None
    designed: DesignedSection | None
    comparison: ComparisonSection | None
    evidence: tuple[EvidenceRow, ...]
    limitations: tuple[str, ...]
    sources: tuple[SourceRow, ...]


# -- building ---------------------------------------------------------------------------------


def _artifact(state: WorkflowState, kind: str) -> Any:
    a = state.latest_artifact(kind)
    return a.content if a is not None else None


def _values(e: CandidateEvaluation, hard_only: bool = False) -> tuple[Value, ...]:
    out = []
    for res, ev, outcome in zip(e.resolutions, e.evidence.requirements, e.outcomes, strict=True):
        if hard_only and not ev.hard:
            continue
        out.append(_value(res.requirement.label, ev, outcome.resolution.value, outcome.gap))
    return tuple(out)


def _value(label: str, ev: RequirementEvidence, resolution: str, gap: str | None) -> Value:
    return Value(
        requirement=label,
        property=ev.property,
        value=ev.numeric_value,
        unit=ev.unit,
        evidence_type=ev.evidence_type,
        source=ev.model or ev.source,
        resolution=resolution,
        gap=gap,
        check=ev.check,
    )


def _candidate_row(e: CandidateEvaluation) -> CandidateRow:
    return CandidateRow(
        rank=e.rank,
        candidate_id=e.candidate_id,
        name=e.name,
        feasibility=e.feasibility.value,
        evidence_classification=e.evidence.classification.value,
        distance=e.distance,
        pessimistic_distance=e.pessimistic_distance,
        coverage=e.coverage,
        hard_evidence=e.evidence.hard_coverage,
        violated=e.evidence.violated,
        evidence_gaps=tuple(dict.fromkeys(e.evidence.missing_critical)),
        values=_values(e, hard_only=True),
    )


def _near_misses(search: ExistingSearchResult) -> tuple[NearMiss, ...]:
    """Per violated hard requirement: the existing material closest to the limit."""
    best: dict[str, NearMiss] = {}
    for e in search.evaluations:
        for res, check in zip(
            (r for r in e.resolutions if r.requirement.priority is Priority.HARD),
            e.checks,
            strict=True,
        ):
            if check.status is not CheckStatus.VIOLATED or check.value is None:
                continue
            lower, upper = check.requirement.bounds()
            limit = lower if lower is not None and check.value < lower else upper
            if limit is None or limit == 0 or check.requirement.unit is None:
                continue
            gap = abs(check.value - limit) / abs(limit)
            label = res.requirement.label
            if label not in best or gap < best[label].relative_violation:
                used = check.used[0] if check.used else None
                from domains.uav_materials.search import source_label

                best[label] = NearMiss(
                    requirement=label,
                    candidate=e.name,
                    value=check.value,
                    limit=limit,
                    unit=check.requirement.unit,
                    relative_violation=gap,
                    source=source_label(used) if used else None,
                )
    return tuple(best.values())


def _evidence_counts(e: CandidateEvaluation) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ev, outcome in zip(e.evidence.requirements, e.outcomes, strict=True):
        key = outcome.gap if outcome.gap in (
            "missing", "ambiguous", "unsupported", "condition_mismatch"
        ) else (ev.evidence_type or "missing")  # fmt: skip
        counts[key] = counts.get(key, 0) + 1
    order = ("measured", "datasheet", "derived", "predicted", "missing", "condition_mismatch",
             "unsupported", "ambiguous")  # fmt: skip
    return {k: counts.get(k, 0) for k in order}


def _predictions(candidate: Any) -> tuple[tuple[Prediction, ...], tuple[Prediction, ...]]:
    laminate = []
    for prop in ("density", "laminate_ex", "laminate_ey", "laminate_gxy", "laminate_nuxy"):
        for m in candidate.material.measurements(prop):
            meta = m.provenance.sources[0].metadata if m.provenance else {}
            laminate.append(
                Prediction(
                    property=prop,
                    value=float(m.value),
                    unit=str(m.unit),
                    direction=str(m.conditions.other.get("direction", "")),
                    equation=str(meta.get("equation", "")),
                    model=m.provenance.sources[0].identifier if m.provenance else "",
                )
            )
    ply = []
    for raw in candidate.metadata.get("laminate", {}).get("ply_predictions", []):
        sources = (raw.get("provenance") or {}).get("sources") or [{}]
        ply.append(
            Prediction(
                property=raw["property"],
                value=float(raw["value"]),
                unit=str(raw["unit"]),
                direction=str((raw.get("conditions") or {}).get("other", {}).get("direction", "")),
                equation=str(sources[0].get("metadata", {}).get("equation", "")),
                model=str(sources[0].get("identifier", PLY_MODEL.label)),
            )
        )
    return tuple(laminate), tuple(ply)


def _designed_section(design: DesignSearchResult) -> DesignedSection:
    grid = (
        f"{len(design.config.fiber_volume_fractions)} fibre volume fractions x "
        f"{len(design.config.layups)} symmetric layups = {len(design.rows)} designs"
    )
    common = {
        "status": design.status,
        "message": design.message,
        "label": "designed composite candidate (virtual design predicted by physics models; "
        "not an experimentally validated material)",
        "unsupported": LAMINATE_MODEL.unsupported,
        "model": f"{LAMINATE_MODEL.label} on {PLY_MODEL.label} plies",
        "model_validation": (
            f"{LAMINATE_MODEL.label}: {LAMINATE_MODEL.validation.status}",
            f"{PLY_MODEL.label}: {PLY_MODEL.validation.status}",
        ),
        "excluded": design.excluded,
        "design_target": tuple(r.label for r in design.design_target.requirements)
        if design.design_target
        else (),
        "selection_rule": design.selection_rule,
        "grid": grid,
        "rows": design.rows,
    }
    best = design.best
    if best is None:
        return DesignedSection(
            design_id=None, fiber=None, matrix=None, fiber_volume_fraction=None, layup=None,
            plies=None, ply_thickness_mm=None, laminate_thickness_mm=None, predictions=(),
            ply_predictions=(), design_target_feasibility=None, **common,
        )  # fmt: skip
    variables = best.design.get("variables", {})
    laminate = best.metadata.get("laminate", {})
    library = constituent_library()
    fiber, matrix = str(variables.get("fiber_id")), str(variables.get("matrix_id"))
    predictions, ply = _predictions(best)
    evaluation = design.best_evaluation
    return DesignedSection(
        design_id=best.candidate_id,
        fiber=f"{library[fiber].name} ({fiber})" if fiber in library else fiber,
        matrix=f"{library[matrix].name} ({matrix})" if matrix in library else matrix,
        fiber_volume_fraction=float(variables.get("fiber_volume_fraction", 0.0)),
        layup=str(laminate.get("layup")),
        plies=len(laminate.get("plies", [])),
        ply_thickness_mm=float(laminate.get("ply_thickness_m", 0.0)) * 1000,
        laminate_thickness_mm=float(laminate.get("thickness_m", 0.0)) * 1000,
        predictions=predictions,
        ply_predictions=ply,
        design_target_feasibility=evaluation.feasibility.value if evaluation else None,
        **common,
    )


DESIGN_SOURCES = {
    "NASA-RP-1351": "classical lamination theory: Q, Qbar, A/B/D and laminate constants",
    "NASA-TM-83320": "ply micromechanics equations; matrix constituent values",
    "NASA-TP-3290": "ply tensile-strength equation; fibre and matrix constituent values",
    "NASA-TP-2515": "fibre and matrix constituent values (cross-check)",
}


def _sources(
    existing: Iterable[CandidateEvaluation], designed: DesignSearchResult | None
) -> tuple[SourceRow, ...]:
    rows: dict[str, SourceRow] = {}
    for e in existing:
        for ev in e.evidence.requirements:
            ident = (ev.source or "").split(" ")[0]
            if ident in SOURCE_TITLES and ident not in rows:
                rows[ident] = SourceRow(
                    identifier=ident,
                    description=SOURCE_TITLES[ident],
                    used_for="existing-material values",
                )
    if designed is not None and designed.best is not None:
        for ident, used_for in DESIGN_SOURCES.items():
            rows.setdefault(
                ident,
                SourceRow(identifier=ident, description=SOURCE_TITLES[ident], used_for=used_for),
            )
    return tuple(rows.values())


def _run_info(state: WorkflowState, current: StepRecord | None = None) -> RunInfo:
    run = state.metadata.get("run", {})
    observations = {o.id: o for o in state.observations}
    steps = tuple(
        StepRecord(
            step=h.step,
            capability_id=h.decision.capability_id,
            intent=str(h.decision.intent),
            router_id=h.decision.router_id,
            success=observations[h.observation_id].success if h.observation_id else None,
        )
        for h in state.history
    ) + ((current,) if current is not None else ())
    return RunInfo(
        mode=str(run.get("mode", "UNSPECIFIED")),
        note=str(run.get("note", "")),
        router=dict(run.get("router", {})),
        interpreter=dict(run.get("interpreter", {})),
        steps=steps,
    )


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "-"
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.{digits}g}"


def _recommendation(
    decision: DecisionSection | None,
    designed: DesignedSection | None,
    existing: ExistingSection | None,
    interpretation: InterpretationSection | None,
) -> str:
    if interpretation is None or interpretation.status in ("failed", "empty"):
        return (
            "No engineering requirement could be interpreted from the request, so no material "
            "was searched or designed. Restate the request with explicit limits."
        )
    if decision is None:
        return "The workflow stopped before a decision was made; see the orchestration trace."
    if decision.outcome == "use_existing":
        return (
            f"Recommend the existing material {decision.selected}: it meets every check of the "
            f"{decision.policy_label.split(' - ')[0].lower()} with sufficient evidence."
        )
    if decision.outcome == "needs_information":
        return (
            "More information is needed: the request states no numeric acceptance limit, so no "
            "material can be judged suitable. The best-ranked existing materials are listed for "
            "orientation only."
        )
    if decision.outcome == "no_suitable_existing":
        return (
            "No existing material meets the acceptance policy, and no composite design is "
            f"proposed because {decision.reasons[2]}. See the decision reasons for the evidence "
            "that would be needed."
        )
    if designed is None or designed.design_id is None:
        return (
            "No existing material meets the acceptance policy, and the composite design space "
            "cannot answer the stated hard requirements, so no design is proposed."
        )
    values = {p.property: p for p in designed.predictions}
    parts = ", ".join(
        f"{name} {_fmt(values[p].value)} {values[p].unit}"
        for p, name in (
            ("density", "density"),
            ("laminate_ex", "Ex"),
            ("laminate_ey", "Ey"),
            ("laminate_gxy", "Gxy"),
        )
        if p in values
    )
    verdict = (
        "satisfies every hard requirement the laminate model can answer"
        if designed.status == "designed"
        else "is the closest design but does not satisfy every hard requirement the laminate "
        "model can answer"
    )
    return (
        "No existing material meets the acceptance policy. Designed composite candidate "
        f"(virtual design): {designed.fiber} / {designed.matrix}, fibre volume fraction "
        f"{_fmt(designed.fiber_volume_fraction)}, layup {designed.layup}, predicted {parts}. It "
        f"{verdict}; requirements the model cannot answer (for example strength, failure and "
        "corrosion) remain open and need test evidence."
    )


def build_report(
    state: WorkflowState,
    *,
    incomplete_reason: str | None = None,
    current_step: StepRecord | None = None,
) -> MaterialWorkflowReport:
    assert isinstance(state, UAVMaterialsState)
    interp: InterpretationOutcome | None = _artifact(state, INTERPRETATION)
    search: ExistingSearchResult | None = _artifact(state, EXISTING_SEARCH)
    decision: MaterialDecision | None = _artifact(state, DECISION)
    design: DesignSearchResult | None = _artifact(state, DESIGN_SEARCH)
    assessment: DesignAssessment | None = _artifact(state, DESIGN_ASSESSMENT)

    interpretation = None
    if interp is not None:
        interpretation = InterpretationSection(
            status=interp.status,
            interpreter=str(interp.interpreter.get("name") or interp.interpreter.get("kind")),
            application=interp.application,
            design_allowed=interp.design_allowed,
            requirements=tuple(
                RequirementRow(
                    label=r.label,
                    priority=r.priority.value,
                    weight=r.weight,
                    quote=str(r.provenance.sources[0].metadata.get("quote", ""))
                    if r.provenance
                    else "",
                )
                for r in interp.requirements
            ),
            concerns=tuple(
                f"{c.aspect.value}"
                + (f" ({c.environment.value})" if c.environment else "")
                + f': "{c.source_text}" - {c.note or "no limit stated"}'
                for c in interp.concerns
            ),
            missing_information=interp.missing_information,
            integrity_findings=tuple(f"{f.item}: {f.problem}" for f in interp.findings),
            error=interp.error,
        )

    existing = None
    if search is not None:
        existing = ExistingSection(
            dataset=search.dataset,
            evaluated=len(search.evaluations),
            counts=search.counts,
            top=tuple(_candidate_row(e) for e in search.evaluations[:TOP_EXISTING]),
            near_misses=_near_misses(search),
        )

    decision_section = None
    if decision is not None:
        by_id = {e.candidate_id: e.name for e in (search.evaluations if search else ())}
        decision_section = DecisionSection(
            outcome=decision.outcome.value,
            selected=by_id.get(decision.selected_candidate_id or ""),
            reasons=decision.reasons,
            policy_label=decision.policy.label,
            policy=decision.policy.model_dump(mode="json", exclude={"label", "rationale"}),
            counts=decision.counts,
            design_supported=decision.design_supported,
            design_unsupported=decision.design_unsupported,
        )

    designed = _designed_section(design) if design is not None else None

    comparison = None
    evidence: list[EvidenceRow] = []
    ref: CandidateEvaluation | None = None
    if assessment is not None:
        ref = assessment.reference_existing
        rows = []
        d = assessment.designed
        dv = _values(d)
        ev = _values(ref) if ref else (None,) * len(dv)
        for i, res in enumerate(d.resolutions):
            rows.append(
                ComparisonRow(
                    requirement=res.requirement.label,
                    hard=res.requirement.priority is Priority.HARD,
                    existing=ev[i],
                    designed=dv[i],
                )
            )
        design_name = (
            f"designed {designed.layup} laminate, fibre volume fraction "
            f"{_fmt(designed.fiber_volume_fraction)} ({designed.design_id})"
            if designed is not None and designed.design_id
            else d.name
        )
        comparison = ComparisonSection(
            existing_name=ref.name if ref else None,
            existing_reason=assessment.reference_reason,
            designed_name=design_name,
            rows=tuple(rows),
        )
        if ref is not None:
            evidence.append(
                EvidenceRow(subject=f"existing: {ref.name}", counts=_evidence_counts(ref))
            )
        evidence.append(
            EvidenceRow(subject="designed composite candidate", counts=_evidence_counts(d))
        )
    elif search is not None and decision is not None:
        selected = next(
            (e for e in search.evaluations if e.candidate_id == decision.selected_candidate_id),
            search.evaluations[0] if search.evaluations else None,
        )
        if selected is not None:
            label = "selected existing" if decision.selected_candidate_id else "top-ranked existing"
            evidence.append(
                EvidenceRow(subject=f"{label}: {selected.name}", counts=_evidence_counts(selected))
            )

    limitations = list(STATIC_LIMITATIONS)
    if interpretation is not None:
        limitations += [f"Missing information: {m}" for m in interpretation.missing_information]
        limitations += [f"Stated concern not evaluated: {c}" for c in interpretation.concerns]
        kind = interp.interpreter.get("kind") if interp is not None else None
        mode_note = (
            "Requirements were interpreted by the deterministic offline parser, which handles "
            "common phrasings only; review the interpreted requirements."
            if kind == "rule_based"
            else "Requirements were produced offline by a deterministic scripted chat model "
            "(no language model was called) and validated (quotes, stated numbers and units); "
            "review the interpreted requirements."
            if kind == "scripted"
            else "Requirements were interpreted by a language model and validated (quotes, "
            "stated numbers and units); review the interpreted requirements."
        )
        limitations.append(mode_note)
    if designed is not None:
        limitations += [
            f"Not addressed by the design search ({x.status}): {x.requirement} - {x.reason}"
            for x in designed.excluded
        ]

    complete = interpretation is not None and (
        interpretation.status in ("failed", "empty")
        or (
            decision is not None
            and (
                decision.outcome.value != "design"
                or (design is not None and (design.best is None or assessment is not None))
            )
        )
    )
    status_note = (
        incomplete_reason
        or ("all workflow steps required by the decision completed" if complete else "")
        or "the workflow ended before every step required by the decision completed"
    )
    return MaterialWorkflowReport(
        status="complete" if complete and incomplete_reason is None else "incomplete",
        status_note=status_note,
        user_goal=state.request_text or "",
        recommendation=_recommendation(decision_section, designed, existing, interpretation),
        run=_run_info(state, current_step),
        interpretation=interpretation,
        existing=existing,
        decision=decision_section,
        designed=designed,
        comparison=comparison,
        evidence=tuple(evidence),
        limitations=tuple(limitations),
        sources=_sources(
            [*(search.evaluations[:TOP_EXISTING] if search else ()), *((ref,) if ref else ())],
            design,
        ),
    )


# -- Markdown ----------------------------------------------------------------------


def _qty(v: float | None, unit: str | None) -> str:
    """A value with its unit; dimensionless ("1") values show no unit."""
    return f"{_fmt(v)} {unit}" if unit and unit != "1" else _fmt(v)


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.0%}"


def _cell(text: object) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _value_cell(v: Value | None) -> str:
    if v is None:
        return "-"
    if v.gap in ("ambiguous", "unsupported"):
        return f"{v.gap}"
    if v.value is None:
        return f"{v.gap or 'missing'}"
    text = _qty(v.value, v.unit)
    detail = ", ".join(x for x in (v.evidence_type, v.check) if x)
    return f"{text} ({detail})" if detail else text


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def render_markdown(report: MaterialWorkflowReport) -> str:
    r = report
    lines: list[str] = [f"# {r.title}", ""]
    lines += [
        f"> **Mode: {r.run.mode}**" + (f" - {r.run.note}" if r.run.note else ""),
        f"> Report status: {r.status} ({r.status_note}).",
        "> Every value in this report comes from the material dataset or the physics models; "
        "no language model produced any material property.",
        "",
        "## User Goal",
        "",
        f"> {r.user_goal}",
        "",
        "## Summary",
        "",
        r.recommendation,
        "",
    ]

    lines += ["## Interpreted Engineering Requirements", ""]
    i = r.interpretation
    if i is None:
        lines += ["The request was not interpreted.", ""]
    else:
        lines += [
            f"Interpreter: `{i.interpreter}` - status `{i.status}` - design allowed: "
            f"{'yes' if i.design_allowed else 'no'}"
            + (f" - application: {i.application}" if i.application else ""),
            "",
        ]
        if i.error:
            lines += [f"Interpretation error: {i.error}", ""]
        if i.requirements:
            lines += _table(
                ("Requirement", "Type", "Weight", "From the request"),
                ((q.label, q.priority, _fmt(q.weight), f'"{q.quote}"') for q in i.requirements),
            )
            lines.append("")
        for title, items in (
            ("Concerns (no limit stated; not scored)", i.concerns),
            ("Missing information", i.missing_information),
            ("Integrity findings (rejected interpretation items)", i.integrity_findings),
        ):
            if items:
                lines += [f"**{title}**", "", *[f"- {x}" for x in items], ""]

    lines += ["## Existing Material Search", ""]
    e = r.existing
    if e is None:
        lines += ["No existing-material search was run.", ""]
    else:
        c = e.counts
        lines += [
            f"Dataset: {e.dataset}.",
            "",
            f"{e.evaluated} materials evaluated: {c.get('feasible', 0)} feasible, "
            f"{c.get('undetermined', 0)} undetermined, {c.get('infeasible', 0)} infeasible. "
            f"Evidence classes: {c.get('infeasible', 0)} engineering failures, "
            f"{c.get('evidence_gap', 0)} evidence gaps (not failures).",
            "",
        ]
        lines += _table(
            (
                "Rank",
                "Material",
                "Feasibility",
                "Evidence",
                "Hard evidence",
                "Distance",
                "Pessimistic",
                "Coverage",
                "Violated / gaps",
            ),  # fmt: skip
            (
                (
                    row.rank,
                    row.name,
                    row.feasibility,
                    row.evidence_classification,
                    f"{row.hard_evidence[0]}/{row.hard_evidence[1]}",
                    _fmt(row.distance),
                    _fmt(row.pessimistic_distance),
                    _pct(row.coverage),
                    "; ".join(
                        [
                            *(f"violates {v}" for v in row.violated),
                            *(f"no evidence: {g}" for g in row.evidence_gaps),
                        ]  # fmt: skip
                    )
                    or "-",
                )
                for row in e.top
            ),
        )
        lines += ["", "Hard-requirement values of the top-ranked materials:", ""]
        value_rows: list[tuple[object, ...]] = []
        for row in e.top:
            if all(v.value is None for v in row.values):
                value_rows.append(
                    (row.name, "all hard requirements", "no value in the dataset", "-")
                )
                continue
            value_rows += [
                (row.name, v.requirement, _value_cell(v), v.source or "-") for v in row.values
            ]
        lines += _table(("Material", "Requirement", "Value", "Source"), value_rows)
        if e.near_misses:
            lines += ["", "Closest existing material per violated hard requirement:", ""]
            lines += _table(
                ("Requirement", "Closest material", "Value", "Limit", "Off by", "Source"),
                (
                    (
                        n.requirement,
                        n.candidate,
                        f"{_fmt(n.value)} {n.unit}",
                        f"{_fmt(n.limit)} {n.unit}",
                        f"{n.relative_violation:.1%}",
                        n.source or "-",
                    )
                    for n in e.near_misses
                ),
            )
        lines.append("")

    lines += ["## Decision", ""]
    d = r.decision
    if d is None:
        lines += ["No decision was made.", ""]
    else:
        lines += [f"**Outcome: `{d.outcome}`**" + (f" - {d.selected}" if d.selected else ""), ""]
        lines += [f"- {x}" for x in d.reasons]
        lines += ["", f"Policy ({d.policy_label}):", ""]
        lines += _table(("Setting", "Value"), ((k, v) for k, v in d.policy.items()))
        if d.design_unsupported:
            lines += ["", "Hard requirements the composite design space cannot address:", ""]
            lines += [f"- {x}" for x in d.design_unsupported]
        lines.append("")

    if r.designed is not None:
        s = r.designed
        lines += ["## Designed Composite Candidate", "", f"_{s.label}._", "", s.message + ".", ""]
        if s.design_id is not None:
            lines += _table(
                ("Design variable", "Value"),
                (
                    ("fibre", s.fiber),
                    ("matrix", s.matrix),
                    ("fibre volume fraction", _fmt(s.fiber_volume_fraction)),
                    ("stacking sequence", s.layup),
                    ("plies", s.plies),
                    ("ply thickness", f"{_fmt(s.ply_thickness_mm)} mm (equal plies)"),
                    ("laminate thickness", f"{_fmt(s.laminate_thickness_mm)} mm"),
                    ("model", s.model),
                ),
            )
            lines += ["", "Predicted laminate properties (evidence type: predicted):", ""]
            lines += _table(
                ("Property", "Value", "Direction", "Equation"),
                (
                    (p.property, _qty(p.value, p.unit), p.direction, p.equation)
                    for p in s.predictions
                ),
            )
            if s.ply_predictions:
                lines += ["", "Ply (lamina) predictions used by the laminate model:", ""]
                lines += _table(
                    ("Property", "Value", "Direction", "Equation"),
                    (
                        (p.property, _qty(p.value, p.unit), p.direction, p.equation)
                        for p in s.ply_predictions
                    ),
                )
            lines += ["", "Not predicted by the model: " + ", ".join(s.unsupported) + ".", ""]
            lines += [f"- {x}" for x in s.model_validation]
            lines.append("")
        lines += [
            f"Design search: {s.grid}; selection: {s.selection_rule}.",
            "",
            "Design target (requirements the laminate model answers): "
            + (", ".join(s.design_target) or "none"),
            "",
        ]
        if s.excluded:
            lines += ["Excluded from the design search:", ""]
            lines += [f"- {x.requirement} ({x.status}): {x.reason}" for x in s.excluded]
            lines.append("")
        if s.rows:
            lines += _table(
                (
                    "Rank",
                    "Layup",
                    "Vf",
                    "Feasibility",
                    "Density kg/m^3",
                    "Ex GPa",
                    "Ey GPa",
                    "Gxy GPa",
                    "nuxy",
                    "Pessimistic",
                ),  # fmt: skip
                (
                    (
                        row.rank,
                        row.layup,
                        _fmt(row.fiber_volume_fraction),
                        row.feasibility + (f" ({', '.join(row.violated)})" if row.violated else ""),
                        _fmt(row.properties.get("density")),
                        _fmt(row.properties.get("laminate_ex")),
                        _fmt(row.properties.get("laminate_ey")),
                        _fmt(row.properties.get("laminate_gxy")),
                        _fmt(row.properties.get("laminate_nuxy")),
                        _fmt(row.pessimistic_distance),
                    )
                    for row in s.rows
                ),
            )
            lines.append("")

    if r.comparison is not None:
        cmp = r.comparison
        lines += [
            "## Existing vs Designed",
            "",
            f"Existing: {cmp.existing_name or '-'} ({cmp.existing_reason}). "
            f"Designed: {cmp.designed_name}. Both are evaluated against the full interpreted "
            "requirements by the same evaluator.",
            "",
        ]
        lines += _table(
            ("Requirement", "Type", "Existing", "Designed"),
            (
                (
                    row.requirement,
                    "hard" if row.hard else "soft",
                    _value_cell(row.existing),
                    _value_cell(row.designed),
                )  # fmt: skip
                for row in cmp.rows
            ),
        )
        lines.append("")

    lines += ["## Evidence / Confidence", ""]
    if r.evidence:
        keys = list(r.evidence[0].counts)
        lines += _table(
            ("Subject", *keys), ((row.subject, *row.counts.values()) for row in r.evidence)
        )
        lines += [
            "",
            "Counts are per requirement: how each one was answered (measured, datasheet, "
            "derived, predicted) or why it could not be (missing, condition mismatch, "
            "unsupported, ambiguous).",
        ]
    else:
        lines.append("No candidate was evaluated.")
    lines.append("")

    lines += ["## Limitations", "", *[f"- {x}" for x in r.limitations], ""]
    lines += ["## Sources", ""]
    lines += [f"- **{s.identifier}** - {s.description} ({s.used_for})" for s in r.sources] or [
        "- none"
    ]
    lines += ["", "## Orchestration Trace", ""]
    router = r.run.router
    lines += [
        f"Router: `{router.get('router_id', '-')}` ({router.get('type', '-')}); adapter: "
        f"{_adapter_label(router)}. Interpreter: {r.run.interpreter.get('name') or r.run.interpreter.get('kind', '-')}.",  # noqa: E501
        "",
    ]
    lines += _table(
        ("Step", "Capability", "Intent", "Router", "Result"),
        (
            (
                s.step,
                s.capability_id or "-",
                s.intent,
                s.router_id or "-",
                _step_result(s),
            )
            for s in r.run.steps
        ),
    )
    lines.append("")
    return "\n".join(lines)


def _step_result(s: StepRecord) -> str:
    if s.success is None:
        return "this report" if s.capability_id == "uavm.generate_material_report" else "-"
    return "ok" if s.success else "failed"


def _adapter_label(router: dict[str, Any]) -> str:
    adapter = router.get("adapter") or {}
    if not isinstance(adapter, dict) or not adapter:
        return "-"
    return f"{adapter.get('provider', '-')} / {adapter.get('model', '-')}"


def numbers_in_markdown(text: str) -> list[str]:
    """Numeric tokens of a rendered report (used by integrity tests)."""
    return re.findall(r"(?<![\w.\-/])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?%?", text)
