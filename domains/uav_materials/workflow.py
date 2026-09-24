"""The request-driven UAV materials workflow as ordinary JevPilot capabilities.

    interpret_requirements → search_existing_materials → assess_existing_candidates
        → (design_composite_candidate → evaluate_designed_candidate) → generate_material_report

Each capability reads what it needs from the workflow state and takes no
inputs, so any router (rules, an LLM, Jev) only has to choose *which* step
runs next; executable preconditions make only the sensible steps available.
The loop, the router and the termination policy are the unmodified JevPilot
core. Nothing here asks a language model for a material property: the
interpreter translates the request, and every number comes from the
dataset or the physics models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.decision import (
    ExistingMaterialAcceptancePolicy,
    MaterialDecision,
    decide,
)
from domains.uav_materials.evaluation import (
    CandidateEvaluation,
    EvaluationContext,
    evaluate_candidates,
)
from domains.uav_materials.evidence import GapClass
from domains.uav_materials.interpretation import (
    InterpretationOutcome,
    RequirementInterpreter,
    RuleBasedInterpreter,
)
from domains.uav_materials.inverse_design import (
    DesignSearchConfig,
    DesignSearchResult,
    search_designs,
)
from domains.uav_materials.requirements import EngineeringTarget
from domains.uav_materials.search import Feasibility
from domains.uav_materials.state import UAVMaterialsState, WorkflowProgress
from jevpilot import (
    Artifact,
    Capability,
    CapabilityResult,
    CapabilitySpec,
    ControlAction,
    EvaluationResult,
    ExecutionContext,
    FrozenModel,
    Observation,
    SourceRef,
    StateReducer,
    WorkflowState,
)

DOMAIN = "uav_materials"
INTERPRETATION = "uav.interpretation"
EXISTING_SEARCH = "uav.existing_search"
DECISION = "uav.decision"
DESIGN_SEARCH = "uav.design_search"
DESIGN_ASSESSMENT = "uav.design_assessment"
REPORT = "uav.report"
REPORT_MARKDOWN = "uav.report_markdown"
DATASET = (
    "46 existing-material records built offline from committed raw extracts of "
    "MIL-HDBK-5J, NRL report AD0609618 and NASA reports (see data/uav_materials/sources)"
)


class NoInputs(BaseModel):
    """The capability reads everything it needs from the workflow state."""

    model_config = ConfigDict(extra="forbid")


class StepSummary(FrozenModel):
    """Compact, router-visible result of one workflow step."""

    stage: str
    status: str
    message: str
    details: dict[str, Any] = {}


class ExistingSearchResult(FrozenModel):
    target: EngineeringTarget
    evaluations: tuple[CandidateEvaluation, ...]  # ranked
    counts: dict[str, int]
    dataset: str = DATASET


class DesignAssessment(FrozenModel):
    """The designed candidate against the user's full target, next to an existing one."""

    designed: CandidateEvaluation  # full target, existing-material evaluation context
    reference_existing: CandidateEvaluation | None
    reference_reason: str


def _latest(state: WorkflowState, kind: str) -> Any:
    artifact = state.latest_artifact(kind)
    return artifact.content if artifact is not None else None


def _progress(state: WorkflowState) -> WorkflowProgress | None:
    if isinstance(state, UAVMaterialsState) and state.request_text is not None:
        return state.progress
    return None


def reference_existing(
    search: ExistingSearchResult, decision: MaterialDecision
) -> tuple[CandidateEvaluation | None, str]:
    """The existing material shown next to a design: the selected one, else the best-ranked
    candidate whose hard constraints could all be checked (an evidence gap tells little)."""
    by_id = {e.candidate_id: e for e in search.evaluations}
    if decision.selected_candidate_id in by_id:
        return by_id[decision.selected_candidate_id], "the existing material selected by the policy"
    for e in search.evaluations:
        known, total = e.evidence.hard_coverage
        if total and known == total:
            return e, (
                "the best-ranked existing material whose hard constraints could all be checked "
                "(candidates ranked above it are evidence gaps)"
                if e.rank and e.rank > 1
                else "the best-ranked existing material"
            )
    return (search.evaluations[0] if search.evaluations else None), "the best-ranked candidate"


# -- capabilities --------------------------------------------------------------------------------


class _WorkflowCapability(Capability):
    def applicable(self, progress: WorkflowProgress) -> bool:
        raise NotImplementedError

    def is_applicable(self, state: WorkflowState) -> bool:
        progress = _progress(state)
        return progress is not None and self.applicable(progress)

    @staticmethod
    def result(
        summary: StepSummary, *artifacts: Artifact, sources: tuple[SourceRef, ...] = ()
    ) -> CapabilityResult:  # noqa: E501
        return CapabilityResult(output=summary, artifacts=artifacts, sources=sources)


class InterpretRequirements(_WorkflowCapability):
    spec = CapabilitySpec(
        id="uavm.interpret_requirements",
        name="interpret_requirements",
        description="Translate the user's natural-language request into validated engineering "
        "requirements (hard limits, weighted preferences, concerns, missing information). Every "
        "requirement must quote the request; numbers must be written by the user.",
        domain=DOMAIN,
        input_schema=NoInputs,
        output_schema=StepSummary,
        preconditions=("the request has not been interpreted yet",),
        effects=("sets progress.requirements", "adds a uav.interpretation artifact"),
        tags=frozenset({"requirements", "interpretation"}),
    )

    def __init__(self, interpreter: RequirementInterpreter) -> None:
        self.interpreter = interpreter

    def applicable(self, progress: WorkflowProgress) -> bool:
        return progress.requirements == "pending"

    def execute(self, inputs: NoInputs, ctx: ExecutionContext) -> CapabilityResult:
        state = ctx.state
        assert isinstance(state, UAVMaterialsState) and state.request_text is not None
        try:
            outcome = self.interpreter.interpret(state.request_text)
        except Exception as exc:  # the report explains an interpretation failure
            outcome = InterpretationOutcome(
                request_text=state.request_text,
                interpreter=self.interpreter.describe(),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        summary = StepSummary(
            stage="requirements",
            status=outcome.status,
            message=f"{len(outcome.hard)} hard requirement(s), {len(outcome.soft)} preference(s), "
            f"{len(outcome.concerns)} concern(s), {len(outcome.findings)} integrity finding(s)",
            details={
                "hard": len(outcome.hard),
                "soft": len(outcome.soft),
                "design_allowed": outcome.design_allowed,
                "missing_information": len(outcome.missing_information),
            },
        )
        artifact = Artifact(name="interpreted requirements", kind=INTERPRETATION, content=outcome)
        source = SourceRef(kind="interpreter", identifier=str(outcome.interpreter.get("name", "")))
        return self.result(summary, artifact, sources=(source,))


class SearchExistingMaterials(_WorkflowCapability):
    spec = CapabilitySpec(
        id="uavm.search_existing_materials",
        name="search_existing_materials",
        description="Screen and rank the real existing-material dataset (46 records with "
        "measurement-level provenance) against the interpreted requirements with the unified, "
        "evidence-aware evaluator.",
        domain=DOMAIN,
        input_schema=NoInputs,
        output_schema=StepSummary,
        preconditions=("requirements are interpreted", "existing materials not searched yet"),
        effects=("sets progress.existing_search", "adds a uav.existing_search artifact"),
        tags=frozenset({"search", "existing_materials"}),
    )

    def applicable(self, progress: WorkflowProgress) -> bool:
        return progress.requirements in ("ok", "partial") and progress.existing_search == "pending"

    def execute(self, inputs: NoInputs, ctx: ExecutionContext) -> CapabilityResult:
        outcome: InterpretationOutcome = _latest(ctx.state, INTERPRETATION)
        target = outcome.target()
        assert target is not None
        ranked = evaluate_candidates(target, real_candidates())
        counts = {f.value: sum(1 for e in ranked if e.feasibility is f) for f in Feasibility}
        counts.update(
            {g.value: sum(1 for e in ranked if e.evidence.classification is g) for g in GapClass}
        )
        result = ExistingSearchResult(target=target, evaluations=ranked, counts=counts)
        summary = StepSummary(
            stage="existing_search",
            status="done",
            message=f"{len(ranked)} existing materials evaluated: {counts['feasible']} feasible, "
            f"{counts['undetermined']} undetermined, {counts['infeasible']} infeasible",
            details=counts,
        )
        return self.result(
            summary,
            Artifact(name="existing-material search", kind=EXISTING_SEARCH, content=result),
            sources=(SourceRef(kind="dataset", identifier="uav-materials-real/1"),),
        )


class AssessExistingCandidates(_WorkflowCapability):
    spec = CapabilitySpec(
        id="uavm.assess_existing_candidates",
        name="assess_existing_candidates",
        description="Apply the configured acceptance policy to the ranked existing materials and "
        "decide: use an existing material, enter composite design, or ask for missing "
        "information. Evidence gaps are never counted as material failures.",
        domain=DOMAIN,
        input_schema=NoInputs,
        output_schema=StepSummary,
        preconditions=("existing materials searched", "no decision yet"),
        effects=("sets progress.decision", "adds a uav.decision artifact"),
        tags=frozenset({"decision", "policy"}),
    )

    def __init__(self, policy: ExistingMaterialAcceptancePolicy) -> None:
        self.policy = policy

    def applicable(self, progress: WorkflowProgress) -> bool:
        return progress.existing_search == "done" and progress.decision == "pending"

    def execute(self, inputs: NoInputs, ctx: ExecutionContext) -> CapabilityResult:
        outcome: InterpretationOutcome = _latest(ctx.state, INTERPRETATION)
        search: ExistingSearchResult = _latest(ctx.state, EXISTING_SEARCH)
        decision = decide(
            search.evaluations,
            search.target,
            design_allowed=outcome.design_allowed,
            policy=self.policy,
        )
        summary = StepSummary(
            stage="decision",
            status=decision.outcome.value,
            message="; ".join(decision.reasons[:1]),
            details={"selected": decision.selected_candidate_id, **decision.counts},
        )
        return self.result(
            summary, Artifact(name="existing-material decision", kind=DECISION, content=decision)
        )


class DesignCompositeCandidate(_WorkflowCapability):
    spec = CapabilitySpec(
        id="uavm.design_composite_candidate",
        name="design_composite_candidate",
        description="Bounded inverse design: enumerate fibre volume fraction x symmetric "
        "stacking sequence in the validated carbon/epoxy laminate design space, predict each "
        "design with NASA-sourced micromechanics and classical lamination theory, and select "
        "the best by the unified evaluator. No language model generates materials.",
        domain=DOMAIN,
        input_schema=NoInputs,
        output_schema=StepSummary,
        preconditions=("the decision is to design a composite candidate",),
        effects=("sets progress.design_search", "adds a uav.design_search artifact"),
        tags=frozenset({"design", "inverse_design", "physics_model"}),
    )

    def __init__(self, config: DesignSearchConfig) -> None:
        self.config = config

    def applicable(self, progress: WorkflowProgress) -> bool:
        return progress.decision == "design" and progress.design_search == "pending"

    def execute(self, inputs: NoInputs, ctx: ExecutionContext) -> CapabilityResult:
        search: ExistingSearchResult = _latest(ctx.state, EXISTING_SEARCH)
        result = search_designs(search.target, self.config)
        best = result.rows[0] if result.rows else None
        summary = StepSummary(
            stage="design_search",
            status=result.status,
            message=result.message,
            details={
                "designs": len(result.rows),
                "best_layup": best.layup if best else None,
                "best_fiber_volume_fraction": best.fiber_volume_fraction if best else None,
            },
        )
        return self.result(
            summary,
            Artifact(name="composite design search", kind=DESIGN_SEARCH, content=result),
            sources=(SourceRef(kind="model", identifier=result.model),),
        )


class EvaluateDesignedCandidate(_WorkflowCapability):
    spec = CapabilitySpec(
        id="uavm.evaluate_designed_candidate",
        name="evaluate_designed_candidate",
        description="Evaluate the best designed composite against the user's full requirements "
        "in the same evaluation context as the existing materials, and pair it with the "
        "reference existing material for a side-by-side comparison.",
        domain=DOMAIN,
        input_schema=NoInputs,
        output_schema=StepSummary,
        preconditions=("a designed candidate exists", "it has not been evaluated yet"),
        effects=("sets progress.design_evaluation", "adds a uav.design_assessment artifact"),
        tags=frozenset({"evaluation", "comparison"}),
    )

    def applicable(self, progress: WorkflowProgress) -> bool:
        return (
            progress.design_search in ("designed", "no_feasible_design")
            and progress.design_evaluation == "pending"
        )

    def execute(self, inputs: NoInputs, ctx: ExecutionContext) -> CapabilityResult:
        search: ExistingSearchResult = _latest(ctx.state, EXISTING_SEARCH)
        decision: MaterialDecision = _latest(ctx.state, DECISION)
        design: DesignSearchResult = _latest(ctx.state, DESIGN_SEARCH)
        assert design.best is not None
        context = EvaluationContext.fit(search.target, real_candidates())
        designed = context.evaluate(design.best)
        reference, why = reference_existing(search, decision)
        assessment = DesignAssessment(
            designed=designed, reference_existing=reference, reference_reason=why
        )
        summary = StepSummary(
            stage="design_evaluation",
            status="done",
            message=f"designed candidate against the full target: {designed.feasibility.value}, "
            f"evidence {designed.evidence.classification.value}",
            details={"feasibility": designed.feasibility.value},
        )
        return self.result(
            summary,
            Artifact(
                name="designed candidate assessment", kind=DESIGN_ASSESSMENT, content=assessment
            ),
        )


class GenerateMaterialReport(_WorkflowCapability):
    spec = CapabilitySpec(
        id="uavm.generate_material_report",
        name="generate_material_report",
        description="Write the final engineering report (structured and Markdown) from the "
        "computed workflow state: requirements, existing-material search, decision, designed "
        "candidate, comparison, evidence, limitations and sources. No values are generated.",
        domain=DOMAIN,
        input_schema=NoInputs,
        output_schema=StepSummary,
        preconditions=("every step the decision requires is complete", "no report yet"),
        effects=("sets progress.report", "adds uav.report and uav.report_markdown artifacts"),
        tags=frozenset({"report"}),
    )

    def applicable(self, progress: WorkflowProgress) -> bool:
        if progress.report != "pending":
            return False
        return (
            progress.requirements in ("empty", "failed")
            or progress.decision in ("use_existing", "needs_information", "no_suitable_existing")
            or progress.design_search == "no_supported_requirements"
            or progress.design_evaluation == "done"
        )

    def execute(self, inputs: NoInputs, ctx: ExecutionContext) -> CapabilityResult:
        from domains.uav_materials.reporting import StepRecord, build_report, render_markdown

        router_id = ctx.state.history[-1].decision.router_id if ctx.state.history else None
        this_step = StepRecord(
            step=ctx.step,
            capability_id=self.spec.id,
            intent="invoke",
            router_id=router_id,
            success=None,
        )
        report = build_report(ctx.state, current_step=this_step)
        markdown = render_markdown(report)
        summary = StepSummary(
            stage="report", status="done", message=f"report written ({report.status})"
        )
        return self.result(
            summary,
            Artifact(name="material workflow report", kind=REPORT, content=report),
            Artifact(
                name="material workflow report (Markdown)",
                kind=REPORT_MARKDOWN,
                content=markdown,
                media_type="text/markdown",
            ),
        )


def workflow_capabilities(
    interpreter: RequirementInterpreter | None = None,
    policy: ExistingMaterialAcceptancePolicy | None = None,
    design_config: DesignSearchConfig | None = None,
) -> list[Capability]:
    return [
        InterpretRequirements(interpreter or RuleBasedInterpreter()),
        SearchExistingMaterials(),
        AssessExistingCandidates(policy or ExistingMaterialAcceptancePolicy()),
        DesignCompositeCandidate(design_config or DesignSearchConfig()),
        EvaluateDesignedCandidate(),
        GenerateMaterialReport(),
    ]


WORKFLOW_ORDER = tuple(c.spec.id for c in workflow_capabilities())


# -- reducer, evaluator, reference routing policy -------------------------------------------------

_STAGE_FIELD = {
    "requirements": "requirements",
    "existing_search": "existing_search",
    "decision": "decision",
    "design_search": "design_search",
    "design_evaluation": "design_evaluation",
    "report": "report",
}


def reduce_progress(state: WorkflowState, observation: Observation) -> WorkflowState:
    """Record a workflow step's status in the compact progress field."""
    if not isinstance(state, UAVMaterialsState) or state.request_text is None:
        return state
    result = observation.result
    if not observation.success or not isinstance(result, StepSummary):
        return state
    field = _STAGE_FIELD.get(result.stage)
    if field is None:
        return state
    progress = state.progress.model_copy(
        update={field: result.status, "summary": f"{result.stage}: {result.message}"[:300]}
    )
    return state.evolve(progress=progress)


class ProgressReducer(StateReducer):
    def reduce(self, state: WorkflowState, observation: Observation) -> WorkflowState:
        return reduce_progress(state, observation)


def evaluate_workflow(state: UAVMaterialsState, evaluator_id: str) -> EvaluationResult:
    p = state.progress
    stages = [p.requirements, p.existing_search, p.decision, p.design_search, p.report]
    done = p.report == "done"
    progress = sum(s != "pending" for s in stages) / len(stages)
    return EvaluationResult(
        evaluator_id=evaluator_id,
        step=state.step,
        goal_progress=1.0 if done else progress,
        remaining_gaps=() if done else ("final engineering report not yet written",),
        recommendation=ControlAction.TERMINATE_SUCCESS if done else ControlAction.CONTINUE,
        rationale="report written" if done else f"in progress ({p.summary or 'not started'})",
    )


def reference_routing_payload(request: Any) -> dict[str, Any]:
    """Deterministic reference routing policy for OFFLINE_DEMO runs.

    Takes a RoutingRequest (duck-typed: ``capability_ids``) and invokes the first
    offered workflow capability in :data:`WORKFLOW_ORDER`; finishes when none is
    offered. Preconditions do the rest. This is a scripted stand-in for a routing
    model, labelled as such, and says nothing about live routing quality.
    """
    offered = set(request.capability_ids)
    for capability_id in WORKFLOW_ORDER:
        if capability_id in offered:
            return {
                "action": "invoke",
                "capability_id": capability_id,
                "inputs": {},
                "reason": "offline demo reference policy: next workflow step whose "
                "preconditions hold",
                "confidence": None,
            }
    return {
        "action": "finish",
        "capability_id": None,
        "inputs": {},
        "reason": "offline demo reference policy: no workflow step is available",
        "confidence": None,
    }
