"""Artificial "records pipeline" domain for routing benchmarks.

Goal: publish a validated summary of the records at a source, on a channel.
It needs no specialist knowledge. It exists to exercise routing patterns:
sequential dependencies, preconditions, redundant and unnecessary
capabilities, retry after a transient failure, replanning after an
observation, missing information and termination.

Goal parameters (all optional): ``source`` (missing → information only a
human can give), ``flaky`` (first fetch fails transiently), ``dirty``
(validation fails until the data is cleaned). ``success_criteria.channel``
names the channel to publish on.

Context conventions (shared with the decision fixtures): ``raw`` (id of the
fetched dataset artifact), ``parsed``, ``validated`` (True/False/absent),
``cleaned``, ``summarized``, ``published`` (channel).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from jevpilot import (
    Artifact,
    ArtifactKind,
    Candidate,
    Capability,
    CapabilityResult,
    CapabilitySpec,
    ControlAction,
    DomainModule,
    EvaluationResult,
    Evaluator,
    ExecutionContext,
    Goal,
    Rule,
    StateEffects,
    WorkflowState,
)
from jevpilot.exceptions import CapabilityError
from jevpilot.routing import RoutingRequest

DOMAIN = "pipeline"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceInput(_Strict):
    source: str


class NoInput(_Strict):
    pass


class SummarizeInput(_Strict):
    style: Literal["brief", "detailed"]


class PublishInput(_Strict):
    channel: Literal["internal", "public"]


def _spec(cid: str, description: str, **kw: Any) -> CapabilitySpec:
    return CapabilitySpec(id=f"pipe.{cid}", name=cid, description=description, domain=DOMAIN, **kw)


class Fetch(Capability):
    spec = _spec(
        "fetch",
        "Fetch raw records from a source location.",
        input_schema=SourceInput,
        tags=frozenset({"acquire"}),
        effects=("adds a raw dataset artifact", "sets context.raw"),
    )

    def execute(self, inputs: SourceInput, ctx: ExecutionContext) -> CapabilityResult:
        params = ctx.state.goal.parameters
        tried = any(o.capability_id == self.spec.id for o in ctx.state.observations)
        if params.get("flaky") and not tried:
            raise CapabilityError("source temporarily unavailable", retryable=True)
        return _fetched(inputs.source)


class FetchMirror(Capability):
    spec = _spec(
        "fetch_mirror",
        "Fetch the same raw records from a read-only mirror of the source.",
        input_schema=SourceInput,
        tags=frozenset({"acquire"}),
        effects=("adds a raw dataset artifact", "sets context.raw"),
    )

    def execute(self, inputs: SourceInput, ctx: ExecutionContext) -> CapabilityResult:
        return _fetched(inputs.source)


def _fetched(source: str) -> CapabilityResult:
    records = [{"id": i, "value": i * 10} for i in range(5)]
    art = Artifact(name="raw_records", kind=ArtifactKind.DATASET, content=records)
    return CapabilityResult(
        output={"records": len(records)},
        artifacts=(art,),
        effects=StateEffects(context_updates={"raw": art.id, "fetched_from": source}),
    )


class Parse(Capability):
    spec = _spec(
        "parse",
        "Parse the fetched raw records into structured rows.",
        input_schema=NoInput,
        preconditions=("raw records fetched",),
        effects=("sets context.parsed",),
    )

    def is_applicable(self, state: WorkflowState) -> bool:
        return "raw" in state.context

    def execute(self, inputs: NoInput, ctx: ExecutionContext) -> CapabilityResult:
        return CapabilityResult(
            output={"rows": 5}, effects=StateEffects(context_updates={"parsed": True})
        )


class Validate(Capability):
    spec = _spec(
        "validate",
        "Check parsed rows for consistency; reports issues if any are found.",
        input_schema=NoInput,
        preconditions=("records parsed",),
        effects=("sets context.validated to true or false",),
    )

    def is_applicable(self, state: WorkflowState) -> bool:
        return bool(state.context.get("parsed"))

    def execute(self, inputs: NoInput, ctx: ExecutionContext) -> CapabilityResult:
        dirty = ctx.state.goal.parameters.get("dirty") and not ctx.state.context.get("cleaned")
        issues = ["duplicate ids", "missing values"] if dirty else []
        return CapabilityResult(
            output={"valid": not issues, "issues": issues},
            effects=StateEffects(context_updates={"validated": not issues}),
        )


class Clean(Capability):
    spec = _spec(
        "clean",
        "Remove duplicate and incomplete rows. Validation must be re-run afterwards.",
        input_schema=NoInput,
        preconditions=("records parsed",),
        effects=("sets context.cleaned", "clears context.validated"),
    )

    def is_applicable(self, state: WorkflowState) -> bool:
        return bool(state.context.get("parsed"))

    def execute(self, inputs: NoInput, ctx: ExecutionContext) -> CapabilityResult:
        return CapabilityResult(
            output={"removed": 2},
            effects=StateEffects(context_updates={"cleaned": True, "validated": None}),
        )


class Summarize(Capability):
    spec = _spec(
        "summarize",
        "Summarize validated rows in the requested style.",
        input_schema=SummarizeInput,
        preconditions=("records validated",),
        effects=("adds a candidate summary", "sets context.summarized"),
    )

    def is_applicable(self, state: WorkflowState) -> bool:
        return state.context.get("validated") is True

    def execute(self, inputs: SummarizeInput, ctx: ExecutionContext) -> CapabilityResult:
        summary = {"style": inputs.style, "rows": 5}
        return CapabilityResult(
            output=summary,
            effects=StateEffects(
                context_updates={"summarized": True}, candidates=(Candidate(content=summary),)
            ),
        )


class Publish(Capability):
    spec = _spec(
        "publish",
        "Publish the summary on a channel. Irreversible.",
        input_schema=PublishInput,
        preconditions=("summary exists",),
        effects=("sets context.published",),
        side_effects=True,
    )

    def is_applicable(self, state: WorkflowState) -> bool:
        return bool(state.context.get("summarized"))

    def execute(self, inputs: PublishInput, ctx: ExecutionContext) -> CapabilityResult:
        return CapabilityResult(
            output={"channel": inputs.channel},
            effects=StateEffects(context_updates={"published": inputs.channel}),
        )


class CleanupCache(Capability):
    spec = _spec(
        "cleanup_cache",
        "Clear the local scratch cache. Never required to reach any goal.",
        input_schema=NoInput,
        tags=frozenset({"maintenance"}),
    )

    def execute(self, inputs: NoInput, ctx: ExecutionContext) -> CapabilityResult:
        return CapabilityResult(output={"cleared": True})


class PublishedEvaluator(Evaluator):
    evaluator_id = "pipeline.published"

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        wanted = state.goal.success_criteria.get("channel")
        published = state.context.get("published")
        base: dict[str, Any] = {"evaluator_id": self.evaluator_id, "step": state.step}
        if published is not None:
            ok = published == wanted
            return EvaluationResult(
                **base,
                goal_progress=1.0 if ok else 0.0,
                recommendation=(
                    ControlAction.TERMINATE_SUCCESS if ok else ControlAction.TERMINATE_FAILURE
                ),
                rationale=f"published on {published!r}, wanted {wanted!r}",
            )
        if state.context.get("validated") is False:
            return EvaluationResult(
                **base,
                goal_progress=0.4,
                remaining_gaps=("validation failed",),
                recommendation=ControlAction.REPLAN,
                rationale="validation found issues; approach must change",
            )
        done = sum(bool(state.context.get(k)) for k in ("raw", "parsed", "validated", "summarized"))
        return EvaluationResult(
            **base,
            goal_progress=done / 5,
            rationale="not yet published",
            remaining_gaps=("publish",),
        )


class PipelineDomain(DomainModule):
    name = DOMAIN
    version = "0.1.0"
    description = "Artificial benchmark domain: fetch, parse, validate, summarize, publish."

    def capabilities(self) -> Sequence[Capability]:
        return [
            Fetch(),
            FetchMirror(),
            Parse(),
            Validate(),
            Clean(),
            Summarize(),
            Publish(),
            CleanupCache(),
        ]

    def evaluators(self) -> Sequence[Evaluator]:
        return [PublishedEvaluator()]

    def new_workflow(self, **params: Any) -> WorkflowState:
        channel = params.pop("channel", "internal")
        goal = Goal(
            description=f"publish a validated summary of the records on the {channel} channel",
            parameters=params,
            success_criteria={"channel": channel},
        )
        return self.create_state(goal)


# -- routing heuristics (benchmark-local) --------------------------------------


def routing_rules() -> list[Rule]:
    """Deterministic baseline: later pipeline stages first, first match wins."""
    ctx = _Ctx
    return [
        Rule(
            "pipe.publish",
            when=lambda s: ctx(s).has("summarized") and not ctx(s).has("published"),
            inputs=lambda s: {"channel": s.goal.success_criteria.get("channel", "internal")},
            reason="summary ready; publish on the requested channel",
        ),
        Rule(
            "pipe.summarize",
            when=lambda s: s.context.get("validated") is True and not ctx(s).has("summarized"),
            inputs={"style": "brief"},
            reason="validated; summarize",
        ),
        Rule(
            "pipe.clean",
            when=lambda s: s.context.get("validated") is False and not ctx(s).has("cleaned"),
            reason="validation failed; clean before re-validating",
        ),
        Rule(
            "pipe.validate",
            when=lambda s: ctx(s).has("parsed") and s.context.get("validated") is None,
            reason="parsed; validate",
        ),
        Rule(
            "pipe.parse",
            when=lambda s: ctx(s).has("raw") and not ctx(s).has("parsed"),
            reason="raw records available; parse",
        ),
        Rule(
            "pipe.fetch",
            when=lambda s: "source" in s.goal.parameters and not ctx(s).has("raw"),
            inputs=lambda s: {"source": s.goal.parameters["source"]},
            reason="nothing fetched yet",
        ),
    ]


class _Ctx:
    def __init__(self, state: WorkflowState) -> None:
        self.context = state.context

    def has(self, key: str) -> bool:
        return bool(self.context.get(key))


def request_policy(request: RoutingRequest) -> dict[str, Any]:
    """Simulated model for this suite: reads only the JSON routing request.

    It mirrors :func:`routing_rules` so that, with fault injection switched
    off, fake model routers and the rule baseline agree. It is test
    scaffolding, not a model of how Jev or any LLM behaves.
    """
    ctx = request.state_summary.context
    goal = request.goal
    offered = set(request.capability_ids)

    def pick(cid: str, inputs: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "action": "invoke",
            "capability_id": cid,
            "inputs": inputs,
            "reason": reason,
            "confidence": 0.9,
        }

    if ctx.get("published"):
        return {
            "action": "finish",
            "capability_id": None,
            "inputs": {},
            "reason": "published",
            "confidence": 0.9,
        }
    if ctx.get("summarized") and "pipe.publish" in offered:
        channel = goal["success_criteria"].get("channel", "internal")
        return pick("pipe.publish", {"channel": channel}, "publish")
    if ctx.get("validated") is True and "pipe.summarize" in offered:
        return pick("pipe.summarize", {"style": "brief"}, "summarize")
    if ctx.get("validated") is False and not ctx.get("cleaned") and "pipe.clean" in offered:
        return pick("pipe.clean", {}, "clean after failed validation")
    if ctx.get("parsed") and ctx.get("validated") is None and "pipe.validate" in offered:
        return pick("pipe.validate", {}, "validate")
    if ctx.get("raw") and not ctx.get("parsed") and "pipe.parse" in offered:
        return pick("pipe.parse", {}, "parse")
    source = goal["parameters"].get("source")
    if not ctx.get("raw"):
        if source is None:
            return {
                "action": "ask_human",
                "capability_id": None,
                "inputs": {},
                "reason": "source location unknown",
                "confidence": 0.8,
            }
        return pick("pipe.fetch", {"source": source}, "fetch")
    return {
        "action": "finish",
        "capability_id": None,
        "inputs": {},
        "reason": "nothing left to do",
        "confidence": 0.5,
    }
