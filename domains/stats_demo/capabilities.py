"""Sampling/summary capabilities (architecture demo only).

Deliberately unlike the arithmetic domain: no state subclass, artifacts as
the main output, uncertainty on every estimate, and explicit provenance links
between artifacts.
"""

from __future__ import annotations

import math
import platform
import random
import statistics

from pydantic import BaseModel, Field

from jevpilot import (
    Artifact,
    ArtifactKind,
    Candidate,
    Capability,
    CapabilityResult,
    CapabilitySpec,
    ExecutionContext,
    SourceRef,
    StateEffects,
    Uncertainty,
    UncertaintyKind,
    WorkflowState,
)

DOMAIN = "stats"


class SampleInput(BaseModel):
    n: int = Field(ge=2)
    seed: int = 0


class NoInput(BaseModel):
    pass


class Summary(BaseModel):
    n: int
    mean: float
    std: float
    stderr: float
    ci_low: float
    ci_high: float

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2


class SampleData(Capability):
    spec = CapabilitySpec(
        id="stats.sample",
        name="sample",
        description="Draw n noisy measurements from the (simulated) process in the goal.",
        domain=DOMAIN,
        input_schema=SampleInput,
        tags=frozenset({"acquire", "data"}),
        effects=("adds a dataset artifact",),
        cost_estimate=0.1,  # per sample
    )

    def execute(self, inputs: SampleInput, ctx: ExecutionContext) -> CapabilityResult:
        params = ctx.state.goal.parameters
        rng = random.Random(inputs.seed)
        data = [rng.gauss(params["true_mean"], params["noise"]) for _ in range(inputs.n)]
        dataset = Artifact(
            name=f"sample_n{inputs.n}",
            kind=ArtifactKind.DATASET,
            content=data,
            media_type="application/json",
            metadata={"n": inputs.n},
        )
        return CapabilityResult(
            output={"n": inputs.n},
            artifacts=(dataset,),
            sources=(
                SourceRef(
                    kind="generator",
                    identifier="random.Random.gauss",
                    version=platform.python_version(),
                    metadata={"seed": inputs.seed},
                ),
            ),
            effects=StateEffects(context_updates={"sample_size": inputs.n}),
        )


class Summarize(Capability):
    spec = CapabilitySpec(
        id="stats.summarize",
        name="summarize",
        description="Estimate the mean of the latest dataset with a 95% interval.",
        domain=DOMAIN,
        input_schema=NoInput,
        output_schema=Summary,
        tags=frozenset({"analyze"}),
        preconditions=("a dataset artifact exists",),
        effects=("adds candidate estimate", "updates uncertainty['mean']"),
    )

    def is_applicable(self, state: WorkflowState) -> bool:
        return state.latest_artifact(ArtifactKind.DATASET) is not None

    def execute(self, inputs: NoInput, ctx: ExecutionContext) -> CapabilityResult:
        dataset = ctx.state.latest_artifact(ArtifactKind.DATASET)
        assert dataset is not None and dataset.provenance is not None
        data: list[float] = dataset.content
        mean, std = statistics.fmean(data), statistics.stdev(data)
        stderr = std / math.sqrt(len(data))
        summary = Summary(
            n=len(data),
            mean=mean,
            std=std,
            stderr=stderr,
            ci_low=mean - 1.96 * stderr,
            ci_high=mean + 1.96 * stderr,
        )
        unc = Uncertainty(
            kind=UncertaintyKind.MEASUREMENT,
            interval=(summary.ci_low, summary.ci_high),
            dispersion=stderr,
            metadata={"interval": "95% normal approximation", "dispersion": "standard error"},
        )
        return CapabilityResult(
            output=summary,
            uncertainty=unc,
            derived_from=(dataset.provenance.id,),
            effects=StateEffects(
                context_updates={"summary_of": dataset.id, "ci_half_width": summary.half_width},
                candidates=(Candidate(content={"mean": mean}, uncertainty=unc),),
                uncertainty_updates={"mean": unc},
            ),
        )


class WriteReport(Capability):
    spec = CapabilitySpec(
        id="stats.report",
        name="report",
        description="Write a short report for the latest estimate.",
        domain=DOMAIN,
        input_schema=NoInput,
        tags=frozenset({"report"}),
        preconditions=("a candidate estimate exists",),
        effects=("adds a report artifact",),
    )

    def is_applicable(self, state: WorkflowState) -> bool:
        return bool(state.candidate_solutions)

    def execute(self, inputs: NoInput, ctx: ExecutionContext) -> CapabilityResult:
        cand = ctx.state.candidate_solutions[-1]
        assert cand.provenance is not None
        unc = cand.uncertainty
        text = f"Estimated mean {cand.content['mean']:.3f}"
        if unc and unc.interval:
            text += f" (95% CI {unc.interval[0]:.3f} to {unc.interval[1]:.3f})"
        report = Artifact(name="estimate_report", kind=ArtifactKind.REPORT, content=text)
        return CapabilityResult(
            output=text,
            artifacts=(report,),
            derived_from=(cand.provenance.id,),
            effects=StateEffects(
                context_updates={"report_for": ctx.state.context.get("summary_of")}
            ),
        )


def all_capabilities() -> list[Capability]:
    return [SampleData(), Summarize(), WriteReport()]
