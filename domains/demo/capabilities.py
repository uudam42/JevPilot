"""Trivial arithmetic capabilities (architecture demo only)."""

from __future__ import annotations

from pydantic import BaseModel

from domains.demo.state import NumberState
from jevpilot import (
    Candidate,
    Capability,
    CapabilityResult,
    CapabilitySpec,
    ExecutionContext,
    StateEffects,
    Uncertainty,
    WorkflowState,
)

DOMAIN = "arithmetic"


class AddInput(BaseModel):
    amount: float


class MultiplyInput(BaseModel):
    factor: float


class NoInput(BaseModel):
    pass


class ValueOutput(BaseModel):
    value: float


class CompareOutput(BaseModel):
    value: float
    target: float
    equal: bool
    difference: float


class _NumberCapability(Capability):
    def is_applicable(self, state: WorkflowState) -> bool:
        return isinstance(state, NumberState)

    @staticmethod
    def number_state(ctx: ExecutionContext) -> NumberState:
        assert isinstance(ctx.state, NumberState)
        return ctx.state


class Add(_NumberCapability):
    spec = CapabilitySpec(
        id="arith.add",
        name="add",
        description="Add an amount to the current value.",
        domain=DOMAIN,
        input_schema=AddInput,
        output_schema=ValueOutput,
        tags=frozenset({"transform"}),
        effects=("value := value + amount",),
        cost_estimate=1.0,
    )

    def execute(self, inputs: AddInput, ctx: ExecutionContext) -> CapabilityResult:
        value = self.number_state(ctx).value + inputs.amount
        return CapabilityResult(output=ValueOutput(value=value), uncertainty=Uncertainty.certain())


class Multiply(_NumberCapability):
    spec = CapabilitySpec(
        id="arith.multiply",
        name="multiply",
        description="Multiply the current value by a factor.",
        domain=DOMAIN,
        input_schema=MultiplyInput,
        output_schema=ValueOutput,
        tags=frozenset({"transform"}),
        preconditions=("value != 0",),
        effects=("value := value * factor",),
        cost_estimate=1.0,
    )

    def is_applicable(self, state: WorkflowState) -> bool:
        return super().is_applicable(state) and getattr(state, "value", 0) != 0

    def execute(self, inputs: MultiplyInput, ctx: ExecutionContext) -> CapabilityResult:
        value = self.number_state(ctx).value * inputs.factor
        return CapabilityResult(output=ValueOutput(value=value), uncertainty=Uncertainty.certain())


class Compare(_NumberCapability):
    spec = CapabilitySpec(
        id="arith.compare",
        name="compare",
        description="Compare the current value with the target; proposes a candidate if equal.",
        domain=DOMAIN,
        input_schema=NoInput,
        output_schema=CompareOutput,
        tags=frozenset({"check"}),
        effects=("records comparison", "adds candidate solution when equal"),
        cost_estimate=0.1,
    )

    def execute(self, inputs: NoInput, ctx: ExecutionContext) -> CapabilityResult:
        s = self.number_state(ctx)
        out = CompareOutput(
            value=s.value, target=s.target, equal=s.value == s.target, difference=s.target - s.value
        )
        candidates = (Candidate(content={"value": s.value}, score=1.0),) if out.equal else ()
        return CapabilityResult(
            output=out,
            effects=StateEffects(candidates=candidates),
            uncertainty=Uncertainty.certain(),
        )


def all_capabilities() -> list[Capability]:
    return [Add(), Multiply(), Compare()]
