"""Shared fixtures for routing tests: a tiny generic workflow with schema'd capabilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from jevpilot import (
    CapabilityRegistry,
    CapabilityResult,
    ControlAction,
    EvaluationResult,
    Evaluator,
    Goal,
    StateEffects,
    WorkflowState,
)
from tests.conftest import make_capability


class TextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class Wrote(Evaluator):
    """Success once context.written is set."""

    evaluator_id = "wrote"

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        done = "written" in state.context
        return EvaluationResult(
            evaluator_id=self.evaluator_id,
            recommendation=ControlAction.TERMINATE_SUCCESS if done else ControlAction.CONTINUE,
        )


def write_fn(inputs: Any, ctx: Any) -> CapabilityResult:
    return CapabilityResult(
        output=inputs.text, effects=StateEffects(context_updates={"written": inputs.text})
    )


def registry_with_writer() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register(make_capability("t.write", write_fn, input_schema=TextInput))
    reg.register(make_capability("t.noop"))
    return reg


def new_state() -> WorkflowState:
    return WorkflowState(goal=Goal(description="write hello", parameters={"text": "hello"}))


SPECS = registry_with_writer().list()
