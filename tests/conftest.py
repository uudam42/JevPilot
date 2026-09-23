"""Shared test helpers. Uses only JevPilot's public API plus throwaway capabilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel

from jevpilot import (
    CapabilityRegistry,
    CapabilitySpec,
    ExecutionContext,
    FunctionCapability,
    Goal,
    WorkflowState,
)


class EchoInput(BaseModel):
    text: str


def make_capability(
    cid: str,
    fn: Callable[[Any, ExecutionContext], Any] = lambda inputs, ctx: None,
    *,
    applicable: Callable[[WorkflowState], bool] | None = None,
    **spec: Any,
) -> FunctionCapability:
    spec.setdefault("name", cid)
    spec.setdefault("description", f"test capability {cid}")
    return FunctionCapability(CapabilitySpec(id=cid, **spec), fn, applicable=applicable)


@pytest.fixture
def state() -> WorkflowState:
    return WorkflowState(goal=Goal(description="test goal"))


@pytest.fixture
def registry() -> CapabilityRegistry:
    return CapabilityRegistry()
