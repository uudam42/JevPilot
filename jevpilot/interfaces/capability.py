"""Capabilities: anything the system knows how to do.

Agents, tools, models, simulators, databases and optimisers are all modelled
as capabilities. The core treats every capability identically: it reads the
:class:`CapabilitySpec` metadata, checks applicability, validates inputs and
calls :meth:`Capability.execute`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from jevpilot.core import (
    Artifact,
    FrozenModel,
    SourceRef,
    StateEffects,
    Uncertainty,
    WorkflowState,
)


class CapabilitySpec(FrozenModel):
    """Structured, router-facing description of a capability.

    Routers only ever see specs, never executable capability objects, which
    enforces the router/executor separation.
    """

    id: str  # globally unique; convention "<domain>.<name>"
    name: str
    description: str
    version: str = "0.1.0"
    domain: str | None = None
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    tags: frozenset[str] = frozenset()
    preconditions: tuple[str, ...] = ()  # human/LLM-readable; executable check is is_applicable
    effects: tuple[str, ...] = ()
    side_effects: bool = False
    cost_estimate: float | None = None
    latency_estimate: float | None = None  # seconds
    requirements: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable description, including JSON Schemas, for routers."""
        data = self.model_dump(mode="json", exclude={"input_schema", "output_schema"})
        data["tags"] = sorted(self.tags)
        data["input_schema"] = self.input_schema.model_json_schema() if self.input_schema else None
        data["output_schema"] = (
            self.output_schema.model_json_schema() if self.output_schema else None
        )
        return data


class ExecutionContext(FrozenModel):
    """Read-only view a capability receives during execution."""

    workflow_id: str
    step: int
    execution_id: str
    decision_id: str | None
    state: WorkflowState


class CapabilityResult(FrozenModel):
    """What a capability returns. Returning any other value is shorthand for
    ``CapabilityResult(output=value)``."""

    output: Any = None
    artifacts: tuple[Artifact, ...] = ()
    effects: StateEffects = Field(default_factory=StateEffects)
    uncertainty: Uncertainty | None = None
    sources: tuple[SourceRef, ...] = ()
    derived_from: tuple[str, ...] = ()  # upstream provenance ids this result depends on
    metadata: dict[str, Any] = Field(default_factory=dict)


class Capability(ABC):
    """Base class for executable capabilities.

    Subclasses provide :attr:`spec` and :meth:`execute`. Failures should be
    signalled by raising (ideally :class:`~jevpilot.exceptions.CapabilityError`);
    the executor converts exceptions into failed observations.
    """

    spec: CapabilitySpec

    @property
    def id(self) -> str:
        return self.spec.id

    def is_applicable(self, state: WorkflowState) -> bool:
        """Executable precondition check. Defaults to always applicable."""
        return True

    @abstractmethod
    def execute(self, inputs: Any, ctx: ExecutionContext) -> CapabilityResult | Any:
        """Run the capability.

        ``inputs`` is an instance of ``spec.input_schema`` when one is declared,
        otherwise the raw ``dict`` from the routing decision.
        """


ExecuteFn = Callable[[Any, ExecutionContext], "CapabilityResult | Any"]
ApplicableFn = Callable[[WorkflowState], bool]


class FunctionCapability(Capability):
    """Adapter turning plain functions into a capability."""

    def __init__(
        self,
        spec: CapabilitySpec,
        fn: ExecuteFn,
        *,
        applicable: ApplicableFn | None = None,
    ) -> None:
        self.spec = spec
        self._fn = fn
        self._applicable = applicable

    def is_applicable(self, state: WorkflowState) -> bool:
        return self._applicable(state) if self._applicable else True

    def execute(self, inputs: Any, ctx: ExecutionContext) -> CapabilityResult | Any:
        return self._fn(inputs, ctx)
