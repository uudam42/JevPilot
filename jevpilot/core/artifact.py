"""Generic artifacts: any non-trivial output of a capability."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from jevpilot.core._base import FrozenModel, new_id
from jevpilot.core.provenance import Provenance
from jevpilot.core.uncertainty import Uncertainty


class ArtifactKind(StrEnum):
    """Common artifact kinds. Any string is accepted; these are conventions."""

    JSON = "json"
    TEXT = "text"
    TABLE = "table"
    DATASET = "dataset"
    PLOT = "plot"
    SIMULATION_RESULT = "simulation_result"
    FILE = "file"
    MODEL = "model"
    REPORT = "report"
    OBJECT = "object"


class Artifact(FrozenModel):
    """A produced object, stored inline (``content``) or by reference (``uri``).

    ``provenance`` is optional at construction time because capabilities build
    artifacts before the executor knows execution details; the executor stamps
    it, and :class:`~jevpilot.orchestration.state_manager.StateManager` refuses
    artifacts that reach state without it.
    """

    id: str = Field(default_factory=lambda: new_id("art"))
    name: str
    kind: str = ArtifactKind.OBJECT
    content: Any = None
    uri: str | None = None
    media_type: str | None = None
    schema_ref: str | None = None
    uncertainty: Uncertainty | None = None
    provenance: Provenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_traceable(self) -> bool:
        return self.provenance is not None
