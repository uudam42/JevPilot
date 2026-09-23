"""Candidate solutions proposed during a workflow."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from jevpilot.core._base import FrozenModel, new_id
from jevpilot.core.provenance import Provenance
from jevpilot.core.uncertainty import Uncertainty


class Candidate(FrozenModel):
    """A proposed answer/design/solution. ``content`` is domain-defined."""

    id: str = Field(default_factory=lambda: new_id("cand"))
    content: Any
    score: float | None = None
    uncertainty: Uncertainty | None = None
    provenance: Provenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
