"""Lightweight provenance records.

Every observation, artifact and candidate solution that enters workflow state
carries a :class:`Provenance` describing *what produced it, from which inputs,
when, and from which upstream sources*.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from jevpilot.core._base import FrozenModel, new_id, utcnow


class SourceRef(FrozenModel):
    """Reference to an external origin of information.

    ``kind`` is free-form (e.g. ``"tool"``, ``"model"``, ``"database"``,
    ``"dataset"``, ``"human"``, ``"publication"``) so domains can describe
    sources without the core enumerating them.
    """

    kind: str
    identifier: str
    version: str | None = None
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Provenance(FrozenModel):
    """Where a piece of information came from."""

    id: str = Field(default_factory=lambda: new_id("prov"))
    workflow_id: str | None = None
    step: int | None = None
    execution_id: str | None = None
    capability_id: str | None = None
    capability_version: str | None = None
    domain: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    inputs_digest: str | None = None
    sources: tuple[SourceRef, ...] = ()
    derived_from: tuple[str, ...] = ()  # ids of upstream provenance records
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
