"""Generic, domain-neutral representation of uncertainty and confidence."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from jevpilot.core._base import FrozenModel


class UncertaintyKind(StrEnum):
    """Broad source of uncertainty. Producers may refine it via ``metadata``."""

    UNSPECIFIED = "unspecified"
    MODEL = "model"  # confidence of a model/estimator
    MEASUREMENT = "measurement"  # instrument or observation noise
    DATA_QUALITY = "data_quality"
    MISSING_INFORMATION = "missing_information"
    CONFLICT = "conflict"  # disagreeing observations


class Uncertainty(FrozenModel):
    """How much a piece of information should be trusted.

    Every field is optional so producers report only what they know. The
    *meaning* of ``interval`` and ``dispersion`` (e.g. a 95% interval, a
    standard deviation) is defined by the producer and recorded in
    ``metadata``; the core never interprets them.
    """

    kind: UncertaintyKind = UncertaintyKind.UNSPECIFIED
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    interval: tuple[float, float] | None = None
    dispersion: float | None = Field(default=None, ge=0.0)
    missing: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def certain(cls) -> Uncertainty:
        """Information that is exact by construction (e.g. a deterministic computation)."""
        return cls(kind=UncertaintyKind.MODEL, confidence=1.0)

    @classmethod
    def unknown(cls, notes: str | None = None) -> Uncertainty:
        """Nothing is known about the reliability of the information."""
        return cls(notes=notes)
