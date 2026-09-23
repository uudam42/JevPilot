"""Structured, machine-readable execution tracing.

The controller emits :class:`TraceEvent` records to any number of
:class:`TraceSink` implementations, so a run can be reconstructed step by step.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from jevpilot.core._base import FrozenModel, to_jsonable, utcnow


class TraceEventType(StrEnum):
    WORKFLOW_STARTED = "workflow_started"
    PLAN_CREATED = "plan_created"
    STATE_SNAPSHOT = "state_snapshot"
    ROUTING_DECISION = "routing_decision"
    ROUTING_FAILURE = "routing_failure"
    OBSERVATION = "observation"
    EVALUATION = "evaluation"
    CONTROL_DECISION = "control_decision"
    ERROR = "error"
    WORKFLOW_COMPLETED = "workflow_completed"


class TraceEvent(FrozenModel):
    seq: int
    workflow_id: str
    step: int
    type: TraceEventType
    timestamp: datetime = Field(default_factory=utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceSink(Protocol):
    def write(self, event: TraceEvent) -> None: ...


class InMemoryTraceSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def write(self, event: TraceEvent) -> None:
        self.events.append(event)


class JsonlTraceSink:
    """Appends one JSON object per event to a file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: TraceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")


class LoggingTraceSink:
    """Emits each event as a single JSON log line via :mod:`logging`."""

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO) -> None:
        self.logger = logger or logging.getLogger("jevpilot.trace")
        self.level = level

    def write(self, event: TraceEvent) -> None:
        self.logger.log(self.level, event.model_dump_json())


class Tracer:
    """Assigns sequence numbers and fans events out to sinks."""

    def __init__(self, sinks: list[TraceSink] | tuple[TraceSink, ...] = ()) -> None:
        self._sinks = list(sinks)
        self._seq = 0

    def emit(self, workflow_id: str, step: int, type: TraceEventType, **payload: Any) -> TraceEvent:
        event = TraceEvent(
            seq=self._seq,
            workflow_id=workflow_id,
            step=step,
            type=type,
            payload=to_jsonable(payload),
        )
        self._seq += 1
        for sink in self._sinks:
            sink.write(event)
        return event


def read_jsonl_trace(path: str | Path) -> list[TraceEvent]:
    """Load a trace written by :class:`JsonlTraceSink`."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [TraceEvent.model_validate(json.loads(line)) for line in lines if line.strip()]
