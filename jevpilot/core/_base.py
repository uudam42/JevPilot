"""Shared base model and small helpers for core data types."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python


class FrozenModel(BaseModel):
    """Immutable base model.

    Instances cannot be mutated through attribute assignment; changes are made
    by producing new instances (``model_copy(update=...)``). Container fields
    are tuples wherever ordering matters so accidental in-place mutation is
    impossible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


def new_id(prefix: str) -> str:
    """Return a short, unique, prefixed identifier (e.g. ``obs_1a2b3c4d5e6f``)."""
    return f"{prefix}_{uuid4().hex[:12]}"


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


def to_jsonable(value: Any) -> Any:
    """Convert arbitrary values (including Pydantic models) into JSON-safe data."""
    return to_jsonable_python(value, fallback=repr)


def stable_digest(value: Any) -> str:
    """Deterministic SHA-256 digest of a JSON-serialisable view of ``value``."""
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
