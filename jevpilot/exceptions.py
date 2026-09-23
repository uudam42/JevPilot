"""Exception hierarchy for JevPilot.

All framework errors derive from :class:`JevPilotError` so callers can catch
framework failures without catching unrelated exceptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class JevPilotError(Exception):
    """Base class for all JevPilot errors."""


class RegistryError(JevPilotError):
    """Raised for invalid registry operations."""


class DuplicateCapabilityError(RegistryError):
    """A capability with the same id is already registered."""


class CapabilityNotFoundError(RegistryError):
    """No capability with the requested id is registered."""


class DomainLoadError(RegistryError):
    """A domain module could not be imported or registered."""


class CapabilityError(JevPilotError):
    """Raised by a capability to signal a controlled execution failure.

    Capabilities may raise this (instead of an arbitrary exception) to state
    explicitly whether retrying could help.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class ProvenanceError(JevPilotError):
    """Raised when data would enter workflow state without provenance."""


class RoutingError(JevPilotError):
    """Raised when a router fails to produce a valid decision.

    Routing failures are explicit: the controller records them (trace and
    ``state.history``) and hands them to the control policy, and a
    :class:`~jevpilot.routing.fallback.FallbackRouter` may catch them. Any
    other exception escaping a router is treated as an orchestration bug.

    ``attempts`` holds the ``RoutingAttempt`` records gathered before the
    failure so traces keep the request, latency and usage of failed calls.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        attempts: Sequence[Any] = (),
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.details: dict[str, Any] = dict(details or {})
        self.attempts: tuple[Any, ...] = tuple(attempts)


class InvalidCapabilityError(RoutingError):
    """The decision names a capability that was not offered to the router."""


class InvalidRoutingInputError(RoutingError):
    """The decision's inputs do not satisfy the capability's input schema."""


class MalformedRoutingDecisionError(RoutingError):
    """The router (or its model) produced output that is not a valid decision."""


class RouterTimeoutError(RoutingError):
    """The router did not decide within its time budget."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class RouterAdapterError(RoutingError):
    """The model/provider adapter behind a router failed."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class LowConfidenceRoutingError(RoutingError):
    """The decision's confidence is below a threshold the router was configured with."""
