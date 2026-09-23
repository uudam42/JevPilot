"""Offline stand-in for a Jev routing model. No network, no model, fully deterministic."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from jevpilot.adapters.scripting import Policy, Script, ScriptItem
from jevpilot.core import ModelUsage
from jevpilot.routing.contracts import LLMCompletion, RoutingModelResponse
from jevpilot.routing.request import RoutingRequest


class FakeJevAdapter:
    """Implements :class:`~jevpilot.routing.contracts.RoutingModelAdapter` from a script or policy.

    ``requests`` keeps every request received, for inspection in tests.
    ``usage`` is reported only if given: fakes never make up token counts.
    """

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        *,
        policy: Policy | None = None,
        on_exhausted: str = "error",
        model: str = "fake-jev",
        usage: ModelUsage | None = None,
        label: str | None = None,
    ) -> None:
        self._script = Script(script, policy=policy, on_exhausted=on_exhausted)
        self.model = model
        self.usage = usage
        self.label = label
        self.requests: list[RoutingRequest] = []

    def infer(
        self, request: RoutingRequest, *, timeout_s: float | None = None
    ) -> RoutingModelResponse:
        self.requests.append(request)
        item = self._script.next(request)
        if isinstance(item, BaseException):
            raise copy.copy(item)  # fresh instance: scripts may reuse one exception
        if isinstance(item, RoutingModelResponse):
            return item
        if isinstance(item, LLMCompletion):
            return RoutingModelResponse(output=item.text, model=item.model, usage=item.usage)
        return RoutingModelResponse(output=item, model=self.model, usage=self.usage)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "fake",
            "kind": "jev",
            "model": self.model,
            "label": self.label,
            **self._script.describe(),
        }
