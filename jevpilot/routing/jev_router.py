"""Jev-backed router.

Jev is one routing policy among several, not a dependency of the framework.
This router adds nothing Jev-specific to the loop. It sends the canonical
:class:`RoutingRequest` to a :class:`RoutingModelAdapter`, and the adapter
owns all Jev-specific formatting and transport. The decision goes through
the same strict parsing and validation as every other model-backed router.
Swapping ``JevRouter`` for ``LLMRouter``, ``RuleRouter`` or a future learned
router requires no change anywhere else.
"""

from __future__ import annotations

from typing import Any

from jevpilot.routing.contracts import RoutingModelAdapter, RoutingModelResponse
from jevpilot.routing.model_router import ModelRouter
from jevpilot.routing.request import RoutingRequest


class JevRouter(ModelRouter):
    router_id = "jev_router"

    def __init__(self, adapter: RoutingModelAdapter, **options: Any) -> None:
        super().__init__(**options)
        self.adapter = adapter

    def _infer(self, request: RoutingRequest) -> RoutingModelResponse:
        return self.adapter.infer(request, timeout_s=self.timeout_s)

    def adapter_config(self) -> dict[str, Any]:
        return self.adapter.describe()
