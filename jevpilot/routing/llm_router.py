"""LLM-backed router: routing request → prompt → text completion → validated decision."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jevpilot.exceptions import MalformedRoutingDecisionError
from jevpilot.routing.contracts import LLMAdapter, LLMCompletion, LLMPrompt, RoutingModelResponse
from jevpilot.routing.model_router import ModelRouter
from jevpilot.routing.prompts import build_routing_prompt
from jevpilot.routing.request import RoutingRequest

PromptBuilder = Callable[[RoutingRequest], LLMPrompt]


class LLMRouter(ModelRouter):
    """Routes with any text-completion model behind an :class:`LLMAdapter`.

    The adapter is injected, so tests and benchmarks run offline with
    :class:`~jevpilot.adapters.llm.FakeLLMAdapter`. The prompt comes from
    :mod:`jevpilot.routing.prompts` unless ``prompt_builder`` overrides it.
    Free text never reaches the executor; it is parsed and validated first.
    """

    router_id = "llm_router"

    def __init__(
        self,
        adapter: LLMAdapter,
        *,
        prompt_builder: PromptBuilder = build_routing_prompt,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.adapter = adapter
        self.prompt_builder = prompt_builder

    def _infer(self, request: RoutingRequest) -> RoutingModelResponse:
        prompt = self.prompt_builder(request)
        completion = self.adapter.complete(prompt, timeout_s=self.timeout_s)
        if not isinstance(completion, LLMCompletion):
            raise MalformedRoutingDecisionError(
                f"LLM adapter returned {type(completion).__name__}, not LLMCompletion"
            )
        return RoutingModelResponse(
            output=completion.text,
            model=completion.model,
            usage=completion.usage,
            metadata={"stop_reason": completion.stop_reason, "prompt_version": prompt.version},
        )

    def adapter_config(self) -> dict[str, Any]:
        return self.adapter.describe()

    def config(self) -> dict[str, Any]:
        builder = getattr(self.prompt_builder, "__qualname__", repr(self.prompt_builder))
        return {**super().config(), "prompt_builder": builder}
