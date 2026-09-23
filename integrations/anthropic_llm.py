"""Claude (Anthropic Messages API) as an :class:`~jevpilot.routing.contracts.LLMAdapter`.

Install with ``pip install 'jevpilot[anthropic]'``. Credentials are resolved by
the SDK from the environment (``ANTHROPIC_API_KEY`` or an ``ant auth login``
profile) and are never passed through JevPilot. The model can be overridden
with ``JEVPILOT_ANTHROPIC_MODEL``.

Usage in a benchmark or workflow::

    from integrations.anthropic_llm import AnthropicLLMAdapter
    router = LLMRouter(AnthropicLLMAdapter(), timeout_s=60)

Token usage comes from the API response. Cost is estimated only when you pass
``price_per_mtok``; otherwise it is recorded as unknown (``None``).
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from jevpilot.core import ModelUsage
from jevpilot.exceptions import RouterAdapterError, RouterTimeoutError
from jevpilot.routing.contracts import LLMCompletion, LLMPrompt

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLMAdapter:
    """Sends the routing prompt as one Messages API request and returns its text.

    ``refusal_fallbacks`` (on by default) lets the API re-run a policy-declined
    request on Anthropic's recommended fallback model. The model that actually
    answered is reported in ``LLMCompletion.model``, so benchmark records show
    any substitution. Set it to ``False`` for strictly single-model experiments.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        effort: str | None = None,
        refusal_fallbacks: bool = True,
        price_per_mtok: tuple[float, float] | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or os.environ.get("JEVPILOT_ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.effort = effort
        self.refusal_fallbacks = refusal_fallbacks
        self.price_per_mtok = price_per_mtok
        if client is None:
            client = _sdk().Anthropic()
        self._client = client

    def complete(self, prompt: LLMPrompt, *, timeout_s: float | None = None) -> LLMCompletion:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": prompt.system,
            "messages": [{"role": "user", "content": prompt.user}],
        }
        if self.effort is not None:
            request["output_config"] = {"effort": self.effort}
        client = self._client.with_options(timeout=timeout_s) if timeout_s else self._client
        try:
            if self.refusal_fallbacks:
                response = client.beta.messages.create(
                    **request, betas=[FALLBACK_BETA], fallbacks="default"
                )
            else:
                response = client.messages.create(**request)
        except Exception as exc:
            if type(exc).__name__ == "APITimeoutError":
                raise RouterTimeoutError(f"Anthropic request timed out: {exc}") from exc
            raise RouterAdapterError(
                f"Anthropic request failed: {type(exc).__name__}: {exc}"
            ) from exc

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise RouterAdapterError(
                f"model declined the routing request (category {category})",
                retryable=False,
                details={"stop_reason": stop_reason, "category": category},
            )
        if stop_reason == "max_tokens":
            raise RouterAdapterError(
                f"response truncated at max_tokens={self.max_tokens}", retryable=False
            )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return LLMCompletion(
            text=text,
            model=getattr(response, "model", None) or self.model,
            usage=self._usage(getattr(response, "usage", None)),
            stop_reason=stop_reason,
            metadata={"request_id": getattr(response, "_request_id", None)},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "anthropic",
            "kind": "llm",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "effort": self.effort,
            "refusal_fallbacks": self.refusal_fallbacks,
            "sampling": "provider default",
        }

    def _usage(self, usage: Any) -> ModelUsage | None:
        if usage is None:
            return None
        tokens_in = getattr(usage, "input_tokens", None)
        tokens_out = getattr(usage, "output_tokens", None)
        cost = None
        if self.price_per_mtok is not None and tokens_in is not None and tokens_out is not None:
            price_in, price_out = self.price_per_mtok
            cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        return ModelUsage(
            model_calls=1, input_tokens=tokens_in, output_tokens=tokens_out, estimated_cost_usd=cost
        )


def _sdk() -> Any:
    try:
        return importlib.import_module("anthropic")
    except ImportError as exc:
        raise ImportError(
            "AnthropicLLMAdapter needs the Anthropic SDK: pip install 'jevpilot[anthropic]'"
        ) from exc
