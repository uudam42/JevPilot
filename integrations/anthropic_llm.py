"""Claude (Anthropic Messages API) as an :class:`~jevpilot.routing.contracts.LLMAdapter`.

Install with ``pip install 'jevpilot[anthropic]'``. Credentials are resolved by
the SDK (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or an ``ant auth login``
profile) and never pass through JevPilot. The model comes from the ``model``
argument, else ``JEVPILOT_ANTHROPIC_MODEL``, else :data:`DEFAULT_MODEL`.

Two operating modes:

* **strict** (default): one model, no server-side substitution. Every
  completion must be served by the requested model, otherwise the call fails
  with :class:`RouterAdapterError`. Use this for experiments, where a result
  labelled "model X" must come from model X.
* **production** (``refusal_fallbacks=True``): a policy-declined request is
  re-run server-side on Anthropic's recommended fallback model. The serving
  model is still recorded, but results are no longer single-model.

Token usage is taken from the API response. Cost is estimated only when
``price_per_mtok`` is given, and the prices used are recorded in
:meth:`describe`. Otherwise cost is unknown (``None``).
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from jevpilot.core import ModelUsage
from jevpilot.exceptions import RouterAdapterError, RouterTimeoutError
from jevpilot.routing.contracts import LLMCompletion, LLMPrompt

ADAPTER_VERSION = "anthropic-llm-adapter/2"
PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-opus-5"
MODEL_ENV = "JEVPILOT_ANTHROPIC_MODEL"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLMAdapter:
    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 16000,
        effort: str | None = None,
        refusal_fallbacks: bool = False,
        price_per_mtok: tuple[float, float] | None = None,
        pricing_source: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.effort = effort
        self.refusal_fallbacks = refusal_fallbacks
        self.price_per_mtok = price_per_mtok
        self.pricing_source = pricing_source
        self._sdk: Any = _optional_sdk()
        if client is None:
            if self._sdk is None:
                raise ImportError(
                    "AnthropicLLMAdapter needs the Anthropic SDK: pip install 'jevpilot[anthropic]'"
                )
            client = self._sdk.Anthropic()
        self._client = client

    @property
    def strict(self) -> bool:
        return not self.refusal_fallbacks

    # -- LLMAdapter -------------------------------------------------------------

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
            raise self._translate(exc) from exc

        served = getattr(response, "model", None)
        identity = self._identity(served)
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise RouterAdapterError(
                f"model declined the routing request (category {category})",
                retryable=False,
                details={**identity, "stop_reason": stop_reason, "category": category},
            )
        if stop_reason == "max_tokens":
            raise RouterAdapterError(
                f"response truncated at max_tokens={self.max_tokens}",
                retryable=False,
                details=identity,
            )
        if self.strict and not _same_model(self.model, served):
            raise RouterAdapterError(
                f"strict mode: requested {self.model!r} but {served!r} served the request",
                retryable=False,
                details=identity,
            )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage_obj = getattr(response, "usage", None)
        return LLMCompletion(
            text=text,
            model=served,  # never inferred: None if the provider did not report it
            usage=self._usage(usage_obj),
            stop_reason=stop_reason,
            metadata={
                **identity,
                "request_id": getattr(response, "_request_id", None),
                "cache_read_input_tokens": getattr(usage_obj, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(
                    usage_obj, "cache_creation_input_tokens", None
                ),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": PROVIDER,
            "kind": "llm",
            "adapter_version": ADAPTER_VERSION,
            "sdk_version": getattr(self._sdk, "__version__", None),
            "model": self.model,
            "mode": "strict" if self.strict else "production",
            "refusal_fallbacks": self.refusal_fallbacks,
            "max_tokens": self.max_tokens,
            "effort": self.effort or "provider default",
            "thinking": "provider default",
            "sampling": "provider default (sampling parameters are not sent)",
            "pricing": (
                {"usd_per_mtok_in_out": list(self.price_per_mtok), "source": self.pricing_source}
                if self.price_per_mtok
                else None
            ),
        }

    # -- preflight ---------------------------------------------------------------

    def verify(self) -> dict[str, Any]:
        """Confirm the configured model exists for these credentials (one cheap metadata call)."""
        try:
            info = self._client.models.retrieve(self.model)
        except Exception as exc:
            raise self._translate(exc) from exc
        return {
            "provider": PROVIDER,
            "requested_model": self.model,
            "model_id": getattr(info, "id", None),
            "display_name": getattr(info, "display_name", None),
            "created_at": str(getattr(info, "created_at", None)),
            "max_input_tokens": getattr(info, "max_input_tokens", None),
            "max_tokens": getattr(info, "max_tokens", None),
        }

    # -- helpers -----------------------------------------------------------------

    def _identity(self, served: str | None) -> dict[str, Any]:
        return {
            "provider": PROVIDER,
            "requested_model": self.model,
            "actual_model": served,
            "adapter_version": ADAPTER_VERSION,
            "mode": "strict" if self.strict else "production",
        }

    def _translate(self, exc: Exception) -> Exception:
        details = self._identity(None)
        sdk = self._sdk
        name = type(exc).__name__
        if (sdk is not None and isinstance(exc, sdk.APITimeoutError)) or name == "APITimeoutError":
            return RouterTimeoutError(f"Anthropic request timed out: {exc}", details=details)
        retryable = True
        if sdk is not None and isinstance(exc, sdk.APIStatusError):
            status = getattr(exc, "status_code", None)
            retryable = status is not None and (status == 429 or status >= 500)
            details["status_code"] = status
        return RouterAdapterError(
            f"Anthropic request failed: {name}: {exc}", retryable=retryable, details=details
        )

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


def _same_model(requested: str, served: str | None) -> bool:
    """True when the served id is the requested one (or a dated snapshot of it)."""
    return served is not None and (served == requested or served.startswith(requested + "-"))


def _optional_sdk() -> Any:
    try:
        return importlib.import_module("anthropic")
    except ImportError:
        return None
