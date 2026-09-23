"""AnthropicLLMAdapter against the real ``anthropic`` SDK via a mock HTTP transport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from jevpilot import LLMRouter
from jevpilot.exceptions import RouterAdapterError, RouterTimeoutError
from tests.routing.helpers import SPECS, new_state

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from integrations.anthropic_llm import AnthropicLLMAdapter  # noqa: E402

DECISION = (
    '{"action": "invoke", "capability_id": "t.write", "inputs": {"text": "hi"}, '
    '"reason": "write it", "confidence": 0.6}'
)
Handler = Callable[[Any], Any]


def message(
    text: str = DECISION, *, model: str = "claude-opus-5", stop: str = "end_turn"
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 55,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def client(handler: Handler) -> Any:
    transport = httpx2.MockTransport(handler)
    return anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=transport),
    )


def respond(body: dict[str, Any], status: int = 200, seen: list[Any] | None = None) -> Handler:
    def handler(request: Any) -> Any:
        if seen is not None:
            seen.append(request)
        return httpx2.Response(status, json=body, headers={"request-id": "req_123"})

    return handler


def test_strict_request_uses_plain_messages_endpoint_and_records_identity() -> None:
    seen: list[Any] = []
    adapter = AnthropicLLMAdapter(client=client(respond(message(), seen=seen)))
    outcome = LLMRouter(adapter, timeout_s=20).route(new_state(), SPECS)
    assert outcome.decision.capability_id == "t.write" and outcome.decision.confidence == 0.6
    (req,) = seen
    body = json.loads(req.content)
    assert req.url.path == "/v1/messages" and "anthropic-beta" not in req.headers
    assert body["model"] == "claude-opus-5" and body["max_tokens"] == 16000
    assert "fallbacks" not in body and "temperature" not in body
    assert "Choose only from" in body["system"] and "t.write" in body["messages"][0]["content"]
    (attempt,) = outcome.attempts
    assert attempt.model == "claude-opus-5"
    assert attempt.usage and (attempt.usage.input_tokens, attempt.usage.output_tokens) == (1200, 55)
    assert attempt.usage.estimated_cost_usd is None
    meta = attempt.metadata
    assert meta["provider"] == "anthropic" and meta["requested_model"] == "claude-opus-5"
    assert meta["actual_model"] == "claude-opus-5" and meta["mode"] == "strict"
    assert meta["request_id"] == "req_123" and meta["prompt_version"] == "routing-prompt/1"


def test_production_mode_sends_the_fallback_beta() -> None:
    seen: list[Any] = []
    adapter = AnthropicLLMAdapter(
        client=client(respond(message(), seen=seen)), refusal_fallbacks=True
    )
    LLMRouter(adapter).route(new_state(), SPECS)
    body = json.loads(seen[0].content)
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in seen[0].headers["anthropic-beta"]
    assert adapter.describe()["mode"] == "production"


def test_strict_mode_rejects_a_substituted_model() -> None:
    adapter = AnthropicLLMAdapter(client=client(respond(message(model="claude-opus-4-8"))))
    with pytest.raises(RouterAdapterError, match="strict mode") as info:
        LLMRouter(adapter).select_next(new_state(), SPECS)
    assert info.value.details["actual_model"] == "claude-opus-4-8"
    snapshot = AnthropicLLMAdapter(client=client(respond(message(model="claude-opus-5-20260101"))))
    assert LLMRouter(snapshot).select_next(new_state(), SPECS).capability_id == "t.write"


def test_model_is_configuration_driven(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEVPILOT_ANTHROPIC_MODEL", "claude-sonnet-5")
    seen: list[Any] = []
    adapter = AnthropicLLMAdapter(
        client=client(respond(message(model="claude-sonnet-5"), seen=seen))
    )
    LLMRouter(adapter).select_next(new_state(), SPECS)
    assert json.loads(seen[0].content)["model"] == "claude-sonnet-5"
    assert AnthropicLLMAdapter(model="x", client=object()).model == "x"


@pytest.mark.parametrize(
    ("body", "status", "error", "retryable"),
    [
        (message(stop="refusal"), 200, RouterAdapterError, False),
        (message(stop="max_tokens"), 200, RouterAdapterError, False),
        (
            {"type": "error", "error": {"type": "not_found_error", "message": "model"}},
            404,
            RouterAdapterError,
            False,
        ),
        (
            {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
            429,
            RouterAdapterError,
            True,
        ),
        (
            {"type": "error", "error": {"type": "api_error", "message": "boom"}},
            500,
            RouterAdapterError,
            True,
        ),
    ],
)
def test_provider_failures_are_explicit(
    body: dict[str, Any], status: int, error: type[Exception], retryable: bool
) -> None:
    adapter = AnthropicLLMAdapter(client=client(respond(body, status)))
    with pytest.raises(error) as info:
        LLMRouter(adapter).select_next(new_state(), SPECS)
    assert info.value.retryable is retryable  # type: ignore[attr-defined]


def test_timeout_maps_to_router_timeout() -> None:
    def slow(request: Any) -> Any:
        raise httpx2.ReadTimeout("read timed out", request=request)

    with pytest.raises(RouterTimeoutError):
        LLMRouter(AnthropicLLMAdapter(client=client(slow)), timeout_s=1).select_next(
            new_state(), SPECS
        )


def test_malformed_model_text_is_a_routing_failure_not_an_action() -> None:
    from jevpilot.exceptions import MalformedRoutingDecisionError

    adapter = AnthropicLLMAdapter(client=client(respond(message("Sure! I'd call t.write."))))
    with pytest.raises(MalformedRoutingDecisionError):
        LLMRouter(adapter).select_next(new_state(), SPECS)


def test_verify_checks_the_model_exists() -> None:
    info = {
        "type": "model",
        "id": "claude-opus-5",
        "display_name": "Claude Opus 5",
        "created_at": "2026-01-01T00:00:00Z",
    }
    seen: list[Any] = []
    result = AnthropicLLMAdapter(client=client(respond(info, seen=seen))).verify()
    assert seen[0].url.path == "/v1/models/claude-opus-5"
    assert result["display_name"] == "Claude Opus 5"
    missing = {"type": "error", "error": {"type": "not_found_error", "message": "no model"}}
    with pytest.raises(RouterAdapterError):
        AnthropicLLMAdapter(client=client(respond(missing, 404))).verify()
