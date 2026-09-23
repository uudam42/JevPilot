"""The real Claude adapter, exercised offline with a stub client (no SDK, no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from integrations.anthropic_llm import AnthropicLLMAdapter
from jevpilot import LLMRouter
from jevpilot.exceptions import RouterAdapterError, RouterTimeoutError
from tests.routing.helpers import SPECS, new_state

DECISION = (
    '{"action": "invoke", "capability_id": "t.write", "inputs": {"text": "hi"}, '
    '"reason": "write it", "confidence": 0.6}'
)


class StubMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response, self.error, self.calls = response, error, []  # type: ignore[var-annotated]

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class StubClient:
    def __init__(self, messages: StubMessages) -> None:
        self.messages = messages
        self.beta = SimpleNamespace(messages=messages)
        self.timeouts: list[float] = []

    def with_options(self, *, timeout: float) -> StubClient:
        self.timeouts.append(timeout)
        return self


def response(text: str = DECISION, stop: str = "end_turn") -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop,
        stop_details=None,
        model="claude-opus-5",
        usage=SimpleNamespace(input_tokens=900, output_tokens=40),
    )


def test_request_shape_usage_and_decision() -> None:
    msgs = StubMessages(response())
    client = StubClient(msgs)
    adapter = AnthropicLLMAdapter(client=client, price_per_mtok=(5.0, 25.0))
    outcome = LLMRouter(adapter, timeout_s=30).route(new_state(), SPECS)
    assert outcome.decision.capability_id == "t.write"
    (call,) = msgs.calls
    assert call["model"] == "claude-opus-5" and call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "Choose only from" in call["system"] and "t.write" in call["messages"][0]["content"]
    assert client.timeouts == [30]
    usage = outcome.attempts[0].usage
    assert usage and (usage.input_tokens, usage.output_tokens) == (900, 40)
    assert usage.estimated_cost_usd == pytest.approx((900 * 5 + 40 * 25) / 1e6)


def test_cost_is_unknown_without_a_price() -> None:
    adapter = AnthropicLLMAdapter(
        client=StubClient(StubMessages(response())), refusal_fallbacks=False
    )
    usage = LLMRouter(adapter).route(new_state(), SPECS).attempts[0].usage
    assert usage and usage.estimated_cost_usd is None


@pytest.mark.parametrize(
    ("resp", "error", "expected"),
    [
        (response(stop="refusal"), None, RouterAdapterError),
        (response(stop="max_tokens"), None, RouterAdapterError),
        (None, type("APITimeoutError", (Exception,), {})("slow"), RouterTimeoutError),
        (None, ConnectionError("down"), RouterAdapterError),
    ],
)
def test_provider_failures_become_routing_failures(resp: Any, error: Any, expected: Any) -> None:
    adapter = AnthropicLLMAdapter(client=StubClient(StubMessages(resp, error)))
    with pytest.raises(expected):
        LLMRouter(adapter).select_next(new_state(), SPECS)
