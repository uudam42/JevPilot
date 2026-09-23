"""TypeSafeJevAdapter against the real ``typesafe-sdk`` with a mock HTTP transport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from jevpilot import JevRouter, RoutingIntent
from jevpilot.exceptions import (
    InvalidCapabilityError,
    InvalidRoutingInputError,
    RouterAdapterError,
    RouterTimeoutError,
)
from tests.routing.helpers import SPECS, new_state

typesafe_sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

from integrations.typesafe_jev import ASK_HUMAN, FINISH, TypeSafeJevAdapter  # noqa: E402


def choice(label: str, p: float = 0.8, others: tuple[str, ...] = ()) -> dict[str, Any]:
    probs = {label: p, **{o: (1 - p) / max(len(others), 1) for o in others}}
    return {"type": "choice", "choice": label, "confidence": p, "probabilities": probs}


def body(answers: dict[str, Any], model: str = "jev-2026-05") -> dict[str, Any]:
    return {"model": model, "usage": {"input_tokens": 800, "output_tokens": 3}, "answers": answers}


def scripted(*bodies: dict[str, Any], seen: list[Any] | None = None, status: int = 200) -> Any:
    queue = list(bodies)

    def handler(request: Any) -> Any:
        if seen is not None:
            seen.append(json.loads(request.content) if request.content else request.url.path)
        return httpx2.Response(status, json=queue.pop(0))

    return handler


def adapter(handler: Callable[[Any], Any]) -> TypeSafeJevAdapter:
    client = typesafe_sdk.TypeSafeClient(
        api_key="test-key",
        transport=httpx2.MockTransport(handler),
        retry=typesafe_sdk.RetryPolicy(max_retries=0),
    )
    return TypeSafeJevAdapter(client=client)


def test_two_call_decision_with_grounded_string_input() -> None:
    seen: list[Any] = []
    handler = scripted(
        body({"next_action": choice("t.write", 0.9, ("t.noop", FINISH, ASK_HUMAN))}),
        body({"input_0": choice("v0", 0.7, ("__none__",))}),
        seen=seen,
    )
    outcome = JevRouter(adapter(handler)).route(new_state(), SPECS)
    d = outcome.decision
    assert (d.capability_id, d.inputs, d.confidence) == ("t.write", {"text": "hello"}, 0.9)
    first, second = seen
    assert first["model"] == "jev-latest"
    assert "available_capabilities" not in first["state"]  # capabilities are the labels
    criteria = first["questions"]["next_action"]["criteria"]
    assert set(criteria) == {"t.write", "t.noop", FINISH, ASK_HUMAN}
    assert criteria["t.write"]["input_schema"]["required"] == ["text"]
    assert second["state"]["selected_capability"] == "t.write"
    grounded = second["questions"]["input_0"]["criteria"]
    assert grounded["v0"] == {"value": "hello", "found_at": "goal.parameters.text"}
    (attempt,) = outcome.attempts
    assert attempt.model == "jev-2026-05"
    assert attempt.usage and attempt.usage.model_calls == 2 and attempt.usage.input_tokens == 1600
    assert attempt.metadata["next_action_probabilities"]["t.write"] == 0.9
    assert attempt.metadata["requested_model"] == "jev-latest"
    assert attempt.metadata["request_format"] == "jevpilot-jev-questions/1"


@pytest.mark.parametrize(
    ("label", "intent"), [(FINISH, RoutingIntent.FINISH), (ASK_HUMAN, RoutingIntent.ASK_HUMAN)]
)
def test_no_action_labels_need_one_call(label: str, intent: RoutingIntent) -> None:
    seen: list[Any] = []
    d = JevRouter(adapter(scripted(body({"next_action": choice(label)}), seen=seen))).select_next(
        new_state(), SPECS
    )
    assert d.intent is intent and len(seen) == 1


def test_capability_without_required_inputs_needs_one_call() -> None:
    seen: list[Any] = []
    d = JevRouter(
        adapter(scripted(body({"next_action": choice("t.noop")}), seen=seen))
    ).select_next(new_state(), SPECS)
    assert d.capability_id == "t.noop" and len(seen) == 1


def test_label_outside_the_offered_set_is_rejected() -> None:
    with pytest.raises(InvalidCapabilityError):
        JevRouter(adapter(scripted(body({"next_action": choice("t.rm_rf")})))).select_next(
            new_state(), SPECS
        )


def test_no_grounded_value_means_invalid_inputs_not_a_guess() -> None:
    handler = scripted(
        body({"next_action": choice("t.write")}), body({"input_0": choice("__none__")})
    )
    with pytest.raises(InvalidRoutingInputError):
        JevRouter(adapter(handler)).select_next(new_state(), SPECS)


def test_malformed_response_fails_safely() -> None:
    with pytest.raises(RouterAdapterError):
        JevRouter(adapter(scripted({"model": "jev", "answers": {}}))).select_next(
            new_state(), SPECS
        )


def test_http_errors_and_timeouts() -> None:
    with pytest.raises(RouterAdapterError) as info:
        JevRouter(adapter(scripted({"detail": "busy"}, status=429))).select_next(new_state(), SPECS)
    assert info.value.retryable is True

    def slow(request: Any) -> Any:
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(RouterTimeoutError):
        JevRouter(adapter(slow), timeout_s=1).select_next(new_state(), SPECS)


def test_verify_lists_models() -> None:
    listed = {
        "models": [{"name": "jev-latest", "description": "Jev", "release_date": "2026-05-01"}]
    }
    assert adapter(scripted(listed)).verify()["release_date"] == "2026-05-01"
    with pytest.raises(RouterAdapterError):
        adapter(scripted({"models": []})).verify()


def test_enum_bool_and_bounded_int_inputs() -> None:
    from pydantic import BaseModel, Field

    from integrations.typesafe_jev import _input_questions
    from jevpilot import CapabilitySpec
    from jevpilot.routing import RoutingRequest

    class In(BaseModel):
        mode: str = Field(json_schema_extra={"enum": ["a", "b"]})
        flag: bool
        level: int = Field(ge=1, le=3)

    spec = CapabilitySpec(id="x.cap", name="cap", description="d", input_schema=In)
    req = RoutingRequest.from_state(new_state(), [spec])
    kinds = [q.kind for q in _input_questions(req.available_capabilities[0], req)]
    assert kinds == ["choice", "noul", "score"]
