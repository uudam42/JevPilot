"""LLMRouter and JevRouter with offline fake adapters: every scripted failure mode."""

from __future__ import annotations

import time
from typing import Any

import pytest

from jevpilot import JevRouter, LLMRouter, ModelUsage, RoutingIntent
from jevpilot.adapters import FakeJevAdapter, FakeLLMAdapter, ScriptExhaustedError
from jevpilot.adapters import scripting as fake
from jevpilot.exceptions import (
    InvalidCapabilityError,
    InvalidRoutingInputError,
    LowConfidenceRoutingError,
    MalformedRoutingDecisionError,
    RouterAdapterError,
    RouterTimeoutError,
    RoutingError,
)
from jevpilot.routing import LLMCompletion, ModelRouter, RoutingModelResponse, RoutingRequest
from tests.routing.helpers import SPECS, new_state


def jev(*script: Any, **kw: Any) -> JevRouter:
    return JevRouter(FakeJevAdapter(list(script)), **kw)


def llm(*script: Any, **kw: Any) -> LLMRouter:
    return LLMRouter(FakeLLMAdapter(list(script)), **kw)


BOTH = [jev, llm]


@pytest.mark.parametrize("make", BOTH)
def test_valid_decision(make: Any) -> None:
    router: ModelRouter = make(fake.valid("t.write", {"text": "hi"}, confidence=0.8))
    outcome = router.route(new_state(), SPECS)
    d = outcome.decision
    assert (d.capability_id, d.inputs, d.confidence) == ("t.write", {"text": "hi"}, 0.8)
    assert d.router_id == router.router_id
    (attempt,) = outcome.attempts
    assert attempt.success and attempt.request is not None
    assert attempt.request["available_capabilities"][0]["id"] == "t.write"
    assert attempt.usage is None  # fakes never invent usage


@pytest.mark.parametrize(
    ("item", "error"),
    [
        (fake.invalid_capability("t.delete_everything"), InvalidCapabilityError),
        (fake.invalid_inputs("t.write", {"text": 5}), InvalidRoutingInputError),
        (fake.invalid_inputs("t.write", {}), InvalidRoutingInputError),  # missing required
        (fake.invalid_inputs("t.write", {"text": "a", "extra": 1}), InvalidRoutingInputError),
        (fake.malformed(), MalformedRoutingDecisionError),
        (fake.timeout(), RouterTimeoutError),
        (fake.adapter_error(), RouterAdapterError),
        (TimeoutError("socket"), RouterTimeoutError),
    ],
)
@pytest.mark.parametrize("make", BOTH)
def test_failures_are_explicit_and_traced(make: Any, item: Any, error: type[Exception]) -> None:
    router = make(item)
    with pytest.raises(error) as info:
        router.select_next(new_state(), SPECS)
    assert isinstance(info.value, RoutingError)
    (attempt,) = info.value.attempts
    assert not attempt.success and attempt.error.type == error.__name__
    assert attempt.request is not None and attempt.router_id == router.router_id


def test_low_confidence_is_stored_not_trusted_unless_threshold_configured() -> None:
    d = jev(fake.low_confidence("t.write", {"text": "a"}, 0.05)).select_next(new_state(), SPECS)
    assert d.confidence == 0.05  # accepted and recorded
    strict = jev(fake.low_confidence("t.write", {"text": "a"}, 0.05), min_confidence=0.5)
    with pytest.raises(LowConfidenceRoutingError):
        strict.select_next(new_state(), SPECS)
    missing = jev(fake.valid("t.write", {"text": "a"}), min_confidence=0.5)
    with pytest.raises(LowConfidenceRoutingError):
        missing.select_next(new_state(), SPECS)


def test_soft_deadline_discards_late_answers() -> None:
    def slow(request: RoutingRequest) -> dict[str, Any]:
        time.sleep(0.02)
        return fake.valid("t.write", {"text": "a"})

    with pytest.raises(RouterTimeoutError):
        jev(slow, timeout_s=0.001).select_next(new_state(), SPECS)


def test_usage_is_recorded_when_the_adapter_reports_it() -> None:
    usage = ModelUsage(model_calls=1, input_tokens=120, output_tokens=30)
    item = RoutingModelResponse(
        output=fake.valid("t.write", {"text": "a"}), model="m1", usage=usage
    )
    (attempt,) = jev(item).route(new_state(), SPECS).attempts
    assert attempt.usage == usage and attempt.model == "m1"
    comp = LLMCompletion(
        text='{"action": "finish", "capability_id": null, "inputs": {}, "reason": "done"}',
        model="m2",
        usage=usage,
    )
    outcome = llm(comp).route(new_state(), SPECS)
    assert outcome.decision.intent is RoutingIntent.FINISH
    assert outcome.attempts[0].usage == usage


def test_llm_router_sends_versioned_prompt_with_the_request() -> None:
    adapter = FakeLLMAdapter([fake.valid("t.write", {"text": "a"})])
    router = LLMRouter(adapter)
    router.select_next(new_state(), SPECS)
    (prompt,) = adapter.prompts
    assert prompt.version == "routing-prompt/1"
    assert "Choose only from the capabilities" in prompt.system
    assert '"t.write"' in prompt.user and prompt.response_schema["required"]
    assert router.config()["adapter"]["provider"] == "fake"


def test_jev_adapter_receives_the_canonical_request() -> None:
    adapter = FakeJevAdapter([fake.finish()])
    JevRouter(adapter).select_next(new_state(), SPECS)
    (request,) = adapter.requests
    assert isinstance(request, RoutingRequest)
    assert request.capability_ids == ("t.write", "t.noop")


def test_exhausted_script_is_an_adapter_error_not_a_silent_default() -> None:
    router = jev()
    with pytest.raises(RouterAdapterError) as info:
        router.select_next(new_state(), SPECS)
    assert isinstance(info.value.__cause__, ScriptExhaustedError)
    lenient = JevRouter(FakeJevAdapter([], on_exhausted="finish"))
    assert lenient.select_next(new_state(), SPECS).intent is RoutingIntent.FINISH


def test_adapter_returning_wrong_type_is_malformed() -> None:
    class Bad:
        def infer(self, request: RoutingRequest, *, timeout_s: float | None = None) -> Any:
            return {"capability_id": "t.write"}

        def describe(self) -> dict[str, Any]:
            return {}

    with pytest.raises(MalformedRoutingDecisionError):
        JevRouter(Bad()).select_next(new_state(), SPECS)


def test_fault_injection_is_seeded_and_reproducible() -> None:
    def base(request: RoutingRequest) -> dict[str, Any]:
        return fake.valid("t.write", {"text": "a"})

    def outcomes(seed: int) -> list[str]:
        router = JevRouter(FakeJevAdapter(policy=fake.with_faults(base, {"malformed": 0.5}, seed)))
        out = []
        for _ in range(20):
            try:
                router.select_next(new_state(), SPECS)
                out.append("ok")
            except RoutingError as exc:
                out.append(type(exc).__name__)
        return out

    assert outcomes(3) == outcomes(3)
    assert set(outcomes(3)) == {"ok", "MalformedRoutingDecisionError"}
    with pytest.raises(ValueError):
        fake.with_faults(base, {"gremlins": 0.1})
