"""Reliability of TypeSafeJevAdapter: discovery, bounded retries, error classes, redaction.

Real ``typesafe-sdk`` over a mock HTTP transport; no network, no real key.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from jevpilot import JevRouter
from jevpilot.exceptions import (
    MalformedRoutingDecisionError,
    RouterAdapterError,
    RouterTimeoutError,
)
from tests.routing.helpers import SPECS, new_state

typesafe_sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

from integrations.typesafe_jev import (  # noqa: E402
    FINISH,
    JevUnavailableError,
    TypeSafeJevAdapter,
)

MODELS = {
    "models": [
        {"name": "jev-1.12.0", "description": "older", "release_date": "2026-06-01"},
        {"name": "jev-1.13.0", "description": "newer", "release_date": "2026-08-01"},
    ]
}


def finish_body(confidence: float = 0.9, model: str = "jev-1.13.0") -> dict[str, Any]:
    answer = {
        "type": "choice",
        "choice": FINISH,
        "confidence": confidence,
        "probabilities": {FINISH: confidence},
    }
    return {"model": model, "usage": {"input_tokens": 10}, "answers": {"next_action": answer}}


Reply = tuple[int, Any, dict[str, str]]


def server(*replies: Reply, log: list[str] | None = None) -> Callable[[Any], Any]:
    queue = list(replies)

    def handler(request: Any) -> Any:
        if log is not None:
            body = json.loads(request.content) if request.content else {}
            log.append(f"{request.method} {request.url.path} {body.get('model', '')}".strip())
        status, body, headers = queue.pop(0)
        return httpx2.Response(status, json=body, headers=headers)

    return handler


def ok(body: Any) -> Reply:
    return (200, body, {})


def fail(status: int, headers: dict[str, str] | None = None, detail: str = "no") -> Reply:
    return (status, {"detail": detail}, headers or {})


def adapter(
    handler: Callable[[Any], Any], **options: Any
) -> tuple[TypeSafeJevAdapter, list[float]]:
    sleeps: list[float] = []
    client = typesafe_sdk.TypeSafeClient(
        api_key="test-key-not-real",
        transport=httpx2.MockTransport(handler),
        retry=typesafe_sdk.RetryPolicy(max_retries=0),
    )
    options.setdefault("sleep", sleeps.append)
    return TypeSafeJevAdapter(client=client, **options), sleeps


def route(a: TypeSafeJevAdapter, timeout_s: float | None = None) -> Any:
    return JevRouter(a, timeout_s=timeout_s).route(new_state(), SPECS)


# -- model discovery ----------------------------------------------------------------------


def test_model_is_discovered_from_the_models_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)
    log: list[str] = []
    a, _ = adapter(server(ok(MODELS), ok(finish_body()), log=log))
    outcome = route(a)
    assert log == ["GET /v1/models", "POST /v1/systemone jev-1.13.0"]
    assert (a.model, a.model_source) == ("jev-1.13.0", "discovered")
    assert outcome.attempts[0].metadata["requested_model"] == "jev-1.13.0"
    assert a.describe()["model_source"] == "discovered"


def test_configured_model_is_used_without_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-1.12.0")
    log: list[str] = []
    a, _ = adapter(server(ok(finish_body(model="jev-1.12.0")), log=log))
    route(a)
    assert log == ["POST /v1/systemone jev-1.12.0"] and a.model_source == "environment"


def test_no_listed_models_is_a_clear_non_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)
    a, _ = adapter(server(ok({"models": []})))
    with pytest.raises(RouterAdapterError, match="no Jev models") as info:
        a.verify()
    assert info.value.retryable is False and info.value.details["failure"] == "no_models"


def test_verify_rejects_a_configured_model_that_is_not_listed() -> None:
    a, _ = adapter(server(ok(MODELS)), model="jev-9")
    with pytest.raises(RouterAdapterError, match="not listed"):
        a.verify()


def test_verify_reports_authentication_ok_and_the_resolved_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)
    a, _ = adapter(server(ok(MODELS)))
    checked = a.verify()
    assert checked["authentication"] == "ok" and checked["requested_model"] == "jev-1.13.0"
    assert checked["release_date"] == "2026-08-01"


# -- error classes and bounded retries ------------------------------------------------------


def test_authentication_failure_is_not_retried_and_names_the_variable() -> None:
    log: list[str] = []
    a, sleeps = adapter(server(fail(401), log=log), model="jev-1.13.0", max_retries=3)
    with pytest.raises(RouterAdapterError) as info:
        route(a)
    err = info.value
    assert (err.retryable, err.details["failure"], err.details["status_code"]) == (
        False,
        "authentication",
        401,
    )
    assert "TYPESAFE_API_KEY" in str(err) and len(log) == 1 and sleeps == []


@pytest.mark.parametrize(
    ("status", "kind"), [(403, "permission"), (404, "not_found"), (422, "invalid_request")]
)
def test_client_errors_are_not_retried(status: int, kind: str) -> None:
    log: list[str] = []
    a, _ = adapter(server(fail(status), log=log), model="jev-1.13.0", max_retries=3)
    with pytest.raises(RouterAdapterError) as info:
        route(a)
    assert info.value.details["failure"] == kind and len(log) == 1


def test_rate_limit_is_retried_honouring_retry_after() -> None:
    a, sleeps = adapter(
        server(fail(429, {"retry-after": "2"}), ok(finish_body())), model="jev-1.13.0"
    )
    outcome = route(a)
    assert outcome.decision.intent == "finish" and sleeps == [2.0]
    (event,) = outcome.attempts[0].metadata["transport_events"]
    assert event["kind"] == "rate_limited" and event["retried"] is True
    assert outcome.attempts[0].metadata["retries"] == 1


def test_server_errors_are_retried_with_backoff_then_succeed() -> None:
    a, sleeps = adapter(
        server(fail(503), fail(529), ok(finish_body())), model="jev-1.13.0", max_retries=2
    )
    outcome = route(a)
    assert outcome.decision.intent == "finish" and sleeps == [0.5, 1.0]


def test_retries_are_bounded() -> None:
    log: list[str] = []
    a, sleeps = adapter(server(*[fail(500)] * 3, log=log), model="jev-1.13.0", max_retries=2)
    with pytest.raises(RouterAdapterError) as info:
        route(a)
    assert len(log) == 3 and len(sleeps) == 2
    assert info.value.retryable is True and info.value.details["failure"] == "server_error"
    assert len(info.value.details["transport_events"]) == 3


def test_retries_never_pass_the_router_deadline() -> None:
    log: list[str] = []
    a, sleeps = adapter(
        server(fail(429, {"retry-after": "30"}), ok(finish_body()), log=log),
        model="jev-1.13.0",
        max_retries=3,
    )
    with pytest.raises(RouterAdapterError):
        route(a, timeout_s=5)
    assert len(log) == 1 and sleeps == []


def test_transport_timeout_becomes_a_router_timeout() -> None:
    def slow(request: Any) -> Any:
        raise httpx2.ReadTimeout("slow", request=request)

    a, _ = adapter(slow, model="jev-1.13.0", max_retries=0)
    with pytest.raises(RouterTimeoutError):
        route(a, timeout_s=1)


# -- malformed answers and confidence --------------------------------------------------------


def test_answer_without_a_label_is_malformed() -> None:
    body = finish_body()
    body["answers"] = {"something_else": body["answers"]["next_action"]}
    a, _ = adapter(server(ok(body)), model="jev-1.13.0")
    with pytest.raises(RouterAdapterError, match="malformed") as info:
        route(a)
    assert info.value.retryable is False


def test_confidence_round_off_is_clamped_and_out_of_range_is_rejected() -> None:
    a, _ = adapter(server(ok(finish_body(1.0 + 1e-12))), model="jev-1.13.0")
    assert route(a).decision.confidence == 1.0
    bad = finish_body()
    bad["answers"]["next_action"]["confidence"] = 1.5
    b, _ = adapter(server(ok(bad)), model="jev-1.13.0")
    with pytest.raises((MalformedRoutingDecisionError, RouterAdapterError)):
        route(b)


# -- credentials -------------------------------------------------------------------------------


def test_missing_key_is_reported_without_building_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevUnavailableError, match="TYPESAFE_API_KEY is not set"):
        TypeSafeJevAdapter()


def test_error_messages_never_contain_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "tsk-" + "x" * 32  # a fake key, echoed back by a hostile or broken server
    monkeypatch.setenv("TYPESAFE_API_KEY", secret)
    a, _ = adapter(server(fail(500, detail=f"bad key {secret}")), model="jev-1.13.0", max_retries=0)
    with pytest.raises(RouterAdapterError) as info:
        route(a)
    assert secret not in str(info.value)
    assert secret not in json.dumps(info.value.details, default=str)
    assert secret not in repr(a.describe())
