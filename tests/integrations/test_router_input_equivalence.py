"""Claude and Jev receive the same underlying information (iteration spec §7).

Both real SDKs run over a mock transport; the wire payloads are captured and
the routing information is reconstructed from each. Provider formatting
differs (prompt JSON vs. Jev state + choice criteria); the information must not.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from experiments.routing.generalization import dataset as ds
from experiments.routing.generalization.world import load_catalog
from jevpilot import JevRouter, LLMRouter, RoutingRequest

anthropic = pytest.importorskip("anthropic")
typesafe_sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

from integrations.anthropic_llm import AnthropicLLMAdapter  # noqa: E402
from integrations.typesafe_jev import ASK_HUMAN, FINISH, TypeSafeJevAdapter  # noqa: E402

CASE = "eval_u4_trend_outliers_mirror@1"  # history with a failed observation + evaluations


def _case() -> tuple[Any, list[Any]]:
    catalog = load_catalog()
    case = next(c for c in ds.decision_cases("eval", catalog) if c.case_id == CASE)
    pres = ds.present(case, catalog)
    return pres.state, list(pres.specs)


def _claude_payload(state: Any, specs: list[Any]) -> dict[str, Any]:
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [
                    {
                        "type": "text",
                        "text": '{"action": "finish", "capability_id": null, '
                        '"inputs": {}, "reason": "x"}',
                    }
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = anthropic.Anthropic(
        api_key="k",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    LLMRouter(AnthropicLLMAdapter(client=client)).select_next(state, specs)
    user = seen[0]["messages"][0]["content"]
    return json.loads(user[user.index("{") :])


def _jev_payload(state: Any, specs: list[Any]) -> dict[str, Any]:
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "answers": {
                    "next_action": {
                        "type": "choice",
                        "choice": FINISH,
                        "confidence": 1.0,
                        "probabilities": {FINISH: 1.0},
                    }
                },
            },
        )

    client = typesafe_sdk.TypeSafeClient(
        api_key="k",
        transport=httpx2.MockTransport(handler),
        retry=typesafe_sdk.RetryPolicy(max_retries=0),
    )
    JevRouter(TypeSafeJevAdapter(client=client, model="jev-test")).select_next(state, specs)
    body = seen[0]
    criteria = body["questions"]["next_action"]["criteria"]
    caps = [{"id": k, **v} for k, v in criteria.items() if k not in (FINISH, ASK_HUMAN)]
    return {**body["state"], "available_capabilities": caps}


def test_claude_and_jev_receive_identical_routing_information() -> None:
    state, specs = _case()
    canonical = json.loads(
        RoutingRequest.from_state(state, specs, metadata={"router": "x"}).to_json()
    )
    claude, jev = _claude_payload(state, specs), _jev_payload(state, specs)
    for payload in (claude, jev):
        payload["metadata"] = {"router": "x"}  # router label is the only permitted difference
    assert claude == canonical
    assert jev == canonical
    # the fields the spec names explicitly, including history with a failed observation
    for key in (
        "goal",
        "state_summary",
        "requirements",
        "constraints",
        "available_capabilities",
        "previous_actions",
        "previous_evaluations",
    ):
        assert claude[key] == jev[key], key
    assert any(a["success"] is False and a["error"] for a in jev["previous_actions"])
    assert all(c["input_schema"] is not None for c in jev["available_capabilities"])
