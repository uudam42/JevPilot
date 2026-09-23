"""Optional live smoke tests: ONE real routing call per provider (iteration spec §3/§4).

Skipped unless ``JEVPILOT_LIVE_TESTS=1`` and that provider's key is set:

    export JEVPILOT_LIVE_TESTS=1   # plus TYPESAFE_API_KEY and/or ANTHROPIC_API_KEY
    pytest tests/integrations/test_live_smoke.py -k jev -s
    pytest tests/integrations/test_live_smoke.py -k claude -s

Each test makes exactly one routing call (Jev: 1–2 API requests), never retries
to obtain a particular answer, and writes its observations (no credentials) to
``experiments/routing/results/live_smoke/<provider>.json``. The answer itself
is not asserted to be correct: this checks the integration, not the model.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from jevpilot import JevRouter, LLMRouter, RoutingOutcome, validate_decision
from jevpilot.routing import ModelRouter
from tests.integrations.live_problem import ACCEPTABLE, OFFERED, smoke_problem

LIVE = os.environ.get("JEVPILOT_LIVE_TESTS") == "1"
OUT = Path(__file__).resolve().parents[2] / "experiments" / "routing" / "results" / "live_smoke"
pytestmark = pytest.mark.skipif(not LIVE, reason="live tests disabled (set JEVPILOT_LIVE_TESTS=1)")


def _route_once(router: ModelRouter) -> tuple[RoutingOutcome, float]:
    state, specs = smoke_problem()
    t0 = time.perf_counter()
    outcome = router.route(state, specs)  # routing errors propagate: they are findings
    wall = time.perf_counter() - t0
    validate_decision(outcome.decision, specs)
    return outcome, wall


def _check_and_record(
    provider: str, outcome: RoutingOutcome, wall: float, preflight: dict[str, Any]
) -> dict[str, Any]:
    (attempt,) = outcome.attempts
    d = outcome.decision
    assert attempt.success and attempt.latency_s > 0
    assert d.capability_id is None or d.capability_id in OFFERED
    request = attempt.request
    assert request is not None
    json.dumps(request)  # plain data only: no executable capability objects were sent
    assert [c["id"] for c in request["available_capabilities"]] == list(OFFERED)
    record = {
        "provider": provider,
        "preflight": preflight,
        "decision": {
            "intent": str(d.intent),
            "capability_id": d.capability_id,
            "inputs": d.inputs,
            "confidence": d.confidence,
            "reason": d.reason,
        },
        "acceptable": d.capability_id in ACCEPTABLE,
        "latency_s": attempt.latency_s,
        "wall_s": wall,
        "served_model": attempt.model,
        "usage": attempt.usage.model_dump() if attempt.usage else None,
        "metadata": attempt.metadata,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{provider}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps(record, indent=2, default=str))
    return record


@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")
def test_live_jev_single_decision() -> None:
    from integrations.typesafe_jev import TypeSafeJevAdapter

    adapter = TypeSafeJevAdapter()
    preflight = adapter.verify()
    outcome, wall = _route_once(JevRouter(adapter, timeout_s=60))
    record = _check_and_record("typesafe_jev", outcome, wall, preflight)
    assert record["metadata"]["requested_model"] == adapter.model
    assert record["served_model"]  # identity reported by the API, never inferred
    assert record["decision"]["confidence"] is not None
    assert "next_action_probabilities" in record["metadata"]


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_live_claude_single_decision_strict() -> None:
    from integrations.anthropic_llm import AnthropicLLMAdapter

    adapter = AnthropicLLMAdapter(refusal_fallbacks=False)  # strict single-model mode
    assert adapter.strict
    preflight = adapter.verify()
    outcome, wall = _route_once(LLMRouter(adapter, timeout_s=120))
    record = _check_and_record("anthropic_claude", outcome, wall, preflight)
    meta = record["metadata"]
    assert meta["mode"] == "strict" and meta["requested_model"] == adapter.model
    assert record["served_model"] and meta["stop_reason"] == "end_turn"
    usage = record["usage"]
    assert usage and usage["input_tokens"] and usage["output_tokens"] is not None
