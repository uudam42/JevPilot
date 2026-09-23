"""Optional live smoke tests: ONE minimal real call per provider.

Skipped unless ``JEVPILOT_LIVE_TESTS=1`` and the provider's credentials are set:

    JEVPILOT_LIVE_TESTS=1 ANTHROPIC_API_KEY=... pytest tests/integrations/test_live_smoke.py
    JEVPILOT_LIVE_TESTS=1 TYPESAFE_API_KEY=...  pytest tests/integrations/test_live_smoke.py

They cost a small amount of money and are never part of the default suite.
"""

from __future__ import annotations

import os

import pytest

from jevpilot import JevRouter, LLMRouter
from tests.routing.helpers import SPECS, new_state

LIVE = os.environ.get("JEVPILOT_LIVE_TESTS") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="live tests disabled (set JEVPILOT_LIVE_TESTS=1)")


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_live_anthropic_single_decision() -> None:
    from integrations.anthropic_llm import AnthropicLLMAdapter

    adapter = AnthropicLLMAdapter()
    assert adapter.verify()["model_id"]
    outcome = LLMRouter(adapter, timeout_s=120).route(new_state(), SPECS)
    (attempt,) = outcome.attempts
    assert attempt.model and attempt.metadata["requested_model"] == adapter.model
    assert attempt.usage and attempt.usage.input_tokens


@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")
def test_live_jev_single_decision() -> None:
    from integrations.typesafe_jev import TypeSafeJevAdapter

    adapter = TypeSafeJevAdapter()
    assert adapter.model in adapter.verify()["listed_models"]
    outcome = JevRouter(adapter, timeout_s=60).route(new_state(), SPECS)
    (attempt,) = outcome.attempts
    assert attempt.model and attempt.usage and attempt.usage.model_calls
