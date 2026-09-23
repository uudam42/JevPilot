"""Model output is untrusted: only a strict JSON decision shape is accepted."""

import pytest

from jevpilot import RoutingIntent
from jevpilot.exceptions import MalformedRoutingDecisionError
from jevpilot.routing import decision_from_output


def parse(output: object):  # type: ignore[no-untyped-def]
    return decision_from_output(output, router_id="r")


def test_valid_invoke_text_and_mapping() -> None:
    d = parse(
        '{"action": "invoke", "capability_id": "a", "inputs": {"x": 1}, '
        '"reason": "go", "confidence": 0.7}'
    )
    assert (d.capability_id, d.inputs, d.intent, d.confidence) == (
        "a",
        {"x": 1},
        RoutingIntent.INVOKE,
        0.7,
    )
    assert d.router_id == "r"
    assert parse({"capability_id": "a"}).intent is RoutingIntent.INVOKE  # action inferred


def test_code_fence_is_tolerated() -> None:
    assert (
        parse(
            '```json\n{"action": "finish", "capability_id": null, "inputs": {}, '
            '"reason": "done"}\n```'
        ).intent
        is RoutingIntent.FINISH
    )


def test_no_action_intents() -> None:
    d = parse({"action": "ask_human", "capability_id": None, "inputs": {}, "reason": "?"})
    assert d.is_no_action and d.intent is RoutingIntent.ASK_HUMAN


@pytest.mark.parametrize(
    "output",
    [
        "run the parser next",
        "[1, 2]",
        '{"capability_id": "a"} trailing',
        {"action": "invoke", "capability_id": "", "inputs": {}},
        {"action": "invoke", "capability_id": 3, "inputs": {}},
        {"action": "execute_python", "capability_id": "a"},
        {"action": "invoke", "capability_id": "a", "inputs": "x=1"},
        {"action": "finish", "capability_id": "a", "inputs": {}},
        {"action": "finish", "capability_id": None, "inputs": {"x": 1}},
        {"action": "invoke", "capability_id": "a", "confidence": 1.5},
        {"action": "invoke", "capability_id": "a", "confidence": True},
        {"action": "invoke", "capability_id": "a", "reason": 5},
        {"action": "invoke", "capability_id": "a", "router_id": "spoofed"},
        {"action": "invoke", "capability_id": "a", "__import__": "os"},
        {"action": "invoke", "capability_id": "a", "inputs": {"f": object()}},
        None,
    ],
)
def test_malformed_output_is_an_explicit_failure(output: object) -> None:
    with pytest.raises(MalformedRoutingDecisionError) as info:
        parse(output)
    assert info.value.retryable is False


def test_failure_details_keep_only_a_short_excerpt() -> None:
    with pytest.raises(MalformedRoutingDecisionError) as info:
        parse("x" * 5000)
    assert len(info.value.details["response_excerpt"]) == 300
