from jevpilot import Rule, RuleRouter, ScriptedRouter, WorkflowState
from tests.conftest import make_capability

SPECS = [make_capability("a").spec, make_capability("b").spec]


def test_scripted_router_replays_then_stops(state: WorkflowState) -> None:
    r = ScriptedRouter([("a", {"x": 1}), ("b", {})])
    d1, d2, d3 = (r.select_next(state, SPECS) for _ in range(3))
    assert (d1.capability_id, d1.inputs) == ("a", {"x": 1})
    assert d2.capability_id == "b"
    assert d3.is_no_action
    r.reset()
    assert r.select_next(state, SPECS).capability_id == "a"


def test_rule_router_first_match_among_available(state: WorkflowState) -> None:
    r = RuleRouter(
        [
            Rule("missing"),
            Rule("a", when=lambda s: s.step > 0),
            Rule("b", inputs=lambda s: {"step": s.step}),
        ]
    )
    d = r.select_next(state, SPECS)
    assert d.capability_id == "b" and d.inputs == {"step": 0}
    assert d.alternatives == ("a",)
    assert r.select_next(state.evolve(step=1), SPECS).capability_id == "a"


def test_rule_router_no_match(state: WorkflowState) -> None:
    assert RuleRouter([Rule("zzz")]).select_next(state, SPECS).is_no_action
