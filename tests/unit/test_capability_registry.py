import pytest

from jevpilot import CapabilityRegistry, WorkflowState
from jevpilot.exceptions import CapabilityNotFoundError, DuplicateCapabilityError
from tests.conftest import EchoInput, make_capability


def test_register_get_list_unregister(registry: CapabilityRegistry) -> None:
    cap = make_capability("x.one")
    registry.register(cap)
    assert "x.one" in registry and len(registry) == 1
    assert registry.get("x.one") is cap
    assert [s.id for s in registry.list()] == ["x.one"]
    registry.unregister("x.one")
    assert len(registry) == 0
    with pytest.raises(CapabilityNotFoundError):
        registry.get("x.one")


def test_duplicate_rejected_unless_replace(registry: CapabilityRegistry) -> None:
    registry.register(make_capability("x.one"))
    with pytest.raises(DuplicateCapabilityError):
        registry.register(make_capability("x.one"))
    registry.register(make_capability("x.one", description="v2"), replace=True)
    assert registry.get("x.one").spec.description == "v2"


def test_find_by_tags_domain_text(registry: CapabilityRegistry) -> None:
    registry.register(make_capability("a.sim", tags=frozenset({"sim", "slow"}), domain="a"))
    registry.register(
        make_capability("b.query", tags=frozenset({"db"}), domain="b", description="look things up")
    )
    assert [s.id for s in registry.find(tags=["sim"])] == ["a.sim"]
    assert [s.id for s in registry.find(domain="b")] == ["b.query"]
    assert [s.id for s in registry.find(text="LOOK")] == ["b.query"]
    assert registry.find(tags=["sim", "db"]) == []


def test_available_respects_preconditions(
    registry: CapabilityRegistry, state: WorkflowState
) -> None:
    def boom(_s: WorkflowState) -> bool:
        raise RuntimeError("broken precondition")

    registry.register(make_capability("yes"))
    registry.register(make_capability("no", applicable=lambda s: False))
    registry.register(make_capability("broken", applicable=boom))
    assert [s.id for s in registry.available(state)] == ["yes"]


def test_spec_describe_is_json_ready(registry: CapabilityRegistry) -> None:
    import json

    cap = make_capability("x.echo", input_schema=EchoInput, tags=frozenset({"b", "a"}))
    desc = cap.spec.describe()
    json.dumps(desc)
    assert desc["tags"] == ["a", "b"]
    assert desc["input_schema"]["properties"]["text"]["type"] == "string"
