from collections.abc import Sequence

import pytest

from jevpilot import Capability, CapabilityRegistry, DomainModule, DomainRegistry, WorkflowState
from jevpilot.exceptions import DomainLoadError, DuplicateCapabilityError
from tests.conftest import make_capability


class ToyDomain(DomainModule):
    def __init__(self, name: str, cap_ids: Sequence[str]) -> None:
        self.name = name
        self._ids = cap_ids

    def capabilities(self) -> Sequence[Capability]:
        return [make_capability(i, domain=self.name) for i in self._ids]


def test_load_and_unload(registry: CapabilityRegistry) -> None:
    domains = DomainRegistry(registry)
    domains.load(ToyDomain("toy", ["toy.a", "toy.b"]))
    assert "toy" in domains and len(registry) == 2
    domains.unload("toy")
    assert "toy" not in domains and len(registry) == 0


def test_load_is_atomic_on_clash(registry: CapabilityRegistry) -> None:
    domains = DomainRegistry(registry)
    domains.load(ToyDomain("one", ["shared.x"]))
    with pytest.raises(DuplicateCapabilityError):
        domains.load(ToyDomain("two", ["two.y", "shared.x"]))
    assert "two.y" not in registry and "two" not in domains


def test_load_from_path(registry: CapabilityRegistry) -> None:
    domains = DomainRegistry(registry)
    d = domains.load_from_path("domains.demo:ArithmeticDomain")
    assert d.name == "arithmetic" and "arith.add" in registry
    with pytest.raises(DomainLoadError):
        domains.load_from_path("domains.nope:Missing")
    with pytest.raises(DomainLoadError):
        domains.load_from_path("no_colon_here")


def test_entry_point_discovery(registry: CapabilityRegistry) -> None:
    assert {"arithmetic", "stats"} <= set(DomainRegistry.discover())
    domains = DomainRegistry(registry)
    assert domains.load_entry_point("stats").name == "stats"


def test_create_state_tags_domain() -> None:
    s = ToyDomain("toy", []).create_state("do it")
    assert isinstance(s, WorkflowState) and s.metadata["domain"] == "toy"
