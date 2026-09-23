"""Loads domain modules and wires their contributions into the registries.

Domains are passed in as objects, or referenced by string
(``"package.module:ClassName"``) or by entry point
(group ``jevpilot.domains``). The core therefore never has a static import of
any domain.
"""

from __future__ import annotations

import builtins
import importlib
from importlib.metadata import entry_points

from jevpilot.exceptions import DomainLoadError, DuplicateCapabilityError
from jevpilot.interfaces.domain import DomainModule
from jevpilot.interfaces.evaluator import Evaluator
from jevpilot.interfaces.reducer import StateReducer
from jevpilot.registry.capability_registry import CapabilityRegistry

ENTRY_POINT_GROUP = "jevpilot.domains"


class DomainRegistry:
    def __init__(self, capabilities: CapabilityRegistry) -> None:
        self._capabilities = capabilities
        self._domains: dict[str, DomainModule] = {}
        self._owned: dict[str, list[str]] = {}

    # -- loading ------------------------------------------------------------

    def load(self, domain: DomainModule) -> DomainModule:
        """Register a domain and all of its capabilities (atomically)."""
        if domain.name in self._domains:
            raise DomainLoadError(f"domain {domain.name!r} is already loaded")
        caps = list(domain.capabilities())
        clashes = [c.spec.id for c in caps if c.spec.id in self._capabilities]
        if clashes:
            raise DuplicateCapabilityError(
                f"domain {domain.name!r} redefines capabilities: {clashes}"
            )
        for cap in caps:
            self._capabilities.register(cap)
        self._domains[domain.name] = domain
        self._owned[domain.name] = [c.spec.id for c in caps]
        return domain

    def load_from_path(self, path: str) -> DomainModule:
        """Import and load ``"module.path:ClassOrFactory"``."""
        return self.load(_instantiate(path))

    def load_entry_point(self, name: str) -> DomainModule:
        matches = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == name]
        if not matches:
            raise DomainLoadError(f"no domain entry point named {name!r}")
        return self.load(_coerce(matches[0].load(), name))

    @staticmethod
    def discover() -> list[str]:
        """Names of domains advertised through installed entry points."""
        return sorted(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))

    def unload(self, name: str) -> None:
        if name not in self._domains:
            raise DomainLoadError(f"domain {name!r} is not loaded")
        for cid in self._owned.pop(name):
            if cid in self._capabilities:
                self._capabilities.unregister(cid)
        del self._domains[name]

    # -- queries ------------------------------------------------------------

    def get(self, name: str) -> DomainModule:
        try:
            return self._domains[name]
        except KeyError:
            raise DomainLoadError(f"domain {name!r} is not loaded") from None

    def list(self) -> list[DomainModule]:
        return list(self._domains.values())

    def evaluators(self) -> builtins.list[Evaluator]:
        return [e for d in self._domains.values() for e in d.evaluators()]

    def reducers(self) -> builtins.list[StateReducer]:
        return [r for d in self._domains.values() for r in d.reducers()]

    def __contains__(self, name: object) -> bool:
        return name in self._domains


def _instantiate(path: str) -> DomainModule:
    module_name, sep, attr = path.partition(":")
    if not sep or not attr:
        raise DomainLoadError(f"expected 'module:attribute', got {path!r}")
    try:
        target = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise DomainLoadError(f"cannot import domain {path!r}: {exc}") from exc
    return _coerce(target, path)


def _coerce(target: object, label: str) -> DomainModule:
    obj = target() if callable(target) and not isinstance(target, DomainModule) else target
    if not isinstance(obj, DomainModule):
        raise DomainLoadError(f"{label!r} did not produce a DomainModule")
    return obj
