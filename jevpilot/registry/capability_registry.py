"""Dynamic registry of capabilities."""

from __future__ import annotations

import builtins
import logging
from collections.abc import Callable, Iterable

from jevpilot.core import WorkflowState
from jevpilot.exceptions import CapabilityNotFoundError, DuplicateCapabilityError
from jevpilot.interfaces.capability import Capability, CapabilitySpec

logger = logging.getLogger(__name__)


class CapabilityRegistry:
    """Holds capabilities registered by domains or plugins at runtime.

    Discovery methods (:meth:`list`, :meth:`find`, :meth:`available`) return
    :class:`CapabilitySpec` descriptors for routers; :meth:`get` returns the
    executable object and is intended for the executor only.
    """

    def __init__(self) -> None:
        self._caps: dict[str, Capability] = {}

    def register(self, capability: Capability, *, replace: bool = False) -> None:
        cid = capability.spec.id
        if cid in self._caps and not replace:
            raise DuplicateCapabilityError(f"capability {cid!r} is already registered")
        self._caps[cid] = capability

    def unregister(self, capability_id: str) -> Capability:
        try:
            return self._caps.pop(capability_id)
        except KeyError:
            raise CapabilityNotFoundError(capability_id) from None

    def get(self, capability_id: str) -> Capability:
        try:
            return self._caps[capability_id]
        except KeyError:
            raise CapabilityNotFoundError(capability_id) from None

    def list(self) -> list[CapabilitySpec]:
        return [c.spec for c in self._caps.values()]

    def find(
        self,
        *,
        tags: Iterable[str] | None = None,
        domain: str | None = None,
        text: str | None = None,
        predicate: Callable[[CapabilitySpec], bool] | None = None,
    ) -> builtins.list[CapabilitySpec]:
        """Filter specs by tags (all must match), domain, free text, or predicate."""
        wanted = set(tags or ())
        needle = text.lower() if text else None
        out = []
        for spec in self.list():
            if wanted and not wanted <= spec.tags:
                continue
            if domain is not None and spec.domain != domain:
                continue
            if needle and needle not in f"{spec.id} {spec.name} {spec.description}".lower():
                continue
            if predicate and not predicate(spec):
                continue
            out.append(spec)
        return out

    def available(self, state: WorkflowState) -> builtins.list[CapabilitySpec]:
        """Specs whose executable preconditions hold in ``state``.

        A capability whose ``is_applicable`` raises is treated as unavailable.
        """
        out = []
        for cap in self._caps.values():
            try:
                ok = cap.is_applicable(state)
            except Exception:
                logger.exception("is_applicable failed for %s", cap.spec.id)
                ok = False
            if ok:
                out.append(cap.spec)
        return out

    def __contains__(self, capability_id: object) -> bool:
        return capability_id in self._caps

    def __len__(self) -> int:
        return len(self._caps)
