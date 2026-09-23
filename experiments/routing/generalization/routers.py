"""Router construction for generalization experiments, with strict offline/live separation.

=========  =========================================================================
mode       routers
=========  =========================================================================
offline    ``rule``: RuleRouter(dev-derived rules).
           ``llm`` / ``jev``: LLMRouter / JevRouter over **fake** adapters driven by
           :func:`naive_request_policy`. Infrastructure tests only.
live       ``rule``: same RuleRouter.
           ``llm``: LLMRouter + AnthropicLLMAdapter (strict single-model by default).
           ``jev``: JevRouter + TypeSafeJevAdapter.
           A router whose SDK, credentials or model preflight is missing is reported
           **unavailable**. It is never replaced with anything else.
=========  =========================================================================

Fallback chains (``jev>rule``, ``llm>rule``, ``jev>llm>rule``) are *routing
systems*, run only in the separate ``fallback`` experiment.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from experiments.routing.generalization.baselines import naive_request_policy, toolkit_rules
from jevpilot import FallbackRouter, JevRouter, LLMRouter, Router, RuleRouter
from jevpilot.adapters import FakeJevAdapter, FakeLLMAdapter, with_faults

MODES = ("offline", "live")
ROUTERS = ("rule", "llm", "jev")
FAKE_PROVIDER = "fake"
OFFLINE_LABEL = "OFFLINE FAKE: naive token-overlap policy; infrastructure test, not a model"


class FakeAdapterInLiveModeError(RuntimeError):
    """A live run was about to use a fake adapter."""


@dataclass
class LiveOptions:
    strict: bool = True
    llm_model: str | None = None
    llm_effort: str | None = None
    llm_max_tokens: int = 16000
    llm_price: tuple[float, float] | None = None
    jev_model: str | None = None
    jev_price_input: float | None = None
    pricing_source: str | None = None
    timeout_s: float = 120.0


@dataclass
class RouterSlot:
    """One router in a run: a factory, or the reason it is unavailable."""

    name: str
    kind: str  # rule | fake | live
    factory: Callable[[int], Router] | None = None
    unavailable: str | None = None
    preflight: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.factory is not None


def rule_router() -> Router:
    return RuleRouter(toolkit_rules())


def _offline(name: str, faults: dict[str, float]) -> Callable[[int], Router]:
    def make(seed: int) -> Router:
        policy = with_faults(naive_request_policy, faults, seed=seed)
        if name == "llm":
            return LLMRouter(FakeLLMAdapter(policy=policy, label=OFFLINE_LABEL))
        return JevRouter(FakeJevAdapter(policy=policy, label=OFFLINE_LABEL))

    return make


def _live_adapter(name: str, opts: LiveOptions) -> Any:
    if name == "llm":
        module: Any = importlib.import_module("integrations.anthropic_llm")
        return module.AnthropicLLMAdapter(
            model=opts.llm_model,
            effort=opts.llm_effort,
            max_tokens=opts.llm_max_tokens,
            refusal_fallbacks=not opts.strict,
            price_per_mtok=opts.llm_price,
            pricing_source=opts.pricing_source,
        )
    module = importlib.import_module("integrations.typesafe_jev")
    return module.TypeSafeJevAdapter(
        model=opts.jev_model,
        price_per_mtok_input=opts.jev_price_input,
        pricing_source=opts.pricing_source,
    )


def assert_not_fake(router: Router) -> None:
    """Refuse any router that (anywhere in a fallback chain) uses a fake adapter."""
    routers = router.chain if isinstance(router, FallbackRouter) else (router,)
    for r in routers:
        adapter = getattr(r, "adapter", None)
        if adapter is None:
            continue
        if isinstance(adapter, FakeJevAdapter | FakeLLMAdapter) or (
            adapter.describe().get("provider") == FAKE_PROVIDER
        ):
            raise FakeAdapterInLiveModeError(f"{r.router_id} uses a fake adapter in live mode")


def build_slot(
    name: str,
    mode: str,
    *,
    faults: dict[str, float] | None = None,
    live: LiveOptions | None = None,
    preflight: bool = True,
) -> RouterSlot:
    if name == "rule":
        return RouterSlot(name, "rule", lambda seed: rule_router(), config=rule_router().config())
    if name not in ("llm", "jev"):
        raise ValueError(f"unknown router {name!r}; choose from {ROUTERS}")
    if mode == "offline":
        make = _offline(name, faults or {})
        return RouterSlot(name, "fake", make, config={**make(0).config(), "faults": faults or {}})
    opts = live or LiveOptions()
    try:
        adapter = _live_adapter(name, opts)
    except Exception as exc:  # SDK missing, credentials missing, …
        return RouterSlot(name, "live", unavailable=f"{type(exc).__name__}: {exc}")
    checked: dict[str, Any] = {}
    if preflight:
        try:
            checked = adapter.verify()
        except Exception as exc:
            return RouterSlot(name, "live", unavailable=f"preflight failed: {exc}")
    router: Router = (
        LLMRouter(adapter, timeout_s=opts.timeout_s)
        if name == "llm"
        else JevRouter(adapter, timeout_s=opts.timeout_s)
    )
    assert_not_fake(router)
    return RouterSlot(name, "live", lambda seed: router, preflight=checked, config=router.config())


def build_chain(chain: str, slots: dict[str, RouterSlot]) -> RouterSlot:
    """``"jev>llm>rule"``: a fallback *system* built from already-built slots."""
    parts = chain.split(">")
    missing = [p for p in parts if not slots.get(p) or not slots[p].available]
    if missing:
        return RouterSlot(chain, "system", unavailable=f"component(s) unavailable: {missing}")
    factories = [slots[p].factory for p in parts]

    def make(seed: int) -> Router:
        routers = [f(seed) for f in factories if f is not None]
        return FallbackRouter(routers[0], *routers[1:], router_id=f"system({chain})")

    return RouterSlot(chain, "system", make, config=make(0).config())
