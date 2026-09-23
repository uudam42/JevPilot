"""Ground truth for the toolkit world, computed by exhaustive search. BENCHMARK-ONLY.

The oracle is how the benchmark knows what is correct. Routers must never see
it; ``tests/benchmark/test_leakage.py`` enforces that.

At any state (facts, known goal parameters, capabilities known to be
permanently unavailable) the oracle computes the shortest number of successful
actions that reach the goal without producing a forbidden fact. From that:

* ``invoke``: a goal-directed action exists. **Acceptable** = offered
  capabilities that are applicable now and lie on *some* shortest plan (so
  ``A→B`` and ``B→A`` are both fine when both are optimal). **Preferred** =
  the cheapest of those.
* ``finish``: the goal already holds. Acceptable: ``finish`` (or ``idle``).
* ``ask_human``: the goal is reachable only if a missing goal parameter
  (information) were provided. Acceptable: ``ask_human`` / ``idle``, and also
  progress actions that do not need the missing information.
* ``impossible``: no offered capability can ever reach the goal. Acceptable:
  ``ask_human`` / ``idle``. Invoking anything is unnecessary.

Transient failures are ignored (the oracle assumes a retry can succeed).
Permanent failures that the router has *observed* remove that capability.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache

from experiments.routing.generalization.world import CapabilityDef

INF = 10**9


@dataclass(frozen=True)
class Problem:
    offered: tuple[CapabilityDef, ...]
    goal: frozenset[str]
    forbidden: frozenset[str]
    known_params: frozenset[str]


@dataclass(frozen=True)
class Expectation:
    kind: str  # invoke | finish | ask_human | impossible
    distance: int | None  # shortest remaining plan length (None: unreachable)
    acceptable: frozenset[str]  # canonical capability ids
    preferred: frozenset[str]
    no_action_intents: frozenset[str]
    forbidden: frozenset[str]
    unnecessary: frozenset[str]


def _usable(d: CapabilityDef, p: Problem, excluded: frozenset[str]) -> bool:
    return (
        d.id not in excluded
        and d.required_params <= p.known_params
        and not (d.produces & p.forbidden)
    )


def _relevant(caps: Iterable[CapabilityDef], goal: frozenset[str]) -> frozenset[str]:
    relevant, caps = set(goal), list(caps)
    changed = True
    while changed:
        changed = False
        for d in caps:
            if d.produces & relevant and not d.requires <= relevant:
                relevant |= d.requires
                changed = True
    return frozenset(relevant)


@lru_cache(maxsize=200_000)
def _distance(facts: frozenset[str], p: Problem, excluded: frozenset[str]) -> int:
    caps = [d for d in p.offered if _usable(d, p, excluded)]
    relevant = _relevant(caps, p.goal)
    caps = [d for d in caps if d.produces & relevant]
    start = facts & relevant
    if p.goal <= start:
        return 0
    seen = {start}
    queue: deque[tuple[frozenset[str], int]] = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        for d in caps:
            if d.requires <= node and not d.produces & relevant <= node:
                nxt = node | (d.produces & relevant)
                if p.goal <= nxt:
                    return depth + 1
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))
    return INF


def distance(
    facts: frozenset[str], p: Problem, excluded: frozenset[str] = frozenset()
) -> int | None:
    d = _distance(facts, p, excluded)
    return None if d >= INF else d


def _progress_actions(
    facts: frozenset[str], p: Problem, excluded: frozenset[str], search: Problem, dist: int
) -> frozenset[str]:
    """Offered, usable (under ``p``) capabilities that shorten a shortest plan under ``search``."""
    out = set()
    for d in p.offered:
        if not _usable(d, p, excluded) or not d.requires <= facts or d.produces <= facts:
            continue
        if _distance(facts | d.produces, search, excluded) == dist - 1:
            out.add(d.id)
    return frozenset(out)


def expect(
    facts: frozenset[str],
    p: Problem,
    *,
    excluded: frozenset[str] = frozenset(),
    all_params: frozenset[str] = frozenset(),
) -> Expectation:
    """What a correct router may do at this state (see module docstring)."""
    forbidden = frozenset(d.id for d in p.offered if d.produces & p.forbidden)
    offered = frozenset(d.id for d in p.offered)
    by_id = {d.id: d for d in p.offered}

    def build(
        kind: str, dist: int | None, acceptable: frozenset[str], no_action: Iterable[str]
    ) -> Expectation:
        preferred = acceptable
        if acceptable:
            cheapest = min(by_id[c].cost for c in acceptable)
            preferred = frozenset(c for c in acceptable if by_id[c].cost == cheapest)
        return Expectation(
            kind,
            dist,
            acceptable,
            preferred,
            frozenset(no_action),
            forbidden,
            offered - acceptable - forbidden,
        )

    if p.goal <= facts:
        return build("finish", 0, frozenset(), ("finish", "idle"))
    d0 = _distance(facts, p, excluded)
    if d0 < INF:
        return build("invoke", d0, _progress_actions(facts, p, excluded, p, d0), ())
    informed = Problem(p.offered, p.goal, p.forbidden, p.known_params | all_params)
    d1 = _distance(facts, informed, excluded)
    if d1 < INF:
        progress = _progress_actions(facts, p, excluded, informed, d1)
        e = build("ask_human", None, progress, ("ask_human", "idle"))
        # Asking early is preferred; progress without the missing information is acceptable.
        return Expectation(
            e.kind, None, e.acceptable, frozenset(), e.no_action_intents, e.forbidden, e.unnecessary
        )
    return build("impossible", None, frozenset(), ("ask_human", "idle"))


def reference_plan(
    facts: frozenset[str], p: Problem, excluded: frozenset[str] = frozenset()
) -> list[str]:
    """One shortest plan (preferred action first, ties by id). Empty if unreachable or done."""
    plan: list[str] = []
    by_id = {d.id: d for d in p.offered}
    while True:
        e = expect(facts, p, excluded=excluded)
        if e.kind != "invoke" or not e.preferred:
            return plan
        step = sorted(e.preferred)[0]
        plan.append(step)
        facts = facts | by_id[step].produces


def param_names(defs: Mapping[str, CapabilityDef]) -> frozenset[str]:
    return frozenset(p for d in defs.values() for p in d.required_params)
