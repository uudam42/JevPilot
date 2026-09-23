"""UAV materials iteration 7: engineering requirements → material-system observables.

python examples/uav_requirement_semantics.py

1. a real metal under the unchanged UAV target: every requirement resolves as before;
2. the AS/IMLS V_f=0.60 ply under the same target: resolved / ambiguous / unsupported;
3. the same ply under a direction-aware DEMONSTRATION target (architecture test only;
   not a replacement for the UAV target), where E11 and S11T legitimately apply.
"""

from __future__ import annotations

from uav_material_search import target  # sibling example (examples/ is on sys.path)

from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.demo import directional_demo_target, example_ply
from domains.uav_materials.evaluation import CandidateEvaluation, EvaluationContext


def show(title: str, e: CandidateEvaluation) -> None:
    print(f"\n{title}\n  {e.name}")
    print(
        f"  system={e.material_system}  feasibility={e.feasibility}  "
        f"evidence={e.evidence.classification}"
    )
    for o in e.outcomes:
        role = "hard" if o.hard else "soft"
        props = ", ".join(o.properties) or "-"
        print(
            f"    {role} {o.requirement:52s} {o.resolution:11s} -> {props:36s} "
            f"check={o.check or '-':12s} gap={o.gap or '-'}"
        )


def main() -> None:
    real = real_candidates()
    uav = EvaluationContext.fit(target(), real)
    bar = next(c for c in real if c.candidate_id == "mil5j-3.7.6.0-d")
    show("1. existing metal, unchanged UAV target", uav.evaluate(bar))

    ply = example_ply()
    e = uav.evaluate(ply)
    show("2. AS/IMLS V_f=0.60 ply, unchanged UAV target", e)
    for o in e.outcomes:
        if o.gap in ("ambiguous", "unsupported"):
            print(f"      {o.gap}: {o.requirement}: {o.reason}")

    demo = EvaluationContext.fit(directional_demo_target(), real)
    d = demo.evaluate(ply)
    show("3. same ply, direction-aware DEMONSTRATION target", d)
    for c in d.checks:
        print(f"      {c.requirement.property:36s} {c.status.value:10s} {c.reason[:70]}")
    show("   (a metal under the demonstration target, isotropy assumed)", demo.evaluate(bar))


if __name__ == "__main__":
    main()
