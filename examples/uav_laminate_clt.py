"""UAV materials iteration 8: lamina → symmetric laminate (Classical Lamination Theory).

python examples/uav_laminate_clt.py

1. CLT reproduces the NASA RP-1351 worked numbers (Q, Qbar(45), A, B);
2. a deterministic layup sweep (not optimisation): stacking sequence → Ex, Ey, Gxy, nuxy;
3. one laminate through the unified evaluator: the unchanged UAV target, then a
   direction-aware laminate DEMONSTRATION target (architecture test only);
4. one provenance chain from a laminate property back to the constituent documents.
"""

from __future__ import annotations

from uav_material_search import target  # sibling example (examples/ is on sys.path)

from domains.uav_materials.clt import (
    LaminatePredictor,
    abd,
    layup_design,
    reduced_stiffness,
    transformed_stiffness,
)
from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.demo import example_laminate, laminate_demo_target
from domains.uav_materials.evaluation import CandidateEvaluation, EvaluationContext


def validation() -> None:
    print("RP-1351 validation (inputs E1=20.01e6, E2=1.301e6, G12=1.001e6 psi, nu12=0.3, "
          "nu21=0.02 as printed; t=0.005 in):")  # fmt: skip
    q = reduced_stiffness(20.01e6, 1.301e6, 1.001e6, 0.3, nu21=0.02)
    q45 = transformed_stiffness(q, 45)
    t = 0.005
    a9, b9, _, _ = abd([(transformed_stiffness(q, 90), t), (q, t), (q45, t)])
    _, b2, _, _ = abd([(q45, t), (q, t)])
    rows = [
        ("Q11 psi", q[0][0], 20_130_785), ("Q12", q[0][1], 392_656),
        ("Q22", q[1][1], 1_308_853), ("Q66", q[2][2], 1_001_000),
        ("Qbar11(45)", q45[0][0], 6_557_237), ("Qbar12(45)", q45[0][1], 4_555_238),
        ("Qbar16(45)", q45[0][2], 4_705_483), ("Qbar66(45)", q45[2][2], 5_163_582),
        ("ex.9 A11 lb/in", a9[0][0], 139_984), ("ex.9 A16", a9[0][2], 23_527),
        ("ex.9 A66", a9[2][2], 35_828), ("ex.9 B11 lb", b9[0][0], 131.2),
        ("ex.9 B22", b9[1][1], -339.4), ("ex.9 B16", b9[0][2], 117.6),
        ("ex.2 B11", b2[0][0], 170), ("ex.2 B22", b2[1][1], -66), ("ex.2 B16", b2[0][2], -59),
    ]  # fmt: skip
    for name, ours, printed in rows:
        print(
            f"  {name:16s} ours {ours:14,.1f}   printed {printed:14,}   diff {ours - printed:+.2f}"
        )


def sweep() -> None:
    print("\nlayup sweep, AS/IMLS V_f=0.60, 0.127 mm plies (deterministic; not optimisation):")
    print(f"  {'layup':16s} {'plies':>5s} {'h mm':>6s} {'Ex GPa':>8s} {'Ey GPa':>8s} "
          f"{'Gxy GPa':>8s} {'nuxy':>7s}  B=0")  # fmt: skip
    predictor = LaminatePredictor()
    for half in ("0", "90", "0/90", "45/-45", "0/45/-45/90", "0/0/45/-45", "0/60/-60"):
        lam = predictor.stiffness(layup_design(half, generation_method="sweep"))
        c = lam.constants
        print(
            f"  {lam.layup:16s} {len(lam.plies):5d} {lam.thickness_m * 1000:6.3f} "
            f"{c['laminate_ex'] / 1e9:8.2f} {c['laminate_ey'] / 1e9:8.2f} "
            f"{c['laminate_gxy'] / 1e9:8.2f} {c['laminate_nuxy']:7.4f}  {lam.b_is_zero}"
        )


def show(title: str, e: CandidateEvaluation) -> None:
    print(f"\n{title}\n  {e.name}")
    print(f"  system={e.material_system}  feasibility={e.feasibility}  "
          f"evidence={e.evidence.classification}")  # fmt: skip
    for o in e.outcomes:
        value = next(
            (f"{c.value:.4g} {c.requirement.unit}" for c in e.checks
             if o.properties and c.requirement.property == o.properties[0] and c.value is not None),
            "",
        )  # fmt: skip
        print(f"    {'hard' if o.hard else 'soft'} {o.requirement:50s} {o.resolution:11s} "
              f"-> {', '.join(o.properties) or '-':28s} {o.check or '-':12s} {value:12s} "
              f"gap={o.gap or '-'}")  # fmt: skip


def chain(e: CandidateEvaluation) -> None:
    m = next(u for u in e.properties_used if u.property == "laminate_ex")
    print(f"\nprovenance chain for laminate_ex = {m.value} ({m.source}):")
    lam = example_laminate()
    measurement = lam.material.measurements("laminate_ex")[0]
    assert measurement.provenance is not None
    for s in measurement.provenance.sources:
        meta = s.metadata
        detail = meta.get("equation") or meta.get("documents") or ""
        print(f"  [{meta.get('chain_level', '')[:1]}] {s.kind:15s} {s.identifier:48s} {detail}")


def main() -> None:
    validation()
    sweep()
    real = real_candidates()
    laminate = example_laminate("0/45/-45/90")
    uav = EvaluationContext.fit(target(), real)
    show("[0/45/-45/90]s under the unchanged UAV target", uav.evaluate(laminate))
    demo = EvaluationContext.fit(laminate_demo_target(), real)
    e = demo.evaluate(laminate)
    show("[0/45/-45/90]s under the laminate DEMONSTRATION target", e)
    show("[0/90]s under the laminate DEMONSTRATION target", demo.evaluate(example_laminate("0/90")))
    chain(e)


if __name__ == "__main__":
    main()
