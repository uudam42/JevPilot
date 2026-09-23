"""UAV materials iteration 6: first engineering forward model (unidirectional CFRP ply).

python examples/uav_composite_micromechanics.py

AS graphite fibre / IMLS epoxy (NASA ICAN data bank), Chamis micromechanics.
1. one design → predicted ply properties, with what is NOT predicted;
2. a deterministic V_f sweep (not optimisation);
3. the design through the unified evaluator against the unchanged UAV target.
"""

from __future__ import annotations

from uav_material_search import target  # sibling example (examples/ is on sys.path)

from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.design import CandidateDesign
from domains.uav_materials.evaluation import EvaluationContext
from domains.uav_materials.micromechanics import (
    MODEL,
    ContinuousFiberCompositeDesignSpace,
    ContinuousFiberMicromechanicsPredictor,
)

SPACE = ContinuousFiberCompositeDesignSpace()
PREDICTOR = ContinuousFiberMicromechanicsPredictor()
SHORT = {
    "density": ("rho", "kg/m^3"),
    "ply_longitudinal_modulus": ("E11", "GPa"),
    "ply_transverse_modulus": ("E22", "GPa"),
    "ply_inplane_shear_modulus": ("G12", "GPa"),
    "ply_major_poisson_ratio": ("nu12", "1"),
    "ply_longitudinal_tensile_strength": ("S11T", "MPa"),
    "ply_longitudinal_tensile_failure_strain": ("eps1T", "%"),
}


def design(vf: float) -> CandidateDesign:
    return SPACE.design(
        {"fiber_id": "AS--", "matrix_id": "IMLS", "fiber_volume_fraction": vf},
        generation_method="manual (example)",
    )


def main() -> None:
    d = design(0.60)
    profile = PREDICTOR.predict(d)
    print(f"model {MODEL.label} ({MODEL.kind}); validation: {MODEL.validation.status}")
    print("design: AS-- graphite fibre / IMLS epoxy, V_f = 0.60 (unidirectional ply)")
    for m in profile.predictions:
        src = m.provenance.sources[0].metadata if m.provenance else {}
        print(
            f"  {m.property:42s} {m.value:10.4g} {m.unit:7s} [{m.conditions.other['direction']}]"
            f"  {src.get('equation')}"
        )
    print(f"  not predicted: {', '.join(profile.not_predicted)}")

    print("\nV_f sweep (deterministic; not optimisation):")
    print("  V_f   " + "  ".join(f"{s[0]:>7s}" for s in SHORT.values()))
    for vf in (0.50, 0.55, 0.60, 0.65):
        values = {m.property: m.value for m in PREDICTOR.predict(design(vf)).predictions}
        print(f"  {vf:.2f}  " + "  ".join(f"{values[p]:7.4g}" for p in SHORT))

    context = EvaluationContext.fit(target(), real_candidates())
    e = context.evaluate(SPACE.candidate_from_design(d))
    print(
        f"\nunified evaluation vs the unchanged target: feasibility={e.feasibility}, "
        f"evidence={e.evidence.classification}"
    )
    for c in e.checks:
        print(f"  {c.requirement.property:24s} {c.status.value:12s} {c.reason[:70]}")
    print(
        f"  distance={e.distance}  pessimistic={e.pessimistic_distance:.4f}  "
        f"coverage={e.coverage:.0%}"
    )
    for w in e.warnings:
        print(f"  note: {w}")


if __name__ == "__main__":
    main()
