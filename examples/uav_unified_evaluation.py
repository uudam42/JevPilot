"""UAV materials iteration 5: one evaluator for existing and designed candidates.

python examples/uav_unified_evaluation.py

1. A real material (MIL-HDBK-5J 7075 bar) goes through the unified evaluator and
   agrees with the similarity search.
2. A designed candidate goes Design → SyntheticLinearPredictor → Candidate →
   the same evaluator, on the same normalisation scale as the real materials.

The designed candidate is SYNTHETIC and NOT ENGINEERING-VALID: its predictor is a
fake linear map that exists only to test the architecture.
"""

from __future__ import annotations

from uav_material_search import target  # sibling example (examples/ is on sys.path)

from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.design import SYNTHETIC_SPACE, SyntheticLinearPredictor
from domains.uav_materials.evaluation import CandidateEvaluation, EvaluationContext
from domains.uav_materials.search import search_materials


def show(e: CandidateEvaluation) -> None:
    print(
        f"{e.name}\n  origin={e.origin}  feasibility={e.feasibility}  "
        f"evidence={e.evidence.classification}"
    )
    dist = "-" if e.distance is None else f"{e.distance:.4f}"
    pess = "-" if e.pessimistic_distance is None else f"{e.pessimistic_distance:.4f}"
    cov = "-" if e.coverage is None else f"{e.coverage:.0%}"
    print(f"  distance={dist}  pessimistic={pess}  coverage={cov}")
    for c in e.checks:
        print(f"    {c.requirement.property:24s} {c.status.value:12s} {c.reason[:80]}")
    for u in e.properties_used:
        role = "hard" if u.hard else "soft"
        print(
            f"    uses {role} {u.label:20s} {u.value or '':>12s}  {u.evidence_type:9s} "
            f"{u.model or u.source or ''}"
        )
    print(f"  evidence types: {e.evidence_types}  prediction status: {e.prediction_status}")
    print(f"  uncertainty known for {e.uncertainty_known[0]}/{e.uncertainty_known[1]} values")
    for w in e.warnings:
        print(f"  WARNING: {w}")


def main() -> None:
    real = real_candidates()
    context = EvaluationContext.fit(target(), real)  # the existing materials set the scale

    bar = next(c for c in real if c.candidate_id == "mil5j-3.7.6.0-d")
    unified = context.evaluate(bar)
    ranked = {a.candidate_id: a for a in search_materials(target(), real).assessments}
    assert unified.distance == ranked[bar.candidate_id].distance
    show(unified)
    print("  (agrees with search_materials: same checks, distance and coverage)\n")

    design = SYNTHETIC_SPACE.design(
        {"fraction_a": 0.4, "fraction_b": 0.6}, generation_method="manual (example)"
    )
    virtual = SYNTHETIC_SPACE.candidate_from_design(design, SyntheticLinearPredictor(), None)
    show(context.evaluate(virtual))


if __name__ == "__main__":
    main()
