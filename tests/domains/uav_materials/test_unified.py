"""Unified existing-material + designed-candidate evaluation (offline, synthetic model only)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from domains.uav_materials import MaterialCandidate, Measurement
from domains.uav_materials.datasets import load_real_materials, real_candidates
from domains.uav_materials.design import (
    SYNTHETIC_SPACE,
    BoundedDesignSpace,
    CandidateDesign,
    DesignInvalid,
    DesignVariable,
    ModelIdentity,
    PredictedPropertyProfile,
    PropertyPredictor,
    SyntheticLinearPredictor,
    VariableKind,
    candidate_from_prediction,
)
from domains.uav_materials.evaluation import (
    CandidatePropertyProfile,
    EvaluationContext,
    candidate_from_record,
    evaluate_candidate,
    evaluate_candidates,
)
from domains.uav_materials.evidence import GapClass, GapReason
from domains.uav_materials.schema import EvidenceType, MaterialFamily
from domains.uav_materials.search import CheckStatus, Feasibility, search_materials
from examples.uav_material_search import target
from jevpilot import Provenance, SourceRef

ROOT = Path(__file__).resolve().parents[3]
PREDICTOR = SyntheticLinearPredictor()


def design(a: float = 0.4, b: float = 0.6) -> CandidateDesign:
    return SYNTHETIC_SPACE.design({"fraction_a": a, "fraction_b": b}, generation_method="test")


def virtual(a: float = 0.4, b: float = 0.6) -> MaterialCandidate:
    return SYNTHETIC_SPACE.candidate_from_design(design(a, b), PREDICTOR)


@pytest.fixture(scope="module")
def context() -> EvaluationContext:
    return EvaluationContext.fit(target(), real_candidates())


# -- existing material → unified candidate ----------------------------------------------------


def test_existing_material_adapter_wraps_without_copying() -> None:
    record = load_real_materials()[0]
    c = candidate_from_record(record)
    assert c.origin == "existing" and c.candidate_id == record.material_id
    assert c.material is record
    view = CandidatePropertyProfile.of(c)
    m = view.measurements("tensile_strength")[0]
    assert any(m is x for x in record.strength)  # the same object: provenance intact
    assert view.evidence_types("tensile_strength") == (EvidenceType.DATASHEET,)


# -- design space and designs -----------------------------------------------------------------


def test_design_preserves_space_variables_constraints_and_method() -> None:
    d = design()
    assert (d.design_space, d.design_space_version) == ("synthetic_two_fraction_v0", "0")
    assert d.variables == {"fraction_a": 0.4, "fraction_b": 0.6}
    assert [v.name for v in d.variable_specs] == ["fraction_a", "fraction_b"]
    assert d.constraints[0].name == "fractions_at_most_one"
    assert d.generation_method == "test"
    assert design().design_id == d.design_id != design(0.5, 0.5).design_id


@pytest.mark.parametrize(
    ("values", "problem"),
    [
        ({"fraction_a": 0.7, "fraction_b": 0.6}, "fractions_at_most_one"),
        ({"fraction_a": -0.1, "fraction_b": 0.2}, "lower bound"),
        ({"fraction_a": 0.1}, "missing"),
        ({"fraction_a": 0.1, "fraction_b": 0.1, "fraction_c": 0.1}, "unknown variable"),
        ({"fraction_a": "x", "fraction_b": 0.1}, "not a number"),
        ({"fraction_a": float("nan"), "fraction_b": 0.1}, "not finite"),
    ],
)
def test_design_space_rejects_invalid_designs(values: dict[str, object], problem: str) -> None:
    with pytest.raises(DesignInvalid, match=problem):
        SYNTHETIC_SPACE.design(values, generation_method="test")  # type: ignore[arg-type]


def test_design_space_checks_space_identity_and_other_variable_kinds() -> None:
    other = BoundedDesignSpace(
        "other_space",
        "1",
        MaterialFamily.COMPOSITE,
        (
            DesignVariable(
                name="fibre", kind=VariableKind.CATEGORICAL, choices=("glass", "carbon")
            ),
            DesignVariable(name="plies", kind=VariableKind.INTEGER, lower=1, upper=64),
        ),
    )
    ok = other.design({"fibre": "carbon", "plies": 8}, generation_method="test")
    assert other.validate_design(ok) == []
    assert any("belongs to other_space" in p for p in SYNTHETIC_SPACE.validate_design(ok))
    with pytest.raises(DesignInvalid, match="not one of"):
        other.design({"fibre": "kevlar", "plies": 8}, generation_method="test")
    with pytest.raises(DesignInvalid, match="not an integer"):
        other.design({"fibre": "glass", "plies": 2.5}, generation_method="test")
    with pytest.raises(DesignInvalid):
        PREDICTOR.predict(ok)  # the synthetic model covers only its own space


# -- predictor --------------------------------------------------------------------------------


def test_predictor_contract_and_deterministic_output() -> None:
    assert isinstance(PREDICTOR, PropertyPredictor)
    one, two = PREDICTOR.predict(design()), SyntheticLinearPredictor().predict(design())
    values = {m.property: (m.value, m.unit) for m in one.predictions}
    assert values == {
        "density": (2100.0, "kg/m^3"),
        "tensile_strength": (440.0, "MPa"),
        "youngs_modulus": (52.0, "GPa"),
    }
    assert [(m.value, m.provenance.id) for m in one.predictions if m.provenance] == [
        (m.value, m.provenance.id) for m in two.predictions if m.provenance
    ]


def test_predicted_property_provenance() -> None:
    p = PREDICTOR.predict(design())
    assert p.model.label == "synthetic-linear/0" and not p.model.engineering_valid
    assert p.model.validation.status == "not_validated" and p.model.assumptions
    for m in p.predictions:
        assert m.basis == "predicted" and m.evidence_type is EvidenceType.PREDICTED
        assert m.provenance_status == "synthetic" and "NOT ENGINEERING-VALID" in m.notes
        assert m.provenance is not None
        src = m.provenance.sources[0]
        assert (src.kind, src.identifier, src.version) == ("model", "synthetic-linear/0", "0")
        assert src.metadata["design_id"] == p.design_id
        assert src.metadata["engineering_valid"] is False and src.metadata["assumptions"]
    c = virtual()
    kept = c.material.measurements("density")[0]
    assert kept.provenance and kept.provenance.sources[0].identifier == "synthetic-linear/0"
    assert c.design["design_space"] == "synthetic_two_fraction_v0"
    assert c.metadata["model"]["name"] == "synthetic-linear"


def test_prediction_never_masquerades_as_measurement() -> None:
    p = PREDICTOR.predict(design())
    measured = p.predictions[0].model_copy(update={"basis": "measured"})
    with pytest.raises(ValueError, match="basis=predicted"):
        PredictedPropertyProfile(design_id=p.design_id, model=p.model, predictions=(measured,))
    uncited = p.predictions[0].model_copy(
        update={"provenance": Provenance(sources=(SourceRef(kind="human", identifier="x"),))}
    )
    with pytest.raises(ValueError, match="must cite model"):
        PredictedPropertyProfile(design_id=p.design_id, model=p.model, predictions=(uncited,))
    with pytest.raises(ValueError, match="cannot be predictions"):
        MaterialCandidate(candidate_id="x", origin="existing", material=virtual().material)
    with pytest.raises(ValueError, match="cannot be an existing"):
        candidate_from_prediction(design(), p, MaterialFamily.OTHER, origin="existing")  # type: ignore[arg-type]


# -- unified evaluation -----------------------------------------------------------------------


def test_unified_hard_constraints_on_predictions(context: EvaluationContext) -> None:
    e = context.evaluate(virtual(0.4, 0.6))
    status = {c.requirement.property: c.status for c in e.checks}
    assert status["density"] is CheckStatus.SATISFIED  # 2100 <= 3000
    assert status["tensile_strength"] is CheckStatus.SATISFIED  # 440 >= 350
    weak = context.evaluate(virtual(0.4, 0.0))  # 200 MPa < 350 MPa
    assert weak.feasibility is Feasibility.INFEASIBLE
    # a violation shown only by a synthetic model is not an engineering failure
    assert weak.evidence.classification is GapClass.EVIDENCE_GAP
    tensile = next(r for r in weak.evidence.requirements if r.property == "tensile_strength")
    assert tensile.reasons == (GapReason.INSUFFICIENT_PROVENANCE,)


def test_unified_distance_uses_the_existing_ranking_formulas(context: EvaluationContext) -> None:
    e = context.evaluate(virtual())
    assert e.distance == context.extractor.score(virtual().material).distance
    modulus = next(c for c in e.contributions if c.property == "youngs_modulus")
    assert modulus.value == 52 and modulus.penalty == pytest.approx((65 - 52) / (120 - 65))
    assert e.coverage == pytest.approx(0.85)  # yield strength (weight 0.15) is not predicted


def test_missing_predicted_properties_stay_missing(context: EvaluationContext) -> None:
    e = context.evaluate(virtual())
    undetermined = {c.requirement.property for c in e.checks if c.status == "undetermined"}
    assert undetermined == {"elongation_at_break", "corrosion_rate", "max_service_temperature"}
    yield_c = next(c for c in e.contributions if c.property == "yield_strength")
    assert yield_c.value is None and yield_c.penalty is None
    assert e.pessimistic_distance is not None and e.distance is not None
    assert e.pessimistic_distance > e.distance
    assert "elongation_at_break" in e.evidence.missing_critical


def test_synthetic_candidate_evaluation_exposes_evidence_quality(
    context: EvaluationContext,
) -> None:
    e = context.evaluate(virtual())
    assert e.origin == "virtual" and e.engineering_valid is False
    assert e.prediction_status == "all" and e.evidence_types == {"predicted": 5}
    assert e.models == ("synthetic-linear/0",) and e.uncertainty_known == (0, 5)
    assert any("NOT ENGINEERING-VALID" in w.upper() for w in e.warnings)
    assert e.evidence.relies_on_predictions


def test_sourced_predictions_are_flagged_not_rejected() -> None:
    """Without a model-validation policy, a documented prediction is not auto-rejected."""
    model = ModelIdentity(name="fixture", version="1", kind="analytical", engineering_valid=True)
    d = design()
    pred = Measurement(
        property="density",
        value=1800,
        unit="kg/m^3",
        basis="predicted",
        provenance=Provenance(sources=(SourceRef(kind="model", identifier="fixture/1"),)),
        provenance_status="sourced",
    )
    profile = PredictedPropertyProfile(design_id=d.design_id, model=model, predictions=(pred,))
    c = candidate_from_prediction(d, profile, MaterialFamily.OTHER)
    e = evaluate_candidate(target(), c)
    density = next(r for r in e.evidence.requirements if r.property == "density" and r.hard)
    assert density.sufficient and density.evidence_type == "predicted"
    assert density.model == "fixture/1"
    assert e.evidence.relies_on_predictions and e.prediction_status == "all"
    assert e.engineering_valid is True and e.warnings  # still reported as a prediction


# -- regression: existing materials -----------------------------------------------------------


def test_existing_ranking_regression() -> None:
    cands = real_candidates()
    search = search_materials(target(), cands).assessments
    unified = evaluate_candidates(target(), cands)
    assert [a.candidate_id for a in search] == [e.candidate_id for e in unified]
    for a, e in zip(search, unified, strict=True):
        assert (a.feasibility, a.distance, a.pessimistic_distance, a.score_coverage, a.rank) == (
            e.feasibility,
            e.distance,
            e.pessimistic_distance,
            e.coverage,
            e.rank,
        )
        assert [c.status for c in a.checks] == [c.status for c in e.checks]
        assert a.contributions == e.contributions
        assert e.prediction_status == "none" and e.engineering_valid is None


def test_designed_candidate_does_not_change_existing_results(context: EvaluationContext) -> None:
    before = context.evaluate(real_candidates()[0])
    context.evaluate(virtual())
    assert context.evaluate(real_candidates()[0]) == before


# -- no LLM ----------------------------------------------------------------------------------


def test_design_and_evaluation_have_no_llm_dependency() -> None:
    for name in ("design.py", "evaluation.py", "evidence.py"):
        tree = ast.parse((ROOT / "domains/uav_materials" / name).read_text())
        imported = {
            (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(tree)
            if isinstance(n, ast.Import | ast.ImportFrom)
            for a in n.names
        }
        assert not {i for i in imported if "anthropic" in i or "typesafe" in i or "llm" in i}
        assert not {i for i in imported if i.startswith("integrations")}
