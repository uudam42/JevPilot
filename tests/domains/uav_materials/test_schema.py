"""Units, measurements, property families, material records and candidates."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from domains.uav_materials import (
    CandidateOrigin,
    MaterialCandidate,
    MaterialRecord,
    TestConditions,
)
from domains.uav_materials.fixtures import load_synthetic_materials
from domains.uav_materials.properties import PROPERTIES, Category
from domains.uav_materials.units import UnitError, compatible, convert
from tests.domains.uav_materials.helpers import SYN, meas

MATS = {m.name: m for m in load_synthetic_materials()}


def test_units_convert_within_a_dimension_only() -> None:
    assert convert(570, "MPa", "GPa") == pytest.approx(0.57)
    assert convert(23, "degC", "K") == pytest.approx(296.15)
    assert convert(1.6, "g/cm^3", "kg/m^3") == pytest.approx(1600)
    assert convert(50, "%", "1") == pytest.approx(0.5)
    assert not compatible("MPa", "kg/m^3")
    with pytest.raises(UnitError):
        convert(1, "MPa", "kg/m^3")
    with pytest.raises(UnitError):
        convert(1, "furlongs", "m/s")


def test_valid_measurement_keeps_physical_meaning() -> None:
    m = meas(
        "tensile_strength",
        570,
        "MPa",
        test_method="synthetic method",
        conditions={"temperature": {"value": 23, "unit": "degC"}},
    )
    assert m.category is Category.STRENGTH and m.value_in("GPa") == pytest.approx(0.57)
    assert m.provenance == SYN and m.conditions.temperature is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": 570, "unit": "kg/m^3"},  # stress in density units
        {"value": 570, "unit": "bananas"},  # unknown unit
        {"value": 570},  # value without unit
        {"value": 570, "unit": "MPa", "missing": "unknown"},  # both value and missing
        {},  # neither
    ],
)
def test_invalid_measurements_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((ValidationError, UnitError)):
        meas("tensile_strength", **kwargs)  # type: ignore[arg-type]


def test_unknown_property_rejected() -> None:
    with pytest.raises(ValidationError):
        meas("awesomeness", 1, "1")


def test_missing_is_explicit_never_zero() -> None:
    for reason in ("unknown", "not_measured", "not_applicable"):
        m = meas("glass_transition_temperature", missing=reason)
        assert m.is_missing and m.value is None and m.value_in("K") is None
    sparse = MATS["SyntheticCompositeF"]
    assert sparse.status("youngs_modulus") == "not_measured"
    assert sparse.status("corrosion_rate") == "unknown"  # not recorded at all
    assert MATS["SyntheticMetalA"].status("water_absorption") == "not_applicable"
    assert sparse.status("density") == "measured"


def test_provenance_state_is_explicit() -> None:
    unsourced = meas("density", 1600, "kg/m^3", provenance_status="unsourced", provenance=None)
    assert unsourced.provenance is None
    with pytest.raises(ValidationError):
        meas("density", 1600, "kg/m^3", provenance_status="sourced", provenance=None)
    with pytest.raises(ValidationError):
        meas("density", 1600, "kg/m^3", provenance_status="unsourced")  # has a source
    assert MATS["SyntheticCompositeF"].provenance_status == "unsourced"


def test_unreported_conditions_are_not_fabricated() -> None:
    m = meas("tensile_strength", 900, "MPa")
    assert m.conditions.reported() == set() and m.conditions.temperature is None
    with pytest.raises((ValidationError, UnitError)):
        TestConditions(temperature={"value": 20, "unit": "MPa"})  # type: ignore[arg-type]


def test_strength_measurements_stay_distinct() -> None:
    metal = MATS["SyntheticMetalA"]
    names = [m.property for m in metal.strength]
    assert {"tensile_strength", "yield_strength", "compressive_strength", "shear_strength"} <= set(
        names
    )
    assert all(PROPERTIES[n].category is Category.STRENGTH for n in names)
    assert (
        metal.measurements("yield_strength")[0].value
        != metal.measurements("tensile_strength")[0].value
    )


@pytest.mark.parametrize(
    "prop,value,unit", [("youngs_modulus", 70, "GPa"), ("elongation_at_break", 5, "%")]
)
def test_deformation_cannot_be_stored_as_strength(prop: str, value: float, unit: str) -> None:
    with pytest.raises(ValidationError):
        MaterialRecord(
            material_id="x",
            name="SyntheticX",
            family="other",
            provenance_status="synthetic",
            provenance=SYN,
            strength=(meas(prop, value, unit),),
        )
    assert PROPERTIES["youngs_modulus"].category is Category.DEFORMATION


def test_environment_dependent_measurements_keep_their_conditions() -> None:
    corrosion = MATS["SyntheticMetalB"].measurements("corrosion_rate")
    media = {m.conditions.medium: m.value for m in corrosion}
    assert len(media) == 2 and all(m.comparable_conditions for m in corrosion)
    assert all(m.conditions.exposure_duration is not None for m in corrosion)
    no_env = meas("corrosion_rate", 0.1, "mm/year", conditions={"medium": "somewhere"})
    assert not no_env.comparable_conditions  # stored, but not comparable without a class
    wa = MATS["SyntheticPolymerD"].measurements("water_absorption")[0]
    assert wa.conditions.exposure_duration and wa.conditions.exposure_duration.to("h") == 24


def test_temperature_dependent_behaviour() -> None:
    metal = MATS["SyntheticMetalA"]
    at_300 = [
        m
        for m in metal.measurements("tensile_strength")
        if m.conditions.temperature and math.isclose(m.conditions.temperature.to("degC"), 300)
    ]
    assert at_300 and at_300[0].value == 720
    retention = metal.measurements("tensile_strength_retention")[0]
    assert retention.category is Category.TEMPERATURE and retention.conditions.temperature
    polymer = MATS["SyntheticPolymerD"]
    assert polymer.measurements("glass_transition_temperature")[0].value_in("K") == pytest.approx(
        333.15
    )
    assert metal.status("glass_transition_temperature") == "not_applicable"
    with pytest.raises(ValidationError):  # a temperature-effect ratio is not a strength
        MaterialRecord(
            material_id="y",
            name="SyntheticY",
            family="metal",
            provenance_status="synthetic",
            provenance=SYN,
            strength=(meas("tensile_strength_retention", 0.8, "1"),),
        )


def test_existing_vs_proposed_candidates() -> None:
    record = MATS["SyntheticMetalB"]
    existing = MaterialCandidate(candidate_id="c1", origin="existing", material=record)
    assert existing.origin is CandidateOrigin.EXISTING
    with pytest.raises(ValidationError):
        MaterialCandidate(
            candidate_id="c2", origin="existing", material=record, parents=("syn-metal-a",)
        )
    predicted = MaterialRecord(
        material_id="virt-1",
        name="SyntheticVirtual1",
        family="composite",
        provenance_status="synthetic",
        provenance=SYN,
        density=(meas("density", 1400, "kg/m^3", basis="predicted"),),
    )
    with pytest.raises(ValidationError):
        MaterialCandidate(candidate_id="c3", origin="existing", material=predicted)
    proposed = MaterialCandidate(
        candidate_id="c4",
        origin="proposed",
        material=predicted,
        design={"note": "future optimizer output"},
    )
    assert proposed.material.density[0].basis == "predicted"
    with pytest.raises(ValidationError):
        MaterialCandidate(
            candidate_id="c5", origin="composite", material=predicted, parents=("syn-metal-a",)
        )
    composite = MaterialCandidate(
        candidate_id="c6",
        origin="composite",
        material=predicted,
        parents=("syn-metal-a", "syn-polymer-d"),
    )
    assert composite.parents


def test_fixtures_are_clearly_synthetic() -> None:
    assert len(MATS) == 7
    for m in MATS.values():
        assert m.name.startswith("Synthetic") and m.metadata["synthetic"] is True
        for x in m.all_measurements():
            assert x.provenance_status in ("synthetic", "unsourced")
            assert x.test_method is None or "synthetic" in x.test_method
