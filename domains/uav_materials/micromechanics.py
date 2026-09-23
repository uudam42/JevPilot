"""Continuous-fibre polymer-matrix composite: lamina micromechanics (forward model v1).

    z = (fiber_id, matrix_id, V_f)  →  f(z) = unidirectional ply properties  →  x̂

Scope: one unidirectional ply (lamina), in the ply material axes (1 = fibre
direction, 2 = in-plane transverse), void-free, dry, room temperature. No
laminate, stacking sequence or orientation (future: classical lamination
theory).

Equations (Chamis' simplified micromechanics; k_f = V_f, k_m = 1 − V_f):

=========================  ==========================================  =====================
property                   equation                                    source
=========================  ==========================================  =====================
density                    ρ = k_f ρ_f + k_m ρ_m                       TM-83320 fig. 5;
                                                                       TP-3290 app. D
E11 (ply_longitudinal_…)   E11 = k_f E_f11 + k_m E_m                   TM-83320 fig. 6
E22 (ply_transverse_…)     E22 = E_m / (1 − √k_f (1 − E_m/E_f22))      TM-83320 fig. 6
G12 (ply_inplane_shear_…)  G12 = G_m / (1 − √k_f (1 − G_m/G_f12))      TM-83320 fig. 6;
                                                                       TP-3290 app. D
ν12 (ply_major_poisson_…)  ν12 = k_f ν_f12 + k_m ν_m                   TM-83320 fig. 6
S11T (ply_longitudinal_    S11T = S_fT (k_f + k_m E_m / E_f11)         TP-3290 app. D p.84
  tensile_strength)                                                    ("same in all routines")
ε1T (ply_longitudinal_     ε1T = S11T / E11                            TM-83320 sec. 2
  tensile_failure_strain)                                              (linear elastic to
                                                                       fracture)
=========================  ==========================================  =====================

E22 and G12 use Chamis' form, not Halpin–Tsai: it has no fitted
reinforcement parameter (ξ), so no constant is introduced without a source.
G_m = E_m / (2(1 + ν_m)) for the isotropic matrix (TM-83320 sec. 4.0).

Checked against the sources' own worked numbers (tests): TM-83320 examples 4.1,
4.2, 7.2, 8.2 and the ICAN (TP-2515) sample output. That verifies the
implementation, not the physics: TM-83320 states "No comparisons were provided
between predicted values and available measured data". Validity is therefore
recorded per property, and no prediction claims more than its equation.

Not predicted (left missing, never inferred from constituent identity):
corrosion rate, water absorption, maximum service temperature, isotropic
Young's modulus / tensile / yield strength, metallic elongation at break.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from domains.uav_materials.constituents import (
    Constituent,
    ConstituentProperty,
    constituent_library,
)
from domains.uav_materials.design import (
    BoundedDesignSpace,
    CandidateDesign,
    DesignInvalid,
    DesignVariable,
    ModelIdentity,
    ModelValidation,
    PredictedPropertyProfile,
    PropertyPredictor,
    VariableKind,
    candidate_from_prediction,
)
from domains.uav_materials.profile import OperatingEnvironment
from domains.uav_materials.schema import (
    CandidateOrigin,
    MaterialCandidate,
    MaterialFamily,
    Measurement,
    MeasurementBasis,
    ProvenanceStatus,
    TemperatureRegime,
    TestConditions,
)
from jevpilot import Provenance, SourceRef, Uncertainty, stable_digest

TM83320 = "Chamis, C. C.: Simplified Composite Micromechanics Equations for Hygral, Thermal and Mechanical Properties. NASA TM-83320, 1983 (NTRS 19830011546)"  # noqa: E501
TP3290 = "Murthy, P. L. N.; Ginty, C. A.; Sanfeliz, J. G.: Second Generation Integrated Composite Analyzer (ICAN) Computer Code. NASA TP-3290, 1993 (NTRS 19930008950)"  # noqa: E501
TP2515 = "Murthy, P. L. N.; Chamis, C. C.: Integrated Composite Analyzer (ICAN) Users and Programmers Manual. NASA TP-2515, 1986 (NTRS 19860012143)"  # noqa: E501

# Engineering range of V_f. Not a validated applicability domain: it is the span the
# sources' own worked examples exercise (TM-83320 examples k_f = 0.50 ... 0.65; ICAN
# sample 0.55). Physically, V_f < π/4 (contiguous fibres in a square array).
VF_RANGE = (0.50, 0.65)
VF_RANGE_BASIS = (
    "modeling assumption: the span of fibre volume ratios in the sources' worked examples "
    "(NASA TM-83320: 0.50-0.65; ICAN TP-2515 sample: 0.55); not a validated domain"
)
VF_PHYSICAL_MAX = math.pi / 4  # contiguous fibres, square array

ASSUMPTIONS = (
    "unidirectional continuous fibres; one ply; properties in ply material axes (1 = fibre)",
    "void-free: k_m = 1 - k_f (the sources' void correction is not used)",
    "ply and constituents linearly elastic to fracture (TM-83320 section 2)",
    "perfect fibre-matrix bond; transversely isotropic ply",
    "constituent properties: dry, room temperature, indicative NASA data-bank values",
)
EQUATIONS = {
    "density": "rho = kf*rho_f + km*rho_m",
    "ply_longitudinal_modulus": "E11 = kf*Ef11 + km*Em",
    "ply_transverse_modulus": "E22 = Em / (1 - sqrt(kf)*(1 - Em/Ef22))",
    "ply_inplane_shear_modulus": "G12 = Gm / (1 - sqrt(kf)*(1 - Gm/Gf12))",
    "ply_major_poisson_ratio": "nu12 = kf*nuf12 + km*num",
    "ply_longitudinal_tensile_strength": "S11T = SfT*(kf + km*Em/Ef11)",
    "ply_longitudinal_tensile_failure_strain": "eps1T = S11T / E11",
}


@dataclass(frozen=True)
class PropertyNote:
    source: str
    direction: str
    validity: str  # what this equation is good for, per property
    limitations: str


NOTES: dict[str, PropertyNote] = {
    "density": PropertyNote(
        "NASA TM-83320 fig. 5; NASA TP-3290 app. D",
        "scalar (no direction)",
        "exact volume-average mixture rule (mass conservation) for a void-free ply",
        "voids lower the real density; resin-rich regions shift it",
    ),
    "ply_longitudinal_modulus": PropertyNote(
        "NASA TM-83320 fig. 6; NASA TP-3290 app. D",
        "1 (fibre axis) only; not an isotropic Young's modulus",
        "rule of mixtures (iso-strain); well established for fibre-direction stiffness",
        "fibre misalignment and waviness lower it; tension/compression not distinguished",
    ),
    "ply_transverse_modulus": PropertyNote(
        "NASA TM-83320 fig. 6",
        "2 (in-plane, transverse to fibres)",
        "semi-empirical (Chamis); matrix-dominated, more uncertain than E11",
        "sensitive to matrix properties, moisture, temperature and fibre packing",
    ),
    "ply_inplane_shear_modulus": PropertyNote(
        "NASA TM-83320 fig. 6; NASA TP-3290 app. D",
        "1-2 in-plane shear",
        "semi-empirical (Chamis); matrix-dominated",
        "shear response of real plies is non-linear; initial modulus only",
    ),
    "ply_major_poisson_ratio": PropertyNote(
        "NASA TM-83320 fig. 6",
        "12 (strain in 2 from load in 1)",
        "rule of mixtures",
        "",
    ),
    "ply_longitudinal_tensile_strength": PropertyNote(
        "NASA TP-3290 app. D p.84",
        "1 (fibre axis), tension only",
        "first-order estimate: fibre-breakage controlled, iso-strain",
        "ignores fibre strength scatter and size effects, defects, misalignment and "
        "fibre-matrix interface failure; typically optimistic; not a design allowable",
    ),
    "ply_longitudinal_tensile_failure_strain": PropertyNote(
        "NASA TM-83320 section 2 (linear elastic to fracture)",
        "1 (fibre axis), tension only",
        "consequence of the two estimates above; brittle, no plastic strain. With these "
        "equations it reduces exactly to the fibre failure strain SfT/Ef11, independent of V_f",
        "not metallic elongation at break; inherits the strength estimate's limitations",
    ),
}
UNSUPPORTED = (
    "corrosion_rate",
    "water_absorption",
    "max_service_temperature",
    "youngs_modulus",  # isotropic; a unidirectional ply has none
    "tensile_strength",  # isotropic
    "yield_strength",
    "elongation_at_break",  # metallic; see ply_longitudinal_tensile_failure_strain
)

MODEL = ModelIdentity(
    name="continuous-fiber-micromechanics",
    version="1",
    kind="analytical",
    assumptions=ASSUMPTIONS,
    engineering_valid=True,
    validation=ModelValidation(
        status="implementation_checked; not validated against measured ply data",
        dataset="worked examples of NASA TM-83320 (4.1, 4.2, 7.2, 8.2) and the ICAN "
        "sample output of NASA TP-2515 (AS/IMLS)",
        applicable_domain="unidirectional continuous-fibre polymer-matrix ply, void-free, "
        f"dry, room temperature, fibre volume ratio {VF_RANGE[0]}-{VF_RANGE[1]}",
        uncertainty="not quantified",
        version="1",
    ),
    equations=EQUATIONS,
    sources=(TM83320, TP3290, TP2515),
    applicable_domain=f"unidirectional ply; V_f in {VF_RANGE} ({VF_RANGE_BASIS})",
    unsupported=UNSUPPORTED,
)


# -- the equations (pure functions, constituent values in consistent units) -----------------


def density(kf: float, rho_f: float, rho_m: float, km: float | None = None) -> float:
    km = 1 - kf if km is None else km
    return kf * rho_f + km * rho_m


def longitudinal_modulus(kf: float, ef11: float, em: float, km: float | None = None) -> float:
    km = 1 - kf if km is None else km
    return kf * ef11 + km * em


def transverse_modulus(kf: float, ef22: float, em: float) -> float:
    return em / (1 - math.sqrt(kf) * (1 - em / ef22))


def inplane_shear_modulus(kf: float, gf12: float, gm: float) -> float:
    return gm / (1 - math.sqrt(kf) * (1 - gm / gf12))


def major_poisson_ratio(kf: float, nuf12: float, num: float, km: float | None = None) -> float:
    km = 1 - kf if km is None else km
    return kf * nuf12 + km * num


def longitudinal_tensile_strength(
    kf: float, sft: float, em: float, ef11: float, km: float | None = None
) -> float:
    km = 1 - kf if km is None else km
    return sft * (kf + km * em / ef11)


# -- design space -----------------------------------------------------------------------------


class ContinuousFiberCompositeDesignSpace(BoundedDesignSpace):
    """(fiber_id, matrix_id, V_f) over the sourced constituent library."""

    def __init__(self, library: Mapping[str, Constituent] | None = None) -> None:
        self.library = dict(library or constituent_library())
        fibers = tuple(sorted(k for k, c in self.library.items() if c.role == "fiber"))
        matrices = tuple(sorted(k for k, c in self.library.items() if c.role == "matrix"))
        super().__init__(
            space_id="continuous_fiber_ud_ply",
            version="1",
            family=MaterialFamily.COMPOSITE,
            variables=(
                DesignVariable(name="fiber_id", kind=VariableKind.CATEGORICAL, choices=fibers),
                DesignVariable(name="matrix_id", kind=VariableKind.CATEGORICAL, choices=matrices),
                DesignVariable(
                    name="fiber_volume_fraction",
                    kind=VariableKind.CONTINUOUS,
                    lower=VF_RANGE[0],
                    upper=VF_RANGE[1],
                    description=VF_RANGE_BASIS,
                ),
            ),
            origin=CandidateOrigin.COMPOSITE,
        )

    def validate_design(self, design: CandidateDesign) -> list[str]:
        out = super().validate_design(design)
        vf = design.variables.get("fiber_volume_fraction")
        if isinstance(vf, int | float) and not 0 < vf < VF_PHYSICAL_MAX:
            out.append(f"fiber_volume_fraction={vf} outside (0, pi/4): not a physical ply")
        return out

    def candidate_from_design(
        self,
        design: CandidateDesign,
        predictor: PropertyPredictor | None = None,
        environment: OperatingEnvironment | None = None,
    ) -> MaterialCandidate:
        problems = self.validate_design(design)
        if problems:
            raise DesignInvalid("; ".join(problems))
        model = predictor or ContinuousFiberMicromechanicsPredictor(self.library)
        parents = tuple(f"constituent:{design.variables[k]}" for k in ("fiber_id", "matrix_id"))
        return candidate_from_prediction(
            design, model.predict(design, environment), self.family, self.origin, parents
        )


# -- predictor ----------------------------------------------------------------------------


class ContinuousFiberMicromechanicsPredictor:
    """f(z) for a unidirectional ply. Deterministic; no fitted constants; no LLM."""

    model = MODEL

    def __init__(self, library: Mapping[str, Constituent] | None = None) -> None:
        self.library = dict(library or constituent_library())

    def predict(
        self, design: CandidateDesign, environment: OperatingEnvironment | None = None
    ) -> PredictedPropertyProfile:
        if design.design_space != "continuous_fiber_ud_ply":
            raise DesignInvalid(f"{MODEL.label} covers continuous_fiber_ud_ply only")
        fiber = self.library[str(design.variables["fiber_id"])]
        matrix = self.library[str(design.variables["matrix_id"])]
        if fiber.role != "fiber" or matrix.role != "matrix":
            raise DesignInvalid("fiber_id must name a fibre and matrix_id a matrix")
        kf = float(design.variables["fiber_volume_fraction"])

        f, m = fiber.properties, matrix.properties
        rho = density(kf, f["density"].value_in("kg/m^3"), m["density"].value_in("kg/m^3"))
        ef11, ef22 = f["axial_modulus"].value_in("GPa"), f["transverse_modulus"].value_in("GPa")
        gf12, em = f["inplane_shear_modulus"].value_in("GPa"), m["modulus"].value_in("GPa")
        gm = m["shear_modulus"].value_in("GPa")
        e11 = longitudinal_modulus(kf, ef11, em)
        s11t = longitudinal_tensile_strength(kf, f["tensile_strength"].value_in("MPa"), em, ef11)
        values: dict[str, tuple[float, str, list[ConstituentProperty]]] = {
            "density": (rho, "kg/m^3", [f["density"], m["density"]]),
            "ply_longitudinal_modulus": (e11, "GPa", [f["axial_modulus"], m["modulus"]]),
            "ply_transverse_modulus": (
                transverse_modulus(kf, ef22, em),
                "GPa",
                [f["transverse_modulus"], m["modulus"]],
            ),
            "ply_inplane_shear_modulus": (
                inplane_shear_modulus(kf, gf12, gm),
                "GPa",
                [f["inplane_shear_modulus"], m["shear_modulus"]],
            ),
            "ply_major_poisson_ratio": (
                major_poisson_ratio(kf, f["major_poisson_ratio"].value, m["poisson_ratio"].value),
                "1",
                [f["major_poisson_ratio"], m["poisson_ratio"]],
            ),
            "ply_longitudinal_tensile_strength": (
                s11t,
                "MPa",
                [f["tensile_strength"], m["modulus"], f["axial_modulus"]],
            ),
            "ply_longitudinal_tensile_failure_strain": (
                s11t / (e11 * 1000) * 100,  # MPa / MPa → %
                "%",
                [f["tensile_strength"], f["axial_modulus"], m["modulus"]],
            ),
        }
        predictions = []
        for prop, (value, unit, inputs) in values.items():
            note = NOTES[prop]
            predictions.append(
                Measurement(
                    property=prop,
                    value=value,
                    unit=unit,
                    basis=MeasurementBasis.PREDICTED,
                    conditions=TestConditions(
                        temperature_regime=TemperatureRegime.ROOM,
                        specimen="unidirectional ply (lamina), void-free",
                        other={
                            "direction": note.direction,
                            "moisture": "dry",
                            "fiber_volume_fraction": kf,
                        },
                    ),
                    test_method=None,
                    provenance=Provenance(
                        id="prov_" + stable_digest([MODEL.label, design.design_id, prop])[:16],
                        sources=(
                            SourceRef(
                                kind="model",
                                identifier=MODEL.label,
                                version=MODEL.version,
                                metadata={
                                    "equation": EQUATIONS[prop],
                                    "equation_source": note.source,
                                    "validity": note.validity,
                                    "limitations": note.limitations,
                                    "assumptions": list(ASSUMPTIONS),
                                    "design_id": design.design_id,
                                },
                            ),
                            *(
                                SourceRef(
                                    kind="constituent",
                                    identifier=f"{c.constituent_id}:{p.symbol}",
                                    metadata={
                                        "value": p.value,
                                        "unit": p.unit,
                                        "evidence_type": p.evidence_type.value,
                                        "data_quality": p.data_quality,
                                        "provenance_id": p.provenance.id,
                                        "documents": sorted(
                                            {s.identifier for s in p.provenance.sources}
                                        ),
                                    },
                                )
                                for c, p in _owners(inputs, fiber, matrix)
                            ),
                        ),
                        metadata={"model_kind": MODEL.kind},
                    ),
                    provenance_status=ProvenanceStatus.SOURCED,
                    uncertainty=Uncertainty.unknown(
                        "not quantified: indicative constituent values; model error unknown"
                    ),
                    notes=f"PREDICTED ({MODEL.label}); {note.validity}",
                )
            )
        return PredictedPropertyProfile(
            design_id=design.design_id,
            model=MODEL,
            predictions=tuple(predictions),
            not_predicted=UNSUPPORTED,
        )


def _owners(
    inputs: list[ConstituentProperty], fiber: Constituent, matrix: Constituent
) -> list[tuple[Constituent, ConstituentProperty]]:
    out = []
    for p in inputs:
        owner = fiber if any(p is q for q in fiber.properties.values()) else matrix
        out.append((owner, p))
    return out
