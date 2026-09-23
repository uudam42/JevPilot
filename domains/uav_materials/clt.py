"""Classical Lamination Theory (CLT): lamina → symmetric laminate stiffness (model v1).

    constituents ─micromechanics─▶ ply E1, E2, G12, ν12 ─▶ Q ─▶ Q̄(θ) ─▶ stack ─▶ A, B, D
                                                                       └─▶ Ex, Ey, Gxy, νxy

Source: Nettles, A. T.: Basic Mechanics of Laminated Composite Plates.
NASA RP-1351, Marshall Space Flight Center, 1994 (NTRS 19950009349;
"Unclassified-unlimited"). Its documentation page notes: "Strength of
laminated composites is not covered", and this model covers stiffness only.

=====================  =========================================================  ===========
quantity               equation                                                   RP-1351
=====================  =========================================================  ===========
reciprocal Poisson     ν21 = ν12 E2 / E1                                          sec. II
reduced stiffness Q    Q11 = E1/(1−ν12ν21), Q22 = E2/(1−ν12ν21),                  eq. (10)
                       Q12 = ν12 E2/(1−ν12ν21), Q66 = G12
transformed Q̄(θ)       m = cos θ, n = sin θ:                                      eq. (22)
                       Q̄11 = Q11 m⁴ + 2(Q12+2Q66) m²n² + Q22 n⁴
                       Q̄22 = Q11 n⁴ + 2(Q12+2Q66) m²n² + Q22 m⁴
                       Q̄12 = (Q11+Q22−4Q66) m²n² + Q12 (m⁴+n⁴)
                       Q̄66 = (Q11+Q22−2Q12−2Q66) m²n² + Q66 (m⁴+n⁴)
                       Q̄16 = (Q11−Q12−2Q66) m³n + (Q12−Q22+2Q66) m n³
                       Q̄26 = (Q11−Q12−2Q66) m n³ + (Q12−Q22+2Q66) m³n
A, B, D                A = Σ Q̄ (z_k − z_{k−1}), B = ½ Σ Q̄ (z_k² − z_{k−1}²),       eqs. (53),
                       D = ⅓ Σ Q̄ (z_k³ − z_{k−1}³)   (z from −h/2, bottom → top)    (54), (56)
engineering constants  a = A⁻¹ (B = 0): Ex = 1/(h a11), Ey = 1/(h a22),            sec. V.B
                       Gxy = 1/(h a66), νxy = −a12/a11
=====================  =========================================================  ===========

Angles: counter-clockwise from laminate x to the fibre direction, in degrees.
RP-1351 notes the sense is a convention; its printed Q̄16 > 0 at +45° matches.

Scope and assumptions: symmetric laminates only (so B = 0, checked
numerically), identical plies from one fibre/matrix/V_f, equal ply
thickness, perfect bonding, plane stress, linear elasticity. The constants
are *membrane* (in-plane) properties; bending is described by D. No strength
or failure: ply S11T says nothing about laminate strength.

Validation (tests): the Q, Q̄(45°), A and B values printed in RP-1351
examples 2 and 9 (text layer, quote-verified) are reproduced. Ex/Ey/Gxy/νxy
have no machine-readable published example; they are checked by exact
identities ([0]s = ply constants, [90]s = swapped, quasi-isotropic A).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from domains.uav_materials.constituents import Constituent, constituent_library
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
    parse_sequence,
)
from domains.uav_materials.micromechanics import (
    MODEL as PLY_MODEL,
)
from domains.uav_materials.micromechanics import (
    VF_PHYSICAL_MAX,
    VF_RANGE,
    VF_RANGE_BASIS,
    ContinuousFiberMicromechanicsPredictor,
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
from jevpilot import FrozenModel, Provenance, SourceRef, Uncertainty, stable_digest

Matrix = list[list[float]]

RP1351 = (
    "Nettles, A. T.: Basic Mechanics of Laminated Composite Plates. NASA RP-1351, "
    "NASA Marshall Space Flight Center, 1994 (NTRS 19950009349)"
)
SYSTEM = "continuous_fiber_laminate"
PLY_THICKNESS_RANGE_MM = (0.05, 0.30)
PLY_THICKNESS_BASIS = (
    "modeling assumption: a range around the ply thicknesses of the sources' examples "
    "(RP-1351: 0.005 in = 0.127 mm; ICAN TP-2515/TP-3290: 0.005-0.010 in); not validated"
)
MAX_HALF_PLIES = 8

EQUATIONS = {
    "Q": "Q11=E1/(1-nu12*nu21), Q22=E2/(1-nu12*nu21), Q12=nu12*E2/(1-nu12*nu21), Q66=G12; "
    "nu21=nu12*E2/E1  [RP-1351 eq. (10)]",
    "Qbar": "Qbar(theta) by the fourth-order rotation of Q  [RP-1351 eq. (22)]",
    "ABD": "A=sum Qbar*(z_k-z_k-1); B=1/2 sum Qbar*(z_k^2-z_k-1^2); D=1/3 sum Qbar*(z_k^3-z_k-1^3)"
    "  [RP-1351 eqs. (53), (54), (56)]",
    "laminate_ex": "Ex = 1/(h*a11), a = inv(A)  [RP-1351 sec. V.B, symmetric laminate]",
    "laminate_ey": "Ey = 1/(h*a22), a = inv(A)  [RP-1351 sec. V.B]",
    "laminate_gxy": "Gxy = 1/(h*a66), a = inv(A)  [RP-1351 sec. V.B]",
    "laminate_nuxy": "nuxy = -a12/a11, a = inv(A)  [RP-1351 sec. V.B]",
    "density": "rho_laminate = rho_ply (identical void-free plies)",
}
ASSUMPTIONS = (
    "classical lamination theory: plane stress, perfectly bonded plies, linear elasticity",
    "symmetric stacking only (B = 0 checked numerically); in-plane (membrane) constants",
    "all plies identical: one fibre/matrix/V_f, equal ply thickness",
    "ply properties from continuous-fiber-micromechanics/1 (dry, room temperature)",
    "no strength or failure prediction",
)
UNSUPPORTED = (
    "corrosion_rate",
    "water_absorption",
    "max_service_temperature",
    "tensile_strength",
    "yield_strength",
    "elongation_at_break",
    "youngs_modulus",  # isotropic; a laminate has directional Ex, Ey
)

MODEL = ModelIdentity(
    name="continuous-fiber-laminate-clt",
    version="1",
    kind="analytical",
    assumptions=ASSUMPTIONS,
    engineering_valid=True,
    validation=ModelValidation(
        status="Q, Qbar, A, B reproduce NASA RP-1351 examples 2 and 9; engineering constants "
        "checked by exact identities; not validated against measured laminate data",
        dataset="NASA RP-1351 (1994) worked examples, text layer",
        applicable_domain="symmetric laminates of identical unidirectional plies, in-plane "
        "stiffness, dry room temperature",
        uncertainty="not quantified",
        version="1",
    ),
    equations=EQUATIONS,
    sources=(RP1351, *PLY_MODEL.sources),
    applicable_domain="symmetric laminate; in-plane stiffness only",
    unsupported=UNSUPPORTED,
)


# -- equations (pure; any consistent units) ---------------------------------------------------


def reduced_stiffness(
    e1: float, e2: float, g12: float, nu12: float, nu21: float | None = None
) -> Matrix:
    """Q in the ply axes; ν21 = ν12 E2/E1 unless given (RP-1351 prints it rounded)."""
    nu21 = nu12 * e2 / e1 if nu21 is None else nu21
    d = 1 - nu12 * nu21
    q11, q22, q12 = e1 / d, e2 / d, nu12 * e2 / d
    return [[q11, q12, 0.0], [q12, q22, 0.0], [0.0, 0.0, g12]]


def transformed_stiffness(q: Matrix, theta_deg: float) -> Matrix:
    """Q̄(θ) in laminate axes (indices 0,1,2 = 1,2,6)."""
    q11, q12, q22, q66 = q[0][0], q[0][1], q[1][1], q[2][2]
    t = math.radians(theta_deg)
    m, n = math.cos(t), math.sin(t)
    m2, n2 = m * m, n * n
    b11 = q11 * m2 * m2 + 2 * (q12 + 2 * q66) * m2 * n2 + q22 * n2 * n2
    b22 = q11 * n2 * n2 + 2 * (q12 + 2 * q66) * m2 * n2 + q22 * m2 * m2
    b12 = (q11 + q22 - 4 * q66) * m2 * n2 + q12 * (m2 * m2 + n2 * n2)
    b66 = (q11 + q22 - 2 * q12 - 2 * q66) * m2 * n2 + q66 * (m2 * m2 + n2 * n2)
    b16 = (q11 - q12 - 2 * q66) * m2 * m * n + (q12 - q22 + 2 * q66) * m * n2 * n
    b26 = (q11 - q12 - 2 * q66) * m * n2 * n + (q12 - q22 + 2 * q66) * m2 * m * n
    return [[b11, b12, b16], [b12, b22, b26], [b16, b26, b66]]


def abd(plies: Sequence[tuple[Matrix, float]]) -> tuple[Matrix, Matrix, Matrix, float]:
    """A, B, D and total thickness h for plies (Q̄, thickness) listed bottom → top."""
    h = sum(t for _, t in plies)
    a = [[0.0] * 3 for _ in range(3)]
    b = [[0.0] * 3 for _ in range(3)]
    d = [[0.0] * 3 for _ in range(3)]
    z0 = -h / 2
    for qbar, t in plies:
        z1 = z0 + t
        for i in range(3):
            for j in range(3):
                a[i][j] += qbar[i][j] * (z1 - z0)
                b[i][j] += qbar[i][j] * (z1**2 - z0**2) / 2
                d[i][j] += qbar[i][j] * (z1**3 - z0**3) / 3
        z0 = z1
    return a, b, d, h


def inverse3(m: Matrix) -> Matrix:
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if det == 0:
        raise ZeroDivisionError("singular extensional stiffness")
    cof = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]
    return [[x / det for x in row] for row in cof]


def engineering_constants(a: Matrix, h: float) -> dict[str, float]:
    """In-plane constants of a symmetric laminate from its extensional stiffness A."""
    s = inverse3(a)
    return {
        "laminate_ex": 1 / (h * s[0][0]),
        "laminate_ey": 1 / (h * s[1][1]),
        "laminate_gxy": 1 / (h * s[2][2]),
        "laminate_nuxy": -s[0][1] / s[0][0],
    }


def symmetric_stack(half: Sequence[float]) -> list[float]:
    """[θ1/…/θn]s → full stack, top → mid → bottom (mirror); order is symmetric anyway."""
    return [*half, *reversed(half)]


def is_symmetric_b(b: Matrix, a: Matrix, h: float, rel: float = 1e-9) -> bool:
    scale = max(abs(x) for row in a for x in row) * h
    return all(abs(x) <= rel * scale for row in b for x in row)


# -- laminate result ---------------------------------------------------------------------------


class Ply(FrozenModel):
    index: int  # 1 = bottom
    angle_deg: float
    thickness_m: float
    z_bottom_m: float
    z_top_m: float
    qbar_pa: Matrix


class LaminateStiffness(FrozenModel):
    """A, B, D with units, the ply stack, and the ply predictions they came from."""

    layup: str  # e.g. "[0/45/-45/90]s"
    plies: tuple[Ply, ...]  # bottom → top
    thickness_m: float
    ply_thickness_m: float
    q_pa: Matrix  # ply reduced stiffness in ply axes
    a: Matrix
    b: Matrix
    d: Matrix
    units: dict[str, str] = Field(
        default_factory=lambda: {"A": "N/m", "B": "N", "D": "N*m", "Q": "Pa", "thickness": "m"}
    )
    b_is_zero: bool
    constants: dict[str, float]  # Pa for moduli, 1 for nuxy
    ply_predictions: tuple[Measurement, ...]  # E11, E22, G12, nu12 (with their provenance)


def laminate_stiffness(
    ply: Mapping[str, Measurement], half_angles: Sequence[float], ply_thickness_m: float
) -> LaminateStiffness:
    e1 = _pa(ply["ply_longitudinal_modulus"])
    e2 = _pa(ply["ply_transverse_modulus"])
    g12 = _pa(ply["ply_inplane_shear_modulus"])
    nu12 = float(ply["ply_major_poisson_ratio"].value_in("1") or 0)
    q = reduced_stiffness(e1, e2, g12, nu12)
    angles = symmetric_stack(half_angles)
    stack = [(transformed_stiffness(q, th), ply_thickness_m) for th in angles]
    a, b, d, h = abd(stack)
    plies, z = [], -h / 2
    for k, (th, (qbar, t)) in enumerate(zip(angles, stack, strict=True), start=1):
        plies.append(
            Ply(index=k, angle_deg=th, thickness_m=t, z_bottom_m=z, z_top_m=z + t, qbar_pa=qbar)
        )
        z += t
    return LaminateStiffness(
        layup="[" + "/".join(f"{x:g}" for x in half_angles) + "]s",
        plies=tuple(plies),
        thickness_m=h,
        ply_thickness_m=ply_thickness_m,
        q_pa=q,
        a=a,
        b=b,
        d=d,
        b_is_zero=is_symmetric_b(b, a, h),
        constants=engineering_constants(a, h),
        ply_predictions=tuple(ply[k] for k in _PLY_INPUTS),
    )


_PLY_INPUTS = (
    "ply_longitudinal_modulus",
    "ply_transverse_modulus",
    "ply_inplane_shear_modulus",
    "ply_major_poisson_ratio",
)


def _pa(m: Measurement) -> float:
    v = m.value_in("Pa")
    assert v is not None
    return v


# -- design space --------------------------------------------------------------------------------


class ContinuousFiberLaminateDesignSpace(BoundedDesignSpace):
    """(fiber, matrix, V_f, half stacking sequence, ply thickness) → symmetric laminate."""

    def __init__(self, library: Mapping[str, Constituent] | None = None) -> None:
        self.library = dict(library or constituent_library())
        fibers = tuple(sorted(k for k, c in self.library.items() if c.role == "fiber"))
        matrices = tuple(sorted(k for k, c in self.library.items() if c.role == "matrix"))
        super().__init__(
            space_id=SYSTEM,
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
                DesignVariable(
                    name="half_stack_deg",
                    kind=VariableKind.SEQUENCE,
                    lower=-90,
                    upper=90,
                    max_length=MAX_HALF_PLIES,
                    description="ply angles from the outer surface to the midplane; the "
                    "laminate is their mirror-symmetric stack [..]s",
                ),
                DesignVariable(
                    name="ply_thickness_mm",
                    kind=VariableKind.CONTINUOUS,
                    lower=PLY_THICKNESS_RANGE_MM[0],
                    upper=PLY_THICKNESS_RANGE_MM[1],
                    unit="mm",
                    description=PLY_THICKNESS_BASIS + "; all plies have this thickness",
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
        model = predictor or LaminatePredictor(self.library)
        parents = tuple(f"constituent:{design.variables[k]}" for k in ("fiber_id", "matrix_id"))
        candidate = candidate_from_prediction(
            design, model.predict(design, environment), self.family, self.origin, parents
        )
        if isinstance(model, LaminatePredictor):
            lam = model.stiffness(design)
            meta = {**candidate.metadata, "laminate": lam.model_dump(mode="json")}
            candidate = candidate.model_copy(update={"metadata": meta})
        return candidate


# -- predictor -----------------------------------------------------------------------------------


class LaminatePredictor:
    """f(z) for a symmetric laminate: micromechanics ply → CLT. Deterministic; no LLM."""

    model = MODEL

    def __init__(self, library: Mapping[str, Constituent] | None = None) -> None:
        self.library = dict(library or constituent_library())
        self.ply_model = ContinuousFiberMicromechanicsPredictor(self.library)
        self.ply_space_id = "continuous_fiber_ud_ply"

    def ply_design(self, design: CandidateDesign) -> CandidateDesign:
        from domains.uav_materials.micromechanics import ContinuousFiberCompositeDesignSpace

        space = ContinuousFiberCompositeDesignSpace(self.library)
        return space.design(
            {k: design.variables[k] for k in ("fiber_id", "matrix_id", "fiber_volume_fraction")},
            generation_method=f"ply of laminate {design.design_id}",
        )

    def stiffness(self, design: CandidateDesign) -> LaminateStiffness:
        if design.design_space != SYSTEM:
            raise DesignInvalid(f"{MODEL.label} covers {SYSTEM} only")
        half = parse_sequence(str(design.variables["half_stack_deg"]))
        assert half
        ply = {m.property: m for m in self.ply_model.predict(self.ply_design(design)).predictions}
        t = float(design.variables["ply_thickness_mm"]) / 1000
        lam = laminate_stiffness(ply, half, t)
        if not lam.b_is_zero:
            raise DesignInvalid(f"{lam.layup}: B is not zero; only symmetric laminates")
        return lam

    def predict(
        self, design: CandidateDesign, environment: OperatingEnvironment | None = None
    ) -> PredictedPropertyProfile:
        lam = self.stiffness(design)
        ply = {m.property: m for m in lam.ply_predictions}
        density = self.ply_model.predict(self.ply_design(design)).predictions[0]
        assert density.property == "density"
        predictions = [
            self._measurement(design, lam, prop, value, unit, direction, ply)
            for prop, value, unit, direction in (
                ("laminate_ex", lam.constants["laminate_ex"] / 1e9, "GPa", "laminate x"),
                ("laminate_ey", lam.constants["laminate_ey"] / 1e9, "GPa", "laminate y"),
                ("laminate_gxy", lam.constants["laminate_gxy"] / 1e9, "GPa", "laminate x-y"),
                ("laminate_nuxy", lam.constants["laminate_nuxy"], "1", "laminate x-y"),
            )
        ]
        predictions.append(
            self._measurement(
                design, lam, "density", float(density.value or 0), "kg/m^3", "scalar",
                {"density": density},
            )
        )  # fmt: skip
        return PredictedPropertyProfile(
            design_id=design.design_id,
            model=MODEL,
            predictions=tuple(predictions),
            not_predicted=UNSUPPORTED,
        )

    def _measurement(
        self,
        design: CandidateDesign,
        lam: LaminateStiffness,
        prop: str,
        value: float,
        unit: str,
        direction: str,
        inputs: Mapping[str, Measurement],
    ) -> Measurement:
        """One laminate prediction whose provenance keeps the whole chain, level by level."""
        clt = SourceRef(
            kind="model",
            identifier=MODEL.label,
            version=MODEL.version,
            metadata={
                "chain_level": "1 laminate property <- CLT equation",
                "equation": EQUATIONS[prop],
                "equation_source": RP1351,
                "layup": lam.layup,
                "thickness_m": lam.thickness_m,
                "ply_thickness_m": lam.ply_thickness_m,
                "equal_ply_thickness": True,
                "b_is_zero": lam.b_is_zero,
                "abd_units": lam.units,
                "design_id": design.design_id,
            },
        )
        plies = SourceRef(
            kind="ply_stack",
            identifier=f"{lam.layup}:Qbar",
            metadata={
                "chain_level": "2 CLT <- ply Qbar(theta) matrices",
                "equation": EQUATIONS["Qbar"],
                "q_pa": lam.q_pa,
                "plies": [
                    {"index": p.index, "angle_deg": p.angle_deg, "qbar_pa": p.qbar_pa}
                    for p in lam.plies
                ],
            },
        )
        ply_refs = [
            SourceRef(
                kind="ply_prediction",
                identifier=f"{PLY_MODEL.label}:{m.property}",
                metadata={
                    "chain_level": "3 ply Q <- lamina micromechanics prediction",
                    "value": m.value,
                    "unit": m.unit,
                    "provenance_id": m.provenance.id if m.provenance else None,
                    "equation": m.provenance.sources[0].metadata.get("equation")
                    if m.provenance
                    else None,
                },
            )
            for m in inputs.values()
        ]
        constituent_refs = {
            s.identifier: SourceRef(
                kind="constituent",
                identifier=s.identifier,
                metadata={
                    **s.metadata,
                    "chain_level": "4 lamina <- constituent values <- documents",
                },
            )
            for m in inputs.values()
            if m.provenance
            for s in m.provenance.sources
            if s.kind == "constituent"
        }
        return Measurement(
            property=prop,
            value=value,
            unit=unit,
            basis=MeasurementBasis.PREDICTED,
            conditions=TestConditions(
                temperature_regime=TemperatureRegime.ROOM,
                specimen=f"symmetric laminate {lam.layup}, h = {lam.thickness_m * 1000:.4g} mm",
                other={
                    "direction": direction,
                    "moisture": "dry",
                    "layup": lam.layup,
                    "laminate_thickness_mm": lam.thickness_m * 1000,
                    "response": "in-plane (membrane)" if prop != "density" else "scalar",
                },
            ),
            provenance=Provenance(
                id="prov_" + stable_digest([MODEL.label, design.design_id, prop])[:16],
                sources=(clt, plies, *ply_refs, *constituent_refs.values()),
                metadata={"model_kind": MODEL.kind},
            ),
            provenance_status=ProvenanceStatus.SOURCED,
            uncertainty=Uncertainty.unknown(
                "not quantified: indicative constituent values; micromechanics and CLT error"
            ),
            notes=f"PREDICTED ({MODEL.label} on {PLY_MODEL.label} plies)",
        )


def layup_design(
    half: str,
    vf: float = 0.60,
    ply_thickness_mm: float = 0.127,
    fiber: str = "AS--",
    matrix: str = "IMLS",
    generation_method: str = "manual",
) -> CandidateDesign:
    """Convenience: a design in the laminate space (validates it)."""
    values: dict[str, Any] = {
        "fiber_id": fiber,
        "matrix_id": matrix,
        "fiber_volume_fraction": vf,
        "half_stack_deg": half,
        "ply_thickness_mm": ply_thickness_mm,
    }
    return ContinuousFiberLaminateDesignSpace().design(values, generation_method=generation_method)
