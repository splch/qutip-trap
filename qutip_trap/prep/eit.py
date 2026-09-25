"""EIT cooling (PLAN.md Section 4.2.3; Morigi, Eschner and Keitel 2000; Morigi 2003; Roos 2000; Lechner 2016; M3).

Closed forms in the plan's sign, Delta = omega_drive - omega_transition, so cooling needs Delta > 0 (blue) and the
tuning rule reads Omega_r^2 = 4 nu (nu + Delta) (Section 13 "EIT detuning sign": Morigi's Delta is the negative of
the plan's, her optimum 4 nu (nu - Delta_Morigi)). With Delta_r = Delta_g = Delta the bare rate coefficients are

    A_+- = (Omega_g^2/gamma) gamma^2 nu^2 / { gamma^2 nu^2 + 4 [Omega_r^2/4 - nu (nu -+ Delta)]^2 },

the upper sign (heating) pairing with nu - Delta and the lower (cooling) with nu + Delta, the Fock rate equation carrying
eta^2 OUTSIDE with eta the two-photon Lamb-Dicke parameter (the projected difference eta_1 cos phi_1 - eta_2 cos phi_2 of
the legs, zero in a Doppler-free geometry; Morigi 2003), the steady state

    <n>_S = A_+/(A_- - A_+) = [gamma^2 nu^2 + 4 (Omega_r^2/4 - nu (nu + Delta))^2] / [4 Delta nu (Omega_r^2 - 4 nu^2)],

equal to (gamma/4 Delta)^2 exactly at delta = nu, with poles at Delta = 0 and Omega_r = 2 nu where cooling vanishes
(``CoolingError``), and the light shift delta = (sqrt(Delta^2 + Omega_r^2) - |Delta|)/2 of the ground-like dressed
state, so that the calibration direction is Delta and the target nu (Omega_r = 2 sqrt(nu (nu + Delta))). The RMP's
(Gamma/4 Delta_r)^2 is approximate: at its own tuning Omega_r^2 = 4 nu Delta the exact value is (Gamma^2 + 4 nu^2)/(16 Delta (Delta - nu)).
Two Rabi frequencies compose as the quadrature sum Omega_r = sqrt(Omega_1^2 + Omega_2^2) in the closed form (Section 13).
No carrier recoil term appears because W(Delta) = 0 at the dark resonance (Morigi 2003: P0 L2 P0 = 0), a property of
the scheme and not a convention; the bandwidth advantage is the dressed linewidth Gamma' ~ Gamma (2 delta/Omega_r)^2 ~ Gamma nu/Delta.
Level C is the Zeeman-resolved master equation of the actual manifolds through ``BlochModel`` (``lambda_level_c_model``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from qutip_trap.dynamics.multilevel import ModeSpec, MultiLevelOptions
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import BlochModel, CoolingError, RateCoefficients
from qutip_trap.light.recoil import RecoilMode
from qutip_trap.prep.validity import assert_weak_drive
from qutip_trap.species.raman import AtomicStructure


def from_morigi_sign(delta_morigi_rad_s: float) -> float:
    """Delta = -Delta_Morigi: Morigi's detuning convention is the negative of the plan's (Section 13 "EIT detuning sign")."""
    return -delta_morigi_rad_s


def composed_coupling_rad_s(omegas: Sequence[float]) -> float:
    """Omega_r = sqrt(sum Omega_j^2): how several coupling Rabi frequencies enter the closed form (Section 4.2.8, Morigi fixture)."""
    return math.sqrt(sum(float(o) ** 2 for o in omegas))


def light_shift_rad_s(delta_rad_s: float, omega_r_rad_s: float) -> float:
    """delta = (sqrt(Delta^2 + Omega_r^2) - |Delta|)/2 ~ Omega_r^2/(4 Delta): the shift of the narrow, ground-like dressed state
    (Morigi 2000 Eq. 1; the broad one moves by -Delta - delta)."""
    return 0.5 * (math.sqrt(delta_rad_s**2 + omega_r_rad_s**2) - abs(delta_rad_s))


def coupling_for_target_rad_s(nu_rad_s: float, delta_rad_s: float) -> float:
    """Omega_r = 2 sqrt(nu (nu + Delta)): the coupling that puts the light shift exactly on the mode, delta = nu (Section 4.2.3)."""
    if delta_rad_s <= 0.0:
        raise CoolingError(
            "EIT cooling needs a blue-detuned coupling beam, Delta > 0 in the plan's sign (Section 4.2.3)"
        )
    return 2.0 * math.sqrt(nu_rad_s * (nu_rad_s + delta_rad_s))


def dressed_linewidth_rad_s(gamma_rad_s: float, delta_rad_s: float, omega_r_rad_s: float) -> float:
    """Gamma' ~ Gamma (2 delta/Omega_r)^2 ~ Gamma nu/Delta: the narrow dressed state's width, the cooling bandwidth."""
    return gamma_rad_s * (2.0 * light_shift_rad_s(delta_rad_s, omega_r_rad_s) / omega_r_rad_s) ** 2


def two_photon_lamb_dicke(eta_1: float, cos_1: float, eta_2: float, cos_2: float) -> float:
    """eta = eta_1 cos phi_1 - eta_2 cos phi_2: each leg's full-wavevector eta projected separately on the mode (Morigi 2003)."""
    return eta_1 * cos_1 - eta_2 * cos_2


def eit_rate_coefficients(
    omega_g_rad_s: float,
    omega_r_rad_s: float,
    nu_rad_s: float,
    delta_rad_s: float,
    gamma_rad_s: float,
    *,
    allow_saturation: bool = False,
) -> RateCoefficients:
    """The bare A_+- of Section 4.2.3 in s^-1 (eta^2 outside), carrier weight zero (no scattering at the dark resonance).

    The perturbative form needs a weak probe, Omega_g << gamma (Section 4.2.8 vii; ``EitClosedForm.weak_probe``), and
    refuses a saturated one unless ``allow_saturation`` - which is how the plan's own Morigi fixture, Omega_g = 17 MHz
    against gamma = 20 MHz, is evaluated (the recorded M3a finding on that fixture).
    """
    if nu_rad_s <= 0.0 or gamma_rad_s <= 0.0:
        raise ValueError("nu and gamma are positive")
    assert_weak_drive(
        omega_g_rad_s,
        gamma_rad_s,
        allow_saturation=allow_saturation,
        what="the EIT bare A_+- (Omega_g << gamma)",
    )

    def a(sign: float) -> float:
        bracket = omega_r_rad_s**2 / 4.0 - nu_rad_s * (nu_rad_s - sign * delta_rad_s)
        return (
            (omega_g_rad_s**2 / gamma_rad_s)
            * gamma_rad_s**2
            * nu_rad_s**2
            / (gamma_rad_s**2 * nu_rad_s**2 + 4.0 * bracket**2)
        )

    return RateCoefficients(a(+1.0), a(-1.0), 0.0)


def eit_steady_state_nbar(
    omega_r_rad_s: float, nu_rad_s: float, delta_rad_s: float, gamma_rad_s: float
) -> float:
    """<n>_S = [gamma^2 nu^2 + 4 (Omega_r^2/4 - nu (nu + Delta))^2]/[4 Delta nu (Omega_r^2 - 4 nu^2)], Omega_g-independent; raises at the
    poles Delta = 0 and Omega_r = 2 nu and whenever A_- <= A_+ (Delta < 0, or Omega_r < 2 nu with Delta > 0)."""
    denominator = 4.0 * delta_rad_s * nu_rad_s * (omega_r_rad_s**2 - 4.0 * nu_rad_s**2)
    if denominator <= 0.0:
        raise CoolingError(
            "EIT closed form: A_- <= A_+ (stability needs Delta (Omega_r^2 - 4 nu^2) > 0; the poles Delta = 0 and Omega_r = 2 nu "
            "are where cooling vanishes, Section 4.2.3)"
        )
    numerator = (
        gamma_rad_s**2 * nu_rad_s**2
        + 4.0 * (omega_r_rad_s**2 / 4.0 - nu_rad_s * (nu_rad_s + delta_rad_s)) ** 2
    )
    return numerator / denominator


def eit_nbar_at_tuning(gamma_rad_s: float, delta_rad_s: float) -> float:
    """(gamma/4 Delta)^2: the steady state exactly at delta = nu (Section 4.2.3)."""
    return (gamma_rad_s / (4.0 * delta_rad_s)) ** 2


def eit_nbar_at_rmp_tuning(gamma_rad_s: float, nu_rad_s: float, delta_rad_s: float) -> float:
    """(Gamma^2 + 4 nu^2)/(16 Delta (Delta - nu)): the exact value at the RMP's approximate tuning Omega_r^2 = 4 nu Delta
    (0.00361702 at gamma = 1, nu = 0.3, Delta = 5; the (Delta + nu) form gives 0.00320755, Section 4.2.3)."""
    return (gamma_rad_s**2 + 4.0 * nu_rad_s**2) / (16.0 * delta_rad_s * (delta_rad_s - nu_rad_s))


def eit_cooling_rate_per_s(
    eta: float,
    omega_g_rad_s: float,
    omega_r_rad_s: float,
    nu_rad_s: float,
    delta_rad_s: float,
    gamma_rad_s: float,
    *,
    allow_saturation: bool = False,
) -> float:
    """W = eta^2 (A_- - A_+): the exponential relaxation rate of <n> (Lechner's 19e3 s^-1 is this quantity)."""
    rc = eit_rate_coefficients(
        omega_g_rad_s,
        omega_r_rad_s,
        nu_rad_s,
        delta_rad_s,
        gamma_rad_s,
        allow_saturation=allow_saturation,
    )
    return rc.cooling_rate_per_s(eta)


@dataclass(frozen=True)
class EitClosedForm:
    """The Section 4.2.3 description of one mode: coefficients, steady state, light shift and bandwidth."""

    nu_rad_s: float
    delta_rad_s: float
    omega_g_rad_s: float
    omega_r_rad_s: float
    gamma_rad_s: float
    eta: float

    @property
    def coefficients(self) -> RateCoefficients:
        """The bare A_+-; a fixture outside the weak-probe regime (``weak_probe`` False) is evaluated anyway, since the
        record already reports the violation."""
        return eit_rate_coefficients(
            self.omega_g_rad_s,
            self.omega_r_rad_s,
            self.nu_rad_s,
            self.delta_rad_s,
            self.gamma_rad_s,
            allow_saturation=not self.weak_probe(),
        )

    @property
    def nbar(self) -> float:
        return eit_steady_state_nbar(self.omega_r_rad_s, self.nu_rad_s, self.delta_rad_s, self.gamma_rad_s)

    @property
    def cooling_rate_per_s(self) -> float:
        return self.coefficients.cooling_rate_per_s(self.eta)

    @property
    def light_shift_rad_s(self) -> float:
        return light_shift_rad_s(self.delta_rad_s, self.omega_r_rad_s)

    @property
    def bandwidth_rad_s(self) -> float:
        return dressed_linewidth_rad_s(self.gamma_rad_s, self.delta_rad_s, self.omega_r_rad_s)

    def weak_probe(self, ratio: float = 0.1) -> bool:
        """Omega_g << Omega_r, the regime the closed form assumes (Section 4.2.3; the plan's Morigi fixture violates it)."""
        return self.omega_g_rad_s < ratio * self.omega_r_rad_s


def lambda_level_c_model(
    structure: AtomicStructure,
    beams: Sequence[Beam],
    mode: ModeSpec,
    *,
    states: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    recoil: RecoilMode = "minimal",
    leak: str = "renormalize",
) -> BlochModel:
    """Level C for EIT: the Zeeman-resolved manifolds of the real ion (S1/2 and P1/2 for 40Ca+, the D3/2 branch renormalized as an
    instantaneous repump unless the repumper is among the beams) with one mode and the recoil kernel (Section 4.2.3 [corrected])."""
    return BlochModel(
        structure,
        beams,
        levels=levels,
        states=states,
        mode=mode,
        options=MultiLevelOptions(leak=leak, recoil=recoil),  # type: ignore[arg-type]
    )
