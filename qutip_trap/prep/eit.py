"""EIT cooling (Morigi, Eschner and Keitel 2000): the closed forms and the level-C model.

Delta = omega_drive - omega_transition (the negative of Morigi's), so cooling needs Delta > 0 and the tuning rule is
Omega_r^2 = 4 nu (nu + Delta); eta is the two-photon Lamb-Dicke parameter, kept outside the bare coefficients.
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
    """Delta = -Delta_Morigi: Morigi's detuning convention is the negative of this module's."""
    return -delta_morigi_rad_s


def composed_coupling_rad_s(omegas: Sequence[float]) -> float:
    """Omega_r = sqrt(sum Omega_j^2): how several coupling Rabi frequencies enter the closed form."""
    return math.sqrt(sum(float(o) ** 2 for o in omegas))


def light_shift_rad_s(delta_rad_s: float, omega_r_rad_s: float) -> float:
    """delta = (sqrt(Delta^2 + Omega_r^2) - |Delta|)/2 ~ Omega_r^2/(4 Delta): the shift of the narrow, ground-like dressed
    state (Morigi et al. 2000 Eq. 1)."""
    return 0.5 * (math.sqrt(delta_rad_s**2 + omega_r_rad_s**2) - abs(delta_rad_s))


def coupling_for_target_rad_s(nu_rad_s: float, delta_rad_s: float) -> float:
    """Omega_r = 2 sqrt(nu (nu + Delta)): the coupling that puts the light shift on the mode, delta = nu."""
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
    """A_+- = (Omega_g^2/gamma) gamma^2 nu^2 / {gamma^2 nu^2 + 4 [Omega_r^2/4 - nu (nu -+ Delta)]^2} in s^-1, with no
    carrier term (nothing scatters at the dark resonance); needs Omega_g << gamma unless ``allow_saturation``."""
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
    """<n>_S = [gamma^2 nu^2 + 4 (Omega_r^2/4 - nu (nu + Delta))^2]/[4 Delta nu (Omega_r^2 - 4 nu^2)], independent of
    Omega_g; raises CoolingError wherever A_- <= A_+, the poles Delta = 0 and Omega_r = 2 nu included."""
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
    """(gamma/4 Delta)^2: the steady state exactly at delta = nu."""
    return (gamma_rad_s / (4.0 * delta_rad_s)) ** 2


def eit_nbar_at_rmp_tuning(gamma_rad_s: float, nu_rad_s: float, delta_rad_s: float) -> float:
    """(Gamma^2 + 4 nu^2)/(16 Delta (Delta - nu)): the exact value at the approximate tuning Omega_r^2 = 4 nu Delta of
    Leibfried et al. 2003."""
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
    """W = eta^2 (A_- - A_+): the exponential relaxation rate of <n>."""
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
    """The EIT closed forms for one mode: coefficients, steady state, light shift and bandwidth."""

    nu_rad_s: float
    delta_rad_s: float
    omega_g_rad_s: float
    omega_r_rad_s: float
    gamma_rad_s: float
    eta: float

    @property
    def coefficients(self) -> RateCoefficients:
        """The bare A_+-, evaluated even outside the weak-probe regime (``weak_probe`` reports that)."""
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
        """Omega_g << Omega_r, the regime the closed form assumes."""
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
    """Level C for EIT: the ion's Zeeman-resolved manifolds with one mode and the recoil kernel."""
    return BlochModel(
        structure,
        beams,
        levels=levels,
        states=states,
        mode=mode,
        options=MultiLevelOptions(leak=leak, recoil=recoil),  # type: ignore[arg-type]
    )
