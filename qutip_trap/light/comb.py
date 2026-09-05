"""Frequency-comb (mode-locked) Raman drives (PLAN.md Section 4.3.7; Appendix E, Run 5 additions; milestone M2).

Conventions (Section 13, comb rows): every rf and laser input is an ORDINARY frequency (Hz) and every Hamiltonian
coefficient angular; ``tau_s`` is the sech FIELD-envelope parameter of sech(pi t/(2 tau)), never a FWHM (field FWHM
1.6768 tau, intensity FWHM 1.1222 tau; ``check_critique_v3.py``); a single tooth counted from the carrier carries
sech(2 pi k nu_rep tau), a tooth PAIR indexed by separation l carries sech(pi l nu_rep tau) (0.863911 against the
wrong 0.595332 at l = 105, 120 MHz, 14 ps), and the comb Stark sum sech^2(pi l nu_rep tau); ``carrier_order`` is a
signed tooth-index difference; ``aom_offset_hz`` is signed and its sign selects red versus blue; the intensity sum
rule sum_k g_k^2 = g_0^2 is asymptotic in nu_rep tau, so ``tooth_amplitudes`` renormalizes to it; the fourth-order
shift is the single sum over the separation l with the guard omega_a != l omega_rep and NEVER j != 0, summed to
|l - j| >~ 1000 and validated by doubling (the |k| ~ 5 to 10 plateau is a factor 2.093 low); a comb drive is a SET of
tones sharing one operator sigma_+ (x) prod D_m (Section 5.2), the near-resonant beat notes explicit and the rest folded
into the static fourth-order shift, never both (Section 4.3.7).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Tone
    from qutip_trap.species.model import Level

M2 = "milestone M2 (light/comb.py, PLAN.md Section 4.3.7)"


def sech(x: float | np.ndarray) -> np.ndarray | float:
    return 1.0 / np.cosh(np.clip(np.abs(x), 0.0, 700.0))


FIELD_FWHM_OVER_TAU = (4.0 / math.pi) * math.acosh(2.0)
"""1.6768: the field FWHM of sech(pi t/(2 tau))."""
INTENSITY_FWHM_OVER_TAU = (4.0 / math.pi) * math.acosh(math.sqrt(2.0))
"""1.1222: the sech^2 intensity FWHM."""
ZETA3_COEFFICIENT = 0.852556797635
"""7 zeta(3)/pi^2 = int_0^inf sech^2 v tanh v / v dv, the single-comb fourth-order closed-form coefficient (printed 0.853)."""


def tau_field_from(value_s: float, convention: str) -> float:
    """The FIELD sech parameter from a value quoted in another convention: intensity sech (tau_int = 2 tau_field) or a FWHM."""
    if convention == "field_sech":
        return value_s
    if convention == "intensity_sech":
        return value_s / 2.0
    if convention == "intensity_sech2":
        return value_s / 2.0
    if convention == "field_fwhm":
        return value_s / FIELD_FWHM_OVER_TAU
    if convention == "intensity_fwhm":
        return value_s / INTENSITY_FWHM_OVER_TAU
    raise ValueError(f"unknown tau convention {convention!r}")


@dataclass(frozen=True)
class CombSpec:
    """The mode-locked train that generates a Drive's tone set."""

    rep_rate_hz: float
    """nu_rep, ordinary Hz; the tooth spacing."""
    tau_s: float
    """sech ENVELOPE parameter in the convention ``tau_convention`` names, never a FWHM."""
    tau_convention: Literal["field_sech", "intensity_sech", "intensity_sech2"]
    carrier_order: int
    """j = round(omega_HF / omega_rep), signed."""
    aom_offset_hz: float
    """Delta_nu_M = nu_1 - nu_2, SIGNED; its sign selects red vs blue."""
    pair_order_max: int = 1200
    chain: Literal["I", "II"] = "I"
    """Omega = g^2/(2 Delta) (I) versus g^2/Delta (II)."""
    i_sat_w_m2: float | None = None
    """None = derive pi h c Gamma_partial/(3 lambda^3); a value here is the lumped D1 convention of chain II."""
    rep_rate_trajectory: Callable[[float], float] | None = None
    locked_tooth: int | None = None

    def __post_init__(self) -> None:
        if self.rep_rate_hz <= 0.0 or self.tau_s <= 0.0:
            raise ValueError("rep_rate_hz and tau_s must be positive")
        if self.pair_order_max < 1000:
            raise ValueError(
                "the C_{n,a} Stark sum needs pair_order_max >~ 1000 (Section 4.3.7); partial sums plateau falsely"
            )
        # chain and i_sat travel together: chain I with the derived pi h c Gamma/(3 lambda^3), chain II with the
        # source's lumped value; the cross-product counts the D1 line strength twice (Appendix E).
        if self.chain == "I" and self.i_sat_w_m2 is not None:
            raise ValueError("chain I derives I_sat from the transition; do not pass the lumped i_sat_w_m2")
        if self.chain == "II" and self.i_sat_w_m2 is None:
            raise ValueError("chain II needs the source's lumped i_sat_w_m2")

    # ---- envelope and teeth ------------------------------------------------------------------------------------------

    @property
    def tau_field_s(self) -> float:
        return tau_field_from(self.tau_s, self.tau_convention)

    @property
    def omega_rep_rad_s(self) -> float:
        return 2.0 * math.pi * self.rep_rate_hz

    def tooth_weight(self, k: int) -> float:
        """g_k/g_0 (unnormalized): sqrt(pi nu_rep tau) sech(2 pi k nu_rep tau) for a SINGLE tooth k from the comb centre."""
        return math.sqrt(math.pi * self.rep_rate_hz * self.tau_field_s) * float(
            sech(2.0 * math.pi * k * self.rep_rate_hz * self.tau_field_s)
        )

    def pair_weight(self, l: int) -> float:  # noqa: E741  (l is the tooth-separation index of Section 4.3.7)
        """sech(pi l nu_rep tau) for a tooth PAIR at separation l; NOT sech(2 pi l ...) (Section 13)."""
        return 1.0 / math.cosh(math.pi * l * self.rep_rate_hz * self.tau_field_s)

    def beat_note_hz(self, order: int) -> float:
        """|order * nu_rep + aom_offset_hz|."""
        return abs(order * self.rep_rate_hz + self.aom_offset_hz)

    def tooth_amplitudes(self, n_teeth: int) -> np.ndarray:
        """g_k/g_0 for |k| <= n_teeth, RENORMALIZED to the sum rule sum_k g_k^2 = g_0^2 (which the prefactor only
        satisfies asymptotically in nu_rep tau: 1.000000000000 at 1.68e-3, 3.14 at 1)."""
        if n_teeth < 1:
            raise ValueError("n_teeth >= 1")
        k = np.arange(-n_teeth, n_teeth + 1)
        g = np.sqrt(math.pi * self.rep_rate_hz * self.tau_field_s) * np.asarray(
            sech(2.0 * math.pi * k * self.rep_rate_hz * self.tau_field_s)
        )
        norm = math.sqrt(float(np.sum(g**2)))
        return np.asarray(g / norm)

    def sum_rule_raw(self, n_teeth: int = 400_000) -> float:
        """sum_k g_k^2/g_0^2 with the analytic prefactor and no renormalization: the asymptotic sum rule's deviation from 1."""
        k = np.arange(-n_teeth, n_teeth + 1)
        g = np.sqrt(math.pi * self.rep_rate_hz * self.tau_field_s) * np.asarray(
            sech(2.0 * math.pi * k * self.rep_rate_hz * self.tau_field_s)
        )
        return float(np.sum(g**2))

    def pair_sum_exact(self, l: int, n_teeth: int = 400_000) -> float:  # noqa: E741
        """sum_k g_k g_{k+l}/g_0^2 by explicit convolution (the closed form sech(pi l nu_rep tau) is itself ~5% high)."""
        k = np.arange(-n_teeth, n_teeth + 1)
        g = np.sqrt(math.pi * self.rep_rate_hz * self.tau_field_s) * np.asarray(
            sech(2.0 * math.pi * k * self.rep_rate_hz * self.tau_field_s)
        )
        if l == 0:
            return float(np.sum(g**2))
        return float(np.sum(g[: len(g) - l] * g[l:]))

    # ---- resonance and lock arithmetic -----------------------------------------------------------------------------

    def resonance_index(self, omega_q_hz: float) -> float:
        """q = nu_q/nu_rep: integer for a single comb driving the qubit, half-integer under a 1-in-2 pulse picker (Hayes)."""
        return omega_q_hz / self.rep_rate_hz

    def solve_offset(self, target_hz: float) -> tuple[int, float]:
        """(j, Delta_nu_M): the signed tooth order and AOM offset with |j nu_rep + Delta_nu_M| = target and the smallest |offset|.

        The same call builds the feed-forward chain, so substituting nu_rep -> nu_rep + delta_r(t) makes the drift
        cancellation emergent rather than assumed (Section 4.3.7).
        """
        if target_hz <= 0.0:
            raise ValueError("the target is a positive ordinary frequency")
        best: tuple[int, float] | None = None
        j0 = int(round(target_hz / self.rep_rate_hz))
        for j in range(j0 - 2, j0 + 3):
            for sign in (+1.0, -1.0):
                off = sign * target_hz - j * self.rep_rate_hz
                if best is None or abs(off) < abs(best[1]):
                    best = (j, off)
        assert best is not None
        return best

    def tones(
        self,
        omega_q_hz: float,
        mode_hz: float | None,
        *,
        omega0_hz: float = 1.0,
        gate_time_s: float | None = None,
        phase_rad: float = 0.0,
        sideband: int = 0,
    ) -> tuple[Tone, ...]:
        """The comb's tone set: one Tone per beat note j nu_rep + Delta_nu_M with envelope Omega_0 sech(pi j nu_rep tau).

        Detunings are measured from the carrier (Section 13): mu_j = (j nu_rep + Delta_nu_M) - nu_q, sideband-shifted by
        ``sideband`` x mode_hz for the resonance target. Near-resonant beat notes stay explicit: |mu_j - mu_res| <~ 10/t_g
        (about 100 kHz for a 100 us gate; every tone when gate_time_s is None), the rest fold into ``stark4``.
        """
        from qutip_trap.control.pulses import Tone

        target = omega_q_hz + (sideband * mode_hz if mode_hz is not None and sideband else 0.0)
        window = math.inf if gate_time_s is None else 10.0 / gate_time_s
        j_res, _ = self.solve_offset(target)
        out: list[Tone] = []
        for j in range(j_res - self.pair_order_max, j_res + self.pair_order_max + 1):
            if j == 0:
                continue  # a tooth does not beat with itself: the l = 0 term is the intensity (the Stark shift), not a drive
            mu = (j * self.rep_rate_hz + self.aom_offset_hz) - omega_q_hz
            mu_res = (j_res * self.rep_rate_hz + self.aom_offset_hz) - omega_q_hz
            if abs(mu - mu_res) > window:
                continue
            out.append(
                Tone(
                    detuning_hz=float(mu),
                    phase_rad=float(phase_rad),
                    envelope_hz=float(omega0_hz * self.pair_weight(j)),
                )
            )
        return tuple(out)

    # ---- fourth-order Stark shift -----------------------------------------------------------------------------------

    def comb_factor(self, omega_a_hz: float, *, l_max: int | None = None) -> float:
        """C_{n,a}-type single sum over the tooth separation l: sum_l sech^2(pi l nu_rep tau) delta/(omega_a - l omega_rep)
        normalized so that its k = 0 term at the nearest beat note reproduces Lee's C (0.746343 alone, 0.943789 converged
        at 120 MHz, 14 ps, 12.642821 GHz); the guard is omega_a != l omega_rep for every l, never j != 0."""
        lmax = self.pair_order_max if l_max is None else l_max
        j = int(round(omega_a_hz / self.rep_rate_hz))
        delta = omega_a_hz - j * self.rep_rate_hz
        if delta == 0.0:
            raise ZeroDivisionError(
                "omega_a sits exactly on a beat note: the fourth-order sum is singular (guard delta != 0)"
            )
        ls = np.arange(j - lmax, j + lmax + 1)
        num = np.asarray(sech(math.pi * ls * self.rep_rate_hz * self.tau_field_s)) ** 2
        den = omega_a_hz - ls * self.rep_rate_hz
        return float(delta * np.sum(num / den))

    def stark4_hz(
        self,
        levels: Sequence[Level] | Sequence[float],
        *,
        couplings_hz: Sequence[float] | None = None,
        n_index: int = 0,
    ) -> np.ndarray:
        """E^{(4)}_n/2pi = sum_{a != n} (Omega_{n,a}^2/4) sum_l sech^2(pi l nu_rep tau)/(omega_a - l omega_rep) (Section 4.3.7).

        ``levels``: the manifold's sublevel frequencies in Hz (or Level objects, read through ``energy_hz``) relative to
        any origin; ``couplings_hz``: Omega_{n,a}/2pi per sublevel (the polarization-weighted Omega_0, carrying NO sech
        factor); returns the shift of level ``n_index`` and, doubling l_max, asserts stability to 1e-6 relative.
        """
        freqs = np.array([float(lv.energy_hz) if hasattr(lv, "energy_hz") else float(lv) for lv in levels])
        if couplings_hz is None:
            raise ValueError("stark4_hz needs the per-sublevel couplings Omega_{n,a}/2pi")
        omegas = np.asarray(couplings_hz, dtype=float)
        if omegas.shape != freqs.shape:
            raise ValueError("one coupling per sublevel")

        def total(lmax: int) -> float:
            s = 0.0
            for a in range(len(freqs)):
                if a == n_index or omegas[a] == 0.0:
                    continue
                w_a = freqs[a] - freqs[n_index]
                j = int(round(w_a / self.rep_rate_hz))
                ls = np.arange(j - lmax, j + lmax + 1)
                den = w_a - ls * self.rep_rate_hz
                if np.any(den == 0.0):
                    raise ZeroDivisionError(
                        "a sublevel splitting sits exactly on a beat note (guard omega_a != l omega_rep)"
                    )
                num = np.asarray(sech(math.pi * ls * self.rep_rate_hz * self.tau_field_s)) ** 2
                s += (omegas[a] ** 2 / 4.0) * float(np.sum(num / den))
            return s

        val = total(self.pair_order_max)
        check = total(2 * self.pair_order_max)
        if abs(check - val) > 1e-6 * max(abs(val), 1e-300):
            raise ValueError(
                f"the fourth-order sum has not converged at pair_order_max = {self.pair_order_max}: {val} vs {check}"
            )
        return np.array([val])

    # ---- guards ------------------------------------------------------------------------------------------------------

    def guards(
        self, omega_q_hz: float, eta: float, nbar: float, t_train_s: float, mode_hz: float
    ) -> dict[str, bool]:
        """The validity hierarchy of Section 4.3.7 as named booleans the engine logs before every solve."""
        tau = self.tau_field_s
        wq = 2.0 * math.pi * omega_q_hz
        n_pulses = t_train_s * self.rep_rate_hz
        return {
            "teeth_resolved": n_pulses > 10.0,
            "bandwidth_straddles_qubit": wq * tau < 1.0,
            "adiabatic_elimination": True,  # |Delta| >> 1/tau is the caller's fine-structure check (needs the species)
            "small_pulse_area": True,  # theta per pulse << 1 needs Omega_0; asserted where the intensity chain is known
            "sum_rule": self.rep_rate_hz * tau < 0.1,
            "sideband_resolved": 2.0 * math.pi * mode_hz * t_train_s > 10.0,
            "lamb_dicke": eta * math.sqrt(nbar + 1.0) < 1.0,
        }


def rosen_zener_ceiling(nu_hf_hz: float, pulse_duration_s: float) -> float:
    """sech^2(pi nu_HF T): the single-pulse transfer ceiling (Campbell Eq. 6; Mizrahi Eq. 18) with nu_HF ORDINARY."""
    return float(sech(math.pi * nu_hf_hz * pulse_duration_s)) ** 2


def pulse_count_map(f_rep_hz: float, t_s: float, theta_per_pulse_rad: float) -> tuple[float, float]:
    """(N, Omega_0) = (f_rep t, f_rep theta) with f_rep ordinary; the printed 2 pi omega_rep forms are wrong by (2 pi)^2."""
    return f_rep_hz * t_s, f_rep_hz * theta_per_pulse_rad


def harmonic_gain(drift_rep_hz: float, tooth_index: int) -> float:
    """delta nu_sb = n delta nu_rep: the beat note is a tooth-index difference (157 x 1 Hz/min at n = 157)."""
    return tooth_index * drift_rep_hz


def lock_residual_factor(tooth: int, locked_tooth: int, sideband: Literal["lower", "upper", "bare"]) -> int:
    """Per-tooth residual coefficient under the feed-forward lock: (m - n) lower, (m + n) upper, m bare (Section 4.3.7)."""
    if sideband == "lower":
        return tooth - locked_tooth
    if sideband == "upper":
        return tooth + locked_tooth
    return tooth


def amplitude_noise_error(alpha_per_hz: float, omega0_hz: float, t_s: float) -> float:
    """epsilon = (pi^2/2) alpha Omega_0^2 T with Omega_0 an ORDINARY frequency in Hz (rad/s would be (2 pi)^2 too large)."""
    return 0.5 * math.pi**2 * alpha_per_hz * omega0_hz**2 * t_s


__all__ = [
    "FIELD_FWHM_OVER_TAU",
    "INTENSITY_FWHM_OVER_TAU",
    "ZETA3_COEFFICIENT",
    "CombSpec",
    "amplitude_noise_error",
    "harmonic_gain",
    "lock_residual_factor",
    "pulse_count_map",
    "rosen_zener_ceiling",
    "sech",
    "tau_field_from",
]
