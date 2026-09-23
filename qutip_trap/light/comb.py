"""Frequency-comb (mode-locked) Raman drives.

Rf and laser inputs are ordinary frequencies (Hz), Hamiltonian coefficients angular; ``tau_field_s`` is tau of the field
envelope sech(pi t/(2 tau)), never a FWHM. Near-resonant beat notes stay explicit tones and the rest fold into the static
fourth-order shift, never both.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Tone
    from qutip_trap.species.model import Level

M2 = "milestone M2 (light/comb.py, PLAN.md Section 4.3.7)"


def sech(x: float | np.ndarray) -> np.ndarray | float:
    """sech|x|, with |x| clamped at 700 so cosh cannot overflow to inf (sech(700) = 9.9e-305 is negligible)."""
    return 1.0 / np.cosh(np.clip(np.abs(x), 0.0, 700.0))


FIELD_FWHM_OVER_TAU = (4.0 / math.pi) * math.acosh(2.0)
"""1.6768: the field FWHM of sech(pi t/(2 tau))."""
INTENSITY_FWHM_OVER_TAU = (4.0 / math.pi) * math.acosh(math.sqrt(2.0))
"""1.1222: the sech^2 intensity FWHM."""
ZETA3_COEFFICIENT = 0.852556797635
"""7 zeta(3)/pi^2 = int_0^inf sech^2 v tanh v / v dv, the single-comb fourth-order closed-form coefficient."""
SUM_DEPTH_HALF_WIDTHS = 8.0
"""Half-widths 1/(pi nu_rep tau) of the sech^2 Stark weight the l-sum spans (eight pass the 1e-6 doubling check)."""
SINGULAR_REL_TOL = 1e-9
"""|delta_{n,a}|/nu_rep below which a splitting sits on a beat note and the l-sum is refused."""
MARGINAL_COUPLING_FACTOR = 1.0
"""|delta_{n,a}|/|Omega_{n,a}| below which the expansion is marginal at full power: warned, never raised."""


def _assert_converged(value: float, doubled: float, lmax: int, what: str, rtol: float = 1e-6) -> None:
    """The doubling check: partial sums plateau falsely (a factor 2.093 low at |k| ~ 5 to 10), so compare at 2 x l_max."""
    if abs(doubled - value) > rtol * max(abs(value), 1e-300):
        raise ValueError(
            f"{what} has not converged at l_max = {lmax}: {value} vs {doubled} at 2 l_max (raise pair_order_max, or "
            "leave it None so the envelope sets the depth; Section 4.3.7)"
        )


TauConvention = Literal["field_sech", "intensity_sech", "intensity_sech2", "field_fwhm", "intensity_fwhm"]
"""Every convention ``tau_field_from`` converts."""


def tau_field_from(value_s: float, convention: str) -> float:
    """The field sech parameter tau of sech(pi t/(2 tau)) from a value in another convention (both intensity spellings
    give value/2)."""
    if convention == "field_sech":
        return value_s
    if convention in ("intensity_sech", "intensity_sech2"):
        return value_s / 2.0
    if convention == "field_fwhm":
        return value_s / FIELD_FWHM_OVER_TAU
    if convention == "intensity_fwhm":
        return value_s / INTENSITY_FWHM_OVER_TAU
    raise ValueError(f"unknown tau convention {convention!r}; known: {TauConvention.__args__}")  # type: ignore[attr-defined]


@dataclass(frozen=True)
class CombSpec:
    """The mode-locked train that generates a Drive's tone set."""

    rep_rate_hz: float
    tau_s: float
    """Pulse-width parameter in the convention ``tau_convention`` names."""
    tau_convention: TauConvention
    carrier_order: int
    """j = round(omega_HF / omega_rep), signed."""
    aom_offset_hz: float
    """Delta_nu_M = nu_1 - nu_2, signed; its sign selects red vs blue."""
    pair_order_max: int | None = None
    """|l - j| retained in the fourth-order sum; None derives it from the envelope (no explicit value may be shallower)."""
    chain: Literal["I", "II"] = "I"
    """Omega = g^2/(2 Delta) (I) versus g^2/Delta (II)."""
    i_sat_w_m2: float | None = None
    """None = derive pi h c Gamma_partial/(3 lambda^3); a value here is the lumped D1 convention of chain II."""
    rep_rate_trajectory: Callable[[float], float] | None = None
    locked_tooth: int | None = None

    def __post_init__(self) -> None:
        if self.rep_rate_hz <= 0.0 or self.tau_s <= 0.0:
            raise ValueError("rep_rate_hz and tau_s must be positive")
        if self.pair_order_max is not None and self.pair_order_max < self.derived_sum_depth:
            raise ValueError(
                f"the C_(n,a) Stark sum needs |l - j| >~ {self.derived_sum_depth} for this envelope "
                f"({SUM_DEPTH_HALF_WIDTHS:g} x the sech^2 half-width 1/(pi nu_rep tau) = {self.sum_half_width:.1f}); "
                f"pair_order_max = {self.pair_order_max} plateaus falsely (Section 4.3.7)"
            )
        # chain I takes the derived I_sat, chain II the lumped one; mixing them counts the D1 line strength twice
        if self.chain == "I" and self.i_sat_w_m2 is not None:
            raise ValueError("chain I derives I_sat from the transition; do not pass the lumped i_sat_w_m2")
        if self.chain == "II" and self.i_sat_w_m2 is None:
            raise ValueError("chain II needs the source's lumped i_sat_w_m2")

    @property
    def tau_field_s(self) -> float:
        return tau_field_from(self.tau_s, self.tau_convention)

    @property
    def omega_rep_rad_s(self) -> float:
        return 2.0 * math.pi * self.rep_rate_hz

    @property
    def sum_half_width(self) -> float:
        """1/(pi nu_rep tau): the half-width, in tooth separation l, of the sech^2 fourth-order weight."""
        return 1.0 / (math.pi * self.rep_rate_hz * self.tau_field_s)

    @property
    def derived_sum_depth(self) -> int:
        """ceil(SUM_DEPTH_HALF_WIDTHS/(pi nu_rep tau)): the depth the envelope itself demands."""
        return int(math.ceil(SUM_DEPTH_HALF_WIDTHS * self.sum_half_width))

    @property
    def sum_depth(self) -> int:
        """The l range actually summed: the explicit ``pair_order_max`` or ``derived_sum_depth``."""
        return self.derived_sum_depth if self.pair_order_max is None else self.pair_order_max

    def tooth_weights(self, k: np.ndarray | float | int) -> np.ndarray:
        """g_k/g_0 (unnormalized) = sqrt(pi nu_rep tau) sech(2 pi k nu_rep tau) for single teeth k from the comb centre."""
        x = np.asarray(k, dtype=float)
        return np.asarray(
            math.sqrt(math.pi * self.rep_rate_hz * self.tau_field_s)
            * np.asarray(sech(2.0 * math.pi * x * self.rep_rate_hz * self.tau_field_s))
        )

    def tooth_weight(self, k: int) -> float:
        """g_k/g_0 (unnormalized) for one tooth (``tooth_weights`` at a scalar)."""
        return float(self.tooth_weights(k))

    def pair_weight(self, l: int) -> float:  # noqa: E741  (l is the tooth-separation index)
        """sech(pi l nu_rep tau) for a tooth pair at separation l (not sech(2 pi l ...))."""
        return 1.0 / math.cosh(math.pi * l * self.rep_rate_hz * self.tau_field_s)

    def beat_note_hz(self, order: int) -> float:
        """|order * nu_rep + aom_offset_hz|."""
        return abs(order * self.rep_rate_hz + self.aom_offset_hz)

    def tooth_amplitudes(self, n_teeth: int) -> np.ndarray:
        """g_k/g_0 for |k| <= n_teeth, renormalized to the sum rule sum_k g_k^2 = g_0^2 (asymptotic in nu_rep tau)."""
        if n_teeth < 1:
            raise ValueError("n_teeth >= 1")
        g = self.tooth_weights(np.arange(-n_teeth, n_teeth + 1))
        norm = math.sqrt(float(np.sum(g**2)))
        return np.asarray(g / norm)

    def sum_rule_raw(self, n_teeth: int = 400_000) -> float:
        """sum_k g_k^2/g_0^2 with the analytic prefactor and no renormalization: the asymptotic sum rule's deviation from 1."""
        g = self.tooth_weights(np.arange(-n_teeth, n_teeth + 1))
        return float(np.sum(g**2))

    def pair_sum_exact(self, l: int, n_teeth: int = 400_000) -> float:  # noqa: E741
        """sum_k g_k g_{k+l}/g_0^2 by explicit convolution (the closed form sech(pi l nu_rep tau) is itself ~5% high)."""
        g = self.tooth_weights(np.arange(-n_teeth, n_teeth + 1))
        if l == 0:
            return float(np.sum(g**2))
        return float(np.sum(g[: len(g) - l] * g[l:]))

    def resonance_index(self, omega_q_hz: float) -> float:
        """q = nu_q/nu_rep: integer for a single comb driving the qubit, half-integer under a 1-in-2 pulse picker (Hayes)."""
        return omega_q_hz / self.rep_rate_hz

    def solve_offset(self, target_hz: float) -> tuple[int, float]:
        """(j, Delta_nu_M): the signed tooth order and smallest AOM offset with |j nu_rep + Delta_nu_M| = target."""
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

    def resonance_target_hz(self, omega_q_hz: float, mode_hz: float | None, sideband: int = 0) -> float:
        """nu_q + sideband x nu_mode: the beat-note frequency the drive is tuned to."""
        return omega_q_hz + (sideband * mode_hz if mode_hz is not None and sideband else 0.0)

    def explicit_orders(self, resonance_hz: float, gate_time_s: float | None) -> frozenset[int]:
        """The beat-note orders j kept explicit, |mu_j - mu_res| <= 10/t_g (every one within ``sum_depth`` when
        ``gate_time_s`` is None); ``comb_factor`` and ``stark4_hz`` exclude exactly these."""
        j_res, _ = self.solve_offset(resonance_hz)
        if gate_time_s is None:
            return frozenset(j for j in range(j_res - self.sum_depth, j_res + self.sum_depth + 1) if j != 0)
        if gate_time_s <= 0.0:
            raise ValueError("gate_time_s is a positive duration")
        span = int(math.floor((10.0 / gate_time_s) / self.rep_rate_hz))
        return frozenset(j for j in range(j_res - span, j_res + span + 1) if j != 0)

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
        """One Tone per explicit beat note, detuned mu_j = (j nu_rep + Delta_nu_M) - nu_q from the carrier, with envelope
        Omega_0 sech(pi j nu_rep tau)."""
        from qutip_trap.control.pulses import Tone

        target = self.resonance_target_hz(omega_q_hz, mode_hz, sideband)
        return tuple(
            Tone(
                detuning_hz=float((j * self.rep_rate_hz + self.aom_offset_hz) - omega_q_hz),
                phase_rad=float(phase_rad),
                envelope_hz=float(omega0_hz * self.pair_weight(j)),
            )
            for j in sorted(self.explicit_orders(target, gate_time_s))
        )

    def beat_notes_hz(self, orders: np.ndarray) -> np.ndarray:
        """l nu_rep + Delta_nu_M for a vector of orders: the one beat-note grid the tone set and the l-sum share."""
        return np.asarray(orders, dtype=float) * self.rep_rate_hz + self.aom_offset_hz

    def _nearest_order(self, omega_a_hz: float) -> int:
        """argmin_l |omega_a - (l nu_rep + Delta_nu_M)|: the l of the nearest beat note."""
        return int(round((omega_a_hz - self.aom_offset_hz) / self.rep_rate_hz))

    def _residual_detuning_hz(
        self,
        omega_a_hz: float,
        coupling_hz: float,
        what: str,
        excluded: frozenset[int] = frozenset(),
    ) -> float:
        """delta_{n,a} = omega_a - (j nu_rep + Delta_nu_M) at the nearest folded beat note; raises ZeroDivisionError at
        |delta| <= ``SINGULAR_REL_TOL`` nu_rep (never guarding j != 0) and warns at |delta| <= |Omega_{n,a}|."""
        j = self._nearest_folded_order(omega_a_hz, excluded)
        delta = omega_a_hz - float(self.beat_notes_hz(np.array([j]))[0])
        floor = SINGULAR_REL_TOL * self.rep_rate_hz
        if abs(delta) <= floor:
            raise ZeroDivisionError(
                f"{what}: the splitting {omega_a_hz:.6g} Hz sits on beat note l = {j} to within "
                f"{abs(delta):.3g} Hz <= {SINGULAR_REL_TOL:g} nu_rep = {floor:.3g} Hz, so the l-sum is singular "
                "(the guard is delta_(n,a) != 0, never j != 0; Section 4.3.7). A comb locked to this splitting needs "
                "the explicit/folded partition: pass gate_time_s and resonance_hz"
            )
        if coupling_hz and abs(delta) <= MARGINAL_COUPLING_FACTOR * abs(coupling_hz):
            warnings.warn(
                f"{what}: |delta_(n,a)| = {abs(delta):.4g} Hz does not exceed |Omega_(n,a)| = "
                f"{abs(coupling_hz):.4g} Hz, so the fourth-order expansion parameter is order unity and the closed "
                "form is not claimed at this power (Section 4.3.7, marginal at full power; Section 12)",
                RuntimeWarning,
                stacklevel=3,
            )
        return delta

    def _nearest_folded_order(self, omega_a_hz: float, excluded: frozenset[int]) -> int:
        """The nearest beat-note order not held explicit; l = 0 is kept (the operative case for the Zeeman partners)."""
        j0 = self._nearest_order(omega_a_hz)
        for step in range(0, 4 * len(excluded) + 2):
            for j in (j0 + step, j0 - step) if step else (j0,):
                if j not in excluded:
                    return j
        raise ValueError("every nearby beat note is held explicit: nothing folds into the static shift")

    def _l_sum(
        self, omega_a_hz: float, lmax: int, excluded: frozenset[int], *, coupling_hz: float, what: str
    ) -> float:
        """sum_l sech^2(pi l nu_rep tau)/(omega_a - (l nu_rep + Delta_nu_M)) over the folded orders |l - j| <= lmax."""
        self._residual_detuning_hz(omega_a_hz, coupling_hz, what, excluded)
        j = self._nearest_order(omega_a_hz)
        ls = np.arange(j - lmax, j + lmax + 1)
        if excluded:
            ls = ls[~np.isin(ls, np.fromiter(excluded, dtype=int, count=len(excluded)))]
        den = omega_a_hz - self.beat_notes_hz(ls)
        num = np.asarray(sech(math.pi * ls * self.rep_rate_hz * self.tau_field_s)) ** 2
        return float(np.sum(num / den))

    def comb_factor(
        self,
        omega_a_hz: float,
        *,
        l_max: int | None = None,
        gate_time_s: float | None = None,
        resonance_hz: float | None = None,
        coupling_hz: float = 0.0,
        validate: bool = False,
    ) -> float:
        """C_{n,a} = delta sum_l sech^2(pi l nu_rep tau)/(omega_a - l omega_rep - omega_A), normalized so its k = 0 term
        reproduces Lee's C; ``validate`` refuses a change above 1e-6 relative when l_max doubles."""
        lmax = self.sum_depth if l_max is None else l_max
        excluded = frozenset() if resonance_hz is None else self.explicit_orders(resonance_hz, gate_time_s)
        # _l_sum re-guards with coupling_hz = 0, so the marginal-power warning is emitted once per call
        delta = self._residual_detuning_hz(omega_a_hz, coupling_hz, "comb_factor", excluded)
        val = delta * self._l_sum(omega_a_hz, lmax, excluded, coupling_hz=0.0, what="comb_factor")
        if validate:
            check = delta * self._l_sum(omega_a_hz, 2 * lmax, excluded, coupling_hz=0.0, what="comb_factor")
            _assert_converged(val, check, lmax, "comb_factor")
        return val

    def stark4_hz(
        self,
        levels: Sequence[Level] | Sequence[float],
        *,
        couplings_hz: Sequence[float] | None = None,
        n_index: int = 0,
        gate_time_s: float | None = None,
        resonance_hz: float | None = None,
    ) -> np.ndarray:
        """E^{(4)}_n/2pi = sum_{a != n} (Omega_{n,a}^2/4) sum_l sech^2(pi l nu_rep tau)/(omega_a - l omega_rep - omega_A).

        ``levels`` are sublevel frequencies in Hz (or Levels) and ``couplings_hz`` Omega_{n,a}/2pi per sublevel without
        the sech factor; the shift of level ``n_index`` is checked to 1e-6 when l_max doubles.
        """
        freqs = np.array([float(lv.energy_hz) if hasattr(lv, "energy_hz") else float(lv) for lv in levels])
        if couplings_hz is None:
            raise ValueError("stark4_hz needs the per-sublevel couplings Omega_{n,a}/2pi")
        omegas = np.asarray(couplings_hz, dtype=float)
        if omegas.shape != freqs.shape:
            raise ValueError("one coupling per sublevel")
        excluded = frozenset() if resonance_hz is None else self.explicit_orders(resonance_hz, gate_time_s)
        # guard and warn once per sublevel, before the two sums of the doubling check
        for a in range(len(freqs)):
            if a != n_index and omegas[a] != 0.0:
                self._residual_detuning_hz(
                    freqs[a] - freqs[n_index], float(omegas[a]), f"stark4_hz level {a}", excluded
                )

        def total(lmax: int) -> float:
            s = 0.0
            for a in range(len(freqs)):
                if a == n_index or omegas[a] == 0.0:
                    continue
                w_a = freqs[a] - freqs[n_index]
                s += (omegas[a] ** 2 / 4.0) * self._l_sum(
                    w_a, lmax, excluded, coupling_hz=0.0, what=f"stark4_hz level {a}"
                )
            return s

        val = total(self.sum_depth)
        check = total(2 * self.sum_depth)
        _assert_converged(val, check, self.sum_depth, "the fourth-order sum")
        return np.array([val])

    def guards(
        self,
        omega_q_hz: float,
        eta: float,
        nbar: float,
        t_train_s: float,
        mode_hz: float,
        *,
        detuning_hz: float | None = None,
        fine_structure_hz: float | None = None,
        theta_per_pulse_rad: float | None = None,
    ) -> dict[str, bool]:
        """The comb's validity conditions as named booleans; ``adiabatic_elimination`` and ``small_pulse_area`` appear
        only when their inputs are given, so "not evaluated" never reads as "satisfied"."""
        tau = self.tau_field_s
        wq = 2.0 * math.pi * omega_q_hz
        n_pulses = t_train_s * self.rep_rate_hz
        out = {
            "teeth_resolved": n_pulses > 10.0,
            "bandwidth_straddles_qubit": wq * tau < 1.0,
            "sum_rule": self.rep_rate_hz * tau < 0.1,
            "sideband_resolved": 2.0 * math.pi * mode_hz * t_train_s > 10.0,
            "lamb_dicke": eta * math.sqrt(nbar + 1.0) < 1.0,
        }
        if detuning_hz is not None:
            # the bandwidth must sit well inside the detuning, and |Delta| <~ omega_FS or the D1 and D2 amplitudes cancel
            ok = 1.0 / tau < 0.1 * abs(detuning_hz)
            if fine_structure_hz is not None:
                ok = ok and abs(detuning_hz) <= abs(fine_structure_hz)
            out["adiabatic_elimination"] = bool(ok)
        if theta_per_pulse_rad is not None:
            out["small_pulse_area"] = abs(theta_per_pulse_rad) < 0.1
        return out

    def unevaluated_guards(
        self, *, detuning_hz: float | None = None, theta_per_pulse_rad: float | None = None
    ) -> tuple[str, ...]:
        """The validity clauses ``guards()`` cannot evaluate from the inputs given."""
        missing: list[str] = []
        if detuning_hz is None:
            missing.append("adiabatic_elimination (needs the fine-structure detuning Delta)")
        if theta_per_pulse_rad is None:
            missing.append("small_pulse_area (needs the intensity-to-Rabi chain's theta per pulse)")
        return tuple(missing)


def comb_build_notes(
    comb: CombSpec, duration_s: float, n_tones: int, stark_shift_hz: float
) -> tuple[str, ...]:
    """The ``approximations`` notes the Hamiltonian builder records for a mode-locked drive (the clauses the pulse
    itself decides; ``light.raman.comb_drive`` reports the rest)."""
    notes = [
        f"comb drive: {n_tones} explicit beat note(s) at nu_rep = {comb.rep_rate_hz:.6g} Hz, the rest folded into a "
        f"static fourth-order shift of {stark_shift_hz:.6g} Hz (l-sum depth {comb.sum_depth}); never both"
    ]
    n_pulses = duration_s * comb.rep_rate_hz
    if n_pulses <= 10.0:
        notes.append(
            f"comb drive: guard 'teeth_resolved' FAILS, {n_pulses:.3g} pulses in the train (Section 4.3.7)"
        )
    if comb.rep_rate_hz * comb.tau_field_s >= 0.1:
        notes.append(
            f"comb drive: guard 'sum_rule' FAILS, nu_rep tau = {comb.rep_rate_hz * comb.tau_field_s:.3g} "
            "(the tooth-amplitude sum rule is asymptotic in nu_rep tau; Section 4.3.7)"
        )
    return tuple(notes)


def rosen_zener_ceiling(nu_hf_hz: float, pulse_duration_s: float) -> float:
    """sech^2(pi nu_HF T): the single-pulse transfer ceiling with nu_HF ordinary (Campbell Eq. 6)."""
    return float(sech(math.pi * nu_hf_hz * pulse_duration_s)) ** 2


def pulse_count_map(f_rep_hz: float, t_s: float, theta_per_pulse_rad: float) -> tuple[float, float]:
    """(N, Omega_0) = (f_rep t, f_rep theta) with f_rep ordinary."""
    return f_rep_hz * t_s, f_rep_hz * theta_per_pulse_rad


def harmonic_gain(drift_rep_hz: float, tooth_index: int) -> float:
    """delta nu_sb = n delta nu_rep: the beat note is a tooth-index difference."""
    return tooth_index * drift_rep_hz


def lock_residual_factor(tooth: int, locked_tooth: int, sideband: Literal["lower", "upper", "bare"]) -> int:
    """Per-tooth residual coefficient under the feed-forward lock: (m - n) lower, (m + n) upper, m bare."""
    if sideband == "lower":
        return tooth - locked_tooth
    if sideband == "upper":
        return tooth + locked_tooth
    return tooth


def amplitude_noise_error(alpha_per_hz: float, omega0_hz: float, t_s: float) -> float:
    """epsilon = (pi^2/2) alpha Omega_0^2 T with Omega_0 an ordinary frequency in Hz."""
    return 0.5 * math.pi**2 * alpha_per_hz * omega0_hz**2 * t_s


__all__ = [
    "FIELD_FWHM_OVER_TAU",
    "INTENSITY_FWHM_OVER_TAU",
    "MARGINAL_COUPLING_FACTOR",
    "SINGULAR_REL_TOL",
    "SUM_DEPTH_HALF_WIDTHS",
    "ZETA3_COEFFICIENT",
    "CombSpec",
    "amplitude_noise_error",
    "comb_build_notes",
    "harmonic_gain",
    "lock_residual_factor",
    "pulse_count_map",
    "rosen_zener_ceiling",
    "sech",
    "tau_field_from",
]
