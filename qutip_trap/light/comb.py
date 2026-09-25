"""Frequency-comb (mode-locked) Raman drives (PLAN.md Section 4.3.7).

Rates and offsets are ordinary frequencies (Hz). ``tau_s`` is the sech FIELD-envelope parameter of sech(pi t/(2 tau)),
never a FWHM (field FWHM 1.6768 tau, intensity FWHM 1.1222 tau); a tooth PAIR at separation l carries sech(pi l nu_rep tau)
and the fourth-order Stark sum sech^2(pi l nu_rep tau). A comb drive is a SET of tones sharing one operator: the beat
notes within 10/t_g of the resonance stay explicit tones and the rest fold into the static fourth-order shift
E^(4)_n = sum_a (Omega_{n,a}^2/4) sum_l sech^2(pi l nu_rep tau)/(omega_a - (l nu_rep + Delta_nu_M)), never both. The l-sum
is guarded on its detuning (delta != 0, never j != 0: j = 0 is the Zeeman partners' case), runs eight sech^2 half-widths
deep about the nearest beat note and is validated by doubling that depth (the |k| ~ 5 to 10 plateau is a factor 2.093 low).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Tone
    from qutip_trap.species.model import Level


def sech(x: float | np.ndarray) -> np.ndarray | float:
    """sech|x|, with |x| clamped at 700 where cosh overflows (sech(700) = 9.9e-305 is already zero in double precision)."""
    return 1.0 / np.cosh(np.clip(np.abs(x), 0.0, 700.0))


FIELD_FWHM_OVER_TAU = (4.0 / math.pi) * math.acosh(2.0)
"""1.6768: the field FWHM of sech(pi t/(2 tau))."""
INTENSITY_FWHM_OVER_TAU = (4.0 / math.pi) * math.acosh(math.sqrt(2.0))
"""1.1222: the sech^2 intensity FWHM."""
SUM_DEPTH_HALF_WIDTHS = 8.0
"""l-sum depth in units of the sech^2 half-width 1/(pi nu_rep tau): its tail beyond k half-widths is ~4 e^{-2k} of the sum,
and eight is where the 1e-6 doubling check passes (1516 teeth at 120 MHz and 14 ps, 3184 at 80 MHz and 10 ps)."""
SINGULAR_REL_TOL = 1e-9
"""|delta_{n,a}|/nu_rep below which a splitting sits ON a beat note and the l-sum is refused (exact float equality is no
guard: a rep rate that nearly divides the splitting gives an enormous finite shift)."""
MARGINAL_COUPLING_FACTOR = 1.0
"""|delta_{n,a}| below this times |Omega_{n,a}| is the source's "marginal at full power" band: warned, never raised."""


def _assert_converged(value: float, doubled: float, lmax: int, what: str, rtol: float = 1e-6) -> None:
    if abs(doubled - value) > rtol * max(abs(value), 1e-300):
        raise ValueError(
            f"{what} has not converged at l_max = {lmax}: {value} vs {doubled} at 2 l_max (raise pair_order_max, or "
            "leave it None so the envelope sets the depth)"
        )


TauConvention = Literal["field_sech", "intensity_sech", "intensity_sech2", "field_fwhm", "intensity_fwhm"]


def tau_field_from(value_s: float, convention: TauConvention) -> float:
    """The FIELD sech parameter from a value quoted in another convention (the two intensity spellings name one parameter
    of sech^2(pi t/(2 tau)), whose argument is twice the field one)."""
    if convention == "field_sech":
        return value_s
    if convention in ("intensity_sech", "intensity_sech2"):
        return value_s / 2.0
    if convention == "field_fwhm":
        return value_s / FIELD_FWHM_OVER_TAU
    if convention == "intensity_fwhm":
        return value_s / INTENSITY_FWHM_OVER_TAU
    raise ValueError(f"unknown tau convention {convention!r}")


@dataclass(frozen=True)
class CombSpec:
    """The mode-locked train that generates a Drive's tone set.

    ``carrier_order`` is j = round(omega_HF/omega_rep), signed; ``aom_offset_hz`` is Delta_nu_M = nu_1 - nu_2, signed (its
    sign selects red or blue); ``pair_order_max`` overrides the l-sum depth the envelope derives and is validated against
    it; ``chain`` I (Omega = g^2/(2 Delta), I_sat derived) or II (Omega = g^2/Delta with the source's lumped
    ``i_sat_w_m2``).
    """

    rep_rate_hz: float
    tau_s: float
    tau_convention: TauConvention
    carrier_order: int
    aom_offset_hz: float
    pair_order_max: int | None = None
    chain: Literal["I", "II"] = "I"
    i_sat_w_m2: float | None = None

    def __post_init__(self) -> None:
        if self.rep_rate_hz <= 0.0 or self.tau_s <= 0.0:
            raise ValueError("rep_rate_hz and tau_s must be positive")
        if self.pair_order_max is not None and self.pair_order_max < self.derived_sum_depth:
            raise ValueError(
                f"the C_(n,a) Stark sum needs |l - j| >~ {self.derived_sum_depth} for this envelope "
                f"({SUM_DEPTH_HALF_WIDTHS:g} x the sech^2 half-width 1/(pi nu_rep tau) = {self.sum_half_width:.1f}); "
                f"pair_order_max = {self.pair_order_max} plateaus falsely"
            )
        # chain I derives I_sat from the transition, chain II carries the source's lumped value; the cross-product
        # counts the D1 line strength twice
        if self.chain == "I" and self.i_sat_w_m2 is not None:
            raise ValueError("chain I derives I_sat from the transition; do not pass the lumped i_sat_w_m2")
        if self.chain == "II" and self.i_sat_w_m2 is None:
            raise ValueError("chain II needs the source's lumped i_sat_w_m2")

    @property
    def tau_field_s(self) -> float:
        return tau_field_from(self.tau_s, self.tau_convention)

    @property
    def sum_half_width(self) -> float:
        """1/(pi nu_rep tau): the half-width, in tooth separation l, of the sech^2 fourth-order weight."""
        return 1.0 / (math.pi * self.rep_rate_hz * self.tau_field_s)

    @property
    def derived_sum_depth(self) -> int:
        return int(math.ceil(SUM_DEPTH_HALF_WIDTHS * self.sum_half_width))

    @property
    def sum_depth(self) -> int:
        """The l range summed: ``pair_order_max`` or the derived depth."""
        return self.derived_sum_depth if self.pair_order_max is None else self.pair_order_max

    def pair_weight(self, l: int) -> float:  # noqa: E741  (l is the tooth-separation index)
        """sech(pi l nu_rep tau) for a tooth PAIR at separation l (not sech(2 pi l ...), the single-tooth argument)."""
        return 1.0 / math.cosh(math.pi * l * self.rep_rate_hz * self.tau_field_s)

    def solve_offset(self, target_hz: float) -> tuple[int, float]:
        """(j, Delta_nu_M): the signed tooth order and AOM offset with |j nu_rep + Delta_nu_M| = target, smallest |offset|."""
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
        """The beat-note orders j kept as explicit tones, |mu_j - mu_res| <= 10/t_g (every one when ``gate_time_s`` is
        None): ``tones`` emits exactly these and the l-sums exclude exactly these."""
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
        """One Tone per explicit beat note j nu_rep + Delta_nu_M, detuned from the carrier by (j nu_rep + Delta_nu_M) - nu_q,
        with envelope Omega_0 sech(pi j nu_rep tau); j = 0 is skipped (a tooth does not beat with itself)."""
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
        """l nu_rep + Delta_nu_M: the one beat-note grid the tone set and the l-sum share."""
        return np.asarray(orders, dtype=float) * self.rep_rate_hz + self.aom_offset_hz

    def beat_note_hz(self, order: int) -> float:
        """|order nu_rep + Delta_nu_M|, the frequency of one beat note."""
        return abs(order * self.rep_rate_hz + self.aom_offset_hz)

    def _nearest_order(self, omega_a_hz: float) -> int:
        return int(round((omega_a_hz - self.aom_offset_hz) / self.rep_rate_hz))

    def _residual_detuning_hz(
        self, omega_a_hz: float, coupling_hz: float, what: str, excluded: frozenset[int] = frozenset()
    ) -> float:
        """delta_{n,a} = omega_a - (j nu_rep + Delta_nu_M) at the nearest FOLDED beat note (``excluded`` holds the explicit
        tones, which are no term of the sum, so a comb locked to the qubit keeps the far teeth's shift): refused within
        SINGULAR_REL_TOL nu_rep of zero, warned inside the marginal band |delta| <= |Omega_{n,a}|."""
        j = self._nearest_folded_order(omega_a_hz, excluded)
        delta = omega_a_hz - float(self.beat_notes_hz(np.array([j]))[0])
        floor = SINGULAR_REL_TOL * self.rep_rate_hz
        if abs(delta) <= floor:
            raise ZeroDivisionError(
                f"{what}: the splitting {omega_a_hz:.6g} Hz sits on beat note l = {j} to within "
                f"{abs(delta):.3g} Hz <= {SINGULAR_REL_TOL:g} nu_rep = {floor:.3g} Hz, so the l-sum is singular "
                "(the guard is delta_(n,a) != 0, never j != 0). A comb locked to this splitting needs the "
                "explicit/folded partition: pass gate_time_s and resonance_hz"
            )
        if coupling_hz and abs(delta) <= MARGINAL_COUPLING_FACTOR * abs(coupling_hz):
            warnings.warn(
                f"{what}: |delta_(n,a)| = {abs(delta):.4g} Hz does not exceed |Omega_(n,a)| = "
                f"{abs(coupling_hz):.4g} Hz, so the fourth-order expansion parameter is order unity and the closed "
                "form is not claimed at this power (marginal at full power)",
                RuntimeWarning,
                stacklevel=3,
            )
        return delta

    def _nearest_folded_order(self, omega_a_hz: float, excluded: frozenset[int]) -> int:
        j0 = self._nearest_order(omega_a_hz)
        for step in range(0, 4 * len(excluded) + 2):
            for j in (j0 + step, j0 - step) if step else (j0,):
                if j not in excluded:
                    return j
        raise ValueError("every nearby beat note is held explicit: nothing folds into the static shift")

    def _l_sum(
        self, omega_a_hz: float, lmax: int, excluded: frozenset[int], *, coupling_hz: float, what: str
    ) -> float:
        """sum_l sech^2(pi l nu_rep tau)/(omega_a - (l nu_rep + Delta_nu_M)) over |l - j| <= lmax about the nearest beat
        note j, the explicit orders excluded."""
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
        """C_{n,a} = delta sum_l sech^2(pi l nu_rep tau)/(omega_a - l omega_rep - omega_A), normalized so that the l = j term
        alone is Lee's C (0.746343, converging to 0.943789 at 120 MHz, 14 ps, 12.642821 GHz); ``validate`` repeats the sum
        at 2 l_max and refuses a change above 1e-6."""
        lmax = self.sum_depth if l_max is None else l_max
        excluded = frozenset() if resonance_hz is None else self.explicit_orders(resonance_hz, gate_time_s)
        # guard once and normalize by the delta the guard measured (the sums re-check with coupling_hz = 0 so the
        # marginal-band warning is emitted once)
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
        """E^(4)_n/2pi = sum_{a != n} (Omega_{n,a}^2/4) sum_l sech^2(pi l nu_rep tau)/(omega_a - l omega_rep - omega_A).

        ``levels`` are the manifold's sublevel frequencies in Hz (or Level records, read through ``energy_hz``) about any
        origin; ``couplings_hz`` are Omega_{n,a}/2pi per sublevel with no sech factor; ``gate_time_s`` with
        ``resonance_hz`` excludes the explicit tones. Returns the shift of level ``n_index``, checked by doubling l_max.
        """
        from qutip_trap.species.model import Level

        freqs = np.array([float(lv.energy_hz) if isinstance(lv, Level) else float(lv) for lv in levels])
        if couplings_hz is None:
            raise ValueError("stark4_hz needs the per-sublevel couplings Omega_{n,a}/2pi")
        omegas = np.asarray(couplings_hz, dtype=float)
        if omegas.shape != freqs.shape:
            raise ValueError("one coupling per sublevel")
        excluded = frozenset() if resonance_hz is None else self.explicit_orders(resonance_hz, gate_time_s)
        for a in range(len(freqs)):  # guard (and warn about) every contributing sublevel once
            if a != n_index and omegas[a] != 0.0:
                self._residual_detuning_hz(
                    freqs[a] - freqs[n_index], float(omegas[a]), f"stark4_hz level {a}", excluded
                )

        def total(lmax: int) -> float:
            s = 0.0
            for a in range(len(freqs)):
                if a == n_index or omegas[a] == 0.0:
                    continue
                s += (omegas[a] ** 2 / 4.0) * self._l_sum(
                    freqs[a] - freqs[n_index], lmax, excluded, coupling_hz=0.0, what=f"stark4_hz level {a}"
                )
            return s

        val = total(self.sum_depth)
        _assert_converged(val, total(2 * self.sum_depth), self.sum_depth, "the fourth-order sum")
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
        """The pulse-train-to-continuous-wave validity hierarchy as named booleans.

        ``adiabatic_elimination`` (omega_q << 1/tau << |Delta| <~ omega_FS) and ``small_pulse_area`` (theta << 1) appear only
        when their inputs are supplied, so "not evaluated" is never read as "satisfied"; ``unevaluated_guards`` names them.
        """
        tau = self.tau_field_s
        out = {
            "teeth_resolved": t_train_s * self.rep_rate_hz > 10.0,
            "bandwidth_straddles_qubit": 2.0 * math.pi * omega_q_hz * tau < 1.0,
            "sum_rule": self.rep_rate_hz * tau < 0.1,
            "sideband_resolved": 2.0 * math.pi * mode_hz * t_train_s > 10.0,
            "lamb_dicke": eta * math.sqrt(nbar + 1.0) < 1.0,
        }
        if detuning_hz is not None:
            ok = 1.0 / tau < 0.1 * abs(detuning_hz)  # the bandwidth well inside the detuning
            if fine_structure_hz is not None:
                ok = ok and abs(detuning_hz) <= abs(fine_structure_hz)  # or the D1 and D2 amplitudes cancel
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
    """What the Hamiltonian builder records about a mode-locked drive: the partition and the two clauses it can evaluate
    from the pulse (teeth resolved over the train, the sum rule nu_rep tau << 1)."""
    notes = [
        f"comb drive: {n_tones} explicit beat note(s) at nu_rep = {comb.rep_rate_hz:.6g} Hz, the rest folded into a "
        f"static fourth-order shift of {stark_shift_hz:.6g} Hz (l-sum depth {comb.sum_depth}); never both"
    ]
    n_pulses = duration_s * comb.rep_rate_hz
    if n_pulses <= 10.0:
        notes.append(f"comb drive: guard 'teeth_resolved' FAILS, {n_pulses:.3g} pulses in the train")
    if comb.rep_rate_hz * comb.tau_field_s >= 0.1:
        notes.append(
            f"comb drive: guard 'sum_rule' FAILS, nu_rep tau = {comb.rep_rate_hz * comb.tau_field_s:.3g} "
            "(the tooth-amplitude sum rule is asymptotic in nu_rep tau)"
        )
    return tuple(notes)
