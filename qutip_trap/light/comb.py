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
    """sech|x|, with |x| clamped at 700 because cosh overflows float64 there (cosh(710) = inf).

    sech(700) = 9.9e-305, so every clamped value is already indistinguishable from zero in double precision and the
    clamp changes no result; without it the far tail of a tooth sum returns nan rather than 0 (M2 audit E19).
    """
    return 1.0 / np.cosh(np.clip(np.abs(x), 0.0, 700.0))


FIELD_FWHM_OVER_TAU = (4.0 / math.pi) * math.acosh(2.0)
"""1.6768: the field FWHM of sech(pi t/(2 tau))."""
INTENSITY_FWHM_OVER_TAU = (4.0 / math.pi) * math.acosh(math.sqrt(2.0))
"""1.1222: the sech^2 intensity FWHM."""
ZETA3_COEFFICIENT = 0.852556797635
"""7 zeta(3)/pi^2 = int_0^inf sech^2 v tanh v / v dv, the single-comb fourth-order closed-form coefficient (printed 0.853)."""
SUM_DEPTH_HALF_WIDTHS = 8.0
"""l_max = ceil(k/(pi nu_rep tau)) at k = 8: the sech^2 Stark weight is centred on the resonant separation with half-width
1/(pi nu_rep tau) (189 at 120 MHz and 14 ps, 398 at 80 MHz and 10 ps), and the sech^2 tail beyond k half-widths is
about 4 e^{-2k} of the sum, so eight is where the 1e-6 doubling check of ``stark4_hz`` passes (five leaves 1.7e-5 at the
Lee point, measured). The plan's "sum to |k| >~ 1000" is then a CONSEQUENCE (1516 at 120 MHz and 14 ps, 3184 at 80 MHz
and 10 ps) rather than a literal that refuses the second operating point (M2 audit E5)."""
SINGULAR_REL_TOL = 1e-9
"""|delta_{n,a}|/nu_rep below which a splitting counts as sitting ON a beat note and the l-sum is refused: the plan's guard
is delta_{n,a} != 0 and never j != 0, but exact float equality is defeated by any comb whose rep rate nearly divides the
splitting (nu_rep = nu_q/158 gave 1.12e17 Hz silently; M2 audit E6)."""
MARGINAL_COUPLING_FACTOR = 1.0
"""|delta_{n,a}| below this times |Omega_{n,a}| is the "marginal at full power" band: the expansion parameter is order unity
and the plan does not claim the closed form there (Omega_0/2pi = 45.7 MHz against delta_00,10/2pi = 42.8 MHz at the source's
own operating point), so it is REPORTED as a warning and never raised - raising would refuse that operating point."""


def _assert_converged(value: float, doubled: float, lmax: int, what: str, rtol: float = 1e-6) -> None:
    """The doubling check of Section 4.3.7: the |k| ~ 5 to 10 plateau is a factor 2.093 low, so successive partial sums
    converge falsely and only 2 x l_max settles it."""
    if abs(doubled - value) > rtol * max(abs(value), 1e-300):
        raise ValueError(
            f"{what} has not converged at l_max = {lmax}: {value} vs {doubled} at 2 l_max (raise pair_order_max, or "
            "leave it None so the envelope sets the depth; Section 4.3.7)"
        )


TauConvention = Literal["field_sech", "intensity_sech", "intensity_sech2", "field_fwhm", "intensity_fwhm"]
"""Every convention ``tau_field_from`` converts. Appendix E's ``CombSpec.tau_convention`` Literal listed only the first
three while the function already accepted the two FWHM names, so a legal argument was a type error (M2 audit E19)."""


def tau_field_from(value_s: float, convention: str) -> float:
    """The FIELD sech parameter tau of sech(pi t/(2 tau)) from a value quoted in another convention.

    ``intensity_sech`` and ``intensity_sech2`` are the SAME conversion and not the same envelope: both name a parameter
    of the intensity profile |sech(pi t/(2 tau))|^2 = sech^2(pi t/(2 tau)), whose sech argument is twice the field one,
    so tau_field = value/2 either way; the two spellings exist because sources use both for that one parameter. A
    genuinely different envelope, sech(pi t/tau), is the retired convention of PLAN.md:1454 and is NOT accepted here -
    its widths are the 0.8384/0.5611 negative control of ``check_comb.py``.
    """
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
    """nu_rep, ordinary Hz; the tooth spacing."""
    tau_s: float
    """sech ENVELOPE parameter in the convention ``tau_convention`` names, never a FWHM."""
    tau_convention: TauConvention
    carrier_order: int
    """j = round(omega_HF / omega_rep), signed."""
    aom_offset_hz: float
    """Delta_nu_M = nu_1 - nu_2, SIGNED; its sign selects red vs blue."""
    pair_order_max: int | None = None
    """|l - j| retained in the fourth-order sum. None = derived from the envelope (``sum_depth``); an explicit value
    overrides it and is validated against the derived depth. Appendix E declares a literal 1200, which is 3x too
    shallow at the plan's own 80 MHz / 10 ps operating point (M2 audit E5; ledger conv.comb_sum_depth)."""
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

    @property
    def sum_half_width(self) -> float:
        """1/(pi nu_rep tau): the half-width, in tooth SEPARATION l, of the sech^2 fourth-order weight."""
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
        """g_k/g_0 (unnormalized): sqrt(pi nu_rep tau) sech(2 pi k nu_rep tau) for SINGLE teeth k from the comb centre.

        The one place the single-tooth amplitude is written; ``tooth_weight``, ``tooth_amplitudes``, ``sum_rule_raw``
        and ``pair_sum_exact`` all read it, where four copies of the expression used to sit (M2 audit E19).
        """
        x = np.asarray(k, dtype=float)
        return np.asarray(
            math.sqrt(math.pi * self.rep_rate_hz * self.tau_field_s)
            * np.asarray(sech(2.0 * math.pi * x * self.rep_rate_hz * self.tau_field_s))
        )

    def tooth_weight(self, k: int) -> float:
        """g_k/g_0 (unnormalized) for one tooth (``tooth_weights`` at a scalar)."""
        return float(self.tooth_weights(k))

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

    def resonance_target_hz(self, omega_q_hz: float, mode_hz: float | None, sideband: int = 0) -> float:
        """nu_q + sideband x nu_mode: the beat-note frequency the drive is tuned to (Section 13, detuning symbols)."""
        return omega_q_hz + (sideband * mode_hz if mode_hz is not None and sideband else 0.0)

    def explicit_orders(self, resonance_hz: float, gate_time_s: float | None) -> frozenset[int]:
        """The beat-note orders j that stay EXPLICIT as QobjEvo coefficients: |mu_j - mu_res| <= 10/t_g (Section 4.3.7).

        ``tones()`` emits exactly these and ``comb_factor``/``stark4_hz`` exclude exactly these, so the explicit and the
        folded sets PARTITION the comb's beat notes: "near-resonant tones stay explicit ... while only far-detuned ones
        fold into the static fourth-order shift below, never both" (PLAN.md:515). Before the M2 fix every retained tone
        was also inside the static sum, a factor 4.78 of double-counted shift at the Lee point (audit E7).

        ``gate_time_s=None`` means no cut: every tone is explicit and nothing folds.
        """
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
        """The comb's tone set: one Tone per beat note j nu_rep + Delta_nu_M with envelope Omega_0 sech(pi j nu_rep tau).

        Detunings are measured from the carrier (Section 13): mu_j = (j nu_rep + Delta_nu_M) - nu_q, sideband-shifted by
        ``sideband`` x mode_hz for the resonance target. Near-resonant beat notes stay explicit: |mu_j - mu_res| <~ 10/t_g
        (about 100 kHz for a 100 us gate; every tone when gate_time_s is None), the rest fold into ``stark4_hz``, and the
        two sets partition the comb (``explicit_orders``).
        """
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

    # ---- fourth-order Stark shift -----------------------------------------------------------------------------------

    def beat_notes_hz(self, orders: np.ndarray) -> np.ndarray:
        """l nu_rep + Delta_nu_M for a vector of orders: the ONE beat-note grid the tone set and the l-sum share.

        ``tones()`` places beat notes at j nu_rep + Delta_nu_M; before the M2 fix the fourth-order sum placed them at
        l nu_rep, so a two-comb drive (the operating configuration, since the AOM sign selects red from blue) had its
        explicit tones and its static shift on different grids (audit E7).
        """
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
        """delta_{n,a} = omega_a - (j nu_rep + Delta_nu_M) at the nearest FOLDED beat note, guarded and reported.

        The plan's guard is delta_{n,a} != 0 and NEVER j != 0 (j = 0 is the operative case for the Zeeman partners), but
        exact float equality is no guard at all: any comb whose rep rate nearly divides the splitting slips through
        (nu_rep = nu_q/158 returned 1.12e17 Hz silently). The threshold is ``SINGULAR_REL_TOL`` x nu_rep, stated in the
        message. ``excluded`` is the explicit tone set: a beat note that stays a QobjEvo coefficient is not a term of
        this perturbative sum, so its vanishing detuning is not a singularity of it - which is what lets a comb LOCKED
        to the qubit (delta = 0 by construction) still carry a fourth-order shift from the far-detuned teeth. Without
        the partition such a comb is refused, and correctly so: Section 12 lists "the missing fourth-order shift for a
        resonant carrier drive" as an open item.

        |delta| below |Omega_{n,a}| is the source's own "marginal at full power" band and is WARNED, not refused
        (PLAN.md:530 keeps that operating point and only declines to claim the closed form there).
        """
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
        """The nearest beat-note order that is NOT held explicit.

        l = 0 is NOT skipped: it is the operative case for the Zeeman partners (omega_Zee/2pi ~ 7 MHz < nu_rep/2), where
        C_{10,11} = C_{10,1-1} exactly and that equality is what makes the elliptical-polarization cancellation survive
        the comb generalization. The plan is explicit that the guard is delta_{n,a} != 0 and never j != 0 - a j != 0
        guard deletes exactly those terms (Section 4.3.7). Only ``tones()`` skips j = 0, because a tooth does not beat
        with itself and there is no drive at zero beat frequency.
        """
        j0 = self._nearest_order(omega_a_hz)
        for step in range(0, 4 * len(excluded) + 2):
            for j in (j0 + step, j0 - step) if step else (j0,):
                if j not in excluded:
                    return j
        raise ValueError("every nearby beat note is held explicit: nothing folds into the static shift")

    def _l_sum(
        self, omega_a_hz: float, lmax: int, excluded: frozenset[int], *, coupling_hz: float, what: str
    ) -> float:
        """sum_l sech^2(pi l nu_rep tau)/(omega_a - (l nu_rep + Delta_nu_M)) over the FOLDED orders only.

        The range is |l - j| <= lmax about the NEAREST beat note j, which is the plan's |k| <= K convention (k = l - j),
        while the guard above is evaluated at the nearest FOLDED order: the two differ only when the resonant note is
        held explicit, and the range must stay centred on j for the partial sums to be the ones Section 4.3.7 tabulates.
        The sech^2 weight itself peaks at l = 0, which the range covers because lmax >> j at every operating point.
        """
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
        """C_{n,a}-type single sum over the tooth separation l: delta sum_l sech^2(pi l nu_rep tau)/(omega_a - l omega_rep
        - omega_A), normalized so that its k = 0 term at the nearest beat note reproduces Lee's C (0.746343 alone,
        0.943789 converged at 120 MHz, 14 ps, 12.642821 GHz); the guard is delta != 0 for every l, never j != 0.

        ``gate_time_s`` with ``resonance_hz`` excludes the orders ``tones()`` keeps explicit ("never both");
        ``validate=True`` repeats the sum at 2 l_max and refuses a change above 1e-6 relative, the same doubling check
        ``stark4_hz`` runs (``comb_factor`` had none at all before the M2 fix; audit E5).
        """
        lmax = self.sum_depth if l_max is None else l_max
        excluded = frozenset() if resonance_hz is None else self.explicit_orders(resonance_hz, gate_time_s)
        # guard once and normalize by the same delta the guard measured; _l_sum re-checks with coupling_hz = 0 so the
        # "marginal at full power" warning is emitted once per call and not once per sum
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
        """E^{(4)}_n/2pi = sum_{a != n} (Omega_{n,a}^2/4) sum_l sech^2(pi l nu_rep tau)/(omega_a - l omega_rep - omega_A)
        (Section 4.3.7).

        ``levels``: the manifold's sublevel frequencies in Hz (or Level objects, read through ``energy_hz``) relative to
        any origin; ``couplings_hz``: Omega_{n,a}/2pi per sublevel (the polarization-weighted Omega_0, carrying NO sech
        factor); ``gate_time_s`` with ``resonance_hz`` excludes the beat notes ``tones()`` keeps explicit, so the two
        paths partition the comb and no tone is counted twice. Returns the shift of level ``n_index`` and, doubling
        l_max, asserts stability to 1e-6 relative.
        """
        freqs = np.array([float(lv.energy_hz) if hasattr(lv, "energy_hz") else float(lv) for lv in levels])
        if couplings_hz is None:
            raise ValueError("stark4_hz needs the per-sublevel couplings Omega_{n,a}/2pi")
        omegas = np.asarray(couplings_hz, dtype=float)
        if omegas.shape != freqs.shape:
            raise ValueError("one coupling per sublevel")
        excluded = frozenset() if resonance_hz is None else self.explicit_orders(resonance_hz, gate_time_s)
        # guard (and warn about) every contributing sublevel ONCE, before the two sums of the doubling check
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

    # ---- guards ------------------------------------------------------------------------------------------------------

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
        """The validity hierarchy of Section 4.3.7 as named booleans the engine logs before every solve.

        The clauses that need inputs this record does not hold appear ONLY when those inputs are supplied, so a caller
        cannot mistake "not evaluated" for "satisfied": ``adiabatic_elimination`` needs the fine-structure detuning
        (omega_q << 1/tau << |Delta| <~ omega_FS) and ``small_pulse_area`` the per-pulse area theta << 1, which comes
        from the intensity-to-Rabi chain (Section 12: not implemented). Before the M2 fix both were the literal
        ``True`` (audit E8/C). ``unevaluated_guards`` names what is missing.
        """
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
            # 1/tau << |Delta|: the bandwidth must sit well inside the detuning for the elimination to hold, and
            # |Delta| <~ omega_FS or the D1 and D2 amplitudes cancel (Section 4.3.7)
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
        """The validity clauses ``guards()`` cannot evaluate from the inputs given (Section 12's open items)."""
        missing: list[str] = []
        if detuning_hz is None:
            missing.append("adiabatic_elimination (needs the fine-structure detuning Delta)")
        if theta_per_pulse_rad is None:
            missing.append("small_pulse_area (needs the intensity-to-Rabi chain's theta per pulse)")
        return tuple(missing)


def comb_build_notes(
    comb: CombSpec, duration_s: float, n_tones: int, stark_shift_hz: float
) -> tuple[str, ...]:
    """What the Hamiltonian builder records about a mode-locked drive (Section 5.7's ``approximations``).

    Only the clauses the builder can evaluate from the pulse itself: the train length against the tooth spacing
    (teeth_resolved), the sum rule nu_rep tau << 1, and the explicit/folded partition that ``CombSpec.explicit_orders``
    fixed. The clauses that need the species or the intensity chain are reported by ``CombSpec.unevaluated_guards`` at
    the point the drive is constructed (``light.raman.comb_drive``), where those inputs exist.
    """
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
