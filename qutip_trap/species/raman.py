"""Single-photon couplings, Raman couplings, light shifts and scattering from the level structure (PLAN.md
Sections 4.5.2, 4.5.4, 4.5.5; milestone M0a).

Everything here is an explicit sum over the field-dressed sublevels of every tabulated E1-connected level of a
:class:`~qutip_trap.species.model.Species` at a :class:`~qutip_trap.device.model.Field`. Conventions:

- single-photon Rabi frequency Omega_{ea} = E_0 <e|d . eps|a>/hbar in the (hbar Omega/2) convention (Section 13),
  with <e|d . eps|a> = sum_q eps_q <e|T_q|a> over the helicity components of the beam about B_hat and
  <e|T_q|a> = (-1)^q conj(<a|T_{-q}|e>) from the lower-upper element of Steck's factorization;
- detuning Delta_e^{(j)} = omega_j - (omega_e - omega_a) with the DRESSED energies, so the three-index detuning
  of Section 4.5.6 (per ground state, per intermediate state, per beam) is automatic: a single Delta per
  intermediate level would return exactly zero for the 0 <-> 0 clock light shift (Section 9.13);
- two-photon Rabi frequency Omega_{g1 g2} = sum_e conj(Omega^{(2)}_{e g2}) Omega^{(1)}_{e g1}/(2 Delta_e^{(1)}) and
  light shift delta_g = sum_j sum_e |Omega^{(j)}_{eg}|^2/(4 Delta_e^{(j)}) (Section 4.5.4; red light lowers a level);
- Kramers-Heisenberg scattering amplitude to (b, q') under beam j: r_{b q'} = sum_e sqrt(Gamma_e) c_{e->b q'}
  Omega^{(j)}_{ea}/(2 Delta_e^{(j)}), with c_{e->b q'} the normalized decay amplitude (sum_{b q'} |c|^2 = 1) and
  sqrt(Gamma_e) INSIDE the coherent sum over e, one factor per path (Section 4.5.5, derivation audit 2026-09-04);
  the rate is Gamma_{a->b} = sum_{q'} |r_{b q'}|^2 (coherent over e, incoherent over the emitted polarization);
- differential Rayleigh dephasing Gamma_el = sum_{q'} |r^{(u)}_{u q'} - r^{(d)}_{d q'}|^2 (Uys et al. 2010), the
  square of the DIFFERENCE of the two qubit states' elastic amplitudes, entering as (1/2) sqrt(Gamma_el) sigma_z.

The phases of Raman couplings between different states depend on the eigenvector sign gauge of the dressed
states and on the azimuthal phase convention of the sigma components; their magnitudes and every rate do not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

from qutip_trap.light.beams import Beam
from qutip_trap.species.dipole import (
    dipole_operator_uncoupled,
    field_amplitude_v_per_m,
    reduced_element_from_partial_rate,
)
from qutip_trap.species.model import Species, Transition, level_j, parse_state_label
from qutip_trap.species.polarization import Vec, spherical_components
from qutip_trap.species.wigner import parity_sign
from qutip_trap.species.zeeman import ZeemanSpectrum, hyperfine_zeeman
from qutip_trap.units import C_M_PER_S, HBAR_J_S, TWO_PI


@dataclass(frozen=True)
class DressedState:
    """One field-dressed sublevel: its level, quantum-number label, absolute energy and uncoupled-basis vector."""

    level: str
    label: str
    energy_hz: float
    vector: np.ndarray

    @property
    def full_label(self) -> str:
        return f"{self.level} {self.label}"


@lru_cache(maxsize=64)
def _operators(nuclear_spin: float, J_lower: Fraction, J_upper: Fraction) -> dict[int, np.ndarray]:
    return {q: dipole_operator_uncoupled(nuclear_spin, J_lower, J_upper, q) for q in (-1, 0, 1)}


def structure_at(species: Species, b_gauss: float, b_hat: Vec) -> AtomicStructure:
    """The :class:`AtomicStructure` of ``species`` at the field (B_gauss, b_hat), built once per distinct field and SHARED.

    A structure is read-only after construction and memoizes its dipole elements and decay amplitudes, so every caller that
    re-derives couplings or scattering rates for the same device (the intrinsic budget per pulse, the played chain per drive,
    the scattering channels per segment, the calibration seeds) reuses the elements instead of recomputing the exact
    Wigner algebra: 6.6 s of a 27 s two-ion Bell run were that recomputation (performance pass 2026-09-09).
    """
    return _structure_cached(species, float(b_gauss), tuple(float(x) for x in np.asarray(b_hat, dtype=float)))


@lru_cache(maxsize=64)
def _structure_cached(species: Species, b_gauss: float, b_hat: tuple[float, ...]) -> AtomicStructure:
    return AtomicStructure(species, b_gauss, b_hat)


class AtomicStructure:
    """A species at a static field: dressed spectra, E1 operators and beam couplings (the engine behind the Species API).

    Instances are read-only after construction and memoize the field-independent algebra per instance (reduced elements,
    the <a|T_q|b> elements between two dressed sublevels, decay amplitudes, total rates and branching totals); obtain them
    through :func:`structure_at` so that callers share one per (species, field).
    """

    def __init__(self, species: Species, b_gauss: float, b_hat: Vec) -> None:
        self.species = species
        self.B_gauss = float(b_gauss)
        self.b_hat = np.asarray(b_hat, dtype=float)
        self.spectra: dict[str, ZeemanSpectrum] = {
            lv.name: hyperfine_zeeman(lv, species.nuclear_spin, species.mu_I_nuclear_magnetons).spectrum(
                self.B_gauss
            )
            for lv in species.levels
        }
        self.e1: dict[tuple[str, str], Transition] = {
            (t.lower, t.upper): t for t in species.transitions if t.multipole == "E1"
        }
        if not self.e1:
            # Section 4.5.4/4.5.5 are explicit sums over the E1-connected levels: with no E1 record every
            # coupling, light shift and scattering rate would silently return 0 or {} (defect B7, 88Sr+).
            # The one legitimate case is an I = 0 optical qubit on the E2 pair itself, which Section 4.5.7
            # drives without any intermediate sum.
            e2_pairs = {frozenset((t.lower, t.upper)) for t in species.transitions if t.multipole == "E2"}
            qubit_levels = frozenset(parse_state_label(lab)[0] for lab in species.qubit)
            if qubit_levels not in e2_pairs:
                raise ValueError(
                    f"{species.name}: no E1 transition is tabulated, so no Section 4.5.4 intermediate sum "
                    f"exists; the qubit pair {sorted(qubit_levels)} is not a tabulated E2 pair either "
                    f"(E2 pairs: {[sorted(p) for p in e2_pairs]})"
                )
        self._states: dict[str, DressedState] = {}
        self._states_by_level: dict[str, tuple[DressedState, ...]] = {}
        for lv in species.levels:
            sp = self.spectra[lv.name]
            per_level: list[DressedState] = []
            for k, lab in enumerate(sp.labels):
                st = DressedState(lv.name, lab, float(sp.energies_hz[k]), sp.eigenvectors[:, k].copy())
                self._states[f"{lv.name} {lab}"] = st
                per_level.append(st)
            self._states_by_level[lv.name] = tuple(per_level)
        self._upper_of: dict[str, list[str]] = {}
        self._lower_of: dict[str, list[str]] = {}
        for lo, up in self.e1:
            self._upper_of.setdefault(lo, []).append(up)
            self._lower_of.setdefault(up, []).append(lo)
        # the memo tables (performance pass 2026-09-09): keyed by the dressed states' labels, which name one object each
        # in this structure; a state from ANOTHER structure (a different field) bypasses them by the identity check
        self._reduced: dict[tuple[str, str], float] = {}
        self._elements: dict[tuple[str, str, int], complex] = {}
        self._decays: dict[str, dict[tuple[str, int], complex]] = {}
        self._gamma: dict[str, float] = {}
        self._branching: dict[str, float] = {}
        self._eps: dict[tuple[complex, ...], np.ndarray] = {}

    # ---- states --------------------------------------------------------------------------------------

    def state(self, label: str) -> DressedState:
        """Look up 'S1/2 F=1 mF=0' (quantum numbers in any order, '+1/2' accepted)."""
        level, rest = parse_state_label(label)
        sp = self.spectra.get(level)
        if sp is None:
            raise KeyError(f"{self.species.name} has no level {level!r}")
        k = sp.index(rest)
        return self._states[f"{level} {sp.labels[k]}"]

    def states_of(self, level: str) -> list[DressedState]:
        return list(self._states_by_level.get(level, ()))

    def upper_levels_of(self, level: str) -> list[str]:
        """Levels connected to ``level`` by a tabulated E1 transition in which ``level`` is the lower one."""
        return list(self._upper_of.get(level, ()))

    def lower_levels_of(self, level: str) -> list[str]:
        return list(self._lower_of.get(level, ()))

    def _own(self, state: DressedState) -> bool:
        """Whether ``state`` is this structure's own object for its label (the memo tables key by label)."""
        return self._states.get(state.full_label) is state

    # ---- dipole elements -----------------------------------------------------------------------------

    def reduced_element_c_m(self, lower: str, upper: str) -> float:
        """|<J||d||J'>| of the tabulated E1 transition lower-upper from its PARTIAL rate (Section 4.5.2)."""
        key = (lower, upper)
        cached = self._reduced.get(key)
        if cached is not None:
            return cached
        tr = self.e1.get(key)
        if tr is None:
            raise KeyError(f"{self.species.name}: no tabulated E1 transition {lower}-{upper}")
        omega = TWO_PI * C_M_PER_S / tr.wavelength_vac_m
        value = reduced_element_from_partial_rate(
            tr.partial_rate_rad_s, omega, level_j(lower), level_j(upper)
        )
        self._reduced[key] = value
        return value

    def _lower_upper_element(self, a: DressedState, b: DressedState, q: int) -> complex:
        """<a|T_q|b> in C m for a in the lower and b in the upper level of an E1 transition (q = m_a - m_b)."""
        own = self._own(a) and self._own(b)
        key = (a.full_label, b.full_label, q)
        if own:
            cached = self._elements.get(key)
            if cached is not None:
                return cached
        ops = _operators(self.species.nuclear_spin, level_j(a.level), level_j(b.level))
        value = complex(np.vdot(a.vector, ops[q] @ b.vector) * self.reduced_element_c_m(a.level, b.level))
        if own:
            self._elements[key] = value
        return value

    def dipole_element_c_m(self, bra: DressedState, ket: DressedState, q: int) -> complex:
        """<bra|T_q|ket> in C m in the field-dressed basis, either ordering of lower and upper (Appendix E)."""
        if (ket.level, bra.level) in self.e1:  # ket lower, bra upper: <b|T_q|a> = (-1)^q conj(<a|T_{-q}|b>)
            return parity_sign(q) * complex(np.conj(self._lower_upper_element(ket, bra, -q)))
        if (bra.level, ket.level) in self.e1:  # bra lower, ket upper
            return self._lower_upper_element(bra, ket, q)
        raise KeyError(
            f"{self.species.name}: {bra.level} and {ket.level} are not connected by a tabulated E1 transition"
        )

    # ---- couplings -----------------------------------------------------------------------------------

    def field_amplitude(self, beam: Beam, position_m: Sequence[float] | None) -> float:
        pos = np.asarray(beam.pointing_m if position_m is None else position_m, dtype=float)
        return field_amplitude_v_per_m(beam.intensity_at(pos))

    def _spherical(self, polarization: Sequence[complex] | np.ndarray) -> np.ndarray:
        """The helicity components of a beam polarization about B_hat, memoized per polarization vector."""
        key = tuple(complex(x) for x in np.asarray(polarization))
        eps = self._eps.get(key)
        if eps is None:
            eps = np.asarray(spherical_components(polarization, self.b_hat))
            self._eps[key] = eps
        return eps

    def single_photon_coupling_rad_s(
        self, a: DressedState, e: DressedState, beam: Beam, position_m: Sequence[float] | None = None
    ) -> complex:
        """Omega_{ea} = E_0 sum_q eps_q <e|T_q|a> / hbar (rad/s, complex) for a lower and e upper."""
        eps = self._spherical(beam.polarization)
        element = sum(eps[q + 1] * self.dipole_element_c_m(e, a, q) for q in (-1, 0, 1))
        return complex(self.field_amplitude(beam, position_m) * element / HBAR_J_S)

    @staticmethod
    def beam_omega_rad_s(beam: Beam) -> float:
        return TWO_PI * C_M_PER_S / beam.wavelength_m

    def detuning_rad_s(self, a: DressedState, e: DressedState, beam: Beam) -> float:
        """Delta_e = omega_L - (omega_e - omega_a) with the dressed energies (plan sign: red negative)."""
        return self.beam_omega_rad_s(beam) - TWO_PI * (e.energy_hz - a.energy_hz)

    def couplings_from(
        self, a: DressedState, beam: Beam, position_m: Sequence[float] | None = None
    ) -> list[tuple[DressedState, complex, float]]:
        """(e, Omega_{ea}, Delta_e) for every upper sublevel E1-connected to a's level."""
        out: list[tuple[DressedState, complex, float]] = []
        for up in self.upper_levels_of(a.level):
            for e in self.states_of(up):
                omega = self.single_photon_coupling_rad_s(a, e, beam, position_m)
                out.append((e, omega, self.detuning_rad_s(a, e, beam)))
        return out

    def nearest_resonance_in_linewidths(
        self, a: DressedState, beam: Beam, position_m: Sequence[float] | None = None
    ) -> float:
        """min_e |Delta_e|/Gamma_e over the intermediate sublevels: how far from resonance the sums are.

        PLAN.md 4.5.6: the second-order sums "have no i gamma/2 in their denominators and are used only far
        from resonance, the multi-level Bloch solve of Section 4.2.8 taking over within a few linewidths".
        This is the diagnostic that says which side of that line a call is on; ``inf`` when the beam couples
        nothing. :meth:`refuse_if_near_resonance` turns it into a refusal.
        """
        worst = math.inf
        for e, _om, delta in self.couplings_from(a, beam, position_m):
            gamma = self.total_decay_rate_rad_s(e.level)
            if gamma > 0.0:
                worst = min(worst, abs(delta) / gamma)
        return worst

    def refuse_if_near_resonance(
        self,
        a: DressedState,
        beam: Beam,
        position_m: Sequence[float] | None = None,
        *,
        min_linewidths: float = 10.0,
    ) -> None:
        """Raise when the beam is within ``min_linewidths`` Gamma_e of any dressed intermediate sublevel.

        Not called by the second-order sums themselves: PLAN.md 4.5.6 requires the residual excited
        population to be "reported with its size", which ``residual_excited_population`` does, and a hard
        refusal inside every sum would break the legitimate near-resonant uses of ``couplings_from`` by the
        Bloch layer. Callers that mean "far from resonance" call this first.
        """
        margin = self.nearest_resonance_in_linewidths(a, beam, position_m)
        if margin < min_linewidths:
            raise ValueError(
                f"{self.species.name}: the beam at {beam.wavelength_m * 1e9:.4f} nm is {margin:.3g} "
                f"linewidths from a dressed intermediate sublevel of {a.full_label}; the second-order sums "
                f"of Section 4.5.4 have no i gamma/2 and are valid only far from resonance (the multi-level "
                f"Bloch solve of Section 4.2.8 takes over within a few linewidths)"
            )

    def light_shift_rad_s(
        self, a: DressedState, beams: Sequence[Beam], position_m: Sequence[float] | None = None
    ) -> float:
        """delta_a = sum_j sum_e |Omega^{(j)}_{ea}|^2 / (4 Delta_e^{(j)}) (Section 4.5.4)."""
        total = 0.0
        for beam in beams:
            for _e, omega, delta in self.couplings_from(a, beam, position_m):
                total += abs(omega) ** 2 / (4.0 * delta)
        return total

    def raman_coupling_rad_s(
        self,
        g1: DressedState,
        g2: DressedState,
        beam1: Beam,
        beam2: Beam,
        position_m: Sequence[float] | None = None,
    ) -> complex:
        """Omega_{g1 g2} = sum_e conj(Omega^{(2)}_{e g2}) Omega^{(1)}_{e g1} / (2 Delta_e^{(1)}) over the shared upper sublevels."""
        c1 = {e.full_label: (om, d) for e, om, d in self.couplings_from(g1, beam1, position_m)}
        c2 = {e.full_label: om for e, om, _ in self.couplings_from(g2, beam2, position_m)}
        total = 0.0 + 0.0j
        for key, (om1, d1) in c1.items():
            om2 = c2.get(key)
            if om2 is not None:
                total += np.conj(om2) * om1 / (2.0 * d1)
        return complex(total)

    def residual_excited_population(
        self, a: DressedState, beam: Beam, position_m: Sequence[float] | None = None
    ) -> float:
        """sum_e |Omega_{ea}|^2/(4 Delta_e^2): the adiabatic-elimination residual the scattering channels use (Section 4.5.4)."""
        return sum(abs(om) ** 2 / (4.0 * d * d) for _e, om, d in self.couplings_from(a, beam, position_m))

    # ---- decay and scattering ------------------------------------------------------------------------

    def total_decay_rate_rad_s(self, level: str) -> float:
        """Gamma_e of a level, from its tabulated transitions (all of which must carry the same total rate)."""
        cached = self._gamma.get(level)
        if cached is not None:
            return cached
        rates = {tr.label: tr.gamma_rad_s for (_lo, up), tr in self.e1.items() if up == level}
        if not rates:
            raise KeyError(f"{self.species.name}: level {level} has no tabulated E1 decay")
        first = next(iter(rates.values()))
        disagree = {lab: g for lab, g in rates.items() if not math.isclose(g, first, rel_tol=1e-9)}
        if disagree:
            raise ValueError(
                f"{self.species.name}: the E1 transitions out of {level} disagree on its total decay rate "
                f"(Section 13: Transition.gamma_hz is the UPPER level's total rate): {rates}"
            )
        self._gamma[level] = first
        return first

    def tabulated_branching_total(self, level: str) -> float:
        """sum_lo branching(lo -> level) over the TABULATED E1 channels out of ``level``.

        1 when the E1 decay of ``level`` is fully tabulated; less when a channel is declared on the
        :class:`~qutip_trap.species.model.Level` as ``untabulated_branching`` (Section 4.5.5): the missing
        fraction must never be renormalized away, because that is what silently moved 40Ca+ P3/2's 5.9% of
        D-state leakage into the 393 nm cycling line.
        """
        cached = self._branching.get(level)
        if cached is None:
            cached = sum(tr.branching for (_lo, up), tr in self.e1.items() if up == level)
            self._branching[level] = cached
        return cached

    def decay_amplitudes(self, e: DressedState) -> dict[tuple[str, int], complex]:
        """sqrt(Gamma_e) c_{e->b q'}: decay amplitudes of e into every lower sublevel b and polarization index q'.

        Section 4.5.5 fixes the normalization through <b|d_q'|e> = sqrt(3 pi eps0 hbar c^3 Gamma_e/omega_e^3)
        c_{e->b q'}, so |c|^2 is the PARTIAL-RATE fraction and carries omega_{e,b}^3 **per channel**: each raw
        element is weighted by omega_{e,lo}^(3/2) before the normalization. Without that weight the branching
        weight per channel is Gamma_partial/omega^3 rather than Gamma_partial, which overstated 171Yb+'s
        P1/2 -> D3/2 leakage by 118x (see ``dynamics/multilevel.py``, which already carries one omega^3 per line).
        sum_{b q'} |c|^2 then equals :meth:`tabulated_branching_total` (1 for a fully tabulated level), never
        an unconditional 1; keys are (b.full_label, q') with q' = m_b - m_e the emitted tensor index.
        """
        own = self._own(e)
        if own:
            memo = self._decays.get(e.full_label)
            if memo is not None:
                return dict(memo)
        raw: dict[tuple[str, int], complex] = {}
        for lo in self.lower_levels_of(e.level):
            tr = self.e1[(lo, e.level)]
            weight = (TWO_PI * C_M_PER_S / tr.wavelength_vac_m) ** 1.5  # per-CHANNEL omega^(3/2)
            for b in self.states_of(lo):
                for q in (-1, 0, 1):
                    val = weight * self._lower_upper_element(b, e, q)
                    if val != 0.0:
                        raw[(b.full_label, q)] = val
        norm = math.sqrt(sum(abs(v) ** 2 for v in raw.values()))
        scale = (
            math.sqrt(self.total_decay_rate_rad_s(e.level) * self.tabulated_branching_total(e.level)) / norm
        )
        out = {k: scale * v for k, v in raw.items()}
        if own:
            self._decays[e.full_label] = dict(out)
        return out

    def scattering_amplitudes_by_path(
        self, a: DressedState, beam: Beam, position_m: Sequence[float] | None = None
    ) -> dict[tuple[str, int], dict[str, complex]]:
        """r_{b q'} split by intermediate level: {(b, q'): {upper level: partial sum over its sublevels}} (units sqrt(1/s))."""
        out: dict[tuple[str, int], dict[str, complex]] = {}
        for e, omega, delta in self.couplings_from(a, beam, position_m):
            for key, amp in self.decay_amplitudes(e).items():
                paths = out.setdefault(key, {})
                paths[e.level] = paths.get(e.level, 0.0 + 0.0j) + amp * omega / (2.0 * delta)
        return out

    def scattering_amplitudes(
        self, a: DressedState, beam: Beam, position_m: Sequence[float] | None = None
    ) -> dict[tuple[str, int], complex]:
        """r_{b q'} = sum_e sqrt(Gamma_e) c_{e->b q'} Omega_{ea}/(2 Delta_e), coherent over every intermediate sublevel."""
        return {
            k: complex(sum(v.values()))
            for k, v in self.scattering_amplitudes_by_path(a, beam, position_m).items()
        }

    def scattering_rates(
        self, a: DressedState, beam: Beam, position_m: Sequence[float] | None = None
    ) -> dict[str, float]:
        """Gamma_{a->b} in s^-1 for every reachable lower sublevel b (b = a is the Rayleigh rate), incoherent over q'."""
        rates: dict[str, float] = {}
        for (b_label, _q), amp in self.scattering_amplitudes(a, beam, position_m).items():
            rates[b_label] = rates.get(b_label, 0.0) + abs(amp) ** 2
        return rates

    def rayleigh_dephasing_rate(
        self, up: DressedState, down: DressedState, beam: Beam, position_m: Sequence[float] | None = None
    ) -> float:
        """Gamma_el = sum_{q'} |r^{(u)}_{u q'} - r^{(d)}_{d q'}|^2 (Uys et al. 2010): the elastic-amplitude DIFFERENCE, squared."""
        ru = self.scattering_amplitudes(up, beam, position_m)
        rd = self.scattering_amplitudes(down, beam, position_m)
        total = 0.0
        for q in (-1, 0, 1):
            a_u = ru.get((up.full_label, q), 0.0 + 0.0j)
            a_d = rd.get((down.full_label, q), 0.0 + 0.0j)
            total += abs(a_u - a_d) ** 2
        return total


__all__ = ["AtomicStructure", "DressedState", "structure_at"]
