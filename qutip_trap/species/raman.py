"""Single-photon couplings, Raman couplings, light shifts and scattering from the level structure (PLAN.md Section 4.5):
explicit sums over the field-dressed sublevels of every tabulated E1-connected level.

- Single-photon Rabi frequency Omega_{ea} = E_0 <e|d . eps|a>/hbar in the (hbar Omega/2) convention, with
  <e|d . eps|a> = sum_q eps_q <e|T_q|a> over the beam's helicity components about B_hat and
  <e|T_q|a> = (-1)^q conj(<a|T_{-q}|e>) from the lower-upper element.
- Detuning Delta_e^{(j)} = omega_j - (omega_e - omega_a) with the DRESSED energies, so the detuning is per ground state,
  per intermediate state and per beam (one Delta per intermediate level would give zero for the 0 <-> 0 clock shift).
- Two-photon Rabi frequency Omega_{g1 g2} = sum_e conj(Omega^{(2)}_{e g2}) Omega^{(1)}_{e g1}/(2 Delta_e^{(1)}); light shift
  delta_g = sum_j sum_e |Omega^{(j)}_{eg}|^2/(4 Delta_e^{(j)}) (red light lowers a level).
- Kramers-Heisenberg amplitude to (b, q') under beam j: r_{b q'} = sum_e sqrt(Gamma_e) c_{e->b q'} Omega^{(j)}_{ea}/
  (2 Delta_e^{(j)}), sqrt(Gamma_e) INSIDE the coherent sum over e and c_{e->b q'} the normalized decay amplitude; the rate
  is Gamma_{a->b} = sum_{q'} |r_{b q'}|^2 (coherent over e, incoherent over the emitted polarization).
- Differential Rayleigh dephasing Gamma_el = sum_{q'} |r^{(u)}_{u q'} - r^{(d)}_{d q'}|^2 (Uys et al. 2010), entering as
  (1/2) sqrt(Gamma_el) sigma_z.

Raman phases between different states depend on the eigenvector sign gauge and the azimuthal phase of the sigma
components; their magnitudes and every rate do not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import NamedTuple

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


class Coupling(NamedTuple):
    """An upper sublevel E1-connected to a lower one: Omega_{ea} and Delta_e, both in rad/s."""

    state: DressedState
    omega_rad_s: complex
    detuning_rad_s: float


@lru_cache(maxsize=64)
def _operators(nuclear_spin: float, J_lower: Fraction, J_upper: Fraction) -> dict[int, np.ndarray]:
    return {q: dipole_operator_uncoupled(nuclear_spin, J_lower, J_upper, q) for q in (-1, 0, 1)}


def structure_at(species: Species, b_gauss: float, b_hat: Vec) -> AtomicStructure:
    """The :class:`AtomicStructure` of ``species`` at the field (B_gauss, b_hat), built once per distinct field and shared
    by every caller (its memoized elements are the expensive part)."""
    return _structure_cached(species, float(b_gauss), tuple(float(x) for x in np.asarray(b_hat, dtype=float)))


@lru_cache(maxsize=64)
def _structure_cached(species: Species, b_gauss: float, b_hat: tuple[float, ...]) -> AtomicStructure:
    return AtomicStructure(species, b_gauss, b_hat)


class AtomicStructure:
    """A species at a static field: dressed spectra, E1 operators and beam couplings (the engine behind the Species API).

    Read-only after construction; it memoizes the dipole elements between dressed sublevels and the decay amplitudes, so
    obtain it through :func:`structure_at`, which shares one per (species, field).
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
            # with no E1 record every coupling, light shift and scattering rate would silently be 0; the one legitimate
            # case is an optical qubit on a tabulated E2 pair, driven without any intermediate sum (Section 4.5.7)
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
        self._gamma: dict[str, float] = {}  # each E1 line out of a level carries its total rate
        for (lo, up), tr in self.e1.items():
            self._upper_of.setdefault(lo, []).append(up)
            self._lower_of.setdefault(up, []).append(lo)
            self._gamma.setdefault(up, tr.gamma_rad_s)
        self._branching = {
            up: sum(self.e1[lo, up].branching for lo in los) for up, los in self._lower_of.items()
        }
        # memo tables keyed by the dressed states' labels, which name one object each in this structure; a state from
        # another structure (a different field) bypasses them by the identity check of ``_own``
        self._reduced: dict[tuple[str, str], float] = {}
        self._elements: dict[tuple[str, str, int], complex] = {}
        self._decays: dict[str, dict[tuple[str, int], complex]] = {}
        self._eps: dict[tuple[complex, ...], np.ndarray] = {}

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
        """<bra|T_q|ket> in C m in the field-dressed basis, either ordering of lower and upper."""
        if (ket.level, bra.level) in self.e1:  # ket lower, bra upper: <b|T_q|a> = (-1)^q conj(<a|T_{-q}|b>)
            return parity_sign(q) * complex(np.conj(self._lower_upper_element(ket, bra, -q)))
        if (bra.level, ket.level) in self.e1:  # bra lower, ket upper
            return self._lower_upper_element(bra, ket, q)
        raise KeyError(
            f"{self.species.name}: {bra.level} and {ket.level} are not connected by a tabulated E1 transition"
        )

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
    ) -> list[Coupling]:
        """Every upper sublevel E1-connected to a's level."""
        return [
            Coupling(
                e, self.single_photon_coupling_rad_s(a, e, beam, position_m), self.detuning_rad_s(a, e, beam)
            )
            for up in self.upper_levels_of(a.level)
            for e in self.states_of(up)
        ]

    def light_shift_rad_s(
        self, a: DressedState, beams: Sequence[Beam], position_m: Sequence[float] | None = None
    ) -> float:
        """delta_a = sum_j sum_e |Omega^{(j)}_{ea}|^2 / (4 Delta_e^{(j)}) (Section 4.5.4)."""
        total = 0.0
        for beam in beams:
            for c in self.couplings_from(a, beam, position_m):
                total += abs(c.omega_rad_s) ** 2 / (4.0 * c.detuning_rad_s)
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
        c2 = {c.state.full_label: c.omega_rad_s for c in self.couplings_from(g2, beam2, position_m)}
        total = 0.0 + 0.0j
        for c in self.couplings_from(g1, beam1, position_m):
            om2 = c2.get(c.state.full_label)
            if om2 is not None:
                total += np.conj(om2) * c.omega_rad_s / (2.0 * c.detuning_rad_s)
        return complex(total)

    def residual_excited_population(
        self, a: DressedState, beam: Beam, position_m: Sequence[float] | None = None
    ) -> float:
        """sum_e |Omega_{ea}|^2/(4 Delta_e^2): the adiabatic-elimination residual excited population (Section 4.5.4)."""
        return sum(
            abs(c.omega_rad_s) ** 2 / (4.0 * c.detuning_rad_s * c.detuning_rad_s)
            for c in self.couplings_from(a, beam, position_m)
        )

    def total_decay_rate_rad_s(self, level: str) -> float:
        """Gamma_e (rad/s) of an E1-decaying level: the total rate its tabulated E1 lines carry."""
        return self._gamma[level]

    def tabulated_branching_total(self, level: str) -> float:
        """sum_lo branching(lo -> level) over the TABULATED E1 channels out of ``level`` (0 without any): 1 when its decay
        is fully tabulated, less by the ``Level.untabulated_branching`` it declares, which is never renormalized away."""
        return self._branching.get(level, 0.0)

    def decay_amplitudes(self, e: DressedState) -> dict[tuple[str, int], complex]:
        """sqrt(Gamma_e) c_{e->b q'}: decay amplitudes of e into every lower sublevel b and polarization index q'.

        <b|d_q'|e> = sqrt(3 pi eps0 hbar c^3 Gamma_e/omega_e^3) c_{e->b q'} (Section 4.5.5) makes |c|^2 the PARTIAL-RATE
        fraction, so each raw element carries omega_{e,lo}^(3/2) per channel before the normalization, and
        sum_{b q'} |c|^2 equals :meth:`tabulated_branching_total`. Keys are (b.full_label, q'), q' = m_b - m_e.
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
        for c in self.couplings_from(a, beam, position_m):
            level = c.state.level
            for key, amp in self.decay_amplitudes(c.state).items():
                paths = out.setdefault(key, {})
                paths[level] = paths.get(level, 0.0 + 0.0j) + amp * c.omega_rad_s / (2.0 * c.detuning_rad_s)
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
