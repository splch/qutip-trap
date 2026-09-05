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


class AtomicStructure:
    """A species at a static field: dressed spectra, E1 operators and beam couplings (the engine behind the Species API)."""

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
        self._states: dict[str, DressedState] = {}
        for lv in species.levels:
            sp = self.spectra[lv.name]
            for k, lab in enumerate(sp.labels):
                st = DressedState(lv.name, lab, float(sp.energies_hz[k]), sp.eigenvectors[:, k].copy())
                self._states[f"{lv.name} {lab}"] = st

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
        return [st for st in self._states.values() if st.level == level]

    def upper_levels_of(self, level: str) -> list[str]:
        """Levels connected to ``level`` by a tabulated E1 transition in which ``level`` is the lower one."""
        return [up for (lo, up) in self.e1 if lo == level]

    def lower_levels_of(self, level: str) -> list[str]:
        return [lo for (lo, up) in self.e1 if up == level]

    # ---- dipole elements -----------------------------------------------------------------------------

    def reduced_element_c_m(self, lower: str, upper: str) -> float:
        """|<J||d||J'>| of the tabulated E1 transition lower-upper from its PARTIAL rate (Section 4.5.2)."""
        tr = self.e1.get((lower, upper))
        if tr is None:
            raise KeyError(f"{self.species.name}: no tabulated E1 transition {lower}-{upper}")
        omega = TWO_PI * C_M_PER_S / tr.wavelength_vac_m
        return reduced_element_from_partial_rate(tr.partial_rate_rad_s, omega, level_j(lower), level_j(upper))

    def _lower_upper_element(self, a: DressedState, b: DressedState, q: int) -> complex:
        """<a|T_q|b> in C m for a in the lower and b in the upper level of an E1 transition (q = m_a - m_b)."""
        ops = _operators(self.species.nuclear_spin, level_j(a.level), level_j(b.level))
        return complex(np.vdot(a.vector, ops[q] @ b.vector) * self.reduced_element_c_m(a.level, b.level))

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

    def single_photon_coupling_rad_s(
        self, a: DressedState, e: DressedState, beam: Beam, position_m: Sequence[float] | None = None
    ) -> complex:
        """Omega_{ea} = E_0 sum_q eps_q <e|T_q|a> / hbar (rad/s, complex) for a lower and e upper."""
        eps = spherical_components(beam.polarization, self.b_hat)
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
        """Gamma_e of a level, from its tabulated transitions (all of which carry the same total rate)."""
        rates = {tr.gamma_rad_s for (lo, up), tr in self.e1.items() if up == level}
        if not rates:
            raise KeyError(f"{self.species.name}: level {level} has no tabulated E1 decay")
        return rates.pop()

    def decay_amplitudes(self, e: DressedState) -> dict[tuple[str, int], complex]:
        """sqrt(Gamma_e) c_{e->b q'}: normalized decay amplitudes of e into every lower sublevel b and polarization index q'.

        c_{e->b q'} = <b|T_{q'}|e> d_red / N_e with N_e^2 = sum_{b q'} |<b|T_{q'}|e> d_red|^2, so sum_{b q'} |c|^2 = 1 exactly;
        keys are (b.full_label, q') with q' = m_b - m_e the emitted tensor index.
        """
        raw: dict[tuple[str, int], complex] = {}
        for lo in self.lower_levels_of(e.level):
            for b in self.states_of(lo):
                for q in (-1, 0, 1):
                    val = self._lower_upper_element(b, e, q)
                    if val != 0.0:
                        raw[(b.full_label, q)] = val
        norm = math.sqrt(sum(abs(v) ** 2 for v in raw.values()))
        root_gamma = math.sqrt(self.total_decay_rate_rad_s(e.level))
        return {k: root_gamma * v / norm for k, v in raw.items()}

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


__all__ = ["AtomicStructure", "DressedState"]
