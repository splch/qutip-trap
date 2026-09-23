"""Species data model: a :class:`Species` of fine-structure :class:`Level` and :class:`Transition` records.

Everything hyperfine-resolved is derived (``zeeman.py``, ``dipole.py``, ``raman.py``), never typed in. ``A_hfs_hz`` is
SIGNED (inverted multiplets have A < 0); wavelengths are vacuum wavelengths.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import TYPE_CHECKING, Literal

from qutip_trap.units import (
    C_M_PER_S,
    H_J_S,
    TWO_PI,
)

if TYPE_CHECKING:  # runtime imports live inside the methods to keep the species package import-cycle free
    from qutip_trap.device.model import Field
    from qutip_trap.light.beams import Beam
    from qutip_trap.species.raman import AtomicStructure
    from qutip_trap.species.zeeman import ClockPoint, HyperfineZeeman, ZeemanSpectrum

_LEVEL_NAME = re.compile(r"^(?:\d)?([SPDFGH])(?:\[\d+/2\])?(\d+/2)$")
_LABEL = re.compile(r"^(?P<level>[^ ]+)(?: (?P<rest>.+))?$")


@cache
def level_j(name: str) -> Fraction:
    """The electronic angular momentum J encoded in a level name ("S1/2" -> 1/2, "3D[3/2]1/2" -> 1/2)."""
    m = _LEVEL_NAME.match(name)
    if m is None:
        raise ValueError(f"level name {name!r} is not of the form S1/2, P3/2, D5/2, F7/2 or 3D[3/2]1/2")
    return Fraction(m.group(2))


@cache
def level_l(name: str) -> int:
    """The orbital label of a level name as an integer (S=0 ... H=5); for bracket levels ("3D[3/2]1/2") the outer
    electron's, not an L for the Lande formula."""
    m = _LEVEL_NAME.match(name)
    if m is None:
        raise ValueError(f"level name {name!r} is not a recognised level name")
    return "SPDFGH".index(m.group(1))


@dataclass(frozen=True)
class Level:
    """One fine-structure level. ``energy_hz`` is measured from the ground level."""

    name: str
    energy_hz: float
    lifetime_s: float | None
    A_hfs_hz: float
    B_hfs_hz: float
    g_J: float
    citations: tuple[str, ...]
    lifetime_systematics_s: tuple[tuple[str, float], ...] = ()
    """Named one-sided systematic corrections to ``lifetime_s``, kept apart from its statistical error."""
    untabulated_branching: tuple[tuple[str, float], ...] = ()
    """(description, branching) of decay channels with no ``Transition`` record. ``Species`` requires the
    tabulated E1 branchings out of an E1 upper level plus these to sum to 1, so a deficit is never renormalized."""

    def __post_init__(self) -> None:
        level_j(self.name)  # validates the name
        for what, fraction in self.untabulated_branching:
            if not what:
                raise ValueError(f"{self.name}: an untabulated decay channel must name itself")
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"{self.name}: untabulated branching {what!r} must lie in (0, 1]")
        if self.energy_hz < 0.0:
            raise ValueError(f"{self.name}: energy_hz must be >= 0 (measured from the ground level)")
        if self.lifetime_s is not None and self.lifetime_s <= 0.0:
            raise ValueError(f"{self.name}: lifetime_s must be positive or None")
        if not self.citations:
            raise ValueError(f"{self.name}: a Level must cite its numbers (M0: every number cited)")
        if self.B_hfs_hz != 0.0 and self.J < 1:
            raise ValueError(f"{self.name}: the electric-quadrupole hyperfine constant needs J >= 1")

    @property
    def J(self) -> Fraction:
        return level_j(self.name)

    @property
    def gamma_total_hz(self) -> float | None:
        """Gamma/2pi of the level's total decay, 1/(2 pi tau), or None when the lifetime is not tabulated."""
        if self.lifetime_s is None:
            return None
        return 1.0 / (TWO_PI * self.lifetime_s)


@dataclass(frozen=True)
class Transition:
    """One fine-structure transition."""

    lower: str
    upper: str
    wavelength_vac_m: float
    gamma_hz: float
    """Gamma/2pi of the upper level's TOTAL decay (an ordinary frequency)."""
    branching: float
    """Fine-structure branching of the upper level's decay into ``lower``."""
    multipole: Literal["E1", "E2", "M1"]
    citations: tuple[str, ...]
    quadrupole_element_au: float | None = None
    """A reduced quadrupole element printed by a source, in atomic units (e a0^2), if any."""
    quadrupole_convention: Literal["johnson_1_15", "racah_c2"] | None = None
    """Which normalization ``quadrupole_element_au`` is in; the two differ by a factor 5 in the rate."""

    def __post_init__(self) -> None:
        if self.wavelength_vac_m <= 0.0:
            raise ValueError(f"{self.label}: wavelength_vac_m must be positive")
        if self.gamma_hz <= 0.0:
            raise ValueError(f"{self.label}: gamma_hz must be positive (a total decay rate over 2 pi)")
        if not 0.0 <= self.branching <= 1.0:
            raise ValueError(f"{self.label}: branching must lie in [0, 1]")
        if not self.citations:
            raise ValueError(f"{self.label}: a Transition must cite its numbers")
        if (self.quadrupole_element_au is None) != (self.quadrupole_convention is None):
            raise ValueError(f"{self.label}: a quadrupole element and its convention travel together")
        if self.quadrupole_element_au is not None and self.multipole != "E2":
            raise ValueError(f"{self.label}: only an E2 transition carries a quadrupole element")

    @property
    def label(self) -> str:
        return f"{self.lower}-{self.upper}"

    @property
    def gamma_rad_s(self) -> float:
        """2 pi gamma_hz, the angular total rate of the upper level."""
        return TWO_PI * self.gamma_hz

    @property
    def partial_rate_rad_s(self) -> float:
        """2 pi gamma_hz branching: the angular partial rate into ``lower``, what I_sat takes."""
        return TWO_PI * self.gamma_hz * self.branching

    @property
    def i_sat_w_m2(self) -> float:
        """I_sat = pi h c Gamma_partial / (3 lambda_vac^3) in W/m^2 (Gamma_partial angular), for unit relative strength."""
        return math.pi * H_J_S * C_M_PER_S * self.partial_rate_rad_s / (3.0 * self.wavelength_vac_m**3)

    @property
    def frequency_hz(self) -> float:
        return C_M_PER_S / self.wavelength_vac_m


def parse_state_label(label: str) -> tuple[str, str]:
    """Split a state label such as "S1/2 F=0 mF=0" or "S1/2 mJ=-1/2" into (level name, quantum numbers)."""
    m = _LABEL.match(label)
    if m is None:
        raise ValueError(f"malformed state label {label!r}")
    return m.group("level"), m.group("rest") or ""


def parse_transition_label(label: str) -> tuple[str, str]:
    """Split "S1/2-P1/2" into (lower, upper); both must be level names (the '-' after ']' is unambiguous)."""
    parts = label.split("-")
    if len(parts) != 2:
        raise ValueError(f"malformed transition label {label!r}; expected 'lower-upper'")
    return parts[0], parts[1]


@dataclass(frozen=True)
class Species:
    """Immutable atomic data for one isotope."""

    name: str
    mass_u: float
    """ION mass in u: the cited atomic mass minus one electron mass."""
    nuclear_spin: float
    mu_I_nuclear_magnetons: float
    levels: tuple[Level, ...]
    transitions: tuple[Transition, ...]
    qubit: tuple[str, str]
    """State labels, "S1/2 F=0 mF=0" or "S1/2 mJ=-1/2"."""
    cycling: str
    repumps: tuple[str, ...]
    shelving: str | None

    def __post_init__(self) -> None:
        if self.mass_u <= 0.0:
            raise ValueError("mass_u must be positive")
        if self.nuclear_spin < 0.0 or (2.0 * self.nuclear_spin) != int(round(2.0 * self.nuclear_spin)):
            raise ValueError("nuclear_spin must be a non-negative integer or half-integer")
        if self.nuclear_spin == 0.0 and self.mu_I_nuclear_magnetons != 0.0:
            raise ValueError("a spin-zero nucleus has no magnetic moment")
        names = [lv.name for lv in self.levels]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate level names in {self.name}: {names}")
        if not any(lv.energy_hz == 0.0 for lv in self.levels):
            raise ValueError(f"{self.name}: no level at energy 0 (the ground level)")
        for lv in self.levels:
            if self.nuclear_spin < 1.0 and lv.B_hfs_hz != 0.0:
                raise ValueError(f"{self.name} {lv.name}: B_hfs needs I >= 1")
            if self.nuclear_spin == 0.0 and lv.A_hfs_hz != 0.0:
                raise ValueError(f"{self.name} {lv.name}: a spin-zero nucleus has no hyperfine structure")
        by_name = {lv.name: lv for lv in self.levels}
        upper_branching: dict[str, float] = {}
        e1_branching: dict[str, float] = {}
        e1_rates: dict[str, dict[str, float]] = {}
        for tr in self.transitions:
            for end in (tr.lower, tr.upper):
                if end not in by_name:
                    raise ValueError(f"{self.name}: transition {tr.label} names unknown level {end!r}")
            lo, up = by_name[tr.lower], by_name[tr.upper]
            if up.energy_hz <= lo.energy_hz:
                raise ValueError(f"{self.name}: {tr.label} must have upper above lower")
            lam = C_M_PER_S / (up.energy_hz - lo.energy_hz)
            if not math.isclose(lam, tr.wavelength_vac_m, rel_tol=1e-6):
                raise ValueError(
                    f"{self.name}: {tr.label} wavelength {tr.wavelength_vac_m:.6e} m disagrees with the level "
                    f"energies ({lam:.6e} m)"
                )
            if up.lifetime_s is not None and not math.isclose(
                tr.gamma_hz, 1.0 / (TWO_PI * up.lifetime_s), rel_tol=1e-6
            ):
                raise ValueError(
                    f"{self.name}: {tr.label} gamma_hz {tr.gamma_hz:.6e} disagrees with 1/(2 pi tau) of {up.name}"
                )
            upper_branching[tr.upper] = upper_branching.get(tr.upper, 0.0) + tr.branching
            if tr.multipole == "E1":
                e1_branching[tr.upper] = e1_branching.get(tr.upper, 0.0) + tr.branching
                e1_rates.setdefault(tr.upper, {})[tr.label] = tr.gamma_hz
        for up_name, total in upper_branching.items():
            if total > 1.0 + 1e-9:
                raise ValueError(f"{self.name}: branchings out of {up_name} sum to {total} > 1")
        # the scattering sums read branchings as partial-rate fractions, so a deficit must be declared
        for up_name, total in e1_branching.items():
            declared = sum(f for _what, f in by_name[up_name].untabulated_branching)
            if not math.isclose(total + declared, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"{self.name}: the E1 branchings out of {up_name} sum to {total!r} and the declared "
                    f"untabulated channels to {declared!r}, not 1 within 1e-9; either tabulate the missing "
                    f"transitions or declare them in {up_name}.untabulated_branching (Section 4.5.5)"
                )
        # gamma_hz is the upper level's TOTAL rate, so every E1 line out of one level must carry the same value
        for up_name, rates in e1_rates.items():
            first = next(iter(rates.values()))
            if any(not math.isclose(g, first, rel_tol=1e-9) for g in rates.values()):
                raise ValueError(
                    f"{self.name}: the E1 transitions out of {up_name} disagree on its total decay rate "
                    f"gamma_hz: {rates}"
                )
        for lab in self.qubit:
            lv_name, _ = parse_state_label(lab)
            if lv_name not in by_name:
                raise ValueError(f"{self.name}: qubit label {lab!r} names unknown level {lv_name!r}")
        labels = [self.cycling, *self.repumps] + ([self.shelving] if self.shelving is not None else [])
        tabulated = {tr.label for tr in self.transitions}
        for lab in labels:
            lo_name, up_name = parse_transition_label(lab)
            for end in (lo_name, up_name):
                if end not in by_name:
                    raise ValueError(f"{self.name}: transition label {lab!r} names unknown level {end!r}")
            # a designated line must be a tabulated Transition, not just two known level names
            if lab not in tabulated:
                raise ValueError(
                    f"{self.name}: the designated transition {lab!r} has no Transition record; add it to the "
                    f"table or move it into MISSING (tabulated: {sorted(tabulated)})"
                )

    # ---- lookups -------------------------------------------------------------------------------------

    def level(self, name: str) -> Level:
        for lv in self.levels:
            if lv.name == name:
                return lv
        raise KeyError(f"{self.name} has no level {name!r}")

    def transition(self, label: str) -> Transition:
        for tr in self.transitions:
            if tr.label == label:
                return tr
        raise KeyError(f"{self.name} has no tabulated transition {label!r}")

    # ---- derived quantities --------------------------------------------------------------------------

    def _hyperfine_zeeman(self, level: str) -> HyperfineZeeman:
        from qutip_trap.species.zeeman import hyperfine_zeeman

        return hyperfine_zeeman(self.level(level), self.nuclear_spin, self.mu_I_nuclear_magnetons)

    def _structure(self, field: Field) -> AtomicStructure:
        from qutip_trap.species.raman import structure_at

        return structure_at(self, field.B_gauss, field.direction)

    def zeeman_spectrum(self, level: str, B_gauss: float) -> ZeemanSpectrum:
        """The hyperfine-Zeeman spectrum of ``level`` at ``B_gauss`` with adiabatic labels and field derivatives."""
        return self._hyperfine_zeeman(level).spectrum(B_gauss)

    def transition_frequency_hz(self, a: str, b: str, B_gauss: float) -> tuple[float, float, float]:
        """(nu_b - nu_a in Hz, d nu/dB in Hz/G, d^2 nu/dB^2 in Hz/G^2) at B; taylor_c2 is HALF the third entry."""
        from qutip_trap.species.zeeman import transition_sensitivity

        la, ra = parse_state_label(a)
        lb, rb = parse_state_label(b)
        s = transition_sensitivity(self._hyperfine_zeeman(la), ra, self._hyperfine_zeeman(lb), rb, B_gauss)
        return s.frequency_hz, s.dnu_dB_hz_per_g, s.d2nu_dB2_hz_per_g2

    def clock_points(self, a: str, b: str, B_lo_gauss: float, B_hi_gauss: float) -> tuple[ClockPoint, ...]:
        """Fields in [B_lo, B_hi] where d nu/dB = 0 for the pair, with |nu| and both curvature conventions."""
        from qutip_trap.species.zeeman import clock_points

        la, ra = parse_state_label(a)
        lb, rb = parse_state_label(b)
        return clock_points(
            self._hyperfine_zeeman(la), ra, self._hyperfine_zeeman(lb), rb, B_lo_gauss, B_hi_gauss
        )

    def dipole_element(self, a: str, b: str, q: int, field: Field) -> complex:
        """<b|d_q|a> in C m in the field-dressed eigenbasis (T_q the spherical tensor component, m_b = m_a + q)."""
        st = self._structure(field)
        return st.dipole_element_c_m(st.state(b), st.state(a), q)

    def rabi_frequency_hz(self, a: str, b: str, beam: Beam, field: Field) -> complex:
        """Omega_{ba}/2pi for the E1 (or, for I = 0, E2) transition a -> b driven by ``beam`` at the beam's peak intensity.

        E1: E_0 sum_q eps_q <b|T_q|a>/hbar in the (hbar Omega/2) convention, complex. E2: the real quadrupole scalar
        (e E_0 k/(2 hbar)) |<S||r^2 C^(2)||D>| |Lambda| |g^(q)|; ``NotImplementedError`` when I != 0."""
        from qutip_trap.species.quadrupole import rabi_frequency_e2_rad_s, reduced_element_from_lifetime_m2
        from qutip_trap.species.zeeman import parse_quantum_numbers
        from qutip_trap.units import TWO_PI

        la, ra = parse_state_label(a)
        lb, rb = parse_state_label(b)
        e2 = next(
            (t for t in self.transitions if t.multipole == "E2" and {t.lower, t.upper} == {la, lb}), None
        )
        if e2 is not None:
            if self.nuclear_spin != 0.0:
                raise NotImplementedError(
                    "hyperfine-resolved E2 couplings are not specified (Section 4.5.7 treats I = 0)"
                )
            st = self._structure(field)
            lower, upper = (a, b) if e2.lower == la else (b, a)
            m = parse_quantum_numbers(parse_state_label(lower)[1])["mJ"]
            mp = parse_quantum_numbers(parse_state_label(upper)[1])["mJ"]
            red = reduced_element_from_lifetime_m2(
                e2.wavelength_vac_m, e2.partial_rate_rad_s, level_j(e2.upper)
            )
            e0 = st.field_amplitude(beam, None)
            return complex(
                rabi_frequency_e2_rad_s(
                    e0,
                    beam.wavelength_m,
                    red,
                    level_j(e2.lower),
                    m,
                    level_j(e2.upper),
                    mp,
                    beam.polarization,
                    beam.k_hat,
                    field.direction,
                )
                / TWO_PI
            )
        st = self._structure(field)
        return complex(st.single_photon_coupling_rad_s(st.state(a), st.state(b), beam) / TWO_PI)

    def raman_coupling_hz(self, g1: str, g2: str, beam1: Beam, beam2: Beam, field: Field) -> complex:
        """Omega_{g1 g2}/2pi = sum_e conj(Omega^{(2)}_{e g2}) Omega^{(1)}_{e g1}/(2 Delta_e^{(1)}) / 2pi."""
        from qutip_trap.units import TWO_PI

        st = self._structure(field)
        return complex(st.raman_coupling_rad_s(st.state(g1), st.state(g2), beam1, beam2) / TWO_PI)

    def light_shift_hz(self, g: str, beam: Beam, field: Field) -> float:
        """delta_g/2pi = sum_e |Omega_{eg}|^2/(4 Delta_e) / 2pi for one beam (red light lowers the level)."""
        from qutip_trap.units import TWO_PI

        st = self._structure(field)
        return st.light_shift_rad_s(st.state(g), (beam,)) / TWO_PI

    def scattering_rates_hz(self, a: str, beam: Beam, field: Field) -> dict[str, float]:
        """Scattering rates in s^-1 (not over 2 pi, despite ``_hz``) by final-state label; ``a`` itself is Rayleigh."""
        st = self._structure(field)
        return st.scattering_rates(st.state(a), beam)

    def rayleigh_dephasing_hz(self, beam: Beam, field: Field) -> float:
        """Gamma_el of the qubit pair in s^-1 (Uys et al. 2010); the collapse operator is (1/2) sqrt(Gamma_el) sigma_z."""
        st = self._structure(field)
        up, down = self.qubit[1], self.qubit[0]
        return st.rayleigh_dephasing_rate(st.state(up), st.state(down), beam)


__all__ = [
    "Level",
    "Species",
    "Transition",
    "level_j",
    "level_l",
    "parse_state_label",
    "parse_transition_label",
]
