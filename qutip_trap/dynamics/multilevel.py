"""The multi-level mode of the ONE Hamiltonian builder (PLAN.md Sections 3.2, 4.2.8, 4.5, 5.7, 8.1; milestone M3a).

Cooling, optical pumping and detection share one Hamiltonian and one set of collapse operators built here from the
level structure of Section 4.5 and re-exported by :mod:`qutip_trap.dynamics.hamiltonian`:

    H/hbar = sum_p (E_p - f_{level(p)}) |p><p|
           + sum_beams sum_{(L,U) addressed} sum_{p in L, q in U} (1/2) Omega^{(b)}_{qp} e^{-i delta_b t} |q><p| (x) D(i eta_b) + h.c.,

with E_p the field-dressed energies of Section 4.5.1 (rad/s), f the frame frequency assigned to each fine-structure
level, Omega^{(b)}_{qp} = E_0 sum_q eps_q <q|T_q|p>/hbar the polarization-resolved Rabi frequency of Section 4.5.2 in
the (hbar Omega/2) convention, D(i eta_b) the exact displacement operator of the beam's projection on the one mode
(Section 5.1.1; identity without a mode) and delta_b the beam's residual against the frame.

Rotating frame (Section 4.2.8): frame frequencies are assigned manifold by manifold along the beams, starting from
the lowest included level at f = 0, so that every beam's detuning from every transition it drives sits on the
diagonal; a beam whose frequency is inconsistent with two already assigned manifolds (two beams on one transition at
different frequencies, or a closed loop of beams whose frequencies do not sum consistently) leaves a nonzero
residual delta_b, the Hamiltonian is then explicitly time periodic at the beat and the build says so
(``FrameAssignment.static`` is False); levels connected only by decay take their frame from the transition
frequency, which leaves the dissipator invariant. A level a beam is not near (|omega_b - omega_LU| >= omega_LU/2)
is not addressed by it.

Dissipator (Sections 4.5.2, 8.1, 4.2.8): one collapse operator per fine-structure decay line (L, U) and emitted
polarization index q, C_q = sqrt(g_LU) T_q^- with g_LU = omega_LU^3/(3 pi eps0 hbar c^3) and T_q^- the lower<-upper
block of the dipole operator in C m, so that sum_q C_q^dagger C_q = Gamma_{U->L} P_U by the Wigner-Eckart sum rule,
the excited-to-ground feeding term carries the product of two matrix elements that transfers excited coherence into
ground coherence, and there is never one operator per excited sublevel. With a mode and recoil on, each C_q is
resolved over emission directions with the kernel of :mod:`qutip_trap.light.recoil`. Population that would decay
into a level or sublevel the build excludes follows the ``leak`` policy: ``include`` (default) pulls every tabulated
decay target in, ``sink`` routes it to a bookkeeping state so the trace is preserved and the leak rate is measurable,
``renormalize`` folds it back into the included branches (the declared "instantaneous repump" approximation).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import qutip as qt

from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.light.beams import Beam
from qutip_trap.light.recoil import (
    RecoilMode,
    angular_factor,
    marginal_quadrature,
    minimal_quadrature,
    recoil_lamb_dicke,
    vector_channels,
)
from qutip_trap.species.polarization import spherical_components
from qutip_trap.species.raman import AtomicStructure, DressedState
from qutip_trap.units import C_M_PER_S, EPSILON_0_F_PER_M, HBAR_J_S, TWO_PI

M3A = "milestone M3a (dynamics/multilevel.py, PLAN.md Section 4.2.8)"

LeakPolicy = Literal["include", "sink", "renormalize"]
SINK = "sink"
"""Label of the bookkeeping state that receives population leaving the included manifold under ``leak='sink'``."""


# ---- inputs ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeSpec:
    """One motional mode for the level-C solve: frequency, mass, laboratory axis and truncation (Section 5.1)."""

    omega_rad_s: float
    mass_kg: float
    axis: tuple[float, float, float]
    d: int
    """Fock levels."""
    expected_n_max: int = 0
    """Top of the populated range the truncation must cover (Section 5.1.1 margin bookkeeping)."""

    def __post_init__(self) -> None:
        if self.omega_rad_s <= 0.0 or self.mass_kg <= 0.0:
            raise ValueError("mode frequency and mass are positive")
        if self.d < 2:
            raise ValueError("a mode needs at least two Fock levels")
        n = math.sqrt(sum(a * a for a in self.axis))
        if not math.isclose(n, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"ModeSpec.axis must be a unit vector, got norm {n}")
        if not 0 <= self.expected_n_max <= self.d - 1:
            raise ValueError("expected_n_max lies in [0, d - 1]")

    @property
    def x0_m(self) -> float:
        """sqrt(hbar/(2 m omega)) (Section 13, "Ladder operators")."""
        return math.sqrt(HBAR_J_S / (2.0 * self.mass_kg * self.omega_rad_s))

    def eta(self, k_vector_rad_per_m: Sequence[float] | np.ndarray) -> float:
        """(k . e_m) x0: the Lamb-Dicke parameter of one beam's wavevector on this mode (single ion, b = 1)."""
        return float(np.dot(np.asarray(k_vector_rad_per_m, dtype=float), np.asarray(self.axis))) * self.x0_m


@dataclass(frozen=True)
class MultiLevelOptions:
    """Options of the multi-level Hamiltonian builder (Sections 4.2.8, 8.1): the leakage policy, the recoil quadrature, the
    frame-residual tolerance (rad/s), the addressing window (a fraction of the transition frequency), the optical phase of
    each beam at the ion (radians) and each beam's linewidth (angular FWHM, rad/s) as a phase-diffusion collapse operator."""

    leak: LeakPolicy = "include"
    recoil: RecoilMode = "off"
    recoil_nodes: int = 16
    """Gauss-Legendre nodes of the one-dimensional marginal quadrature (``recoil='marginal'``)."""
    recoil_grid: tuple[int, int] = (6, 8)
    """(n_theta, n_phi) of the direction quadrature (``recoil='vector'``)."""
    frame_tolerance_rad_s: float = 1e3
    """Residuals below this are frame round-off (a beam frequency of 5e15 rad/s is a double to about 1 rad/s), not beats."""
    address_window: float = 0.5
    """A beam addresses a transition when |omega_b - omega_LU| < address_window x omega_LU."""
    beam_phases_rad: tuple[float, ...] = ()
    """Optical phase of each beam at the ion (default 0); only relative phases between beams on one transition matter."""
    laser_linewidth_rad_s: tuple[float, ...] = ()
    """delta omega_L of each beam (FWHM, angular): the phase-diffusion rate its light puts on the optical coherences.

    Empty (the default) means an ideal monochromatic laser. Section 8.1 requires "-(gamma/2 + delta omega_L/2) on
    optical coherences": the Lindblad form supplies -gamma/2 by itself, and a nonzero entry here adds the missing
    -delta omega_L/2 as one phase-diffusion collapse operator per beam,

        C_b = sqrt(delta omega_L,b / 4) (P_upper(b) - P_lower(b)),

    the projector difference on the manifolds beam b connects (Berkeland and Boshier, Phys. Rev. A 65, 033413 (2002)
    Eq. 12: a Lorentzian laser spectrum of full width delta omega_L is a Wiener phase whose Lindblad generator damps
    the driven transition's coherence at delta omega_L/2 and leaves every population and every within-manifold
    coherence untouched). This is the constant-rate Lorentzian of Section 8.1, not the laser's spectrum as a
    spectral density, which Section 12 grants may be omitted."""

    def __post_init__(self) -> None:
        if self.recoil_nodes < 3:
            raise ValueError("at least three quadrature nodes (Section 4.2.8)")
        if not 0.0 < self.address_window < 1.0:
            raise ValueError("address_window is a fraction of the transition frequency in (0, 1)")
        if self.frame_tolerance_rad_s <= 0.0:
            raise ValueError("frame_tolerance_rad_s is positive")
        if any(w < 0.0 for w in self.laser_linewidth_rad_s):
            raise ValueError("a laser linewidth is non-negative")


# ---- records --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameEdge:
    """One (beam, transition) coupling and its residual against the assigned frames."""

    beam: int
    lower: str
    upper: str
    omega_beam_rad_s: float
    residual_rad_s: float
    sets_frame: bool
    """True for the beam that fixed the upper manifold's frame along this edge."""


@dataclass(frozen=True)
class FrameAssignment:
    """Frame frequency per included level, every (beam, transition) edge and the beats the residuals leave (Section 4.2.8)."""

    frame_rad_s: dict[str, float]
    edges: tuple[FrameEdge, ...]
    beats_rad_s: tuple[float, ...]
    """Distinct nonzero residuals (magnitudes); empty when the frame is consistent."""
    period_s: float | None
    """2 pi over the common beat when the beats are commensurate; None if static or incommensurate."""

    @property
    def static(self) -> bool:
        return not self.beats_rad_s

    def inconsistent_edges(self) -> tuple[FrameEdge, ...]:
        return tuple(e for e in self.edges if e.residual_rad_s != 0.0)


@dataclass(frozen=True)
class CouplingRecord:
    """What one beam does to one sublevel pair: Rabi frequency, detuning, Lamb-Dicke parameter."""

    beam: int
    lower: str
    upper: str
    omega_rad_s: complex
    """Omega_{qp}, complex, (hbar Omega/2) convention."""
    detuning_rad_s: float
    """omega_b - (E_q - E_p) with the dressed energies (plan sign: red negative)."""
    eta: float
    """(k_b . e_m) x0 on the mode; 0 without a mode."""
    residual_rad_s: float


@dataclass(frozen=True)
class EmissionChannel:
    """One decay line of the build: which upper level decays into which lower level (or the sink) and at what rate."""

    lower: str
    upper: str
    rate_rad_s: float
    """Gamma_{U->L}: the tabulated partial rate (s^-1); for a renormalized line the scaled mean rate."""
    wavenumber_rad_per_m: float
    n_operators: int
    kind: Literal["decay", "sink", "renormalized"]
    eta_em: float
    """k_em x0 on the mode; 0 without a mode."""
    alpha: dict[int, float] = field(default_factory=dict)
    """alpha_q of the mode axis per polarization index, when recoil is on."""
    operator_slice: tuple[int, int] = (0, 0)
    """[start, stop) of this channel's operators in ``MultiLevelBuild.c_ops``."""
    operator_qs: tuple[int, ...] = ()
    """The polarization index q of each operator in the slice (9 for a vector-form channel, which mixes q)."""


@dataclass(frozen=True)
class MultiLevelBuild:
    """The Hamiltonian, collapse operators and bookkeeping of one multi-level build (Section 5.7 diagnostics)."""

    H: qt.Qobj | qt.QobjEvo
    c_ops: tuple[qt.Qobj, ...]
    labels: tuple[str, ...]
    """Internal basis labels in order (full state labels, plus ``SINK`` last when present)."""
    levels: tuple[str, ...]
    frame: FrameAssignment
    couplings: tuple[CouplingRecord, ...]
    channels: tuple[EmissionChannel, ...]
    level_rates_rad_s: dict[str, float]
    """Total decay rate of every included level that decays (the lifetime, or the sum of tabulated partial rates)."""
    approximations: tuple[str, ...]
    space: HilbertSpace | None
    mode: ModeSpec | None
    options: MultiLevelOptions
    n_beams: int
    dephasing_slice: tuple[int, int] = (0, 0)
    """[start, stop) of the laser-linewidth phase-diffusion operators in ``c_ops`` (empty when no linewidth is given).

    They are Lindblad operators like the decay ones, but they carry no photon and no recoil, so the decay sum rule of
    Section 4.2.8 excludes them (``decay_sum_rule_residual``) and no ``EmissionChannel`` claims them."""

    # ---- bookkeeping ---------------------------------------------------------------------------------------

    @property
    def n_internal(self) -> int:
        return len(self.labels)

    @property
    def static(self) -> bool:
        return self.frame.static and not isinstance(self.H, qt.QobjEvo)

    @property
    def dimension(self) -> int:
        return self.n_internal if self.space is None else self.space.dimension

    def index(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(f"{label!r} is not a state of this build; states: {self.labels}") from None

    @staticmethod
    def level_of(label: str) -> str:
        return label.split(" ", 1)[0]

    def states_of(self, level: str) -> tuple[str, ...]:
        return tuple(lab for lab in self.labels if lab != SINK and self.level_of(lab) == level)

    # ---- operators (joint space when a mode is present) ----------------------------------------------------

    def internal(self, op: qt.Qobj) -> qt.Qobj:
        """Embed an operator on the internal factor into the build's space (identity on the mode)."""
        if op.shape != (self.n_internal, self.n_internal):
            raise ValueError("operator does not act on the internal factor")
        if self.space is None:
            return op
        return self.space.embed(op, 0)

    def projector(self, label: str) -> qt.Qobj:
        return self.internal(qt.basis(self.n_internal, self.index(label)).proj())

    def manifold_projector(self, labels: Sequence[str]) -> qt.Qobj:
        p = qt.qzero(self.n_internal)
        for lab in labels:
            p = p + qt.basis(self.n_internal, self.index(lab)).proj()
        return self.internal(p)

    def level_projector(self, level: str) -> qt.Qobj:
        return self.manifold_projector(self.states_of(level))

    def number(self) -> qt.Qobj:
        if self.space is None:
            raise KeyError("this build has no motional mode")
        return self.space.number(0)

    def liouvillian(self) -> qt.Qobj | qt.QobjEvo:
        return qt.liouvillian(self.H, list(self.c_ops))

    def internal_state(self, label: str) -> qt.Qobj:
        """|label> on the internal factor (a ket)."""
        return qt.basis(self.n_internal, self.index(label))

    def populations(self, rho: qt.Qobj) -> dict[str, float]:
        """Population of every internal state (the mode traced out)."""
        r = rho if rho.isoper else qt.ket2dm(rho)
        if self.space is not None:
            r = self.space.marginal(r, (0,))
        diag = np.real(np.diag(np.asarray(r.full())))
        return {lab: float(diag[k]) for k, lab in enumerate(self.labels)}

    def level_populations(self, rho: qt.Qobj) -> dict[str, float]:
        pops = self.populations(rho)
        out: dict[str, float] = {}
        for lab, p in pops.items():
            key = SINK if lab == SINK else self.level_of(lab)
            out[key] = out.get(key, 0.0) + p
        return out


# ---- coefficients (module-level functions: they pickle, Section 5.3) ------------------------------------------------------


def _beat_coefficient(
    t: float,
    residual: float,
    phase: float,
    mod_kind: str,
    mod_omega: float,
    mod_depth: float,
    q_index: int,
    **_: object,
) -> complex:
    c = 0.5 * np.exp(1j * (phase - residual * t))
    if mod_kind == "aom":
        # the two circular components counter-shifted: a polarization rotating about B at the modulation frequency
        c = c * np.exp(-1j * q_index * mod_omega * t)
    elif mod_kind in ("pem", "eom"):
        # phase modulation of the circular components against the pi component, index mod_depth
        c = c * np.exp(1j * q_index * mod_depth * math.sin(mod_omega * t))
    return complex(c)


def _beat_coefficient_conj(t: float, **kwargs: Any) -> complex:
    return complex(np.conj(_beat_coefficient(t, **kwargs)))


# ---- frame assignment -----------------------------------------------------------------------------------------------------


def _beam_omega(beam: Beam) -> float:
    return TWO_PI * C_M_PER_S / beam.wavelength_m


def _transition_omega(st: AtomicStructure, lower: str, upper: str) -> float:
    tr = st.e1[(lower, upper)]
    return TWO_PI * C_M_PER_S / tr.wavelength_vac_m


def addressed_transitions(
    st: AtomicStructure, beam: Beam, levels: Sequence[str], window: float
) -> list[tuple[str, str]]:
    """The tabulated E1 transitions among ``levels`` that ``beam`` is within ``window`` (fractional) of."""
    w_b = _beam_omega(beam)
    out: list[tuple[str, str]] = []
    for lo, up in st.e1:
        if lo in levels and up in levels:
            w = _transition_omega(st, lo, up)
            if abs(w_b - w) < window * w:
                out.append((lo, up))
    return out


def _common_period(beats: Sequence[float]) -> float | None:
    """2 pi / gcd of the beats when they are commensurate (rational ratios to 1e-9), else None."""
    if not beats:
        return None
    g = min(beats)
    for b in beats:
        ratio = Fraction(b / g).limit_denominator(1000)
        if abs(float(ratio) - b / g) > 1e-9:
            return None
        g = g / ratio.denominator
    return TWO_PI / g


def assign_frames(
    st: AtomicStructure,
    beams: Sequence[Beam],
    levels: Sequence[str],
    *,
    window: float = 0.5,
    tolerance_rad_s: float = 1e3,
) -> FrameAssignment:
    """Frame frequencies manifold by manifold along the beams, with the inconsistency check of Section 4.2.8.

    Breadth-first from the lowest included level (f = 0): the first beam addressing a transition out of an assigned
    level fixes the other level's frame to f + omega_b (or f - omega_b downward); every later beam on an edge whose
    ends are both assigned is checked, and a residual above ``tolerance_rad_s`` is a beat. Levels no beam reaches are
    attached through their tabulated transitions (frame difference = transition frequency), which leaves the
    dissipator invariant, and an isolated level takes its own energy as frame.
    """
    if not levels:
        raise ValueError("no levels to assign frames to")
    energies = {lv: TWO_PI * st.species.level(lv).energy_hz for lv in levels}
    root = min(levels, key=lambda lv: energies[lv])
    frame: dict[str, float] = {root: 0.0}
    edges_all: list[tuple[int, str, str, float]] = []
    for b, beam in enumerate(beams):
        for lo, up in addressed_transitions(st, beam, levels, window):
            edges_all.append((b, lo, up, _beam_omega(beam)))
    records: list[FrameEdge] = []
    used: set[int] = set()
    changed = True
    while changed:
        changed = False
        for k, (b, lo, up, w_b) in enumerate(edges_all):
            if k in used:
                continue
            if lo in frame and up not in frame:
                frame[up] = frame[lo] + w_b
            elif up in frame and lo not in frame:
                frame[lo] = frame[up] - w_b
            else:
                continue
            records.append(FrameEdge(b, lo, up, w_b, 0.0, True))
            used.add(k)
            changed = True
    changed = True
    while changed:
        changed = False
        for lo, up in st.e1:
            if lo in levels and up in levels:
                if lo in frame and up not in frame:
                    frame[up] = frame[lo] + _transition_omega(st, lo, up)
                    changed = True
                elif up in frame and lo not in frame:
                    frame[lo] = frame[up] - _transition_omega(st, lo, up)
                    changed = True
    for lv in levels:
        if lv not in frame:
            frame[lv] = energies[lv]
    beats: list[float] = []
    for k, (b, lo, up, w_b) in enumerate(edges_all):
        if k in used:
            continue
        residual = w_b - (frame[up] - frame[lo])
        if abs(residual) < tolerance_rad_s:
            residual = 0.0
        else:
            beats.append(abs(residual))
        records.append(FrameEdge(b, lo, up, w_b, residual, False))
    distinct = tuple(sorted({round(x, 6) for x in beats}))
    return FrameAssignment(frame, tuple(records), distinct, _common_period(distinct))


# ---- the build ------------------------------------------------------------------------------------------------------------


def _polarization_components(beam: Beam, b_hat: np.ndarray) -> np.ndarray:
    if beam.polarization_amplitudes is not None:
        return np.asarray(beam.polarization_amplitudes, dtype=complex)
    return spherical_components(beam.polarization, b_hat)


def _g_lu(st: AtomicStructure, lower: str, upper: str) -> float:
    """omega^3/(3 pi eps0 hbar c^3): |<p|d|q>|^2 g is the decay rate of the pair (Section 4.5.2)."""
    w = _transition_omega(st, lower, upper)
    return float(w**3 / (3.0 * math.pi * EPSILON_0_F_PER_M * HBAR_J_S * C_M_PER_S**3))


def _decay_block(
    st: AtomicStructure, lowers: Sequence[DressedState], uppers: Sequence[DressedState], q: int
) -> np.ndarray:
    """<p|T_q|q'> in C m for p in lowers (rows) and q' in uppers (columns)."""
    block = np.zeros((len(lowers), len(uppers)), dtype=complex)
    for i, p in enumerate(lowers):
        for j, e in enumerate(uppers):
            block[i, j] = st.dipole_element_c_m(p, e, q)
    return block


def _place(
    matrix: np.ndarray, rows: Sequence[int], cols: Sequence[int], scale: np.ndarray, n: int
) -> qt.Qobj:
    """The internal operator with ``matrix`` (scaled per column) in the block rows x cols.

    ``matrix`` must already carry its rate prefactor (units sqrt(1/s)): QuTiP's auto-tidyup zeroes elements below
    1e-12 when a Qobj is created, and raw dipole elements in C m are 1e-29.
    """
    full = np.zeros((n, n), dtype=complex)
    full[np.ix_(rows, cols)] = matrix * scale[np.newaxis, :]
    return qt.Qobj(full, dims=[[n], [n]])


class _Embedder:
    """Operators on the build's space: the internal factor alone or tensored with a mode operator."""

    def __init__(self, space: HilbertSpace | None) -> None:
        self.space = space

    def __call__(self, op_int: qt.Qobj, mode_op: qt.Qobj | None = None) -> qt.Qobj:
        if self.space is None:
            return op_int
        if mode_op is None:
            return self.space.embed(op_int, 0)
        return self.space.embed_many({0: op_int, 1: mode_op})


def build_multilevel(
    structure: AtomicStructure,
    beams: Sequence[Beam],
    *,
    levels: Sequence[str] | None = None,
    states: Sequence[str] | None = None,
    position_m: Sequence[float] | None = None,
    mode: ModeSpec | None = None,
    options: MultiLevelOptions | None = None,
) -> MultiLevelBuild:
    """Assemble H and the collapse operators of the internal levels (times one mode) for the given beams.

    ``levels``: fine-structure level names to include (default: every level a beam addresses; with ``leak='include'``
    every tabulated decay target of an included upper level is added too). ``states``: optional subset of sublevel
    labels ("S1/2 F=1 mF=0", ...). ``position_m``: the ion's position for the beam intensities (default: origin).
    """
    st = structure
    opts = options or MultiLevelOptions()
    b_hat = np.asarray(st.b_hat, dtype=float)
    approximations: list[str] = []
    if mode is None and opts.recoil != "off":
        raise ValueError("recoil needs a motional mode; use recoil='off' for an internal-only build")
    if opts.beam_phases_rad and len(opts.beam_phases_rad) != len(beams):
        raise ValueError("beam_phases_rad has one entry per beam")
    phases = list(opts.beam_phases_rad) if opts.beam_phases_rad else [0.0] * len(beams)
    pos = np.zeros(3) if position_m is None else np.asarray(position_m, dtype=float)

    # ---- level set ----
    all_levels = [lv.name for lv in st.species.levels]
    if levels is None:
        chosen: list[str] = []
        for beam in beams:
            for lo, up in addressed_transitions(st, beam, all_levels, opts.address_window):
                for lv in (lo, up):
                    if lv not in chosen:
                        chosen.append(lv)
        if not chosen:
            raise ValueError("no beam addresses a tabulated E1 transition; give `levels` explicitly")
    else:
        chosen = list(dict.fromkeys(levels))
        for lv in chosen:
            st.species.level(lv)
    if opts.leak == "include":
        changed = True
        while changed:
            changed = False
            for lo, up in list(st.e1):
                if up in chosen and lo not in chosen:
                    chosen.append(lo)
                    changed = True
    level_order = sorted(chosen, key=lambda lv: st.species.level(lv).energy_hz)

    # ---- internal basis ----
    dressed: list[DressedState] = []
    for lv in level_order:
        for s in st.states_of(lv):
            if states is None or s.full_label in states:
                dressed.append(s)
    if states is not None:
        present = {d.full_label for d in dressed}
        missing = [s for s in states if s not in present]
        if missing:
            raise KeyError(f"states not in the included levels: {missing}")
    if not dressed:
        raise ValueError("the build has no states")
    labels = [d.full_label for d in dressed]
    if opts.leak == "sink":
        labels.append(SINK)
    n_int = len(labels)
    idx = {lab: k for k, lab in enumerate(labels)}
    dressed_by_level: dict[str, list[DressedState]] = {
        lv: [d for d in dressed if d.level == lv] for lv in level_order
    }

    # ---- frame ----
    frame = assign_frames(
        st, beams, level_order, window=opts.address_window, tolerance_rad_s=opts.frame_tolerance_rad_s
    )

    # ---- space ----
    space: HilbertSpace | None = None
    etas_by_beam: dict[int, float] = {}
    eta_em_by_line: dict[tuple[str, str], float] = {}
    cos_chi = 0.0
    mode_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    if mode is not None:
        mode_axis = mode.axis
        cos_chi = float(np.dot(np.asarray(mode.axis), b_hat))
        for b, beam in enumerate(beams):
            etas_by_beam[b] = mode.eta(tuple(float(x) for x in beam.k_vector()))
        for lo, up in st.e1:
            if up in level_order:
                eta_em_by_line[(lo, up)] = recoil_lamb_dicke(
                    _transition_omega(st, lo, up) / C_M_PER_S, mode.x0_m
                )
        eta_max = max([abs(e) for e in etas_by_beam.values()] + list(eta_em_by_line.values()) + [0.0])
        space = HilbertSpace(
            ion_dims=(n_int,),
            resolved=(ModeTruncation(0, mode.d, (0, mode.expected_n_max), eta_max * (1.0 + 1e-9)),),
            enr_group=None,
            frozen=(),
        )
    embed = _Embedder(space)

    # ---- diagonal ----
    diag = np.zeros(n_int)
    for k, d in enumerate(dressed):
        diag[k] = TWO_PI * d.energy_hz - frame.frame_rad_s[d.level]
    h_static = embed(qt.Qobj(np.diag(diag), dims=[[n_int], [n_int]]))
    if mode is not None and space is not None:
        h_static = h_static + mode.omega_rad_s * space.number(0)

    # ---- couplings ----
    terms: list[Any] = []
    couplings: list[CouplingRecord] = []
    for b, beam in enumerate(beams):
        w_b = _beam_omega(beam)
        e0 = st.field_amplitude(beam, tuple(float(x) for x in pos))
        eps = _polarization_components(beam, b_hat)
        eta_b = etas_by_beam.get(b, 0.0)
        addressed = addressed_transitions(st, beam, level_order, opts.address_window)
        for lo, up in addressed_transitions(st, beam, all_levels, opts.address_window):
            if (lo, up) not in addressed:
                approximations.append(
                    f"beam {b} couples {lo}-{up} but that level is excluded: its light shift and scattering are dropped"
                )
        for lo, up in addressed:
            edge = next(e for e in frame.edges if e.beam == b and e.lower == lo and e.upper == up)
            residual = edge.residual_rad_s
            ops_by_q: dict[int, np.ndarray] = {}
            for p in dressed_by_level[lo]:
                for q_state in dressed_by_level[up]:
                    total = 0.0 + 0.0j
                    for qi, qpol in enumerate((-1, 0, 1)):
                        elem = eps[qi] * st.dipole_element_c_m(q_state, p, qpol)
                        if elem != 0.0:
                            block = ops_by_q.setdefault(qpol, np.zeros((n_int, n_int), dtype=complex))
                            block[idx[q_state.full_label], idx[p.full_label]] += e0 * elem / HBAR_J_S
                            total += e0 * elem / HBAR_J_S
                    if total != 0.0:
                        couplings.append(
                            CouplingRecord(
                                b,
                                p.full_label,
                                q_state.full_label,
                                complex(total),
                                w_b - TWO_PI * (q_state.energy_hz - p.energy_hz),
                                eta_b,
                                residual,
                            )
                        )
            if not ops_by_q:
                continue
            mode_op = None if space is None else space.displacement_factor(0, eta_b)
            if beam.modulation is None:
                op_int = qt.Qobj(sum(ops_by_q.values()), dims=[[n_int], [n_int]])
                op = embed(op_int, mode_op).to("CSR")
                if residual == 0.0:
                    c = 0.5 * np.exp(1j * phases[b])
                    h_static = h_static + c * op + np.conj(c) * op.dag()
                else:
                    args = {
                        "residual": residual,
                        "phase": phases[b],
                        "mod_kind": "",
                        "mod_omega": 0.0,
                        "mod_depth": 0.0,
                        "q_index": 0,
                    }
                    terms.append([op, qt.coefficient(_beat_coefficient, args=args)])
                    terms.append([op.dag(), qt.coefficient(_beat_coefficient_conj, args=args)])
            else:
                mod = beam.modulation
                approximations.append(
                    f"beam {b}: polarization modulation ({mod.kind}) makes the Liouvillian time periodic; steadystate is "
                    "replaced by the period propagator (Section 8.1) [background]"
                )
                for qpol, block in ops_by_q.items():
                    op = embed(qt.Qobj(block, dims=[[n_int], [n_int]]), mode_op).to("CSR")
                    args = {
                        "residual": residual,
                        "phase": phases[b],
                        "mod_kind": mod.kind,
                        "mod_omega": TWO_PI * mod.frequency_hz,
                        "mod_depth": mod.depth_rad,
                        "q_index": qpol,
                    }
                    terms.append([op, qt.coefficient(_beat_coefficient, args=args)])
                    terms.append([op.dag(), qt.coefficient(_beat_coefficient_conj, args=args)])
    beats = list(frame.beats_rad_s)
    for beam in beams:
        if beam.modulation is not None:
            beats.append(TWO_PI * beam.modulation.frequency_hz)
    if len(beats) != len(frame.beats_rad_s):
        distinct = tuple(sorted({round(x, 6) for x in beats}))
        frame = FrameAssignment(frame.frame_rad_s, frame.edges, distinct, _common_period(distinct))
    if not frame.static:
        approximations.append(
            "frame graph inconsistent: the Hamiltonian is time periodic at "
            + ", ".join(f"{b / TWO_PI:.6g} Hz" for b in frame.beats_rad_s)
            + " and is integrated as such (Section 4.2.8 Floquet fallback)"
        )

    # ---- dissipator ----
    c_ops: list[qt.Qobj] = []
    channels: list[EmissionChannel] = []
    level_rates: dict[str, float] = {}
    for up in level_order:
        uppers = dressed_by_level[up]
        lower_lines = [(lo, u) for (lo, u) in st.e1 if u == up]
        if not uppers or not lower_lines:
            continue
        lifetime = st.species.level(up).lifetime_s
        total_rate = (
            1.0 / lifetime
            if lifetime is not None
            else sum(st.e1[ln].partial_rate_rad_s for ln in lower_lines)
        )
        level_rates[up] = total_rate
        col_idx = [idx[e.full_label] for e in uppers]
        included_rate = np.zeros(len(uppers))
        blocks: list[tuple[str, dict[int, np.ndarray], float]] = []
        for lo, _u in lower_lines:
            lowers = dressed_by_level.get(lo, [])
            if not lowers:
                continue
            g = _g_lu(st, lo, up)
            per_q: dict[int, np.ndarray] = {}
            for q in (-1, 0, 1):
                blk = _decay_block(st, lowers, uppers, q)
                if np.any(blk != 0.0):
                    per_q[q] = blk
                    included_rate += g * np.sum(np.abs(blk) ** 2, axis=0)
            if per_q:
                blocks.append((lo, per_q, g))
        deficit = total_rate - included_rate
        deficit[np.abs(deficit) < 1e-9 * total_rate] = 0.0
        scale = np.ones(len(uppers))
        has_deficit = bool(np.any(deficit > 0.0))
        if has_deficit:
            if opts.leak == "renormalize":
                scale = np.sqrt(total_rate / included_rate)
                approximations.append(
                    f"{up}: decay outside the included states ({np.max(deficit) / TWO_PI:.4g} Hz of "
                    f"{total_rate / TWO_PI:.4g} Hz) folded back into the included lines (instantaneous repump)"
                )
            elif opts.leak == "include":
                approximations.append(
                    f"{up}: {np.max(deficit) / TWO_PI:.4g} Hz of its decay goes to excluded states or untabulated "
                    "lines and is dropped (trace-decreasing); include the states or use leak='sink'"
                )
        for lo, per_q, g in blocks:
            eta_em = eta_em_by_line.get((lo, up), 0.0)
            k_em = _transition_omega(st, lo, up) / C_M_PER_S
            row_idx = [idx[p.full_label] for p in dressed_by_level[lo]]
            n_ops = 0
            start = len(c_ops)
            op_qs: list[int] = []
            alphas: dict[int, float] = {}
            if opts.recoil == "vector" and space is not None:
                chans = vector_channels(
                    b_hat, mode_axis, n_theta=opts.recoil_grid[0], n_phi=opts.recoil_grid[1]
                )
                shape = next(iter(per_q.values())).shape
                for ch in chans:
                    matrix = np.zeros(shape, dtype=complex)
                    for qi, q in enumerate((-1, 0, 1)):
                        if q in per_q:
                            matrix += ch.amplitudes[qi] * per_q[q]
                    if not np.any(matrix != 0.0):
                        continue
                    kick = space.displacement_factor(0, -eta_em * ch.u)
                    # the rate prefactor multiplies the NumPy array: QuTiP tidies elements below 1e-12 away on creation
                    op_int = _place(math.sqrt(ch.weight * g) * matrix, row_idx, col_idx, scale, n_int)
                    c_ops.append(embed(op_int, kick).to("CSR"))
                    op_qs.append(9)
                    n_ops += 1
                alphas = {q: angular_factor(q, cos_chi) for q in per_q}
            else:
                for q, blk in per_q.items():
                    op_int = _place(math.sqrt(g) * blk, row_idx, col_idx, scale, n_int)
                    if opts.recoil == "off" or space is None:
                        c_ops.append(embed(op_int).to("CSR"))
                        op_qs.append(q)
                        n_ops += 1
                        continue
                    alpha = angular_factor(q, cos_chi)
                    alphas[q] = alpha
                    quad = (
                        minimal_quadrature(alpha)
                        if opts.recoil == "minimal"
                        else marginal_quadrature(q, cos_chi, opts.recoil_nodes)
                    )
                    for u, p_w in zip(quad.nodes, quad.weights):
                        if p_w <= 0.0:
                            continue
                        kick = space.displacement_factor(0, -eta_em * float(u))
                        c_ops.append(embed(math.sqrt(p_w) * op_int, kick).to("CSR"))
                        op_qs.append(q)
                        n_ops += 1
            kind: Literal["decay", "sink", "renormalized"] = "decay"
            rate = st.e1[(lo, up)].partial_rate_rad_s
            if opts.leak == "renormalize" and has_deficit:
                kind = "renormalized"
                rate = rate * float(np.mean(scale**2))
            channels.append(
                EmissionChannel(
                    lo, up, rate, k_em, n_ops, kind, eta_em, alphas, (start, len(c_ops)), tuple(op_qs)
                )
            )
        if opts.leak == "sink" and has_deficit:
            n_ops = 0
            start = len(c_ops)
            for j, e in enumerate(uppers):
                if deficit[j] <= 0.0:
                    continue
                op_int = qt.basis(n_int, idx[SINK]) * qt.basis(n_int, idx[e.full_label]).dag()
                c_ops.append(math.sqrt(float(deficit[j])) * embed(op_int).to("CSR"))
                n_ops += 1
            channels.append(
                EmissionChannel(
                    SINK, up, float(np.max(deficit)), 0.0, n_ops, "sink", 0.0, {}, (start, len(c_ops)), ()
                )
            )
            approximations.append(
                f"{up}: decay outside the included states ({np.max(deficit) / TWO_PI:.4g} Hz) routed to the sink state"
            )

    # ---- laser linewidth on the optical coherences (Section 8.1; Berkeland and Boshier 2002 Eq. 12) ----
    dephasing_start = len(c_ops)
    if opts.laser_linewidth_rad_s:
        if len(opts.laser_linewidth_rad_s) != len(beams):
            raise ValueError("laser_linewidth_rad_s has one entry per beam")
        for b, width in enumerate(opts.laser_linewidth_rad_s):
            if width <= 0.0:
                continue
            addressed = addressed_transitions(st, beams[b], level_order, opts.address_window)
            if not addressed:
                continue
            weight = np.zeros(n_int)
            for lo, up in addressed:
                for d in dressed_by_level.get(up, []):
                    weight[idx[d.full_label]] += 1.0
                for d in dressed_by_level.get(lo, []):
                    weight[idx[d.full_label]] -= 1.0
            if not np.any(weight != 0.0):
                continue
            op_int = qt.Qobj(np.diag(math.sqrt(0.25 * width) * weight), dims=[[n_int], [n_int]])
            c_ops.append(embed(op_int).to("CSR"))
            approximations.append(
                f"beam {b}: laser linewidth {width / TWO_PI:.4g} Hz added as phase diffusion on its optical "
                "coherences (-delta omega_L/2; Section 8.1, Berkeland and Boshier 2002 Eq. 12), a constant "
                "Lorentzian rate and not the laser's spectrum (Section 12) [background]"
            )
        if any(
            width > 0.0 and len(addressed_transitions(st, beams[b], level_order, opts.address_window)) > 1
            for b, width in enumerate(opts.laser_linewidth_rad_s)
        ):
            approximations.append(
                "a beam addressing several transitions gets ONE phase-diffusion operator over the union of its "
                "manifolds: exact for the coherences it drives, and a level that is both an upper and a lower of "
                "that beam cancels out of it [background]"
            )

    h_static = h_static.to("CSR")
    H: qt.Qobj | qt.QobjEvo = qt.QobjEvo([h_static, *terms]) if terms else h_static
    return MultiLevelBuild(
        H=H,
        c_ops=tuple(c_ops),
        labels=tuple(labels),
        levels=tuple(level_order),
        frame=frame,
        couplings=tuple(couplings),
        channels=tuple(channels),
        level_rates_rad_s=level_rates,
        approximations=tuple(approximations),
        space=space,
        mode=mode,
        options=opts,
        n_beams=len(beams),
        dephasing_slice=(dephasing_start, len(c_ops)),
    )


def decay_sum_rule_residual(build: MultiLevelBuild) -> float:
    """max over decaying sublevels of |(sum_k C_k^dagger C_k)_{ee} - Gamma_e| / Gamma_e (Section 4.2.8: the kicks are unitary).

    Zero to round-off under ``sink`` and ``renormalize``; under ``include`` it reports the dropped (untabulated) share.
    """
    decay = [
        c for k, c in enumerate(build.c_ops) if not build.dephasing_slice[0] <= k < build.dephasing_slice[1]
    ]
    if not decay:
        return 0.0
    total = decay[0].dag() * decay[0]
    for c in decay[1:]:
        total = total + c.dag() * c
    if build.space is not None:
        tot = np.asarray(total.full()).reshape(build.space.dims + build.space.dims)
        internal = np.trace(tot, axis1=1, axis2=3) / build.space.dims[1]
    else:
        internal = np.asarray(total.full())
    worst = 0.0
    for lab in build.labels:
        if lab == SINK:
            continue
        gamma = build.level_rates_rad_s.get(build.level_of(lab))
        if gamma is None:
            continue
        got = float(np.real(internal[build.index(lab), build.index(lab)]))
        worst = max(worst, abs(got - gamma) / gamma)
    return worst
