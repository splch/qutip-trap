"""Mode selection for a schedule: each crystal mode is resolved, frozen, dropped or carried in an ENR group.

A resolved mode is carried with a Fock cap, a frozen one leaves the joint space (its Fock state entering through the
exact Debye-Waller factor) and a dropped one the dynamics; :func:`classify` decides from the played entangling
waveforms' closed-form contributions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from qutip_trap.control.shaping import GateModes, excursion_by_mode, gate_modes, waveform_integrals
from qutip_trap.hilbert.operators import populated_range, required_margin
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation, enr_dimension
from qutip_trap.hilbert.truncation import warn_cap_clamped
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import PlayedGate, Schedule
    from qutip_trap.control.table import Waveform
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions

ModeClass3 = Literal["resolved", "frozen", "dropped", "enr"]
"""A mode's class: resolved, frozen, dropped, or ``enr`` (in an excitation-number-restricted group)."""

DETUNING_GUARD_FACTOR = 20.0
"""The frozen-spectator sanity check |mu - l omega_m| > 20 eta_m Omega sqrt(n_m + 1), at most (1/20)^2 = 2.5e-3 of
off-resonant excitation per sideband; violations are reported, never enforced."""

DROP_ALPHA_MAX = 1e-6
"""|alpha|^2 (2 nbar + 1) below which, with ``DROP_CHI_MAX_RAD`` and ``DW_SPREAD_DROP_MAX``, a mode is dropped."""
DROP_CHI_MAX_RAD = 1e-4

DW_SPREAD_DROP_MAX = 3e-4
"""The largest shot-to-shot spread of its Debye-Waller factor (rad of rotation angle) that a dropped mode may carry: a
pulse that closes a mode's loop leaves |alpha|^2 (2 nbar + 1) near zero however strongly it couples, and the calibration
absorbs the factor's mean but not its thermal spread eta^2 sqrt(nbar (nbar + 1)). Above it the mode is frozen instead;
(3e-4/2)^2 = 2.3e-8 of pi-pulse infidelity sits below ``DROP_ALPHA_MAX``."""


@dataclass(frozen=True)
class ModeContribution:
    """One mode's closed-form contribution to one played gate."""

    mode: int
    alpha2_weighted: float
    """max over the gate ions of |alpha_{i,m}(tau)|^2 (2 nbar_m + 1)."""
    chi_rad: float
    """|chi_m| of the pair."""
    radius: float
    """The coherent excursion that sizes the cap: max over the pulse of sum_i |alpha_{i,m}(t)|
    (``control.shaping.excursion_by_mode``)."""
    eta_max: float
    dw_spread_rad: float = 0.0
    """max over the gate ions of eta_{i,m}^2 sqrt(nbar_m (nbar_m + 1)), the Debye-Waller spread (rad) that
    ``DW_SPREAD_DROP_MAX`` bounds; zero by default."""


def waveform_contributions(
    waveform: Waveform, modes: GateModes, pair: tuple[int, int]
) -> dict[int, ModeContribution]:
    """Per mode of ``modes``: the residual displacement, entangling angle and loop radius of ``waveform`` on ``pair``."""
    ints = waveform_integrals(waveform, modes)
    excursion = excursion_by_mode(waveform, modes) if waveform.segments is not None else {}
    a, b = pair
    out: dict[int, ModeContribution] = {}
    for k, m in enumerate(modes.modes):
        eta_max = max(abs(modes.eta[i][k]) for i in modes.ions)
        nbar = modes.nbar[k]
        alpha2 = 0.0
        for i in modes.ions:
            al = ints.alpha.get((i, m), 0.0)
            alpha2 = max(alpha2, abs(al) ** 2 * (2.0 * nbar + 1.0))
        chi = abs(ints.chi_by_mode.get((a, b, m), ints.chi_by_mode.get((b, a, m), 0.0)))
        spread = eta_max**2 * math.sqrt(max(nbar, 0.0) * (max(nbar, 0.0) + 1.0))
        out[m] = ModeContribution(m, alpha2, chi, float(excursion.get(m, 0.0)), eta_max, float(spread))
    return out


def classify(
    contribution: ModeContribution | None,
    *,
    coupled: bool,
    freeze_alpha_max: float,
    freeze_chi_max_rad: float,
) -> ModeClass3:
    """One mode's class from its contribution over the played entangling waveforms: dropped below (``DROP_ALPHA_MAX``,
    ``DROP_CHI_MAX_RAD``) with a Debye-Waller spread below ``DW_SPREAD_DROP_MAX``, frozen below (``freeze_alpha_max``,
    ``freeze_chi_max_rad``), else resolved. A mode no entangling gate touches (``contribution is None``) is frozen when
    ``coupled`` (a carrier pulse needs its Debye-Waller factor), else dropped."""
    if contribution is None:
        return "frozen" if coupled else "dropped"
    if (
        contribution.alpha2_weighted < DROP_ALPHA_MAX
        and contribution.chi_rad < DROP_CHI_MAX_RAD
        and contribution.dw_spread_rad < DW_SPREAD_DROP_MAX
    ):
        return "dropped"
    if contribution.alpha2_weighted < freeze_alpha_max and contribution.chi_rad < freeze_chi_max_rad:
        return "frozen"
    return "resolved"


def cap_requirement(
    radius: float,
    nbar: float,
    eta_max: float,
    *,
    d_min: int,
    extra: int = 0,
    tail: float = 1e-6,
) -> tuple[int, int]:
    """(d, highest expected Fock index) the cap rule asks for before any ceiling: the populated range at ``tail`` (the
    definition the engine's margin check reads) of the thermal mode displaced by ``radius``, plus the margin for its
    eta."""
    n_hi = populated_range(max(radius, 0.0), max(nbar, 0.0), tail=tail)
    return max(n_hi + 1 + required_margin(eta_max) + extra, d_min), n_hi


def cap_for(
    radius: float,
    nbar: float,
    eta_max: float,
    *,
    d_min: int,
    d_max: int,
    extra: int = 0,
    tail: float = 1e-6,
) -> ModeTruncation:
    """The cap rule, :func:`cap_requirement` clamped to ``d_max`` (the declared range too; the caller reports a clamp),
    as a ``ModeTruncation`` with mode -1 for the caller to set."""
    d_want, n_hi = cap_requirement(radius, nbar, eta_max, d_min=d_min, extra=extra, tail=tail)
    d = min(d_want, d_max)
    return ModeTruncation(-1, d, (0, min(n_hi, d - 1)), max(eta_max * 1.5, 1e-3))


@dataclass(frozen=True)
class SpaceSelection:
    """The joint space of a run (a declaration: no operator on it is allocated yet) and the class of every mode, with
    the contributions and the size-guard verdict (``budget``)."""

    space: HilbertSpace
    mode_class: dict[int, ModeClass3]
    contribution: dict[int, tuple[float, float]]
    """Per mode touched by an entangling gate: (max |alpha|^2 (2 nbar + 1), max |chi|) over the played gates."""
    nbar: dict[int, float]
    notes: tuple[str, ...] = field(default_factory=tuple)
    frozen_excitation: dict[int, float] = field(default_factory=dict)
    """Per frozen mode, the off-resonant excitation bound summed over the schedule's pulses."""
    guard_violations: tuple[str, ...] = field(default_factory=tuple)
    """The detuning-guard failures (reported, never enforced)."""
    budget: tuple[bool, int, int] = (True, 0, 0)
    """(inside the guards, joint dimension, drive-operator non-zero estimate) of the declared space
    (``run.levels.within_budget``); a caller that supplies its own space keeps the default."""
    dw_spread: dict[int, float] = field(default_factory=dict)
    """Per mode touched by an entangling gate, the Debye-Waller spread (rad), eta^2 sqrt(nbar (nbar + 1))."""

    @property
    def frozen_contribution(self) -> dict[int, tuple[float, float]]:
        return {m: c for m, c in self.contribution.items() if self.mode_class[m] == "frozen"}

    @property
    def dropped_modes(self) -> tuple[int, ...]:
        return tuple(sorted(m for m, c in self.mode_class.items() if c == "dropped"))

    @property
    def dropped_contribution(self) -> tuple[float, float]:
        """(sum |alpha_m|^2 (2 nbar_m + 1), sum |chi_m|) over the dropped modes, whose loss nothing absorbs."""
        a = sum(c[0] for m, c in self.contribution.items() if self.mode_class[m] == "dropped")
        x = sum(c[1] for m, c in self.contribution.items() if self.mode_class[m] == "dropped")
        return float(a), float(x)

    @property
    def dropped_dw_spread_rad(self) -> float:
        """Summed Debye-Waller spread (rad) of the dropped modes: the part of their loss the calibration, which absorbs
        the factor's mean, cannot."""
        return float(sum(self.dw_spread.get(m, 0.0) for m, c in self.mode_class.items() if c == "dropped"))

    @property
    def resolved_modes(self) -> tuple[int, ...]:
        return tuple(m.mode for m in self.space.resolved)

    @property
    def enr_modes(self) -> tuple[int, ...]:
        return tuple(sorted(m for m, c in self.mode_class.items() if c == "enr"))


def gate_modes_for(device: Device, gate: PlayedGate, nbar: Mapping[int, float]) -> GateModes:
    b = gate.beams
    if len(b) != 2:
        raise ValueError("the mode selection reads Raman (two-beam) entangling drives")
    return gate_modes(device, gate.pair, (b[0], b[1]), nbar=nbar)


def best_contributions(
    device: Device, gates: Sequence[PlayedGate], nbar: Mapping[int, float]
) -> dict[int, ModeContribution]:
    """Per mode, the worst (largest) closed-form contribution over the played entangling gates."""
    best: dict[int, ModeContribution] = {}
    for gate in gates:
        modes = gate_modes_for(device, gate, nbar)
        for m, c in waveform_contributions(gate.waveform, modes, gate.pair).items():
            prev = best.get(m)
            if prev is None:
                best[m] = c
            else:
                best[m] = ModeContribution(
                    m,
                    max(prev.alpha2_weighted, c.alpha2_weighted),
                    max(prev.chi_rad, c.chi_rad),
                    max(prev.radius, c.radius),
                    max(prev.eta_max, c.eta_max),
                    max(prev.dw_spread_rad, c.dw_spread_rad),
                )
    return best


def coupled_modes(device: Device, pulses: Sequence[Pulse], ions: Sequence[int] | None = None) -> set[int]:
    """The modes any of ``pulses`` couples to (|eta| > 1e-12) on its addressed ions (restricted to ``ions`` when given)."""
    from qutip_trap.light.raman import lamb_dicke_parameters

    out: set[int] = set()
    for pulse in pulses:
        dk = pulse.drive.delta_k(device.beams)
        if float(np.linalg.norm(dk)) == 0.0:
            continue
        for ion in pulse.drive.ions:
            if ions is not None and ion not in ions:
                continue
            etas, _ = lamb_dicke_parameters(device, ion, dk)
            out.update(m for m, e in etas.items() if abs(e) > 1e-12)
    return out


def _peak_rabi_rad_s(pulse: Pulse) -> float:
    peak = 0.0
    for tone in pulse.drive.tones:
        env = tone.envelope_hz
        if callable(env):
            grid = np.linspace(0.0, pulse.duration_s, 101)
            peak = max(peak, float(np.max(np.abs([float(env(x)) for x in grid]))))
        elif isinstance(env, np.ndarray):
            peak = max(peak, float(np.max(np.abs(env))))
        else:
            peak = max(peak, abs(float(env)))
    return TWO_PI * peak


def _tone_detunings_rad_s(pulse: Pulse) -> list[float]:
    out: list[float] = []
    for tone in pulse.drive.tones:
        mu = tone.detuning_hz
        if callable(mu):
            grid = np.linspace(0.0, pulse.duration_s, 101)
            vals = [TWO_PI * float(mu(x)) for x in grid]
            # the schedule's excursion: the value nearest each sideband is the worst case, so keep the extremes
            out.extend([min(vals), max(vals)])
        else:
            out.append(TWO_PI * float(mu))
    return out


def frozen_excitation_bounds(
    device: Device,
    pulses: Sequence[Pulse],
    frozen: Sequence[int],
    nbar: Mapping[int, float],
    *,
    sidebands: Sequence[int] = (-1, 1),
) -> tuple[dict[int, float], tuple[str, ...]]:
    """Per frozen mode, the off-resonant excitation bound: (eta_m Omega sqrt(nbar_m + 1)/(mu - l omega_m))^2 summed
    over the pulses, tones, addressed ions and ``sidebands`` l, with the detuning-guard violations as notes. A tone
    exactly on a sideband gives inf."""
    from qutip_trap.light.raman import lamb_dicke_parameters

    bounds: dict[int, float] = {int(m): 0.0 for m in frozen}
    notes: list[str] = []
    for pulse in pulses:
        dk = pulse.drive.delta_k(device.beams)
        if float(np.linalg.norm(dk)) == 0.0:
            continue
        omega_peak = _peak_rabi_rad_s(pulse)
        if omega_peak == 0.0:
            continue
        mus = _tone_detunings_rad_s(pulse)
        for ion in pulse.drive.ions:
            etas, _ = lamb_dicke_parameters(device, ion, dk)
            for m in frozen:
                eta = abs(float(etas[m]))
                if eta == 0.0:
                    continue
                w_m = device.crystal.modes[m].omega_rad_s
                coupling = eta * omega_peak * math.sqrt(float(nbar.get(m, 0.0)) + 1.0)
                for mu in mus:
                    for ell in sidebands:
                        gap = abs(mu - ell * w_m)
                        if gap == 0.0:
                            bounds[m] = float("inf")
                            notes.append(
                                f"pulse {pulse.gate_id!r}, ion {ion}: a tone sits on sideband {ell:+d} of frozen mode {m}"
                            )
                            continue
                        bounds[m] += (coupling / gap) ** 2
                        if gap <= DETUNING_GUARD_FACTOR * coupling:
                            notes.append(
                                f"pulse {pulse.gate_id!r}, ion {ion}, frozen mode {m}, sideband {ell:+d}: |mu - l omega| = "
                                f"{gap / TWO_PI:.4g} Hz <= 20 eta Omega sqrt(n + 1) = {DETUNING_GUARD_FACTOR * coupling / TWO_PI:.4g} Hz "
                                f"(Section 5.2 guard; excitation {(coupling / gap) ** 2:.2e})"
                            )
    return bounds, tuple(dict.fromkeys(notes))


def select_space(
    device: Device,
    schedule: Schedule,
    options: SolverOptions,
    *,
    nbar: Mapping[int, float] | None = None,
    caps: Mapping[int, int] | None = None,
    d_min: int = 6,
    d_max: int | None = None,
    extra_levels: int = 0,
    ion_dims: Sequence[int] | None = None,
    enr: tuple[Sequence[int], int] | None = None,
) -> SpaceSelection:
    """Classify every mode for ``schedule`` and declare the product space of the resolved ones; ``enr`` = (modes, N_exc)
    carries the named modes as one ENR factor whatever their class. A clamped cap (``d_max`` None: the options'
    ceiling) warns and is noted; the size-guard verdict is taken before any operator is allocated."""
    from qutip_trap.run.levels import within_budget

    d_ceiling = int(options.mode_dimension_max if d_max is None else d_max)
    n_ions = device.crystal.n_ions
    n_modes = len(device.crystal.modes)
    nb = {m: float((nbar or {}).get(m, 0.0)) for m in range(n_modes)}
    best = best_contributions(device, schedule.gates, nb)
    # modes any drive couples to at all (single-qubit carrier pulses included): frozen at least
    coupled = coupled_modes(device, schedule.pulses)
    classes: dict[int, ModeClass3] = {}
    for m in range(n_modes):
        classes[m] = classify(
            best.get(m),
            coupled=m in coupled,
            freeze_alpha_max=options.freeze_alpha_max,
            freeze_chi_max_rad=options.freeze_chi_max_rad,
        )
    notes: list[str] = []
    enr_group: tuple[tuple[int, ...], int] | None = None
    if enr is not None:
        group = tuple(int(m) for m in enr[0])
        n_exc = int(enr[1])
        if not group or n_exc < 0 or any(m < 0 or m >= n_modes for m in group):
            raise ValueError("enr names crystal modes and a non-negative excitation cap")
        for m in group:
            if classes[m] == "resolved":
                notes.append(
                    f"mode {m}: the criterion resolves it (|alpha|^2(2n+1) = {best[m].alpha2_weighted:.2e}, |chi| = "
                    f"{best[m].chi_rad:.3e} rad) but it is carried in the ENR group at N_exc = {n_exc}; the top shell is its "
                    "boundary (Section 5.1)"
                )
            classes[m] = "enr"
        enr_group = (group, n_exc)
    resolved: list[ModeTruncation] = []
    for m in range(n_modes):
        if classes[m] != "resolved":
            continue
        c = best[m]
        # the same boundary threshold the engine's margin check reads
        tail = float(options.boundary_population_max)
        d_want, n_hi_want = cap_requirement(
            c.radius, nb[m], c.eta_max, d_min=d_min, extra=extra_levels, tail=tail
        )
        tr = cap_for(c.radius, nb[m], c.eta_max, d_min=d_min, d_max=d_ceiling, extra=extra_levels, tail=tail)
        d = int(caps[m]) if caps is not None and m in caps else tr.d
        if d_want > d and (caps is None or m not in caps):
            warn_cap_clamped(m, d_want, n_hi_want, d, d_ceiling)
            notes.append(
                f"mode {m}: the cap rule asks for d = {d_want} (expected occupation up to n = {n_hi_want}) but "
                f"mode_dimension_max = {d_ceiling} clamps it to d = {d} (declared range up to n = {min(n_hi_want, d - 1)}); "
                "the Section 5.1.1 oracle check and the Section 5.5 margin check are evaluated over the clamped range, and "
                "the boundary monitor grows the cap only up to the engine's retry budget (Sections 5.2, 5.3: a Doppler-cooled "
                "nbar ~ 20 mode needs d_m >~ 150)"
            )
        resolved.append(ModeTruncation(m, d, (0, min(tr.expected_n_range[1], d - 1)), tr.eta_max))
    frozen = tuple(m for m in range(n_modes) if classes[m] in ("frozen", "dropped"))
    # a dropped mode is in ``frozen`` too (no tensor factor) and in addition has no Debye-Waller factor or Fock branch
    dropped = tuple(m for m in range(n_modes) if classes[m] == "dropped")
    dims = tuple(int(x) for x in (ion_dims if ion_dims is not None else [2] * n_ions))
    # HilbertSpace allocates no operator, so the guards below are evaluated before anything is built
    space = HilbertSpace(dims, tuple(resolved), enr_group, frozen, (), dropped)
    budget = within_budget(space, options)
    if not budget[0]:
        notes.append(
            f"the declared joint space is outside the Section 11.5 guards (dimension {budget[1]} against "
            f"{options.joint_dimension_max}, drive non-zeros {budget[2]} against {options.nnz_max}): no operator on it is "
            "built and the run is routed to GATE_LOCAL"
        )
    for m, c in best.items():
        notes.append(
            f"mode {m}: {classes[m]} (|alpha|^2(2n+1) = {c.alpha2_weighted:.2e}, |chi| = {c.chi_rad:.3e} rad, "
            f"radius {c.radius:.3f}, Debye-Waller spread {c.dw_spread_rad:.2e} rad)"
        )
    dropped_spread = sum(best[m].dw_spread_rad for m in dropped if m in best)
    if dropped_spread > 0.0:
        notes.append(
            f"dropped modes {list(dropped)}: their Debye-Waller factors leave the dynamics entirely; the calibration absorbs "
            f"the MEAN of each factor (the spot check and the run drop the same mode) but not its shot-to-shot spread, "
            f"{dropped_spread:.2e} rad summed, each term below DW_SPREAD_DROP_MAX = {DW_SPREAD_DROP_MAX:g} rad (Section 5.2)"
        )
    frozen_only = [m for m in range(n_modes) if classes[m] == "frozen"]
    excitation, guard = frozen_excitation_bounds(device, schedule.pulses, frozen_only, nb)
    for m, v in excitation.items():
        if v > 0.0:
            notes.append(f"mode {m}: frozen; off-resonant excitation bound {v:.2e} (Section 5.2)")
    return SpaceSelection(
        space=space,
        mode_class=classes,
        contribution={m: (c.alpha2_weighted, c.chi_rad) for m, c in best.items()},
        dw_spread={m: c.dw_spread_rad for m, c in best.items()},
        nbar=nb,
        notes=tuple(notes),
        frozen_excitation=excitation,
        guard_violations=guard,
        budget=budget,
    )


def drive_operator_nonzeros(space: HilbertSpace) -> int:
    """The estimated non-zeros of the merged drive operator: N 2^N prod_m d_m^2 for two-level ions, times the square of
    the ENR factor's dimension when a group is carried (its sum-generator exponential is dense within the block)."""
    n = space.n_ions
    prod = 1
    for m in space.resolved:
        prod *= m.d * m.d
    if space.enr_group is not None:
        d_enr = enr_dimension(len(space.enr_group[0]), space.enr_group[1])
        prod *= d_enr * d_enr
    return int(n * (2**n) * prod)


__all__ = [
    "DETUNING_GUARD_FACTOR",
    "DROP_ALPHA_MAX",
    "DROP_CHI_MAX_RAD",
    "DW_SPREAD_DROP_MAX",
    "ModeClass3",
    "ModeContribution",
    "SpaceSelection",
    "best_contributions",
    "cap_for",
    "cap_requirement",
    "classify",
    "coupled_modes",
    "drive_operator_nonzeros",
    "frozen_excitation_bounds",
    "gate_modes_for",
    "select_space",
    "waveform_contributions",
]
