"""Resolved, frozen and dropped modes for a schedule, and the joint space that carries them (PLAN.md Sections 5.1, 5.2,
5.4, 5.5, 11.3, 11.5; Appendix E ``HilbertSpace.for_``; milestone M6).

Every crystal mode is assigned to exactly one of three classes per run (Section 5.2 [corrected: critique, 2026-09-04]),
from the closed-form residual displacement and entangling-angle contribution of the Section 4.4.3 integrals evaluated with
each played waveform's actual detuning schedule and the prepared occupations (``control.shaping.waveform_integrals``):

- dropped when max_gates |alpha_m|^2 (2 nbar_m + 1) < 1e-6 and |chi_m| < 1e-4 (nothing absorbs a dropped mode's loss);
- frozen when above that pair but |alpha_m|^2 (2 nbar_m + 1) < ``SolverOptions.freeze_alpha_max`` and |chi_m| <
  ``SolverOptions.freeze_chi_max_rad`` (default 0.05 rad, the miscalibration the entangling-gate calibration absorbs): the
  mode leaves the joint space, its Fock state is drawn per shot and enters through the exact Debye-Waller factor, and its
  chi_m is reported as the frozen contribution;
- resolved otherwise: carried in the product space with a cap that follows the loop radius eta Omega/epsilon of the gate
  that drives it hardest, the thermal occupation and the Section 5.1.1 margin for its eta.

Modes that only single-qubit carrier pulses touch are frozen (their Debye-Waller factor is what the carrier sees); modes no
drive couples to are dropped. ``HilbertSpace.for_(device, schedule, options)`` is this selection with the device's prepared
occupations; the dimension and non-zero guards of Section 11.5 are ``resolve_level``'s.

M9a adds what Sections 5.2 and 11.3 ask the selection to report and offer: the summed contribution of the dropped modes
(``SpaceSelection.dropped_contribution``: nothing absorbs it), the off-resonant excitation bound of every frozen spectator,
sum over the tones, the gate ions and the nearest sidebands of (eta_m Omega sqrt(nbar_m + 1)/(mu - l omega_m))^2, with the
detuning guard |mu - l omega_m| > 20 eta_m Omega sqrt(n + 1) kept as the sanity check whose violations are noted
(``frozen_excitation_bounds``), and the ENR option (``enr``): a group of cold modes carried dynamically as ONE
excitation-number-restricted factor with the sum-generator displacement (Sections 5.1, 5.1.1; 11.3 item 1), a fourth class
``enr`` beside resolved, frozen and dropped, never composed with the product-space treatment on the same modes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from qutip_trap.control.shaping import GateModes, excursion_by_mode, gate_modes, waveform_integrals
from qutip_trap.hilbert.operators import populated_range, required_margin
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation, enr_dimension
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import PlayedGate, Schedule
    from qutip_trap.control.table import Waveform
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions

ModeClass3 = Literal["resolved", "frozen", "dropped", "enr"]
"""The three classes of Section 5.2 plus ``enr``, the ENR option's class (Section 11.3 item 1; M9a)."""

DETUNING_GUARD_FACTOR = 20.0
"""Section 5.2: |mu_tone - l omega_m| > 20 eta_m Omega sqrt(n_m + 1) is the frozen-spectator sanity check, (1/20)^2 = 2.5e-3 of
off-resonant excitation per sideband; violations are reported, never enforced (the contribution criterion decides)."""

DROP_ALPHA_MAX = 1e-6
"""|alpha|^2 (2 nbar + 1) below which, together with DROP_CHI_MAX_RAD, a mode is dropped (Section 11.3 item 2)."""
DROP_CHI_MAX_RAD = 1e-4


@dataclass(frozen=True)
class ModeContribution:
    """One mode's closed-form contribution to one played gate (Section 4.4.3 integrals)."""

    mode: int
    alpha2_weighted: float
    """max over the gate ions of |alpha_{i,m}(tau)|^2 (2 nbar_m + 1)."""
    chi_rad: float
    """|chi_m| of the pair."""
    radius: float
    """The coherent excursion that sizes the cap: max over the pulse of sum_i |alpha_{i,m}(t)|, the displacement of the S = +-N
    spin branch from the closed-form trajectories (``control.shaping.excursion_by_mode``; M9a. M6 used the single-loop radius
    eta_max Omega_peak/epsilon_m, which a segmented pulse exceeds by factors, so the Section 5.5 margin check tripped)."""
    eta_max: float


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
        out[m] = ModeContribution(m, alpha2, chi, float(excursion.get(m, 0.0)), eta_max)
    return out


def classify(
    contribution: ModeContribution | None,
    *,
    coupled: bool,
    freeze_alpha_max: float,
    freeze_chi_max_rad: float,
) -> ModeClass3:
    if contribution is None:
        return "frozen" if coupled else "dropped"
    if contribution.alpha2_weighted < DROP_ALPHA_MAX and contribution.chi_rad < DROP_CHI_MAX_RAD:
        return "dropped" if not coupled else "frozen"
    if contribution.alpha2_weighted < freeze_alpha_max and contribution.chi_rad < freeze_chi_max_rad:
        return "frozen"
    return "resolved"


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
    """The cap rule (Sections 5.1.1, 5.5): the populated range of the thermal mode displaced by the loop radius, at the boundary
    threshold ``tail`` (``hilbert.operators.populated_range``, the same definition the engine's margin check reads), plus the
    Section 5.1.1 margin for the mode's eta; the M6 rule (radius + sqrt nbar)^2 + 2 sqrt(radius^2 + nbar + 1/4) + 1 sat one
    level under the margin check on the two-ion fixture and cost every pulse a cap-raising retry (M9a)."""
    n_hi = populated_range(max(radius, 0.0), max(nbar, 0.0), tail=tail)
    d = min(max(n_hi + 1 + required_margin(eta_max) + extra, d_min), d_max)
    return ModeTruncation(-1, d, (0, min(n_hi, d - 1)), max(eta_max * 1.5, 1e-3))


@dataclass(frozen=True)
class SpaceSelection:
    """The joint space of a run and the class of every mode (Section 5.2), with the frozen contributions reported."""

    space: HilbertSpace
    mode_class: dict[int, ModeClass3]
    contribution: dict[int, tuple[float, float]]
    """Per mode touched by an entangling gate: (max |alpha|^2 (2 nbar + 1), max |chi|) over the played gates."""
    nbar: dict[int, float]
    notes: tuple[str, ...] = field(default_factory=tuple)
    frozen_excitation: dict[int, float] = field(default_factory=dict)
    """Per frozen mode, the summed off-resonant excitation bound of Section 5.2 over the schedule's pulses (M9a)."""
    guard_violations: tuple[str, ...] = field(default_factory=tuple)
    """The detuning-guard sanity check of Section 5.2 where it fails (reported, never enforced; M9a)."""

    @property
    def frozen_contribution(self) -> dict[int, tuple[float, float]]:
        return {m: c for m, c in self.contribution.items() if self.mode_class[m] == "frozen"}

    @property
    def dropped_modes(self) -> tuple[int, ...]:
        return tuple(sorted(m for m, c in self.mode_class.items() if c == "dropped"))

    @property
    def dropped_contribution(self) -> tuple[float, float]:
        """(sum |alpha_m|^2 (2 nbar_m + 1), sum |chi_m|) over the dropped modes: the summed dropped contribution Section 11.3
        item 2 asks the report to state, since nothing absorbs a dropped mode's loss (M9a)."""
        a = sum(c[0] for m, c in self.contribution.items() if self.mode_class[m] == "dropped")
        x = sum(c[1] for m, c in self.contribution.items() if self.mode_class[m] == "dropped")
        return float(a), float(x)

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
    """Section 5.2's error bound of the frozen-spectator option, per frozen mode: the off-resonant excitation probability
    (eta_m Omega sqrt(n_m + 1)/(mu - l omega_m))^2 summed over the pulses, their tones, the addressed ions and the nearest
    sidebands l = -/+ 1 (n_m = nbar_m, the thermal mean), together with the detuning-guard violations
    |mu - l omega_m| <= 20 eta_m Omega sqrt(n_m + 1) as notes. A pulse whose tone sits exactly ON a sideband of a frozen
    mode has an unbounded excitation and is reported as such (inf): the mode cannot be frozen for that pulse."""
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
    d_max: int = 64,
    extra_levels: int = 0,
    ion_dims: Sequence[int] | None = None,
    enr: tuple[Sequence[int], int] | None = None,
) -> SpaceSelection:
    """Classify every mode for ``schedule`` and build the product space of the resolved ones (Sections 5.2, 5.5).

    ``enr`` = (modes, N_exc) carries the named modes as one ENR factor instead of their criterion class (Section 11.3 item 1,
    the option for cold undriven groups); a mode the criterion would resolve may be placed there, with a note, since the top
    ENR shell is then the boundary the monitor watches (Section 5.1: the ENR displacement is silently wrong near the cap)."""
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
        tr = cap_for(c.radius, nb[m], c.eta_max, d_min=d_min, d_max=d_max, extra=extra_levels)
        d = int(caps[m]) if caps is not None and m in caps else tr.d
        resolved.append(ModeTruncation(m, d, (0, min(tr.expected_n_range[1], d - 1)), tr.eta_max))
    frozen = tuple(m for m in range(n_modes) if classes[m] in ("frozen", "dropped"))
    dims = tuple(int(x) for x in (ion_dims if ion_dims is not None else [2] * n_ions))
    space = HilbertSpace(dims, tuple(resolved), enr_group, frozen)
    for m, c in best.items():
        notes.append(
            f"mode {m}: {classes[m]} (|alpha|^2(2n+1) = {c.alpha2_weighted:.2e}, |chi| = {c.chi_rad:.3e} rad, "
            f"radius {c.radius:.3f})"
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
        nbar=nb,
        notes=tuple(notes),
        frozen_excitation=excitation,
        guard_violations=guard,
    )


def drive_operator_nonzeros(space: HilbertSpace) -> int:
    """The Section 11.2 estimate of the merged drive operator's non-zeros: N 2^N prod_m d_m^2 for two-level ions, times the
    square of the ENR factor's dimension when a group is carried (its sum-generator exponential is dense within the block)."""
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
    "ModeClass3",
    "ModeContribution",
    "SpaceSelection",
    "best_contributions",
    "cap_for",
    "classify",
    "coupled_modes",
    "drive_operator_nonzeros",
    "frozen_excitation_bounds",
    "gate_modes_for",
    "select_space",
    "waveform_contributions",
]
