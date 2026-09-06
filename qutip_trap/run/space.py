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
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from qutip_trap.control.shaping import GateModes, gate_modes, waveform_integrals
from qutip_trap.hilbert.operators import required_margin
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.control.schedule import PlayedGate, Schedule
    from qutip_trap.control.table import Waveform
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions

ModeClass3 = Literal["resolved", "frozen", "dropped"]

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
    """The loop radius eta_max Omega_peak/epsilon_m that sizes the cap."""
    eta_max: float


def waveform_contributions(
    waveform: Waveform, modes: GateModes, pair: tuple[int, int]
) -> dict[int, ModeContribution]:
    """Per mode of ``modes``: the residual displacement, entangling angle and loop radius of ``waveform`` on ``pair``."""
    ints = waveform_integrals(waveform, modes)
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
        radius = 0.0
        if waveform.segments is not None:
            for seg in waveform.segments:
                peak = 0.0
                for amp in seg.amplitude_hz.values():
                    if callable(amp):
                        grid = np.linspace(0.0, seg.duration_s, 201)
                        peak = max(peak, float(np.max(np.abs([amp(x) for x in grid]))))
                    else:
                        peak = max(peak, abs(float(amp)))
                for leg in seg.legs:
                    mu = seg.detuning_hz[leg]
                    if callable(mu):
                        grid = np.linspace(0.0, seg.duration_s, 201)
                        mu_val = float(min(abs(mu(x)) for x in grid))
                    else:
                        mu_val = abs(float(mu))
                    eps = abs(modes.omega_rad_s[k] - TWO_PI * mu_val)
                    if eps > 0.0:
                        radius = max(radius, eta_max * TWO_PI * peak / eps)
        out[m] = ModeContribution(m, alpha2, chi, radius, eta_max)
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
    radius: float, nbar: float, eta_max: float, *, d_min: int, d_max: int, extra: int = 0
) -> ModeTruncation:
    """The cap rule of ``calibration.entangling.gate_space``: the populated range from the loop radius and the thermal
    occupation plus the Section 5.1.1 margin for the mode's eta."""
    n_hi = int(
        math.ceil((radius + math.sqrt(max(nbar, 0.0))) ** 2 + 2.0 * math.sqrt(radius**2 + nbar + 0.25) + 1.0)
    )
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

    @property
    def frozen_contribution(self) -> dict[int, tuple[float, float]]:
        return {m: c for m, c in self.contribution.items() if self.mode_class[m] == "frozen"}

    @property
    def dropped_modes(self) -> tuple[int, ...]:
        return tuple(sorted(m for m, c in self.mode_class.items() if c == "dropped"))

    @property
    def resolved_modes(self) -> tuple[int, ...]:
        return tuple(m.mode for m in self.space.resolved)


def gate_modes_for(device: Device, gate: PlayedGate, nbar: Mapping[int, float]) -> GateModes:
    b = gate.beams
    if len(b) != 2:
        raise ValueError("the mode selection reads Raman (two-beam) entangling drives")
    return gate_modes(device, gate.pair, (b[0], b[1]), nbar=nbar)


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
) -> SpaceSelection:
    """Classify every mode for ``schedule`` and build the product space of the resolved ones (Sections 5.2, 5.5)."""
    from qutip_trap.light.raman import lamb_dicke_parameters

    n_ions = device.crystal.n_ions
    n_modes = len(device.crystal.modes)
    nb = {m: float((nbar or {}).get(m, 0.0)) for m in range(n_modes)}
    best: dict[int, ModeContribution] = {}
    for gate in schedule.gates:
        modes = gate_modes_for(device, gate, nb)
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
    # modes any drive couples to at all (single-qubit carrier pulses included): frozen at least
    coupled: set[int] = set()
    for pulse in schedule.pulses:
        dk = pulse.drive.delta_k(device.beams)
        if float(np.linalg.norm(dk)) == 0.0:
            continue
        for ion in pulse.drive.ions:
            etas, _ = lamb_dicke_parameters(device, ion, dk)
            coupled.update(m for m, e in etas.items() if abs(e) > 1e-12)
    classes: dict[int, ModeClass3] = {}
    for m in range(n_modes):
        classes[m] = classify(
            best.get(m),
            coupled=m in coupled,
            freeze_alpha_max=options.freeze_alpha_max,
            freeze_chi_max_rad=options.freeze_chi_max_rad,
        )
    resolved: list[ModeTruncation] = []
    notes: list[str] = []
    for m in range(n_modes):
        if classes[m] != "resolved":
            continue
        c = best[m]
        tr = cap_for(c.radius, nb[m], c.eta_max, d_min=d_min, d_max=d_max, extra=extra_levels)
        d = int(caps[m]) if caps is not None and m in caps else tr.d
        resolved.append(ModeTruncation(m, d, (0, min(tr.expected_n_range[1], d - 1)), tr.eta_max))
    frozen = tuple(m for m in range(n_modes) if classes[m] != "resolved")
    dims = tuple(int(x) for x in (ion_dims if ion_dims is not None else [2] * n_ions))
    space = HilbertSpace(dims, tuple(resolved), None, frozen)
    for m, c in best.items():
        notes.append(
            f"mode {m}: {classes[m]} (|alpha|^2(2n+1) = {c.alpha2_weighted:.2e}, |chi| = {c.chi_rad:.3e} rad, "
            f"radius {c.radius:.3f})"
        )
    return SpaceSelection(
        space=space,
        mode_class=classes,
        contribution={m: (c.alpha2_weighted, c.chi_rad) for m, c in best.items()},
        nbar=nb,
        notes=tuple(notes),
    )


def drive_operator_nonzeros(space: HilbertSpace) -> int:
    """The Section 11.2 estimate of the merged drive operator's non-zeros: N 2^N prod_m d_m^2 for two-level ions."""
    n = space.n_ions
    prod = 1
    for m in space.resolved:
        prod *= m.d * m.d
    return int(n * (2**n) * prod)


__all__ = [
    "DROP_ALPHA_MAX",
    "DROP_CHI_MAX_RAD",
    "ModeClass3",
    "ModeContribution",
    "SpaceSelection",
    "cap_for",
    "classify",
    "drive_operator_nonzeros",
    "gate_modes_for",
    "select_space",
    "waveform_contributions",
]
