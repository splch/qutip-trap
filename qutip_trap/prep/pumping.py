"""Optical pumping: the internal-state preparation and its recoil heating (PLAN.md Section 4.2.6).

The pumping beams' Bloch model evolves the scrambled state Doppler cooling leaves (a uniform mixture over the resonant
ground manifold, by default) and gives the residual population outside the target (the preparation error), the pumping
time, the scattered photons and, through the recoil kernel with the ion's participation, the motional heating
(a few photons at k x0 = 0.053 leave Delta n ~ 3e-3). 171Yb+ pumps to |F=0, m=0> on S1/2 F=1 -> P1/2 F=1 with 1/3
branching into |0> per excitation, three photons on average.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import qutip as qt

from qutip_trap.dynamics.multilevel import SINK
from qutip_trap.light.bloch import BlochModel, PumpingTrace, operator_angular_factor
from qutip_trap.light.recoil import emission_lamb_dicke
from qutip_trap.noise.levels import SINK as REGISTER_SINK
from qutip_trap.trap.crystal import Crystal

TOLERANCE = 1e-4
"""1 - P(target) at which the pump counts as done (``time_to_reach_s``)."""


@dataclass(frozen=True)
class PumpingResult:
    """What optical pumping produced: the internal state, its error, the duration, the photons and their recoil."""

    trace: PumpingTrace
    populations: dict[str, float]
    preparation_error: float
    """1 - P(target) at the end of the pump."""
    steady_state_error: float
    """1 - P(target) of the beams' steady state: the floor from off-resonant excitation of the target."""
    time_to_reach_s: float | None
    """First sampled time at which P(target) >= 1 - TOLERANCE."""
    photons_scattered: float
    motional_heating_quanta: dict[int, float]
    """Delta n per mode deposited by the scattered photons' recoil (empty without a crystal)."""

    def qubit_density_matrix(self, qubit: tuple[str, str]) -> qt.Qobj:
        """The diagonal 2 x 2 state on the qubit pair (index 0 the lower level); population outside the pair counts as the
        upper qubit state, the worst case for a preparation-error budget."""
        lower, upper = qubit
        p_lower = self.populations.get(lower, 0.0)
        p_upper = self.populations.get(upper, 0.0)
        p_upper += max(0.0, 1.0 - p_lower - p_upper)
        return qt.Qobj(np.diag([p_lower, p_upper]), dims=[[2], [2]])

    def qudit_density_matrix(self, labels: Sequence[str]) -> qt.Qobj:
        """The d x d diagonal state on a register factor with the given level labels: every resolved label takes its
        pumped population, the SINK (if present) the remainder, else the remainder counts as the upper qubit level."""
        pops = [0.0 if lab == REGISTER_SINK else float(self.populations.get(lab, 0.0)) for lab in labels]
        rest = max(0.0, 1.0 - sum(pops))
        if REGISTER_SINK in labels:
            pops[list(labels).index(REGISTER_SINK)] = rest
        else:
            pops[1] += rest
        d = len(labels)
        return qt.Qobj(np.diag(pops), dims=[[d], [d]])


def scrambled_initial_state(model: BlochModel, labels: Sequence[str] | None = None) -> qt.Qobj:
    """The uniform mixture over ``labels`` (default: the resonant ground manifold of the beams)."""
    b = model.build
    if labels is None:
        ground, _excited = model.resonant_manifold()
        labels = ground
    if not labels:
        raise ValueError("no ground states to scramble over")
    n = b.n_internal
    rho = np.zeros((n, n))
    for lab in labels:
        rho[b.index(lab), b.index(lab)] = 1.0 / len(labels)
    return qt.Qobj(rho, dims=[[n], [n]])


def optical_pumping(
    model: BlochModel,
    target: Sequence[str],
    *,
    duration_s: float,
    initial: qt.Qobj | Sequence[str] | None = None,
    samples: int = 4001,
    crystal: Crystal | None = None,
    ion: int = 0,
) -> PumpingResult:
    """Evolve the internal state under the pumping beams for ``duration`` and read off the preparation error, time,
    photons and (with a crystal) recoil heating.

    ``initial``: a density matrix on the model's internal factor, state labels to scramble over, or None (the resonant
    ground manifold). An accurate photon count needs the sampling to resolve the photon rate's coherent oscillations
    (5 ns steps; 100 ns aliases 3.00 photons to 2.83).
    """
    if model.build.space is not None:
        raise ValueError(
            "optical pumping runs on the internal-only model; the recoil is booked through the kernel"
        )
    rho0 = initial if isinstance(initial, qt.Qobj) else scrambled_initial_state(model, initial)
    trace = model.evolve(rho0, np.linspace(0.0, duration_s, samples))
    ss = model.steadystate()
    return PumpingResult(
        trace=trace,
        populations={lab: float(arr[-1]) for lab, arr in trace.populations.items()},
        preparation_error=float(1.0 - trace.population(list(target))[-1]),
        steady_state_error=float(1.0 - sum(ss.populations.get(lab, 0.0) for lab in target)),
        time_to_reach_s=trace.time_to_reach(list(target), 1.0 - TOLERANCE),
        photons_scattered=float(trace.photons_scattered[-1]),
        motional_heating_quanta={} if crystal is None else pump_recoil_heating(model, trace, crystal, ion),
    )


def pump_recoil_heating(
    model: BlochModel, trace: PumpingTrace, crystal: Crystal, ion: int
) -> dict[int, float]:
    """Delta n_m = sum_operators N_k alpha_q(k)(chi_m) eta_em,{i,m}^2: each emitted photon's mean recoil into every mode,
    with the channel's own wavenumber and polarization index and the ion's participation."""
    b = model.build
    photons = trace.photons_per_operator()
    b_hat = model.structure.b_hat
    out: dict[int, float] = {}
    for m, mode in enumerate(crystal.modes):
        cos_chi = float(np.dot(np.asarray(mode.e_hat, dtype=float), b_hat))
        total = 0.0
        for ch in b.channels:
            if ch.kind == "sink" or ch.lower == SINK:
                continue
            eta_em = emission_lamb_dicke(crystal, ion, ch.wavenumber_rad_per_m, m)
            start, stop = ch.operator_slice
            for k in range(start, stop):
                total += float(photons[k]) * operator_angular_factor(ch, k - start, cos_chi) * eta_em**2
        out[m] = total
    return out
