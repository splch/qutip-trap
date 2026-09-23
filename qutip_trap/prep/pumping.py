"""Optical pumping evolved on the multi-level ``BlochModel`` from the state Doppler cooling leaves: the preparation
error, the pumping time, the scattered photons and their recoil heating.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import qutip as qt

from qutip_trap.dynamics.multilevel import SINK
from qutip_trap.light.bloch import BlochModel, PumpingTrace, operator_angular_factor
from qutip_trap.light.recoil import emission_lamb_dicke
from qutip_trap.trap.crystal import Crystal

LeakToQubit = Literal["to_upper", "renormalize", "full"]


@dataclass(frozen=True)
class PumpingResult:
    """What optical pumping produced: the internal state, its error, the duration, the photons and their recoil."""

    target: tuple[str, ...]
    trace: PumpingTrace
    final: qt.Qobj
    """The internal density matrix at the end of the pump (the full included manifold)."""
    populations: dict[str, float]
    preparation_error: float
    """1 - P(target) at the end of the pump: the SPAM preparation error."""
    steady_state_error: float
    """1 - P(target) of the beams' steady state: the floor from off-resonant excitation of the target."""
    time_to_reach_s: float | None
    """First sampled time at which P(target) >= 1 - tolerance."""
    tolerance: float
    photons_scattered: float
    photons_per_line: dict[str, float]
    motional_heating_quanta: dict[int, float] = field(default_factory=dict)
    """Delta n per mode deposited by the scattered photons' recoil, when a crystal and ion were given."""
    approximations: tuple[str, ...] = ()

    def qubit_density_matrix(self, qubit: tuple[str, str], *, leak: LeakToQubit = "to_upper") -> qt.Qobj:
        """The 2 x 2 state on the qubit pair (index 0 the lower level); population outside it counts as upper
        (``to_upper``, the worst case), is folded back proportionally (``renormalize``) or is refused (``full``)."""
        lower, upper = qubit
        p_lower = self.populations.get(lower, 0.0)
        p_upper = self.populations.get(upper, 0.0)
        rest = max(0.0, 1.0 - p_lower - p_upper)
        if leak == "to_upper":
            p_upper += rest
        elif leak == "renormalize":
            total = p_lower + p_upper
            if total <= 0.0:
                raise ValueError("no population in the qubit pair")
            p_lower, p_upper = p_lower / total, p_upper / total
        elif leak == "full":
            if rest > 1e-12:
                raise ValueError(
                    f"{rest:.3g} of the population sits outside the qubit pair; use a register factor of dimension > 2"
                )
        else:
            raise ValueError("leak is 'to_upper', 'renormalize' or 'full'")
        # the pump erases the coherence between the qubit levels, so the state is diagonal
        return qt.Qobj(np.diag([p_lower, p_upper]), dims=[[2], [2]])

    def qudit_density_matrix(self, labels: Sequence[str]) -> qt.Qobj:
        """The d x d diagonal state on a register factor with the given level labels: each label takes its pumped
        population and the SINK (if present) the remainder; without a SINK the remainder goes to the upper qubit level."""
        from qutip_trap.noise.levels import SINK

        pops = [0.0 if lab == SINK else float(self.populations.get(lab, 0.0)) for lab in labels]
        rest = max(0.0, 1.0 - sum(pops))
        if SINK in labels:
            pops[list(labels).index(SINK)] = rest
        else:
            pops[1] += rest
        d = len(labels)
        return qt.Qobj(np.diag(pops), dims=[[d], [d]])


def scrambled_initial_state(model: BlochModel, labels: Sequence[str] | None = None) -> qt.Qobj:
    """The uniform mixture over ``labels`` (default: the resonant ground manifold of the beams): what Doppler cooling leaves."""
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
    tolerance: float = 1e-4,
    crystal: Crystal | None = None,
    ion: int = 0,
) -> PumpingResult:
    """Pump for ``duration_s`` from ``initial`` (an internal density matrix, labels to scramble over, or None: the
    resonant ground manifold); ``samples`` must resolve the photon rate's oscillations for an accurate photon count."""
    b = model.build
    if b.space is not None:
        raise ValueError(
            "optical pumping runs on the internal-only model; the recoil is booked through the kernel"
        )
    if isinstance(initial, qt.Qobj):
        rho0 = initial
    else:
        rho0 = scrambled_initial_state(model, initial)
    times = np.linspace(0.0, duration_s, samples)
    trace = model.evolve(rho0, times)
    p_target = trace.population(list(target))
    ss = model.steadystate()
    ss_error = 1.0 - sum(ss.populations.get(lab, 0.0) for lab in target)
    final_pops = {lab: float(arr[-1]) for lab, arr in trace.populations.items()}
    approximations = list(b.approximations)
    heating: dict[int, float] = {}
    if crystal is not None:
        heating = pump_recoil_heating(model, trace, crystal, ion)
    return PumpingResult(
        target=tuple(target),
        trace=trace,
        final=trace.final,
        populations=final_pops,
        preparation_error=float(1.0 - p_target[-1]),
        steady_state_error=float(ss_error),
        time_to_reach_s=trace.time_to_reach(list(target), 1.0 - tolerance),
        tolerance=tolerance,
        photons_scattered=float(trace.photons_scattered[-1]),
        photons_per_line=trace.photons_per_line(),
        motional_heating_quanta=heating,
        approximations=tuple(approximations),
    )


def pump_recoil_heating(
    model: BlochModel, trace: PumpingTrace, crystal: Crystal, ion: int
) -> dict[int, float]:
    """Delta n_m = sum_k N_k alpha_q(k)(chi_m) eta_em,{i,m}^2: the emitted photons' mean recoil into every mode, with each
    channel's own wavenumber and polarization index."""
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


def pumping_time_scale_s(model: BlochModel) -> float:
    """1/(slowest nonzero Liouvillian rate): the exponential time scale of the pump's approach to its steady state."""
    b = model.build
    L = np.asarray(b.liouvillian().full())
    vals = np.linalg.eigvals(L)
    rates = np.sort(-vals.real)
    nonzero = rates[rates > 1e-9 * rates.max()]
    if nonzero.size == 0:
        return math.inf
    return float(1.0 / nonzero[0])


__all__ = [
    "LeakToQubit",
    "PumpingResult",
    "optical_pumping",
    "pump_recoil_heating",
    "pumping_time_scale_s",
    "scrambled_initial_state",
]
