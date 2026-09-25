"""Optical pumping and internal-state preparation from first principles (PLAN.md Section 4.2.6; milestone M3).

The pumping beams' polarization-resolved Rabi frequencies, the branching ratios of Section 4.5.2 and the frame of
Section 4.2.8 are the ``BlochModel`` of M3a; its time evolution from the scrambled state Doppler cooling leaves (a
uniform mixture over the ground manifold the cooling light drove, by default) gives the residual population in
non-target states (the SPAM preparation error of Section 8), the pumping time, the mean number of scattered photons
and, through the recoil kernel with the ion's participation (``light.recoil``), the motional heating of the pump
(Section 4.2.8: a few photons at k x0 = 0.053 leave Delta n ~ 3e-3, the quantitative content of the sources' "Delta n ~ 0").

Species anchors (Section 4.2.6): 171Yb+ pumps to |F=0, m=0> with a 2.1 GHz sideband on S1/2 F=1 -> P1/2 F=1 (1/3 branching
into |0> per excitation, three photons; the M3a test), 9Be+ into |2,2> in about 7 us, 43Ca+ prepares the stretch state
with error below 1e-4. The stage order (Doppler first, pump last) is enforced by ``prep.sequence``, never here.
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
    """1 - P(target) at the end of the pump: the SPAM preparation error of Section 8."""
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
        """The 2 x 2 state on the qubit pair (index 0 the lower level, Section 13 computational ordering).

        Population outside the pair (leaked Zeeman sublevels) is counted as the upper qubit state (``to_upper``, the worst
        case for a preparation-error budget), folded back proportionally (``renormalize``), or refused (``full``, which
        asks the caller for a d > 2 register factor instead).
        """
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
        # the pumped state carries no coherence between the qubit levels (the pump erases it): a diagonal state
        return qt.Qobj(np.diag([p_lower, p_upper]), dims=[[2], [2]])

    def qudit_density_matrix(self, labels: Sequence[str]) -> qt.Qobj:
        """The d x d diagonal state on a register factor with the given level labels (``noise/levels.py``; M7): every resolved
        atomic label takes its pumped population, the SINK (if present) the remainder, and without a SINK the remainder is
        counted as the upper qubit level (the ``to_upper`` policy)."""
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
    """Evolve the internal state under the pumping beams for ``duration`` and read off the preparation error, time, photons and recoil.

    ``initial``: a density matrix on the model's internal factor, state labels to scramble over, or None (the resonant ground
    manifold). The sampling must resolve the coherent oscillations of the photon rate for an accurate photon count (the M3a
    finding: 5 ns steps; 100 ns aliased 3.00 photons to 2.83), hence the default of 4001 samples.
    """
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
    """Delta n_m = sum_operators N_k alpha_q(k)(chi_m) eta_em,{i,m}^2: each emitted photon's mean recoil into every mode, with the
    channel's own wavenumber and polarization index and the ion's participation (Section 4.2.8, "Sizes")."""
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
    """1/(slowest nonzero Liouvillian rate) of the pump: the exponential time scale of the approach to the target.

    A property of the Liouvillian alone, so it takes no target state (the dead ``target`` argument is dropped)."""
    b = model.build
    L = np.asarray(b.liouvillian().full())
    vals = np.linalg.eigvals(L)
    rates = np.sort(-vals.real)
    nonzero = rates[rates > 1e-9 * rates.max()]
    if nonzero.size == 0:
        return math.inf
    return float(1.0 / nonzero[0])
