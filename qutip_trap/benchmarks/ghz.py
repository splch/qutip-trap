"""GHZ-state fidelity on the simulated device (PLAN.md Section 7.9 "Entangling gate").

The laboratory protocol (Sackett et al. 2000; Leibfried et al. 2005; Wright et al. 2019): prepare the GHZ state with H(q_0),
CNOT(q_0, q_1), ..., CNOT(q_{N-2}, q_{N-1}) and measure P_0 = P(0...0) and P_1 = P(1...1); on fresh shots follow the same
circuit by a pi/2 analysis pulse GPi2(phi) on every qubit and measure the parity Pi(phi) = <prod_q Z_q>, which oscillates as
C cos(N phi + phi_0) with C = 2 |rho_{0...0,1...1}| (only the sigma_+^{(x)N} term carries e^{iN phi}).

Since P_0, P_1 and C/2 are the corner elements of rho,

    (P_0 + P_1 + C)/2 = (rho_{0...0,0...0} + rho_{1...1,1...1})/2 + |rho_{0...0,1...1}|
                      = max_theta <GHZ_theta| rho |GHZ_theta>,   |GHZ_theta> = (|0...0> + e^{i theta}|1...1>)/sqrt 2,

an equality with the phase-optimised GHZ fidelity and therefore an UPPER bound on the fidelity against any fixed-phase GHZ
state (``conv.ghz_parity_bound``). The result reports both exact numbers the simulator can read off the register state:
``register_fidelity_max_phase``, which the bound estimates, and ``register_fidelity`` against the compiled circuit's own
fixed-phase target. Every point is one ``run`` of ``shots``; the analysis pulses are native gates of the circuit, so their
phases are frame-propagated by the compiler as a laboratory's are.

The budget alongside: the intrinsic scales of the circuit, the channels of its native gate kinds composed into the product
floor F_gates = prod_gates (1 - eps_gate) (the crosstalk-free composite of Wright et al. 2019), and the readout losses
(P_0 + P_1)(1 - sum_q eps_q) and C prod_q (1 - 2 eps_q) with eps_q = (eps_B + eps_D)/2 per qubit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.benchmarks.budget import (
    BenchmarkBudget,
    _shot_sigma,
    channels_for,
    gather_counts_and_intrinsic,
    spam_of,
)
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.experiments.fitting import fit_fringe
from qutip_trap.run.job import register_fidelity
from qutip_trap.run.results import Result

if TYPE_CHECKING:
    from qutip_trap.machine import Machine


def ghz_circuit(qubits: Sequence[int], n_qubits: int) -> Circuit:
    """H on the first qubit, then a CNOT chain along ``qubits``; measures ``qubits``."""
    qs = tuple(int(q) for q in qubits)
    ops = [Operation("h", (qs[0],), ()), *(Operation("cnot", (a, b), ()) for a, b in zip(qs[:-1], qs[1:]))]
    return Circuit(n_qubits, tuple(ops), qs)


def parity_circuit(qubits: Sequence[int], n_qubits: int, phi_rad: float) -> Circuit:
    """The GHZ circuit followed by GPi2(phi) on every qubit."""
    base = ghz_circuit(qubits, n_qubits)
    return Circuit(
        n_qubits,
        (*base.ops, *(Operation("gpi2", (q,), (float(phi_rad),)) for q in base.measure)),
        base.measure,
    )


def ghz_coherence(result: Result) -> tuple[float, float, float]:
    """(rho_{0...0,0...0}, rho_{1...1,1...1}, |rho_{0...0,1...1}|) of the run's recombined register state
    (``keep_final_state=True``), in the register order (ion 0 the first tensor factor; the qubit levels of a qudit
    register)."""
    rho = result.final_state
    if rho is None:
        raise ValueError("run with keep_final_state=True to read the register state's GHZ coherence")
    mat = np.asarray(rho.full(), dtype=complex)
    i1 = 0
    stride = 1
    for d in reversed([int(x) for x in rho.dims[0]]):
        i1 += stride
        stride *= d
    return float(np.real(mat[0, 0])), float(np.real(mat[i1, i1])), float(abs(mat[0, i1]))


def parity_of(result: Result) -> tuple[float, float]:
    """(<prod Z>, sigma) from a result's histogram: sum_s (-1)^|s| P(s), sigma = sqrt((1 - Pi^2)/n_eff)."""
    par = sum((-1.0) ** key.count("1") * float(p) for key, p in result.probabilities.items())
    return par, _shot_sigma(result, 1.0 - par * par)


@dataclass(frozen=True)
class GHZResult:
    """The GHZ-state benchmark (module docstring): the populations P0 = P(0...0) and P1 = P(1...1) with their shot-noise
    sigmas, the parity fringe (n_phases, 3: phase, parity, sigma) with its fit (contrast, phi0_rad, offset, chi2_per_dof),
    the laboratory bound (P_0 + P_1 + C)/2 with its uncertainty (an upper bound on the fixed-phase fidelity), the two exact
    register fidelities of the populations run, the budget alongside and every ``Result``, the populations run first."""

    qubits: tuple[int, ...]
    shots: int
    analysis_phases_rad: tuple[float, ...]
    populations: dict[str, tuple[float, float]]
    parity: np.ndarray
    fit: dict[str, tuple[float, float]]
    fidelity_bound: tuple[float, float]
    register_fidelity: float
    """<GHZ| rho |GHZ> against the compiled circuit's own fixed-phase target."""
    register_fidelity_max_phase: float
    """max_theta <GHZ_theta| rho |GHZ_theta>, the quantity ``fidelity_bound`` estimates; never below ``register_fidelity``."""
    converged: bool
    results: tuple[Result, ...]
    budget: BenchmarkBudget | None = None
    notes: tuple[str, ...] = ()

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)


def ghz_fidelity(
    machine: Machine,
    qubits: Sequence[int],
    *,
    shots: int = 400,
    analysis_phases_rad: Sequence[float] | None = None,
    seed: int = 0,
    budget: bool = True,
) -> GHZResult:
    """The GHZ benchmark of the module docstring on ``qubits`` (two or more), every point a ``Machine.run`` of ``shots``;
    ``analysis_phases_rad`` default to eight phases over one period 2 pi/N of the parity oscillation, and the populations
    run keeps its final state for the exact register fidelities."""
    qs = tuple(int(q) for q in qubits)
    if len(qs) < 2 or len(set(qs)) != len(qs):
        raise ValueError("a GHZ state needs at least two distinct qubits")
    n_ions = machine.device.crystal.n_ions
    if any(q < 0 or q >= n_ions for q in qs):
        raise ValueError("qubits index the device's ions")
    if shots < 1:
        raise ValueError("shots is positive")
    n = len(qs)
    phases = (
        tuple(float(x) for x in np.linspace(0.0, 2.0 * math.pi / n, 8, endpoint=False))
        if analysis_phases_rad is None
        else tuple(float(x) for x in analysis_phases_rad)
    )
    if len(phases) < 4:
        raise ValueError("at least four analysis phases for the three-parameter parity fit")
    root = SeedSpec(int(seed))

    def run_seed(k: int) -> int:
        return int(root.child(0, 0, k, 0, "ghz_runs").generate_state(1)[0])

    res_pop = machine.run(ghz_circuit(qs, n_ions), shots, seed=run_seed(0), keep_final_state=True)
    key0, key1 = "0" * n, "1" * n
    p0 = float(res_pop.probabilities.get(key0, 0.0))
    p1 = float(res_pop.probabilities.get(key1, 0.0))
    s0 = float(res_pop.error_bars.get(key0, 1.0 / shots))
    s1 = float(res_pop.error_bars.get(key1, 1.0 / shots))
    results = [res_pop]
    rows: list[tuple[float, float, float]] = []
    for k, phi in enumerate(phases):
        res = machine.run(parity_circuit(qs, n_ions, phi), shots, seed=run_seed(k + 1))
        rows.append((phi, *parity_of(res)))
        results.append(res)
    parity = np.array(rows)
    fringe = fit_fringe(parity[:, 0], parity[:, 1], parity[:, 2], n)
    fit = {
        "contrast": fringe.contrast,
        "phi0_rad": fringe.phase_rad,
        "offset": fringe.offset,
        "chi2_per_dof": (fringe.chi2_per_dof, 0.0),
    }
    c, sc = fringe.contrast
    f_exact = register_fidelity(res_pop)
    rho00, rho11, coherence = ghz_coherence(res_pop)
    f_max_phase = 0.5 * (rho00 + rho11) + coherence
    notes: list[str] = [
        "(P_0 + P_1 + C)/2 = max_theta <GHZ_theta| rho |GHZ_theta> exactly, so it estimates"
        f" register_fidelity_max_phase ({f_max_phase:.5f} here) and is an UPPER bound on the fixed-phase"
        f" register_fidelity ({f_exact:.5f} here), not a lower one (conv.ghz_parity_bound)"
    ]
    return GHZResult(
        qubits=qs,
        shots=shots,
        analysis_phases_rad=phases,
        populations={"P0": (p0, s0), "P1": (p1, s1)},
        parity=parity,
        fit=fit,
        fidelity_bound=(0.5 * (p0 + p1 + c), 0.5 * math.sqrt(s0 * s0 + s1 * s1 + sc * sc)),
        register_fidelity=float(f_exact),
        register_fidelity_max_phase=float(f_max_phase),
        converged=fringe.converged,
        results=tuple(results),
        budget=_ghz_budget(machine, res_pop, qs) if budget else None,
        notes=tuple(notes),
    )


def _ghz_budget(machine: Machine, res_pop: Result, qs: tuple[int, ...]) -> BenchmarkBudget:
    """The GHZ budget: the product floor of the circuit's channels and the readout losses (module docstring)."""
    counts, intrinsic = gather_counts_and_intrinsic([res_pop], [1])
    spam = spam_of(res_pop, qs)
    channels, infid = channels_for(machine, list(counts), qs)
    f_gates = 1.0
    for kind in counts:
        f_gates *= (1.0 - infid[kind]) ** counts[kind]
    eps_ro = [0.5 * (spam.get(f"q{q}", (0.0, 0.0))[0] + spam.get(f"q{q}", (0.0, 0.0))[1]) for q in qs]
    pop_factor = 1.0 - sum(eps_ro)
    contrast_factor = float(np.prod([1.0 - 2.0 * e for e in eps_ro]))
    return BenchmarkBudget(
        unit="circuit",
        qubits=qs,
        counts=counts,
        intrinsic=intrinsic,
        spam=spam,
        channels=channels,
        channel_infidelity=infid,
        predicted={
            "F_gates": f_gates,
            "P0_plus_P1": f_gates * pop_factor,
            "contrast": f_gates * contrast_factor,
            "fidelity_bound": 0.5 * f_gates * (pop_factor + contrast_factor),
            "intrinsic_total": float(intrinsic["total"]),
        },
        notes=(
            "F_gates = prod over the GHZ circuit's native pieces of (1 - reduced average infidelity of the kind's GATE_LOCAL "
            "channel): the crosstalk-free product floor of Section 7.9 with the simulator's own channels",
            "P0_plus_P1 = F_gates (1 - sum_q eps_q) and contrast = F_gates prod_q (1 - 2 eps_q) with eps_q = (eps_B + eps_D)/2:"
            " the readout loss of the populations and of the parity",
            "intrinsic_total = the closed-form intrinsic scales of the whole circuit, summed over every native"
            " piece's whole schedule entry (the crosstalk on ions outside the benchmarked set included), so it bounds"
            " a larger error than F_gates measures",
        ),
    )
