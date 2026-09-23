"""GHZ-state fidelity on the simulated device (Sackett et al. 2000; Leibfried et al. 2005): P_0 = P(0...0) and
P_1 = P(1...1) after H and a CNOT chain, and the contrast C of the parity C cos(N phi + phi_0) + B with GPi2(phi) on every
qubit after it. (P_0 + P_1 + C)/2 is the fidelity with the best-phase GHZ state (|0...0> + e^{i theta}|1...1>)/sqrt 2,
so it is an UPPER bound on the fixed-phase GHZ fidelity; both exact register fidelities are reported beside it."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from qutip_trap.benchmarks.budget import (
    BenchmarkBudget,
    channels_for,
    gather_counts_and_intrinsic,
    spam_of,
)
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.experiments.fitting import weighted_fit
from qutip_trap.run.job import machine_with_run_kwargs, register_fidelity
from qutip_trap.run.results import Result

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.machine import Machine


def ghz_circuit(qubits: Sequence[int], n_qubits: int) -> Circuit:
    """H on the first qubit, then a CNOT chain along ``qubits``; measures ``qubits``."""
    qs = tuple(int(q) for q in qubits)
    ops: list[Operation] = [Operation("h", (qs[0],), ())]
    for a, b in zip(qs[:-1], qs[1:]):
        ops.append(Operation("cnot", (a, b), ()))
    return Circuit(n_qubits, tuple(ops), qs)


def parity_circuit(qubits: Sequence[int], n_qubits: int, phi_rad: float) -> Circuit:
    """The GHZ circuit followed by GPi2(phi) on every qubit."""
    base = ghz_circuit(qubits, n_qubits)
    ops = list(base.ops) + [Operation("gpi2", (q,), (float(phi_rad),)) for q in base.measure]
    return Circuit(n_qubits, tuple(ops), base.measure)


def ghz_coherence(result: Result) -> tuple[float, float, float]:
    """(rho_{0...0,0...0}, rho_{1...1,1...1}, |rho_{0...0,1...1}|) of the run's register state (``keep_final_state=True``),
    ion 0 the first tensor factor, the levels 0 and 1 of each qudit."""
    rho = result.final_state
    if rho is None:
        raise ValueError("run with keep_final_state=True to read the register state's GHZ coherence")
    dims = [int(x) for x in rho.dims[0]]
    mat = np.asarray(rho.full(), dtype=complex)
    i1 = 0
    stride = 1
    for d in reversed(dims):
        i1 += stride
        stride *= d
    return float(np.real(mat[0, 0])), float(np.real(mat[i1, i1])), float(abs(mat[0, i1]))


def parity_of(result: Result) -> tuple[float, float]:
    """(<prod Z>, sigma) from a result's histogram: sum_s (-1)^|s| P(s), sigma = sqrt((1 - Pi^2)/n_eff)."""
    par = 0.0
    for key, p in result.probabilities.items():
        par += (-1.0) ** key.count("1") * float(p)
    n_eff = max(float(result.diagnostics.effective_sample_size), 1.0)
    return par, math.sqrt(max(1.0 - par * par, 0.0) / n_eff) or 1.0 / n_eff


def fit_parity(
    phases: np.ndarray, parity: np.ndarray, sigma: np.ndarray, n_qubits: int
) -> tuple[dict[str, tuple[float, float]], bool]:
    """C cos(N phi + phi_0) + B by weighted least squares from several starting phases; the best chi^2 wins."""

    def model(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.asarray(float(p[0]) * np.cos(n_qubits * np.asarray(x) + float(p[1])) + float(p[2]))

    c0 = 0.5 * float(parity.max() - parity.min())
    best: tuple[float, Any] | None = None
    for guess in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        fit = weighted_fit(model, [max(c0, 1e-3), guess, float(parity.mean())], phases, parity, sigma=sigma)
        if best is None or fit.chi2_per_dof < best[0]:
            best = (fit.chi2_per_dof, fit)
    assert best is not None
    fit = best[1]
    c = float(fit.params[0])
    phi0 = float(fit.params[1]) + (math.pi if c < 0.0 else 0.0)
    out = {
        "contrast": (abs(c), float(fit.errors[0])),
        "phi0_rad": ((phi0 + math.pi) % (2.0 * math.pi) - math.pi, float(fit.errors[1])),
        "offset": fit.value(2),
        "chi2_per_dof": (fit.chi2_per_dof, 0.0),
    }
    return out, fit.converged


@dataclass(frozen=True)
class GHZResult:
    """The GHZ benchmark: the populations, the parity fringe against the analysis phase (radians) with its fit, the bound
    (P_0 + P_1 + C)/2, the two exact register fidelities, the budget and every ``Result`` behind them."""

    qubits: tuple[int, ...]
    shots: int
    analysis_phases_rad: tuple[float, ...]
    populations: dict[str, tuple[float, float]]
    """``P0`` = P(0...0) and ``P1`` = P(1...1) with their shot-noise sigmas."""
    parity: np.ndarray
    """(n_phases, 3): phase, parity, sigma."""
    fit: dict[str, tuple[float, float]]
    """contrast, phi0_rad, offset, chi2_per_dof."""
    fidelity_bound: tuple[float, float]
    """(P_0 + P_1 + C)/2 with its uncertainty, an estimate of ``register_fidelity_max_phase``."""
    register_fidelity: float
    """Exact <GHZ| rho |GHZ> of the populations run against the compiled circuit's own fixed-phase target."""
    register_fidelity_max_phase: float
    """Exact max_theta <GHZ_theta| rho |GHZ_theta> of the same state; never below ``register_fidelity``."""
    converged: bool
    results: tuple[Result, ...]
    """The populations run first, then one per analysis phase."""
    budget: BenchmarkBudget | None = None
    notes: tuple[str, ...] = ()

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)


def ghz_fidelity(
    machine: Machine | Device,
    qubits: Sequence[int],
    *,
    shots: int = 400,
    analysis_phases_rad: Sequence[float] | None = None,
    seed: int = 0,
    budget: bool = True,
    **run_kwargs: Any,
) -> GHZResult:
    """The GHZ benchmark on ``qubits`` (two or more), every point a ``Machine.run`` of ``shots``; the analysis phases
    default to eight over one parity period 2 pi/N. A bare ``Device`` (wrapped in a default machine) and legacy
    ``run_kwargs`` are accepted with a deprecation warning."""
    m = machine_with_run_kwargs(
        machine, run_kwargs, caller="qutip_trap.benchmarks.ghz.ghz_fidelity", stacklevel=2, call_keywords=True
    )
    device = m.device
    qs = tuple(int(q) for q in qubits)
    if len(qs) < 2 or len(set(qs)) != len(qs):
        raise ValueError("a GHZ state needs at least two distinct qubits")
    n_ions = device.crystal.n_ions
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
    seed0 = int(root.child(0, 0, 0, 0, "ghz_runs").generate_state(1)[0])
    res_pop = m.run(ghz_circuit(qs, n_ions), shots, seed=seed0, keep_final_state=True)
    key0, key1 = "0" * n, "1" * n
    p0 = float(res_pop.probabilities.get(key0, 0.0))
    p1 = float(res_pop.probabilities.get(key1, 0.0))
    s0 = float(res_pop.error_bars.get(key0, 1.0 / shots))
    s1 = float(res_pop.error_bars.get(key1, 1.0 / shots))
    results = [res_pop]
    rows: list[tuple[float, float, float]] = []
    for k, phi in enumerate(phases):
        seed_k = int(root.child(0, 0, k + 1, 0, "ghz_runs").generate_state(1)[0])
        res = m.run(parity_circuit(qs, n_ions, phi), shots, seed=seed_k)
        par, sg = parity_of(res)
        rows.append((phi, par, sg))
        results.append(res)
    parity = np.array(rows)
    fit, converged = fit_parity(parity[:, 0], parity[:, 1], parity[:, 2], n)
    c, sc = fit["contrast"]
    f_bound = 0.5 * (p0 + p1 + c)
    s_bound = 0.5 * math.sqrt(s0 * s0 + s1 * s1 + sc * sc)
    f_exact = register_fidelity(res_pop)
    rho00, rho11, coherence = ghz_coherence(res_pop)
    f_max_phase = 0.5 * (rho00 + rho11) + coherence
    notes: list[str] = [
        "(P_0 + P_1 + C)/2 = max_theta <GHZ_theta| rho |GHZ_theta> exactly, so it estimates"
        f" register_fidelity_max_phase ({f_max_phase:.5f} here) and is an UPPER bound on the fixed-phase"
        f" register_fidelity ({f_exact:.5f} here), not a lower one (conv.ghz_parity_bound)"
    ]
    bud: BenchmarkBudget | None = None
    if budget:
        counts, intrinsic = gather_counts_and_intrinsic([res_pop], [1])
        spam = spam_of(res_pop, qs)
        kinds = [name for name in counts if not name.startswith("total")]
        channels, infid = channels_for(m, kinds, qs)
        f_gates = 1.0
        for kind in kinds:
            f_gates *= (1.0 - infid[kind]) ** counts[kind]
        eps_ro = [0.5 * (spam.get(f"q{q}", (0.0, 0.0))[0] + spam.get(f"q{q}", (0.0, 0.0))[1]) for q in qs]
        pop_factor = 1.0 - sum(eps_ro)
        contrast_factor = float(np.prod([1.0 - 2.0 * e for e in eps_ro]))
        predicted = {
            "F_gates": f_gates,
            "register_fidelity": f_gates,
            "P0_plus_P1": f_gates * pop_factor,
            "contrast": f_gates * contrast_factor,
            "fidelity_bound": 0.5 * f_gates * (pop_factor + contrast_factor),
            "intrinsic_total": float(intrinsic.get("total", 0.0)),
        }
        bud = BenchmarkBudget(
            unit="circuit",
            qubits=qs,
            counts=counts,
            intrinsic=intrinsic,
            spam=spam,
            channels=channels,
            channel_infidelity=infid,
            predicted=predicted,
            notes=(
                "F_gates = prod over the GHZ circuit's native pieces of (1 - reduced average infidelity of the kind's GATE_LOCAL "
                "channel): the crosstalk-free product floor of Section 7.9 with the simulator's own channels",
                "P0_plus_P1 = F_gates (1 - sum_q eps_q) and contrast = F_gates prod_q (1 - 2 eps_q) with eps_q = (eps_B + eps_D)/2:"
                " the readout loss of the populations and of the parity",
                "intrinsic_total = the Section 9.6 closed-form scales of the whole circuit, summed over every native"
                " piece's whole schedule entry (the crosstalk on ions outside the benchmarked set included), so it bounds"
                " a larger error than F_gates measures",
            ),
        )
    return GHZResult(
        qubits=qs,
        shots=shots,
        analysis_phases_rad=phases,
        populations={"P0": (p0, s0), "P1": (p1, s1)},
        parity=parity,
        fit=fit,
        fidelity_bound=(f_bound, s_bound),
        register_fidelity=float(f_exact),
        register_fidelity_max_phase=float(f_max_phase),
        converged=converged,
        results=tuple(results),
        budget=bud,
        notes=tuple(notes),
    )


__all__ = [
    "GHZResult",
    "fit_parity",
    "ghz_circuit",
    "ghz_coherence",
    "ghz_fidelity",
    "parity_circuit",
    "parity_of",
]
