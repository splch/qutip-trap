"""GHZ-state fidelity on the simulated device (PLAN.md Section 10 M10; Section 7.9 "Entangling gate"; Section 9.6 rows 1 and 2).

The laboratory protocol (Sackett et al. 2000; Leibfried et al. 2005; Wright et al. 2019 for the two-qubit case in Section
7.9): prepare the GHZ state with the circuit H(q_0), CNOT(q_0, q_1), ..., CNOT(q_{N-2}, q_{N-1}) and measure the populations
P_0 = P(0...0) and P_1 = P(1...1); then, on fresh shots, follow the same circuit by a pi/2 analysis pulse GPi2(phi) on every
qubit and measure the parity Pi(phi) = <prod_q Z_q>, which oscillates as C cos(N phi + phi_0) with the contrast C = 2 |rho_{0...0,
1...1}|; the fidelity with the ideal GHZ state is bounded from below by F >= (P_0 + P_1 + C)/2 (an equality for the ideal
state, a lower bound in general). Every point is one ``run`` of ``shots`` (the analysis pulses are ordinary native gates of the
circuit, so their phases are frame-propagated by the compiler exactly as the laboratory's are), and the simulator adds what
no laboratory has: the exact register fidelity <GHZ| rho |GHZ> of the recombined register state (``register_fidelity``).

The budget alongside: the Section 9.6 closed-form scales of the circuit, the Section 6.8 channels of its native gate kinds
composed into the product floor F_gates = prod_gates (1 - eps_gate) (the crosstalk-free composite of Wright et al. 2019 in
Section 7.9, with the simulator's own per-gate channels in place of the published fidelities), and the readout loss
(P_0 + P_1) x (1 - sum_q eps_q) and C x prod_q (1 - 2 eps_q) with eps_q = (eps_B + eps_D)/2 per qubit.
"""

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
from qutip_trap.run.job import register_fidelity, run
from qutip_trap.run.results import Result

if TYPE_CHECKING:
    from qutip_trap.device.model import Device


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
    """F >= (P_0 + P_1 + C)/2 with its uncertainty: the laboratory's number."""
    register_fidelity: float
    """<GHZ| rho |GHZ> of the exact register state of the populations run (simulator only)."""
    converged: bool
    results: tuple[Result, ...]
    """The populations run first, then one per analysis phase."""
    budget: BenchmarkBudget | None = None
    notes: tuple[str, ...] = ()

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)


def ghz_fidelity(
    device: Device,
    qubits: Sequence[int],
    *,
    shots: int = 400,
    analysis_phases_rad: Sequence[float] | None = None,
    seed: int = 0,
    budget: bool = True,
    **run_kwargs: Any,
) -> GHZResult:
    """The GHZ benchmark of the module docstring on ``qubits`` (two or more), every point a ``run`` of ``shots``.

    ``analysis_phases_rad`` default to eight phases over one period 2 pi/N of the parity oscillation. ``run_kwargs`` go to
    ``run``; ``keep_final_state`` is forced on for the populations run (the exact register fidelity)."""
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
    kw = {k: v for k, v in run_kwargs.items() if k != "keep_final_state"}
    seed0 = int(root.child(0, 0, 0, 0, "ghz_runs").generate_state(1)[0])
    res_pop = run(ghz_circuit(qs, n_ions), device, shots, seed=seed0, keep_final_state=True, **kw)
    key0, key1 = "0" * n, "1" * n
    p0 = float(res_pop.probabilities.get(key0, 0.0))
    p1 = float(res_pop.probabilities.get(key1, 0.0))
    s0 = float(res_pop.error_bars.get(key0, 1.0 / shots))
    s1 = float(res_pop.error_bars.get(key1, 1.0 / shots))
    results = [res_pop]
    rows: list[tuple[float, float, float]] = []
    for k, phi in enumerate(phases):
        seed_k = int(root.child(0, 0, k + 1, 0, "ghz_runs").generate_state(1)[0])
        res = run(parity_circuit(qs, n_ions, phi), device, shots, seed=seed_k, **kw)
        par, sg = parity_of(res)
        rows.append((phi, par, sg))
        results.append(res)
    parity = np.array(rows)
    fit, converged = fit_parity(parity[:, 0], parity[:, 1], parity[:, 2], n)
    c, sc = fit["contrast"]
    f_bound = 0.5 * (p0 + p1 + c)
    s_bound = 0.5 * math.sqrt(s0 * s0 + s1 * s1 + sc * sc)
    f_exact = register_fidelity(res_pop)
    notes: list[str] = []
    bud: BenchmarkBudget | None = None
    if budget:
        counts, intrinsic = gather_counts_and_intrinsic([res_pop], [1])
        spam = spam_of(res_pop, qs)
        kinds = [name for name in counts if not name.startswith("total")]
        channels, infid = channels_for(device, kinds, qs, **kw)
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
                "intrinsic_total = the Section 9.6 closed-form scales of the whole circuit",
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
        converged=converged,
        results=tuple(results),
        budget=bud,
        notes=tuple(notes),
    )


__all__ = ["GHZResult", "fit_parity", "ghz_circuit", "ghz_fidelity", "parity_circuit", "parity_of"]
