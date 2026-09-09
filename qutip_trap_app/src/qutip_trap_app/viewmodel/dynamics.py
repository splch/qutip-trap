"""Level 3, the dynamics: inside one pulse for one dynamical sample (PLAN.md Section 14.2, row 3).

Reads a :class:`ZoomTrace` (a re-simulated gate step, :mod:`qutip_trap_app.resim`): qubit populations and coherences
against time, <n_m>(t), the phase-space trajectories <a_m>(t), the Fock distributions at the step's start and end, the
jumps, the quasi-static noise values drawn for the sample, and for two qubits the concurrence and the Pauli correlators
against time. What the core does not expose is named in ``Record.core_gaps`` rather than drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import Record, ZoomTrace
from qutip_trap_app.viewmodel.catalogue import Shown
from qutip_trap_app.viewmodel.circuit import concurrence, pauli_expectations, single_ion_reduced


@dataclass(frozen=True)
class Series:
    quantity: str
    label: str
    times_s: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        Shown(self.quantity, None)  # validates the catalogue id


@dataclass(frozen=True)
class Loop:
    mode: int
    alpha: np.ndarray
    """Complex <a_m>(t)."""
    quantity: str = "alpha_m"

    @property
    def closes(self) -> float:
        """|alpha(end) - alpha(start)|: how far the loop failed to close."""
        return float(abs(self.alpha[-1] - self.alpha[0])) if self.alpha.size else 0.0


@dataclass(frozen=True)
class PulseDynamics:
    step_index: int
    gate_id: str
    t_start_s: float
    t_end_s: float
    times_s: np.ndarray
    populations: tuple[Series, ...]
    coherences: tuple[Series, ...]
    nbar: tuple[Series, ...]
    loops: tuple[Loop, ...]
    fock_start: dict[int, np.ndarray]
    fock_end: dict[int, np.ndarray]
    pauli: tuple[Series, ...]
    concurrence: Series | None
    jumps: tuple[Shown, ...]
    noise_values: tuple[Shown, ...]
    boundary_population: tuple[Shown, ...]
    debye_waller: tuple[Shown, ...]
    engine: tuple[str, ...]
    wall_time: Shown
    norm_deficit: Shown
    """1 - Tr rho at the end of the step: the integrator's norm drift (normalize_output is off, Section 5.3), a numerics fact shown rather than hidden."""
    unavailable: tuple[str, ...]


def pulse_dynamics(record: Record, z: ZoomTrace) -> PulseDynamics:
    tr = z.trace
    step = record.step(z.step_index)
    n = record.n_qubits
    t = tr.times_s
    pops = tuple(
        Series("P1", f"P1 ion {i}", t, np.asarray(np.real(tr.expectations[f"P1[{i}]"]), dtype=float))
        for i in range(n)
        if f"P1[{i}]" in tr.expectations
    )
    coh: list[Series] = []
    if tr.reduced_internal.size and all(d == 2 for d in tr.ion_dims):
        for i in range(n):
            vals = [abs(single_ion_reduced(rho, i, n)[0, 1]) for rho in tr.reduced_internal]
            coh.append(Series("coherence", f"|rho_01| ion {i}", t, np.asarray(vals, dtype=float)))
    nbar = tuple(
        Series("mode_nbar", f"<n> mode {m}", t, np.asarray(v, dtype=float))
        for m, v in sorted(tr.mode_nbar.items())
    )
    loops = tuple(Loop(m, np.asarray(a, dtype=complex)) for m, a in sorted(tr.alpha_m.items()))
    pauli: list[Series] = []
    conc: Series | None = None
    if n == 2 and tr.reduced_internal.size and all(d == 2 for d in tr.ion_dims):
        keys = ("XX", "YY", "ZZ", "ZI", "IZ")
        cols: dict[str, list[float]] = {k: [] for k in keys}
        cvals: list[float] = []
        for rho in tr.reduced_internal:
            pe = pauli_expectations(rho, 2)
            for k in keys:
                cols[k].append(pe[k])
            cvals.append(concurrence(rho))
        pauli = [
            Series("pauli_expectation", f"<{k}>", t, np.asarray(v, dtype=float)) for k, v in cols.items()
        ]
        conc = Series("concurrence", "concurrence", t, np.asarray(cvals, dtype=float))
    sample = record.noise_samples[z.sample_index]
    noise_vals = tuple(Shown("noise_sample_value", v, k) for k, v in sorted(sample.values.items()))
    dw = tuple(
        Shown("debye_waller", c[0], f"frozen mode {m}: |alpha|^2 (2 nbar + 1); chi loss {c[1]:.3g} rad")
        for m, c in sorted(record.diagnostics.frozen_contribution.items())
    )
    unavailable = [
        "per-time Fock distributions inside the pulse (core gap: Traces carries <n_m>(t) only; start and end shown)",
    ]
    if n != 2:
        unavailable.append("concurrence and Pauli correlators are computed for two-qubit registers")
    return PulseDynamics(
        step_index=z.step_index,
        gate_id=step.gate_id,
        t_start_s=step.t_start_s,
        t_end_s=step.t_end_s,
        times_s=t,
        populations=pops,
        coherences=tuple(coh),
        nbar=nbar,
        loops=loops,
        fock_start=dict(z.fock_start),
        fock_end=dict(z.fock_end),
        pauli=tuple(pauli),
        concurrence=conc,
        jumps=tuple(Shown("jump", t_j, channel) for t_j, channel in tr.jumps),
        noise_values=noise_vals,
        boundary_population=tuple(
            Shown("boundary_population", v, f"mode {m}") for m, v in sorted(tr.boundary_population.items())
        ),
        debye_waller=dw,
        engine=z.integrators + (z.method,),
        wall_time=Shown("wall_time", z.wall_time_s, f"re-simulation at dimension {z.engine_dimension}"),
        norm_deficit=Shown(
            "norm_deficit", 1.0 - float(np.real(np.trace(tr.final_internal))), "at the step's end"
        ),
        unavailable=tuple(unavailable),
    )


__all__ = ["Loop", "PulseDynamics", "Series", "pulse_dynamics"]
