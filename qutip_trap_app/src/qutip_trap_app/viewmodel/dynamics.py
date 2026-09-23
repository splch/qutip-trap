"""Level 3, the dynamics: inside one gate step for one sample and one branch, from its fine re-simulation or, until
then, from the run's own stored points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap_app import core
from qutip_trap_app.record import BoundaryState, ProcessMatrixRecord, Record, TraceRecord, ZoomTrace
from qutip_trap_app.resim import boundary_key
from qutip_trap_app.viewmodel.circuit import concurrence, pauli_expectations, single_ion_reduced
from qutip_trap_app.viewmodel.shown import Shown

FOCK_FLOOR = 1e-6
"""Fock populations below this are not drawn."""


@dataclass(frozen=True)
class Series:
    label: str
    times_s: np.ndarray
    values: np.ndarray

    def last(self, unit: str = "") -> Shown:
        return Shown(f"{self.label} at the end", float(self.values[-1]) if self.values.size else None, unit)


@dataclass(frozen=True)
class Loop:
    """One phase-space curve: a spin-branch loop of the played waveform (``ion`` set), or the trace's spin-averaged
    <a_m>(t) (``ion`` None), in which the branches' displacements cancel."""

    mode: int
    alpha: np.ndarray
    label: str
    ion: int | None = None
    chi_m_rad: float | None = None
    chi_closed_form_rad: float | None = None

    @property
    def closes(self) -> float:
        """|alpha(end) - alpha(start)|: how far the loop failed to close."""
        return float(abs(self.alpha[-1] - self.alpha[0])) if self.alpha.size else 0.0

    @property
    def excursion(self) -> float:
        """max |alpha(t) - alpha(start)|: the loop's reach."""
        return float(np.max(np.abs(self.alpha - self.alpha[0]))) if self.alpha.size else 0.0


@dataclass(frozen=True)
class PulseDynamics:
    step_index: int
    gate_id: str
    times_s: np.ndarray
    populations: tuple[Series, ...]
    coherences: tuple[Series, ...]
    nbar: tuple[Series, ...]
    loops: tuple[Loop, ...]
    """The played waveform's spin-branch loops, one per (ion, mode); empty for a step with no entangling waveform."""
    mean_alpha: tuple[Loop, ...]
    fock_start: dict[int, np.ndarray]
    fock_end: dict[int, np.ndarray]
    pauli: tuple[Series, ...]
    concurrence: Series | None
    jumps: tuple[tuple[float, str], ...]
    sample_values: tuple[Shown, ...]
    boundary_population: tuple[Shown, ...]
    frozen: tuple[Shown, ...]
    wall_time: Shown
    norm_deficit: Shown | None
    """1 - Tr rho at the end of the step: the integrator's norm drift; None when the trace stores no register state."""


def pulse_dynamics(record: Record, z: ZoomTrace) -> PulseDynamics:
    tr = z.trace
    step = record.step(z.step_index)
    n = record.n_ions
    t = tr.times_s
    two_level = tr.reduced_internal.size > 0 and all(d == 2 for d in record.space.ion_dims)
    coherences = (
        tuple(
            Series(
                f"|rho_01| ion {i}",
                t,
                np.array([abs(single_ion_reduced(rho, i, n)[0, 1]) for rho in tr.reduced_internal]),
            )
            for i in range(n)
        )
        if two_level
        else ()
    )
    pauli: tuple[Series, ...] = ()
    conc: Series | None = None
    if n == 2 and two_level:
        keys = ("XX", "YY", "ZZ", "ZI", "IZ")
        pe = [pauli_expectations(rho, 2) for rho in tr.reduced_internal]
        pauli = tuple(Series(f"<{k}>", t, np.array([p[k] for p in pe])) for k in keys)
        conc = Series("concurrence", t, np.array([concurrence(rho) for rho in tr.reduced_internal]))
    deficit = (
        None
        if tr.reduced_internal.size == 0
        else Shown(
            "Norm deficit",
            1.0 - float(np.real(np.trace(tr.reduced_internal[-1]))),
            detail="1 - Tr rho at the step's end: the integrator runs unnormalized",
        )
    )
    return PulseDynamics(
        step_index=z.step_index,
        gate_id=step.gate_id,
        times_s=t,
        populations=tuple(
            Series(f"P1 ion {i}", t, np.real(np.asarray(tr.expectations[f"P1[{i}]"])))
            for i in range(n)
            if f"P1[{i}]" in tr.expectations
        ),
        coherences=coherences,
        nbar=tuple(Series(f"<n> mode {m}", t, v) for m, v in sorted(tr.mode_nbar.items())),
        loops=tuple(
            Loop(
                lp.mode,
                lp.alpha,
                f"ion {lp.ion}, mode {lp.mode}",
                ion=lp.ion,
                chi_m_rad=lp.chi_m_rad,
                chi_closed_form_rad=lp.chi_closed_form_rad,
            )
            for lp in record.branch_loops
            if lp.gate_id == step.gate_id
        ),
        mean_alpha=tuple(Loop(m, a, f"<a_{m}> spin-averaged") for m, a in sorted(tr.alpha_m.items())),
        fock_start=dict(z.fock_start),
        fock_end=dict(z.fock_end),
        pauli=pauli,
        concurrence=conc,
        jumps=tr.jumps,
        sample_values=tuple(Shown(k, v) for k, v in sorted(record.sample_values[z.sample_index].items())),
        boundary_population=tuple(
            Shown(f"Population at the cap, mode {m}", v) for m, v in sorted(tr.boundary_population.items())
        ),
        frozen=tuple(
            Shown(f"Frozen mode {m}", c[0], detail=f"|alpha|^2 (2 nbar + 1); lost angle {c[1]:.3g} rad")
            for m, c in sorted(record.diagnostics.frozen_contribution.items())
        ),
        wall_time=Shown("Wall time", z.wall_time_s, "s", f"at dimension {z.engine_dimension}"),
        norm_deficit=deficit,
    )


def recorded_zoom(record: Record, step_index: int, sample_index: int = 0, branch: int = 0) -> ZoomTrace:
    """The run's own stored points inside one step, shaped as a zoom, so Level 3 opens before any re-simulation. Raises
    ``KeyError`` when the record has no such step or no trace for the (sample, branch)."""
    if not 0 <= step_index < len(record.schedule.steps):
        raise KeyError(f"the record has no step {step_index} ({len(record.schedule.steps)} steps)")
    step = record.step(step_index)
    tr = record.trace(sample_index, branch)
    keep = np.flatnonzero((tr.times_s >= step.t_start_s - 1e-12) & (tr.times_s <= step.t_end_s + 1e-12))
    if keep.size == 0:
        raise KeyError(f"the run stored no point inside step {step_index}")
    sliced = TraceRecord(
        sample_index=tr.sample_index,
        branch=tr.branch,
        weight=tr.weight,
        times_s=tr.times_s[keep],
        expectations={k: np.asarray(v)[keep] for k, v in tr.expectations.items()},
        reduced_internal=tr.reduced_internal[keep] if tr.reduced_internal.size else tr.reduced_internal,
        mode_nbar={m: v[keep] for m, v in tr.mode_nbar.items()},
        alpha_m={m: v[keep] for m, v in tr.alpha_m.items()},
        jumps=tuple(j for j in tr.jumps if step.t_start_s <= j[0] <= step.t_end_s),
        boundary_population=dict(tr.boundary_population),
        mode_marginal=None if tr.mode_marginal is None else {m: v[keep] for m, v in tr.mode_marginal.items()},
    )
    start = record.cached(boundary_key(step_index, sample_index, branch), BoundaryState)
    end = record.cached(boundary_key(step_index + 1, sample_index, branch), BoundaryState)
    return ZoomTrace(
        key=f"recorded/step{step_index}/s{sample_index}/b{branch}",
        step_index=step_index,
        sample_index=sample_index,
        branch=branch,
        n_store=0,
        trace=sliced,
        fock_start={} if start is None else dict(start.fock),
        fock_end={} if end is None else dict(end.fock),
        integrators=("recorded",),
        method=record.diagnostics.integrator,
        wall_time_s=0.0,
        engine_dimension=record.space.dimension,
    )


@dataclass(frozen=True)
class FockHeatmap:
    mode: int
    times_s: np.ndarray
    values: np.ndarray
    """(frames, levels) Fock populations, the levels cut above the highest one populated."""


def fock_heatmaps(z: ZoomTrace, frames: int = 25) -> tuple[FockHeatmap, ...]:
    """The Fock populations of every carried mode inside the step at ``frames`` evenly spaced stored times, from the
    marginals the zoom stored; empty when it stored none."""
    tr = z.trace
    if tr.mode_marginal is None or tr.times_s.size == 0:
        return ()
    rows = np.unique(np.round(np.linspace(0, tr.times_s.size - 1, frames)).astype(int))
    out = []
    for m, dist in sorted(tr.mode_marginal.items()):
        values = np.asarray(dist[rows], dtype=float)
        populated = np.flatnonzero(np.max(values, axis=0) > FOCK_FLOOR)
        top = int(populated[-1]) + 2 if populated.size else 1
        out.append(FockHeatmap(m, tr.times_s[rows], values[:, : min(top, values.shape[1])]))
    return tuple(out)


@dataclass(frozen=True)
class ProcessView:
    gate_id: str
    infidelity: Shown
    entanglement_infidelity: Shown
    depolarizing_rate: Shown
    cp_residual: Shown
    tp_residual: Shown
    pauli: tuple[Shown, ...]
    choi_abs: np.ndarray
    ideal_choi_abs: np.ndarray
    wall_time: Shown


def process_view(pm: ProcessMatrixRecord) -> ProcessView:
    return ProcessView(
        gate_id=pm.gate_id,
        infidelity=Shown(
            "Average gate infidelity",
            pm.average_gate_infidelity,
            detail=f"{pm.n_inputs} input states, {pm.n_traj} trajectories",
        ),
        entanglement_infidelity=Shown("Entanglement infidelity", pm.entanglement_infidelity),
        depolarizing_rate=Shown("Depolarizing rate", pm.depolarizing_rate),
        cp_residual=Shown("Distance to a completely positive map", pm.cp_tp_residual[0]),
        tp_residual=Shown("Distance to a trace-preserving map", pm.cp_tp_residual[1]),
        pauli=tuple(
            Shown(f"p({k})", v, detail="Pauli twirl")
            for k, v in sorted(pm.pauli_twirled.items(), key=lambda kv: -kv[1])
        ),
        choi_abs=np.abs(pm.choi),
        ideal_choi_abs=np.abs(np.asarray(core.choi_from_unitary(pm.ideal), dtype=complex)),
        wall_time=Shown("Wall time", pm.wall_time_s, "s"),
    )
