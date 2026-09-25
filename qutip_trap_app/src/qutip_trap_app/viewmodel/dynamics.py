"""Level 3, the dynamics: inside one pulse for one dynamical sample (PLAN.md Section 14.2).

Reads a :class:`ZoomTrace` (a re-simulated gate step, or the run's own coarse trace of it): qubit populations and
coherences against time, <n_m>(t), the spin-branch loops alpha_im(t) of the played waveform (Section 4.4.1) beside the
exact trace's spin-averaged <a_m>(t), the Fock distributions, the jumps, the sample's quasi-static noise values, and for two
qubits the concurrence and the Pauli correlators against time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import ProcessMatrixRecord, Record, TraceRecord, ZoomTrace
from qutip_trap_app.viewmodel.catalogue import Shown
from qutip_trap_app.viewmodel.circuit import concurrence, pauli_expectations, single_ion_reduced
from qutip_trap_app.viewmodel.learn import LoopKey


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
    """One phase-space curve: a spin-branch loop alpha_im(t) of the played waveform (``quantity`` "branch_alpha", ``ion``
    set) or the exact trace's spin-averaged <a_m>(t) ("alpha_m", ``ion`` None), a residue in which the branches'
    displacements cancel for a register with <S_phi> = 0."""

    mode: int
    alpha: np.ndarray
    """Complex alpha(t)."""
    quantity: str = "alpha_m"
    ion: int | None = None
    label: str = ""
    chi_m_rad: float | None = None
    """The played waveform's stored per-mode angle (a branch loop only)."""
    chi_closed_form_rad: float | None = None
    """2 Im int conj(alpha_a) d alpha_b over the pair's loops on this mode (Section 4.4.3; a branch loop only)."""

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
    gate_id: str
    times_s: np.ndarray
    populations: tuple[Series, ...]
    coherences: tuple[Series, ...]
    nbar: tuple[Series, ...]
    loops: tuple[Loop, ...]
    """The played waveform's spin-branch loops, one per (ion, mode); empty for a step without an entangling waveform."""
    mean_alpha: tuple[Loop, ...]
    """The exact trace's spin-averaged <a_m>(t) per mode: a residue, not a loop."""
    fock_start: dict[int, np.ndarray]
    fock_end: dict[int, np.ndarray]
    pauli: tuple[Series, ...]
    concurrence: Series | None
    jumps: tuple[Shown, ...]
    noise_values: tuple[Shown, ...]
    boundary_population: tuple[Shown, ...]
    debye_waller: tuple[Shown, ...]
    wall_time: Shown
    norm_deficit: Shown | None
    """1 - Tr rho at the end of the step (normalize_output is off, Section 5.3); None when the trace carries no register
    state at the step's end."""
    unavailable: tuple[str, ...]


def pulse_dynamics(record: Record, z: ZoomTrace) -> PulseDynamics:
    tr = z.trace
    step = record.step(z.step_index)
    n = record.n_ions  # the traces' P1[i] and reduced internal states span every ion of the crystal
    t = tr.times_s
    two_level = bool(tr.reduced_internal.size) and all(d == 2 for d in tr.ion_dims)
    pops = tuple(
        Series("P1", f"P1 ion {i}", t, np.asarray(np.real(tr.expectations[f"P1[{i}]"]), dtype=float))
        for i in range(n)
        if f"P1[{i}]" in tr.expectations
    )
    coh = tuple(
        Series(
            "coherence",
            f"|rho_01| ion {i}",
            t,
            np.asarray(
                [abs(single_ion_reduced(rho, i, n)[0, 1]) for rho in tr.reduced_internal], dtype=float
            ),
        )
        for i in range(n)
        if two_level
    )
    loops = tuple(
        Loop(
            lp.mode,
            np.asarray(lp.alpha, dtype=complex),
            "branch_alpha",
            ion=lp.ion,
            label=f"ion {lp.ion}, mode {lp.mode}",
            chi_m_rad=lp.chi_m_rad,
            chi_closed_form_rad=lp.chi_closed_form_rad,
        )
        for lp in record.branch_loops
        if lp.gate_id == step.gate_id
    )
    pauli: list[Series] = []
    conc: Series | None = None
    if n == 2 and two_level:
        keys = ("XX", "YY", "ZZ", "ZI", "IZ")
        expectations = [pauli_expectations(rho, 2) for rho in tr.reduced_internal]
        pauli = [
            Series("pauli_expectation", f"<{k}>", t, np.asarray([pe[k] for pe in expectations], dtype=float))
            for k in keys
        ]
        cvals = np.asarray([concurrence(rho) for rho in tr.reduced_internal], dtype=float)
        conc = Series("concurrence", "concurrence", t, cvals)
    sample = record.noise_samples[z.sample_index]
    unavailable: list[str] = []
    if tr.mode_marginal is None:
        unavailable.append(
            "per-time Fock distributions inside the pulse: this trace stores none (re-simulate for them; start and end shown)"
        )
    if n != 2:
        unavailable.append("concurrence and Pauli correlators are computed for two-qubit registers")
    return PulseDynamics(
        gate_id=step.gate_id,
        times_s=t,
        populations=pops,
        coherences=coh,
        nbar=tuple(
            Series("mode_nbar", f"<n> mode {m}", t, np.asarray(v, dtype=float))
            for m, v in sorted(tr.mode_nbar.items())
        ),
        loops=loops,
        mean_alpha=tuple(
            Loop(m, np.asarray(a, dtype=complex), "alpha_m", label=f"mode {m}")
            for m, a in sorted(tr.alpha_m.items())
        ),
        fock_start=dict(z.fock_start),
        fock_end=dict(z.fock_end),
        pauli=tuple(pauli),
        concurrence=conc,
        jumps=tuple(Shown("jump", t_j, channel) for t_j, channel in tr.jumps),
        noise_values=tuple(Shown("noise_sample_value", v, k) for k, v in sorted(sample.values.items())),
        boundary_population=tuple(
            Shown("boundary_population", v, f"mode {m}") for m, v in sorted(tr.boundary_population.items())
        ),
        debye_waller=tuple(
            Shown("debye_waller", c[0], f"frozen mode {m}: |alpha|^2 (2 nbar + 1); chi loss {c[1]:.3g} rad")
            for m, c in sorted(record.diagnostics.frozen_contribution.items())
        ),
        wall_time=Shown("wall_time", z.wall_time_s, f"re-simulation at dimension {z.engine_dimension}"),
        norm_deficit=norm_deficit(tr, "at the step's end"),
        unavailable=tuple(unavailable),
    )


def norm_deficit(tr: TraceRecord, detail: str) -> Shown | None:
    """1 - Tr rho at the trace's end; None when it carries no register state there."""
    if tr.final_internal.size == 0:
        return None
    return Shown("norm_deficit", 1.0 - float(np.real(np.trace(tr.final_internal))), detail)


def recorded_zoom(record: Record, step_index: int, sample_index: int = 0, branch: int = 0) -> ZoomTrace:
    """The run's own stored points inside one step as a ZoomTrace, so that :func:`pulse_dynamics` shows the recorded coarse
    trace before any re-simulation. Raises ``KeyError`` when the record has no such step (a circuit that plays no pulse) or
    stores no trace for the (sample, branch)."""
    if not 0 <= step_index < len(record.schedule.steps):
        raise KeyError(
            f"the record has no step {step_index} ({len(record.schedule.steps)} steps: no pulse was played)"
        )
    step = record.step(step_index)
    tr = record.trace(sample_index, branch)
    t = np.asarray(tr.times_s, dtype=float)
    keep = np.flatnonzero((t >= step.t_start_s - 1e-12) & (t <= step.t_end_s + 1e-12))
    if keep.size == 0:
        raise KeyError(f"the run stored no point inside step {step_index}")
    sliced = TraceRecord(
        sample_index=tr.sample_index,
        sample_id=tr.sample_id,
        branch=tr.branch,
        weight=tr.weight,
        times_s=t[keep],
        expectations={k: np.asarray(v)[keep] for k, v in tr.expectations.items()},
        reduced_internal=tr.reduced_internal[keep] if tr.reduced_internal.size else tr.reduced_internal,
        ion_dims=tr.ion_dims,
        mode_nbar={m: np.asarray(v)[keep] for m, v in tr.mode_nbar.items()},
        alpha_m={m: np.asarray(v)[keep] for m, v in tr.alpha_m.items()},
        jumps=tuple(j for j in tr.jumps if step.t_start_s <= j[0] <= step.t_end_s),
        boundary_population=dict(tr.boundary_population),
        # the run's final internal state belongs to the end of the run, not of this step: with no register state stored
        # inside the step, an empty array says the state at its end is unknown
        final_internal=tr.reduced_internal[keep[-1]]
        if tr.reduced_internal.size
        else np.zeros((0, 0), dtype=complex),
        final_mode_reduced={},
        final_nbar={m: float(np.asarray(v)[keep[-1]]) for m, v in tr.mode_nbar.items()},
        final_joint=None,
        joint_dims=tr.joint_dims,
    )
    boundary = record.boundary(step_index, sample_index, branch)
    end_boundary = record.boundary(step_index + 1, sample_index, branch)
    return ZoomTrace(
        key=f"recorded/step{step_index}/s{sample_index}/b{branch}",
        step_index=step_index,
        sample_index=sample_index,
        branch=branch,
        n_store=0,
        options_digest="recorded",
        trace=sliced,
        fock_end={}
        if end_boundary is None
        else {m: np.real(np.diag(r)).astype(float) for m, r in end_boundary.mode_reduced.items()},
        fock_start={}
        if boundary is None
        else {m: np.real(np.diag(r)).astype(float) for m, r in boundary.mode_reduced.items()},
        integrators=("recorded",),
        method=record.diagnostics.integrator,
        approximations=record.diagnostics.approximations,
        wall_time_s=0.0,
        engine_dimension=record.space.dimension,
    )


# ---- Fock heatmaps ---------------------------------------------------------------------------------------------------

FOCK_FRAMES = 8
"""A Fock heatmap shows the step's start and this many equally spaced points of the trace up to its end."""


@dataclass(frozen=True)
class FockHeatmap:
    mode: int
    times_s: np.ndarray
    values: np.ndarray
    """(K + 1, d) P(n, t_k)."""
    nbar: tuple[Shown, ...]
    largest_populated: int
    """The highest n with P(n) above 1e-6 in any frame."""


def fock_heatmaps(z: ZoomTrace) -> tuple[FockHeatmap, ...]:
    """Per resolved mode the Fock populations P(n, t) at ``FOCK_FRAMES`` + 1 points of the zoom's trace, read off the
    populations the trace stores at every point; empty when it stores none (the run's own coarse trace)."""
    tr = z.trace
    if tr.mode_marginal is None:
        return ()
    frames = [round(k * (tr.times_s.size - 1) / FOCK_FRAMES) for k in range(FOCK_FRAMES + 1)]
    out: list[FockHeatmap] = []
    for m, dist in sorted(tr.mode_marginal.items()):
        values = np.asarray(dist[frames], dtype=float)
        populated = np.flatnonzero(np.max(values, axis=0) > 1e-6)
        out.append(
            FockHeatmap(
                mode=int(m),
                times_s=np.asarray(tr.times_s[frames], dtype=float),
                values=values,
                nbar=tuple(
                    Shown("mode_nbar", float(tr.mode_nbar[m][i]), f"mode {m}, frame {k}")
                    for k, i in enumerate(frames)
                ),
                largest_populated=int(populated[-1]) if populated.size else 0,
            )
        )
    return tuple(out)


# ---- the process matrix of the finished pulse ------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessView:
    infidelity: Shown
    entanglement_infidelity: Shown
    depolarizing_rate: Shown
    cp_residual: Shown
    tp_residual: Shown
    pauli: tuple[Shown, ...]
    choi_abs: np.ndarray
    ideal_choi_abs: np.ndarray
    axis_labels: tuple[str, ...]
    """The Choi matrices' row and column labels: input and output basis states side by side."""
    wall_time: Shown
    method: str


def process_view(pm: ProcessMatrixRecord) -> ProcessView:
    from qutip_trap_app.core import choi_from_unitary

    d = int(round(np.sqrt(pm.choi.shape[0])))
    states = [format(k, f"0{max(1, int(np.log2(d)))}b") for k in range(d)]
    return ProcessView(
        infidelity=Shown(
            "channel_infidelity",
            pm.average_gate_infidelity,
            f"{pm.n_inputs} inputs, {pm.n_traj} trajectories",
        ),
        entanglement_infidelity=Shown("entanglement_infidelity", pm.entanglement_infidelity),
        depolarizing_rate=Shown("depolarizing_rate", pm.depolarizing_rate),
        cp_residual=Shown("cp_tp_residual", pm.cp_tp_residual[0], "against the completely positive cone"),
        tp_residual=Shown("cp_tp_residual", pm.cp_tp_residual[1], "against the trace-preserving set"),
        pauli=tuple(
            Shown("pauli_twirl", v, k) for k, v in sorted(pm.pauli_twirled.items(), key=lambda kv: -kv[1])
        ),
        choi_abs=np.abs(pm.choi),
        ideal_choi_abs=np.abs(np.asarray(choi_from_unitary(pm.ideal), dtype=complex)),
        axis_labels=tuple(f"{a}{b}" for a in states for b in states),
        wall_time=Shown("wall_time", pm.wall_time_s, "tomography"),
        method=pm.method,
    )


def closure_table(dyn: PulseDynamics) -> tuple[dict[LoopKey, float], dict[LoopKey, float]]:
    """(end-to-start distance, largest excursion) per (ion, mode) spin-branch loop of the played waveform: what a closure
    prediction is scored against."""
    keyed = [((lp.ion if lp.ion is not None else -1, lp.mode), lp) for lp in dyn.loops]
    return {key: lp.closes for key, lp in keyed}, {key: lp.excursion for key, lp in keyed}
