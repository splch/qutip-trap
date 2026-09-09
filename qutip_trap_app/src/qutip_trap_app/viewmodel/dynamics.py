"""Level 3, the dynamics: inside one pulse for one dynamical sample (PLAN.md Section 14.2, row 3).

Reads a :class:`ZoomTrace` (a re-simulated gate step, :mod:`qutip_trap_app.resim`): qubit populations and coherences
against time, <n_m>(t), the spin-branch loops alpha_im(t) of the played waveform (Section 4.4.1) beside the exact
trace's spin-averaged <a_m>(t), the Fock distributions at the step's start and end, the
jumps, the quasi-static noise values drawn for the sample, and for two qubits the concurrence and the Pauli correlators
against time. What the core does not expose is named in ``Record.core_gaps`` rather than drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import FockMovie, ProcessMatrixRecord, Record, TraceRecord, ZoomTrace
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


LoopKey = tuple[int, int]
"""(ion, mode) of one spin-branch loop."""


@dataclass(frozen=True)
class Loop:
    """One phase-space curve: a spin-branch loop alpha_im(t) of the played waveform (``quantity`` "branch_alpha", ``ion``
    set, Section 4.4.1) or the exact trace's spin-averaged <a_m>(t) ("alpha_m", ``ion`` None), which is a residue and not a
    loop: the branches' displacements cancel in it for a register with <S_phi> = 0."""

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
    eta: float | None = None

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
    t_start_s: float
    t_end_s: float
    times_s: np.ndarray
    populations: tuple[Series, ...]
    coherences: tuple[Series, ...]
    nbar: tuple[Series, ...]
    loops: tuple[Loop, ...]
    """The played waveform's spin-branch loops on the run's modes, one per (ion, mode); empty for a step without an entangling waveform."""
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
    loops = tuple(
        Loop(
            lp.mode,
            np.asarray(lp.alpha, dtype=complex),
            "branch_alpha",
            ion=lp.ion,
            label=f"ion {lp.ion}, mode {lp.mode}",
            chi_m_rad=lp.chi_m_rad,
            chi_closed_form_rad=lp.chi_closed_form_rad,
            eta=lp.eta,
        )
        for lp in record.branch_loops
        if lp.gate_id == step.gate_id
    )
    mean_alpha = tuple(
        Loop(m, np.asarray(a, dtype=complex), "alpha_m", label=f"mode {m}")
        for m, a in sorted(tr.alpha_m.items())
    )
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
        mean_alpha=mean_alpha,
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


# ---- the recorded coarse trace of a step, shown at once while the zoom computes (Section 14.7) ------------------------------------


def recorded_zoom(record: Record, step_index: int, sample_index: int = 0, branch: int = 0) -> ZoomTrace:
    """The run's own stored points inside one step as a ZoomTrace-shaped object (the segment boundaries the run stored,
    Section 14.3), so that :func:`pulse_dynamics` shows the recorded coarse trace before any re-simulation."""
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
        final_internal=tr.reduced_internal[keep[-1]] if tr.reduced_internal.size else tr.final_internal,
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


# ---- Fock heatmaps (Section 14.2 row 3) ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FockHeatmap:
    mode: int
    times: tuple[Shown, ...]
    """Frame times, each a Shown (the truncation points of the movie)."""
    times_s: np.ndarray
    levels: np.ndarray
    values: np.ndarray
    """(K + 1, d) P(n, t_k)."""
    nbar: tuple[Shown, ...]
    largest_populated: int
    """The highest n with P(n) above 1e-6 in any frame: what the heatmap should show."""
    method: str


def fock_heatmaps(movie: FockMovie, *, floor: float = 1e-6) -> tuple[FockHeatmap, ...]:
    out: list[FockHeatmap] = []
    for m, dist in sorted(movie.distributions.items()):
        populated = np.flatnonzero(np.max(dist, axis=0) > floor)
        top = int(populated[-1]) if populated.size else 0
        out.append(
            FockHeatmap(
                mode=m,
                times=tuple(Shown("frame_time", float(t), f"frame {k}") for k, t in enumerate(movie.times_s)),
                times_s=np.asarray(movie.times_s, dtype=float),
                levels=np.arange(dist.shape[1]),
                values=np.asarray(dist, dtype=float),
                nbar=tuple(
                    Shown("mode_nbar", float(v), f"mode {m}, frame {k}") for k, v in enumerate(movie.nbar[m])
                ),
                largest_populated=top,
                method=movie.method,
            )
        )
    return tuple(out)


# ---- the process matrix of the finished pulse (Section 14.2 row 3) -----------------------------------------------------------------


@dataclass(frozen=True)
class ProcessView:
    gate_id: str
    ions: tuple[int, ...]
    infidelity: Shown
    entanglement_infidelity: Shown
    depolarizing_rate: Shown
    cp_residual: Shown
    tp_residual: Shown
    pauli: tuple[Shown, ...]
    choi_abs: np.ndarray
    ideal_choi_abs: np.ndarray
    labels: tuple[str, ...]
    n_inputs: int
    wall_time: Shown
    method: str


def process_view(pm: ProcessMatrixRecord) -> ProcessView:
    from qutip_trap_app.core import choi_from_unitary

    d = pm.choi.shape[0]
    n = int(round(np.log2(int(round(np.sqrt(d))))))
    labels = tuple(format(i, f"0{n}b") + "->" + format(j, f"0{n}b") for i in range(2**n) for j in range(2**n))
    ideal = np.asarray(choi_from_unitary(pm.ideal), dtype=complex)
    return ProcessView(
        gate_id=pm.gate_id,
        ions=pm.ions,
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
        ideal_choi_abs=np.abs(ideal),
        labels=labels,
        n_inputs=pm.n_inputs,
        wall_time=Shown("wall_time", pm.wall_time_s, "tomography"),
        method=pm.method,
    )


def closure_table(dyn: PulseDynamics) -> tuple[dict[LoopKey, float], dict[LoopKey, float]]:
    """(end-to-start distance, largest excursion) per (ion, mode) spin-branch loop of the played waveform: what a closure
    prediction is scored against. The spin-averaged <a_m>(t) is not in it (it cancels between the branches)."""
    keyed = [((lp.ion if lp.ion is not None else -1, lp.mode), lp) for lp in dyn.loops]
    closes = {key: lp.closes for key, lp in keyed}
    excursions = {key: lp.excursion for key, lp in keyed}
    return closes, excursions


__all__ = [
    "FockHeatmap",
    "Loop",
    "LoopKey",
    "ProcessView",
    "PulseDynamics",
    "Series",
    "closure_table",
    "fock_heatmaps",
    "process_view",
    "pulse_dynamics",
    "recorded_zoom",
]
