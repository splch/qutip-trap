"""Building a run record from a channel replay (PLAN.md Section 14.3: one record, many views; milestone M11.2).

A replay produces no dynamics traces, no joint space run and no photon records; its record therefore carries the schedule
and compiled circuit the replay played, the mode classes the run WOULD have used (``select_space`` on the same schedule,
so the closure indicators of Level 2 read the same numbers), the register after every gate (what Level 1 shows), the
channels applied with their residuals (``Record.replay``), and results read out through the table's readout errors. Its
fidelity level is ``CHANNEL_REPLAY`` and every view labels it derived.
"""

from __future__ import annotations

import datetime as _dt
import math
import platform

import numpy as np

from qutip_trap_app import __version__ as app_version
from qutip_trap_app import core
from qutip_trap_app.record import (
    RECORD_FORMAT,
    ChannelEntryRecord,
    ChannelPieceRecord,
    CircuitRecord,
    CompiledRecord,
    DiagnosticsRecord,
    JobSpec,
    NoiseSampleRecord,
    PreparationRecord,
    ReadoutRecord,
    Record,
    ReplayGate,
    ReplayRecord,
    ResultsRecord,
    branch_loops,
    device_card,
    schedule_record,
    space_record,
    table_record,
)
from qutip_trap_app.replay import ChannelLibrary, ReplayOutcome
from qutip_trap_app.viewmodel.circuit import fidelity_to_ket, target_ket


def _bitstrings(bits: np.ndarray, measured: tuple[int, ...]) -> np.ndarray:
    """The declared bits of every ion (register order, ion 0 first) restricted to the measured qubits in ascending order: the
    Result convention (column j = the j-th measured qubit; qubit q is ion q), the same columns the core's pipeline keeps."""
    arr = np.asarray(bits, dtype=np.uint8)
    return np.asarray(arr[:, list(measured)], dtype=np.uint8) if arr.ndim == 2 else arr


def _counts(bits: np.ndarray) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
    n = bits.shape[0]
    counts: dict[str, int] = {}
    for row in bits:
        key = "".join(str(int(b)) for b in row[::-1])  # qubit 0 rightmost (conv.result_bit_order)
        counts[key] = counts.get(key, 0) + 1
    probs = {k: v / n for k, v in counts.items()}
    bars = {k: math.sqrt(max(p * (1.0 - p), 0.0) / n) for k, p in probs.items()}
    return counts, probs, bars


def channel_entry_record(entry: object) -> ChannelEntryRecord:
    from qutip_trap_app.replay import ChannelEntry

    assert isinstance(entry, ChannelEntry)
    return ChannelEntryRecord(
        key=entry.key,
        kind=entry.kind,
        addressed=entry.addressed,
        angle=entry.angle,
        pieces=tuple(
            ChannelPieceRecord(
                gate_id=p.gate_id,
                ions=p.ions,
                choi=np.asarray(p.choi, dtype=complex),
                ideal=np.asarray(p.ideal, dtype=complex),
                summary=p.summary,
                residual_bound=p.residual_bound,
                frozen_excitation=p.frozen_excitation,
                dropped_crosstalk=p.dropped_crosstalk,
                cp_residual=p.cp_residual,
                tp_residual=p.tp_residual,
                residual_displacement=dict(p.residual_displacement),
                nbar_after=dict(p.nbar_after),
                local_dimension=p.local_dimension,
                engine_runs=p.engine_runs,
            )
            for p in entry.pieces
        ),
        covariance_residual=entry.covariance_residual,
        wall_time_s=entry.wall_time_s,
        notes=entry.notes,
    )


def build_replay_record(
    job: JobSpec,
    device: core.Device,
    table: core.CalibrationTable,
    outcome: ReplayOutcome,
    library: ChannelLibrary,
) -> Record:
    n = device.crystal.n_ions
    circuit = job.circuit.to_core()
    opts = job.solver_options()
    bits = _bitstrings(outcome.bits, outcome.measured)
    counts, probs, bars = _counts(bits)
    sched_rec = schedule_record(outcome.schedule)
    ideal = target_ket([(t.unitary, t.ions) for t in sched_rec.targets], n)
    register_fid = fidelity_to_ket(outcome.final, ideal)
    versions = {
        "qutip_trap": str(core.core_version),
        "qutip_trap_app": str(app_version),
        "python": platform.python_version(),
    }
    used = {g.key: library.entries[g.key] for g in outcome.applied if g.key in library.entries}
    replay_rec = ReplayRecord(
        engine="channel replay (app-side, Section 5.4): the core's Section 6.8 channel summaries applied per gate, labelled derived",
        gates=tuple(
            ReplayGate(
                g.gate_id,
                g.key,
                g.ions,
                dict(g.phases),
                g.residual,
                g.average_gate_infidelity,
                g.depolarizing_rate,
            )
            for g in outcome.applied
        ),
        register_after=np.asarray(outcome.register_after, dtype=complex),
        channels={k: channel_entry_record(e) for k, e in used.items()},
        residual_terms=dict(outcome.residual_terms),
        residual_total=float(outcome.residual_total),
        bright_levels=outcome.bright_levels,
        spam_used=dict(outcome.spam),
        notes=outcome.notes,
    )
    tomography = {g.gate_id: g.average_gate_infidelity for g in outcome.applied}
    selection = outcome.selection
    prep = outcome.preparation
    shots = int(bits.shape[0])
    diagnostics = DiagnosticsRecord(
        level="CHANNEL_REPLAY",
        integrator="channel replay (Kraus maps of the extracted channels)",
        tolerances=(float(opts.atol), float(opts.rtol)),
        samples=1,
        trajectories=1,
        branches=1,
        shots_per_sample_realized=(shots,),
        root_seed=int(job.seed),
        boundary_population={},
        boundary_population_max=float(opts.boundary_population_max),
        margin_levels={},
        margin_reached={},
        populated_n_max={},
        cap_growth={},
        dropped_branch_weight=0.0,
        frozen_excitation_bound=dict(selection.frozen_excitation),
        dropped_contribution=(
            float(selection.dropped_contribution[0]),
            float(selection.dropped_contribution[1]),
        ),
        frozen_contribution={
            int(m): (float(c[0]), float(c[1])) for m, c in selection.frozen_contribution.items()
        },
        intrinsic_budget={},
        approximations=(
            "CHANNEL_REPLAY: every gate applied as its extracted Section 6.8 channel; the correlations between gates are traced out "
            f"and bounded by the derivation residual {outcome.residual_total:.3e} (Section 9.8); the readout is the table's (eps_B, eps_D)",
        )
        + tuple(outcome.notes),
        kernel="none",
        workers=1,
        propagator_cache_hits=0,
        wall_clock_span_s=0.0,
        wall_time_s=float(outcome.wall_time_s),
        convergence=None,
        run_state_order=tuple(range(n)),
        run_state_events=(),
    )
    readout = ReadoutRecord(
        mode="replay",
        window_s=float(table.detection["window_s"].value) if "window_s" in table.detection else 0.0,
        discriminator="table (eps_B, eps_D) per ion",
        threshold=float(table.detection["threshold"].value) if "threshold" in table.detection else None,
        levels=np.asarray(outcome.levels, dtype=np.uint8),
        bits_declared=np.asarray(outcome.bits, dtype=np.uint8),
        time_used_s=np.zeros((shots, n), dtype=float),
        photon_records=None,
        posteriors=None,
        sub_bin_records=None,
        arrival_offsets=None,
        arrival_times_s=None,
        rates=(),
        leakage={},
        crosstalk_discrepancy=None,
    )
    results = ResultsRecord(
        bitstrings=bits,
        heralds=np.zeros(shots, dtype=np.uint8),
        counts=counts,
        probabilities=probs,
        error_bars=bars,
        target_probabilities={str(k): float(v) for k, v in core.ideal_probabilities(circuit).items()},
        effective_sample_size=float(shots),
        discarded_shots=0,
        sample_of_shot=np.zeros(shots, dtype=np.int64),
        spam=dict(outcome.spam),
        final_state=np.asarray(outcome.final, dtype=complex),
        register_fidelity=register_fid,
    )
    return Record(
        format=RECORD_FORMAT,
        created_utc=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        versions=versions,
        job=job,
        device_hash=device.hash(),
        table=table_record(table),
        device_card=device_card(device, spam=outcome.spam, intrinsic_budget={}, tomography=tomography),
        compiled=CompiledRecord(
            native=CircuitRecord.from_core(outcome.compile.circuit),
            final_frame_rad={int(q): float(v) for q, v in outcome.compile.final_frame_rad.items()},
            n_pulses=int(outcome.compile.n_pulses),
            n_entangling=int(outcome.compile.n_entangling),
            block_residuals=tuple(float(r) for r in outcome.compile.block_residuals),
            circuit_residual=None
            if outcome.compile.circuit_residual is None
            else float(outcome.compile.circuit_residual),
            entangler=str(outcome.compile.entangler),
            notes=tuple(outcome.compile.notes),
            target_unitary=np.asarray(core.circuit_unitary(circuit), dtype=complex)
            if circuit.n_qubits <= 6
            else None,
        ),
        schedule=sched_rec,
        space=space_record(selection.space, selection),
        preparation=PreparationRecord(
            nbar={int(m): float(v) for m, v in prep.nbar.items()},
            duration_s=float(prep.duration_s),
            preparation_error={int(i): float(prep.preparation_error(i)) for i in prep.pumps},
            provenance=tuple(prep.provenance),
            notes=tuple(prep.notes),
        ),
        noise_samples=(NoiseSampleRecord(sample_id=0, t_s=0.0, values={}, grids={}),),
        branches=(),
        traces=(),
        gate_local=None,
        readout=readout,
        results=results,
        diagnostics=diagnostics,
        notes=tuple(outcome.notes),
        core_gaps=core.CORE_GAPS,
        branch_loops=branch_loops(outcome.schedule, device, {int(m): float(v) for m, v in prep.nbar.items()}),
        joint_store_dimension_max=4096,
        replay=replay_rec,
    )


__all__ = ["build_replay_record", "channel_entry_record"]
