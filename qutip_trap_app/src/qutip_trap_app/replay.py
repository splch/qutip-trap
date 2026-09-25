"""The app-side channel replay, Level 0's default engine (PLAN.md Sections 5.4, 6.8).

For every distinct native gate piece a circuit needs (kind, addressed ions, entangling angle) the core's GATE_LOCAL tomography
extracts the Section 6.8 channel of that gate played once from the prepared motional state. The channel is extracted at
phase zero and applied at the played phase by conjugating it with the virtual-Z rotation of every ion in the step, which is
exact when the physics is covariant under a rotation of the qubit frame; the library measures that covariance once per gate
kind at a second phase and adds the deviation to the residual. The register is carried as a density matrix and read out
through the calibration table's (eps_B, eps_D). The correlations between gates that the replay traces out are bounded by the
reported residual (Section 9.8); the record has no dynamics traces and its level, CHANNEL_REPLAY, is labelled derived.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from qutip_trap_app import core
from qutip_trap_app.record import (
    ChannelEntryRecord,
    ChannelPieceRecord,
    DiagnosticsRecord,
    JobSpec,
    NoiseSampleRecord,
    Progress,
    ReadoutRecord,
    Record,
    RecordError,
    ReplayGate,
    ReplayRecord,
    ResultsRecord,
    branch_loops,
    channel_summary_record,
    compiled_record,
    device_card,
    embed_on_register,
    schedule_record,
    space_record,
    table_record,
)

COVARIANCE_CHECK_PHASE_RAD = 1.0
"""The phase at which the library extracts a second channel per kind to measure the frame covariance of the extraction."""

ENTANGLING_ANGLE_DIGITS = 9


def channel_key(kind: str, ions: Sequence[int], angle: float | None) -> str:
    base = f"{kind}[{','.join(str(int(i)) for i in ions)}]"
    return base if angle is None else f"{base}@{round(float(angle), ENTANGLING_ANGLE_DIGITS):.9g}"


def conjugate_choi(choi: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    """The Choi state of R E(R^dag . R) R^dag from the Choi state of E, in the core's (input, output) convention
    (``choi_from_unitary``: |Phi_U> = sum_i |i> (x) U|i>, so a frame rotation acts as conj(R) (x) R)."""
    m = np.kron(unitary.conj(), unitary)
    return np.asarray(m @ choi @ m.conj().T)


def frame_rotation(ions: Sequence[int], phases: Mapping[int, float]) -> np.ndarray:
    """The product of RZ(phi_i) over ``ions`` in matrix order: the rotation that carries a phase-0 channel to the phases."""
    out = np.array([[1.0 + 0.0j]])
    for ion in ions:
        out = np.kron(out, core.rz(float(phases.get(ion, 0.0))))
    return out


def _embed(op: np.ndarray, sub_ions: Sequence[int], ions: Sequence[int]) -> np.ndarray:
    """``op`` on ``sub_ions`` as an operator on ``ions`` (both in matrix order), identity elsewhere."""
    position = {ion: k for k, ion in enumerate(ions)}
    return embed_on_register(op, [position[i] for i in sub_ions], len(ions))


# ---- the channel library --------------------------------------------------------------------------------------------------------


def one_gate_circuit(
    kind: str, ions: Sequence[int], angle: float | None, n_qubits: int, phase_rad: float = 0.0
) -> core.Circuit:
    """The native gate ``kind`` alone on ``ions`` at ``phase_rad`` (the entangling gates at their ``angle``)."""
    if kind in ("gpi", "gpi2"):
        params: tuple[float, ...] = (float(phase_rad),)
    elif angle is None:
        raise RecordError(f"a {kind} channel needs its angle")
    else:
        params = (float(phase_rad), float(phase_rad), float(angle)) if kind == "ms" else (float(angle),)
    op = core.Operation(kind, tuple(int(i) for i in ions), params)
    return core.Circuit(n_qubits, (op,), tuple(range(n_qubits)))


def extract_pieces(
    job: JobSpec,
    device: core.Device,
    table: core.CalibrationTable,
    kind: str,
    ions: Sequence[int],
    angle: float | None,
    *,
    phase_rad: float = 0.0,
) -> tuple[ChannelPieceRecord, ...]:
    """One gate played once at GATE_LOCAL from the prepared motional state: the Section 6.8 summaries of its pieces."""
    circuit = one_gate_circuit(kind, ions, angle, device.crystal.n_ions, phase_rad)
    machine = dataclasses.replace(job.machine(device, table), level=core.FidelityLevel.GATE_LOCAL)
    res = machine.run(circuit, 1, seed=job.seed)
    gl = res.diagnostics.gate_local
    assert gl is not None, "a GATE_LOCAL run reports its steps"
    steps = {s.gate_id: s for s in core.gate_steps(core.last_record(res).schedule) if s.kind == "gate"}
    pieces: list[ChannelPieceRecord] = []
    for st in gl.steps:
        if st.kind != "gate" or st.summary is None:
            continue
        ideal = np.eye(2 ** len(st.ions), dtype=complex)
        for tg in steps[st.gate_id].targets:
            ideal = _embed(tg.unitary(), tg.ions, st.ions) @ ideal
        pieces.append(
            ChannelPieceRecord(
                gate_id=str(st.gate_id),
                ions=tuple(int(i) for i in st.ions),
                choi=np.asarray(st.summary.choi, dtype=complex),
                ideal=ideal,
                summary=channel_summary_record(st.summary),
                residual_bound=float(st.residual_bound),
                frozen_excitation=float(sum(st.frozen_excitation.values())),
                dropped_crosstalk=float(st.dropped_crosstalk),
                cp_residual=float(st.cp_residual),
                tp_residual=float(st.tp_residual),
            )
        )
    if not pieces:
        raise RecordError(f"no channel extracted for {kind} on {tuple(ions)}")
    return tuple(pieces)


@dataclass
class ChannelLibrary:
    """The channels of one device, table and solver options, extracted on demand and kept for the session."""

    entries: dict[str, ChannelEntryRecord] = field(default_factory=dict)
    covariance_by_kind: dict[str, float] = field(default_factory=dict)
    """The frame-covariance residual measured per gate kind (once per kind); every entry of that kind carries it."""
    notes: list[str] = field(default_factory=list)
    _kraus: dict[str, list[np.ndarray]] = field(default_factory=dict, repr=False)

    def get(
        self,
        job: JobSpec,
        device: core.Device,
        table: core.CalibrationTable,
        kind: str,
        ions: Sequence[int],
        angle: float | None,
        *,
        progress: Progress | None = None,
    ) -> ChannelEntryRecord:
        """The entry, extracting it (and the kind's covariance check) on a miss."""
        key = channel_key(kind, ions, angle)
        entry = self.entries.get(key)
        if entry is None:
            if progress:
                progress("channels", None, f"deriving the channel of {key} by process tomography")
            pieces = extract_pieces(job, device, table, kind, ions, angle)
            entry = ChannelEntryRecord(key, kind, tuple(int(i) for i in ions), angle, pieces, None)
            self.entries[key] = entry
        if kind not in self.covariance_by_kind and kind in ("gpi", "gpi2", "ms"):
            if progress:
                progress(
                    "channels",
                    None,
                    f"measuring the frame covariance of {kind} at phase {COVARIANCE_CHECK_PHASE_RAD:g} rad",
                )
            residual = self._covariance(job, device, table, entry)
            self.covariance_by_kind[kind] = residual
            self.notes.append(
                f"{kind}: frame covariance residual {residual:.3e} at phase {COVARIANCE_CHECK_PHASE_RAD:g} rad"
            )
        if kind in self.covariance_by_kind and entry.covariance_residual != self.covariance_by_kind[kind]:
            entry = dataclasses.replace(entry, covariance_residual=self.covariance_by_kind[kind])
            self.entries[key] = entry
        return entry

    def _covariance(
        self, job: JobSpec, device: core.Device, table: core.CalibrationTable, entry: ChannelEntryRecord
    ) -> float:
        """Extract the gate again at a non-zero phase: max |Choi_phi - R Choi_0 R^dag| over its pieces."""
        phi = COVARIANCE_CHECK_PHASE_RAD
        pieces = extract_pieces(job, device, table, entry.kind, entry.addressed, entry.angle, phase_rad=phi)
        worst = 0.0
        for p0, p1 in zip(entry.pieces, pieces):
            predicted = conjugate_choi(p0.choi, frame_rotation(p0.ions, {ion: phi for ion in p0.ions}))
            worst = max(worst, float(np.max(np.abs(predicted - p1.choi))))
        return worst

    def kraus(self, entry: ChannelEntryRecord, piece_index: int) -> list[np.ndarray]:
        k = f"{entry.key}#{piece_index}"
        if k not in self._kraus:
            self._kraus[k] = [
                np.asarray(m, dtype=complex) for m in core.kraus_operators(entry.pieces[piece_index].choi)
            ]
        return self._kraus[k]


# ---- the replay -----------------------------------------------------------------------------------------------------------------


def preparation_for(job: JobSpec, device: core.Device) -> core.PreparationRun:
    """The device's recipe run, as the core prepares it (Section 4.2.6): the device's own, else the standard one on the
    entangling drives' Raman pair."""
    if device.preparation is not None:
        return core.run_preparation(device, device.preparation)
    pair = next(
        (
            (int(d.beams[0]), int(d.beams[1]))
            for d in job.entangling_drives.values()
            if d.kind == "raman" and len(d.beams) == 2
        ),
        None,
    )
    return core.run_preparation(device, core.standard_recipe(device, raman_pair=pair))


def readout_polarity(device: core.Device) -> tuple[int, ...]:
    """The bright computational level per ion from the species' readout scheme under the ion's detection beams (Section 13
    row 'Bright/dark polarity': a property of the scheme, never of the qubit)."""
    out: list[int] = []
    for i, sp in enumerate(device.crystal.species):
        beams = [device.beams[k] for k in core.detection_beams(device, i)]
        _rates, scheme, _model = core.detection_rates_for_ion(
            sp, device.field.B_gauss, device.field.direction, beams
        )
        out.append(int(scheme.bright_level))
    return tuple(out)


def replay(
    job: JobSpec,
    device: core.Device,
    table: core.CalibrationTable,
    library: ChannelLibrary,
    *,
    progress: Progress | None = None,
) -> Record:
    """Compile, schedule, derive the channels the schedule needs, apply them to the register, read it out: the record."""
    t_start = time.perf_counter()
    n = device.crystal.n_ions
    opts = job.options
    if progress:
        progress("compiling", 0.0, "compiling to native gates")
    machine = job.machine(device, table)
    report = machine.compile(job.circuit.to_core())
    if progress:
        progress("scheduling", 0.05, "scheduling pulses from the calibration table")
    sched = core.schedule(
        report.circuit,
        machine.device,
        table,
        t0_s=0.0,
        parallel=None,
        crosstalk_suppression="none",
        stark_compensation=True,
    )
    if progress:
        progress("preparing", 0.1, "the preparation recipe and the mode classes")
    prep = preparation_for(job, device)
    selection = core.select_space(device, sched, opts, nbar=prep.nbar, caps=job.caps, ion_dims=[2] * n)
    targets = sorted(sched.targets, key=lambda t: (t.t_start_s, t.gate_id))
    # the initial register: every ion pumped to |0> with its preparation error in |1> (Section 4.2.6)
    rho = np.array([[1.0 + 0.0j]])
    for i in range(n):
        p = float(prep.preparation_error(i)) if i in prep.pumps else 0.0
        rho = np.kron(rho, np.diag([1.0 - p, p]).astype(complex))
    applied: list[ReplayGate] = []
    states: list[np.ndarray] = []
    terms = {
        "residual_displacement": 0.0,
        "frozen_excitation": 0.0,
        "dropped_crosstalk": 0.0,
        "cp_tp_projection": 0.0,
        "frame_covariance": 0.0,
    }
    for k, tg in enumerate(targets):
        name, params = tg.native
        if progress:
            progress("replaying", 0.15 + 0.7 * k / len(targets), f"gate {k + 1}/{len(targets)}: {tg.gate_id}")
        angle = None
        phases: dict[int, float] = {}
        if name in ("gpi", "gpi2"):
            phases = {int(tg.ions[0]): float(params[0])}
        elif name == "ms":
            angle = float(params[2])
            phases = {int(tg.ions[0]): float(params[0]), int(tg.ions[1]): float(params[1])}
        elif name == "zz":
            angle = float(params[0])
        else:
            raise RecordError(f"the replay knows no native gate {name!r}")
        entry = library.get(job, device, table, name, tg.ions, angle, progress=progress)
        for j, piece in enumerate(entry.pieces):
            # neighbours of the step see the same light as the addressed ion(s): their frame turns by the addressed phase
            # (the mean of the two for an MS pair), which the covariance check measures rather than assumes
            default_phase = float(np.mean(list(phases.values()))) if phases else 0.0
            rot = frame_rotation(piece.ions, {ion: phases.get(ion, default_phase) for ion in piece.ions})
            new = np.zeros_like(rho)
            for km in library.kraus(entry, j):
                big = embed_on_register(rot @ km @ rot.conj().T, piece.ions, n)
                new += big @ rho @ big.conj().T
            rho = new
            terms["residual_displacement"] += piece.residual_bound
            terms["frozen_excitation"] += piece.frozen_excitation
            terms["dropped_crosstalk"] += piece.dropped_crosstalk
            terms["cp_tp_projection"] += piece.cp_residual + piece.tp_residual
        cov = entry.covariance_residual or 0.0
        terms["frame_covariance"] += cov
        # the Stark frame the scheduler absorbed (Section 7.5 item 7) is inside the target unitary and the extracted channel
        states.append(rho.copy())
        applied.append(
            ReplayGate(
                gate_id=str(tg.gate_id),
                key=entry.key,
                average_gate_infidelity=entry.average_gate_infidelity,
            )
        )
    if progress:
        progress("sampling", 0.9, f"reading out {job.shots} shots through the table's readout errors")
    bright = readout_polarity(device)
    eps_b = float(table.detection["eps_B"].value)
    eps_d = float(table.detection["eps_D"].value)
    spam = {f"q{i}": (eps_b, eps_d) for i in range(n)}
    for i in range(n):
        spam[f"q{i}.state_preparation"] = (float(prep.preparation_error(i)) if i in prep.pumps else 0.0, 0.0)
    probs = np.clip(np.real(np.diag(rho)), 0.0, None)
    probs = probs / probs.sum()
    rng = np.random.default_rng(core.SeedSpec(job.seed).child(0, 0, 0, 0, "channel_replay_readout"))
    idx = rng.choice(probs.size, size=int(job.shots), p=probs)
    levels = np.array(
        [[(x >> (n - 1 - i)) & 1 for i in range(n)] for x in idx], dtype=np.uint8
    )  # register order: ion 0 first
    declared = levels.copy()
    for i in range(n):
        is_bright = levels[:, i] == bright[i]
        flip = np.where(is_bright, rng.random(len(idx)) < eps_b, rng.random(len(idx)) < eps_d)
        declared[:, i] = np.where(flip, 1 - levels[:, i], levels[:, i])
    # the histogram's columns are the qubits the schedule's terminal measurement reads, as in the core's pipeline
    measured = next((e.ions for e in sched.events if e.kind == "measure"), ()) or tuple(range(n))
    bits = np.asarray(declared[:, [int(q) for q in measured]], dtype=np.uint8)
    counts, probabilities = core.aggregate(bits)
    shots = int(bits.shape[0])
    total = float(sum(terms.values()))
    notes = tuple(library.notes) + (
        "channel replay (app-side, Section 5.4): every gate applied as the Section 6.8 channel of that gate kind extracted once by "
        "GATE_LOCAL tomography from the prepared motional state and conjugated to the played phase; correlations between gates "
        "are traced out and bounded by the reported residual",
        f"readout: the calibration table's eps_B = {eps_b:.3g}, eps_D = {eps_d:.3g} per ion with bright levels {bright}",
    )
    sched_rec = schedule_record(sched)
    ideal = np.zeros(2**n, dtype=complex)
    ideal[0] = 1.0
    for t in sched_rec.targets:
        ideal = embed_on_register(t.unitary, t.ions, n) @ ideal
    replay_rec = ReplayRecord(
        gates=tuple(applied),
        register_after=np.stack(states) if states else np.zeros((0, 2**n, 2**n), complex),
        channels={g.key: library.entries[g.key] for g in applied},
        residual_terms=terms,
        residual_total=total,
    )
    diagnostics = DiagnosticsRecord(
        level="CHANNEL_REPLAY",
        integrator="channel replay (Kraus maps of the extracted channels)",
        tolerances=(float(opts.atol), float(opts.rtol)),
        samples=1,
        trajectories=1,
        branches=1,
        shots_per_sample_realized=(shots,),
        boundary_population={},
        boundary_population_max=float(opts.boundary_population_max),
        margin_levels={},
        margin_reached={},
        dropped_branch_weight=0.0,
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
            f"and bounded by the derivation residual {total:.3e} (Section 9.8); the readout is the table's (eps_B, eps_D)",
        )
        + notes,
        kernel="none",
        workers=1,
        wall_time_s=time.perf_counter() - t_start,
        convergence=None,
    )
    return Record(
        job=job,
        device_hash=device.hash(),
        table=table_record(table),
        device_card=device_card(
            device,
            spam=spam,
            intrinsic_budget={},
            tomography={g.gate_id: g.average_gate_infidelity for g in applied},
        ),
        compiled=compiled_record(report),
        schedule=sched_rec,
        space=space_record(selection.space, selection),
        noise_samples=(NoiseSampleRecord(sample_id=0, t_s=0.0, values={}, grids={}),),
        branches=(),
        traces=(),
        gate_local=None,
        readout=ReadoutRecord(
            window_s=float(table.detection["window_s"].value),
            threshold=float(table.detection["threshold"].value),
            levels=levels,
            time_used_s=np.zeros((shots, n), dtype=float),
            photon_records=None,
            posteriors=None,
        ),
        results=ResultsRecord(
            bitstrings=bits,
            heralds=np.zeros(shots, dtype=np.uint8),
            counts=counts,
            probabilities=probabilities,
            error_bars=core.binomial_error_bars(probabilities, shots),
            target_probabilities={
                str(k): float(v) for k, v in core.ideal_probabilities(job.circuit.to_core()).items()
            },
            effective_sample_size=float(shots),
            discarded_shots=0,
            sample_of_shot=np.zeros(shots, dtype=np.int64),
            final_state=rho,
            register_fidelity=float(np.real(ideal.conj() @ rho @ ideal)),
        ),
        diagnostics=diagnostics,
        branch_loops=branch_loops(sched, device, {int(m): float(v) for m, v in prep.nbar.items()}),
        replay=replay_rec,
    )
