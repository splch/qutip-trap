"""The app-side channel-replay engine (PLAN.md Sections 5.4, 6.8, 14.2 row 0, 14.6; Section 9.11 row "Channel derivation";
milestone M11.2).

Section 5.4: "Replaying derived per-gate channels for large circuits is not a core fidelity level: the separate application
package of Section 14 builds that on its own side from the channel summaries the core already exports (Section 6.8), and the
core neither knows nor needs to know about it." Section 14.2 makes it Level 0's default engine, "labelled as derived".

How it works. For every distinct native gate piece a circuit needs (kind, addressed ions, entangling angle) the engine asks
the core for the Section 6.8 channel summary of that gate played ONCE from the prepared motional state: a one-gate circuit
run at GATE_LOCAL, whose ``Diagnostics.gate_local`` carries the step's Choi matrix after the CP/TP projection, its residual
displacement bound, its frozen-spectator excitation and dropped crosstalk, and the projection residuals. The channel is
extracted at phase zero and applied to the register at the phase the schedule plays by conjugating it with the virtual-Z
rotation of every ion in the step, which is exact when the physics is covariant under a rotation of the qubit frame (the
drive phase shifts, the neighbours' leaked light shifts with it, the dissipators are invariant); the library MEASURES that
covariance once per gate kind by extracting a second channel at a non-zero phase, and the deviation joins the residual.
The register is carried as a density matrix (Section 5.4 allows this to 12 qubits), the measurement is the calibration
table's (eps_B, eps_D) per ion with the species' readout polarity, and every shot is a draw from that.

What the replay cannot know is stated, not hidden: the correlations between gates that GATE_LOCAL traces out, and that the
replay traces out too, are bounded by the summed residual displacement |alpha_m|^2 (2 nbar_m + 1) of the extracted steps
(Section 9.8); a hotter motional state widens that bound and the discrepancy together, which is what Section 9.11 tests.
The replay reports ``residual_total`` as the channel-derivation residual and "verify deeper" (:mod:`qutip_trap_app.verify`)
runs the same job at the next engine and compares.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from qutip_trap_app import core
from qutip_trap_app.codec import dumps
from qutip_trap_app.record import (
    ChannelSummaryRecord,
    JobSpec,
    RecordError,
    channel_summary_record,
    options_record,
)

Progress = Callable[[str, float | None, str], None]
"""progress(stage, fraction or None, message)."""

REGISTER_DM_MAX_QUBITS = 12
"""Section 5.4: the register is a density matrix to 12 qubits; the pure-state sampling path beyond is not in M11.2."""

COVARIANCE_CHECK_PHASE_RAD = 1.0
"""The phase at which the library extracts a second channel per kind to measure the frame covariance of the extraction."""

ENTANGLING_ANGLE_DIGITS = 9

RESONANT_WINDOW_M = 1e-9
"""Beams within this wavelength window of the species' cycling line are its detection light (the core's roles module uses a
frequency window; a nanometre separates 369.5 nm detection light from 355 nm gate light by far)."""


def channel_key(kind: str, ions: Sequence[int], angle: float | None) -> str:
    base = f"{kind}[{','.join(str(int(i)) for i in ions)}]"
    return base if angle is None else f"{base}@{round(float(angle), ENTANGLING_ANGLE_DIGITS):.9g}"


def rz(phi_rad: float) -> np.ndarray:
    """exp(-i phi sigma_z/2): the frame rotation whose conjugation carries a phase-0 channel to phase phi (Section 7.6)."""
    return np.diag([np.exp(-0.5j * phi_rad), np.exp(0.5j * phi_rad)]).astype(complex)


def conjugate_choi(choi: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    """The Choi state of R E(R^dag . R) R^dag from the Choi state of E, in the core's (input, output) convention
    (``choi_from_unitary``: |Phi_U> = sum_i |i> (x) U|i>, so a frame rotation acts as conj(R) (x) R)."""
    m = np.kron(unitary.conj(), unitary)
    return np.asarray(m @ choi @ m.conj().T)


def embed_on(op: np.ndarray, sub_ions: Sequence[int], all_ions: Sequence[int]) -> np.ndarray:
    """``op`` on ``sub_ions`` (matrix order) as an operator on ``all_ions`` (matrix order), identity elsewhere."""
    from qutip_trap_app.viewmodel.circuit import embed_operator

    pos = {ion: k for k, ion in enumerate(all_ions)}
    return embed_operator(op, tuple(pos[i] for i in sub_ions), len(all_ions))


# ---- the channel library --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelPiece:
    """One gate step's channel at phase zero: what the core's GATE_LOCAL tomography extracted (Section 5.4 (a))."""

    gate_id: str
    ions: tuple[int, ...]
    """The step's ions in matrix order: the addressed ions and the crosstalk neighbours above the threshold."""
    choi: np.ndarray
    ideal: np.ndarray
    """The target unitary on ``ions`` at phase zero (identity on the neighbours)."""
    summary: ChannelSummaryRecord
    residual_bound: float
    frozen_excitation: float
    dropped_crosstalk: float
    cp_residual: float
    tp_residual: float
    residual_displacement: dict[int, float]
    nbar_after: dict[int, float]
    local_dimension: int
    engine_runs: int

    @property
    def derivation_residual(self) -> float:
        """What the replay of this piece cannot account for: the traced-out correlations (Section 9.8's bound), the frozen
        spectators' excitation, the dropped crosstalk and the CP/TP projection residuals."""
        return (
            self.residual_bound
            + self.frozen_excitation
            + self.dropped_crosstalk
            + self.cp_residual
            + self.tp_residual
        )


@dataclass(frozen=True)
class ChannelEntry:
    key: str
    kind: str
    addressed: tuple[int, ...]
    angle: float | None
    pieces: tuple[ChannelPiece, ...]
    covariance_residual: float | None
    """max |Choi(phi) - conj(Choi(0))| measured at ``COVARIANCE_CHECK_PHASE_RAD``; None until checked."""
    wall_time_s: float
    notes: tuple[str, ...]

    @property
    def derivation_residual(self) -> float:
        return sum(p.derivation_residual for p in self.pieces) + (self.covariance_residual or 0.0)

    @property
    def average_gate_infidelity(self) -> float:
        return float(sum(p.summary.average_gate_infidelity for p in self.pieces))

    @property
    def depolarizing_rate(self) -> float:
        return float(sum(p.summary.depolarizing_rate for p in self.pieces))


def options_digest(options: core.SolverOptions) -> str:
    return hashlib.sha256(dumps(options_record(options))).hexdigest()[:16]


def one_gate_circuit(
    kind: str, ions: Sequence[int], angle: float | None, n_qubits: int, phase_rad: float = 0.0
) -> core.Circuit:
    ions_t = tuple(int(i) for i in ions)
    if kind in ("gpi", "gpi2"):
        op = core.Operation(kind, ions_t, (float(phase_rad),))
    elif kind == "ms":
        if angle is None:
            raise RecordError("an ms channel needs its angle")
        op = core.Operation("ms", ions_t, (float(phase_rad), float(phase_rad), float(angle)))
    elif kind == "zz":
        if angle is None:
            raise RecordError("a zz channel needs its angle")
        op = core.Operation("zz", ions_t, (float(angle),))
    else:
        raise RecordError(f"not a native gate kind the replay can extract: {kind!r}")
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
) -> tuple[tuple[ChannelPiece, ...], tuple[str, ...]]:
    """One gate played once at GATE_LOCAL from the prepared motional state: the Section 6.8 summaries of its pieces."""
    circuit = one_gate_circuit(kind, ions, angle, device.crystal.n_ions, phase_rad)
    machine = dataclasses.replace(job.machine(device, table), level=core.FidelityLevel.GATE_LOCAL)
    res = machine.run(circuit, 1, seed=job.seed)
    gl = res.diagnostics.gate_local
    if gl is None:
        raise RecordError("the one-gate GATE_LOCAL run returned no gate-local report")
    rec = core.last_record(res)
    steps = {s.gate_id: s for s in core.gate_steps(rec.schedule) if s.kind == "gate"}
    pieces: list[ChannelPiece] = []
    for st in gl.steps:
        if st.kind != "gate" or st.summary is None:
            continue
        gs = steps.get(st.gate_id)
        if gs is None:
            raise RecordError(f"GATE_LOCAL step {st.gate_id!r} has no schedule counterpart")
        ideal = np.eye(2 ** len(st.ions), dtype=complex)
        for tg in gs.targets:
            ideal = embed_on(np.asarray(tg.unitary(), dtype=complex), tg.ions, st.ions) @ ideal
        pieces.append(
            ChannelPiece(
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
                residual_displacement={int(m): float(v) for m, v in st.residual_displacement.items()},
                nbar_after={int(m): float(v) for m, v in st.nbar_after.items()},
                local_dimension=int(np.prod(st.space_dims)),
                engine_runs=int(st.engine_runs),
            )
        )
    if not pieces:
        raise RecordError(f"no channel extracted for {kind} on {tuple(ions)}")
    return tuple(pieces), tuple(gl.notes)


@dataclass
class ChannelLibrary:
    """The channels of one device and table, extracted on demand and kept for the session (a device change empties it)."""

    device_hash: str
    table_seed: int
    options_digest: str
    entries: dict[str, ChannelEntry] = field(default_factory=dict)
    covariance_checked: set[str] = field(default_factory=set)
    """Gate kinds (``gpi2``, ``ms``, ...) whose frame covariance has been measured; one check per kind per library."""
    covariance_by_kind: dict[str, float] = field(default_factory=dict)
    """The measured residual per kind; every entry of that kind carries it."""
    notes: list[str] = field(default_factory=list)
    _kraus: dict[str, list[np.ndarray]] = field(default_factory=dict, repr=False)

    @classmethod
    def for_job(cls, job: JobSpec, device: core.Device, table: core.CalibrationTable) -> ChannelLibrary:
        return cls(device.hash(), int(table.seed), options_digest(job.solver_options()))

    def matches(self, job: JobSpec, device: core.Device, table: core.CalibrationTable) -> bool:
        return (
            self.device_hash == device.hash()
            and self.table_seed == int(table.seed)
            and self.options_digest == options_digest(job.solver_options())
        )

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
        check_covariance: bool = True,
    ) -> ChannelEntry:
        """The entry, extracting it (and the kind's covariance check) on a miss."""
        key = channel_key(kind, ions, angle)
        entry = self.entries.get(key)
        if entry is None:
            if progress:
                progress("channels", None, f"deriving the channel of {key} by process tomography")
            t0 = time.perf_counter()
            pieces, notes = extract_pieces(job, device, table, kind, ions, angle)
            entry = ChannelEntry(
                key, kind, tuple(int(i) for i in ions), angle, pieces, None, time.perf_counter() - t0, notes
            )
            self.entries[key] = entry
        if check_covariance and kind not in self.covariance_checked and kind in ("gpi", "gpi2", "ms"):
            if progress:
                progress(
                    "channels",
                    None,
                    f"measuring the frame covariance of {kind} at phase {COVARIANCE_CHECK_PHASE_RAD:g} rad",
                )
            residual = self.covariance_check(job, device, table, entry)
            self.covariance_checked.add(kind)
            self.covariance_by_kind[kind] = residual
            self.notes.append(
                f"{kind}: frame covariance residual {residual:.3e} at phase {COVARIANCE_CHECK_PHASE_RAD:g} rad"
            )
        if kind in self.covariance_by_kind and entry.covariance_residual != self.covariance_by_kind[kind]:
            entry = replace(entry, covariance_residual=self.covariance_by_kind[kind])
            self.entries[key] = entry
        return entry

    def covariance_check(
        self, job: JobSpec, device: core.Device, table: core.CalibrationTable, entry: ChannelEntry
    ) -> float:
        """Extract the same gate at a non-zero phase and compare with the conjugated phase-0 channel: max |Choi_phi - R Choi_0 R^dag|."""
        phi = COVARIANCE_CHECK_PHASE_RAD
        pieces, _notes = extract_pieces(
            job, device, table, entry.kind, entry.addressed, entry.angle, phase_rad=phi
        )
        worst = 0.0
        for p0, p1 in zip(entry.pieces, pieces):
            phases = {ion: phi for ion in p0.ions}
            predicted = conjugate_choi(p0.choi, frame_rotation(p0.ions, phases))
            worst = max(worst, float(np.max(np.abs(predicted - p1.choi))))
        return worst

    def kraus(self, entry: ChannelEntry, piece_index: int) -> list[np.ndarray]:
        k = f"{entry.key}#{piece_index}"
        if k not in self._kraus:
            self._kraus[k] = [
                np.asarray(m, dtype=complex) for m in core.kraus_operators(entry.pieces[piece_index].choi)
            ]
        return self._kraus[k]


def frame_rotation(ions: Sequence[int], phases: Mapping[int, float]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for ion in ions:
        out = np.kron(out, rz(float(phases.get(ion, 0.0))))
    return out


# ---- the replay -----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayGateApplied:
    gate_id: str
    key: str
    ions: tuple[int, ...]
    phases: dict[int, float]
    residual: float
    average_gate_infidelity: float
    depolarizing_rate: float


@dataclass(frozen=True)
class ReplayOutcome:
    compile: Any
    """The core ``CompileReport``."""
    schedule: Any
    """The core ``Schedule``."""
    selection: Any
    """The core ``SpaceSelection``."""
    preparation: Any
    """The core ``PreparationRun``."""
    targets: tuple[Any, ...]
    """The schedule's ``GateTarget`` pieces in time order."""
    applied: tuple[ReplayGateApplied, ...]
    register_after: np.ndarray
    """(n_gates, 2^n, 2^n): the register after each applied piece, register order."""
    final: np.ndarray
    levels: np.ndarray
    """(shots, n_ions) the sampled computational level of every ion, register order (ion 0 first)."""
    bits: np.ndarray
    """(shots, n_ions) the declared bit of every ion through the table's readout errors."""
    measured: tuple[int, ...]
    """The circuit's measured qubits in ascending order (its terminal ``measure`` plus every trailing measure operation, the
    rule the core's pipeline applies): the columns of the record's bitstrings, a subset of the ions when the circuit says so."""
    bright_levels: tuple[int, ...]
    spam: dict[str, tuple[float, float]]
    residual_terms: dict[str, float]
    residual_total: float
    wall_time_s: float
    notes: tuple[str, ...]


def preparation_for(job: JobSpec, device: core.Device) -> Any:
    """The device's recipe run, as ``run()`` prepares it (Section 4.2.6): the device's own, else the standard one on the
    entangling drives' Raman pair."""
    if device.preparation is not None:
        return core.run_preparation(device, device.preparation)
    pair: tuple[int, int] | None = None
    for d in job.entangling_drives.values():
        if d.kind == "raman" and len(d.beams) == 2:
            pair = (int(d.beams[0]), int(d.beams[1]))
            break
    return core.run_preparation(device, core.standard_recipe(device, raman_pair=pair))


def readout_polarity(device: core.Device) -> tuple[int, ...]:
    """The bright computational level per ion from the species' readout scheme under the device's beams (Section 13 row
    'Bright/dark polarity': a property of the scheme, never of the qubit)."""
    out: list[int] = []
    cache: dict[str, int] = {}
    for sp in device.crystal.species:
        if sp.name not in cache:
            lam = float(sp.transition(sp.cycling).wavelength_vac_m)
            # the resonant detection light only: the far-detuned gate beams would make the Bloch build time dependent
            detection = [b for b in device.beams if abs(b.wavelength_m - lam) < RESONANT_WINDOW_M]
            if not detection:
                raise RecordError(
                    f"{sp.name}: no beam within {RESONANT_WINDOW_M * 1e9:.1f} nm of the cycling line {sp.cycling}"
                )
            _rates, scheme, _model = core.detection_rates_for_ion(
                sp, device.field.B_gauss, device.field.direction, detection
            )
            cache[sp.name] = int(scheme.bright_level)
        out.append(cache[sp.name])
    return tuple(out)


def replay(
    job: JobSpec,
    device: core.Device,
    table: core.CalibrationTable,
    library: ChannelLibrary,
    *,
    progress: Progress | None = None,
) -> ReplayOutcome:
    """Compile, schedule, derive the channels the schedule needs, apply them to the register and read it out."""
    t_start = time.perf_counter()
    n = device.crystal.n_ions
    if n > REGISTER_DM_MAX_QUBITS:
        raise RecordError(
            f"the replay carries the register as a density matrix up to {REGISTER_DM_MAX_QUBITS} qubits; {n} ions need the sampling path (not in M11.2)"
        )
    if not library.matches(job, device, table):
        raise RecordError("the channel library was built for another device, table or solver options")
    circuit = job.circuit.to_core()
    opts = job.solver_options()
    if progress:
        progress("compiling", 0.0, "compiling to native gates")
    machine = job.machine(device, table)
    report = machine.compile(circuit)
    if progress:
        progress("scheduling", 0.05, "scheduling pulses from the calibration table")
    sched = core.schedule(
        report.circuit,
        machine.device,
        table,
        t0_s=0.0,
        parallel=None,
        crosstalk_suppression="none",
        stark_compensation=job.stark_compensation,
    )
    if progress:
        progress("preparing", 0.1, "the preparation recipe and the mode classes")
    prep = preparation_for(job, device)
    selection = core.select_space(device, sched, opts, nbar=prep.nbar, caps=job.caps, ion_dims=[2] * n)
    targets = tuple(sorted(sched.targets, key=lambda t: (t.t_start_s, t.gate_id)))
    # the initial register: every ion pumped to |0> with its preparation error in |1> (Section 4.2.6)
    rho = np.array([[1.0 + 0.0j]])
    for i in range(n):
        p = float(prep.preparation_error(i)) if i in prep.pumps else 0.0
        rho = np.kron(rho, np.diag([1.0 - p, p]).astype(complex))
    applied: list[ReplayGateApplied] = []
    states: list[np.ndarray] = []
    terms = {
        "residual_displacement": 0.0,
        "frozen_excitation": 0.0,
        "dropped_crosstalk": 0.0,
        "cp_tp_projection": 0.0,
        "frame_covariance": 0.0,
    }
    n_targets = len(targets)
    for k, tg in enumerate(targets):
        name, params = tg.native
        if progress:
            progress(
                "replaying", 0.15 + 0.7 * k / max(n_targets, 1), f"gate {k + 1}/{n_targets}: {tg.gate_id}"
            )
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
        gate_residual = 0.0
        for j, piece in enumerate(entry.pieces):
            # neighbours of the step see the same light as the addressed ion(s): their frame turns by the addressed phase
            # (the mean of the two for an MS pair), which the covariance check measures rather than assumes
            default_phase = float(np.mean(list(phases.values()))) if phases else 0.0
            full = {ion: phases.get(ion, default_phase) for ion in piece.ions}
            rot = frame_rotation(piece.ions, full)
            kraus = [rot @ km @ rot.conj().T for km in library.kraus(entry, j)]
            new = np.zeros_like(rho)
            for km in kraus:
                big = embed_on(km, piece.ions, tuple(range(n)))
                new += big @ rho @ big.conj().T
            rho = new
            gate_residual += piece.derivation_residual
            terms["residual_displacement"] += piece.residual_bound
            terms["frozen_excitation"] += piece.frozen_excitation
            terms["dropped_crosstalk"] += piece.dropped_crosstalk
            terms["cp_tp_projection"] += piece.cp_residual + piece.tp_residual
        cov = entry.covariance_residual or 0.0
        terms["frame_covariance"] += cov
        gate_residual += cov
        # the Stark frame the scheduler absorbed (Section 7.5 item 7) is inside the target unitary and the extracted channel
        states.append(rho.copy())
        applied.append(
            ReplayGateApplied(
                gate_id=str(tg.gate_id),
                key=entry.key,
                ions=tuple(int(i) for i in tg.ions),
                phases=phases,
                residual=gate_residual,
                average_gate_infidelity=entry.average_gate_infidelity,
                depolarizing_rate=entry.depolarizing_rate,
            )
        )
    if progress:
        progress("sampling", 0.9, f"reading out {job.shots} shots through the table's readout errors")
    bright = readout_polarity(device)
    eps_b = float(table.detection["eps_B"].value) if "eps_B" in table.detection else 0.0
    eps_d = float(table.detection["eps_D"].value) if "eps_D" in table.detection else 0.0
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
    bits = levels.copy()
    for i in range(n):
        is_bright = levels[:, i] == bright[i]
        flip = np.where(is_bright, rng.random(len(idx)) < eps_b, rng.random(len(idx)) < eps_d)
        bits[:, i] = np.where(flip, 1 - levels[:, i], levels[:, i])
    # the measured set is the compiled circuit's terminal measure unioned with every trailing measure operation, exactly the
    # rule the core's pipeline applies (run/pipeline.py), so the two engines histogram the same qubits
    declared: list[int] = list(report.circuit.measure)
    for op in report.circuit.ops:
        if op.name == "measure":
            declared.extend(int(q) for q in op.qubits if q not in declared)
    measured = tuple(sorted(int(q) for q in declared)) if declared else tuple(range(n))
    total = float(sum(terms.values()))
    notes = tuple(library.notes) + (
        "channel replay (app-side, Section 5.4): every gate applied as the Section 6.8 channel of that gate kind extracted once by "
        "GATE_LOCAL tomography from the prepared motional state and conjugated to the played phase; correlations between gates "
        "are traced out and bounded by the reported residual",
        f"readout: the calibration table's eps_B = {eps_b:.3g}, eps_D = {eps_d:.3g} per ion with bright levels {bright}",
    )
    return ReplayOutcome(
        compile=report,
        schedule=sched,
        selection=selection,
        preparation=prep,
        targets=targets,
        applied=tuple(applied),
        register_after=np.stack(states) if states else np.zeros((0, 2**n, 2**n), complex),
        final=rho,
        levels=levels,
        bits=bits,
        measured=measured,
        bright_levels=bright,
        spam=spam,
        residual_terms=terms,
        residual_total=total,
        wall_time_s=time.perf_counter() - t_start,
        notes=notes,
    )


def hotter_recipe(
    device: core.Device, *, sideband_pulses_per_order: int, raman_pair: tuple[int, int] | None = None
) -> Any:
    """A preparation recipe with fewer sideband-cooling pulses per order: the deliberately hotter motional state of Section
    9.11 (the recipe stays the device's standard one otherwise)."""
    base = core.standard_recipe(device, raman_pair=raman_pair)
    sb = base.sideband
    if sb is None:
        raise RecordError("the standard recipe of this device has no sideband cooling stage to shorten")
    pulses = {order: min(int(k), int(sideband_pulses_per_order)) for order, k in sb.pulses_per_order.items()}
    return replace(base, sideband=replace(sb, pulses_per_order=pulses))


__all__ = [
    "COVARIANCE_CHECK_PHASE_RAD",
    "ENTANGLING_ANGLE_DIGITS",
    "REGISTER_DM_MAX_QUBITS",
    "ChannelEntry",
    "ChannelLibrary",
    "ChannelPiece",
    "Progress",
    "ReplayGateApplied",
    "ReplayOutcome",
    "channel_key",
    "conjugate_choi",
    "embed_on",
    "extract_pieces",
    "frame_rotation",
    "hotter_recipe",
    "one_gate_circuit",
    "options_digest",
    "preparation_for",
    "readout_polarity",
    "replay",
    "rz",
]
