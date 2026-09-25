"""Level 1, the circuit: the compiled native-gate timeline, the register after each gate, the phase register (PLAN.md
Section 14.2).

The register after gate k is the reduced internal density matrix at the gate's end, read from the recorded traces and
weighted over the initial mixture's branches and the run's samples; :func:`register_from_joint` computes the same quantity
from a joint Level 3 state by partial trace. The register order has ion 0 as the FIRST tensor factor
(``conv.computational_ordering``); bitstring keys read qubit 0 rightmost.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import (
    ChannelSummaryRecord,
    Record,
    TableRecord,
    TargetRecord,
    TraceRecord,
    WaveformRecord,
)
from qutip_trap_app.viewmodel.catalogue import Shown


class RegisterUnavailable(KeyError):
    """The record holds nothing the register after a gate can be read from: no recorded trace, no replay register and no
    GATE_LOCAL register. The message names the gap; Level 1 shows it in place of the register."""


PAULI: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# ---- linear algebra on the register ----------------------------------------------------------------------------------


def register_from_joint(joint: np.ndarray, joint_dims: tuple[int, ...], n_ions: int) -> np.ndarray:
    """The reduced register density matrix (register order) of a joint ket or density matrix over ion and mode factors."""
    dims = list(joint_dims)
    d_int = int(np.prod(dims[:n_ions]))
    d_mot = int(np.prod(dims[n_ions:])) if len(dims) > n_ions else 1
    arr = np.asarray(joint, dtype=complex)
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        psi = arr.reshape(d_int, d_mot)
        return np.asarray(psi @ psi.conj().T)
    rho = arr.reshape(d_int, d_mot, d_int, d_mot)
    return np.asarray(np.einsum("iaja->ij", rho))


def embed_operator(op: np.ndarray, ions: tuple[int, ...], n_ions: int) -> np.ndarray:
    """``op`` on ``ions`` (in the matrix's factor order) as an operator on the whole two-level register in register order."""
    rest = [i for i in range(n_ions) if i not in ions]
    full = np.kron(np.asarray(op, dtype=complex), np.eye(2 ** len(rest), dtype=complex))
    order = list(ions) + rest  # the ion on each tensor factor of ``full``
    perm = [order.index(i) for i in range(n_ions)]
    t = full.reshape([2] * (2 * n_ions)).transpose(perm + [n_ions + p for p in perm])
    return np.asarray(t.reshape(2**n_ions, 2**n_ions))


def populations(rho: np.ndarray, n_ions: int) -> dict[str, float]:
    """diag(rho) keyed by bitstring with qubit 0 RIGHTMOST (the histogram convention), from the register order."""
    return {format(idx, f"0{n_ions}b")[::-1]: float(p) for idx, p in enumerate(np.real(np.diag(rho)))}


def pauli_expectations(rho: np.ndarray, n_ions: int) -> dict[str, float]:
    """<P> for every Pauli string (label character k = ion k) with the identity string omitted."""
    out: dict[str, float] = {}
    for labels in itertools.product("IXYZ", repeat=n_ions):
        if set(labels) == {"I"}:
            continue
        op = np.array([[1.0 + 0.0j]])
        for ch in labels:
            op = np.kron(op, PAULI[ch])
        out["".join(labels)] = float(np.real(np.trace(rho @ op)))
    return out


def single_ion_reduced(rho: np.ndarray, ion: int, n_ions: int) -> np.ndarray:
    """The 2 x 2 reduced state of one ion of a two-level register (register order, ion 0 first)."""
    r = np.asarray(rho, dtype=complex).reshape([2] * (2 * n_ions))
    for j in sorted((k for k in range(n_ions) if k != ion), reverse=True):
        r = np.trace(r, axis1=j, axis2=r.ndim // 2 + j)
    return np.asarray(r.reshape(2, 2))


def bloch_vectors(rho: np.ndarray, n_ions: int) -> dict[int, tuple[float, float, float]]:
    """(<X>, <Y>, <Z>) of every ion's reduced state."""
    out: dict[int, tuple[float, float, float]] = {}
    for i in range(n_ions):
        r = single_ion_reduced(rho, i, n_ions)
        x, y, z = (float(np.real(np.trace(r @ PAULI[a]))) for a in "XYZ")
        out[i] = (x, y, z)
    return out


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def concurrence(rho: np.ndarray) -> float:
    """Wootters' concurrence of a two-qubit density matrix."""
    if rho.shape != (4, 4):
        raise ValueError("concurrence is defined here for two qubits")
    yy = np.kron(PAULI["Y"], PAULI["Y"])
    r = rho @ yy @ rho.conj() @ yy
    ev = np.sort(np.sqrt(np.clip(np.real(np.linalg.eigvals(r)), 0.0, None)))[::-1]
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


def fidelity_to_ket(rho: np.ndarray, ket: np.ndarray) -> float:
    v = np.asarray(ket, dtype=complex).reshape(-1)
    return float(np.real(v.conj() @ rho @ v))


def pair_waveform(table: TableRecord, pair: tuple[int, int]) -> WaveformRecord | None:
    """The table's entangling waveform of a pair of ions, whichever order the pair is keyed in."""
    a, b = pair
    return table.waveforms.get(f"{a},{b}") or table.waveforms.get(f"{b},{a}")


# ---- the timeline ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateView:
    index: int
    gate_id: str
    name: Shown
    params_rad: tuple[float, ...]
    ions: tuple[int, ...]
    t_start_s: float
    t_end_s: float
    duration: Shown
    unitary: Shown
    matrix: np.ndarray
    """The target unitary (``TargetRecord.unitary``), on ``ions`` in matrix order."""
    step_index: int
    calibrated: tuple[Shown, ...]
    """The table entries the gate was scheduled from: carrier Rabi frequency per ion, or the pair's entangling angle."""
    error_estimate: Shown | None
    channel: ChannelSummaryRecord | None
    stark_frame: tuple[Shown, ...]


def _target_by_time(record: Record) -> tuple[TargetRecord, ...]:
    return tuple(sorted(record.schedule.targets, key=lambda t: (t.t_start_s, t.gate_id)))


def timeline(record: Record) -> tuple[GateView, ...]:
    table = record.table
    card = record.device_card
    out: list[GateView] = []
    for k, tg in enumerate(_target_by_time(record)):
        calibrated: list[Shown] = []
        if tg.native_name in ("ms", "zz"):
            wf = pair_waveform(table, (tg.ions[0], tg.ions[1]))
            if wf is not None:
                detail = f"signed |chi| of the pair's waveform ({wf.phi_s.status})"
                calibrated.append(Shown("entangling_angle", wf.chi_total_rad, detail))
        else:
            drive = record.job.gate_drives.get(tg.ions[0])
            if drive is not None:
                entry = table.entries.get(f"rabi[({tg.ions[0]}, {drive.beams[0] if drive.beams else -1})]")
                if entry is not None:
                    detail = f"{entry.status}, {entry.provenance_id}"
                    calibrated.append(Shown("rabi_frequency", entry.value, detail))
        step = record.step_of_gate(tg.gate_id)
        est = card.gate_error_estimates.get(tg.gate_id)
        channel = None
        if record.gate_local is not None:
            for st in record.gate_local.steps:
                if st.gate_id == step.gate_id and st.summary is not None:
                    channel = st.summary
        if record.replay is not None:
            for g in record.replay.gates:
                if g.gate_id == tg.gate_id and g.key in record.replay.channels:
                    channel = record.replay.channels[g.key].pieces[0].summary
        d = 2 ** len(tg.ions)
        out.append(
            GateView(
                index=k,
                gate_id=tg.gate_id,
                name=Shown("native_gate", tg.native_name, "phases frame-applied (Section 7.6)"),
                params_rad=tg.native_params,
                ions=tg.ions,
                t_start_s=tg.t_start_s,
                t_end_s=tg.t_end_s,
                duration=Shown("gate_duration", tg.t_end_s - tg.t_start_s),
                unitary=Shown("gate_unitary", "U", f"{d} x {d} target unitary"),
                matrix=np.asarray(tg.unitary, dtype=complex),
                step_index=step.index,
                calibrated=tuple(calibrated),
                error_estimate=None if est is None else Shown("gate_error_estimate", est),
                channel=channel,
                stark_frame=tuple(
                    Shown("stark_frame", v, f"ion {q}") for q, v in sorted(tg.stark_frame_rad.items())
                ),
            )
        )
    return tuple(out)


# ---- the register after gate k ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisterView:
    rho: np.ndarray
    """Reduced register density matrix, register order."""
    populations: dict[str, Shown]
    bloch: dict[int, tuple[float, float, float]]
    target_bloch: dict[int, tuple[float, float, float]]
    """The ideal register's Bloch vectors after this gate (the target unitaries so far applied to |0...0>)."""
    pauli: dict[str, Shown]
    purity: Shown
    fidelity: Shown
    weights_note: str


def _weights(record: Record, sample_index: int | None, branch: int | None) -> list[tuple[TraceRecord, float]]:
    traces = [
        tr
        for tr in record.traces
        if (sample_index is None or tr.sample_index == sample_index)
        and (branch is None or tr.branch == branch)
    ]
    if not traces:
        raise RegisterUnavailable("no recorded traces match (a GATE_LOCAL run stores none: see core_gaps)")
    shots = record.diagnostics.shots_per_sample_realized or (1,) * record.n_samples
    weights = []
    for tr in traces:
        w_sample = 1.0 if sample_index is not None else float(shots[tr.sample_index]) / float(sum(shots))
        w_branch = 1.0 if branch is not None else tr.weight
        weights.append((tr, w_sample * w_branch))
    total = sum(w for _tr, w in weights)
    return [(tr, w / total) for tr, w in weights]


def target_ket(targets: Sequence[tuple[np.ndarray, tuple[int, ...]]], n_ions: int) -> np.ndarray:
    """The target unitaries (each on its ions in matrix order) applied in sequence to |0...0>, in the register order."""
    ket = np.zeros(2**n_ions, dtype=complex)
    ket[0] = 1.0
    for unitary, ions in targets:
        ket = embed_operator(unitary, tuple(ions), n_ions) @ ket
    return ket


def target_ket_after(record: Record, gate_index: int) -> np.ndarray:
    """The target unitaries of gates 0..k applied to |0...0>, in the register order."""
    return target_ket(
        [(tg.unitary, tg.ions) for tg in _target_by_time(record)[: gate_index + 1]], record.n_ions
    )


GATE_LOCAL_REGISTER_NOTE = (
    "recorded: the GATE_LOCAL walk's register after this step (the first dynamical sample)"
)


def gate_local_register_after(record: Record, gate_index: int) -> np.ndarray:
    """The register after gate ``gate_index`` of a GATE_LOCAL record: the walk's own register after the gate's step
    (``GateLocalStep.register_after``, the idle intervals' one-qubit channels included)."""
    gl = record.gate_local
    if gl is None:
        raise RegisterUnavailable("this record stores no GATE_LOCAL steps (see core_gaps)")
    tg = _target_by_time(record)[gate_index]
    step = record.step_of_gate(tg.gate_id)
    for st in gl.steps:
        if st.gate_id == step.gate_id:
            if st.register_after is None:
                raise RegisterUnavailable(
                    f"the record carries no register after {st.gate_id}: written before 0.4.0, or a register the core does "
                    "not store (a pure-state ensemble, or above its store cap)"
                )
            return np.asarray(st.register_after, dtype=complex)
    raise RegisterUnavailable(f"no GATE_LOCAL step carries gate {tg.gate_id!r}")


def register_after(
    record: Record, gate_index: int, *, sample_index: int | None = None, branch: int | None = None
) -> RegisterView:
    """The register after gate ``gate_index``: from the recorded traces (the reduced state at the gate's end time, weighted
    over the branches and samples unless one is selected); else the channel replay's register; else the GATE_LOCAL walk's.
    Raises :class:`RegisterUnavailable` when the record holds none of the three. The register spans every ion
    (``record.n_ions``): a circuit on fewer qubits leaves the trailing ions in |0>."""
    t_end = _target_by_time(record)[gate_index].t_end_s
    if not record.traces and record.replay is not None:
        rho = np.asarray(record.replay.register_after[gate_index], dtype=complex)
        note = "derived: the channel replay's register after this gate's channel"
    elif not record.traces and record.gate_local is not None:
        rho, note = gate_local_register_after(record, gate_index), GATE_LOCAL_REGISTER_NOTE
    else:
        n = record.n_ions
        rho = np.zeros((2**n, 2**n), dtype=complex)
        for tr, w in _weights(record, sample_index, branch):
            rho += w * tr.reduced_internal[tr.index_at(t_end)]
        note = "branch- and shot-weighted over the recorded traces"
    n = record.n_ions
    ket = target_ket_after(record, gate_index)
    return RegisterView(
        rho=rho,
        populations={k: Shown("population", v) for k, v in populations(rho, n).items()},
        bloch=bloch_vectors(rho, n),
        target_bloch=bloch_vectors(np.outer(ket, ket.conj()), n),
        pauli={k: Shown("pauli_expectation", v) for k, v in pauli_expectations(rho, n).items()},
        purity=Shown("purity", purity(rho)),
        fidelity=Shown("state_fidelity", fidelity_to_ket(rho, ket)),
        weights_note=note,
    )


@dataclass(frozen=True)
class PhaseRegister:
    """The virtual-Z frame per ion at the end of the circuit and the increments the scheduler absorbed (Section 7.6)."""

    final_frame: dict[int, Shown]
    stark_increments: tuple[tuple[str, int, Shown], ...]
    rule: str


def phase_register(record: Record) -> PhaseRegister:
    incs = [
        (tg.gate_id, q, Shown("stark_frame", v))
        for tg in _target_by_time(record)
        for q, v in sorted(tg.stark_frame_rad.items())
        if v != 0.0
    ]
    return PhaseRegister(
        final_frame={q: Shown("phase_frame", v) for q, v in sorted(record.schedule.phase_frame.items())},
        stark_increments=tuple(incs),
        rule="a virtual Z by theta shifts every later pulse's phase: phi -> phi - theta (conv.virtual_z_propagation)",
    )


def compile_report(record: Record) -> tuple[Shown, ...]:
    c = record.compiled
    out = [Shown("compile_residual", r, f"block {k}") for k, r in enumerate(c.block_residuals)]
    if c.circuit_residual is not None:
        out.append(Shown("compile_residual", c.circuit_residual, "whole circuit"))
    return tuple(out)


def infidelity_budget_check(record: Record) -> tuple[float | None, float, bool]:
    """(1 - register fidelity, the closed-form budget total, inside?) for the run (Section 9.6)."""
    fid = record.results.register_fidelity
    budget = float(record.diagnostics.intrinsic_budget.get("total", math.nan))
    if fid is None:
        return None, budget, False
    return 1.0 - fid, budget, (1.0 - fid) <= budget
