"""Level 1, the circuit: the compiled native-gate timeline, the register after each gate, the phase register
(PLAN.md Section 14.2, row 1; Section 9.11 row "Coarse-graining identity").

The register state after gate k is read from the recorded traces (the reduced internal density matrix at the gate's end,
weighted over the initial mixture's branches and the run's samples), and :func:`register_from_joint` computes the same
quantity from a joint Level 3 state by partial trace. Section 9.11: "populations and Pauli expectations shown at Level 1
after gate k, computed from the recorded Level 3 joint state, equal the reduced-density-matrix values to 1e-12".

Orders (Section 13): the register order has ion 0 as the FIRST tensor factor (``conv.computational_ordering``); the
compiler's bit order has qubit 0 as the least-significant index bit, and bitstring keys read qubit 0 rightmost.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import ChannelSummaryRecord, Record, TargetRecord, TraceRecord
from qutip_trap_app.viewmodel.catalogue import Shown

PAULI: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# ---- linear algebra on the register ------------------------------------------------------------------------------------------------


def register_from_joint(joint: np.ndarray, joint_dims: tuple[int, ...], n_ions: int) -> np.ndarray:
    """The reduced register density matrix (register order) of a joint ket or density matrix over ion and mode factors:
    the coarse-graining of Level 3 into Level 1."""
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
    out = np.zeros((2**n_ions, 2**n_ions), dtype=complex)
    k = len(ions)
    op_t = np.asarray(op, dtype=complex).reshape([2] * (2 * k))
    others = [i for i in range(n_ions) if i not in ions]
    for rest in itertools.product((0, 1), repeat=len(others)):
        for a in itertools.product((0, 1), repeat=k):
            for b in itertools.product((0, 1), repeat=k):
                val = op_t[a + b]
                if val == 0.0:
                    continue
                row = [0] * n_ions
                col = [0] * n_ions
                for ion, bit_a, bit_b in zip(ions, a, b):
                    row[ion] = bit_a
                    col[ion] = bit_b
                for ion, bit in zip(others, rest):
                    row[ion] = bit
                    col[ion] = bit
                r = int("".join(str(x) for x in row), 2)
                c = int("".join(str(x) for x in col), 2)
                out[r, c] += val
    return out


def populations(rho: np.ndarray, n_ions: int) -> dict[str, float]:
    """diag(rho) keyed by bitstring with qubit 0 RIGHTMOST (the histogram convention), from the register order."""
    diag = np.real(np.diag(rho))
    out: dict[str, float] = {}
    for idx, p in enumerate(diag):
        bits = format(idx, f"0{n_ions}b")  # register order: ion 0 the most-significant character
        out[bits[::-1]] = float(p)
    return out


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


def bloch_vectors(rho: np.ndarray, n_ions: int) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for i in range(n_ions):
        comps = []
        for axis in "XYZ":
            labels = ["I"] * n_ions
            labels[i] = axis
            op = np.array([[1.0 + 0.0j]])
            for ch in labels:
                op = np.kron(op, PAULI[ch])
            comps.append(float(np.real(np.trace(rho @ op))))
        out[i] = (comps[0], comps[1], comps[2])
    return out


def single_ion_reduced(rho: np.ndarray, ion: int, n_ions: int) -> np.ndarray:
    """The 2 x 2 reduced state of one ion of a two-level register (register order, ion 0 first)."""
    r = np.asarray(rho, dtype=complex).reshape([2] * (2 * n_ions))
    for j in sorted((k for k in range(n_ions) if k != ion), reverse=True):
        half = r.ndim // 2
        r = np.trace(r, axis1=j, axis2=half + j)
    return np.asarray(r.reshape(2, 2))


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


# ---- the timeline --------------------------------------------------------------------------------------------------------------------


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
            pair = f"{tg.ions[0]},{tg.ions[1]}"
            wf = table.waveforms.get(pair) or table.waveforms.get(f"{tg.ions[1]},{tg.ions[0]}")
            if wf is not None:
                calibrated.append(
                    Shown(
                        "entangling_angle",
                        wf.chi_total_rad,
                        f"signed |chi| of the pair's waveform ({wf.phi_s.status})",
                    )
                )
        else:
            drive = record.job.gate_drives.get(tg.ions[0])
            if drive is not None:
                key = f"rabi[({tg.ions[0]}, {drive.beams[0] if drive.beams else -1})]"
                entry = table.entries.get(key)
                if entry is not None:
                    calibrated.append(
                        Shown("rabi_frequency", entry.value, f"{entry.status}, {entry.provenance_id}")
                    )
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
                unitary=Shown(
                    "gate_unitary", "U", f"{2 ** len(tg.ions)} x {2 ** len(tg.ions)} target unitary"
                ),
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


# ---- the register after gate k -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisterView:
    gate_index: int
    t_s: float
    rho: np.ndarray
    """Reduced register density matrix, register order."""
    populations: dict[str, Shown]
    bloch: dict[int, tuple[float, float, float]]
    pauli: dict[str, Shown]
    purity: Shown
    target_ket: np.ndarray
    """The ideal register state after this gate (the target unitaries so far applied to |0...0>), register order."""
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
        raise KeyError("no recorded traces match (a GATE_LOCAL run stores none: see core_gaps)")
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
        [(tg.unitary, tg.ions) for tg in _target_by_time(record)[: gate_index + 1]], record.n_qubits
    )


def register_after(
    record: Record, gate_index: int, *, sample_index: int | None = None, branch: int | None = None
) -> RegisterView:
    """The register after gate ``gate_index``: from the recorded traces (the reduced state at the gate's end time, weighted
    over the branches and samples unless one is selected), or from the channel replay's own register sequence for a
    CHANNEL_REPLAY record (labelled derived). A GATE_LOCAL record stores neither (``core_gaps``)."""
    n = record.n_qubits
    tg = _target_by_time(record)[gate_index]
    t_end = tg.t_end_s
    if not record.traces and record.replay is not None:
        rho = np.asarray(record.replay.register_after[gate_index], dtype=complex)
        return _register_view(
            record, gate_index, t_end, rho, "derived: the channel replay's register after this gate's channel"
        )
    rho = np.zeros((2**n, 2**n), dtype=complex)
    for tr, w in _weights(record, sample_index, branch):
        idx = tr.index_at(t_end)
        rho += w * tr.reduced_internal[idx]
    return _register_view(
        record, gate_index, t_end, rho, "branch- and shot-weighted over the recorded traces"
    )


def register_from_state(record: Record, gate_index: int, rho: np.ndarray) -> RegisterView:
    tg = _target_by_time(record)[gate_index]
    return _register_view(record, gate_index, tg.t_end_s, rho, "from a joint state by partial trace")


def _register_view(record: Record, gate_index: int, t_s: float, rho: np.ndarray, note: str) -> RegisterView:
    n = record.n_qubits
    ket = target_ket_after(record, gate_index)
    return RegisterView(
        gate_index=gate_index,
        t_s=t_s,
        rho=rho,
        populations={k: Shown("population", v) for k, v in populations(rho, n).items()},
        bloch=bloch_vectors(rho, n),
        pauli={k: Shown("pauli_expectation", v) for k, v in pauli_expectations(rho, n).items()},
        purity=Shown("purity", purity(rho)),
        target_ket=ket,
        fidelity=Shown("state_fidelity", fidelity_to_ket(rho, ket)),
        weights_note=note,
    )


@dataclass(frozen=True)
class PhaseRegister:
    """The virtual-Z frame per ion at the end of the circuit and the increments the scheduler absorbed (Section 7.6)."""

    final_frame: dict[int, Shown]
    compiler_frame: dict[int, Shown]
    stark_increments: tuple[tuple[str, int, Shown], ...]
    rule: str


def phase_register(record: Record) -> PhaseRegister:
    incs = []
    for tg in _target_by_time(record):
        for q, v in sorted(tg.stark_frame_rad.items()):
            if v != 0.0:
                incs.append((tg.gate_id, q, Shown("stark_frame", v)))
    return PhaseRegister(
        final_frame={q: Shown("phase_frame", v) for q, v in sorted(record.schedule.phase_frame.items())},
        compiler_frame={
            q: Shown("phase_frame", v) for q, v in sorted(record.compiled.final_frame_rad.items())
        },
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


__all__ = [
    "PAULI",
    "GateView",
    "PhaseRegister",
    "RegisterView",
    "bloch_vectors",
    "compile_report",
    "concurrence",
    "embed_operator",
    "fidelity_to_ket",
    "infidelity_budget_check",
    "pauli_expectations",
    "phase_register",
    "populations",
    "purity",
    "register_after",
    "register_from_joint",
    "register_from_state",
    "single_ion_reduced",
    "target_ket",
    "target_ket_after",
    "timeline",
]
