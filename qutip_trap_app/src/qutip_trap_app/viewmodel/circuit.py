"""Level 1, the circuit: the compiled native gates in time order and the register after each of them, from the recorded
traces or from a GATE_LOCAL run's own register."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import Record, TargetRecord, TraceRecord
from qutip_trap_app.viewmodel.shown import Shown


class RegisterUnavailable(KeyError):
    """The record holds nothing the register after a gate can be read from; the message says why."""


PAULI: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# ---- linear algebra on the register ----------------------------------------------------------------------------------------------


def register_from_joint(joint: np.ndarray, joint_dims: tuple[int, ...], n_ions: int) -> np.ndarray:
    """The reduced register density matrix of a joint ket or density matrix over ion and mode factors."""
    d_int = int(np.prod(joint_dims[:n_ions]))
    d_mot = int(np.prod(joint_dims[n_ions:])) if len(joint_dims) > n_ions else 1
    arr = np.asarray(joint, dtype=complex)
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        psi = arr.reshape(d_int, d_mot)
        return np.asarray(psi @ psi.conj().T)
    return np.asarray(np.einsum("iaja->ij", arr.reshape(d_int, d_mot, d_int, d_mot)))


def embed_operator(op: np.ndarray, ions: tuple[int, ...], n_ions: int) -> np.ndarray:
    """``op`` on ``ions`` (in the matrix's factor order) as an operator on the whole two-level register."""
    others = [i for i in range(n_ions) if i not in ions]
    order = [*ions, *others]
    # op (x) 1 acts on the factors in ``order``; its tensor axes are moved back to the register order
    full = np.kron(np.asarray(op, dtype=complex), np.eye(2 ** len(others), dtype=complex))
    axes = [order.index(i) for i in range(n_ions)]
    tensor = full.reshape([2] * (2 * n_ions)).transpose(axes + [n_ions + a for a in axes])
    return np.asarray(tensor.reshape(2**n_ions, 2**n_ions))


def pauli_operator(labels: str) -> np.ndarray:
    """The Pauli string on the register, character k acting on ion k."""
    op = np.array([[1.0 + 0.0j]])
    for ch in labels:
        op = np.kron(op, PAULI[ch])
    return op


def populations(rho: np.ndarray, n_ions: int) -> dict[str, float]:
    """diag(rho) keyed by bitstring with qubit 0 rightmost (the histogram's keys)."""
    return {format(k, f"0{n_ions}b")[::-1]: float(p) for k, p in enumerate(np.real(np.diag(rho)))}


def pauli_expectations(rho: np.ndarray, n_ions: int) -> dict[str, float]:
    """<P> for every Pauli string but the identity."""
    return {
        "".join(labels): float(np.real(np.trace(rho @ pauli_operator("".join(labels)))))
        for labels in itertools.product("IXYZ", repeat=n_ions)
        if set(labels) != {"I"}
    }


def bloch_vectors(rho: np.ndarray, n_ions: int) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for i in range(n_ions):
        comps = [
            float(np.real(np.trace(rho @ pauli_operator("I" * i + axis + "I" * (n_ions - i - 1)))))
            for axis in "XYZ"
        ]
        out[i] = (comps[0], comps[1], comps[2])
    return out


def single_ion_reduced(rho: np.ndarray, ion: int, n_ions: int) -> np.ndarray:
    """The 2 x 2 reduced state of one ion of a two-level register."""
    r = np.asarray(rho, dtype=complex).reshape([2] * (2 * n_ions))
    for j in sorted((k for k in range(n_ions) if k != ion), reverse=True):
        r = np.trace(r, axis1=j, axis2=r.ndim // 2 + j)
    return np.asarray(r.reshape(2, 2))


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def concurrence(rho: np.ndarray) -> float:
    """Wootters' concurrence of a two-qubit density matrix."""
    if rho.shape != (4, 4):
        raise ValueError("concurrence is defined here for two qubits")
    yy = np.kron(PAULI["Y"], PAULI["Y"])
    ev = np.sort(np.sqrt(np.clip(np.real(np.linalg.eigvals(rho @ yy @ rho.conj() @ yy)), 0.0, None)))[::-1]
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


def fidelity_to_ket(rho: np.ndarray, ket: np.ndarray) -> float:
    v = np.asarray(ket, dtype=complex).reshape(-1)
    return float(np.real(v.conj() @ rho @ v))


# ---- the timeline ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateView:
    index: int
    gate_id: str
    name: str
    params_rad: tuple[float, ...]
    """Phases with the virtual-Z frame applied."""
    ions: tuple[int, ...]
    t_start_s: float
    t_end_s: float
    duration: Shown
    step_index: int
    unitary: np.ndarray
    """The ideal unitary on ``ions`` in matrix order."""


def _targets(record: Record) -> tuple[TargetRecord, ...]:
    return tuple(sorted(record.schedule.targets, key=lambda t: (t.t_start_s, t.gate_id)))


def timeline(record: Record) -> tuple[GateView, ...]:
    return tuple(
        GateView(
            index=k,
            gate_id=tg.gate_id,
            name=tg.native_name,
            params_rad=tg.native_params,
            ions=tg.ions,
            t_start_s=tg.t_start_s,
            t_end_s=tg.t_end_s,
            duration=Shown("Duration", tg.t_end_s - tg.t_start_s, "s"),
            step_index=record.step_of_gate(tg.gate_id).index,
            unitary=tg.unitary,
        )
        for k, tg in enumerate(_targets(record))
    )


# ---- the register after gate k ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisterView:
    rho: np.ndarray
    """The reduced register density matrix, register order."""
    populations: dict[str, float]
    bloch: dict[int, tuple[float, float, float]]
    target_bloch: dict[int, tuple[float, float, float]]
    pauli: dict[str, float]
    purity: Shown
    fidelity: Shown
    """Against the ideal register after this gate: the target unitaries so far applied to |0...0>."""
    note: str


def target_ket(targets: Sequence[tuple[np.ndarray, tuple[int, ...]]], n_ions: int) -> np.ndarray:
    """The unitaries (each on its ions in matrix order) applied in sequence to |0...0>, in the register order."""
    ket = np.zeros(2**n_ions, dtype=complex)
    ket[0] = 1.0
    for unitary, ions in targets:
        ket = embed_operator(unitary, tuple(ions), n_ions) @ ket
    return ket


def target_ket_after(record: Record, gate_index: int) -> np.ndarray:
    return target_ket([(tg.unitary, tg.ions) for tg in _targets(record)[: gate_index + 1]], record.n_ions)


def gate_local_register_after(record: Record, gate_index: int) -> np.ndarray:
    """The register after a gate of a GATE_LOCAL run: the walk's own register after the gate's step."""
    gl = record.gate_local
    if gl is None:
        raise RegisterUnavailable("this record stores neither traces nor the GATE_LOCAL walk's registers")
    step = record.step_of_gate(_targets(record)[gate_index].gate_id)
    for st in gl.steps:
        if st.gate_id == step.gate_id:
            if st.register_after is None:
                raise RegisterUnavailable(
                    f"the core kept no register after {st.gate_id} (a pure-state ensemble, or above its store cap)"
                )
            return np.asarray(st.register_after, dtype=complex)
    raise RegisterUnavailable(f"no GATE_LOCAL step carries gate {step.gate_id!r}")


def _weights(record: Record, sample_index: int | None, branch: int | None) -> list[tuple[TraceRecord, float]]:
    traces = [
        tr
        for tr in record.traces
        if (sample_index is None or tr.sample_index == sample_index)
        and (branch is None or tr.branch == branch)
    ]
    if not traces:
        raise RegisterUnavailable("no recorded trace matches")
    shots = record.diagnostics.shots_per_sample or (1,) * record.n_samples
    weights = [
        (
            tr,
            (1.0 if sample_index is not None else shots[tr.sample_index] / sum(shots))
            * (1.0 if branch is not None else tr.weight),
        )
        for tr in traces
    ]
    total = sum(w for _tr, w in weights)
    return [(tr, w / total) for tr, w in weights]


def register_after(
    record: Record, gate_index: int, *, sample_index: int | None = None, branch: int | None = None
) -> RegisterView:
    """The register after gate ``gate_index``: from the recorded traces at the gate's end (weighted over the branches and
    samples unless one is selected), else from a GATE_LOCAL run's own register. The register spans every ion."""
    n = record.n_ions
    if record.traces:
        t_end = _targets(record)[gate_index].t_end_s
        rho = np.zeros((2**n, 2**n), dtype=complex)
        for tr, w in _weights(record, sample_index, branch):
            rho += w * tr.reduced_internal[tr.index_at(t_end)]
        note = "the recorded traces at the gate's end, weighted over the branches and samples"
    else:
        rho = gate_local_register_after(record, gate_index)
        note = "the GATE_LOCAL walk's register after this step (the first dynamical sample)"
    return register_view(record, gate_index, rho, note)


def register_view(record: Record, gate_index: int, rho: np.ndarray, note: str) -> RegisterView:
    n = record.n_ions
    ket = target_ket_after(record, gate_index)
    return RegisterView(
        rho=rho,
        populations=populations(rho, n),
        bloch=bloch_vectors(rho, n),
        target_bloch=bloch_vectors(np.outer(ket, ket.conj()), n),
        pauli=pauli_expectations(rho, n),
        purity=Shown("Purity", purity(rho), detail="Tr rho^2: 1 for a pure state"),
        fidelity=Shown("Fidelity to the ideal", fidelity_to_ket(rho, ket), detail="<ideal| rho |ideal>"),
        note=note,
    )
