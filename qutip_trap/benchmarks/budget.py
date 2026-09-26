"""The simulator's own error budget, reported beside every benchmark (PLAN.md Section 6.8).

Three layers of the simulator's own accounting predict what a benchmark should return, so that the benchmark measures the
machine and the budget says how much of it the known physics accounts for:

1. the closed-form error scales every run reports (``Diagnostics.intrinsic_budget``), their summed terms per native gate
   kind and per benchmark unit (a Clifford, a circuit): scales that bound the coherent errors, not their values;
2. the channel of every native gate kind the benchmark compiled to (``gpi2`` and ``gpi`` per ion, ``ms`` and ``zz`` per
   pair) by state-based process tomography: the kind played once as a one-gate circuit at GATE_LOCAL from the prepared
   motional state, its Choi matrix reduced to the benchmarked qubits (the crosstalk neighbours prepared in |0> and traced
   out) and compared with its ideal unitary; the predictions compose these to first order (average infidelities add for
   small independent errors, Magesan et al. 2011);
3. the readout and preparation errors (``Result.spam``), which set an RB curve's SPAM offsets and the readout loss of a
   population or parity.

Every prediction states its composition rule in ``notes``; none of it is fed back into the simulation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.dynamics.engine import ChannelSummary
from qutip_trap.dynamics.tomography import local_ideal
from qutip_trap.noise.summary import (
    apply_choi,
    average_gate_infidelity,
    choi_from_unitary,
    entanglement_infidelity,
)
from qutip_trap.run.gate_local import gate_steps
from qutip_trap.run.job import last_record
from qutip_trap.run.levels import FidelityLevel
from qutip_trap.run.results import IntrinsicBudget, Result

if TYPE_CHECKING:
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.machine import Machine

FULL_MS_RAD = math.pi / 2.0
"""The fully entangling native angle (Section 7.1): MS theta = pi/2, ZZ theta = pi/2."""


def kind_of(name: str, ions: Sequence[int]) -> str:
    """The native gate kind key: ``gpi2[0]``, ``gpi[1]``, ``ms[0,1]``, ``zz[0,1]``."""
    return f"{name}[{','.join(str(int(q)) for q in ions)}]"


def kinds_of_schedule(schedule: Schedule) -> dict[str, str]:
    """gate piece id -> kind key over the schedule's ``targets`` (the native pieces the scheduler played)."""
    return {t.gate_id: kind_of(t.native[0], t.ions) for t in schedule.targets}


def intrinsic_by_kind(budget: IntrinsicBudget, kinds: Mapping[str, str]) -> dict[str, float]:
    """The closed-form intrinsic scales of one run summed per kind: every record's summed terms (``IntrinsicBudget.by_gate``)
    charged to the schedule piece its gate id belongs to (``IntrinsicBudget.by_piece``), so the kinds add up to the run's
    total. Each record is summed over
    the kind's whole schedule entry, the crosstalk it inflicts outside the benchmarked set included (``gpi2[0]``'s
    ``crosstalk`` is the neighbour's rotation error), so the total bounds a larger error than the channel infidelities,
    which are reduced to the benchmarked qubits."""
    out: dict[str, float] = {}
    for piece, val in budget.by_piece(kinds).items():
        out[kinds[piece]] = out.get(kinds[piece], 0.0) + val
    return out


def reduced_choi(choi: np.ndarray, n_factors: int, keep: Sequence[int]) -> np.ndarray:
    """The trace-1 Choi matrix of the channel reduced to the qubit factors ``keep`` (positions in the step's ion order), the
    other factors prepared in |0> and traced out after the map: E_keep(rho) = Tr_rest E(rho (x) |0><0|)."""
    d_all = 2**n_factors
    if choi.shape != (d_all * d_all, d_all * d_all):
        raise ValueError("Choi matrix and factor count disagree")
    keep = [int(k) for k in keep]
    d_k = 2 ** len(keep)
    out = np.zeros((d_k * d_k, d_k * d_k), dtype=complex)
    for i in range(d_k):
        for j in range(d_k):
            # the input |i><j| on the kept factors (x) |0><0| on the rest, in the step's factor order
            rho_in = np.outer(_product_ket(i, keep, n_factors), _product_ket(j, keep, n_factors).conj())
            red = _partial_trace(apply_choi(choi, rho_in), n_factors, keep)
            out[i * d_k : (i + 1) * d_k, j * d_k : (j + 1) * d_k] += red
    # block (i, j) of the Choi matrix is E(|i><j|)/d in the convention of noise/summary.py (first factor the input copy)
    return np.asarray(out / d_k)


def _product_ket(index: int, keep: Sequence[int], n_factors: int) -> np.ndarray:
    """The basis ket with the bits of ``index`` on the kept factors (the lowest kept factor the most significant bit) and
    |0> on every other factor, in the step's factor order."""
    bits = dict(zip(sorted(keep), ((index >> (len(keep) - 1 - k)) & 1 for k in range(len(keep)))))
    full = np.zeros([2] * n_factors, dtype=complex)
    full[tuple(bits.get(f, 0) for f in range(n_factors))] = 1.0
    return full.reshape(-1)


def _partial_trace(rho: np.ndarray, n_factors: int, keep: Sequence[int]) -> np.ndarray:
    r = rho.reshape([2] * (2 * n_factors))
    for k in sorted((k for k in range(n_factors) if k not in keep), reverse=True):
        r = np.trace(r, axis1=k, axis2=k + r.ndim // 2)
    d_k = 2 ** len(keep)
    return np.asarray(r.reshape(d_k, d_k))


def reduced_ideal(u: np.ndarray, n_factors: int, keep: Sequence[int]) -> np.ndarray:
    """The tensor factor of the ideal ``u`` on the qubit factors ``keep``, normalized to a unitary.

    A step's ideal is the product over its ions of each ion's own target (the native gate on the addressed ion, the
    identity on the crosstalk neighbours, ``dynamics.tomography.local_ideal``), so it factors as U_keep (x) U_rest:
    reshaped to R[(i, j), (k, l)] = U_keep[i, j] U_rest[k, l] it has rank one and its leading left singular vector is U_keep
    up to a scalar, normalized away (a global phase leaves ``choi_from_unitary`` unchanged). The <0_rest| U |0_rest>
    block would be U_keep scaled by prod_rest <0|U_rest|0>, zero for a GPi on a traced-out ion."""
    keep_list = [int(k) for k in keep]
    rest = [k for k in range(n_factors) if k not in keep_list]
    if not rest:
        return np.asarray(u, dtype=complex)
    d_k, d_r = 2 ** len(keep_list), 2 ** len(rest)
    arr = np.asarray(u, dtype=complex).reshape([2] * (2 * n_factors))
    order = keep_list + [k + n_factors for k in keep_list] + rest + [k + n_factors for k in rest]
    left, sv, _vh = np.linalg.svd(arr.transpose(order).reshape(d_k * d_k, d_r * d_r))
    if sv[0] <= 0.0 or (len(sv) > 1 and sv[1] > 1e-8 * sv[0]):
        raise ValueError("the step's ideal unitary does not factor over the benchmarked qubits")
    out = np.asarray(left[:, 0].reshape(d_k, d_k))
    scale = math.sqrt(float(np.real(np.trace(out.conj().T @ out))) / d_k)
    if scale <= 0.0:
        raise ValueError("the step's ideal unitary has no factor on the benchmarked qubits")
    out = out / scale
    if np.max(np.abs(out.conj().T @ out - np.eye(d_k))) > 1e-8:
        raise ValueError("the step's ideal unitary does not factor into unitaries on the benchmarked qubits")
    return out


@dataclass(frozen=True)
class StepChannel:
    """One gate step's channel (Section 6.8) with its ideal unitary on the step's ions (the native gate at its
    frame-applied phase followed by the Stark frame, the identity on the crosstalk neighbours; the first ion the first
    factor)."""

    gate_id: str
    ions: tuple[int, ...]
    summary: ChannelSummary
    ideal: np.ndarray
    engine_runs: int
    local_dimension: int

    def infidelity_on(self, qubits: Sequence[int]) -> float:
        """The average gate infidelity of the channel reduced to the benchmarked ``qubits`` among the step's ions (the others
        prepared in |0> and traced out), against the ideal's factor on them. A step addressing an ion outside ``qubits``
        (a neighbour's pulse, which simultaneous RB charges to the qubit it rotates by crosstalk) has the identity there,
        so the number is the crosstalk error per neighbour pulse."""
        keep = [k for k, q in enumerate(self.ions) if q in set(int(x) for x in qubits)]
        if not keep:
            return 0.0
        n = len(self.ions)
        if len(keep) == n:
            choi, ideal = self.summary.choi, self.ideal
        else:
            choi, ideal = reduced_choi(self.summary.choi, n, keep), reduced_ideal(self.ideal, n, keep)
        eps = entanglement_infidelity(choi, choi_from_unitary(ideal))
        return float(average_gate_infidelity(max(eps, 0.0), 2 ** len(keep)))


@dataclass(frozen=True)
class GateChannel:
    """The GATE_LOCAL channels of one native gate kind played once as a one-gate circuit: one step per piece of its
    scheduler expansion (the ZZ wrapper on an MS device has several)."""

    kind: str
    steps: tuple[StepChannel, ...]
    notes: tuple[str, ...] = ()

    def infidelity_on(self, qubits: Sequence[int]) -> float:
        """First-order composition over the pieces of the reduced average infidelities."""
        return float(sum(s.infidelity_on(qubits) for s in self.steps))


_CHANNEL_CACHE: dict[tuple[object, ...], GateChannel] = {}


def one_gate_circuit(kind: str, n_qubits: int) -> Circuit:
    """The one-gate circuit of a kind key at phase 0 and the fully entangling angle."""
    name, rest = kind.split("[", 1)
    ions = tuple(int(x) for x in rest.rstrip("]").split(","))
    if name in ("gpi", "gpi2"):
        op = Operation(name, ions, (0.0,))
    elif name == "ms":
        op = Operation("ms", ions, (0.0, 0.0, FULL_MS_RAD))
    elif name == "zz":
        op = Operation("zz", ions, (FULL_MS_RAD,))
    else:
        raise ValueError(f"not a native gate kind: {kind!r}")
    return Circuit(n_qubits, (op,), ions)


def gate_channel(machine: Machine, kind: str) -> GateChannel:
    """The channel of one native gate kind by GATE_LOCAL tomography on ``machine`` (module docstring, item 2), cached
    within the process per ``Machine.hash()`` (the device with its roles, the table and the option objects) and kind."""
    local = replace(machine, level=FidelityLevel.GATE_LOCAL)
    key: tuple[object, ...] = (local.hash(), kind)
    if key in _CHANNEL_CACHE:
        return _CHANNEL_CACHE[key]
    res = local.run(one_gate_circuit(kind, local.device.crystal.n_ions), 1)
    gl = res.diagnostics.gate_local
    assert gl is not None
    walk = {s.gate_id: s for s in gate_steps(last_record(res).schedule) if s.kind == "gate"}
    steps: list[StepChannel] = []
    for st in gl.steps:
        if st.kind != "gate" or st.summary is None:
            continue
        gs = walk.get(st.gate_id)
        if gs is None:
            raise RuntimeError(f"GATE_LOCAL step {st.gate_id!r} has no schedule counterpart")
        ideal = local_ideal(st.ions, [2] * len(st.ions), gs.targets)
        if ideal is None:
            raise RuntimeError(f"GATE_LOCAL step {st.gate_id!r} has no ideal unitary")
        steps.append(
            StepChannel(
                gate_id=st.gate_id,
                ions=tuple(int(q) for q in st.ions),
                summary=st.summary,
                ideal=ideal,
                engine_runs=int(st.engine_runs),
                local_dimension=int(np.prod(st.space_dims)),
            )
        )
    _CHANNEL_CACHE[key] = GateChannel(kind=kind, steps=tuple(steps), notes=tuple(gl.notes))
    return _CHANNEL_CACHE[key]


@dataclass(frozen=True)
class BenchmarkBudget:
    """What the simulator's own physics accounts for, beside a benchmark's measured number (module docstring): per native
    gate kind the average pieces per ``unit`` (``clifford``, ``computational gate`` or ``circuit``), the intrinsic scales
    per unit (and their ``total``), the channels and their average infidelities reduced to the benchmarked ``qubits``; per
    qubit ``q{i}`` -> (eps_B, eps_D) and ``q{i}.state_preparation`` -> (eps_prep, 0); the predictions and their rules."""

    unit: str
    qubits: tuple[int, ...]
    counts: dict[str, float]
    intrinsic: dict[str, float]
    spam: dict[str, tuple[float, float]]
    channels: dict[str, GateChannel]
    channel_infidelity: dict[str, float]
    predicted: dict[str, float]
    notes: tuple[str, ...]


def gather_counts_and_intrinsic(
    results: Sequence[Result], units: Sequence[int]
) -> tuple[dict[str, float], dict[str, float]]:
    """Per kind, the average piece count and the average intrinsic scale per unit over several runs (``units[k]``
    benchmark units in run ``k``), and the intrinsic ``total`` per unit."""
    counts: dict[str, float] = {}
    intrinsic: dict[str, float] = {}
    total_units = float(sum(units))
    if total_units <= 0.0:
        raise ValueError("at least one benchmark unit")
    intrinsic_total = 0.0
    for res in results:
        kinds = kinds_of_schedule(last_record(res).schedule)
        for kind in kinds.values():
            counts[kind] = counts.get(kind, 0.0) + 1.0
        for kind, val in intrinsic_by_kind(res.diagnostics.intrinsic_budget, kinds).items():
            intrinsic[kind] = intrinsic.get(kind, 0.0) + val
        intrinsic_total += res.diagnostics.intrinsic_budget.total
    counts = {k: v / total_units for k, v in counts.items()}
    intrinsic = {k: v / total_units for k, v in intrinsic.items()}
    intrinsic["total"] = intrinsic_total / total_units
    return counts, intrinsic


def spam_of(result: Result, qubits: Sequence[int]) -> dict[str, tuple[float, float]]:
    """The run's readout (eps_B, eps_D) and preparation error of each benchmarked qubit (``BenchmarkBudget.spam``)."""
    out: dict[str, tuple[float, float]] = {}
    for q in qubits:
        for key in (f"q{q}", f"q{q}.state_preparation"):
            if key in result.spam:
                out[key] = (float(result.spam[key][0]), float(result.spam[key][1]))
    return out


def channels_for(
    machine: Machine, kinds: Sequence[str], qubits: Sequence[int]
) -> tuple[dict[str, GateChannel], dict[str, float]]:
    """The channel of every kind and its average infidelity reduced to ``qubits`` (``gate_channel`` per kind)."""
    channels = {kind: gate_channel(machine, kind) for kind in sorted(set(kinds))}
    return channels, {kind: ch.infidelity_on(qubits) for kind, ch in channels.items()}


def _shot_sigma(result: Result, variance: float) -> float:
    """sqrt(variance/n_eff) over the run's effective sample size, at least one count in n_eff."""
    n_eff = max(float(result.diagnostics.effective_sample_size), 1.0)
    return math.sqrt(max(variance, 0.0) / n_eff) or 1.0 / n_eff
