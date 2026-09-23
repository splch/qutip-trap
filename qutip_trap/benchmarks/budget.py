"""The simulator's own error budget beside a benchmark's measured number: the closed-form scales every run reports
(``Diagnostics.intrinsic_budget``) summed per native gate kind, which bound the coherent errors rather than value them;
each kind's GATE_LOCAL channel from a one-gate circuit, reduced to the benchmarked qubits, whose average infidelities
compose to first order (Magesan et al. 2011); and the SPAM errors of ``Result.spam``."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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
from qutip_trap.run.results import Diagnostics, Result

if TYPE_CHECKING:
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.machine import Machine

FULL_MS_RAD = math.pi / 2.0
"""The fully entangling native angle: MS theta = pi/2, ZZ theta = pi/2."""


def kind_of(name: str, ions: Sequence[int]) -> str:
    """The native gate kind key: ``gpi2[0]``, ``gpi[1]``, ``ms[0,1]``, ``zz[0,1]``."""
    return f"{name}[{','.join(str(int(q)) for q in ions)}]"


def kinds_of_schedule(schedule: Schedule) -> dict[str, str]:
    """gate piece id -> kind key over the schedule's ``targets`` (the native pieces the scheduler played)."""
    return {t.gate_id: kind_of(t.native[0], t.ions) for t in schedule.targets}


def gate_piece_of(key: str, gate_ids: Iterable[str]) -> str | None:
    """The schedule ``gate_id`` an ``intrinsic_budget`` key belongs to: the longest gate id that prefixes the key at a ``.``
    or ``/`` separator (gate ids such as ``zz[k]/loop1`` contain a slash themselves)."""
    best: str | None = None
    for gid in gate_ids:
        if key == gid or (key.startswith(gid) and key[len(gid)] in "./"):
            if best is None or len(gid) > len(best):
                best = gid
    return best


def intrinsic_by_kind(diagnostics: Diagnostics, kinds: Mapping[str, str]) -> dict[str, float]:
    """One run's closed-form scales summed per kind: each gate's scale entries and its Raman, leakage and Rayleigh-dephasing
    probabilities. Entries cover the gate's whole schedule entry, crosstalk on ions outside the benchmarked set included, so
    the sum bounds a larger error than :meth:`StepChannel.infidelity_on`."""
    out: dict[str, float] = {}
    for key, val in diagnostics.intrinsic_budget.items():
        if key == "total":
            continue
        # keys such as "ms[2].residual_displacement", "gpi2[0].ion0.P_raman", "zz[2]/loop1.residual_displacement"
        piece = gate_piece_of(key, kinds)
        if piece is None:
            continue
        rest = key[len(piece) :]
        scale = "/" not in rest and rest.count(".") == 1
        scattering = rest.endswith((".P_raman", ".P_leak", ".rayleigh_dephasing"))
        if not (scale or scattering):
            continue
        kind = kinds[piece]
        out[kind] = out.get(kind, 0.0) + float(val)
    return out


def reduced_choi(choi: np.ndarray, n_factors: int, keep: Sequence[int]) -> np.ndarray:
    """The trace-1 Choi matrix of the channel reduced to the qubit factors ``keep`` (positions in the step's ion order):
    E_keep(rho) = Tr_rest E(rho (x) |0><0|)."""
    d_all = 2**n_factors
    if choi.shape != (d_all * d_all, d_all * d_all):
        raise ValueError("Choi matrix and factor count disagree")
    keep = [int(k) for k in keep]
    rest = [k for k in range(n_factors) if k not in keep]
    d_k = 2 ** len(keep)
    out = np.zeros((d_k * d_k, d_k * d_k), dtype=complex)
    del rest
    for i in range(d_k):
        for j in range(d_k):
            psi_i = _product_ket(i, len(keep), keep, n_factors)
            psi_j = _product_ket(j, len(keep), keep, n_factors)
            rho_in = np.outer(psi_i, psi_j.conj())
            rho_out = apply_choi(choi, rho_in)
            red = _partial_trace(rho_out, n_factors, keep)
            out[i * d_k : (i + 1) * d_k, j * d_k : (j + 1) * d_k] += red
    # block (i, j) of a Choi matrix is E(|i><j|)/d in noise/summary.py's convention
    return np.asarray(out / d_k)


def _product_ket(index: int, n_keep: int, keep: Sequence[int], n_factors: int) -> np.ndarray:
    """The basis ket with ``index``'s bits on the kept factors (the first the most significant) and |0> on the others."""
    bits = [(index >> (n_keep - 1 - k)) & 1 for k in range(n_keep)]
    full = np.zeros([2] * n_factors, dtype=complex)
    idx: list[int] = []
    kb = 0
    for f in range(n_factors):
        if f in keep:
            idx.append(bits[kb])
            kb += 1
        else:
            idx.append(0)
    full[tuple(idx)] = 1.0
    return full.reshape(-1)


def _partial_trace(rho: np.ndarray, n_factors: int, keep: Sequence[int]) -> np.ndarray:
    r = rho.reshape([2] * (2 * n_factors))
    keep_list = list(keep)
    rest = [k for k in range(n_factors) if k not in keep_list]
    for k in sorted(rest, reverse=True):
        r = np.trace(r, axis1=k, axis2=k + r.ndim // 2)
    d_k = 2 ** len(keep_list)
    return np.asarray(r.reshape(d_k, d_k))


@dataclass(frozen=True)
class StepChannel:
    """One gate step's channel with its ideal unitary on the step's ions."""

    gate_id: str
    ions: tuple[int, ...]
    summary: ChannelSummary
    ideal: np.ndarray
    """The ideal on ``ions`` (first ion the first factor): the ``GateTarget`` unitaries, identity on the neighbours."""
    engine_runs: int
    local_dimension: int

    def infidelity_on(self, qubits: Sequence[int]) -> float:
        """Average gate infidelity of the channel reduced to ``qubits`` (the step's other ions prepared in |0> and traced
        out) against the ideal's factor on them; for a gate on a neighbour of ``qubits`` that factor is the identity, so
        this is the crosstalk error per neighbour pulse."""
        keep = [k for k, q in enumerate(self.ions) if q in set(int(x) for x in qubits)]
        if not keep:
            return 0.0
        n = len(self.ions)
        if len(keep) == n:
            choi = self.summary.choi
            ideal = self.ideal
        else:
            choi = reduced_choi(self.summary.choi, n, keep)
            ideal = reduced_ideal(self.ideal, n, keep)
        eps = entanglement_infidelity(choi, choi_from_unitary(ideal))
        return float(average_gate_infidelity(max(eps, 0.0), 2 ** len(keep)))


def reduced_ideal(u: np.ndarray, n_factors: int, keep: Sequence[int]) -> np.ndarray:
    """The factor U_keep of the ideal ``u`` = U_keep (x) U_rest on the qubit factors ``keep``, up to a global phase: the
    leading singular vector of the rank-one reshaping; raises ``ValueError`` when ``u`` does not factor into unitaries."""
    keep_list = [int(k) for k in keep]
    rest = [k for k in range(n_factors) if k not in keep_list]
    if not rest:
        return np.asarray(u, dtype=complex)
    d_k, d_r = 2 ** len(keep_list), 2 ** len(rest)
    arr = np.asarray(u, dtype=complex).reshape([2] * (2 * n_factors))
    order = keep_list + [k + n_factors for k in keep_list] + rest + [k + n_factors for k in rest]
    r = arr.transpose(order).reshape(d_k * d_k, d_r * d_r)
    left, sv, _vh = np.linalg.svd(r)
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
class GateChannel:
    """The GATE_LOCAL channels of one native gate kind played as a one-gate circuit, one step per scheduler piece (the zz
    wrapper on an MS device has several)."""

    kind: str
    steps: tuple[StepChannel, ...]
    level: str
    notes: tuple[str, ...] = ()

    def infidelity_on(self, qubits: Sequence[int]) -> float:
        """First-order composition over the pieces of the reduced average infidelities."""
        return float(sum(s.infidelity_on(qubits) for s in self.steps))

    @property
    def average_gate_infidelity(self) -> float:
        return float(sum(s.summary.average_gate_infidelity for s in self.steps))

    @property
    def depolarizing_rate(self) -> float:
        return float(sum(s.summary.depolarizing_rate for s in self.steps))


_CHANNEL_CACHE: dict[tuple[object, ...], GateChannel] = {}


def clear_budget_cache() -> None:
    """Empty the process-wide :func:`gate_channel` cache (keyed by ``Machine.hash()`` and kind); call it after changing
    physics the machine hash does not see."""
    _CHANNEL_CACHE.clear()


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
    """The channel of one native gate kind by GATE_LOCAL tomography on ``machine``, cached per ``Machine.hash()`` and kind."""
    from dataclasses import replace

    from qutip_trap.run.levels import FidelityLevel

    m = machine
    local = replace(m, level=FidelityLevel.GATE_LOCAL)
    device = local.device
    key: tuple[object, ...] = (local.hash(), kind)
    if key in _CHANNEL_CACHE:
        return _CHANNEL_CACHE[key]
    n = device.crystal.n_ions
    res = local.run(one_gate_circuit(kind, n), 1)
    rec = last_record(res)
    gl = res.diagnostics.gate_local
    assert gl is not None
    walk = {s.gate_id: s for s in gate_steps(rec.schedule) if s.kind == "gate"}
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
    ch = GateChannel(kind=kind, steps=tuple(steps), level="GATE_LOCAL", notes=tuple(gl.notes))
    _CHANNEL_CACHE[key] = ch
    return ch


@dataclass(frozen=True)
class BenchmarkBudget:
    """The simulator's own error budget beside a benchmark's measured number."""

    unit: str
    """What counts and scales are per: ``clifford``, ``computational gate`` (Knill RB) or ``circuit``."""
    qubits: tuple[int, ...]
    counts: dict[str, float]
    """Average number of native gate pieces of each kind per unit."""
    intrinsic: dict[str, float]
    """Closed-form scales per kind per unit, and ``total``."""
    spam: dict[str, tuple[float, float]]
    """Per benchmarked qubit ``q{i}`` -> (eps_B, eps_D) and ``q{i}.state_preparation`` -> (eps_prep, 0)."""
    channels: dict[str, GateChannel] = field(default_factory=dict)
    channel_infidelity: dict[str, float] = field(default_factory=dict)
    """Per kind, the average infidelity of the channel reduced to the benchmarked qubits."""
    predicted: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def channel_error_per_unit(self) -> float | None:
        """sum_kind counts x reduced average infidelity (first-order composition), None without channels."""
        if not self.channel_infidelity:
            return None
        return float(sum(self.counts.get(k, 0.0) * v for k, v in self.channel_infidelity.items()))

    @property
    def readout_error_mean(self) -> dict[int, float]:
        """Per benchmarked qubit, (eps_B + eps_D)/2."""
        out: dict[int, float] = {}
        for q in self.qubits:
            eb, ed = self.spam.get(f"q{q}", (0.0, 0.0))
            out[int(q)] = 0.5 * (float(eb) + float(ed))
        return out

    @property
    def preparation_error(self) -> dict[int, float]:
        return {int(q): float(self.spam.get(f"q{q}.state_preparation", (0.0, 0.0))[0]) for q in self.qubits}


def gather_counts_and_intrinsic(
    results: Sequence[Result], units: Sequence[int]
) -> tuple[dict[str, float], dict[str, float]]:
    """Per kind, the average piece count and intrinsic scale per unit over runs (``units[k]`` units in run ``k``)."""
    counts: dict[str, float] = {}
    intrinsic: dict[str, float] = {}
    total_units = float(sum(units))
    if total_units <= 0.0:
        raise ValueError("at least one benchmark unit")
    intrinsic_total = 0.0
    for res in results:
        rec = last_record(res)
        kinds = kinds_of_schedule(rec.schedule)
        for kind in kinds.values():
            counts[kind] = counts.get(kind, 0.0) + 1.0
        for kind, val in intrinsic_by_kind(res.diagnostics, kinds).items():
            intrinsic[kind] = intrinsic.get(kind, 0.0) + val
        intrinsic_total += float(res.diagnostics.intrinsic_budget.get("total", 0.0))
    counts = {k: v / total_units for k, v in counts.items()}
    intrinsic = {k: v / total_units for k, v in intrinsic.items()}
    intrinsic["total"] = intrinsic_total / total_units
    return counts, intrinsic


def spam_of(result: Result, qubits: Sequence[int]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for q in qubits:
        for key in (f"q{q}", f"q{q}.state_preparation"):
            if key in result.spam:
                out[key] = (float(result.spam[key][0]), float(result.spam[key][1]))
    return out


def channels_for(
    machine: Machine, kinds: Sequence[str], qubits: Sequence[int]
) -> tuple[dict[str, GateChannel], dict[str, float]]:
    """The :func:`gate_channel` of every kind and its average infidelity reduced to ``qubits``."""
    m = machine
    channels: dict[str, GateChannel] = {}
    infidelity: dict[str, float] = {}
    for kind in sorted(set(kinds)):
        ch = gate_channel(m, kind)
        channels[kind] = ch
        infidelity[kind] = ch.infidelity_on(qubits)
    return channels, infidelity


__all__ = [
    "FULL_MS_RAD",
    "BenchmarkBudget",
    "GateChannel",
    "StepChannel",
    "channels_for",
    "clear_budget_cache",
    "gate_channel",
    "gate_piece_of",
    "gather_counts_and_intrinsic",
    "intrinsic_by_kind",
    "kind_of",
    "kinds_of_schedule",
    "one_gate_circuit",
    "reduced_choi",
    "reduced_ideal",
    "spam_of",
]
