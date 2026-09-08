"""The simulator's own error budget, reported beside every benchmark (PLAN.md Section 10 M10 "with the simulator's own error
budget reported alongside"; Sections 6.8, 8.4, 9.6; Section 13 rows "RB error rate", "Depolarizing normalization").

Three layers of the simulator's own accounting are composed into a prediction of what the benchmark should return, so that
the benchmark measures the machine and the budget says what the machine's known physics accounts for:

1. the closed-form scales of Section 9.6 that every run reports (``Diagnostics.intrinsic_budget``: residual displacement,
   Debye-Waller loss, the off-resonant carrier scale, the frozen spectators' chi, the single-qubit pulses' sideband scale and
   addressing crosstalk, the scattering probabilities), summed per native gate kind and per benchmark unit (a Clifford, a
   circuit); these are SCALES that bound the coherent errors, not their values;
2. the Section 6.8 channel summaries of the native gate set by state-based process tomography (Section 5.4): every native gate
   kind the benchmark compiled to (``gpi2`` and ``gpi`` per ion, ``ms`` and ``zz`` per pair) is played once as a one-gate
   circuit through ``run(level="GATE_LOCAL")`` from the device's prepared motional state, and the step's Choi matrix, reduced
   to the benchmarked qubits with the crosstalk neighbours traced out in |0>, gives the gate's average infidelity against its
   ideal unitary (the native gate at its frame-applied phase followed by the Stark frame, ``GateTarget``); the prediction
   composes them to first order (average infidelities add for small independent errors, Magesan et al. 2011);
3. the readout and preparation errors of Section 8.4 (``Result.spam``), which set the SPAM offsets of an RB curve and the
   readout loss of a population or parity.

Every prediction states its composition rule in ``predicted`` and ``notes``; none of it is fed back into the simulation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
from qutip_trap.run.job import last_record, run
from qutip_trap.run.results import Diagnostics, Result

if TYPE_CHECKING:
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.device.model import Device

FULL_MS_RAD = math.pi / 2.0
"""The fully entangling native angle (Section 7.1): MS theta = pi/2, ZZ theta = pi/2."""


def kind_of(name: str, ions: Sequence[int]) -> str:
    """The native gate kind key: ``gpi2[0]``, ``gpi[1]``, ``ms[0,1]``, ``zz[0,1]``."""
    return f"{name}[{','.join(str(int(q)) for q in ions)}]"


def kinds_of_schedule(schedule: Schedule) -> dict[str, str]:
    """gate piece id -> kind key over the schedule's ``targets`` (the native pieces the scheduler played)."""
    return {t.gate_id: kind_of(t.native[0], t.ions) for t in schedule.targets}


def intrinsic_by_kind(diagnostics: Diagnostics, kinds: Mapping[str, str]) -> dict[str, float]:
    """The Section 9.6 closed-form scales of one run summed per kind (the accounting of ``run.job.intrinsic_budget``: every
    non-scattering entry, and of the scattering estimates the Raman, leakage and Rayleigh-dephasing probabilities)."""
    out: dict[str, float] = {}
    for key, val in diagnostics.intrinsic_budget.items():
        if key == "total":
            continue
        # "ms[2].residual_displacement", "gpi2[0].crosstalk", "gpi2[0].ion0.P_raman", "ms[2]/seg0/ion0.ion0.P_raman"
        piece = key.split("/")[0].split(".")[0]
        if piece not in kinds:
            continue
        scale = "/" not in key and key.count(".") == 1
        scattering = key.endswith((".P_raman", ".P_leak", ".rayleigh_dephasing"))
        if not (scale or scattering):
            continue
        kind = kinds[piece]
        out[kind] = out.get(kind, 0.0) + float(val)
    return out


def reduced_choi(choi: np.ndarray, n_factors: int, keep: Sequence[int]) -> np.ndarray:
    """The trace-1 Choi matrix of the channel reduced to the qubit factors ``keep`` (positions in the step's ion order) with the
    other factors prepared in |0> and traced out after the map: E_keep(rho) = Tr_rest E(rho (x) |0><0|)."""
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
            # the input |i><j| on the kept factors (x) |0><0| on the rest, assembled in the step's factor order
            psi_i = _product_ket(i, len(keep), keep, n_factors)
            psi_j = _product_ket(j, len(keep), keep, n_factors)
            rho_in = np.outer(psi_i, psi_j.conj())
            rho_out = apply_choi(choi, rho_in)
            red = _partial_trace(rho_out, n_factors, keep)
            out[i * d_k : (i + 1) * d_k, j * d_k : (j + 1) * d_k] += red
    # block (i, j) of the Choi matrix is E(|i><j|)/d in the convention of noise/summary.py (first factor the input copy)
    return np.asarray(out / d_k)


def _product_ket(index: int, n_keep: int, keep: Sequence[int], n_factors: int) -> np.ndarray:
    """The basis ket with the bits of ``index`` on the kept factors (first kept factor the most significant bit) and |0> on
    every other factor, in the step's factor order."""
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
    """One gate step's channel (Section 6.8) with its ideal unitary on the step's ions."""

    gate_id: str
    ions: tuple[int, ...]
    summary: ChannelSummary
    ideal: np.ndarray
    """The ideal unitary on ``ions`` (first ion the first factor): the native gate at its frame-applied phase followed by the
    Stark frame (``GateTarget.unitary``), identity on the crosstalk neighbours."""
    engine_runs: int
    local_dimension: int

    def infidelity_on(self, qubits: Sequence[int]) -> float:
        """The average gate infidelity of the channel reduced to the benchmarked ``qubits`` among the step's ions (the other
        ions traced out in |0>), against the ideal reduced the same way."""
        keep = [k for k, q in enumerate(self.ions) if q in set(int(x) for x in qubits)]
        if not keep:
            return 0.0
        n = len(self.ions)
        if len(keep) == n:
            choi = self.summary.choi
            ideal = self.ideal
        else:
            choi = reduced_choi(self.summary.choi, n, keep)
            ideal = _reduced_ideal(self.ideal, n, keep)
        eps = entanglement_infidelity(choi, choi_from_unitary(ideal))
        return float(average_gate_infidelity(max(eps, 0.0), 2 ** len(keep)))


def _reduced_ideal(u: np.ndarray, n_factors: int, keep: Sequence[int]) -> np.ndarray:
    """The block <0_rest| U |0_rest> of an ideal that is a product of a unitary on ``keep`` and the identity on the rest."""
    rest = [k for k in range(n_factors) if k not in keep]
    arr = u.reshape([2] * (2 * n_factors))
    for k in sorted(rest, reverse=True):
        arr = np.take(np.take(arr, 0, axis=k + arr.ndim // 2), 0, axis=k)
    d_k = 2 ** len(keep)
    out = np.asarray(arr.reshape(d_k, d_k))
    if np.max(np.abs(out.conj().T @ out - np.eye(d_k))) > 1e-8:
        raise ValueError("the step's ideal unitary is not the identity on the traced-out ions")
    return out


@dataclass(frozen=True)
class GateChannel:
    """The Section 6.8 channels of one native gate kind, played once as a one-gate circuit at GATE_LOCAL (a kind whose
    scheduler expansion has several pieces, the zz wrapper on an MS device, carries one step per piece)."""

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


def gate_channel(device: Device, kind: str, **run_kwargs: Any) -> GateChannel:
    """The channel of one native gate kind by GATE_LOCAL tomography (module docstring item 2), cached per device, kind, table
    and solver options within the process."""
    table = run_kwargs.get("table")
    opts = run_kwargs.get("options")
    key: tuple[object, ...] = (
        device.hash(),
        kind,
        None if table is None else (table.device_hash, table.seed, table.surrogate),
        None if opts is None else tuple(sorted((k, repr(v)) for k, v in vars(opts).items())),
        bool(run_kwargs.get("noise", True)),
        str(run_kwargs.get("crosstalk_suppression", "none")),
    )
    if key in _CHANNEL_CACHE:
        return _CHANNEL_CACHE[key]
    kw = {k: v for k, v in run_kwargs.items() if k not in ("level", "keep_final_state", "readout")}
    n = device.crystal.n_ions
    res = run(one_gate_circuit(kind, n), device, 1, level="GATE_LOCAL", **kw)
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
    """What the simulator's own physics accounts for, beside a benchmark's measured number (module docstring)."""

    unit: str
    """What the counts and scales are per: ``clifford`` or ``circuit``."""
    qubits: tuple[int, ...]
    counts: dict[str, float]
    """Average number of native gate pieces of each kind per unit."""
    intrinsic: dict[str, float]
    """Section 9.6 closed-form scales per kind per unit, and ``total``."""
    spam: dict[str, tuple[float, float]]
    """Per benchmarked qubit ``q{i}`` -> (eps_B, eps_D) and ``q{i}.state_preparation`` -> (eps_prep, 0)."""
    channels: dict[str, GateChannel] = field(default_factory=dict)
    """Section 6.8 channels per kind (empty when tomography was not requested)."""
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
        """Per benchmarked qubit, (eps_B + eps_D)/2 (Section 13 row 'Readout figure of merit')."""
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
    """Per kind, the average piece count and the average intrinsic scale per unit over several runs (``units[k]`` benchmark
    units in run ``k``)."""
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
    device: Device, kinds: Sequence[str], qubits: Sequence[int], **run_kwargs: Any
) -> tuple[dict[str, GateChannel], dict[str, float]]:
    """The Section 6.8 channels of every kind and their average infidelity reduced to ``qubits``."""
    channels: dict[str, GateChannel] = {}
    infidelity: dict[str, float] = {}
    for kind in sorted(set(kinds)):
        ch = gate_channel(device, kind, **run_kwargs)
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
    "gather_counts_and_intrinsic",
    "intrinsic_by_kind",
    "kind_of",
    "kinds_of_schedule",
    "one_gate_circuit",
    "reduced_choi",
    "spam_of",
]
