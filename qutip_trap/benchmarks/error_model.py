"""The inverse direction: the phenomenological error model of a simulated device (docs/api_implementation_plan.md 2.6;
docs/api_proposal.md Sections 1 item 8 and 4.8). Every vendor emulator surveyed takes numbers like IonQ's ``r_1q``/``r_2q``,
Quantinuum's ``p1``/``p2``/``p_meas`` or the QDK estimator's gate times and error rates, and none derives them from
physics; ``error_model(machine)`` emits them from the machine: per native gate kind the average gate infidelity of its
GATE_LOCAL channel (``gate_channel``, Section 6.8) reduced to the gate's own qubits, the SPAM errors from the detection
calibration and the preparation recipe, the durations from the schedule, the dephasing and heating rates from the noise
model, each in the naming discipline that survived three vendor API generations: ``p_*`` a probability, ``*_rate`` per
second, ``*_ratio`` a fraction of another probability, ``*_scale`` a multiplier.

Conversions (stated once, here, and repeated on each exporter):

- the channel's ``infidelity_on`` is the AVERAGE gate infidelity r = 1 - F_avg; the ledger's ``conv.depolarizing_normalization``
  writes the depolarizing channel with the entanglement infidelity eps as its rate, r = d eps/(d + 1);
- IonQ's noise model is the maximally-mixed weight lambda: Lambda(rho) = (1 - lambda) rho + lambda 1/d, F_avg = 1 - lambda (1 - 1/d),
  so ``r_1q = 2 r`` (F_avg = 1 - r_1q/2) and ``r_2q = 4 r/3`` (F_avg = 1 - 3 r_2q/4), i.e. lambda = eps 4^n/(4^n - 1);
- Quantinuum's ``p1``/``p2`` are taken as the same depolarizing weights, ``p_meas`` as (P(read 1 | prepared 0), P(read 0 |
  prepared 1)) = (eps_D, eps_B) with |0> the dark state, ``p_init`` the preparation error, the dephasing rate per second;
- the QDK resource estimator takes gate times as unit-suffixed strings (``"5 µs"``) and error rates as probabilities per
  gate; the idle error rate is the white dephasing probability over one single-qubit gate time, 1 - exp(-gamma t).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qutip_trap.machine import Machine

QDK_TIME_PATTERN = re.compile(r"^\d+(\.\d+)?(e[+-]?\d+)? (ns|µs|ms|s)$")
"""The form of a time the QDK resource estimator's ``qubitParams`` accept: a number, one space, a unit."""

SINGLE_QUBIT_KINDS: tuple[str, ...] = ("gpi", "gpi2")
"""The single-qubit native gate kinds the model characterises, per qubit."""


def qdk_time(seconds: float) -> str:
    """``seconds`` as the estimator's unit-suffixed string, in the largest unit that keeps the number at or above one."""
    if seconds < 0.0 or not math.isfinite(seconds):
        raise ValueError("a gate time is a finite, non-negative number of seconds")
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns")):
        if seconds >= scale:
            return f"{seconds / scale:.6g} {unit}"
    return f"{seconds / 1e-9:.6g} ns"


@dataclass(frozen=True)
class ErrorModel:
    """The phenomenological summary of one machine (``error_model``): per native gate kind the average gate infidelity
    ``infidelity[kind]`` (r = 1 - F_avg, the channel reduced to the gate's qubits, the crosstalk neighbours traced out) and
    its duration ``durations_s[kind]``; the depolarizing weights ``p_1q`` (mean over the single-qubit kinds of 2 r) and
    ``p_2q`` (mean over the entangling kinds of 4 r/3; None without a pair); per qubit ``p_meas`` = (eps_D, eps_B), the
    probabilities of reading 1 when |0> (dark) was prepared and 0 when |1> (bright) was, and ``p_init`` the preparation
    error; the white qubit dephasing rate and the heating rate per mode from the noise model (1/s and quanta/s; empty when
    the model has none); the machine hash the numbers belong to. The exporters state their conversions."""

    infidelity: dict[str, float]
    durations_s: dict[str, float]
    p_1q: float
    p_2q: float | None
    p_meas: dict[int, tuple[float, float]]
    p_init: dict[int, float]
    dephasing_rate_per_s: dict[int, float]
    heating_rate_per_s: dict[int, float]
    machine_hash: str
    qubits: tuple[int, ...]
    entangler: str
    provenance: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def single_qubit_kinds(self) -> tuple[str, ...]:
        return tuple(k for k in self.infidelity if k.split("[", 1)[0] in SINGLE_QUBIT_KINDS)

    @property
    def two_qubit_kinds(self) -> tuple[str, ...]:
        return tuple(k for k in self.infidelity if k.split("[", 1)[0] not in SINGLE_QUBIT_KINDS)

    @property
    def t_1q_s(self) -> float:
        """The mean single-qubit gate duration (s)."""
        return float(
            sum(self.durations_s[k] for k in self.single_qubit_kinds) / max(1, len(self.single_qubit_kinds))
        )

    @property
    def t_2q_s(self) -> float | None:
        kinds = self.two_qubit_kinds
        return None if not kinds else float(sum(self.durations_s[k] for k in kinds) / len(kinds))

    @property
    def p_meas_mean(self) -> tuple[float, float]:
        if not self.p_meas:
            return (0.0, 0.0)
        n = len(self.p_meas)
        return (sum(v[0] for v in self.p_meas.values()) / n, sum(v[1] for v in self.p_meas.values()) / n)

    @property
    def dephasing_rate_mean_per_s(self) -> float:
        return (
            float(sum(self.dephasing_rate_per_s.values()) / len(self.dephasing_rate_per_s))
            if self.dephasing_rate_per_s
            else 0.0
        )

    def to_ionq_noise(self) -> dict[str, float]:
        """IonQ's noise model parameters ``{"r_1q", "r_2q"}``: the maximally-mixed weight of a depolarizing channel after
        each ideal gate, with F_avg = 1 - r_1q/2 on one qubit and 1 - 3 r_2q/4 on two, so r_1q = 2 r and r_2q = 4 r/3 for the
        average gate infidelity r (equivalently lambda = eps 4^n/(4^n - 1) for the entanglement infidelity eps of
        ``conv.depolarizing_normalization``). ``r_2q`` is left out when the machine has no entangling pair."""
        out = {"r_1q": float(self.p_1q)}
        if self.p_2q is not None:
            out["r_2q"] = float(self.p_2q)
        return out

    def to_quantinuum_error_params(self) -> dict[str, Any]:
        """Quantinuum's ``UserErrorParams`` vocabulary: ``p1`` and ``p2`` the depolarizing weights of ``to_ionq_noise`` (the
        same convention, F_avg = 1 - p (1 - 1/d)), ``p_meas`` = (P(1 | 0 prepared), P(0 | 1 prepared)) averaged over the
        qubits, ``p_init`` the mean preparation error, ``linear_dephasing_rate`` the mean white qubit dephasing rate in 1/s
        and ``quadratic_dephasing_rate`` 0.0 (the simulator's white dephasing is a linear rate; a quadratic term would be a
        drift, which the noise model samples rather than summarises). The field names follow the vendor survey of
        docs/api_proposal.md; the units are the ones stated here."""
        return {
            "p1": float(self.p_1q),
            "p2": None if self.p_2q is None else float(self.p_2q),
            "p_meas": self.p_meas_mean,
            "p_init": float(sum(self.p_init.values()) / len(self.p_init)) if self.p_init else 0.0,
            "linear_dephasing_rate": self.dephasing_rate_mean_per_s,
            "quadratic_dephasing_rate": 0.0,
        }

    def to_qdk_qubit_params(self) -> dict[str, Any]:
        """The QDK resource estimator's gate-based ``qubitParams``: the gate and measurement times as the estimator's
        unit-suffixed strings (``qdk_time``: ``"5 µs"``, matching ``QDK_TIME_PATTERN``), the one- and two-qubit and T-gate
        error rates as the depolarizing weights of ``to_ionq_noise`` (the T gate is a single-qubit gate here), the
        measurement error rate as the mean of the two readout errors, and ``idleErrorRate`` the white dephasing
        probability over one single-qubit gate time, 1 - exp(-gamma t_1q). The measurement time is the detection window."""
        p01, p10 = self.p_meas_mean
        t_2q = self.t_2q_s
        out: dict[str, Any] = {
            "name": "qutip-trap",
            "instructionSet": "GateBased",
            "oneQubitMeasurementTime": qdk_time(self.measurement_time_s),
            "oneQubitGateTime": qdk_time(self.t_1q_s),
            "tGateTime": qdk_time(self.t_1q_s),
            "oneQubitMeasurementErrorRate": 0.5 * (p01 + p10),
            "oneQubitGateErrorRate": float(self.p_1q),
            "tGateErrorRate": float(self.p_1q),
            "idleErrorRate": float(1.0 - math.exp(-self.dephasing_rate_mean_per_s * self.t_1q_s)),
        }
        if t_2q is not None and self.p_2q is not None:
            out["twoQubitGateTime"] = qdk_time(t_2q)
            out["twoQubitGateErrorRate"] = float(self.p_2q)
        return out

    @property
    def measurement_time_s(self) -> float:
        return float(self.durations_s.get("measure", 0.0))


def error_model(machine: Machine | Any, *, qubits: Sequence[int] | None = None) -> ErrorModel:
    """The ``ErrorModel`` of ``machine`` (a ``Device`` is wrapped in a default machine): one GATE_LOCAL tomography per
    single-qubit kind and qubit (``gpi[i]``, ``gpi2[i]``) and per adjacent pair of the machine's entangler (``ms[i,j]`` or
    ``zz[i,j]``), through ``gate_channel`` (cached per machine hash and kind); the SPAM from the machine's table (the cached
    closed-form surrogate when none is pinned) and the preparation recipe; the durations from ``Machine.schedule`` of a
    one-gate circuit per kind, the detection window as the measurement time. ``qubits`` restricts the characterised ions
    (default: all)."""
    from dataclasses import replace

    from qutip_trap.benchmarks.budget import gate_channel, kind_of, one_gate_circuit
    from qutip_trap.calibration import calibrate
    from qutip_trap.machine import as_machine
    from qutip_trap.prep.recipe import recipe_of, run_preparation

    m = as_machine(machine)
    device = m.device
    n = device.crystal.n_ions
    qs = tuple(range(n)) if qubits is None else tuple(int(q) for q in qubits)
    if any(q < 0 or q >= n for q in qs) or len(set(qs)) != len(qs):
        raise ValueError("qubits are distinct ions of the device")
    entangler = m.physics.entangler
    kinds: list[tuple[str, tuple[int, ...]]] = [
        (kind_of(name, (q,)), (q,)) for q in qs for name in SINGLE_QUBIT_KINDS
    ]
    pairs = [(a, b) for a, b in zip(qs, qs[1:])]
    kinds.extend((kind_of(entangler, pair), pair) for pair in pairs)
    table = m.table if m.table is not None else calibrate(m, pairs=pairs).table
    pinned = replace(m, table=table)
    infidelity: dict[str, float] = {}
    durations: dict[str, float] = {}
    notes: list[str] = []
    for kind, ions in kinds:
        channel = gate_channel(pinned, kind)
        infidelity[kind] = channel.infidelity_on(ions)
        sched = pinned.schedule(one_gate_circuit(kind, n))
        starts = [p.t_start_s for p in sched.pulses]
        ends = [p.t_end_s for p in sched.pulses]
        durations[kind] = float(max(ends) - min(starts)) if sched.pulses else 0.0
        notes.extend(x for x in channel.notes if x not in notes)
    durations["measure"] = float(device.detector.window_s)
    single = [2.0 * infidelity[k] for k, ions in kinds if len(ions) == 1]
    double = [4.0 * infidelity[k] / 3.0 for k, ions in kinds if len(ions) == 2]
    detection = table.detection
    eps_b = float(detection["eps_B"].value) if "eps_B" in detection else 0.0
    eps_d = float(detection["eps_D"].value) if "eps_D" in detection else 0.0
    if "eps_B" not in detection or "eps_D" not in detection:
        notes.append("the table carries no detection calibration: p_meas is reported as zero")
    prep = run_preparation(device, recipe_of(device))
    provenance = {
        "infidelity": "conv.depolarizing_normalization",
        "p_1q": "conv.depolarizing_normalization",
        "p_2q": "conv.depolarizing_normalization",
        "p_meas": "conv.readout_figure_of_merit",
        "p_init": "conv.readout_figure_of_merit",
    }
    return ErrorModel(
        infidelity=infidelity,
        durations_s=durations,
        p_1q=float(sum(single) / len(single)) if single else 0.0,
        p_2q=float(sum(double) / len(double)) if double else None,
        p_meas={q: (eps_d, eps_b) for q in qs},
        p_init={q: float(prep.preparation_error(q)) for q in qs},
        dephasing_rate_per_s={
            i: float(g) for i, g in device.noise.qubit_white_dephasing_per_s(device).items() if i in qs
        },
        heating_rate_per_s={
            mode: float(r) for mode, r in device.noise.heating_rates_quanta_per_s(device).items()
        },
        machine_hash=pinned.hash(),
        qubits=qs,
        entangler=entangler,
        provenance=provenance,
        notes=tuple(notes),
    )
