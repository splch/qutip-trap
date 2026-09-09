"""Request semantics of PLAN.md Section 14.4 at Levels 1 and 2 (milestone M11.4; Section 9.11 row "Request semantics").

Section 14.4: "A change at a shallower level is a request rather than an edit: asking for XX(0.3) at Level 1 invokes the pulse
solver; setting a detuning by hand at Level 2 is applied as written, and the gate at Level 1 then displays its actual unitary,
not the requested one. Requests the device cannot satisfy are shown with the reason, for example loop closure impossible
within the power limit or a requested angle outside the calibrated range."

Both requests make a NEW job from the record's job (nothing on the record is edited):

- :func:`request_angle` replaces the native MS gate's angle in the compiled circuit. The scheduler then plays the pair's
  calibrated waveform rescaled by sqrt(|chi|/|chi_cal|) (the s-squared law of Section 4.4.7 (7), which is the pulse
  solver's own amplitude law), the job runs at the full engine, and the finished pulse's process matrix is what Level 1
  shows beside the requested unitary. The request is refused, with the reason, when the angle lies outside the native
  gate's range or when the rescaled amplitude exceeds the calibrated carrier Rabi frequency of the entangling drive (the
  power limit).
- :func:`request_detuning` writes a beat-note offset into the job's ``waveform_overrides``; ``record.calibrate_for`` shifts
  the pair's waveform (blue legs up, red legs down) and the scheduler plays it as written, amplitude unchanged. The loops
  then fail to close by what the physics says, and the played gate's process matrix differs from the requested unitary.

:func:`fitted_ms_angle` reads the entangling angle back from a process matrix: the MS angle whose ideal channel has the
largest entanglement fidelity with the measured one, so "the simulated unitary is XX(0.3)" is a number, not a feeling.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from qutip_trap_app import core
from qutip_trap_app.record import (
    CircuitRecord,
    JobSpec,
    OpRecord,
    ProcessMatrixRecord,
    Record,
    TargetRecord,
    WaveformRecord,
)

MAX_ANGLE_RAD = math.pi / 2.0
"""|chi| of XX(chi) the native MS gate can express: MS(phi0, phi1, theta) with theta = 2|chi| in [0, pi] (Section 7.6)."""

MAX_DETUNING_FRACTION = 0.5
"""A hand-set beat-note offset is accepted up to this fraction of the beat note itself: beyond it a tone crosses the carrier
and the pulse is no longer the bichromatic drive the waveform was solved as (Section 7.5's scans stay well inside)."""

RequestKind = Literal["angle", "detuning"]


class RequestError(ValueError):
    """A request that cannot even be formed (an unknown gate, a gate that is not an entangling gate)."""


@dataclass(frozen=True)
class GateRequest:
    """One request at Level 1 or 2 and what became of it: the job to run, or the refusal with its reason."""

    kind: RequestKind
    gate_id: str
    step_index: int
    pair: tuple[int, int]
    requested: float
    """chi in rad (``angle``) or the beat-note offset in Hz (``detuning``)."""
    calibrated_chi_rad: float | None
    scale: float | None
    """The amplitude factor the scheduler applies (``angle``): sqrt(|chi|/|chi_cal|)."""
    peak_rabi_hz: float | None
    """The rescaled waveform's peak Rabi frequency (``angle``)."""
    carrier_rabi_hz: float | None
    """The calibrated carrier Rabi frequency of the entangling drive, the power limit."""
    job: JobSpec | None
    refusal: str | None
    note: str

    @property
    def accepted(self) -> bool:
        return self.job is not None


@dataclass(frozen=True)
class RequestOutcome:
    """Requested against actual for one entangling gate of a record that carries the gate's process matrix (Level 1)."""

    gate_id: str
    requested_theta_rad: float
    """The MS angle theta the target unitary encodes (2|chi|)."""
    fitted_theta_rad: float
    """The MS angle whose ideal channel is closest to the measured one."""
    fitted_fidelity: float
    """Entanglement fidelity of the measured channel with MS(phi0, phi1, fitted theta)."""
    infidelity_to_requested: float
    """Average gate infidelity of the measured channel against the requested unitary (the tomography's own figure)."""
    requests: tuple[str, ...]
    hand_set_detuning_hz: float | None

    @property
    def requested_chi_rad(self) -> float:
        return 0.5 * self.requested_theta_rad

    @property
    def fitted_chi_rad(self) -> float:
        return 0.5 * self.fitted_theta_rad


# ---- helpers ----------------------------------------------------------------------------------------------------------------------


def gate_index(gate_id: str) -> int:
    """The native-circuit operation index a gate id names (``"ms[2]"`` -> 2, Section 7.3's ids)."""
    if "[" not in gate_id or not gate_id.endswith("]"):
        raise RequestError(f"{gate_id!r} is not a native gate id of the form name[k]")
    return int(gate_id[gate_id.index("[") + 1 : -1])


def target_of(record: Record, gate_id: str) -> TargetRecord:
    for tg in record.schedule.targets:
        if tg.gate_id == gate_id:
            return tg
    raise RequestError(f"no gate {gate_id!r} in this record")


def pair_waveform(record: Record, pair: tuple[int, int]) -> WaveformRecord | None:
    a, b = pair
    return record.table.waveforms.get(f"{a},{b}") or record.table.waveforms.get(f"{b},{a}")


def peak_amplitude_hz(wf: WaveformRecord) -> float:
    """The largest Rabi frequency any segment of the waveform asks of any ion and leg."""
    peak = 0.0
    for seg in wf.segments or ():
        for fn in seg.amplitude_hz.values():
            if fn.kind == "constant" and fn.value is not None:
                peak = max(peak, abs(float(fn.value)))
            elif fn.samples is not None and fn.samples.size:
                peak = max(peak, float(np.max(np.abs(fn.samples))))
    return peak


def carrier_rabi_hz(record: Record, ion: int) -> float | None:
    """The table's calibrated carrier Rabi frequency of the entangling drive on ``ion`` (Section 7.3's key), or None when
    the table carries none."""
    drive = record.job.entangling_drives.get(ion)
    if drive is None:
        return None
    beam = drive.beams[0] if drive.beams else -1
    entry = record.table.entries.get(f"rabi[({ion}, {beam})]")
    if entry is None or entry.status == "uncalibrated" or entry.value <= 0.0:
        return None
    return float(entry.value)


def beat_note_hz(record: Record, tg: TargetRecord) -> float | None:
    """|mu| of the gate's first pulse's first tone: the beat note the hand-set offset is measured against."""
    for k in tg.pulse_indices:
        p = record.schedule.pulses[k]
        for t in p.tones:
            fn = t.detuning_hz
            if fn.kind == "constant" and fn.value is not None:
                return abs(float(fn.value))
            if fn.samples is not None and fn.samples.size:
                return abs(float(fn.samples[0]))
    return None


def ms_matrix(phi0_rad: float, phi1_rad: float, theta_rad: float) -> np.ndarray:
    """MS(phi0, phi1, theta) = cos(theta/2) 1 - i sin(theta/2) GPi(phi0) (x) GPi(phi1) (Section 7.1; conv.native_ms_matrix)."""
    g0 = np.array([[0.0, np.exp(-1j * phi0_rad)], [np.exp(1j * phi0_rad), 0.0]], dtype=complex)
    g1 = np.array([[0.0, np.exp(-1j * phi1_rad)], [np.exp(1j * phi1_rad), 0.0]], dtype=complex)
    g = np.kron(g0, g1)
    return math.cos(theta_rad / 2.0) * np.eye(4, dtype=complex) - 1j * math.sin(theta_rad / 2.0) * g


# ---- the two requests ------------------------------------------------------------------------------------------------------------


def request_angle(record: Record, gate_id: str, chi_rad: float) -> GateRequest:
    """XX(chi) requested for the native MS gate ``gate_id``: the job that plays it, or the refusal (Section 14.4)."""
    tg = target_of(record, gate_id)
    if tg.native_name != "ms" or len(tg.ions) != 2:
        raise RequestError(f"{gate_id} is a {tg.native_name} gate; an angle is requested of an MS gate")
    pair = (int(tg.ions[0]), int(tg.ions[1]))
    step = record.step_of_gate(gate_id)
    k = gate_index(gate_id)
    ops = list(record.compiled.native.ops)
    if k >= len(ops) or ops[k].name != "ms":
        raise RequestError(f"{gate_id} does not index an MS operation of the compiled circuit")
    wf = pair_waveform(record, pair)
    chi_cal = None if wf is None else abs(float(wf.chi_total_rad))
    common = {
        "kind": "angle",
        "gate_id": gate_id,
        "step_index": step.index,
        "pair": pair,
        "requested": float(chi_rad),
        "calibrated_chi_rad": chi_cal,
    }
    if not math.isfinite(chi_rad) or chi_rad == 0.0:
        return GateRequest(
            **common,  # type: ignore[arg-type]
            scale=None,
            peak_rabi_hz=None,
            carrier_rabi_hz=None,
            job=None,
            refusal="XX(0) plays no pulse: request a non-zero angle",
            note="",
        )
    if abs(chi_rad) > MAX_ANGLE_RAD + 1e-12:
        return GateRequest(
            **common,  # type: ignore[arg-type]
            scale=None,
            peak_rabi_hz=None,
            carrier_rabi_hz=None,
            job=None,
            refusal=(
                f"|chi| = {abs(chi_rad):.4g} rad is outside the calibrated range: the native MS gate spans theta = 2|chi| "
                f"in [0, pi], so |chi| <= pi/2 (Section 7.6)"
            ),
            note="",
        )
    if wf is None or chi_cal is None or chi_cal <= 0.0:
        return GateRequest(
            **common,  # type: ignore[arg-type]
            scale=None,
            peak_rabi_hz=None,
            carrier_rabi_hz=None,
            job=None,
            refusal=f"the calibration table carries no entangling waveform for ions {pair}",
            note="",
        )
    scale = math.sqrt(abs(chi_rad) / chi_cal)
    peak_cal = peak_amplitude_hz(wf)
    peak = peak_cal * scale
    carriers = [c for c in (carrier_rabi_hz(record, i) for i in pair) if c is not None]
    # the power limit is the calibrated carrier Rabi frequency of the entangling drive, when the calibrated waveform itself
    # sits below it; a table whose entangling tones already exceed that entry (the example device: the closure needs more
    # than the carrier entry) declares no limit the app can apply, and the request says so instead of inventing one
    carrier = min(carriers) if carriers and min(carriers) >= peak_cal else None
    if carrier is not None and peak > carrier:
        return GateRequest(
            **common,  # type: ignore[arg-type]
            scale=scale,
            peak_rabi_hz=peak,
            carrier_rabi_hz=carrier,
            job=None,
            refusal=(
                f"loop closure impossible within the power limit: XX({chi_rad:.4g}) rescales the calibrated waveform by "
                f"{scale:.3f} to a peak Rabi frequency of {peak / 1e3:.1f} kHz, above the calibrated carrier Rabi frequency "
                f"{carrier / 1e3:.1f} kHz of the entangling drive"
            ),
            note="",
        )
    phi0, phi1, _theta = ops[k].params
    theta = 2.0 * abs(chi_rad)
    if chi_rad < 0.0:
        phi1 = (
            phi1 + math.pi
        )  # GPi(phi + pi) = -GPi(phi) flips the generator's sign (Section 7.2's XX template)
    ops[k] = OpRecord("ms", ops[k].qubits, (float(phi0), float(phi1), float(theta)))
    note = (
        f"XX({chi_rad:.4g}) requested at Level 1 for {gate_id}: MS(theta = {theta:.4g}) replaces the compiled gate; the "
        f"scheduler rescales the calibrated waveform (|chi| = {chi_cal:.4g}) by {scale:.3f} to a peak Rabi frequency of "
        f"{peak / 1e3:.1f} kHz"
        + (
            f" against a {carrier / 1e3:.1f} kHz carrier"
            if carrier is not None
            else "; the device declares no power limit the app can apply (an ideal hardware chain has no saturation)"
        )
    )
    job = dataclasses.replace(
        record.job,
        circuit=CircuitRecord(record.compiled.native.n_qubits, tuple(ops), record.compiled.native.measure),
        requests=tuple(record.job.requests) + (note,),
        label=f"request: XX({chi_rad:.4g}) on {gate_id}",
    )
    return GateRequest(
        **common,  # type: ignore[arg-type]
        scale=scale,
        peak_rabi_hz=peak,
        carrier_rabi_hz=carrier,
        job=job,
        refusal=None,
        note=note,
    )


def request_detuning(record: Record, gate_id: str, offset_hz: float) -> GateRequest:
    """A beat-note offset set by hand at Level 2 for the entangling gate ``gate_id``: the job that plays the pair's waveform
    shifted as written, or the refusal (Section 14.4)."""
    tg = target_of(record, gate_id)
    if tg.native_name not in ("ms", "zz") or len(tg.ions) != 2:
        raise RequestError(
            f"{gate_id} is a {tg.native_name} gate; a beat-note offset belongs to an entangling gate"
        )
    pair = (int(tg.ions[0]), int(tg.ions[1]))
    step = record.step_of_gate(gate_id)
    wf = pair_waveform(record, pair)
    mu = beat_note_hz(record, tg)
    common = {
        "kind": "detuning",
        "gate_id": gate_id,
        "step_index": step.index,
        "pair": pair,
        "requested": float(offset_hz),
        "calibrated_chi_rad": None if wf is None else abs(float(wf.chi_total_rad)),
        "scale": 1.0,
        "peak_rabi_hz": None if wf is None else peak_amplitude_hz(wf),
        "carrier_rabi_hz": None,
    }
    if not math.isfinite(offset_hz):
        return GateRequest(**common, job=None, refusal="the offset must be a finite frequency", note="")  # type: ignore[arg-type]
    if wf is None or wf.segments is None:
        return GateRequest(
            **common,  # type: ignore[arg-type]
            job=None,
            refusal=f"the calibration table carries no segmented entangling waveform for ions {pair} to shift",
            note="",
        )
    if mu is not None and abs(offset_hz) > MAX_DETUNING_FRACTION * mu:
        return GateRequest(
            **common,  # type: ignore[arg-type]
            job=None,
            refusal=(
                f"an offset of {offset_hz / 1e3:.2f} kHz is more than half the {mu / 1e3:.2f} kHz beat note: a tone would "
                "cross the carrier and the pulse would no longer be the bichromatic drive the waveform was solved as"
            ),
            note="",
        )
    key = f"{pair[0]},{pair[1]}"
    overrides = {**record.job.waveform_overrides, key: float(offset_hz)}
    note = (
        f"beat-note detuning set by hand at Level 2 for {gate_id}: every blue leg +{offset_hz / 1e3:.3g} kHz, every red leg "
        f"-{offset_hz / 1e3:.3g} kHz, amplitude unchanged (applied as written, Section 14.4)"
    )
    job = dataclasses.replace(
        record.job,
        waveform_overrides=overrides,
        requests=tuple(record.job.requests) + (note,),
        label=f"request: {offset_hz / 1e3:+.3g} kHz on {gate_id}",
    )
    return GateRequest(**common, job=job, refusal=None, note=note)  # type: ignore[arg-type]


# ---- reading the actual unitary back -----------------------------------------------------------------------------------------------


def ms_phases_of(unitary: np.ndarray) -> tuple[float, float, float]:
    """(phi0, phi1, theta) of an MS target unitary, independent of its global phase: theta from |U_00| = cos(theta/2), the
    phase sum from U_03/U_00 = -i tan(theta/2) e^{-i(phi0 + phi1)} (the element coupling |00> to |11>) and the phase
    difference from U_12/U_11 = -i tan(theta/2) e^{-i(phi0 - phi1)} (|01> to |10>)."""
    c = float(np.clip(abs(unitary[0, 0]), 0.0, 1.0))
    theta = 2.0 * math.acos(c)
    if math.sin(theta / 2.0) < 1e-9 or c < 1e-9:
        return 0.0, 0.0, theta
    sum_phase = -float(np.angle(unitary[0, 3] / unitary[0, 0] / (-1j)))
    diff_phase = -float(np.angle(unitary[1, 2] / unitary[1, 1] / (-1j)))
    phi0 = 0.5 * (sum_phase + diff_phase)
    phi1 = 0.5 * (sum_phase - diff_phase)
    return float(phi0), float(phi1), float(theta)


def fitted_ms_angle(
    choi: np.ndarray, phi0_rad: float, phi1_rad: float, *, n_grid: int = 2001
) -> tuple[float, float]:
    """(theta, entanglement fidelity) of the MS(phi0, phi1, theta) whose ideal channel is closest to the measured Choi
    matrix over theta in [0, pi]: a grid search refined by a parabolic step around the best point."""
    thetas = np.linspace(0.0, math.pi, n_grid)

    def fidelity(theta: float) -> float:
        ideal = core.choi_from_unitary(ms_matrix(phi0_rad, phi1_rad, theta))
        return 1.0 - float(core.entanglement_infidelity(choi, ideal))

    values = np.array([fidelity(float(t)) for t in thetas])
    k = int(np.argmax(values))
    best_theta, best_f = float(thetas[k]), float(values[k])
    if 0 < k < n_grid - 1:
        y0, y1, y2 = values[k - 1], values[k], values[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom < 0.0:
            h = float(thetas[1] - thetas[0])
            shift = 0.5 * h * (y0 - y2) / denom
            cand = best_theta + shift
            fc = fidelity(cand)
            if fc > best_f:
                best_theta, best_f = cand, fc
    return best_theta, best_f


def request_outcome(record: Record, gate_id: str, pm: ProcessMatrixRecord) -> RequestOutcome:
    """What Level 1 shows for an entangling gate whose step has a process matrix: the requested MS angle (from the target
    unitary), the angle the measured channel is closest to, and the tomography's infidelity against the requested unitary."""
    tg = target_of(record, gate_id)
    if tg.native_name != "ms" or len(tg.ions) != 2:
        raise RequestError(f"{gate_id} is not an MS gate")
    phi0, phi1, theta = ms_phases_of(np.asarray(tg.unitary, dtype=complex))
    fitted, fid = fitted_ms_angle(np.asarray(pm.choi, dtype=complex), phi0, phi1)
    key = f"{tg.ions[0]},{tg.ions[1]}"
    alt = f"{tg.ions[1]},{tg.ions[0]}"
    hand = record.job.waveform_overrides.get(key, record.job.waveform_overrides.get(alt))
    return RequestOutcome(
        gate_id=gate_id,
        requested_theta_rad=theta,
        fitted_theta_rad=fitted,
        fitted_fidelity=fid,
        infidelity_to_requested=float(pm.average_gate_infidelity),
        requests=tuple(record.job.requests),
        hand_set_detuning_hz=None if hand is None else float(hand),
    )


__all__ = [
    "MAX_ANGLE_RAD",
    "MAX_DETUNING_FRACTION",
    "GateRequest",
    "RequestError",
    "RequestKind",
    "RequestOutcome",
    "beat_note_hz",
    "carrier_rabi_hz",
    "fitted_ms_angle",
    "gate_index",
    "ms_matrix",
    "ms_phases_of",
    "pair_waveform",
    "peak_amplitude_hz",
    "request_angle",
    "request_detuning",
    "request_outcome",
    "target_of",
]
