"""The played chain: from the pulses the scheduler requests to the fields the ions see (PLAN.md Section 7.3).

The scheduler never reads the device's true values: every pulse it emits carries the CalibrationTable's beliefs and is
marked ``Drive.programmed``. A requested Rabi frequency is a number in the table's units, so the ion sees

    Omega_phys = Omega_req x Omega_derived(ion, beams) / Omega_table(ion, beams),

the device's true rf-power-to-Omega map divided by the calibrated one; the light shift it sees is the derived shift at the
played intensity, delta_derived x sum_tones (Omega_phys,tone/Omega_derived)^p with p the drive kind's scaling power; the
crosstalk its neighbours see is the derived intensity profile. ``physical_schedule`` rewrites every programmed drive this
way and leaves drives built from a ``DerivedDrive`` alone. With a surrogate table (seeds = derived values) the chain is the
identity; with a table fitted by simulated experiments the calibration's error becomes the over-rotation, detuning and
crosstalk error a laboratory's would.

Crosstalk: the chain replaces the ratios of the neighbours the pulse LISTS by the derived values, adds none the table does
not carry, and reports the largest unlisted derived ratio.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.control.pulses import Drive, Pulse, ScaledFn
from qutip_trap.control.schedule import MICROWAVE_BEAM_KEY, Schedule, stark_scaling_power
from qutip_trap.control.table import CalibrationTable, usable

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

UNLISTED_CROSSTALK_REPORT = 1e-3
"""A derived crosstalk ratio above this on a neighbour the table does not list is reported in the chain's notes."""

Envelope = Callable[[float], float] | np.ndarray | float


def _scale_envelope(env: Envelope, factor: float) -> Envelope:
    if factor == 1.0:
        return env
    if callable(env):
        return ScaledFn(env, factor)
    if isinstance(env, np.ndarray):
        return np.asarray(env, dtype=float) * factor
    return float(env) * factor


@dataclass(frozen=True, eq=False)
class _PhysicalStarkFn:
    """tau -> delta_derived sum_tones (|Omega_tone(tau)|/Omega_derived)^p, an array envelope read at its nearest sample
    (picklable when the envelopes are)."""

    stark_hz: float
    omega_hz: float
    power: int
    envelopes: tuple[Envelope, ...]
    duration_s: float

    def __call__(self, tau: float) -> float:
        total = 0.0
        for e in self.envelopes:
            if callable(e):
                val = abs(float(e(tau)))
            elif isinstance(e, np.ndarray):
                k = min(int(round(tau / max(self.duration_s, 1e-300) * (len(e) - 1))), len(e) - 1)
                val = abs(float(e[max(k, 0)]))
            else:
                val = abs(float(e))
            total += (val / self.omega_hz) ** self.power
        return self.stark_hz * total


class _Truth:
    """Derived values per (ion, beams, kind), computed once per schedule."""

    def __init__(self, device: Device) -> None:
        self.device = device
        self._rabi: dict[tuple[int, tuple[int, ...], str], tuple[float, float]] = {}
        self._crosstalk: dict[tuple[int, tuple[int, ...], str], dict[int, complex] | None] = {}

    def rabi_and_stark(self, ion: int, beams: tuple[int, ...], kind: str) -> tuple[float, float]:
        """(carrier Rabi frequency, differential Stark shift) in Hz the device derives for the drive."""
        key = (ion, beams, kind)
        if key not in self._rabi:
            from qutip_trap.light.raman import derive_optical_drive, derive_raman_drive

            if kind == "raman":
                dd = derive_raman_drive(self.device, ion, (beams[0], beams[1]), scattering=False)
            else:
                dd = derive_optical_drive(self.device, ion, beams[0], scattering=False)
            self._rabi[key] = (float(dd.carrier_rabi_hz), float(dd.stark_shift_hz))
        return self._rabi[key]

    def crosstalk(self, ion: int, beams: tuple[int, ...], kind: str) -> dict[int, complex] | None:
        """The derived ratios, or None when the device derives no coupling for the addressed ion (a beam geometry whose
        quadrupole coupling vanishes): the table's beliefs are then kept."""
        key = (ion, beams, kind)
        if key not in self._crosstalk:
            from qutip_trap.light.raman import crosstalk_ratios

            try:
                self._crosstalk[key] = dict(crosstalk_ratios(self.device, ion, beams, kind=kind))  # type: ignore[arg-type]
            except ZeroDivisionError:
                self._crosstalk[key] = None
        return self._crosstalk[key]


def physical_drive(
    device: Device,
    drive: Drive,
    duration_s: float,
    table: CalibrationTable,
    truth: _Truth,
    notes: list[str],
    tag: str,
) -> Drive:
    """The drive as the ions see it: physical envelopes, light shift and crosstalk (module docstring)."""
    if not drive.programmed:
        return drive
    if drive.kind not in ("raman", "optical_E1", "optical_E2"):
        # a microwave, gradient or light-shift drive has no derivable rf-power-to-Omega map here: the request is played as is
        return replace(drive, programmed=False)
    ion = drive.ions[0]
    key = (ion, drive.beams[0] if drive.beams else MICROWAVE_BEAM_KEY)
    omega_true, stark_true = truth.rabi_and_stark(ion, drive.beams, drive.kind)
    belief = table.rabi.get(key)
    if usable(belief) and belief is not None and belief.value > 0.0 and omega_true == 0.0:
        # the table can drive the pulse and the device derives zero coupling for this beam geometry: refuse rather than
        # play a pulse the beams cannot produce
        raise ValueError(
            f"{tag}: the device derives no {drive.kind} coupling for beams {drive.beams} on ion {ion} (Omega = 0), while "
            f"the table carries a usable Rabi entry of {belief.value:.6g} Hz for {key}: the requested Rabi frequency "
            "cannot be played. Fix the beam geometry (polarization, k_hat and B direction) or the drive assignment."
        )
    if usable(belief) and belief is not None and belief.value > 0.0 and omega_true > 0.0:
        ratio = omega_true / float(belief.value)
    else:
        ratio = 1.0
        notes.append(
            f"{tag}: no usable Rabi entry for {key} in the table; the requested Rabi frequency is played as the physical one"
        )
    tones = tuple(replace(t, envelope_hz=_scale_envelope(t.envelope_hz, ratio)) for t in drive.tones)
    # the physical light shift at the played intensity
    power = stark_scaling_power(drive.kind)
    stark: float | Callable[[float], float] = 0.0
    if omega_true > 0.0 and stark_true != 0.0:
        envs = [t.envelope_hz for t in tones]
        constants = [float(e) for e in envs if isinstance(e, (int, float))]
        if len(constants) == len(envs):
            stark = stark_true * float(sum((abs(e) / omega_true) ** power for e in constants))
        else:
            stark = _PhysicalStarkFn(stark_true, omega_true, power, tuple(envs), duration_s)
    # crosstalk: the derived ratios of the neighbours the table lists
    derived = truth.crosstalk(ion, drive.beams, drive.kind)
    if derived is None:
        notes.append(
            f"{tag}: the device derives no coupling for ion {ion} under beams {drive.beams}; the table's crosstalk kept"
        )
        derived = {j: complex(v) for j, v in drive.crosstalk.items()}
    xt: dict[int, complex] = {}
    for j, believed in drive.crosstalk.items():
        if j in derived:
            xt[j] = complex(derived[j])
        else:
            xt[j] = complex(believed)
            notes.append(
                f"{tag}: the device derives no light on ion {j} under beams {drive.beams}; the table's ratio kept"
            )
    unlisted = {j: abs(v) for j, v in derived.items() if j not in xt and j not in drive.ions}
    if unlisted:
        worst = max(unlisted, key=lambda j: unlisted[j])
        if unlisted[worst] >= UNLISTED_CROSSTALK_REPORT:
            notes.append(
                f"{tag}: derived crosstalk {unlisted[worst]:.3g} on ion {worst} is not listed in the table and is not simulated"
            )
    return replace(drive, tones=tones, stark_shift_hz=stark, crosstalk=xt, programmed=False)


def physical_schedule(
    device: Device, schedule: Schedule, table: CalibrationTable
) -> tuple[Schedule, tuple[str, ...]]:
    """The schedule as the ions see it: every programmed drive converted through the device's derived values."""
    truth = _Truth(device)
    notes: list[str] = []
    pulses: list[Pulse] = []
    changed = False
    for p in schedule.pulses:
        if not p.drive.programmed:
            pulses.append(p)
            continue
        changed = True
        local: list[str] = []
        d = physical_drive(
            device, p.drive, p.duration_s, table, truth, local, p.gate_id or f"pulse@{p.t_start_s:.9g}"
        )
        for note in local:
            if note not in notes:
                notes.append(note)
        pulses.append(Pulse(d, p.t_start_s, p.t_end_s, p.gate_id, p.closes_modes))
    if not changed:
        return schedule, ()
    return replace(schedule, pulses=tuple(pulses), phase_frame=dict(schedule.phase_frame)), tuple(notes)
