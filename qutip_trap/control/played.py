"""The played chain: the scheduler's programmed drives (the CalibrationTable's beliefs) converted into what the ions see.

Omega_phys = Omega_req x Omega_derived / Omega_table per (ion, beams); the light shift is the derived one at the played
intensity, delta_derived sum_tones (Omega_phys,tone/Omega_derived)^p; listed neighbours' crosstalk ratios become the
derived ones (unlisted ones are only reported). Drives not ``programmed`` are already physical and pass unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.control.pulses import Drive, Pulse, ScaledFn, Tone
from qutip_trap.control.schedule import MICROWAVE_BEAM_KEY, Schedule, stark_scaling_power
from qutip_trap.control.table import CalibrationTable, usable

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

UNLISTED_CROSSTALK_REPORT = 1e-3
"""A derived crosstalk ratio at least this large on a neighbour the table does not list is reported in the notes."""


def _scale_envelope(
    env: Callable[[float], float] | np.ndarray | float, factor: float
) -> Callable[[float], float] | np.ndarray | float:
    if factor == 1.0:
        return env
    if callable(env):
        return ScaledFn(env, factor)
    if isinstance(env, np.ndarray):
        return np.asarray(env, dtype=float) * factor
    return float(env) * factor


def _peak_hz(env: Callable[[float], float] | np.ndarray | float, duration_s: float) -> float:
    if callable(env):
        grid = np.linspace(0.0, duration_s, 65)
        return float(np.max(np.abs([float(env(x)) for x in grid])))
    if isinstance(env, np.ndarray):
        return float(np.max(np.abs(env)))
    return abs(float(env))


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
        """The derived crosstalk ratios, or None when the device derives no coupling for the addressed ion."""
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
    """The drive as the ions see it, appending notes to ``notes``; ValueError when the table holds a usable Rabi entry for
    beams the device derives no coupling for."""
    if not drive.programmed:
        return drive
    if drive.kind not in ("raman", "optical_E1", "optical_E2"):
        # no derivable rf-power-to-Omega map for these kinds: the request is played as is
        return replace(drive, programmed=False)
    ion = drive.ions[0]
    key = (ion, drive.beams[0] if drive.beams else MICROWAVE_BEAM_KEY)
    omega_true, stark_true = truth.rabi_and_stark(ion, drive.beams, drive.kind)
    belief = table.rabi.get(key)
    if usable(belief) and belief is not None and belief.value > 0.0 and omega_true == 0.0:
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
    tones = tuple(
        Tone(
            detuning_hz=t.detuning_hz,
            phase_rad=t.phase_rad,
            envelope_hz=_scale_envelope(t.envelope_hz, ratio),
            theta_bessel_rad=t.theta_bessel_rad,
        )
        for t in drive.tones
    )
    power = stark_scaling_power(drive.kind)
    if omega_true > 0.0 and stark_true != 0.0:
        envs = [t.envelope_hz for t in tones]
        constants = [float(e) for e in envs if isinstance(e, (int, float))]
        if len(constants) == len(envs):
            stark: float | Callable[[float], float] = stark_true * float(
                sum((abs(e) / omega_true) ** power for e in constants)
            )
        else:

            def stark_fn(
                tau: float, _envs: list[Callable[[float], float] | np.ndarray | float] = envs
            ) -> float:
                total = 0.0
                for e in _envs:
                    if callable(e):
                        val = abs(float(e(tau)))
                    elif isinstance(e, np.ndarray):
                        k = min(int(round(tau / max(duration_s, 1e-300) * (len(e) - 1))), len(e) - 1)
                        val = abs(float(e[max(k, 0)]))
                    else:
                        val = abs(float(e))
                    total += (val / omega_true) ** power
                return stark_true * total

            stark = stark_fn
    else:
        stark = 0.0
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
    """The schedule as the ions see it and the chain's notes; the input schedule itself when no drive is programmed."""
    truth = _Truth(device)
    notes: list[str] = []
    seen: set[str] = set()
    pulses: list[Pulse] = []
    changed = False
    for p in schedule.pulses:
        if not p.drive.programmed:
            pulses.append(p)
            continue
        changed = True
        tag = p.gate_id or f"pulse@{p.t_start_s:.9g}"
        local: list[str] = []
        d = physical_drive(device, p.drive, p.duration_s, table, truth, local, tag)
        for n in local:
            if n not in seen:
                seen.add(n)
                notes.append(n)
        pulses.append(Pulse(d, p.t_start_s, p.t_end_s, p.gate_id, p.closes_modes))
    if not changed:
        return schedule, ()
    return (
        Schedule(
            tuple(pulses),
            schedule.idle,
            schedule.events,
            dict(schedule.phase_frame),
            transports=schedule.transports,
            gates=schedule.gates,
            targets=schedule.targets,
            t0_s=schedule.t0_s,
        ),
        tuple(notes),
    )


__all__ = ["UNLISTED_CROSSTALK_REPORT", "physical_drive", "physical_schedule"]
