"""Calibration: simulated device -> ``CalibrationTable``.

``method="closed_form"`` (the default) is the surrogate of Section 7.5: the device's derived values as seed entries, the
closed-form entangling waveforms corrected by exact spot checks, and the detection threshold and window from the readout
model. ``method="experiments"`` runs the simulated experiments in dependency order (field, micromotion, modes, Rabi, Stark,
qubit frequency, crosstalk, entangling scans, detection, heating), refusing a fit whose upstream entry is uncalibrated.
Tables are cached per device configuration and seed, and invalidated, never regenerated silently, when a device parameter
changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from qutip_trap.calibration.cache import DEFAULT_CACHE, CalibrationCache
from qutip_trap.calibration.experiments import ALIASES, ORDER, CalibrationReport, full_calibration

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.machine import Machine

EXPERIMENTS: tuple[str, ...] = ORDER + tuple(ALIASES)
"""The experiment names ``calibrate`` accepts: the dependency order of Section 7.5 plus the ``ALIASES`` for its parts
(``mode_spectroscopy``/``thermometry`` are the sideband spectroscopy, ``crosstalk_phase`` the crosstalk scan,
``ms_phase_scan`` the phase alignment ``ms_scan`` runs). Every name resolves to an experiment that runs."""


CalibrationMethod = Literal["closed_form", "experiments"]
"""``closed_form``: the Section 7.5 surrogate (the derived seeds, the closed-form waveforms with exact spot checks, the
detection calibration); ``experiments``: the simulated experiments of M8 in the dependency order."""


def _report(
    device: Device,
    *,
    seed: int,
    experiments: Sequence[str],
    surrogate: bool,
    t0_s: float,
    cache: CalibrationCache | None,
    refresh: bool,
    **kwargs: Any,
) -> Any:
    """The report of the path chosen: the ``SurrogateReport`` of the closed forms or the ``CalibrationReport`` of the
    simulated experiments, through the cache."""
    unknown = [e for e in experiments if e != "all" and e not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown calibration experiments {unknown}; known: {EXPERIMENTS}")
    exp = tuple(experiments)
    key = None
    if cache is not None:
        key = cache.key(device, seed=seed, surrogate=surrogate, experiments=exp, t0_s=t0_s, kwargs=kwargs)
        if not refresh and cache.get(key, device) is not None and key in cache.reports:
            return cache.reports[key]
    if surrogate:
        from qutip_trap.calibration.surrogate import surrogate_table

        report: Any = surrogate_table(device, seed=seed, t0_s=t0_s, **kwargs)
    else:
        report = full_calibration(device, seed=seed, t0_s=t0_s, experiments=exp, **kwargs)
    if cache is not None and key is not None:
        cache.put(key, report.table, report)
    return report


def _closed_form_report(sur: Any, *, t0_s: float) -> CalibrationReport:
    """The ``CalibrationReport`` of the closed-form path: the surrogate's table beside the surrogate itself, no experiment
    results and no refusals (every entry is a ``seed`` or a spot check), the quiet sample the seeds are fitted under."""
    from qutip_trap.noise.sampling import quiet_sample

    return CalibrationReport(
        table=sur.table,
        surrogate=sur,
        results={},
        refused={},
        notes=tuple(sur.notes),
        sample=quiet_sample(0, t0_s),
        experiments=(),
    )


def _machine_kwargs(machine: Machine, scans: Mapping[str, Any]) -> dict[str, Any]:
    """What the machine tells the calibration: its roles as the drive maps, its numerics as the solver options, its
    builder options and caps; ``scans`` (the scan settings) come after and win."""
    roles = machine.device.roles.resolve(machine.device)
    out: dict[str, Any] = {
        "gate_drives": roles.gate,
        "entangling_drives": roles.entangling,
        "options": machine.numerics.to_solver_options(machine.physics),
        "builder_options": machine.physics.builder,
        "caps": machine.numerics.truncation.caps,
    }
    out.update(scans)
    return out


def calibrate(
    machine: Machine,
    *,
    method: CalibrationMethod = "closed_form",
    experiments: Sequence[str] = ("all",),
    seed: int = 0,
    t0_s: float | None = None,
    cache: CalibrationCache | None = DEFAULT_CACHE,
    refresh: bool = False,
    **scans: Any,
) -> CalibrationReport:
    """Calibrate a machine: the ``CalibrationReport`` of ``method`` (its ``table`` is what the scheduler reads),
    ``"closed_form"`` for the Section 7.5 surrogate or ``"experiments"`` for the simulated experiments in the dependency
    order, restricted to ``experiments`` (the other entries stay surrogate seeds). The machine supplies the drive maps (its
    roles), the solver options, the builder options and the caps; ``scans`` are the scan settings of
    :func:`~qutip_trap.calibration.surrogate.surrogate_table` (``pairs``, ``detection_records``, ``detection_windows_s``,
    ``spot_check``, ...) or of :func:`~qutip_trap.calibration.experiments.full_calibration` (``pairs``, ``scans`` of type
    ``CalibrationScans``, a ``sample``); ``t0_s`` defaults to the machine's ``Physics.t0_s``. Reports are cached per (device
    hash, seed, arguments) in ``cache`` (None disables it); ``refresh`` recomputes; a device whose hash changed never hits a
    cached table. ``Machine.calibrated(method=)`` is this call with the table pinned on the machine.
    """
    if method not in ("closed_form", "experiments"):
        raise ValueError("method is 'closed_form' or 'experiments'")
    closed_form = method == "closed_form"
    t0 = float(machine.physics.t0_s if t0_s is None else t0_s)
    report = _report(
        machine.device,
        seed=seed,
        experiments=experiments,
        surrogate=closed_form,
        t0_s=t0,
        cache=cache,
        refresh=refresh,
        **_machine_kwargs(machine, scans),
    )
    if closed_form:
        return _closed_form_report(report, t0_s=t0)
    out: CalibrationReport = report
    return out
