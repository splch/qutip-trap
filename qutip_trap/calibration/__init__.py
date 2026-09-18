"""Calibration emulation: simulated experiments -> CalibrationTable (PLAN.md Section 7.5; milestones M4, M5, M6, M8).

Sits ABOVE control/ and dynamics/ (Section 3.2): it consumes the experiments package and produces the plain data object
``control.table.CalibrationTable``. The default calibration is the SURROGATE of Section 7.5 (``surrogate=True``): the
device's derived values as ``seed`` entries, the closed-form entangling waveforms corrected by exact spot checks
(``calibration.entangling``, M4), and the detection threshold and window from the simulated readout model
(``calibration.readout``, M5), assembled by ``calibration.surrogate`` (M6). The full simulated-experiment path
(``surrogate=False``, ``calibration.experiments``, M8) follows the dependency graph field -> micromotion -> modes -> rabi,
stark, qubit frequency -> crosstalk -> entangling scans -> detection, heating, refusing a downstream fit whose upstream entry
is ``uncalibrated`` (Appendix E), and reports the surrogate's error beside every entry it replaces. Tables are cached per
device configuration and noise seed (``calibration.cache``) and invalidated, never regenerated silently, when a device
parameter changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

from qutip_trap._compat import deprecated
from qutip_trap.calibration.cache import DEFAULT_CACHE, CalibrationCache
from qutip_trap.calibration.experiments import (
    ALIASES,
    ORDER,
    UPSTREAM,
    CalibrationError,
    CalibrationReport,
    CalibrationScans,
    full_calibration,
    upstream_status,
)

if TYPE_CHECKING:
    from qutip_trap.control.table import CalibrationTable
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
    simulated experiments, through the cache (the 0.1.0 body of ``calibrate_with_report``)."""
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


@overload
def calibrate(
    machine: Machine,
    *,
    method: CalibrationMethod = ...,
    experiments: Sequence[str] = ...,
    seed: int = ...,
    t0_s: float | None = ...,
    cache: CalibrationCache | None = ...,
    refresh: bool = ...,
    surrogate: bool | None = ...,
    **scans: Any,
) -> CalibrationReport: ...


@overload
def calibrate(
    machine: Device,
    *,
    method: CalibrationMethod = ...,
    experiments: Sequence[str] = ...,
    seed: int = ...,
    t0_s: float | None = ...,
    cache: CalibrationCache | None = ...,
    refresh: bool = ...,
    surrogate: bool | None = ...,
    **scans: Any,
) -> CalibrationTable: ...


def calibrate(
    machine: Machine | Device,
    *,
    method: CalibrationMethod = "closed_form",
    experiments: Sequence[str] = ("all",),
    seed: int = 0,
    t0_s: float | None = None,
    cache: CalibrationCache | None = DEFAULT_CACHE,
    refresh: bool = False,
    surrogate: bool | None = None,
    **scans: Any,
) -> CalibrationReport | CalibrationTable:
    """Calibrate a machine (docs/api_implementation_plan.md 2.2): the ``CalibrationReport`` of ``method`` (its ``table`` is
    what the scheduler reads), ``"closed_form"`` for the Section 7.5 surrogate or ``"experiments"`` for the M8 simulated
    experiments in the dependency order, restricted to ``experiments`` (the other entries stay surrogate seeds). The machine
    supplies the drive maps (its roles), the solver options, the builder options and the caps; ``scans`` are the scan
    settings of :func:`~qutip_trap.calibration.surrogate.surrogate_table` (``pairs``, ``detection_records``,
    ``detection_windows_s``, ``spot_check``, ...) or of :func:`~qutip_trap.calibration.experiments.full_calibration`
    (``pairs``, ``scans`` of type ``CalibrationScans``, a ``sample``); ``t0_s`` defaults to the machine's ``Physics.t0_s``.
    Reports are cached per (device hash, seed, arguments) in ``cache`` (the process-wide ``DEFAULT_CACHE``; None disables
    it); ``refresh`` recomputes; a device whose hash changed never hits a cached table (Section 7.5: invalidated, not
    regenerated silently). ``Machine.calibrated(method=)`` is this call with the table pinned on the machine.

    A bare ``Device`` is the 0.1.0 shape and keeps its 0.1.0 result, the ``CalibrationTable`` (its ``surrogate=True/False``
    keyword is deprecated in favour of ``method``; the drive keywords in ``scans`` are deprecated in favour of the device's
    roles); it warns since 0.4.0 (``machine.warn_bare_device``: wrap it, ``Machine(device)``, and read ``.table`` off the
    report). ``calibrate_with_report`` is the deprecated name of the report on a device.
    """
    from qutip_trap._compat import message, warn
    from qutip_trap.device.model import Device as _Device
    from qutip_trap.machine import DRIVE_KEYWORDS, warn_bare_device

    unknown = [e for e in experiments if e != "all" and e not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown calibration experiments {unknown}; known: {EXPERIMENTS}")
    if surrogate is not None:
        warn(
            message(
                "the 'surrogate' argument of qutip_trap.calibration.calibrate",
                "v0.5",
                "Pass method='closed_form' (surrogate=True) or method='experiments' (surrogate=False).",
            ),
            stacklevel=2,
        )
        method = "closed_form" if surrogate else "experiments"
    if method not in ("closed_form", "experiments"):
        raise ValueError("method is 'closed_form' or 'experiments'")
    for key, fix in DRIVE_KEYWORDS.items():
        if key in scans:
            warn(
                message(f"the {key!r} argument of qutip_trap.calibration.calibrate", "v0.5", fix),
                stacklevel=2,
            )
    closed_form = method == "closed_form"
    if isinstance(machine, _Device):
        warn_bare_device("qutip_trap.calibration.calibrate", stacklevel=2)
        report = _report(
            machine,
            seed=seed,
            experiments=experiments,
            surrogate=closed_form,
            t0_s=0.0 if t0_s is None else float(t0_s),
            cache=cache,
            refresh=refresh,
            **scans,
        )
        table: CalibrationTable = report.table
        return table
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


@deprecated(
    deadline="v0.5",
    fix="Call calibrate(Machine(device), method='closed_form' or 'experiments'); it returns the CalibrationReport.",
)
def calibrate_with_report(
    device: Device,
    *,
    seed: int = 0,
    experiments: Sequence[str] = ("all",),
    surrogate: bool = True,
    t0_s: float = 0.0,
    cache: CalibrationCache | None = DEFAULT_CACHE,
    refresh: bool = False,
    **kwargs: Any,
) -> CalibrationReport | Any:
    """The 0.1.0 name of ``calibrate`` returning the full report on a device: the ``CalibrationReport`` of the simulated
    experiments (``surrogate=False``) or the ``SurrogateReport`` of the closed-form path; both carry the table. Deprecated
    in 0.3.0: ``calibrate(machine, method=...)`` returns a ``CalibrationReport`` on every call."""
    return _report(
        device,
        seed=seed,
        experiments=experiments,
        surrogate=surrogate,
        t0_s=t0_s,
        cache=cache,
        refresh=refresh,
        **kwargs,
    )


__all__ = [
    "ALIASES",
    "DEFAULT_CACHE",
    "EXPERIMENTS",
    "ORDER",
    "UPSTREAM",
    "CalibrationCache",
    "CalibrationError",
    "CalibrationMethod",
    "CalibrationReport",
    "CalibrationScans",
    "calibrate",
    "calibrate_with_report",
    "full_calibration",
    "upstream_status",
]
