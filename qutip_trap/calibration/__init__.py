"""Calibration: the ``CalibrationTable`` the scheduler reads. The default method is the closed-form surrogate with exact
spot checks (``calibration.surrogate``); ``method="experiments"`` fits the table from simulated experiments in the
laboratory's dependency order (``calibration.experiments``). Reports are cached per device hash, seed and arguments, so
a changed device never reuses a stale table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from qutip_trap.calibration.cache import DEFAULT_CACHE, CalibrationCache
from qutip_trap.calibration.experiments import ALIASES, ORDER, CalibrationReport, full_calibration

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.machine import Machine

EXPERIMENTS: tuple[str, ...] = ORDER + tuple(ALIASES)
"""The experiment names ``calibrate`` accepts: the dependency order plus the aliases for its parts."""

CalibrationMethod = Literal["closed_form", "experiments"]
"""``closed_form``: the surrogate; ``experiments``: the simulated experiments in the dependency order."""


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
    """The ``SurrogateReport`` of the closed forms or the ``CalibrationReport`` of the experiments, through the cache."""
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
    """The ``CalibrationReport`` of the closed-form path: the surrogate's table beside the surrogate itself."""
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
    """What the machine tells the calibration (its drives, solver and builder options and caps); ``scans`` win."""
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
    """Calibrate ``machine``: the ``CalibrationReport`` of ``method``, whose ``table`` the scheduler reads.

    ``experiments`` restricts the experiments method (the other entries stay surrogate seeds). ``scans`` are the scan
    settings of ``surrogate_table`` (``pairs``, ``detection_records``, ``detection_windows_s``, ``spot_check``, ...) or of
    ``full_calibration`` (``pairs``, ``scans``, ``sample``); ``t0_s`` defaults to ``machine.physics.t0_s``. ``cache``
    None disables caching and ``refresh`` recomputes. ``Machine.calibrated`` pins the table on the machine."""
    if method not in ("closed_form", "experiments"):
        raise ValueError("method is 'closed_form' or 'experiments'")
    t0 = float(machine.physics.t0_s if t0_s is None else t0_s)
    report = _report(
        machine.device,
        seed=seed,
        experiments=experiments,
        surrogate=method == "closed_form",
        t0_s=t0,
        cache=cache,
        refresh=refresh,
        **_machine_kwargs(machine, scans),
    )
    if method == "closed_form":
        return _closed_form_report(report, t0_s=t0)
    out: CalibrationReport = report
    return out
