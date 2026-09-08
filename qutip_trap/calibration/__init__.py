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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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

EXPERIMENTS: tuple[str, ...] = ORDER + tuple(ALIASES)
"""The experiment names ``calibrate`` accepts: the dependency order of Section 7.5 plus the ``ALIASES`` for its parts
(``mode_spectroscopy``/``thermometry`` are the sideband spectroscopy, ``crosstalk_phase`` the crosstalk scan,
``ms_phase_scan`` the phase alignment ``ms_scan`` runs). Every name resolves to an experiment that runs."""


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
    """``calibrate`` returning the full report: the ``CalibrationReport`` of the simulated experiments (``surrogate=False``) or the
    ``SurrogateReport`` of the closed-form path; both carry the table."""
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


def calibrate(
    device: Device,
    *,
    seed: int = 0,
    experiments: Sequence[str] = ("all",),
    surrogate: bool = True,
    t0_s: float = 0.0,
    cache: CalibrationCache | None = DEFAULT_CACHE,
    refresh: bool = False,
    **kwargs: Any,
) -> CalibrationTable:
    """The CalibrationTable of ``device`` at ``t0_s``: the Section 7.5 surrogate (default), or the M8 simulated experiments.

    ``surrogate=True`` passes the keyword arguments to :func:`qutip_trap.calibration.surrogate.surrogate_table` (gate drives,
    pairs, detection records, spot-check options); ``surrogate=False`` to :func:`qutip_trap.calibration.experiments.full_calibration`
    (gate drives, pairs, ``scans`` of type ``CalibrationScans``, a ``sample``). ``experiments`` restricts the simulated-experiment
    path to a subset, run in the dependency order of Section 7.5 with the other entries kept as surrogate seeds. Tables are cached
    per (device hash, seed, arguments) in ``cache`` (the process-wide ``DEFAULT_CACHE`` by default; None disables it); ``refresh``
    recomputes. A device whose hash changed never hits a cached table (Section 7.5: invalidated, not regenerated silently).
    """
    report = calibrate_with_report(
        device,
        seed=seed,
        experiments=experiments,
        surrogate=surrogate,
        t0_s=t0_s,
        cache=cache,
        refresh=refresh,
        **kwargs,
    )
    table: CalibrationTable = report.table
    return table


__all__ = [
    "ALIASES",
    "DEFAULT_CACHE",
    "EXPERIMENTS",
    "ORDER",
    "UPSTREAM",
    "CalibrationCache",
    "CalibrationError",
    "CalibrationReport",
    "CalibrationScans",
    "calibrate",
    "calibrate_with_report",
    "full_calibration",
    "upstream_status",
]
