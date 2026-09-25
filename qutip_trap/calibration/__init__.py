"""Calibration: simulated device -> ``CalibrationTable`` (PLAN.md Section 7.5).

``method="closed_form"`` (the default) is the surrogate: the device's derived values as seed entries, the closed-form
entangling waveforms corrected by exact spot checks, and the detection threshold and window from the readout model.
``method="experiments"`` runs the simulated experiments in dependency order, refusing a fit whose upstream entry is
uncalibrated. Reports are cached per device configuration and seed, and invalidated, never regenerated silently, when a
device parameter changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from qutip_trap.calibration.cache import DEFAULT_CACHE, CalibrationCache, cached_surrogate
from qutip_trap.calibration.experiments import CalibrationReport, full_calibration

if TYPE_CHECKING:
    from qutip_trap.machine import Machine

CalibrationMethod = Literal["closed_form", "experiments"]
"""``closed_form``: the surrogate (derived seeds, closed-form waveforms with exact spot checks, the detection calibration);
``experiments``: the simulated experiments in the dependency order."""


def calibrate(
    machine: Machine,
    *,
    method: CalibrationMethod = "closed_form",
    experiments: Sequence[str] = ("all",),
    seed: int = 0,
    t0_s: float | None = None,
    cache: CalibrationCache | None = DEFAULT_CACHE,
    **scans: Any,
) -> CalibrationReport:
    """Calibrate a machine: the ``CalibrationReport`` whose ``table`` the scheduler reads. The machine supplies the drive
    roles, the solver options, the builder options and the caps; ``scans`` are the settings of
    :func:`~qutip_trap.calibration.surrogate.surrogate_table` (``pairs``, ``detection_records``, ``detection_windows_s``,
    ``spot_check``) or of :func:`~qutip_trap.calibration.experiments.full_calibration` (``pairs``, ``scans`` a
    ``CalibrationScans``, and the surrogate's settings), ``experiments`` restricts the simulated experiments (the others
    stay surrogate seeds), ``t0_s`` defaults to the machine's ``Physics.t0_s``. Reports are cached in ``cache`` (None
    disables it); a device whose hash changed never hits a cached table."""
    from qutip_trap.noise.sampling import quiet_sample

    device = machine.device
    t0 = float(machine.physics.t0_s if t0_s is None else t0_s)
    kwargs: dict[str, Any] = {
        "options": machine.numerics.to_solver_options(machine.physics),
        "builder_options": machine.physics.builder,
        "caps": machine.numerics.truncation.caps,
        **scans,
    }
    if method == "closed_form":
        sur = cached_surrogate(device, seed=seed, t0_s=t0, cache=cache, **kwargs)
        return CalibrationReport(
            table=sur.table,
            surrogate=sur,
            results={},
            refused={},
            notes=tuple(sur.notes),
            sample=quiet_sample(0, t0),
            experiments=(),
        )
    if method != "experiments":
        raise ValueError("method is 'closed_form' or 'experiments'")

    def build() -> CalibrationReport:
        return full_calibration(device, seed=seed, t0_s=t0, experiments=experiments, **kwargs)

    if cache is None:
        return build()
    return cache.report(
        device,
        device.roles.resolve(device),
        "experiments",
        seed=seed,
        t0_s=t0,
        kwargs={"experiments": tuple(experiments), **kwargs},
        build=build,
    )
