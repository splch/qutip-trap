"""Calibration: simulated device -> ``CalibrationTable`` (PLAN.md Section 7.5).

``method="closed_form"`` (the default) is the surrogate: the device's derived values as seed entries, the closed-form
entangling waveforms corrected by exact spot checks, and the detection threshold and window from the readout model.
``method="experiments"`` runs the simulated experiments in dependency order, refusing a fit whose upstream entry is
uncalibrated. Reports are cached per device configuration and seed, and invalidated, never regenerated silently, when a
device parameter changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from qutip_trap.calibration.experiments import CalibrationReport, full_calibration
from qutip_trap.hashing import canonical_digest

if TYPE_CHECKING:
    from qutip_trap.calibration.surrogate import SurrogateReport
    from qutip_trap.control.schedule import GateDrive
    from qutip_trap.device.model import Device, ResolvedRoles
    from qutip_trap.machine import Machine


_R = TypeVar("_R")


class CalibrationCache:
    """Calibration reports by key; a service object with state, not a frozen record."""

    __slots__ = ("reports",)

    def __init__(self) -> None:
        self.reports: dict[str, object] = {}

    def report(
        self,
        device: Device,
        roles: ResolvedRoles,
        method: str,
        *,
        seed: int,
        t0_s: float,
        kwargs: Mapping[str, Any],
        build: Callable[[], _R],
    ) -> _R:
        """The report ``build`` makes, cached under the device's hash and ``roles``, the method, the seed, the time and the
        arguments."""
        key = canonical_digest(
            (
                device.hash(),
                roles.gate,
                roles.entangling,
                method,
                int(seed),
                float(t0_s),
                dict(sorted(kwargs.items())),
            )
        )
        if key not in self.reports:
            self.reports[key] = build()
        return cast(_R, self.reports[key])


DEFAULT_CACHE = CalibrationCache()
"""The process-wide cache ``calibrate`` and ``run`` share."""


def cached_surrogate(
    device: Device,
    *,
    seed: int,
    t0_s: float,
    cache: CalibrationCache | None = DEFAULT_CACHE,
    gate_drives: Mapping[int, GateDrive] | None = None,
    entangling_drives: Mapping[int, GateDrive] | None = None,
    **kwargs: Any,
) -> SurrogateReport:
    """``surrogate_table`` through ``cache`` (None: always built); explicit drive maps override the device's roles."""
    from qutip_trap.calibration.surrogate import surrogate_table

    roles = device.roles.resolve(device, gate_drives=gate_drives, entangling_drives=entangling_drives)

    def build() -> SurrogateReport:
        return surrogate_table(
            device, seed=seed, t0_s=t0_s, gate_drives=roles.gate, entangling_drives=roles.entangling, **kwargs
        )

    if cache is None:
        return build()
    return cache.report(device, roles, "closed_form", seed=seed, t0_s=t0_s, kwargs=kwargs, build=build)


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
    roles, the numerics and the physics (the builder options and the hardware chain); ``scans`` are the settings of
    :func:`~qutip_trap.calibration.surrogate.surrogate_table` (``pairs``, ``detection_records``, ``detection_windows_s``,
    ``spot_check``) or of :func:`~qutip_trap.calibration.experiments.full_calibration` (``pairs``, ``scans`` a
    ``CalibrationScans``, and the surrogate's settings), ``experiments`` restricts the simulated experiments (the others
    stay surrogate seeds), ``t0_s`` defaults to the machine's ``Physics.t0_s``. Reports are cached in ``cache`` (None
    disables it); a device whose hash changed never hits a cached table."""
    from qutip_trap.noise.sampling import quiet_sample

    device = machine.device
    t0 = float(machine.physics.t0_s if t0_s is None else t0_s)
    physics = machine.physics
    kwargs: dict[str, Any] = {"options": machine.numerics, "builder_options": physics.builder, **scans}
    if method == "closed_form":
        sur = cached_surrogate(
            device, seed=seed, t0_s=t0, cache=cache, hardware_chain=physics.hardware_chain, **kwargs
        )
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
        return full_calibration(
            device, seed=seed, t0_s=t0, experiments=experiments, physics=physics, **kwargs
        )

    if cache is None:
        return build()
    return cache.report(
        device,
        device.roles.resolve(device),
        "experiments",
        seed=seed,
        t0_s=t0,
        kwargs={"experiments": tuple(experiments), "physics": physics, **kwargs},
        build=build,
    )
