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
from qutip_trap.calibration.surrogate import SurrogateReport, canonical_pairs, surrogate_table
from qutip_trap.hashing import canonical_digest

if TYPE_CHECKING:
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
    pairs: Sequence[Sequence[int]] | None = None,
    **kwargs: Any,
) -> SurrogateReport:
    """``surrogate_table`` through ``cache`` (None: always built); explicit drive maps override the device's roles. The
    request is keyed and built with its pairs as ``canonical_pairs`` orders them, so a run's circuit pairs and a caller's
    list of the same pairs share one entry."""
    roles = device.roles.resolve(device, gate_drives=gate_drives, entangling_drives=entangling_drives)
    request = {**kwargs, "pairs": canonical_pairs(device, pairs)}

    def build() -> SurrogateReport:
        return surrogate_table(
            device,
            seed=seed,
            t0_s=t0_s,
            gate_drives=roles.gate,
            entangling_drives=roles.entangling,
            **request,
        )

    if cache is None:
        return build()
    return cache.report(device, roles, "closed_form", seed=seed, t0_s=t0_s, kwargs=request, build=build)


CalibrationMethod = Literal["closed_form", "experiments"]
"""``closed_form``: the surrogate (derived seeds, closed-form waveforms with exact spot checks, the detection calibration);
``experiments``: the simulated experiments in the dependency order."""


def calibrate(
    machine: Machine,
    *,
    method: CalibrationMethod = "closed_form",
    experiments: Sequence[str] = ("all",),
    pairs: Sequence[Sequence[int]] | None = None,
    seed: int = 0,
    t0_s: float | None = None,
    cache: CalibrationCache | None = DEFAULT_CACHE,
    **scans: Any,
) -> CalibrationReport:
    """Calibrate a machine: the ``CalibrationReport`` whose ``table`` the scheduler reads. The machine supplies the drive
    roles, the numerics and the physics every spot check and experiment plays under (the builder options, the hardware
    chain, the Stark compensation and the crosstalk echo); ``pairs`` restricts the entangling waveforms (default: every pair
    of the crystal, in either key order); ``scans`` are the other settings of
    :func:`~qutip_trap.calibration.surrogate.surrogate_table` (``detection_records``, ``detection_windows_s``,
    ``spot_check``) or of :func:`~qutip_trap.calibration.experiments.full_calibration` (``scans`` a ``CalibrationScans``,
    and the surrogate's settings), ``experiments`` restricts the simulated experiments (the others stay surrogate seeds),
    ``t0_s`` defaults to the machine's ``Physics.t0_s``. Reports are cached in ``cache`` (None disables it), keyed by the
    request with its pairs as ``canonical_pairs`` orders them; a device whose hash changed never hits a cached table."""
    from qutip_trap.noise.sampling import quiet_sample

    device = machine.device
    t0 = float(machine.physics.t0_s if t0_s is None else t0_s)
    kwargs: dict[str, Any] = {
        "options": machine.numerics,
        "physics": machine.physics,
        "pairs": canonical_pairs(device, pairs),
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
