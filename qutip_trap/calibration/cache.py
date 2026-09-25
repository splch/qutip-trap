"""The calibration cache: reports kept per device configuration, drive roles, seed, time and arguments (PLAN.md Section
7.5).

The key is ``Device.hash()`` (which leaves out the beam roles, so the resolved drive maps join it), the method, the seed,
the laboratory time and a digest of the calibration's arguments: a changed device parameter never hits a table of the old
configuration, and the new configuration has none until it is calibrated. In-process only (a table may hold callables).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from qutip_trap.hashing import canonical_digest

if TYPE_CHECKING:
    from qutip_trap.calibration.surrogate import SurrogateReport
    from qutip_trap.control.schedule import GateDrive
    from qutip_trap.device.model import Device, ResolvedRoles

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
