"""The in-process calibration cache: a table is returned only for the device whose ``Device.hash()`` it was fitted for."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from qutip_trap.hashing import canonical_digest

if TYPE_CHECKING:
    from qutip_trap.calibration.surrogate import SurrogateReport
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device


def _kwargs_digest(kwargs: Mapping[str, Any]) -> str:
    return canonical_digest({k: v for k, v in sorted(kwargs.items())})


CacheKey = tuple[str, int, bool, tuple[str, ...], float, str]


class CalibrationCache:
    """Tables and reports keyed by (device hash, seed, surrogate, experiments, t0, kwargs digest), with hit/miss counts."""

    __slots__ = ("hits", "misses", "reports", "tables")

    def __init__(self) -> None:
        self.tables: dict[CacheKey, CalibrationTable] = {}
        self.reports: dict[CacheKey, object] = {}
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def key(
        device: Device,
        *,
        seed: int,
        surrogate: bool,
        experiments: tuple[str, ...],
        t0_s: float,
        kwargs: Mapping[str, Any],
    ) -> CacheKey:
        return (
            device.hash(),
            int(seed),
            bool(surrogate),
            tuple(experiments),
            float(t0_s),
            _kwargs_digest(kwargs),
        )

    def get(self, key: CacheKey, device: Device) -> CalibrationTable | None:
        """The table cached under ``key`` if it was fitted for ``device``, else None."""
        table = self.tables.get(key)
        if table is None or not table.is_current_for(device.hash()):
            self.misses += 1
            return None
        self.hits += 1
        return table

    def put(
        self,
        key: CacheKey,
        table: CalibrationTable,
        report: object = None,
    ) -> None:
        if key[0] != table.device_hash:
            raise ValueError("a table is cached under the hash of the device it was fitted for (Section 7.5)")
        self.tables[key] = table
        if report is not None:
            self.reports[key] = report

    def invalidate(self, device: Device) -> int:
        """Drop every table of ``device``'s configuration; returns how many were dropped."""
        h = device.hash()
        keys = [k for k in self.tables if k[0] == h]
        for k in keys:
            del self.tables[k]
            self.reports.pop(k, None)
        return len(keys)

    def tables_for(self, device: Device) -> tuple[CalibrationTable, ...]:
        h = device.hash()
        return tuple(t for k, t in self.tables.items() if k[0] == h)

    def clear(self) -> None:
        self.tables.clear()
        self.reports.clear()
        self.hits = 0
        self.misses = 0


DEFAULT_CACHE = CalibrationCache()
"""The process-wide cache ``calibrate`` and ``run`` share."""


def cached_surrogate(
    device: Device, *, seed: int, t0_s: float, cache: CalibrationCache | None = None, **kwargs: Any
) -> SurrogateReport:
    """``surrogate_table`` through the cache: the same device, seed, time and arguments return the same report."""
    from qutip_trap.calibration.surrogate import surrogate_table

    store = DEFAULT_CACHE if cache is None else cache
    key = store.key(device, seed=seed, surrogate=True, experiments=("surrogate",), t0_s=t0_s, kwargs=kwargs)
    table = store.get(key, device)
    report = store.reports.get(key)
    if table is not None and report is not None:
        return report  # type: ignore[return-value]
    report = surrogate_table(device, seed=seed, t0_s=t0_s, **kwargs)
    store.put(key, report.table, report)
    return report


__all__ = ["DEFAULT_CACHE", "CalibrationCache", "cached_surrogate"]
