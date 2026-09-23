"""Trajectory, branch and sample parallelism through QuTiP's maps.

Every ``QobjEvo`` coefficient must pickle (so the builder never emits closures); a task in a worker runs its own
trajectories in-process, never in a nested pool; each task carries its keyed seeds, and results keep the input order.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from qutip.settings import available_cpu_count
from qutip.solver.parallel import loky_pmap, parallel_map

if TYPE_CHECKING:
    from qutip_trap.dynamics.engine import SolverOptions

M9B = "milestone M9b (dynamics/parallel.py, PLAN.md Section 11.3 item 9)"

MapKind = Literal["serial", "parallel", "loky"]


MEMORY_FRACTION_FOR_WORKERS = 0.5
"""The share of physical memory the forked workers may occupy together."""
MIN_PARENT_BYTES = 512 * 1024**2
"""The smallest parent footprint the cap assumes, so a fresh process still gets every CPU."""


def peak_rss_bytes() -> int:
    """This process's peak resident set size in bytes; 0 under WebAssembly, which has no ``resource`` module."""
    if sys.platform == "emscripten":
        return 0
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def physical_memory_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


def memory_worker_cap() -> int:
    """How many forked workers fit in ``MEMORY_FRACTION_FOR_WORKERS`` of physical memory, each counted at the parent's
    peak (reference counting dirties a forked worker's copy-on-write pages)."""
    parent = max(peak_rss_bytes(), MIN_PARENT_BYTES)
    return max(1, int(MEMORY_FRACTION_FOR_WORKERS * physical_memory_bytes() // parent))


WORKERS_ENV = "QUTIP_TRAP_MAX_WORKERS"
"""Environment cap on the default worker count (``SolverOptions.workers = None``); an explicit ``workers=`` is not capped."""


def worker_count(options: SolverOptions) -> int:
    """The processes the maps may use: ``SolverOptions.workers``, else every CPU QuTiP sees capped by ``WORKERS_ENV``; 1
    under ``map="serial"``; never more than ``memory_worker_cap()``."""
    if options.map == "serial":
        return 1
    if options.workers is not None:
        wanted = int(options.workers)
    else:
        wanted = int(available_cpu_count())
        env = os.environ.get(WORKERS_ENV, "").strip()
        if env:
            wanted = min(wanted, max(1, int(env)))
    return max(1, min(wanted, memory_worker_cap()))


def map_tasks[T, R](
    task: Callable[[T], R],
    items: Sequence[T],
    *,
    map_kind: MapKind,
    workers: int,
    min_tasks: int = 2,
) -> list[R]:
    """``[task(item) for item in items]`` through QuTiP's ``parallel`` or ``loky`` map (arguments and results must
    pickle); in-process when the map is serial, one worker is available or fewer than ``min_tasks`` items are given."""
    items_list = list(items)
    if map_kind == "serial" or workers <= 1 or len(items_list) < max(min_tasks, 1):
        return [task(item) for item in items_list]
    mapper = parallel_map if map_kind == "parallel" else loky_pmap
    out: list[R] = mapper(
        task,
        items_list,
        map_kw={"num_cpus": min(int(workers), len(items_list))},
        progress_bar="",
    )
    return out


__all__ = [
    "M9B",
    "MEMORY_FRACTION_FOR_WORKERS",
    "MIN_PARENT_BYTES",
    "MapKind",
    "map_tasks",
    "memory_worker_cap",
    "peak_rss_bytes",
    "physical_memory_bytes",
    "worker_count",
]
