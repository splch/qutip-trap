"""Trajectory, branch and sample parallelism through QuTiP's maps (PLAN.md Section 3.4).

The parallel maps pickle every task, so the builder's coefficients are module-level classes and functions, never closures
(``map="loky"`` would pickle closures through cloudpickle). One level runs in parallel at a time: a task that runs in a
worker runs its own trajectories in-process. Every task carries its keyed seeds, so its result does not depend on the
worker or the completion order; results come back in the order of the inputs.
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

MapKind = Literal["serial", "parallel", "loky"]

MEMORY_FRACTION_FOR_WORKERS = 0.5
"""The share of physical memory the forked workers may occupy together."""
MIN_PARENT_BYTES = 512 * 1024**2
"""The smallest parent footprint the memory cap reckons with, so a fresh process still gets every CPU."""
WORKERS_ENV = "QUTIP_TRAP_MAX_WORKERS"
"""Environment cap on the default worker count (``SolverOptions.workers = None``); an explicit ``workers`` is not capped.
A process that already runs beside others (a pytest-xdist worker) sets it to 1."""


def peak_rss_bytes() -> int:
    """This process's peak resident set size (``ru_maxrss``: bytes on macOS, kilobytes on Linux); 0 under WebAssembly,
    which has no ``resource`` module and one CPU."""
    if sys.platform == "emscripten":
        return 0
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def physical_memory_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


def memory_worker_cap() -> int:
    """How many forked workers fit beside this process: QuTiP's ``parallel`` map forks, and each copy-on-write worker can
    grow to the parent's size, so the cap keeps (workers x parent peak) inside ``MEMORY_FRACTION_FOR_WORKERS`` of memory."""
    parent = max(peak_rss_bytes(), MIN_PARENT_BYTES)
    return max(1, int(MEMORY_FRACTION_FOR_WORKERS * physical_memory_bytes() // parent))


def worker_count(options: SolverOptions) -> int:
    """The processes the maps may use: 1 under ``map="serial"``, else ``options.workers`` or every CPU QuTiP sees capped by
    ``WORKERS_ENV``, never more than ``memory_worker_cap()``."""
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
    task: Callable[[T], R], items: Sequence[T], *, map_kind: MapKind, workers: int
) -> list[R]:
    """``[task(item) for item in items]`` through QuTiP's serial, ``parallel`` or ``loky`` map, in the order of ``items``.

    In-process when the map is serial, one worker is available or fewer than two items are given; under the parallel
    maps every argument and result must pickle.
    """
    items_list = list(items)
    if map_kind == "serial" or workers <= 1 or len(items_list) < 2:
        return [task(item) for item in items_list]
    mapper = parallel_map if map_kind == "parallel" else loky_pmap
    out: list[R] = mapper(
        task,
        items_list,
        map_kw={"num_cpus": min(int(workers), len(items_list))},
        progress_bar="",
    )
    return out
