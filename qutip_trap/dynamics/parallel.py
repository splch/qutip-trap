"""Trajectory, branch and sample parallelism through QuTiP's maps (PLAN.md Sections 3.4, 9.9, 11.3 item 9; milestone M9b).

Section 11.3 item 9: trajectories and quasi-static samples are spread over the cores with the parallel ``map``, which requires
every ``QobjEvo`` coefficient to be a module-level named function, a ``Coefficient`` or an array (a lambda or closure raises
``PicklingError`` under ``map="parallel"`` on 5.3.1, and only ``map="loky"``, an optional dependency, survives it through
cloudpickle), so the builder never emits closures (``control/pulses.py`` holds the picklable time functions). The same maps
serve the initial-mixture branches and the dynamical samples of ``run()`` and the tomography inputs of GATE_LOCAL, one level
at a time: a task that runs in a worker runs its own trajectories in-process (``map="serial"``), never a nested pool.

Reproducibility (Section 3.4): every task carries its keyed seeds, so the result of a task is independent of the worker it ran
on and of the completion order; results are returned in the order of the inputs. Agreement over 1 and N workers is a tolerance
test at 1e-12 rather than a bitwise one because sums over trajectories accumulate in completion order inside QuTiP.
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
"""The share of physical memory the forked workers may occupy together (the other half stays with the parent, the page cache
and the rest of the machine)."""
MIN_PARENT_BYTES = 512 * 1024**2
"""The smallest parent footprint the cap reckons with, so a fresh process still gets every CPU."""


def peak_rss_bytes() -> int:
    """This process's peak resident set size (``ru_maxrss``: bytes on macOS, kilobytes on Linux); 0 under WebAssembly,
    which has no ``resource`` module (and one CPU, so no pool)."""
    if sys.platform == "emscripten":
        return 0
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def physical_memory_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


def memory_worker_cap() -> int:
    """How many forked workers the machine can carry beside this process.

    QuTiP's ``parallel`` map forks (``multiprocessing.get_context("fork")``), so every worker starts as a copy-on-write image
    of the parent, and Python's reference counting dirties the pages a worker touches: a long session (a test run, an
    application that has built many Hamiltonians) whose parent has grown to several gigabytes turns N workers into up to N
    copies of that footprint. Two kernel watchdog panics on 2026-09-08 were exactly that, 18 workers forked from a multi-
    gigabyte pytest process on a 48 GB machine. The cap keeps the workers' worst-case total (each as large as the parent's
    peak) inside ``MEMORY_FRACTION_FOR_WORKERS`` of physical memory.
    """
    parent = max(peak_rss_bytes(), MIN_PARENT_BYTES)
    return max(1, int(MEMORY_FRACTION_FOR_WORKERS * physical_memory_bytes() // parent))


WORKERS_ENV = "QUTIP_TRAP_MAX_WORKERS"
"""Environment cap on the DEFAULT worker count (``SolverOptions.workers = None``). A process that already runs beside others
(a pytest-xdist worker, one job of several on a CI runner) sets it to 1 so that no run forks a pool of its own; an explicit
``workers=`` request is not touched, so the parallel-map tests still exercise the pools."""


def worker_count(options: SolverOptions) -> int:
    """The processes the maps may use: ``SolverOptions.workers``, else every CPU QuTiP sees (``available_cpu_count``) capped by
    ``WORKERS_ENV`` when it is set; 1 under ``map="serial"``; never more than ``memory_worker_cap()`` allows (ledger
    ``conv.worker_memory_cap``)."""
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
    """``[task(item) for item in items]`` through QuTiP's serial, ``multiprocessing`` (``parallel``) or ``loky`` map.

    Runs in-process when the map is serial, when one worker is available or when fewer than ``min_tasks`` items are given (a
    pool costs more than it saves for a single task). Under the parallel maps every argument and result crosses a process
    boundary and must pickle. Results come back in the order of ``items``.
    """
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
