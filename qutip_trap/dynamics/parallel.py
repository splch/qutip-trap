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

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from qutip.settings import available_cpu_count
from qutip.solver.parallel import loky_pmap, parallel_map

if TYPE_CHECKING:
    from qutip_trap.dynamics.engine import SolverOptions

M9B = "milestone M9b (dynamics/parallel.py, PLAN.md Section 11.3 item 9)"

MapKind = Literal["serial", "parallel", "loky"]


def worker_count(options: SolverOptions) -> int:
    """The processes the maps may use: ``SolverOptions.workers``, else every CPU QuTiP sees (``available_cpu_count``); 1 under
    ``map="serial"``."""
    if options.map == "serial":
        return 1
    if options.workers is not None:
        return max(1, int(options.workers))
    return max(1, int(available_cpu_count()))


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


__all__ = ["M9B", "MapKind", "map_tasks", "worker_count"]
