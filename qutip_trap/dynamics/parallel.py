"""Trajectory, branch and sample parallelism through QuTiP's maps (PLAN.md Section 3.4).

The parallel maps pickle every task, so the builder's coefficients are module-level classes and functions, never closures
(``map="loky"`` would pickle closures through cloudpickle). One level runs in parallel at a time: a task that runs in a
worker runs its own trajectories in-process. Every task carries its keyed seeds, so its result does not depend on the
worker, the completion order or the chunk it travels in; results come back in the order of the inputs.

The tasks travel in contiguous chunks, each one pickle, so the items of a chunk share in the worker what they share in the
parent, as the items of an in-process map do: what an object memoizes per instance (a device's digest, which fingerprints
every segment Hamiltonian) is computed once per chunk, and a task whose item carries state one run leaves for the next (an
engine's propagator cache) runs on its own copy of it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal, NamedTuple

from qutip.settings import available_cpu_count
from qutip.solver.parallel import loky_pmap, parallel_map

if TYPE_CHECKING:
    from qutip_trap.options import Numerics

MapKind = Literal["serial", "parallel", "loky"]

MEMORY_FRACTION_FOR_WORKERS = 0.5
"""The share of physical memory the forked workers may occupy together."""
MIN_PARENT_BYTES = 512 * 1024**2
"""The smallest parent footprint the memory cap reckons with, so a fresh process still gets every CPU."""
WORKERS_ENV = "QUTIP_TRAP_MAX_WORKERS"
"""Environment cap on the default worker count (``Numerics.workers = None``); an explicit ``workers`` is not capped.
A process that already runs beside others (a pytest-xdist worker) sets it to 1."""
CHUNKS_PER_WORKER = 16
"""The contiguous chunks a parallel map sends its items in, per worker. QuTiP's pools spend a few tenths of a millisecond of
the parent's time on every task and keep one task per worker in flight, which a task of a few milliseconds cannot hide:
sent one per task over 18 workers, the 193 483 branch runs of ``presets.ca40_optical(2)`` kept the parent busy for 82 s of
a 94 s run while the workers waited (44 s in chunks), and each run unpickled a device of its own and recomputed its digest,
2.8 ms of CPU per run where the runs of a chunk, sharing one device, take 1.6 ms. Sixteen chunks per worker pay the pool's
cost a few hundred times per map, and the last chunks to finish idle the other workers for about a sixteenth of it; a map
of fewer items sends one per chunk."""


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


def worker_count(options: Numerics) -> int:
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


class _Indexed[T](NamedTuple):
    """An item and its position in the map's input, which puts results that arrive in completion order back in order."""

    position: int
    item: T


@dataclass(frozen=True)
class _Chunk[T, R]:
    """``task`` applied to every item of a contiguous chunk in order, each result returned beside its item's position (a
    module-level class, so it pickles with the task)."""

    task: Callable[[T], R]

    def __call__(self, chunk: Sequence[_Indexed[T]]) -> list[tuple[int, R]]:
        return [(entry.position, self.task(entry.item)) for entry in chunk]


def map_tasks[T, R](
    task: Callable[[T], R],
    items: Sequence[T],
    *,
    map_kind: MapKind,
    workers: int,
    on_done: Callable[[int], None] | None = None,
) -> list[R]:
    """``[task(item) for item in items]`` through QuTiP's serial, ``parallel`` or ``loky`` map, in the order of ``items``;
    ``on_done`` is called with the number of items finished each time some finish (in completion order under a parallel
    map, a chunk at a time, so a caller can report progress while the others run).

    In-process when the map is serial, one worker is available or fewer than two items are given; under the parallel
    maps every argument and result must pickle, and the items travel in ``CHUNKS_PER_WORKER`` contiguous chunks per worker,
    each one pickle, whose items share in the worker what they share here (the module docstring).
    """
    items_list = list(items)
    if map_kind == "serial" or workers <= 1 or len(items_list) < 2:
        out: list[R] = []
        for item in items_list:
            out.append(task(item))
            if on_done is not None:
                on_done(len(out))
        return out
    if map_kind == "loky" and find_spec("loky") is None:
        raise ImportError(
            "Numerics(map='loky') maps through the loky package, which is not installed: install it (pip install loky) "
            "or use map='parallel'"
        )
    mapper = parallel_map if map_kind == "parallel" else loky_pmap
    indexed = [_Indexed(k, item) for k, item in enumerate(items_list)]
    n_chunks = min(len(indexed), int(workers) * CHUNKS_PER_WORKER)
    edges = [len(indexed) * c // n_chunks for c in range(n_chunks + 1)]
    chunks = [indexed[lo:hi] for lo, hi in zip(edges[:-1], edges[1:])]
    results: dict[int, R] = {}

    def collect(finished: list[tuple[int, R]]) -> None:
        results.update(finished)
        if on_done is not None:
            on_done(len(results))

    mapper(
        _Chunk(task),
        chunks,
        reduce_func=collect,
        map_kw={"num_cpus": min(int(workers), len(chunks))},
        progress_bar="",
    )
    return [results[k] for k in range(len(items_list))]
