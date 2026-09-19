"""Suite-wide configuration.

Under pytest-xdist every worker is already one of several processes, so the engine's default trajectory map must not fork a
pool of its own inside each of them (nested pools: N workers x every CPU, the oversubscription and the forked copies of a
multi-gigabyte pytest process that ``dynamics/parallel.memory_worker_cap`` guards against). ``QUTIP_TRAP_MAX_WORKERS`` caps
the DEFAULT worker count only; a test that asks for ``workers=`` explicitly still gets its pool (``tests/test_parallel.py``).

``QUTIP_TRAP_TEST_WATCHDOG_S`` (seconds; CI sets 1800) arms a per-test watchdog that neither pytest-timeout method provides: a
test stuck inside one long C call that holds the interpreter lock (a multi-gigabyte sparse factorization, a runaway integrator)
starves both the SIGALRM handler and the timer thread, and the CI runs of 1371cf0 and 80b89ed sat for hours on such tests
with the 30-minute timeout never firing. ``faulthandler.dump_traceback_later`` is a C-level thread that needs no lock: at the
deadline it writes every thread's Python stack to stderr and ends the process, xdist reports the test as a crash and starts a
replacement worker, and the dump names the line the test was on. xdist does not relay a worker's stderr, so the dump goes to
``.pytest_watchdog/<worker>.txt`` (``QUTIP_TRAP_TEST_WATCHDOG_DIR``), which CI prints after the step. A
``@pytest.mark.timeout(N)`` on a test sets its own deadline.
"""

from __future__ import annotations

import faulthandler
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ.setdefault("QUTIP_TRAP_MAX_WORKERS", "1")

_WATCHDOG_ENV = "QUTIP_TRAP_TEST_WATCHDOG_S"


@pytest.fixture(autouse=True)
def _watchdog(request: pytest.FixtureRequest) -> Iterator[None]:
    default = os.environ.get(_WATCHDOG_ENV, "").strip()
    if not default:
        yield
        return
    marker = request.node.get_closest_marker("timeout")
    seconds = float(marker.args[0]) if marker is not None and marker.args else float(default)
    directory = Path(os.environ.get("QUTIP_TRAP_TEST_WATCHDOG_DIR", ".pytest_watchdog"))
    directory.mkdir(parents=True, exist_ok=True)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    with (directory / f"{worker}.txt").open("a", encoding="utf-8") as dump:
        dump.write(f"watchdog: {seconds:.0f} s for {request.node.nodeid}\n")
        dump.flush()
        faulthandler.dump_traceback_later(seconds, repeat=False, file=dump, exit=True)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
            dump.write(f"done: {request.node.nodeid}\n")
