"""Suite-wide configuration.

Under pytest-xdist every worker is already one of several processes, so the engine's default trajectory map must not fork a
pool of its own inside each of them (nested pools: N workers x every CPU, the oversubscription and the forked copies of a
multi-gigabyte pytest process that ``dynamics/parallel.memory_worker_cap`` guards against). ``QUTIP_TRAP_MAX_WORKERS`` caps
the DEFAULT worker count only; a test that asks for ``workers=`` explicitly still gets its pool (``tests/test_parallel.py``).
"""

from __future__ import annotations

import os

if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ.setdefault("QUTIP_TRAP_MAX_WORKERS", "1")
