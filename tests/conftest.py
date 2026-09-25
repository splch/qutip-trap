"""Under pytest-xdist each worker caps the engine's default worker count at one (``QUTIP_TRAP_MAX_WORKERS``), so no test
forks a trajectory pool inside a worker unless it asks for ``workers=``."""

import os

if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ.setdefault("QUTIP_TRAP_MAX_WORKERS", "1")
