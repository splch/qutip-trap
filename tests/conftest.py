"""Under pytest-xdist every worker is already one of several processes, so the engine's default trajectory map must not
fork a pool of its own inside each of them. ``QUTIP_TRAP_MAX_WORKERS`` caps the default worker count only; a test that asks
for ``workers=`` explicitly still gets its pool."""

import os

if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ.setdefault("QUTIP_TRAP_MAX_WORKERS", "1")
