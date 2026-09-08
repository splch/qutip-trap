"""qutip-trap: a first-principles trapped-ion quantum computer simulator built on QuTiP.

The specification is ``PLAN.md`` at the repository root; the public surface is Appendix E, re-exported by
:mod:`qutip_trap.api`. Release 0.1.0 completes milestones M0 to M10 of PLAN.md Section 10: the atomic layer, the
trap and crystal, the one Hamiltonian builder, cooling and preparation, entangling gates, readout, end-to-end
circuits, noise channels, calibration emulation, the two scaling milestones and the benchmark emulation, with the
documentation under ``docs/``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
