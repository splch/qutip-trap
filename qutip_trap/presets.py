"""The example devices as machines: ``yb171_chain(n)`` and ``ca40_optical(n)`` return a ``Machine`` on the device
``qutip_trap.device.presets`` builds (its knobs pass through), with the closed-form calibration cached per device, the
default option objects and the AUTO level. The numbers are illustrative, not a published apparatus.
"""

from __future__ import annotations

from typing import Any

from qutip_trap.device import presets as device_presets
from qutip_trap.machine import Machine


def yb171_chain(n_ions: int = 2, **knobs: Any) -> Machine:
    """The example 171Yb+ chain: a global 355 nm Raman pair for the entangling gates, one addressing pair per ion, the
    oblique 369.5 nm detection beam, Crain's SNSPD detector, a quiet noise model and near-ideal electronics unless
    overridden (``device.presets.yb171_chain``)."""
    return device_presets.yb171_chain(n_ions, **knobs).machine()


def ca40_optical(n_ions: int = 1, **knobs: Any) -> Machine:
    """The example 40Ca+ optical-qubit chain: the 729 nm quadrupole beam for the single-qubit gates, the 397 and 866 nm
    cooling and detection light, Myerson's PMT chain, no entangling drive (``device.presets.ca40_optical``)."""
    return device_presets.ca40_optical(n_ions, **knobs).machine()
