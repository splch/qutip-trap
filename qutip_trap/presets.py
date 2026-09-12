"""The example devices as machines (docs/api_implementation_plan.md 1.6; 0.2.0): ``yb171_chain(n)`` and ``ca40_optical(n)``
return a ``Machine`` on the device ``qutip_trap.device.presets`` builds, with its beam roles, the closed-form calibration
cached per device, the default option objects and the AUTO level; every knob of the device presets (``s_o``,
``address_waist_m``, ``omega_hz``, ``noise``, ``hardware``, ``detector``, ``recipe``, ``phase_continuous``, ``reset_beam``)
passes through. ``DevicePreset.machine()`` is the bridge for code that holds a preset. The numbers are ILLUSTRATIVE (a
realizable laboratory configuration, not a published apparatus), as the device presets' docstrings say.
"""

from __future__ import annotations

from typing import Any

from qutip_trap.device import presets as device_presets
from qutip_trap.device.presets import DevicePreset
from qutip_trap.machine import Machine


def yb171_chain(n_ions: int = 2, **knobs: Any) -> Machine:
    """The example 171Yb+ chain as a ``Machine``: a global 355 nm Raman pair for the entangling gates, one addressing pair per
    ion, the oblique 369.5 nm detection beam, Crain's SNSPD detector, a quiet noise model and near-ideal electronics unless
    overridden (``device.presets.yb171_chain`` for the knobs and the numbers)."""
    return device_presets.yb171_chain(n_ions, **knobs).machine()


def ca40_optical(n_ions: int = 1, **knobs: Any) -> Machine:
    """The example 40Ca+ optical-qubit chain as a ``Machine``: the 729 nm quadrupole beam for the single-qubit gates, the 397
    and 866 nm cooling and detection light, Myerson's PMT chain, no entangling drive (``device.presets.ca40_optical``)."""
    return device_presets.ca40_optical(n_ions, **knobs).machine()


__all__ = ["DevicePreset", "ca40_optical", "yb171_chain"]
