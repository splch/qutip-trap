"""Devices for the M9a scaling tests (fixture numbers, not physics claims).

``tilted_pair_device``: the two-ion 171Yb+ chain of the M4 fixtures with the global 355 nm Raman pair rotated by ``theta`` in the
xy-plane, so that Delta k = 2k (cos theta, sin theta, 0) couples the drive weakly to the y (transverse_2) modes: eta_y = eta_x
tan theta, about 0.008 at 6 degrees against 0.08 on the x modes. A symmetric single-mode pulse on the x-COM then leaves the
y-COM (2.9 MHz, 120 kHz below the tone) as a genuine frozen spectator of Section 5.2, |alpha|^2 (2 nbar + 1) ~ 1e-4 and
|chi| ~ 1e-3 rad, the case the frozen-against-joint comparison of Section 9.9 and the ENR option of Section 11.3 need.

Both purposes are now exercised: the frozen-spectator comparison in ``tests/test_scaling_modes.py`` and the ENR option end to
end (``run(..., enr_group=(Y_MODES_TWO_IONS, N_exc))``) in ``tests/test_m9_scaling.py``. M9a shipped the ENR half unused,
which is why nothing caught the branch enumeration above the cap or the ENR boundary trip (M9a audit D5).
"""

from __future__ import annotations

import math

from qutip_trap.api import Beam, Device, Field
from qutip_trap.species import species
from qutip_trap.trap.crystal import solve_crystal
from tests.fixtures import make_detector, make_hardware, make_noise
from tests.m4_fixtures import ms_trap

Y_MODES_TWO_IONS = (4, 5)
"""transverse_2 family of the two-ion crystal: index 4 the rocking mode (2.720 MHz), 5 the COM (2.900 MHz)."""


def tilted_pair(theta_rad: float, power_w: float = 0.3, waist_m: float = 60e-6) -> tuple[Beam, Beam]:
    """Counter-propagating 355 nm beams along +-(cos theta, sin theta, 0), polarizations perpendicular to their k and to each
    other (the in-plane transverse direction and z), B along x: the clock transition is driven with a y-component of Delta k."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    b1 = Beam(355e-9, (c, s, 0.0), (-s, c, 0.0), waist_m, power_w, (0.0, 0.0, 0.0))
    b2 = Beam(355e-9, (-c, -s, 0.0), (0.0, 0.0, 1.0), waist_m, power_w, (0.0, 0.0, 0.0))
    return b1, b2


def tilted_pair_device(theta_rad: float = math.radians(6.0), n_ions: int = 2) -> Device:
    trap = ms_trap((3.0e6, 2.9e6, 1.0e6))
    yb = species("171Yb+")
    crystal = solve_crystal(trap, tuple([yb] * n_ions))
    return Device(
        crystal=crystal,
        trap=trap,
        field=Field(5.0, (1.0, 0.0, 0.0), None),
        beams=tilted_pair(theta_rad),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
    )
