"""Devices for the M6 end-to-end tests (fixture numbers, not physics claims).

The M4 171Yb+ chain (B along x, trap (3.0, 2.9, 1.0) MHz) with the beams a laboratory needs to run circuits: a GLOBAL
355 nm Raman pair along +-x for the entangling gates (a 60 um waist that both ions of a short chain see to 0.5 %, at a power
that gives a carrier Rabi frequency near 100 kHz), one INDIVIDUALLY ADDRESSED 355 nm pair per ion for the single-qubit gates
(a 2.5 um waist pointed at the ion, whose Gaussian tail on the neighbours 3.5 um away is the few-percent Rabi crosstalk of
Wright et al., Section 6.6), the resonant 369.5 nm light that cools, detects and pumps along an oblique path k = (1, 1, 1)/
sqrt 3 so that every mode projects on it (an unaddressed mode is uncoolable, Section 4.2) with its linear polarization at
Berkeland's magic angle to B (Section 8.1), and Crain's SNSPD detector. ``gate_drives`` and ``entangling_drives`` name
which pair plays which gate, because a device with several Raman pairs is ambiguous to the scheduler's inference.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from qutip_trap.api import Beam, Device
from qutip_trap.control.schedule import GateDrive
from qutip_trap.light.bloch import beam_for_transition
from qutip_trap.prep.recipe import PreparationRecipe, magic_angle_polarization, standard_recipe
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD
from qutip_trap.species import species
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.crystal import solve_crystal
from tests.m4_fixtures import ms_trap
from tests.readout_fixtures import snspd_detector

OBLIQUE = (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
DETECTION_WAIST_M = 30e-6
GLOBAL_WAIST_M = 60e-6
GLOBAL_POWER_W = 0.3
ADDRESS_WAIST_M = 2.5e-6
ADDRESS_POWER_W = 0.3 * (ADDRESS_WAIST_M / GLOBAL_WAIST_M) ** 2
"""The same on-axis intensity as the global pair, so every carrier Rabi frequency sits near 100 kHz."""


def oblique_detection_beam(
    s_o: float = 2.45, b_gauss: float = 5.0, waist_m: float = DETECTION_WAIST_M
) -> Beam:
    """The 369.5 nm detection beam along (1, 1, 1)/sqrt 3 at I/I_sat = s_o on axis (Crain's operating point), magic-angle
    linear polarization to B = x, wide enough that every ion of a short chain sees the same intensity to a percent."""
    yb = species("171Yb+")
    st = AtomicStructure(yb, b_gauss, (1.0, 0.0, 0.0))
    line = yb.transition("S1/2-P1/2")
    power = s_o * line.i_sat_w_m2 * math.pi * waist_m**2 / 2.0
    pol = magic_angle_polarization(OBLIQUE, (1.0, 0.0, 0.0))
    return beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=0 mF=0",
        0.0,
        OBLIQUE,
        (complex(pol[0]), complex(pol[1]), complex(pol[2])),
        power_w=power,
        waist_m=waist_m,
    )


def raman_pair(power_w: float, waist_m: float, pointing_m: tuple[float, float, float]) -> tuple[Beam, Beam]:
    """Counter-propagating 355 nm beams along +-x, lin-perp-lin (y, z) so that the clock transition is driven with B along x."""
    b1 = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), waist_m, power_w, pointing_m)
    b2 = Beam(355e-9, (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), waist_m, power_w, pointing_m)
    return b1, b2


@dataclasses.dataclass(frozen=True)
class CircuitFixture:
    device: Device
    gate_drives: dict[int, GateDrive]
    entangling_drives: dict[int, GateDrive]
    detection_beam: int

    @property
    def n_ions(self) -> int:
        return self.device.crystal.n_ions


def circuit_fixture(
    n_ions: int = 2,
    *,
    s_o: float = 2.45,
    leakage: dict[int, float] | None = None,
    recipe: PreparationRecipe | None = None,
    with_recipe: bool = True,
    address_waist_m: float = ADDRESS_WAIST_M,
    omega_hz: tuple[float, float, float] = (3.0e6, 2.9e6, 1.0e6),
    phase_continuous: bool = False,
) -> CircuitFixture:
    """``phase_continuous=False``: the tones of every gate are programmed from the gate's own start (an AWG per pulse), so the
    bichromatic beat note begins at the phase it was calibrated with; True keeps the tone oscillators running and exposes
    Roos's spin-axis tilt at the gate's start phase (Section 7.10, M6 finding)."""
    trap = ms_trap(omega_hz)
    yb = species("171Yb+")
    crystal = solve_crystal(trap, tuple([yb] * n_ions))
    beams: list[Beam] = list(raman_pair(GLOBAL_POWER_W, GLOBAL_WAIST_M, (0.0, 0.0, 0.0)))
    gate_drives: dict[int, GateDrive] = {}
    for i in range(n_ions):
        pos = tuple(float(v) for v in crystal.positions_m[i])
        power = ADDRESS_POWER_W * (address_waist_m / ADDRESS_WAIST_M) ** 2
        k = len(beams)
        beams.extend(raman_pair(power, address_waist_m, pos))  # type: ignore[arg-type]
        gate_drives[i] = GateDrive("raman", (k, k + 1))
    det_index = len(beams)
    beams.append(oblique_detection_beam(s_o, 5.0))
    from qutip_trap.api import Field
    from tests.fixtures import make_hardware, make_noise

    dev = Device(
        crystal=crystal,
        trap=trap,
        field=Field(5.0, (1.0, 0.0, 0.0), None),
        beams=tuple(beams),
        noise=make_noise(),
        detector=snspd_detector(leakage=leakage),
        hardware=dataclasses.replace(make_hardware(), phase_continuous=phase_continuous),
    )
    if recipe is not None:
        dev = dataclasses.replace(dev, preparation=recipe)
    elif with_recipe:
        dev = dataclasses.replace(dev, preparation=standard_recipe(dev, raman_pair=(0, 1)))
    entangling = {i: GateDrive("raman", (0, 1)) for i in range(n_ions)}
    return CircuitFixture(dev, gate_drives, entangling, det_index)


def crain_efficiency() -> float:
    return float(CRAIN_YB171_SNSPD.efficiency)


def bell_target() -> np.ndarray:
    """(|00> + |11>)/sqrt 2 in the qubit-0-least-significant order."""
    v = np.zeros(4, dtype=complex)
    v[0] = v[3] = 1.0 / math.sqrt(2.0)
    return v
