"""Fixtures for the M5 readout tests: 171Yb+ detection beams on the M4 chain device, and record models built from the
published apparatus presets (Section 8.4) so the discriminators can be exercised at the sources' operating points.

The chain device of ``tests/m4_fixtures.py`` has B along x; the detection beam propagates along y with its linear
polarization at the magic angle arccos(1/sqrt 3) to B (Section 8.1), which destabilizes the coherent dark states of the
F = 1 -> F' = 0 cycle. Fixture numbers, not physics claims.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import qutip as qt

from qutip_trap.device.model import Device, Field
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import beam_for_transition
from qutip_trap.readout.detection import Detector, RecordModel
from qutip_trap.readout.fluorescence import FluorescenceRates, ReadoutScheme, rates_from_detected
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD, MYERSON_CA40_PMT
from qutip_trap.species import species
from qutip_trap.species.polarization import linear_polarization
from qutip_trap.species.raman import AtomicStructure
from tests.m4_fixtures import chain_device

MAGIC_ANGLE_RAD = math.acos(1.0 / math.sqrt(3.0))
DETECTION_WAIST_M = 20e-6


def yb_detection_beam(s_o: float, b_gauss: float = 5.0, detuning_rad_s: float = 0.0) -> Beam:
    """A 369.5 nm beam along +y, linear polarization at the magic angle to B = x, with I/I_sat = s_o on axis."""
    yb = species("171Yb+")
    st = AtomicStructure(yb, b_gauss, (1.0, 0.0, 0.0))
    line = yb.transition("S1/2-P1/2")
    power = s_o * line.i_sat_w_m2 * math.pi * DETECTION_WAIST_M**2 / 2.0
    pol = tuple(linear_polarization((0.0, 1.0, 0.0), MAGIC_ANGLE_RAD, (1.0, 0.0, 0.0)))
    return beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=0 mF=0",
        detuning_rad_s,
        (0.0, 1.0, 0.0),
        pol,  # type: ignore[arg-type]
        power_w=power,
        waist_m=DETECTION_WAIST_M,
    )


def snspd_detector(window_s: float = 22e-6, leakage: dict[int, float] | None = None) -> Detector:
    """Crain's apparatus: eps_sys = 4.356 %, 4.2 cps background."""
    return Detector(
        kind="snspd",
        efficiency=CRAIN_YB171_SNSPD.efficiency,
        background_cps=CRAIN_YB171_SNSPD.background_per_s,
        psf_leakage=dict(leakage or {}),
        dead_time_s=None,
        afterpulse_prob=None,
        window_s=window_s,
        numerical_aperture=0.6,
    )


def yb_readout_device(n_ions: int = 2, s_o: float = 2.45, leakage: dict[int, float] | None = None) -> Device:
    """The M4 chain device plus a detection beam and Crain's SNSPD detector."""
    dev = chain_device(n_ions)
    return dataclasses.replace(
        dev,
        beams=tuple(dev.beams) + (yb_detection_beam(s_o, dev.field.B_gauss),),
        detector=snspd_detector(leakage=leakage),
    )


def crain_record_model(window_s: float = 22e-6, leakage: dict[int, float] | None = None) -> RecordModel:
    """Crain's measured triple as a record model: 472 kcps detected, R_d = 341 Hz, R_b = 16.4 Hz, 4.2 cps background."""
    return RecordModel.from_rates(CRAIN_YB171_SNSPD.rates(), snspd_detector(window_s, leakage))


def myerson_record_model(window_s: float = 420e-6) -> RecordModel:
    """Myerson's 40Ca+ apparatus: R_B = 55800 s^-1, R_D = 442 s^-1, shelf lifetime 1168 ms."""
    det = Detector(
        kind="pmt",
        efficiency=MYERSON_CA40_PMT.efficiency,
        background_cps=MYERSON_CA40_PMT.background_per_s,
        psf_leakage={},
        dead_time_s=None,
        afterpulse_prob=None,
        window_s=window_s,
    )
    return RecordModel.from_rates(MYERSON_CA40_PMT.rates(), det)


def two_state_rates(
    detected_bright_per_s: float, efficiency: float, r_d: float, r_b: float
) -> FluorescenceRates:
    return rates_from_detected(
        detected_bright_per_s, efficiency, dark_pumping_per_s=r_d, bright_pumping_per_s=r_b
    )


def register_space(n_ions: int) -> HilbertSpace:
    """An internal-only register space (no resolved modes): what the readout stage reads after the pulses."""
    return HilbertSpace(tuple([2] * n_ions), (), None, ())


def bell_state(space: HilbertSpace) -> qt.Qobj:
    zero = qt.tensor(*[qt.basis(2, 0) for _ in range(space.n_ions)])
    one = qt.tensor(*[qt.basis(2, 1) for _ in range(space.n_ions)])
    return (zero + one).unit()


def product_state(space: HilbertSpace, bits: tuple[int, ...]) -> qt.Qobj:
    return qt.tensor(*[qt.basis(2, b) for b in bits])


YB_DIRECT = ReadoutScheme.direct(1)
"""171Yb+: |1> = F = 1 bright, |0> = F = 0 dark."""
CA_OPTICAL = ReadoutScheme.shelving(1)
"""40Ca+ optical qubit: |1> = D5/2 is the shelf, |0> = S1/2 bright (polarity inverted)."""


def field_x(b_gauss: float = 5.0) -> Field:
    return Field(b_gauss, (1.0, 0.0, 0.0), None)


def bits_to_array(rows: list[tuple[int, ...]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.uint8)
