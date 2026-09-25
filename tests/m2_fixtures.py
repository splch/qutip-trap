"""Single-ion and two-ion devices for the M2 tests: derived Raman drives on the 171Yb+ clock transition.

The clock transition |0,0> <-> |1,0> needs eps_1^* x eps_2 parallel to B (the vector light shift), so a counter-propagating
pair along x with lin-perp-lin polarizations (y, z) takes B along x, and the 90-degree pair (x-beam polarized y, y-beam
polarized x) takes B along z. Fixture powers and waists give a 30 kHz carrier, not a physics claim.
"""

from __future__ import annotations

import dataclasses

from qutip_trap.control.table import CalEntry
from qutip_trap.device.model import Device, Field
from qutip_trap.light.beams import Beam
from qutip_trap.species import species
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import RfDrive
from tests.fixtures import make_calibration_table, make_detector, make_hardware, make_noise


def explicit_trap(rf: RfDrive | None = None, stray: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Trap:
    return Trap(
        omega_hz=(3.0e6, 2.9e6, 1.0e6),
        axis_angle_rad=0.0,
        rf=rf,
        dc=None,
        geometry=None,
        stray_field_v_per_m=stray,
        shim_voltages_v={},
    )


def counter_propagating_pair(power_w: float = 10e-3, waist_m: float = 20e-6) -> tuple[Beam, Beam]:
    b1 = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), waist_m, power_w, (0.0, 0.0, 0.0))
    b2 = Beam(355e-9, (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), waist_m, power_w, (0.0, 0.0, 0.0))
    return b1, b2


def single_ion_raman_device(
    rf: RfDrive | None = None, stray: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Device:
    """One 171Yb+ ion, counter-propagating 355 nm pair along x, B along x: eta ~ 0.111 on the 3 MHz x mode (mode index 1)."""
    trap = explicit_trap(rf, stray)
    yb = species("171Yb+")
    return Device(
        crystal=solve_crystal(trap, (yb,)),
        trap=trap,
        field=Field(5.0, (1.0, 0.0, 0.0)),
        beams=counter_propagating_pair(),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
    )


def two_ion_raman_device(waist_m: float = 20e-6) -> Device:
    """Two 171Yb+ ions along z, the 90-degree pair (x-beam polarized y, y-beam polarized x) pointed at ion 0, B along z."""
    trap = explicit_trap()
    yb = species("171Yb+")
    crystal = solve_crystal(trap, (yb, yb))
    x0 = tuple(float(v) for v in crystal.positions_m[0])
    b1 = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), waist_m, 10e-3, x0)
    b2 = Beam(355e-9, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), waist_m, 10e-3, x0)
    return Device(
        crystal=crystal,
        trap=trap,
        field=Field(5.0, (0.0, 0.0, 1.0)),
        beams=(b1, b2),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
    )


def microwave_device() -> Device:
    """One 171Yb+ ion, no beams: single-qubit gates are microwave pulses (eta = 0)."""
    trap = explicit_trap()
    yb = species("171Yb+")
    return Device(
        crystal=solve_crystal(trap, (yb,)),
        trap=trap,
        field=Field(5.0, (0.0, 0.0, 1.0)),
        beams=(),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
    )


def table_with_rabi(entries: dict[tuple[int, int], float], status: str = "calibrated"):  # type: ignore[no-untyped-def]
    table = make_calibration_table()
    rabi = {
        k: CalEntry(v, 1.0, status, "rabi_scan", "conv.rabi_frequency", 0.0, 0) for k, v in entries.items()
    }  # type: ignore[arg-type]
    return dataclasses.replace(table, rabi=rabi)
