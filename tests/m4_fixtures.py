"""Two- and three-ion 171Yb+ devices for the M4 entangling-gate tests: global counter-propagating 355 nm Raman beams along x
(B along x, lin-perp-lin y/z polarizations, so the clock transition is driven), a wide waist so both ions see the same
intensity, and the calibration tables the scheduler reads. Fixture numbers, not physics claims; the trap frequencies follow the
realizable Section 11.1 fixture (x 3.0, y 2.9, z 1.0 MHz), whose two x modes are the COM at 3.000 MHz and the rocking mode at
2.828 MHz with Lamb-Dicke parameters 0.080 and -/+ 0.0824 (Section 9.13 row 18)."""

from __future__ import annotations

import dataclasses
import math

from qutip_trap.api import Beam, CalEntry, CalibrationTable, Device, Field, Trap, Waveform
from qutip_trap.control.schedule import GateDrive
from qutip_trap.control.shaping import GateModes, gate_modes
from qutip_trap.species import species
from qutip_trap.trap.crystal import solve_crystal
from tests.fixtures import make_calibration_table, make_detector, make_hardware, make_noise

X_MODES_TWO_IONS = (2, 3)
"""transverse_1 family of the two-ion crystal: index 2 the rocking mode (2.828 MHz), 3 the COM (3.000 MHz)."""
X_COM_TWO_IONS = 3


def ms_trap(omega_hz: tuple[float, float, float] = (3.0e6, 2.9e6, 1.0e6)) -> Trap:
    return Trap(
        omega_hz=omega_hz,
        axis_angle_rad=0.0,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )


def global_pair_along_x(power_w: float = 0.3, waist_m: float = 60e-6) -> tuple[Beam, Beam]:
    """Counter-propagating 355 nm beams along +-x pointed at the chain centre; |Delta k| = 2k along x. At 0.3 W in a 60 um waist
    the derived carrier Rabi frequency is 100.8 kHz (the 10 mW, 200 um pair of the first M4 fixtures gave 303 Hz while its tables
    claimed 100 kHz; since M8 plays the physical Rabi frequency of a requested one, the beams must deliver what the table says)."""
    b1 = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), waist_m, power_w, (0.0, 0.0, 0.0))
    b2 = Beam(355e-9, (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), waist_m, power_w, (0.0, 0.0, 0.0))
    return b1, b2


def chain_device(n_ions: int, omega_hz: tuple[float, float, float] = (3.0e6, 2.9e6, 1.0e6)) -> Device:
    trap = ms_trap(omega_hz)
    yb = species("171Yb+")
    crystal = solve_crystal(trap, tuple([yb] * n_ions))
    return Device(
        crystal=crystal,
        trap=trap,
        field=Field(5.0, (1.0, 0.0, 0.0), None),
        beams=global_pair_along_x(),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
    )


def two_ion_device() -> Device:
    return chain_device(2)


def three_ion_device() -> Device:
    return chain_device(3)


def raman_gate_drives(n_ions: int) -> dict[int, GateDrive]:
    return {i: GateDrive("raman", (0, 1)) for i in range(n_ions)}


def two_ion_modes(device: Device, nbar: dict[int, float] | None = None) -> GateModes:
    """The x-family modes the along-x pair couples to (the y and z modes have eta = 0 exactly)."""
    return gate_modes(device, (0, 1), (0, 1), nbar=nbar)


def derived_seeds(
    device: Device, drives: dict[int, GateDrive]
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """(carrier Rabi frequency, differential Stark shift) per (ion, table key beam) the device derives for ``drives``: the
    seeds a surrogate table carries, so that the played chain of M8 (requested -> physical) is the identity."""
    from qutip_trap.light.raman import derive_optical_drive, derive_raman_drive

    rabi: dict[tuple[int, int], float] = {}
    stark: dict[tuple[int, int], float] = {}
    for ion, spec in drives.items():
        if spec.kind == "raman":
            dd = derive_raman_drive(device, ion, (spec.beams[0], spec.beams[1]), scattering=False)
        elif spec.kind in ("optical_E1", "optical_E2"):
            dd = derive_optical_drive(device, ion, spec.beams[0], scattering=False)
        else:
            continue
        rabi[(ion, spec.table_key_beam)] = float(dd.carrier_rabi_hz)
        stark[(ion, spec.table_key_beam)] = float(dd.stark_shift_hz)
    return rabi, stark


def table_with_waveform(
    pair: tuple[int, int],
    waveform: Waveform,
    rabi_hz: dict[tuple[int, int], float] | None = None,
    *,
    device: Device | None = None,
    drives: dict[int, GateDrive] | None = None,
    stark_hz: dict[tuple[int, int], float] | None = None,
) -> CalibrationTable:
    """A table carrying ``waveform`` for ``pair`` and carrier Rabi entries: ``rabi_hz`` explicitly, or the derived values of
    ``drives`` on ``device`` (with the derived Stark shifts) when both are given."""
    table = make_calibration_table()
    if rabi_hz is None and device is not None and drives is not None:
        rabi_hz, derived_stark = derived_seeds(device, drives)
        stark_hz = derived_stark if stark_hz is None else stark_hz
    rabi = {
        k: CalEntry(v, 1.0, "calibrated", "rabi_scan", "conv.rabi_frequency", 0.0, 0)
        for k, v in (rabi_hz or {}).items()
    }
    stark = {
        k: CalEntry(v, 0.1, "calibrated", "stark_scan", "conv.two_photon_rabi", 0.0, 0)
        for k, v in (stark_hz or {}).items()
    }
    return dataclasses.replace(table, ms={pair: waveform}, rabi=rabi, stark=stark)


def bell_state_fidelity(rho, chi_rad: float = math.pi / 4.0) -> float:  # type: ignore[no-untyped-def]
    """Overlap with (|00> - i sin/cos ...): the ideal XX(chi)|00> = cos chi |00> - i sin chi |11>; a diagnostic for tests."""
    import numpy as np

    target = np.zeros(4, dtype=complex)
    target[0] = math.cos(chi_rad)
    target[3] = -1j * math.sin(chi_rad)
    mat = np.asarray(rho.full())
    return float(np.real(target.conj() @ mat @ target))
