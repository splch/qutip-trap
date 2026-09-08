"""Example devices (PLAN.md Section 3.2 ``device/presets.py``; milestone M10).

A preset is a complete ``Device`` plus the ``GateDrive`` maps the scheduler needs when a device carries several Raman pairs
(Section 7.3), so that the documentation examples and the benchmarks of ``qutip_trap.benchmarks`` run on a device without
importing the test tree. The numbers are ILLUSTRATIVE (a realizable laboratory configuration, not a published apparatus):

``yb171_chain(n_ions)`` is the 171Yb+ chain the validation suite runs on since milestone M6 (``tests/m6_fixtures.py``; a test
asserts the two build the same device, hash for hash): B = 5 G along x, secular frequencies (3.0, 2.9, 1.0) MHz, a GLOBAL
355 nm counter-propagating Raman pair along +-x (0.3 W in a 60 um waist, a carrier Rabi frequency near 100 kHz) for the
entangling gates, one INDIVIDUALLY ADDRESSED 355 nm pair per ion for the single-qubit gates (a 2.5 um waist pointed at the
ion, whose Gaussian tail on the neighbours 3.5 um away is the few-percent Rabi crosstalk of Wright et al. 2019, Section 6.6),
the resonant 369.5 nm cooling/detection/pumping light along the oblique path (1, 1, 1)/sqrt 3 so that every mode projects on
it (an unaddressed mode is uncoolable, Section 4.2) with its linear polarization at Berkeland's magic angle to B (Section
8.1), Crain et al. 2019's SNSPD detection chain (eps_sys = 4.356 %, 4.2 cps background), a QUIET noise model (no heating, no
drift: pass ``noise=`` for a noisy machine) and near-ideal control electronics (32-bit phase and 24-bit amplitude words, no
modulator rise time, 1 us dead time; pass ``hardware=`` for the Section 7.10 chain), and the standard preparation recipe of
Section 4.2.6 (Doppler cooling at the detuning that minimizes the gate modes' occupation, pulsed Raman sideband cooling of
the coupled modes, the optical pump on F = 1 -> F' = 1) derived from these beams.

Everything downstream (Lamb-Dicke parameters, Rabi frequencies, Stark shifts, scattering rates, mode structure, pulse
parameters) is derived from these inputs by the package (Section 3.1); nothing here is a physics claim.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from qutip_trap.control.hardware import HardwareChain
from qutip_trap.control.schedule import GateDrive
from qutip_trap.device.model import Device, Field
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import beam_for_transition
from qutip_trap.noise.model import NoiseModel
from qutip_trap.noise.spectra import Drift, NoiseSpectrum
from qutip_trap.prep.recipe import PreparationRecipe, magic_angle_polarization, standard_recipe
from qutip_trap.readout.detection import Detector
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD
from qutip_trap.species import species
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.trap.model import Trap

OBLIQUE = (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
DETECTION_WAIST_M = 30e-6
GLOBAL_WAIST_M = 60e-6
GLOBAL_POWER_W = 0.3
ADDRESS_WAIST_M = 2.5e-6
ADDRESS_POWER_W = 0.3 * (ADDRESS_WAIST_M / GLOBAL_WAIST_M) ** 2
"""The same on-axis intensity as the global pair, so every carrier Rabi frequency sits near 100 kHz."""
TRAP_HZ = (3.0e6, 2.9e6, 1.0e6)
FIELD_GAUSS = 5.0


@dataclasses.dataclass(frozen=True)
class DevicePreset:
    """A device with the drive maps its scheduler needs (Section 7.3) and the index of its detection beam."""

    name: str
    device: Device
    gate_drives: dict[int, GateDrive]
    """Which beams play the single-qubit gates of each ion."""
    entangling_drives: dict[int, GateDrive]
    """Which beams play the entangling gates (the global pair)."""
    detection_beam: int
    notes: tuple[str, ...] = ()

    @property
    def n_ions(self) -> int:
        return self.device.crystal.n_ions

    def run_kwargs(self) -> dict[str, object]:
        """The keyword arguments ``run``, ``calibrate`` and the benchmarks take for this device's drive maps."""
        return {"gate_drives": self.gate_drives, "entangling_drives": self.entangling_drives}


def quiet_noise_model() -> NoiseModel:
    """A noise model with no heating, no field or drift content and no collisions (every Drift of zero rms)."""
    omega = np.linspace(-2.0 * math.pi * 1e7, 2.0 * math.pi * 1e7, 5)
    flat = NoiseSpectrum(omega_rad_s=omega, S=np.zeros(5), unit="(V/m)^2/(rad/s)")
    quiet = Drift(rms=0.0, tau_s=1.0, servo_bandwidth_hz=None)
    return NoiseModel(
        S_E=flat,
        correlation_length_m=0.0,
        S_B=None,
        mains=None,
        laser_phase=None,
        laser_intensity=None,
        rf_amplitude_noise=None,
        rf_phase_noise=None,
        rf_amplitude_drift=quiet,
        mode_drift_differential=quiet,
        rabi_drift=quiet,
        beam_phase_drift=quiet,
        field_drift=quiet,
        stray_field_drift=quiet,
        pointing_drift=quiet,
        rabi_amplitude=None,
        collisions=None,
    )


def ideal_hardware(*, phase_continuous: bool = False, dead_time_s: float = 1e-6) -> HardwareChain:
    """Near-ideal control electronics (Section 7.10: a device without these parameters gets ideal electronics and says so):
    32-bit phase and 24-bit amplitude words, no modulator rise time, infinite amplifier bandwidth. ``phase_continuous=False``
    programs every gate's tones from the gate's own start, so the bichromatic beat note begins at its calibrated phase."""
    return HardwareChain(
        dds_phase_bits=32,
        dds_amplitude_bits=24,
        aom_rise_s=0.0,
        amplifier_bandwidth_hz=float("inf"),
        dead_time_s=dead_time_s,
        phase_continuous=phase_continuous,
    )


def crain_snspd_detector(window_s: float = 22e-6) -> Detector:
    """Crain et al. 2019's detection chain: eps_sys = 4.356 %, 4.2 cps background, a 0.6 NA objective."""
    return Detector(
        kind="snspd",
        efficiency=CRAIN_YB171_SNSPD.efficiency,
        background_cps=CRAIN_YB171_SNSPD.background_per_s,
        psf_leakage={},
        dead_time_s=None,
        afterpulse_prob=None,
        window_s=window_s,
        numerical_aperture=0.6,
    )


def secular_trap(omega_hz: tuple[float, float, float] = TRAP_HZ) -> Trap:
    """A trap given by its secular frequencies (x, y, z) with the principal axes along the laboratory axes."""
    return Trap(
        omega_hz=omega_hz,
        axis_angle_rad=0.0,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )


def oblique_detection_beam(
    s_o: float = 2.45, b_gauss: float = FIELD_GAUSS, waist_m: float = DETECTION_WAIST_M
) -> Beam:
    """The 369.5 nm resonant beam along (1, 1, 1)/sqrt 3 at I/I_sat = s_o on axis (Crain's operating point), linear polarization
    at the magic angle to B = x, wide enough that every ion of a short chain sees the same intensity to a percent."""
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


def raman_pair_along_x(
    power_w: float, waist_m: float, pointing_m: tuple[float, float, float]
) -> tuple[Beam, Beam]:
    """Counter-propagating 355 nm beams along +-x, lin-perp-lin (y, z) so that the clock transition is driven with B along x;
    |Delta k| = 2k along x."""
    b1 = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), waist_m, power_w, pointing_m)
    b2 = Beam(355e-9, (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), waist_m, power_w, pointing_m)
    return b1, b2


def yb171_chain(
    n_ions: int = 2,
    *,
    s_o: float = 2.45,
    address_waist_m: float = ADDRESS_WAIST_M,
    omega_hz: tuple[float, float, float] = TRAP_HZ,
    noise: NoiseModel | None = None,
    hardware: HardwareChain | None = None,
    detector: Detector | None = None,
    recipe: PreparationRecipe | None = None,
    phase_continuous: bool = False,
) -> DevicePreset:
    """The example 171Yb+ chain (module docstring): ``n_ions`` ions, a global Raman pair, one addressing pair per ion, the oblique
    detection beam, Crain's detector, a quiet noise model and near-ideal electronics unless overridden."""
    if n_ions < 1:
        raise ValueError("a chain has at least one ion")
    trap = secular_trap(omega_hz)
    yb = species("171Yb+")
    crystal = solve_crystal(trap, tuple([yb] * n_ions))
    beams: list[Beam] = list(raman_pair_along_x(GLOBAL_POWER_W, GLOBAL_WAIST_M, (0.0, 0.0, 0.0)))
    gate_drives: dict[int, GateDrive] = {}
    for i in range(n_ions):
        pos = tuple(float(v) for v in crystal.positions_m[i])
        power = ADDRESS_POWER_W * (address_waist_m / ADDRESS_WAIST_M) ** 2
        k = len(beams)
        beams.extend(raman_pair_along_x(power, address_waist_m, (pos[0], pos[1], pos[2])))
        gate_drives[i] = GateDrive("raman", (k, k + 1))
    det_index = len(beams)
    beams.append(oblique_detection_beam(s_o, FIELD_GAUSS))
    dev = Device(
        crystal=crystal,
        trap=trap,
        field=Field(FIELD_GAUSS, (1.0, 0.0, 0.0), None),
        beams=tuple(beams),
        noise=noise if noise is not None else quiet_noise_model(),
        detector=detector if detector is not None else crain_snspd_detector(),
        hardware=hardware if hardware is not None else ideal_hardware(phase_continuous=phase_continuous),
    )
    dev = dataclasses.replace(
        dev, preparation=recipe if recipe is not None else standard_recipe(dev, raman_pair=(0, 1))
    )
    entangling = {i: GateDrive("raman", (0, 1)) for i in range(n_ions)}
    return DevicePreset(
        name=f"171Yb+ chain of {n_ions} (example device, illustrative numbers)",
        device=dev,
        gate_drives=gate_drives,
        entangling_drives=entangling,
        detection_beam=det_index,
        notes=(
            "example device: a realizable configuration, not a published apparatus",
            "quiet noise model and near-ideal electronics unless noise= and hardware= are given",
        ),
    )


__all__ = [
    "ADDRESS_POWER_W",
    "ADDRESS_WAIST_M",
    "DETECTION_WAIST_M",
    "FIELD_GAUSS",
    "GLOBAL_POWER_W",
    "GLOBAL_WAIST_M",
    "OBLIQUE",
    "TRAP_HZ",
    "DevicePreset",
    "crain_snspd_detector",
    "ideal_hardware",
    "oblique_detection_beam",
    "quiet_noise_model",
    "raman_pair_along_x",
    "secular_trap",
    "yb171_chain",
]
