"""Example devices: a 171Yb+ chain (``yb171_chain``) and a 40Ca+ optical-qubit chain (``ca40_optical``). The numbers are
illustrative: a realizable laboratory configuration, not a published apparatus; everything downstream is derived."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap._compat import deprecated
from qutip_trap.control.hardware import HardwareChain
from qutip_trap.control.schedule import GateDrive
from qutip_trap.device.model import BeamRoles, Device, Field
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import beam_for_transition
from qutip_trap.noise.model import NoiseModel
from qutip_trap.prep.recipe import PreparationRecipe, magic_angle_polarization, standard_recipe
from qutip_trap.readout.detection import Detector
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD, MYERSON_CA40_PMT
from qutip_trap.species import species
from qutip_trap.species.raman import structure_at
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.trap.model import Trap

if TYPE_CHECKING:
    from qutip_trap.machine import Machine

OBLIQUE = (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
DETECTION_WAIST_M = 30e-6
GLOBAL_WAIST_M = 60e-6
GLOBAL_POWER_W = 0.3
ADDRESS_WAIST_M = 2.5e-6
ADDRESS_POWER_W = 0.3 * (ADDRESS_WAIST_M / GLOBAL_WAIST_M) ** 2
"""The same on-axis intensity as the global pair."""
TRAP_HZ = (3.0e6, 2.9e6, 1.0e6)
FIELD_GAUSS = 5.0


@dataclasses.dataclass(frozen=True)
class DevicePreset:
    """An example device with its drive maps and detection beam index, which restate ``device.roles``."""

    name: str
    device: Device
    gate_drives: dict[int, GateDrive]
    entangling_drives: dict[int, GateDrive]
    detection_beam: int
    notes: tuple[str, ...] = ()

    @property
    def n_ions(self) -> int:
        return self.device.crystal.n_ions

    def machine(self) -> Machine:
        """A ``Machine`` on this preset's device."""
        from qutip_trap.machine import Machine

        return Machine(self.device, name=self.name)


@deprecated(deadline="v0.5", fix="Call NoiseModel() instead; every default of the model means off.")
def quiet_noise_model() -> NoiseModel:
    """A noise model with every channel off: ``NoiseModel()``."""
    return NoiseModel()


def ideal_hardware(*, phase_continuous: bool = False, dead_time_s: float = 1e-6) -> HardwareChain:
    """Near-ideal control electronics. ``phase_continuous=False`` starts every gate's tones at the gate's own start, so the
    bichromatic beat note begins at its calibrated phase."""
    return HardwareChain(
        dds_phase_bits=32,
        dds_amplitude_bits=24,
        aom_rise_s=0.0,
        amplifier_bandwidth_hz=float("inf"),
        dead_time_s=dead_time_s,
        phase_continuous=phase_continuous,
    )


def crain_snspd_detector(window_s: float = 22e-6) -> Detector:
    """Crain et al. 2019's SNSPD detection chain."""
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
    """The resonant 369.5 nm beam along (1, 1, 1)/sqrt 3, so every mode projects on it, at I/I_sat = ``s_o`` on axis
    (default: Crain's operating point), linearly polarized at the magic angle to B = x."""
    yb = species("171Yb+")
    st = structure_at(yb, b_gauss, (1.0, 0.0, 0.0))
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
    """The example 171Yb+ chain: B = 5 G along x, a global 355 nm Raman pair along +-x for the entangling gates, one
    individually addressed pair per ion for the single-qubit gates, the oblique detection beam, Crain's detector, the
    standard preparation recipe, and a quiet noise model and near-ideal electronics unless overridden."""
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
    entangling = {i: GateDrive("raman", (0, 1)) for i in range(n_ions)}
    dev = Device(
        crystal=crystal,
        trap=trap,
        field=Field(FIELD_GAUSS, (1.0, 0.0, 0.0), None),
        beams=tuple(beams),
        noise=noise if noise is not None else NoiseModel(),
        detector=detector if detector is not None else crain_snspd_detector(),
        hardware=hardware if hardware is not None else ideal_hardware(phase_continuous=phase_continuous),
        roles=BeamRoles(gate=gate_drives, entangling=entangling, detection=det_index),
    )
    dev = dataclasses.replace(
        dev, preparation=recipe if recipe is not None else standard_recipe(dev, raman_pair=(0, 1))
    )
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


# ---- 40Ca+ optical qubit --------------------------------------------------------------------------------------------------

CA40_TRAP_HZ = (3.0e6, 2.9e6, 2.0e6)
"""The 40Ca+ preset's secular frequencies. The axial mode is 2.0 MHz, not the 171Yb+ preset's 1.0 MHz, because this device
has no sideband cooling: at 1.0 MHz the Doppler-limited eta^2 (2 nbar + 1) exceeds the 0.25 Lamb-Dicke validity threshold
of the level-A rate model (it scales roughly as 1/omega^2)."""
CA40_729_WAIST_M = 200e-6
CA40_729_POWER_W = 5e-3
CA40_729_K = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
"""The 729 nm beam's direction, in the xy plane at 45 degrees to B = x."""
CA40_729_POLARIZATION = (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
"""At 135 degrees, so the Delta m = 0 geometric factor of the E2 tensor, which needs both the polarization and the
wavevector to have a component along B, is non-zero (k = y with pol = x would derive exactly zero)."""
CA40_397_WAIST_M = 30e-6
CA40_866_WAIST_M = 30e-6
CA40_854_WAIST_M = 30e-6


CA40_DETECTION_S_397 = 2.0
CA40_DETECTION_S_866 = 10.0
"""The detection cycle's 397 and 866 nm saturation parameters. The derived bright scatter rate peaks near these values and
falls at higher intensity, because the magic-angle linear polarization of the 397/866 cycle traps population in coherent
dark states; it sits well below Myerson et al. 2008's, which the preset compensates with a 2 ms detection window."""
CA40_DETECTION_WINDOW_S = 2e-3


def myerson_ca40_pmt_detector(window_s: float = CA40_DETECTION_WINDOW_S) -> Detector:
    """Myerson et al. 2008's 40Ca+ PMT shelving-readout chain."""
    return Detector(
        kind="pmt",
        efficiency=MYERSON_CA40_PMT.efficiency,
        background_cps=MYERSON_CA40_PMT.background_per_s,
        psf_leakage={},
        dead_time_s=None,
        afterpulse_prob=None,
        window_s=window_s,
    )


def ca40_optical_recipe(
    device: Device,
    *,
    doppler_duration_s: float = 2e-3,
    pump_duration_s: float = 100e-6,
    s_397: float = 0.5,
    s_866: float = 3.0,
    s_pump_397: float = 0.05,
    s_pump_866: float = 10.0,
) -> PreparationRecipe:
    """The 40Ca+ optical qubit's preparation (``standard_recipe`` covers only F = 0 hyperfine qubits): 397 nm Doppler
    cooling with the 866 nm repump, then 397 nm sigma- pumping into S1/2 mJ = -1/2, weak against a strong repump because
    the population left in the long-lived D3/2 level is the preparation error. No sideband cooling (this device would cool
    on the unscheduled 729 nm line), so the Doppler occupations stand."""
    ca = species("40Ca+")
    if {sp.name for sp in device.crystal.species} != {"40Ca+"}:
        raise ValueError("ca40_optical_recipe is the 40Ca+ optical-qubit recipe")
    st = structure_at(ca, device.field.B_gauss, device.field.direction)
    line397 = ca.transition("S1/2-P1/2")
    line866 = ca.transition("D3/2-P1/2")
    gamma = line397.gamma_rad_s
    b_hat = tuple(float(x) for x in device.field.direction)
    pol_cool = magic_angle_polarization(OBLIQUE, b_hat)
    power397 = s_397 * line397.i_sat_w_m2 * math.pi * CA40_397_WAIST_M**2 / 2.0
    power866 = s_866 * line866.i_sat_w_m2 * math.pi * CA40_866_WAIST_M**2 / 2.0
    pump_power397 = s_pump_397 * line397.i_sat_w_m2 * math.pi * CA40_397_WAIST_M**2 / 2.0
    pump_power866 = s_pump_866 * line866.i_sat_w_m2 * math.pi * CA40_866_WAIST_M**2 / 2.0
    cool397 = beam_for_transition(
        st,
        "S1/2 mJ=-1/2",
        "P1/2 mJ=-1/2",
        -0.5 * gamma,
        OBLIQUE,
        (complex(pol_cool[0]), complex(pol_cool[1]), complex(pol_cool[2])),
        power_w=power397,
        waist_m=CA40_397_WAIST_M,
    )
    repump866 = beam_for_transition(
        st,
        "D3/2 mJ=-1/2",
        "P1/2 mJ=-1/2",
        0.0,
        OBLIQUE,
        (complex(pol_cool[0]), complex(pol_cool[1]), complex(pol_cool[2])),
        power_w=power866,
        waist_m=CA40_866_WAIST_M,
    )
    # sigma- about B: the circular basis vector orthogonal to b_hat, so the beam propagates along B
    e1, e2 = _orthonormal_to(b_hat)  # type: ignore[arg-type]
    sigma_minus = tuple(
        complex(e1[j]) / math.sqrt(2.0) - 1j * complex(e2[j]) / math.sqrt(2.0) for j in range(3)
    )
    pump397 = beam_for_transition(
        st,
        "S1/2 mJ=+1/2",
        "P1/2 mJ=-1/2",
        0.0,
        b_hat,  # type: ignore[arg-type]
        (sigma_minus[0], sigma_minus[1], sigma_minus[2]),
        power_w=pump_power397,
        waist_m=CA40_397_WAIST_M,
    )
    pump866 = beam_for_transition(
        st,
        "D3/2 mJ=-1/2",
        "P1/2 mJ=-1/2",
        0.0,
        OBLIQUE,
        (complex(pol_cool[0]), complex(pol_cool[1]), complex(pol_cool[2])),
        power_w=pump_power866,
        waist_m=CA40_866_WAIST_M,
    )
    return PreparationRecipe(
        doppler_beams=(cool397, repump866),
        doppler_duration_s=doppler_duration_s,
        pump_beams=(pump397, pump866),
        pump_duration_s=pump_duration_s,
        sideband=None,
        pump_target=(ca.qubit[0],),
        levels=("S1/2", "P1/2", "D3/2"),
        notes=(
            f"40Ca+ optical qubit: 397 nm Doppler cooling at -Gamma/2 (I/I_sat = {s_397:g}) along the oblique path with the "
            f"866 nm D3/2 repump on resonance (I/I_sat = {s_866:g})",
            f"pump into {ca.qubit[0]!r}: 397 nm sigma- light along B at I/I_sat = {s_pump_397:g} with the 866 nm repump at "
            f"{s_pump_866:g} (the I = 0 analogue of the 171Yb+ F = 1 -> F' = 1 pump); the weak pump against the strong "
            "repump keeps the metastable D3/2 population, which is what the preparation error is, below 1e-4",
            "no sideband-cooling stage: Section 4.2.2's stage is pulsed RAMAN cooling on a beam pair, and this device would "
            "cool on the 729 nm quadrupole line, which the first release does not schedule; the Doppler occupations stand",
            "the 854 nm D5/2-P3/2 beam is the shelf reset of the readout stage, not part of this recipe",
        ),
    )


def _orthonormal_to(direction: tuple[float, float, float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Two unit vectors completing ``direction`` to a right-handed orthonormal triad."""
    n = np.asarray(direction, dtype=float)
    n = n / float(np.linalg.norm(n))
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - float(np.dot(seed, n)) * n
    e1 = e1 / float(np.linalg.norm(e1))
    e2 = np.cross(n, e1)
    return tuple(float(x) for x in e1), tuple(float(x) for x in e2)


def ca40_optical(
    n_ions: int = 1,
    *,
    omega_hz: tuple[float, float, float] = CA40_TRAP_HZ,
    noise: NoiseModel | None = None,
    hardware: HardwareChain | None = None,
    detector: Detector | None = None,
    recipe: PreparationRecipe | None = None,
    phase_continuous: bool = False,
    reset_beam: bool = False,
) -> DevicePreset:
    """An example 40Ca+ optical-qubit chain (S1/2 mJ = -1/2 <-> D5/2 mJ = -1/2 at 729 nm): one global 729 nm beam for the
    single-qubit gates, 397/866 nm light for cooling and shelving detection, Myerson's PMT chain, ``ca40_optical_recipe``,
    and no entangling drive (the scheduler refuses two-qubit gates). ``reset_beam=True`` adds the 854 nm shelf-reset beam,
    off by default because a device's beams are always on and an ungated 854 nm beam depumps the shelf during detection."""
    if n_ions < 1:
        raise ValueError("a chain has at least one ion")
    trap = secular_trap(omega_hz)
    ca = species("40Ca+")
    crystal = solve_crystal(trap, tuple([ca] * n_ions))
    st = structure_at(ca, FIELD_GAUSS, (1.0, 0.0, 0.0))
    line397 = ca.transition("S1/2-P1/2")
    line866 = ca.transition("D3/2-P1/2")
    line854 = ca.transition("D5/2-P3/2")
    pol = magic_angle_polarization(OBLIQUE, (1.0, 0.0, 0.0))
    pol_c = (complex(pol[0]), complex(pol[1]), complex(pol[2]))
    b729 = Beam(
        729.348e-9,
        CA40_729_K,
        CA40_729_POLARIZATION,
        CA40_729_WAIST_M,
        CA40_729_POWER_W,
        (0.0, 0.0, 0.0),
    )
    b397 = beam_for_transition(
        st,
        "S1/2 mJ=-1/2",
        "P1/2 mJ=-1/2",
        -0.5 * line397.gamma_rad_s,
        OBLIQUE,
        pol_c,
        power_w=CA40_DETECTION_S_397 * line397.i_sat_w_m2 * math.pi * CA40_397_WAIST_M**2 / 2.0,
        waist_m=CA40_397_WAIST_M,
    )
    b866 = beam_for_transition(
        st,
        "D3/2 mJ=-1/2",
        "P1/2 mJ=-1/2",
        0.0,
        OBLIQUE,
        pol_c,
        power_w=CA40_DETECTION_S_866 * line866.i_sat_w_m2 * math.pi * CA40_866_WAIST_M**2 / 2.0,
        waist_m=CA40_866_WAIST_M,
    )
    beams: list[Beam] = [b729, b397, b866]
    if reset_beam:
        beams.append(
            beam_for_transition(
                st,
                "D5/2 mJ=-1/2",
                "P3/2 mJ=-1/2",
                0.0,
                OBLIQUE,
                pol_c,
                power_w=3.0 * line854.i_sat_w_m2 * math.pi * CA40_854_WAIST_M**2 / 2.0,
                waist_m=CA40_854_WAIST_M,
            )
        )
    gate = {i: GateDrive("optical_E2", (0,)) for i in range(n_ions)}
    dev = Device(
        crystal=crystal,
        trap=trap,
        field=Field(FIELD_GAUSS, (1.0, 0.0, 0.0), None),
        beams=tuple(beams),
        noise=noise if noise is not None else NoiseModel(),
        detector=detector if detector is not None else myerson_ca40_pmt_detector(),
        hardware=hardware if hardware is not None else ideal_hardware(phase_continuous=phase_continuous),
        roles=BeamRoles(gate=gate, entangling={}, detection=1),
    )
    dev = dataclasses.replace(dev, preparation=recipe if recipe is not None else ca40_optical_recipe(dev))
    return DevicePreset(
        name=f"40Ca+ optical-qubit chain of {n_ions} (example device, illustrative numbers)",
        device=dev,
        gate_drives=gate,
        entangling_drives={},
        detection_beam=1,
        notes=(
            "example device: a realizable configuration, not a published apparatus",
            "no entangling drive: the Section 4.4.4 light-shift force needs a far-detuned 398.5 nm pair this preset omits",
            "no sideband cooling: 729 nm resolved-sideband cooling is not scheduled in the first release",
            "detection is Myerson et al. 2008's shelving readout (eps_sys = 0.19 %, 442 cps, 420 us)",
            f"854 nm shelf-reset beam {'PRESENT (it depumps the shelf during detection: no per-stage beam gating)' if reset_beam else 'omitted (reset_beam=True adds it)'}",
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
    "CA40_729_K",
    "CA40_729_POLARIZATION",
    "CA40_729_POWER_W",
    "CA40_729_WAIST_M",
    "CA40_DETECTION_S_397",
    "CA40_DETECTION_S_866",
    "CA40_DETECTION_WINDOW_S",
    "CA40_TRAP_HZ",
    "DevicePreset",
    "ca40_optical",
    "ca40_optical_recipe",
    "crain_snspd_detector",
    "myerson_ca40_pmt_detector",
    "ideal_hardware",
    "oblique_detection_beam",
    "quiet_noise_model",
    "raman_pair_along_x",
    "secular_trap",
    "yb171_chain",
]
