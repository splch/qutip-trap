"""Example devices (PLAN.md Section 3.2 ``device/presets.py``; milestone M10).

A preset is a complete ``Device`` plus the ``GateDrive`` maps the scheduler needs when a device carries several Raman pairs
(Section 7.3), so that the documentation examples and the benchmarks of ``qutip_trap.benchmarks`` run on a device without
importing the test tree; since 0.2.0 the device itself carries those maps as ``Device.roles`` (docs/api_implementation_plan.md
1.1). The numbers are ILLUSTRATIVE (a realizable laboratory configuration, not a published apparatus):

``yb171_chain(n_ions)`` is the 171Yb+ chain the validation suite runs on since milestone M6 (``tests/m6_fixtures.py``; a test
asserts the two build the same device, hash for hash): B = 5 G along x, secular frequencies (3.0, 2.9, 2.0) MHz, a GLOBAL
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
"""The same on-axis intensity as the global pair, so every carrier Rabi frequency sits near 100 kHz."""
TRAP_HZ = (3.0e6, 2.9e6, 1.0e6)
FIELD_GAUSS = 5.0


@dataclasses.dataclass(frozen=True)
class DevicePreset:
    """An example device with the drive maps its scheduler needs (Section 7.3) and the index of its detection beam. Since
    0.2.0 the ``device`` carries the same three maps as ``Device.roles`` (docs/api_implementation_plan.md 1.1), so ``run``,
    ``calibrate`` and the benchmarks need no drive keyword; the fields here restate them for the record and the app.
    ``run_kwargs()``, deprecated in 0.2.0, was removed in 0.4.0 (docs/deprecations.md)."""

    name: str
    device: Device
    gate_drives: dict[int, GateDrive]
    """Which beams play the single-qubit gates of each ion (``device.roles.gate``)."""
    entangling_drives: dict[int, GateDrive]
    """Which beams play the entangling gates (the global pair; ``device.roles.entangling``)."""
    detection_beam: int
    notes: tuple[str, ...] = ()

    @property
    def n_ions(self) -> int:
        return self.device.crystal.n_ions

    def machine(self) -> Machine:
        """The preset as the executor of 0.2.0: a ``Machine`` on its device (which carries the roles), the closed-form
        calibration cached per device, the default option objects and the AUTO level (docs/api_implementation_plan.md 1.1)."""
        from qutip_trap.machine import Machine

        return Machine(self.device, name=self.name)


@deprecated(deadline="v0.5", fix="Call NoiseModel() instead; every default of the model means off.")
def quiet_noise_model() -> NoiseModel:
    """A noise model with no heating, no field or drift content and no collisions (every Drift of zero rms): since 0.3.0
    this is ``NoiseModel()`` itself (docs/api_implementation_plan.md 2.5), field for field and digest for digest."""
    return NoiseModel()


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


# ---- 40Ca+ optical qubit (Section 3.2's preset list; the M6 escape from the F = 0 recipe restriction) ----------------------

CA40_TRAP_HZ = (3.0e6, 2.9e6, 2.0e6)
"""The 40Ca+ preset's secular frequencies. The axial mode is 2.0 MHz rather than the 171Yb+ preset's 1.0 MHz because this
device has no sideband-cooling stage (see ``ca40_optical_recipe``): at 1.0 MHz the Doppler-limited occupation nbar = 25.96 with
eta = 0.1028 gives eta^2 (2 nbar + 1) = 0.559, which the Section 4.2.8 (vii) Lamb-Dicke validity guard of the level-A rate model
refuses (the threshold is 0.25). eta^2 (2 nbar + 1) scales roughly as 1/omega^2, and at 2.0 MHz it is 0.141 exactly
(eta = 0.07267, nbar = 12.89).

Both occupations fell 14 % on 2026-09-08 (28.09 at 1.0 MHz and 15.05 at 2.0 MHz, giving 0.604 and 0.164) when the 397 nm
total rate became Hettrich et al. 2015's measured lifetime rather than PLAN.md 9.13's quoted 21.57 MHz read as a total; the
margin against the 0.25 threshold therefore grew, and the 2.0 MHz choice stands either way (ledger
conv.ca40_linewidth_reading)."""
CA40_729_WAIST_M = 200e-6
CA40_729_POWER_W = 5e-3
CA40_729_K = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
"""The 729 nm beam runs in the xy plane at 45 degrees to B = x, so its Delta k projects on the axial and both radial families."""
CA40_729_POLARIZATION = (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
"""At 135 degrees, so the Delta m = 0 geometric factor of the E2 tensor is non-zero.

g^(0) = c^(0)_ij eps_i n_j with c^(0) = (2/3) diag(-1/2, -1/2, 1) in the atomic frame needs eps_z(atomic) n_z(atomic) != 0,
so the obvious k = y_lab with pol = x_lab derives EXACTLY zero E2 Rabi frequency (M4 finding, ``tests/m4_fixtures.py``
``CA40_729_E2_BEAM``, which this geometry matches: 34735.4 Hz at 5 mW in a 200 um waist with B = 5 G along x); k = (1,1,0)
with pol = z and k = y with pol = (x + z)/sqrt 2 also derive zero. A preset whose beams cannot deliver the Rabi frequency
its table claims makes the played chain of Section 7.3 play a fiction, so the geometry is part of the preset."""
CA40_397_WAIST_M = 30e-6
CA40_866_WAIST_M = 30e-6
CA40_854_WAIST_M = 30e-6


CA40_DETECTION_S_397 = 2.0
CA40_DETECTION_S_866 = 10.0
"""The detection cycle's saturation parameters. The derived bright scatter rate peaks near these values at R_o = 3.49e6/s and
FALLS at higher intensity (1.31e6/s at (10, 30), 1.96e5/s at (50, 100)): the magic-angle LINEAR polarization of the 397/866
cycle traps population in coherent dark states, the effect a laboratory destabilizes with a second 866 nm polarization or a
larger field. R_o is therefore 8 times below Myerson et al. 2008's 2.9e7/s, and the preset compensates with a 2 ms detection
window (13.3 detected photons at eps_sys = 0.19 %) instead of their 420 us (2.9).

All three rates moved on 2026-09-08 with the 40Ca+ P1/2 rate (ledger conv.ca40_linewidth_reading; 3.57e6, 1.99e6 and
3.06e5/s before, 13.6 photons). The peak barely moved, 2 %, but the saturated points fell 34 % and 36 % even though the
total rate ROSE 6.9 %, and the driver is the total rate and not Ramm's branching: the total rate alone takes the (10, 30)
point from 1.99e6 to 1.22e6/s (-38 %), while the branching alone raises it to 2.12e6/s (+6.7 %), and the two together give
1.31e6/s. Rescaling B with Gamma (5 G ->
4.678 G under the old rate) recovers 80 % of the drop, which identifies the cause as the fixed 5 G field, whose Zeeman
splitting shrinks in units of the linewidth as Gamma grows and so destabilizes the coherent dark states less."""
CA40_DETECTION_WINDOW_S = 2e-3


def myerson_ca40_pmt_detector(window_s: float = CA40_DETECTION_WINDOW_S) -> Detector:
    """Myerson et al. 2008's 40Ca+ shelving readout chain: eps_sys = 0.19 % and 442 detected background counts per second
    (``readout.presets.MYERSON_CA40_PMT``), at the window ``CA40_DETECTION_WINDOW_S`` explains."""
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
    """The 40Ca+ optical qubit's preparation (Section 4.2.6 order Doppler -> sideband -> pump), written by hand because
    ``standard_recipe`` covers hyperfine qubits whose lower level is an F = 0 state (``conv.preparation_single_species_f0``).

    Doppler cooling is the 397 nm S1/2-P1/2 line at -Gamma/2 along the oblique path (every mode projects on it, Section 4.2)
    with the 866 nm D3/2-P1/2 repump on resonance, which closes the cooling cycle. The "optical pump" that prepares the
    lower qubit level S1/2 mJ = -1/2 is 397 nm sigma- light along B with the same 866 nm repump: sigma- drives
    S1/2 mJ = +1/2 -> P1/2 mJ = -1/2 and leaves mJ = -1/2 dark, so the population accumulates there (the I = 0 analogue of
    the F = 1 -> F' = 1 pump of the 171Yb+ recipe).

    The pump runs a WEAK 397 nm beam against a STRONG 866 nm repump (``s_pump_397`` = 0.05 against ``s_pump_866`` = 10),
    because what is left in the metastable D3/2 level when the beams switch off IS the preparation error: D3/2 lives about a
    second, so its residual population never decays into the qubit. The pumped-state error is 1.43e-1 at the Doppler stage's
    own (0.5, 3.0) in 20 us, 1.06e-5 at the default (0.05, 10) in 100 us and 8.2e-10 at (0.02, 20) in 200 us - the
    laboratory equivalent is switching the 397 nm light off before the 866 nm light, which this recipe's two-beam pump stage
    cannot express. (The three durations differ and the previous version of this note did not say so; at a common 100 us the
    three are 3.7e-3, 1.06e-5 and 2.0e-5, so (0.02, 20) is not uniformly better - a weaker 397 nm beam also pumps more
    slowly.) All three fell on 2026-09-08 with the 40Ca+ P1/2 rate, by 1.3x, 5.8x and 10.6x (1.8e-1, 6.1e-5 and 8.7e-9
    before). The default point splits about evenly between the two changes: 6.11e-5 -> 2.51e-5 from Ramm et al. 2013's
    branching 0.06435 replacing Section 8.1's 0.06 alone, and -> 2.81e-5 from the 6.9 % higher total rate alone. They act
    differently, though. The branching acts through the 866/397 intensity ratio: I_sat(866) follows the 866 nm PARTIAL rate,
    which rose 14.6 % where the 397 nm one fell 0.5 %, so at fixed s_pump the repump gets relatively stronger and leaves
    less in D3/2. The total rate acts mostly through the FIXED 100 us pump duration, which buys 6.9 % more pumping measured
    in 1/Gamma: at a duration scaled by the same 1.0687 (93.57 us) the total-rate-only error is 5.17e-5 instead of 2.81e-5,
    i.e. the fixed duration accounts for 72 % of its effect and the fixed B = 5 G for the rest (ledger
    conv.ca40_linewidth_reading, conv.ca40_branching).

    There is NO sideband-cooling stage: Section 4.2.2's stage is pulsed RAMAN cooling on a two-beam pair
    (``SidebandCoolingSpec.beams`` is a Raman pair whose Delta k sets eta per mode), and this device cools its motion on the
    729 nm quadrupole line instead, which the first release does not schedule. The Doppler occupations therefore stand, and
    ``run`` reports them; the 854 nm D5/2-P3/2 beam the device carries is the shelf reset, which belongs to the readout
    stage rather than to this recipe."""
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
    """A 40Ca+ optical-qubit chain (S1/2 mJ = -1/2 <-> D5/2 mJ = -1/2 at 729 nm), the second species of Section 3.2's preset
    list and the M6 answer to "every non-F = 0 qubit needs a hand-written recipe and none exists".

    B = 5 G along x, secular frequencies (3.0, 2.9, 2.0) MHz, one global 729 nm quadrupole beam in the xy plane at 45 degrees
    (5 mW in a 200 um waist; the geometry that derives a non-zero Delta m = 0 E2 Rabi frequency, see
    ``CA40_729_POLARIZATION``) for the single-qubit gates, the resonant 397 nm S1/2-P1/2 light along the oblique path
    (1, 1, 1)/sqrt 3 for Doppler cooling and shelving detection, the 866 nm D3/2-P1/2 repump that closes the cooling cycle,
    Myerson et al. 2008's PMT chain, a quiet noise model and near-ideal electronics unless overridden, and
    ``ca40_optical_recipe``.

    ``reset_beam=True`` adds the 854 nm D5/2-P3/2 beam that empties the shelf between shots. It is OFF by default because a
    Device's beams are always on: the readout's Bloch model takes every resonant beam near the cycling cycle as detection
    light (``light.roles.detection_beams`` returns the 397, 866 AND 854 nm beams), so an 854 nm beam that is never gated
    depumps the shelved dark state DURING detection and destroys the contrast - measured through the product POVM (eps_B, eps_D) = (2.9e-2, 0.97) on 2026-09-08, the shelf being pumped bright
    with it against (3.4e-4, 1.8e-3) without. Per-stage beam gating is not in the first release, so the reset beam is an
    opt-in the caller adds when modelling the reset itself. (eps_B rose from 2.5e-4 on 2026-09-08 with the 40Ca+ P1/2 rate,
    which lowered the derived bright rate R_o at this saturation and so the bright-record separation; eps_D, set by the
    1.168 s shelf lifetime against the 2 ms window, did not move. Ledger conv.ca40_linewidth_reading.)

    There is NO entangling drive: the light-shift force of Section 4.4.4 needs a far-detuned pair near 398.5 nm
    (``tests/test_light_shift_gate.py``), which this preset does not carry, so ``entangling_drives`` is empty and a circuit
    with a two-qubit gate is refused by the scheduler. Single-qubit circuits and the whole preparation and readout chain
    run."""
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
