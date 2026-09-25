"""Shared test fixtures: small devices, toy species, readout models and the constants the end-to-end runs share.

Numbers are fixture values that exercise the package, not physics results. The 171Yb+ chains sit in the trap
(3.0, 2.9, 1.0) MHz of the Section 11.1 fixture: for two ions the x-COM mode is 3.000 MHz and the x-rocking mode 2.828 MHz.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np

from qutip_trap.calibration.surrogate import SurrogateReport, surrogate_table
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.control.hardware import HardwareChain
from qutip_trap.control.schedule import GateDrive
from qutip_trap.control.shaping import GateModes, gate_modes
from qutip_trap.control.table import CalEntry, CalibrationTable, Waveform
from qutip_trap.device.model import Device, Field
from qutip_trap.device.presets import (
    CA40_729_K,
    CA40_729_POLARIZATION,
    CA40_729_WAIST_M,
    crain_snspd_detector,
    ideal_hardware,
    myerson_ca40_pmt_detector,
    raman_pair_along_x,
    secular_trap,
    yb171_chain,
)
from qutip_trap.dynamics.multilevel import MultiLevelOptions
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.beams import Beam, PolGradientBeams
from qutip_trap.light.bloch import BlochModel, beam_for_transition
from qutip_trap.light.raman import derive_optical_drive, derive_raman_drive
from qutip_trap.machine import Machine
from qutip_trap.noise.model import NoiseModel
from qutip_trap.options import Numerics
from qutip_trap.readout.detection import Detector, RecordModel
from qutip_trap.readout.fluorescence import FluorescenceRates, ReadoutScheme, rates_from_detected
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD, MYERSON_CA40_PMT
from qutip_trap.run.results import Diagnostics, Progress, Result, RunState, aggregate
from qutip_trap.species import species
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.polarization import linear_polarization
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.crystal import Crystal, Mode, solve_crystal
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import RfDrive
from qutip_trap.units import ATOMIC_MASS_KG, C_M_PER_S, TWO_PI, hz_from_wavenumber_cm, lande_g_j

# ---- the end-to-end runs ------------------------------------------------------------------------------------------------

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
"""The detection windows (s) the surrogate's readout calibration chooses from."""
FAST = Numerics(branch_weight_min=1e-3)
"""Fock branches below 1e-3 weight dropped."""


@functools.cache
def two_ion_surrogate(detection_records: int) -> SurrogateReport:
    """The closed-form calibration of ``yb171_chain(2)`` for the pair (0, 1), built once per process and read-only."""
    return surrogate_table(
        yb171_chain(2).device,
        pairs=[(0, 1)],
        detection_records=detection_records,
        detection_windows_s=WINDOWS,
    )


def run(
    circuit: Circuit,
    device: Device,
    shots: int,
    *,
    seed: int = 0,
    keep_final_state: bool = False,
    progress: Callable[[Progress], None] | None = None,
    **machine: Any,
) -> Result:
    """``Machine(device, **machine).run(circuit, shots, ...)``."""
    return Machine(device, **machine).run(
        circuit, shots, seed=seed, keep_final_state=keep_final_state, progress=progress
    )


# ---- records -------------------------------------------------------------------------------------------------------------

REALISTIC_HARDWARE = HardwareChain(
    dds_phase_bits=16,
    dds_amplitude_bits=14,
    aom_rise_s=50e-9,
    amplifier_bandwidth_hz=1e8,
    dead_time_s=1e-6,
    phase_continuous=True,
)
"""A 16-bit phase / 14-bit amplitude DDS, a 50 ns modulator rise and a 100 MHz amplifier (Section 7.10)."""


def quiet_device(crystal: Crystal, trap: Trap, field: Field, beams: tuple[Beam, ...]) -> Device:
    """A device with a PMT (2 % efficiency, 100 cps background, 100 us window), the quiet noise model and near-ideal
    phase-continuous electronics."""
    return Device(
        crystal=crystal,
        trap=trap,
        field=field,
        beams=beams,
        noise=NoiseModel(),
        detector=Detector("pmt", 0.02, 100.0, {}, None, None, 100e-6),
        hardware=ideal_hardware(phase_continuous=True),
    )


def make_device() -> Device:
    """Two 171Yb+ ions with the modes written out (axial 1.000/1.732, x 2.828/3.000, y 2.720/2.900 MHz), B along z and two
    355 nm Raman beams crossing at 90 degrees, both polarized along z."""
    yb = species("171Yb+")
    sq = 1.0 / math.sqrt(2.0)
    com = np.array([sq, sq])
    rock = np.array([-sq, sq])
    modes = (
        Mode("axial", 0, 1.000e6, (0.0, 0.0, 1.0), com),
        Mode("axial", 1, math.sqrt(3.0) * 1.000e6, (0.0, 0.0, 1.0), rock),
        Mode("transverse_1", 0, 2.828e6, (1.0, 0.0, 0.0), rock),
        Mode("transverse_1", 1, 3.000e6, (1.0, 0.0, 0.0), com),
        Mode("transverse_2", 0, 2.720e6, (0.0, 1.0, 0.0), rock),
        Mode("transverse_2", 1, 2.900e6, (0.0, 1.0, 0.0), com),
    )
    crystal = Crystal(
        species=(yb, yb), positions_m=np.array([[0.0, 0.0, -2.7e-6], [0.0, 0.0, 2.7e-6]]), modes=modes
    )
    b1 = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 20e-6, 10e-3, (0.0, 0.0, 0.0))
    b2 = Beam(355e-9, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 20e-6, 10e-3, (0.0, 0.0, 0.0))
    return quiet_device(crystal, secular_trap(), Field(5.0, (0.0, 0.0, 1.0)), (b1, b2))


def make_run_state() -> RunState:
    return RunState(order=(0, 1), dark=frozenset(), lost=frozenset(), events=())


def make_calibration_table() -> CalibrationTable:
    entry = CalEntry(
        value=5.0,
        uncertainty=0.001,
        status="seed",
        experiment="field_scan",
        provenance_id="conv.curvature_naming",
        fitted_at_s=0.0,
        sample_id=0,
    )
    return CalibrationTable(
        device_hash="fixture",
        seed=0,
        surrogate=True,
        qubit_freq={},
        rabi={},
        stark={},
        crosstalk={},
        modes={},
        nbar={},
        ms={},
        field=entry,
        micromotion={},
        detection={},
        heating={},
    )


def make_space() -> HilbertSpace:
    return HilbertSpace(
        ion_dims=(2, 2),
        resolved=(ModeTruncation(mode=3, d=8, expected_n_range=(0, 3), eta_max=0.1),),
        enr_group=None,
        frozen=(0, 1, 2, 4, 5),
    )


def make_diagnostics() -> Diagnostics:
    return Diagnostics(
        level="JOINT_EXACT",
        space=make_space(),
        mode_class={3: "resolved", 0: "frozen", 1: "frozen", 2: "frozen", 4: "frozen", 5: "frozen"},
        run_state=make_run_state(),
        wall_clock_span_s=0.0,
        boundary_population={3: 0.0},
        margin_levels={3: 0},
        dropped_modes=(),
        frozen_contribution={},
        integrator="dop853",
        tolerances=(1e-10, 1e-8),
        samples=1,
        trajectories=1,
        shots_per_sample=1,
        effective_sample_size=1.0,
        root_seed=0,
        calibration=make_calibration_table(),
        approximations=(),
    )


def make_result(bitstrings: np.ndarray) -> Result:
    counts, probabilities = aggregate(bitstrings)
    return Result(
        bitstrings=np.asarray(bitstrings, dtype=np.uint8),
        bit_order="qubit0_lsb",
        counts=counts,
        probabilities=probabilities,
        error_bars={k: 0.0 for k in counts},
        photon_records=None,
        posteriors=None,
        noise_samples=(),
        heralds=np.zeros(len(bitstrings), dtype=np.uint8),
        discarded_shots=0,
        run_state=make_run_state(),
        spam={},
        final_state=None,
        diagnostics=make_diagnostics(),
    )


@dataclasses.dataclass(frozen=True)
class MassOnly:
    """A stand-in for ``Species`` carrying only the mass, for crystals quoted with isotope masses."""

    mass_u: float


# ---- 171Yb+ devices with derived drives -----------------------------------------------------------------------------------

KX = 1
"""The 3 MHz x mode of the single-ion device."""
X_COM_TWO_IONS = 3
"""The x-COM mode (3.000 MHz) of the two-ion chain; mode 2 is the x-rocking mode (2.828 MHz)."""


def single_ion_raman_device(
    rf: RfDrive | None = None, stray: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Device:
    """One 171Yb+ ion with B along x and a 10 mW, 20 um counter-propagating pair along x: eta = 0.111 on the x mode."""
    trap = dataclasses.replace(secular_trap(), rf=rf, stray_field_v_per_m=stray)
    return quiet_device(
        solve_crystal(trap, (species("171Yb+"),)),
        trap,
        Field(5.0, (1.0, 0.0, 0.0)),
        raman_pair_along_x(10e-3, 20e-6, (0.0, 0.0, 0.0)),
    )


def two_ion_raman_device(waist_m: float = 20e-6) -> Device:
    """Two 171Yb+ ions along z with B along z and a 90-degree pair (x-beam polarized y, y-beam polarized x) on ion 0."""
    trap = secular_trap()
    yb = species("171Yb+")
    crystal = solve_crystal(trap, (yb, yb))
    x0 = tuple(float(v) for v in crystal.positions_m[0])
    b1 = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), waist_m, 10e-3, x0)
    b2 = Beam(355e-9, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), waist_m, 10e-3, x0)
    return quiet_device(crystal, trap, Field(5.0, (0.0, 0.0, 1.0)), (b1, b2))


def microwave_device() -> Device:
    """One 171Yb+ ion and no beams: single-qubit gates are microwave pulses (eta = 0)."""
    trap = secular_trap()
    return quiet_device(solve_crystal(trap, (species("171Yb+"),)), trap, Field(5.0, (0.0, 0.0, 1.0)), ())


def chain_device(n_ions: int, omega_hz: tuple[float, float, float] = (3.0e6, 2.9e6, 1.0e6)) -> Device:
    """``n_ions`` 171Yb+ ions with B along x and one global 0.3 W, 60 um pair along +-x (a 147 kHz carrier)."""
    trap = secular_trap(omega_hz)
    crystal = solve_crystal(trap, tuple([species("171Yb+")] * n_ions))
    return quiet_device(
        crystal, trap, Field(5.0, (1.0, 0.0, 0.0)), raman_pair_along_x(0.3, 60e-6, (0.0, 0.0, 0.0))
    )


def raman_gate_drives(n_ions: int) -> dict[int, GateDrive]:
    return {i: GateDrive("raman", (0, 1)) for i in range(n_ions)}


def two_ion_modes(device: Device, nbar: dict[int, float] | None = None) -> GateModes:
    """The x-family modes the along-x pair couples to (the y and z modes have eta = 0 exactly)."""
    return gate_modes(device, (0, 1), (0, 1), nbar=nbar)


def derived_seeds(
    device: Device, drives: dict[int, GateDrive]
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """(carrier Rabi frequency, differential Stark shift) per (ion, table key beam) that the device derives for ``drives``,
    so that a table seeded with them plays the physical drive it requests."""
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


def ca_light_shift_device() -> Device:
    """Two 40Ca+ ions (optical S1/2-D5/2 qubit) with B along x: a counter-propagating 398.5 nm pair along x, 3 THz below the
    S-P1/2 line and circular about B, for the light-shift force, and a 166 mW 729 nm quadrupole beam in the xy plane whose
    derived E2 Rabi frequency is 200144 Hz."""
    ca = species("40Ca+")
    trap = secular_trap()
    lam = C_M_PER_S / (C_M_PER_S / 396.959e-9 - 3.0e12)
    s = 1.0 / math.sqrt(2.0)
    b1 = Beam(lam, (1.0, 0.0, 0.0), (0.0, s, 1j * s), 200e-6, 10e-3, (0.0, 0.0, 0.0))
    b2 = Beam(lam, (-1.0, 0.0, 0.0), (0.0, s, 1j * s), 200e-6, 10e-3, (0.0, 0.0, 0.0))
    e2 = Beam(729.348e-9, CA40_729_K, CA40_729_POLARIZATION, CA40_729_WAIST_M, 166e-3, (0.0, 0.0, 0.0))
    return quiet_device(solve_crystal(trap, (ca, ca)), trap, Field(5.0, (1.0, 0.0, 0.0)), (b1, b2, e2))


# ---- toy species -----------------------------------------------------------------------------------------------------------

HALF = Fraction(1, 2)
GAMMA_HZ = 19.6e6
"""gamma/2pi of the toy species' excited levels (Ozeri 2007 Table I for 9Be+)."""
G_J_S12 = 2.00226
"""9Be+'s S1/2 g_J, 2.00226(2); also the value assumed for 25Mg+, whose absolute g_J is not measured (only g_I/g_J)."""
WAVELENGTH_M = 369.5e-9
MASS_KG = 170.93578 * ATOMIC_MASS_KG
TWO_LEVEL_GROUND = "S0/2 mJ=0"
TWO_LEVEL_EXCITED = "P2/2 mJ=0"
TWO_LEVEL_EXCITED_PLUS = "P2/2 mJ=1"
LAMBDA_EXCITED = "P0/2 mJ=0"
LAMBDA_GROUND_MINUS = "S2/2 mJ=-1"
LAMBDA_GROUND_PLUS = "S2/2 mJ=1"


def _be_species(
    name: str,
    nuclear_spin: float,
    mass_u: float,
    mu_i: float,
    a_s12: float,
    g_j_s: float,
    qubit: tuple[str, str],
) -> Species:
    """S1/2, P1/2 and P3/2 at the NIST Be II energies with hyperfine-free P levels; Gamma(P3/2) follows Gamma(P1/2) as
    omega^3 (one radial integral)."""
    e12 = float(hz_from_wavenumber_cm(31928.744))
    e32 = float(hz_from_wavenumber_cm(31935.320))
    gamma_p32_hz = GAMMA_HZ * (e32 / e12) ** 3
    cites = ("Steck",)
    s12 = Level("S1/2", 0.0, None, a_s12, 0.0, g_j_s, cites)
    p12 = Level("P1/2", e12, 1.0 / (TWO_PI * GAMMA_HZ), 0.0, 0.0, lande_g_j(1, HALF, HALF), cites)
    p32 = Level(
        "P3/2", e32, 1.0 / (TWO_PI * gamma_p32_hz), 0.0, 0.0, lande_g_j(1, HALF, Fraction(3, 2)), cites
    )
    t12 = Transition("S1/2", "P1/2", C_M_PER_S / e12, GAMMA_HZ, 1.0, "E1", cites)
    t32 = Transition("S1/2", "P3/2", C_M_PER_S / e32, gamma_p32_hz, 1.0, "E1", cites)
    return Species(
        name=name,
        mass_u=mass_u,
        nuclear_spin=nuclear_spin,
        mu_I_nuclear_magnetons=mu_i,
        levels=(s12, p12, p32),
        transitions=(t12, t32),
        qubit=qubit,
        cycling="S1/2-P3/2",
        repumps=(),
        shelving=None,
    )


def be9_like(*, a_scale: float = 1.0) -> Species:
    """I = 3/2 with the 9Be+ ground-state constants (A_hfs times ``a_scale``, mu_I, g_J) and hyperfine-free P levels, the
    approximation of Ozeri's and Wineland's closed forms."""
    return _be_species(
        "9Be+-like fixture",
        1.5,
        9.012183065 - 0.00054858,
        -1.177432,
        -625.008837048e6 * a_scale,
        G_J_S12,
        ("S1/2 F=2 mF=0", "S1/2 F=1 mF=0"),
    )


def spin_zero_like() -> Species:
    """I = 0 with S1/2, P1/2 and P3/2: the single-electron angular algebra of Ozeri 2005."""
    return _be_species(
        "spin-zero fixture", 0.0, 9.0, 0.0, 0.0, lande_g_j(0, HALF, HALF), ("S1/2 mJ=-1/2", "S1/2 mJ=1/2")
    )


def field_z(b_gauss: float = 1.0) -> Field:
    return Field(B_gauss=b_gauss, direction=(0.0, 0.0, 1.0))


def _closed_atom(
    name: str,
    ground: str,
    excited: str,
    g_ground: float,
    g_excited: float,
    *,
    gamma_hz: float = GAMMA_HZ,
    wavelength_m: float = WAVELENGTH_M,
    mass_kg: float = MASS_KG,
) -> Species:
    cites = ("Steck",)
    lower = Level(ground, 0.0, None, 0.0, 0.0, g_ground, cites)
    upper = Level(excited, C_M_PER_S / wavelength_m, 1.0 / (TWO_PI * gamma_hz), 0.0, 0.0, g_excited, cites)
    return Species(
        name=name,
        mass_u=mass_kg / ATOMIC_MASS_KG,
        nuclear_spin=0.0,
        mu_I_nuclear_magnetons=0.0,
        levels=(lower, upper),
        transitions=(Transition(ground, excited, wavelength_m, gamma_hz, 1.0, "E1", cites),),
        qubit=(f"{ground} mJ=0", f"{excited} mJ=0"),
        cycling=f"{ground}-{excited}",
        repumps=(),
        shelving=None,
    )


def two_level_atom(
    *, gamma_hz: float = GAMMA_HZ, wavelength_m: float = WAVELENGTH_M, mass_kg: float = MASS_KG
) -> Species:
    """J = 0 -> J' = 1 (levels "S0/2", "P2/2"): pi light on |g> <-> |e, 0> is an exactly closed two-level system, so the RMP
    Lorentzian, Stenholm's sideband floor and Itano's Doppler limit hold exactly (171Yb+ 369.5 nm numbers by default)."""
    return _closed_atom(
        "two-level fixture",
        "S0/2",
        "P2/2",
        0.0,
        1.0,
        gamma_hz=gamma_hz,
        wavelength_m=wavelength_m,
        mass_kg=mass_kg,
    )


def lambda_atom() -> Species:
    """J = 1 -> J' = 0 (levels "S2/2", "P0/2"): sigma+ and pi light drive |g,-1> and |g,0> to |e>; with |g,+1> excluded
    (``leak='renormalize'``) the two arms share the decay 1/2 : 1/2, Morigi's ideal Lambda system."""
    return _closed_atom("Lambda fixture", "S2/2", "P0/2", 1.0, 0.0)


def structure(
    species: Species, b_gauss: float = 1e-6, b_hat: tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> AtomicStructure:
    """The dressed structure at a negligible field that only orients B."""
    return AtomicStructure(species, b_gauss, b_hat)


def gamma_rad_s() -> float:
    return TWO_PI * GAMMA_HZ


def power_for_rabi(
    st: AtomicStructure,
    lower: str,
    upper: str,
    omega_rad_s: float,
    waist_m: float,
    polarization: tuple[complex, complex, complex],
    k_hat: tuple[float, float, float],
) -> float:
    """The beam power that gives |Omega_eg| = omega on the pair (polarization-resolved element)."""
    lam = TWO_PI * C_M_PER_S / (TWO_PI * (st.state(upper).energy_hz - st.state(lower).energy_hz))
    probe = Beam(lam, k_hat, polarization, waist_m, 1e-3, (0.0, 0.0, 0.0))
    om = abs(st.single_photon_coupling_rad_s(st.state(lower), st.state(upper), probe))
    return 1e-3 * (omega_rad_s / om) ** 2


def circular_beam(
    st: AtomicStructure,
    lower: str,
    upper: str,
    omega_rad_s: float,
    detuning_rad_s: float,
    *,
    q: int = 1,
    k_sign: int = 1,
    waist_m: float = 20e-6,
) -> Beam:
    """A sigma+ (q = +1, eps = -(x + i y)/sqrt2) or sigma- (q = -1, eps = (x - i y)/sqrt2) beam about B = z along +-z
    (``k_sign``) with Rabi frequency ``omega`` on lower -> upper; eps_q drives m -> m + q whatever the direction of k."""
    if q not in (1, -1):
        raise ValueError("q is +1 or -1")
    pol = (-q / math.sqrt(2.0) + 0.0j, -1j / math.sqrt(2.0), 0.0 + 0.0j)
    k_hat = (0.0, 0.0, float(k_sign))
    power = power_for_rabi(st, lower, upper, omega_rad_s, waist_m, pol, k_hat)
    omega = TWO_PI * (st.state(upper).energy_hz - st.state(lower).energy_hz) + detuning_rad_s
    return Beam(TWO_PI * C_M_PER_S / omega, k_hat, pol, waist_m, power, (0.0, 0.0, 0.0))


def sigma_plus_beam(
    st: AtomicStructure,
    lower: str,
    upper: str,
    omega_rad_s: float,
    detuning_rad_s: float,
    *,
    waist_m: float = 20e-6,
) -> Beam:
    """A sigma+ beam along +z with Rabi frequency ``omega`` on lower -> upper."""
    return circular_beam(st, lower, upper, omega_rad_s, detuning_rad_s, q=1, k_sign=1, waist_m=waist_m)


# ---- 171Yb+ detection ---------------------------------------------------------------------------------------------------

YB = species("171Yb+")
GAMMA_S = YB.transition("S1/2-P1/2").partial_rate_rad_s
"""The partial S1/2-P1/2 rate, 2 pi x 19.62 MHz: the Gamma of the detection-rate closed forms."""
D_HFP = TWO_PI * 2.105e9
"""The P1/2 hyperfine splitting (rad/s)."""
D_HFS = TWO_PI * 12_642_812_118.5
"""The S1/2 hyperfine splitting (rad/s)."""
BRIGHT = tuple(f"S1/2 F=1 mF={m}" for m in (-1, 0, 1))
DARK = ("S1/2 F=0 mF=0",)
MAGIC_ANGLE_RAD = math.acos(1.0 / math.sqrt(3.0))
YB_DIRECT = ReadoutScheme.direct(1)
"""171Yb+: |1> = F = 1 bright, |0> = F = 0 dark."""
CA_OPTICAL = ReadoutScheme.shelving(1)
"""40Ca+ optical qubit: |1> = D5/2 is the shelf, |0> = S1/2 bright."""


def detection_model(s0: float, b_gauss: float, delta_rad_s: float = 0.0, **kw: object) -> BlochModel:
    """171Yb+ S1/2 + P1/2 with the D3/2 branch folded back, B along z, and a 20 um detection beam along x at the magic
    angle with I/I_sat = s0 on the partial-rate I_sat."""
    st = AtomicStructure(YB, b_gauss, (0.0, 0.0, 1.0))
    waist = 20e-6
    pol = tuple(linear_polarization((1.0, 0.0, 0.0), MAGIC_ANGLE_RAD, (0.0, 0.0, 1.0)))
    power = s0 * YB.transition("S1/2-P1/2").i_sat_w_m2 * math.pi * waist**2 / 2.0
    beam = beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=0 mF=0",
        delta_rad_s,
        (1.0, 0.0, 0.0),
        pol,
        power_w=power,
        waist_m=waist,
    )
    return BlochModel(
        st, [beam], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"), **kw
    )


def yb_detection_beam(s_o: float, b_gauss: float = 5.0, detuning_rad_s: float = 0.0) -> Beam:
    """A 20 um 369.5 nm beam along +y, linearly polarized at the magic angle to B = x, with I/I_sat = s_o on axis."""
    st = AtomicStructure(YB, b_gauss, (1.0, 0.0, 0.0))
    waist = 20e-6
    power = s_o * YB.transition("S1/2-P1/2").i_sat_w_m2 * math.pi * waist**2 / 2.0
    pol = tuple(linear_polarization((0.0, 1.0, 0.0), MAGIC_ANGLE_RAD, (1.0, 0.0, 0.0)))
    return beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=0 mF=0",
        detuning_rad_s,
        (0.0, 1.0, 0.0),
        pol,
        power_w=power,
        waist_m=waist,
    )


def crain_record_model(window_s: float = 22e-6) -> RecordModel:
    """Crain's measured rates as a record model: 472 kcps detected, R_d = 341 Hz, R_b = 16.4 Hz, 4.2 cps background."""
    return RecordModel.from_rates(CRAIN_YB171_SNSPD.rates(), crain_snspd_detector(window_s))


def myerson_record_model(window_s: float = 420e-6) -> RecordModel:
    """Myerson's 40Ca+ apparatus: R_B = 55800/s, R_D = 442/s, a 1168 ms shelf lifetime."""
    return RecordModel.from_rates(MYERSON_CA40_PMT.rates(), myerson_ca40_pmt_detector(window_s))


def two_state_rates(
    detected_bright_per_s: float, efficiency: float, r_d: float, r_b: float
) -> FluorescenceRates:
    return rates_from_detected(
        detected_bright_per_s, efficiency, dark_pumping_per_s=r_d, bright_pumping_per_s=r_b
    )


def lin_perp_lin_pair(
    wavelength_m: float,
    power_w: float,
    waist_m: float,
    detuning_hz: float,
    *,
    phase_rad: float = 0.0,
    beat_hz: float = 0.0,
) -> PolGradientBeams:
    """A counter-propagating pair along z polarized along x and y; ``beat_hz`` offsets beam b for a moving gradient."""
    a = Beam(wavelength_m, (0.0, 0.0, 1.0), (1.0 + 0j, 0j, 0j), waist_m, power_w, (0.0, 0.0, 0.0))
    b = Beam(wavelength_m, (0.0, 0.0, -1.0), (0j, 1.0 + 0j, 0j), waist_m, power_w, (0.0, 0.0, 0.0))
    return PolGradientBeams(a, b, detuning_hz, beat_hz, phase_rad, "jg12_je12")
