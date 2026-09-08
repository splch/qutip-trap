"""The near-field microwave-gradient drive, end to end (PLAN.md Section 4.4.5, 9.4 'Microwave gradient', 9.13; M4).

Srinivas et al., Nature 597, 209 (2021) drive two 25Mg+ ions with two microwave tones symmetrically detuned by +-delta
from the ac-Zeeman-shifted qubit frequency plus an oscillating magnetic-field gradient at omega_g near the motional
frequency, giving

    H_I(t) = hbar Omega_g J_2(4 Omega_mu/delta)(sigma_z1 - sigma_z2)(a e^{i Delta t} + a^dag e^{-i Delta t}),
    Omega_g = (r_0/4)[grad(B_g . r_hat_q) . r_hat](d omega_0/dB),   Omega_mu = (B_x/2 hbar)<dn|mu_x|up>,

with intrinsic dynamical decoupling where J_0(4 Omega_mu/delta) = 0, i.e. Omega_mu/delta = 0.6012. Before M4's fix the
``gradient`` drive kind had no branch in the builder at all and fell through to the sigma_+ spin-flip path, so a user who
declared one got an ordinary microwave with no sigma_z force and nothing in ``approximations``.

The fixture is a 40Ca+ ZEEMAN qubit (S1/2 mJ = -1/2 <-> +1/2 at 5 G, 2.8025 MHz/G) rather than 25Mg+: the gradient
couples through d omega_0/dB, which VANISHES at a clock point, and 25Mg+'s species table is incomplete (its P-level
constants are not cited), so the 25Mg+ numbers are checked against the ledger's hyperfine-Zeeman table directly in
``test_srinivas_own_parameters``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import qutip as qt
from scipy.special import jv

from qutip_trap.api import Device, Field, Trap
from qutip_trap.control.pulses import Drive, Pulse, Tone
from qutip_trap.control.table import CalEntry, Segment, Waveform
from qutip_trap.device.model import GradientField
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.light.microwave import (
    derive_gradient_drive,
    field_sensitivity_rad_s_per_t,
    gradient_drive,
)
from qutip_trap.noise.sampling import NoiseSample, key_qubit_offset_hz, quiet_sample
from qutip_trap.species import species
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.units import HBAR_J_S, TWO_PI
from qutip_trap.validation.two_qubit_closed_forms import (
    intrinsic_dynamical_decoupling_ratio,
    srinivas_effective_coupling_rad_s,
    srinivas_gradient_rabi_rad_s,
)
from tests.fixtures import make_detector, make_hardware, make_noise

ROCK = 2
"""The transverse-x mode with the ANTISYMMETRIC pattern (-1, +1)/sqrt2: the one the (sigma_z1 - sigma_z2) force drives."""
DELTA_HZ = 200e3
"""delta/2pi, the microwave tones' symmetric detuning; the gradient sits at omega_m - 2 delta so the dressed force is resonant."""
GRADIENT_T_PER_M = 152.0
"""Srinivas et al.'s measured gradient amplitude, 152(15) T/m."""
RABI_HZ_PER_MICROTESLA = 14012.475693
"""Omega/2pi of the 40Ca+ Zeeman qubit at 5 G under a 1 uT field along y (mu_B/h x 1 uT for a Delta m_J = 1 transition)."""


def zeeman_calcium() -> object:
    """40Ca+ with its S1/2 Zeeman pair designated as the qubit: field-sensitive (2.8025 MHz/G) and magnetic-dipole coupled,
    which is what a gradient gate needs. The optical S1/2-D5/2 qubit the species table designates cannot carry a
    magnetic-dipole microwave drive at all (the two levels are different fine-structure levels)."""
    return dataclasses.replace(species("40Ca+"), qubit=("S1/2 mJ=-1/2", "S1/2 mJ=1/2"))


def gradient_device(*, ratio: float | None = None) -> Device:
    """Two 40Ca+ Zeeman qubits in the Section 11.1 trap with near-field gradient electrodes at ``ratio`` = Omega_mu/delta
    (default: the intrinsic-dynamical-decoupling point 0.6012)."""
    ratio = intrinsic_dynamical_decoupling_ratio() if ratio is None else ratio
    ion = zeeman_calcium()
    trap = Trap(
        omega_hz=(3.0e6, 2.9e6, 1.0e6),
        axis_angle_rad=0.0,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    crystal = solve_crystal(trap, (ion, ion))  # type: ignore[arg-type]
    tone_rabi_hz = 2.0 * ratio * DELTA_HZ
    b1 = (0.0, tone_rabi_hz / RABI_HZ_PER_MICROTESLA * 1e-6, 0.0)
    gradient = GradientField(
        gradient_t_per_m=GRADIENT_T_PER_M,
        frequency_hz=crystal.modes[ROCK].omega_hz - 2.0 * DELTA_HZ,
        axis=(1.0, 0.0, 0.0),
        b1_tesla_lab=b1,
        phase_rad=0.0,
    )
    return Device(
        crystal=crystal,
        trap=trap,
        field=Field(5.0, (1.0, 0.0, 0.0), None),
        beams=(),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
        gradient=gradient,
    )


def gradient_space() -> HilbertSpace:
    return HilbertSpace((2, 2), (ModeTruncation(ROCK, 8, (0, 1), 0.2),), None, (0, 1, 3, 4, 5))


def test_gradient_couplings_are_derived_and_r0_carries_the_total_two_ion_mass() -> None:
    """Section 4.4.5 with Section 13's r_0 convention: Omega_g = (r_0/4) grad(B)(d omega_0/dB) with r_0 built from the
    TOTAL mass of the two ions, and the per-(ion, mode) force coefficient w_{i,m} of the laboratory Hamiltonian is
    +-2 Omega_g EXACTLY on a spatially antisymmetric mode - which is what makes the plan's (sigma_z1 - sigma_z2)
    normalization the right one. Nothing is entered as a coupling: d omega_0/dB comes from the hyperfine-Zeeman
    diagonalization, Omega_mu from the magnetic-dipole matrix element of the configured field, and w_{i,m} from the
    crystal's mass-weighted mode pattern."""
    dev = gradient_device()
    dd = derive_gradient_drive(dev, (0, 1))
    assert dd.field_sensitivity_rad_s_per_t == pytest.approx(1.76086e11, rel=1e-5)
    assert dd.field_sensitivity_rad_s_per_t == pytest.approx(TWO_PI * 2.8024951386e6 * 1e4, rel=1e-9), (
        "2.8025 MHz/G, the 40Ca+ Zeeman qubit's slope, in rad/s per tesla"
    )
    delta = TWO_PI * DELTA_HZ
    assert dd.microwave_rabi_rad_s / delta == pytest.approx(0.6012, abs=1e-4)
    assert dd.bessel_argument(delta) == pytest.approx(4.0 * 0.6012, abs=4e-4)
    assert float(jv(0, dd.bessel_argument(delta))) == pytest.approx(0.0, abs=1e-4)
    assert dd.tone_rabi_hz == pytest.approx(2.0 * dd.microwave_rabi_rad_s / TWO_PI, rel=1e-12), (
        "Srinivas's Omega_mu is HALF the tone Rabi frequency of the (hbar Omega/2) convention"
    )
    # r_0 with the total mass, and the closed form
    mass_total = float(sum(dev.crystal.masses_kg[i] for i in (0, 1)))
    r0 = math.sqrt(HBAR_J_S / (2.0 * mass_total * dev.crystal.modes[ROCK].omega_rad_s))
    omega_g = srinivas_gradient_rabi_rad_s(r0, GRADIENT_T_PER_M, dd.field_sensitivity_rad_s_per_t)
    assert dd.gradient_rabi_rad_s[ROCK] == pytest.approx(omega_g, rel=1e-12)
    assert omega_g / TWO_PI == pytest.approx(5035.3, rel=1e-4)
    for ion, sign in ((0, -1.0), (1, +1.0)):
        assert dd.coupling_rad_s[(ion, ROCK)] / omega_g == pytest.approx(2.0 * sign, rel=1e-9), (
            "w_{i,m} = +-2 Omega_g on the antisymmetric mode: the total-mass r_0 of Section 13"
        )
    assert srinivas_effective_coupling_rad_s(
        omega_g, dd.microwave_rabi_rad_s, delta
    ) / TWO_PI == pytest.approx(2174.0, rel=1e-4)
    # a device with no gradient electrodes refuses
    with pytest.raises(ValueError, match="Device.gradient"):
        derive_gradient_drive(dataclasses.replace(dev, gradient=None), (0, 1))


def test_field_sensitivity_vanishes_on_a_clock_transition() -> None:
    """A gradient gate needs a field-SENSITIVE qubit: at a clock point d omega_0/dB = 0 and Omega_g vanishes with it, which
    is why the fixture designates the Zeeman pair and why Srinivas et al. drive |F=3,mF=3> <-> |F=2,mF=2> of 25Mg+ rather
    than its 212.78 G clock line."""
    yb = species("171Yb+")
    field = Field(1e-6, (1.0, 0.0, 0.0), None)
    assert abs(field_sensitivity_rad_s_per_t(yb, field)) < 1e3, (
        "the 171Yb+ clock line is flat in B at zero field"
    )
    assert abs(field_sensitivity_rad_s_per_t(zeeman_calcium(), field)) > 1e11  # type: ignore[arg-type]


def test_builder_dressed_gradient_operator_is_the_j2_weighted_sigma_z_force() -> None:
    """Section 9.4 'Microwave gradient': with ``gradient_form="dressed"`` the built Hamiltonian's spin part at t = 0 is
    EXACTLY Omega_g J_2(4 Omega_mu/delta)(sigma_z1 - sigma_z0)(a + a^dag) on top of the free motion, and the elimination
    is recorded in ``approximations`` (the pattern of ``test_builder_light_shift_operator_is_the_level_weighted_force``)."""
    dev = gradient_device()
    dd = derive_gradient_drive(dev, (0, 1))
    delta = TWO_PI * DELTA_HZ
    drive = gradient_drive(dd, detuning_hz=DELTA_HZ)
    space = gradient_space()
    built = build_hamiltonian(
        dev,
        [Pulse(drive, 0.0, 100e-6, "grad", ())],
        space,
        options=BuilderOptions(gradient_form="dressed", micromotion="none", include_stark=False),
    )
    a = space.annihilation(ROCK)
    force = srinivas_effective_coupling_rad_s(
        dd.gradient_rabi_rad_s[ROCK], dd.microwave_rabi_rad_s, delta
    ) * (space.sigma_z(1) - space.sigma_z(0))
    expected = dev.crystal.modes[ROCK].omega_rad_s * space.number(ROCK) + force * (a + a.dag())
    assert (built.H(0.0) - expected).norm() < 1e-9 * expected.norm()
    assert built.n_drive_terms == 4, "one (sigma_z (x) a, its conjugate) pair per ion, no spin flip"
    assert any("adiabatically eliminated" in note for note in built.approximations)
    assert any("frozen mode 3" in note for note in built.approximations), (
        "the COM mode is frozen in this space: its gradient force is dropped and recorded"
    )


def test_builder_bare_gradient_keeps_both_microwave_tones_and_the_force() -> None:
    """The default ``gradient_form="bare"`` builds the two microwave tones as ordinary carrier terms BESIDE the laboratory
    sigma_z force w_{i,m} cos(omega_g t + phi_g), so J_2(4 Omega_mu/delta) and the J_0 = 0 decoupling emerge from the exact
    dynamics rather than being put in by hand (the plan's 'everything derived' preference)."""
    dev = gradient_device()
    dd = derive_gradient_drive(dev, (0, 1))
    drive = gradient_drive(dd, detuning_hz=DELTA_HZ)
    space = gradient_space()
    built = build_hamiltonian(
        dev,
        [Pulse(drive, 0.0, 100e-6, "grad", ())],
        space,
        options=BuilderOptions(micromotion="none", include_stark=False),
    )
    assert built.n_drive_terms == 8, "two carrier terms and two force terms per ion"
    assert any("emerges from the dynamics" in note for note in built.approximations)
    h0 = built.H(0.0)
    sx0 = space.sigma_plus(0) + space.sigma_minus(0)
    assert abs((h0.dag() * sx0).tr()) > 1e6, "the microwave carrier is present in the bare form"
    # the force is still exactly the laboratory one: w_{i,m} cos(phi_g) at t = 0
    a = space.annihilation(ROCK)
    force = dd.coupling_rad_s[(1, ROCK)] * (space.sigma_z(1) - space.sigma_z(0)) * (a + a.dag())
    residual = h0 - dev.crystal.modes[ROCK].omega_rad_s * space.number(ROCK) - force
    assert abs((residual.dag() * (space.sigma_z(0) * (a + a.dag()))).tr()) < 1e-3 * abs(
        (force.dag() * (space.sigma_z(0) * (a + a.dag()))).tr()
    )
    # the counter-rotating (omega_m + omega_g) half of the laboratory force is what the dressed form drops
    dressed = build_hamiltonian(
        dev,
        [Pulse(drive, 0.0, 100e-6, "grad", ())],
        space,
        options=BuilderOptions(gradient_form="dressed", micromotion="none", include_stark=False),
    )
    assert dressed.n_drive_terms < built.n_drive_terms


def test_gradient_drive_refuses_a_malformed_tone_pair() -> None:
    """A gradient drive carries exactly the two tones at -/+ delta: anything else is refused rather than built as
    something else (the old silent fall-through to the sigma_+ carrier path)."""
    dev = gradient_device()
    dd = derive_gradient_drive(dev, (0, 1))
    space = gradient_space()

    def build(drive: Drive) -> None:
        build_hamiltonian(dev, [Pulse(drive, 0.0, 1e-6, "g", ())], space, options=BuilderOptions())

    one_tone = Drive("gradient", (0, 1), (Tone(DELTA_HZ, 0.0, 1e3),), (), 0.0, {})
    with pytest.raises(ValueError, match="two microwave tones"):
        build(one_tone)
    lopsided = Drive(
        "gradient", (0, 1), (Tone(DELTA_HZ, 0.0, 1e3), Tone(2 * DELTA_HZ, 0.0, 1e3)), (), 0.0, {}
    )
    with pytest.raises(ValueError, match=r"\+-delta"):
        build(lopsided)
    no_electrodes = dataclasses.replace(dev, gradient=None)
    with pytest.raises(ValueError, match="Device.gradient"):
        build_hamiltonian(
            no_electrodes,
            [Pulse(gradient_drive(dd, detuning_hz=DELTA_HZ), 0.0, 1e-6, "g", ())],
            space,
            options=BuilderOptions(),
        )
    with pytest.raises(NotImplementedError, match="Schroedinger frame only"):
        build_hamiltonian(
            dev,
            [Pulse(gradient_drive(dd, detuning_hz=DELTA_HZ), 0.0, 1e-6, "g", ())],
            space,
            options=BuilderOptions(frame="interaction"),
        )


@pytest.mark.slow
def test_intrinsic_dynamical_decoupling_at_the_bessel_zero() -> None:
    """Section 9.4 'Microwave gradient': the IDD claim, from the exact dynamics rather than from a Bessel-zero lookup.

    A slow (here static) qubit-frequency drift enters the dressed frame weighted by J_0(4 Omega_mu/delta), so at
    Omega_mu/delta = 0.6012 the gate is first-order insensitive to it and at 0.3 it is not. Measured over 100 us with a
    2 kHz offset on both ions, starting from |+ +>|0>: infidelity 6.9e-5 at the decoupling point against 3.06e-1 at
    Omega_mu/delta = 0.3, a suppression of 4400."""
    space = gradient_space()
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi0 = space.product_state(qt.tensor(plus, plus), {ROCK: qt.basis(8, 0)})
    duration = 100e-6
    infidelities = {}
    for ratio in (intrinsic_dynamical_decoupling_ratio(), 0.3):
        dev = gradient_device(ratio=ratio)
        drive = gradient_drive(derive_gradient_drive(dev, (0, 1)), detuning_hz=DELTA_HZ)
        pulse = Pulse(drive, 0.0, duration, "grad", ())
        finals = []
        for drift_hz in (0.0, 2000.0):
            sample = (
                quiet_sample()
                if drift_hz == 0.0
                else NoiseSample(0, {key_qubit_offset_hz(0): drift_hz, key_qubit_offset_hz(1): drift_hz}, {})
            )
            built = build_hamiltonian(
                dev,
                [pulse],
                space,
                sample=sample,
                options=BuilderOptions(micromotion="none", include_stark=False),
            )
            finals.append(
                qt.sesolve(
                    built.H,
                    psi0,
                    [0.0, duration],
                    options={"atol": 1e-10, "rtol": 1e-8, "nsteps": 10**8},
                ).states[-1]
            )
        infidelities[ratio] = 1.0 - abs(finals[0].overlap(finals[1])) ** 2
    idd = infidelities[intrinsic_dynamical_decoupling_ratio()]
    plain = infidelities[0.3]
    assert idd < 1e-3, idd
    assert plain > 0.1, plain
    assert plain / idd > 1e3, (idd, plain)


def gradient_waveform(dev: Device, *, chi_rad: float) -> Waveform:
    """A one-segment ``gradient`` Waveform: the two microwave tones at -/+ delta on both ions, the sigma_z force's angle
    stored per mode. The gradient family's closure is Srinivas's Walsh sign modulation and not the AM/FM/PM algebra of
    Section 4.4.3 (the force is a Bessel FUNCTION of the tone amplitude, not linear in it; ledger conv.gradient_drive),
    so the waveform is written out rather than solved for."""
    dd = derive_gradient_drive(dev, (0, 1))
    entry = CalEntry(0.0, 0.0, "seed", "fixture", "conv.spin_motion_phases", 0.0, 0)
    amp: dict[tuple[int, str], float] = {}
    phase: dict[tuple[int, str], float] = {}
    for ion in (0, 1):
        for leg in ("blue", "red"):
            amp[(ion, leg)] = dd.tone_rabi_hz
            phase[(ion, leg)] = 0.0
    seg = Segment(50e-6, amp, phase, {"blue": DELTA_HZ, "red": -DELTA_HZ})  # type: ignore[arg-type]
    return Waveform(
        segments=(seg,),
        fourier=None,
        duration_s=50e-6,
        phi_s=entry,
        phi_m=entry,
        chi_m={ROCK: chi_rad},
        alpha_m={ROCK: 0j},
        kind="gradient",
    )


def test_scheduler_plays_a_gradient_waveform_through_the_sigma_z_echo_path() -> None:
    """Section 4.4.5 through the scheduler: a ``gradient`` waveform plays as ZZ(theta) on the same two-loop spin-echo path
    a light-shift waveform uses (the physics is the same sigma_z force), each loop emitting one ``Drive`` of kind
    ``gradient`` per ion, and the sign still comes only from the detuning side."""
    import dataclasses as dc

    from qutip_trap.api import Circuit, Operation
    from qutip_trap.control.schedule import GateDrive, ScheduleError, schedule
    from tests.fixtures import make_calibration_table

    dev = gradient_device()
    wf = gradient_waveform(dev, chi_rad=-math.pi / 8.0)
    table = dc.replace(
        make_calibration_table(),
        ms={(0, 1): wf},
        rabi={
            (i, -1): CalEntry(50e3, 1.0, "calibrated", "rabi_scan", "conv.rabi_frequency", 0.0, 0)
            for i in (0, 1)
        },
    )
    micro = {i: GateDrive("microwave", ()) for i in (0, 1)}
    ent = {i: GateDrive("gradient", ()) for i in (0, 1)}
    circuit = Circuit(2, (Operation("zz", (0, 1), (math.pi / 2,)),), (0, 1))
    sched = schedule(circuit, dev, table, gate_drives=micro, entangling_drives=ent)
    kinds = [p.drive.kind for p in sched.pulses]
    assert kinds.count("gradient") == 4, "two ions x two half-angle loops"
    assert kinds.count("microwave") == 4, "the echo and un-echo pi pulses on both ions"
    for pulse in sched.pulses:
        if pulse.drive.kind != "gradient":
            continue
        assert len(pulse.drive.tones) == 2
        detunings = sorted(float(t.detuning_hz) for t in pulse.drive.tones)  # type: ignore[arg-type]
        assert detunings == pytest.approx([-DELTA_HZ, DELTA_HZ])
    assert [g.kind for g in sched.gates] == ["zz", "zz"]
    # a gradient waveform on a light-shift drive is refused, and so is MS(...) on a gradient waveform
    with pytest.raises(ScheduleError, match="gradient gate drive"):
        schedule(circuit, dev, table, gate_drives=micro, entangling_drives=micro)
    ms_circuit = Circuit(2, (Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2)),), (0, 1))
    with pytest.raises(ScheduleError, match="microwave-gradient one"):
        schedule(ms_circuit, dev, table, gate_drives=micro, entangling_drives=ent)
    # the sign guard of the sigma_z path
    wrong = gradient_waveform(dev, chi_rad=+math.pi / 8.0)
    with pytest.raises(ScheduleError, match="opposite sign"):
        schedule(
            circuit,
            dev,
            dc.replace(table, ms={(0, 1): wrong}),
            gate_drives=micro,
            entangling_drives=ent,
        )


def test_srinivas_own_parameters() -> None:
    """Section 9.4 'Microwave gradient' and 9.13 '25Mg+ clock point', with Srinivas et al.'s own numbers instead of the
    arbitrary fixture values the first pass restated the implementation with.

    Their qubit is |F=3, mF=3> <-> |F=2, mF=2> of the 25Mg+ S1/2 manifold at 21.3 mT, NOT the 212.78 G clock line: the
    hyperfine-Zeeman diagonalization of the cited 25Mg+ constants puts that transition at 1.326467 GHz with
    d f_0/dB = 1.97333 MHz/G at 212.78 G, and the paper prints omega~_0 ~ 2 pi x 1.326 GHz for the ac-Zeeman-shifted
    qubit - agreement to four digits, and the clock pair at the same field comes out at 1.686462 GHz with 21 Hz/G, the
    9.13 row's own anchor.

    With the paper's gradient (152(15) T/m), mode (omega_r ~ 2 pi x 6.9 MHz) and gradient frequency
    (omega_g = 2 pi x 5 MHz), the derived Omega_g/2pi is 2870.8 Hz and the dressed coupling at the decoupling point is
    Omega_g J_2(4 x 0.6012)/2pi = 1239.5 Hz. The paper's delta ~ (omega_r - omega_g)/2 = 2 pi x 950 kHz is exactly the
    2 delta = omega_r - omega_g resonance the dressed force needs, and a single closed loop at the maximally entangling
    (Omega_eff/Delta_gate) = 1/4 would take 202 us, so the paper's 740 us eight-segment Walsh sequence is the right
    scale for the closure (reported, not asserted: the Walsh sequence's phase efficiency is not a plan number)."""
    from qutip_trap.species import MODULES
    from qutip_trap.species.model import Level
    from qutip_trap.species.zeeman import HyperfineZeeman
    from tests.test_zeeman_anchors import MG25_G_J_ASSUMED

    table = MODULES["25Mg+"].TABLE
    level = Level("S1/2", 0.0, None, table["mg25.S12.A_hfs_hz"].value, 0.0, MG25_G_J_ASSUMED, ("Steck",))
    hz = HyperfineZeeman(level, 2.5, table["mg25.mu_I_nuclear_magnetons"].value)
    spectrum = hz.spectrum(212.78)
    i_up, i_dn = spectrum.index("F=3 mF=3"), spectrum.index("F=2 mF=2")
    f0 = abs(spectrum.energies_hz[i_up] - spectrum.energies_hz[i_dn])
    slope_hz_per_gauss = abs(spectrum.dE_dB_hz_per_g[i_up] - spectrum.dE_dB_hz_per_g[i_dn])
    assert f0 == pytest.approx(1.326e9, rel=1e-3), "the paper's omega~_0 ~ 2 pi x 1.326 GHz"
    assert f0 == pytest.approx(1.326467253e9, rel=1e-9), "recomputed here from the cited 25Mg+ constants"
    assert slope_hz_per_gauss == pytest.approx(1.97333e6, rel=1e-5)
    i_c_up, i_c_dn = spectrum.index("F=3 mF=1"), spectrum.index("F=2 mF=1")
    assert abs(spectrum.energies_hz[i_c_up] - spectrum.energies_hz[i_c_dn]) == pytest.approx(
        1.686462e9, abs=1e3
    ), "the 9.13 row's clock point at the same field, first-order insensitive (21 Hz/G)"
    assert abs(spectrum.dE_dB_hz_per_g[i_c_up] - spectrum.dE_dB_hz_per_g[i_c_dn]) < 25.0
    # Omega_g from the paper's gradient, mode and (total) two-ion mass
    from qutip_trap.units import ATOMIC_MASS_KG

    mass_total = 2.0 * table["mg25.mass_atomic_u"].value * ATOMIC_MASS_KG
    omega_r = TWO_PI * 6.9e6
    r0 = math.sqrt(HBAR_J_S / (2.0 * mass_total * omega_r))
    sensitivity = TWO_PI * slope_hz_per_gauss * 1e4
    omega_g = srinivas_gradient_rabi_rad_s(r0, GRADIENT_T_PER_M, sensitivity)
    assert r0 == pytest.approx(3.82844e-9, rel=1e-5)
    assert omega_g / TWO_PI == pytest.approx(2870.8, rel=1e-4)
    delta = TWO_PI * 0.95e6
    assert delta == pytest.approx(0.5 * (omega_r - TWO_PI * 5.0e6), rel=1e-9), (
        "delta ~ (omega_r - omega_g)/2 IS the 2 delta = omega_r - omega_g resonance of the dressed force"
    )
    omega_mu = intrinsic_dynamical_decoupling_ratio() * delta
    effective = srinivas_effective_coupling_rad_s(omega_g, omega_mu, delta)
    assert effective / TWO_PI == pytest.approx(1239.5, rel=1e-4)
    assert float(jv(0, 4.0 * omega_mu / delta)) == pytest.approx(0.0, abs=1e-9)
    assert omega_mu / TWO_PI == pytest.approx(571.1e3, rel=1e-3)
    # one closed loop at the maximally entangling ratio, against the paper's 740 us of Walsh segments
    loop_s = TWO_PI / (4.0 * effective)
    assert loop_s == pytest.approx(202e-6, rel=0.01)
    assert 0.1 < 2.0 * loop_s / 740e-6 < 1.0, "the eight-segment 740 us sequence is the right scale"


def test_gradient_couplings_scale_as_the_configured_gradient() -> None:
    """Device parameters in, everything derived: doubling the electrodes' gradient doubles every force coefficient and
    Omega_g, and leaves Omega_mu (a property of the microwave field, not of the gradient) alone."""
    dev = gradient_device()
    base = derive_gradient_drive(dev, (0, 1))
    doubled = derive_gradient_drive(
        dataclasses.replace(
            dev, gradient=dataclasses.replace(dev.gradient, gradient_t_per_m=2.0 * GRADIENT_T_PER_M)
        ),
        (0, 1),
    )
    assert doubled.microwave_rabi_rad_s == pytest.approx(base.microwave_rabi_rad_s, rel=1e-12)
    for key, value in base.coupling_rad_s.items():
        assert doubled.coupling_rad_s[key] == pytest.approx(2.0 * value, rel=1e-12)
    for mode, value in base.gradient_rabi_rad_s.items():
        assert doubled.gradient_rabi_rad_s[mode] == pytest.approx(2.0 * value, rel=1e-12)
    # the y and z modes are orthogonal to the gradient axis: no force at all
    for mode in (0, 1, 4, 5):
        assert abs(base.coupling_rad_s[(0, mode)]) < 1e-9 * abs(base.coupling_rad_s[(0, ROCK)]), mode
    # and the COM mode's force is IN phase on both ions (a spin-independent push, not a (sigma_z1 - sigma_z2) force)
    assert base.coupling_rad_s[(0, 3)] == pytest.approx(base.coupling_rad_s[(1, 3)], rel=1e-9)
    assert np.sign(base.coupling_rad_s[(0, ROCK)]) != np.sign(base.coupling_rad_s[(1, ROCK)])
