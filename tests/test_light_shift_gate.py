"""The sigma_z-force (ZZ) gates: the light-shift gate from the atomic layer and the near-field microwave-gradient gate
(PLAN.md Section 4.4).

Srinivas et al., Nature 597, 209 (2021) drive two ions with two microwave tones at +-delta from the ac-Zeeman-shifted
qubit frequency plus a magnetic-field gradient oscillating at omega_g near the motional frequency:

    H_I(t) = hbar Omega_g J_2(4 Omega_mu/delta)(sigma_z1 - sigma_z2)(a e^{i Delta t} + a^dag e^{-i Delta t}),
    Omega_g = (r_0/4)[grad(B_g . r_hat_q) . r_hat](d omega_0/dB),   Omega_mu = (B_x/2 hbar)<dn|mu_x|up>,

with intrinsic dynamical decoupling where J_0(4 Omega_mu/delta) = 0, i.e. Omega_mu/delta = 0.6012. The gradient fixture
is a 40Ca+ ZEEMAN qubit (S1/2 mJ = -1/2 <-> +1/2 at 5 G, 2.8025 MHz/G): the gradient couples through d omega_0/dB, which
vanishes at a clock point, and the 25Mg+ species table lacks cited P-level constants, so the paper's own 25Mg+ numbers
are checked against the hyperfine-Zeeman diagonalization directly.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import qutip as qt
from scipy.special import jv

from qutip_trap.calibration.entangling import calibrate_entangling_angle, exact_gate_check, gate_space
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.control.pulses import Drive, Pulse, Tone
from qutip_trap.control.schedule import GateDrive, ScheduleError, schedule
from qutip_trap.control.shaping import gate_modes, solve_amplitude_modulation
from qutip_trap.control.table import CalEntry, Segment, Waveform
from qutip_trap.device.model import BeamRoles, Device, Field, GradientField
from qutip_trap.device.presets import secular_trap
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.beams import Beam
from qutip_trap.light.microwave import (
    derive_gradient_drive,
    field_sensitivity_rad_s_per_t,
    gradient_drive,
)
from qutip_trap.light.raman import (
    derive_light_shift_drive,
    derive_raman_drive,
    light_shift_drive,
    two_photon_self_couplings_hz,
)
from qutip_trap.noise.sampling import NoiseSample, key_qubit_offset_hz, quiet_sample
from qutip_trap.species import MODULES, species
from qutip_trap.species.model import Level, Species
from qutip_trap.species.zeeman import HyperfineZeeman
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI
from tests.fixtures import make_calibration_table, make_detector, make_hardware, make_noise
from tests.m4_fixtures import CA40_729_E2_BEAM, derived_seeds, table_with_waveform, two_ion_device
from tests.oracles import (
    intrinsic_dynamical_decoupling_ratio,
    srinivas_effective_coupling_rad_s,
    srinivas_gradient_rabi_rad_s,
)
from tests.test_zeeman_anchors import MG25_G_J_ASSUMED

C_M_PER_S = 299792458.0
X_COM = 3
X_ROCK = 2
"""The transverse-x mode with the ANTISYMMETRIC pattern (-1, +1)/sqrt2: the one a (sigma_z1 - sigma_z2) force drives."""


def ca_light_shift_device() -> Device:
    """Two 40Ca+ ions (optical S1/2-D5/2 qubit), a counter-propagating 398.5 nm pair along x (3 THz below the S-P1/2 line, both
    circular about B || x) for the light-shift force, and the ``CA40_729_E2_BEAM`` quadrupole beam for the E2 single-qubit
    pulses - rotated into the xy plane so that its Rabi frequency is DERIVED (200144 Hz at 166 mW) rather than supplied."""
    ca = species("40Ca+")
    trap = secular_trap()
    crystal = solve_crystal(trap, (ca, ca))
    lam = C_M_PER_S / (C_M_PER_S / 396.959e-9 - 3.0e12)
    s = 1.0 / math.sqrt(2.0)
    b1 = Beam(lam, (1.0, 0.0, 0.0), (0.0, s, 1j * s), 200e-6, 10e-3, (0.0, 0.0, 0.0))
    b2 = Beam(lam, (-1.0, 0.0, 0.0), (0.0, s, 1j * s), 200e-6, 10e-3, (0.0, 0.0, 0.0))
    return Device(
        crystal=crystal,
        trap=trap,
        field=Field(5.0, (1.0, 0.0, 0.0)),
        beams=(b1, b2, CA40_729_E2_BEAM),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
    )


@pytest.fixture(scope="module")
def ca_device():  # type: ignore[no-untyped-def]
    dev = ca_light_shift_device()
    ent = {
        i: GateDrive(
            "light_shift",
            (0, 1),
            light_shift=derive_light_shift_drive(dev, i, (0, 1), scattering=False).light_shift,
        )
        for i in (0, 1)
    }
    sq = {i: GateDrive("optical_E2", (2,)) for i in (0, 1)}
    modes = gate_modes(dev, (0, 1), (0, 1))
    return dev, ent, sq, modes


def test_light_shift_couplings_from_the_atomic_layer(ca_device) -> None:  # type: ignore[no-untyped-def]
    """Essentially only the S1/2 level of the optical qubit sees the 398 nm two-photon shift (weights (-1.99748, 0.00252)), the
    S-to-D5/2 Raman spin flip the same beams drive is 411 THz off resonance, and the differential coupling is half the
    difference of the two levels' self-couplings (Zhu 2006 Eq. 2)."""
    dev, _ent, _sq, modes = ca_device
    dn, up = two_photon_self_couplings_hz(dev, 0, (0, 1))
    # the D5/2 level's own two-photon shift is 1.3e-3 of the S level's
    assert abs(dn) > 1e3 and abs(up) < 2e-3 * abs(dn)
    ls = derive_light_shift_drive(dev, 0, (0, 1), scattering=False)
    assert ls.kind == "light_shift" and ls.light_shift is not None
    assert ls.rabi_hz == pytest.approx((up - dn) / 2.0)
    assert dn == pytest.approx(-10554.4, rel=1e-4) and up == pytest.approx(13.311, rel=1e-3)
    assert ls.light_shift.level_weights == pytest.approx((-1.997481, 0.002519), abs=1e-6)
    assert ls.light_shift.level_weights[1] - ls.light_shift.level_weights[0] == pytest.approx(2.0, rel=1e-12)
    assert ls.light_shift.spin_flip_weight == pytest.approx(-0.16812, rel=1e-3), (
        "the same beams drive an S-D two-photon matrix element, 411 THz off resonance (excitation ~1e-23)"
    )
    assert ls.light_shift.qubit_freq_hz == pytest.approx(C_M_PER_S / 729.348e-9, rel=1e-3)
    assert ls.stark_shift_hz == pytest.approx((up - dn).real, rel=1e-6), (
        "the static shift delta(D) - delta(S) is the difference of the two levels' two-photon self-couplings (equal beams)"
    )
    assert ls.stark_shift_hz == pytest.approx(10567.7, rel=1e-4)
    rr = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    assert rr.rabi_hz == pytest.approx(-888.3, rel=2e-2)
    assert modes.modes == (X_ROCK, X_COM) and abs(modes.eta[0][1]) == pytest.approx(0.1448, abs=1e-3)
    drive = light_shift_drive(ls, beat_hz=2.98e6)
    assert (
        drive.kind == "light_shift"
        and drive.tones[0].detuning_hz == 2.98e6
        and drive.light_shift is ls.light_shift
    )
    with pytest.raises(ValueError, match="light_shift drive carries"):
        Drive("raman", (0,), (Tone(0.0, 0.0, 1.0),), (0, 1), 0.0, {}, light_shift=ls.light_shift)
    # a clock qubit under linear light has no differential shift: the derivation refuses it
    with pytest.raises(ValueError, match="no state-dependent force"):
        derive_light_shift_drive(two_ion_device(), 0, (0, 1), scattering=False)


def test_builder_light_shift_operator_is_the_level_weighted_force(ca_device) -> None:  # type: ignore[no-untyped-def]
    """The drive term is (1/2) Omega_LS e^{-i(mu t - phi)} [w_dn P_0 + w_up P_1] (x) D + h.c. = Omega_LS cos(...) [sigma_z + (w_up + w_dn)/2] (x) D:
    at t = 0 with phi = 0 the Hamiltonian's spin part on ion 0 is Omega_LS (w_dn P_0 + w_up P_1) (x) (D + D^dag)/2."""
    dev, _ent, _sq, _modes = ca_device
    ls = derive_light_shift_drive(dev, 0, (0, 1), scattering=False)
    omega_ls = TWO_PI * abs(ls.rabi_hz)
    w_dn, w_up = ls.light_shift.level_weights  # type: ignore[union-attr]
    drive = light_shift_drive(ls, beat_hz=2.98e6, include_stark=False)
    space = HilbertSpace((2, 2), (ModeTruncation(X_COM, 6, (0, 1), 0.2),), None, (0, 1, 2, 4, 5))
    built = build_hamiltonian(
        dev, [Pulse(drive, 0.0, 1e-6, "ls", ())], space, options=BuilderOptions(frozen_debye_waller=False)
    )
    h0 = built.H(0.0)
    d_op = space.displacement_factor(X_COM, ls.etas[X_COM])
    force = complex(w_dn) * qt.basis(2, 0).proj() + complex(w_up) * qt.basis(2, 1).proj()
    expected = TWO_PI * dev.crystal.modes[X_COM].omega_hz * space.number(X_COM) + omega_ls * space.embed_many(
        {0: force, 2: 0.5 * (d_op + d_op.dag())}
    )
    assert (h0 - expected).norm() < 1e-9 * expected.norm()
    assert not any("spin flip" in a and "keeps" in a for a in built.approximations)


@pytest.mark.slow
def test_light_shift_zz_gate_in_the_echo_form(ca_device) -> None:  # type: ignore[no-untyped-def]
    """Two light-shift pulses of two-body angle pi/8 around a pi pulse on both ions give ZZ(pi/2) = exp(-i (pi/4) Z Z) on the
    optical qubit; the sign follows the detuning side; the exact spot check calibrates the pulse angle; the AM solver closes the
    rocking mode for a light-shift force too."""
    dev, ent, sq, modes = ca_device
    # the table's 729 nm E2 entries are the derived values (200144 Hz at 166 mW in a 200 um waist), so the played chain is
    # the identity; the E2 drive has no differential light shift
    rabi, stark = derived_seeds(dev, sq)
    assert rabi[(0, 2)] == pytest.approx(2.00144e5, rel=1e-4), (
        "the derived E2 Rabi frequency, not a supplied one"
    )
    wf = Waveform.symmetric(
        modes,
        gate_mode=X_COM,
        loops=1,
        epsilon_hz=20e3,
        kind="light_shift",
        detuning_side="outside",
        chi_target_rad=math.pi / 8,
    )
    assert wf.kind == "light_shift" and wf.segments[0].legs == ("blue",)
    assert wf.chi_total_rad < 0.0
    space = gate_space(modes, 2, waveform=wf)
    plain = gate_space(modes, 2, waveform=wf, force_weight=1.0)
    assert [t.d for t in space.resolved] >= [t.d for t in plain.resolved]
    assert space.dims != plain.dims, (
        "the light-shift force operator's spectral radius is 2 for weights (-2, 0), so the cap the excursion asks for is "
        "larger than the +-1 assumption's"
    )
    table = table_with_waveform((0, 1), wf, rabi_hz=rabi, stark_hz=stark)
    check, trace = exact_gate_check(
        dev, wf, (0, 1), ent, table, space=space, chi_target_rad=math.pi / 8, single_qubit_drives=sq
    )
    assert max(trace.boundary_population.values()) < 1e-10, (
        "the Fock cap holds the sigma_z force's excursion, which is TWICE the +-1 assumption of the closed-form "
        "trajectory for level weights (-2, 0) (control.shaping.LIGHT_SHIFT_FORCE_WEIGHT)"
    )
    assert 0.85 < check.chi_rad / (math.pi / 8) < 0.95, (
        "the surrogate over-predicts the angle by the Debye-Waller corrections at eta = 0.145"
    )
    assert check.fidelity > 0.97 and check.leakage < 0.02
    with pytest.raises(ValueError, match="spin-echo"):
        exact_gate_check(dev, wf, (0, 1), ent, table, space=space, chi_target_rad=math.pi / 8)
    inside = Waveform.symmetric(
        modes,
        gate_mode=X_COM,
        loops=1,
        epsilon_hz=20e3,
        kind="light_shift",
        detuning_side="inside",
        chi_target_rad=math.pi / 8,
    )
    check_in, _ = exact_gate_check(
        dev,
        inside,
        (0, 1),
        ent,
        table_with_waveform((0, 1), inside, rabi_hz=rabi, stark_hz=stark),
        space=space,
        chi_target_rad=math.pi / 8,
        single_qubit_drives=sq,
    )
    assert check_in.fidelity < 0.05, "exp(+i |chi| Z Z) is ZZ of the opposite sign"
    # AM closure of both x modes with the sigma_z force, calibrated exactly
    am = solve_amplitude_modulation(
        modes, mu_hz=3.02e6, duration_s=100e-6, kind="light_shift", chi_target_rad=math.pi / 8
    )
    assert am.chi_rad < 0.0 and all(abs(a) < 1e-10 for a in am.integrals.alpha.values())
    space_am = gate_space(modes, 2, waveform=am.waveform)
    run = calibrate_entangling_angle(
        dev,
        am.waveform,
        (0, 1),
        ent,
        table_with_waveform((0, 1), am.waveform, rabi_hz=rabi, stark_hz=stark),
        space=space_am,
        chi_target_rad=math.pi / 8,
        single_qubit_drives=sq,
        tolerance_rad=3e-4,
    )
    assert run.converged and run.checks[-1].fidelity > 0.99 and run.checks[-1].leakage < 5e-3


# ---- the microwave-gradient gate -------------------------------------------------------------------------------------

DELTA_HZ = 200e3
"""delta/2pi, the microwave tones' symmetric detuning; the gradient sits at omega_m - 2 delta so the dressed force is resonant."""
GRADIENT_T_PER_M = 152.0
"""Srinivas et al.'s measured gradient amplitude, 152(15) T/m."""
RABI_HZ_PER_MICROTESLA = 14012.475693
"""Omega/2pi of the 40Ca+ Zeeman qubit at 5 G under a 1 uT field along y (mu_B/h x 1 uT for a Delta m_J = 1 transition)."""


def zeeman_calcium() -> Species:
    """40Ca+ with its S1/2 Zeeman pair as the qubit: field-sensitive (2.8025 MHz/G) and magnetic-dipole coupled, which a
    gradient gate needs (the optical S1/2-D5/2 qubit, two fine-structure levels, has no magnetic-dipole drive)."""
    return dataclasses.replace(species("40Ca+"), qubit=("S1/2 mJ=-1/2", "S1/2 mJ=1/2"))


def gradient_device(*, ratio: float | None = None) -> Device:
    """Two 40Ca+ Zeeman qubits with near-field gradient electrodes at ``ratio`` = Omega_mu/delta (default: the
    intrinsic-dynamical-decoupling point 0.6012)."""
    ratio = intrinsic_dynamical_decoupling_ratio() if ratio is None else ratio
    ion = zeeman_calcium()
    trap = secular_trap()
    crystal = solve_crystal(trap, (ion, ion))
    tone_rabi_hz = 2.0 * ratio * DELTA_HZ
    b1 = (0.0, tone_rabi_hz / RABI_HZ_PER_MICROTESLA * 1e-6, 0.0)
    gradient = GradientField(
        gradient_t_per_m=GRADIENT_T_PER_M,
        frequency_hz=crystal.modes[X_ROCK].omega_hz - 2.0 * DELTA_HZ,
        axis=(1.0, 0.0, 0.0),
        b1_tesla_lab=b1,
        phase_rad=0.0,
    )
    return Device(
        crystal=crystal,
        trap=trap,
        field=Field(5.0, (1.0, 0.0, 0.0)),
        beams=(),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
        gradient=gradient,
    )


def gradient_space() -> HilbertSpace:
    return HilbertSpace((2, 2), (ModeTruncation(X_ROCK, 8, (0, 1), 0.2),), None, (0, 1, 3, 4, 5))


def test_gradient_couplings_are_derived_and_r0_carries_the_total_two_ion_mass() -> None:
    """Omega_g = (r_0/4) grad(B)(d omega_0/dB) with r_0 from the TOTAL two-ion mass, and the laboratory force coefficient
    w_{i,m} is +-2 Omega_g exactly on the antisymmetric mode, so (sigma_z1 - sigma_z2) is the right normalization. Every
    coupling is derived: d omega_0/dB from the hyperfine-Zeeman diagonalization, Omega_mu from the magnetic-dipole matrix
    element of the configured field, w_{i,m} from the crystal's mass-weighted mode pattern."""
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
    mass_total = float(sum(dev.crystal.masses_kg[i] for i in (0, 1)))
    r0 = math.sqrt(HBAR_J_S / (2.0 * mass_total * dev.crystal.modes[X_ROCK].omega_rad_s))
    omega_g = srinivas_gradient_rabi_rad_s(r0, GRADIENT_T_PER_M, dd.field_sensitivity_rad_s_per_t)
    assert dd.gradient_rabi_rad_s[X_ROCK] == pytest.approx(omega_g, rel=1e-12)
    assert omega_g / TWO_PI == pytest.approx(5035.3, rel=1e-4)
    for ion, sign in ((0, -1.0), (1, +1.0)):
        assert dd.coupling_rad_s[(ion, X_ROCK)] / omega_g == pytest.approx(2.0 * sign, rel=1e-9), (
            "w_{i,m} = +-2 Omega_g on the antisymmetric mode: the total-mass r_0"
        )
    assert srinivas_effective_coupling_rad_s(
        omega_g, dd.microwave_rabi_rad_s, delta
    ) / TWO_PI == pytest.approx(2174.0, rel=1e-4)
    with pytest.raises(ValueError, match="Device.gradient"):
        derive_gradient_drive(dataclasses.replace(dev, gradient=None), (0, 1))


def test_field_sensitivity_vanishes_on_a_clock_transition() -> None:
    """At a clock point d omega_0/dB = 0 and Omega_g vanishes with it, which is why a gradient gate needs a field-sensitive
    qubit (Srinivas et al. drive |F=3,mF=3> <-> |F=2,mF=2> of 25Mg+, not its 212.78 G clock line)."""
    yb = species("171Yb+")
    field = Field(1e-6, (1.0, 0.0, 0.0))
    assert abs(field_sensitivity_rad_s_per_t(yb, field)) < 1e3, (
        "the 171Yb+ clock line is flat in B at zero field"
    )
    assert abs(field_sensitivity_rad_s_per_t(zeeman_calcium(), field)) > 1e11


def test_builder_dressed_gradient_operator_is_the_j2_weighted_sigma_z_force() -> None:
    """With ``gradient_form="dressed"`` the Hamiltonian at t = 0 is exactly the free motion plus
    Omega_g J_2(4 Omega_mu/delta)(sigma_z1 - sigma_z0)(a + a^dag), and the elimination is recorded in ``approximations``."""
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
    a = space.annihilation(X_ROCK)
    force = srinivas_effective_coupling_rad_s(
        dd.gradient_rabi_rad_s[X_ROCK], dd.microwave_rabi_rad_s, delta
    ) * (space.sigma_z(1) - space.sigma_z(0))
    expected = dev.crystal.modes[X_ROCK].omega_rad_s * space.number(X_ROCK) + force * (a + a.dag())
    assert (built.H(0.0) - expected).norm() < 1e-9 * expected.norm()
    assert built.n_drive_terms == 4, "one (sigma_z (x) a, its conjugate) pair per ion, no spin flip"
    assert any("adiabatically eliminated" in note for note in built.approximations)
    assert any("frozen mode 3" in note for note in built.approximations), (
        "the COM mode is frozen in this space: its gradient force is dropped and recorded"
    )


def test_builder_bare_gradient_keeps_both_microwave_tones_and_the_force() -> None:
    """The default ``gradient_form="bare"`` builds the two microwave tones as ordinary carrier terms BESIDE the laboratory
    sigma_z force w_{i,m} cos(omega_g t + phi_g), so J_2(4 Omega_mu/delta) and the J_0 = 0 decoupling emerge from the exact
    dynamics rather than being put in by hand."""
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
    a = space.annihilation(X_ROCK)
    force = dd.coupling_rad_s[(1, X_ROCK)] * (space.sigma_z(1) - space.sigma_z(0)) * (a + a.dag())
    residual = h0 - dev.crystal.modes[X_ROCK].omega_rad_s * space.number(X_ROCK) - force
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
    something else."""
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
    """The intrinsic dynamical decoupling from the exact dynamics rather than from a Bessel-zero lookup.

    A slow (here static) qubit-frequency drift enters the dressed frame weighted by J_0(4 Omega_mu/delta), so at
    Omega_mu/delta = 0.6012 the gate is first-order insensitive to it and at 0.3 it is not. Measured over 100 us with a
    2 kHz offset on both ions, starting from |+ +>|0>: infidelity 6.9e-5 at the decoupling point against 3.06e-1 at
    Omega_mu/delta = 0.3, a suppression of 4400."""
    space = gradient_space()
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi0 = space.product_state(qt.tensor(plus, plus), {X_ROCK: qt.basis(8, 0)})
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
    stored per mode. Its closure is Srinivas's Walsh sign modulation, not the AM/FM/PM algebra (the force is a Bessel
    function of the tone amplitude, not linear in it), so the waveform is written out rather than solved for."""
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
        duration_s=50e-6,
        phi_s=entry,
        phi_m=entry,
        chi_m={X_ROCK: chi_rad},
        alpha_m={X_ROCK: 0j},
        kind="gradient",
    )


def test_scheduler_plays_a_gradient_waveform_through_the_sigma_z_echo_path() -> None:
    """The scheduler plays a ``gradient`` waveform as ZZ(theta) on the light-shift gate's two-loop spin-echo path (the
    same sigma_z force), one ``gradient`` Drive per ion and loop; the sign comes only from the detuning side."""
    dev = gradient_device()
    wf = gradient_waveform(dev, chi_rad=-math.pi / 8.0)
    table = dataclasses.replace(
        make_calibration_table(),
        ms={(0, 1): wf},
        rabi={
            (i, -1): CalEntry(50e3, 1.0, "calibrated", "rabi_scan", "conv.rabi_frequency", 0.0, 0)
            for i in (0, 1)
        },
    )
    micro = {i: GateDrive("microwave", ()) for i in (0, 1)}
    ent = {i: GateDrive("gradient", ()) for i in (0, 1)}
    dev = dataclasses.replace(dev, roles=BeamRoles(gate=micro, entangling=ent))
    circuit = Circuit(2, (Operation("zz", (0, 1), (math.pi / 2,)),), (0, 1))
    sched = schedule(circuit, dev, table)
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
        schedule(circuit, dataclasses.replace(dev, roles=BeamRoles(gate=micro, entangling=micro)), table)
    ms_circuit = Circuit(2, (Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2)),), (0, 1))
    with pytest.raises(ScheduleError, match="microwave-gradient one"):
        schedule(ms_circuit, dev, table)
    # the sign guard of the sigma_z path
    wrong = gradient_waveform(dev, chi_rad=+math.pi / 8.0)
    with pytest.raises(ScheduleError, match="opposite sign"):
        schedule(circuit, dev, dataclasses.replace(table, ms={(0, 1): wrong}))


def test_srinivas_own_parameters() -> None:
    """Srinivas et al.'s own numbers. Their qubit is |F=3, mF=3> <-> |F=2, mF=2> of the 25Mg+ S1/2 manifold at 21.3 mT, NOT
    the 212.78 G clock line: the hyperfine-Zeeman diagonalization of the cited 25Mg+ constants puts that transition at
    1.326467 GHz with d f_0/dB = 1.97333 MHz/G at 212.78 G against the paper's omega~_0 ~ 2 pi x 1.326 GHz, and the clock
    pair at the same field at 1.686462 GHz with 21 Hz/G.

    With the paper's gradient (152(15) T/m), mode (omega_r ~ 2 pi x 6.9 MHz) and gradient frequency
    (omega_g = 2 pi x 5 MHz), Omega_g/2pi = 2870.8 Hz and the dressed coupling at the decoupling point is
    Omega_g J_2(4 x 0.6012)/2pi = 1239.5 Hz. The paper's delta ~ (omega_r - omega_g)/2 = 2 pi x 950 kHz is exactly the
    2 delta = omega_r - omega_g resonance the dressed force needs, and one closed loop at the maximally entangling
    (Omega_eff/Delta_gate) = 1/4 takes 202 us, the scale of the paper's 740 us eight-segment Walsh sequence."""
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
    ), "the clock point at the same field, first-order insensitive (21 Hz/G)"
    assert abs(spectrum.dE_dB_hz_per_g[i_c_up] - spectrum.dE_dB_hz_per_g[i_c_dn]) < 25.0
    # Omega_g from the paper's gradient, mode and (total) two-ion mass
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
    """Doubling the electrodes' gradient doubles every force coefficient and Omega_g, and leaves Omega_mu (a property of
    the microwave field, not of the gradient) alone."""
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
        assert abs(base.coupling_rad_s[(0, mode)]) < 1e-9 * abs(base.coupling_rad_s[(0, X_ROCK)]), mode
    # and the COM mode's force is IN phase on both ions (a spin-independent push, not a (sigma_z1 - sigma_z2) force)
    assert base.coupling_rad_s[(0, X_COM)] == pytest.approx(base.coupling_rad_s[(1, X_COM)], rel=1e-9)
    assert np.sign(base.coupling_rad_s[(0, X_ROCK)]) != np.sign(base.coupling_rad_s[(1, X_ROCK)])
