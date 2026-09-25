"""The light-shift (sigma_z sigma_z) gate from the atomic layer and the Section 4.4.4 closed forms (PLAN.md Sections 4.4.4, 4.4.5,
4.4.6, 9.4, 9.17 'Baldwin echo')."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.calibration.entangling import calibrate_entangling_angle, exact_gate_check, gate_space
from qutip_trap.control.native import equal_up_to_global_phase
from qutip_trap.control.native import zz as native_zz
from qutip_trap.control.pulses import Drive, Tone
from qutip_trap.control.schedule import GateDrive
from qutip_trap.control.shaping import gate_modes, solve_amplitude_modulation
from qutip_trap.control.table import Waveform
from qutip_trap.device.model import Device, Field
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.light.beams import Beam
from qutip_trap.light.raman import (
    derive_light_shift_drive,
    derive_raman_drive,
    light_shift_drive,
    two_photon_self_couplings_hz,
)
from qutip_trap.species import species
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.trap.model import Trap
from qutip_trap.units import TWO_PI
from qutip_trap.validation.two_qubit_closed_forms import (
    baldwin_echo_unitary,
    baldwin_loop_phase,
    ballance_dephasing_coefficient,
    ballance_dephasing_error,
    ballance_heating_error,
    ballance_thermal_error,
    hughes_gate_angle_derived,
    hughes_gate_angle_printed,
    intrinsic_dynamical_decoupling_ratio,
    ms_two_body_angle,
    srinivas_effective_coupling_rad_s,
    srinivas_gradient_rabi_rad_s,
)
from tests.fixtures import make_detector, make_hardware, make_noise
from tests.m4_fixtures import CA40_729_E2_BEAM, derived_seeds, table_with_waveform

C_M_PER_S = 299792458.0


def ca_light_shift_device() -> Device:
    """Two 40Ca+ ions (optical S1/2-D5/2 qubit), a counter-propagating 398.5 nm pair along x (3 THz below the S-P1/2 line, both
    circular about B || x) for the light-shift force, and the ``CA40_729_E2_BEAM`` quadrupole beam for the E2 single-qubit
    pulses - rotated into the xy plane so that its Rabi frequency is DERIVED (200144 Hz at 166 mW) rather than supplied."""
    ca = species("40Ca+")
    trap = Trap(
        omega_hz=(3.0e6, 2.9e6, 1.0e6),
        axis_angle_rad=0.0,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    crystal = solve_crystal(trap, (ca, ca))
    lam = C_M_PER_S / (C_M_PER_S / 396.959e-9 - 3.0e12)
    s = 1.0 / math.sqrt(2.0)
    b1 = Beam(lam, (1.0, 0.0, 0.0), (0.0, s, 1j * s), 200e-6, 10e-3, (0.0, 0.0, 0.0))
    b2 = Beam(lam, (-1.0, 0.0, 0.0), (0.0, s, 1j * s), 200e-6, 10e-3, (0.0, 0.0, 0.0))
    b729 = CA40_729_E2_BEAM
    return Device(
        crystal=crystal,
        trap=trap,
        field=Field(5.0, (1.0, 0.0, 0.0), None),
        beams=(b1, b2, b729),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
    )


X_COM = 3
X_ROCK = 2


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
    difference of the two levels' self-couplings (Section 4.4.4; Zhu 2006 Eq. 2).

    Four pins moved on 2026-09-08 with the 40Ca+ P-level rates (ledger conv.ca40_linewidth_reading): the S1/2 shift is set by
    the 397/393 nm dipole elements, whose squares scale with the PARTIAL rates, and the D5/2 shift by the 854/850 nm ones.
      - the S1/2 self-coupling -9973.1 -> -10554.4 Hz and so the differential Omega_LS/2pi 4993.0 -> 5283.8 Hz, +5.8 %:
        the 397 nm partial rate rose 6.4 % (20.276 -> 21.569 MHz) and the 393 nm one 2.4 % (21.872 -> 22.407).
      - level_weights (-1.99740, 0.00260) -> (-1.99748, 0.00252): the D5/2 share of the pair fell, because the 854/850 nm
        partial rates rose only 2.4 % against the S levels' 6.4 % and 2.4 % mix.
      - spin_flip_weight -0.17366 -> -0.16812, -3.2 %, for the same reason (an S-D product over an S-S one).
      - the Raman rabi_hz -867.1 -> -888.3 Hz, +2.4 %, is the S-D two-photon element, which carries one 397/393 factor and
        one 854/850 factor."""
    dev, _ent, _sq, modes = ca_device
    dn, up = two_photon_self_couplings_hz(dev, 0, (0, 1))
    # the D5/2 level's own two-photon shift from the newly tabulated D-P dipole couplings is 1.3e-3 of the S level's
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
    from tests.m4_fixtures import two_ion_device

    with pytest.raises(ValueError, match="no state-dependent force"):
        derive_light_shift_drive(two_ion_device(), 0, (0, 1), scattering=False)


def test_builder_light_shift_operator_is_the_level_weighted_force(ca_device) -> None:  # type: ignore[no-untyped-def]
    """The drive term is (1/2) Omega_LS e^{-i(mu t - phi)} [w_dn P_0 + w_up P_1] (x) D + h.c. = Omega_LS cos(...) [sigma_z + (w_up + w_dn)/2] (x) D:
    at t = 0 with phi = 0 the Hamiltonian's spin part on ion 0 is Omega_LS (w_dn P_0 + w_up P_1) (x) (D + D^dag)/2."""
    dev, _ent, _sq, _modes = ca_device
    ls = derive_light_shift_drive(dev, 0, (0, 1), scattering=False)
    omega_ls = TWO_PI * abs(ls.rabi_hz)
    w_dn, w_up = ls.light_shift.level_weights  # type: ignore[union-attr]
    from qutip_trap.control.pulses import Pulse

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
    """Section 4.4.4: two light-shift pulses of two-body angle pi/8 around a pi pulse on both ions give ZZ(pi/2) = exp(-i (pi/4) Z Z) on
    the optical qubit; the sign follows the detuning side; the exact spot check calibrates the pulse angle; the AM solver closes the
    rocking mode for a light-shift force too."""
    dev, ent, sq, modes = ca_device
    # every number the table carries for the 729 nm E2 echo pulses is DERIVED from the beam's geometry and power
    # (200144 Hz at 166 mW in a 200 um waist), so the played chain of M8 (requested -> physical) is the identity here
    # and the fixture's 200 kHz is delivered rather than supplied; the E2 drive has no differential light shift
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
    assert wf.kind == "light_shift" and wf.segments is not None and wf.segments[0].legs == ("blue",)
    assert wf.chi_total_rad < 0.0
    space = gate_space(modes, 2, waveform=wf)
    plain = gate_space(modes, 2, waveform=wf, force_weight=1.0)
    assert [t.d for t in space.resolved] >= [t.d for t in plain.resolved]
    assert space.dims != plain.dims, (
        "the light-shift force operator's spectral radius is 2 for weights (-2, 0), so the cap the excursion asks for is "
        "larger than the +-1 assumption's (M4 finding)"
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


def test_baldwin_echo_gives_diag_1_i_i_1() -> None:
    """Section 9.17 'Baldwin echo': H = eta Omega sum_j (-1)^j (1 + sigma_z^j)(a e^{i delta t} + h.c.) integrated through R_x(pi) U R_-x(pi) U with
    8 pi (eta Omega/delta)^2 = pi/4 gives diag(1, i, i, 1) up to a global phase; Phi_loop = 2 pi (eta Omega/delta)^2 on S^2 per loop."""
    delta = TWO_PI * 20e3
    ratio = 1.0 / math.sqrt(32.0)  # 8 pi (eta Omega/delta)^2 = pi/4
    eta = 0.1
    omega = ratio * delta / eta
    u = baldwin_echo_unitary(eta, omega, delta, d=24)
    target = np.diag([1.0, 1j, 1j, 1.0])
    # ONE sign, not a disjunction: for tones ABOVE the mode (delta > 0 here) the echo gives the CONJUGATE of the
    # paper's printed diag(1, i, i, 1), which is what ledger anchor.m4.baldwin_echo records; the printed sign is the
    # other detuning side, and asserting it here is the negative control
    assert equal_up_to_global_phase(u, target.conj(), atol=2e-5)
    assert not equal_up_to_global_phase(u, target, atol=1e-2), (
        "the two detuning sides are distinguishable: the disjunction the first pass asserted was blind to a sign error"
    )
    below = baldwin_echo_unitary(eta, omega, -delta, d=24)
    assert equal_up_to_global_phase(below, target, atol=2e-5), "tones below the mode give the printed sign"
    assert baldwin_loop_phase(eta, omega, delta) == pytest.approx(2.0 * math.pi / 32.0)
    assert 4.0 * baldwin_loop_phase(eta, omega, delta) == pytest.approx(math.pi / 4.0), (
        "8 pi (eta Omega/delta)^2 over the two echo loops"
    )
    # which sign the detuning side gives, as one value
    assert equal_up_to_global_phase(u, native_zz(-math.pi / 2), atol=2e-5)
    assert equal_up_to_global_phase(below, native_zz(math.pi / 2), atol=2e-5)


def test_ballance_srinivas_hughes_closed_forms() -> None:
    """Section 9.4 rows: Ballance's budget forms, Srinivas's J_2 Hamiltonian and the IDD ratio 0.601, Hughes's theta_g = pi/4 with
    the factor-2 between the printed integral and the displayed Hamiltonian (recorded)."""
    assert ballance_thermal_error(0.123, 0.02) == pytest.approx(0.25 * math.pi**2 * 0.123**4 * 0.02 * 1.04)
    assert ballance_heating_error(2.2, 100e-6, 2) == pytest.approx(2.2 * 100e-6 / 4.0)
    with pytest.raises(ValueError):
        ballance_heating_error(2000.0, 100e-6, 1)
    assert [ballance_dephasing_coefficient(k) for k in (1, 2, 4)] == pytest.approx(
        [0.6875, 0.296875, 0.13671875]
    )
    assert ballance_dephasing_error(100e-6, 200e-3, 2) == pytest.approx(0.297 * 100e-6 / 200e-3, rel=2e-3), (
        "0.15e-3, the paper's line"
    )
    assert intrinsic_dynamical_decoupling_ratio() == pytest.approx(0.6012, abs=1e-4)
    om_g = srinivas_gradient_rabi_rad_s(1e-8, 35.0, TWO_PI * 1e10)
    assert om_g == pytest.approx(0.25 * 1e-8 * 35.0 * TWO_PI * 1e10)
    from scipy.special import jv

    assert srinivas_effective_coupling_rad_s(om_g, 0.6012 * 1e5, 1e5) == pytest.approx(
        om_g * jv(2, 4 * 0.6012)
    )
    # Hughes: a constant-delta square pulse closed after K loops has theta_g = pi K (Omega_g/delta)^2 = int Omega_g^2/(2 delta) dt
    delta = TWO_PI * 20e3
    loops = 1
    omega_g = delta / (2.0 * math.sqrt(loops))  # Omega_g = eta Omega at the maximally entangling closure
    tau = 2.0 * math.pi * loops / delta
    derived = hughes_gate_angle_derived(lambda t: omega_g, lambda t: delta, tau)
    assert derived == pytest.approx(math.pi / 4.0, rel=1e-9)
    assert derived == pytest.approx(ms_two_body_angle(1.0, 1.0, omega_g, delta, loops), rel=1e-9)
    printed = hughes_gate_angle_printed(lambda t: omega_g, lambda t: delta, lambda t: 0.0, tau)
    assert printed == pytest.approx(2.0 * derived, rel=1e-9), (
        "the printed integral is twice the displayed Hamiltonian's angle"
    )
