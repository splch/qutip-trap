"""The Hamiltonian builder and engine against the closed forms of Sections 4.3.1, 4.2.7, 6.2 and 9.2 (JOINT_EXACT, one ion)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt
from scipy.special import jv

from qutip_trap.api import HilbertSpace, ModeTruncation, Pulse, RfDrive, Schedule, SeedSpec, SolverOptions
from qutip_trap.dynamics.channels import device_heating_rates, heating_channels, qubit_dephasing_channels
from qutip_trap.dynamics.engine import JointExactEngine
from qutip_trap.dynamics.evolve import evolve
from qutip_trap.dynamics.frames import interaction_picture
from qutip_trap.dynamics.hamiltonian import (
    BuilderOptions,
    CurvatureSpec,
    build_hamiltonian,
    carrier_debye_waller_frozen,
)
from qutip_trap.hilbert.operators import debye_waller_factor, rabi_matrix_element, thermal_populations
from qutip_trap.light.microwave import square_microwave_drive
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.noise.sampling import (
    KEY_RABI_SCALE,
    NoiseSample,
    key_frozen_n,
    key_qubit_offset_hz,
    quiet_sample,
)
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI
from qutip_trap.validation.spin_motion_closed_forms import (
    cetina_population,
    cetina_theta,
    frozen_thermal_population,
    gaussian_curvature_per_m2,
    generalized_rabi_rad_s,
    resonant_transition_amplitude,
    sideband_rabi_rad_s,
    two_level_population,
)
from tests.m2_fixtures import microwave_device, single_ion_raman_device, two_ion_raman_device

KX = 1  # the 3 MHz x mode of the single-ion fixture
WX = TWO_PI * 3.0e6


@pytest.fixture(scope="module")
def raman():  # type: ignore[no-untyped-def]
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    space = HilbertSpace((2,), (ModeTruncation(KX, 12, (0, 3), 0.2),), None, (0, 2))
    return dev, dd, space


def _run(
    dev,
    drive,
    t,
    space,
    n0=0,
    opts=None,
    n_store=2,
    thermal=None,
    sample=None,
    channels=(),
    store_times=(),
    sopts=None,
):  # type: ignore[no-untyped-def]
    pulse = Pulse(drive, 0.0, t, "p", ())
    state = space.initial_state([0], fock={KX: n0} if thermal is None else None, thermal=thermal)
    eng = JointExactEngine(
        builder_options=opts,
        store_per_segment=n_store,
        channels=tuple(channels),
        store_times_s=tuple(store_times),
    )
    tr = eng.run_pulses(
        dev,
        Schedule((pulse,), (), (), {0: 0.0}),
        state,
        space,
        sample or quiet_sample(),
        SeedSpec(0),
        sopts or SolverOptions(),
    )
    return tr, eng


def test_carrier_flopping_with_debye_waller_and_rabi_scale(raman) -> None:  # type: ignore[no-untyped-def]
    dev, dd, space = raman
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]
    t_pi = math.pi / (om * math.exp(-(eta**2) / 2))
    tr, eng = _run(dev, square_drive(dd, include_stark=False), t_pi, space, n_store=21)
    pred = np.sin(0.5 * om * math.exp(-(eta**2) / 2) * tr.times_s) ** 2
    # The residual is the off-resonant sidebands, ~(eta Omega/omega_m)^2. The bound was 3e-6 while 171Yb+
    # had no P3/2 record: the M0a fix of 2026-09-08 (audit item E4) gave the Raman intermediate sum its
    # second path, and for this fixture's polarization the two paths ADD (the P3/2 partial sum is +0.4577 of
    # the P1/2 one, the ratio (w_32/w_12)(Delta_12/Delta_32)^2 with Delta/2pi = +33.19 and -66.65 THz), so
    # carrier_rabi_hz rose from 30278.49 to 44136.18 Hz and the residual grew as Omega^2, from 1.26e-6 to
    # 2.67e-6. Bound restated at 5e-6, still an order below the physics it is guarding.
    residual = float(np.max(np.abs(tr.expectations["P1[0]"] - pred)))
    assert residual < 5e-6, f"off-resonant sidebands ~ (eta Omega/omega_m)^2, got {residual:.3e}"
    assert residual == pytest.approx((eta * om / WX) ** 2, rel=0.3), (
        "the residual IS the off-resonant-sideband scaling, so it must track (eta Omega/omega_m)^2"
    )
    assert tr.boundary_population[KX] < 1e-12
    rep = eng.last_report
    assert rep is not None and rep.segments[0].integrator == "dop853"
    # the Section 5.3 step-density band is a Schroedinger-picture measurement (the state rotates at the Fock energies), so it
    # is checked on the Schroedinger-picture integration; the default rotating frame of dynamics/rotating.py integrates the
    # same pulse with fewer steps per period of the highest mode and the same physics (performance pass 2026-09-09)
    tr_s, eng_s = _run(
        dev,
        square_drive(dd, include_stark=False),
        t_pi,
        space,
        n_store=21,
        sopts=SolverOptions(rotating_frame=False),
    )
    rep_s = eng_s.last_report
    assert (
        rep_s is not None and rep_s.segments[0].frame == "schrodinger" and rep.segments[0].frame == "rotating"
    )
    assert 10 < rep_s.segments[0].steps_per_period < 60, (
        "Section 5.3's step-density budget is 14 to 45 steps per period of the highest mode, measured with solve_ivp; "
        "QuTiP's dop853 wrapper reports about 2x that on the same problem (12 against 27 in the plan's own bench), and "
        "the count moves with the fixture's Rabi frequency, so the window is widened and the discrepancy is recorded "
        "in the ledger as anchor.m2.picture_equivalence_and_step_density rather than absorbed into the band"
    )
    assert rep.segments[0].steps_per_period < 0.5 * rep_s.segments[0].steps_per_period
    assert float(np.max(np.abs(tr.expectations["P1[0]"] - tr_s.expectations["P1[0]"]))) < 1e-7
    # the sample's Rabi scale multiplies the drive
    tr2, _ = _run(
        dev,
        square_drive(dd, include_stark=False),
        t_pi,
        space,
        sample=NoiseSample(0, {KEY_RABI_SCALE: 0.5}, {}),
    )
    assert tr2.expectations["P1[0]"][-1] == pytest.approx(math.sin(0.25 * math.pi) ** 2, abs=1e-5)
    # on |n = 2> the carrier is reduced by e^{-eta^2/2} L_2(eta^2)
    tr3, _ = _run(dev, square_drive(dd, include_stark=False), t_pi / 2, space, n0=2)
    om2 = om * debye_waller_factor(2, eta)
    assert tr3.expectations["P1[0]"][-1] == pytest.approx(math.sin(0.5 * om2 * t_pi / 2) ** 2, abs=2e-5)


def test_blue_and_red_sideband_flopping_and_the_sideband_phase(raman) -> None:  # type: ignore[no-untyped-def]
    """Omega_{1,0} = Omega eta e^{-eta^2/2}, Omega_{0,1} likewise; the transition amplitude carries phi + pi/2 (Wineland Eq. 21);
    the full model deviates from the two-level form by the carrier's light shift of the sideband, ~(Omega/(2 eta omega))^2."""
    dev, dd, space = raman
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]
    om10 = sideband_rabi_rad_s(om, eta, 0, 1)
    assert om10 == pytest.approx(om * eta * math.exp(-(eta**2) / 2), rel=1e-12)
    t = math.pi / om10
    tr, _ = _run(dev, square_drive(dd, detuning_hz=3.0e6, include_stark=False), t, space, n_store=21)
    pred = np.array([two_level_population(om10, 0.0, x) for x in tr.times_s])
    budget = (om / (2 * eta * WX)) ** 2
    assert np.max(np.abs(tr.expectations["P1[0]"] - pred)) < 3 * budget
    assert tr.mode_occupations[KX][-1] == pytest.approx(tr.expectations["P1[0]"][-1], abs=3 * budget), (
        "|down,0> -> |up,1>"
    )
    for phi in (0.0, 0.7, -1.3):
        tr2, _ = _run(
            dev, square_drive(dd, detuning_hz=3.0e6, phase_rad=phi, include_stark=False), t / 2, space
        )
        psi = tr2.final.joint.full().ravel()
        amp = psi[12 + 1] * np.exp(1j * WX * t / 2)  # undo the motional free rotation of |1>
        pred_amp = resonant_transition_amplitude(om, eta, phi, 0, 1, t / 2)
        assert abs(amp) == pytest.approx(abs(pred_amp), abs=2e-3)
        assert np.angle(amp / pred_amp) == pytest.approx(0.0, abs=2e-3)
    om01 = sideband_rabi_rad_s(om, eta, 1, 0)
    tr3, _ = _run(dev, square_drive(dd, detuning_hz=-3.0e6, include_stark=False), math.pi / om01, space, n0=1)
    assert tr3.expectations["P1[0]"][-1] == pytest.approx(1.0, abs=3 * budget)
    assert om01 / (eta * om) == pytest.approx(rabi_matrix_element(0, 1, eta) / eta, rel=1e-12)


def test_rwa_option_is_the_exact_jaynes_cummings_model_and_the_detuned_two_level_propagator(raman) -> None:  # type: ignore[no-untyped-def]
    """In the interaction picture with rwa the single sideband term reproduces the two-level closed forms to 1e-9, including
    the generalized Rabi frequency sqrt(Omega_{n'n}^2 + Delta^2) (Section 4.3.1, Wineland Eq. 21 in the plan's convention)."""
    dev, dd, space = raman
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]
    om10 = sideband_rabi_rad_s(om, eta, 0, 1)
    opts = BuilderOptions(frame="interaction", rwa=True)
    for delta in (0.0, om10, -0.6 * om10):
        t = 1.3 * math.pi / om10
        tr, eng = _run(
            dev,
            square_drive(dd, detuning_hz=3.0e6 + delta / TWO_PI, include_stark=False),
            t,
            space,
            opts=opts,
            n_store=9,
        )
        pred = np.array([two_level_population(om10, delta, x) for x in tr.times_s])
        assert np.max(np.abs(tr.expectations["P1[0]"] - pred)) < 1e-9
        assert generalized_rabi_rad_s(om10, delta) == pytest.approx(math.hypot(om10, delta))
        rep = eng.last_report
        assert rep is not None and any("rwa keeps 1 sideband" in a for a in rep.approximations)


def test_interaction_picture_equals_the_schroedinger_picture(raman) -> None:  # type: ignore[no-untyped-def]
    """Section 9.13: the two pictures agree to 1e-8 in the final state; k_max truncation reports the dropped weight and the
    interaction picture holds the evaluation count flat while the sideband count grows."""
    dev, dd, space = raman
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]
    t = 0.4 * math.pi / sideband_rabi_rad_s(om, eta, 0, 1)
    drive = square_drive(dd, detuning_hz=3.0e6, include_stark=False)
    tr_s, eng_s = _run(dev, drive, t, space)
    tr_i, eng_i = _run(dev, drive, t, space, opts=BuilderOptions(frame="interaction"))
    u0 = (-1j * t * WX * space.number(KX)).expm()
    assert (tr_s.final.joint - u0 * tr_i.final.joint).norm() < 2e-8
    tr_k, eng_k = _run(dev, drive, t, space, opts=BuilderOptions(frame="interaction", k_max=2))
    assert (tr_s.final.joint - u0 * tr_k.final.joint).norm() < 1e-4
    rep_k = eng_k.last_report
    assert rep_k is not None and any("dropped weight" in a for a in rep_k.approximations)
    ev_s = eng_s.last_report.segments[0].rhs_evaluations
    ev_i = eng_i.last_report.segments[0].rhs_evaluations
    assert ev_i is not None and ev_s is not None and ev_i < ev_s
    pic = interaction_picture(space, 0, {KX: eta}, k_max=1)
    assert len(pic.terms) == 3 and pic.dropped_weight > 0.0
    with pytest.raises(ValueError):
        BuilderOptions(rwa=True)


def test_lamb_dicke_expansion_is_an_approximation_of_order_eta_squared(raman) -> None:  # type: ignore[no-untyped-def]
    dev, dd, space = raman
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]
    t = 0.4 * math.pi / sideband_rabi_rad_s(om, eta, 0, 1)
    drive = square_drive(dd, detuning_hz=3.0e6, include_stark=False)
    exact, _ = _run(dev, drive, t, space)
    first, eng = _run(dev, drive, t, space, opts=BuilderOptions(lamb_dicke_order=1))
    diff = abs(exact.expectations["P1[0]"][-1] - first.expectations["P1[0]"][-1])
    assert (
        0.2 * eta**2 * exact.expectations["P1[0]"][-1] < diff < 5 * eta**2 * exact.expectations["P1[0]"][-1]
    )
    assert any("expanded to order 1" in a for a in eng.last_report.approximations)
    zeroth, _ = _run(dev, drive, t, space, opts=BuilderOptions(lamb_dicke_order=0))
    assert zeroth.expectations["P1[0]"][-1] < 1e-2, "no motional coupling at order 0 on a sideband"


def test_micromotion_j0_factor_and_modulated_drive() -> None:
    """Section 4.3.6: an unlocked drive's carrier is reduced by J_0(beta); the rf-locked modulated drive averages to the same
    carrier when the rf is far above every other frequency; C0 enters eta once, through Crystal.lamb_dicke."""
    dev = single_ion_raman_device(rf=RfDrive(frequency_hz=30e6, voltage_peak_v=100.0), stray=(50.0, 0.0, 0.0))
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    assert dd.micromotion is not None and dd.c0_applied
    beta = dd.micromotion.total
    assert 0.3 < beta < 0.5
    space = HilbertSpace((2,), (ModeTruncation(KX, 12, (0, 3), 0.2),), None, (0, 2))
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]
    t = math.pi / (om * jv(0, beta) * math.exp(-(eta**2) / 2))
    tr, eng = _run(
        dev, square_drive(dd, include_stark=False), t, space, opts=BuilderOptions(micromotion="carrier_j0")
    )
    assert tr.expectations["P1[0]"][-1] == pytest.approx(1.0, abs=1e-5)
    rec = eng.last_report.drive_records[0]
    assert rec.micromotion_beta == pytest.approx(beta) and rec.carrier_factor == pytest.approx(jv(0, beta))
    tr_m, _ = _run(
        dev,
        square_drive(dd, include_stark=False, rf_locked=True, rf_phase_rad=0.3),
        t,
        space,
        opts=BuilderOptions(micromotion="modulated"),
    )
    assert tr_m.expectations["P1[0]"][-1] == pytest.approx(1.0, abs=1e-4)
    tr_none, _ = _run(
        dev, square_drive(dd, include_stark=False), t, space, opts=BuilderOptions(micromotion="none")
    )
    assert tr_none.expectations["P1[0]"][-1] == pytest.approx(
        math.sin(0.5 * om * math.exp(-(eta**2) / 2) * t) ** 2, abs=1e-5
    )
    with pytest.raises(ValueError, match="rf phase"):
        _run(
            dev, square_drive(dd, include_stark=False), t, space, opts=BuilderOptions(micromotion="modulated")
        )


def test_frozen_spectator_debye_waller_factor_from_the_sample_or_the_seeds(raman) -> None:  # type: ignore[no-untyped-def]
    """Section 5.2: a frozen mode multiplies Omega by e^{-eta^2/2} L_n(eta^2) for the shot's Fock state n, sampled once per shot."""
    dev, dd, _ = raman
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]
    frozen_space = HilbertSpace((2,), (), None, (0, 1, 2))
    t = math.pi / om
    for n in (0, 3):
        sample = NoiseSample(0, {key_frozen_n(KX): float(n)}, {})
        tr, eng = _run(dev, square_drive(dd, include_stark=False), t, frozen_space, sample=sample)
        dw = carrier_debye_waller_frozen({KX: eta}, {KX: n})
        assert dw == pytest.approx(debye_waller_factor(n, eta))
        assert tr.expectations["P1[0]"][-1] == pytest.approx(math.sin(0.5 * om * dw * t) ** 2, abs=1e-9)
        assert eng.last_report.frozen_n[KX] == n
    # without a sample key the state's nbar is drawn through the keyed seeds: reproducible
    space0 = frozen_space
    state = space0.initial_state([0], thermal={KX: 2.0})
    eng = JointExactEngine()
    sch = Schedule((Pulse(square_drive(dd, include_stark=False), 0.0, t, "p", ()),), (), (), {0: 0.0})
    eng.run_pulses(dev, sch, state, space0, quiet_sample(), SeedSpec(7), SolverOptions())
    n_a = eng.last_report.frozen_n[KX]
    eng.run_pulses(dev, sch, state, space0, quiet_sample(), SeedSpec(7), SolverOptions())
    assert eng.last_report.frozen_n[KX] == n_a


def test_thermal_carrier_flopping_matches_the_fock_sum(raman) -> None:  # type: ignore[no-untyped-def]
    """Carrier Rabi flopping on a thermal resolved mode is sum_n P_n sin^2(Omega_n t/2) (Section 4.2.7, the calibration fit model)."""
    dev, dd, _ = raman
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]
    space = HilbertSpace((2,), (ModeTruncation(KX, 24, (0, 12), 0.2),), None, (0, 2))
    t = 2.5 * math.pi / om
    tr, _ = _run(dev, square_drive(dd, include_stark=False), t, space, thermal={KX: 1.5}, n_store=6)
    for tt, p1 in zip(tr.times_s, tr.expectations["P1[0]"]):
        assert p1 == pytest.approx(frozen_thermal_population(om, eta, 1.5, tt), abs=2e-5)
    assert tr.boundary_population[KX] < 1e-6


def test_stark_term_qubit_shift_and_idle_free_evolution(raman) -> None:  # type: ignore[no-untyped-def]
    """H_Stark = (delta_St/2) sigma_z and H_int = (Delta/2) sigma_z: a superposition acquires the phase 2 pi (delta_St + Delta) t
    relative to the frame; idle intervals evolve under H_0 alone (Section 3.4)."""
    dev = microwave_device()
    space = HilbertSpace((2,), (), None, (0, 1, 2))
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    state = space.initial_state(plus)
    stark = 250.0
    drive = square_microwave_drive(
        0, 0.0 + 1e-30, stark_shift_hz=stark
    )  # a Stark-only 'pulse' (zero Rabi frequency)
    t = 1e-3
    pulse = Pulse(drive, 0.0, t, "stark", ())
    eng = JointExactEngine(qubit_shifts_hz={0: 100.0})
    tr = eng.run_pulses(
        dev,
        Schedule((pulse,), (), (), {0: 0.0}),
        state,
        space,
        NoiseSample(0, {key_qubit_offset_hz(0): 50.0}, {}),
        SeedSpec(0),
        SolverOptions(),
    )
    psi = tr.final.joint.full().ravel()
    assert np.angle(psi[1] / psi[0]) == pytest.approx(-TWO_PI * (stark + 100.0 + 50.0) * t, abs=1e-6)
    # an idle interval: the same phase from the qubit shifts alone
    sch = Schedule((), ((0.0, t),), (), {0: 0.0})
    tr2 = eng.run_pulses(dev, sch, state, space, quiet_sample(), SeedSpec(0), SolverOptions())
    psi2 = tr2.final.joint.full().ravel()
    assert np.angle(psi2[1] / psi2[0]) == pytest.approx(-TWO_PI * 100.0 * t, abs=1e-6)


def test_heating_and_dephasing_channels_in_mesolve(raman) -> None:  # type: ignore[no-untyped-def]
    """Idle evolution with sqrt(Gamma) a and sqrt(Gamma) a^dag heats at Gamma quanta per second; sqrt(gamma/2) sigma_z decays the
    coherence at gamma (Section 13 rows)."""
    dev0, _, _ = raman
    import dataclasses

    from tests.fixtures import make_noise

    dev = dataclasses.replace(dev0, noise=make_noise(s_e_two_sided=1e-13))
    space = HilbertSpace((2,), (ModeTruncation(KX, 14, (0, 4), 0.2),), None, (0, 2))
    rates = device_heating_rates(dev, space)
    assert set(rates) == {KX} and rates[KX] > 0.0
    gamma_h = 2000.0
    ch = heating_channels(space, {KX: gamma_h}) + qubit_dephasing_channels(space, {0: 5e3})
    assert len(ch) == 3 and ch[0].channel == "heating_down" and ch[2].channel == "qubit_dephasing"
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    state = space.initial_state(plus, fock={KX: 0})
    t = 1e-4
    eng = JointExactEngine(channels=ch)
    tr = eng.run_pulses(
        dev,
        Schedule((), ((0.0, t),), (), {0: 0.0}),
        state,
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    assert tr.final.motional.nbar[KX] == pytest.approx(gamma_h * t, rel=1e-3)
    rho = tr.final.internal.full()
    assert abs(rho[0, 1]) == pytest.approx(0.5 * math.exp(-5e3 * t), rel=1e-3)


def test_boundary_monitor_grows_the_cap(raman) -> None:  # type: ignore[no-untyped-def]
    """Section 5.5: a cap too low for the pulse trips the boundary monitor and the run repeats with the cap raised."""
    dev, dd, _ = raman
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]
    tiny = HilbertSpace((2,), (ModeTruncation(KX, 4, (0, 2), 0.2),), None, (0, 2))
    t = 2.0 * math.pi / sideband_rabi_rad_s(om, eta, 0, 1)
    tr, eng = _run(dev, square_drive(dd, detuning_hz=3.0e6, include_stark=False), t, tiny, n0=2)
    rep = eng.last_report
    assert rep is not None and rep.growth_retries >= 1 and rep.space.truncation(KX).d > 4
    assert max(tr.boundary_population.values()) <= SolverOptions().boundary_population_max


def test_crosstalk_drives_the_neighbour_at_the_ratio() -> None:
    """Section 6.6: the leaked drive on a neighbour is the same term with Omega -> eps Omega and the neighbour's own eta."""
    dev = two_ion_raman_device(waist_m=10e-6)
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    from qutip_trap.light.raman import crosstalk_ratios

    eps = crosstalk_ratios(dev, 0, (0, 1))[1]
    space = HilbertSpace((2, 2), (), None, (0, 1, 2, 3, 4, 5))
    om = TWO_PI * dd.carrier_rabi_hz
    t = math.pi / om
    drive = square_drive(dd, include_stark=False, crosstalk={1: eps})
    dw0 = math.prod(debye_waller_factor(0, dd.etas[m]) for m in range(6))
    pulse = Pulse(drive, 0.0, t, "x", ())
    eng = JointExactEngine()
    tr = eng.run_pulses(
        dev,
        Schedule((pulse,), (), (), {0: 0.0, 1: 0.0}),
        space.initial_state([0, 0]),
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    assert tr.expectations["P1[0]"][-1] == pytest.approx(math.sin(0.5 * om * dw0 * t) ** 2, abs=1e-6)
    dd1 = derive_raman_drive(dev, 1, (0, 1), scattering=False)
    dw1 = math.prod(debye_waller_factor(0, dd1.etas[m]) for m in range(6))
    assert tr.expectations["P1[1]"][-1] == pytest.approx(
        math.sin(0.5 * abs(eps) * om * dw1 * t) ** 2, abs=1e-6
    )
    # include_crosstalk=False drops the neighbour term and records the drop
    built = build_hamiltonian(dev, [pulse], space, options=BuilderOptions(include_crosstalk=False))
    assert any("crosstalk" in a for a in built.approximations)
    off = JointExactEngine(builder_options=BuilderOptions(include_crosstalk=False)).run_pulses(
        dev,
        Schedule((pulse,), (), (), {0: 0.0, 1: 0.0}),
        space.initial_state([0, 0]),
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    assert off.expectations["P1[1]"][-1] == pytest.approx(0.0, abs=1e-12), (
        "the neighbour is not driven at all"
    )


def test_beam_curvature_cetina_forms() -> None:
    """Section 6.2: Omega(x) = Omega_0 (1 + (Omega''/2 Omega_0) x^2) on a thermal axial mode reproduces the frozen-Fock secular sum
    exactly and Cetina's algebraic contrast C = prod (1 + theta^2 Omega^2 t^2)^{-1/2} with the phase LAG sum arctan(theta Omega t)
    in the continuum limit; Omega''/Omega = -2/w^2 = -2.6424e12 at w = 870 nm; xi_0 = 17.19 nm at 100 kHz for 171Yb+ (Section 9.12)."""
    assert gaussian_curvature_per_m2(870e-9) == pytest.approx(-2.6424e12, rel=1e-4)
    m = 170.93578 * ATOMIC_MASS_KG
    assert 1e9 * math.sqrt(HBAR_J_S / (2 * m * TWO_PI * 100e3)) == pytest.approx(17.19, abs=0.01)
    assert 1e9 * math.sqrt(HBAR_J_S / (2 * m * TWO_PI * 140e3)) == pytest.approx(14.53, abs=0.01)
    from qutip_trap.api import Device, Field
    from qutip_trap.species import species
    from qutip_trap.trap.crystal import solve_crystal
    from tests.fixtures import make_detector, make_hardware, make_noise
    from tests.m2_fixtures import explicit_trap

    trap = explicit_trap()
    trap = trap.__class__(**{**trap.__dict__, "omega_hz": (3.0e6, 2.9e6, 0.2e6)})
    yb = species("171Yb+")
    dev = Device(
        crystal=solve_crystal(trap, (yb,)),
        trap=trap,
        field=Field(5.0, (0.0, 0.0, 1.0), None),
        beams=(),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
    )
    kz = 0
    kappa = gaussian_curvature_per_m2(2e-6)
    xi = math.sqrt(HBAR_J_S / (2 * m * TWO_PI * 0.2e6))
    nbar = 4.0
    om = TWO_PI * 50e3
    space = HilbertSpace((2,), (ModeTruncation(kz, 60, (0, 40), 0.0),), None, (1, 2))
    drive = square_microwave_drive(0, 50e3)
    t_end = 10 * math.pi / om
    pulse = Pulse(drive, 0.0, t_end, "c", ())
    eng = JointExactEngine(
        builder_options=BuilderOptions(curvature={0: CurvatureSpec(kappa, (0.0, 0.0, 1.0))}),
        store_per_segment=11,
    )
    tr = eng.run_pulses(
        dev,
        Schedule((pulse,), (), (), {0: 0.0}),
        space.initial_state([0], thermal={kz: nbar}),
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    p = thermal_populations(nbar, 60)
    p = p / p.sum()  # the prepared state is the normalized truncated thermal state
    for t, p1 in zip(tr.times_s, tr.expectations["P1[0]"]):
        secular = sum(p[n] * math.sin(0.5 * om * (1 + kappa * xi**2 * (n + 0.5)) * t) ** 2 for n in range(60))
        assert p1 == pytest.approx(secular, abs=1e-6)
        theta = cetina_theta(1.0, xi, kappa, nbar)
        assert p1 == pytest.approx(cetina_population([theta], om, t), abs=2e-3)
    # the continuum form converges to the frozen sum as nbar grows (exponential energies, Cetina's regime)
    for nb, tol in ((20.0, 5e-5), (100.0, 2e-4)):
        th = cetina_theta(1.0, xi, kappa, nb)
        pp = thermal_populations(nb, int(60 + 40 * nb))
        tt = 8 * math.pi / om
        sec = sum(
            pp[n] * math.sin(0.5 * om * (1 + kappa * xi**2 * (n + 0.5)) * tt) ** 2 for n in range(len(pp))
        )
        assert sec == pytest.approx(cetina_population([th], om, tt), abs=tol)


def test_evolve_ladder_records_the_integrator_and_never_uses_multistep() -> None:
    h = qt.QobjEvo([qt.sigmaz(), [qt.sigmax(), lambda t, **kw: math.cos(t)]])
    ev = evolve(h, qt.basis(2, 0), [0.0, 1.0], options=SolverOptions())
    assert ev.integrator == "dop853" and ev.retries == () and ev.final.norm() == pytest.approx(1.0, abs=1e-8)
    ev2 = evolve(h, qt.basis(2, 0), [0.0, 1.0], options=SolverOptions(integrators=("vern9",)))
    assert ev2.integrator == "vern9"
    with pytest.raises(ValueError, match="multistep"):
        SolverOptions(integrators=("adams",))
