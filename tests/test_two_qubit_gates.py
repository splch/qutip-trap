"""Two-qubit gate physics through the JOINT_EXACT engine against the closed forms (PLAN.md Section 4.4), and the reference MS
pulse (one 100 us loop on the two-ion chain's x-COM mode at d_m = 12) against the native matrix inside its intrinsic budget."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import qutip as qt
from scipy.integrate import cumulative_simpson, simpson

from qutip_trap.calibration.entangling import (
    calibrate_entangling_angle,
    exact_gate_check,
    ms_schedule,
    spot_check_space,
)
from qutip_trap.control.native import ms as native_ms
from qutip_trap.control.schedule import PhaseFrame, PlayedGate, Schedule, entangling_pulses, ms_spin_phases
from qutip_trap.control.shaping import (
    CHI_MAXIMAL_RAD,
    SINE_MOTION_PHASE_RAD,
    GateModes,
    SampledEnvelope,
    SegmentedEnvelope,
    gate_modes,
    integrals_sampled,
    integrals_segmented,
    solve_amplitude_modulation,
    solve_fourier_amplitude_modulation,
    solve_frequency_modulation,
    symmetric_pulse,
    waveform_from_segmented,
)
from qutip_trap.control.table import Waveform
from qutip_trap.device.model import Device, Field
from qutip_trap.device.presets import secular_trap
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec
from qutip_trap.dynamics.hamiltonian import BuilderOptions
from qutip_trap.dynamics.operators import rabi_matrix_element
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.beams import Beam
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.options import Numerics, Physics
from qutip_trap.published import (
    kirchmair_populations,
    ms_alpha,
    ms_gamma,
    roos_force_saturation,
    thermal_debye_waller_infidelity,
)
from qutip_trap.run.job import (
    intrinsic_budget,
    roos_beat_phase_tilt,
    roos_bessel_saturation,
    sideband_lamb_dicke_deficit,
)
from qutip_trap.run.space import SpaceSelection, select_space
from qutip_trap.species import species
from qutip_trap.trap.crystal import Crystal, Mode
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI
from tests.fixtures import (
    X_COM_TWO_IONS,
    chain_device,
    derived_seeds,
    quiet_device,
    raman_gate_drives,
    table_with_waveform,
    two_ion_modes,
)
from tests.oracles import (
    entanglement_infidelity_from_displacements,
    ms_propagator,
    spectator_loop_error,
    state_infidelity_uniform_input,
)

KET00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
ETA_ANCHOR = 0.05
NU_ANCHOR_HZ = 1.0e6
EPS_ANCHOR_HZ = 10e3


def anchor_device(eta: float = ETA_ANCHOR) -> Device:
    """Two 171Yb+ ions whose x-COM mode at 1 MHz carries ``eta`` (0.05 unless given) exactly (the 355 nm pair's crossing
    angle is chosen for it)."""
    yb = species("171Yb+")
    mass = yb.mass_u * ATOMIC_MASS_KG
    x0 = math.sqrt(HBAR_J_S / (2.0 * mass * TWO_PI * NU_ANCHOR_HZ))
    dk = eta / (x0 / math.sqrt(2.0))
    k = TWO_PI / 355e-9
    half = math.asin(dk / (2.0 * k))
    sq = 1.0 / math.sqrt(2.0)
    com = np.array([sq, sq])
    rock = np.array([-sq, sq])
    modes = (
        Mode("axial", 0, 0.5e6, (0.0, 0.0, 1.0), com),
        Mode("axial", 1, math.sqrt(3.0) * 0.5e6, (0.0, 0.0, 1.0), rock),
        Mode("transverse_1", 0, 0.8e6, (1.0, 0.0, 0.0), rock),
        Mode("transverse_1", 1, NU_ANCHOR_HZ, (1.0, 0.0, 0.0), com),
        Mode("transverse_2", 0, 0.7e6, (0.0, 1.0, 0.0), rock),
        Mode("transverse_2", 1, 0.9e6, (0.0, 1.0, 0.0), com),
    )
    crystal = Crystal(
        species=(yb, yb), positions_m=np.array([[0.0, 0.0, -3e-6], [0.0, 0.0, 3e-6]]), modes=modes
    )
    b1 = Beam(355e-9, (math.sin(half), 0.0, math.cos(half)), (0.0, 1.0, 0.0), 200e-6, 10e-3, (0.0, 0.0, 0.0))
    b2 = Beam(
        355e-9,
        (-math.sin(half), 0.0, math.cos(half)),
        (math.cos(half), 0.0, math.sin(half)),
        200e-6,
        10e-3,
        (0.0, 0.0, 0.0),
    )
    return quiet_device(crystal, secular_trap((1.0e6, 0.9e6, 0.5e6)), Field(5.0, (1.0, 0.0, 0.0)), (b1, b2))


ANCHOR_MODE = 3
ONE_MODE_SPACE = HilbertSpace((2, 2), (ModeTruncation(ANCHOR_MODE, 16, (0, 4), 0.1),), None, (0, 1, 2, 4, 5))
ONE_MODE_OPTIONS = BuilderOptions(frozen_debye_waller=False)
"""The rocking mode (eta = 0.056, frozen) is switched off entirely: the anchor's fixture has one mode."""


def _run(
    dev: Device, wf: Waveform, space: HilbertSpace, opts: BuilderOptions, internal=(0, 0), nbar=None, store=2
):
    # the closed forms carry no light shift: derived Rabi entries, no Stark entry, and an engine without the table
    table = table_with_waveform((0, 1), wf, device=dev, drives=raman_gate_drives(2), stark_hz={})
    sched = ms_schedule(dev, wf, (0, 1), raman_gate_drives(2), table, physics=Physics(builder=opts))
    eng = JointExactEngine(builder_options=opts, store_per_segment=store)
    tr = eng.run_pulses(
        dev,
        sched,
        space.initial_state(
            internal if isinstance(internal, qt.Qobj) else list(internal), thermal=nbar or {}
        ),
        space,
        quiet_sample(),
        SeedSpec(0),
        Numerics(),
    )
    return tr, eng


@pytest.fixture(scope="module")
def anchor():
    dev = anchor_device()
    modes = gate_modes(dev, (0, 1), (0, 1)).subset([ANCHOR_MODE])
    assert modes.eta[0] == pytest.approx((ETA_ANCHOR,), rel=1e-12)
    return dev, modes


def test_ms_closure_anchors_reproduce_through_the_package(anchor) -> None:
    """One loop from |dd> at eta = 0.05, nu = 1 MHz, eps = 10 kHz, d_m = 16: concurrence 0.9999 and populations (0.5074, 0, 0,
    0.4926) at eta Omega/eps = 1/2, and 0.3825 with (0.962, 0, 0, 0.038) at 1/4, to 6e-5 and 6e-4."""
    dev, modes = anchor
    for ratio, conc, p_dd, p_uu in ((0.5, 0.9999, 0.5074, 0.4926), (0.25, 0.3825, 0.962, 0.038)):
        wf = Waveform.symmetric(
            modes,
            gate_mode=ANCHOR_MODE,
            loops=1,
            epsilon_hz=EPS_ANCHOR_HZ,
            kernel="rwa",
            chi_target_rad=math.pi * ratio**2,
        )
        assert wf.segments[0].amplitude_hz[(0, "blue")] == pytest.approx(
            ratio * EPS_ANCHOR_HZ / ETA_ANCHOR, rel=1e-12
        )
        tr, _ = _run(dev, wf, ONE_MODE_SPACE, ONE_MODE_OPTIONS)
        rho = tr.final.internal
        pops = np.real(np.diag(rho.full()))
        assert qt.concurrence(rho) == pytest.approx(conc, abs=6e-5)
        assert pops[0] == pytest.approx(p_dd, abs=6e-4) and pops[3] == pytest.approx(p_uu, abs=6e-4)
        assert pops[1] < 1e-4 and pops[2] < 1e-4


def test_sine_motion_phase_tilts_the_spin_axis_by_the_carrier_rotation(anchor) -> None:
    """With the tone phases pi apart (Choi's sine convention) the carrier tilts the entangling axis by psi = 2 Omega/mu = 0.2
    rad (Roos 2008) and the closed-loop |dd> gate leaks sin^2(psi) into |du>, |ud> to 10 %."""
    dev, modes = anchor
    wf = Waveform.symmetric(
        modes,
        gate_mode=ANCHOR_MODE,
        loops=1,
        epsilon_hz=EPS_ANCHOR_HZ,
        kernel="rwa",
        phi_m_rad=SINE_MOTION_PHASE_RAD,
    )
    tr, _ = _run(dev, wf, ONE_MODE_SPACE, ONE_MODE_OPTIONS)
    rho = tr.final.internal
    pops = np.real(np.diag(rho.full()))
    omega = TWO_PI * wf.segments[0].amplitude_hz[(0, "blue")]
    mu = TWO_PI * wf.segments[0].detuning_hz["blue"]
    psi = 2.0 * omega / mu
    assert psi == pytest.approx(0.2, rel=2e-2)
    assert pops[1] + pops[2] == pytest.approx(math.sin(psi) ** 2, rel=0.1)
    assert 0.94 < qt.concurrence(rho) < 0.97
    assert tr.final.motional.nbar[ANCHOR_MODE] < 1e-4, (
        "the loop itself closes: the leakage is the axis tilt, not a residual displacement"
    )


X_COM_ALONE = HilbertSpace((2, 2), (ModeTruncation(X_COM_TWO_IONS, 14, (0, 5), 0.12),), None, (0, 1, 2, 4, 5))
X_MODES = HilbertSpace(
    (2, 2),
    (ModeTruncation(2, 10, (0, 3), 0.12), ModeTruncation(X_COM_TWO_IONS, 10, (0, 3), 0.12)),
    None,
    (0, 1, 4, 5),
)


def _played_from(
    device: Device,
    wf: Waveform,
    t_start_s: float,
    ket: qt.Qobj,
    *,
    reset: bool,
    space: HilbertSpace = X_COM_ALONE,
) -> qt.Qobj:
    """The internal state after ``wf`` on the two-ion chain's modes of ``space`` (no light shift, the other modes switched
    off) from ``ket``|0>, its tones running since t = 0 and the gate started at ``t_start_s``."""
    drives = raman_gate_drives(2)
    table = table_with_waveform((0, 1), wf, device=device, drives=drives, stark_hz={})
    spins, _ = ms_spin_phases(wf, (0, 1), (0.0, 0.0), PhaseFrame())
    pulses = entangling_pulses(
        wf,
        drives,
        spin_phases_rad=spins,
        t_start_s=t_start_s,
        table=table,
        gate_id="ms",
        beat_phase_reset=reset,
        stark_compensation=False,
    )
    gate = PlayedGate("ms", "ms", (0, 1), wf, drives[0].beams, t_start_s, t_start_s + wf.duration_s)
    sched = Schedule(tuple(pulses), (), (), {0: 0.0, 1: 0.0}, gates=(gate,), t0_s=t_start_s)
    engine = JointExactEngine(builder_options=BuilderOptions(include_stark=False, frozen_debye_waller=False))
    traces = engine.run_pulses(
        device, sched, space.initial_state(ket), space, quiet_sample(), SeedSpec(0), Numerics()
    )
    return traces.final.internal


def test_the_beat_phase_tilt_is_set_by_the_amplitude_the_gate_switches_on_with() -> None:
    """Roos's spin-axis tilt at beat phase zeta = pi/2 against the exact engine on the two-ion chain's x-COM mode with
    phase-continuous tones: the |++> output moves from the zeta = 0 one by the budget's sin^2(psi), psi = (2 Omega_on/mu)
    sin(zeta), for a square pulse (1.77e-3 against 1.81e-3) and for Leung's FM pulse at its starting detuning (6.95e-3
    against 6.48e-3); the Fourier-sine AM switches on at zero amplitude, the carrier follows it adiabatically and its output
    does not move (below 1e-5), where its time-mean amplitude would give 4.4e-3."""
    device = chain_device(2)
    assert device.hardware.phase_continuous
    reset_hw = dataclasses.replace(
        device, hardware=dataclasses.replace(device.hardware, phase_continuous=False)
    )
    modes = two_ion_modes(device).subset([X_COM_TWO_IONS])
    square = Waveform.symmetric(
        modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=EPSILON_HZ, kernel="rwa", all_modes=False
    )
    fourier = solve_fourier_amplitude_modulation(modes, mu_hz=2.99e6, duration_s=100e-6, n_basis=4).waveform
    fm = solve_frequency_modulation(modes, duration_s=100e-6, n_vertices=5, mu0_hz=3.012e6).waveform
    plus = qt.tensor((qt.basis(2, 0) + qt.basis(2, 1)).unit(), (qt.basis(2, 0) + qt.basis(2, 1)).unit())
    moved: dict[str, tuple[float, float]] = {}
    for name, wf in (("square", square), ("fourier", fourier), ("fm", fm)):
        mu = wf.segments[0].detuning_hz["blue"]
        t_g = 0.25 / float(mu(0.0) if callable(mu) else mu)
        at_zero = _played_from(device, wf, 0.0, plus, reset=False)
        at_quarter = _played_from(device, wf, t_g, plus, reset=False)
        assert float(np.real((at_zero * at_zero).tr())) > 1.0 - 1e-6, "the |++> output is pure"
        change = 1.0 - float(np.real((at_zero * at_quarter).tr()))
        moved[name] = (change, roos_beat_phase_tilt(wf, t_g, beat_reset=False))
    assert moved["square"][1] == pytest.approx(1.81e-3, rel=1e-2)
    assert moved["square"][0] == pytest.approx(moved["square"][1], rel=0.05), moved
    assert moved["fm"][1] == pytest.approx(6.48e-3, rel=1e-2)
    assert moved["fm"][0] == pytest.approx(moved["fm"][1], rel=0.1), moved
    assert moved["fourier"][1] == 0.0 and moved["fourier"][0] < 1e-5, moved
    # the budget reads the device's hardware: on phase-continuous tones the square pulse at t_g, on the resetting chain none
    for dev, want in ((device, moved["square"][1]), (reset_hw, 0.0)):
        t_g = 0.25 / float(square.segments[0].detuning_hz["blue"])
        drives = raman_gate_drives(2)
        gate = PlayedGate("ms", "ms", (0, 1), square, drives[0].beams, t_g, t_g + square.duration_s)
        pulses = entangling_pulses(
            square,
            drives,
            spin_phases_rad=ms_spin_phases(square, (0, 1), (0.0, 0.0), PhaseFrame())[0],
            t_start_s=t_g,
            table=table_with_waveform((0, 1), square, device=dev, drives=drives, stark_hz={}),
            gate_id="ms",
            beat_phase_reset=not dev.hardware.phase_continuous,
        )
        sched = Schedule(tuple(pulses), (), (), {0: 0.0, 1: 0.0}, gates=(gate,), t0_s=t_g)
        selection = select_space(
            dev, sched, Numerics(caps={X_COM_TWO_IONS: 12}), nbar=dict.fromkeys(range(6), 0.0)
        )
        assert intrinsic_budget(dev, sched, selection)["ms.beat_phase_tilt"] == pytest.approx(want, abs=1e-15)


def test_a_resetting_chain_plays_an_fm_gate_a_quarter_beat_in_as_it_plays_at_t_zero() -> None:
    """The builder starts a detuning schedule's beat at 2 pi mu(0) t_start in absolute time, and the per-gate reset cancels
    it: Leung's FM pulse started a quarter beat into the schedule on hardware that programs each gate from its own start
    leaves every input where the pulse at t = 0 leaves it (trace distance below 1e-8 from |++>, |00>, |01> and |+ +i>,
    where the running beat moves them by 8.3e-2), and the budget's tilt at t_g is its tilt at t = 0."""
    device = chain_device(2)
    reset_hw = dataclasses.replace(
        device, hardware=dataclasses.replace(device.hardware, phase_continuous=False)
    )
    modes = two_ion_modes(device).subset([X_COM_TWO_IONS])
    fm = solve_frequency_modulation(modes, duration_s=100e-6, n_vertices=5, mu0_hz=3.012e6).waveform
    mu = fm.segments[0].detuning_hz["blue"]
    assert callable(mu)
    t_g = 0.25 / float(mu(0.0))
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    plus_i = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    for ket in (qt.tensor(plus, plus), qt.tensor(zero, zero), qt.tensor(zero, one), qt.tensor(plus, plus_i)):
        at_zero = _played_from(reset_hw, fm, 0.0, ket, reset=True)
        at_quarter = _played_from(reset_hw, fm, t_g, ket, reset=True)
        assert qt.tracedist(at_zero, at_quarter) < 1e-8
    assert roos_beat_phase_tilt(fm, t_g, beat_reset=True) == pytest.approx(
        roos_beat_phase_tilt(fm, 0.0, beat_reset=True), abs=1e-15
    )


@pytest.mark.slow
def test_a_stepped_envelope_moves_less_than_its_switch_on_tilt() -> None:
    """Five-segment AM closing both x modes (33, 107, 148, 107, 33 kHz): started at zeta = pi/2 its |++> output moves by
    1.2e-4, inside the switch-on tilt sin^2(2 Omega_on/mu) = 5.2e-4, which the later steps, each at its own beat phase,
    partly cancel here; the mean segment amplitude would give 3.4e-3."""
    device = chain_device(2)
    modes = two_ion_modes(device).subset([2, X_COM_TWO_IONS])
    wf = solve_amplitude_modulation(modes, mu_hz=2.95e6, duration_s=150e-6, kernel="rwa").waveform
    assert len(wf.segments) == 5
    plus = qt.tensor((qt.basis(2, 0) + qt.basis(2, 1)).unit(), (qt.basis(2, 0) + qt.basis(2, 1)).unit())
    t_g = 0.25 / 2.95e6
    at_zero = _played_from(device, wf, 0.0, plus, reset=False, space=X_MODES)
    at_quarter = _played_from(device, wf, t_g, plus, reset=False, space=X_MODES)
    change = 1.0 - float(np.real((at_zero * at_quarter).tr()))
    tilt = roos_beat_phase_tilt(wf, t_g, beat_reset=False)
    omega_on = float(wf.segments[0].amplitude_hz[(0, "blue")])
    assert tilt == pytest.approx(math.sin(2.0 * omega_on / 2.95e6) ** 2, rel=1e-9)
    assert 2e-5 < change < tilt, (change, tilt)


def test_the_carrier_saturates_the_force_at_twice_the_per_tone_rabi_frequency_over_mu() -> None:
    """Roos 2008 Eq. 17 in the per-tone Omega: the carrier scales the MS force by (J_0 + J_2)(2 Omega/mu), so chi by its
    square. One loop at eps = 10 kHz on the 1 MHz anchor mode with the carrier and both first sidebands (lamb_dicke_order =
    1, no Debye-Waller factor) against the closed form's pi/4, at eta = 0.1 and 0.025 (Omega/mu = 0.0505 and 0.202): the
    ratio of the two angles is the ratio of (J_0 + J_2)^2 at x = 2 Omega/mu to 2e-4 (0.96238 against 0.96233), where Roos's
    printed 4 Omega/mu read in this Omega would give 0.856; the counter-rotating S_y^2 term's drive-independent eps/(2 nu)
    offset (+0.005 on each angle) cancels in the ratio."""
    opts = BuilderOptions(lamb_dicke_order=1, include_stark=False, frozen_debye_waller=False)
    chi: dict[float, float] = {}
    for eta in (0.1, 0.025):
        dev = anchor_device(eta)
        modes = gate_modes(dev, (0, 1), (0, 1)).subset([ANCHOR_MODE])
        wf = Waveform.symmetric(modes, gate_mode=ANCHOR_MODE, loops=1, epsilon_hz=EPS_ANCHOR_HZ, kernel="rwa")
        omega = TWO_PI * float(wf.segments[0].amplitude_hz[(0, "blue")])
        mu = TWO_PI * float(wf.segments[0].detuning_hz["blue"])
        assert omega / mu == pytest.approx(
            EPS_ANCHOR_HZ / (2.0 * eta * (NU_ANCHOR_HZ - EPS_ANCHOR_HZ)), rel=1e-9
        )
        tr, _ = _run(dev, wf, ONE_MODE_SPACE, opts)
        p11 = float(np.real(tr.final.internal.full()[3, 3]))
        chi[omega / mu] = math.asin(math.sqrt(p11)) / (math.pi / 4.0)
    (weak, w_chi), (strong, s_chi) = sorted(chi.items())
    assert weak == pytest.approx(0.0505, rel=1e-3) and strong == pytest.approx(0.202, rel=1e-3)
    measured = s_chi / w_chi
    per_tone = (roos_force_saturation(strong, 1.0) / roos_force_saturation(weak, 1.0)) ** 2
    roos_printed = (roos_force_saturation(2.0 * strong, 1.0) / roos_force_saturation(2.0 * weak, 1.0)) ** 2
    assert measured == pytest.approx(per_tone, abs=2e-4), (measured, per_tone)
    assert abs(measured - roos_printed) > 0.1


def test_exact_ms_propagator_first_order_lamb_dicke(anchor) -> None:
    """In the first-order Lamb-Dicke, RWA interaction picture the state equals the MS propagator D(alpha S) exp(i gamma S^2)
    |dd, 0> to 1e-9 at every stored time and the populations follow Kirchmair's nbar = 0 envelopes to 1e-9."""
    dev, modes = anchor
    wf = Waveform.symmetric(modes, gate_mode=ANCHOR_MODE, loops=1, epsilon_hz=EPS_ANCHOR_HZ, kernel="rwa")
    opts = BuilderOptions(frame="interaction", rwa=True, lamb_dicke_order=1, frozen_debye_waller=False)
    tr, eng = _run(dev, wf, ONE_MODE_SPACE, opts, store=41)
    omega = TWO_PI * wf.segments[0].amplitude_hz[(0, "blue")]
    eps = TWO_PI * EPS_ANCHOR_HZ
    assert eng.last_report is not None and any("rwa keeps" in a for a in eng.last_report.approximations)
    psi0 = ONE_MODE_SPACE.initial_state([0, 0]).joint
    assert psi0 is not None
    worst = 0.0
    for t, state in zip(tr.times_s, tr.reduced_internal):
        alpha = ms_alpha(ETA_ANCHOR, omega, eps, t)
        gamma = ms_gamma(ETA_ANCHOR, omega, eps, t)
        # alpha carries the -(hbar eta Omega/2) sign; the MS(0, 0) schedule's axes are X on ion 0 and -X on ion 1 (S = X_0 - X_1)
        u = ms_propagator(-alpha, gamma, 16, phi_rad=0.0, phi2_rad=math.pi)
        ref = (u * psi0).ptrace([0, 1])
        worst = max(worst, 1.0 - float(qt.fidelity(ref, state)) ** 2)
        p = np.real(np.diag(state.full()))
        p0, p1, p2 = kirchmair_populations(abs(alpha), gamma, 0.0)
        # Kirchmair's bright state is |d> (the fluorescing S1/2 level): p_2 = P_00, p_0 = P_11
        # measured worst deviations: 3.95e-11 on the populations and 2.28e-11 on the propagator infidelity
        assert p[0] == pytest.approx(p2, abs=1e-9) and p[3] == pytest.approx(p0, abs=1e-9)
        assert p[1] + p[2] == pytest.approx(p1, abs=1e-9)
    assert worst < 1e-9
    # the sign: alpha's phase convention and the sign of the geometric phase are checked through the full state, the closure exactly
    assert tr.final.motional.nbar[ANCHOR_MODE] < 1e-9


def test_thermal_envelopes_and_debye_waller_references(anchor) -> None:
    """The first-order force on a thermal mode (nbar = 1) follows Kirchmair Eq. 14 to 3e-4, and the three thermal Debye-Waller
    references are 2, 3 and 4.25 in units of (pi^2/4) eta^4 at nbar = 1."""
    dev, modes = anchor
    wf = Waveform.symmetric(modes, gate_mode=ANCHOR_MODE, loops=1, epsilon_hz=EPS_ANCHOR_HZ, kernel="rwa")
    omega = TWO_PI * wf.segments[0].amplitude_hz[(0, "blue")]
    eps = TWO_PI * EPS_ANCHOR_HZ
    space = HilbertSpace((2, 2), (ModeTruncation(ANCHOR_MODE, 30, (0, 12), 0.1),), None, (0, 1, 2, 4, 5))
    first_order = BuilderOptions(frame="interaction", rwa=True, lamb_dicke_order=1, frozen_debye_waller=False)
    tr, _ = _run(dev, wf, space, first_order, nbar={ANCHOR_MODE: 1.0}, store=21)
    for t, state in zip(tr.times_s, tr.reduced_internal):
        alpha = ms_alpha(ETA_ANCHOR, omega, eps, t)
        gamma = ms_gamma(ETA_ANCHOR, omega, eps, t)
        p = np.real(np.diag(state.full()))
        p0, p1, p2 = kirchmair_populations(abs(alpha), gamma, 1.0)
        assert p[0] == pytest.approx(p2, abs=3e-4)
        assert p[1] + p[2] == pytest.approx(p1, abs=3e-4)
        assert p[3] == pytest.approx(p0, abs=3e-4)
    assert thermal_debye_waller_infidelity(0.1, 1.0, "mean") / ((math.pi**2 / 4) * 1e-4) == pytest.approx(2.0)
    assert thermal_debye_waller_infidelity(0.1, 1.0, "n0") / ((math.pi**2 / 4) * 1e-4) == pytest.approx(3.0)
    assert thermal_debye_waller_infidelity(0.1, 1.0, "minus_half") / (
        (math.pi**2 / 4) * 1e-4
    ) == pytest.approx(4.25)


def test_symmetrized_kernel_is_exact_by_block_diagonal_integration() -> None:
    """With non-proportional envelopes chi from the four sigma_x blocks' exact phases equals the symmetrized kernel K_ij + K_ji
    to 1e-8, and Choi's two printed readings 2 K_ij, 2 K_ji miss it by equal and opposite amounts (26.8 % here)."""
    omega = TWO_PI * 1.0e6
    mu = TWO_PI * 0.96e6
    # a wide-open loop: at closure (F = 0) the two printed readings coincide
    tau = 2.0 * math.pi / (omega - mu) * 0.6
    eta = 0.08
    d = 24
    t = np.linspace(0.0, tau, 4001)
    w = TWO_PI * 60e3
    env_a = w * np.sin(math.pi * t / tau) ** 2
    env_b = w * (t / tau) * np.sin(math.pi * t / tau) ** 2
    modes = GateModes(ions=(0, 1), modes=(0,), omega_rad_s=(omega,), eta={0: (eta,), 1: (eta,)}, nbar=(0.0,))
    ints = integrals_sampled(SampledEnvelope(t, {0: env_a, 1: env_b}, mu * t), modes, "choi")
    a = qt.destroy(d)
    phases = {}
    for s1 in (1, -1):
        for s2 in (1, -1):
            # H = eta [s1 Omega_a(t) + s2 Omega_b(t)] cos(mu t) (a e^{-i omega t} + h.c.) in the motional interaction picture
            def coef(tt: float, s1: int = s1, s2: int = s2, **kwargs: object) -> complex:
                oa = w * math.sin(math.pi * tt / tau) ** 2
                ob = w * (tt / tau) * math.sin(math.pi * tt / tau) ** 2
                return complex(eta * (s1 * oa + s2 * ob) * math.cos(mu * tt) * np.exp(-1j * omega * tt))

            def coef_conj(tt: float, c=coef, **kwargs: object) -> complex:
                return complex(np.conj(c(tt)))

            h = qt.QobjEvo([[a, coef], [a.dag(), coef_conj]])
            psi = qt.sesolve(
                h, qt.basis(d, 0), [0.0, tau], options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 10**7}
            ).states[-1]
            ov = complex(qt.basis(d, 0).overlap(psi))
            phases[(s1, s2)] = np.angle(ov)
    chi_exact = (phases[(1, 1)] + phases[(-1, -1)] - phases[(1, -1)] - phases[(-1, 1)]) / 4.0
    # measured 2.70e-9 at these 4001 points (2.4e-11 at 20001)
    assert ints.chi_of(0, 1) == pytest.approx(chi_exact, rel=1e-8)
    assert ints.chi_of(1, 0) == pytest.approx(ints.chi_of(0, 1), rel=1e-12)
    f = np.cos(mu * t) * np.exp(1j * omega * t)

    def k_of(env_i: np.ndarray, env_j: np.ndarray) -> float:
        return float(
            eta
            * eta
            * simpson(np.imag(env_i * f * np.conj(cumulative_simpson(env_j * f, x=t, initial=0.0))), x=t)
        )

    kab, kba = k_of(env_a, env_b), k_of(env_b, env_a)
    assert kab + kba == pytest.approx(chi_exact, rel=1e-8)
    # the full-square integral of the printed integrand is the ANTISYMMETRIC combination K_ab - K_ba, not chi
    a_a = complex(simpson(env_a * f, x=t))
    a_b = complex(simpson(env_b * f, x=t))
    full_square = float(eta * eta * np.imag(a_a * np.conj(a_b)))
    assert full_square == pytest.approx(kab - kba, rel=1e-8), (
        "int_0^tau dt' int_0^tau dt of the printed integrand = K_ab - K_ba"
    )
    # the two printed readings 2 K_ab and 2 K_ba miss chi by equal and opposite amounts
    dev_ab = 2 * kab / chi_exact - 1.0
    dev_ba = 2 * kba / chi_exact - 1.0
    assert abs(dev_ab) > 0.1 and abs(dev_ba) > 0.1
    assert dev_ab == pytest.approx(-dev_ba, rel=1e-6), "equal and opposite, exactly"
    assert abs(dev_ab) == pytest.approx(0.268098, rel=1e-3), (
        "fixture-dependent: only the equal-and-opposite structure is universal"
    )


def test_residual_displacement_conversions_by_direct_integration(anchor) -> None:
    """On an open loop (1.1 loops) of the first-order force the final <n> is eps_ent = sum |alpha|^2 to 1e-3, and 1 - F_ent from
    the four sigma_x inputs is the exact uniform-input form to 2e-4 and eps_ent to 10 %."""
    dev, modes = anchor
    eps = TWO_PI * EPS_ANCHOR_HZ
    tau = 1.1 * 2.0 * math.pi / eps
    sp = symmetric_pulse(modes, gate_mode=ANCHOR_MODE, loops=1, epsilon_hz=EPS_ANCHOR_HZ, kernel="rwa")
    omega = TWO_PI * sp.diagnostics["rabi_hz"]
    env = SegmentedEnvelope((tau,), {0: (omega,), 1: (omega,)}, TWO_PI * (NU_ANCHOR_HZ - EPS_ANCHOR_HZ))
    ints = integrals_segmented(env, modes, "rwa")
    wf = waveform_from_segmented(env, ints, modes)
    alphas = [ints.alpha[(0, ANCHOR_MODE)], ints.alpha[(1, ANCHOR_MODE)]]
    eps_ent = entanglement_infidelity_from_displacements(alphas, [0.0, 0.0])
    assert 0.02 < eps_ent < 0.1
    opts = BuilderOptions(frame="interaction", rwa=True, lamb_dicke_order=1, frozen_debye_waller=False)
    tr, _ = _run(dev, wf, ONE_MODE_SPACE, opts)
    assert tr.final.motional.nbar[ANCHOR_MODE] == pytest.approx(eps_ent, rel=1e-3), (
        "final <n> = sum_j |alpha_j|^2 for the |00> input"
    )
    # entanglement fidelity from the four sigma_x-basis inputs: F_ent = (1/16) sum_{s s'} <m_s'|m_s> e^{-i(theta_s - theta_s')}
    # the MS(0, 0) schedule's force axes are X on ion 0 and -X on ion 1 (S = X_0 - X_1): x eigenstates are preserved
    xplus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    xminus = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
    motional = {}
    for s in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        spin = qt.tensor(xplus if s[0] > 0 else xminus, xplus if s[1] > 0 else xminus)
        tr_s, _ = _run(dev, wf, ONE_MODE_SPACE, opts, internal=spin)
        joint = tr_s.final.joint
        assert joint is not None
        arr = np.asarray(joint.full()).reshape(4, 16)
        spin_vec = np.asarray(spin.full()).ravel()
        # <s| psi> as a motional ket (the S eigenstates are preserved by the force)
        motional[s] = spin_vec.conj() @ arr
        assert np.linalg.norm(motional[s]) == pytest.approx(1.0, abs=1e-6)
    chi = ints.chi_of(0, 1)
    keys = list(motional)
    f_ent = 0.0
    for s in keys:
        for sp_ in keys:
            # S^2 = 2 - 2 X_0 X_1: the two-body phase is -chi s_0 s_1 (the scheduler's sign flip on ion 1)
            th = -chi * (s[0] * s[1] - sp_[0] * sp_[1])
            f_ent += np.real(np.vdot(motional[sp_], motional[s]) * np.exp(-1j * th))
    f_ent /= 16.0
    # exact: F_ent = (1/16) sum_{s s'} exp(-|beta_s - beta_s'|^2/2), which state_infidelity_uniform_input evaluates; eps_ent is its
    # first-order expansion (the second-order term is -5 |alpha|^4 here, 6% of eps_ent at |alpha|^2 = 0.024)
    assert 1.0 - f_ent == pytest.approx(state_infidelity_uniform_input(alphas, 0.0), abs=2e-4)
    assert 1.0 - f_ent == pytest.approx(eps_ent, rel=0.1)


def test_spectator_loop_error_and_am_closure_exactly() -> None:
    """The symmetric COM pulse leaves sum_j |alpha_jm|^2 quanta in the open rocking-mode loop (to 5 %, the spectator-loop form to
    1e-9) and a fidelity below 0.995; the AM pulse closes every loop (leakage < 2e-4, fidelity > 0.999)."""
    dev = chain_device(2)
    modes = two_ion_modes(dev)
    drives = raman_gate_drives(2)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=20e3)
    space = spot_check_space(dev, modes, wf, (0, 1), Numerics())[0]
    table = table_with_waveform((0, 1), wf, device=dev, drives=raman_gate_drives(2))
    check, _ = exact_gate_check(dev, wf, (0, 1), drives, table, space=space, physics=Physics())
    rock = 2
    # the worst ion's alpha, equal for both by symmetry
    closed_form = sum(abs(wf.alpha_m[rock]) ** 2 for _ in (0, 1))
    assert check.residual_quanta[rock] == pytest.approx(closed_form, rel=0.05)
    assert check.residual_quanta[X_COM_TWO_IONS] < 1e-3
    # the derived spectator-loop form assumes the single-mode closure Omega = pi sqrt K/(eta_g t_g) and the rwa force: exact for it
    single = symmetric_pulse(
        modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=20e3, kernel="rwa", all_modes=False
    )
    ints = integrals_segmented(single.envelope, modes, "rwa")
    omega_g = modes.omega_rad_s[modes.index(X_COM_TWO_IONS)]
    omega_r = modes.omega_rad_s[modes.index(rock)]
    mu = single.envelope.mu_rad_s
    derived = spectator_loop_error(2, 1, omega_g, [(omega_r, mu - omega_r, 0.0)], single.waveform.duration_s)
    assert derived == pytest.approx(sum(abs(ints.alpha[(i, rock)]) ** 2 for i in (0, 1)), rel=1e-9)
    assert check.leakage > 5e-3 and check.fidelity < 0.995
    am = solve_amplitude_modulation(modes, mu_hz=2.914e6, duration_s=100e-6)
    space_am = spot_check_space(dev, modes, am.waveform, (0, 1), Numerics())[0]
    table_am = table_with_waveform((0, 1), am.waveform, device=dev, drives=raman_gate_drives(2))
    check_am, tr = exact_gate_check(
        dev, am.waveform, (0, 1), drives, table_am, space=space_am, physics=Physics()
    )
    assert check_am.leakage < 2e-4
    assert all(v < 5e-4 for v in check_am.residual_quanta.values())
    assert check_am.fidelity > 0.999
    assert 0.97 < check_am.chi_rad / CHI_MAXIMAL_RAD < 1.0, (
        "the surrogate over-predicts chi by the Debye-Waller and carrier corrections"
    )
    assert max(tr.boundary_population.values()) < 1e-8


def test_ms_gate_from_the_scheduler_matches_the_native_matrix_up_to_the_open_spectator() -> None:
    """At arbitrary phases the exact two-mode gate matches MS(phi_0, phi_1, 2 chi) up to the open rocking-mode loop's leakage
    (2e-3) and overlaps the opposite-sign gate by less than 0.02."""
    dev = chain_device(2)
    modes = two_ion_modes(dev)
    drives = raman_gate_drives(2)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=20e3)
    space = spot_check_space(dev, modes, wf, (0, 1), Numerics())[0]
    table = table_with_waveform((0, 1), wf, device=dev, drives=raman_gate_drives(2))
    base, _ = exact_gate_check(dev, wf, (0, 1), drives, table, space=space, physics=Physics())
    for phases in ((0.3, 1.1), (-0.7, 2.0)):
        check, _ = exact_gate_check(
            dev,
            wf,
            (0, 1),
            drives,
            table,
            space=space,
            physics=Physics(),
            phases_rad=phases,
            chi_target_rad=base.chi_rad,
        )
        assert check.fidelity == pytest.approx(1.0 - base.leakage, abs=2e-3)
        target_wrong = qt.Qobj(native_ms(phases[0], phases[1], -2.0 * base.chi_rad) @ KET00)
        target_wrong.dims = [[2, 2], [1, 1]]
        assert float(np.real(qt.expect(check.internal, target_wrong))) < 0.02


# ---- the reference MS pulse against the native matrix ---------------------------------------------------------------

EPSILON_HZ = 10e3
"""The reference pulse's tones sit eps/2pi = 10 kHz inside the sideband, so one loop closes in 100 us."""


def _single_mode_fixture():
    """(device, drives, the one-mode GateModes of the x-COM, the closed-form pulse, the d_m = 12 space)."""
    device = chain_device(2)
    drives = raman_gate_drives(2)
    full = two_ion_modes(device)
    k = full.modes.index(X_COM_TWO_IONS)
    modes = GateModes(
        ions=(0, 1),
        modes=(X_COM_TWO_IONS,),
        omega_rad_s=(full.omega_rad_s[k],),
        eta={0: (full.eta[0][k],), 1: (full.eta[1][k],)},
        nbar=(0.0,),
    )
    waveform = symmetric_pulse(
        modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=EPSILON_HZ, all_modes=False
    ).waveform
    space, _classes = spot_check_space(device, modes, waveform, (0, 1), Numerics(caps={X_COM_TWO_IONS: 12}))
    return device, drives, modes, waveform, space


def _exact_play(builder: BuilderOptions):
    """The reference pulse calibrated exactly under ``builder`` and played once from |00> as one scheduled gate with the pair
    scheduler's spin phases: (device, modes, the calibrated waveform, the schedule, the infidelity to MS(0, 0, pi/2)|00>)."""
    device, drives, modes, waveform0, space = _single_mode_fixture()
    rabi, _stark = derived_seeds(device, drives)
    run = calibrate_entangling_angle(
        device,
        waveform0,
        (0, 1),
        drives,
        table_with_waveform((0, 1), waveform0, rabi_hz=rabi),
        space=space,
        physics=Physics(builder=builder),
        tolerance_rad=1e-7,
        max_iterations=8,
    )
    assert run.converged
    waveform = run.waveform
    table = table_with_waveform((0, 1), waveform, rabi_hz=rabi)
    spins, _ = ms_spin_phases(waveform, (0, 1), (0.0, 0.0), PhaseFrame())
    pulses = entangling_pulses(
        waveform,
        drives,
        spin_phases_rad=spins,
        t_start_s=0.0,
        table=table,
        gate_id="ms11",
        stark_compensation=False,
    )
    sched = Schedule(
        tuple(pulses),
        (),
        (),
        {0: 0.0, 1: 0.0},
        gates=(PlayedGate("ms11", "ms", (0, 1), waveform, drives[0].beams, 0.0, waveform.duration_s),),
    )
    traces = JointExactEngine(table=table, builder_options=builder).run_pulses(
        device,
        sched,
        space.initial_state([0, 0]),
        space,
        quiet_sample(),
        SeedSpec(0),
        Numerics(),
    )
    target = qt.Qobj((native_ms(0.0, 0.0, math.pi / 2) @ KET00).reshape(-1, 1), dims=[[2, 2], [1, 1]])
    infidelity = 1.0 - float(np.real(qt.expect(traces.final.internal, target)))
    return device, modes, waveform, sched, infidelity


def test_the_section_11_1_fixture_is_the_plan_s_pulse() -> None:
    """The Section 11.1 pulse: joint dimension 48 at d_m = 12, a 100 us loop with chi = pi/4, eta = 0.078577 and
    eta Omega/eps = 0.499583 (1/2 to 1e-3)."""
    _device, _drives, modes, waveform, space = _single_mode_fixture()
    assert space.dims == [2, 2, 12] and space.dimension == 48
    assert waveform.duration_s == pytest.approx(1.0 / EPSILON_HZ, rel=1e-12)
    assert waveform.chi_total_rad == pytest.approx(math.pi / 4, rel=1e-9)
    eta = abs(modes.eta[0][0])
    assert eta == pytest.approx(0.078577, abs=1e-6)
    amp = float(waveform.segments[0].amplitude_hz[(0, "blue")])
    # eta Omega/eps = 1/2 is the leading-order closure; the exact chi = pi/4 solve lands at 0.499583, a 8.3e-4 relative
    # correction from the |alpha|^2 term of the kernel
    assert eta * amp / EPSILON_HZ == pytest.approx(0.499583, rel=1e-5)
    assert eta * amp / EPSILON_HZ == pytest.approx(0.5, rel=1e-3)


@pytest.mark.slow
def test_the_section_11_1_pulse_reproduces_the_native_ms_matrix_inside_its_intrinsic_budget() -> None:
    """The exact play of the calibrated reference pulse misses MS(0, 0, pi/2) by 4.66e-5 (2 %), inside the off-resonant
    carrier term (Omega_tone/(2 nu))^2 = 1.143e-4, which the budget's carrier_scale (Omega_tone/(2 mu))^2 = 1.151e-4
    reports at the tones' detuning mu = nu - eps, with the budget's other terms at their closed forms."""
    device, modes, waveform, sched, infidelity = _exact_play(BuilderOptions(include_stark=False))
    omega_tone = TWO_PI * float(waveform.segments[0].amplitude_hz[(0, "blue")])
    nu = modes.omega_rad_s[0]
    off_resonant = (omega_tone / (2.0 * nu)) ** 2
    # 1.1430e-4 at the calibrated amplitude (1.1228e-4 at the closed-form one)
    assert off_resonant == pytest.approx(1.1430e-4, rel=2e-3)
    assert off_resonant == pytest.approx(1.1e-4, abs=5e-6)
    # the identity holds inside it, and is not trivially small: it IS the off-resonant carrier, twice over (two ions)
    assert infidelity == pytest.approx(4.659e-5, rel=2e-2), infidelity
    assert infidelity < off_resonant
    assert infidelity > 0.25 * off_resonant
    selection = select_space(
        device, sched, Numerics(caps={X_COM_TWO_IONS: 12}), nbar=dict.fromkeys(range(6), 0.0)
    )
    budget = intrinsic_budget(device, sched, selection)
    # (Omega_tone/(2 mu))^2 at mu/2pi = 2.99 MHz: the off-resonant term above to (nu/mu)^2 - 1 = 0.67 %
    mu = TWO_PI * float(waveform.segments[0].detuning_hz["blue"])
    assert budget["ms11.carrier_scale"] == pytest.approx((omega_tone / (2.0 * mu)) ** 2, rel=1e-12)
    assert budget["ms11.carrier_scale"] == pytest.approx(1.1507e-4, rel=2e-3)
    assert budget["ms11.carrier_scale"] == pytest.approx(off_resonant, rel=1e-2)
    assert infidelity < budget["ms11.carrier_scale"]
    # the loop closes, so the residual displacement is below 1e-3, and the ground state has no thermal Debye-Waller loss
    assert budget["ms11.debye_waller"] == 0.0
    assert budget["ms11.residual_displacement"] < 1e-3
    # the sideband element's Lamb-Dicke deficit: 1 - <n+1|D(i eta)|n>/(eta sqrt(n+1)) at the cap's top n = 11, reported
    # beside the budget and NOT summed (its mean is the s^2 calibration's rescaling, its spread the Debye-Waller term)
    assert budget["ms11.sideband_lamb_dicke_deficit"] == pytest.approx(3.0554e-2, rel=1e-2)
    # the Bessel force saturation is Roos Eq. 17: f = 1 - (J_0 + J_2)(2 Omega/mu) in the per-tone Omega at the tone
    # amplitude and the tone-to-carrier detuning, entered as the uncalibrated angle error's infidelity sin^2(pi f/2)
    gate = sched.gates[0]
    seg = gate.waveform.segments[0]
    omega_hz = max(abs(float(a)) for a in seg.amplitude_hz.values())
    mu_hz = min(abs(float(d)) for d in seg.detuning_hz.values())
    f = 1.0 - roos_force_saturation(TWO_PI * omega_hz, TWO_PI * mu_hz)
    # x = 2 Omega/mu = 0.0429 at Omega/2pi = 64.1 kHz against mu/2pi = 2.99 MHz: f = x^2/8
    assert mu_hz == pytest.approx(2.99e6, rel=1e-6)
    assert f == pytest.approx((2.0 * omega_hz / mu_hz) ** 2 / 8.0, rel=1e-3) and 2e-4 < f < 3e-4, f
    assert budget["ms11.bessel_saturation"] == pytest.approx(math.sin(math.pi * f / 2.0) ** 2, rel=1e-9)
    assert budget["ms11.bessel_saturation"] == pytest.approx(roos_bessel_saturation(gate.waveform), rel=1e-12)
    assert budget["ms11.bessel_saturation"] < 1e-5, "far inside the off-resonant carrier term"
    ms_terms = (
        budget["ms11.residual_displacement"]
        + budget["ms11.debye_waller"]
        + budget["ms11.carrier_scale"]
        + budget["ms11.bessel_saturation"]
        + budget["ms11.beat_phase_tilt"]
    )
    assert budget["total"] >= ms_terms * (1.0 - 1e-12), "the total carries the five summed MS terms"
    assert budget["total"] < budget["ms11.sideband_lamb_dicke_deficit"], (
        "the Lamb-Dicke deficit (3e-2 at the cap's top) is reported, not summed"
    )


@pytest.mark.slow
def test_the_same_identity_reaches_1e_6_with_lamb_dicke_order_and_rwa_on() -> None:
    """With lamb_dicke_order = 1 and the RWA the same pulse reproduces the native matrix to 2.32e-10 (20 %), below 1e-6."""
    *_, infidelity = _exact_play(
        BuilderOptions(lamb_dicke_order=1, rwa=True, frame="interaction", include_stark=False)
    )
    assert infidelity < 1e-6, infidelity
    assert infidelity == pytest.approx(2.32e-10, rel=0.2), infidelity


def test_sideband_lamb_dicke_deficit_against_eta_sqrt_n_plus_one() -> None:
    """The Lamb-Dicke deficit is 1 - |<n+1|D(i eta)|n>|/(eta sqrt(n+1)) at the top carried Fock level to 1e-12 (Wineland 1998
    Eq. 18), monotone in eta and zero at eta = 0."""

    def _selection(d: int) -> SpaceSelection:
        space = HilbertSpace((2, 2), (ModeTruncation(X_COM_TWO_IONS, d, (0, d - 1), 0.1),), None, ())
        return SpaceSelection(
            space=space,
            mode_class={X_COM_TWO_IONS: "resolved"},
            contribution={},
            nbar={X_COM_TWO_IONS: 0.0},
        )

    def _modes(eta: float) -> GateModes:
        return GateModes(
            ions=(0, 1),
            modes=(X_COM_TWO_IONS,),
            omega_rad_s=(TWO_PI * 3.0e6,),
            eta={0: (eta,), 1: (eta,)},
            nbar=(0.0,),
        )

    # the closed form at the top of the carried range
    for eta, d in ((0.08, 12), (0.05, 8), (0.15, 6)):
        n = d - 1
        want = abs(1.0 - rabi_matrix_element(n + 1, n, eta) / (eta * math.sqrt(n + 1.0)))
        assert sideband_lamb_dicke_deficit(_modes(eta), _selection(d)) == pytest.approx(want, rel=1e-12)
    # monotone in eta at a fixed cap, and vanishing as eta -> 0 (the linear-in-eta limit)
    values = [sideband_lamb_dicke_deficit(_modes(e), _selection(12)) for e in (1e-4, 0.01, 0.05, 0.08, 0.15)]
    assert values == sorted(values)
    assert values[0] < 1e-7 and values[-1] > 0.1
    # a mode with eta = 0 contributes nothing (no force to saturate)
    assert sideband_lamb_dicke_deficit(_modes(0.0), _selection(12)) == 0.0
