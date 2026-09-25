"""Two-qubit gate physics through the JOINT_EXACT engine against the closed forms (PLAN.md Section 4.4), and the reference MS
pulse (one 100 us loop on the two-ion chain's x-COM mode at d_m = 12) against the native matrix inside its intrinsic budget."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt
from scipy.integrate import cumulative_simpson, simpson

from qutip_trap.calibration.entangling import (
    calibrate_entangling_angle,
    exact_gate_check,
    gate_space,
    ms_schedule,
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
from qutip_trap.options import Numerics
from qutip_trap.published import (
    kirchmair_populations,
    ms_alpha,
    ms_gamma,
    roos_force_saturation,
    thermal_debye_waller_infidelity,
)
from qutip_trap.run.job import intrinsic_budget, roos_bessel_saturation, sideband_lamb_dicke_deficit
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


def anchor_device() -> Device:
    """Two 171Yb+ ions whose x-COM mode at 1 MHz carries eta = 0.05 exactly (the 355 nm pair's crossing angle is chosen for
    it)."""
    yb = species("171Yb+")
    mass = yb.mass_u * ATOMIC_MASS_KG
    x0 = math.sqrt(HBAR_J_S / (2.0 * mass * TWO_PI * NU_ANCHOR_HZ))
    dk = ETA_ANCHOR / (x0 / math.sqrt(2.0))
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
):  # type: ignore[no-untyped-def]
    # the closed forms carry no light shift: derived Rabi entries, no Stark entry, and an engine without the table
    table = table_with_waveform((0, 1), wf, device=dev, drives=raman_gate_drives(2), stark_hz={})
    sched = ms_schedule(wf, (0, 1), raman_gate_drives(2), table)
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
def anchor():  # type: ignore[no-untyped-def]
    dev = anchor_device()
    modes = gate_modes(dev, (0, 1), (0, 1)).subset([ANCHOR_MODE])
    assert modes.eta[0] == pytest.approx((ETA_ANCHOR,), rel=1e-12)
    return dev, modes


def test_ms_closure_anchors_reproduce_through_the_package(anchor) -> None:  # type: ignore[no-untyped-def]
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


def test_sine_motion_phase_tilts_the_spin_axis_by_the_carrier_rotation(anchor) -> None:  # type: ignore[no-untyped-def]
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


def test_exact_ms_propagator_first_order_lamb_dicke(anchor) -> None:  # type: ignore[no-untyped-def]
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


def test_thermal_envelopes_and_debye_waller_references(anchor) -> None:  # type: ignore[no-untyped-def]
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

            def coef_conj(tt: float, c=coef, **kwargs: object) -> complex:  # type: ignore[no-untyped-def]
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


def test_residual_displacement_conversions_by_direct_integration(anchor) -> None:  # type: ignore[no-untyped-def]
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
    space = gate_space(modes, 2, waveform=wf)
    table = table_with_waveform((0, 1), wf, device=dev, drives=raman_gate_drives(2))
    check, _ = exact_gate_check(dev, wf, (0, 1), drives, table, space=space)
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
    space_am = gate_space(modes, 2, waveform=am.waveform)
    table_am = table_with_waveform((0, 1), am.waveform, device=dev, drives=raman_gate_drives(2))
    check_am, tr = exact_gate_check(dev, am.waveform, (0, 1), drives, table_am, space=space_am)
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
    space = gate_space(modes, 2, waveform=wf)
    table = table_with_waveform((0, 1), wf, device=dev, drives=raman_gate_drives(2))
    base, _ = exact_gate_check(dev, wf, (0, 1), drives, table, space=space)
    for phases in ((0.3, 1.1), (-0.7, 2.0)):
        check, _ = exact_gate_check(
            dev, wf, (0, 1), drives, table, space=space, phases_rad=phases, chi_target_rad=base.chi_rad
        )
        assert check.fidelity == pytest.approx(1.0 - base.leakage, abs=2e-3)
        target_wrong = qt.Qobj(native_ms(phases[0], phases[1], -2.0 * base.chi_rad) @ KET00)
        target_wrong.dims = [[2, 2], [1, 1]]
        assert float(np.real(qt.expect(check.internal, target_wrong))) < 0.02


# ---- the reference MS pulse against the native matrix ---------------------------------------------------------------

EPSILON_HZ = 10e3
"""The reference pulse's tones sit eps/2pi = 10 kHz inside the sideband, so one loop closes in 100 us."""


def _single_mode_fixture():  # type: ignore[no-untyped-def]
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
    space = gate_space(
        modes, 2, waveform=waveform, frozen=(0, 1, 2, 4, 5), n_modes_total=6, d_min=12, d_max=12
    )
    return device, drives, modes, waveform, space


def _exact_play(builder: BuilderOptions):  # type: ignore[no-untyped-def]
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
        tolerance_rad=1e-7,
        builder_options=builder,
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
    carrier term (Omega_tone/(2 nu))^2 = 1.143e-4 and the budget's carrier_scale 5.14e-4, with the budget's other terms at
    their closed forms."""
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
    # (Omega_tone/nu_min)^2 with nu_min the x-rocking mode at 2.8284 MHz
    assert budget["ms11.carrier_scale"] == pytest.approx(5.144e-4, rel=2e-2)
    assert budget["ms11.carrier_scale"] > 4.0 * off_resonant
    assert infidelity < budget["ms11.carrier_scale"]
    # the loop closes, so the residual displacement is below 1e-3, and the ground state has no thermal Debye-Waller loss
    assert budget["ms11.debye_waller"] == 0.0
    assert budget["ms11.residual_displacement"] < 1e-3
    # the sideband element's Lamb-Dicke deficit: 1 - <n+1|D(i eta)|n>/(eta sqrt(n+1)) at the cap's top n = 11, reported
    # beside the budget and NOT summed (its mean is the s^2 calibration's rescaling, its spread the Debye-Waller term)
    assert budget["ms11.sideband_lamb_dicke_deficit"] == pytest.approx(3.0554e-2, rel=1e-2)
    # the Bessel force saturation is Roos Eq. 17: f = 1 - (J_0 + J_2)(4 Omega/mu) at the tone amplitude and the
    # tone-to-carrier detuning, entered as the uncalibrated angle error's infidelity sin^2(pi f/2)
    gate = sched.gates[0]
    seg = gate.waveform.segments[0]
    omega_hz = max(abs(float(a)) for a in seg.amplitude_hz.values())
    mu_hz = min(abs(float(d)) for d in seg.detuning_hz.values())
    f = 1.0 - roos_force_saturation(TWO_PI * omega_hz, TWO_PI * mu_hz)
    # x = 4 Omega/mu = 0.085 at Omega/2pi = 64 kHz against mu/2pi = 3.01 MHz: f = x^2/8
    assert 1e-4 < f < 2e-3, f
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
