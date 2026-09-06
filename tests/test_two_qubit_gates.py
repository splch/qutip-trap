"""Two-qubit gate physics through the JOINT_EXACT engine against the closed forms of Section 4.4 (PLAN.md Sections 4.4.1,
4.4.3, 4.4.7, 6.2, 9.4, 9.16 rows 4.4-4, 4.4-5, 4.4-7, 6-3, 13-8)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.api import (
    Beam,
    Crystal,
    Device,
    Field,
    HilbertSpace,
    Mode,
    ModeTruncation,
    SeedSpec,
    SolverOptions,
    Trap,
)
from qutip_trap.calibration.entangling import exact_gate_check, gate_space, ms_schedule
from qutip_trap.control.native import ms as native_ms
from qutip_trap.control.shaping import (
    CHI_MAXIMAL_RAD,
    SINE_MOTION_PHASE_RAD,
    GateModes,
    SampledEnvelope,
    gate_modes,
    integrals_sampled,
    solve_amplitude_modulation,
    symmetric_pulse,
)
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import JointExactEngine
from qutip_trap.dynamics.hamiltonian import BuilderOptions
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.species import species
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI
from qutip_trap.validation.two_qubit_closed_forms import (
    ballance_dephasing_coefficient,
    chi_of_n,
    entanglement_infidelity_from_displacements,
    kirchmair_populations,
    landsman_average_gate_infidelity,
    ms_alpha,
    ms_closure_ratio,
    ms_gamma,
    ms_propagator,
    overlap_fidelity,
    sideband_coupling_squared_difference,
    sorensen_molmer_thermal_fidelity,
    spectator_loop_error,
    state_infidelity_uniform_input,
    thermal_debye_waller_infidelity,
    traced_out_fidelity,
    zhu_state_infidelity_as_printed,
)
from tests.fixtures import make_detector, make_hardware, make_noise
from tests.m4_fixtures import (
    X_COM_TWO_IONS,
    raman_gate_drives,
    table_with_waveform,
    two_ion_device,
    two_ion_modes,
)

ETA_ANCHOR = 0.05
NU_ANCHOR_HZ = 1.0e6
EPS_ANCHOR_HZ = 10e3


def anchor_device() -> Device:
    """Two 171Yb+ ions whose x-COM mode at 1 MHz carries eta = 0.05 exactly: the check_ms_closure.py fixture as a Device (the
    crossing angle of the 355 nm pair is chosen for it; the other modes carry eta = 0 or are frozen in the tests)."""
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
    trap = Trap(
        omega_hz=(1.0e6, 0.9e6, 0.5e6),
        axis_angle_rad=0.0,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    return Device(
        crystal=crystal,
        trap=trap,
        field=Field(5.0, (1.0, 0.0, 0.0), None),
        beams=(b1, b2),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
    )


ANCHOR_MODE = 3
ONE_MODE_SPACE = HilbertSpace((2, 2), (ModeTruncation(ANCHOR_MODE, 16, (0, 4), 0.1),), None, (0, 1, 2, 4, 5))
ONE_MODE_OPTIONS = BuilderOptions(frozen_debye_waller=False)
"""The rocking mode (eta = 0.056, frozen) is switched off entirely: the anchor's fixture has one mode."""


def _run(
    dev: Device, wf: Waveform, space: HilbertSpace, opts: BuilderOptions, internal=(0, 0), nbar=None, store=2
):  # type: ignore[no-untyped-def]
    table = table_with_waveform((0, 1), wf)
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
        SolverOptions(),
    )
    return tr, eng


@pytest.fixture(scope="module")
def anchor():  # type: ignore[no-untyped-def]
    dev = anchor_device()
    modes = gate_modes(dev, (0, 1), (0, 1)).subset([ANCHOR_MODE])
    assert modes.eta[0] == pytest.approx((ETA_ANCHOR,), rel=1e-12)
    return dev, modes


def test_check_ms_closure_anchors_reproduce_through_the_package(anchor) -> None:  # type: ignore[no-untyped-def]
    """Section 9.4 row 2 [recomputed here, check_ms_closure.py]: eta = 0.05, nu = 2 pi x 1 MHz, eps = 2 pi x 10 kHz, d_m = 16, one loop
    from |dd>: concurrence 0.9999 with populations (0.4926, 0, 0, 0.5074) at eta Omega/eps = 1/2 and 0.38 (chi = pi/16) at 1/4, the
    tones at equal phase as the script plays them."""
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
    assert ms_closure_ratio(1) == 0.5


def test_sine_motion_phase_tilts_the_spin_axis_by_the_carrier_rotation(anchor) -> None:  # type: ignore[no-untyped-def]
    """M4 finding (Roos 2008's psi = (4 Omega_Roos/delta) sin zeta): with the tone phases differing by pi (Choi's sine convention) the
    carrier's frame rotation has the mean 2 Omega/mu, the entangling axis tilts by 0.2 rad and the |dd> gate leaks sin^2(0.2)
    into |du>, |ud>; with equal tone phases the tilt vanishes."""
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
    """Section 9.4 row 1: with the displacement expanded to first order and the resonant sidebands kept (frame='interaction', rwa,
    lamb_dicke_order=1) the drive IS the spin-dependent force and the state equals D(alpha S_y) exp[i gamma S_y^2]|dd, 0> with
    alpha = (eta Omega/(2 eps))(e^{i eps t} - 1), gamma = lambda t - chi sin(eps t), at every stored time; the populations follow
    Kirchmair's envelopes with nbar = 0."""
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
        # the force axis is phi_s + pi/2 = pi/2: S_y for equal tone phases at zero (Section 4.3.4)
        # the plan's -(hbar eta Omega/2) sign of alpha; the MS(0, 0) schedule's axes are X on ion 0 and -X on ion 1 (S = X_0 - X_1)
        u = ms_propagator(-alpha, gamma, 16, phi_rad=0.0, phi2_rad=math.pi)
        ref = (u * psi0).ptrace([0, 1])
        worst = max(worst, 1.0 - float(qt.fidelity(ref, state)) ** 2)
        p = np.real(np.diag(state.full()))
        p0, p1, p2 = kirchmair_populations(abs(alpha), gamma, 0.0)
        # Kirchmair's bright state is |d> (the fluorescing S1/2 level): p_2 = P_00, p_0 = P_11
        assert p[0] == pytest.approx(p2, abs=2e-6) and p[3] == pytest.approx(p0, abs=2e-6)
        assert p[1] + p[2] == pytest.approx(p1, abs=2e-6)
    assert worst < 1e-5
    # the sign: alpha's phase convention and the sign of the geometric phase are checked through the full state, the closure exactly
    assert tr.final.motional.nbar[ANCHOR_MODE] < 1e-9


def test_thermal_envelopes_and_debye_waller_references(anchor) -> None:  # type: ignore[no-untyped-def]
    """Kirchmair Eq. 14 with a thermal mode (first-order force: exact at nbar = 1), the three thermal references of Section 4.4.7 (1) in
    units of (pi^2/4) eta^4 (2.000, 3.000, 4.250 at nbar = 1), Sorensen-Molmer's Var(n) form and the Debye-Waller law chi(n) = chi_0
    [1 - eta^2 (2n + 1)] from the exact Laguerre elements (9.16 row 4.4-7)."""
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
    # the three thermal references in units of (pi^2/4) eta^4 (the exact thermal curve of the calibrated gate is in
    # tests/test_gate_calibration.py, Fock-resolved)
    assert thermal_debye_waller_infidelity(0.1, 1.0, "mean") / ((math.pi**2 / 4) * 1e-4) == pytest.approx(2.0)
    assert thermal_debye_waller_infidelity(0.1, 1.0, "n0") / ((math.pi**2 / 4) * 1e-4) == pytest.approx(3.0)
    assert thermal_debye_waller_infidelity(0.1, 1.0, "minus_half") / (
        (math.pi**2 / 4) * 1e-4
    ) == pytest.approx(4.25)
    assert sorensen_molmer_thermal_fidelity(2, 0.1, 2.0) == pytest.approx(
        1.0 - math.pi**2 * 2 * 1e-4 * 2.0 / 8.0
    )
    for n in (1, 2, 3):
        assert sideband_coupling_squared_difference(0.123, n) / (
            0.123**2 * (1.0 - 0.123**2 * (2 * n + 1))
        ) == pytest.approx([1.0007, 1.0020, 1.0039][n - 1], abs=3e-4)
        assert chi_of_n(1.0, 0.123, n) == pytest.approx(1.0 - 0.123**2 * (2 * n + 1))


def test_symmetrized_kernel_is_exact_by_block_diagonal_integration() -> None:
    """Section 9.16 row 4.4-5: the first-order Lamb-Dicke Hamiltonian is block diagonal in the sigma_x basis, so chi follows exactly from
    the four blocks' phases; with non-proportional envelopes it equals the symmetrized kernel and neither printed reading."""
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
    assert ints.chi_of(0, 1) == pytest.approx(chi_exact, rel=2e-4)
    from scipy.integrate import cumulative_simpson, simpson

    f = np.cos(mu * t) * np.exp(1j * omega * t)

    def k_of(env_i: np.ndarray, env_j: np.ndarray) -> float:
        return float(
            eta
            * eta
            * simpson(np.imag(env_i * f * np.conj(cumulative_simpson(env_j * f, x=t, initial=0.0))), x=t)
        )

    kab, kba = k_of(env_a, env_b), k_of(env_b, env_a)
    assert kab + kba == pytest.approx(chi_exact, rel=2e-4)
    assert abs(2 * kab / chi_exact - 1.0) > 0.1 and abs(2 * kba / chi_exact - 1.0) > 0.1


def test_residual_displacement_conversions_by_direct_integration(anchor) -> None:  # type: ignore[no-untyped-def]
    """Section 9.4 'Residual displacement': an open loop (1.1 loops) with the first-order force: eps_ent = sum |alpha|^2 equals the final
    mode energy exactly, 1 - F_ent = eps_ent to first order, Landsman's (4/5) eps_ent is the average gate infidelity, the |dd> state
    infidelity equals eps_ent (a uniform input), so Zhu's printed sum/4 is one quarter of what direct integration gives."""
    dev, modes = anchor
    eps = TWO_PI * EPS_ANCHOR_HZ
    tau = 1.1 * 2.0 * math.pi / eps
    sp = symmetric_pulse(modes, gate_mode=ANCHOR_MODE, loops=1, epsilon_hz=EPS_ANCHOR_HZ, kernel="rwa")
    omega = TWO_PI * sp.diagnostics["rabi_hz"]
    from qutip_trap.control.shaping import SegmentedEnvelope, integrals_segmented, waveform_from_segmented

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
        motional[s] = (
            spin_vec.conj() @ arr
        )  # <s| psi> as a motional ket (the S eigenstates are preserved by the force)
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
    assert landsman_average_gate_infidelity(1.0 - f_ent) == pytest.approx(0.8 * (1.0 - f_ent))
    assert state_infidelity_uniform_input(alphas, 0.0) == pytest.approx(eps_ent, rel=0.1)
    assert zhu_state_infidelity_as_printed([alphas[0]], [alphas[1]], [0.0]) == pytest.approx(eps_ent / 4.0)
    a1 = alphas[0]
    assert overlap_fidelity(a1) == pytest.approx(math.exp(-(abs(a1) ** 2))) and traced_out_fidelity(
        a1
    ) == pytest.approx(0.5 * (1 + math.exp(-2 * abs(a1) ** 2)))
    assert 1.0 - traced_out_fidelity(a1) == pytest.approx(0.5 * (1.0 - overlap_fidelity(a1) ** 2), rel=1e-12)


def test_spectator_loop_error_and_am_closure_exactly() -> None:
    """Section 4.4.7: the open rocking-mode loop of the symmetric COM pulse costs sum_j |alpha_jm|^2 quanta, the derived
    spectator-loop form pi^2 N K (omega_g/omega_m) sin^2(delta t_g/2)/(delta t_g)^2 with 2n + 1 = 1; the five-segment AM pulse closes it
    (M4 exit criterion: JOINT_EXACT at two ions with both resolved x modes)."""
    dev = two_ion_device()
    modes = two_ion_modes(dev)
    drives = raman_gate_drives(2)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=20e3)
    space = gate_space(modes, 2, waveform=wf)
    table = table_with_waveform((0, 1), wf)
    check, _ = exact_gate_check(dev, wf, (0, 1), drives, table, space=space)
    rock = 2
    closed_form = sum(
        abs(wf.alpha_m[rock]) ** 2 for _ in (0, 1)
    )  # the worst ion's alpha, equal for both by symmetry
    assert check.residual_quanta[rock] == pytest.approx(closed_form, rel=0.05)
    assert check.residual_quanta[X_COM_TWO_IONS] < 1e-3
    # the derived spectator-loop form assumes the single-mode closure Omega = pi sqrt K/(eta_g t_g) and the rwa force: exact for it
    from qutip_trap.control.shaping import integrals_segmented, symmetric_pulse

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
    table_am = table_with_waveform((0, 1), am.waveform)
    check_am, tr = exact_gate_check(dev, am.waveform, (0, 1), drives, table_am, space=space_am)
    assert check_am.leakage < 2e-4
    assert all(v < 5e-4 for v in check_am.residual_quanta.values())
    assert check_am.fidelity > 0.999
    assert 0.97 < check_am.chi_rad / CHI_MAXIMAL_RAD < 1.0, (
        "the surrogate over-predicts chi by the Debye-Waller and carrier corrections"
    )
    assert max(tr.boundary_population.values()) < 1e-8


def test_motional_dephasing_coefficient_alpha_k() -> None:
    """Section 9.16 row 6-3 and Section 6.2: L = a^dag a sqrt(2/tau) on the K-loop force model, one Liouvillian exponential per sigma_x
    block pair; (1 - F) tau/t_g -> (8K + 3)/(16 K^2) = 11/16, 19/64, 35/256 (Ballance prints 0.686 for K = 1)."""
    d = 24
    a = qt.destroy(d)
    n = a.dag() * a
    for loops, target in ((1, 11 / 16), (2, 19 / 64), (4, 35 / 256)):
        eps = 1.0
        t_g = 2.0 * math.pi * loops / eps
        force = eps / (4.0 * math.sqrt(loops))
        tau = t_g / 1e-4
        lind = math.sqrt(2.0 / tau) * n
        rho0 = qt.basis(d, 0).proj()
        s_vals = [2, 0, 0, -2]
        pure = {}
        blocks = {}
        for i, s in enumerate(s_vals):
            h_s = eps * n + s * force * (a + a.dag())
            pure[i] = (-1j * h_s * t_g).expm() * qt.basis(d, 0)
            for j, sp in enumerate(s_vals):
                h_sp = eps * n + sp * force * (a + a.dag())
                lio = -1j * (qt.spre(h_s) - qt.spost(h_sp)) + qt.lindblad_dissipator(lind)
                blocks[(i, j)] = qt.vector_to_operator((lio * t_g).expm() * qt.operator_to_vector(rho0))
        th = np.angle([complex(qt.basis(d, 0).overlap(pure[i])) for i in range(4)])
        f_ent = float(
            np.real(
                sum(blocks[(i, j)].tr() * np.exp(-1j * (th[i] - th[j])) for i in range(4) for j in range(4))
                / 16.0
            )
        )
        assert (1.0 - f_ent) * tau / t_g == pytest.approx(target, rel=3e-4)
        assert ballance_dephasing_coefficient(loops) == pytest.approx(target)
    assert ballance_dephasing_coefficient(1) == 0.6875 != pytest.approx(0.686, abs=1e-4)


def test_ms_gate_from_the_scheduler_matches_the_native_matrix_up_to_the_open_spectator() -> None:
    """The exact two-mode gate of the symmetric pulse against MS(phi_0, phi_1, 2 chi_exact): the phase conventions (axes at phi_i, the
    sign) hold for arbitrary phases, the residual being the open rocking-mode loop."""
    dev = two_ion_device()
    modes = two_ion_modes(dev)
    drives = raman_gate_drives(2)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=20e3)
    space = gate_space(modes, 2, waveform=wf)
    table = table_with_waveform((0, 1), wf)
    base, _ = exact_gate_check(dev, wf, (0, 1), drives, table, space=space)
    for phases in ((0.3, 1.1), (-0.7, 2.0)):
        check, _ = exact_gate_check(
            dev, wf, (0, 1), drives, table, space=space, phases_rad=phases, chi_target_rad=base.chi_rad
        )
        assert check.fidelity == pytest.approx(1.0 - base.leakage, abs=2e-3)
        target_wrong = qt.Qobj(
            native_ms(phases[0], phases[1], -2.0 * base.chi_rad) @ np.array([1, 0, 0, 0], dtype=complex)
        )
        target_wrong.dims = [[2, 2], [1, 1]]
        assert float(np.real(qt.expect(check.internal, target_wrong))) < 0.02
