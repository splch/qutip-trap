"""Section 9.7 error budgets through the JOINT_EXACT engine and the force model (PLAN.md Sections 4.4.7, 6.2 to 6.6; Section
9.7 rows 'Heating during gate', 'Motional dephasing', 'Dephasing correlation', 'Intensity noise', 'Crosstalk unitary',
'Parallel gates', 'Depolarizing summary', 'Over-rotation twirl'; Section 9.16 rows 4.4-8, 6-3, 6-4; Section 9.12 'Fang echo
identities'; M7)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt
from scipy.linalg import expm

from qutip_trap.api import HilbertSpace, ModeTruncation, SolverOptions, Waveform
from qutip_trap.calibration.entangling import exact_gate_check
from qutip_trap.dynamics.channels import (
    heating_channels,
    motional_dephasing_channels,
)
from qutip_trap.noise.summary import (
    average_gate_infidelity,
    choi_from_unitary,
    depolarizing_choi,
    depolarizing_rate,
    entanglement_infidelity,
    over_rotation_twirl_probability,
    pauli_string,
    pauli_twirl,
)
from qutip_trap.validation.noise_closed_forms import (
    intensity_noise_amplitude_n_ions,
    intensity_noise_error_n_ions,
    intensity_noise_intercept_n_ions,
)
from qutip_trap.validation.two_qubit_closed_forms import (
    ballance_dephasing_coefficient,
    ballance_dephasing_error,
    ballance_heating_error,
    dephasing_error_derived,
    fang_bell_fidelity,
    fang_crosstalk_unitary,
    fang_printed_bell_fidelity,
    fang_spectator_excitation,
    intensity_noise_error_derived,
    intensity_noise_error_printed,
    inter_pair_phase_scaling,
    kirchmair_heating_error,
    landsman_parallel_gate_bound,
    sigma_phi,
)
from tests.m4_fixtures import (
    X_COM_TWO_IONS,
    raman_gate_drives,
    table_with_waveform,
    two_ion_device,
    two_ion_modes,
)

FAST = SolverOptions(mesolve_dimension_max=4096)


def _gate(loops: int, epsilon_hz: float):  # type: ignore[no-untyped-def]
    """A K-loop symmetric MS gate closed on the COM alone (``all_modes=False``) with the rocking mode frozen (its n = 0
    Debye-Waller factor kept): dims 2 x 2 x 12. The two-mode closure would over-rotate by the frozen mode's angle and distort the
    loss (M7 diagnostic: 0.84 of the closed form with it, 0.96 without)."""
    dev = two_ion_device()
    modes = two_ion_modes(dev)
    wf = Waveform.symmetric(
        modes, gate_mode=X_COM_TWO_IONS, loops=loops, epsilon_hz=epsilon_hz, all_modes=False
    )
    space = HilbertSpace((2, 2), (ModeTruncation(X_COM_TWO_IONS, 12, (0, 4), 0.15),), None, (0, 1, 2, 4, 5))
    table = table_with_waveform((0, 1), wf, device=dev, drives=raman_gate_drives(2))
    return dev, wf, space, table


def _check(dev, wf, space, table, channels=()):  # type: ignore[no-untyped-def]
    check, _ = exact_gate_check(
        dev, wf, (0, 1), raman_gate_drives(2), table, space=space, channels=channels, options=FAST
    )
    return check


@pytest.mark.slow
@pytest.mark.parametrize("loops", [1, 2])
def test_heating_during_the_gate_costs_ndot_tg_over_2k(loops: int) -> None:
    """Ballance: eps_h = ndot t_g/(2K) for the K-loop gate; Kirchmair: Delta F = Gamma_h t_g/2 at K = 1 (the heating channel of Section
    4.1.5 on the gate mode, mesolve)."""
    dev, wf, space, table = _gate(loops, 20e3 * loops)
    base = _check(dev, wf, space, table)
    ndot = 400.0
    noisy = _check(dev, wf, space, table, channels=heating_channels(space, {X_COM_TWO_IONS: ndot}))
    t_g = wf.duration_s
    loss = base.fidelity - noisy.fidelity
    assert loss == pytest.approx(ballance_heating_error(ndot, t_g, loops), rel=0.08)
    if loops == 1:
        assert ballance_heating_error(ndot, t_g, 1) == kirchmair_heating_error(ndot, t_g)
    assert loss > 2e-3


@pytest.mark.slow
@pytest.mark.parametrize("loops", [1, 2])
def test_motional_dephasing_costs_alpha_k_tg_over_tau(loops: int) -> None:
    """Ballance: L = a^dag a sqrt(2/tau) gives eps_d = alpha_K t_g/tau with alpha_K = (8K + 3)/(16 K^2) (0.6875, 0.2969), not 0.686."""
    dev, wf, space, table = _gate(loops, 20e3 * loops)
    base = _check(dev, wf, space, table)
    t_g = wf.duration_s
    tau = t_g / 5e-3
    noisy = _check(dev, wf, space, table, channels=motional_dephasing_channels(space, {X_COM_TWO_IONS: tau}))
    loss = base.fidelity - noisy.fidelity
    assert loss == pytest.approx(ballance_dephasing_error(t_g, tau, loops), rel=0.1)
    assert ballance_dephasing_coefficient(1) == 11 / 16 and ballance_dephasing_coefficient(4) == 35 / 256


def _force_model(n_ions: int, loops: int, d: int = 16):  # type: ignore[no-untyped-def]
    """The rotating-frame MS force H = eps n + f (a + a^dag) S_x closing after K loops with chi = pi/4 per pair (eps = 1)."""
    a = qt.destroy(d)
    n = a.dag() * a
    eps = 1.0
    t_g = 2.0 * math.pi * loops / eps
    force = eps / (4.0 * math.sqrt(loops))
    sx = sum(
        qt.tensor(*[qt.sigmax() if k == i else qt.qeye(2) for k in range(n_ions)]) for i in range(n_ions)
    )
    idq = qt.tensor(*[qt.qeye(2)] * n_ions)
    h = eps * qt.tensor(idq, n) + force * qt.tensor(sx, a + a.dag())
    return h, t_g, force, sx, a, n_ions


def _final_spin_fidelity(h, t_g, c_ops, psi_spin, rho_mode, d, n_ions):  # type: ignore[no-untyped-def]
    rho0 = qt.tensor(qt.ket2dm(psi_spin), rho_mode)
    opts = {
        "atol": 1e-11,
        "rtol": 1e-9,
        "nsteps": 10**6,
    }  # the losses are 1e-6 to 1e-5: both solves need tight tolerances
    ideal = qt.mesolve(h, rho0, [0.0, t_g], [], options=opts).final_state
    noisy = qt.mesolve(h, rho0, [0.0, t_g], c_ops, options=opts).final_state
    keep = list(range(n_ions))
    rho_i = ideal.ptrace(keep)
    rho_n = noisy.ptrace(keep)
    # the ideal spin state is pure at closure
    return float(np.real((rho_n * rho_i).tr()))


@pytest.mark.slow
def test_dephasing_correlation_local_against_global_at_three_ions() -> None:
    """Bermudez derived (Section 4.4.7): eps_d <= N t_g/(2 T2) for uncorrelated and N^2 t_g/(4 T2) for globally correlated field noise,
    T2 the single-ion 1/e time; the two coincide at N = 2 and differ by 1.5 at N = 3, where the bounds are checked from above and the
    ordering below (Bermudez's printed 2 t_g N/T2 and 2 t_g N^2/T2 are 4 and 8 times high)."""
    d = 12
    h, t_g, _f, _sx, _a, n = _force_model(3, 1, d)
    t2 = t_g / 2e-3
    gamma = 1.0 / t2
    zs = [qt.tensor(*[qt.sigmaz() if k == i else qt.qeye(2) for k in range(n)]) for i in range(n)]
    ida = qt.qeye(d)
    local = [math.sqrt(gamma / 2.0) * qt.tensor(z, ida) for z in zs]
    glob = [math.sqrt(gamma / 2.0) * qt.tensor(sum(zs[1:], zs[0]), ida)]
    psi = qt.tensor(*[qt.basis(2, 0)] * n)
    f_local = _final_spin_fidelity(h, t_g, local, psi, qt.fock_dm(d, 0), d, n)
    f_glob = _final_spin_fidelity(h, t_g, glob, psi, qt.fock_dm(d, 0), d, n)
    bound_local = dephasing_error_derived(n, t_g, t2, correlated=False)
    bound_glob = dephasing_error_derived(n, t_g, t2, correlated=True)
    assert bound_local == pytest.approx(3e-3) and bound_glob == pytest.approx(4.5e-3)
    assert 1.0 - f_local <= bound_local * 1.02 and 1.0 - f_glob <= bound_glob * 1.02
    assert 1.0 - f_glob > 1.0 - f_local, "global field noise costs more than local at N = 3"
    assert 1.0 - f_local > 0.3 * bound_local
    assert dephasing_error_derived(2, t_g, t2, False) == dephasing_error_derived(2, t_g, t2, True)


@pytest.mark.slow
def test_intensity_noise_channel_two_term_budget_and_its_gamma_i_convention() -> None:
    """Section 9.16 row 4.4-8 in the force model: c_op = sqrt(2k) H_int on the K-loop MS force with Gamma_I = k Omega^2 the carrier-contrast
    decay rate (L = sqrt(2k)(Omega/2) sigma_x decays <sigma_z> at exactly k Omega^2). Fitting eps_I = A (2 nbar + 1) + B over nbar = 0, 1/2,
    1 at all five (N, K) the row asks for gives the two-term structure with the Gamma_I-independent ratio B/A = 3(N - 1)/(4K) of the
    DERIVED form (Bermudez's printed (N - 1)/4 would give 1/2), and A = **N** Gamma_I t_g eta^2/2: the plan's derived A is exact at
    N = 1 under the carrier-contrast Gamma_I and its eps_I first term is missing a factor N for an N-ion drive
    (``anchor.m7.intensity_noise_n_ions``; the retired ``anchor.m7.intensity_noise_factor_two`` read the N = 2 case as a factor 2 and
    inferred a Gamma_I convention instead). Measured A/plan_A: 1.00002, 2.00035, 3.00622, 2.00003, 1.99998, 3.00038 at
    (N, K) = (1,1), (2,1), (3,1), (2,2), (2,3), (3,2)."""
    d = 40
    eta = 0.1
    k = 2e-6
    nbars = [0.0, 0.5, 1.0]
    for n_ions, loops in ((1, 1), (2, 1), (3, 1), (2, 2), (2, 3), (3, 2)):
        h, t_g, force, sx, a, n = _force_model(n_ions, loops, d)
        omega = 2.0 * force / eta  # f = eta Omega/2 is the sideband coupling of the (hbar Omega/2) convention
        gamma_i = k * omega**2
        h_int = force * qt.tensor(sx, a + a.dag())
        c_ops = [math.sqrt(2.0 * k) * h_int]
        psi = qt.tensor(*[qt.basis(2, 0)] * n_ions)
        losses = []
        for nb in nbars:
            rho_m = qt.fock_dm(d, 0) if nb == 0.0 else qt.thermal_dm(d, nb)
            losses.append(1.0 - _final_spin_fidelity(h, t_g, c_ops, psi, rho_m, d, n))
        slope, intercept = np.polyfit([2 * nb + 1 for nb in nbars], losses, 1)
        plan_a = intensity_noise_error_derived(
            gamma_i, t_g, eta, 0.5, n_ions, loops
        ) - intensity_noise_error_derived(gamma_i, t_g, eta, 0.0, n_ions, loops)
        assert slope == pytest.approx(float(n_ions) * plan_a, rel=0.01), (
            f"A = N x the plan's Gamma_I t_g eta^2/2 at (N, K) = ({n_ions}, {loops})"
        )
        assert slope == pytest.approx(intensity_noise_amplitude_n_ions(gamma_i, t_g, eta, n_ions), rel=0.01)
        if n_ions > 1:
            assert intercept / slope == pytest.approx(3.0 * (n_ions - 1) / (4.0 * loops), rel=0.01), (
                "the derived ratio, not the printed (N - 1)/4"
            )
            assert intercept == pytest.approx(
                intensity_noise_intercept_n_ions(gamma_i, t_g, eta, n_ions, loops), rel=0.01
            )
            assert intensity_noise_error_n_ions(gamma_i, t_g, eta, 0.5, n_ions, loops) == pytest.approx(
                slope * 2.0 + intercept, rel=0.01
            )
        else:
            assert abs(intercept / slope) < 0.01, "no inter-ion term at N = 1"
        if (n_ions, loops) == (2, 1):
            unit = gamma_i * t_g * eta**2
            printed_ratio = (intensity_noise_error_printed(gamma_i, t_g, eta, 0.0, 2) - unit / 2.0) / (
                unit / 2.0
            )
            assert printed_ratio == pytest.approx(0.5) and intercept / slope != pytest.approx(0.5, rel=0.1)
    # the Gamma_I definition itself, on the N = 2 model the loop above left behind
    _h, _t_g, force, _sx, _a, _n = _force_model(2, 1, d)
    omega = 2.0 * force / eta
    gamma_i = k * omega**2
    # the carrier-contrast definition of Gamma_I: d<sigma_z>/dt = -k Omega^2 <sigma_z> under sqrt(2k)(Omega/2) sigma_x
    rho = qt.basis(2, 0).proj()
    lind = qt.lindblad_dissipator(math.sqrt(2.0 * k) * 0.5 * omega * qt.sigmax())
    drho = qt.vector_to_operator(lind * qt.operator_to_vector(rho))
    assert float(qt.expect(qt.sigmaz(), drho)) == pytest.approx(-gamma_i, rel=1e-12)


def test_fang_crosstalk_unitary_closed_forms_and_echo_identities() -> None:
    """Section 9.7 'Crosstalk unitary' and Section 9.16 row 6-4: F = cos^2 theta_13 cos^2 theta_23 and P_ion3 = [1 - cos 2 theta_13 cos 2
    theta_23]/2 exactly, independent of chi and phi_beam (0.900790 and 0.097217 at (0.1644, -0.2763); printed 0.974422 / 0.025450);
    Section 9.12: [prod_j Z^(j) U(theta/2)]^2 = exp(-i theta X1 X2) and [Y1 Y2 U(theta/2)]^2 = exp(-i theta X1 X2) to first order in the
    leaked angles, the echo schemes of Section 6.6."""
    rng = np.random.default_rng(0)
    bell = np.zeros(8, complex)
    bell[0], bell[6] = 1 / math.sqrt(2), -1j / math.sqrt(2)
    psi0 = np.zeros(8, complex)
    psi0[0] = 1.0
    worst = 0.0
    for _ in range(300):
        chi, phi = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
        t13, t23 = rng.uniform(-0.5, 0.5, 2)
        u = fang_crosstalk_unitary(math.pi / 4, t13, t23, phi)
        psi = u @ psi0
        p3 = sum(abs(psi[k]) ** 2 for k in range(8) if k % 2 == 1)
        worst = max(
            worst,
            abs(abs(np.vdot(bell, psi)) ** 2 - fang_bell_fidelity(t13, t23)),
            abs(p3 - fang_spectator_excitation(t13, t23)),
        )
        _ = chi
    assert worst < 1e-12
    assert fang_bell_fidelity(0.1644, -0.2763) == pytest.approx(0.900790, abs=1e-6)
    assert fang_spectator_excitation(0.1644, -0.2763) == pytest.approx(0.097217, abs=1e-6)
    assert fang_printed_bell_fidelity(0.1644, -0.2763) == pytest.approx(0.974422, abs=1e-6)
    assert fang_spectator_excitation(math.pi / 2, math.pi / 2) == pytest.approx(0.0, abs=1e-12)
    # echo identities: the leaked terms flip sign under a physical Z(pi) on the spectator or Y(pi) on both targets
    theta, t13, t23, phi = 0.7, 0.05, -0.03, 0.4
    x1x2 = np.kron(np.kron(pauli_string("X"), pauli_string("X")), np.eye(2))
    target = expm(-1j * theta * x1x2)
    half = fang_crosstalk_unitary(theta / 2, t13 / 2, t23 / 2, phi)
    z3 = np.kron(np.eye(4), pauli_string("Z"))
    y12 = np.kron(np.kron(pauli_string("Y"), pauli_string("Y")), np.eye(2))
    plain = half @ half
    echo_z = z3 @ half @ z3 @ half
    echo_y = y12 @ half @ y12 @ half
    dist = lambda u: np.linalg.norm(u - target, 2)  # noqa: E731
    assert dist(echo_z) < 0.02 * dist(plain) and dist(echo_y) < 0.02 * dist(plain)
    assert dist(echo_z) < 2e-3 and dist(plain) > 0.05
    assert np.allclose(sigma_phi(0.0), pauli_string("X")) and np.allclose(
        sigma_phi(math.pi / 2), pauli_string("Y")
    )


def _inter_pair_unitary(thetas):  # type: ignore[no-untyped-def]
    """exp[-i sum_rs Theta_rs Z_r Z_s], the commuting unitary two parallel gates add on top of the two XX(chi) they
    intend (Section 6.6, Landsman 2019). Pairs (0, 1) and (2, 3), so the four inter-pair terms are (0,2), (0,3), (1,2),
    (1,3). Built directly as a four-qubit operator, NOT through ``schedule(parallel=True)``, which serializes MS gates."""
    h = 0
    for th, (r, s) in zip(thetas, [(0, 2), (0, 3), (1, 2), (1, 3)]):
        ops = [qt.qeye(2)] * 4
        ops[r] = qt.sigmaz()
        ops[s] = qt.sigmaz()
        h = h + th * qt.tensor(*ops)
    return (-1j * h).expm()


def _diamond_lower_bound(u, n_random: int = 40, seed: int = 0) -> float:  # type: ignore[no-untyped-def]
    """A LOWER bound on (1/2)||E||_diamond for the unitary error channel E(rho) = U rho U^dag - rho.

    The diamond norm is by definition the supremum of ||(E (x) I)(rho)||_1/2 over states on the DOUBLED space, so any
    input gives a lower bound; for two pure states (1/2)||psi psi^dag - phi phi^dag||_1 = sqrt(1 - |<psi|phi>|^2).
    Sampled at the maximally entangled (Choi) input and at random pure states of the eight-qubit doubled space.
    """
    mat = np.asarray(u.full())
    d = mat.shape[0]
    ext = np.kron(mat, np.eye(d))
    phi = np.eye(d).reshape(-1) / math.sqrt(d)
    best = math.sqrt(max(0.0, 1.0 - abs(np.vdot(phi, ext @ phi)) ** 2))
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        psi = rng.normal(size=d * d) + 1j * rng.normal(size=d * d)
        psi = psi / np.linalg.norm(psi)
        best = max(best, math.sqrt(max(0.0, 1.0 - abs(np.vdot(psi, ext @ psi)) ** 2)))
    return best


def test_landsman_parallel_gate_bound_holds_against_the_simulated_channel_and_is_vacuous_above_one() -> None:
    """Section 9.7 row 'Parallel gates': (1/2)||E||_diamond <= sum_rs |Theta_rs| with the 1/2 load-bearing, and the bound
    is vacuous above 1 because ||E||_diamond <= 2 always, "so the simulator reports the exact simulated channel rather
    than the bound whenever the bound exceeds 1" (Section 6.6 **[corrected]**).

    Audit E-15: the test used to check only the right-hand side's arithmetic. The inequality itself is now tested against
    the EXACT four-qubit commuting channel through the diamond norm's own definition (a sampled lower bound over the
    doubled space), in the direction that can falsify it. Measured lower-bound-to-bound ratios: 0.6686 at
    (0.1, -0.05, 0.02, 0.01), 0.5533 at four equal 0.01, 0.5588 at (0.6, 0.5, 0.1, 0.1), 0.2880 at four equal 0.8 - and
    0.9996 when a SINGLE phase is non-zero, which is where the triangle-inequality sum is saturated."""
    cases = [
        [0.1, -0.05, 0.02, 0.01],
        [0.01, 0.01, 0.01, 0.01],
        [0.6, 0.5, 0.1, 0.1],
        [0.3, 0.3, 0.3, 0.3],
        [0.8, 0.8, 0.8, 0.8],
        [0.05, 0.0, 0.0, 0.0],
    ]
    for thetas in cases:
        bound, vacuous = landsman_parallel_gate_bound(thetas)
        assert bound == pytest.approx(sum(abs(t) for t in thetas))
        assert vacuous == (bound > 1.0)
        lower = _diamond_lower_bound(_inter_pair_unitary(thetas))
        assert lower <= bound + 1e-12, (thetas, lower, bound)
        assert lower <= 1.0 + 1e-12, "||E||_diamond <= 2 always, which is why the bound is vacuous above 1"
        if vacuous:
            # the fallback Section 6.6 requires: the exact channel is BELOW 1 where the bound says nothing
            assert lower < 1.0 and bound > 1.0, (thetas, lower, bound)
    # the bound is tight exactly when one phase carries everything (the triangle inequality is saturated)
    single = _diamond_lower_bound(_inter_pair_unitary([0.05, 0.0, 0.0, 0.0]))
    assert single / 0.05 == pytest.approx(0.9996, abs=2e-3), single
    spread = _diamond_lower_bound(_inter_pair_unitary([0.0125] * 4))
    assert spread / 0.05 < 0.6, "four equal phases make the sum a factor two loose"
    # a vanishing separation is not an excuse: Theta falls as 1/n^3 with the pair separation in sites
    assert inter_pair_phase_scaling(2, 0.08) == pytest.approx(0.01)
    assert inter_pair_phase_scaling(4, 0.08) == pytest.approx(0.08 / 64.0)
    assert landsman_parallel_gate_bound([inter_pair_phase_scaling(4, 0.08)] * 4)[1] is False


def test_depolarizing_summary_and_over_rotation_twirl() -> None:
    """Section 9.7: Lambda_eps's entanglement infidelity equals eps exactly (Chen 2023); the twirl of exp(-i alpha XX) is p_xx = sin^2
    alpha (Trout 2018); the average gate infidelity is (4/5) of the entanglement one for two qubits."""
    for n in (1, 2):
        for eps in (1e-3, 0.05):
            c = depolarizing_choi(eps, n)
            ideal = choi_from_unitary(np.eye(2**n))
            assert entanglement_infidelity(c, ideal) == pytest.approx(eps, rel=1e-10)
            assert depolarizing_rate(c, ideal) == pytest.approx(eps, rel=1e-10)
            assert np.allclose(c, c.conj().T) and np.trace(c).real == pytest.approx(1.0)
    alpha = 0.23
    u = expm(-1j * alpha * pauli_string("XX"))
    tw = pauli_twirl(choi_from_unitary(u), 2)
    assert tw["XX"] == pytest.approx(math.sin(alpha) ** 2, abs=1e-12) and tw["II"] == pytest.approx(
        math.cos(alpha) ** 2, abs=1e-12
    )
    assert (
        sum(tw.values()) == pytest.approx(1.0)
        and over_rotation_twirl_probability(alpha) == math.sin(alpha) ** 2
    )
    assert average_gate_infidelity(1e-3, 4) == pytest.approx(0.8e-3)
    assert entanglement_infidelity(choi_from_unitary(u), choi_from_unitary(np.eye(4))) == pytest.approx(
        math.sin(alpha) ** 2, abs=1e-12
    )
