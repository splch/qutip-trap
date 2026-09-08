"""The Section 9.4 / 9.12 / 9.16 two-qubit rows the M4 audit found unasserted (PLAN.md Sections 4.4.1-4.4.7, 9.4, 9.12, 9.16, 13).

Every test here pins a row that existed only as a closed form or a printed line: the n-independence of the bichromatic
coupling (Sorensen-Molmer 1999 Fig. 4), the N = 4 GHZ time (Molmer-Sorensen 1999), the three spin normalizations, the
operator identity S_x^2 = 2(1 + sigma_x sigma_x), the withdrawn doubled Hamiltonian as a negative control, the alpha
prefactor controls, and the published consistency anchors (Kirchmair, Ballance, the records table). Consistency anchors
report against the published uncertainty or 20 % (Section 9's rule); closed forms and identities are pinned at 1e-9 and
1e-12 respectively.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
import qutip as qt
from scipy.special import jv

from qutip_trap.units import TWO_PI
from qutip_trap.validation.two_qubit_closed_forms import (
    PUBLISHED_RECORDS,
    ballance_heating_error,
    ballance_thermal_error,
    debye_waller_error_derived,
    effective_bichromatic_coupling_rad_s,
    entanglement_infidelity_from_displacements,
    ghz_coupling_rad_s,
    ghz_time_s,
    kirchmair_heating_error,
    leung_zero_temperature_error,
    ms_closure_ratio,
    ms_two_body_angle,
    recalibrated_gate_time_s,
    spin_projectors,
    state_infidelity_uniform_input,
    zhu_beta_bar,
)

# ---- Section 9.4 "n-independence": Sorensen-Molmer 1999 Fig. 4 ------------------------------------------------------------------

SM99_NU = 1.0
"""Fig. 4 is quoted in units of the trap frequency: delta = 0.9 nu, Omega = 0.1 nu, eta = 0.1, Gamma = 2e-4 nu."""
SM99_DELTA = 0.9 * SM99_NU
SM99_OMEGA = 0.1 * SM99_NU
SM99_ETA = 0.1
SM99_GAMMA = 2e-4 * SM99_NU


def _sm99_drive_coefficient(t: float, **_: object) -> float:
    """Omega cos(delta t): the two tones at +-delta from the carrier, in the frame rotating at the qubit frequency."""
    return SM99_OMEGA * math.cos(SM99_DELTA * t)


def _sm99_inversion(n_th: float, *, d: int = 14, t_max: float = 1800.0, n_points: int = 361) -> np.ndarray:
    """mesolve of the two-ion, one-mode bichromatic model with a thermal reservoir: (t, P_ee, P_gg).

    The Lamb-Dicke expansion is kept to second order, as Sorensen-Molmer's own model is, so the drive operator stays
    sparse; the reservoir is the Section 6 thermal pair sqrt(Gamma(n_th + 1)) a and sqrt(Gamma n_th) a^dag."""
    a = qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(d))
    x = qt.destroy(d) + qt.create(d)
    disp = qt.qeye(d) + 1j * SM99_ETA * x - 0.5 * (SM99_ETA * x) ** 2
    sp = qt.tensor(qt.sigmap(), qt.qeye(2), qt.qeye(d)) + qt.tensor(qt.qeye(2), qt.sigmap(), qt.qeye(d))
    drive = sp * qt.tensor(qt.qeye(2), qt.qeye(2), disp)
    ham = qt.QobjEvo([SM99_NU * a.dag() * a, [(drive + drive.dag()).to("CSR"), _sm99_drive_coefficient]])
    c_ops = [math.sqrt(SM99_GAMMA * (n_th + 1.0)) * a]
    if n_th > 0.0:
        c_ops.append(math.sqrt(SM99_GAMMA * n_th) * a.dag())
    down, up = qt.basis(2, 0), qt.basis(2, 1)
    rho0 = qt.tensor(qt.tensor(down, down).proj(), qt.thermal_dm(d, n_th) if n_th > 0.0 else qt.fock_dm(d, 0))
    p_ee = qt.tensor(up.proj(), up.proj(), qt.qeye(d))
    p_gg = qt.tensor(down.proj(), down.proj(), qt.qeye(d))
    times = np.linspace(0.0, t_max, n_points)
    res = qt.mesolve(
        ham,
        rho0,
        times,
        c_ops=c_ops,
        e_ops=[p_ee, p_gg],
        options={"atol": 1e-9, "rtol": 1e-7, "nsteps": 10**7},
    )
    return np.vstack([times, np.asarray(res.expect[0]), np.asarray(res.expect[1])])


@pytest.mark.slow
def test_bichromatic_coupling_is_n_independent() -> None:
    """Section 9.4 'n-independence': Omega~ = -(Omega eta)^2/(nu - delta) drives |gg n> <-> |ee n> at the SAME rate for
    n_th = 0 and n_th = 2 (Sorensen-Molmer 1999 Fig. 4's parameters, with the reservoir at Gamma = 2e-4 nu).

    P_ee = sin^2(Omega~ t/2) in the plan's (hbar Omega/2) convention, so the populations cross at t = pi/(2|Omega~|).
    Measured: 1520 (n_th = 0) and 1525 (n_th = 2) against pi/(2|Omega~|) = 1570.8, i.e. within 3.3 % of the closed form
    and within 0.4 % of each other - the n-independence the row asks for. The 3 % offset is the ac Stark correction of a
    drive at Omega/nu = 0.1 plus the second-order Lamb-Dicke truncation, and is a consistency anchor, not a closed form."""
    omega_tilde = effective_bichromatic_coupling_rad_s(SM99_OMEGA, SM99_ETA, SM99_NU, SM99_DELTA)
    assert omega_tilde == pytest.approx(-1e-3 * SM99_NU, rel=1e-12), "twice the monochromatic value, negative"
    half_inversion = math.pi / (2.0 * abs(omega_tilde))
    crossings = []
    for n_th in (0.0, 2.0):
        times, p_ee, p_gg = _sm99_inversion(n_th)
        idx = np.flatnonzero(p_ee > p_gg)
        assert idx.size, f"n_th = {n_th}: the inversion did not reach P_ee > P_gg within {times[-1]}"
        crossings.append(float(times[idx[0]]))
        assert p_ee[idx[0]] == pytest.approx(0.5, abs=0.05)
    for t_cross in crossings:
        assert t_cross == pytest.approx(half_inversion, rel=0.06)
    assert crossings[0] == pytest.approx(crossings[1], rel=0.02), (
        "the inversion period is n-independent: the row's claim"
    )


# ---- Section 9.4 "GHZ": Molmer-Sorensen 1999 at N = 4 ----------------------------------------------------------------------------


def test_ghz_at_four_ions_from_the_closed_form() -> None:
    """Section 9.4 'GHZ': H = 4 chi J_x^2 on |0000> gives the GHZ state at t = pi/(8 chi) for N = 4, and
    chi = eta^2 Omega^2 nu/(2(nu^2 - delta^2)) is that chi (Molmer-Sorensen 1999). Direct matrix exponential, so the
    plan's 1e-9 closed-form tolerance applies."""
    n = 4
    j_x = 0.5 * sum(
        (qt.tensor([qt.sigmax() if i == j else qt.qeye(2) for i in range(n)]) for j in range(n)),
        0.0 * qt.tensor([qt.qeye(2)] * n),
    )
    chi = 1.0
    t_ghz = ghz_time_s(chi)
    assert t_ghz == pytest.approx(math.pi / 8.0, rel=1e-12)
    psi = (-1j * (4.0 * chi * j_x * j_x) * t_ghz).expm() * qt.tensor([qt.basis(2, 0)] * n)
    pops = np.abs(np.asarray(psi.full()).ravel()) ** 2
    assert pops[0] == pytest.approx(0.5, abs=1e-9) and pops[-1] == pytest.approx(0.5, abs=1e-9)
    assert float(np.sum(pops[1:-1])) < 1e-9, "no population outside |0000> and |1111>"
    ghz = (qt.tensor([qt.basis(2, 0)] * n) + 1j * qt.tensor([qt.basis(2, 1)] * n)).unit()
    assert abs(ghz.overlap(psi)) ** 2 == pytest.approx(1.0, rel=1e-9)
    # the row's chi: eta^2 Omega^2 nu/(2(nu^2 - delta^2)) with the Fig. 4 ratios
    nu, delta = TWO_PI * 1.0e6, TWO_PI * 0.9e6
    omega, eta = TWO_PI * 100e3, 0.1
    chi_row = ghz_coupling_rad_s(eta, omega, nu, delta)
    assert chi_row == pytest.approx(eta**2 * omega**2 * nu / (2.0 * (nu**2 - delta**2)), rel=1e-12)
    assert ghz_time_s(chi_row) == pytest.approx(math.pi / (8.0 * chi_row), rel=1e-12)


@pytest.mark.slow
def test_ghz_at_four_ions_through_the_microscopic_bichromatic_drive() -> None:
    """The same GHZ state from the MICROSCOPIC bichromatic drive rather than the effective H = 4 chi J_x^2: four qubits x
    one mode of 12 Fock levels is 2^4 x 12 = 192, well inside the 4096 space guard of Section 5.1, so the microscopic run
    fits and the M4 exit criterion's 2-to-3-ion limit is not binding here.

    At Sorensen-Molmer's own Fig. 4 ratios (eta = 0.1, Omega = 0.1 nu, delta = 0.9 nu) the effective Hamiltonian is a
    second-order approximation, so this is a consistency anchor: the populations land on 0.5/0.5 to 5 % and the state is a
    GHZ state UP TO LOCAL Z PHASES (the off-resonant carrier and the ac Stark shift rotate each qubit's phase, which
    exp(i 4 chi J_x^2) does not), so the overlap is maximized over the relative phase - the entanglement structure is what
    the row claims, not a particular phase convention."""
    n, d = 4, 12
    nu, eta = 1.0, 0.1
    delta, omega = 0.9 * nu, 0.1 * nu
    chi = ghz_coupling_rad_s(eta, omega, nu, delta)
    t_ghz = ghz_time_s(chi)
    ident = [qt.qeye(2)] * n + [qt.qeye(d)]
    a = qt.tensor(*(ident[:-1] + [qt.destroy(d)]))
    x = qt.destroy(d) + qt.create(d)
    disp = qt.qeye(d) + 1j * eta * x - 0.5 * (eta * x) ** 2 - 1j / 6.0 * (eta * x) ** 3
    sp = 0.0 * a
    for j in range(n):
        sp = sp + qt.tensor(*([qt.sigmap() if i == j else qt.qeye(2) for i in range(n)] + [qt.qeye(d)]))
    drive = sp * qt.tensor(*(ident[:-1] + [disp]))

    def coef(t: float, **_: object) -> float:
        return float(omega * math.cos(delta * t))

    ham = qt.QobjEvo([nu * a.dag() * a, [(drive + drive.dag()).to("CSR"), coef]])
    psi0 = qt.tensor(*([qt.basis(2, 0)] * n + [qt.basis(d, 0)]))
    psi = qt.sesolve(ham, psi0, [0.0, t_ghz], options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 10**8}).states[
        -1
    ]
    arr = np.asarray(psi.full()).reshape(2**n, d)
    assert float(1.0 - np.sum(np.abs(arr[:, 0]) ** 2)) < 0.03, "the motional loop closes to 3 %"
    spin = arr[:, 0] / np.linalg.norm(arr[:, 0])
    pops = np.abs(spin) ** 2
    assert pops[0] == pytest.approx(0.5, abs=0.05) and pops[-1] == pytest.approx(0.5, abs=0.05)
    assert float(np.sum(pops[1:-1])) < 0.05, "the population stays in |0000> and |1111>"
    best = max(
        abs((spin[0] + cmath.exp(-1j * phi) * spin[-1]) / math.sqrt(2.0)) ** 2
        for phi in np.linspace(0.0, TWO_PI, 721)
    )
    assert best > 0.9, f"a GHZ state up to local Z phases: best overlap {best:.4f}"


# ---- Section 13 / 9.16 rows 13-8, 13-1, "Spin operator in MS formulas" ----------------------------------------------------------


def test_spin_operator_identity_and_the_three_normalizations() -> None:
    """Section 13 'Spin operator in MS formulas' and 9.16 row 13-8: ||S_x^2 - 2(1 + sigma_x sigma_x)|| = 0 with spectrum
    {-2, 0, 0, 2} (an operator identity, tolerance 1e-12), and the ONE closure eta Omega/eps = 1/(2 sqrt K) carries
    A = -pi/2 on J_y^2, -pi/8 on S_y^2 and chi = pi/4 on sigma_y sigma_y."""
    s, _p0, _pp, _pm = spin_projectors(0.0)
    one = qt.tensor(qt.qeye(2), qt.qeye(2))
    sxsx = qt.tensor(qt.sigmax(), qt.sigmax())
    assert (s * s - 2.0 * (one + sxsx)).norm() < 1e-12
    assert np.sort(np.real(s.eigenenergies())) == pytest.approx([-2.0, 0.0, 0.0, 2.0], abs=1e-12)
    # the three normalizations of the same phase: S = 2 J, S^2 = 2 + 2 sigma sigma
    chi = math.pi / 4.0
    on_s2 = chi / 2.0  # exp(i chi sigma sigma) = exp(i (chi/2) S^2) up to a global phase
    on_j2 = 4.0 * on_s2  # J = S/2, so J^2 = S^2/4
    assert on_s2 == pytest.approx(math.pi / 8.0, rel=1e-12)
    assert on_j2 == pytest.approx(math.pi / 2.0, rel=1e-12)
    # Roos's Omega_c = |eps|/(4 eta) for a Rabi frequency WITHOUT the 1/2 is |eps|/(2 eta) in the plan's Omega
    eta, eps = 0.05, TWO_PI * 10e3
    omega_plan = ms_closure_ratio(1) * eps / eta
    assert omega_plan == pytest.approx(eps / (2.0 * eta), rel=1e-12)
    assert 0.5 * omega_plan == pytest.approx(eps / (4.0 * eta), rel=1e-12), "Roos's own convention"
    assert ms_two_body_angle(eta, eta, omega_plan, eps, 1) == pytest.approx(chi, rel=1e-12)


@pytest.mark.slow
def test_pure_force_model_gives_theta_pi_over_eight_and_a_maximally_entangled_pair() -> None:
    """Section 9.16 row 13-8: integrating H/hbar = F S_x(a e^{-i eps t} + h.c.) with F = eta Omega/2 and
    r = eta Omega/|eps| = 1/(2 sqrt K) for K = 1, 2, 3 gives exp(i Theta S_x^2) with Theta = 0.392699 = pi/8, chi = pi/4,
    concurrence 1.000000000, P_gg = P_ee = 0.5 and leakage <= 2e-16; r = 1/(4 sqrt K) does not close maximally."""
    eta, eps, d = 0.05, TWO_PI * 10e3, 30
    for loops in (1, 2, 3):
        for ratio, target in (
            (ms_closure_ratio(loops), math.pi / 8.0),
            (0.5 * ms_closure_ratio(loops), None),
        ):
            omega = ratio * eps / eta
            force = eta * omega / 2.0
            tau = TWO_PI * loops / eps
            s_x = qt.tensor(qt.sigmax(), qt.qeye(2), qt.qeye(d)) + qt.tensor(
                qt.qeye(2), qt.sigmax(), qt.qeye(d)
            )
            op = force * s_x * qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(d))

            def plain(t: float, _eps: float = eps, **_: object) -> complex:
                return complex(cmath.exp(-1j * _eps * t))

            def conj(t: float, _eps: float = eps, **_: object) -> complex:
                return complex(cmath.exp(1j * _eps * t))

            ham = qt.QobjEvo([[op, plain], [op.dag(), conj]])
            cols = []
            leakage = 0.0
            for basis in range(4):
                spin = qt.basis(4, basis)
                spin.dims = [[2, 2], [1, 1]]
                psi = qt.sesolve(
                    ham,
                    qt.tensor(spin, qt.basis(d, 0)),
                    [0.0, tau],
                    options={"atol": 1e-13, "rtol": 1e-11, "nsteps": 10**7, "method": "dop853"},
                ).states[-1]
                arr = np.asarray(psi.full()).reshape(4, d)
                leakage = max(leakage, float(abs(1.0 - np.sum(np.abs(arr[:, 0]) ** 2))))
                cols.append(arr[:, 0])
            unitary = np.array(cols).T
            phases = sorted(cmath.phase(x) for x in np.linalg.eigvals(unitary))
            theta = max(abs(p) for p in phases) / 4.0
            if target is None:
                assert theta != pytest.approx(math.pi / 8.0, rel=0.05), (
                    f"K = {loops}: r = 1/(4 sqrt K) must NOT close maximally"
                )
                continue
            assert leakage < 2e-15, (loops, leakage)
            assert theta == pytest.approx(target, rel=1e-9), (loops, theta)
            # the maximally entangled pair: |dd> -> (|dd> - i|uu>)/sqrt2
            psi = unitary @ np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
            rho = qt.Qobj(np.outer(psi, psi.conj()), dims=[[2, 2], [2, 2]])
            # the row prints concurrence 1.000000000; reconstructing the 4 x 4 unitary from four independent
            # dop853 runs and projecting onto |0> costs about 5e-7 of it, so 2e-6 is the achievable pin here while
            # Theta itself, the physics, is pinned at 1e-9
            assert qt.concurrence(rho) == pytest.approx(1.0, abs=2e-6)
            pops = np.real(np.diag(rho.full()))
            assert pops[0] == pytest.approx(0.5, abs=1e-6) and pops[3] == pytest.approx(0.5, abs=1e-6)


@pytest.mark.slow
def test_withdrawn_doubled_hamiltonian_is_the_negative_control() -> None:
    """Section 9.16 row 4.4-4: the plan's WITHDRAWN displayed force -(hbar eta Omega) S_y (a^dag e^{i eps t} + h.c.), twice
    the derived one, lands maximally entangling at eta Omega/eps = 1/4 (populations (0.5, 0, 0, 0.5), concurrence 1.0) and
    on a product state at 1/2 - the factor-2 error the derivation audit of 2026-09-04 caught. The correct
    -(hbar eta Omega/2) S_y form does the opposite (``check_ms_closure.py``)."""
    # the row's own settings: Fock 60 to 90 and atol 1e-13, which is what gets the concurrence to 1e-10
    eta, eps, d = 0.05, TWO_PI * 10e3, 60
    tau = TWO_PI / eps

    def integrate(ratio: float, factor: float) -> tuple[float, np.ndarray]:
        omega = ratio * eps / eta
        a = qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(d))
        s_y = qt.tensor(qt.sigmay(), qt.qeye(2), qt.qeye(d)) + qt.tensor(qt.qeye(2), qt.sigmay(), qt.qeye(d))
        op = -(eta * omega * factor) * s_y * a.dag()

        def plain(t: float, **_: object) -> complex:
            return complex(cmath.exp(1j * eps * t))

        def conj(t: float, **_: object) -> complex:
            return complex(cmath.exp(-1j * eps * t))

        ham = qt.QobjEvo([[op, plain], [op.dag(), conj]])
        psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1), qt.basis(d, 0))
        psi = qt.sesolve(
            ham, psi0, [0.0, tau], options={"atol": 1e-13, "rtol": 1e-11, "nsteps": 10**7, "method": "dop853"}
        ).states[-1]
        rho = psi.ptrace([0, 1])
        return float(qt.concurrence(rho)), np.real(np.diag(rho.full()))

    conc_quarter, pops_quarter = integrate(0.25, 1.0)
    conc_half, pops_half = integrate(0.5, 1.0)
    assert conc_quarter == pytest.approx(1.0, abs=1e-6)
    assert pops_quarter == pytest.approx([0.5, 0.0, 0.0, 0.5], abs=1e-4)
    assert conc_half < 1e-8, "the doubled force over-rotates to a product state at the correct closure"
    assert pops_half == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-4)
    # and the derived (halved) force does the opposite at the same two ratios
    assert integrate(0.5, 0.5)[0] == pytest.approx(1.0, abs=1e-6)
    assert integrate(0.25, 0.5)[0] < 0.4


def test_alpha_prefactor_negative_controls() -> None:
    """Section 9.16 row 13-1: the sideband coupling is eta Omega/2, so alpha = (eta Omega/(2 eps))(e^{i eps t} - 1) and the
    row's two withdrawn readings - eta Omega/(1 x eps) and eta Omega/(4 x eps) - miss it by exactly 2x and 1/2x, hence the
    two-body angle chi ~ |alpha|^2 by 4x and 1/4x. The exact integration that picks the 1/2 is
    ``test_withdrawn_doubled_hamiltonian_is_the_negative_control`` and ``check_ms_closure.py``'s negative control."""
    from qutip_trap.validation.two_qubit_closed_forms import ms_alpha

    eta, eps = 0.05, TWO_PI * 10e3
    omega = ms_closure_ratio(1) * eps / eta
    t_half = math.pi / eps
    derived = ms_alpha(eta, omega, eps, t_half)
    assert abs(derived) == pytest.approx(eta * omega / eps, rel=1e-12), (
        "|alpha| at eps t = pi is 2 x eta Omega/(2 eps)"
    )
    for denominator, ratio in ((1.0, 2.0), (4.0, 0.5)):
        withdrawn = eta * omega / (denominator * eps) * (cmath.exp(1j * eps * t_half) - 1.0)
        assert abs(withdrawn) / abs(derived) == pytest.approx(ratio, rel=1e-12), denominator
        assert abs(withdrawn) ** 2 / abs(derived) ** 2 == pytest.approx(ratio**2, rel=1e-12)
    assert ms_two_body_angle(eta, eta, 2.0 * omega, eps, 1) == pytest.approx(
        4.0 * ms_two_body_angle(eta, eta, omega, eps, 1), rel=1e-12
    ), "chi scales as the square of the force prefactor"


# ---- Section 9.4 / 9.12 consistency anchors --------------------------------------------------------------------------------------


def test_ballance_carrier_factor() -> None:
    """Section 9.12 'Ballance carrier factor': beta = |Delta k| x 38 nm = 0.8505 for a 397 nm counter-propagating pair,
    J_0(beta) = 0.8272 (the paper quotes 0.83, which is exact at 37.7 nm) and J_1^2/J_0^2 = 0.220, the 22 %
    off-resonant micromotion channel Section 4.4.4 flags as [corrected]."""
    delta_k = TWO_PI * math.sqrt(2.0) / 397e-9
    beta = delta_k * 38e-9
    assert beta == pytest.approx(0.8505, abs=5e-5)
    assert float(jv(0, beta)) == pytest.approx(0.8272, abs=5e-5)
    assert float(jv(1, beta)) ** 2 / float(jv(0, beta)) ** 2 == pytest.approx(0.220, abs=5e-4)
    assert float(jv(0, delta_k * 37.7e-9)) == pytest.approx(0.83, abs=5e-3), "the paper's rounded 0.83"


def test_cross_apparatus_heating_budget_and_the_cooling_floor() -> None:
    """Section 9.12 'Cross-apparatus heating budget' and 'Cooling stage sets the gate floor'. The heating pair is
    algebraically identical at K = 1 and gives 8.1e-3 at Gamma_h = 65 /s, tau = 250 us; the eps_nbar floor at eta = 0.123
    is 1.1e-5 / 2.8e-5 / 3.9e-5 / 0.238 as the row prints them, which the formula reproduces to 12 % (a consistency
    anchor with the plan's 20 % default: the printed digits carry Ballance's own rounding of eta)."""
    assert kirchmair_heating_error(65.0, 250e-6) == pytest.approx(8.1e-3, rel=0.02)
    assert ballance_heating_error(65.0, 250e-6, 1) == pytest.approx(
        kirchmair_heating_error(65.0, 250e-6), rel=1e-12
    )
    printed = {0.02: 1.1e-5, 0.05: 2.8e-5, 0.06: 3.9e-5, 15.0: 0.238}
    for nbar, value in printed.items():
        assert ballance_thermal_error(0.123, nbar) == pytest.approx(value, rel=0.2), nbar


def test_published_two_qubit_records() -> None:
    """Section 9.4 'Records': the three tracked laboratory numbers, never reproduced from first principles."""
    assert set(PUBLISHED_RECORDS) == {"fastest_gate_2018", "lowest_error_2025", "helios_2025"}
    assert PUBLISHED_RECORDS["fastest_gate_2018"][0] == pytest.approx(1.6e-6, rel=1e-12)
    assert PUBLISHED_RECORDS["lowest_error_2025"][0] == pytest.approx(8.4e-5, rel=1e-12)
    assert PUBLISHED_RECORDS["helios_2025"][0] == pytest.approx(7.9e-4, rel=1e-12)
    assert all(
        0.0 < value < 1.0 or key == "fastest_gate_2018" for key, (value, _) in PUBLISHED_RECORDS.items()
    )
    assert all(text for _value, text in PUBLISHED_RECORDS.values())


# ---- Section 4.4.7 derived forms against direct integration ----------------------------------------------------------------------


def test_zhu_beta_bar_identity_and_leung_zero_temperature_limit() -> None:
    """Section 9.4 'Residual displacement': Zhu's beta_k = coth[sqrt(mu_k/4) ln(1 + 1/nbar_1)] IS 2 nbar_k + 1 at
    nbar_k = 1/((1 + 1/nbar_1)^sqrt(mu_k) - 1) (an identity, 1e-12), and Leung's sum |alpha|^2 is eps_ent at nbar = 0."""
    for mu_k in (1.0, 2.0, 4.0, 9.0):
        for nbar_1 in (0.1, 0.5, 3.0):
            nbar_k = 1.0 / ((1.0 + 1.0 / nbar_1) ** math.sqrt(mu_k) - 1.0)
            assert zhu_beta_bar(mu_k, nbar_1) == pytest.approx(2.0 * nbar_k + 1.0, rel=1e-12)
    alphas = [0.03 + 0.04j, -0.02, 0.01j]
    assert leung_zero_temperature_error(alphas) == pytest.approx(
        entanglement_infidelity_from_displacements(alphas, [0.0] * 3), rel=1e-12
    )
    assert leung_zero_temperature_error(alphas) == pytest.approx(sum(abs(a) ** 2 for a in alphas), rel=1e-12)


def test_recalibration_rule_and_the_derived_debye_waller_error() -> None:
    """Section 4.4.7: t_g(nbar) = t_g(0)(1 + eta^2(2 nbar + 1)) (no 1/N), and the derived Debye-Waller error
    (pi^2/8) N(N-1)(eta^4/N^2) Var(n) with Var(n) = nbar^2 + nbar, which is the Sorensen-Molmer fidelity's own bracket.

    The recalibration rule is checked against the Debye-Waller law chi(n) = chi_0[1 - eta^2(2n+1)] it comes from: the gate
    time must grow by exactly the factor the thermally averaged angle shrank by."""
    from qutip_trap.validation.two_qubit_closed_forms import (
        chi_of_n,
        sorensen_molmer_thermal_fidelity,
        thermal_average,
    )

    eta, nbar, t_g0 = 0.1, 1.0, 100e-6
    assert recalibrated_gate_time_s(t_g0, eta, nbar) == pytest.approx(
        t_g0 * (1.0 + eta**2 * (2.0 * nbar + 1.0)), rel=1e-12
    )
    mean_chi = thermal_average([chi_of_n(1.0, eta, n) for n in range(400)], nbar)
    assert mean_chi == pytest.approx(1.0 - eta**2 * (2.0 * nbar + 1.0), rel=1e-9), (
        "the thermal average of chi(n) is chi_0[1 - eta^2(2 nbar + 1)]"
    )
    assert recalibrated_gate_time_s(t_g0, eta, nbar) == pytest.approx(t_g0 * (2.0 - mean_chi), rel=1e-9)
    var_n = nbar * (nbar + 1.0)
    # the derived Debye-Waller error carries the explicit eta^4/N^2 of Section 4.4.7 (the per-ion eta of an N-ion COM
    # mode), so it is 1/N^2 of the Sorensen-Molmer fidelity's own bracket, which is written in the single-ion eta
    for n_ions in (2, 3, 5, 17):
        assert debye_waller_error_derived(n_ions, eta, nbar) == pytest.approx(
            (1.0 - sorensen_molmer_thermal_fidelity(n_ions, eta, var_n)) / n_ions**2, rel=1e-12
        ), n_ions
        assert debye_waller_error_derived(n_ions, eta, nbar) == pytest.approx(
            math.pi**2 / 8.0 * (n_ions - 1) / n_ions * eta**4 * var_n, rel=1e-12
        ), n_ions
    assert debye_waller_error_derived(2, eta, nbar) == pytest.approx(
        math.pi**2 / 16.0 * eta**4 * var_n, rel=1e-12
    )


def test_thermal_weight_of_eps_ent_by_direct_integration() -> None:
    """Section 9.4 'Residual displacement': eps_ent = sum_{j,m} |alpha_{j,m}|^2 (2 nbar_m + 1) carries a THERMAL weight the
    first pass only ever validated at nbar = 0. The exact state infidelity of a uniform-input state under the same
    residual displacements is the multi-branch Gaussian overlap
    1 - (1/16) sum_{s,s'} exp(-|beta_s - beta_s'|^2 (nbar + 1/2)), whose first-order expansion IS eps_ent: measured
    ratios 0.9923 (nbar = 0), 0.9773 (1) and 0.9627 (2) at |alpha| ~ 0.06, the deficit being the fourth-order term.

    The overlap formula itself is checked against an independent QuTiP evaluation with explicit displacement operators on
    a thermal density matrix (agreement 1e-6 at d = 60), so the weight is pinned by integration and not by citation."""
    alphas = [0.06 + 0.02j, -0.05]
    ratios = []
    for nbar, expected in ((0.0, 0.992323), (1.0, 0.977287), (2.0, 0.962661)):
        eps_ent = entanglement_infidelity_from_displacements(alphas, [nbar, nbar])
        assert eps_ent == pytest.approx((2.0 * nbar + 1.0) * sum(abs(a) ** 2 for a in alphas), rel=1e-12)
        exact = state_infidelity_uniform_input(alphas, nbar)
        assert exact / eps_ent == pytest.approx(expected, rel=1e-4)
        ratios.append(exact / eps_ent)
    assert ratios == sorted(ratios, reverse=True), "the fourth-order deficit grows with nbar"
    # the overlap formula against explicit displacement operators on a thermal mode
    d = 60
    for nbar in (0.0, 1.0):
        rho = qt.thermal_dm(d, nbar) if nbar > 0.0 else qt.fock_dm(d, 0)
        total = 0.0
        signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for s in signs:
            beta_s = alphas[0] * s[0] + alphas[1] * s[1]
            for s_p in signs:
                beta_p = alphas[0] * s_p[0] + alphas[1] * s_p[1]
                op = qt.displace(d, beta_s).dag() * qt.displace(d, beta_p)
                total += float(np.real((rho * op).tr()))
        assert 1.0 - total / 16.0 == pytest.approx(state_infidelity_uniform_input(alphas, nbar), abs=2e-6)


def test_roos_bessel_saturation_of_the_force() -> None:
    """Section 9.4 'Beyond leading order': Roos Eq. 17's carrier saturation of the spin-dependent force, J_0(x) + J_2(x) at
    x = 4 Omega/delta, which is exactly 2 J_1(x)/x. At the two-ion fixture's Omega/mu ~ 0.03 the force is reduced by
    1.80e-3, the same order as the off-resonant carrier scale (Omega/nu)^2 the intrinsic budget of Section 4.4.7 already
    reports, so it belongs in that ledger of scales (``run/job.py``, M6's file: this test supplies the term)."""
    from qutip_trap.validation.two_qubit_closed_forms import roos_force_saturation

    for ratio, value in ((0.03, 0.998201080), (0.1, 0.980132890), (0.2, 0.922105115)):
        assert roos_force_saturation(ratio, 1.0) == pytest.approx(value, rel=1e-9)
        x = 4.0 * ratio
        assert roos_force_saturation(ratio, 1.0) == pytest.approx(2.0 * float(jv(1, x)) / x, rel=1e-12), (
            "J_0(x) + J_2(x) = 2 J_1(x)/x"
        )
    assert 1.0 - roos_force_saturation(0.03, 1.0) == pytest.approx(1.80e-3, rel=0.01)
    assert roos_force_saturation(0.0, 1.0) == pytest.approx(1.0, rel=1e-12), "no saturation at zero drive"
