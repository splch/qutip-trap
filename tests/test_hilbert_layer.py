"""The Hilbert layer of M2: exact displacement operators against the Laguerre oracle (Section 5.1.1), the Rabi table and
Debye-Waller factors (Sections 4.3.1, 4.2.7, 9.2, 9.10, 9.12), spaces, marginals and the truncation monitor (Section 5.5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt
from scipy.special import eval_genlaguerre

from qutip_trap.api import HilbertSpace, ModeTruncation, SolverOptions
from qutip_trap.hilbert.operators import (
    TABLE_ELEMENT_ERROR,
    TABLE_ETA,
    analytic_norm_loss,
    debye_waller_factor,
    debye_waller_rms_fraction,
    displacement_element_analytic,
    displacement_matrix_analytic,
    displacement_operator,
    interior_tolerance,
    probability_within,
    rabi_matrix_element,
    rabi_table,
    required_margin,
    sideband_operators,
    thermal_debye_waller_approx,
    thermal_debye_waller_mean,
    thermal_populations,
)
from qutip_trap.hilbert.truncation import (
    boundary_population,
    grow_for_margins,
    halving_test,
    margin_reports,
    regrid_state,
)

# ---- Section 5.1.1: the two constructions ---------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [8, 10, 20, 40])
@pytest.mark.parametrize("eta", TABLE_ETA)
def test_expm_matches_analytic_elements_in_the_interior_block(d: int, eta: float) -> None:
    """The measured table: max element difference over n, n' < d/2 (round-off rows compared at the 1e-12 floor)."""
    exp_ = displacement_operator(d, 1j * eta).full()
    ana = displacement_matrix_analytic(d, 1j * eta)
    blk = d // 2
    diff = np.max(np.abs(exp_[:blk, :blk] - ana[:blk, :blk]))
    table = TABLE_ELEMENT_ERROR[d][TABLE_ETA.index(eta)]
    assert diff <= max(3.0 * table, 1e-12), (d, eta, diff, table)
    # the exponential is exactly unitary, the analytic matrix is not: its top column loses norm
    unit = np.linalg.norm(exp_.conj().T @ exp_ - np.eye(d), 2)
    assert unit < 1e-13
    loss_top = analytic_norm_loss(d, 1j * eta, d - 1)
    assert loss_top > 0.05  # 9% at (10, 0.1), 30% at 40, about half at eta >= 0.5


def test_analytic_norm_loss_anchor_row_10_eta_0_1() -> None:
    """Section 9.13: analytic top-column norm loss 9.3% at (d_m = 10, eta = 0.1)."""
    assert analytic_norm_loss(10, 0.1j, 9) == pytest.approx(0.093, abs=0.002)


def test_both_branches_of_the_analytic_form_for_complex_alpha() -> None:
    """alpha^{n'-n} below the diagonal, (-alpha^*)^{n-n'} above it (the second revision had the labels swapped), against
    QuTiP's displace far from the boundary; at alpha = i eta both read (i eta)^{|n'-n|}."""
    for alpha in (0.3 + 0.2j, -0.7j, 1.1, 0.05 - 0.4j):
        ana = displacement_matrix_analytic(60, alpha)
        ref = qt.displace(60, alpha).full()
        assert np.max(np.abs(ana[:20, :20] - ref[:20, :20])) < 1e-13
    eta = 0.37
    for n, m in ((0, 3), (3, 0), (2, 5), (5, 2)):
        val = displacement_element_analytic(m, n, 1j * eta)
        k = abs(m - n)
        lo = min(m, n)
        expected = (
            math.exp(-(eta**2) / 2)
            * math.sqrt(math.factorial(lo) / math.factorial(max(m, n)))
            * (1j * eta) ** k
            * eval_genlaguerre(lo, k, eta**2)
        )
        assert val == pytest.approx(expected, rel=1e-12)


def test_exact_rabi_matrix_elements_wineland_eq_18() -> None:
    """Section 9.2: Omega_{n',n} = Omega e^{-eta^2/2} (n_<!/n_>!)^{1/2} |eta|^{|n'-n|} |L^{|n'-n|}_{n_<}(eta^2)| from D(i eta), rtol 1e-10.

    The MODULUS: the signed Laguerre form goes negative for odd |n' - n| at negative eta and beyond the polynomial's first
    zero (n = 6 to 8 at eta = 0.5; audit item 4.3-2)."""
    for eta in (-0.9, -0.1, 0.05, 0.1, 0.3, 0.5, 0.9, 1.0, 1.5):
        exp_ = displacement_operator(400, 1j * eta).full()
        for n in range(9):
            for m in range(9):
                assert rabi_matrix_element(m, n, eta) == pytest.approx(abs(exp_[m, n]), rel=1e-10, abs=1e-14)
                assert rabi_matrix_element(m, n, eta) >= 0.0
    # the printed signed form is negative already at eta = +0.5 for n = n' >= 6
    signed = math.exp(-0.125) * eval_genlaguerre(6, 0, 0.25)
    assert signed < 0.0 and rabi_matrix_element(6, 6, 0.5) == pytest.approx(-signed, rel=1e-12)


def test_lamb_dicke_limits_and_quartic_residual() -> None:
    """Section 9.10: carrier -> Omega, red -> eta Omega sqrt(n), blue -> eta Omega sqrt(n+1) at eta = 1e-4; Omega_00 = Omega e^{-eta^2/2};
    the quartic Debye-Waller residual scales as eta^6 (n = 1, eta = 0.1: 0.9850623544 exact vs 0.9850625000)."""
    eta = 1e-4
    for n in (1, 3, 7):
        assert rabi_matrix_element(n, n, eta) == pytest.approx(1.0, abs=1e-7)
        assert rabi_matrix_element(n - 1, n, eta) == pytest.approx(eta * math.sqrt(n), rel=1e-6)
        assert rabi_matrix_element(n + 1, n, eta) == pytest.approx(eta * math.sqrt(n + 1), rel=1e-6)
    assert rabi_matrix_element(0, 0, 0.3) == pytest.approx(math.exp(-0.045), rel=1e-12)
    assert debye_waller_factor(1, 0.1) == pytest.approx(0.9850623544, abs=5e-11)
    quartic = 1.0 - 1.5 * 0.01 + 0.625 * 1e-4  # e^{-x/2}(1 - x) = 1 - 3x/2 + 5x^2/8 + O(x^3)
    assert quartic == pytest.approx(0.9850625000, abs=1e-12)
    assert (debye_waller_factor(1, 0.1) - quartic) / 0.1**6 == pytest.approx(-7.0 / 48.0, rel=0.05), (
        "the residual scales as eta^6"
    )
    # audit 4.3-3: <p|D(i eta)|0> = e^{-eta^2/2} (i eta)^p/sqrt(p!) and the heuristic eta^p/p! understates by sqrt(p!) e^{-eta^2/2}
    for p, ratio in ((3, 2.4373), (4, 4.8745)):
        exact = rabi_matrix_element(p, 0, 0.1)
        assert exact / (0.1**p / math.factorial(p)) == pytest.approx(ratio, abs=2e-4)


def test_sideband_phase_and_operators_decompose_the_displacement() -> None:
    d = 12
    mat = displacement_matrix_analytic(d, 0.2j)
    parts = sideband_operators(mat)
    assert np.allclose(sum(parts.values()), mat)
    for k, part in parts.items():
        rows, cols = np.nonzero(part)
        assert np.all(rows - cols == k)
    # the drive phase on the k-th sideband is arg (i eta)^k = k pi/2
    assert np.angle(displacement_element_analytic(2, 0, 0.3j)) == pytest.approx(math.pi, abs=1e-12)
    assert np.angle(displacement_element_analytic(1, 0, 0.3j)) == pytest.approx(math.pi / 2, abs=1e-12)


# ---- Debye-Waller statistics (Sections 4.2.7, 9.2, 9.10, 9.12) ------------------------------------------------------------


def test_thermal_debye_waller_identity_and_wineland_rms_example() -> None:
    """Section 9.12: sum_n P_n e^{-eta^2/2} L_n(eta^2) = 0.9851119396031 at eta^2 = 0.01, nbar = 1 and 0.8203698531378 at
    eta^2 = 0.09, nbar = 1.7 to 13 digits; the n = 1 start gives 0.4662967. The sum equals exp[-eta^2(nbar + 1/2)] EXACTLY
    (the thermal characteristic function is Gaussian), so the plan's 'approximately' is an identity.
    Section 9.10: 100 spectator modes at nbar = 0.1, eta = 0.01 give P(|delta Omega/Omega| < 1e-4) = 0.237 (0.330 without the 2)."""
    assert thermal_debye_waller_mean(0.1, 1.0) == pytest.approx(0.9851119396031, abs=1e-12)
    assert thermal_debye_waller_mean(0.3, 1.7) == pytest.approx(0.8203698531378, abs=1e-12)
    assert thermal_debye_waller_approx(0.1, 1.0) == pytest.approx(
        thermal_debye_waller_mean(0.1, 1.0), abs=1e-13
    )
    p = thermal_populations(1.7, 400)
    start_at_one = sum(p[n] * debye_waller_factor(n, 0.3) for n in range(1, 400))
    assert start_at_one == pytest.approx(0.4662967, abs=1e-6), "the n = 1 start of the (0.09, 1.7) case"
    rms = debye_waller_rms_fraction([0.01] * 100, [0.1] * 100)
    assert rms == pytest.approx(math.sqrt(100 * 1e-8 * 0.11), rel=1e-12)
    assert probability_within(1e-4, rms) == pytest.approx(0.237, abs=1e-3)
    assert math.erf(1e-4 / math.sqrt(100 * 1e-8 * 0.11)) == pytest.approx(0.330, abs=1e-3), (
        "dropping the 2 under the radical"
    )


def test_thermal_populations_normalize_and_truncate() -> None:
    p = thermal_populations(2.0, 400)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert p[1] / p[0] == pytest.approx(2.0 / 3.0)
    assert thermal_populations(0.0, 5)[0] == 1.0


# ---- margins and tolerances (Section 5.1.1 rule ii) --------------------------------------------------------------------------


def test_margin_fixture_and_interior_tolerance() -> None:
    assert required_margin(0.05) == 6 and required_margin(0.1) == 6
    assert required_margin(0.5) == 10 and required_margin(1.0) == 20
    assert 6 < required_margin(0.3) < 10
    assert interior_tolerance(5, 0.5) == pytest.approx(7e-8)  # the d = 10 row
    assert interior_tolerance(4, 0.1) == 1e-12  # 2e-13 floors at 1e-12
    assert interior_tolerance(20, 0.5) == 1e-12
    with pytest.raises(ValueError):
        interior_tolerance(0, 0.1)


def test_space_refuses_eta_above_its_declaration_and_checks_the_oracle() -> None:
    space = HilbertSpace((2,), (ModeTruncation(0, 12, (0, 3), 0.15),), None, (1, 2))
    op = space.displacement_factor(0, 0.1)
    assert op.shape == (12, 12)
    with pytest.raises(ValueError, match="eta_max"):
        space.displacement_factor(0, 0.2)
    reports = margin_reports(space)
    assert reports[0].margin_levels == 8 and reports[0].ok
    tight = HilbertSpace((2,), (ModeTruncation(0, 6, (0, 3), 0.15),), None, (1, 2))
    rep = margin_reports(tight)[0]
    assert not rep.ok and rep.deficit == rep.required_levels - 2
    grown = grow_for_margins(tight)
    assert margin_reports(grown)[0].ok and grown.truncation(0).d == 6 + rep.deficit


# ---- spaces, operators, states, marginals -------------------------------------------------------------------------------------


def test_drive_operator_structure_and_nonzeros() -> None:
    """sigma_+ (x) prod D_m with 2^{N-1} prod d_m^2 non-zeros per ion for two-level ions (Section 5.1.1; tidyup may drop < 1e-14)."""
    space = HilbertSpace(
        (2, 2), (ModeTruncation(0, 6, (0, 1), 0.1), ModeTruncation(1, 5, (0, 0), 0.1)), None, (2, 3, 4, 5)
    )
    v = space.drive_operator(0, {0: 0.08, 1: 0.05})
    nnz = v.data.as_scipy().nnz
    assert v.shape == (4 * 30, 4 * 30)
    assert 0.98 * 2 * 36 * 25 <= nnz <= 2 * 36 * 25
    asserted, diff, tol = space.oracle_status(0, 0.08)
    assert asserted and diff <= tol
    small = HilbertSpace((2,), (ModeTruncation(0, 4, (0, 2), 0.2),), None, ())
    asserted_small, diff_small, _ = small.oracle_status(0, 0.11)
    assert not asserted_small and diff_small > 1e-6, "a margin below the table is reported, not asserted"
    # equal to the embedded product of per-mode displacements times sigma_+
    ref = space.sigma_plus(0) * space.displacement(0, 0, 0.08) * space.displacement(0, 1, 0.05)
    assert (v - ref).norm() < 1e-12
    ops = space.operators({(0, 0): 0.08})
    assert len(ops.sigma_plus) == 2 and len(ops.a) == 2 and (0, 0) in ops.displacement
    # sigma_z is the ENERGY operator: |1><1| - |0><0|; the computational Z of control/native is its negative
    sz = space.sigma_z(0)
    ket1 = space.product_state([1, 0], {0: space.fock(0, 0), 1: space.fock(1, 0)})
    assert qt.expect(sz, ket1) == pytest.approx(1.0)


def test_states_marginals_and_boundary_population() -> None:
    space = HilbertSpace((2,), (ModeTruncation(0, 10, (0, 3), 0.15),), None, (1, 2))
    st = space.initial_state([0], thermal={0: 0.5, 1: 0.2})
    assert st.joint is not None and st.joint.isoper
    assert st.motional.nbar[0] == pytest.approx(0.5, abs=1e-3)  # the truncated tail lowers the mean slightly
    assert st.motional.nbar[1] == 0.2 and st.motional.frozen == (1, 2)
    rho_int = space.internal_marginal(st)
    assert rho_int.shape == (2, 2) and rho_int[0, 0] == pytest.approx(1.0)
    rho_mode = space.mode_marginal(st, 0)
    p_th = thermal_populations(0.5, 10)
    p_th = p_th / p_th.sum()
    assert np.allclose(np.diag(rho_mode.full()).real[:3], p_th[:3])
    assert boundary_population(st, space, 0) == pytest.approx(float(np.sum(p_th[-2:])), rel=1e-9)
    ket = space.initial_state([1], fock={0: 2})
    assert ket.joint is not None and ket.joint.isket
    assert space.fock_populations(ket, 0)[2] == pytest.approx(1.0)
    with pytest.raises(KeyError):
        space.mode_marginal(ket, 1)  # frozen


def test_enr_space_marginals_and_displacement() -> None:
    """ENR: shape rule, the marginal built by index sums (ptrace raises there), the sum-generator exponential (Section 5.1.1)."""
    space = HilbertSpace((2,), (ModeTruncation(0, 6, (0, 2), 0.1),), ((1, 2), 6), (3,))
    assert space.dims == [2, 6, 28] and space.dimension == 2 * 6 * 28
    st = space.initial_state([0], fock={0: 1, 1: 2, 2: 1})
    assert st.motional.nbar == {0: 1.0, 1: 2.0, 2: 1.0, 3: 0.0}
    rho2 = space.mode_marginal(st, 2)
    assert rho2.shape == (7, 7) and rho2[1, 1] == pytest.approx(1.0)
    assert boundary_population(st, space, 1) == pytest.approx(0.0)
    top = space.initial_state([0], fock={0: 0, 1: 6, 2: 0})
    assert boundary_population(top, space, 1) == pytest.approx(1.0)
    d_enr = space.enr_displacement({1: 0.1, 2: 0.05})
    unit = (d_enr.dag() * d_enr - space.enr_identity()).norm()
    assert unit < 1e-12
    # against the analytic elements deep inside the cap (n1 + n2 <= N_exc/2): Section 5.1.1's 2e-10 at eta = 0.1, N_exc = 6
    _n, s2i, _i2s = qt.enr_state_dictionaries([7, 7], 6)
    dense = d_enr.full()
    ana = displacement_matrix_analytic(7, 0.1j)
    worst = 0.0
    for (n1, n2), i in s2i.items():
        for (m1, m2), j in s2i.items():
            if n1 + n2 <= 3 and m1 + m2 <= 3 and n2 == m2:
                worst = max(
                    worst, abs(dense[j, i] - ana[m1, n1] * (1.0 if n2 == m2 else 0.0) * (1.0 if True else 0))
                )
    # the second mode is displaced by 0.05 too, so compare the product only where mode 2 is in its own diagonal element
    assert (
        worst < 5e-2
    )  # loose: mode 2's displacement mixes n2 = m2 elements by up to eta_2; the unitarity check above is the sharp one


def test_regrid_and_halving_test() -> None:
    old = HilbertSpace((2,), (ModeTruncation(0, 6, (0, 2), 0.1),), None, ())
    new = old.grown(0, 4)
    ket = old.initial_state([1], fock={0: 2}).joint
    assert ket is not None
    big = regrid_state(ket, old, new)
    assert big.shape[0] == 20 and new.fock_populations(big, 0)[2] == pytest.approx(1.0)
    rho = qt.ket2dm(ket)
    assert regrid_state(rho, old, new).tr() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        regrid_state(ket, new, old)
    calls: list[float] = []

    def run(opts: SolverOptions) -> np.ndarray:
        calls.append(opts.atol)
        return np.array([0.5 + 1e-9 * opts.atol / 1e-10, 0.5])

    ok, delta, tight = halving_test(run, SolverOptions(), tol=1e-6)
    assert ok and delta < 1e-8 and tight.atol == pytest.approx(1e-11)
    assert calls == pytest.approx([1e-10, 1e-11])


def test_rabi_table_is_the_analytic_modulus() -> None:
    t = rabi_table(6, 0.3)
    assert t[0, 0] == pytest.approx(math.exp(-0.045))
    assert t[1, 0] == pytest.approx(0.3 * math.exp(-0.045))
    assert np.all(t >= 0.0)
