"""The Hilbert layer: exact displacement operators against the Laguerre oracle (PLAN.md Section 5.1.1), the Rabi table
and Debye-Waller factors, spaces, marginals and the truncation monitor."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt
from scipy.special import eval_genlaguerre

from qutip_trap.hilbert.operators import (
    TABLE_ELEMENT_ERROR,
    TABLE_ETA,
    debye_waller_factor,
    displacement_element_analytic,
    displacement_leakage,
    displacement_matrix_analytic,
    displacement_operator,
    interior_element_error,
    interior_tolerance,
    rabi_matrix_element,
    rabi_table,
    required_margin,
    sideband_operators,
    thermal_populations,
)
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation, enr_dimension
from qutip_trap.hilbert.truncation import boundary_population, regrid_state

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
    assert _top_column_loss(d, eta) > 0.05  # 9% at (10, 0.1), 30% at 40, about half at eta >= 0.5


def _top_column_loss(d: int, eta: float) -> float:
    return float(1.0 - np.sum(np.abs(displacement_matrix_analytic(d, 1j * eta)[:, d - 1]) ** 2))


def test_analytic_norm_loss_anchor_row_10_eta_0_1() -> None:
    """The analytic top-column norm loss is 9.3% at (d_m = 10, eta = 0.1)."""
    assert _top_column_loss(10, 0.1) == pytest.approx(0.093, abs=0.002)


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
    """Omega_{n',n} = Omega e^{-eta^2/2} (n_<!/n_>!)^{1/2} |eta|^{|n'-n|} |L^{|n'-n|}_{n_<}(eta^2)| from D(i eta), rtol 1e-10.

    The modulus: the signed Laguerre form goes negative for odd |n' - n| at negative eta and beyond the polynomial's first
    zero (n = 6 to 8 at eta = 0.5)."""
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
    """Carrier -> Omega, red -> eta Omega sqrt(n), blue -> eta Omega sqrt(n+1) at eta = 1e-4; Omega_00 = Omega e^{-eta^2/2};
    the quartic Debye-Waller residual scales as eta^6 (n = 1, eta = 0.1: 0.9850623544 exact vs 0.9850625000)."""
    eta = 1e-4
    for n in (1, 3, 7):
        assert rabi_matrix_element(n, n, eta) == pytest.approx(1.0, abs=1e-7)
        assert rabi_matrix_element(n - 1, n, eta) == pytest.approx(eta * math.sqrt(n), rel=1e-6)
        assert rabi_matrix_element(n + 1, n, eta) == pytest.approx(eta * math.sqrt(n + 1), rel=1e-6)
    assert rabi_matrix_element(0, 0, 0.3) == pytest.approx(math.exp(-0.045), rel=1e-12)
    assert debye_waller_factor(1, 0.1) == pytest.approx(0.9850623544, abs=5e-11)
    quartic = 1.0 - 1.5 * 0.01 + 0.625 * 1e-4  # e^{-x/2}(1 - x) = 1 - 3x/2 + 5x^2/8 + O(x^3)
    assert (debye_waller_factor(1, 0.1) - quartic) / 0.1**6 == pytest.approx(-7.0 / 48.0, rel=0.05), (
        "the residual scales as eta^6"
    )
    # <p|D(i eta)|0> = e^{-eta^2/2} (i eta)^p/sqrt(p!): the heuristic eta^p/p! understates it by sqrt(p!) e^{-eta^2/2}
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


def _thermal_mean_debye_waller(eta: float, nbar: float, start: int = 0) -> float:
    n_max = int(50 + 40 * nbar)
    p = thermal_populations(nbar, n_max + 1)
    return float(sum(p[n] * debye_waller_factor(n, eta) for n in range(start, n_max + 1)))


def test_thermal_debye_waller_identity() -> None:
    """sum_n P_n e^{-eta^2/2} L_n(eta^2) = 0.9851119396031 at eta^2 = 0.01, nbar = 1 and 0.8203698531378 at
    eta^2 = 0.09, nbar = 1.7 to 13 digits (the n = 1 start gives 0.4662967); the sum is exp[-eta^2(nbar + 1/2)] exactly,
    the thermal characteristic function being Gaussian."""
    assert _thermal_mean_debye_waller(0.1, 1.0) == pytest.approx(0.9851119396031, abs=1e-12)
    assert _thermal_mean_debye_waller(0.3, 1.7) == pytest.approx(0.8203698531378, abs=1e-12)
    assert _thermal_mean_debye_waller(0.1, 1.0) == pytest.approx(math.exp(-0.01 * 1.5), abs=1e-13)
    assert _thermal_mean_debye_waller(0.3, 1.7, start=1) == pytest.approx(0.4662967, abs=1e-6)


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
    assert space.truncation(0).margin_levels == 8 >= required_margin(0.15)
    tight = HilbertSpace((2,), (ModeTruncation(0, 6, (0, 3), 0.15),), None, (1, 2))
    assert tight.truncation(0).margin_levels == 2 < required_margin(0.15)
    grown = tight.grown(0, required_margin(0.15) - 2)
    assert grown.truncation(0).margin_levels == required_margin(0.15)


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
    ref = (
        space.sigma_plus(0)
        * space.embed(space.displacement_factor(0, 0.08), space.mode_factor(0))
        * space.embed(space.displacement_factor(1, 0.05), space.mode_factor(1))
    )
    assert (v - ref).norm() < 1e-12
    # sigma_z is the energy operator |1><1| - |0><0|; the computational Z of control/native is its negative
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
                worst = max(worst, abs(dense[j, i] - ana[m1, n1]))
    # loose: the second mode is displaced by 0.05 too, which mixes the n2 = m2 elements by up to eta_2; the unitarity check
    # above is the sharp one
    assert worst < 5e-2


def test_hilbert_space_dimensions() -> None:
    space = HilbertSpace(
        (2, 2), (ModeTruncation(0, 8, (0, 3), 0.1), ModeTruncation(1, 6, (0, 2), 0.1)), ((2, 3), 6), (4, 5)
    )
    assert enr_dimension(2, 6) == 28
    assert space.dims == [2, 2, 8, 6, 28]
    assert space.dimension == 4 * 48 * 28
    with pytest.raises(ValueError, match="never in two classes"):
        HilbertSpace((2,), (ModeTruncation(0, 4, (0, 1), 0.1),), None, (0,))
    with pytest.raises(ValueError):
        ModeTruncation(0, 4, (0, 5), 0.1)


def test_regrid() -> None:
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


def test_rabi_table_is_the_analytic_modulus() -> None:
    t = rabi_table(6, 0.3)
    assert t[0, 0] == pytest.approx(math.exp(-0.045))
    assert t[1, 0] == pytest.approx(0.3 * math.exp(-0.045))
    assert np.all(t >= 0.0)


# ---- the derived margin -----------------------------------------------------------------------------------------------------------


def test_derived_margin_holds_the_element_tolerance_and_the_leakage_and_never_exceeds_the_fixture() -> None:
    """``required_margin`` with a declared element tolerance and a leakage tail: the smallest margin at which the exponential's
    interior elements over n <= n_hi agree with the analytic ones to the tolerance and one displacement from n_hi leaks less than
    the tail past the cap, measured directly; smaller than the fixture at small eta, growing with n_hi (the element error at a
    fixed margin does), and never above the fixture."""
    for eta, n_hi, expected in ((0.1, 2, 3), (0.1, 5, 3), (0.05, 2, 2), (0.3, 5, 6)):
        m = required_margin(eta, tail=1e-7, n_hi=n_hi, element_tol=1e-8)
        assert m == expected, (eta, n_hi, m)
        assert interior_element_error(n_hi + 1 + m, eta, n_hi) <= 1e-8
        assert displacement_leakage(eta, n_hi, m) < 1e-7
        assert m <= required_margin(eta)
        if m > 1:
            assert (
                interior_element_error(n_hi + m, eta, n_hi) > 1e-8
                or displacement_leakage(eta, n_hi, m - 1) >= 1e-7
            )
    # the element error at a fixed margin grows with the populated range, so the derived margin does too
    assert required_margin(0.1, n_hi=20, element_tol=1e-8) > required_margin(0.1, n_hi=2, element_tol=1e-8)
    # the fixture is the ceiling: an absurdly tight tolerance falls back to it
    assert required_margin(0.1, n_hi=2, element_tol=1e-30) == required_margin(0.1) == 6
    assert required_margin(0.5, n_hi=2, element_tol=1e-30, tail=1e-30) == 10
    # the analytic leakage decays with the margin and grows with eta and n_hi
    assert displacement_leakage(0.1, 2, 3) < displacement_leakage(0.1, 2, 2) < displacement_leakage(0.1, 2, 1)
    assert displacement_leakage(0.1, 10, 3) > displacement_leakage(0.1, 2, 3)
    assert displacement_leakage(0.3, 2, 3) > displacement_leakage(0.1, 2, 3)
    assert displacement_leakage(0.1, 2, 3) == pytest.approx(6.2e-9, rel=0.1)
    with pytest.raises(ValueError):
        required_margin(0.1, n_hi=2, element_tol=-1.0)
    with pytest.raises(ValueError):
        required_margin(0.1, n_hi=2, tail=2.0)


def test_a_declared_element_tolerance_is_asserted_at_construction_and_reported() -> None:
    """A ``ModeTruncation`` that carries ``element_tol`` asserts the rule (ii) oracle to that number at any margin (the table
    asserts margins of four and more only), ``oracle_status`` reports it, and growth keeps the declaration."""
    m = required_margin(0.1, tail=1e-7, n_hi=2, element_tol=1e-8)
    derived = HilbertSpace((2,), (ModeTruncation(0, 3 + m, (0, 2), 0.15, element_tol=1e-8),), None, (1, 2))
    op = derived.displacement_factor(0, 0.1)
    assert op.shape == (3 + m, 3 + m)
    asserted, diff, tol = derived.oracle_status(0, 0.1)
    assert asserted and tol == 1e-8 and diff <= tol
    grown = derived.grown(0, 2)
    assert grown.truncation(0).element_tol == 1e-8 and grown.truncation(0).d == 5 + m
    # the same cap without the declaration is below the table's smallest row: reported, not asserted
    plain = HilbertSpace((2,), (ModeTruncation(0, 3 + m, (0, 2), 0.15),), None, (1, 2))
    asserted_plain, diff_plain, _tol = plain.oracle_status(0, 0.1)
    assert (not asserted_plain) and diff_plain == pytest.approx(diff)
    # a tolerance the cap cannot meet is refused at construction of the operator, not silently passed
    tight = HilbertSpace((2,), (ModeTruncation(0, 4, (0, 2), 0.3, element_tol=1e-12),), None, (1, 2))
    with pytest.raises(ValueError, match="raise the cap"):
        tight.displacement_factor(0, 0.3)
    with pytest.raises(ValueError):
        ModeTruncation(0, 6, (0, 2), 0.1, element_tol=0.0)
