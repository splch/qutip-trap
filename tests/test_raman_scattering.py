"""Raman couplings, light shifts and scattering from the level structure (PLAN.md Sections 4.5.4, 4.5.5) against the
closed forms of Ozeri 2005/2007, Wineland 2003 and Uys 2010, and against a full multi-level master equation."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
import qutip as qt

from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import TWO_PI
from qutip_trap.validation.atomic_closed_forms import (
    ozeri_gamma_raman,
    ozeri_gamma_total,
    ozeri_p_raman,
    ozeri_p_total,
    ozeri_p_total_optimum_delta,
    ozeri_raman_rabi_half,
    uys_bounds,
    uys_gamma_el,
    wineland_bracket,
    wineland_clock_light_shift,
    wineland_delta_over_omega_clock,
    wineland_p_se_clock,
    wineland_p_se_zeeman_22_11,
    wineland_ratio_function,
)
from tests.atomic_fixtures import (
    GAMMA_HZ,
    be9_like,
    field_z,
    fine_structure_omega,
    lin_perp_lin_pair,
    pi_beam_perp,
    sigma_plus_along_z,
    spin_zero_like,
    stretched_g_half,
    toy_spin_zero,
)

HALF = Fraction(1, 2)


# ---- closed forms as printed --------------------------------------------------------------------------------


def test_wineland_optimum_and_table_1_numbers() -> None:
    from scipy.optimize import minimize_scalar

    r = minimize_scalar(
        wineland_ratio_function, bounds=(0.05, 0.95), method="bounded", options={"xatol": 1e-12}
    )
    assert r.x == pytest.approx(math.sqrt(2.0) - 1.0, abs=1e-6) and r.fun == pytest.approx(
        2.0 * math.sqrt(2.0), abs=1e-6
    )
    wrong = minimize_scalar(wineland_bracket, bounds=(0.05, 0.95), method="bounded", options={"xatol": 1e-12})
    assert wrong.x == pytest.approx(1.0 / (1.0 + 2.0 ** (1.0 / 3.0)), abs=1e-6), (
        "minimizing the bracket alone is the wrong optimum"
    )
    gamma, omega_f, omega_0 = TWO_PI * 19.4e6, TWO_PI * 198e9, TWO_PI * 1.25e9
    assert wineland_p_se_clock(gamma, omega_f) == pytest.approx(8.706e-4, rel=1e-3)
    assert wineland_p_se_zeeman_22_11(gamma, omega_f) == pytest.approx(1.0053e-3, rel=1e-3)
    assert wineland_delta_over_omega_clock(omega_0, omega_f) == pytest.approx(3.571e-2, rel=1e-3)
    assert 2.0 * math.sqrt(2.0) * math.pi == pytest.approx(8.885766, abs=1e-6)
    assert 8.0 * math.pi / math.sqrt(6.0) == pytest.approx(10.260399, abs=1e-6)


def test_ozeri_scaling_laws_of_section_9_10() -> None:
    gamma, omega_f = 1.0, 1.0
    for x, ref in ((10.0, 9.8088), (100.0, 9.4568), (1000.0, 9.4279)):
        assert ozeri_p_total(gamma, omega_f, x * omega_f) == pytest.approx(ref, abs=1e-3)
    d_plus, d_minus = ozeri_p_total_optimum_delta(omega_f)
    assert ozeri_p_total(gamma, omega_f, d_plus) == pytest.approx(2.0 * math.sqrt(2.0) * math.pi, rel=1e-12)
    assert ozeri_p_total(gamma, omega_f, d_minus) == pytest.approx(2.0 * math.sqrt(2.0) * math.pi, rel=1e-12)
    assert ozeri_p_raman(gamma, omega_f, 0.5 * omega_f) == pytest.approx(8.0 * math.pi / 3.0, rel=1e-12)
    assert ozeri_p_raman(gamma, omega_f, 0.5 * omega_f) < ozeri_p_raman(gamma, omega_f, 0.4 * omega_f)


def test_uys_formula_bounds() -> None:
    """0 <= Gamma_el <= 2(Gamma_dd + Gamma_uu); amplitudes a = 1.0, b = -0.9 give Gamma_el = 3.61 against Gamma_dd + Gamma_uu = 1.81."""
    dd = np.array([[1.0]])
    uu = np.array([[-0.9]])
    g_el = uys_gamma_el(dd, uu, 1.0, 1.0)
    assert g_el == pytest.approx(3.61, abs=1e-12)
    lo, hi = uys_bounds(1.0, 0.81)
    assert lo <= g_el <= hi and hi == pytest.approx(3.62)
    assert uys_gamma_el(dd, dd, 1.0, 1.0) == 0.0, (
        "equal amplitudes: no dephasing however large the elastic rates"
    )


# ---- the explicit sums against the closed forms ------------------------------------------------------------------


@pytest.fixture(scope="module")
def be() -> AtomicStructure:
    return AtomicStructure(be9_like(), 1.0, (0.0, 0.0, 1.0))


def _clock_states(st: AtomicStructure):  # type: ignore[no-untyped-def]
    low = st.state("S1/2 F=2 mF=0")  # F = 2 lies BELOW F = 1 for 9Be+ (negative moment)
    high = st.state("S1/2 F=1 mF=0")
    assert high.energy_hz > low.energy_hz
    return low, high


def test_raman_coupling_reduces_to_ozeris_closed_form(be: AtomicStructure) -> None:
    """Omega_{g1 g2} (plan convention) = 2 x (g_b g_r/3)(b_- r_- - b_+ r_+) omega_f/[Delta(Delta - omega_f)] for lin-perp-lin beams.

    With the P hyperfine set to zero the only deviation is O(omega_0/Delta) from referencing Delta to one ground state.
    """
    sp = be.species
    omega_f = fine_structure_omega(sp)
    low, high = _clock_states(be)
    omega_q = TWO_PI * (high.energy_hz - low.energy_hz)
    for x in (0.414, -0.7, 2.5):
        delta = x * omega_f
        b, r = lin_perp_lin_pair(sp, delta, ground_energy_hz=low.energy_hz, omega_q_rad_s=omega_q)
        g_b = stretched_g_half(sp, b, field_z())
        g_r = stretched_g_half(sp, r, field_z())
        mine = abs(be.raman_coupling_rad_s(low, high, b, r))
        # lin-perp-lin on a clock line gives b_- = r_- = b_+ = -r_+ = 1/sqrt2 (PLAN.md 683), so
        # (b_- r_- - b_+ r_+) = 1/2 + 1/2 = 1; the plan's Omega is twice Ozeri's half-convention Omega_R.
        # The audit of 2026-09-07 (item E14) found this call dead: the line below it overwrote `closed`
        # with the same expression evaluated at (b_- r_- - b_+ r_+) = 1, so the (b, r) polarization
        # structure was never exercised. Both forms are now asserted, and the amplitudes are the plan's.
        h = 1.0 / math.sqrt(2.0)
        closed = 2.0 * abs(ozeri_raman_rabi_half(g_b, g_r, h, h, h, -h, delta, omega_f))
        bare = 2.0 * abs(g_b * g_r / 3.0 * omega_f / (delta * (delta - omega_f)))
        assert closed == pytest.approx(bare, rel=1e-12), "(b_- r_- - b_+ r_+) must be exactly 1 here"
        assert mine == pytest.approx(closed, rel=3e-2)
        # NEGATIVE CONTROL: same-handed pairs cancel, which is what the signed amplitudes are for
        assert ozeri_raman_rabi_half(g_b, g_r, h, h, h, h, delta, omega_f) == 0.0


def test_raman_coupling_closed_form_tightens_as_the_hyperfine_splitting_shrinks() -> None:
    """The residual is the O(omega_0/Delta) reference ambiguity: scaling A_S by 1e-3 brings the agreement to 2e-4.

    Evaluated at a negligible field: at 1 G a 1.25 MHz splitting would already be Paschen-Back mixed and the F labels would not be
    the states the closed form assumes.
    """
    sp = be9_like(a_scale=1e-3)
    st = AtomicStructure(sp, 1e-6, (0.0, 0.0, 1.0))
    omega_f = fine_structure_omega(sp)
    low, high = st.state("S1/2 F=2 mF=0"), st.state("S1/2 F=1 mF=0")
    omega_q = TWO_PI * (high.energy_hz - low.energy_hz)
    delta = 0.414 * omega_f
    b, r = lin_perp_lin_pair(sp, delta, ground_energy_hz=low.energy_hz, omega_q_rad_s=omega_q)
    g = stretched_g_half(sp, b, field_z())
    mine = abs(st.raman_coupling_rad_s(low, high, b, r))
    closed = 2.0 * abs(g * g / 3.0 * omega_f / (delta * (delta - omega_f)))
    assert mine == pytest.approx(closed, rel=2e-4)


def test_total_and_raman_scattering_rates_reduce_to_ozeris_forms(be: AtomicStructure) -> None:
    """Gamma_total = (gamma/3) g^2 [1/Delta^2 + 2/(Delta - omega_f)^2] per lin-perp-B beam; the inelastic (spin-changing) part is
    (2 gamma/9) g^2 [omega_f/(Delta(Delta - omega_f))]^2 (Ozeri Eqs. 14-15)."""
    sp = be.species
    gamma = TWO_PI * GAMMA_HZ
    omega_f = fine_structure_omega(sp)
    low, _ = _clock_states(be)
    for x in (0.414, -1.5, 3.0):
        delta = x * omega_f
        b, _r = lin_perp_lin_pair(sp, delta, ground_energy_hz=low.energy_hz)
        g = stretched_g_half(sp, b, field_z())
        rates = be.scattering_rates(low, b)
        total = sum(rates.values())
        inelastic = total - rates[low.full_label]
        assert total == pytest.approx(ozeri_gamma_total(gamma, g, 0.0, 1.0, 0.0, delta, omega_f), rel=3e-3)
        assert inelastic == pytest.approx(
            ozeri_gamma_raman(gamma, g, 0.0, 1.0, 0.0, delta, omega_f), rel=3e-2
        )
        # residual excited population times the (single) decay rate is the total rate
        assert be.residual_excited_population(low, b) * gamma == pytest.approx(total, rel=1e-3)


def test_scattering_probability_per_pi_pulse_has_ozeris_minimum(be: AtomicStructure) -> None:
    """P_total = Gamma_total x pi/|Omega_R| = (pi gamma/omega_f)(2 Delta^2 + (Delta - omega_f)^2)/|Delta(Delta - omega_f)| with its
    minimum 2 sqrt2 pi gamma/omega_f at Delta = (sqrt2 - 1) omega_f (Section 4.3.2), all power and waist dependence cancelling."""
    sp = be.species
    gamma = TWO_PI * GAMMA_HZ
    omega_f = fine_structure_omega(sp)
    low, high = _clock_states(be)
    omega_q = TWO_PI * (high.energy_hz - low.energy_hz)
    x_opt = math.sqrt(2.0) - 1.0
    values = []
    for x in (0.25, x_opt, 0.6, -2.0):
        delta = x * omega_f
        b, r = lin_perp_lin_pair(sp, delta, ground_energy_hz=low.energy_hz, omega_q_rad_s=omega_q)
        total = sum(be.scattering_rates(low, b).values()) + sum(be.scattering_rates(low, r).values())
        omega_r = abs(be.raman_coupling_rad_s(low, high, b, r))
        p = total * math.pi / omega_r
        assert p == pytest.approx(ozeri_p_total(gamma, omega_f, delta), rel=3e-2)
        values.append(p)
    assert values[1] < values[0] and values[1] < values[2]
    assert values[1] == pytest.approx(wineland_p_se_clock(gamma, omega_f), rel=3e-2)
    # doubling the power leaves P unchanged
    b2, r2 = lin_perp_lin_pair(
        sp, x_opt * omega_f, power_w=2e-3, ground_energy_hz=low.energy_hz, omega_q_rad_s=omega_q
    )
    total2 = sum(be.scattering_rates(low, b2).values()) + sum(be.scattering_rates(low, r2).values())
    assert total2 * math.pi / abs(be.raman_coupling_rad_s(low, high, b2, r2)) == pytest.approx(
        values[1], rel=1e-9
    )


def test_clock_light_shift_needs_the_three_index_detuning(be: AtomicStructure) -> None:
    """delta_{0<->0} = -(g_b^2 + g_r^2)(omega_0/3)[1/Delta^2 + 2/(Delta - omega_F)^2] (Wineland Eq. 2.17) comes out of the sum
    because each ground state carries its own detuning; the summed line strength sum_e |Omega_{eg}|^2 is the same for both clock
    states, so a single Delta per intermediate level would return exactly zero (Section 9.13)."""
    sp = be.species
    omega_f = fine_structure_omega(sp)
    low, high = _clock_states(be)
    omega_0 = TWO_PI * (high.energy_hz - low.energy_hz)
    delta = 0.414 * omega_f
    b, r = lin_perp_lin_pair(sp, delta, ground_energy_hz=low.energy_hz, omega_q_rad_s=omega_0)
    g_b, g_r = stretched_g_half(sp, b, field_z()), stretched_g_half(sp, r, field_z())
    mine = be.light_shift_rad_s(high, (b, r)) - be.light_shift_rad_s(low, (b, r))
    assert mine == pytest.approx(wineland_clock_light_shift(g_b, g_r, omega_0, delta, omega_f), rel=3e-2)
    # the negative control: identical summed line strengths -> zero differential shift at a common detuning
    for beam in (b, r):
        s_low = sum(abs(om) ** 2 for _e, om, _d in be.couplings_from(low, beam))
        s_high = sum(abs(om) ** 2 for _e, om, _d in be.couplings_from(high, beam))
        assert s_low == pytest.approx(s_high, rel=1e-12)
    # tight version at a small splitting and a negligible field (see the Raman test above)
    small = be9_like(a_scale=1e-3)
    st = AtomicStructure(small, 1e-6, (0.0, 0.0, 1.0))
    lo2, hi2 = st.state("S1/2 F=2 mF=0"), st.state("S1/2 F=1 mF=0")
    w0 = TWO_PI * (hi2.energy_hz - lo2.energy_hz)
    b2, r2 = lin_perp_lin_pair(small, delta, ground_energy_hz=lo2.energy_hz, omega_q_rad_s=w0)
    g2 = stretched_g_half(small, b2, field_z())
    mine2 = st.light_shift_rad_s(hi2, (b2, r2)) - st.light_shift_rad_s(lo2, (b2, r2))
    assert mine2 == pytest.approx(wineland_clock_light_shift(g2, g2, w0, delta, omega_f), rel=2e-3)


def test_ozeri_2005_single_electron_amplitudes() -> None:
    """Section 9.10: Raman a^(1/2) = -sqrt2/3, a^(3/2) = +sqrt2/3 (equal and opposite); Rayleigh with sigma+ on m = -1/2: 2/3 and 1/3;
    with pi on m = -1/2: 1/3 and 2/3; stretched state under sigma+: 0 and 1; in Gamma_{i,f} = g^2 gamma |a^(1/2)/Delta + a^(3/2)/(Delta - Delta_f)|^2."""
    sp = spin_zero_like()
    st = AtomicStructure(sp, 1.0, (0.0, 0.0, 1.0))
    gamma = TWO_PI * GAMMA_HZ
    omega_f = fine_structure_omega(sp)
    down, up = st.state("S1/2 mJ=-1/2"), st.state("S1/2 mJ=1/2")
    delta = 0.3 * omega_f

    def amplitudes(a, beam, b):  # type: ignore[no-untyped-def]
        """Per-path amplitudes a^(J') = r_J' Delta_J'/(g sqrt gamma); exactly one emitted polarization q' is nonzero per (b, path)."""
        g = stretched_g_half(sp, beam, field_z())
        out: dict[str, float] = {}
        for (b_label, _q), by_level in st.scattering_amplitudes_by_path(a, beam).items():
            if b_label != b.full_label:
                continue
            for lev, r in by_level.items():
                d = delta if lev == "P1/2" else delta - omega_f
                out[lev] = out.get(lev, 0.0) + (r * d / (g * math.sqrt(gamma))).real
        return out

    sig = sigma_plus_along_z(sp, delta)
    ram = amplitudes(down, sig, up)
    assert abs(ram["P1/2"]) == pytest.approx(math.sqrt(2.0) / 3.0, rel=2e-3)
    assert abs(ram["P3/2"]) == pytest.approx(math.sqrt(2.0) / 3.0, rel=2e-3)
    assert ram["P1/2"] * ram["P3/2"] < 0.0, "the two fine-structure paths interfere destructively"
    ray = amplitudes(down, sig, down)
    assert abs(ray["P1/2"]) == pytest.approx(2.0 / 3.0, rel=2e-3) and abs(ray["P3/2"]) == pytest.approx(
        1.0 / 3.0, rel=2e-3
    )
    assert ray["P1/2"] * ray["P3/2"] > 0.0, "Rayleigh amplitudes share a sign but not a magnitude"
    ray_pi = amplitudes(down, pi_beam_perp(sp, delta), down)
    assert abs(ray_pi["P1/2"]) == pytest.approx(1.0 / 3.0, rel=2e-3) and abs(ray_pi["P3/2"]) == pytest.approx(
        2.0 / 3.0, rel=2e-3
    )
    stretched = amplitudes(up, sig, up)
    assert stretched.get("P1/2", 0.0) == pytest.approx(0.0, abs=1e-12) and abs(
        stretched["P3/2"]
    ) == pytest.approx(1.0, rel=2e-3)


def test_leakage_rates_sum_and_rayleigh_dephasing_vanishes_for_clock_states(be: AtomicStructure) -> None:
    sp = be.species
    omega_f = fine_structure_omega(sp)
    low, high = _clock_states(be)
    b, _ = lin_perp_lin_pair(sp, 5.0 * omega_f, ground_energy_hz=low.energy_hz)
    rates = be.scattering_rates(low, b)
    assert set(rates) <= {s.full_label for s in be.states_of("S1/2")}
    assert all(v >= 0.0 for v in rates.values())
    # far from the fine structure the two clock states' elastic amplitudes coincide by symmetry (Section 4.5.5)
    g_el = be.rayleigh_dephasing_rate(high, low, b)
    total = sum(rates.values())
    assert g_el < 1e-3 * total
    # a Zeeman-type pair does dephase near the fine structure
    zeeman_up, zeeman_down = be.state("S1/2 F=2 mF=2"), be.state("S1/2 F=2 mF=-2")
    b_near, _ = lin_perp_lin_pair(sp, 0.3 * omega_f, ground_energy_hz=low.energy_hz)
    assert be.rayleigh_dephasing_rate(zeeman_up, zeeman_down, b_near) > 1e-2 * sum(
        be.scattering_rates(zeeman_up, b_near).values()
    )


# ---- the full master equation --------------------------------------------------------------------------------------


def _state_index(st: AtomicStructure) -> dict[str, int]:
    states = st.states_of("S1/2") + st.states_of("P1/2") + st.states_of("P3/2")
    return {s.full_label: k for k, s in enumerate(states)}


def _hamiltonian(st: AtomicStructure, beam):  # type: ignore[no-untyped-def]
    """Rotating-frame Hamiltonian on the S, P1/2, P3/2 sublevels (rad/s): ground energies E_a - E_ref on the diagonal, excited
    energies E_e - E_ref - omega_L, and (Omega_ea/2)|e><a| + h.c., so that every pair sees its own Delta_e = omega_L - (E_e - E_a)."""
    states = st.states_of("S1/2") + st.states_of("P1/2") + st.states_of("P3/2")
    idx = {s.full_label: k for k, s in enumerate(states)}
    dim = len(states)
    e_ref = min(s.energy_hz for s in st.states_of("S1/2"))
    omega_l = st.beam_omega_rad_s(beam)
    h = np.zeros((dim, dim), dtype=complex)
    for s in states:
        h[idx[s.full_label], idx[s.full_label]] = TWO_PI * (s.energy_hz - e_ref) - (
            omega_l if s.level != "S1/2" else 0.0
        )
    for a in st.states_of("S1/2"):
        for e, om, _d in st.couplings_from(a, beam):
            h[idx[e.full_label], idx[a.full_label]] += om / 2.0
            h[idx[a.full_label], idx[e.full_label]] += np.conj(om) / 2.0
    return states, idx, h


def _dressed(st: AtomicStructure, beam, bare_label: str) -> np.ndarray:  # type: ignore[no-untyped-def]
    """The eigenvector of the rotating-frame Hamiltonian with the largest weight on a bare ground state (adiabatic following)."""
    _states, idx, h = _hamiltonian(st, beam)
    _w, v = np.linalg.eigh(h)
    k = int(np.argmax(np.abs(v[idx[bare_label], :])))
    vec = v[:, k]
    return np.asarray(vec * np.exp(-1j * np.angle(vec[idx[bare_label]])))


def _weak_beam(st: AtomicStructure, beam, a, omega_over_delta: float = 0.02):  # type: ignore[no-untyped-def]
    """The same beam with its power rescaled so that max_e |Omega_ea| = omega_over_delta x min_e |Delta_e| (perturbative regime).

    The toy fixture's reduced dipole element scales as sqrt(Gamma/omega^3) and is enormous at MHz transition frequencies, so the
    power is set from the coupling rather than guessed.
    """
    from qutip_trap.light.beams import Beam

    couplings = st.couplings_from(a, beam)
    omega_max = max(abs(om) for _e, om, _d in couplings)
    delta_min = min(abs(d) for _e, _om, d in couplings)
    factor = (omega_over_delta * delta_min / omega_max) ** 2
    return Beam(
        beam.wavelength_m, beam.k_hat, beam.polarization, beam.waist_m, beam.power_w * factor, beam.pointing_m
    )


def _multilevel_mesolve(st: AtomicStructure, beam, initial, t_final: float, n: int = 201):  # type: ignore[no-untyped-def]
    """Master equation with one collapse operator per emitted polarization index q', coherent over every (e, b) pair (Section 8.1)."""
    states, idx, h = _hamiltonian(st, beam)
    dim = len(states)
    c_ops = []
    for q in (-1, 0, 1):
        L = np.zeros((dim, dim), dtype=complex)
        for e in st.states_of("P1/2") + st.states_of("P3/2"):
            for (b_label, qp), amp in st.decay_amplitudes(e).items():
                if qp == q:
                    L[idx[b_label], idx[e.full_label]] += amp
        c_ops.append(qt.Qobj(L))
    times = np.linspace(0.0, t_final, n)
    res = qt.mesolve(
        qt.Qobj(h),
        qt.Qobj(initial),
        times,
        c_ops=c_ops,
        options={"atol": 1e-13, "rtol": 1e-11, "nsteps": 10**8},
    )
    return states, idx, times, res


def test_scattering_rates_against_mesolve_with_unequal_fine_structure_rates() -> None:
    """Audit row 4.5-7: with Gamma(P1/2) != Gamma(P3/2) only the sqrt(Gamma_e)-inside form matches the master equation.

    The toy level scheme (P levels 100 and 150 MHz above S) keeps the rotating-frame dynamics unstiff, and the initial state is the
    DRESSED ground state: a sudden switch-on from the bare state leaves an undamped excited-amplitude transient that biases the rate.
    """
    sp = toy_spin_zero(gamma_p12_s=1.0e5, gamma_p32_s=3.0e5)
    st = AtomicStructure(sp, 1.0, (0.0, 0.0, 1.0))
    omega_f = fine_structure_omega(sp)
    delta = (
        0.2 * omega_f
    )  # 2 pi x 10 MHz from P1/2 and -40 MHz from P3/2: far detuned (Delta/Gamma ~ 600) and interfering
    down, up = st.state("S1/2 mJ=-1/2"), st.state("S1/2 mJ=1/2")
    beam = _weak_beam(
        st, sigma_plus_along_z(sp, delta, power_w=1e-9, waist_m=1e-3), down, omega_over_delta=0.02
    )
    omega_max = max(abs(om) for _e, om, _d in st.couplings_from(down, beam))
    assert omega_max / abs(delta) <= 0.02 + 1e-12
    predicted = st.scattering_rates(down, beam)
    psi = _dressed(st, beam, down.full_label)
    # fit window: 0.1% total transfer, so depletion of the source state biases the slope by about 0.05%
    t_final = 1e-3 / sum(predicted.values())
    _states, idx, times, res = _multilevel_mesolve(st, beam, np.outer(psi, psi.conj()), t_final, n=101)
    p_up = np.array([r[idx[up.full_label], idx[up.full_label]].real for r in res.states])
    slope = np.polyfit(times, p_up, 1)[0]
    assert slope == pytest.approx(predicted[up.full_label], rel=5e-3)
    # the Gamma_e-outside form (one shared Gamma times the residual population) is wrong by a large factor here
    outside = st.residual_excited_population(down, beam)
    for gamma_shared in (1.0e5, 3.0e5):
        assert not math.isclose(outside * gamma_shared, predicted[up.full_label], rel_tol=0.3)


@pytest.mark.slow
def test_rayleigh_dephasing_against_mesolve() -> None:
    """The qubit coherence decays at (Gamma_Ram + Gamma_el)/2 with Gamma_el = sum_q' |r_u - r_d|^2 (Uys Eqs. 6-8)."""
    sp = toy_spin_zero(gamma_p12_s=1.0e5, gamma_p32_s=1.0e5)
    st = AtomicStructure(sp, 1.0, (0.0, 0.0, 1.0))
    omega_f = fine_structure_omega(sp)
    down, up = st.state("S1/2 mJ=-1/2"), st.state("S1/2 mJ=1/2")
    beam = _weak_beam(
        st, sigma_plus_along_z(sp, 0.3 * omega_f, power_w=2e-9, waist_m=1e-3), down, omega_over_delta=0.02
    )
    g_du = st.scattering_rates(down, beam).get(up.full_label, 0.0)
    g_ud = st.scattering_rates(up, beam).get(down.full_label, 0.0)
    g_el = st.rayleigh_dephasing_rate(up, down, beam)
    assert g_el > 0.0
    psi = (_dressed(st, beam, down.full_label) + _dressed(st, beam, up.full_label)) / math.sqrt(2.0)
    t_final = 0.05 / (g_du + g_ud + g_el)
    _states, idx, times, res = _multilevel_mesolve(st, beam, np.outer(psi, psi.conj()), t_final, n=101)
    coh = np.array([abs(r[idx[up.full_label], idx[down.full_label]]) for r in res.states])
    rate = -np.polyfit(times, np.log(coh), 1)[0]
    assert rate == pytest.approx((g_du + g_ud + g_el) / 2.0, rel=1e-2)
    # the negative control of Section 9.13: sqrt(Gamma_el/2) sigma_z would decay twice as fast as (1/2) sqrt(Gamma_el) sigma_z
    ts = np.linspace(0, 2.0, 51)
    plus = qt.Qobj([[0.5, 0.5], [0.5, 0.5]])
    r1 = qt.mesolve(0 * qt.qeye(2), plus, ts, c_ops=[0.5 * math.sqrt(1.7) * qt.sigmaz()])
    r2 = qt.mesolve(0 * qt.qeye(2), plus, ts, c_ops=[math.sqrt(1.7 / 2.0) * qt.sigmaz()])
    k1 = -np.polyfit(ts, np.log([abs(s[0, 1]) for s in r1.states]), 1)[0]
    k2 = -np.polyfit(ts, np.log([abs(s[0, 1]) for s in r2.states]), 1)[0]
    assert k1 == pytest.approx(0.85, rel=1e-4) and k2 == pytest.approx(1.70, rel=1e-4)
    assert max(abs(s[0, 0].real - 0.5) for s in r1.states) < 1e-12, "the Rayleigh channel is pure dephasing"


def test_species_api_end_to_end_on_the_fixture() -> None:
    """The Appendix E methods route through the same engine and speak Hz."""
    from qutip_trap.device.model import Field

    sp = be9_like()
    field = Field(B_gauss=1.0, direction=(0.0, 0.0, 1.0), noise=None)
    st = AtomicStructure(sp, 1.0, (0.0, 0.0, 1.0))
    omega_f = fine_structure_omega(sp)
    low, high = st.state("S1/2 F=2 mF=0"), st.state("S1/2 F=1 mF=0")
    b, r = lin_perp_lin_pair(
        sp,
        0.414 * omega_f,
        ground_energy_hz=low.energy_hz,
        omega_q_rad_s=TWO_PI * (high.energy_hz - low.energy_hz),
    )
    assert sp.raman_coupling_hz("S1/2 F=2 mF=0", "S1/2 F=1 mF=0", b, r, field) == pytest.approx(
        st.raman_coupling_rad_s(low, high, b, r) / TWO_PI
    )
    assert sp.light_shift_hz("S1/2 F=2 mF=0", b, field) == pytest.approx(
        st.light_shift_rad_s(low, (b,)) / TWO_PI
    )
    rates = sp.scattering_rates_hz("S1/2 F=2 mF=0", b, field)
    assert rates == st.scattering_rates(low, b)
    assert sp.rayleigh_dephasing_hz(b, field) == pytest.approx(st.rayleigh_dephasing_rate(high, low, b))
    e = st.states_of("P3/2")[0]
    omega = sp.rabi_frequency_hz("S1/2 F=2 mF=0", e.full_label, b, field)
    assert omega == pytest.approx(st.single_photon_coupling_rad_s(low, e, b) / TWO_PI)
    d = sp.dipole_element("S1/2 F=2 mF=0", e.full_label, 0, field)
    assert isinstance(d, complex)
    # a pure sigma+ beam drives only m -> m + 1 (Section 4.5.3)
    sig = sigma_plus_along_z(sp, 0.414 * omega_f)
    couplings = list(st.couplings_from(st.state("S1/2 F=2 mF=1"), sig))
    # audit item B11: the guard here used to read `abs(om) > 1e-9 * abs(om) + 1e-30 and abs(om) > 0`, which
    # is self-referential and reduces to `abs(om) > 0`, so it conveyed nothing. The intended threshold is
    # RELATIVE to the largest coupling in the set.
    largest = max(abs(om) for _e2, om, _d in couplings)
    assert largest > 0.0
    driven = 0
    for e2, om, _d in couplings:
        mf = st.spectra[e2.level].mF[st.spectra[e2.level].index(e2.label)]
        if abs(om) > 1e-9 * largest:
            assert mf == 2, f"sigma+ out of mF = 1 must reach mF = 2 only, not {mf}"
            driven += 1
    assert driven > 0, "at least one sigma+ coupling must survive the relative threshold"
