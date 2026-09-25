"""Raman couplings, light shifts, decay amplitudes and scattering from the level structure (PLAN.md Section 4.5) against the closed forms of Ozeri 2005/2007 and Wineland 2003 and against a full multi-level master equation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt
from scipy.optimize import brentq

from qutip_trap.light.beams import Beam
from qutip_trap.species import species
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import C_M_PER_S, TWO_PI
from tests.atomic_fixtures import (
    GAMMA_HZ,
    be9_like,
    beam_at_detuning,
    field_z,
    fine_structure_omega,
    lin_perp_lin_pair,
    pi_beam_perp,
    sigma_plus_along_z,
    spin_zero_like,
    stretched_g_half,
    toy_spin_zero,
)
from tests.oracles import (
    ozeri_gamma_raman,
    ozeri_gamma_total,
    ozeri_p_total,
    ozeri_raman_rabi_half,
    wineland_clock_light_shift,
    wineland_p_se_clock,
)

B_GAUSS = 4.0
Z_HAT = (0.0, 0.0, 1.0)


@pytest.fixture(scope="module")
def be() -> AtomicStructure:
    return AtomicStructure(be9_like(), 1.0, Z_HAT)


@pytest.fixture(scope="module")
def yb() -> Species:
    return species("171Yb+")


def _clock_states(st: AtomicStructure):  # type: ignore[no-untyped-def]
    low = st.state("S1/2 F=2 mF=0")  # F = 2 lies BELOW F = 1 for 9Be+ (negative moment)
    high = st.state("S1/2 F=1 mF=0")
    assert high.energy_hz > low.energy_hz
    return low, high


# ---- the explicit sums against the closed forms ------------------------------------------------------------------


def test_raman_coupling_reduces_to_ozeris_closed_form(be: AtomicStructure) -> None:
    """Omega_{g1 g2} = 2 x (g_b g_r/3)(b_- r_- - b_+ r_+) omega_f/[Delta(Delta - omega_f)] for lin-perp-lin beams, where
    b_- = r_- = b_+ = -r_+ = 1/sqrt2 (PLAN.md 4.3.2); the residual is O(omega_0/Delta) from referencing Delta to one ground
    state. Same-handed pairs cancel."""
    sp = be.species
    omega_f = fine_structure_omega(sp)
    low, high = _clock_states(be)
    omega_q = TWO_PI * (high.energy_hz - low.energy_hz)
    h = 1.0 / math.sqrt(2.0)
    for x in (0.414, -0.7, 2.5):
        delta = x * omega_f
        b, r = lin_perp_lin_pair(sp, delta, ground_energy_hz=low.energy_hz, omega_q_rad_s=omega_q)
        g_b, g_r = stretched_g_half(sp, b, field_z()), stretched_g_half(sp, r, field_z())
        closed = 2.0 * abs(ozeri_raman_rabi_half(g_b, g_r, h, h, h, -h, delta, omega_f))
        assert closed == pytest.approx(
            2.0 * abs(g_b * g_r / 3.0 * omega_f / (delta * (delta - omega_f))), rel=1e-12
        )
        assert abs(be.raman_coupling_rad_s(low, high, b, r)) == pytest.approx(closed, rel=3e-2)
        assert ozeri_raman_rabi_half(g_b, g_r, h, h, h, h, delta, omega_f) == 0.0


def test_raman_coupling_closed_form_tightens_as_the_hyperfine_splitting_shrinks() -> None:
    """The residual is the O(omega_0/Delta) reference ambiguity: scaling A_S by 1e-3 brings the agreement to 2e-4 (at a
    negligible field, where the F labels are the states the closed form assumes)."""
    sp = be9_like(a_scale=1e-3)
    st = AtomicStructure(sp, 1e-6, Z_HAT)
    omega_f = fine_structure_omega(sp)
    low, high = st.state("S1/2 F=2 mF=0"), st.state("S1/2 F=1 mF=0")
    delta = 0.414 * omega_f
    b, r = lin_perp_lin_pair(
        sp, delta, ground_energy_hz=low.energy_hz, omega_q_rad_s=TWO_PI * (high.energy_hz - low.energy_hz)
    )
    g = stretched_g_half(sp, b, field_z())
    closed = 2.0 * abs(g * g / 3.0 * omega_f / (delta * (delta - omega_f)))
    assert abs(st.raman_coupling_rad_s(low, high, b, r)) == pytest.approx(closed, rel=2e-4)


def test_total_and_raman_scattering_rates_reduce_to_ozeris_forms(be: AtomicStructure) -> None:
    """Gamma_total = (gamma/3) g^2 [1/Delta^2 + 2/(Delta - omega_f)^2] per lin-perp-B beam; the inelastic part is
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
        assert total == pytest.approx(ozeri_gamma_total(gamma, g, 0.0, 1.0, 0.0, delta, omega_f), rel=3e-3)
        inelastic = total - rates[low.full_label]
        assert inelastic == pytest.approx(
            ozeri_gamma_raman(gamma, g, 0.0, 1.0, 0.0, delta, omega_f), rel=3e-2
        )
        # the residual excited population times the (single) decay rate is the total rate
        assert be.residual_excited_population(low, b) * gamma == pytest.approx(total, rel=1e-3)


def test_scattering_probability_per_pi_pulse_has_ozeris_minimum(be: AtomicStructure) -> None:
    """P_total = Gamma_total x pi/|Omega_R| = (pi gamma/omega_f)(2 Delta^2 + (Delta - omega_f)^2)/|Delta(Delta - omega_f)|,
    minimum 2 sqrt2 pi gamma/omega_f at Delta = (sqrt2 - 1) omega_f, power independent (Section 4.3.2)."""
    sp = be.species
    gamma = TWO_PI * GAMMA_HZ
    omega_f = fine_structure_omega(sp)
    low, high = _clock_states(be)
    omega_q = TWO_PI * (high.energy_hz - low.energy_hz)
    x_opt = math.sqrt(2.0) - 1.0

    def p_per_pi(x: float, power_w: float = 1e-3) -> float:
        b, r = lin_perp_lin_pair(
            sp, x * omega_f, power_w=power_w, ground_energy_hz=low.energy_hz, omega_q_rad_s=omega_q
        )
        total = sum(be.scattering_rates(low, b).values()) + sum(be.scattering_rates(low, r).values())
        return total * math.pi / abs(be.raman_coupling_rad_s(low, high, b, r))

    values = [p_per_pi(x) for x in (0.25, x_opt, 0.6, -2.0)]
    for x, p in zip((0.25, x_opt, 0.6, -2.0), values):
        assert p == pytest.approx(ozeri_p_total(gamma, omega_f, x * omega_f), rel=3e-2)
    assert values[1] < values[0] and values[1] < values[2]
    assert values[1] == pytest.approx(wineland_p_se_clock(gamma, omega_f), rel=3e-2)
    assert p_per_pi(x_opt, power_w=2e-3) == pytest.approx(values[1], rel=1e-9)


def test_clock_light_shift_needs_the_three_index_detuning(be: AtomicStructure) -> None:
    """delta_{0<->0} = -(g_b^2 + g_r^2)(omega_0/3)[1/Delta^2 + 2/(Delta - omega_F)^2] (Wineland Eq. 2.17) comes out because
    each ground state carries its own detuning; the summed line strength is the same for both clock states, so one Delta
    per intermediate level would give exactly zero."""
    sp = be.species
    omega_f = fine_structure_omega(sp)
    low, high = _clock_states(be)
    omega_0 = TWO_PI * (high.energy_hz - low.energy_hz)
    delta = 0.414 * omega_f
    b, r = lin_perp_lin_pair(sp, delta, ground_energy_hz=low.energy_hz, omega_q_rad_s=omega_0)
    g_b, g_r = stretched_g_half(sp, b, field_z()), stretched_g_half(sp, r, field_z())
    mine = be.light_shift_rad_s(high, (b, r)) - be.light_shift_rad_s(low, (b, r))
    assert mine == pytest.approx(wineland_clock_light_shift(g_b, g_r, omega_0, delta, omega_f), rel=3e-2)
    for beam in (b, r):
        s_low = sum(abs(om) ** 2 for _e, om, _d in be.couplings_from(low, beam))
        s_high = sum(abs(om) ** 2 for _e, om, _d in be.couplings_from(high, beam))
        assert s_low == pytest.approx(s_high, rel=1e-12)
    # tight at a small splitting and a negligible field
    small = be9_like(a_scale=1e-3)
    st = AtomicStructure(small, 1e-6, Z_HAT)
    lo2, hi2 = st.state("S1/2 F=2 mF=0"), st.state("S1/2 F=1 mF=0")
    w0 = TWO_PI * (hi2.energy_hz - lo2.energy_hz)
    b2, r2 = lin_perp_lin_pair(small, delta, ground_energy_hz=lo2.energy_hz, omega_q_rad_s=w0)
    g2 = stretched_g_half(small, b2, field_z())
    mine2 = st.light_shift_rad_s(hi2, (b2, r2)) - st.light_shift_rad_s(lo2, (b2, r2))
    assert mine2 == pytest.approx(wineland_clock_light_shift(g2, g2, w0, delta, omega_f), rel=2e-3)


def _elliptical_beam(sp: Species, delta_rad_s: float, angle_rad: float, ground_energy_hz: float) -> Beam:
    """A beam along B = z running from pure sigma+ (angle 0) to pure sigma- (angle pi/2)."""
    c, s = math.cos(angle_rad) / math.sqrt(2.0), math.sin(angle_rad) / math.sqrt(2.0)
    pol = (complex(c + s, 0.0), complex(0.0, c - s), 0.0 + 0.0j)
    return beam_at_detuning(sp, delta_rad_s, Z_HAT, pol, ground_energy_hz=ground_energy_hz)


def test_the_9be_zeeman_qubit_shift_has_a_polarization_null_and_the_clock_one_does_not() -> None:
    """The |2,2> <-> |1,1> differential shift crosses zero between sigma+ and sigma-; the clock shift, sharing one prefactor
    and one bracket with R_SE, is polarization independent and therefore unnullable (PLAN.md 4.3.2)."""
    sp = be9_like()
    st = AtomicStructure(sp, 1.0, Z_HAT)
    delta = 0.414213562373 * fine_structure_omega(sp)

    def shift(lower: str, upper: str, angle: float) -> float:
        beam = _elliptical_beam(sp, delta, angle, st.state(upper).energy_hz)
        return st.light_shift_rad_s(st.state(upper), (beam,)) - st.light_shift_rad_s(st.state(lower), (beam,))

    def zeeman(angle: float) -> float:
        return shift("S1/2 F=1 mF=1", "S1/2 F=2 mF=2", angle)

    assert zeeman(0.0) * zeeman(math.pi / 2.0) < 0.0
    null = brentq(zeeman, 0.0, math.pi / 2.0, xtol=1e-14)
    assert zeeman(null) == pytest.approx(0.0, abs=1e-6 * abs(zeeman(0.0)))
    assert zeeman(null - 0.05) * zeeman(null + 0.05) < 0.0
    clock = np.array(
        [shift("S1/2 F=1 mF=0", "S1/2 F=2 mF=0", a) for a in np.linspace(0.0, math.pi / 2.0, 41)]
    )
    assert np.all(clock > 0.0) or np.all(clock < 0.0)
    # the summed line strength out of either state of each pair is the same: the shifts come only from the detunings
    for lower, upper in (("S1/2 F=1 mF=0", "S1/2 F=2 mF=0"), ("S1/2 F=1 mF=1", "S1/2 F=2 mF=2")):
        for angle in (0.0, 0.3, math.pi / 4.0, math.pi / 2.0):
            beam = _elliptical_beam(sp, delta, angle, st.state(upper).energy_hz)
            lo = sum(abs(om) ** 2 for _e, om, _d in st.couplings_from(st.state(lower), beam))
            hi = sum(abs(om) ** 2 for _e, om, _d in st.couplings_from(st.state(upper), beam))
            assert lo == pytest.approx(hi, rel=1e-12), (lower, angle)


def test_ozeri_2005_single_electron_amplitudes() -> None:
    """Raman a^(1/2) = -sqrt2/3, a^(3/2) = +sqrt2/3 (equal and opposite); Rayleigh with sigma+ on m = -1/2: 2/3 and 1/3; with
    pi on m = -1/2: 1/3 and 2/3; the stretched state under sigma+: 0 and 1 (Ozeri 2005)."""
    sp = spin_zero_like()
    st = AtomicStructure(sp, 1.0, Z_HAT)
    gamma = TWO_PI * GAMMA_HZ
    omega_f = fine_structure_omega(sp)
    down, up = st.state("S1/2 mJ=-1/2"), st.state("S1/2 mJ=1/2")
    delta = 0.3 * omega_f

    def amplitudes(a, beam, b):  # type: ignore[no-untyped-def]
        """Per-path a^(J') = r_J' Delta_J'/(g sqrt gamma); one emitted polarization q' is nonzero per (b, path)."""
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
    assert abs(ray["P1/2"]) == pytest.approx(2.0 / 3.0, rel=2e-3)
    assert abs(ray["P3/2"]) == pytest.approx(1.0 / 3.0, rel=2e-3)
    assert ray["P1/2"] * ray["P3/2"] > 0.0
    ray_pi = amplitudes(down, pi_beam_perp(sp, delta), down)
    assert abs(ray_pi["P1/2"]) == pytest.approx(1.0 / 3.0, rel=2e-3)
    assert abs(ray_pi["P3/2"]) == pytest.approx(2.0 / 3.0, rel=2e-3)
    stretched = amplitudes(up, sig, up)
    assert stretched.get("P1/2", 0.0) == pytest.approx(0.0, abs=1e-12)
    assert abs(stretched["P3/2"]) == pytest.approx(1.0, rel=2e-3)


def test_leakage_rates_sum_and_rayleigh_dephasing_vanishes_for_clock_states(be: AtomicStructure) -> None:
    sp = be.species
    omega_f = fine_structure_omega(sp)
    low, high = _clock_states(be)
    b, _ = lin_perp_lin_pair(sp, 5.0 * omega_f, ground_energy_hz=low.energy_hz)
    rates = be.scattering_rates(low, b)
    assert set(rates) <= {s.full_label for s in be.states_of("S1/2")}
    assert all(v >= 0.0 for v in rates.values())
    # far from the fine structure the two clock states' elastic amplitudes coincide by symmetry (Section 4.5.5)
    assert be.rayleigh_dephasing_rate(high, low, b) < 1e-3 * sum(rates.values())
    # a Zeeman-type pair does dephase near the fine structure
    zeeman_up, zeeman_down = be.state("S1/2 F=2 mF=2"), be.state("S1/2 F=2 mF=-2")
    b_near, _ = lin_perp_lin_pair(sp, 0.3 * omega_f, ground_energy_hz=low.energy_hz)
    assert be.rayleigh_dephasing_rate(zeeman_up, zeeman_down, b_near) > 1e-2 * sum(
        be.scattering_rates(zeeman_up, b_near).values()
    )


# ---- the full master equation --------------------------------------------------------------------------------------


def _hamiltonian(st: AtomicStructure, beam: Beam):  # type: ignore[no-untyped-def]
    """Rotating-frame Hamiltonian on the S, P1/2, P3/2 sublevels (rad/s): ground energies E_a - E_ref on the diagonal,
    excited energies E_e - E_ref - omega_L, and (Omega_ea/2)|e><a| + h.c."""
    states = st.states_of("S1/2") + st.states_of("P1/2") + st.states_of("P3/2")
    idx = {s.full_label: k for k, s in enumerate(states)}
    e_ref = min(s.energy_hz for s in st.states_of("S1/2"))
    omega_l = st.beam_omega_rad_s(beam)
    h = np.zeros((len(states), len(states)), dtype=complex)
    for s in states:
        h[idx[s.full_label], idx[s.full_label]] = TWO_PI * (s.energy_hz - e_ref) - (
            omega_l if s.level != "S1/2" else 0.0
        )
    for a in st.states_of("S1/2"):
        for e, om, _d in st.couplings_from(a, beam):
            h[idx[e.full_label], idx[a.full_label]] += om / 2.0
            h[idx[a.full_label], idx[e.full_label]] += np.conj(om) / 2.0
    return states, idx, h


def _dressed(st: AtomicStructure, beam: Beam, bare_label: str) -> np.ndarray:
    """The rotating-frame eigenvector with the largest weight on a bare ground state (adiabatic following)."""
    _states, idx, h = _hamiltonian(st, beam)
    _w, v = np.linalg.eigh(h)
    vec = v[:, int(np.argmax(np.abs(v[idx[bare_label], :])))]
    return np.asarray(vec * np.exp(-1j * np.angle(vec[idx[bare_label]])))


def _weak_beam(st: AtomicStructure, beam: Beam, a, omega_over_delta: float = 0.02) -> Beam:  # type: ignore[no-untyped-def]
    """The beam with its power set so that max_e |Omega_ea| = omega_over_delta x min_e |Delta_e| (perturbative regime)."""
    couplings = st.couplings_from(a, beam)
    omega_max = max(abs(om) for _e, om, _d in couplings)
    delta_min = min(abs(d) for _e, _om, d in couplings)
    factor = (omega_over_delta * delta_min / omega_max) ** 2
    return Beam(
        beam.wavelength_m, beam.k_hat, beam.polarization, beam.waist_m, beam.power_w * factor, beam.pointing_m
    )


def _multilevel_mesolve(st: AtomicStructure, beam: Beam, initial: np.ndarray, t_final: float, n: int = 201):  # type: ignore[no-untyped-def]
    """The master equation with one collapse operator per emitted polarization q', coherent over every (e, b) pair."""
    states, idx, h = _hamiltonian(st, beam)
    c_ops = []
    for q in (-1, 0, 1):
        L = np.zeros((len(states), len(states)), dtype=complex)
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
    return idx, times, res


def test_scattering_rates_against_mesolve_with_unequal_fine_structure_rates() -> None:
    """With Gamma(P1/2) != Gamma(P3/2) only the sqrt(Gamma_e)-inside form matches the master equation (P levels 100 and
    150 MHz above S keep the dynamics unstiff; the initial state is the DRESSED ground state)."""
    sp = toy_spin_zero(gamma_p12_s=1.0e5, gamma_p32_s=3.0e5)
    st = AtomicStructure(sp, 1.0, Z_HAT)
    delta = 0.2 * fine_structure_omega(sp)  # far detuned (Delta/Gamma ~ 600) and interfering
    down, up = st.state("S1/2 mJ=-1/2"), st.state("S1/2 mJ=1/2")
    beam = _weak_beam(st, sigma_plus_along_z(sp, delta, power_w=1e-9, waist_m=1e-3), down)
    assert max(abs(om) for _e, om, _d in st.couplings_from(down, beam)) / abs(delta) <= 0.02 + 1e-12
    predicted = st.scattering_rates(down, beam)
    psi = _dressed(st, beam, down.full_label)
    t_final = 1e-3 / sum(predicted.values())  # 0.1% total transfer: depletion biases the slope by about 0.05%
    idx, times, res = _multilevel_mesolve(st, beam, np.outer(psi, psi.conj()), t_final, n=101)
    p_up = np.array([r[idx[up.full_label], idx[up.full_label]].real for r in res.states])
    assert np.polyfit(times, p_up, 1)[0] == pytest.approx(predicted[up.full_label], rel=5e-3)
    # one shared Gamma times the residual population (the Gamma_e-outside form) is wrong by a large factor here
    outside = st.residual_excited_population(down, beam)
    for gamma_shared in (1.0e5, 3.0e5):
        assert not math.isclose(outside * gamma_shared, predicted[up.full_label], rel_tol=0.3)


@pytest.mark.slow
def test_rayleigh_dephasing_against_mesolve() -> None:
    """The qubit coherence decays at (Gamma_Ram + Gamma_el)/2 with Gamma_el = sum_q' |r_u - r_d|^2 (Uys Eqs. 6-8)."""
    sp = toy_spin_zero(gamma_p12_s=1.0e5, gamma_p32_s=1.0e5)
    st = AtomicStructure(sp, 1.0, Z_HAT)
    down, up = st.state("S1/2 mJ=-1/2"), st.state("S1/2 mJ=1/2")
    beam = _weak_beam(
        st, sigma_plus_along_z(sp, 0.3 * fine_structure_omega(sp), power_w=2e-9, waist_m=1e-3), down
    )
    g_du = st.scattering_rates(down, beam).get(up.full_label, 0.0)
    g_ud = st.scattering_rates(up, beam).get(down.full_label, 0.0)
    g_el = st.rayleigh_dephasing_rate(up, down, beam)
    assert g_el > 0.0
    psi = (_dressed(st, beam, down.full_label) + _dressed(st, beam, up.full_label)) / math.sqrt(2.0)
    t_final = 0.05 / (g_du + g_ud + g_el)
    idx, times, res = _multilevel_mesolve(st, beam, np.outer(psi, psi.conj()), t_final, n=101)
    coh = np.array([abs(r[idx[up.full_label], idx[down.full_label]]) for r in res.states])
    assert -np.polyfit(times, np.log(coh), 1)[0] == pytest.approx((g_du + g_ud + g_el) / 2.0, rel=1e-2)


def test_species_api_end_to_end_on_the_fixture() -> None:
    """The Species methods route through the same engine and speak Hz."""
    from qutip_trap.device.model import Field

    sp = be9_like()
    field = Field(B_gauss=1.0, direction=Z_HAT, noise=None)
    st = AtomicStructure(sp, 1.0, Z_HAT)
    low, high = st.state("S1/2 F=2 mF=0"), st.state("S1/2 F=1 mF=0")
    b, r = lin_perp_lin_pair(
        sp,
        0.414 * fine_structure_omega(sp),
        ground_energy_hz=low.energy_hz,
        omega_q_rad_s=TWO_PI * (high.energy_hz - low.energy_hz),
    )
    assert sp.raman_coupling_hz("S1/2 F=2 mF=0", "S1/2 F=1 mF=0", b, r, field) == pytest.approx(
        st.raman_coupling_rad_s(low, high, b, r) / TWO_PI
    )
    assert sp.light_shift_hz("S1/2 F=2 mF=0", b, field) == pytest.approx(
        st.light_shift_rad_s(low, (b,)) / TWO_PI
    )
    assert sp.scattering_rates_hz("S1/2 F=2 mF=0", b, field) == st.scattering_rates(low, b)
    assert sp.rayleigh_dephasing_hz(b, field) == pytest.approx(st.rayleigh_dephasing_rate(high, low, b))
    e = st.states_of("P3/2")[0]
    omega = sp.rabi_frequency_hz("S1/2 F=2 mF=0", e.full_label, b, field)
    assert omega == pytest.approx(st.single_photon_coupling_rad_s(low, e, b) / TWO_PI)
    assert isinstance(sp.dipole_element("S1/2 F=2 mF=0", e.full_label, 0, field), complex)


# ---- decay amplitudes and branchings on real multi-channel species ----------------------------------------------------


@pytest.mark.parametrize("name", ["171Yb+", "40Ca+", "88Sr+"])
def test_decay_amplitude_shares_reproduce_the_tabulated_branchings(name: str) -> None:
    """sum_{b in lo, q} |c_{e->b q}|^2 / Gamma_e == Transition.branching(lo -> e) to 1e-12 for every E1-reached upper level
    and every dressed sublevel: |c|^2 is the partial-rate fraction, each channel weighted by its own omega^3."""
    st = AtomicStructure(species(name), B_GAUSS, Z_HAT)
    checked = 0
    for level in {up for (_lo, up) in st.e1}:
        gamma_e = st.total_decay_rate_rad_s(level)
        want = {lo: st.e1[(lo, level)].branching for lo in st.lower_levels_of(level)}
        for e in st.states_of(level):
            got: dict[str, float] = dict.fromkeys(want, 0.0)
            for (label, _q), amp in st.decay_amplitudes(e).items():
                got[label.split()[0]] += abs(amp) ** 2 / gamma_e
            for lo, fraction in want.items():
                assert got[lo] == pytest.approx(fraction, rel=1e-12, abs=1e-15), (
                    f"{name} {e.full_label} -> {lo}"
                )
            checked += 1
    assert checked >= 4


def test_the_scattering_budget_of_the_355_nm_drive_leaks_at_the_tabulated_rate(yb: Species) -> None:
    """On the 355 nm clock drive the leakage share out of the qubit manifold is percent level (the D branchings weighted
    by the two paths)."""
    st = AtomicStructure(yb, 5.0, Z_HAT)
    beam = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 20e-6, 10e-3, (0.0, 0.0, 0.0))
    rates = st.scattering_rates(st.state("S1/2 F=0 mF=0"), beam)
    total = sum(rates.values())
    leak = sum(v for k, v in rates.items() if not k.startswith("S1/2"))
    assert 0.0 < leak / total < 0.02
    assert sum(v for k, v in rates.items() if k.startswith("S1/2")) / total > 0.98


def _lin_perp_lin(lam: float) -> tuple[Beam, Beam]:
    """Two linear polarizations perpendicular to each other and to B = z: pure sigma content, no pi."""
    return (
        Beam(lam, Z_HAT, (1.0, 0.0, 0.0), 20e-6, 1e-3, (0.0, 0.0, 0.0)),
        Beam(lam, Z_HAT, (0.0, 1.0, 0.0), 20e-6, 1e-3, (0.0, 0.0, 0.0)),
    )


def test_the_two_paths_enter_with_opposite_signs(yb: Species) -> None:
    """The P1/2 and P3/2 partial Raman sums are antiparallel (PLAN.md 4.5.4): the origin of omega_f/[Delta(Delta - omega_f)]."""
    st = AtomicStructure(yb, 5.0, Z_HAT)
    dn, up = st.state("S1/2 F=0 mF=0"), st.state("S1/2 F=1 mF=0")
    f0 = yb.level("P1/2").energy_hz
    for detuning_thz in (-50.0, -100.0, -200.0, -400.0):
        b1, b2 = _lin_perp_lin(C_M_PER_S / (f0 + detuning_thz * 1e12))
        c1 = {e.full_label: (om, d) for e, om, d in st.couplings_from(dn, b1)}
        c2 = {e.full_label: om for e, om, _ in st.couplings_from(up, b2)}
        parts: dict[str, complex] = {}
        for key, (om1, d1) in c1.items():
            om2 = c2.get(key)
            if om2 is not None:
                parts[key.split()[0]] = parts.get(key.split()[0], 0.0 + 0.0j) + np.conj(om2) * om1 / (
                    2.0 * d1
                )
        assert set(parts) == {"P1/2", "P3/2"}
        ratio = parts["P3/2"] / parts["P1/2"]
        assert ratio.real < 0.0, f"the two paths must interfere destructively at {detuning_thz} THz"
        assert abs(ratio.imag) < 1e-9 * abs(ratio.real)


def test_the_raman_coupling_follows_the_two_path_closed_form_not_one_over_delta(yb: Species) -> None:
    """|Omega_R| |Delta(Delta - omega_f)|/omega_f varies by < 35% over Delta/2pi = -50 to -400 THz, |Omega_R Delta| by 2.6x
    (171Yb+'s 99.84 THz fine structure puts the 1/Delta^2 regime outside the optical band)."""
    st = AtomicStructure(yb, 5.0, Z_HAT)
    dn, up = st.state("S1/2 F=0 mF=0"), st.state("S1/2 F=1 mF=0")
    f0 = yb.level("P1/2").energy_hz
    omega_f_thz = (yb.level("P3/2").energy_hz - f0) / 1e12
    assert omega_f_thz == pytest.approx(99.8432, rel=1e-5)
    two_path: list[float] = []
    one_path: list[float] = []
    for detuning_thz in (-50.0, -100.0, -200.0, -400.0):
        omega = abs(st.raman_coupling_rad_s(dn, up, *_lin_perp_lin(C_M_PER_S / (f0 + detuning_thz * 1e12))))
        two_path.append(omega * abs(detuning_thz * (detuning_thz - omega_f_thz)) / omega_f_thz)
        one_path.append(omega * abs(detuning_thz))
    assert max(two_path) / min(two_path) < 1.35
    assert max(one_path) / min(one_path) > 2.5


def test_the_raman_spin_flip_rate_is_interference_protected_and_the_leakage_is_not(yb: Species) -> None:
    """The spin-flip rate falls faster than 1/Delta^3; the D-state leakage, not protected, as about 1/Delta^2 (4.5.5)."""
    st = AtomicStructure(yb, 5.0, Z_HAT)
    dn, up = st.state("S1/2 F=0 mF=0"), st.state("S1/2 F=1 mF=0")
    f0 = yb.level("P1/2").energy_hz
    flips: list[float] = []
    leaks: list[float] = []
    for detuning_thz in (-100.0, -200.0, -400.0):
        rates = st.scattering_rates(dn, _lin_perp_lin(C_M_PER_S / (f0 + detuning_thz * 1e12))[0])
        flips.append(rates.get(up.full_label, 0.0))
        leaks.append(sum(v for k, v in rates.items() if not k.startswith("S1/2")))
    assert all(flips[i + 1] / flips[i] < 0.1 for i in range(2)), flips
    assert all(0.2 < leaks[i + 1] / leaks[i] < 0.45 for i in range(2)), leaks


def _two_channel_species(branchings: tuple[float, float], deficit: float | None) -> Species:
    """A toy S/D/P species whose P level has two tabulated E1 channels of the given branchings."""
    e_d, e_p, tau = 1.0e13, 8.0e14, 8.0e-9
    gamma = 1.0 / (TWO_PI * tau)
    untab = () if deficit is None else (("an untabulated channel", deficit),)
    levels = (
        Level("S1/2", 0.0, None, 0.0, 0.0, 2.0, ("test",)),
        Level("D3/2", e_d, 1.0, 0.0, 0.0, 0.8, ("test",)),
        Level("P1/2", e_p, tau, 0.0, 0.0, 0.667, ("test",), untabulated_branching=untab),
    )
    trs = (
        Transition("S1/2", "P1/2", C_M_PER_S / e_p, gamma, branchings[0], "E1", ("test",)),
        Transition("D3/2", "P1/2", C_M_PER_S / (e_p - e_d), gamma, branchings[1], "E1", ("test",)),
    )
    return Species(
        "toy",
        40.0,
        0.0,
        0.0,
        levels,
        trs,
        ("S1/2 mJ=-1/2", "D3/2 mJ=-1/2"),
        "S1/2-P1/2",
        ("D3/2-P1/2",),
        None,
    )


def test_an_incomplete_branching_set_is_refused_unless_the_deficit_is_declared() -> None:
    assert _two_channel_species((0.94, 0.06), None).level("P1/2").untabulated_branching == ()
    with pytest.raises(ValueError, match=r"E1 branchings out of P1/2 sum to .* not 1 within 1e-9"):
        _two_channel_species((0.90, 0.06), None)
    sp = _two_channel_species((0.90, 0.06), 0.04)
    assert sp.level("P1/2").untabulated_branching == (("an untabulated channel", 0.04),)
    with pytest.raises(ValueError, match="must lie in"):
        Level("P1/2", 1.0, 1e-8, 0.0, 0.0, 0.667, ("test",), untabulated_branching=(("x", 1.5),))
    with pytest.raises(ValueError, match="must name itself"):
        Level("P1/2", 1.0, 1e-8, 0.0, 0.0, 0.667, ("test",), untabulated_branching=(("", 0.5),))


def test_a_declared_deficit_scales_the_amplitudes_rather_than_being_renormalized_away() -> None:
    """sum_{b q} |c|^2 equals the TABULATED total: a declared 4% leak stays 4% missing."""
    st = AtomicStructure(_two_channel_species((0.90, 0.06), 0.04), B_GAUSS, Z_HAT)
    assert st.tabulated_branching_total("P1/2") == pytest.approx(0.96, rel=1e-12)
    e = st.states_of("P1/2")[0]
    gamma = st.total_decay_rate_rad_s("P1/2")
    shares: dict[str, float] = {}
    for (label, _q), amp in st.decay_amplitudes(e).items():
        shares[label.split()[0]] = shares.get(label.split()[0], 0.0) + abs(amp) ** 2 / gamma
    assert shares["S1/2"] == pytest.approx(0.90, rel=1e-12)
    assert shares["D3/2"] == pytest.approx(0.06, rel=1e-12)


def test_transitions_out_of_one_upper_level_must_agree_on_its_total_rate() -> None:
    e_d, e_p, gamma = 1.0e13, 8.0e14, 2.0e7
    levels = (
        Level("S1/2", 0.0, None, 0.0, 0.0, 2.0, ("test",)),
        Level("D3/2", e_d, 1.0, 0.0, 0.0, 0.8, ("test",)),
        Level("P1/2", e_p, None, 0.0, 0.0, 0.667, ("test",)),
    )
    trs = (
        Transition("S1/2", "P1/2", C_M_PER_S / e_p, gamma, 0.94, "E1", ("test",)),
        Transition("D3/2", "P1/2", C_M_PER_S / (e_p - e_d), 1.5 * gamma, 0.06, "E1", ("test",)),
    )
    with pytest.raises(ValueError, match="disagree on its total decay rate"):
        Species(
            "toy",
            40.0,
            0.0,
            0.0,
            levels,
            trs,
            ("S1/2 mJ=-1/2", "D3/2 mJ=-1/2"),
            "S1/2-P1/2",
            ("D3/2-P1/2",),
            None,
        )


def test_atomic_structure_refuses_a_species_with_no_e1_structure() -> None:
    """An E2-only record answers no Section 4.5.4/4.5.5 question: it builds only when its qubit is the E2 pair itself."""
    levels = (
        Level("S1/2", 0.0, None, 0.0, 0.0, 2.0, ("test",)),
        Level("D5/2", 4.4e14, 0.39, 0.0, 0.0, 1.2, ("test",)),
    )
    trs = (Transition("S1/2", "D5/2", C_M_PER_S / 4.4e14, 1.0 / (TWO_PI * 0.39), 1.0, "E2", ("test",)),)

    def e2_only(qubit: tuple[str, str]) -> Species:
        return Species("e2only", 88.0, 0.0, 0.0, levels, trs, qubit, "S1/2-D5/2", (), "S1/2-D5/2")

    st = AtomicStructure(e2_only(("S1/2 mJ=-1/2", "D5/2 mJ=-1/2")), B_GAUSS, Z_HAT)
    assert st.e1 == {}
    with pytest.raises(KeyError, match="no tabulated E1"):
        st.reduced_element_c_m("S1/2", "D5/2")
    with pytest.raises(ValueError, match="no E1"):
        AtomicStructure(e2_only(("S1/2 mJ=-1/2", "S1/2 mJ=+1/2")), B_GAUSS, Z_HAT)
