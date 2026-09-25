"""Cubic Coulomb couplings from first principles, the three-mode resonance checker and the default-on phase estimate."""

from __future__ import annotations

import dataclasses
import math
from itertools import combinations_with_replacement, permutations

import numpy as np
import pytest

from qutip_trap.control.pulses import Pulse
from qutip_trap.device.presets import secular_trap
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.species import species
from qutip_trap.trap.anharmonic import (
    AnharmonicTerms,
    anharmonic_estimate,
    coulomb_anharmonic_terms,
    coulomb_fourth_derivatives,
    coulomb_third_derivatives,
    dispersive_phase_bound_rad,
    integrated_cubic_phase_rad,
    three_mode_resonances,
)
from qutip_trap.trap.crystal import Crystal, coulomb_hessian_j_per_m2, length_scale_m, solve_crystal
from qutip_trap.units import HBAR_J_S, TWO_PI
from tests.m2_fixtures import two_ion_raman_device

G_RAD_S = TWO_PI * 1.419e3
"""g = eps omega_z for 40Ca+ at omega_z = 2 pi x 2 MHz, eps = x0/(4 l)."""
OMEGA_A_RAD_S = TWO_PI * 2.0e6
DURATION_S = 100e-6


def _marquet_axial_cubic_rad_s(crystal: Crystal) -> dict[tuple[int, int, int], float]:
    """Marquet 2003's axial couplings of an equal-mass chain: 2 eps omega_z D_pqr (mu_p mu_q mu_r)^{-1/4} x multiplicity with
    D_pqr = sum C_mnp' b_m^(p) b_n^(q) b_p'^(r), C = (1/6) d^3 F/du^3 of F = sum_{m<n} 1/|u_m - u_n| and eps = x0/(4 l)."""
    axial = crystal.family("axial")
    n = crystal.n_ions
    mass = float(crystal.masses_kg[0])
    omega_z = axial[0].omega_rad_s
    ell = length_scale_m(mass, omega_z)
    u = (np.asarray(crystal.positions_m, dtype=float) @ crystal.principal_axes)[:, 2] / ell
    c = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            f3 = -6.0 * math.copysign(1.0, u[i] - u[j]) / (u[i] - u[j]) ** 4
            for bits in range(8):
                ions = tuple(j if (bits >> s) & 1 else i for s in range(3))
                c[ions] += (-1.0) ** sum(1 for x in ions if x == j) * f3 / 6.0
    b = np.column_stack([m.eigenvector for m in axial])
    d = np.einsum("mnp,mi,nj,pk->ijk", c, b, b, b)
    mu = np.array([(m.omega_rad_s / omega_z) ** 2 for m in axial])
    eps = math.sqrt(HBAR_J_S / (2.0 * mass * omega_z)) / (4.0 * ell)
    return {
        key: len(set(permutations(key)))
        * 2.0
        * eps
        * omega_z
        * d[key]
        * (mu[key[0]] * mu[key[1]] * mu[key[2]]) ** -0.25
        for key in combinations_with_replacement(range(n), 3)
    }


def test_third_and_fourth_derivatives_against_finite_differences() -> None:
    pos = np.random.default_rng(3).normal(size=(3, 3)) * 3e-6
    v3 = coulomb_third_derivatives(pos)
    v4 = coulomb_fourth_derivatives(pos)
    assert np.allclose(v3, v3.transpose(1, 0, 2)) and np.allclose(v3, v3.transpose(0, 2, 1)), (
        "fully symmetric"
    )
    h = 2e-9
    for c in range(9):
        dp = pos.ravel().copy()
        dm = dp.copy()
        dp[c] += h
        dm[c] -= h
        fd = (
            coulomb_hessian_j_per_m2(dp.reshape(3, 3), np.ones(3))
            - coulomb_hessian_j_per_m2(dm.reshape(3, 3), np.ones(3))
        ) / (2 * h)
        assert np.allclose(v3[:, :, c], fd, rtol=1e-5, atol=1e-5 * np.max(np.abs(v3)))
        fd4 = (coulomb_third_derivatives(dp.reshape(3, 3)) - coulomb_third_derivatives(dm.reshape(3, 3))) / (
            2 * h
        )
        assert np.allclose(v4[:, :, :, c], fd4, rtol=1e-4, atol=1e-4 * np.max(np.abs(v4)))
    # translation invariance: summing any index over all ions of one axis gives zero
    assert abs(np.sum(v3.reshape(3, 3, 3, 3, 3, 3), axis=4)).max() < 1e-9 * np.max(np.abs(v3))


def test_cartesian_route_reproduces_marquet_for_equal_masses_and_com_decouples() -> None:
    ca = species("40Ca+")
    cr = solve_crystal(secular_trap((5e6, 5.5e6, 2e6)), (ca,) * 3)
    terms = coulomb_anharmonic_terms(cr)
    for key, val in _marquet_axial_cubic_rad_s(cr).items():
        if abs(val) > 1e-6:
            assert terms.cubic_rad_s[key] == pytest.approx(val, rel=1e-9)
        else:
            assert abs(terms.cubic_rad_s.get(key, 0.0)) < 1e-6
    com = cr.mode_index("axial", 0)
    for key, val in terms.cubic_rad_s.items():
        if com in key:
            assert abs(val) < 1e-6 * terms.largest_cubic_rad_s(), (
                "D_mn1 = 0: the COM decouples from all mixing"
            )
    stretch, x_rock = cr.mode_index("axial", 1), cr.mode_index("transverse_1", 0)
    assert abs(terms.cubic_rad_s[(stretch, x_rock, x_rock)]) > 0.1 * terms.largest_cubic_rad_s(), (
        "z (x x) terms"
    )
    assert abs(terms.cubic_rad_s.get((x_rock, x_rock, x_rock), 0.0)) < 1e-6, (
        "no transverse-only term for a chain"
    )
    with pytest.raises(ValueError):
        AnharmonicTerms(cubic_rad_s={(2, 1, 0): 1.0})


def test_the_two_ion_stretch_self_coupling_is_marquets_d222() -> None:
    """D_222 = -1.1225 at N = 2: the coefficient is 2 eps omega_z D_222/3^{3/4}, negative because stretching the pair
    softens the Coulomb spring."""
    ca = species("40Ca+")
    cr = solve_crystal(secular_trap((5e6, 5e6, 2e6)), (ca, ca))
    mass = float(cr.masses_kg[0])
    omega_z = TWO_PI * 2e6
    eps = math.sqrt(HBAR_J_S / (2.0 * mass * omega_z)) / (4.0 * length_scale_m(mass, omega_z))
    assert eps == pytest.approx(7.09e-4, rel=2e-3)
    assert coulomb_anharmonic_terms(cr).cubic_rad_s[(1, 1, 1)] == pytest.approx(
        -1.1225 * 2 * eps * omega_z / 3**0.75, rel=1e-4
    )


def test_resonance_checker_finds_a_tuned_three_mode_resonance() -> None:
    """omega_z,stretch = 2 omega_x,rock when omega_x^2 = 7/4 omega_z^2 for two ions (a z x x coupling); none detuned."""
    ca = species("40Ca+")
    cr = solve_crystal(secular_trap((math.sqrt(1.75) * 1.0e6, 1.5e6, 1.0e6)), (ca, ca))
    hits = three_mode_resonances(cr, coulomb_anharmonic_terms(cr), width_hz=1.0)
    stretch, rock = cr.mode_index("axial", 1), cr.mode_index("transverse_1", 0)
    assert any(set(h.modes) == {stretch, rock} and h.sign == +1 for h in hits)
    assert all(abs(h.mismatch_hz) < 1.0 for h in hits)
    quiet = solve_crystal(secular_trap((5e6, 5.5e6, 2e6)), (ca, ca))
    assert three_mode_resonances(quiet, coulomb_anharmonic_terms(quiet), width_hz=1e4) == ()
    with pytest.raises(ValueError):
        dispersive_phase_bound_rad(1.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("mismatch_hz", "phase_rad", "bound_rad"),
    [(1.0e6, 0.0041, 0.0013), (0.3e6, 0.0034, 0.0042), (0.1e6, 0.0032, 0.0127), (0.03e6, 0.0032, 0.0422)],
)
def test_integrated_cubic_phase_against_the_second_order_bound(
    mismatch_hz: float, phase_rad: float, bound_rad: float
) -> None:
    """At g/2pi = 1.419 kHz over 100 us on |1, 0> of the 2:1 model (w_b = 2 w_a - 2 pi Delta) the integrated phase is
    0.0041, 0.0034, 0.0032, 0.0032 rad at Delta = 1, 0.3, 0.1, 0.03 MHz, against g^2 t/Delta = 0.0013 ... 0.0422."""
    got = integrated_cubic_phase_rad(
        G_RAD_S, OMEGA_A_RAD_S, 2.0 * OMEGA_A_RAD_S - TWO_PI * mismatch_hz, DURATION_S
    )
    assert got == pytest.approx(phase_rad, abs=5e-5)
    assert dispersive_phase_bound_rad(G_RAD_S, TWO_PI * mismatch_hz, DURATION_S) == pytest.approx(
        bound_rad, abs=5e-5
    )


def test_the_integrated_phase_is_flat_in_the_mismatch_and_the_resonant_pair_is_larger() -> None:
    """The phase varies by 22 % over a factor 33 in Delta where g^2 t/Delta varies 33x; the resonant |2, 0> <-> |0, 1>
    pair accumulates 0.016 rad at 0.1 MHz."""
    phases = [
        integrated_cubic_phase_rad(G_RAD_S, OMEGA_A_RAD_S, 2.0 * OMEGA_A_RAD_S - TWO_PI * d, DURATION_S)
        for d in (1.0e6, 0.03e6)
    ]
    assert max(phases) / min(phases) < 1.35
    resonant = integrated_cubic_phase_rad(
        G_RAD_S, OMEGA_A_RAD_S, 2.0 * OMEGA_A_RAD_S - TWO_PI * 0.1e6, DURATION_S, state=(2, 0)
    )
    assert resonant == pytest.approx(-0.01589, abs=5e-6)
    with pytest.raises(ValueError, match="does not fit the truncations"):
        integrated_cubic_phase_rad(G_RAD_S, OMEGA_A_RAD_S, OMEGA_A_RAD_S, DURATION_S, state=(9, 0))


def test_the_builder_reports_the_estimate_whether_or_not_the_cubic_term_is_built() -> None:
    """The checker and the estimate reach ``approximations`` for both settings of ``include_anharmonic``;
    ``AnharmonicTerms.resonance_check=False`` silences them and a trap without the record has nothing to check."""
    base = two_ion_raman_device()
    terms = coulomb_anharmonic_terms(base.crystal)
    assert terms.cubic_rad_s and terms.resonance_check is True
    trap = dataclasses.replace(base.trap, anharmonic_terms=terms)
    dev = dataclasses.replace(base, trap=trap)
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    modes = tuple(range(len(dev.crystal.modes)))
    space = HilbertSpace(
        (2, 2), (ModeTruncation(modes[3], 4, (0, 2), 0.2),), None, tuple(m for m in modes if m != 3)
    )
    pulses = [Pulse(square_drive(dd, include_stark=False), 0.0, DURATION_S, "p", ())]
    estimate = anharmonic_estimate(dev.crystal, terms, DURATION_S)
    assert estimate is not None and estimate.phase_rad != 0.0
    assert estimate.coupling_rad_s != 0.0 and estimate.mismatch_hz != 0.0
    assert estimate.bound_rad == pytest.approx(
        estimate.coupling_rad_s**2 * DURATION_S / abs(TWO_PI * estimate.mismatch_hz), rel=1e-12
    )
    for include in (False, True):
        built = build_hamiltonian(dev, pulses, space, options=BuilderOptions(include_anharmonic=include))
        notes = [a for a in built.approximations if "anharmonic resonance check" in a]
        assert len(notes) == 1, f"include_anharmonic={include}: {built.approximations}"
        assert "integrated cubic phase" in notes[0] and "g^2 t/Delta bound" in notes[0]
    silent = dataclasses.replace(
        dev,
        trap=dataclasses.replace(trap, anharmonic_terms=dataclasses.replace(terms, resonance_check=False)),
    )
    built = build_hamiltonian(silent, pulses, space, options=BuilderOptions())
    assert not any("anharmonic resonance check" in a for a in built.approximations)
    built = build_hamiltonian(base, pulses, space, options=BuilderOptions())
    assert not any("anharmonic resonance check" in a for a in built.approximations)


def test_the_estimate_reports_a_tuned_resonance_as_linear_in_time() -> None:
    ca = species("40Ca+")
    tuned = solve_crystal(secular_trap((math.sqrt(1.75) * 1.0e6, 1.5e6, 1.0e6)), (ca, ca))
    est = anharmonic_estimate(tuned, coulomb_anharmonic_terms(tuned), DURATION_S)
    assert est is not None and est.resonances, "the tuned triple is inside the coupling width"
    assert abs(est.resonances[0].mismatch_hz) < est.width_hz
    assert "LINEAR in time" in est.summary()
    quiet = solve_crystal(secular_trap((5e6, 5.5e6, 2e6)), (ca, ca))
    quiet_est = anharmonic_estimate(quiet, coulomb_anharmonic_terms(quiet), DURATION_S)
    assert quiet_est is not None and quiet_est.resonances == ()
    assert "no mode triple within" in quiet_est.summary()
