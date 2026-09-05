"""Electric-quadrupole coupling: the Section 9.14 rows (Run 4Q) and the audit row 4.5-8."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from qutip_trap.api import Beam, Field
from qutip_trap.species import species
from qutip_trap.species.quadrupole import (
    B_TENSORS,
    C_ALPHA_M_PER_S,
    C_TENSORS,
    e2_over_e1_amplitude_ratio,
    geometric_factor,
    geometric_factor_closed_form,
    geometric_factors,
    lambda_3j,
    rabi_frequency_e2_rad_s,
    racah_c2,
    reduced_element_a0_squared,
    reduced_element_from_lifetime_m2,
    stretched_closed_form_rad_s,
)
from qutip_trap.units import A_0_M, TWO_PI

HALF = Fraction(1, 2)
FIVE_HALF = Fraction(5, 2)
B_Z = (0.0, 0.0, 1.0)


def _geometry(phi_deg: float, gamma_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Roos's frame: B = z, k = (sin phi, 0, cos phi), eps = (cos gamma cos phi, sin gamma, -cos gamma sin phi)."""
    p, g = math.radians(phi_deg), math.radians(gamma_deg)
    k = np.array([math.sin(p), 0.0, math.cos(p)])
    eps = np.array([math.cos(g) * math.cos(p), math.sin(g), -math.cos(g) * math.sin(p)])
    return eps, k


def test_c_alpha_and_prefactor_identity() -> None:
    assert C_ALPHA_M_PER_S == pytest.approx(2.187691e6, rel=1e-6)


def test_rank2_tensors_normalization_symmetry_and_generation() -> None:
    """sum_ij |c^(q)|^2 = 2/3 for all q; symmetric and traceless; c^(q) = (-1)^q c^(-q)*; b^(q)_ij r_i r_j = r^2 C^(2)_q."""
    rng = np.random.default_rng(3)
    for q, c in C_TENSORS.items():
        assert float(np.sum(np.abs(c) ** 2)) == pytest.approx(2.0 / 3.0, abs=1e-15)
        assert np.allclose(c, c.T, atol=1e-15) and abs(np.trace(c)) < 1e-15
        assert np.allclose(c, (-1) ** q * np.conj(C_TENSORS[-q]), atol=1e-15)
    for _ in range(20):
        r = rng.normal(size=3)
        for q, b in B_TENSORS.items():
            assert complex(r @ b @ r) == pytest.approx(racah_c2(q, r), abs=2e-16 * float(r @ r))
    # 5x5 overlap (2/3) x identity
    flat = np.array([C_TENSORS[q].ravel() for q in (-2, -1, 0, 1, 2)])
    assert np.allclose(flat.conj() @ flat.T, (2.0 / 3.0) * np.eye(5), atol=1e-15)


@pytest.mark.parametrize(
    "phi_deg,gamma_deg", [(90, 90), (45, 0), (0, 30), (20, 20), (70, 20), (33, 71), (90, 0)]
)
def test_geometric_factors_against_the_closed_forms_and_the_sum_rule(
    phi_deg: float, gamma_deg: float
) -> None:
    eps, k = _geometry(phi_deg, gamma_deg)
    g = geometric_factors(eps, k, B_Z)
    for q in (-2, -1, 0, 1, 2):
        assert abs(g[q]) == pytest.approx(
            geometric_factor_closed_form(q, math.radians(phi_deg), math.radians(gamma_deg)), abs=1e-14
        )
    assert abs(g[0]) ** 2 + 2 * abs(g[1]) ** 2 + 2 * abs(g[2]) ** 2 == pytest.approx(1.0 / 3.0, abs=1e-14)
    assert abs(g[1]) == pytest.approx(abs(g[-1]), abs=1e-15) and abs(g[2]) == pytest.approx(
        abs(g[-2]), abs=1e-15
    )


def test_geometric_factor_special_values_of_section_9_14() -> None:
    eps, k = _geometry(90, 90)
    g = geometric_factors(eps, k, B_Z)
    assert abs(g[0]) < 1e-15 and abs(g[1]) < 1e-15
    assert abs(g[2]) == pytest.approx(0.4082483, abs=1e-7)
    eps, k = _geometry(45, 0)
    g = geometric_factors(eps, k, B_Z)
    assert (
        abs(g[0]) == pytest.approx(0.5, abs=1e-12)
        and abs(g[1]) < 1e-15
        and abs(g[2]) == pytest.approx(0.2041241, abs=1e-7)
    )
    # at phi = 0 only |Delta m| = 1 is driven; at phi = 90 Delta m = 0 is never driven
    for gamma_deg in (0, 30, 60, 90):
        eps, k = _geometry(0, gamma_deg)
        g = geometric_factors(eps, k, B_Z)
        assert (
            abs(g[0]) < 1e-15
            and abs(g[2]) < 1e-15
            and abs(g[1]) == pytest.approx(1 / math.sqrt(6), abs=1e-14)
        )
        eps, k = _geometry(90, gamma_deg)
        assert abs(geometric_factor(0, eps, k, B_Z)) < 1e-15


def test_gamma_definition_negative_control() -> None:
    """gamma = arcsin|eps . y| for every phi; the literal arccos|cos gamma sin phi| reading gives 71.25, 48.36, 27.99, 20.00 degrees at gamma = 20."""
    for phi_deg, literal in ((20, 71.25), (45, 48.36), (70, 27.99), (90, 20.00)):
        eps, _k = _geometry(phi_deg, 20)
        assert math.degrees(math.asin(abs(eps[1]))) == pytest.approx(20.0, abs=1e-9)
        assert math.degrees(
            math.acos(abs(math.cos(math.radians(20)) * math.sin(math.radians(phi_deg))))
        ) == pytest.approx(literal, abs=0.01)


def test_3j_table_mask_and_peak_coupling() -> None:
    """|Lambda| for m = -1/2: 0.408248, 0.365148, 0.316228, 0.258199, 0.182574, 0 for m' = -5/2 .. +5/2; sum_{m,q}|Lambda|^2 = 1/6 at fixed m';
    10 of 12 pairs allowed; peak |Lambda| g = 1/6 on the stretched line at (90, 90), runner-up 0.158114, best |Delta m| = 1: 0.149071."""
    table = [abs(lambda_3j(HALF, -HALF, FIVE_HALF, Fraction(2 * k - 5, 2))) for k in range(6)]
    assert table == pytest.approx([0.408248, 0.365148, 0.316228, 0.258199, 0.182574, 0.0], abs=1e-6)
    for mp in (Fraction(2 * k - 5, 2) for k in range(6)):
        assert sum(lambda_3j(HALF, m, FIVE_HALF, mp) ** 2 for m in (-HALF, HALF)) == pytest.approx(
            1.0 / 6.0, abs=1e-14
        )
    allowed = [
        (m, mp)
        for m in (-HALF, HALF)
        for mp in (Fraction(2 * k - 5, 2) for k in range(6))
        if lambda_3j(HALF, m, FIVE_HALF, mp) != 0.0
    ]
    assert len(allowed) == 10
    eps, k = _geometry(90, 90)
    g = geometric_factors(eps, k, B_Z)
    peak = abs(lambda_3j(HALF, -HALF, FIVE_HALF, -FIVE_HALF)) * abs(g[2])
    assert peak == pytest.approx(1.0 / 6.0, abs=1e-12)
    eps, k = _geometry(45, 0)
    g = geometric_factors(eps, k, B_Z)
    assert abs(lambda_3j(HALF, -HALF, FIVE_HALF, -HALF)) * abs(g[0]) == pytest.approx(0.158114, abs=1e-6)
    eps, k = _geometry(0, 0)
    g = geometric_factors(eps, k, B_Z)
    assert abs(lambda_3j(HALF, -HALF, FIVE_HALF, -Fraction(3, 2))) * abs(g[1]) == pytest.approx(
        0.149071, abs=1e-6
    )


def test_reduced_elements_of_section_9_14() -> None:
    """40Ca+: 2.724e-20 m^2 = 9.73 a0^2 (tau 1.168 s, vacuum lambda, 2j'+1 = 6); 10.29 with Roos's 1.045 s; 5.7 with 2j+1 = 2 (wrong by sqrt 3);
    88Sr+: 13.81 a0^2 at 0.3908 s and 14.70 with the superseded 0.345 s."""
    assert reduced_element_from_lifetime_m2(729.347e-9, 1.0 / 1.168, FIVE_HALF) == pytest.approx(
        2.724e-20, rel=1e-3
    )
    assert reduced_element_a0_squared(729.347e-9, 1.0 / 1.168, FIVE_HALF) == pytest.approx(9.73, abs=5e-3)
    assert reduced_element_a0_squared(729.347e-9, 1.0 / 1.045, FIVE_HALF) == pytest.approx(10.29, abs=5e-3)
    assert reduced_element_a0_squared(729.347e-9, 1.0 / 1.168, HALF) == pytest.approx(
        9.73 / math.sqrt(3.0), abs=1e-2
    )
    assert reduced_element_a0_squared(674.025591e-9, 1.0 / 0.3908, FIVE_HALF) == pytest.approx(
        13.81, abs=5e-3
    )
    assert reduced_element_a0_squared(674.025591e-9, 1.0 / 0.345, FIVE_HALF) == pytest.approx(14.70, abs=5e-3)


def test_stretched_closed_form_equals_the_pipeline_and_roos_worked_numbers() -> None:
    """(e E0 k/(2 hbar)) x element x 1/6 at (90, 90), (m, m') = (-1/2, -5/2) equals (e E0/hbar) sqrt(5 lambda^3 A/(64 pi^3 c alpha)) to 1e-12.

    Roos's worked example: E0 = 2.3e5 V/m, lambda_air 729.147 nm, tau 1.045 s -> 1.1495 MHz (printed 1.2). Section 9.14 then prints
    1.088 MHz "with 2.309e5 V/m, lambda_vac 729.347 nm, tau 1.168 s"; the closed form gives 1.088 MHz and the 460 ns pi time only at
    E0 = 2.3e5 V/m, and 1.0920 MHz at 2.309e5 V/m, so the row pairs its result with the other field value (a 0.4% inconsistency in
    the plan, recorded here rather than absorbed into a tolerance).
    """
    eps, k = _geometry(90, 90)
    cases = (
        (2.3e5, 729.147e-9, 1.045, 1.1495),
        (2.3e5, 729.347e-9, 1.168, 1.0877),
        (2.309e5, 729.347e-9, 1.168, 1.0920),
    )
    for e0, lam, tau, ref_mhz in cases:
        a = 1.0 / tau
        red = reduced_element_from_lifetime_m2(lam, a, FIVE_HALF)
        omega = rabi_frequency_e2_rad_s(e0, lam, red, HALF, -HALF, FIVE_HALF, -FIVE_HALF, eps, k, B_Z)
        assert omega == pytest.approx(stretched_closed_form_rad_s(e0, lam, a), rel=1e-12)
        assert omega / TWO_PI * 1e-6 == pytest.approx(ref_mhz, abs=5e-4)
    omega = stretched_closed_form_rad_s(2.3e5, 729.347e-9, 1.0 / 1.168)
    assert math.pi / omega * 1e9 == pytest.approx(460.0, abs=1.0)
    assert 90.0 / (12.0**2 * 8.0) == 5.0 / 64.0


def test_field_from_power_round_trip() -> None:
    """100 mW in a 30 um waist: I0 = 7.074e7 W/m^2, E0 = 2.3087e5 V/m; I = (1/2) c eps0 E0^2."""
    from qutip_trap.species.dipole import field_amplitude_v_per_m
    from qutip_trap.units import C_M_PER_S, EPSILON_0_F_PER_M

    i0 = 2.0 * 0.1 / (math.pi * (30e-6) ** 2)
    assert i0 == pytest.approx(7.074e7, rel=1e-4)
    e0 = field_amplitude_v_per_m(i0)
    assert e0 == pytest.approx(2.3087e5, rel=1e-4)
    assert 0.5 * C_M_PER_S * EPSILON_0_F_PER_M * e0**2 == pytest.approx(i0, rel=1e-12)


def test_e2_over_e1_suppression() -> None:
    assert e2_over_e1_amplitude_ratio(729.347e-9) == pytest.approx(2.28e-4, rel=3e-3)
    assert e2_over_e1_amplitude_ratio(674.026e-9) == pytest.approx(2.47e-4, rel=3e-3)


def test_sigma_plus_along_b_drives_delta_m_plus_one_with_the_c_tensors() -> None:
    """Audit row 4.5-8: sigma+ along B requires |g^(-1)| = 1/sqrt3 and |g^(+1)| = 0, driving Delta m = +1 with q = m - m'."""
    eps = np.array([-1.0, -1.0j, 0.0]) / math.sqrt(2.0)
    g = geometric_factors(eps, np.array([0.0, 0.0, 1.0]), B_Z)
    assert abs(g[-1]) == pytest.approx(1.0 / math.sqrt(3.0), abs=1e-14)
    assert abs(g[1]) < 1e-15 and abs(g[0]) < 1e-15 and abs(g[2]) < 1e-15 and abs(g[-2]) < 1e-15
    # q = m - m' = -1 means m' = m + 1
    assert lambda_3j(HALF, -HALF, FIVE_HALF, HALF) != 0.0


def test_species_api_e2_rabi_frequency_and_lamb_dicke_input() -> None:
    """Species.rabi_frequency_hz on the 40Ca+ 729 nm line: 100 mW in a 30 um waist is E0 = 2.3087e5 V/m, so the stretched line at
    (90, 90) degrees gives 1.0918 MHz with the table's tau = 1.168 s and vacuum wavelength (see the E0 pairing note above)."""
    ca = species("40Ca+")
    eps, k = _geometry(90, 90)
    beam = Beam(
        ca.transition("S1/2-D5/2").wavelength_vac_m, tuple(k), tuple(eps), 30e-6, 0.1, (0.0, 0.0, 0.0)
    )
    field = Field(B_gauss=4.0, direction=B_Z, noise=None)
    omega = ca.rabi_frequency_hz("S1/2 mJ=-1/2", "D5/2 mJ=-5/2", beam, field)
    assert abs(omega) * 1e-6 == pytest.approx(1.0918, abs=1e-3)
    assert ca.rabi_frequency_hz("S1/2 mJ=-1/2", "D5/2 mJ=5/2", beam, field) == 0.0, (
        "|Delta m| = 3 is forbidden"
    )
    # single 40Ca+ at 1 MHz: x0 = 11.246 nm, k x0 = 0.0969 along the beam (Section 9.14)
    from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S

    x0 = math.sqrt(HBAR_J_S / (2.0 * ca.mass_u * ATOMIC_MASS_KG * TWO_PI * 1e6))
    assert x0 * 1e9 == pytest.approx(11.246, abs=2e-3)
    assert TWO_PI / beam.wavelength_m * x0 == pytest.approx(0.0969, abs=2e-4)
    assert A_0_M > 0
