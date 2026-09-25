"""Electric-quadrupole coupling and its ac Stark shift (PLAN.md Section 4.5.7)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.constants import physical_constants

from qutip_trap.device.model import Field
from qutip_trap.light.beams import Beam
from qutip_trap.light.raman import derive_optical_drive, quadrupole_stark_shift_hz
from qutip_trap.species import species
from qutip_trap.species.dipole import field_amplitude_v_per_m
from qutip_trap.species.quadrupole import (
    B_TENSORS,
    C_ALPHA_M_PER_S,
    C_TENSORS,
    e2_stark_shift_rad_s,
    geometric_factor,
    lambda_3j,
    rabi_frequency_e2_rad_s,
    reduced_element_from_lifetime_m2,
)
from qutip_trap.units import ATOMIC_MASS_KG, C_M_PER_S, E_C, EPSILON_0_F_PER_M, HBAR_J_S, TWO_PI
from tests.fixtures import HALF, ca_light_shift_device, single_ion_raman_device

FIVE_HALF = Fraction(5, 2)
B_Z = (0.0, 0.0, 1.0)


def _geometry(phi_deg: float, gamma_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Roos's frame: B = z, k = (sin phi, 0, cos phi), eps = (cos gamma cos phi, sin gamma, -cos gamma sin phi)."""
    p, g = math.radians(phi_deg), math.radians(gamma_deg)
    k = np.array([math.sin(p), 0.0, math.cos(p)])
    eps = np.array([math.cos(g) * math.cos(p), math.sin(g), -math.cos(g) * math.sin(p)])
    return eps, k


def _factors(eps: np.ndarray, k: np.ndarray) -> dict[int, complex]:
    return {q: geometric_factor(q, eps, k, B_Z) for q in (-2, -1, 0, 1, 2)}


def _racah_c2(q: int, r: np.ndarray) -> complex:
    """r^2 C^(2)_q = sqrt(4 pi/5) r^2 Y_{2q} in Cartesian form."""
    x, y, z = (float(v) for v in r)
    if q == 0:
        return complex((3.0 * z * z - (x * x + y * y + z * z)) / 2.0)
    if abs(q) == 1:
        return complex(-q * math.sqrt(1.5) * z * (x + 1j * q * y))
    return complex(math.sqrt(3.0 / 8.0) * (x + 1j * (q // 2) * y) ** 2)


def _roos_closed_form(q: int, phi_rad: float, gamma_rad: float) -> float:
    """|g^(q)| in Roos's frame (Roos 2000): (1/2)|cos gamma sin 2phi|, (1/sqrt 6)|cos gamma cos 2phi + i sin gamma cos phi|,
    (1/sqrt 6)|(1/2) cos gamma sin 2phi + i sin gamma sin phi| for q = 0, +-1, +-2."""
    cg, sg = math.cos(gamma_rad), math.sin(gamma_rad)
    if q == 0:
        return 0.5 * abs(cg * math.sin(2.0 * phi_rad))
    if abs(q) == 1:
        return abs(cg * math.cos(2.0 * phi_rad) + 1j * sg * math.cos(phi_rad)) / math.sqrt(6.0)
    return abs(0.5 * cg * math.sin(2.0 * phi_rad) + 1j * sg * math.sin(phi_rad)) / math.sqrt(6.0)


def _stretched_closed_form_rad_s(e0_v_per_m: float, wavelength_vac_m: float, rate_per_s: float) -> float:
    """Omega = (e E_0/hbar) sqrt(5 lambda^3 A/(64 pi^3 c alpha)), the |Delta m| = 2 line at (90, 90) degrees (James Eqs.
    5.11, 5.13; Roos p. 31)."""
    return (
        E_C
        * e0_v_per_m
        / HBAR_J_S
        * math.sqrt(5.0 * wavelength_vac_m**3 * rate_per_s / (64.0 * math.pi**3 * C_ALPHA_M_PER_S))
    )


def test_c_alpha() -> None:
    assert C_ALPHA_M_PER_S == pytest.approx(2.187691e6, rel=1e-6)


def test_rank2_tensors_normalization_symmetry_and_generation() -> None:
    """The rank-2 tensors are symmetric, traceless and orthogonal with sum_ij |c^(q)|^2 = 2/3 and c^(q) = (-1)^q c^(-q)*
    (1e-15), and b^(q)_ij r_i r_j = r^2 C^(2)_q."""
    rng = np.random.default_rng(3)
    for q, c in C_TENSORS.items():
        assert float(np.sum(np.abs(c) ** 2)) == pytest.approx(2.0 / 3.0, abs=1e-15)
        assert np.allclose(c, c.T, atol=1e-15) and abs(np.trace(c)) < 1e-15
        assert np.allclose(c, (-1) ** q * np.conj(C_TENSORS[-q]), atol=1e-15)
    for _ in range(20):
        r = rng.normal(size=3)
        for q, b in B_TENSORS.items():
            assert complex(r @ b @ r) == pytest.approx(_racah_c2(q, r), abs=2e-16 * float(r @ r))
    flat = np.array([C_TENSORS[q].ravel() for q in (-2, -1, 0, 1, 2)])
    assert np.allclose(flat.conj() @ flat.T, (2.0 / 3.0) * np.eye(5), atol=1e-15)


@pytest.mark.parametrize(
    "phi_deg,gamma_deg", [(90, 90), (45, 0), (0, 30), (20, 20), (70, 20), (33, 71), (90, 0)]
)
def test_geometric_factors_against_the_closed_forms_and_the_sum_rule(
    phi_deg: float, gamma_deg: float
) -> None:
    g = _factors(*_geometry(phi_deg, gamma_deg))
    for q in (-2, -1, 0, 1, 2):
        assert abs(g[q]) == pytest.approx(
            _roos_closed_form(q, math.radians(phi_deg), math.radians(gamma_deg)), abs=1e-14
        )
    assert abs(g[0]) ** 2 + 2 * abs(g[1]) ** 2 + 2 * abs(g[2]) ** 2 == pytest.approx(1.0 / 3.0, abs=1e-14)
    assert abs(g[1]) == pytest.approx(abs(g[-1]), abs=1e-15) and abs(g[2]) == pytest.approx(
        abs(g[-2]), abs=1e-15
    )


def test_geometric_factor_special_values() -> None:
    g = _factors(*_geometry(90, 90))
    assert abs(g[0]) < 1e-15 and abs(g[1]) < 1e-15
    assert abs(g[2]) == pytest.approx(0.4082483, abs=1e-7)
    g = _factors(*_geometry(45, 0))
    assert abs(g[0]) == pytest.approx(0.5, abs=1e-12) and abs(g[1]) < 1e-15
    assert abs(g[2]) == pytest.approx(0.2041241, abs=1e-7)
    # at phi = 0 only |Delta m| = 1 is driven; at phi = 90 Delta m = 0 is never driven
    for gamma_deg in (0, 30, 60, 90):
        g = _factors(*_geometry(0, gamma_deg))
        assert abs(g[0]) < 1e-15 and abs(g[2]) < 1e-15
        assert abs(g[1]) == pytest.approx(1 / math.sqrt(6), abs=1e-14)
        eps, k = _geometry(90, gamma_deg)
        assert abs(geometric_factor(0, eps, k, B_Z)) < 1e-15


def test_3j_table_mask_and_peak_coupling() -> None:
    """|Lambda| for m = -1/2 is 0.408248 ... 0 over m' = -5/2 .. +5/2 (1e-6) with sum 1/6 at fixed m' and 10 of 12 pairs
    allowed, and the peak |Lambda g| is 1/6 at (90, 90), the runner-up 0.158114 and the best |Delta m| = 1 0.149071."""
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
    peak = abs(lambda_3j(HALF, -HALF, FIVE_HALF, -FIVE_HALF)) * abs(_factors(*_geometry(90, 90))[2])
    assert peak == pytest.approx(1.0 / 6.0, abs=1e-12)
    runner_up = abs(lambda_3j(HALF, -HALF, FIVE_HALF, -HALF)) * abs(_factors(*_geometry(45, 0))[0])
    assert runner_up == pytest.approx(0.158114, abs=1e-6)
    delta_m1 = abs(lambda_3j(HALF, -HALF, FIVE_HALF, -Fraction(3, 2))) * abs(_factors(*_geometry(0, 0))[1])
    assert delta_m1 == pytest.approx(0.149071, abs=1e-6)


def test_reduced_elements() -> None:
    """The E2 reduced elements are 9.73 a0^2 = 2.724e-20 m^2 for 40Ca+ at 1.168 s (10.29 at Roos's 1.045 s, 9.73/sqrt 3
    with 2j + 1 = 2) and 13.81 and 14.70 a0^2 for 88Sr+ at 0.3908 and 0.345 s, to 5e-3 a0^2."""
    a0_sq = physical_constants["Bohr radius"][0] ** 2
    assert reduced_element_from_lifetime_m2(729.347e-9, 1.0 / 1.168, FIVE_HALF) == pytest.approx(
        2.724e-20, rel=1e-3
    )
    for lam, tau, j_up, ref in (
        (729.347e-9, 1.168, FIVE_HALF, 9.73),
        (729.347e-9, 1.045, FIVE_HALF, 10.29),
        (674.025591e-9, 0.3908, FIVE_HALF, 13.81),
        (674.025591e-9, 0.345, FIVE_HALF, 14.70),
    ):
        assert reduced_element_from_lifetime_m2(lam, 1.0 / tau, j_up) / a0_sq == pytest.approx(ref, abs=5e-3)
    wrong = reduced_element_from_lifetime_m2(729.347e-9, 1.0 / 1.168, HALF) / a0_sq
    assert wrong == pytest.approx(9.73 / math.sqrt(3.0), abs=1e-2)


def test_stretched_closed_form_equals_the_pipeline_and_roos_worked_numbers() -> None:
    """The stretched-line Rabi frequency is James's closed form to 1e-12: 1.1495 MHz for Roos's air-wavelength example
    and 1.0877 and 1.0920 MHz at 2.3e5 and 2.309e5 V/m with the vacuum wavelength (5e-4), a 460 ns pi time."""
    eps, k = _geometry(90, 90)
    for e0, lam, tau, ref_mhz in (
        (2.3e5, 729.147e-9, 1.045, 1.1495),
        (2.3e5, 729.347e-9, 1.168, 1.0877),
        (2.309e5, 729.347e-9, 1.168, 1.0920),
    ):
        red = reduced_element_from_lifetime_m2(lam, 1.0 / tau, FIVE_HALF)
        omega = rabi_frequency_e2_rad_s(e0, lam, red, HALF, -HALF, FIVE_HALF, -FIVE_HALF, eps, k, B_Z)
        assert omega == pytest.approx(_stretched_closed_form_rad_s(e0, lam, 1.0 / tau), rel=1e-12)
        assert omega / TWO_PI * 1e-6 == pytest.approx(ref_mhz, abs=5e-4)
    omega = _stretched_closed_form_rad_s(2.3e5, 729.347e-9, 1.0 / 1.168)
    assert math.pi / omega * 1e9 == pytest.approx(460.0, abs=1.0)


def test_field_from_power_round_trip() -> None:
    """100 mW in a 30 um waist gives E0 = 2.3087e5 V/m (1e-4), and I0 = (1/2) c eps0 E0^2 (1e-12)."""
    i0 = 2.0 * 0.1 / (math.pi * (30e-6) ** 2)
    e0 = field_amplitude_v_per_m(i0)
    assert e0 == pytest.approx(2.3087e5, rel=1e-4)
    assert 0.5 * C_M_PER_S * EPSILON_0_F_PER_M * e0**2 == pytest.approx(i0, rel=1e-12)


def test_sigma_plus_along_b_drives_delta_m_plus_one_with_the_c_tensors() -> None:
    """sigma+ light along B has |g^(-1)| = 1/sqrt3 (1e-14) and every other factor zero, driving Delta m = +1."""
    g = _factors(np.array([-1.0, -1.0j, 0.0]) / math.sqrt(2.0), np.array([0.0, 0.0, 1.0]))
    assert abs(g[-1]) == pytest.approx(1.0 / math.sqrt(3.0), abs=1e-14)
    assert abs(g[1]) < 1e-15 and abs(g[0]) < 1e-15 and abs(g[2]) < 1e-15 and abs(g[-2]) < 1e-15
    assert lambda_3j(HALF, -HALF, FIVE_HALF, HALF) != 0.0


def test_species_api_e2_rabi_frequency_and_lamb_dicke_input() -> None:
    """Species.rabi_frequency_hz gives 1.0918 MHz (1e-3) for 100 mW in 30 um on the 40Ca+ stretched line at (90, 90) and
    zero for |Delta m| = 3, with x0 = 11.246 nm and k x0 = 0.0969 at 1 MHz."""
    ca = species("40Ca+")
    eps, k = _geometry(90, 90)
    beam = Beam(
        ca.transition("S1/2-D5/2").wavelength_vac_m, tuple(k), tuple(eps), 30e-6, 0.1, (0.0, 0.0, 0.0)
    )
    field = Field(B_gauss=4.0, direction=B_Z)
    omega = ca.rabi_frequency_hz("S1/2 mJ=-1/2", "D5/2 mJ=-5/2", beam, field)
    assert abs(omega) * 1e-6 == pytest.approx(1.0918, abs=1e-3)
    assert ca.rabi_frequency_hz("S1/2 mJ=-1/2", "D5/2 mJ=5/2", beam, field) == 0.0
    x0 = math.sqrt(HBAR_J_S / (2.0 * ca.mass_u * ATOMIC_MASS_KG * TWO_PI * 1e6))
    assert x0 * 1e9 == pytest.approx(11.246, abs=2e-3)
    assert TWO_PI / beam.wavelength_m * x0 == pytest.approx(0.0969, abs=2e-4)


# ---- the E2 ac Stark shift ------------------------------------------------------------------------------------------


def _toy_e2_table(zeeman_lower_hz: float, zeeman_upper_hz: float, omega: float) -> tuple[dict, dict, dict]:  # type: ignore[type-arg]
    """Equal couplings on every allowed component, with linear Zeeman ladders on both manifolds."""
    lower = {m: float(m) * zeeman_lower_hz for m in (-HALF, HALF)}
    upper = {Fraction(k, 2): float(Fraction(k, 2)) * zeeman_upper_hz for k in (-5, -3, -1, 1, 3, 5)}
    couplings = {
        (m, mp): (omega if lambda_3j(HALF, m, FIVE_HALF, mp) != 0.0 else 0.0) for m in lower for mp in upper
    }
    return couplings, lower, upper


def test_the_e2_stark_shift_is_second_order_in_the_off_resonant_components() -> None:
    """Doubling every coupling quadruples the E2 Stark shift and halving the Zeeman span doubles it (1e-12)."""
    couplings, lower, upper = _toy_e2_table(1e6, 0.8e6, TWO_PI * 1e5)
    base = e2_stark_shift_rad_s(couplings, lower, upper, -HALF, -HALF)
    assert base != 0.0
    doubled = {key: 2.0 * v for key, v in couplings.items()}
    assert e2_stark_shift_rad_s(doubled, lower, upper, -HALF, -HALF) == pytest.approx(4.0 * base, rel=1e-12)
    narrow_l = {m: 0.5 * v for m, v in lower.items()}
    narrow_u = {m: 0.5 * v for m, v in upper.items()}
    assert e2_stark_shift_rad_s(couplings, narrow_l, narrow_u, -HALF, -HALF) == pytest.approx(
        2.0 * base, rel=1e-12
    )


def test_only_the_components_sharing_a_level_with_the_driven_one_contribute() -> None:
    couplings, lower, upper = _toy_e2_table(1e6, 0.8e6, TWO_PI * 1e5)
    base = e2_stark_shift_rad_s(couplings, lower, upper, -HALF, -HALF)
    disjoint = dict(couplings)
    disjoint[(HALF, Fraction(3, 2))] = 0.0  # shares neither level
    assert e2_stark_shift_rad_s(disjoint, lower, upper, -HALF, -HALF) == pytest.approx(base, rel=1e-15)
    shared = dict(couplings)
    shared[(-HALF, Fraction(-3, 2))] = 0.0  # shares |S, -1/2>
    assert e2_stark_shift_rad_s(shared, lower, upper, -HALF, -HALF) != pytest.approx(base, rel=1e-9)


def test_the_e2_stark_shift_refuses_a_degenerate_component() -> None:
    """A component degenerate with the driven one raises instead of returning infinity, and an undriven component is
    refused."""
    couplings, lower, upper = _toy_e2_table(0.0, 0.0, TWO_PI * 1e5)
    with pytest.raises(ZeroDivisionError, match="degenerate"):
        e2_stark_shift_rad_s(couplings, lower, upper, -HALF, -HALF)
    with pytest.raises(KeyError, match="driven component"):
        e2_stark_shift_rad_s({}, lower, upper, -HALF, -HALF)


def test_the_derived_optical_e2_drive_carries_the_stark_shift() -> None:
    """The 40Ca+ 729 nm drive's Stark shift is the quadrupole sum (1e-12), below the carrier Rabi frequency."""
    dev = ca_light_shift_device()
    assert dev.crystal.species[0].name == "40Ca+"
    e2_beam = next(i for i, b in enumerate(dev.beams) if 7.0e-7 < b.wavelength_m < 7.5e-7)
    dd = derive_optical_drive(dev, 0, e2_beam, scattering=False)
    assert dd.kind == "optical_E2"
    assert dd.stark_shift_hz != 0.0
    assert dd.stark_shift_hz == pytest.approx(quadrupole_stark_shift_hz(dev, 0, e2_beam), rel=1e-12)
    assert "conv.e2_ac_stark_shift" in dd.provenance
    assert abs(dd.stark_shift_hz) < dd.carrier_rabi_hz


def test_a_hyperfine_e2_stark_shift_raises() -> None:
    """The E2 Stark shift is refused for a species with nuclear spin (Section 4.5.7 covers I = 0)."""
    dev = single_ion_raman_device()
    assert dev.crystal.species[0].nuclear_spin != 0.0
    with pytest.raises(NotImplementedError, match="I = 0"):
        quadrupole_stark_shift_hz(dev, 0, 0)
