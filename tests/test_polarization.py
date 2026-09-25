"""Polarization in the atomic frame (PLAN.md Section 4.5.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.species.polarization import (
    atomic_frame,
    linear_polarization,
    spherical_basis,
    spherical_components,
    to_atomic_frame,
)

X, Y, Z = np.eye(3)
SIGMA_PLUS = -(X + 1j * Y) / math.sqrt(2.0)
SIGMA_MINUS = (X - 1j * Y) / math.sqrt(2.0)


# ---- the four statements of Section 4.5.3 ---------------------------------------------------------------


def test_circular_light_along_b_is_pure_sigma() -> None:
    """k along B with eps = -(x + iy)/sqrt2 gives (|eps_-1|^2, |eps_0|^2, |eps_+1|^2) = (0, 0, 1), and the
    opposite handedness (1, 0, 0). A beam along B carries NO pi component."""
    plus = np.abs(spherical_components(SIGMA_PLUS, Z)) ** 2
    minus = np.abs(spherical_components(SIGMA_MINUS, Z)) ** 2
    assert plus == pytest.approx([0.0, 0.0, 1.0], abs=1e-15)
    assert minus == pytest.approx([1.0, 0.0, 0.0], abs=1e-15)


def test_linear_light_along_b_is_half_and_half_with_no_pi_component() -> None:
    """k along B with a LINEAR eps gives (1/2, 0, 1/2): equal sigma+ and sigma-, still no pi."""
    for eps in (X, Y, (X + Y) / math.sqrt(2.0)):
        w = np.abs(spherical_components(eps, Z)) ** 2
        assert w == pytest.approx([0.5, 0.0, 0.5], abs=1e-15)


def test_light_polarized_along_b_is_pure_pi() -> None:
    """k perpendicular to B with eps along B gives (0, 1, 0): eps = B_hat is pure pi whatever k is."""
    for b_hat in (X, Y, Z, np.array([0.6, 0.0, 0.8])):
        w = np.abs(spherical_components(b_hat, b_hat)) ** 2
        assert w == pytest.approx([0.0, 1.0, 0.0], abs=1e-14)
    # and the ``linear_polarization`` helper at angle 0 reproduces it for any transverse k
    eps = linear_polarization(X, 0.0, Z)
    assert np.abs(spherical_components(eps, Z)) ** 2 == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


def test_light_polarized_perpendicular_to_b_is_half_and_half() -> None:
    """k perpendicular to B with eps perpendicular to B gives (1/2, 0, 1/2)."""
    w = np.abs(spherical_components(Y, X)) ** 2
    assert w == pytest.approx([0.5, 0.0, 0.5], abs=1e-15)
    eps = linear_polarization(X, math.pi / 2.0, Z)
    assert np.abs(spherical_components(eps, Z)) ** 2 == pytest.approx([0.5, 0.0, 0.5], abs=1e-12)


@pytest.mark.parametrize("theta_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
def test_the_general_angle_gives_the_sin_squared_over_two_cos_squared_split(theta_deg: float) -> None:
    """At angle theta between k and B with eps in the k-B plane the weights are
    (cos^2 theta/2, sin^2 theta, cos^2 theta/2) -- the interpolation between the two limiting statements
    above: theta = 0 (k along B) gives (1/2, 0, 1/2) and theta = 90 deg (k perpendicular to B, so eps in the
    k-B plane is along B) gives pure pi. |eps_0|^2 = |eps . B_hat|^2 = sin^2 theta is the whole content."""
    theta = math.radians(theta_deg)
    k = np.array([math.sin(theta), 0.0, math.cos(theta)])
    eps = np.array([math.cos(theta), 0.0, -math.sin(theta)])  # in the k-B plane, transverse to k
    assert abs(float(np.dot(k, eps))) < 1e-15
    w = np.abs(spherical_components(eps, Z)) ** 2
    s2, c2 = math.sin(theta) ** 2, math.cos(theta) ** 2
    assert w == pytest.approx([c2 / 2.0, s2, c2 / 2.0], abs=1e-14)


# ---- the sum rule and the frame construction ------------------------------------------------------------


@pytest.mark.parametrize(
    "b_hat", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1 / math.sqrt(3),) * 3]
)
def test_the_sum_rule_holds_for_any_field_direction_and_any_polarization(
    b_hat: tuple[float, ...],
) -> None:
    """sum_q |eps_q|^2 = 1 (``spherical_components`` asserts it internally, so a violation raises)."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        v = rng.normal(size=3) + 1j * rng.normal(size=3)
        eps = v / np.linalg.norm(v)
        comps = spherical_components(eps, b_hat)
        assert float(np.sum(np.abs(comps) ** 2)) == pytest.approx(1.0, abs=1e-12)


def test_an_unnormalized_polarization_is_refused() -> None:
    with pytest.raises(ValueError, match="unit-normalized"):
        spherical_components(np.array([1.0, 1.0, 0.0]), Z)
    with pytest.raises(ValueError, match="unit vector"):
        atomic_frame((0.0, 0.0, 2.0))


@pytest.mark.parametrize(
    "b_hat", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.6, 0.0, 0.8), (1 / math.sqrt(3),) * 3]
)
def test_the_atomic_frame_is_right_handed_and_orthonormal_with_z_along_b(b_hat: tuple[float, ...]) -> None:
    x, y, z = atomic_frame(b_hat)
    assert z == pytest.approx(np.asarray(b_hat) / np.linalg.norm(b_hat), abs=1e-15)
    gram = np.array([[float(np.dot(a, b)) for b in (x, y, z)] for a in (x, y, z)])
    assert gram == pytest.approx(np.eye(3), abs=1e-14)
    assert np.cross(x, y) == pytest.approx(z, abs=1e-14), "right-handed"


@pytest.mark.parametrize("b_hat", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.6, 0.0, 0.8)])
def test_the_spherical_basis_is_orthonormal_and_conjugate_paired(b_hat: tuple[float, ...]) -> None:
    """<e_q|e_q'> = delta_{qq'} and e_{-1} = -conj(e_{+1}) (Section 13's e_{+-1} = -+(x +- iy)/sqrt2)."""
    e_minus, e_zero, e_plus = spherical_basis(b_hat)
    basis = (e_minus, e_zero, e_plus)
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            assert complex(np.vdot(a, b)) == pytest.approx(1.0 if i == j else 0.0, abs=1e-14)
    assert e_minus == pytest.approx(-np.conj(e_plus), abs=1e-15)


def test_to_atomic_frame_agrees_with_the_frame_it_is_built_from() -> None:
    b_hat = (0.6, 0.0, 0.8)
    x, y, z = atomic_frame(b_hat)
    v = np.array([0.3, -0.5, 0.2])
    assert to_atomic_frame(v, b_hat) == pytest.approx(
        np.array([np.dot(x, v), np.dot(y, v), np.dot(z, v)]), abs=1e-15
    )
    # a vector along B has only a z' component
    assert to_atomic_frame(np.asarray(b_hat), b_hat) == pytest.approx([0.0, 0.0, 1.0], abs=1e-15)


def test_linear_polarization_refuses_a_beam_along_b() -> None:
    """A beam along B carries no pi component, so the angle to B is undefined rather than zero."""
    with pytest.raises(ValueError, match="parallel to B_hat"):
        linear_polarization(Z, 0.0, Z)


# ---- the selection rule ------------------------------------------------------------------------------


def test_a_pure_sigma_plus_beam_drives_only_m_to_m_plus_one() -> None:
    """A pure eps_{+1} beam drives m_F -> m_F + 1 only (relative to the largest coupling of the set)."""
    from qutip_trap.light.beams import Beam
    from qutip_trap.species.raman import AtomicStructure
    from qutip_trap.units import C_M_PER_S
    from tests.atomic_fixtures import be9_like, field_z

    sp = be9_like()
    st = AtomicStructure(sp, 1.0, field_z().direction)
    e_p32 = sp.level("P3/2").energy_hz
    beam = Beam(
        wavelength_m=C_M_PER_S / (e_p32 - 0.4 * (e_p32 - sp.level("P1/2").energy_hz)),
        power_w=1e-3,
        waist_m=20e-6,
        k_hat=(0.0, 0.0, 1.0),
        polarization=tuple(SIGMA_PLUS),
        pointing_m=(0.0, 0.0, 0.0),
    )
    assert np.abs(spherical_components(beam.polarization, field_z().direction)) ** 2 == pytest.approx(
        [0.0, 0.0, 1.0], abs=1e-12
    )
    for m_lower in (-1, 0, 1):
        source = st.state(f"S1/2 F=2 mF={m_lower}")
        couplings = list(st.couplings_from(source, beam))
        largest = max(abs(om) for _e, om, _d in couplings)
        assert largest > 0.0
        driven = 0
        for e, om, _d in couplings:
            sp_e = st.spectra[e.level]
            m_upper = sp_e.mF[sp_e.index(e.label)]
            if abs(om) > 1e-9 * largest:
                assert m_upper == m_lower + 1, f"sigma+ reached mF = {m_upper} from {m_lower}"
                driven += 1
        assert driven > 0
