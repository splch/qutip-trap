"""Composite pulses (PLAN.md Sections 4.3.5, 9.2, 9.10, 9.15) against check_composite.py and Mount 2015."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.control.composite import (
    MOUNT_PD6_PHASES,
    SUZUKI_PHASE_TOLERANCE,
    CompositePulse,
    _t2j,
    ap1_phases,
    certificate,
    composite_pulse,
    corpse_angles,
    fidelity_avg,
    fidelity_c,
    fidelity_k,
    fitted_leading_coefficient,
    fold,
    fold_addressing,
    leading_coefficient,
    operator_distance,
    pd2_phases,
    phi_sk1,
    rotation,
    seq_bb1,
    seq_nb1,
    seq_pb1,
    seq_pb1_three_pulse_control,
    seq_pd6_mount,
    seq_scrofulous,
    seq_short_corpse,
    seq_sk1,
    seq_sk1_printed,
    suzuki_factor,
    suzuki_phase,
    toggled_phases,
)

PI = math.pi


def test_primitive_is_exact_and_a_2pi_segment_is_minus_identity() -> None:
    assert np.allclose(rotation(2 * PI, 0.7), -np.eye(2))
    v = rotation(PI, 0.0)
    cp = composite_pulse("primitive", PI)
    # amplitude infidelity exact: 1 - F_K = sin^2(theta eps/2); detuning: (1/2)(1 - cos theta) eps^2 -> eps^2 at theta = pi
    assert cp.infidelity(eps_a=1e-3) == pytest.approx(math.sin(PI * 1e-3 / 2) ** 2, rel=1e-9)
    assert leading_coefficient(cp, "amplitude", 2, 1e-4) == pytest.approx(PI**2 / 4, rel=1e-6)
    assert leading_coefficient(cp, "detuning", 2, 1e-4) == pytest.approx(1.0, rel=1e-6)
    assert fidelity_k(v, v) == 1.0 and fidelity_avg(v, v) == 1.0


def test_fidelity_measure_relations() -> None:
    """1 - F_K = 2(1 - F_C) and 1 - F_avg = (4/3)(1 - F_C) at leading order; the two printed norms are identically constant."""
    v = rotation(PI, 0.0)
    u = fold(seq_sk1(PI), eps_a=0.03)
    fc, fk, fav = fidelity_c(v, u), fidelity_k(v, u), fidelity_avg(v, u)
    assert fk == pytest.approx(fc**2, abs=1e-15)
    assert (1 - fk) / (2 * (1 - fc)) == pytest.approx(1.0, abs=1e-3)
    assert (1 - fav) / (4 * (1 - fc) / 3) == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(u @ v.conj().T, 2) == pytest.approx(1.0, abs=1e-12), (
        "Low-Yoder-Chuang's ||U V^dag|| is identically 1"
    )
    assert 1 - np.linalg.norm(v.conj().T @ u, 2) == pytest.approx(0.0, abs=1e-12), (
        "Brown-Harrow-Chuang's 1 - ||V^dag U|| is 0"
    )


@pytest.mark.parametrize(
    ("family", "amp_slope", "det_slope", "sim_slope", "windows"),
    [
        ("primitive", 2, 2, 2, ((1e-3, 1e-2), (1e-3, 1e-2))),
        ("SK1", 4, 2, 2, ((1e-4, 1e-3), (1e-3, 1e-2))),
        ("BB1", 6, 2, 2, ((3e-3, 3e-2), (1e-3, 1e-2))),
        ("PB1", 6, 2, 2, ((3e-3, 3e-2), (1e-3, 1e-2))),
        ("CORPSE", 2, 4, 2, ((1e-3, 1e-2), (1e-5, 1e-4))),
        ("short_CORPSE", 2, 4, 2, ((1e-3, 1e-2), (1e-5, 1e-4))),
        ("SCROFULOUS", 4, 2, 2, ((1e-4, 1e-3), (1e-3, 1e-2))),
        ("CinSK", 4, 4, 4, ((1e-4, 1e-3), (1e-5, 1e-4))),
        ("CinBB", 6, 4, 4, ((3e-3, 3e-2), (1e-5, 1e-4))),
    ],
)
def test_order_ladder_at_theta_pi(
    family: str,
    amp_slope: int,
    det_slope: int,
    sim_slope: int,
    windows: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """Section 9.15: log-log slopes of 1 - F_K, order n <=> infidelity slope 2(n + 1); CORPSE's detuning exponent is fitted
    below 1e-4 because its eps_D^3 term dominates a wide intermediate regime."""
    cp = composite_pulse(family, PI)
    sa, _ = cp.order_slope("amplitude", windows[0])
    sd, _ = cp.order_slope("detuning", windows[1])
    assert sa == pytest.approx(amp_slope, abs=0.12), (family, sa)
    assert sd == pytest.approx(det_slope, abs=0.12), (family, sd)
    if sim_slope == 4:
        ss, _ = cp.order_slope("simultaneous", (1e-4, 1e-3))
        assert ss == pytest.approx(4, abs=0.2), (family, ss)


def test_leading_coefficients_at_theta_pi() -> None:
    """Section 4.3.5 [recomputed here]: SK1 22.830256 eps_a^4 = (pi^2 sin 2 phi_1)^2; BB1 and CinBB 9.388566 eps_a^6; SCROFULOUS
    4.566051 eps_a^4 and 4.000000 eps_d^2; CORPSE 6.5008e-3 eps_d^4, degraded 115-fold to 0.75 by either concatenation."""
    assert leading_coefficient(composite_pulse("SK1", PI), "amplitude", 4, 1e-3) == pytest.approx(
        22.830256, rel=2e-3
    )
    assert (PI**2 * math.sin(2 * math.acos(-0.25))) ** 2 == pytest.approx(22.830256, rel=1e-6)
    assert leading_coefficient(composite_pulse("SCROFULOUS", PI), "amplitude", 4, 1e-3) == pytest.approx(
        4.566051, rel=2e-3
    )
    assert leading_coefficient(composite_pulse("SCROFULOUS", PI), "detuning", 2, 1e-3) == pytest.approx(
        4.0, rel=1e-4
    )
    assert leading_coefficient(composite_pulse("CORPSE", PI), "detuning", 4, 1e-4) == pytest.approx(
        6.5008e-3, rel=2e-3
    )
    assert leading_coefficient(composite_pulse("CinSK", PI), "detuning", 4, 1e-4) == pytest.approx(
        0.75, rel=2e-3
    )
    assert leading_coefficient(composite_pulse("CinBB", PI), "detuning", 4, 1e-4) == pytest.approx(
        0.75, rel=2e-3
    )
    # PB1 = P2 = PD2 is a second-order amplitude corrector like BB1, with its own leading coefficient
    assert leading_coefficient(composite_pulse("PB1", PI), "amplitude", 6, 3e-4) == pytest.approx(
        118.295935928, rel=1e-6
    )
    assert leading_coefficient(composite_pulse("BB1", PI), "amplitude", 6, 3e-4) == pytest.approx(
        9.388566343, rel=2e-6
    )
    assert leading_coefficient(composite_pulse("BB1", PI), "amplitude", 6, 3e-3) == pytest.approx(
        9.388566, rel=1e-2
    )
    assert leading_coefficient(composite_pulse("CinBB", PI), "amplitude", 6, 3e-3) == pytest.approx(
        9.388566, rel=1e-2
    )


def test_negative_controls_printed_sk1_and_p2_pulse_count() -> None:
    """BHC Eq. 10 as printed is first order and exactly twice the bare-pulse error; P2's phase forced into a three-pulse
    corrector is ~3700x worse than P2 and ~16x worse than no correction."""
    v = rotation(PI / 2, 0.0)
    bare = operator_distance(fold([(PI / 2, 0.0)], eps_a=1e-2), v)
    assert bare == pytest.approx(2 * abs(math.sin(PI / 2 * 1e-2 / 4)), rel=1e-6)
    assert operator_distance(fold(seq_sk1_printed(PI / 2), eps_a=1e-2), v) == pytest.approx(
        2 * bare, rel=1e-3
    )
    corrected = operator_distance(fold(seq_sk1(PI / 2), eps_a=1e-2), v)
    assert corrected < 0.05 * bare  # second order: about 3% of the bare error at eps = 1e-2
    assert operator_distance(fold(seq_sk1(PI / 2), eps_a=1e-3), v) < 0.05 * corrected
    p2 = 1 - fidelity_c(v, fold(seq_pb1(PI / 2), eps_a=0.1))
    three = 1 - fidelity_c(v, fold(seq_pb1_three_pulse_control(PI / 2), eps_a=0.1))
    none = 1 - fidelity_c(v, fold([(PI / 2, 0.0)], eps_a=0.1))
    assert three / p2 > 3000 and three / none > 14


def test_bb1_general_axis_and_the_corrector_identity() -> None:
    """The target phase is ADDED to every entry (Z(phi_0) U Z(phi_0)^dag exactly); W1 = R(pi, phi) R(2 pi, 3 phi) R(pi, phi) is +I."""
    phit = 0.7
    z = np.diag([np.exp(-0.5j * phit), np.exp(0.5j * phit)])
    ua = fold(seq_bb1(PI, phit), eps_a=0.05)
    ub = z @ fold(seq_bb1(PI, 0.0), eps_a=0.05) @ z.conj().T
    assert operator_distance(ua, ub) < 1e-12
    # W1 = +I "for any phi" (Section 4.3.5), not at one phase only
    for theta in (PI / 2, PI, 3 * PI / 2, 2.1):
        p1 = phi_sk1(theta)
        w1 = [(PI, p1), (2 * PI, 3 * p1), (PI, p1)]
        assert operator_distance(fold(w1), np.eye(2)) < 1e-12, theta
    assert operator_distance(fold(seq_bb1(PI)[1:]), np.eye(2)) < 1e-12
    # phi_B2 = arccos(-theta_t/4 pi) at the plan's three target angles (Section 4.3.5 [recomputed here]); the W1
    # identity above is the theta-independent half, this is the theta-dependent one
    for theta, phi_b2 in ((PI / 2, 1.696124158), (PI, 1.823476582), (3 * PI / 2, 1.955193101)):
        assert phi_sk1(theta) == pytest.approx(phi_b2, rel=1e-9), theta
        assert seq_bb1(theta)[1][1] == pytest.approx(phi_b2, rel=1e-9)
    assert phi_sk1(PI) == pytest.approx(math.acos(-0.25), rel=1e-15)
    # NB1 in the addressing model equals B2 in the amplitude model, numerically
    for x in (0.2, 0.1, 0.05):
        n2 = 1 - fidelity_c(np.eye(2), fold_addressing(seq_nb1(PI / 2), x))
        b2 = 1 - fidelity_c(rotation(PI / 2, 0.0), fold(seq_bb1(PI / 2), eps_a=x))
        assert n2 == pytest.approx(b2, rel=1e-9)


def test_corpse_scrofulous_and_durations() -> None:
    """CORPSE 420/300/60 degrees at theta = pi; SCROFULOUS 180_60 180_300 180_60; durations 4 pi + theta (SK1, BB1),
    4 pi + theta - 4k (CORPSE) and 8 pi + theta - 4k (CinSK, CinBB), with 1500 degrees at theta = pi for the concatenations."""
    a = corpse_angles(PI)
    assert [round(math.degrees(x), 6) for x in a] == [420.0, 300.0, 60.0]
    assert np.allclose(fold(seq_short_corpse(PI)), -rotation(PI, 0.0)), "the (0, 1, 0) winding returns -R"
    phases = [round(math.degrees(p) % 360, 3) for _, p in seq_scrofulous(PI)]
    assert phases == [60.0, 300.0, 60.0]
    k = math.asin(math.sin(PI / 2) / 2)
    assert composite_pulse("SK1", PI).total_rotation_rad() == pytest.approx(5 * PI)
    assert composite_pulse("BB1", PI).total_rotation_rad() == pytest.approx(5 * PI)
    assert composite_pulse("CORPSE", PI).total_rotation_rad() == pytest.approx(5 * PI - 4 * k)
    assert composite_pulse("CinSK", PI).total_rotation_rad() == pytest.approx(9 * PI - 4 * k)
    assert composite_pulse("CinBB", PI).total_rotation_rad() == pytest.approx(math.radians(1500.0))


def test_suzuki_ladder_and_its_printed_defect() -> None:
    """f_j = (2^{2j-1} - 2) f_{j-1}: 4, 24, 720, 90720 (P) and 2, 12, 360, 45360 (N, B); phi_P2 = arccos(-theta/8 pi),
    phi_P4 = arccos(-theta/48 pi); P2 is PB1; the corrected odd-k B layer reproduces Eq. 43."""
    assert [suzuki_factor(j, 4) for j in (1, 2, 3, 4)] == [4, 24, 720, 90720]
    assert [suzuki_factor(j, 2) for j in (1, 2, 3, 4)] == [2, 12, 360, 45360]
    # PLAN.md:493 requires the phase to be ROOT-FOUND on the leading eps coefficient and not read off f_j: the
    # agreement is what certifies the transcribed recursion (P/B null the toggled amplitude polygon, N the bare
    # addressing sum), and the printed (2^{2j-1} - 1) recursion is the negative control it catches
    for family in ("P", "N", "B"):
        for j in (1, 2, 3):
            root, residual = suzuki_phase(j, PI / 2, family)
            assert residual < SUZUKI_PHASE_TOLERANCE, (family, j, residual)
            f1 = 4.0 if family == "P" else 2.0
            assert root == pytest.approx(math.acos(-(PI / 2) / (2 * PI * suzuki_factor(j, f1))), abs=1e-12)
    assert suzuki_factor(2, 4, printed=True) == 28.0 and suzuki_factor(2, 2, printed=True) == 14.0
    for family in ("P", "N", "B"):
        _root, residual = suzuki_phase(2, PI / 2, family, printed=True)
        assert residual > 1e-3, (
            family,
            residual,
        )  # 1.5e-3 (P), 3.0e-3 (N, B): six decades above the tolerance
    # the printed f_2 = 28 ladder is FIRST order, not fourth: its amplitude infidelity slope is 2, not 10
    printed_phi = math.acos(-(PI / 2) / (2 * PI * suzuki_factor(2, 4, printed=True)))
    segs = [(PI / 2, 0.0)] + [(a, p) for a, p in _t2j(2, 1.0, printed_phi, "P")]
    eps = np.geomspace(1e-3, 1e-2, 7)
    vals = [1 - fidelity_k(rotation(PI / 2, 0.0), fold(segs, eps_a=e)) for e in eps]
    assert float(np.polyfit(np.log(eps), np.log(vals), 1)[0]) == pytest.approx(2.0, abs=0.1)
    p2 = composite_pulse("P2j", PI / 2, order=1)
    assert np.allclose(np.array(p2.segments), np.array(composite_pulse("PB1", PI / 2).segments))
    p4 = composite_pulse("P2j", PI / 2, order=2)
    assert p4.order == 4 and len(p4.segments) == 37
    assert p4.segments[1][1] == pytest.approx(math.acos(-(PI / 2) / (48 * PI)))
    # P4 corrects to fourth order: infidelity slope 10 (order-aware window)
    slope, _ = p4.order_slope("amplitude", (2e-2, 8e-2))
    assert slope == pytest.approx(10.0, abs=0.5)
    b2 = composite_pulse("B2j", PI / 2, order=1)
    assert operator_distance(fold(b2.segments, eps_a=0.05), fold(seq_bb1(PI / 2), eps_a=0.05)) < 1e-12


def test_low_yoder_chuang_certificate_and_toggling() -> None:
    """The eps-free certificate Phi_L^j = f_L^j(gamma) holds for j <= n and FAILS at n + 1 (AP1 = SK1 at n = 1, PD2 = PB1 at
    n = 2); toggled PD2 at gamma = 1 is Wimperis's BB1 (phi_1, 3 phi_1, 3 phi_1, phi_1) with phi_1 = arccos(-1/4)."""
    for gamma in (0.25, 0.5, 1.0):
        for phases, n in ((ap1_phases(gamma), 1), (pd2_phases(gamma), 2)):
            cert = certificate(phases, gamma, n + 1)
            assert all(cert[j][1] for j in range(1, n + 1))
            assert not cert[n + 1][1] and abs(cert[n + 1][0]) > 0.1
    q = math.acos(-0.25)
    assert np.allclose(toggled_phases(pd2_phases(1.0)), [q, 3 * q, 3 * q, q])
    bbn = composite_pulse("BBn", PI, order=2)
    assert operator_distance(fold(bbn.segments, eps_a=0.05), fold(seq_bb1(PI), eps_a=0.05)) < 1e-12
    cp = composite_pulse("PB1", PI / 2)
    cert2 = cp.certificate()
    assert cert2[1][1] and cert2[2][1] and not cert2[3][1]
    with pytest.raises(ValueError):
        composite_pulse("SCROFULOUS", PI).certificate()


def test_mount_pd6_anchors() -> None:
    """Section 9.10: PD6 keyed by theta_t: 2.447e-2 -> 3.713e-11 at theta_t = pi, eps = 0.1, and 6.156e-3 -> 1.348e-11 at
    pi/2, while cross-keyed rows leave 6.156e-3; B2 stays below 1% for |eps| < 0.4 and PD6 for |eps| < 0.6 (Section 9.2)."""
    pd6_pi = composite_pulse("PDn", PI, order=6)
    pd6_half = composite_pulse("PDn", PI / 2, order=6)
    assert len(pd6_pi.segments) == 13 and pd6_pi.total_rotation_rad() == pytest.approx(13 * PI)
    assert composite_pulse("primitive", PI).infidelity(eps_a=0.1) == pytest.approx(2.447e-2, rel=2e-3)
    assert pd6_pi.infidelity(eps_a=0.1) == pytest.approx(3.713e-11, rel=2e-3)
    assert composite_pulse("primitive", PI / 2).infidelity(eps_a=0.1) == pytest.approx(6.156e-3, rel=2e-3)
    assert pd6_half.infidelity(eps_a=0.1) == pytest.approx(1.348e-11, rel=2e-3)
    cross = (
        [(PI / 2, 0.0)]
        + [(PI, p) for p in MOUNT_PD6_PHASES[PI]]
        + [(PI, p) for p in reversed(MOUNT_PD6_PHASES[PI])]
    )
    assert 1 - fidelity_k(rotation(PI / 2, 0.0), fold(cross, eps_a=0.1)) == pytest.approx(6.156e-3, rel=2e-3)
    b2 = composite_pulse("BB1", PI / 2)
    assert b2.infidelity(eps_a=0.4) < 0.01 < b2.infidelity(eps_a=0.5)
    assert pd6_half.infidelity(eps_a=0.6) < 0.01 < pd6_half.infidelity(eps_a=0.7)
    assert np.allclose(np.array(seq_pd6_mount(PI, target_first=False)[-1]), np.array([PI, 0.0]))
    with pytest.raises(ValueError):
        composite_pulse("PDn", 1.0, order=6)


def test_dc_polygon_and_record_invariants() -> None:
    """Section 9.15: the polygon closes for every amplitude-correcting family and equals (pi, 0, 0) otherwise."""
    for family in ("SK1", "BB1", "PB1", "SCROFULOUS", "CinSK", "CinBB"):
        assert composite_pulse(family, PI).dc_polygon()[1], family
    for family in ("primitive", "CORPSE"):
        poly, closed = composite_pulse(family, PI).dc_polygon()
        assert not closed, family
        assert poly == pytest.approx([PI, 0.0, 0.0], abs=1e-14), family
    with pytest.raises(ValueError):
        CompositePulse("BB1", PI, 0.0, 2, frozenset({"amplitude"}), ((PI, 0.0), (-PI, 0.0)))
    with pytest.raises(ValueError):
        composite_pulse("SK1", 5 * PI)  # arccos domain
    with pytest.raises(NotImplementedError):
        composite_pulse("SKn", PI, order=2)


# ---- Appendix E's two filter-function methods (Section 6.9) --------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "slope"),
    [
        ("primitive", 2.0),
        ("SK1", 4.0),
        ("BB1", 4.0),
        ("PB1", 4.0),
        ("SCROFULOUS", 4.0),
        ("CinSK", 4.0),
        ("CinBB", 4.0),
        ("CORPSE", 2.0),
    ],
)
def test_filter_function_amplitude_low_frequency_slope(family: str, slope: float) -> None:
    """Section 9.15: F_a ~ omega^2 for a family that does not correct amplitude at first order, omega^4 for one that does.

    ``CompositePulse.filter_function_amplitude`` is Appendix E's method; it delegates to the exact A_l/B_l segment sum
    of ``noise.decoupling.amplitude_filter_function`` and must agree with it element by element.
    """
    from qutip_trap.noise.decoupling import amplitude_filter_function, composite_segments, local_slope

    cp = composite_pulse(family, PI)
    omega = np.geomspace(1e-5, 1e-4, 4)
    f = cp.filter_function_amplitude(omega, 1.0)
    assert np.array_equal(f, amplitude_filter_function(composite_segments(cp, 1.0), omega))
    assert local_slope(omega, f) == pytest.approx(slope, abs=1e-3), (family, f)
    # the method carries the Rabi frequency: F_a is a function of omega/Omega
    assert cp.filter_function_amplitude(2.0 * omega, 2.0) == pytest.approx(f, rel=1e-12)


def test_dc_floor_method_fits_the_leading_coefficients_and_the_section_9_15_floors() -> None:
    """Appendix E's ``dc_floor``: c-hat fitted numerically, cached, and combined as c-hat (2m+1)!! (<beta^2>/Omega^2)^{m+1}.

    The fitted c-hats reproduce the Section 4.3.5 values to 2e-7 relative or better (SK1 4.1e-10, BB1 1.5e-7, CORPSE
    1.6e-7 as measured here), so the floors come out at 2.5e-7 relative - the precision of the fit, not of the plan.
    """
    assert fitted_leading_coefficient(composite_pulse("SK1", PI), "amplitude", 4) == pytest.approx(
        22.8302557111, rel=1e-6
    )
    assert fitted_leading_coefficient(composite_pulse("BB1", PI), "amplitude", 6) == pytest.approx(
        9.388566343, rel=1e-6
    )
    assert fitted_leading_coefficient(composite_pulse("CORPSE", PI), "detuning", 4) == pytest.approx(
        0.006500751892, rel=1e-6
    )
    var = 2.07e9 / PI  # <beta^2> in (rad/s)^2 at the Section 9.15 benchmark, Omega = 1.5e6 rad/s
    assert composite_pulse("SK1", PI).dc_floor({"amplitude": var}, 1.5e6) == pytest.approx(
        5.87365e-6, rel=1e-5
    )
    assert composite_pulse("BB1", PI).dc_floor({"amplitude": var}, 1.5e6) == pytest.approx(
        3.53675e-9, rel=1e-5
    )
    assert composite_pulse("CORPSE", PI).dc_floor({"detuning": var}, 1.5e6) == pytest.approx(
        1.67248e-9, rel=1e-5
    )
    # a channel the family does not correct enters at m = 0: BB1 has no detuning correction, so its detuning floor is
    # the primitive's (c-hat = 1 at theta = pi, (2.0 + 1)!! ... m = 0 gives 1!! = 1)
    prim = composite_pulse("primitive", PI)
    bb1 = composite_pulse("BB1", PI)
    assert bb1.dc_floor({"detuning": var}, 1.5e6) == pytest.approx(
        prim.dc_floor({"detuning": var}, 1.5e6), rel=2e-3
    )
    # the channels add
    both = bb1.dc_floor({"amplitude": var, "detuning": var}, 1.5e6)
    assert both == pytest.approx(
        bb1.dc_floor({"amplitude": var}, 1.5e6) + bb1.dc_floor({"detuning": var}, 1.5e6), rel=1e-12
    )
    with pytest.raises(ValueError, match="amplitude and detuning"):
        bb1.dc_floor({"addressing": var}, 1.5e6)
    with pytest.raises(ValueError, match="Rabi frequency"):
        bb1.dc_floor({"amplitude": var}, 0.0)
