"""The ``Trap`` record's Appendix E methods on the explicit, rod and surface paths (Sections 3.3, 4.1.1, 4.1.6; 9.12 row 66)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import DcElectrodes, Device, Electrodes, RfDrive, Trap, solve_crystal
from qutip_trap.species import species
from qutip_trap.trap.mathieu import UnstableMathieuError, beta_exact, c0_wronskian
from qutip_trap.trap.micromotion import (
    MicromotionIndex,
    carrier_amplitude,
    carrier_factor,
    displacement_m,
    excess_amplitude_m,
    modulation_index,
    second_order_doppler_fraction,
    sideband_ratio,
    sideband_weights,
)
from qutip_trap.units import ATOMIC_MASS_KG, E_C, TWO_PI
from tests.fixtures import make_device, make_trap


def test_fixture_trap_reproduces_the_fixture_crystal_frequencies() -> None:
    yb = species("171Yb+")
    cr = solve_crystal(make_trap(), (yb, yb))
    f = {(m.family, m.index): m.omega_hz for m in cr.modes}
    assert f[("transverse_1", 0)] == pytest.approx(math.sqrt(3.0**2 - 1.0**2) * 1e6, rel=1e-12)
    assert f[("transverse_2", 0)] == pytest.approx(math.sqrt(2.9**2 - 1.0**2) * 1e6, rel=1e-12)
    assert f[("axial", 1)] == pytest.approx(math.sqrt(3.0) * 1e6, rel=1e-12)
    assert make_trap().path == "explicit" and make_trap().anharmonic() is None
    assert make_device().hash() == make_device().hash()
    assert isinstance(make_device(), Device)


def test_explicit_path_needs_the_rf_frequency_for_mathieu_parameters() -> None:
    yb = species("171Yb+")
    trap = make_trap()
    with pytest.raises(ValueError, match="rf frequency"):
        trap.mathieu(yb)
    assert trap.secular_hz(yb) == (3.0e6, 2.9e6, 1.0e6)
    with_rf = Trap(
        omega_hz=(3.0e6, 2.9e6, 1.0e6),
        axis_angle_rad=0.2,
        rf=RfDrive(0.0, 30e6),
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    p = with_rf.mathieu(yb)
    assert p.secular_hz == pytest.approx((3.0e6, 2.9e6, 1.0e6), rel=1e-9)
    assert p.mass_kg == pytest.approx(yb.mass_u * ATOMIC_MASS_KG) and p.omega_rf_hz == 30e6
    assert p.C0[2] == 1.0 and p.C0[0] == pytest.approx(c0_wronskian(p.a[0, 0], p.q[0, 0]))
    assert np.allclose(p.principal_axes[:, 0], [math.cos(0.2), math.sin(0.2), 0.0])
    with pytest.raises(ValueError, match="shim"):
        Trap(
            omega_hz=(3.0e6, 2.9e6, 1.0e6),
            axis_angle_rad=0.0,
            rf=None,
            dc=None,
            geometry=None,
            stray_field_v_per_m=(0.0, 0.0, 0.0),
            shim_voltages_v={"s": 1.0},
        )
    with pytest.raises(ValueError):
        Trap(
            omega_hz=None,
            axis_angle_rad=0.0,
            rf=None,
            dc=None,
            geometry=None,
            stray_field_v_per_m=(0.0, 0.0, 0.0),
            shim_voltages_v={},
        )


def test_mixed_species_on_the_explicit_path_scale_with_the_mathieu_parameters() -> None:
    """omega_z ~ 1/sqrt m exactly (static only); the lighter ion's radial frequencies are higher and follow beta(a m_ref/m, q m_ref/m).

    88Sr+ is the reference with 40Ca+ as the second species (mass ratio 2.2); a 171Yb+/40Ca+ pair at q_Yb = 0.28 would put the
    calcium ion at q = 1.2, outside the stability region, which the solver reports as UnstableMathieuError.
    """
    sr, ca, yb = species("88Sr+"), species("40Ca+"), species("171Yb+")
    bare = make_trap()
    with pytest.raises(ValueError, match="rf frequency"):
        bare.single_ion_frequencies_rad_s((yb, ca))
    trap = Trap(
        omega_hz=(3.0e6, 2.9e6, 1.0e6),
        axis_angle_rad=0.0,
        rf=RfDrive(0.0, 60e6),
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    omega, axes, field = trap.single_ion_frequencies_rad_s((sr, ca), reference=0)
    assert omega[0] == pytest.approx(TWO_PI * np.array([3.0e6, 2.9e6, 1.0e6]))
    ratio = sr.mass_u / ca.mass_u
    assert omega[1, 2] == pytest.approx(omega[0, 2] * math.sqrt(ratio), rel=1e-9)
    p = trap.mathieu(sr)
    for k in range(2):
        assert omega[1, k] == pytest.approx(
            beta_exact(p.a[k, k] * ratio, p.q[k, k] * ratio) * TWO_PI * 60e6 / 2, rel=1e-9
        )
    assert omega[1, 0] > omega[0, 0], "at fixed voltages the LIGHTER ion (Ca) is more tightly confined"
    assert np.allclose(axes, np.eye(3)) and np.allclose(field, 0.0)
    cr = solve_crystal(trap, (sr, ca))
    assert len(cr.modes) == 6 and cr.collinear()
    hot = Trap(
        omega_hz=(1.0e6, 1.0e6, 0.2e6),
        axis_angle_rad=0.0,
        rf=RfDrive(0.0, 10.1e6),
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    with pytest.raises(UnstableMathieuError):
        hot.single_ion_frequencies_rad_s((yb, ca))


def test_berkeland_excess_micromotion_numbers_of_section_9_12() -> None:
    """171Yb+ at 1 MHz, q = 0.28: E_dc = 1, 10, 100 V/m give u0 = 14.3, 143, 1429 nm, beta = 0.034, 0.340, 3.40 at 369.5 nm,
    J0 = 0.9997, 0.971, -0.365; J1^2/J0^2 -> (beta/2)^2."""
    yb = species("171Yb+")
    m = yb.mass_u * ATOMIC_MASS_KG
    k = TWO_PI / 369.5e-9
    for e_dc, u0_nm, beta_exp, j0 in (
        (1.0, 14.3, 0.034, 0.9997),
        (10.0, 143.0, 0.340, 0.971),
        (100.0, 1429.0, 3.40, -0.365),
    ):
        u0 = displacement_m(e_dc, m, TWO_PI * 1e6)[0]
        assert u0 * 1e9 == pytest.approx(u0_nm, rel=3e-3)
        amp = excess_amplitude_m(u0, 0.28)[0]
        beta = modulation_index(np.array([k, 0.0, 0.0]), np.array([amp, 0.0, 0.0]))
        # both are SIGNED under the adopted origin a - 2q cos 2xi: u_1 = -(1/2) q u_0 and beta = delta_k . u_1, so a
        # positive q and a positive field give a NEGATIVE index (Section 4.1.1; Section 13 "Floquet function and rf
        # phase origin"). The plan's printed magnitudes are Berkeland's (1/2)|q| u_0
        assert amp < 0.0 and beta == pytest.approx(-beta_exp, rel=1.5e-2)
        assert excess_amplitude_m(u0, -0.28)[0] == pytest.approx(-amp, rel=1e-12), (
            "q_y = -q_x puts the two radial micromotions in antiphase"
        )
        # the plan's J0 values are rounded at 1e-3 and evaluated at its own (rounded) beta: J0(3.40) = -0.3643, J0(3.404) = -0.3650
        assert carrier_amplitude(beta) == pytest.approx(j0, abs=1.5e-3)
        assert carrier_factor(beta) == pytest.approx(j0 * j0, abs=1.5e-3)
    assert sideband_ratio(0.034) == pytest.approx((0.034 / 2) ** 2, rel=1e-3)
    w = sideband_weights(0.34)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9) and w[1] == w[-1]
    assert (
        second_order_doppler_fraction(
            np.array([1.429e-6 * 0.14, 0.0, 0.0]), TWO_PI * 1e6 * 2 * math.sqrt(2) / 0.28
        )
        < 0.0
    )
    with pytest.raises(ValueError):
        MicromotionIndex(0.1, 0.0, "amplitude")  # type: ignore[arg-type]
    assert MicromotionIndex(0.3, 0.4, "peak").total == pytest.approx(0.5)
    assert MicromotionIndex(0.3, 0.4, "rms").as_peak().in_phase == pytest.approx(0.3 * math.sqrt(2))


def test_trap_micromotion_beta_scales_with_stray_field_and_wavevector() -> None:
    yb = species("171Yb+")

    def trap(e_x: float) -> Trap:
        return Trap(
            omega_hz=(1.0e6, 1.0e6, 0.2e6),
            axis_angle_rad=0.0,
            rf=RfDrive(0.0, 10.1e6),
            dc=None,
            geometry=None,
            stray_field_v_per_m=(e_x, 0.0, 0.0),
            shim_voltages_v={},
        )

    k = TWO_PI / 369.5e-9
    p = trap(1.0).mathieu(yb)
    assert p.q[0, 0] == pytest.approx(0.28, abs=0.02), (
        "Berkeland's example: q_x = 2 sqrt 2 omega_x/Omega ~ 0.28"
    )
    b1 = trap(1.0).micromotion_beta(yb, np.array([k, 0.0, 0.0]))
    b10 = trap(10.0).micromotion_beta(yb, np.array([k, 0.0, 0.0]))
    assert b1.convention == "peak" and b1.out_of_phase == 0.0
    assert b10.in_phase == pytest.approx(10 * b1.in_phase, rel=1e-9)
    assert trap(1.0).micromotion_beta(yb, np.array([2 * k, 0.0, 0.0])).in_phase == pytest.approx(
        2 * b1.in_phase, rel=1e-9
    )
    assert trap(1.0).micromotion_beta(yb, np.array([0.0, 0.0, k])).in_phase == 0.0, (
        "axial motion has no micromotion (q_z = 0)"
    )
    assert trap(0.0).micromotion_beta(yb, np.array([k, 0.0, 0.0])).in_phase == 0.0
    omega_ps2 = (TWO_PI * 10.1e6 / 2) ** 2 * (
        p.a[0, 0] + p.q[0, 0] ** 2 / 2
    )  # the static spring is the pseudopotential curvature
    expected = -k * 0.5 * p.q[0, 0] * E_C / (yb.mass_u * ATOMIC_MASS_KG * omega_ps2)
    # SIGNED: u_1 = -(1/2) Q u_0 under the adopted origin a - 2q cos 2xi (Section 13, "Floquet function and rf phase
    # origin"), so sign(beta_ip) = -sign(q_x E_x); the plan's printed 0.034 is the magnitude
    assert b1.in_phase == pytest.approx(expected, rel=1e-9) and expected < 0.0
    assert abs(b1.in_phase) == pytest.approx(0.034, rel=0.1)
    assert trap(-1.0).micromotion_beta(yb, np.array([k, 0.0, 0.0])).in_phase == pytest.approx(
        -b1.in_phase, rel=1e-12
    ), "the index steps by pi as the residual field crosses zero"
    # compensation: a shim field that cancels the stray field nulls the in-phase index (explicit path folds shims into the stray field)
    assert trap(0.0).residual_field_v_per_m() == pytest.approx(np.zeros(3))


def test_rod_trap_path_follows_berkeland_and_the_phase_imbalance_term() -> None:
    yb = species("171Yb+")
    m = yb.mass_u * ATOMIC_MASS_KG
    omega_rf = TWO_PI * 20e6
    geometry = Electrodes("rod_quadrupole", {"R_m": 1.0e-3, "Z0_m": 2.0e-3, "kappa": 0.3, "alpha": 0.8})
    trap = Trap(
        omega_hz=None,
        axis_angle_rad=0.0,
        rf=RfDrive(300.0, 20e6, phase_imbalance_rad=1e-3),
        dc=DcElectrodes({"endcaps": 8.0}),
        geometry=geometry,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    assert trap.path == "linear"
    p = trap.mathieu(yb)
    q_x = 2 * E_C * 300.0 / (m * 1e-6 * omega_rf**2)
    a_z = 8 * E_C * 0.3 * 8.0 / (m * 4e-6 * omega_rf**2)
    assert p.q[0, 0] == pytest.approx(q_x) and p.q[1, 1] == pytest.approx(-q_x) and p.q[2, 2] == 0.0
    assert p.a[2, 2] == pytest.approx(a_z) and p.a[0, 0] == pytest.approx(-a_z / 2)
    assert p.secular_hz[2] == pytest.approx(math.sqrt(a_z) * 20e6 / 2, rel=1e-9)
    assert p.secular_hz[0] == pytest.approx(math.sqrt(-a_z / 2 + q_x**2 / 2) * 20e6 / 2, rel=1e-3), (
        "beta ~ sqrt(a + q^2/2) at q = 0.02"
    )
    k = TWO_PI / 369.5e-9
    idx = trap.micromotion_beta(yb, np.array([k, 0.0, 0.0]))
    assert idx.in_phase == 0.0
    assert idx.out_of_phase == pytest.approx(k * 0.25 * q_x * 1.0e-3 * 0.8 * 1e-3, rel=1e-12)
    bare = Trap(
        omega_hz=None,
        axis_angle_rad=0.0,
        rf=RfDrive(300.0, 20e6, phase_imbalance_rad=1e-3),
        dc=DcElectrodes({"endcaps": 8.0}),
        geometry=Electrodes("rod_quadrupole", {"R_m": 1.0e-3, "Z0_m": 2.0e-3, "kappa": 0.3}),
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    with pytest.raises(ValueError, match="alpha"):
        bare.micromotion_beta(yb, np.array([k, 0.0, 0.0]))


def test_a_rod_or_blade_trap_refuses_shim_voltages_it_cannot_turn_into_a_field() -> None:
    """PLAN 4.1.1: no source gives kappa, R', Z0 or Phi'' for any electrode design, so Berkeland's rod map has no
    shim -> field response. A non-zero shim there used to be dropped from ``residual_field_v_per_m`` and, if it happened
    to be named like the endcap, to change a_z instead of producing a field; it is now refused, as on the explicit path."""
    geometry = Electrodes("rod_quadrupole", {"R_m": 1.0e-3, "Z0_m": 2.0e-3, "kappa": 0.3})

    def rod(shims: dict[str, float]) -> Trap:
        return Trap(
            omega_hz=None,
            axis_angle_rad=0.0,
            rf=RfDrive(300.0, 20e6),
            dc=DcElectrodes({"endcaps": 8.0}),
            geometry=geometry,
            stray_field_v_per_m=(5.0, 0.0, 0.0),
            shim_voltages_v=shims,
        )

    for shims in ({"shim_x": 3.0}, {"endcaps": 1.0}, {"shim_x": -1e-9}):
        with pytest.raises(ValueError, match="no electrode model to convert shim voltages into fields"):
            rod(shims)
    # explicitly zero shims are a record of "nothing applied" and stay legal, as does the bare stray field
    assert rod({}).residual_field_v_per_m() == pytest.approx(np.array([5.0, 0.0, 0.0]))
    assert rod({"shim_x": 0.0}).residual_field_v_per_m() == pytest.approx(np.array([5.0, 0.0, 0.0]))
    # the explicit path keeps its own message
    with pytest.raises(ValueError, match="the explicit-frequency path has no electrode model"):
        Trap(
            omega_hz=(1e6, 1e6, 0.2e6),
            axis_angle_rad=0.0,
            rf=None,
            dc=None,
            geometry=None,
            stray_field_v_per_m=(0.0, 0.0, 0.0),
            shim_voltages_v={"shim_x": 1.0},
        )


def test_surface_trap_path_end_to_end_with_88sr() -> None:
    """House's five-wire geometry with an externally solved axial curvature: Mathieu parameters, a two-ion crystal, shims and stray field."""
    sr = species("88Sr+")
    m = sr.mass_u * ATOMIC_MASS_KG
    omega_rf = TWO_PI * 50e6
    kzz = m * (TWO_PI * 1.0e6) ** 2 / E_C  # V/m^2 for a 1 MHz axial frequency
    geometry = Electrodes(
        "surface_five_wire",
        {
            "a_m": 100e-6,
            "b_m": 120e-6,
            "dc_curvature_xx_v_per_m2": -kzz / 2,
            "dc_curvature_yy_v_per_m2": -kzz / 2,
            "dc_curvature_zz_v_per_m2": kzz,
        },
    )
    trap = Trap(
        omega_hz=None,
        axis_angle_rad=0.0,
        rf=RfDrive(300.0, 50e6),
        dc=DcElectrodes({"centre": 0.0}),
        geometry=geometry,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={"left": 0.0, "right": 0.0},
    )
    assert trap.path == "surface"
    p = trap.mathieu(sr)
    assert p.secular_hz[2] == pytest.approx(1.0e6, rel=1e-9)
    q = (
        2 * E_C * 300.0 * 0.2284260 / (2 * E_C * 300.0 / (1.46e-25 * omega_rf**2)) / (m * omega_rf**2)
    )  # House's Q11 rescaled to the Sr mass
    assert abs(p.q[0, 0]) == pytest.approx(q, rel=1e-6)
    assert p.secular_hz[0] == pytest.approx(p.secular_hz[1], rel=1e-6), (
        "the rf pseudopotential is isotropic and the dc is symmetric here"
    )
    assert np.allclose(np.abs(p.principal_axes[:, 2]), [0.0, 0.0, 1.0])
    cr = solve_crystal(trap, (sr, sr))
    assert len(cr.modes) == 6 and cr.collinear()
    assert cr.family("axial")[0].omega_hz == pytest.approx(1.0e6, rel=1e-9)
    assert np.allclose(cr.positions_m[:, 1], trap.rf_null_m()[1]), "the chain sits at the rf null height"
    # a shim voltage produces a field at the null and hence an in-phase micromotion index
    shimmed = Trap(
        omega_hz=None,
        axis_angle_rad=0.0,
        rf=RfDrive(300.0, 50e6),
        dc=DcElectrodes({"centre": 0.0}),
        geometry=geometry,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={"left": 0.5, "right": -0.5},
    )
    field = shimmed.residual_field_v_per_m()
    assert abs(field[0]) > 0.0
    k = TWO_PI / 422e-9
    idx = shimmed.micromotion_beta(sr, np.array([k, 0.0, 0.0]))
    # signed: sign(beta_ip) = -sign(q_x E_x) under the adopted rf phase origin (Section 13)
    assert idx.in_phase != 0.0 and idx.convention == "peak"
    assert math.copysign(1.0, idx.in_phase) == -math.copysign(1.0, p.q[0, 0] * field[0])
    compensated = Trap(
        omega_hz=None,
        axis_angle_rad=0.0,
        rf=RfDrive(300.0, 50e6),
        dc=DcElectrodes({"centre": 0.0}),
        geometry=geometry,
        stray_field_v_per_m=(-field[0], -field[1], -field[2]),
        shim_voltages_v={"left": 0.5, "right": -0.5},
    )
    assert compensated.micromotion_beta(sr, np.array([k, 0.0, 0.0])).in_phase == pytest.approx(0.0, abs=1e-12)
