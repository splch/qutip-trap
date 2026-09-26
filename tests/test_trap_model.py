"""The ``Trap`` record on the explicit, rod and surface paths: Mathieu parameters, mixed species, the residual field, the
signed excess micromotion and the trap quantities ``Device.derived()`` reports."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.special import jv

from qutip_trap.control.pulses import Pulse
from qutip_trap.device.model import Field
from qutip_trap.device.presets import secular_trap
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian, micromotion_index
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.raman import derive_raman_drive, lamb_dicke_parameters, square_drive
from qutip_trap.light.roles import NoDetectionBeamError, detection_beams
from qutip_trap.run.job import detection_micromotion
from qutip_trap.species import species
from qutip_trap.trap.crystal import build_crystal, solve_crystal
from qutip_trap.trap.mathieu import UnstableMathieuError, beta_exact, c0_wronskian
from qutip_trap.trap.micromotion import MicromotionIndex, modulation_index, second_order_doppler_fraction
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import DcElectrodes, RfDrive
from qutip_trap.trap.surface import Electrodes
from qutip_trap.units import ATOMIC_MASS_KG, E_C, TWO_PI
from tests.fixtures import KX, chain_device, microwave_device, quiet_device, single_ion_raman_device
from tests.oracles import five_wire_null_height_m

K_369 = TWO_PI / 369.5e-9


def _yb_trap(e_x: float = 0.0) -> Trap:
    """Berkeland's example: 171Yb+ at 1 MHz radially with Omega_rf/2pi = 10.1 MHz, so q_x = +0.2785."""
    return dataclasses.replace(
        secular_trap((1.0e6, 1.0e6, 0.2e6)), rf=RfDrive(0.0, 10.1e6), stray_field_v_per_m=(e_x, 0.0, 0.0)
    )


def _house_surface(mass_kg: float, **trap: object) -> Trap:
    """House's five-wire rails (a = 100 um, b = 120 um) with an external axial curvature for a 1 MHz axial frequency."""
    kzz = mass_kg * (TWO_PI * 1.0e6) ** 2 / E_C
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
    base = Trap(
        omega_hz=None,
        axis_angle_rad=0.0,
        rf=RfDrive(300.0, 50e6),
        dc=DcElectrodes({"centre": 0.0}),
        geometry=geometry,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    return dataclasses.replace(base, **trap)


def _rod(**rf: float) -> Trap:
    return Trap(
        omega_hz=None,
        axis_angle_rad=0.0,
        rf=RfDrive(300.0, 20e6, **rf),
        dc=DcElectrodes({"endcaps": 8.0}),
        geometry=Electrodes("rod_quadrupole", {"R_m": 1.0e-3, "Z0_m": 2.0e-3, "kappa": 0.3, "alpha": 0.8}),
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )


# ---- the three paths ----------------------------------------------------------------------------------------------------


def test_fixture_trap_reproduces_the_fixture_crystal_frequencies() -> None:
    yb = species("171Yb+")
    f = {(m.family, m.index): m.omega_hz for m in solve_crystal(secular_trap(), (yb, yb)).modes}
    assert f[("transverse_1", 0)] == pytest.approx(math.sqrt(3.0**2 - 1.0**2) * 1e6, rel=1e-12)
    assert f[("transverse_2", 0)] == pytest.approx(math.sqrt(2.9**2 - 1.0**2) * 1e6, rel=1e-12)
    assert f[("axial", 1)] == pytest.approx(math.sqrt(3.0) * 1e6, rel=1e-12)
    assert secular_trap().path == "explicit" and secular_trap().anharmonic() is None


def test_explicit_path_needs_the_rf_frequency_for_mathieu_parameters() -> None:
    yb = species("171Yb+")
    trap = secular_trap()
    with pytest.raises(ValueError, match="rf frequency"):
        trap.mathieu(yb)
    assert trap.secular_hz(yb) == (3.0e6, 2.9e6, 1.0e6)
    p = dataclasses.replace(
        secular_trap((3.0e6, 2.9e6, 1.0e6)), axis_angle_rad=0.2, rf=RfDrive(0.0, 30e6)
    ).mathieu(yb)
    assert p.secular_hz == pytest.approx((3.0e6, 2.9e6, 1.0e6), rel=1e-9)
    assert p.mass_kg == pytest.approx(yb.mass_u * ATOMIC_MASS_KG) and p.omega_rf_hz == 30e6
    assert p.C0[2] == 1.0 and p.C0[0] == pytest.approx(c0_wronskian(p.a[0, 0], p.q[0, 0]))
    assert np.allclose(p.principal_axes[:, 0], [math.cos(0.2), math.sin(0.2), 0.0])
    with pytest.raises(ValueError, match="the explicit-frequency path has no electrode model"):
        dataclasses.replace(secular_trap((3.0e6, 2.9e6, 1.0e6)), shim_voltages_v={"s": 1.0})
    with pytest.raises(ValueError):
        dataclasses.replace(secular_trap(), omega_hz=None)


def test_mixed_species_on_the_explicit_path_scale_with_the_mathieu_parameters() -> None:
    """On the explicit path omega_z scales as 1/sqrt m and the lighter ion's radial frequencies follow beta(a m_ref/m,
    q m_ref/m) to 1e-9; a 40Ca+ neighbour of 171Yb+ at q = 0.28 (so q = 1.2) is refused as unstable."""
    sr, ca, yb = species("88Sr+"), species("40Ca+"), species("171Yb+")
    with pytest.raises(ValueError, match="rf frequency"):
        dataclasses.replace(secular_trap(), reference_mass_u=yb.mass_u).single_ion_frequencies_rad_s((yb, ca))
    trap = dataclasses.replace(
        secular_trap((3.0e6, 2.9e6, 1.0e6)), rf=RfDrive(0.0, 60e6), reference_mass_u=sr.mass_u
    )
    omega, axes, field = trap.single_ion_frequencies_rad_s((sr, ca))
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
    with pytest.raises(UnstableMathieuError):
        dataclasses.replace(_yb_trap(), reference_mass_u=yb.mass_u).single_ion_frequencies_rad_s((yb, ca))


def test_each_ion_of_a_mixed_crystal_on_the_explicit_path_takes_its_own_mathieu_record() -> None:
    """Explicit frequencies quoted for 88Sr+ are the Sr ion's of a rod trap; the 40Ca+ ion's (a, q), exponents, C0 and signed
    micromotion amplitude then equal the rod's voltage map evaluated at the Ca mass (1e-9), and in a (3.0, 2.9, 1.0) MHz
    Sr trap at 60 MHz the Ca ion's eta carries its own C0 = 1.019948, not Sr's 1.003867."""
    sr, ca = species("88Sr+"), species("40Ca+")
    rod = dataclasses.replace(
        _rod(),
        geometry=Electrodes("rod_quadrupole", {"R_m": 0.5e-3, "Z0_m": 2.0e-3, "kappa": 0.3}),
        stray_field_v_per_m=(10.0, -4.0, 0.0),
    )
    quoted = Trap(
        omega_hz=rod.mathieu(sr).secular_hz,
        axis_angle_rad=0.0,
        rf=rod.rf,
        dc=None,
        geometry=None,
        stray_field_v_per_m=rod.stray_field_v_per_m,
        shim_voltages_v={},
        reference_mass_u=sr.mass_u,
    )
    got, want = quoted.mathieu(ca), rod.mathieu(ca)
    assert got.mass_kg == want.mass_kg == ca.mass_u * ATOMIC_MASS_KG
    assert np.diag(got.q) == pytest.approx(np.diag(want.q), rel=1e-9) and abs(want.q[0, 0]) > 0.36
    assert np.diag(got.a) == pytest.approx(np.diag(want.a), rel=1e-9)
    assert got.beta == pytest.approx(want.beta, rel=1e-9) and got.C0 == pytest.approx(want.C0, rel=1e-9)
    assert quoted.micromotion_amplitude_m(ca) == pytest.approx(rod.micromotion_amplitude_m(ca), rel=1e-9)
    trap = dataclasses.replace(
        secular_trap((3.0e6, 2.9e6, 1.0e6)), rf=RfDrive(0.0, 60e6), reference_mass_u=sr.mass_u
    )
    crystal = solve_crystal(trap, (sr, ca))
    device = quiet_device(crystal, trap, Field(5.0, (1.0, 0.0, 0.0)), ())
    dk = np.array([2.0 * TWO_PI / 729e-9, 0.0, 0.0])
    etas, applied = lamb_dicke_parameters(device, 1, dk)
    assert applied
    for m in (crystal.mode_index("transverse_1", 0), crystal.mode_index("transverse_1", 1)):
        c0 = etas[m] / crystal.lamb_dicke(1, m, dk, micromotion=None)
        assert c0 == pytest.approx(1.019948, abs=1e-6)
    assert trap.mathieu(sr).C0[0] == pytest.approx(1.003867, abs=1e-6)


def test_the_explicit_path_refuses_a_mixed_crystal_whose_reference_mass_it_does_not_know() -> None:
    """Explicit frequencies with no reference mass describe whichever ion is asked about, so a crystal of two masses is
    refused by the solver and by the device; a reference mass on a rod trap, whose voltages set every mass, is refused
    too."""
    sr, ca = species("88Sr+"), species("40Ca+")
    trap = dataclasses.replace(secular_trap((3.0e6, 2.9e6, 1.0e6)), rf=RfDrive(0.0, 60e6))
    with pytest.raises(ValueError, match="reference_mass_u"):
        solve_crystal(trap, (sr, ca))
    named = dataclasses.replace(trap, reference_mass_u=sr.mass_u)
    crystal = solve_crystal(named, (sr, ca))
    with pytest.raises(ValueError, match="reference_mass_u"):
        quiet_device(crystal, trap, Field(5.0, (1.0, 0.0, 0.0)), ())
    assert quiet_device(crystal, named, Field(5.0, (1.0, 0.0, 0.0)), ()).trap is named
    with pytest.raises(ValueError, match="rod and surface paths derive every mass"):
        dataclasses.replace(_rod(), reference_mass_u=sr.mass_u)
    with pytest.raises(ValueError, match="positive"):
        dataclasses.replace(trap, reference_mass_u=0.0)


def test_rod_trap_path_follows_berkeland_and_the_phase_imbalance_term() -> None:
    yb = species("171Yb+")
    m = yb.mass_u * ATOMIC_MASS_KG
    omega_rf = TWO_PI * 20e6
    trap = _rod(phase_imbalance_rad=1e-3)
    assert trap.path == "linear"
    p = trap.mathieu(yb)
    q_x = 2 * E_C * 300.0 / (m * 1e-6 * omega_rf**2)
    a_z = 8 * E_C * 0.3 * 8.0 / (m * 4e-6 * omega_rf**2)
    assert p.q[0, 0] == pytest.approx(q_x) and p.q[1, 1] == pytest.approx(-q_x) and p.q[2, 2] == 0.0
    assert p.a[2, 2] == pytest.approx(a_z) and p.a[0, 0] == pytest.approx(-a_z / 2)
    assert p.secular_hz[2] == pytest.approx(math.sqrt(a_z) * 20e6 / 2, rel=1e-9)
    assert p.secular_hz[0] == pytest.approx(math.sqrt(-a_z / 2 + q_x**2 / 2) * 20e6 / 2, rel=1e-3)
    idx = trap.micromotion_beta(yb, np.array([K_369, 0.0, 0.0]))
    assert idx.in_phase == 0.0
    assert idx.out_of_phase == pytest.approx(K_369 * 0.25 * q_x * 1.0e-3 * 0.8 * 1e-3, rel=1e-12)
    bare = dataclasses.replace(
        trap, geometry=Electrodes("rod_quadrupole", {"R_m": 1.0e-3, "Z0_m": 2.0e-3, "kappa": 0.3})
    )
    with pytest.raises(ValueError, match="alpha"):
        bare.micromotion_beta(yb, np.array([K_369, 0.0, 0.0]))


def test_a_rod_or_blade_trap_refuses_shim_voltages_it_cannot_turn_into_a_field() -> None:
    """A rod trap refuses any non-zero shim voltage (no shim -> field map) and accepts an explicitly zero one, the residual
    field then being the stray field."""
    rod = dataclasses.replace(_rod(), stray_field_v_per_m=(5.0, 0.0, 0.0))
    for shims in ({"shim_x": 3.0}, {"endcaps": 1.0}, {"shim_x": -1e-9}):
        with pytest.raises(ValueError, match="no electrode model to convert shim voltages into fields"):
            dataclasses.replace(rod, shim_voltages_v=shims)
    assert rod.residual_field_v_per_m() == pytest.approx(np.array([5.0, 0.0, 0.0]))
    assert dataclasses.replace(
        rod, shim_voltages_v={"shim_x": 0.0}
    ).residual_field_v_per_m() == pytest.approx(np.array([5.0, 0.0, 0.0]))


def test_surface_trap_path_end_to_end_with_88sr() -> None:
    """On House's five-wire rails q matches House's Q11 (1e-6), a two-ion 88Sr+ chain sits collinear at the rf null, and a
    shim's field drives a signed in-phase index that a matching stray field cancels (1e-12)."""
    sr = species("88Sr+")
    m = sr.mass_u * ATOMIC_MASS_KG
    trap = _house_surface(m, shim_voltages_v={"left": 0.0, "right": 0.0})
    assert trap.path == "surface"
    p = trap.mathieu(sr)
    assert p.secular_hz[2] == pytest.approx(1.0e6, rel=1e-9)
    # House's Q11 = 0.2284260 at 1.46e-25 kg rescaled to the Sr mass
    assert abs(p.q[0, 0]) == pytest.approx(0.2284260 * 1.46e-25 / m, rel=1e-6)
    assert p.secular_hz[0] == pytest.approx(p.secular_hz[1], rel=1e-6), (
        "isotropic rf pseudopotential, symmetric dc"
    )
    assert np.allclose(np.abs(p.principal_axes[:, 2]), [0.0, 0.0, 1.0])
    cr = solve_crystal(trap, (sr, sr))
    assert len(cr.modes) == 6 and cr.collinear()
    assert cr.family("axial")[0].omega_hz == pytest.approx(1.0e6, rel=1e-9)
    assert np.allclose(cr.positions_m[:, 1], trap.rf_null_m()[1]), "the chain sits at the rf null height"
    shimmed = dataclasses.replace(trap, shim_voltages_v={"left": 0.5, "right": -0.5})
    field = shimmed.residual_field_v_per_m()
    assert abs(field[0]) > 0.0
    k = TWO_PI / 422e-9
    idx = shimmed.micromotion_beta(sr, np.array([k, 0.0, 0.0]))
    assert idx.in_phase != 0.0
    assert math.copysign(1.0, idx.in_phase) == -math.copysign(1.0, p.q[0, 0] * field[0])
    compensated = dataclasses.replace(shimmed, stray_field_v_per_m=(-field[0], -field[1], -field[2]))
    assert compensated.micromotion_beta(sr, np.array([k, 0.0, 0.0])).in_phase == pytest.approx(0.0, abs=1e-12)


# ---- the signed excess micromotion ---------------------------------------------------------------------------------------


def test_the_in_phase_amplitude_is_minus_half_q_u0_against_the_pseudopotential_spring() -> None:
    """u_1 = -(1/2) q u_0 with u_0 = Q E/(m (Omega/2)^2 (a + q^2/2)) to 1e-9, negative for q_x E_x > 0 and odd in the
    field."""
    yb = species("171Yb+")
    p = _yb_trap(1.0).mathieu(yb)
    q_x = float(p.q[0, 0])
    assert q_x > 0.0
    amp = _yb_trap(1.0).micromotion_amplitude_m(yb)
    spring = (TWO_PI * 10.1e6 / 2) ** 2 * (p.a[0, 0] + q_x**2 / 2)
    assert amp[0] == pytest.approx(-0.5 * q_x * E_C / (yb.mass_u * ATOMIC_MASS_KG * spring), rel=1e-9)
    assert amp[0] < 0.0
    assert _yb_trap(-1.0).micromotion_amplitude_m(yb)[0] == pytest.approx(-amp[0], rel=1e-12)
    assert np.allclose(amp[1:], 0.0)


def test_trap_micromotion_beta_scales_with_stray_field_and_wavevector() -> None:
    """The signed index -k (q/2) u_0 (1e-9) has Berkeland's magnitude 0.034 at 1 V/m and 369.5 nm (10 %), linear in the
    field and the wavevector, zero along z and odd in the field."""
    yb = species("171Yb+")
    p = _yb_trap(1.0).mathieu(yb)
    assert p.q[0, 0] == pytest.approx(0.28, abs=0.02), "q_x = 2 sqrt 2 omega_x/Omega ~ 0.28"
    b1 = _yb_trap(1.0).micromotion_beta(yb, np.array([K_369, 0.0, 0.0]))
    assert b1.out_of_phase == 0.0
    assert _yb_trap(10.0).micromotion_beta(yb, np.array([K_369, 0.0, 0.0])).in_phase == pytest.approx(
        10 * b1.in_phase
    )
    assert _yb_trap(1.0).micromotion_beta(yb, np.array([2 * K_369, 0.0, 0.0])).in_phase == pytest.approx(
        2 * b1.in_phase
    )
    assert _yb_trap(1.0).micromotion_beta(yb, np.array([0.0, 0.0, K_369])).in_phase == 0.0, "q_z = 0"
    assert _yb_trap(0.0).micromotion_beta(yb, np.array([K_369, 0.0, 0.0])).in_phase == 0.0
    omega_ps2 = (TWO_PI * 10.1e6 / 2) ** 2 * (p.a[0, 0] + p.q[0, 0] ** 2 / 2)
    expected = -K_369 * 0.5 * p.q[0, 0] * E_C / (yb.mass_u * ATOMIC_MASS_KG * omega_ps2)
    assert b1.in_phase == pytest.approx(expected, rel=1e-9) and expected < 0.0
    assert abs(b1.in_phase) == pytest.approx(0.034, rel=0.1)
    assert _yb_trap(-1.0).micromotion_beta(yb, np.array([K_369, 0.0, 0.0])).in_phase == pytest.approx(
        -b1.in_phase, rel=1e-12
    )
    assert _yb_trap(0.0).residual_field_v_per_m() == pytest.approx(np.zeros(3))


def test_the_modulation_index_is_signed_and_its_quadrature_pair_reproduces_both_terms() -> None:
    """beta = delta_k . u_1 is signed, and ``as_modulation`` turns (ip, op) into (beta, offset) with
    beta cos(theta + offset) = ip cos(theta) + op sin(theta) to 1e-12."""
    yb = species("171Yb+")
    dk = np.array([K_369, 0.0, 0.0])
    plus = _yb_trap(1.0).micromotion_beta(yb, dk)
    minus = _yb_trap(-1.0).micromotion_beta(yb, dk)
    assert plus.in_phase < 0.0 < minus.in_phase
    assert modulation_index(dk, np.array([-1e-9, 0.0, 0.0])) == pytest.approx(-K_369 * 1e-9, rel=1e-12)
    for ip, op in ((-0.4, 0.0), (0.4, 0.0), (0.3, 0.4), (-0.3, 0.4), (0.3, -0.4), (0.0, 0.0)):
        index = MicromotionIndex(ip, op)
        beta, offset = index.as_modulation()
        assert beta == pytest.approx(math.hypot(ip, op), rel=1e-12, abs=1e-15) == index.total
        for theta in np.linspace(0.0, TWO_PI, 17):
            assert beta * math.cos(theta + offset) == pytest.approx(
                ip * math.cos(theta) + op * math.sin(theta), rel=1e-12, abs=1e-15
            )


def test_micromotion_index_refuses_a_missing_geometry_instead_of_reporting_beta_zero() -> None:
    """An rf phase imbalance without the rod factors, or rod dc voltages the map cannot read, raises rather than reporting
    beta = 0; no rf record, or no wavevector, gives zero."""
    base = single_ion_raman_device(
        rf=RfDrive(frequency_hz=30e6, voltage_peak_v=100.0), stray=(50.0, 0.0, 0.0)
    )
    dk = np.asarray(derive_raman_drive(base, 0, (0, 1), scattering=False).delta_k, dtype=float)
    assert micromotion_index(base, 0, dk)[0] > 0.0
    no_rf = dataclasses.replace(base, trap=dataclasses.replace(base.trap, rf=None))
    assert micromotion_index(no_rf, 0, dk) == (0.0, 0.0)
    assert micromotion_index(base, 0, np.zeros(3)) == (0.0, 0.0), "a drive with no wavevector"
    imbalanced = dataclasses.replace(
        base,
        trap=dataclasses.replace(
            base.trap, rf=RfDrive(frequency_hz=30e6, voltage_peak_v=100.0, phase_imbalance_rad=1e-3)
        ),
    )
    with pytest.raises(ValueError, match="R_m and alpha"):
        micromotion_index(imbalanced, 0, dk)
    with pytest.raises(ValueError, match="R_m and alpha"):
        detection_micromotion(imbalanced, 0, imbalanced.beams[:1])
    rod = dataclasses.replace(
        base,
        trap=dataclasses.replace(
            base.trap,
            omega_hz=None,
            rf=RfDrive(frequency_hz=20e6, voltage_peak_v=300.0),
            dc=DcElectrodes({"endcaps": 8.0, "extra": 1.0}),
            geometry=Electrodes("rod_quadrupole", {"R_m": 1e-3, "Z0_m": 2e-3, "kappa": 0.3}),
        ),
    )
    with pytest.raises(ValueError, match="one endcap voltage"):
        micromotion_index(rod, 0, dk)


def test_a_derived_drive_raises_on_an_rf_record_it_cannot_evaluate_instead_of_reading_c0_one() -> None:
    """A 5 MHz rf drive under the 3 MHz radial frequency (outside the first stability region), rod dc voltages the map
    cannot read and an rf phase imbalance without the rod factors raise in the derived drive; only a trap with no rf record
    derives C0 = 1 and no micromotion."""
    base = single_ion_raman_device(
        rf=RfDrive(frequency_hz=30e6, voltage_peak_v=100.0), stray=(50.0, 0.0, 0.0)
    )
    dd = derive_raman_drive(base, 0, (0, 1), scattering=False)
    assert dd.c0_applied and dd.micromotion is not None
    no_rf = derive_raman_drive(
        dataclasses.replace(base, trap=dataclasses.replace(base.trap, rf=None)), 0, (0, 1), scattering=False
    )
    assert not no_rf.c0_applied and no_rf.micromotion is None
    assert no_rf.etas[KX] < dd.etas[KX], "C0 > 1 on the rf-derived x mode"
    unstable = dataclasses.replace(
        base, trap=dataclasses.replace(base.trap, rf=RfDrive(frequency_hz=5e6, voltage_peak_v=100.0))
    )
    with pytest.raises(UnstableMathieuError):
        derive_raman_drive(unstable, 0, (0, 1), scattering=False)
    rod = dataclasses.replace(
        base,
        trap=dataclasses.replace(
            base.trap,
            omega_hz=None,
            rf=RfDrive(frequency_hz=20e6, voltage_peak_v=300.0),
            dc=DcElectrodes({"endcaps": 8.0, "extra": 1.0}),
            geometry=Electrodes("rod_quadrupole", {"R_m": 1e-3, "Z0_m": 2e-3, "kappa": 0.3}),
        ),
    )
    with pytest.raises(ValueError, match="one endcap voltage"):
        derive_raman_drive(rod, 0, (0, 1), scattering=False)
    imbalanced = dataclasses.replace(
        base,
        trap=dataclasses.replace(
            base.trap, rf=RfDrive(frequency_hz=30e6, voltage_peak_v=100.0, phase_imbalance_rad=1e-3)
        ),
    )
    with pytest.raises(ValueError, match="R_m and alpha"):
        derive_raman_drive(imbalanced, 0, (0, 1), scattering=False)


def test_the_modulated_builder_first_micromotion_sideband_changes_sign_across_the_compensating_shim() -> None:
    """On the sigma_+ carrier element <up, 0|H|down, 0>, whose coefficient carries e^{+i beta cos(Omega_rf t + delta)} with the
    sign of the drive's e^{+i Delta k . x}, the first micromotion-sideband weight is 2 i J_1(beta) e^{i delta}/J_0(beta)
    (1e-9), flips sign across the compensating field (1e-9) and vanishes at zero field; the sigma_- element carries minus
    it."""
    base = single_ion_raman_device(
        rf=RfDrive(frequency_hz=30e6, voltage_peak_v=100.0), stray=(50.0, 0.0, 0.0)
    )
    space = HilbertSpace((2,), (ModeTruncation(KX, 6, (0, 2), 0.2),), None, (0, 2))
    assert base.trap.rf is not None
    omega_rf = base.trap.rf.omega_rad_s
    period = TWO_PI / omega_rf
    times = np.linspace(0.0, period, 129)[:-1]
    up_0 = int(np.ravel_multi_index((1, 0), space.dims))
    down_0 = int(np.ravel_multi_index((0, 0), space.dims))

    def first_harmonic(stray_x: float, element: tuple[int, int] = (up_0, down_0)) -> complex:
        """(2/T) int c(t) e^{-i Omega_rf t} dt / <c> of the element: its first micromotion-sideband weight."""
        trap = dataclasses.replace(base.trap, stray_field_v_per_m=(stray_x, 0.0, 0.0))
        dev = dataclasses.replace(base, trap=trap, crystal=solve_crystal(trap, base.crystal.species))
        dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
        drive = square_drive(dd, include_stark=False, rf_locked=True, rf_phase_rad=0.0)
        built = build_hamiltonian(
            dev,
            [Pulse(drive, 0.0, 5.0 * period, "p", ())],
            space,
            options=BuilderOptions(micromotion="modulated"),
        )
        vals = np.array([complex(built.drive_parts["p"](float(t)).full()[element]) for t in times])
        dc = complex(np.mean(vals))
        assert abs(dc) > 0.0
        return complex(2.0 * np.mean(vals * np.exp(-1j * omega_rf * times)) / dc)

    dd = derive_raman_drive(base, 0, (0, 1), scattering=False)
    assert dd.micromotion is not None
    beta, offset = dd.micromotion.as_modulation()
    assert dd.micromotion.in_phase < 0.0 and offset == pytest.approx(-math.pi, abs=1e-12)
    assert (beta, offset) == micromotion_index(base, 0, np.asarray(dd.delta_k, dtype=float))
    expected = 2j * jv(1, beta) * np.exp(1j * offset) / jv(0, beta)
    s_plus = first_harmonic(+50.0)
    assert abs(s_plus) > 1e-3
    assert s_plus == pytest.approx(expected, rel=1e-9)
    assert first_harmonic(-50.0) == pytest.approx(-expected, rel=1e-9), "a pi step across compensation"
    assert first_harmonic(0.0) == pytest.approx(0j, abs=1e-9)
    assert first_harmonic(+50.0, (down_0, up_0)) == pytest.approx(-expected, rel=1e-9)


# ---- the equilibrium's response to an added field --------------------------------------------------------------------------


def _field_response_fd(trap: Trap, ions: tuple, field: np.ndarray, step: float = 1e-3) -> np.ndarray:
    """The re-solved crystal's position shift per unit of ``field`` added to the stray field (a central difference)."""

    def positions(sign: float) -> np.ndarray:
        stray = np.asarray(trap.stray_field_v_per_m) + sign * step * field
        return solve_crystal(dataclasses.replace(trap, stray_field_v_per_m=tuple(stray)), ions).positions_m

    return (positions(1.0) - positions(-1.0)) / (2.0 * step)


def test_the_field_displacement_is_the_linear_response_of_the_re_solved_crystal_on_every_trap_path() -> None:
    """On the explicit, rod and surface paths, mixed crystals included, ``field_displacement_m`` is the shift of the re-solved
    equilibrium (1e-6): rigid along z, where the dc curvature is mass-independent, the lighter and stiffer 40Ca+ displaced
    less transversely, and an equal-mass chain in rotated axes moves rigidly by K^{-1} e E."""
    sr, ca, yb = species("88Sr+"), species("40Ca+"), species("171Yb+")
    field = np.array([3.0, -2.0, 5.0])
    explicit = dataclasses.replace(
        secular_trap((3.0e6, 2.9e6, 1.0e6)), rf=RfDrive(0.0, 60e6), reference_mass_u=sr.mass_u
    )
    surface = _house_surface(sr.mass_u * ATOMIC_MASS_KG)
    for trap, ions in ((explicit, (sr, ca)), (_rod(), (yb, ca)), (surface, (sr, ca)), (surface, (sr, sr))):
        shift = solve_crystal(trap, ions).field_displacement_m(field)
        expected = _field_response_fd(trap, ions, field)
        assert shift == pytest.approx(expected, rel=1e-6, abs=1e-6 * float(np.max(np.abs(expected))))
        assert shift[0, 2] == pytest.approx(shift[1, 2], rel=1e-12)
        if ions[1] is ca:
            assert np.all(np.abs(shift[1, :2]) < np.abs(shift[0, :2]))
    rotated = dataclasses.replace(secular_trap((3.0e6, 2.9e6, 1.0e6)), axis_angle_rad=0.3)
    chain = solve_crystal(rotated, (yb, yb, yb))
    axes = chain.principal_axes
    spring = yb.mass_u * ATOMIC_MASS_KG * (TWO_PI * np.asarray(rotated.omega_hz)) ** 2
    rigid = axes @ (axes.T @ (E_C * field) / spring)
    assert chain.field_displacement_m(field) == pytest.approx(np.tile(rigid, (3, 1)), rel=1e-12, abs=1e-20)


# ---- the trap quantities of Device.derived() ------------------------------------------------------------------------------


def _surface_device(stray_x: float = 0.0):
    base = chain_device(2)
    mass = base.crystal.species[0].mass_u * ATOMIC_MASS_KG
    trap = _house_surface(mass, stray_field_v_per_m=(stray_x, 0.0, 0.0))
    return dataclasses.replace(base, trap=trap, crystal=solve_crystal(trap, base.crystal.species))


def test_surface_device_reports_its_ion_height_and_trap_depth() -> None:
    """The surface device reports House's height 92.195445 um (1e-6 um) and depth 0.184465 eV at 1.46e-25 kg scaled as 1/m
    (1e-5); a device without an electrode plane reports neither."""
    dev = _surface_device()
    d = dev.derived()
    assert set(d.values) == set(d.provenance)
    assert d.values["ion_height_m"] == pytest.approx(five_wire_null_height_m(100e-6, 120e-6), rel=1e-12)
    assert d.values["ion_height_m"] * 1e6 == pytest.approx(92.195445, abs=1e-6)
    mass = dev.crystal.species[0].mass_u * ATOMIC_MASS_KG
    assert d.values["trap_depth_ev"] == pytest.approx(0.184465 * 1.46e-25 / mass, rel=1e-5)
    assert d.provenance["ion_height_m"] == d.provenance["trap_depth_ev"] == "conv.surface_electrode_geometry"
    plain = chain_device(2).derived()
    assert "ion_height_m" not in plain.values and "trap_depth_ev" not in plain.values


def test_second_order_doppler_and_the_pseudopotential_error_estimate() -> None:
    """``derived()`` reports <Delta nu/nu> = -<v^2>/(2 c^2) of the excess micromotion and |u_1|/s (1e-12), zero on the rf
    null and absent without an rf record."""
    base = chain_device(2)
    trap = dataclasses.replace(
        base.trap, rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6), stray_field_v_per_m=(20.0, 0.0, 0.0)
    )
    dev = dataclasses.replace(base, trap=trap, crystal=solve_crystal(trap, base.crystal.species))
    d = dev.derived()
    u1 = trap.micromotion_amplitude_m(dev.crystal.species[0])
    assert u1[0] < 0.0
    assert trap.rf is not None
    for i in range(dev.crystal.n_ions):
        got = d.values[f"second_order_doppler[{i}]"]
        assert got == pytest.approx(second_order_doppler_fraction(u1, trap.rf.omega_rad_s), rel=1e-12)
        assert got < 0.0
    pos = np.asarray(dev.crystal.positions_m, dtype=float)
    spacing = float(np.linalg.norm(pos[0] - pos[1]))
    assert d.values["pseudopotential_error"] == pytest.approx(float(np.linalg.norm(u1)) / spacing, rel=1e-12)
    assert 0.0 < d.values["pseudopotential_error"] < 1e-2
    assert _surface_device().derived().values["pseudopotential_error"] == 0.0
    plain = chain_device(2).derived()
    assert not any(k.startswith("second_order_doppler") for k in plain.values)
    assert "pseudopotential_error" not in plain.values
    assert any("need the rf record" in n for n in plain.notes)


def test_derived_raises_on_a_trap_record_it_cannot_evaluate_and_notes_only_an_absent_input() -> None:
    """A 5 MHz rf drive under the 3 MHz radial frequency, or a 40Ca+ ion at q = 1.2 beside the 171Yb+ the frequencies are
    quoted for, is a record derived() cannot evaluate: it raises instead of noting the secular frequencies, the Mathieu
    parameters or the micromotion amplitude away (the devices carry no gate beam, whose derivation would raise on its
    own). The absent inputs are still notes: no rf record, no detection beam."""
    base = microwave_device()
    unstable = dataclasses.replace(base.trap, rf=RfDrive(100.0, 5e6))
    dev = dataclasses.replace(base, trap=unstable, crystal=solve_crystal(unstable, base.crystal.species))
    with pytest.raises(UnstableMathieuError):
        dev.derived()
    yb, ca = species("171Yb+"), species("40Ca+")
    quoted = dataclasses.replace(_yb_trap(5.0), reference_mass_u=yb.mass_u)
    assert quoted.mathieu(yb).q[0, 0] == pytest.approx(0.2785, abs=1e-3)
    omega = TWO_PI * np.array(
        [[1.0e6, 1.0e6, 0.2e6], [4.3e6, 4.3e6, 0.2e6 * math.sqrt(yb.mass_u / ca.mass_u)]]
    )
    mixed = quiet_device(build_crystal((yb, ca), omega), quoted, Field(5.0, (1.0, 0.0, 0.0)), ())
    with pytest.raises(UnstableMathieuError):
        mixed.derived()
    plain = base.derived()
    assert [plain.values[f"secular_hz[{ax}]"] for ax in "xyz"] == [3.0e6, 2.9e6, 1.0e6]
    assert any("need the rf record" in n for n in plain.notes)
    assert any(
        n.startswith("ion 0: no beam of the device is near the 171Yb+ cycling line") for n in plain.notes
    )
    with pytest.raises(NoDetectionBeamError):
        detection_beams(base, 0)


def test_the_pseudopotential_error_grows_linearly_with_the_stray_field() -> None:
    values = []
    for stray in (10.0, 20.0, 40.0):
        base = chain_device(2)
        trap = dataclasses.replace(
            base.trap,
            rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6),
            stray_field_v_per_m=(stray, 0.0, 0.0),
        )
        dev = dataclasses.replace(base, trap=trap, crystal=solve_crystal(trap, base.crystal.species))
        values.append(dev.derived().values["pseudopotential_error"])
    # the ion spacing moves a little with the field, so this is linear to a per cent
    assert values[1] / values[0] == pytest.approx(2.0, rel=2e-2)
    assert values[2] / values[0] == pytest.approx(4.0, rel=4e-2)
