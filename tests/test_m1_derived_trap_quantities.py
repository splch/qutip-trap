"""``Device.derived()`` reports the Section 4.1 "with every device" trap quantities (audit item E.9).

Three clauses of Section 4.1 name quantities the module must report with every device that uses it, none of which
``derived()`` carried: the ion height and trap depth of 4.1.6, the implied second-order Doppler shift of 4.1.1, and the
pseudopotential error estimate that closes 4.1.7. Each key carries a ledger id (``tests/test_provenance_and_layout.py``
enforces that the id exists), and the one quantity whose plan wording is ambiguous - "in units of the mode spacing" -
records the reading taken in ``conv.pseudopotential_error_estimate``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.trap.micromotion import second_order_doppler_fraction
from qutip_trap.trap.pseudopotential import DcElectrodes, RfDrive
from qutip_trap.trap.surface import Electrodes, five_wire_depth_j, five_wire_null_height_m
from qutip_trap.units import ATOMIC_MASS_KG, E_C, TWO_PI
from tests.m4_fixtures import two_ion_device

HOUSE_A, HOUSE_B = 100e-6, 120e-6


def _surface_device(stray_x: float = 0.0):  # type: ignore[no-untyped-def]
    base = two_ion_device()
    mass = base.crystal.species[0].mass_u * ATOMIC_MASS_KG
    kzz = mass * (TWO_PI * 1.0e6) ** 2 / E_C
    geometry = Electrodes(
        "surface_five_wire",
        {
            "a_m": HOUSE_A,
            "b_m": HOUSE_B,
            "dc_curvature_xx_v_per_m2": -kzz / 2,
            "dc_curvature_yy_v_per_m2": -kzz / 2,
            "dc_curvature_zz_v_per_m2": kzz,
        },
    )
    trap = dataclasses.replace(
        base.trap,
        omega_hz=None,
        rf=RfDrive(300.0, 50e6),
        dc=DcElectrodes({"centre": 0.0}),
        geometry=geometry,
        stray_field_v_per_m=(stray_x, 0.0, 0.0),
        shim_voltages_v={},
    )
    return dataclasses.replace(base, trap=trap, crystal=solve_crystal(trap, base.crystal.species))


def test_surface_device_reports_its_ion_height_and_trap_depth() -> None:
    """Section 4.1.6: "the module reports h and the trap depth with every device that uses it"."""
    dev = _surface_device()
    d = dev.derived()
    assert set(d.values) == set(d.provenance)
    h = five_wire_null_height_m(HOUSE_A, HOUSE_B)
    assert d.values["ion_height_m"] == pytest.approx(h, rel=1e-12)
    assert d.values["ion_height_m"] * 1e6 == pytest.approx(92.195445, abs=1e-6), "House's y0"
    mass = dev.crystal.species[0].mass_u * ATOMIC_MASS_KG
    depth = five_wire_depth_j(HOUSE_A, HOUSE_B, 300.0, mass, TWO_PI * 50e6)
    assert d.values["trap_depth_ev"] == pytest.approx(depth / E_C, rel=1e-9)
    assert d.provenance["ion_height_m"] == d.provenance["trap_depth_ev"] == "conv.surface_electrode_geometry"
    # the explicit-frequency path has no electrode plane: no height, no depth, and no claim to either
    plain = two_ion_device().derived()
    assert "ion_height_m" not in plain.values and "trap_depth_ev" not in plain.values


def test_second_order_doppler_and_the_pseudopotential_error_estimate() -> None:
    """Section 4.1.1 "reports the implied second-order Doppler shift for completeness" and the 4.1.7 close,
    "the pseudopotential error estimate (q/2 times the configured off-null displacement ...) with every device"."""
    base = two_ion_device()
    trap = dataclasses.replace(
        base.trap, rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6), stray_field_v_per_m=(20.0, 0.0, 0.0)
    )
    dev = dataclasses.replace(base, trap=trap, crystal=solve_crystal(trap, base.crystal.species))
    d = dev.derived()
    sp = dev.crystal.species[0]
    u1 = trap.micromotion_amplitude_m(sp)
    assert u1[0] < 0.0, "the signed amplitude the estimate is built from (Section 13)"
    for i in range(dev.crystal.n_ions):
        got = d.values[f"second_order_doppler[{i}]"]
        assert got == pytest.approx(second_order_doppler_fraction(u1, trap.rf.omega_rad_s), rel=1e-12)
        assert got < 0.0, "<Delta nu/nu> = -<v^2>/(2 c^2) is negative by construction"
    pos = np.asarray(dev.crystal.positions_m, dtype=float)
    spacing = float(np.linalg.norm(pos[0] - pos[1]))
    assert d.values["pseudopotential_error"] == pytest.approx(float(np.linalg.norm(u1)) / spacing, rel=1e-12)
    assert 0.0 < d.values["pseudopotential_error"] < 1e-2, (
        "a small parameter, as the plan's derivation assumes"
    )
    # on the rf null with no stray field the estimate is EXACTLY zero: the pseudopotential modes are then exact
    on_null = _surface_device()
    assert on_null.derived().values["pseudopotential_error"] == 0.0
    # a device with no rf record cannot derive either, and says so instead of reporting a wrong number
    plain = two_ion_device().derived()
    assert not any(k.startswith("second_order_doppler") for k in plain.values)
    assert "pseudopotential_error" not in plain.values
    assert any("need the rf record" in n for n in plain.notes)


def test_the_estimate_grows_linearly_with_the_stray_field_and_with_q() -> None:
    """|u_1|/s = (q/2)|u_0|/s: linear in the residual field at fixed q (u_0 = Q E/(m omega_ps^2))."""
    values = []
    for stray in (10.0, 20.0, 40.0):
        base = two_ion_device()
        trap = dataclasses.replace(
            base.trap,
            rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6),
            stray_field_v_per_m=(stray, 0.0, 0.0),
        )
        dev = dataclasses.replace(base, trap=trap, crystal=solve_crystal(trap, base.crystal.species))
        values.append(dev.derived().values["pseudopotential_error"])
    # the ion spacing moves a little with the field, so this is linear to a per cent, not exactly
    assert values[1] / values[0] == pytest.approx(2.0, rel=2e-2)
    assert values[2] / values[0] == pytest.approx(4.0, rel=4e-2)
    assert all(math.isfinite(v) for v in values)
