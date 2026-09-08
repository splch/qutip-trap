"""The default-on resonance checker and integrated-phase estimator (PLAN.md Sections 4.1.4, 5.7; Section 9.12 row
"Cubic anharmonicity gate"; audit item E.5).

Section 5.7: "H_anh: the cubic Coulomb coupling of 4.1.4, opt-in because it couples all three mode families into one
space; the resonance checker and the estimated accumulated phase are ON BY DEFAULT." Section 9.12 makes the DIRECTLY
INTEGRATED phase the fixture and keeps g^2 t/Delta_res only as an explicit upper bound: at g/2pi = 1.419 kHz over 100 us
the integrated phase on |1, 0> is 0.0041, 0.0034, 0.0032, 0.0032 rad at mismatches 1, 0.3, 0.1, 0.03 MHz - essentially
flat, against the bound's 0.0013, 0.0042, 0.0127, 0.0422 rad (3x low at 1 MHz, 13x high at 0.03 MHz) - and 0.016 rad on
the resonant pair |2, 0> <-> |0, 1> at 0.1 MHz. The withdrawn first-order figure g t is 0.89 rad.

Before this milestone's fix those numbers lived only inside ``validation/scripts/check_anharmonic.py``, a standalone
script that does not import the package, and nothing in ``dynamics/`` called the checker at all.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from qutip_trap.api import HilbertSpace, ModeTruncation, Pulse
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.trap.anharmonic import (
    anharmonic_estimate,
    coulomb_anharmonic_terms,
    dispersive_phase_bound_rad,
    integrated_cubic_phase_rad,
)
from qutip_trap.units import TWO_PI
from tests.m2_fixtures import two_ion_raman_device

G_RAD_S = TWO_PI * 1.419e3
"""The plan's 40Ca+ value: g = eps omega_z at omega_z = 2 pi x 2 MHz (Section 4.1.4)."""
OMEGA_A_RAD_S = TWO_PI * 2.0e6
DURATION_S = 100e-6


@pytest.mark.parametrize(
    ("mismatch_hz", "phase_rad", "bound_rad"),
    [(1.0e6, 0.0041, 0.0013), (0.3e6, 0.0034, 0.0042), (0.1e6, 0.0032, 0.0127), (0.03e6, 0.0032, 0.0422)],
)
def test_integrated_cubic_phase_of_the_section_9_12_fixture(
    mismatch_hz: float, phase_rad: float, bound_rad: float
) -> None:
    """The four printed integrated phases at their printed digits, and the withdrawn second-order label beside them."""
    omega_b = 2.0 * OMEGA_A_RAD_S - TWO_PI * mismatch_hz
    got = integrated_cubic_phase_rad(G_RAD_S, OMEGA_A_RAD_S, omega_b, DURATION_S)
    assert got == pytest.approx(phase_rad, abs=5e-5), "the printed digits of check_anharmonic.out"
    assert dispersive_phase_bound_rad(G_RAD_S, TWO_PI * mismatch_hz, DURATION_S) == pytest.approx(
        bound_rad, abs=5e-5
    ), "g^2 t/Delta_res is an upper bound, not the scaling"
    # the withdrawn first-order figure, the negative control of the row
    assert G_RAD_S * DURATION_S == pytest.approx(0.892, abs=5e-4)


def test_the_integrated_phase_is_flat_in_the_mismatch_and_the_resonant_pair_is_larger() -> None:
    """The integrated phase varies by 22 % over a factor 33 in Delta where the g^2 t/Delta label varies by 33x; the
    |2, 0> <-> |0, 1> pair, which a^2 b^dag does couple, accumulates 0.016 rad at a 0.1 MHz mismatch."""
    phases = [
        integrated_cubic_phase_rad(G_RAD_S, OMEGA_A_RAD_S, 2.0 * OMEGA_A_RAD_S - TWO_PI * d, DURATION_S)
        for d in (1.0e6, 0.03e6)
    ]
    assert max(phases) / min(phases) < 1.35, "essentially flat in the mismatch"
    bounds = [dispersive_phase_bound_rad(G_RAD_S, TWO_PI * d, DURATION_S) for d in (1.0e6, 0.03e6)]
    assert bounds[1] / bounds[0] == pytest.approx(1.0e6 / 0.03e6, rel=1e-9)
    resonant = integrated_cubic_phase_rad(
        G_RAD_S,
        OMEGA_A_RAD_S,
        2.0 * OMEGA_A_RAD_S - TWO_PI * 0.1e6,
        DURATION_S,
        state=(2, 0),
    )
    assert resonant == pytest.approx(-0.01589, abs=5e-6)
    assert abs(resonant) == pytest.approx(0.016, abs=5e-4)
    # and the Ballance sanity constraint the row names: a coherent 0.89 rad phase would cost 0.19 infidelity
    assert math.sin(0.5 * G_RAD_S * DURATION_S) ** 2 == pytest.approx(0.19, abs=5e-3)
    with pytest.raises(ValueError, match="does not fit the truncations"):
        integrated_cubic_phase_rad(G_RAD_S, OMEGA_A_RAD_S, OMEGA_A_RAD_S, DURATION_S, state=(9, 0))


def test_the_builder_reports_the_estimate_whether_or_not_the_cubic_term_is_built() -> None:
    """Section 5.7 "on by default": the checker and the estimate reach ``approximations`` for both settings of
    ``include_anharmonic``, and ``AnharmonicTerms.resonance_check=False`` is the switch that silences them."""
    base = two_ion_raman_device()
    terms = coulomb_anharmonic_terms(base.crystal)
    assert terms.cubic_rad_s and terms.resonance_check is True, "on by default"
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
    # the cubic coefficient is SIGNED (D_222 < 0 for a two-ion stretch mode); only the mismatch must be off resonance
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
    # a trap with no anharmonic record has nothing to check
    built = build_hamiltonian(base, pulses, space, options=BuilderOptions())
    assert not any("anharmonic resonance check" in a for a in built.approximations)


def test_the_estimate_reports_a_tuned_resonance_as_linear_in_time() -> None:
    """omega_z,stretch = 2 omega_x,rock for two ions: the checker lists the triple inside the coupling width, where the
    growth is linear in time and the dispersive estimate does not apply."""
    from qutip_trap.species import species
    from qutip_trap.trap.crystal import solve_crystal
    from tests.test_crystal import _explicit

    ca = species("40Ca+")
    tuned = solve_crystal(_explicit((math.sqrt(1.75) * 1.0e6, 1.5e6, 1.0e6)), (ca, ca))
    terms = coulomb_anharmonic_terms(tuned)
    est = anharmonic_estimate(tuned, terms, DURATION_S)
    assert est is not None and est.resonances, "the tuned triple is inside the coupling width"
    assert abs(est.resonances[0].mismatch_hz) < est.width_hz
    assert "LINEAR in time" in est.summary()
    quiet = solve_crystal(_explicit((5e6, 5.5e6, 2e6)), (ca, ca))
    quiet_est = anharmonic_estimate(quiet, coulomb_anharmonic_terms(quiet), DURATION_S)
    assert quiet_est is not None and quiet_est.resonances == ()
    assert "no mode triple within" in quiet_est.summary()
