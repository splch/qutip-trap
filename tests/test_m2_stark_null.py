"""The two weak Section 9.2 rows: the photons-per-Stark-radian saturation and the 9Be+ polarization null.

Section 9.2 "Scattering rates" ends with "``Gamma_total/Delta_St -> 0.0154`` for 9Be+" and Section 9.2 "Stark shifts"
asks for the 9Be+ eight-term differential shift, its ``omega_0 << Delta`` limit and its polarization null. Neither
existed anywhere in the repository (M2 audit E9). What can be closed is closed here:

* the exact identity behind the ratio, ``R_SE/|delta_(0<->0)| = gamma/omega_0`` for ANY detuning and polarization
  (Wineland Eqs. 2.17-2.18 share one prefactor and one bracket), and the plan's saturated 0.9579 gamma/Delta_hf;
* the qualitative claim the plan makes for the |2,2> <-> |1,1> line - it HAS a polarization null - asserted against
  the derived ``AtomicStructure`` sum on the 9Be+ fixture, with the clock line as the negative control (it is
  polarization independent and therefore unnullable, PLAN.md:685).

Wineland Eq. 2.11's eight coefficient sets are NOT transcribed: PLAN.md:1885 lists Eqs. 2.5-2.9 and 2.11-2.12 as
untranscribed, so the plan carries no numbers to reproduce and none are invented (ledger
anchor.m2.wineland_eight_term_stark).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import TWO_PI
from qutip_trap.validation.atomic_closed_forms import (
    OZERI_PHOTONS_PER_RADIAN_COEFFICIENT,
    ozeri_photons_per_stark_radian,
    wineland_clock_light_shift,
    wineland_photons_per_stark_radian,
    wineland_r_se_clock,
)
from tests.atomic_fixtures import (
    be9_like,
    beam_at_detuning,
    fine_structure_omega,
)

GAMMA_MONROE_HZ = 19.4e6
"""Gamma/2pi of the 9Be+ 313 nm line as PLAN.md 4.2.1 and 9.13 quote it (Monroe 1995); species/be9.py cites it."""
DELTA_HF_LANGER_HZ = 1.207495843e9
"""The 9Be+ clock-transition frequency at the field-independent point B0 = 11,944.6 uT (Langer 2005; species/be9.py)."""
DELTA_HF_ZERO_FIELD_HZ = 2.0 * 625.008837048e6
"""|A|(I + 1/2) = 1.250018 GHz, the ZERO-FIELD ground hyperfine splitting (species/be9.py S12.A_hfs_hz)."""


def test_photons_per_stark_radian_identity_and_the_saturated_coefficient() -> None:
    """R_SE/|delta_00| = gamma/omega_0 exactly; the far-detuned clock ratio saturates at 0.9579 gamma/Delta_hf."""
    gamma = TWO_PI * GAMMA_MONROE_HZ
    omega_f = TWO_PI * 198e9
    omega_0 = TWO_PI * DELTA_HF_LANGER_HZ
    # the identity, over four decades of detuning and three polarization pairs: it is exact, so 1e-12
    for x in (0.1, 0.414213562373, 2.0, 50.0, -3.0):
        delta = x * omega_f
        for g_b, g_r in ((1.0, 1.0), (1.0, 0.3), (0.7, 0.0)):
            r_se = wineland_r_se_clock(gamma, g_b, g_r, delta, omega_f)
            shift = wineland_clock_light_shift(g_b, g_r, omega_0, delta, omega_f)
            if shift == 0.0:
                continue
            assert r_se / abs(shift) == pytest.approx(
                wineland_photons_per_stark_radian(gamma, omega_0), rel=1e-12
            ), (x, g_b, g_r)
    assert wineland_photons_per_stark_radian(gamma, omega_0) == pytest.approx(
        GAMMA_MONROE_HZ / DELTA_HF_LANGER_HZ, rel=1e-15
    )
    # Ozeri's saturated value is a 4.2 percent correction to that identity, not a factor error
    assert OZERI_PHOTONS_PER_RADIAN_COEFFICIENT == pytest.approx(0.9579, abs=5e-5)
    ratio = ozeri_photons_per_stark_radian(GAMMA_MONROE_HZ, DELTA_HF_LANGER_HZ)
    assert ratio == pytest.approx(0.0154, abs=5e-5), "the 9.2 row's 0.0154 photons per radian for 9Be+"
    # the printed 0.015396 needs Delta_hf = 1.20702 GHz; Langer's clock-point splitting reproduces it to 4e-4, the
    # zero-field |A|(I + 1/2) not at all. A published-number anchor: it reports, it does not fail (Section 9's rule).
    assert ratio == pytest.approx(0.015396, rel=1e-3)
    assert ozeri_photons_per_stark_radian(GAMMA_MONROE_HZ, DELTA_HF_ZERO_FIELD_HZ) == pytest.approx(
        0.014867, rel=1e-4
    )
    with pytest.raises(ZeroDivisionError):
        ozeri_photons_per_stark_radian(gamma, 0.0)


def _differential_shift(structure: AtomicStructure, lower: str, upper: str, beams) -> float:  # type: ignore[no-untyped-def]
    lo = structure.state(lower)
    up = structure.state(upper)
    return structure.light_shift_rad_s(up, beams) - structure.light_shift_rad_s(lo, beams)


def _elliptical_beam(species, delta_rad_s: float, angle_rad: float, ground_energy_hz: float):  # type: ignore[no-untyped-def]
    """A beam along B = z whose polarization runs from pure sigma+ (angle 0) to pure sigma- (angle pi/2).

    Propagating along B admits only sigma components, so the ellipticity angle is the one knob Wineland's null needs:
    the differential shift of a Zeeman-type qubit changes sign with the sigma+/sigma- imbalance.
    """
    pol = (
        complex(math.cos(angle_rad) / math.sqrt(2.0), 0.0),
        complex(0.0, math.cos(angle_rad) / math.sqrt(2.0)),
        0.0 + 0.0j,
    )
    pol_minus = (
        complex(math.sin(angle_rad) / math.sqrt(2.0), 0.0),
        complex(0.0, -math.sin(angle_rad) / math.sqrt(2.0)),
        0.0 + 0.0j,
    )
    total = tuple(a + b for a, b in zip(pol, pol_minus))
    return beam_at_detuning(species, delta_rad_s, (0.0, 0.0, 1.0), total, ground_energy_hz=ground_energy_hz)


def test_the_9be_zeeman_qubit_shift_has_a_polarization_null_and_the_clock_one_does_not() -> None:
    """PLAN.md:447 and 685: the |2,2> <-> |1,1> differential shift has a polarization null; the clock shift, sharing one
    prefactor and one bracket with R_SE, is polarization INDEPENDENT and therefore unnullable.

    Both statements are asserted against the derived explicit sum, not against a transcribed eight-term formula.
    """
    sp = be9_like()
    st = AtomicStructure(sp, 1.0, (0.0, 0.0, 1.0))
    omega_f = fine_structure_omega(sp)
    delta = 0.414213562373 * omega_f
    e_ground = st.state("S1/2 F=2 mF=2").energy_hz

    def zeeman(angle: float) -> float:
        beam = _elliptical_beam(sp, delta, angle, e_ground)
        return _differential_shift(st, "S1/2 F=1 mF=1", "S1/2 F=2 mF=2", (beam,))

    # pure sigma+ and pure sigma- bracket a sign change: there is a null in between
    assert zeeman(0.0) * zeeman(math.pi / 2.0) < 0.0, (zeeman(0.0), zeeman(math.pi / 2.0))
    null = brentq(zeeman, 0.0, math.pi / 2.0, xtol=1e-14)
    assert 0.0 < null < math.pi / 2.0
    assert zeeman(null) == pytest.approx(0.0, abs=1e-6 * abs(zeeman(0.0)))
    # the null is a genuine crossing, not a coincidence of the endpoints: the shift is monotone through it
    assert zeeman(null - 0.05) * zeeman(null + 0.05) < 0.0

    # the clock line: the same polarization scan never crosses zero, because Eq. 2.17 carries no polarization
    # dependence at all - the summed line strength out of each clock state is the same
    def clock(angle: float) -> float:
        beam = _elliptical_beam(sp, delta, angle, st.state("S1/2 F=2 mF=0").energy_hz)
        return _differential_shift(st, "S1/2 F=1 mF=0", "S1/2 F=2 mF=0", (beam,))

    values = np.array([clock(a) for a in np.linspace(0.0, math.pi / 2.0, 41)])
    assert np.all(values > 0.0) or np.all(values < 0.0), (
        "the clock shift is unnullable: no polarization in this family sends it through zero",
        values,
    )
    # both differential shifts come ONLY from the three-index detuning: the total line strength out of any lower
    # sublevel, summed over upper sublevels and polarization components, is the same constant (PLAN.md:647's sum rule),
    # so a single Delta per intermediate LEVEL would return zero for either pair
    for lower, upper in (("S1/2 F=1 mF=0", "S1/2 F=2 mF=0"), ("S1/2 F=1 mF=1", "S1/2 F=2 mF=2")):
        for angle in (0.0, 0.3, math.pi / 4.0, math.pi / 2.0):
            beam = _elliptical_beam(sp, delta, angle, st.state(upper).energy_hz)
            lo = sum(abs(om) ** 2 for _e, om, _d in st.couplings_from(st.state(lower), beam))
            hi = sum(abs(om) ** 2 for _e, om, _d in st.couplings_from(st.state(upper), beam))
            assert lo == pytest.approx(hi, rel=1e-12), (lower, angle)
    # what differs is the per-SUBLEVEL distribution against the sublevels' own detunings, and for the Zeeman pair that
    # redistribution reverses the sign of the weighted difference; for the clock pair it does not
    assert zeeman(0.0) > 0.0 > zeeman(math.pi / 2.0) or zeeman(0.0) < 0.0 < zeeman(math.pi / 2.0)
