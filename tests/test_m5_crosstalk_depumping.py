"""The depumping half of Wineland's readout-crosstalk mechanism (PLAN.md Section 8.5).

Section 8.5: "Wineland's mechanism is a degradation of state-discrimination efficiency by a neighbour's scattered light of
different polarization, not a change in the instrumental eta_d; it is modelled as added counts on a dark neighbour PLUS
DEPUMPING-REDUCED N. Radiative pumping of a neighbour by a bright ion is bounded by I_ion/I_sat = 3 lambda^2/(8 pi^2 x^2)
with I_sat = pi h c Gamma/(3 lambda^3)". Before the 2026-09-07 M5 fix only the added counts existed and
``neighbour_intensity_ratio`` had no consumer at all (audit B4).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.readout.detection import depumped_model, neighbourhood_model, sample_register_records
from qutip_trap.readout.discriminate import ThresholdDiscriminator, register_confusion
from qutip_trap.readout.fluorescence import (
    neighbour_intensity_ratio,
    neighbour_pumping_rates,
    rates_from_detected,
)
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD
from tests.readout_fixtures import YB_DIRECT, crain_record_model

WINDOW_S = 22e-6
YB_WAVELENGTH_M = 369.5e-9
SPACING_M = 14e-6
"""Burrell's 14 um nearest-neighbour spacing, the same geometry as the 4.0 % PSF leakage row of Section 8.5."""


def test_the_pumping_a_bright_neighbour_drives_is_the_plans_bound_times_the_ions_own_rates() -> None:
    """R_d and R_b are linear in intensity with no saturation denominator (Section 8.1), so the leaked light drives the
    same two channels at s_neighbour/s_beam; s_neighbour is the plan's bound 3 lambda^2/(8 pi^2 x^2)."""
    rates = CRAIN_YB171_SNSPD.rates()
    s_beam = 2.45  # Crain's operating point in the plan's full-line convention (Section 8.8)
    s_nb = neighbour_intensity_ratio(YB_WAVELENGTH_M, SPACING_M)
    assert s_nb == pytest.approx(3.0 * YB_WAVELENGTH_M**2 / (8.0 * math.pi**2 * SPACING_M**2), rel=1e-12)
    d_d, d_b = neighbour_pumping_rates(rates, s_beam, YB_WAVELENGTH_M, SPACING_M)
    scale = s_nb / s_beam
    assert d_d == pytest.approx(rates.R_dark_pumping_per_s * scale, rel=1e-12)
    assert d_b == pytest.approx(rates.R_bright_pumping_per_s * scale, rel=1e-12)
    # at 14 um the bound is ~1.1e-5 of the beam, so one neighbour adds ~4 mHz to Crain's 341 Hz R_d: negligible but real
    assert 1e-6 < scale < 1e-4
    assert d_d / rates.R_dark_pumping_per_s == pytest.approx(d_b / rates.R_bright_pumping_per_s, rel=1e-12)
    # the polarization share only scales it down, and it is a share
    half = neighbour_pumping_rates(rates, s_beam, YB_WAVELENGTH_M, SPACING_M, polarization_purity=0.5)
    assert half[0] == pytest.approx(0.5 * d_d, rel=1e-12)
    with pytest.raises(ValueError, match="polarization share"):
        neighbour_pumping_rates(rates, s_beam, YB_WAVELENGTH_M, SPACING_M, polarization_purity=1.5)
    with pytest.raises(ValueError, match="saturation parameter"):
        neighbour_pumping_rates(rates, 0.0, YB_WAVELENGTH_M, SPACING_M)
    # the 111Cd+ anchor of Section 8.5, unchanged: 1.1e-4 at 4 um (Acton's printed 7e-4 is 2 pi too large)
    assert neighbour_intensity_ratio(214.5e-9, 4e-6) == pytest.approx(1.09e-4, rel=2e-2)


def test_depumped_model_touches_only_the_pumping_channels_and_only_for_bright_neighbours() -> None:
    rm = crain_record_model()
    extra = {1: (5.0e3, 2.0e2)}
    beside_bright = depumped_model([rm, rm], 0, ["dark", "bright"], extra)
    beside_dark = depumped_model([rm, rm], 0, ["dark", "dark"], extra)
    assert beside_dark is rm, "a dark neighbour scatters nothing, so it pumps nothing"
    assert beside_bright.detected_bright_per_s == rm.detected_bright_per_s
    assert beside_bright.background_per_s == rm.background_per_s, (
        "Wineland's mechanism is not a change of eta_d or of the background: the added counts are the other half"
    )
    assert beside_bright.rates[("bright", "dark")] == pytest.approx(
        rm.rates[("bright", "dark")] + 5.0e3, rel=1e-12
    )
    assert beside_bright.rates[("dark", "bright")] == pytest.approx(
        rm.rates[("dark", "bright")] + 2.0e2, rel=1e-12
    )
    # several bright neighbours add, because the rates are linear in intensity (Section 8.1)
    three = depumped_model([rm] * 3, 1, ["bright", "dark", "bright"], extra)
    assert three.rates[("bright", "dark")] == pytest.approx(
        rm.rates[("bright", "dark")] + 2.0 * 5.0e3, rel=1e-12
    )
    # without a depumping model nothing changes at all (the pre-fix behaviour, still the default)
    assert depumped_model([rm, rm], 0, ["dark", "bright"], None) is rm


def test_a_bright_ion_beside_a_bright_one_loses_photons_the_depumping_reduced_N() -> None:
    """The exact count distribution of the chain, no Monte Carlo: with the neighbour bright the ion's mean count falls
    because it is pumped dark during the window, which is Section 8.5's "depumping-reduced N"."""
    rm = crain_record_model()
    extra = {1: (5.0e3, 0.0)}
    alone = neighbourhood_model([rm, rm], 0, ["bright", "dark"], {}, extra)
    beside = neighbourhood_model([rm, rm], 0, ["bright", "bright"], {}, extra)
    n_alone = alone.mean_counts("bright", WINDOW_S)
    n_beside = beside.mean_counts("bright", WINDOW_S)
    assert n_beside < n_alone
    # dR_d * t = 0.11, so the ion spends about 5 % less time bright: a first-order, checkable loss
    assert n_beside / n_alone == pytest.approx(1.0 - 0.5 * 5.0e3 * WINDOW_S, rel=2e-2)
    # and the threshold error grows: the discrimination is degraded, which is Wineland's claim
    disc = ThresholdDiscriminator(0.5, WINDOW_S)
    eps_b_alone, _ = disc.error_rates(alone)
    eps_b_beside, _ = disc.error_rates(beside)
    assert eps_b_beside > eps_b_alone


def test_the_register_confusion_and_the_sampled_records_both_carry_the_depumping() -> None:
    rm = crain_record_model()
    schemes = [YB_DIRECT, YB_DIRECT]
    leak = {1: 0.04}
    extra = {1: (5.0e3, 0.0)}
    plain = register_confusion([rm, rm], schemes, ThresholdDiscriminator(0.5, WINDOW_S), leak)
    pumped = register_confusion(
        [rm, rm], schemes, ThresholdDiscriminator(0.5, WINDOW_S), leak, depumping=extra
    )
    assert plain.factored is not None and pumped.factored is not None
    # ion 0 in the bright level (|1> for 171Yb+) with its neighbour bright: declared bright LESS often once depumped
    p_plain = float(plain.factored.tables[0][1, 0, 0])
    p_pumped = float(pumped.factored.tables[0][1, 0, 0])
    assert p_pumped < p_plain
    # the sampled records agree with the frozen-neighbour form within statistics
    shots = 4000
    totals_bright_neighbour = []
    totals_dark_neighbour = []
    for s in range(shots):
        rngs = [np.random.default_rng((s, i, 17)) for i in range(2)]
        recs = sample_register_records(
            [rm, rm], ["bright", "bright"], WINDOW_S, rngs, leakage={}, depumping=extra
        )
        totals_bright_neighbour.append(recs[0].total)
        rngs = [np.random.default_rng((s, i, 17)) for i in range(2)]
        recs = sample_register_records(
            [rm, rm], ["bright", "dark"], WINDOW_S, rngs, leakage={}, depumping=extra
        )
        totals_dark_neighbour.append(recs[0].total)
    beside = neighbourhood_model([rm, rm], 0, ["bright", "bright"], {}, extra)
    alone = neighbourhood_model([rm, rm], 0, ["bright", "dark"], {}, extra)
    mean_b = float(np.mean(totals_bright_neighbour))
    mean_d = float(np.mean(totals_dark_neighbour))
    assert mean_b == pytest.approx(beside.mean_counts("bright", WINDOW_S), rel=0.05)
    assert mean_d == pytest.approx(alone.mean_counts("bright", WINDOW_S), rel=0.05)
    assert mean_b < mean_d


def test_rates_from_detected_feeds_the_bound_without_a_species_constant() -> None:
    """Section 8.8: the apparatus residual is a calibration parameter, never a species constant; the depumping bound is a
    pure function of the wavelength and the ion spacing, so it needs no fitted number at all."""
    ingested = rates_from_detected(472e3, 0.04356, dark_pumping_per_s=341.0, bright_pumping_per_s=16.4)
    d_d, d_b = neighbour_pumping_rates(ingested, 2.45, YB_WAVELENGTH_M, SPACING_M)
    assert d_d > 0.0 and d_b > 0.0
    assert d_d / d_b == pytest.approx(341.0 / 16.4, rel=1e-12)
    # the bound falls as 1/x^2: doubling the spacing quarters it
    far = neighbour_pumping_rates(ingested, 2.45, YB_WAVELENGTH_M, 2.0 * SPACING_M)
    assert far[0] == pytest.approx(d_d / 4.0, rel=1e-12)
