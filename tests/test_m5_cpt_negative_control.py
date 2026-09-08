"""The Section 9.5 "CPT" row's second clause: the lumped F_CPT(eta) = F_no-CPT(eta/3) model is ILLEGAL beside the Bloch
solve (PLAN.md Sections 8.1, 9.5).

Section 8.1: "Coherent population trapping in the F = 1 -> F' = 0 detection cycle caps the 171Yb+ scattering rate at
Gamma/4, the saturation ceiling n_e/(n_e + n_g) of the three-ground-plus-one-excited manifold ... (Olmschenk et al. 2007
quote one third, which is the Lambda repump cycle's ceiling and not this manifold's; the lumped 'eta -> eta/3' model of
the earlier revisions under-predicts detected counts by 3 and is illegal whenever R_o comes from the Bloch solve, whose
assertion is R_o <= Gamma/4; at Crain's fitted s_o = 0.815 the corrected R_o alone gives 0.0880 Gamma, which times
eps_sys = 4.356 % is his 472 kcps)".

The first clause (R_o <= Gamma/4) is pinned in ``tests/test_readout_rates.py``; the second had no test anywhere before the
2026-09-07 M5 audit. Note the plan's own slip, already recorded as ``anchor.m5.yb171_rate_object``: the point
R_o = 0.0880 Gamma is Noek's s = 0.815, that is s_o = 2.45 in the plan's full-line convention (Section 8.8).
"""

from __future__ import annotations

import pytest

from qutip_trap.readout.detection import Detector, RecordModel
from qutip_trap.readout.fluorescence import (
    FluorescenceRates,
    saturation_ceiling,
    yb171_bright_rate_closed,
)
from qutip_trap.units import TWO_PI

GAMMA_S_RAD_S = TWO_PI * 19.62e6
"""Partial S1/2 <-> P1/2 rate of 171Yb+ (PLAN.md Section 8.1: 0.99499 of the 19.72 MHz total from the 8.07(9) ns
lifetime, Section 9.12). Written out here so the negative control needs no species table."""
S_O_CRAIN = 2.45
"""Noek's s = 0.815 in the plan's full-line convention (Section 8.8, "two saturation parameters, never aliased")."""
EPS_SYS_CRAIN = 0.04356


def snspd(efficiency: float = EPS_SYS_CRAIN) -> Detector:
    return Detector(
        kind="snspd",
        efficiency=efficiency,
        background_cps=4.2,
        psf_leakage={},
        dead_time_s=None,
        afterpulse_prob=None,
        window_s=22e-6,
    )


def test_the_lumped_eta_over_three_model_is_a_factor_three_below_the_bloch_solve_rate() -> None:
    r_o = yb171_bright_rate_closed(S_O_CRAIN, GAMMA_S_RAD_S)
    assert r_o / GAMMA_S_RAD_S == pytest.approx(0.0880, rel=3e-3), "Section 9.5's 0.0880 Gamma"
    detected = EPS_SYS_CRAIN * r_o
    assert detected == pytest.approx(472e3, rel=3e-2), "Crain's 472 kcps"
    lumped = (EPS_SYS_CRAIN / 3.0) * r_o
    assert lumped == pytest.approx(157e3, rel=3e-2)
    assert detected / lumped == pytest.approx(3.0, rel=1e-12), (
        "the lumped model under-predicts detected counts by exactly 3 (Section 8.1)"
    )


def test_no_readout_code_path_applies_eta_over_three() -> None:
    """epsilon_sys enters exactly once and unmodified (Section 13, "Detection efficiency"): the detected rate is linear in
    ``Detector.efficiency`` with slope R_o, which leaves no room for a hidden 1/3."""
    rates = FluorescenceRates(
        R_bright_per_s=yb171_bright_rate_closed(S_O_CRAIN, GAMMA_S_RAD_S),
        R_dark_pumping_per_s=341.0,
        R_bright_pumping_per_s=16.4,
    )
    for eff in (EPS_SYS_CRAIN, 2.0 * EPS_SYS_CRAIN, 0.5):
        det = snspd(eff)
        assert rates.detected(det)[0] == pytest.approx(eff * rates.R_bright_per_s, rel=1e-12)
        model = RecordModel.from_rates(rates, det)
        assert model.detected_bright_per_s == pytest.approx(eff * rates.R_bright_per_s, rel=1e-12)
        assert model.background_per_s == pytest.approx(det.background_cps, rel=1e-12)
    # and the ceiling the solve asserts against is this manifold's 1/4, never the Lambda cycle's 1/3 (Olmschenk 2007)
    assert saturation_ceiling(3, 1) == pytest.approx(0.25, rel=1e-12)
    assert saturation_ceiling(2, 1) == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert saturation_ceiling(1, 1) == pytest.approx(0.5, rel=1e-12)
    assert yb171_bright_rate_closed(1e9, GAMMA_S_RAD_S) < 0.25 * GAMMA_S_RAD_S


def test_the_lumped_model_would_break_the_ceiling_bookkeeping_it_claims_to_encode() -> None:
    """The lumped model replaces the ceiling by a change of eta, so it predicts the SAME scattered rate and a smaller
    detected one; the Bloch solve does the opposite (the ceiling is on R_o, eta is untouched), and the two cannot be
    combined without applying the 1/3 twice."""
    r_o = yb171_bright_rate_closed(S_O_CRAIN, GAMMA_S_RAD_S)
    both = (EPS_SYS_CRAIN / 3.0) * r_o
    assert both / (EPS_SYS_CRAIN * r_o) == pytest.approx(1.0 / 3.0, rel=1e-12)
    # the saturated rate already carries the 1/4 ceiling, so a further 1/3 would put it below the Lambda ceiling's share
    saturated = yb171_bright_rate_closed(1e9, GAMMA_S_RAD_S)
    assert saturated == pytest.approx(0.25 * GAMMA_S_RAD_S, rel=1e-8)
    assert saturated / 3.0 < saturation_ceiling(2, 1) * GAMMA_S_RAD_S / 3.0 + 1.0
