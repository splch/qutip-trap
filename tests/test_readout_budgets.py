"""Section 9.5 "Budgets" and the calibration of Section 7.5 item 5: Harty's and Christensen's tables close, Egan's line items,
register scaling, the apparatus presets, the detection calibration and the detection_histogram experiment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.calibration.readout import calibrate_detection, histogram_error_rates
from qutip_trap.experiments import detection_histogram
from qutip_trap.readout.discriminate import BudgetLine, ReadoutBudget
from qutip_trap.readout.fluorescence import shelf_decay_error
from qutip_trap.readout.presets import (
    BURRELL_CA40_CAMERA,
    CHRISTENSEN_BA133,
    CRAIN_YB171_SNSPD,
    EGAN_YB171,
    HARTY_CA43,
    MYERSON_CA40_PMT,
    PRESETS,
)
from tests.readout_fixtures import YB_DIRECT, crain_record_model, myerson_record_model, yb_readout_device


def test_harty_budget_closes_from_its_table_rows() -> None:
    """Harty 2014 Table I: 1.8 + 1.8 + 1.7 + 1.5 (x1e-4) = 6.8e-4, the measured SPAM error, rows state-averaged."""
    budget = ReadoutBudget(
        (
            BudgetLine.averaged("transfer to qubit (3 or 4 m.w. pi-pulses)", 1.8e-4, "Harty2014"),
            BudgetLine.averaged("transfer from qubit (4 m.w. pi-pulses)", 1.8e-4, "Harty2014"),
            BudgetLine.averaged("shelving transfer S(4,+4) -> D5/2", 1.7e-4, "Harty2014"),
            BudgetLine.averaged("time-resolved fluorescence detection", 1.5e-4, "Harty2014"),
        )
    )
    assert budget.error == pytest.approx(6.8e-4, abs=1e-12)
    assert budget.fidelity == pytest.approx(1.0 - 6.8e-4)
    assert budget.by_name()["shelving transfer S(4,+4) -> D5/2"] == (1.7e-4, 1.7e-4)
    assert HARTY_CA43.quoted_error == 6.8e-4


def test_christensen_budget_totals_3_4e_4_against_2_9e_4_measured_and_its_decay_row() -> None:
    rows = {
        "initialization to |0>": 0.1e-4,
        "|0> -> |1> CP Robust 180": 0.5e-4,
        "spontaneous decay during readout": 0.7e-4,
        "shelving |1>": 1.0e-4,
        "off-resonant shelving |0>": 1.0e-4,
        "readout of the S1/2 manifold": 0.1e-4,
    }
    budget = ReadoutBudget(tuple(BudgetLine.averaged(k, v, "Christensen2020") for k, v in rows.items()))
    assert budget.error == pytest.approx(3.4e-4, abs=1e-12)
    measured, sigma = 2.9e-4, 0.6e-4
    assert abs(budget.error - measured) < 1.0 * sigma
    # the decay row: 1 - exp(-4.5 ms/30 s) = 1.5e-4 for a |1> shot, 0.7e-4 state-averaged (Section 8.4)
    assert shelf_decay_error(4.5e-3, 30.0) == pytest.approx(1.5e-4, rel=2e-3)
    assert 0.5 * shelf_decay_error(4.5e-3, 30.0) == pytest.approx(0.75e-4, rel=2e-3)
    # the threshold overlap at bright mean 39 versus dark mean 1 is below 1e-6
    from qutip_trap.readout.detection import poisson_pmf

    n = np.arange(200)
    overlap_bright = float(np.sum(np.asarray(poisson_pmf(n, 39.0))[n <= 12]))
    overlap_dark = float(np.sum(np.asarray(poisson_pmf(n, 1.0))[n > 12]))
    assert overlap_bright < 1e-6 and overlap_dark < 1e-9
    assert CHRISTENSEN_BA133.threshold == 12.5 and CHRISTENSEN_BA133.window_s == 4.5e-3


def test_egan_line_items_and_the_asymmetric_report() -> None:
    """Egan's single-ion budget (Section 6.7): bright 0.71 % (pumping 0.55 %), dark 0.22 % (pumping 0.13 % + background 0.07 %);
    eps_B and eps_D are reported separately and averaged only at the end (Section 13)."""
    budget = ReadoutBudget(
        (
            BudgetLine("bright-to-dark pumping", 0.0055, 0.0, "Egan2021"),
            BudgetLine("other bright-state error", 0.0071 - 0.0055, 0.0, "Egan2021"),
            BudgetLine("dark-to-bright pumping", 0.0, 0.0013, "Egan2021"),
            BudgetLine("background", 0.0, 0.0007, "Egan2021"),
            BudgetLine("other dark-state error", 0.0, 0.0022 - 0.0020, "Egan2021"),
        )
    )
    assert budget.eps_B == pytest.approx(0.0071) and budget.eps_D == pytest.approx(0.0022)
    assert budget.error == pytest.approx(0.5 * (0.0071 + 0.0022))
    worst = ReadoutBudget(budget.lines, convention="min")
    assert worst.error == pytest.approx(0.0071)
    assert EGAN_YB171.quoted_error == pytest.approx(budget.error)
    # a 100 us window at R_d = 55 Hz would give 0.55 % if every pumping event lost the shot: R_d t = 5.5e-3
    assert 55.0 * 100e-6 == pytest.approx(0.0055)


def test_register_scaling_and_usable_size() -> None:
    budget = ReadoutBudget((BudgetLine.averaged("spam", 2.9e-4),))
    assert budget.register_fidelity(10) == pytest.approx((1.0 - 2.9e-4) ** 10)
    assert budget.usable_qubits() == pytest.approx(math.log(2.0) / 2.9e-4)
    assert 2000 < budget.usable_qubits() < 2500
    with pytest.raises(ValueError):
        BudgetLine("negative", -1e-4, 0.0)


def test_presets_carry_apparatus_data_with_sources() -> None:
    assert set(PRESETS) >= {MYERSON_CA40_PMT.name, CRAIN_YB171_SNSPD.name, BURRELL_CA40_CAMERA.name}
    for preset in PRESETS.values():
        assert preset.source and preset.species and 0.0 < preset.efficiency <= 1.0
    rates = MYERSON_CA40_PMT.rates()
    assert rates.R_bright_per_s == pytest.approx(55_800.0 / 0.0019)
    assert rates.shelf_decay_per_s == pytest.approx(1.0 / 1.168)
    assert rates.ceiling is None and rates.provenance[0] == "Myerson2008"
    crain = CRAIN_YB171_SNSPD.rates()
    assert crain.R_dark_pumping_per_s == 341.0 and crain.R_bright_pumping_per_s == 16.4
    # the scattered rate Crain's numbers imply is 0.088 Gamma, below the Gamma/4 ceiling
    from qutip_trap.species import species

    gamma = species("171Yb+").transition("S1/2-P1/2").partial_rate_rad_s
    assert crain.R_bright_per_s / gamma == pytest.approx(0.088, abs=1e-3)
    assert crain.R_bright_per_s < gamma / 4.0
    from qutip_trap.species.sources import SOURCES

    for key in (
        "Myerson2008",
        "Burrell2010",
        "Acton2006",
        "Noek2013",
        "Crain2019",
        "Egan2021",
        "Wineland1998",
    ):
        assert key in SOURCES


def test_calibrate_detection_recovers_the_rates_and_picks_the_interior_optimum() -> None:
    """10^4 bright and 10^4 dark records at Crain's operating point: the mean-count fit recovers eps R_o to 1 %, R_d within
    its uncertainty, and the chosen (n_c, t_b) sits at the exact model's interior optimum."""
    rm = crain_record_model(window_s=60e-6)
    windows = tuple(float(x) for x in np.linspace(6e-6, 60e-6, 10))
    cal = calibrate_detection(
        rm, YB_DIRECT, windows_s=windows, n_records=10_000, rng=np.random.default_rng(0)
    )
    e = cal.entries
    assert e["R_bright_detected_per_s"].value == pytest.approx(472e3, rel=0.01)
    assert e["R_dark_pumping_per_s"].value == pytest.approx(
        341.0, abs=4.0 * max(e["R_dark_pumping_per_s"].uncertainty, 60.0)
    )
    assert e["threshold"].value == 0.5
    assert 15e-6 <= e["window_s"].value <= 30e-6
    assert e["eps_B"].status == "calibrated" and e["eps_B"].experiment == "detection_histogram"
    assert 0.5 * (e["eps_B"].value + e["eps_D"].value) == pytest.approx(5.9e-4, abs=0.6e-4)
    assert cal.povm.per_ion is not None and cal.discriminator.n_c == 0.5
    lab_b, lab_d = histogram_error_rates(np.array([0, 1, 5, 7]), np.array([0, 0, 1, 0]), 0.5)
    assert lab_b == 0.25 and lab_d == 0.25
    assert cal.bright_histogram.sum() == 10_000 and cal.dark_histogram.sum() == 10_000


def test_calibrate_detection_on_the_shelving_scheme_reports_the_shelf_start() -> None:
    from tests.readout_fixtures import CA_OPTICAL

    rm = myerson_record_model()
    cal = calibrate_detection(
        rm,
        CA_OPTICAL,
        windows_s=(200e-6, 300e-6, 400e-6, 500e-6, 700e-6),
        n_records=2000,
        rng=np.random.default_rng(1),
    )
    assert cal.entries["R_dark_pumping_per_s"].value < 50.0
    assert 250e-6 <= cal.entries["window_s"].value <= 500e-6
    assert cal.optimum.best.eps < 3e-4


@pytest.mark.slow
def test_detection_histogram_experiment_runs_the_bloch_model_of_the_device_beams() -> None:
    """The experiment finds the 369.5 nm beam of the device, solves the rates at the ion's position and returns histograms, the
    threshold, the window and the fitted rates; the scattered rate is the Bloch solve's, below Gamma/4."""
    dev = yb_readout_device(2, s_o=2.45)
    res = detection_histogram(dev, 0, 2000, windows_s=tuple(np.linspace(10e-6, 60e-6, 6)), seed=0)
    assert res.model == "detection_histogram" and res.data.shape[0] == 2
    assert res.data[0].sum() == 2000 and res.data[1].sum() == 2000
    fitted = res.fitted
    from qutip_trap.species import species

    gamma = species("171Yb+").transition("S1/2-P1/2").partial_rate_rad_s
    assert 0.0 < fitted["R_bright_scattered_per_s"][0] < gamma / 4.0
    assert fitted["R_bright_detected_per_s"][0] == pytest.approx(
        dev.detector.efficiency * fitted["R_bright_scattered_per_s"][0], rel=0.05
    )
    assert fitted["threshold"][0] >= 0.5 and 10e-6 <= fitted["window_s"][0] <= 60e-6
    with pytest.raises(ValueError):
        detection_histogram(_no_detection_device(), 0, 500)


def _no_detection_device():  # type: ignore[no-untyped-def]
    from tests.m4_fixtures import chain_device

    return chain_device(2)
