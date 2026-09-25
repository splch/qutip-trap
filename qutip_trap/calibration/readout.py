"""Detection calibration: histogram bright and dark records, fit the rates, choose threshold and window (PLAN.md Section 7.5
item 5; milestone M5).

The calibration draws ``n_records`` bright and ``n_records`` dark records from the simulated readout model (the laboratory's
10^4 and 10^4), fits the mean-count curve n̄(τ) = εR_o[(R_b/k) τ + (R_d/k^2)(1 - e^{-kτ})] of Section 8.2 to the bright
records over several windows for (εR_o, R_d, R_b), and chooses (n_c, t_b) at the minimum of the average error of the FITTED
MODEL's exact count distributions. The histograms are the laboratory's own estimate of (eps_B, eps_D) at that point and are
reported beside the model's (``histogram_errors``, and the sigma of the table's ``eps_B``/``eps_D`` entries), so the two
agree only to counting statistics: the choice of (n_c, t_b) is the model's, never the histogram's (M5 fix 2026-09-07; the
first version of this docstring claimed both chose). The entries are stored in ``CalibrationTable.detection`` with their
uncertainties, status and provenance, and the POVM of Section 5.7 is what the fast path of ``run`` reads. Sits above
``control`` and ``dynamics`` like the rest of ``calibration/`` (Section 3.2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qutip_trap.control.table import CalEntry
from qutip_trap.readout.detection import PhotonRecord, RecordModel
from qutip_trap.readout.discriminate import (
    POVM,
    ThresholdDiscriminator,
    ThresholdOptimum,
    optimize_threshold,
    product_povm,
)
from qutip_trap.readout.fluorescence import ReadoutClass, ReadoutScheme, fit_mean_count_curve


@dataclass(frozen=True)
class DetectionCalibration:
    """What the detection experiment fits (Section 7.5 item 5)."""

    entries: dict[str, CalEntry]
    """threshold, window_s, eps_B, eps_D, R_bright_detected_per_s, R_dark_pumping_per_s, R_bright_pumping_per_s."""
    bright_histogram: np.ndarray
    dark_histogram: np.ndarray
    optimum: ThresholdOptimum
    povm: POVM
    fitted_model: RecordModel
    """The record model rebuilt from the FITTED rates (what the scheduler and the fast path see), not the true one."""
    histogram_errors: tuple[float, float] = (0.0, 0.0)
    """(eps_B, eps_D) read straight off the two histograms at the chosen (n_c, t_b), the laboratory's own estimate; the
    table's entries carry the fitted MODEL's values, whose difference from these is counting statistics (Section 7.5 item 5)."""

    @property
    def discriminator(self) -> ThresholdDiscriminator:
        return self.optimum.discriminator


def histogram_error_rates(
    bright_counts: np.ndarray, dark_counts: np.ndarray, n_c: float
) -> tuple[float, float]:
    """The laboratory's estimate: fraction of bright records at or below n_c and of dark records above it."""
    return float(np.mean(bright_counts <= n_c)), float(np.mean(dark_counts > n_c))


def calibrate_detection(
    model: RecordModel,
    scheme: ReadoutScheme,
    *,
    windows_s: Sequence[float],
    n_records: int = 10_000,
    fit_windows_s: Sequence[float] | None = None,
    n_fit_records: int | None = None,
    rng: np.random.Generator | None = None,
    fitted_at_s: float = 0.0,
    sample_id: int = 0,
    dark_start: ReadoutClass | None = None,
) -> DetectionCalibration:
    """Draw records, fit (εR_o, R_d, R_b) from the bright mean-count curve, pick (n_c, t_b) on the fitted model.

    ``windows_s`` are the candidate bin times; the records are drawn at the longest one and truncated for the shorter ones
    (as a time-tagged laboratory record would be), so one set of 2 x n_records shots serves every window. ``dark_start`` is
    the class the scheme's dark qubit level starts in ("dark" or "shelf"); inferred from the scheme when None.
    """
    if n_records < 100:
        raise ValueError("a calibration needs at least 100 records per state (Section 7.5 draws 1e4)")
    gen = rng if rng is not None else np.random.default_rng(0)
    windows = sorted(float(w) for w in windows_s)
    if not windows or windows[0] <= 0.0:
        raise ValueError("windows are positive")
    t_max = windows[-1]
    if dark_start is None:
        other = scheme.start_distribution(1 - scheme.bright_level)
        start: ReadoutClass = max(other, key=other.__getitem__)
    else:
        start = dark_start
    sub_bin = t_max / 200.0
    bright_records = [model.sample_record("bright", t_max, gen, sub_bin_s=sub_bin) for _ in range(n_records)]
    dark_records = [model.sample_record(start, t_max, gen, sub_bin_s=sub_bin) for _ in range(n_records)]

    def stacked(records: Sequence[PhotonRecord]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for r in records:
            assert r.sub_bins is not None
            rows.append(np.asarray(r.sub_bins, dtype=np.int64))
        return np.vstack(rows)

    def counts_at(records: np.ndarray, t: float, bin_s: float) -> np.ndarray:
        """Per record, the count within ``t`` of the window start (the sub-bin sums, exact integers)."""
        k = int(round(t / bin_s))
        return np.asarray(records[:, :k].sum(axis=1), dtype=np.int64)

    bright_stack, dark_stack = stacked(bright_records), stacked(dark_records)

    # the mean-count fit needs windows long against 1/R_d and 1/R_b (Noek fitted n̄(τ) out to 60 ms): at bin-time windows the
    # curvature is invisible and (R_d, R_b) are degenerate, so a separate, longer set of bright records serves the fit
    fit_windows = sorted(
        float(w)
        for w in (fit_windows_s if fit_windows_s is not None else np.geomspace(t_max, 300.0 * t_max, 12))
    )
    t_fit = fit_windows[-1]
    fit_bin = t_fit / 400.0
    n_fit = n_records if n_fit_records is None else n_fit_records
    fit_records = stacked(
        [model.sample_record("bright", t_fit, gen, sub_bin_s=fit_bin) for _ in range(n_fit)]
    )
    taus = np.array(fit_windows)
    nbar = np.array([counts_at(fit_records, t, fit_bin).mean() for t in fit_windows])
    guess = (float(nbar[0] / taus[0]), 1.0 / t_fit, 0.05 / t_fit)
    (r0, rd, rb), (s0, sd, sb) = fit_mean_count_curve(taus.tolist(), nbar.tolist(), guess=guess)
    fitted_rates = {k: v for k, v in model.rates.items()}
    fitted_rates[("bright", "dark")] = rd
    if ("dark", "bright") in fitted_rates or rb > 0.0:
        fitted_rates[("dark", "bright")] = rb
    fitted = RecordModel(
        detected_bright_per_s=r0,
        background_per_s=model.background_per_s,
        rates={k: v for k, v in fitted_rates.items() if v > 0.0},
        dead_time_s=model.dead_time_s,
        afterpulse_prob=model.afterpulse_prob,
    )
    optimum = optimize_threshold(fitted, windows, dark_start=start)
    n_c, t_b = optimum.best.n_c, optimum.best.window_s
    hb, hd = counts_at(bright_stack, t_b, sub_bin), counts_at(dark_stack, t_b, sub_bin)
    eps_b_lab, eps_d_lab = histogram_error_rates(hb, hd, n_c)
    sigma_b = math.sqrt(max(eps_b_lab * (1.0 - eps_b_lab), 1.0 / n_records) / n_records)
    sigma_d = math.sqrt(max(eps_d_lab * (1.0 - eps_d_lab), 1.0 / n_records) / n_records)

    def entry(value: float, unc: float, pid: str) -> CalEntry:
        return CalEntry(value, unc, "calibrated", "detection_histogram", pid, fitted_at_s, sample_id)

    entries = {
        "threshold": entry(n_c, 0.0, "conv.readout_figure_of_merit"),
        "window_s": entry(t_b, 0.0, "conv.readout_figure_of_merit"),
        "eps_B": entry(optimum.best.eps_B, sigma_b, "conv.readout_figure_of_merit"),
        "eps_D": entry(optimum.best.eps_D, sigma_d, "conv.readout_figure_of_merit"),
        "R_bright_detected_per_s": entry(r0, s0, "conv.mean_count_curve"),
        "R_dark_pumping_per_s": entry(rd, sd, "conv.mean_count_curve"),
        "R_bright_pumping_per_s": entry(rb, sb, "conv.mean_count_curve"),
    }
    povm = product_povm([fitted], [scheme], optimum.discriminator)
    return DetectionCalibration(
        entries=entries,
        bright_histogram=np.bincount(hb),
        dark_histogram=np.bincount(hd),
        optimum=optimum,
        povm=povm,
        fitted_model=fitted,
        histogram_errors=(eps_b_lab, eps_d_lab),
    )
