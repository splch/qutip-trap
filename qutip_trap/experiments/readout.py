"""The detection-histogram experiment (PLAN.md Section 7.5 item 5)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.experiments.fitting import _detection_rates
from qutip_trap.experiments.result import DetectionHistogram, ExperimentResult, ScanParameters

if TYPE_CHECKING:
    from qutip_trap.machine import Machine


def detection_histogram(
    machine: Machine,
    ion: int,
    n_records: int,
    *,
    windows_s: Sequence[float] | None = None,
    seed: int = 0,
    t0_s: float = 0.0,
    sample_id: int = 0,
) -> ExperimentResult:
    """Histogram ``n_records`` bright and dark photon records of ``ion``'s simulated readout (the Bloch model of the device's
    detection beams, with the micromotion factor of Section 8.8 that ``run`` applies) and choose the threshold and the
    window among ``windows_s`` (default 0.25 to 2.5 detector windows) that minimize the average error; ``seed`` seeds the
    record generator, ``t0_s`` and ``sample_id`` stamp the entries. ``data`` holds the bright and dark histograms at the
    chosen window (rows); fitted: the threshold, window_s, eps_B, eps_D and the fitted rates, and the scattered bright rate."""
    from qutip_trap.calibration.readout import calibrate_detection
    from qutip_trap.readout.detection import RecordModel

    device = machine.device
    rates, scheme, _model = _detection_rates(device, ion, micromotion=True)
    if windows_s is None:
        windows_s = tuple(float(x) for x in np.geomspace(0.25, 2.5, 12) * device.detector.window_s)
    cal = calibrate_detection(
        RecordModel.from_rates(rates, device.detector),
        scheme,
        windows_s=windows_s,
        n_records=n_records,
        rng=np.random.default_rng(int(seed)),
        fitted_at_s=float(t0_s),
        sample_id=int(sample_id),
    )
    data = np.zeros((2, max(cal.bright_histogram.size, cal.dark_histogram.size)))
    data[0, : cal.bright_histogram.size] = cal.bright_histogram
    data[1, : cal.dark_histogram.size] = cal.dark_histogram
    fitted = {name: (e.value, e.uncertainty) for name, e in cal.entries.items()}
    fitted["R_bright_scattered_per_s"] = (rates.R_bright_per_s, 0.0)
    return DetectionHistogram(
        data=data,
        fitted=fitted,
        model="detection_histogram",
        provenance_id="conv.readout_figure_of_merit",
        requested=ScanParameters({"windows_s": [float(w) for w in windows_s], "n_records": (n_records,)}),
        subject={"ion": int(ion)},
    )
