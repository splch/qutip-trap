"""The detection-histogram experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qutip_trap.experiments.result import DetectionHistogram, ExperimentResult, ScanParameters
from qutip_trap.machine import laboratory_kwargs

if TYPE_CHECKING:
    from qutip_trap.machine import Machine


def detection_histogram(machine: Machine, ion: int, n_records: int, **kw: Any) -> ExperimentResult:
    """Histogram ``n_records`` bright and dark photon counts of ``ion`` over candidate windows; returns a
    ``DetectionHistogram`` with the threshold and window minimizing the average error, and the fitted rates.

    ``detection_beams`` overrides the device's, ``scheme`` the ``ReadoutScheme``, ``levels`` the fine-structure levels,
    ``windows_s`` the candidate bin times (default 0.25 to 2.5 detector windows); ``data`` rows are the two histograms.
    """
    device, kw = laboratory_kwargs(machine, kw)
    from qutip_trap.calibration.readout import calibrate_detection
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.detection import RecordModel
    from qutip_trap.readout.fluorescence import detection_rates_for_ion
    from qutip_trap.run.job import detection_micromotion

    species = device.crystal.species[ion]
    beams = kw.get("detection_beams")
    if beams is None:
        beams = [device.beams[k] for k in detection_beams(device, ion)]
    # the micromotion factor the readout stage of ``run`` applies, so the fitted table matches the run that reads it
    beta, omega_rf = detection_micromotion(device, ion, beams)
    rates, scheme, _model = detection_rates_for_ion(
        species,
        device.field.B_gauss,
        device.field.direction,
        beams,
        position_m=tuple(float(x) for x in device.crystal.positions_m[ion]),
        levels=kw.get("levels"),
        scheme=kw.get("scheme"),
        micromotion_beta=beta,
        omega_rf_rad_s=omega_rf,
    )
    record_model = RecordModel.from_rates(rates, device.detector)
    windows = kw.get("windows_s")
    if windows is None:
        windows = tuple(float(x) for x in np.geomspace(0.25, 2.5, 12) * device.detector.window_s)
    cal = calibrate_detection(
        record_model,
        scheme,
        windows_s=windows,
        n_records=n_records,
        rng=np.random.default_rng(int(kw.get("seed", 0))),
        fitted_at_s=float(kw.get("t0_s", 0.0)),
        sample_id=int(kw.get("sample_id", 0)),
    )
    n = max(cal.bright_histogram.size, cal.dark_histogram.size)
    data = np.zeros((2, n))
    data[0, : cal.bright_histogram.size] = cal.bright_histogram
    data[1, : cal.dark_histogram.size] = cal.dark_histogram
    fitted = {name: (e.value, e.uncertainty) for name, e in cal.entries.items()}
    fitted["R_bright_scattered_per_s"] = (rates.R_bright_per_s, 0.0)
    return DetectionHistogram(
        data=data,
        fitted=fitted,
        model="detection_histogram",
        provenance_id="conv.readout_figure_of_merit",
        requested=ScanParameters({"windows_s": [float(w) for w in windows], "n_records": (n_records,)}),
        subject={"ion": int(ion)},
    )
