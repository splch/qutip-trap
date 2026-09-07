"""The detection-histogram experiment (PLAN.md Section 7.5 item 5; M5)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qutip_trap.experiments.result import ExperimentResult

if TYPE_CHECKING:
    from qutip_trap.device.model import Device


def detection_histogram(device: Device, ion: int, n_records: int, **kw: Any) -> ExperimentResult:
    """Section 7.5 item 5 (M5): histogram bright and dark photon counts on the simulated readout model of ion ``ion`` and
    choose the threshold and window minimizing the average error.

    The rates come from the M3a Bloch model of the device's detection beams at the ion (``detection_beams`` overrides the
    beams near the species' cycling wavelength; ``scheme`` a :class:`~qutip_trap.readout.fluorescence.ReadoutScheme`;
    ``levels`` the included fine-structure levels; ``windows_s`` the candidate bin times, default 0.25 to 2.5 times the
    detector's window; ``seed`` the record generator). ``data`` holds the bright and dark histograms at the chosen window
    (rows) and ``fitted`` the threshold, window, eps_B, eps_D and the fitted rates with their uncertainties.
    """
    from qutip_trap.calibration.readout import calibrate_detection
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.detection import RecordModel
    from qutip_trap.readout.fluorescence import detection_rates_for_ion

    species = device.crystal.species[ion]
    beams = kw.get("detection_beams")
    if beams is None:
        beams = [device.beams[k] for k in detection_beams(device, ion)]
    rates, scheme, _model = detection_rates_for_ion(
        species,
        device.field.B_gauss,
        device.field.direction,
        beams,
        position_m=tuple(float(x) for x in device.crystal.positions_m[ion]),
        levels=kw.get("levels"),
        scheme=kw.get("scheme"),
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
    return ExperimentResult(
        data=data, fitted=fitted, model="detection_histogram", provenance_id="conv.readout_figure_of_merit"
    )


__all__ = ["detection_histogram"]
