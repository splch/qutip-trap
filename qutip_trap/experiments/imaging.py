"""The crystal image: the calibration experiment that detects dark and lost ions (PLAN.md Section 6.7).

The chain is illuminated by the detection beams for ten detection windows. With a camera (a detector carrying ``pixel_m``
and an objective NA) the exposure is a pixel image (``CameraGeometry``: PSF weights integrated over the pixels, background
and read noise), each ion's region of interest summed and its centroid taken; with a PMT or SNSPD the image is the vector
of per-ion counts. An ion in ``run_state.dark`` scatters no light, one in ``run_state.lost`` is absent
and an ion of a species no detection beam addresses is dark by wavelength. Two identical ions that swapped are invisible to
the image (the mode structure detects them, not the camera), and the result says so.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Unpack

import numpy as np

from qutip_trap.experiments.fitting import _detection_rates
from qutip_trap.experiments.result import CrystalImage, ExperimentResult
from qutip_trap.experiments.single_ion import _Lab, _LabOptions

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.machine import Machine
    from qutip_trap.run.results import RunState

_ROI_PIXELS = 3
"""The pixels of an ion's region of interest on the camera."""


def _ion_rates(device: Device, ion: int) -> tuple[float, float] | None:
    """(detected bright photon rate, background rate) of ``ion`` under the detection beams, None when no beam addresses it."""
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.detection import RecordModel

    try:
        detection_beams(device, ion)
    except ValueError:
        return None
    model = RecordModel.from_rates(_detection_rates(device, ion)[0], device.detector)
    return float(model.detected_bright_per_s), float(model.background_per_s)


def crystal_image(
    machine: Machine, *, run_state: RunState | None = None, **kw: Unpack[_LabOptions]
) -> ExperimentResult:
    """Image the chain in ``run_state`` (default nominal) and compare it with the nominal crystal (Section 6.7); with
    ``shots`` the counts are Poisson draws, else exact means. Fitted: n_ions, n_bright, n_dark, n_lost and per ion
    bright[i] (1 or 0), counts[i] and, on a camera, position_m[i]; data: the image (camera) or the per-ion counts."""
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.run.results import RunState

    lab = _Lab.of(machine, kw)
    device = lab.device
    obs = lab.obs
    n = device.crystal.n_ions
    state = run_state or RunState.nominal(n)
    exposure = 10.0 * device.detector.window_s
    notes: list[str] = []
    rates = [_ion_rates(device, i) for i in range(n)]
    present = [i for i in range(n) if i not in state.lost]
    lit = [i for i in present if i not in state.dark and rates[i] is not None]
    bg = max([r[1] for r in rates if r is not None], default=0.0)
    det = device.detector
    rng = (
        np.random.default_rng(obs._rng("crystal_image", 0, 0).integers(0, 2**32 - 1))
        if obs.shots is not None
        else None
    )
    fitted: dict[str, tuple[float, float]] = {"n_ions": (float(n), 0.0)}
    if det.pixel_m is not None and det.numerical_aperture is not None:
        from qutip_trap.readout.detection import CameraGeometry

        positions = np.asarray(device.crystal.positions_m, dtype=float)
        axis = positions[-1] - positions[0] if n > 1 else np.array([0.0, 0.0, 1.0])
        along = positions @ (axis / (np.linalg.norm(axis) or 1.0))
        # the shortest detection wavelength on the addressed ions (with none addressed no ion is lit and the PSF is
        # never evaluated)
        wavelength = min(
            (
                device.beams[k].wavelength_m
                for i in present
                if rates[i] is not None
                for k in detection_beams(device, i)
            ),
            default=369.5e-9,
        )
        span = float(along.max() - along.min()) if n > 1 else 0.0
        geometry = CameraGeometry(
            tuple(float(x) for x in along),
            det.pixel_m,
            max(8, int(math.ceil(span / det.pixel_m)) + 8),
            5,
            wavelength,
            numerical_aperture=det.numerical_aperture,
        )
        mean = np.full((geometry.n_rows, geometry.n_columns), det.read_noise_counts, dtype=float)
        mean += bg * exposure / geometry.n_pixels
        for i in lit:
            r = rates[i]
            assert r is not None
            mean += geometry.weights(i) * r[0] * exposure
        image = rng.poisson(mean).astype(float) if rng is not None else mean
        xs, _ys = geometry.pixel_centres()
        bg_roi = (bg * exposure / geometry.n_pixels + det.read_noise_counts) * _ROI_PIXELS
        for i in range(n):
            roi = geometry.roi(i, _ROI_PIXELS)
            counts = float(np.sum(image.ravel()[roi]))
            bright = counts > bg_roi + 5.0 * math.sqrt(max(bg_roi, 1.0))
            fitted[f"counts[{i}]"] = (counts, math.sqrt(max(counts, 1.0)))
            fitted[f"bright[{i}]"] = (1.0 if bright else 0.0, 0.0)
            if bright:
                w = np.maximum(image.ravel()[roi] - bg_roi / _ROI_PIXELS, 0.0)
                fitted[f"position_m[{i}]"] = (
                    float(np.sum(w * xs.ravel()[roi]) / max(np.sum(w), 1e-300)),
                    det.pixel_m / math.sqrt(max(np.sum(w), 1.0)),
                )
        data = image
    else:
        data = np.zeros(n)
        for i in range(n):
            r = rates[i]
            mean_i = bg * exposure
            if i in lit and r is not None:
                mean_i += r[0] * exposure
            c, s = obs.counts(mean_i, "crystal_image", i, i)
            data[i] = c
            fitted[f"counts[{i}]"] = (c, s)
            fitted[f"bright[{i}]"] = (
                1.0 if c > bg * exposure + 5.0 * math.sqrt(max(bg * exposure, 1.0)) else 0.0,
                0.0,
            )
        notes.append("no camera on the device: the image is the per-ion photon count (a PMT per ion)")
    bright_ions = [i for i in range(n) if fitted[f"bright[{i}]"][0] > 0.0]
    dark_seen = [i for i in range(n) if rates[i] is not None and i not in bright_ions]
    fitted["n_bright"] = (float(len(bright_ions)), 0.0)
    fitted["n_dark"] = (float(len([i for i in dark_seen if i not in state.lost])), 0.0)
    fitted["n_lost"] = (float(len(state.lost)), 0.0)
    if dark_seen:
        notes.append(
            f"ions {dark_seen} read dark under the illuminating light (dark-ion or loss event, Section 6.7)"
        )
    if len(set(sp.name for sp in device.crystal.species)) == 1:
        notes.append(
            "identical ions: a reorder is invisible to the image (the mode structure detects it, Section 6.7)"
        )
    return CrystalImage(
        data=np.asarray(data),
        fitted=fitted,
        model="crystal_image",
        provenance_id="conv.readout_crosstalk_added_counts",
        converged=len(dark_seen) == 0 and not state.lost,
        notes=tuple(notes),
    )
