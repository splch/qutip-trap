"""The crystal image, which detects dark, lost and reordered ions: a camera's pixel image, or per-ion counts on a PMT or
SNSPD. A species no detection beam addresses reads dark, which is how a mixed-species reorder shows."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from qutip_trap.experiments.result import CrystalImage, ExperimentResult
from qutip_trap.experiments.single_ion import _observation
from qutip_trap.machine import laboratory_kwargs

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.machine import Machine


def _ion_rates(device: Device, ion: int) -> tuple[float, float] | None:
    """(detected bright photon rate, background rate) of ``ion`` under the detection beams, None when no beam addresses it."""
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.detection import RecordModel
    from qutip_trap.readout.fluorescence import detection_rates_for_ion

    try:
        idx = detection_beams(device, ion)
    except ValueError:
        return None
    rates, _scheme, _model = detection_rates_for_ion(
        device.crystal.species[ion],
        device.field.B_gauss,
        device.field.direction,
        [device.beams[k] for k in idx],
        position_m=tuple(float(x) for x in device.crystal.positions_m[ion]),
    )
    model = RecordModel.from_rates(rates, device.detector)
    return float(model.detected_bright_per_s), float(model.background_per_s)


def crystal_image(machine: Machine | Device, **kw: Any) -> ExperimentResult:
    """Image the chain once (no scan) against the nominal crystal; returns a ``CrystalImage``.

    ``run_state`` the machine state to image (default nominal), ``exposure_s`` (default ten detection windows), ``species``
    the species whose light is on (default every detection beam), ``psf_sigma_m`` a Gaussian PSF for a camera without an
    NA. Fitted n_ions, n_bright, n_dark, n_lost, bright[i], counts[i] and position_m[i] (camera only).
    """
    device, kw = laboratory_kwargs(machine, kw, caller=crystal_image)
    from qutip_trap.run.results import RunState

    n = device.crystal.n_ions
    run_state: RunState = kw.get("run_state") or RunState.nominal(n)
    exposure = float(kw.get("exposure_s", 10.0 * device.detector.window_s))
    obs = _observation(device, kw)
    species_filter = kw.get("species")
    notes: list[str] = []
    rates: list[tuple[float, float] | None] = []
    for i in range(n):
        if species_filter is not None and device.crystal.species[i].name != species_filter:
            rates.append(None)
            continue
        rates.append(_ion_rates(device, i))
    present = [i for i in range(n) if i not in run_state.lost]
    lit = [i for i in present if i not in run_state.dark and rates[i] is not None]
    bg = max([r[1] for r in rates if r is not None], default=0.0)
    det = device.detector
    camera = det.pixel_m is not None and (
        det.numerical_aperture is not None or kw.get("psf_sigma_m") is not None
    )
    rng = (
        np.random.default_rng(obs._rng("crystal_image", 0, 0).integers(0, 2**32 - 1))
        if obs.shots is not None
        else None
    )
    fitted: dict[str, tuple[float, float]] = {"n_ions": (float(n), 0.0)}
    if camera:
        from qutip_trap.readout.detection import CameraGeometry

        positions = np.asarray(device.crystal.positions_m, dtype=float)
        axis = positions[-1] - positions[0] if n > 1 else np.array([0.0, 0.0, 1.0])
        axis = axis / (np.linalg.norm(axis) or 1.0)
        along = positions @ axis
        wavelength = min(
            (
                device.beams[k].wavelength_m
                for i in present
                for k in (
                    __import__("qutip_trap.light.roles", fromlist=["detection_beams"]).detection_beams(
                        device, i
                    )
                    if rates[i] is not None
                    else ()
                )
            ),
            default=369.5e-9,
        )
        assert det.pixel_m is not None
        span = float(along.max() - along.min()) if n > 1 else 0.0
        n_cols = int(kw.get("n_columns", max(8, int(math.ceil(span / det.pixel_m)) + 8)))
        geometry = CameraGeometry(
            tuple(float(x) for x in along),
            det.pixel_m,
            n_cols,
            int(kw.get("n_rows", 5)),
            wavelength,
            numerical_aperture=det.numerical_aperture if kw.get("psf_sigma_m") is None else None,
            psf_sigma_m=kw.get("psf_sigma_m"),
        )
        mean = np.full((geometry.n_rows, geometry.n_columns), det.read_noise_counts, dtype=float)
        mean += bg * exposure / geometry.n_pixels
        for i in lit:
            r = rates[i]
            assert r is not None
            mean += geometry.weights(i) * r[0] * exposure
        image = rng.poisson(mean).astype(float) if rng is not None else mean
        roi_pixels = int(kw.get("roi_pixels", 3))
        xs, _ys = geometry.pixel_centres()
        for i in range(n):
            roi = geometry.roi(i, roi_pixels)
            counts = float(np.sum(image.ravel()[roi]))
            bg_roi = (bg * exposure / geometry.n_pixels + det.read_noise_counts) * roi_pixels
            bright = counts > bg_roi + 5.0 * math.sqrt(max(bg_roi, 1.0))
            fitted[f"counts[{i}]"] = (counts, math.sqrt(max(counts, 1.0)))
            fitted[f"bright[{i}]"] = (1.0 if bright else 0.0, 0.0)
            if bright:
                w = np.maximum(image.ravel()[roi] - bg_roi / roi_pixels, 0.0)
                fitted[f"position_m[{i}]"] = (
                    float(np.sum(w * xs.ravel()[roi]) / max(np.sum(w), 1e-300)),
                    det.pixel_m / math.sqrt(max(np.sum(w), 1.0)),
                )
        data = image
    else:
        counts_vec = np.zeros(n)
        for i in range(n):
            r = rates[i]
            mean_i = bg * exposure
            if i in lit and r is not None:
                mean_i += r[0] * exposure
            if i in run_state.lost:
                mean_i = bg * exposure
            c, s = obs.counts(mean_i, "crystal_image", i, i)
            counts_vec[i] = c
            bright = c > bg * exposure + 5.0 * math.sqrt(max(bg * exposure, 1.0))
            fitted[f"counts[{i}]"] = (c, s)
            fitted[f"bright[{i}]"] = (1.0 if bright else 0.0, 0.0)
        data = counts_vec
        notes.append("no camera on the device: the image is the per-ion photon count (a PMT per ion)")
    bright_ions = [i for i in range(n) if fitted[f"bright[{i}]"][0] > 0.0]
    nominal_bright = [i for i in range(n) if rates[i] is not None]
    dark_seen = [i for i in nominal_bright if i not in bright_ions]
    fitted["n_bright"] = (float(len(bright_ions)), 0.0)
    fitted["n_dark"] = (float(len([i for i in dark_seen if i not in run_state.lost])), 0.0)
    fitted["n_lost"] = (float(len(run_state.lost)), 0.0)
    if dark_seen:
        notes.append(
            f"ions {dark_seen} read dark under the illuminating light (dark-ion or loss event, Section 6.7)"
        )
    if species_filter is not None:
        expected = [i for i in range(n) if device.crystal.species[i].name == species_filter]
        if sorted(bright_ions) != sorted(expected):
            notes.append(
                f"the bright pattern {bright_ions} differs from the nominal {expected} positions of {species_filter}: a reorder or a dark ion"
            )
    elif len(set(sp.name for sp in device.crystal.species)) == 1:
        notes.append(
            "identical ions: a reorder is invisible to the image (the mode structure detects it, Section 6.7)"
        )
    return CrystalImage(
        data=np.asarray(data),
        fitted=fitted,
        model="crystal_image",
        provenance_id="conv.readout_crosstalk_added_counts",
        converged=len(dark_seen) == 0 and not run_state.lost,
        notes=tuple(notes),
    )


__all__ = ["crystal_image"]
