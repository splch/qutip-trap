"""What a simulated experiment returns: the scan, the fitted parameters with their uncertainties, the fit's quality, the
scan as requested beside the scan the hardware chain played, and the calibration-table entries the fit proposes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from qutip_trap.control.table import CalEntry

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Drive
    from qutip_trap.device.model import Device

CHI2_GOOD_MAX = 3.0
"""A fit whose reduced chi-square is above this is ``"poor"``: its scatter is larger than its error bars explain."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ScanParameters:
    """Scan parameters by name, each a tuple of floats (one per point for the scanned axis, one for a held setting), read
    as attributes (``scan.requested.durations_s``)."""

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = {str(k): tuple(float(x) for x in np.atleast_1d(v)) for k, v in self.values.items()}
        object.__setattr__(self, "values", values)

    def __getattr__(self, name: str) -> tuple[float, ...]:
        values: dict[str, tuple[float, ...]] = self.__dict__.get("values", {})
        if name in values:
            return values[name]
        raise AttributeError(f"{type(self).__name__} has no parameter {name!r}; it has {sorted(values)}")


def _scalar(value: Any, t_s: float) -> float:
    return float(value(0.5 * t_s)) if callable(value) else float(value)


def _peak(envelope: Any, t_s: float) -> float:
    if isinstance(envelope, np.ndarray):
        return float(np.max(np.abs(envelope))) if envelope.size else 0.0
    if callable(envelope):
        return float(max(abs(float(envelope(t))) for t in np.linspace(0.0, t_s, 65)))
    return float(abs(float(envelope)))


def requested_drive(drive: Drive, duration_s: float) -> dict[str, float]:
    """The first tone's detuning (Hz), phase (rad) and peak Rabi amplitude (Hz) of a square pulse of ``drive``."""
    tone = drive.tones[0]
    return {
        "detuning_hz": _scalar(tone.detuning_hz, duration_s),
        "phase_rad": _scalar(tone.phase_rad, duration_s),
        "rabi_hz": _peak(tone.envelope_hz, duration_s),
    }


def realized_drive(device: Device, drive: Drive, duration_s: float) -> dict[str, float]:
    """``requested_drive`` of the pulse the hardware chain of Section 7.10 plays: the DDS frequency grid, phase and amplitude
    words and the amplifier saturation (the modulator response and the jitter move no set point and are left out)."""
    from qutip_trap.control.hardware import apply_hardware_chain
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import Schedule

    pulse = Pulse(drive, 0.0, float(duration_s), "realized", ())
    sched = Schedule((pulse,), (), (), {q: 0.0 for q in range(device.crystal.n_ions)})
    played, _notes = apply_hardware_chain(sched, device.hardware, quantize=True, response=False, jitter=False)
    return requested_drive(played.pulses[0].drive, duration_s)


@dataclass(frozen=True)
class ExperimentResult:
    """A fitted scan (Section 7.5): the scan as measured (one row per point, columns documented per experiment), each fitted
    parameter as (value, one-sigma uncertainty), the fit model and the provenance id behind it, whether the fit converged (a
    failed or edge-of-scan fit leaves its entry ``uncalibrated``), notes, the per-row sigma when shots were drawn, the scan
    as requested and as realized, the reduced chi-square of the principal fit, the subject as the calibration table keys it
    (``{"ion": 0, "beam": 2}``, ``{"pair": (0, 1)}``, ``{"mode": 3}``), the creation time (ISO 8601 UTC, not part of equality)
    and the experiment that made it."""

    data: np.ndarray
    fitted: dict[str, tuple[float, float]]
    model: str
    provenance_id: str
    converged: bool = True
    notes: tuple[str, ...] = ()
    sigma: np.ndarray | None = None
    requested: ScanParameters | None = None
    realized: ScanParameters | None = None
    chi2: float | None = None
    subject: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now, compare=False)
    experiment: str = "experiment"

    def value(self, key: str) -> float:
        return float(self.fitted[key][0])

    def uncertainty(self, key: str) -> float:
        return float(self.fitted[key][1])

    @property
    def quality(self) -> Literal["good", "poor", "failed", "exact"]:
        """``"failed"`` when the fit did not converge, ``"exact"`` for exact populations (no shots, no chi-square),
        ``"good"`` when the reduced chi-square is at most ``CHI2_GOOD_MAX`` or no weighted fit was made, else ``"poor"``:
        the test a caller applies before adopting ``CalibrationTable.updated_with``."""
        if not self.converged:
            return "failed"
        if self.sigma is None and self.chi2 is None:
            return "exact"
        if self.chi2 is None or not math.isfinite(self.chi2) or self.chi2 <= CHI2_GOOD_MAX:
            return "good"
        return "poor"

    def _entry(
        self, key: str, fitted_at_s: float, sample_id: int, provenance_id: str | None = None
    ) -> CalEntry:
        """The fitted ``key`` as a table entry: ``calibrated`` when the fit converged on a finite value and uncertainty,
        else ``uncalibrated`` with a non-finite number stored as 0; stamped with the experiment, ``provenance_id`` (default
        the result's, the model of its principal fit), the time and the noise sample."""
        value, uncertainty = (float(x) for x in self.fitted[key])
        finite = math.isfinite(value) and math.isfinite(uncertainty)
        return CalEntry(
            value if math.isfinite(value) else 0.0,
            uncertainty if math.isfinite(uncertainty) else 0.0,
            "calibrated" if self.converged and finite else "uncalibrated",
            self.experiment,
            self.provenance_id if provenance_id is None else provenance_id,
            float(fitted_at_s),
            int(sample_id),
        )

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """The calibration-table fields this result sets (``CalibrationTable.updated_with``): ``{field: {key: CalEntry}}``,
        or ``{"field": CalEntry}``, each entry ``calibrated`` when the fit converged on a finite value (``_entry``); a result
        that sets none refuses."""
        raise ValueError(
            f"{type(self).__name__} ({self.experiment}) sets no calibration table entry directly"
        )


@dataclass(frozen=True)
class RabiScan(ExperimentResult):
    """``rabi_scan``: sets ``rabi[(ion, beam)]`` from ``f_rabi_hz``."""

    experiment: str = "rabi_scan"

    @property
    def f_rabi_hz(self) -> float | None:
        return self.value("f_rabi_hz") if "f_rabi_hz" in self.fitted else None

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        if "f_rabi_hz" not in self.fitted:
            raise ValueError("RabiScan: no f_rabi_hz was fitted (fewer than four points)")
        key = (int(self.subject["ion"]), int(self.subject["beam"]))
        return {"rabi": {key: self._entry("f_rabi_hz", fitted_at_s, sample_id)}}


@dataclass(frozen=True)
class RamseyFringe(ExperimentResult):
    """``ramsey`` (one fringe) and ``ramsey_frequency`` (two probes), which sets ``qubit_freq[ion]`` from ``qubit_freq_hz``."""

    experiment: str = "ramsey"

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        if "qubit_freq_hz" not in self.fitted:
            raise ValueError("RamseyFringe: only ramsey_frequency (qubit_freq_hz) sets a table entry")
        return {
            "qubit_freq": {int(self.subject["ion"]): self._entry("qubit_freq_hz", fitted_at_s, sample_id)}
        }


@dataclass(frozen=True)
class SidebandSpectrum(ExperimentResult):
    """``sideband_spectroscopy`` and ``mode_spectroscopy``: sets ``modes[mode]``, ``nbar[mode]`` and
    ``lamb_dicke[(ion, mode)]`` from ``mode_hz``, ``nbar`` and ``eta`` where fitted, the occupation under the sideband
    ratio's provenance and |eta| under the Lamb-Dicke convention's."""

    experiment: str = "sideband_spectroscopy"

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        if "mode" not in self.subject:
            raise ValueError("SidebandSpectrum: the scan drove no mode, so it sets no mode entry")
        mode = int(self.subject["mode"])
        out: dict[str, Any] = {}
        if "mode_hz" in self.fitted:
            out["modes"] = {mode: self._entry("mode_hz", fitted_at_s, sample_id)}
        if "nbar" in self.fitted:
            out["nbar"] = {
                mode: self._entry("nbar", fitted_at_s, sample_id, "anchor.m3.thermometry_exactness")
            }
        if "eta" in self.fitted and "ion" in self.subject:
            key = (int(self.subject["ion"]), mode)
            out["lamb_dicke"] = {key: self._entry("eta", fitted_at_s, sample_id, "conv.lamb_dicke")}
        if not out:
            raise ValueError("SidebandSpectrum: neither mode_hz, nbar nor eta was fitted")
        return out


@dataclass(frozen=True)
class ThermometryResult(ExperimentResult):
    """``thermometry``: sets ``nbar[mode]``."""

    experiment: str = "thermometry"

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        return {"nbar": {int(self.subject["mode"]): self._entry("nbar", fitted_at_s, sample_id)}}


@dataclass(frozen=True)
class HeatingRateFit(ExperimentResult):
    """``heating_rate``: sets ``heating[mode]`` from ``ndot_per_s`` (quanta per second)."""

    experiment: str = "heating_rate"

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        return {"heating": {int(self.subject["mode"]): self._entry("ndot_per_s", fitted_at_s, sample_id)}}


@dataclass(frozen=True)
class ParityScan(ExperimentResult):
    """``parity_scan``: the parity contrast and the Bell-fidelity bound; sets no entry."""

    experiment: str = "parity_scan"


@dataclass(frozen=True)
class DetectionHistogram(ExperimentResult):
    """``detection_histogram``: sets ``detection[name]`` for the threshold, the window and (eps_B, eps_D) of the figure of
    merit, and for the three rates of the mean-count fit under that fit's provenance."""

    experiment: str = "detection_histogram"

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        entries = {
            n: self._entry(n, fitted_at_s, sample_id) for n in ("threshold", "window_s", "eps_B", "eps_D")
        }
        for n in ("R_bright_detected_per_s", "R_dark_pumping_per_s", "R_bright_pumping_per_s"):
            entries[n] = self._entry(n, fitted_at_s, sample_id, "conv.mean_count_curve")
        return {"detection": entries}


@dataclass(frozen=True)
class StarkScan(ExperimentResult):
    """``stark_scan``: sets ``stark[(ion, beam)]`` from ``stark_shift_hz``, the shift of the ion's gate beams."""

    experiment: str = "stark_scan"

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        if "stark_shift_hz" not in self.fitted:
            raise ValueError("StarkScan: no stark_shift_hz was fitted")
        key = (int(self.subject["ion"]), int(self.subject["beam"]))
        return {"stark": {key: self._entry("stark_shift_hz", fitted_at_s, sample_id)}}


@dataclass(frozen=True)
class CrosstalkScan(ExperimentResult):
    """``crosstalk_scan``: sets ``crosstalk[(ion, j)]`` from ``eps[j]`` and ``crosstalk_phase[(ion, j)]`` from
    ``phase_rad[j]`` for every neighbour j."""

    experiment: str = "crosstalk_scan"

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        ion = int(self.subject["ion"])
        neighbours = sorted(int(k[4:-1]) for k in self.fitted if k.startswith("eps[") and k.endswith("]"))
        if not neighbours:
            raise ValueError("CrosstalkScan: no neighbour was measured")
        out: dict[str, Any] = {
            "crosstalk": {(ion, j): self._entry(f"eps[{j}]", fitted_at_s, sample_id) for j in neighbours}
        }
        phases = {
            (ion, j): self._entry(f"phase_rad[{j}]", fitted_at_s, sample_id)
            for j in neighbours
            if f"phase_rad[{j}]" in self.fitted
        }
        if phases:
            out["crosstalk_phase"] = phases
        return out


@dataclass(frozen=True)
class FieldScan(ExperimentResult):
    """``field_scan``: sets ``field`` from ``B_gauss``."""

    experiment: str = "field_scan"

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        return {"field": self._entry("B_gauss", fitted_at_s, sample_id)}


@dataclass(frozen=True)
class MicromotionScan(ExperimentResult):
    """``micromotion_scan``: sets ``micromotion["shim[name]"]`` for every shim the scan nulled and
    ``micromotion["beta[beam]"]``, the residual index it left."""

    experiment: str = "micromotion_scan"

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        keys = [k for k in self.fitted if k.startswith(("shim[", "beta[")) and k.endswith("]")]
        return {"micromotion": {k: self._entry(k, fitted_at_s, sample_id) for k in keys}}


@dataclass(frozen=True)
class CrystalImage(ExperimentResult):
    """``crystal_image``: the imaged chain against the nominal crystal; sets no entry."""

    experiment: str = "crystal_image"
