"""``ExperimentResult``, the record every simulated experiment returns, and one typed subclass per experiment whose
attributes are the fitted values (None where the fit did not produce the key)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np

from qutip_trap.control.table import CalEntry

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Drive
    from qutip_trap.device.model import Device

Quality = Literal["good", "poor", "failed", "exact"]
CHI2_GOOD_MAX = 3.0
"""A fit whose reduced chi-square is above this is ``"poor"``: its scatter is larger than its error bars explain."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ScanParameters:
    """Scan parameters by name, one tuple of floats each: one value per point for the scanned axis, one for a setting
    held over the scan; read by attribute (``scan.requested.durations_s[3]``)."""

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            {str(k): tuple(float(x) for x in np.atleast_1d(v)) for k, v in self.values.items()},
        )

    def __getattr__(self, name: str) -> tuple[float, ...]:
        values = self.__dict__.get("values", {})
        if name in values:
            return cast("tuple[float, ...]", values[name])
        raise AttributeError(f"{type(self).__name__} has no parameter {name!r}; it has {sorted(values)}")

    def keys(self) -> tuple[str, ...]:
        return tuple(self.values)

    def as_dict(self) -> dict[str, tuple[float, ...]]:
        return {k: cast("tuple[float, ...]", v) for k, v in self.values.items()}


def _scalar(value: Any, t_s: float) -> float:
    if callable(value):
        return float(value(0.5 * t_s))
    return float(value)


def _peak(envelope: Any, t_s: float) -> float:
    if isinstance(envelope, np.ndarray):
        return float(np.max(np.abs(envelope))) if envelope.size else 0.0
    if callable(envelope):
        ts = np.linspace(0.0, t_s, 65)
        return float(max(abs(float(envelope(t))) for t in ts))
    return float(abs(float(envelope)))


def realized_drive(device: Device, drive: Drive, duration_s: float) -> dict[str, float]:
    """The first tone's detuning (Hz), phase (rad) and peak Rabi amplitude (Hz) the hardware chain plays for one square
    pulse of ``drive``: after the DDS words and the amplifier saturation (response and jitter move no set point)."""
    from qutip_trap.control.hardware import apply_hardware_chain
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import Schedule

    sched = Schedule(
        (Pulse(drive, 0.0, float(duration_s), "realized", ()),),
        (),
        (),
        {q: 0.0 for q in range(device.crystal.n_ions)},
    )
    played, _notes = apply_hardware_chain(sched, device.hardware, quantize=True, response=False, jitter=False)
    tone = played.pulses[0].drive.tones[0]
    return {
        "detuning_hz": _scalar(tone.detuning_hz, duration_s),
        "phase_rad": _scalar(tone.phase_rad, duration_s),
        "rabi_hz": _peak(tone.envelope_hz, duration_s),
    }


def requested_drive(drive: Drive, duration_s: float) -> dict[str, float]:
    """``realized_drive``'s three numbers read from the drive as requested."""
    tone = drive.tones[0]
    return {
        "detuning_hz": _scalar(tone.detuning_hz, duration_s),
        "phase_rad": _scalar(tone.phase_rad, duration_s),
        "rabi_hz": _peak(tone.envelope_hz, duration_s),
    }


@dataclass(frozen=True)
class ExperimentResult:
    """What a simulated experiment returns: the measured scan (one row per point), the fitted parameters, the fit model,
    whether it converged, notes, and the scan as requested and as the hardware realized it."""

    data: np.ndarray
    fitted: dict[str, tuple[float, float]]
    """Parameter -> (value, uncertainty), in the units the key names."""
    model: str
    provenance_id: str
    converged: bool = True
    """False when a fit failed, landed at its scan edge or failed a consistency check; its entry is then ``uncalibrated``."""
    notes: tuple[str, ...] = ()
    sigma: np.ndarray | None = None
    """Per-row statistical uncertainty of the measured column (None for exact populations)."""
    requested: ScanParameters | None = None
    realized: ScanParameters | None = None
    """The scan the electronics could play; None where the axis is not a drive parameter."""
    chi2: float | None = None
    """The reduced chi-square of the principal fit; None when no weighted fit was made."""
    subject: dict[str, Any] = field(default_factory=dict)
    """What the experiment addressed, keyed as in the table: ``{"ion": 0, "beam": 2}``, ``{"pair": (0, 1)}``, ..."""
    created_at: str = field(default_factory=_now, compare=False)
    """ISO 8601 UTC wall-clock time; not part of equality."""

    x_label: ClassVar[str] = ""
    y_label: ClassVar[str] = ""

    def value(self, key: str) -> float:
        return float(self.fitted[key][0])

    def uncertainty(self, key: str) -> float:
        return float(self.fitted[key][1])

    def has(self, key: str) -> bool:
        return key in self.fitted

    def _get(self, key: str) -> float | None:
        return float(self.fitted[key][0]) if key in self.fitted else None

    @property
    def experiment(self) -> str:
        """The name of the experiment that made this result, from the type (and the model where a type serves two)."""
        return (
            EXPERIMENT_OF.get((type(self).__name__, self.model)) or EXPERIMENT_OF[(type(self).__name__, "")]
        )

    def _entry(self, key: str, *, fitted_at_s: float, sample_id: int) -> CalEntry:
        value, uncertainty = self.fitted[key]
        return CalEntry(
            value=float(value),
            uncertainty=float(uncertainty),
            status="calibrated" if self.converged else "uncalibrated",
            experiment=self.experiment,
            provenance_id=self.provenance_id,
            fitted_at_s=float(fitted_at_s),
            sample_id=int(sample_id),
        )

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """The table fields this result sets, for ``CalibrationTable.updated_with``: ``{field: {key: CalEntry}}``, or
        ``{"field": CalEntry}`` for the magnetic field; a result that sets no entry raises ``ValueError``."""
        raise ValueError(
            f"{type(self).__name__} ({self.experiment}) sets no calibration table entry directly"
        )

    @property
    def quality(self) -> Quality:
        """``"failed"`` if the fit did not converge, ``"exact"`` for exact populations (no shots, no chi-square), ``"good"``
        when the reduced chi-square is at most ``CHI2_GOOD_MAX`` or no weighted fit was made, else ``"poor"``."""
        if not self.converged:
            return "failed"
        if self.sigma is None and self.chi2 is None:
            return "exact"
        if self.chi2 is None or not math.isfinite(self.chi2) or self.chi2 <= CHI2_GOOD_MAX:
            return "good"
        return "poor"

    def plot(self, ax: Any = None) -> Any:
        """Plot the measured column against the scanned one on ``ax`` (a new figure when None) and return the axes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("ExperimentResult.plot needs matplotlib: install the 'plot' extra") from exc
        if ax is None:
            _fig, ax = plt.subplots()
        data = np.asarray(self.data)
        if data.ndim == 2 and data.shape[1] >= 2 and data.shape[0] > 0:
            x, y = data[:, 0], data[:, 1]
            if self.sigma is not None and np.shape(self.sigma)[0] == x.shape[0]:
                ax.errorbar(x, y, yerr=np.asarray(self.sigma).reshape(x.shape[0], -1)[:, 0], fmt="o")
            else:
                ax.plot(x, y, "o")
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(f"{type(self).__name__}: {self.model} ({self.quality})")
        return ax


@dataclass(frozen=True)
class RabiScan(ExperimentResult):
    """``rabi_scan``: P1 against pulse duration, fitted with the thermal Debye-Waller envelope."""

    @property
    def f_rabi_hz(self) -> float | None:
        return self._get("f_rabi_hz")

    @property
    def nbar(self) -> float | None:
        return self._get("nbar")

    @property
    def contrast(self) -> float | None:
        return self._get("contrast")

    @property
    def offset(self) -> float | None:
        return self._get("offset")

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """``rabi[(ion, beam)]`` from ``f_rabi_hz``."""
        if "f_rabi_hz" not in self.fitted:
            raise ValueError("RabiScan: no f_rabi_hz was fitted (fewer than four points)")
        key = (int(self.subject["ion"]), int(self.subject["beam"]))
        return {"rabi": {key: self._entry("f_rabi_hz", fitted_at_s=fitted_at_s, sample_id=sample_id)}}


@dataclass(frozen=True)
class RamseyFringe(ExperimentResult):
    """``ramsey`` (one fringe: ``delta_hz``, ``contrast``, ``phi0_rad``, ``offset``) and ``ramsey_frequency`` (two probes:
    ``qubit_freq_hz``, ``qubit_offset_hz``, ``fringe_plus_hz``, ``fringe_minus_hz``)."""

    @property
    def delta_hz(self) -> float | None:
        return self._get("delta_hz")

    @property
    def contrast(self) -> float | None:
        return self._get("contrast")

    @property
    def phi0_rad(self) -> float | None:
        return self._get("phi0_rad")

    @property
    def offset(self) -> float | None:
        return self._get("offset")

    @property
    def qubit_freq_hz(self) -> float | None:
        return self._get("qubit_freq_hz")

    @property
    def qubit_offset_hz(self) -> float | None:
        return self._get("qubit_offset_hz")

    @property
    def fringe_plus_hz(self) -> float | None:
        return self._get("fringe_plus_hz")

    @property
    def fringe_minus_hz(self) -> float | None:
        return self._get("fringe_minus_hz")

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """``qubit_freq[ion]`` from ``qubit_freq_hz`` (the two-probe experiment); a single fringe sets nothing."""
        if "qubit_freq_hz" not in self.fitted:
            raise ValueError("RamseyFringe: only ramsey_frequency (qubit_freq_hz) sets a table entry")
        return {
            "qubit_freq": {
                int(self.subject["ion"]): self._entry(
                    "qubit_freq_hz", fitted_at_s=fitted_at_s, sample_id=sample_id
                )
            }
        }


@dataclass(frozen=True)
class SidebandSpectrum(ExperimentResult):
    """``sideband_spectroscopy`` (the peaks nearest the carrier and the driven mode's sidebands) and ``mode_spectroscopy``
    (the two-stage lineshape fit: ``mode_hz``, ``eta``, ``nbar``, the sideband Rabi frequencies)."""

    @property
    def carrier_hz(self) -> float | None:
        return self._get("carrier_hz")

    @property
    def blue_sideband_hz(self) -> float | None:
        return self._get("blue_sideband_hz")

    @property
    def red_sideband_hz(self) -> float | None:
        return self._get("red_sideband_hz")

    @property
    def mode_hz(self) -> float | None:
        return self._get("mode_hz")

    @property
    def omega_bsb_hz(self) -> float | None:
        return self._get("omega_bsb_hz")

    @property
    def omega_carrier_hz(self) -> float | None:
        return self._get("omega_carrier_hz")

    @property
    def eta(self) -> float | None:
        return self._get("eta")

    @property
    def nbar(self) -> float | None:
        return self._get("nbar")

    @property
    def linewidth_hz(self) -> float | None:
        return self._get("linewidth_hz")

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """``modes[mode]``, ``nbar[mode]``, ``lamb_dicke[(ion, mode)]`` from ``mode_hz``, ``nbar``, ``eta`` if fitted."""
        if "mode" not in self.subject:
            raise ValueError("SidebandSpectrum: the scan drove no mode, so it sets no mode entry")
        mode = int(self.subject["mode"])
        out: dict[str, Any] = {}
        if "mode_hz" in self.fitted:
            out["modes"] = {mode: self._entry("mode_hz", fitted_at_s=fitted_at_s, sample_id=sample_id)}
        if "nbar" in self.fitted:
            out["nbar"] = {mode: self._entry("nbar", fitted_at_s=fitted_at_s, sample_id=sample_id)}
        if "eta" in self.fitted and "ion" in self.subject:
            out["lamb_dicke"] = {
                (int(self.subject["ion"]), mode): self._entry(
                    "eta", fitted_at_s=fitted_at_s, sample_id=sample_id
                )
            }
        if not out:
            raise ValueError("SidebandSpectrum: neither mode_hz, nbar nor eta was fitted")
        return out


@dataclass(frozen=True)
class ThermometryResult(ExperimentResult):
    """``thermometry``: nbar of one mode from the red/blue sideband ratio after equal pulses."""

    @property
    def nbar(self) -> float | None:
        return self._get("nbar")

    @property
    def ratio(self) -> float | None:
        return self._get("ratio")

    @property
    def duration_s(self) -> float | None:
        return self._get("duration_s")

    @property
    def ratio_spread(self) -> float | None:
        return self._get("ratio_spread")

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """``nbar[mode]``."""
        return {
            "nbar": {
                int(self.subject["mode"]): self._entry("nbar", fitted_at_s=fitted_at_s, sample_id=sample_id)
            }
        }


@dataclass(frozen=True)
class HeatingRateFit(ExperimentResult):
    """``heating_rate``: n_dot of one mode from a delay scan of the sideband asymmetry."""

    @property
    def ndot_per_s(self) -> float | None:
        return self._get("ndot_per_s")

    @property
    def nbar0(self) -> float | None:
        return self._get("nbar0")

    @property
    def duration_s(self) -> float | None:
        return self._get("duration_s")

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """``heating[mode]`` from ``ndot_per_s`` (quanta per second)."""
        return {
            "heating": {
                int(self.subject["mode"]): self._entry(
                    "ndot_per_s", fitted_at_s=fitted_at_s, sample_id=sample_id
                )
            }
        }


@dataclass(frozen=True)
class MSScan(ExperimentResult):
    """``ms_scan`` (the closure offset and the amplitude scale: ``closure_offset_hz``, ``closure_scale``, ``chi_unit_rad``,
    ``leakage_min``) and ``ms_phase_scan`` (the spin-phase alignment: ``correction_rad[i]`` per ion of the pair)."""

    @property
    def closure_offset_hz(self) -> float | None:
        return self._get("closure_offset_hz")

    @property
    def closure_offset_used_hz(self) -> float | None:
        return self._get("closure_offset_used_hz")

    @property
    def closure_scale(self) -> float | None:
        return self._get("closure_scale")

    @property
    def chi_unit_rad(self) -> float | None:
        return self._get("chi_unit_rad")

    @property
    def leakage_min(self) -> float | None:
        return self._get("leakage_min")

    def correction_rad(self, ion: int) -> float | None:
        """The spin-phase correction ``ms_phase_scan`` fitted for ``ion`` (None for the population scan)."""
        return self._get(f"correction_rad[{ion}]")


@dataclass(frozen=True)
class ParityScan(ExperimentResult):
    """``parity_scan``: the parity oscillation's contrast and phase, the Bell fidelity bound and the two populations."""

    @property
    def contrast(self) -> float | None:
        return self._get("contrast")

    @property
    def phi0_rad(self) -> float | None:
        return self._get("phi0_rad")

    @property
    def offset(self) -> float | None:
        return self._get("offset")

    @property
    def bell_fidelity_bound(self) -> float | None:
        return self._get("bell_fidelity_bound")

    @property
    def P00(self) -> float | None:  # noqa: N802 - the population's own name
        return self._get("P00")

    @property
    def P11(self) -> float | None:  # noqa: N802 - the population's own name
        return self._get("P11")


@dataclass(frozen=True)
class DetectionHistogram(ExperimentResult):
    """``detection_histogram``: the bright and dark count histograms and the fitted threshold, window and error rates."""

    @property
    def threshold(self) -> float | None:
        return self._get("threshold")

    @property
    def window_s(self) -> float | None:
        return self._get("window_s")

    @property
    def eps_B(self) -> float | None:  # noqa: N802 - Section 8.6's symbol
        return self._get("eps_B")

    @property
    def eps_D(self) -> float | None:  # noqa: N802 - Section 8.6's symbol
        return self._get("eps_D")

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """``detection[name]`` for every fitted entry except the scattered rate, which is not a table entry."""
        names = [k for k in self.fitted if k != "R_bright_scattered_per_s"]
        return {"detection": {n: self._entry(n, fitted_at_s=fitted_at_s, sample_id=sample_id) for n in names}}

    @property
    def errors(self) -> tuple[float, float] | None:
        """(eps_B, eps_D) at the optimum, or None before a fit."""
        b, d = self._get("eps_B"), self._get("eps_D")
        return None if b is None or d is None else (b, d)


@dataclass(frozen=True)
class StarkScan(ExperimentResult):
    """``stark_scan``: the differential light shift ``stark_shift_hz`` (per beam ``stark_shift_hz[b]``) and, in beat-note
    mode, the coupling shift."""

    @property
    def stark_shift_hz(self) -> float | None:
        return self._get("stark_shift_hz")

    @property
    def coupling_shift_hz(self) -> float | None:
        return self._get("coupling_shift_hz")

    @property
    def coupling_shift_expected_hz(self) -> float | None:
        return self._get("coupling_shift_expected_hz")

    @property
    def stark_detuning_hz(self) -> float | None:
        return self._get("stark_detuning_hz")

    def shift_of_beam(self, beam: int) -> float | None:
        return self._get(f"stark_shift_hz[{beam}]")

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """``stark[(ion, beam)]`` from ``stark_shift_hz`` (the total shift of the ion's gate beams under ``subject['beam']``)."""
        if "stark_shift_hz" not in self.fitted:
            raise ValueError("StarkScan: no stark_shift_hz was fitted")
        key = (int(self.subject["ion"]), int(self.subject["beam"]))
        return {"stark": {key: self._entry("stark_shift_hz", fitted_at_s=fitted_at_s, sample_id=sample_id)}}


@dataclass(frozen=True)
class CrosstalkScan(ExperimentResult):
    """``crosstalk_scan``: the driven ion's ``rate_hz`` and per neighbour ``rate_hz[j]``, ``eps[j]`` and ``phase_rad[j]``."""

    @property
    def rate_hz(self) -> float | None:
        return self._get("rate_hz")

    def eps(self, neighbour: int) -> float | None:
        return self._get(f"eps[{neighbour}]")

    def rate_of(self, neighbour: int) -> float | None:
        return self._get(f"rate_hz[{neighbour}]")

    def phase_rad(self, neighbour: int) -> float | None:
        return self._get(f"phase_rad[{neighbour}]")

    @property
    def neighbours(self) -> tuple[int, ...]:
        """The neighbours the scan measured, from the fitted keys."""
        out: list[int] = []
        for key in self.fitted:
            if key.startswith("eps[") and key.endswith("]"):
                out.append(int(key[4:-1]))
        return tuple(sorted(out))

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """``crosstalk[(ion, j)]`` from ``eps[j]`` and ``crosstalk_phase[(ion, j)]`` from ``phase_rad[j]`` per neighbour."""
        ion = int(self.subject["ion"])
        if not self.neighbours:
            raise ValueError("CrosstalkScan: no neighbour was measured")
        out: dict[str, Any] = {
            "crosstalk": {
                (ion, j): self._entry(f"eps[{j}]", fitted_at_s=fitted_at_s, sample_id=sample_id)
                for j in self.neighbours
            }
        }
        phases = {
            (ion, j): self._entry(f"phase_rad[{j}]", fitted_at_s=fitted_at_s, sample_id=sample_id)
            for j in self.neighbours
            if f"phase_rad[{j}]" in self.fitted
        }
        if phases:
            out["crosstalk_phase"] = phases
        return out


@dataclass(frozen=True)
class FieldScan(ExperimentResult):
    """``field_scan``: B from the qubit transition frequency through the atomic layer's nu(B)."""

    @property
    def B_gauss(self) -> float | None:  # noqa: N802 - the field's symbol
        return self._get("B_gauss")

    @property
    def qubit_freq_hz(self) -> float | None:
        return self._get("qubit_freq_hz")

    @property
    def qubit_offset_hz(self) -> float | None:
        return self._get("qubit_offset_hz")

    @property
    def dnu_dB_hz_per_g(self) -> float | None:  # noqa: N802 - the derivative's symbol
        return self._get("dnu_dB_hz_per_g")

    @property
    def b_seed_gauss(self) -> float | None:
        return self._get("b_seed_gauss")

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """``field`` from ``B_gauss``."""
        return {"field": self._entry("B_gauss", fitted_at_s=fitted_at_s, sample_id=sample_id)}


@dataclass(frozen=True)
class MicromotionScan(ExperimentResult):
    """``micromotion_scan``: the nulling shim voltages (``shim[name]``) and the modulation index before the scan."""

    def shim_v(self, name: str) -> float | None:
        return self._get(f"shim[{name}]")

    @property
    def shims_v(self) -> dict[str, float]:
        return {
            k[5:-1]: float(v[0]) for k, v in self.fitted.items() if k.startswith("shim[") and k.endswith("]")
        }

    @property
    def beta_before(self) -> float | None:
        return self._get("beta_before")

    @property
    def carrier_hz(self) -> float | None:
        return self._get("carrier_hz")

    @property
    def sideband_excitation(self) -> float | None:
        return self._get("sideband_excitation")

    def table_updates(self, *, fitted_at_s: float, sample_id: int) -> dict[str, Any]:
        """``micromotion["shim[name]"]`` for every shim the scan nulled."""
        shims = [k for k in self.fitted if k.startswith("shim[") and k.endswith("]")]
        if not shims:
            raise ValueError("MicromotionScan: no shim was fitted")
        return {
            "micromotion": {k: self._entry(k, fitted_at_s=fitted_at_s, sample_id=sample_id) for k in shims}
        }


@dataclass(frozen=True)
class CrystalImage(ExperimentResult):
    """``crystal_image``: the imaged chain against the nominal crystal (counts of bright, dark and lost ions)."""

    @property
    def n_ions(self) -> float | None:
        return self._get("n_ions")

    @property
    def n_bright(self) -> float | None:
        return self._get("n_bright")

    @property
    def n_dark(self) -> float | None:
        return self._get("n_dark")

    @property
    def n_lost(self) -> float | None:
        return self._get("n_lost")


for _cls, _x, _y in (
    (RabiScan, "pulse duration (s)", "P1"),
    (RamseyFringe, "delay (s)", "P1"),
    (SidebandSpectrum, "detuning from the carrier (Hz)", "P1"),
    (ThermometryResult, "sideband", "P1"),
    (HeatingRateFit, "delay (s)", "nbar"),
    (MSScan, "amplitude scale", "population"),
    (ParityScan, "analysis phase (rad)", "parity"),
    (DetectionHistogram, "photon count", "records"),
    (StarkScan, "delay (s)", "P1"),
    (CrosstalkScan, "pulse duration (s)", "P1"),
    (FieldScan, "delay (s)", "P1"),
    (MicromotionScan, "shim voltage (V)", "signal"),
    (CrystalImage, "column", "counts"),
):
    _cls.x_label = _x
    _cls.y_label = _y

EXPERIMENT_OF: dict[tuple[str, str], str] = {
    ("RabiScan", ""): "rabi_scan",
    ("RamseyFringe", ""): "ramsey",
    ("RamseyFringe", "ramsey_two_probe"): "ramsey_frequency",
    ("SidebandSpectrum", ""): "sideband_spectroscopy",
    ("SidebandSpectrum", "sideband_lineshape_two_stage"): "mode_spectroscopy",
    ("ThermometryResult", ""): "thermometry",
    ("HeatingRateFit", ""): "heating_rate",
    ("MSScan", ""): "ms_scan",
    ("MSScan", "ms_spin_phase_scan"): "ms_phase_scan",
    ("ParityScan", ""): "parity_scan",
    ("DetectionHistogram", ""): "detection_histogram",
    ("StarkScan", ""): "stark_scan",
    ("CrosstalkScan", ""): "crosstalk_scan",
    ("FieldScan", ""): "field_scan",
    ("MicromotionScan", ""): "micromotion_scan",
    ("CrystalImage", ""): "crystal_image",
    ("ExperimentResult", ""): "experiment",
}
"""(result type, model) -> the experiment's name; the empty model is the type's default (``ExperimentResult.experiment``)."""

RESULT_TYPES: dict[str, type[ExperimentResult]] = {
    "rabi_scan": RabiScan,
    "ramsey": RamseyFringe,
    "ramsey_frequency": RamseyFringe,
    "sideband_spectroscopy": SidebandSpectrum,
    "mode_spectroscopy": SidebandSpectrum,
    "thermometry": ThermometryResult,
    "heating_rate": HeatingRateFit,
    "ms_scan": MSScan,
    "ms_phase_scan": MSScan,
    "parity_scan": ParityScan,
    "detection_histogram": DetectionHistogram,
    "stark_scan": StarkScan,
    "crosstalk_scan": CrosstalkScan,
    "field_scan": FieldScan,
    "micromotion_scan": MicromotionScan,
    "crystal_image": CrystalImage,
}
"""Experiment name -> the result type it returns."""
