"""The calibration table and its entries (PLAN.md Sections 3.2, 7.5; Appendix E).

``CalibrationTable`` is plain data: ``control.schedule`` reads it and never writes it, and ``control`` never
imports ``calibration`` (Appendix E). Every entry carries its status, the experiment that produced it, its
provenance id, its age and the noise sample it was fitted under (Section 7.5).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from qutip_trap.control.shaping import GateModes, Kernel

Leg = Literal["red", "blue"]
EntryKind = Literal["setpoint", "characterisation"]
"""Qibolab's partition of a calibration (docs/api_proposal.md Section 4.8): a ``setpoint`` is what the scheduler programs
(the frame frequency, the Rabi rate the amplitude word is derived from, the light shift it compensates, the mode frequencies
its sidebands sit at, the entangling waveforms, the micromotion shims, the detection threshold and window); a
``characterisation`` is what was measured about the device and programmed nowhere (the field, the occupations, the heating
rates, the crosstalk ratios and phases, the Lamb-Dicke parameters)."""
ENTRY_KINDS: dict[str, EntryKind] = {
    "qubit_freq": "setpoint",
    "rabi": "setpoint",
    "stark": "setpoint",
    "modes": "setpoint",
    "ms": "setpoint",
    "micromotion": "setpoint",
    "detection": "setpoint",
    "field": "characterisation",
    "nbar": "characterisation",
    "heating": "characterisation",
    "crosstalk": "characterisation",
    "crosstalk_phase": "characterisation",
    "lamb_dicke": "characterisation",
}
"""The kind of every table field (``CalibrationTable.kind_of`` reads an entry's own ``kind`` first)."""
WaveformKind = Literal["ms", "light_shift", "gradient"]
"""``ms``: bichromatic red/blue legs on the spin flip (Section 4.4.1); ``light_shift``: one beat note ("blue") on the
state-dependent light shift, sigma_z sigma_z (Section 4.4.4); ``gradient``: the two microwave tones at -/+ delta of a
near-field magnetic-gradient drive, whose dressed sigma_z force carries J_2(4 Omega_mu/delta) (Section 4.4.5)."""


@dataclass(frozen=True)
class CalEntry:
    """One calibrated number (Section 7.5): the value and its uncertainty in the units of the table field that holds it, the
    status (``seed`` from the closed forms, ``calibrated`` by a simulated experiment, ``uncalibrated`` when a fit was
    refused), the experiment and the provenance id behind it, the laboratory time of the fit (s) and the noise sample it
    was fitted under."""

    value: float
    uncertainty: float
    status: Literal["seed", "calibrated", "uncalibrated"]
    experiment: str
    provenance_id: str
    fitted_at_s: float
    """Calibration age: the laboratory time the fit was made at (Section 7.5)."""
    sample_id: int
    """The noise sample the entry was fitted under."""
    kind: EntryKind | None = dc_field(default=None, metadata={"hash": "skip_default"})
    """``setpoint`` (what the scheduler programs) or ``characterisation`` (what was measured about the device); None derives
    it from the table field that holds the entry (``ENTRY_KINDS``, ``CalibrationTable.kind_of``; 0.3.0). Left out of the
    digest while unset, so every 0.2.0 table digest survives."""

    def to_dict(self) -> dict[str, Any]:
        """The entry as plain JSON-able values (``Result.to_dict``, 0.2.0); ``kind`` only when set."""
        out: dict[str, Any] = {
            "value": float(self.value),
            "uncertainty": float(self.uncertainty),
            "status": self.status,
            "experiment": self.experiment,
            "provenance_id": self.provenance_id,
            "fitted_at_s": float(self.fitted_at_s),
            "sample_id": int(self.sample_id),
        }
        if self.kind is not None:
            out["kind"] = self.kind
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CalEntry:
        return cls(
            value=float(d["value"]),
            uncertainty=float(d["uncertainty"]),
            status=d["status"],
            experiment=str(d["experiment"]),
            provenance_id=str(d["provenance_id"]),
            fitted_at_s=float(d["fitted_at_s"]),
            sample_id=int(d["sample_id"]),
            kind=d.get("kind"),
        )

    def __post_init__(self) -> None:
        if self.uncertainty < 0.0:
            raise ValueError("uncertainty must be non-negative")

    @property
    def usable(self) -> bool:
        """A ``seed`` or ``calibrated`` entry may be scheduled from; an ``uncalibrated`` one refuses (Section 7.3)."""
        return self.status != "uncalibrated"


def usable(entry: CalEntry | None) -> bool:
    """True for a present ``seed`` or ``calibrated`` entry (what the scheduler and a downstream fit may read)."""
    return entry is not None and entry.usable


@dataclass(frozen=True)
class Segment:
    """One segment of a calibrated entangling pulse (Appendix E, ``Waveform.segments``).

    Amplitudes and detunings are constants (the segmented AM family) or callables of the time since the SEGMENT start
    (a Fourier-sine amplitude, an FM detuning schedule), the same two forms ``Tone`` accepts; the scheduler plays each
    segment as one ``Pulse`` per ion so that step discontinuities fall on integration boundaries (Section 7.4).
    """

    duration_s: float
    amplitude_hz: dict[tuple[int, Leg], float | Callable[[float], float]]
    """Omega per (ion, leg), leg in {red, blue}. A per-ion or per-leg imbalance is NOT a table entry: Kirchmair's
    Omega_b = Omega(1 + xi), Omega_r = Omega(1 - xi) is carried by the two legs' own amplitudes, which the pulse solvers
    set through ``waveform_from_segmented(imbalance=...)`` (Section 4.4.1; ledger ``conv.no_stark_imbalance``)."""
    phase_rad: dict[tuple[int, Leg], float]
    detuning_hz: dict[Leg, float | Callable[[float], float]]

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("segment duration must be positive")
        if set(self.amplitude_hz) != set(self.phase_rad):
            raise ValueError("amplitude_hz and phase_rad must be indexed by the same (ion, leg) pairs")
        legs = {leg for _ion, leg in self.amplitude_hz}
        if not legs <= set(self.detuning_hz):
            raise ValueError("every leg with an amplitude needs a detuning")
        for key, v in self.amplitude_hz.items():
            if not callable(v) and float(v) < 0.0:
                raise ValueError(f"amplitude of {key} is a magnitude; a reversed force is a phase of pi")

    @property
    def ions(self) -> tuple[int, ...]:
        return tuple(sorted({ion for ion, _leg in self.amplitude_hz}))

    @property
    def legs(self) -> tuple[Leg, ...]:
        return tuple(leg for leg in ("red", "blue") if any(k[1] == leg for k in self.amplitude_hz))


@dataclass(frozen=True)
class Waveform:
    """A calibrated entangling pulse: what the scheduler plays (Sections 4.4.3, 7.4)."""

    segments: tuple[Segment, ...] | None
    fourier: tuple[complex, ...] | None
    """Or Fourier coefficients of the modulation (Section 4.4.3)."""
    duration_s: float
    phi_s: CalEntry
    phi_m: CalEntry
    chi_m: dict[int, float]
    """Per-mode entangling angle at closure, SIGNED: the pulse applies exp(+i sum_m chi_m sigma sigma) (Section 13)."""
    alpha_m: dict[int, complex]
    """Per-mode residual displacement at closure (the worst gate ion's)."""
    kind: WaveformKind = "ms"

    def __post_init__(self) -> None:
        if (self.segments is None) == (self.fourier is None):
            raise ValueError("a Waveform is either segmented or Fourier-parameterized, not both or neither")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.segments is not None:
            total = sum(s.duration_s for s in self.segments)
            if not math.isclose(total, self.duration_s, rel_tol=1e-9, abs_tol=1e-15):
                raise ValueError(f"segment durations sum to {total}, not duration_s = {self.duration_s}")
            ions = {s.ions for s in self.segments}
            if len(ions) != 1:
                raise ValueError("every segment addresses the same ions")
            for s in self.segments:
                if self.kind == "light_shift" and s.legs != ("blue",):
                    raise ValueError("a light-shift waveform has one leg, 'blue' (the beat note itself)")
                if self.kind in ("ms", "gradient") and s.legs != ("red", "blue"):
                    raise ValueError(
                        "an MS waveform has a red and a blue leg on every ion, and so does a gradient waveform "
                        "(its two microwave tones at -/+ delta)"
                    )

    @property
    def ions(self) -> tuple[int, ...]:
        if self.segments is None:
            raise ValueError("a Fourier-parameterized waveform names no ions")
        return self.segments[0].ions

    @property
    def chi_total_rad(self) -> float:
        """The signed two-body angle: exp(+i chi sigma sigma) = XX(-chi) (Section 13)."""
        return float(sum(self.chi_m.values()))

    @classmethod
    def symmetric(
        cls,
        modes: GateModes,
        *,
        gate_mode: int,
        loops: int = 1,
        epsilon_hz: float | None = None,
        duration_s: float | None = None,
        chi_target_rad: float = math.pi / 4.0,
        kernel: Kernel = "choi",
        pair: tuple[int, int] | None = None,
        phi_s_rad: float = 0.0,
        phi_m_rad: float = 0.0,
        imbalance: float = 0.0,
        detuning_side: Literal["inside", "outside"] = "inside",
        all_modes: bool = True,
        kind: Literal["ms", "light_shift"] = "ms",
    ) -> Waveform:
        """The equal-envelope symmetric-detuning shortcut (Appendix E): one square segment, tones at -/+ (omega_g -/+ eps) closing
        ``gate_mode`` after ``loops`` loops (tau = 2 pi K/eps), the amplitude from |chi| = chi_target over the pair's modes; for
        one mode eta Omega/eps = 1/(2 sqrt K) exactly (Section 4.4.1). Equals the general solver at equal envelopes (Section 9.17).

        ``kind`` is ``ms`` or ``light_shift``: the closure algebra of Section 4.4.3 does not cover a ``gradient`` waveform,
        whose force is a Bessel function of the tone amplitude rather than linear in it (ledger ``conv.gradient_drive``)."""
        from qutip_trap.control.shaping import symmetric_pulse

        return symmetric_pulse(
            modes,
            gate_mode=gate_mode,
            loops=loops,
            epsilon_hz=epsilon_hz,
            duration_s=duration_s,
            chi_target_rad=chi_target_rad,
            kernel=kernel,
            pair=pair,
            phi_s_rad=phi_s_rad,
            phi_m_rad=phi_m_rad,
            imbalance=imbalance,
            detuning_side=detuning_side,
            all_modes=all_modes,
            kind=kind,
        ).waveform


@dataclass(frozen=True)
class CalibrationTable:
    """What the scheduler believes about the device (Sections 3.2, 7.5): the per-ion, per-(ion, beam), per-pair and per-mode
    entries, each a ``CalEntry`` with its status and provenance, the entangling ``Waveform`` per pair, all keyed to the
    ``Device.hash()`` and the noise ``seed`` they were fitted under. Plain data: ``control.schedule`` reads it and never
    writes it, ``calibration`` builds it; ``surrogate`` says whether it came from the closed forms with spot checks or
    from full simulated experiments.

    Lookup precedence (0.3.0; the way OpenQASM 3 states ``defcal`` resolution, most specific first): an entangling
    ``Waveform`` is looked up per PAIR, under either key order (``waveform_for``); a Rabi, Stark or crosstalk entry per
    (ion, beam) or (ion, neighbour), with the beam the drive's first beam (``GateDrive.table_key_beam``; -1 for a microwave
    drive); a qubit frequency, mode frequency, occupation or heating rate per ion or per mode; the field and the detection
    entries once for the device. A more specific entry never falls back to a less specific one: an absent or
    ``uncalibrated`` entry refuses to schedule (Section 7.3). Edits are proposals: ``with_params`` sets entries by field,
    ``updated_with`` maps a typed ``ExperimentResult`` onto the entries it fitted, both returning a new table."""

    device_hash: str
    seed: int
    surrogate: bool
    """Closed-form surrogate with spot checks, or full simulated experiments (Section 7.5)."""
    qubit_freq: dict[int, CalEntry]
    """From the Ramsey-frequency experiment; the scheduler never reads the true value."""
    rabi: dict[tuple[int, int], CalEntry]
    """(ion, beam)."""
    stark: dict[tuple[int, int], CalEntry]
    crosstalk: dict[tuple[int, int], CalEntry]
    modes: dict[int, CalEntry]
    nbar: dict[int, CalEntry]
    ms: dict[tuple[int, int], Waveform]
    """Entangling waveforms per pair; looked up as (a, b) or (b, a) (``waveform_for``)."""
    field: CalEntry
    micromotion: dict[str, CalEntry]
    """Shim voltages and residual beta per beam direction."""
    detection: dict[str, CalEntry]
    heating: dict[int, CalEntry]
    crosstalk_phase: dict[tuple[int, int], CalEntry] = dc_field(default_factory=dict)
    """arg(epsilon_ij) per (ion, neighbour) from the crosstalk scan's phase measurement (Section 7.5 item 8; M8); absent = 0."""
    lamb_dicke: dict[tuple[int, int], CalEntry] = dc_field(default_factory=dict)
    """|eta_{i,m}| per (ion, mode) extracted from the sideband Rabi frequency (Section 7.9; M8); C0 is inside a
    sideband-calibrated eta and outside a carrier-derived one (Section 9.17). NO solver reads it today - ``gate_modes`` has
    no eta override and every caller builds ``GateModes`` from the device's crystal - which is the beliefs-vs-truth gap
    recorded as ``conv.solvers_read_the_device_modes`` rather than a capability this docstring may claim."""
    fitted_at_s: float = 0.0
    """The laboratory time the table was assembled at (the age of its youngest entry is ``fitted_at_s`` too; Section 7.5)."""

    def waveform_for(self, pair: Sequence[int]) -> Waveform | None:
        """The pair's entangling waveform under either key order, None when uncalibrated."""
        a, b = int(pair[0]), int(pair[1])
        return self.ms.get((a, b)) or self.ms.get((b, a))

    def is_current_for(self, device_hash: str) -> bool:
        """Whether the table was fitted for the device with this canonical hash (Section 7.5: a table is invalidated, never
        silently regenerated, when a device parameter changes)."""
        return self.device_hash == device_hash

    def kind_of(self, key: str) -> EntryKind:
        """The kind of the entry under the flat ``key`` of ``entries()``: its own ``kind`` when set, else the kind of the table
        field that holds it (``ENTRY_KINDS``; the waveform phases are setpoints)."""
        entry = self.entries()[key]
        if entry.kind is not None:
            return entry.kind
        name = key.split("[", 1)[0].split(".", 1)[0]
        return ENTRY_KINDS[name]

    def with_params(self, **overrides: Any) -> CalibrationTable:
        """This table with the given fields set (docs/api_implementation_plan.md 2.4; Cirq's ``NoiseProperties.with_params``):
        a mapping field (``rabi``, ``modes``, ``ms``, ...) takes the given entries MERGED over the existing ones (an ``ms``
        pair under either key order replaces the stored pair), a scalar field (``field``, ``fitted_at_s``, ``seed``,
        ``surrogate``) is replaced; the device hash is the device's and cannot be set; an unknown name is refused. The
        scheduler plays the result as written (a miscalibration set by hand, the app's Level 2 request), so the caller
        owns the physics of the edit."""
        import dataclasses

        names = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(k for k in overrides if k not in names or k == "device_hash")
        if unknown:
            raise TypeError(
                f"CalibrationTable.with_params: unknown or unsettable field(s) {unknown}; the fields are "
                f"{sorted(names - {'device_hash'})}"
            )
        changes: dict[str, Any] = {}
        for name, value in overrides.items():
            current = getattr(self, name)
            if isinstance(current, dict):
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"CalibrationTable.with_params: {name} takes a mapping of entries, got {type(value).__name__}"
                    )
                merged = dict(current)
                for key, entry in value.items():
                    stored = key
                    if name == "ms" and isinstance(key, tuple) and len(key) == 2:
                        a, b = int(key[0]), int(key[1])
                        stored = (a, b) if (a, b) in merged or (b, a) not in merged else (b, a)
                    merged[stored] = entry
                changes[name] = merged
            else:
                changes[name] = value
        return dataclasses.replace(self, **changes)

    def updated_with(
        self, result: Any, *, fitted_at_s: float | None = None, sample_id: int = 0
    ) -> CalibrationTable:
        """The proposal a typed ``ExperimentResult`` makes for this table (docs/api_implementation_plan.md 2.4; Qibocal's
        rule: an update is a proposal a caller gates on ``result.quality`` before adopting): the entries the result fitted,
        each a ``CalEntry`` stamped with the value and uncertainty, ``calibrated`` (or ``uncalibrated`` when the fit did not
        converge), the experiment's name, its provenance id, ``fitted_at_s`` (default: this table's) and ``sample_id``, set
        through ``with_params``; the table's own ``fitted_at_s`` moves to the youngest entry. The result names the fields
        it sets (``ExperimentResult.table_updates``); a result that sets none (a parity scan, an image) is refused."""
        at = float(self.fitted_at_s if fitted_at_s is None else fitted_at_s)
        updates = result.table_updates(fitted_at_s=at, sample_id=int(sample_id))
        return self.with_params(**updates, fitted_at_s=max(float(self.fitted_at_s), at))

    def entries(self, kind: EntryKind | None = None) -> dict[str, CalEntry]:
        """Every CalEntry of the table under a flat key (``rabi[(0, 2)]``, ``modes[3]``, ``field``, ...), waveform phases included;
        ``kind`` keeps the setpoints or the characterisation only (``kind_of``)."""
        if kind is not None:
            return {k: e for k, e in self.entries().items() if self.kind_of(k) == kind}
        out: dict[str, CalEntry] = {"field": self.field}
        for name in (
            "qubit_freq",
            "rabi",
            "stark",
            "crosstalk",
            "crosstalk_phase",
            "modes",
            "nbar",
            "micromotion",
            "detection",
            "heating",
            "lamb_dicke",
        ):
            for key, entry in getattr(self, name).items():
                out[f"{name}[{key!r}]"] = entry
        for pair, wf in self.ms.items():
            out[f"ms[{pair!r}].phi_s"] = wf.phi_s
            out[f"ms[{pair!r}].phi_m"] = wf.phi_m
        return out

    def uncalibrated(self) -> tuple[str, ...]:
        """The flat keys of every entry a fit could not establish (what the scheduler refuses to use)."""
        return tuple(k for k, e in self.entries().items() if e.status == "uncalibrated")

    def to_dict(self) -> dict[str, Any]:
        """The table as plain JSON-able values (``Result.to_dict``, 0.2.0): the identity fields, every ``CalEntry`` under its
        flat key (``entries()``), and per entangling pair the scalar summary of its waveform (duration, kind, chi_m, alpha_m
        and the two phase entries); the segments and Fourier coefficients, which may hold callables, are not carried, so a
        table read back has ``ms == {}`` and schedules no entangling gate."""
        waveforms: dict[str, Any] = {}
        for pair, wf in self.ms.items():
            waveforms[repr(pair)] = {
                "duration_s": float(wf.duration_s),
                "kind": wf.kind,
                "chi_m": {str(m): float(x) for m, x in wf.chi_m.items()},
                "alpha_m": {str(m): [float(a.real), float(a.imag)] for m, a in wf.alpha_m.items()},
                "phi_s": wf.phi_s.to_dict(),
                "phi_m": wf.phi_m.to_dict(),
            }
        return {
            "device_hash": self.device_hash,
            "seed": int(self.seed),
            "surrogate": bool(self.surrogate),
            "fitted_at_s": float(self.fitted_at_s),
            "entries": {
                key: entry.to_dict() for key, entry in self.entries().items() if not key.startswith("ms[")
            },
            "waveforms": waveforms,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CalibrationTable:
        """The inverse of :meth:`to_dict`: every scalar entry restored under its field and key, ``ms`` empty (module
        docstring of the method)."""
        import ast

        fields: dict[str, dict[Any, CalEntry]] = {
            name: {}
            for name in (
                "qubit_freq",
                "rabi",
                "stark",
                "crosstalk",
                "crosstalk_phase",
                "modes",
                "nbar",
                "micromotion",
                "detection",
                "heating",
                "lamb_dicke",
            )
        }
        field_entry: CalEntry | None = None
        for flat, value in dict(d["entries"]).items():
            entry = CalEntry.from_dict(value)
            if flat == "field":
                field_entry = entry
                continue
            name, _, rest = flat.partition("[")
            if name not in fields or not rest.endswith("]"):
                raise ValueError(f"unknown calibration entry key {flat!r}")
            fields[name][ast.literal_eval(rest[:-1])] = entry
        if field_entry is None:
            raise ValueError("a calibration table carries its field entry")
        return cls(
            device_hash=str(d["device_hash"]),
            seed=int(d["seed"]),
            surrogate=bool(d["surrogate"]),
            qubit_freq=fields["qubit_freq"],
            rabi=fields["rabi"],
            stark=fields["stark"],
            crosstalk=fields["crosstalk"],
            modes=fields["modes"],
            nbar=fields["nbar"],
            ms={},
            field=field_entry,
            micromotion=fields["micromotion"],
            detection=fields["detection"],
            heating=fields["heating"],
            crosstalk_phase=fields["crosstalk_phase"],
            lamb_dicke=fields["lamb_dicke"],
            fitted_at_s=float(d.get("fitted_at_s", 0.0)),
        )

    @classmethod
    def empty(cls, device_hash: str = "") -> CalibrationTable:
        """No calibration at all: every map empty and the field entry ``uncalibrated``, the table of a result that came from
        outside the simulator (``Result.from_ionq_v1_shots``, ``Result.from_dict`` of a summary without one)."""
        none = CalEntry(math.nan, math.nan, "uncalibrated", "none", "", 0.0, 0)
        return cls(
            device_hash=device_hash,
            seed=0,
            surrogate=False,
            qubit_freq={},
            rabi={},
            stark={},
            crosstalk={},
            modes={},
            nbar={},
            ms={},
            field=none,
            micromotion={},
            detection={},
            heating={},
        )


__all__ = [
    "ENTRY_KINDS",
    "EntryKind",
    "CalEntry",
    "CalibrationTable",
    "Leg",
    "Segment",
    "Waveform",
    "WaveformKind",
    "usable",
]
