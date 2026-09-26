"""The calibration table and its entries (PLAN.md Section 7.5).

``CalibrationTable`` is plain data: ``control.schedule`` reads it and never writes it, and ``control`` never imports
``calibration``. Every entry carries its status, the experiment that produced it, its provenance id, its age and the
noise sample it was fitted under.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

if TYPE_CHECKING:
    from qutip_trap.control.shaping import GateModes, Kernel

Leg = Literal["red", "blue"]
EntryKind = Literal["setpoint", "characterisation"]
"""Qibolab's partition of a calibration: a ``setpoint`` is what the scheduler programs (the frame frequency, the Rabi rate
the amplitude word is derived from, the light shift it compensates, the mode frequencies its sidebands sit at, the
entangling waveforms, the micromotion shims, the detection threshold and window); a ``characterisation`` is what was
measured about the device and programmed nowhere (the field, the occupations, the heating rates, the crosstalk ratios and
phases, the Lamb-Dicke parameters)."""
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
"""The kind of every table field (``CalibrationTable.kind_of``)."""
WaveformKind = Literal["ms", "light_shift", "gradient"]
"""``ms``: bichromatic red/blue legs on the spin flip; ``light_shift``: one beat note ("blue") on the state-dependent
light shift, sigma_z sigma_z; ``gradient``: the two microwave tones at -/+ delta of a near-field magnetic-gradient drive,
whose dressed sigma_z force carries J_2(4 Omega_mu/delta)."""
KeyedGroup = Literal[
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
]
"""The table fields that map keys to ``CalEntry`` records, in the order ``CalibrationTable.entries`` lists them."""
EntryGroup = Literal["field", KeyedGroup, "ms"]
"""An entry group of the table, what a calibration experiment reads and writes (``CalibrationTable.group``,
``CalibrationTable.with_group``): the field entry, a keyed map, or the phase entries of the entangling waveforms (``ms``),
in the order ``CalibrationTable.entries`` lists them."""
_ENTRY_GROUPS: Final[tuple[KeyedGroup, ...]] = get_args(KeyedGroup)
_GROUPS: Final[tuple[EntryGroup, ...]] = get_args(EntryGroup)


def _kind_of_key(key: str) -> EntryKind:
    """The kind of the table field a flat ``entries()`` key names."""
    return ENTRY_KINDS[key.split("[", 1)[0].split(".", 1)[0]]


@dataclass(frozen=True)
class CalEntry:
    """One calibrated number: the value and its uncertainty in the units of the table field that holds it, the status
    (``seed`` from the closed forms, ``calibrated`` by a simulated experiment, ``uncalibrated`` when a fit was refused), the
    experiment and the provenance id behind it, the laboratory time of the fit (s) and the noise sample it was fitted under."""

    value: float
    uncertainty: float
    status: Literal["seed", "calibrated", "uncalibrated"]
    experiment: str
    provenance_id: str
    fitted_at_s: float
    sample_id: int

    def __post_init__(self) -> None:
        if self.uncertainty < 0.0:
            raise ValueError("uncertainty must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """The entry as plain JSON-able values."""
        return {
            "value": float(self.value),
            "uncertainty": float(self.uncertainty),
            "status": self.status,
            "experiment": self.experiment,
            "provenance_id": self.provenance_id,
            "fitted_at_s": float(self.fitted_at_s),
            "sample_id": int(self.sample_id),
        }

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
        )

    @property
    def usable(self) -> bool:
        """A ``seed`` or ``calibrated`` entry may be scheduled from; an ``uncalibrated`` one refuses."""
        return self.status != "uncalibrated"


def usable(entry: CalEntry | None) -> bool:
    """True for a present ``seed`` or ``calibrated`` entry (what the scheduler and a downstream fit may read)."""
    return entry is not None and entry.usable


@dataclass(frozen=True)
class Segment:
    """One segment of a calibrated entangling pulse. Amplitudes and detunings are constants (the segmented AM family) or
    callables of the time since the SEGMENT start (a Fourier-sine amplitude, an FM detuning schedule); the scheduler plays
    each segment as one ``Pulse`` per ion so that step discontinuities fall on integration boundaries."""

    duration_s: float
    amplitude_hz: dict[tuple[int, Leg], float | Callable[[float], float]]
    """Omega per (ion, leg), leg in {red, blue}; a leg imbalance lives in the two legs' own amplitudes."""
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
    """A calibrated entangling pulse: what the scheduler plays."""

    segments: tuple[Segment, ...]
    duration_s: float
    phi_s: CalEntry
    phi_m: CalEntry
    chi_m: dict[int, float]
    """Per-mode entangling angle at closure, SIGNED: the pulse applies exp(+i sum_m chi_m sigma sigma)."""
    alpha_m: dict[int, complex]
    """Per-mode residual displacement at closure (the worst gate ion's)."""
    kind: WaveformKind = "ms"

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        total = sum(s.duration_s for s in self.segments)
        if not math.isclose(total, self.duration_s, rel_tol=1e-9, abs_tol=1e-15):
            raise ValueError(f"segment durations sum to {total}, not duration_s = {self.duration_s}")
        if len({s.ions for s in self.segments}) != 1:
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
        return self.segments[0].ions

    @property
    def chi_total_rad(self) -> float:
        """The signed two-body angle: exp(+i chi sigma sigma) = XX(-chi)."""
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
        phi_m_rad: float = 0.0,
        detuning_side: Literal["inside", "outside"] = "inside",
        all_modes: bool = True,
        kind: Literal["ms", "light_shift"] = "ms",
    ) -> Waveform:
        """The waveform of ``control.shaping.symmetric_pulse``: one square segment closing ``gate_mode`` after ``loops``
        loops (tau = 2 pi K/eps), the amplitude from |chi| = chi_target over the pair's modes (a ``gradient`` waveform is
        not covered by this closure algebra: its force is a Bessel function of the tone amplitude)."""
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
            phi_m_rad=phi_m_rad,
            detuning_side=detuning_side,
            all_modes=all_modes,
            kind=kind,
        ).waveform


@dataclass(frozen=True)
class CalibrationTable:
    """What the scheduler believes about the device: the per-ion, per-(ion, beam), per-pair and per-mode entries, each a
    ``CalEntry``, and the entangling ``Waveform`` per pair, keyed to the ``Device.hash()`` and the noise ``seed`` they were
    fitted under; ``surrogate`` says whether it came from the closed forms with spot checks or from simulated experiments.

    Lookup, most specific first: a waveform per PAIR under either key order (``waveform_for``); a Rabi, Stark or crosstalk
    entry per (ion, beam) or (ion, neighbour), the beam the drive's first beam (``GateDrive.table_key_beam``, -1 for a
    microwave drive); a qubit frequency, mode frequency, occupation or heating rate per ion or mode; the field and the
    detection entries once. An absent or ``uncalibrated`` entry refuses to schedule and never falls back to a less
    specific one. Edits are proposals returning a new table: ``with_params`` sets entries by field, ``updated_with`` maps a
    typed ``ExperimentResult`` onto the entries it fitted."""

    device_hash: str
    seed: int
    surrogate: bool
    qubit_freq: dict[int, CalEntry]
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
    """arg(epsilon_ij) per (ion, neighbour) from the crosstalk scan's phase measurement; absent = 0."""
    lamb_dicke: dict[tuple[int, int], CalEntry] = dc_field(default_factory=dict)
    """|eta_{i,m}| per (ion, mode) from the sideband Rabi frequency (C0 inside); recorded, not read by the pulse solvers,
    which build their ``GateModes`` from the device's crystal."""
    fitted_at_s: float = 0.0
    """The laboratory time the table was assembled at."""

    def waveform_for(self, pair: Sequence[int]) -> Waveform | None:
        """The pair's entangling waveform under either key order, None when uncalibrated."""
        a, b = int(pair[0]), int(pair[1])
        return self.ms.get((a, b)) or self.ms.get((b, a))

    def is_current_for(self, device_hash: str) -> bool:
        """Whether the table was fitted for the device with this canonical hash (a table is invalidated, never silently
        regenerated, when a device parameter changes)."""
        return self.device_hash == device_hash

    def kind_of(self, key: str) -> EntryKind:
        """The kind of the entry under the flat ``key`` of ``entries()``: the kind of the table field that holds it (the
        waveform phases are setpoints)."""
        if key not in self.entries():
            raise KeyError(key)
        return _kind_of_key(key)

    def with_params(self, **overrides: Any) -> CalibrationTable:
        """This table with the given fields set: a mapping field (``rabi``, ``modes``, ``ms``, ...) takes the given entries
        MERGED over the existing ones (an ``ms`` pair under either key order replaces the stored pair), a scalar field is
        replaced; the device hash cannot be set and an unknown name is refused. The scheduler plays the result as written,
        so the caller owns the physics of the edit."""
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
        """The proposal a typed ``ExperimentResult`` makes for this table (a caller gates on ``result.quality`` before
        adopting it): the entries the result fitted (``ExperimentResult.table_updates``), stamped ``calibrated`` (or
        ``uncalibrated`` when the fit did not converge) with the experiment's name, its provenance id, ``fitted_at_s``
        (default: this table's) and ``sample_id``; the table's own ``fitted_at_s`` moves to the youngest entry. A result
        that sets no entry (a parity scan, an image) is refused."""
        at = float(self.fitted_at_s if fitted_at_s is None else fitted_at_s)
        updates = result.table_updates(fitted_at_s=at, sample_id=int(sample_id))
        return self.with_params(**updates, fitted_at_s=max(float(self.fitted_at_s), at))

    def entries(self, kind: EntryKind | None = None) -> dict[str, CalEntry]:
        """Every CalEntry of the table under a flat key (``rabi[(0, 2)]``, ``modes[3]``, ``field``, ...), waveform phases
        included; ``kind`` keeps the setpoints or the characterisation only."""
        if kind is not None:
            return {k: e for k, e in self.entries().items() if _kind_of_key(k) == kind}
        out: dict[str, CalEntry] = {}
        for name in _GROUPS:
            out.update(self.group(name))
        return out

    def group(self, name: EntryGroup) -> dict[str, CalEntry]:
        """The entries of one entry group under their flat keys (``entries``): the field entry, every entry of a keyed map,
        or both phase entries of every entangling waveform (``ms``)."""
        if name == "field":
            return {"field": self.field}
        if name == "ms":
            phases: dict[str, CalEntry] = {}
            for pair, wf in self.ms.items():
                phases[f"ms[{pair!r}].phi_s"] = wf.phi_s
                phases[f"ms[{pair!r}].phi_m"] = wf.phi_m
            return phases
        return {f"{name}[{key!r}]": entry for key, entry in getattr(self, name).items()}

    def with_group(self, name: EntryGroup, entry: Callable[[CalEntry], CalEntry]) -> CalibrationTable:
        """This table with every entry of the group ``name`` (``group``) replaced by ``entry`` of it: a proposal returning a
        new table, as ``with_params`` is."""
        import dataclasses

        if name == "field":
            return dataclasses.replace(self, field=entry(self.field))
        if name == "ms":
            return dataclasses.replace(
                self,
                ms={
                    pair: dataclasses.replace(wf, phi_s=entry(wf.phi_s), phi_m=entry(wf.phi_m))
                    for pair, wf in self.ms.items()
                },
            )
        changed: dict[str, Any] = {name: {key: entry(e) for key, e in getattr(self, name).items()}}
        return dataclasses.replace(self, **changed)

    def uncalibrated(self) -> tuple[str, ...]:
        """The flat keys of every entry a fit could not establish (what the scheduler refuses to use)."""
        return tuple(k for k, e in self.entries().items() if e.status == "uncalibrated")

    def to_dict(self) -> dict[str, Any]:
        """The table as plain JSON-able values: the identity fields, every ``CalEntry`` under its flat key, and per
        entangling pair the scalar summary of its waveform (duration, kind, chi_m, alpha_m and the two phase entries); the
        segments, which may hold callables, are not carried, so a table read back has ``ms == {}``."""
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
        """The inverse of :meth:`to_dict`: every scalar entry restored under its field and key, ``ms`` empty."""
        import ast

        fields: dict[str, dict[Any, CalEntry]] = {name: {} for name in _ENTRY_GROUPS}
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
            fitted_at_s=float(d["fitted_at_s"]),
        )

    @classmethod
    def empty(cls, device_hash: str = "") -> CalibrationTable:
        """No calibration at all: every map empty and the field entry ``uncalibrated``, the table of a result that came from
        outside the simulator."""
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
