"""The calibration table and its entries: plain data the scheduler reads and never writes (``control`` never imports
``calibration``); every entry carries its status, experiment, provenance id, age and noise sample."""

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
"""A ``setpoint`` is what the scheduler programs (frame frequency, Rabi rate, compensated light shift, mode frequencies,
waveforms, shims, detection); a ``characterisation`` was measured about the device and is programmed nowhere."""
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
"""``ms``: red/blue legs on the spin flip; ``light_shift``: one beat note ("blue") on the sigma_z light shift; ``gradient``:
two microwave tones at -/+ delta whose dressed sigma_z force carries J_2(4 Omega_mu/delta)."""


@dataclass(frozen=True)
class CalEntry:
    """One calibrated number, in the units of the table field that holds it; ``status`` is ``seed`` (closed forms),
    ``calibrated`` (a simulated experiment) or ``uncalibrated`` (a refused fit)."""

    value: float
    uncertainty: float
    status: Literal["seed", "calibrated", "uncalibrated"]
    experiment: str
    provenance_id: str
    fitted_at_s: float
    sample_id: int
    """The noise sample the entry was fitted under."""
    kind: EntryKind | None = dc_field(default=None, metadata={"hash": "skip_default"})
    """None takes the kind of the table field that holds the entry (``ENTRY_KINDS``); left out of the digest while unset."""

    def to_dict(self) -> dict[str, Any]:
        """The entry as plain JSON-able values; ``kind`` only when set."""
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
        """A ``seed`` or ``calibrated`` entry may be scheduled from; an ``uncalibrated`` one refuses."""
        return self.status != "uncalibrated"


def usable(entry: CalEntry | None) -> bool:
    """True for a present ``seed`` or ``calibrated`` entry."""
    return entry is not None and entry.usable


@dataclass(frozen=True)
class Segment:
    """One segment of a calibrated entangling pulse; amplitudes and detunings are constants or callables of segment time."""

    duration_s: float
    amplitude_hz: dict[tuple[int, Leg], float | Callable[[float], float]]
    """Omega per (ion, leg), a magnitude; a leg imbalance Omega(1 +- xi) is carried here, not as a table entry."""
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
    """A calibrated entangling pulse, either segmented or Fourier-parameterized."""

    segments: tuple[Segment, ...] | None
    fourier: tuple[complex, ...] | None
    duration_s: float
    phi_s: CalEntry
    phi_m: CalEntry
    chi_m: dict[int, float]
    """Per-mode entangling angle at closure, signed: the pulse applies exp(+i sum_m chi_m sigma sigma)."""
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
        phi_s_rad: float = 0.0,
        phi_m_rad: float = 0.0,
        imbalance: float = 0.0,
        detuning_side: Literal["inside", "outside"] = "inside",
        all_modes: bool = True,
        kind: Literal["ms", "light_shift"] = "ms",
    ) -> Waveform:
        """The equal-envelope symmetric-detuning pulse: one square segment, tones at -/+ (omega_g -/+ eps) closing
        ``gate_mode`` after ``loops`` loops (tau = 2 pi K/eps), amplitude from |chi| = chi_target over the pair's modes (for
        one mode eta Omega/eps = 1/(2 sqrt K)). No ``gradient`` kind: its force is not linear in the tone amplitude."""
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
    """What the scheduler believes about the device: ``CalEntry`` values per ion, (ion, beam), pair and mode and the
    entangling ``Waveform`` per pair, keyed to ``Device.hash()`` and the noise ``seed``. The beam key is the drive's first
    beam (-1 for a microwave drive); a lookup never falls back to a less specific entry, and an absent or ``uncalibrated``
    entry refuses to schedule."""

    device_hash: str
    seed: int
    surrogate: bool
    """True for closed forms with spot checks, False for full simulated experiments."""
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
    """arg(epsilon_ij) per (ion, neighbour); absent = 0."""
    lamb_dicke: dict[tuple[int, int], CalEntry] = dc_field(default_factory=dict)
    """|eta_{i,m}| per (ion, mode) from the sideband Rabi frequency; recorded only (the solvers use the device's eta)."""
    fitted_at_s: float = 0.0

    def waveform_for(self, pair: Sequence[int]) -> Waveform | None:
        """The pair's entangling waveform under either key order, None when uncalibrated."""
        a, b = int(pair[0]), int(pair[1])
        return self.ms.get((a, b)) or self.ms.get((b, a))

    def is_current_for(self, device_hash: str) -> bool:
        """Whether the table was fitted for the device with this canonical hash."""
        return self.device_hash == device_hash

    def kind_of(self, key: str) -> EntryKind:
        """The kind of the entry under the flat ``key`` of ``entries()``: its own, else its field's (``ENTRY_KINDS``)."""
        entry = self.entries()[key]
        if entry.kind is not None:
            return entry.kind
        name = key.split("[", 1)[0].split(".", 1)[0]
        return ENTRY_KINDS[name]

    def with_params(self, **overrides: Any) -> CalibrationTable:
        """This table with the given fields set: a mapping field takes the given entries merged over the existing ones (an
        ``ms`` pair under either key order replaces the stored pair), a scalar field is replaced. ``device_hash`` and
        unknown names raise TypeError; no physics check is made."""
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
        """The table a typed ``ExperimentResult`` proposes: its ``table_updates`` (stamped ``fitted_at_s``, default this
        table's, and ``sample_id``) set through ``with_params``, the table's ``fitted_at_s`` moved to the youngest entry.
        A proposal: callers gate adoption on ``result.quality``."""
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
        """The table as plain JSON-able values: every ``CalEntry`` under its flat key and a scalar summary per waveform.
        Segments and Fourier coefficients (possibly callables) are not carried: a table read back has ``ms == {}``."""
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
        """The inverse of :meth:`to_dict`, with ``ms`` empty."""
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
        """No calibration: every map empty and the field entry ``uncalibrated`` (for a result from outside the simulator)."""
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
