"""Tones, drives and pulses, and the picklable time functions they carry.

A tone's detuning mu is measured from the carrier (delta_{i,m} = mu_i - omega_m); the drive term is (hbar/2) sum_tones
Omega(t) e^{-i(mu t - phi(t))} sigma_+ (x) prod_m D_m(i eta) + h.c.; Delta k = k_1 - k_2 with beam 1 the higher-frequency
beam absorbed from the lower qubit level (Wineland 2003 Eq. 2.3).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from qutip_trap.light.beams import Beam

if TYPE_CHECKING:
    from qutip_trap.light.comb import CombSpec

DriveKind = Literal["raman", "optical_E1", "optical_E2", "microwave", "gradient", "light_shift"]
"""The drive kinds; ``light_shift``: a Raman pair beating near a mode frequency, a state-dependent force, not a spin flip."""

_BEAMS_PER_KIND: dict[str, int] = {
    "raman": 2,
    "optical_E1": 1,
    "optical_E2": 1,
    "microwave": 0,
    "gradient": 0,
    "light_shift": 2,
}


@dataclass(frozen=True)
class LightShiftCouplings:
    """The level structure of a light-shift (sigma_z sigma_z) drive relative to its tone envelope Omega_LS = (Omega_upup -
    Omega_dndn)/2, Omega_gg the two-photon self-coupling of level g: ``level_weights`` = (Omega_dndn, Omega_upup)/Omega_LS
    differ by exactly 2, the drive term on ion i being (1/2) sum_tones Omega_LS(t) e^{-i(mu t - phi)} [w_dn P_dn + w_up P_up]
    (x) D_i + h.c.; ``spin_flip_weight`` = Omega_R/Omega_LS is the same beams' ordinary Raman coupling, off resonant by
    mu - omega_0 (omega_0 from ``qubit_freq_hz``)."""

    level_weights: tuple[complex, complex]
    spin_flip_weight: complex
    qubit_freq_hz: float

    def __post_init__(self) -> None:
        w_dn, w_up = self.level_weights
        if abs((w_up - w_dn) - 2.0) > 1e-9:
            raise ValueError(
                "level_weights are (Omega_dndn, Omega_upup)/Omega_LS with Omega_LS = (up - dn)/2: they differ by 2"
            )
        if self.qubit_freq_hz <= 0.0:
            raise ValueError("qubit_freq_hz is the positive transition frequency")


@dataclass(frozen=True)
class Tone:
    """One frequency component of a drive; a callable field is a function of the time since the pulse start."""

    detuning_hz: Callable[[float], float] | float
    """mu(t) from the carrier; for a ``light_shift`` drive the beat note itself, near a mode frequency."""
    phase_rad: Callable[[float], float] | float
    """phi(t) in the ion frame."""
    envelope_hz: Callable[[float], float] | np.ndarray | float
    """Omega(t) in the (hbar Omega/2) convention: a callable, an array uniformly sampled over the pulse (cubic spline)
    or a constant (square)."""
    theta_bessel_rad: float | None = None
    """Kick backend: Theta_B in exp[i Theta_B sin(Delta k x + phi) sigma_x]; a perfect spin-dependent kick has
    sum_k Theta_{B,k} = pi."""


@dataclass(frozen=True)
class Drive:
    """A physical drive on a set of ions."""

    kind: DriveKind
    ions: tuple[int, ...]
    tones: tuple[Tone, ...]
    beams: tuple[int, ...]
    """Indices into Device.beams: two for raman and light_shift (k_1 - k_2), one for optical_E1/E2, none otherwise."""
    stark_shift_hz: Callable[[float], float] | float
    crosstalk: dict[int, complex]
    """epsilon_ij onto neighbour j: a Rabi (amplitude) ratio, not an intensity ratio."""
    rf_locked: bool = False
    rf_phase_rad: float | None = None
    """Pulse start relative to the trap rf, for an rf-locked pulse only (unlocked = averaged)."""
    comb: CombSpec | None = None
    """Set for a mode-locked Raman drive; then ``tones`` comes from comb.tones() and stark_shift_hz from comb.stark4_hz()."""
    light_shift: LightShiftCouplings | None = None
    """Required for ``kind == "light_shift"`` and refused otherwise."""
    programmed: bool = False
    """True for the CalibrationTable's beliefs (requested Rabi frequencies, made physical by ``control.played``)."""

    def __post_init__(self) -> None:
        if not self.ions:
            raise ValueError("a Drive addresses at least one ion")
        expected = _BEAMS_PER_KIND[self.kind]
        if len(self.beams) != expected:
            raise ValueError(f"a {self.kind} drive references {expected} beam(s), got {len(self.beams)}")
        if not self.rf_locked and self.rf_phase_rad is not None:
            raise ValueError("rf_phase_rad is only meaningful for an rf-locked pulse (unlocked = averaged)")
        if self.comb is not None and self.kind != "raman":
            raise ValueError("a frequency comb generates a Raman drive")
        if (self.kind == "light_shift") != (self.light_shift is not None):
            raise ValueError("a light_shift drive carries its LightShiftCouplings and no other kind does")

    def delta_k(self, beams: Sequence[Beam]) -> np.ndarray:
        """The effective wavevector from ``Device.beams``: k_1 - k_2 for a beam pair, k k_hat for one beam, 0 for none."""
        if self.kind in ("raman", "light_shift"):
            k1 = beams[self.beams[0]].k_vector()
            k2 = beams[self.beams[1]].k_vector()
            return np.asarray(k1 - k2, dtype=float)
        if self.kind in ("optical_E1", "optical_E2"):
            return np.asarray(beams[self.beams[0]].k_vector(), dtype=float)
        return np.zeros(3)


@dataclass(frozen=True)
class Pulse:
    """A ``Drive`` with start and end times and metadata linking it to the gate it implements."""

    drive: Drive
    t_start_s: float
    t_end_s: float
    gate_id: str | None
    closes_modes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.t_end_s <= self.t_start_s:
            raise ValueError("a pulse must end after it starts")

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s


# ---- picklable time functions: the parallel maps pickle every QobjEvo coefficient, so no lambdas or closures ----


@dataclass(frozen=True)
class ConstantFn:
    """tau -> value."""

    value: float

    def __call__(self, tau: float) -> float:
        return self.value


@dataclass(frozen=True, eq=False)
class ScaledFn:
    """tau -> factor x fn(tau) (as a float); picklable when ``fn`` is."""

    fn: Callable[[float], float]
    factor: float = 1.0

    def __call__(self, tau: float) -> float:
        return self.factor * float(self.fn(tau))


class SplineFn:
    """A uniformly sampled array over [0, duration] as a cubic spline times ``scale``, with tau clamped to the interval."""

    __slots__ = ("_spline", "duration_s", "samples", "scale")

    def __init__(self, samples: np.ndarray, duration_s: float, scale: float = 1.0) -> None:
        from scipy.interpolate import CubicSpline

        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("a sampled envelope needs at least two samples")
        self.samples = arr
        self.duration_s = float(duration_s)
        self.scale = float(scale)
        self._spline = CubicSpline(
            np.linspace(0.0, self.duration_s, arr.size), self.scale * arr, extrapolate=True
        )

    def __call__(self, tau: float) -> float:
        return float(self._spline(min(max(tau, 0.0), self.duration_s)))

    def __reduce__(self) -> tuple[object, ...]:
        return (SplineFn, (self.samples, self.duration_s, self.scale))


class InterpFn:
    """A uniformly sampled array over [0, duration], interpolated linearly."""

    __slots__ = ("duration_s", "grid", "samples")

    def __init__(self, samples: np.ndarray, duration_s: float) -> None:
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("a sampled envelope needs at least two samples")
        self.samples = arr
        self.duration_s = float(duration_s)
        self.grid = np.linspace(0.0, self.duration_s, arr.size)

    def __call__(self, tau: float) -> float:
        return float(np.interp(tau, self.grid, self.samples))

    def __reduce__(self) -> tuple[object, ...]:
        return (InterpFn, (self.samples, self.duration_s))


def as_time_function(
    value: Callable[[float], float] | np.ndarray | float, duration_s: float, *, scale: float = 1.0
) -> Callable[[float], float]:
    """A pulse-local callable of tau from a constant, a callable or a uniformly sampled array over [0, duration], times
    ``scale``; picklable when its input is."""
    if callable(value):
        return ScaledFn(value, scale)
    if isinstance(value, np.ndarray):
        return SplineFn(value, duration_s, scale)
    return ConstantFn(scale * float(value))


def fingerprint_pulse(pulse: Pulse, n_samples: int = 33) -> tuple[object, ...]:
    """A value fingerprint of a pulse for caches: every callable is sampled at ``n_samples`` points over the pulse, so
    pulses compare by their values, never by function identity."""
    grid = np.linspace(0.0, pulse.duration_s, n_samples)

    def sampled(v: object) -> object:
        if callable(v):
            return np.array([float(v(x)) for x in grid])
        if isinstance(v, np.ndarray):
            return np.asarray(v, dtype=float)
        return float(v)  # type: ignore[arg-type]

    d = pulse.drive
    tones = tuple(
        (sampled(t.detuning_hz), sampled(t.phase_rad), sampled(t.envelope_hz), t.theta_bessel_rad)
        for t in d.tones
    )
    ls = None
    if d.light_shift is not None:
        ls = (d.light_shift.level_weights, d.light_shift.spin_flip_weight, d.light_shift.qubit_freq_hz)
    return (
        d.kind,
        d.ions,
        d.beams,
        pulse.t_start_s,
        pulse.t_end_s,
        pulse.gate_id,
        tones,
        sampled(d.stark_shift_hz),
        tuple(sorted((int(j), complex(e)) for j, e in d.crosstalk.items())),
        d.rf_locked,
        d.rf_phase_rad,
        d.programmed,
        ls,
    )
