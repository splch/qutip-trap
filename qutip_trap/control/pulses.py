"""Tones, drives and pulses (PLAN.md Sections 3.3, 4.3, 5.2, 7.4; Appendix E).

Conventions (Section 13): a tone's detuning mu is measured from the carrier and delta_{i,m} = mu_i - omega_m is
always two-indexed; the drive term is (hbar/2) sum_tones Omega(t) e^{-i(mu t - phi(t))} sigma_+ (x) prod_m
D_m(i eta) + h.c. (Section 5.7); the effective wavevector is Delta k = k_1 - k_2 with beam 1 the higher-frequency
beam absorbed from the lower qubit level (Wineland 2003 Eq. 2.3, Section 13; the row's 'coupling to the upper level'
wording is ambiguous), |Delta k| = 2k sin(theta_cross/2) for a Raman pair, k for one beam and 0 for a microwave.
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
"""Appendix E's five kinds plus ``light_shift`` (M4, Section 4.4.4): a Raman beam pair whose beat note sits near a MODE
frequency and drives the qubit through its state-dependent two-photon light shift instead of a spin flip."""

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
    """The level structure of a light-shift (sigma_z sigma_z) drive relative to its force amplitude (Section 4.4.4).

    With Omega_gg the two-photon self-coupling of qubit level g under the beam pair (light/raman.py) and Omega_LS =
    (Omega_upup - Omega_dndn)/2 the tone's envelope, ``level_weights`` = (Omega_dndn, Omega_upup)/Omega_LS (they differ by
    exactly 2), so the drive term on ion i is (1/2) sum_tones Omega_LS(t) e^{-i(mu t - phi)} [w_dn P_dn + w_up P_up] (x) D_i
    + h.c. = Omega_LS cos(...) [sigma_z + (w_up + w_dn)/2] (x) ...: Zhu-Monroe-Duan's H = hbar Omega_j cos(Delta k . q_j +
    mu t) sigma_z^j plus the spin-independent force. ``spin_flip_weight`` = Omega_R/Omega_LS is the ordinary Raman coupling
    the same beams drive, far off resonance because the beat note mu sits near a mode and not near the qubit frequency
    ``qubit_freq_hz``: in the qubit frame it rotates at mu - omega_0 and the builder keeps it unless its off-resonant
    excitation is negligible (recorded either way).
    """

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
    detuning_hz: Callable[[float], float] | float
    """mu(t) from the carrier (ordinary Hz in the public API); for a ``light_shift`` drive the BEAT NOTE itself, which sits
    near a mode frequency and not near the qubit frequency (Section 4.4.4)."""
    phase_rad: Callable[[float], float] | float
    """phi_tone(t) in the ion frame (Section 5.2)."""
    envelope_hz: Callable[[float], float] | np.ndarray | float
    """Omega(t), the Rabi frequency in the (hbar Omega/2) convention, as an ordinary frequency: a callable of the
    time since the pulse start, an array uniformly sampled over the pulse (cubic spline), or a constant (square)."""
    theta_bessel_rad: float | None = None
    """Kick backend: the Bessel argument of exp[i Theta_B sin(Delta k x + phi) sigma_x]; a perfect
    spin-dependent kick sits at sum_k Theta_{B,k} = pi (Appendix E, Run 5 amendment)."""


@dataclass(frozen=True)
class Drive:
    """A physical drive on a set of ions (Section 3.3)."""

    kind: DriveKind
    ions: tuple[int, ...]
    tones: tuple[Tone, ...]
    beams: tuple[int, ...]
    """Indices into Device.beams: one for optical_E1/E2, two for raman (k_1 - k_2), none for microwave."""
    stark_shift_hz: Callable[[float], float] | float
    crosstalk: dict[int, complex]
    """epsilon_ij onto neighbours: a Rabi (amplitude) ratio, not an intensity ratio (Section 13)."""
    rf_locked: bool = False
    rf_phase_rad: float | None = None
    """Pulse start relative to the trap rf (Section 4.3.6); unlocked = averaged."""
    comb: CombSpec | None = None
    """Set for a mode-locked Raman drive; then ``tones`` comes from comb.tones() and stark_shift_hz from comb.stark4_hz()."""
    light_shift: LightShiftCouplings | None = None
    """Required for ``kind == "light_shift"`` and refused otherwise (Section 4.4.4)."""
    programmed: bool = False
    """True for a drive the scheduler built from the CalibrationTable (Section 7.3): its envelopes are REQUESTED Rabi
    frequencies in the table's units, its Stark shift and crosstalk the table's beliefs, and the played chain of
    ``control.played`` converts them into what the ions see through the device's derived values (M8). False for a drive built
    from a derived ``DerivedDrive`` (the experiments), whose values are already physical."""

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
        """The effective wavevector, DERIVED from the beams' wavelengths and directions, never a free field.

        Raman: k_1 - k_2 with beam 1 the higher-frequency beam absorbed from the lower qubit level (Section 13, "Effective wavevector");
        single-photon optical: k k_hat; microwave and gradient: 0. Appendix E declares this as a property; it
        takes the device's beam list here because a ``Drive`` stores indices into ``Device.beams``.
        """
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


# ---- picklable time functions (PLAN.md Section 11.3 item 9; M9b) -----------------------------------------------------------------
# The parallel maps of Section 11.3 pickle every QobjEvo coefficient, so nothing on the pulse path may be a lambda or a closure:
# the envelopes, phases and detunings a pulse carries, the hardware chain's shaped and filtered envelopes, the played chain's
# rescaled ones and the scattering channel's intensity scale are all instances of the classes below (or arrays and constants).


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
    """A uniformly sampled array over [0, duration] as a clamped cubic spline (Section 5.5: programmed envelopes are sampled
    at 20 times their bandwidth and interpolated with the default cubic spline), times ``scale``."""

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
    """A uniformly sampled array over [0, duration] interpolated linearly (the scattering channel's intensity scale)."""

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
    ``scale``: the one conversion the builder, the hardware chain and the scattering channel share; picklable when its input is."""
    if callable(value):
        return ScaledFn(value, scale)
    if isinstance(value, np.ndarray):
        return SplineFn(value, duration_s, scale)
    return ConstantFn(scale * float(value))


def fingerprint_pulse(pulse: Pulse, n_samples: int = 33) -> tuple[object, ...]:
    """A value fingerprint of a pulse (the GATE_LOCAL extraction cache, the engine's propagator cache): kind, ions, beams,
    absolute times, every tone sampled over the pulse (a callable envelope or detuning differs by its VALUES, never by the source
    text a closure shares), the Stark shift, the crosstalk and the light-shift couplings."""
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


__all__ = [
    "ConstantFn",
    "Drive",
    "DriveKind",
    "InterpFn",
    "LightShiftCouplings",
    "Pulse",
    "ScaledFn",
    "SplineFn",
    "Tone",
    "as_time_function",
    "fingerprint_pulse",
]
