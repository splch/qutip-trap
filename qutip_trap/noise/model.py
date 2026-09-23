"""The ``NoiseModel`` record and its two products: collapse operators and dynamical samples.

A spectrum's white level becomes a Lindblad operator and its tabulated band a sampled trajectory; a ``Drift`` is a
quasi-static offset per sample. Spectra are two-sided in angular frequency: ``S_E`` (V/m)^2/(rad/s), ``S_B``
T^2/(rad/s), ``laser_phase`` rad^2/(rad/s), ``laser_intensity`` (dI/I)^2/(rad/s), ``rf_amplitude_noise``
(dV/V)^2/(rad/s); ``Drift`` units are in ``DRIFT_UNITS``. ``rf_phase_noise`` is recorded and not yet applied.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from qutip_trap.noise.processes import (
    TAU_C_OVERSAMPLE,
    Trajectory,
    correlated_normals,
    mains_trajectory,
    synthesize,
    time_grid,
)
from qutip_trap.noise.sampling import (
    KEY_FIELD_OFFSET_T,
    KEY_INTENSITY_TRAJECTORY,
    KEY_LASER_OFFSET_HZ,
    KEY_LASER_PHASE_TRAJECTORY,
    KEY_MAINS_PHASE,
    KEY_RABI_SCALE,
    KEY_RF_FRACTION,
    KEY_RF_FRACTION_TRAJECTORY,
    NoiseSample,
    key_beam_offset_m,
    key_beam_phase_rad,
    key_beam_phase_trajectory_rad,
    key_mode_offset_hz,
    key_position_offset_m,
    key_qubit_offset_hz,
    key_qubit_trajectory_hz,
)
from qutip_trap.noise.spectra import Collisions, Drift, Mains, NoiseSpectrum
from qutip_trap.units import E_C, TWO_PI

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.channels import CollapseOp
    from qutip_trap.hilbert.space import HilbertSpace

RF_DERIVED_FAMILIES = ("transverse_1", "transverse_2")
"""The mode families whose frequencies follow the rf amplitude (omega_sec proportional to V_rf)."""

GAUSS_PER_TESLA = 1.0e4


DRIFT_UNITS: dict[str, str] = {
    "rf_amplitude_drift": "1",
    "mode_drift_differential": "Hz",
    "rabi_drift": "1",
    "beam_phase_drift": "rad",
    "field_drift": "T",
    "stray_field_drift": "V/m",
    "pointing_drift": "m",
    "laser_frequency_drift": "Hz",
}
"""The unit of each ``Drift`` field's rms and ramp ("1" = fractional). ``mode_drift_differential`` is drawn per mode,
``beam_phase_drift`` per beam, ``pointing_drift`` per beam and transverse direction, ``stray_field_drift`` per axis."""


def _rms_unit(density_unit: str) -> str:
    """The rms unit of a density unit: ``X^2/(rad/s)`` -> ``X``, else ``sqrt(unit x rad/s)``."""
    m = re.fullmatch(r"\((.*)\)\^2/\(rad/s\)|(.*)\^2/\(rad/s\)", density_unit.strip())
    if m:
        return m.group(1) or m.group(2)
    return f"sqrt({density_unit} x rad/s)"


def quiet_field_spectrum() -> NoiseSpectrum:
    """The zero electric-field spectrum ``NoiseModel()`` carries: five zero points over +-2 pi x 10^7 rad/s."""
    omega = np.linspace(-2.0 * math.pi * 1e7, 2.0 * math.pi * 1e7, 5)
    return NoiseSpectrum(omega_rad_s=omega, S=np.zeros(5), unit="(V/m)^2/(rad/s)")


def quiet_drift() -> Drift:
    """A drift of zero rms (tau_s = 1 s, no servo): the default of every ``Drift`` field of ``NoiseModel``."""
    return Drift(rms=0.0, tau_s=1.0, servo_bandwidth_hz=None)


@dataclass(frozen=True)
class NoiseModel:
    """The noise of a device as spectra, drifts and event rates, never as phenomenological error rates. Every default
    means "off": ``NoiseModel()`` is the quiet model, so a channel is on exactly when its input was set."""

    S_E: NoiseSpectrum = field(default_factory=lambda: quiet_field_spectrum())
    """Electric-field noise, which heats the modes; default: the zero spectrum."""
    correlation_length_m: float | None = 0.0
    """Correlation length of the field noise (0 uncorrelated, inf uniform); None is refused once S_E is non-zero."""
    S_B: NoiseSpectrum | None = None
    mains: Mains | None = None
    laser_phase: NoiseSpectrum | None = None
    laser_intensity: NoiseSpectrum | None = None
    rf_amplitude_noise: NoiseSpectrum | None = None
    rf_phase_noise: NoiseSpectrum | None = None
    rf_amplitude_drift: Drift = field(default_factory=lambda: quiet_drift())
    """Common-mode fractional drift dV/V of every rf-derived mode."""
    mode_drift_differential: Drift = field(default_factory=lambda: quiet_drift())
    rabi_drift: Drift = field(default_factory=lambda: quiet_drift())
    beam_phase_drift: Drift = field(default_factory=lambda: quiet_drift())
    field_drift: Drift = field(default_factory=lambda: quiet_drift())
    stray_field_drift: Drift = field(default_factory=lambda: quiet_drift())
    pointing_drift: Drift = field(default_factory=lambda: quiet_drift())
    """Beam pointing, which moves crosstalk and Rabi rate together."""
    rabi_amplitude: NoiseSpectrum | None = None
    """Two-sided S_a(omega) of the additive Rabi-amplitude noise in (rad/s)^2/(rad/s), for the filter functions only."""
    collisions: Collisions | None = None
    laser_frequency_drift: Drift | None = None
    """The gate laser's frequency against the transition (optical qubits); None = none."""
    beam_phase_noise: NoiseSpectrum | None = None
    """rad^2/(rad/s): each beam's sampled optical path phase, independent per beam (a beat note sees phi_2 - phi_1)."""
    grid_oversample: float = TAU_C_OVERSAMPLE
    """Points per half period of the highest sampled frequency on a trajectory grid (default: dt <= tau_c/10)."""
    extra: dict[str, float] = field(default_factory=dict)
    """Free-form documented device numbers (a measured T2 to compare with, a quoted heating rate)."""

    def heating_rates_quanta_per_s(self, device: Device) -> dict[int, float]:
        """Heating rate (quanta/s) per crystal mode from S_E with the configured correlation length (Brownnutt)."""
        from qutip_trap.trap.heating import heating_rates_per_mode, single_sided_from_spectrum

        if self.S_E.is_zero():
            return {}
        if self.correlation_length_m is None:
            raise ValueError(
                "NoiseModel.correlation_length_m is required once S_E is non-zero: 0 for uncorrelated field noise (every mode "
                "heats at the single-ion rate), inf for a uniform field (only the centre-of-mass modes of an equal-mass chain), "
                "or a length (Section 4.1.5: the simulator never defaults to the uniform limit)"
            )
        # the record, not its arrays: its value() folds onto |omega| and already adds white_level once
        s_e = single_sided_from_spectrum(self.S_E)
        rates = heating_rates_per_mode(device.crystal, s_e, float(self.correlation_length_m))
        return {m: float(r) for m, r in enumerate(rates) if r > 0.0}

    def motional_dephasing_tau_s(self, device: Device) -> dict[int, float]:
        """tau per rf-derived mode from the white rf-amplitude noise: 2/tau = omega_m^2 S_V,white."""
        if self.rf_amplitude_noise is None or self.rf_amplitude_noise.white_level <= 0.0:
            return {}
        out: dict[int, float] = {}
        for m, mode in enumerate(device.crystal.modes):
            if mode.family in RF_DERIVED_FAMILIES:
                d = mode.omega_rad_s**2 * self.rf_amplitude_noise.white_level
                out[m] = 2.0 / d
        return out

    def field_spectrum(self, device: Device) -> NoiseSpectrum | None:
        """S_B of the model, else the Field record's own spectrum, else None."""
        return self.S_B if self.S_B is not None else device.field.noise

    def qubit_white_dephasing_per_s(self, device: Device) -> dict[int, float]:
        """gamma_phi = 2 pi^2 (d nu/dB)^2 S_B,white per ion: the coherence decay rate 1/T2 of the white field noise."""
        spec = self.field_spectrum(device)
        if spec is None or spec.white_level <= 0.0:
            return {}
        out: dict[int, float] = {}
        s_white_g = spec.white_level * GAUSS_PER_TESLA**2  # G^2/(rad/s)
        for i in range(device.crystal.n_ions):
            sp = device.crystal.species[i]
            _f, d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], device.field.B_gauss)
            gamma = 2.0 * math.pi**2 * d1**2 * s_white_g
            if gamma > 0.0:
                out[i] = gamma
        return out

    def intensity_white_density(self) -> float:
        """The white level of ``laser_intensity`` (1/(rad/s)), 0 without one."""
        return 0.0 if self.laser_intensity is None else float(self.laser_intensity.white_level)

    def channels(self, device: Device, space: HilbertSpace) -> tuple[CollapseOp, ...]:
        """The state-independent collapse operators on ``space``: heating, motional and white qubit dephasing."""
        from qutip_trap.dynamics.channels import (
            heating_channels,
            motional_dephasing_channels,
            qubit_dephasing_channels,
        )

        out: list[CollapseOp] = []
        out.extend(heating_channels(space, self.heating_rates_quanta_per_s(device)))
        out.extend(motional_dephasing_channels(space, self.motional_dephasing_tau_s(device)))
        out.extend(qubit_dephasing_channels(space, self.qubit_white_dephasing_per_s(device)))
        return tuple(out)

    @property
    def drifts(self) -> dict[str, Drift]:
        out = {
            "rf_amplitude_drift": self.rf_amplitude_drift,
            "mode_drift_differential": self.mode_drift_differential,
            "rabi_drift": self.rabi_drift,
            "beam_phase_drift": self.beam_phase_drift,
            "field_drift": self.field_drift,
            "stray_field_drift": self.stray_field_drift,
            "pointing_drift": self.pointing_drift,
        }
        if self.laser_frequency_drift is not None:
            out["laser_frequency_drift"] = self.laser_frequency_drift
        return out

    def sampled_spectra(self, device: Device | None) -> dict[str, NoiseSpectrum]:
        """The spectra whose tabulated bands are synthesized per sample."""
        out: dict[str, NoiseSpectrum] = {}
        s_b = self.S_B if device is None else self.field_spectrum(device)
        if s_b is not None and not bool(np.all(np.asarray(s_b.S) == 0.0)):
            out["S_B"] = s_b
        for name in ("laser_phase", "laser_intensity", "rf_amplitude_noise", "beam_phase_noise"):
            spec = getattr(self, name)
            if spec is not None and not bool(np.all(np.asarray(spec.S) == 0.0)):
                out[name] = spec
        return out

    def is_quiet(self, device: Device | None = None) -> bool:
        """No quasi-static drift, no sampled band and no mains: every dynamical sample equals the nominal one."""
        if any(not d.quiet for d in self.drifts.values()):
            return False
        if self.mains is not None and any(a != 0.0 for a in self.mains.amplitudes_t.values()):
            return False
        return not self.sampled_spectra(device)

    def apparatus(self) -> tuple[str, ...]:
        """The sorted, de-duplicated apparatus tags of every non-zero rate of this model."""
        tags: set[str] = set()
        for spec in (
            self.S_E,
            self.S_B,
            self.laser_phase,
            self.laser_intensity,
            self.rf_amplitude_noise,
            self.rf_phase_noise,
            self.rabi_amplitude,
            self.beam_phase_noise,
        ):
            if spec is not None and not spec.is_zero():
                tags.update(spec.provenance)
        for drift in self.drifts.values():
            if not drift.quiet:
                tags.update(drift.provenance)
        return tuple(sorted(tags))

    def undeclared_rate_count(self) -> int:
        """How many non-zero rates carry no apparatus tag."""
        n = 0
        for spec in (
            self.S_E,
            self.S_B,
            self.laser_phase,
            self.laser_intensity,
            self.rf_amplitude_noise,
            self.rf_phase_noise,
            self.rabi_amplitude,
            self.beam_phase_noise,
        ):
            if spec is not None and not spec.is_zero() and not spec.provenance:
                n += 1
        return n + sum(1 for d in self.drifts.values() if not d.quiet and not d.provenance)

    def provenance_sentence(self) -> str:
        """How many apparatus the non-zero rates are stitched from and how many are undeclared; empty when quiet."""
        tags = self.apparatus()
        undeclared = self.undeclared_rate_count()
        if not tags and not undeclared:
            return ""
        parts = []
        if tags:
            parts.append(
                f"this noise budget is stitched from {len(tags)} apparatus ({', '.join(tags)}): published rates come "
                "from different machines and mixing them is what Section 6.1 warns about"
            )
        if undeclared:
            parts.append(
                f"{undeclared} non-zero rate(s) carry no apparatus tag (NoiseSpectrum/Drift.provenance)"
            )
        return "; ".join(parts)

    def summary(self, device: Device | None = None) -> dict[str, tuple[float, str]]:
        """The channels that follow from what was set, name -> (value, unit), empty for the quiet model; the per-mode
        and per-ion rates (heating, motional and qubit dephasing, collisions) need ``device``."""
        out: dict[str, tuple[float, str]] = {}
        if device is not None:
            for m, rate in self.heating_rates_quanta_per_s(device).items():
                out[f"heating_rate_per_s[{m}]"] = (float(rate), "quanta/s")
            for m, tau in self.motional_dephasing_tau_s(device).items():
                out[f"motional_dephasing_tau_s[{m}]"] = (float(tau), "s")
            for i, gamma in self.qubit_white_dephasing_per_s(device).items():
                out[f"qubit_dephasing_per_s[{i}]"] = (float(gamma), "1/s")
            if self.collisions is not None and self.collisions.pressure_pa > 0.0:
                from qutip_trap.noise.collisions import collision_rate_per_ion

                for i in range(device.crystal.n_ions):
                    rate = collision_rate_per_ion(self.collisions, float(device.crystal.masses_kg[i]))
                    out[f"collision_rate_per_ion[{i}]"] = (float(rate), "1/s")
        elif not self.S_E.is_zero():
            out["S_E_white_level"] = (float(self.S_E.white_level), self.S_E.unit)
        density = self.intensity_white_density()
        if density > 0.0:
            out["intensity_noise_density"] = (density, "1/(rad/s)")
        for name, spec in self.sampled_spectra(device).items():
            out[f"{name}_rms"] = (float(math.sqrt(max(spec.variance(), 0.0))), _rms_unit(spec.unit))
        for name, drift in self.drifts.items():
            if not drift.quiet:
                out[f"{name}_rms"] = (float(drift.rms), DRIFT_UNITS[name])
        if self.mains is not None:
            for k, amp in sorted(self.mains.amplitudes_t.items()):
                if amp != 0.0:
                    out[f"mains_amplitude_t[{k}]"] = (float(amp), "T")
        return out

    def from_experiments(self, results: Sequence[Any], *, device: Device) -> NoiseModel:
        """This model with the white field spectrum a set of ``HeatingRateFit`` results implies: each fit gives the
        single-sided S_E(omega_m) = 4 m hbar omega_m n_dot/e^2 (Brownnutt), the two-sided level is the mean of S_E/2.
        Other results and mixed masses are refused; a None ``correlation_length_m`` becomes 0."""
        from qutip_trap.trap.heating import s_e_from_heating_rate

        masses = {float(m) for m in device.crystal.masses_kg}
        if len(masses) != 1:
            raise ValueError(
                "NoiseModel.from_experiments: the heating inversion takes one ion mass; the crystal mixes species"
            )
        mass = masses.pop()
        levels: list[float] = []
        for res in results:
            if type(res).__name__ != "HeatingRateFit":
                raise ValueError(
                    f"NoiseModel.from_experiments inverts heating-rate fits only, not {type(res).__name__}"
                )
            mode = int(res.subject["mode"])
            ndot = float(res.value("ndot_per_s"))
            omega = float(device.crystal.modes[mode].omega_rad_s)
            levels.append(
                0.5 * s_e_from_heating_rate(ndot, mass, omega)
            )  # two-sided, the NoiseSpectrum convention
        if not levels:
            raise ValueError("NoiseModel.from_experiments: no result given")
        level = float(np.mean(levels))
        spectrum = NoiseSpectrum(
            np.array([0.0, 1.0]),
            np.zeros(2),
            "(V/m)^2/(rad/s)",
            white_level=level,
            provenance=("heating_rate experiment (NoiseModel.from_experiments)",),
        )
        extra = dict(self.extra)
        extra["S_E_from_experiments_modes"] = float(len(levels))
        if len(levels) > 1:
            extra["S_E_from_experiments_spread"] = float(np.max(levels) - np.min(levels))
        return replace(
            self,
            S_E=spectrum,
            correlation_length_m=0.0 if self.correlation_length_m is None else self.correlation_length_m,
            extra=extra,
        )

    def grid_omega_max_rad_s(self, device: Device | None) -> float:
        w = 0.0
        for spec in self.sampled_spectra(device).values():
            w = max(w, spec.omega_max_rad_s)
        if self.mains is not None:
            w = max(w, self.mains.omega_max_rad_s)
        return w

    def sample(
        self,
        rng: np.random.Generator,
        *,
        device: Device | None = None,
        t_s: float = 0.0,
        duration_s: float | None = None,
        sample_id: int = 0,
    ) -> NoiseSample:
        """One dynamical sample at shot-clock time ``t_s``: without ``device`` the device-independent draws only, with
        it the per-ion, per-mode and per-beam keys, and with ``duration_s`` also the sampled bands' trajectories."""
        return self.sample_sequence(rng, (t_s,), device=device, duration_s=duration_s, first_id=sample_id)[0]

    def sample_sequence(
        self,
        rng: np.random.Generator,
        times_s: Sequence[float],
        *,
        device: Device | None = None,
        duration_s: float | None = None,
        first_id: int = 0,
        t0_s: float | None = None,
    ) -> tuple[NoiseSample, ...]:
        """Dynamical samples at the shot-clock times ``times_s``: every Drift an Ornstein-Uhlenbeck chain over the times
        plus its ramp (only the ``servo_residual`` with ``servo_bandwidth_hz``), the sampled bands drawn per sample."""
        times = np.asarray(times_s, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times_s is a non-empty sequence of shot-clock times")
        t0 = float(times[0]) if t0_s is None else float(t0_s)
        n_beams = 0 if device is None else len(device.beams)
        n_modes = 0 if device is None else len(device.crystal.modes)
        draws: dict[str, np.ndarray] = {}
        gen = rng.spawn(len(self.drifts) + 4)
        k = 0
        for name, drift in self.drifts.items():
            size = {
                "rf_amplitude_drift": 1,
                "mode_drift_differential": max(n_modes, 1),
                "rabi_drift": 1,
                "beam_phase_drift": max(n_beams, 1),
                "field_drift": 1,
                "stray_field_drift": 3,
                "pointing_drift": max(2 * n_beams, 1),
                "laser_frequency_drift": 1,
            }[name]
            draws[name] = (
                correlated_normals(gen[k], times, drift.tau_s, size)
                if not drift.quiet
                else np.zeros((times.size, size))
            )
            k += 1
        gen_mains, gen_traj = gen[k], gen[k + 1]
        offsets: dict[str, np.ndarray] = {}
        for name, drift in self.drifts.items():
            raw = drift.rms * draws[name] + drift.rate_per_s * (times - t0)[:, None]
            offsets[name] = (
                servo_residual(raw, times, drift.servo_bandwidth_hz)
                if drift.servo_bandwidth_hz is not None and drift.servo_bandwidth_hz > 0.0 and not drift.quiet
                else raw
            )
        out: list[NoiseSample] = []
        for s_idx, t in enumerate(times):
            values: dict[str, float] = {}
            grids: dict[str, np.ndarray] = {}

            def amp(name: str, j: int = 0, _s: int = s_idx) -> float:  # noqa: B023
                return float(offsets[name][_s, j])

            if not self.rabi_drift.quiet:
                values[KEY_RABI_SCALE] = 1.0 + amp("rabi_drift")
            if not self.rf_amplitude_drift.quiet:
                values[KEY_RF_FRACTION] = amp("rf_amplitude_drift")
            if not self.field_drift.quiet:
                values[KEY_FIELD_OFFSET_T] = amp("field_drift")
            if self.laser_frequency_drift is not None and not self.laser_frequency_drift.quiet:
                values[KEY_LASER_OFFSET_HZ] = amp("laser_frequency_drift")
            if self.mains is not None and self.mains.amplitudes_t:
                values[KEY_MAINS_PHASE] = (
                    float(gen_mains.uniform(0.0, TWO_PI)) if self.mains.trigger == "free_running" else 0.0
                )
            if device is not None:
                self._derive_device_keys(device, values, amp)
                if duration_s is not None and duration_s > 0.0:
                    self._synthesize(device, values, grids, gen_traj, float(duration_s))
            out.append(NoiseSample(sample_id=first_id + s_idx, values=values, ou_grids=grids, t_s=float(t)))
        return tuple(out)

    def _derive_device_keys(
        self, device: Device, values: dict[str, float], amp_fn: Callable[[str, int], float]
    ) -> None:
        n_beams = len(device.beams)
        # qubit offsets from the field offset: the exact diagonalization at the shifted field
        db_t = values.get(KEY_FIELD_OFFSET_T, 0.0)
        if db_t != 0.0:
            for i in range(device.crystal.n_ions):
                sp = device.crystal.species[i]
                f0, _d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], device.field.B_gauss)
                f1, _d1b, _d2b = sp.transition_frequency_hz(
                    sp.qubit[0], sp.qubit[1], device.field.B_gauss + db_t * GAUSS_PER_TESLA
                )
                values[key_qubit_offset_hz(i)] = float(f1 - f0)
        frac = values.get(KEY_RF_FRACTION, 0.0)
        for m, mode in enumerate(device.crystal.modes):
            off = 0.0
            if frac != 0.0 and mode.family in RF_DERIVED_FAMILIES:
                off += frac * mode.omega_hz
            if not self.mode_drift_differential.quiet:
                off += amp_fn("mode_drift_differential", m)
            if off != 0.0:
                values[key_mode_offset_hz(m)] = off
        if not self.beam_phase_drift.quiet:
            for b in range(n_beams):
                values[key_beam_phase_rad(b)] = amp_fn("beam_phase_drift", b)
        if not self.pointing_drift.quiet:
            for b in range(n_beams):
                k_hat = np.asarray(device.beams[b].k_hat, dtype=float)
                seed = np.array([1.0, 0.0, 0.0]) if abs(k_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                u = seed - np.dot(seed, k_hat) * k_hat
                u /= np.linalg.norm(u)
                v = np.cross(k_hat, u)
                offset = amp_fn("pointing_drift", 2 * b) * u + amp_fn("pointing_drift", 2 * b + 1) * v
                for ax in range(3):
                    if offset[ax] != 0.0:
                        values[key_beam_offset_m(b, ax)] = float(offset[ax])
        if not self.stray_field_drift.quiet:
            omegas = device.trap.omega_hz
            if omegas is not None:
                for i in range(device.crystal.n_ions):
                    mass = float(device.crystal.masses_kg[i])
                    for ax in range(3):
                        e_field = amp_fn("stray_field_drift", ax)
                        w = TWO_PI * float(omegas[ax])
                        if e_field != 0.0:
                            values[key_position_offset_m(i, ax)] = float(E_C * e_field / (mass * w * w))

    def _synthesize(
        self,
        device: Device,
        values: dict[str, float],
        grids: dict[str, np.ndarray],
        rng: np.random.Generator,
        duration_s: float,
    ) -> None:
        spectra = self.sampled_spectra(device)
        w_max = self.grid_omega_max_rad_s(device)
        if w_max <= 0.0:
            return
        times = time_grid(duration_s, w_max, oversample=self.grid_oversample)
        field_traj: Trajectory | None = None
        if "S_B" in spectra:
            field_traj = synthesize(spectra["S_B"], times, rng)
        if self.mains is not None and self.mains.amplitudes_t:
            mt = mains_trajectory(self.mains, times, values.get(KEY_MAINS_PHASE, 0.0))
            field_traj = mt if field_traj is None else Trajectory(times, field_traj.values + mt.values)
        if field_traj is not None:
            db_g = field_traj.values * GAUSS_PER_TESLA
            b_static = device.field.B_gauss + values.get(KEY_FIELD_OFFSET_T, 0.0) * GAUSS_PER_TESLA
            for i in range(device.crystal.n_ions):
                sp = device.crystal.species[i]
                _f, d1, d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], b_static)
                dnu = d1 * db_g + 0.5 * d2 * db_g**2
                grids[key_qubit_trajectory_hz(i)] = Trajectory(times, dnu).as_grid()
        if "laser_phase" in spectra:
            grids[KEY_LASER_PHASE_TRAJECTORY] = synthesize(spectra["laser_phase"], times, rng).as_grid()
        if "laser_intensity" in spectra:
            grids[KEY_INTENSITY_TRAJECTORY] = synthesize(spectra["laser_intensity"], times, rng).as_grid()
        if "rf_amplitude_noise" in spectra:
            grids[KEY_RF_FRACTION_TRAJECTORY] = synthesize(
                spectra["rf_amplitude_noise"], times, rng
            ).as_grid()
        if "beam_phase_noise" in spectra:
            # independent per beam, so a co-propagating pair (one shared path) must be declared as one beam, not two
            for b in range(len(device.beams)):
                grids[key_beam_phase_trajectory_rad(b)] = synthesize(
                    spectra["beam_phase_noise"], times, rng
                ).as_grid()


def servo_residual(values: np.ndarray, times_s: np.ndarray, bandwidth_hz: float) -> np.ndarray:
    """The part of a drifting parameter a first-order servo of ``bandwidth_hz`` does not remove: re-locking at every
    sample from x_hat_0 = x_0, x_hat_k = x_hat_{k-1} + (1 - e^{-2 pi f_s dt_k})(x_k - x_hat_{k-1}), sample k carries
    x_k - x_hat_{k-1} (0 at k = 0)."""
    x = np.asarray(values, dtype=float)
    t = np.asarray(times_s, dtype=float)
    if bandwidth_hz <= 0.0:
        return x
    out = np.zeros_like(x)
    estimate = x[0].copy()
    for k in range(1, x.shape[0]):
        alpha = 1.0 - math.exp(-TWO_PI * bandwidth_hz * abs(float(t[k] - t[k - 1])))
        out[k] = x[k] - estimate
        estimate = estimate + alpha * (x[k] - estimate)
    return out


__all__ = [
    "DRIFT_UNITS",
    "quiet_drift",
    "quiet_field_spectrum",
    "GAUSS_PER_TESLA",
    "RF_DERIVED_FAMILIES",
    "NoiseModel",
    "servo_residual",
]
