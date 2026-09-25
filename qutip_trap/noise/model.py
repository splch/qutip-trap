"""The ``NoiseModel``: spectra, drifts and collisions, turned into collapse operators and dynamical samples (PLAN.md
Section 6).

Each noise is routed by its correlation time: a white density is a Lindblad operator (``channels``), a slow
parameter a quasi-static offset drawn once per dynamical sample (the ``Drift`` records), anything in between a
trajectory on a fixed grid (the tabulated band of a ``NoiseSpectrum``), and collisions discrete events drawn per shot.

Units: ``S_E`` (V/m)^2/(rad/s), heating through the mode-projected formula with ``correlation_length_m``; ``S_B``
T^2/(rad/s), per-ion transition trajectories through d nu/dB and d^2 nu/dB^2, its white level the dephasing operator
L = sqrt(gamma_phi/2) sigma_z with gamma_phi = 2 pi^2 (d nu/dB)^2 S_B,white; ``mains`` T per harmonic; ``laser_phase``
rad^2/(rad/s) on single-photon optical drives; ``laser_intensity`` (dI/I)^2/(rad/s), its white level the intensity-noise
channel; ``rf_amplitude_noise`` (dV/V)^2/(rad/s) on the transverse modes, its white level the motional dephasing operator
with 2/tau = omega_m^2 S_V,white; ``rf_phase_noise`` is recorded and not applied; ``rabi_amplitude`` (rad/s)^2/(rad/s) is
read by ``noise/decoupling.py``; ``beam_phase_noise`` rad^2/(rad/s) per beam. Drifts: ``rf_amplitude_drift`` and
``rabi_drift`` fractional, ``mode_drift_differential`` Hz per mode, ``beam_phase_drift`` rad per beam, ``field_drift`` T
(exact transition offsets at the shifted field), ``stray_field_drift`` V/m (a displacement e E/(m omega^2) of every ion),
``pointing_drift`` m per beam transverse to it, ``laser_frequency_drift`` Hz (not seen by an rf-referenced Raman beat note).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

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
"""The mode families whose frequencies follow the rf amplitude."""

GAUSS_PER_TESLA = 1.0e4


class _DriftField(NamedTuple):
    """One ``Drift`` field of a ``NoiseModel``: its name, the record, the unit of its rms and its draws per sample."""

    name: str
    drift: Drift
    unit: str
    draws: int


def _rms_unit(density_unit: str) -> str:
    """The unit of an rms from a two-sided density's unit: ``X^2/(rad/s)`` -> ``X``, else the density unit with a note."""
    m = re.fullmatch(r"\((.*)\)\^2/\(rad/s\)|(.*)\^2/\(rad/s\)", density_unit.strip())
    if m:
        return m.group(1) or m.group(2)
    return f"sqrt({density_unit} x rad/s)"


def quiet_drift() -> Drift:
    """A drift of zero rms (tau_s = 1 s, no servo): the default of every ``Drift`` field of ``NoiseModel``."""
    return Drift(rms=0.0, tau_s=1.0, servo_bandwidth_hz=None)


def _quiet_field_spectrum() -> NoiseSpectrum:
    """The zero electric-field spectrum: a five-point zero band over +-2 pi x 10^7 rad/s."""
    omega = np.linspace(-2.0 * math.pi * 1e7, 2.0 * math.pi * 1e7, 5)
    return NoiseSpectrum(omega_rad_s=omega, S=np.zeros(5), unit="(V/m)^2/(rad/s)")


@dataclass(frozen=True)
class NoiseModel:
    """The noise of a device as spectra, drifts and event rates, never as phenomenological error rates (units in the module
    docstring). Every default means off: ``NoiseModel()`` is the quiet model."""

    S_E: NoiseSpectrum = field(default_factory=_quiet_field_spectrum)
    """Electric-field noise, the source of heating."""
    correlation_length_m: float | None = 0.0
    """Spatial correlation length of the field noise (0 uncorrelated, inf uniform); required once S_E is non-zero."""
    S_B: NoiseSpectrum | None = None
    mains: Mains | None = None
    laser_phase: NoiseSpectrum | None = None
    laser_intensity: NoiseSpectrum | None = None
    rf_amplitude_noise: NoiseSpectrum | None = None
    rf_phase_noise: NoiseSpectrum | None = None
    rf_amplitude_drift: Drift = field(default_factory=lambda: quiet_drift())
    mode_drift_differential: Drift = field(default_factory=lambda: quiet_drift())
    rabi_drift: Drift = field(default_factory=lambda: quiet_drift())
    beam_phase_drift: Drift = field(default_factory=lambda: quiet_drift())
    field_drift: Drift = field(default_factory=lambda: quiet_drift())
    stray_field_drift: Drift = field(default_factory=lambda: quiet_drift())
    pointing_drift: Drift = field(default_factory=lambda: quiet_drift())
    rabi_amplitude: NoiseSpectrum | None = None
    """The additive amplitude noise S_a in rad/s; Omega is taken from the pulse at use."""
    collisions: Collisions | None = None
    laser_frequency_drift: Drift | None = None
    beam_phase_noise: NoiseSpectrum | None = None
    """The sampled band of each beam's optical path phase, drawn independently per beam."""
    grid_oversample: float = TAU_C_OVERSAMPLE
    """Points per half period of the highest sampled frequency on a trajectory grid."""

    # ---- Lindblad rates from the white parts and S_E -----------------------------------------------------------------

    def heating_rates_quanta_per_s(self, device: Device) -> dict[int, float]:
        """Gamma_h per crystal mode from S_E with the configured correlation length (Kielpinski, Brownnutt)."""
        from qutip_trap.trap.heating import heating_rates_per_mode, single_sided_from_spectrum

        if self.S_E.is_zero():
            return {}
        if self.correlation_length_m is None:
            raise ValueError(
                "NoiseModel.correlation_length_m is required once S_E is non-zero: 0 for uncorrelated field noise (every mode "
                "heats at the single-ion rate), inf for a uniform field (only the centre-of-mass modes of an equal-mass chain), "
                "or a length (the simulator never defaults to the uniform limit)"
            )
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

    def qubit_white_dephasing_per_s(self, device: Device) -> dict[int, float]:
        """gamma_phi = 2 pi^2 (d nu/dB)^2 S_B,white per ion, 1/T2 of the white component."""
        spec = self.S_B
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
        """The flat two-sided density of dI/I above the tabulated band (1/(rad/s)), 0 without one."""
        return 0.0 if self.laser_intensity is None else float(self.laser_intensity.white_level)

    def channels(self, device: Device, space: HilbertSpace) -> tuple[CollapseOp, ...]:
        """The state-independent collapse operators on ``space``: heating per carried mode, motional dephasing per rf-derived
        mode, white qubit dephasing per ion. The pulse-dependent ones are built per segment by the engine."""
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

    # ---- dynamical samples ---------------------------------------------------------------------------------------------

    def _drift_fields(self, n_modes: int = 0, n_beams: int = 0) -> list[_DriftField]:
        """Every Drift field in declaration order, with the unit of its rms and its draws per sample."""
        out = [
            _DriftField("rf_amplitude_drift", self.rf_amplitude_drift, "1", 1),
            _DriftField("mode_drift_differential", self.mode_drift_differential, "Hz", max(n_modes, 1)),
            _DriftField("rabi_drift", self.rabi_drift, "1", 1),
            _DriftField("beam_phase_drift", self.beam_phase_drift, "rad", max(n_beams, 1)),
            _DriftField("field_drift", self.field_drift, "T", 1),
            _DriftField("stray_field_drift", self.stray_field_drift, "V/m", 3),
            _DriftField("pointing_drift", self.pointing_drift, "m", max(2 * n_beams, 1)),
        ]
        if self.laser_frequency_drift is not None:
            out.append(_DriftField("laser_frequency_drift", self.laser_frequency_drift, "Hz", 1))
        return out

    @property
    def drifts(self) -> dict[str, Drift]:
        return {f.name: f.drift for f in self._drift_fields()}

    def sampled_spectra(self) -> dict[str, NoiseSpectrum]:
        """The spectra whose tabulated bands are synthesized per sample, by field name."""
        named = {
            "S_B": _band(self.S_B),
            "laser_phase": _band(self.laser_phase),
            "laser_intensity": _band(self.laser_intensity),
            "rf_amplitude_noise": _band(self.rf_amplitude_noise),
            "beam_phase_noise": _band(self.beam_phase_noise),
        }
        return {k: s for k, s in named.items() if s is not None}

    def is_quiet(self, device: Device | None = None) -> bool:
        """No quasi-static drift, no sampled band and no mains: every dynamical sample is the nominal one (``device`` is
        not needed)."""
        if any(not d.quiet for d in self.drifts.values()):
            return False
        if self.mains is not None and any(a != 0.0 for a in self.mains.amplitudes_t.values()):
            return False
        return not self.sampled_spectra()

    def _rates(self) -> list[NoiseSpectrum | Drift]:
        """Every non-quiet spectrum and drift: the rates a noise budget is assembled from."""
        spectra = (
            self.S_E,
            self.S_B,
            self.laser_phase,
            self.laser_intensity,
            self.rf_amplitude_noise,
            self.rf_phase_noise,
            self.rabi_amplitude,
            self.beam_phase_noise,
        )
        rates: list[NoiseSpectrum | Drift] = [s for s in spectra if s is not None and not s.is_zero()]
        return rates + [d for d in self.drifts.values() if not d.quiet]

    def apparatus(self) -> tuple[str, ...]:
        """Every apparatus tag declared by a non-quiet rate, sorted and de-duplicated (Section 6.1)."""
        return tuple(sorted({tag for rate in self._rates() for tag in rate.provenance}))

    def undeclared_rate_count(self) -> int:
        """How many non-quiet rates carry no apparatus tag."""
        return sum(1 for rate in self._rates() if not rate.provenance)

    def provenance_sentence(self) -> str:
        """Section 6.1's report that a budget is stitched from several apparatus; empty when the model is quiet."""
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

    def summary(self, device: Device) -> dict[str, tuple[float, str]]:
        """The channels that follow from what was set, name -> (value, unit): per mode the heating rate (quanta/s) and the
        motional dephasing time (s), per ion the white qubit dephasing rate and the collision rate (1/s), the white
        intensity-noise density (1/(rad/s)), the rms of every sampled band and non-quiet drift, the mains amplitude per
        harmonic (T). Empty for the quiet model."""
        out: dict[str, tuple[float, str]] = {}
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
        density = self.intensity_white_density()
        if density > 0.0:
            out["intensity_noise_density"] = (density, "1/(rad/s)")
        for name, spec in self.sampled_spectra().items():
            out[f"{name}_rms"] = (float(math.sqrt(max(spec.variance(), 0.0))), _rms_unit(spec.unit))
        for f in self._drift_fields():
            if not f.drift.quiet:
                out[f"{f.name}_rms"] = (float(f.drift.rms), f.unit)
        if self.mains is not None:
            for k, amp in sorted(self.mains.amplitudes_t.items()):
                if amp != 0.0:
                    out[f"mains_amplitude_t[{k}]"] = (float(amp), "T")
        return out

    def sample(
        self,
        rng: np.random.Generator,
        *,
        device: Device,
        t_s: float = 0.0,
        duration_s: float | None = None,
        sample_id: int = 0,
    ) -> NoiseSample:
        """One dynamical sample at shot-clock time ``t_s``; with ``duration_s`` the sampled bands are synthesized too."""
        return self.sample_sequence(rng, (t_s,), device=device, duration_s=duration_s, first_id=sample_id)[0]

    def sample_sequence(
        self,
        rng: np.random.Generator,
        times_s: Sequence[float],
        *,
        device: Device,
        duration_s: float | None = None,
        first_id: int = 0,
        t0_s: float | None = None,
    ) -> tuple[NoiseSample, ...]:
        """Dynamical samples at the shot-clock times ``times_s`` (Section 7.5): every Drift is an Ornstein-Uhlenbeck chain
        over the times plus its ramp rate x (t - t0), high-passed by its servo when it declares one (the offset is then the
        parameter minus the loop's last estimate, zero at the first sample); the sampled bands are drawn independently per
        sample."""
        times = np.asarray(times_s, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times_s is a non-empty sequence of shot-clock times")
        t0 = float(times[0]) if t0_s is None else float(t0_s)
        drift_fields = self._drift_fields(len(device.crystal.modes), len(device.beams))
        gen = rng.spawn(len(drift_fields) + 4)
        # the offsets: rms x draw + ramp, high-passed by the servo when the Drift declares one; a quiet drift is never read
        offsets: dict[str, np.ndarray] = {}
        for k, f in enumerate(drift_fields):
            if f.drift.quiet:
                continue
            draw = correlated_normals(gen[k], times, f.drift.tau_s, f.draws)
            raw = f.drift.rms * draw + f.drift.rate_per_s * (times - t0)[:, None]
            bandwidth = f.drift.servo_bandwidth_hz
            offsets[f.name] = (
                servo_residual(raw, times, bandwidth) if bandwidth is not None and bandwidth > 0.0 else raw
            )
        gen_mains, gen_traj = gen[len(drift_fields)], gen[len(drift_fields) + 1]
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
        # mode offsets: the common-mode rf fraction on the transverse families plus the differential drift
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
        w_max = max((s.omega_max_rad_s for s in self.sampled_spectra().values()), default=0.0)
        if self.mains is not None:
            w_max = max(w_max, self.mains.omega_max_rad_s)
        if w_max <= 0.0:
            return
        times = time_grid(duration_s, w_max, oversample=self.grid_oversample)
        # the field: the S_B trajectory plus the mains at this sample's trigger phase -> per-ion transition offsets
        field_traj: Trajectory | None = None
        if (s_b := _band(self.S_B)) is not None:
            field_traj = synthesize(s_b, times, rng)
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
        for key, spec in (
            (KEY_LASER_PHASE_TRAJECTORY, self.laser_phase),
            (KEY_INTENSITY_TRAJECTORY, self.laser_intensity),
            (KEY_RF_FRACTION_TRAJECTORY, self.rf_amplitude_noise),
        ):
            if (band := _band(spec)) is not None:
                grids[key] = synthesize(band, times, rng).as_grid()
        if (beam_band := _band(self.beam_phase_noise)) is not None:
            # independent per beam: a co-propagating pair (one shared path) must be declared as one beam, not two
            for b in range(len(device.beams)):
                grids[key_beam_phase_trajectory_rad(b)] = synthesize(beam_band, times, rng).as_grid()


def _band(spectrum: NoiseSpectrum | None) -> NoiseSpectrum | None:
    """The spectrum when it has a non-zero tabulated band to sample, else None."""
    return spectrum if spectrum is not None and bool(np.any(np.asarray(spectrum.S) != 0.0)) else None


def servo_residual(values: np.ndarray, times_s: np.ndarray, bandwidth_hz: float) -> np.ndarray:
    """The part of a drifting parameter a first-order servo of ``bandwidth_hz`` does not remove (Section 7.5).

    The loop's estimate is re-locked at every sample, x_hat_k = x_hat_{k-1} + (1 - e^{-2 pi f_s dt_k})(x_k - x_hat_{k-1})
    from x_hat_0 = x_0, and a sample carries the tracking error x_k - x_hat_{k-1}; for an OU drift of correlation time tau
    the residual variance is sigma^2/(1 + 2 pi f_s tau) in the continuous limit.
    """
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
