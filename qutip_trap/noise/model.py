"""The ``NoiseModel`` record and its two products: collapse operators and dynamical samples (PLAN.md Sections 4.1.5, 6.1
to 6.7, 7.5; Section 13; Appendix E; milestone M7).

Section 6.1 routes every physical noise by its correlation time: white noise is a Lindblad operator (route b,
``channels``), noise slow compared with a shot is a quasi-static parameter drawn once per dynamical sample (route c,
the ``Drift`` records), anything in between is a sampled time series on a fixed grid (route d, the tabulated band of a
``NoiseSpectrum``), and background-gas collisions are discrete events (route e, ``noise/collisions.py``, drawn per shot
by the run). Nothing here is phenomenological: every rate is derived from a spectrum through the Section 13
normalizations, and every offset from a drift's rms through the atomic and crystal layers.

Units of the ``Drift`` fields: ``rf_amplitude_drift`` fractional dV/V (every rf-derived, i.e. transverse, mode moves by
omega_m dV/V, Section 6.2); ``mode_drift_differential`` Hz per mode; ``rabi_drift`` fractional; ``beam_phase_drift`` rad
per beam; ``field_drift`` tesla (converted to per-ion transition offsets through the exact hyperfine-Zeeman
diagonalization at the shifted field, Section 6.3); ``stray_field_drift`` V/m (an equal displacement e E/(m omega^2) of
every ion along each axis, which moves the beams' intensity at the ions); ``pointing_drift`` metres per beam in the plane
transverse to the beam; ``laser_frequency_drift`` Hz (an optical qubit's laser against the transition; a Raman beat note
is rf-referenced and does not see it).

Spectra: ``S_E`` (V/m)^2/(rad/s) two-sided -> heating through the mode-projected multi-ion formula with the configured
correlation length (Section 4.1.5, never the uniform limit by default); ``S_B`` T^2/(rad/s) -> per-ion transition-frequency
trajectories through the computed sensitivities d nu/dB and d^2 nu/dB^2, and its white level to the qubit dephasing
operator L = sqrt(gamma_phi/2) sigma_z with gamma_phi = 2 pi^2 (d nu/dB)^2 S_B,white (Section 13: H_deph = b sigma_z with
b = pi delta nu, S_b = S_delta/4, and L = sqrt(gamma/2) sigma_z decays the coherence at gamma = 2 S_b(0)); ``mains``
(tesla per harmonic with a per-shot trigger phase) added to the same trajectory; ``laser_phase`` rad^2/(rad/s) -> phi_L(t)
on single-photon optical drives; ``laser_intensity`` (dI/I)^2/(rad/s) -> a fractional intensity trajectory on every laser
drive and, from its white level, the intensity-noise channel sqrt(D) H_drive(t) (Section 6.4); ``rf_amplitude_noise``
(dV/V)^2/(rad/s) -> a common-mode fractional trajectory of the transverse mode frequencies and, from its white level, the
motional dephasing operator a^dag a sqrt(2/tau) with 2/tau = omega_m^2 S_V,white (Section 13 row "Motional dephasing
operator": the coherence of |n> + |n'> decays at (n - n')^2/tau); ``rf_phase_noise`` is recorded and not yet applied
(the Omega_rf +- omega_t sideband-heating term of Section 4.1.5 needs the uncompensated stray field's rf gradient, which
the explicit-frequency trap record does not carry); ``rabi_amplitude`` (rad/s)^2/(rad/s) is consumed by the filter
functions of ``noise/decoupling.py`` only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.noise.processes import Trajectory, correlated_normals, mains_trajectory, synthesize, time_grid
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
"""The mode families whose frequencies follow the rf amplitude (omega_sec proportional to V_rf, Section 6.2)."""

GAUSS_PER_TESLA = 1.0e4


@dataclass(frozen=True)
class NoiseModel:
    S_E: NoiseSpectrum
    """Electric-field noise (heating, Section 6.2); stored two-sided like every NoiseSpectrum."""
    correlation_length_m: float | None
    """Spatial correlation length of the field noise (0 uncorrelated, inf uniform); REQUIRED once S_E is non-zero (Section 4.1.5)."""
    S_B: NoiseSpectrum | None
    mains: Mains | None
    laser_phase: NoiseSpectrum | None
    laser_intensity: NoiseSpectrum | None
    rf_amplitude_noise: NoiseSpectrum | None
    rf_phase_noise: NoiseSpectrum | None
    rf_amplitude_drift: Drift
    """Common-mode fractional drift of every rf-derived mode of a family."""
    mode_drift_differential: Drift
    rabi_drift: Drift
    beam_phase_drift: Drift
    field_drift: Drift
    stray_field_drift: Drift
    pointing_drift: Drift
    """Beam pointing, which moves crosstalk and Rabi rate together (Section 6.6)."""
    rabi_amplitude: NoiseSpectrum | None
    """Two-sided S_a(omega) of the ADDITIVE amplitude noise in rad/s (Section 6.9); Omega is taken from the pulse
    at use and never folded as Omega^2 into a device PSD."""
    collisions: Collisions | None
    laser_frequency_drift: Drift | None = None
    """Hz: the gate laser's frequency against the transition (optical qubits); None = none (M7 extension, defaulted)."""
    grid_oversample: float = 4.0
    """Points per half period of the highest sampled frequency on a trajectory grid (Section 5.5)."""
    extra: dict[str, float] = field(default_factory=dict)
    """Free-form documented device numbers (a measured T2 to compare with, a quoted heating rate with its provenance)."""

    # ---- route (b): Lindblad rates from the white parts and S_E -------------------------------------------------------------

    def heating_rates_quanta_per_s(self, device: Device) -> dict[int, float]:
        """Gamma_h per crystal mode from S_E with the configured correlation length (Section 4.1.5, Kielpinski/Brownnutt)."""
        from qutip_trap.trap.heating import heating_rates_per_mode, single_sided_from_spectrum

        if self.S_E.is_zero():
            return {}
        if self.correlation_length_m is None:
            raise ValueError(
                "NoiseModel.correlation_length_m is required once S_E is non-zero: 0 for uncorrelated field noise (every mode "
                "heats at the single-ion rate), inf for a uniform field (only the centre-of-mass modes of an equal-mass chain), "
                "or a length (Section 4.1.5: the simulator never defaults to the uniform limit)"
            )
        tab = single_sided_from_spectrum(np.abs(np.asarray(self.S_E.omega_rad_s)), np.asarray(self.S_E.S))
        white = 2.0 * self.S_E.white_level

        def s_e(omega: float) -> float:
            return float(tab(omega) if not self.S_E.is_zero() else 0.0) + white

        rates = heating_rates_per_mode(device.crystal, s_e, float(self.correlation_length_m))
        return {m: float(r) for m, r in enumerate(rates) if r > 0.0}

    def motional_dephasing_tau_s(self, device: Device) -> dict[int, float]:
        """tau per rf-derived mode from the white rf-amplitude noise: 2/tau = omega_m^2 S_V,white (Section 6.2, 13)."""
        if self.rf_amplitude_noise is None or self.rf_amplitude_noise.white_level <= 0.0:
            return {}
        out: dict[int, float] = {}
        for m, mode in enumerate(device.crystal.modes):
            if mode.family in RF_DERIVED_FAMILIES:
                d = mode.omega_rad_s**2 * self.rf_amplitude_noise.white_level
                out[m] = 2.0 / d
        return out

    def field_spectrum(self, device: Device) -> NoiseSpectrum | None:
        """S_B of the model, else the Field record's own spectrum (the M0 slot), else None."""
        return self.S_B if self.S_B is not None else device.field.noise

    def qubit_white_dephasing_per_s(self, device: Device) -> dict[int, float]:
        """gamma_phi = 2 pi^2 (d nu/dB)^2 S_B,white per ion, the white-field dephasing rate = 1/T2 of the white component."""
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
        """The flat two-sided density of dI/I above the tabulated band (1/(rad/s)), 0 without one."""
        return 0.0 if self.laser_intensity is None else float(self.laser_intensity.white_level)

    def channels(self, device: Device, space: HilbertSpace) -> tuple[CollapseOp, ...]:
        """The state-independent collapse operators of Section 5.7 on ``space`` (Section 13 normalizations): heating per
        carried mode, motional dephasing per rf-derived mode, white qubit dephasing per ion. The pulse-dependent operators
        (scattering with recoil and leakage, the intensity-noise channel) are assembled per segment by the engine."""
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

    # ---- routes (c) and (d): dynamical samples ---------------------------------------------------------------------------

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
        for name in ("laser_phase", "laser_intensity", "rf_amplitude_noise"):
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
        """One dynamical sample at shot-clock time ``t_s`` (Appendix E ``sample(rng)``; the keyword arguments are M7's).

        Without ``device`` only the device-independent raw draws are made (field offset, rf fraction, Rabi scale, mains
        phase, laser offset); with it the per-ion, per-mode and per-beam keys the Hamiltonian builder reads are derived,
        and with ``duration_s`` the trajectories of the sampled bands are synthesized on the grid of Section 5.5.
        """
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
        """Dynamical samples at the shot-clock times ``times_s`` (Section 7.5): every Drift is an Ornstein-Uhlenbeck chain
        over the times with its correlation time (samples far apart in time are independent, close ones correlated), its ramp
        rate x (t - t0) added; the trajectories of the sampled bands are drawn independently per sample."""
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
        out: list[NoiseSample] = []
        for s_idx, t in enumerate(times):
            values: dict[str, float] = {}
            grids: dict[str, np.ndarray] = {}
            dt = float(t - t0)

            def amp(name: str, j: int = 0, _s: int = s_idx, _dt: float = dt) -> float:  # noqa: B023
                d = self.drifts[name]
                return float(d.rms * draws[name][_s, j] + d.rate_per_s * _dt)

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

    # ---- helpers -----------------------------------------------------------------------------------------------------------

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
        # mode offsets: common-mode rf fraction on the transverse families plus the differential drift
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
        # the field: S_B trajectory plus the mains at this sample's trigger phase -> per-ion transition offsets
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


__all__ = ["GAUSS_PER_TESLA", "RF_DERIVED_FAMILIES", "NoiseModel"]
