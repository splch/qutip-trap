"""NoiseModel.channels and NoiseModel.sample (PLAN.md Sections 4.1.5, 6.1 to 6.4, 7.5; Section 13 rows 'Motional dephasing
operator', 'Qubit dephasing operator'; Section 9.7 'Dephasing correlation' Ramsey test; Section 9.17 'Shot clock'; M7)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.control.schedule import Schedule
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.noise.model import GAUSS_PER_TESLA, NoiseModel
from qutip_trap.noise.sampling import (
    KEY_FIELD_OFFSET_T,
    KEY_MAINS_PHASE,
    KEY_RABI_SCALE,
    key_beam_offset_m,
    key_beam_phase_rad,
    key_mode_offset_hz,
    key_qubit_offset_hz,
    key_qubit_trajectory_hz,
    quiet_sample,
)
from qutip_trap.noise.spectra import Drift, Mains, ou_spectrum, white_spectrum
from qutip_trap.trap.heating import heating_rate_quanta_per_s, s_e_from_heating_rate
from qutip_trap.units import ATOMIC_MASS_KG
from tests.fixtures import make_noise
from tests.m2_fixtures import single_ion_raman_device
from tests.m4_fixtures import two_ion_device


def _with(device, **noise_fields):  # type: ignore[no-untyped-def]
    return dataclasses.replace(device, noise=dataclasses.replace(device.noise, **noise_fields))


def test_heating_rates_follow_the_correlation_length_and_refuse_its_absence() -> None:
    """Uncorrelated noise heats every mode at the single-ion rate; a uniform field only the centre-of-mass modes of an equal-mass
    chain (Section 4.1.5); the correlation length is required once S_E is non-zero."""
    dev = two_ion_device()
    mass = dev.crystal.species[0].mass_u * ATOMIC_MASS_KG
    com = dev.crystal.mode_index("transverse_1", 1)
    rock = dev.crystal.mode_index("transverse_1", 0)
    w_com = dev.crystal.modes[com].omega_rad_s
    s_e = s_e_from_heating_rate(100.0, mass, w_com)
    quiet_dev = _with(dev, S_E=white_spectrum(s_e / 2.0, "(V/m)^2/(rad/s)"), correlation_length_m=0.0)
    rates = quiet_dev.noise.heating_rates_quanta_per_s(quiet_dev)
    assert rates[com] == pytest.approx(100.0, rel=1e-9)
    assert rates[rock] == pytest.approx(
        heating_rate_quanta_per_s(s_e, mass, dev.crystal.modes[rock].omega_rad_s), rel=1e-9
    )
    uniform = _with(dev, S_E=white_spectrum(s_e / 2.0, "(V/m)^2/(rad/s)"), correlation_length_m=math.inf)
    ur = uniform.noise.heating_rates_quanta_per_s(uniform)
    assert ur[com] == pytest.approx(200.0, rel=1e-9) and rock not in ur
    bad = _with(dev, S_E=white_spectrum(s_e / 2.0, "(V/m)^2/(rad/s)"), correlation_length_m=None)
    with pytest.raises(ValueError, match="correlation_length_m"):
        bad.noise.heating_rates_quanta_per_s(bad)
    assert make_noise().is_quiet() and make_noise().heating_rates_quanta_per_s(dev) == {}
    space = HilbertSpace(
        (2, 2), (ModeTruncation(com, 6, (0, 2), 0.1),), None, tuple(m for m in range(6) if m != com)
    )
    ch = quiet_dev.noise.channels(quiet_dev, space)
    assert [c.channel for c in ch] == ["heating_down", "heating_up"] and ch[0].mode == com


def test_white_field_noise_is_the_qubit_dephasing_operator_with_gamma_equal_to_one_over_t2() -> None:
    """S_B,white -> gamma_phi = 2 pi^2 (d nu/dB)^2 S_B (the Section 13 chain S_b = S_delta/4, L = sqrt(gamma/2) sigma_z), and a Ramsey
    coherence under the channel decays as e^{-gamma t} = e^{-t/T2} (Section 9.7 Ramsey test)."""
    dev = single_ion_raman_device()
    sp = dev.crystal.species[0]
    _f, d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], dev.field.B_gauss)
    s_white_t = 1e-20  # T^2/(rad/s)
    noisy = _with(dev, S_B=white_spectrum(s_white_t, "T^2/(rad/s)"))
    gamma = noisy.noise.qubit_white_dephasing_per_s(noisy)[0]
    assert gamma == pytest.approx(2.0 * math.pi**2 * (d1 * GAUSS_PER_TESLA) ** 2 * s_white_t, rel=1e-12)
    space = HilbertSpace((2,), (), None, (0, 1, 2))
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    state = space.initial_state(plus)
    t = 2.0 / gamma
    eng = JointExactEngine(device_channels=True)
    tr = eng.run_pulses(
        noisy,
        Schedule((), ((0.0, t),), (), {0: 0.0}),
        state,
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    rho = tr.final.internal.full()
    assert abs(rho[0, 1]) == pytest.approx(0.5 * math.exp(-gamma * t), rel=1e-4)
    assert eng.last_report is not None and eng.last_report.channel_names == ("qubit_dephasing",)


def test_white_rf_amplitude_noise_is_motional_dephasing_on_the_transverse_modes() -> None:
    """2/tau = omega_m^2 S_V,white per rf-derived mode; the coherence of |0> + |1> of that mode decays at 1/tau (Section 13 row)."""
    dev = single_ion_raman_device()
    noisy = _with(
        dev, rf_amplitude_noise=white_spectrum(5e-11, "1/(rad/s)")
    )  # tau = 2/(omega^2 S) ~ 0.1 ms on the 3 MHz mode
    taus = noisy.noise.motional_dephasing_tau_s(noisy)
    assert set(taus) == {1, 2}, (
        "the axial (dc) mode 0 does not follow the rf amplitude; the two radial modes do"
    )
    w = dev.crystal.modes[1].omega_rad_s
    assert taus[1] == pytest.approx(2.0 / (w * w * 5e-11), rel=1e-12)
    space = HilbertSpace((2,), (ModeTruncation(1, 6, (0, 2), 0.1),), None, (0, 2))
    psi = (space.fock(1, 0) + space.fock(1, 1)).unit()
    state = space.initial_state([0], states={1: psi})
    t = taus[1]
    eng = JointExactEngine(device_channels=True)
    tr = eng.run_pulses(
        noisy,
        Schedule((), ((0.0, t),), (), {0: 0.0}),
        state,
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    rho_m = tr.final.motional.reduced[1].full()
    assert abs(rho_m[0, 1]) == pytest.approx(0.5 * math.exp(-1.0), rel=2e-3)


def test_sample_sequence_draws_drifts_at_the_shot_clock_with_their_correlation_and_ramp() -> None:
    """Section 7.5 / 9.17: samples at t and t' correlate as exp(-|t - t'|/tau); a Drift ramp advances linearly; the field offset
    converts to the exact transition offset (the diagonalization at the shifted field) and moves both ions alike."""
    dev = two_ion_device()
    noisy = _with(
        dev,
        field_drift=Drift(1e-7, 1.0, None, rate_per_s=1e-8),
        rabi_drift=Drift(1e-2, 1000.0, None),
        rf_amplitude_drift=Drift(1e-4, 1.0, None),
        mode_drift_differential=Drift(5.0, 1.0, None),
        beam_phase_drift=Drift(0.3, 1.0, None),
        pointing_drift=Drift(1e-7, 1.0, None),
    )
    rng = np.random.default_rng(0)
    times = [0.0, 1e-3, 2e-3, 50.0]
    samples = noisy.noise.sample_sequence(rng, times, device=noisy)
    assert [s.sample_id for s in samples] == [0, 1, 2, 3] and [s.t_s for s in samples] == times
    scales = [s.get(KEY_RABI_SCALE, 1.0) for s in samples]
    assert max(scales[:3]) - min(scales[:3]) < 1e-4, "a 1000 s Rabi drift is one value across 2 ms"
    assert abs(scales[3] - scales[0]) < 5e-2 * math.sqrt(1.0 - math.exp(-2 * 50.0 / 1000.0)) * 5.0
    sp = dev.crystal.species[0]
    for s in samples:
        db = s.values[KEY_FIELD_OFFSET_T]
        f0, _a, _b = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], dev.field.B_gauss)
        f1, _a, _b = sp.transition_frequency_hz(
            sp.qubit[0], sp.qubit[1], dev.field.B_gauss + db * GAUSS_PER_TESLA
        )
        assert s.values[key_qubit_offset_hz(0)] == pytest.approx(f1 - f0, abs=1e-9)
        assert s.values[key_qubit_offset_hz(0)] == s.values[key_qubit_offset_hz(1)]
        frac = s.values["rf_amplitude_fraction"]
        com = dev.crystal.mode_index("transverse_1", 1)
        assert abs(s.values[key_mode_offset_hz(com)] - frac * dev.crystal.modes[com].omega_hz) < 4.0 * 5.0
        assert key_beam_phase_rad(0) in s.values and any(
            key_beam_offset_m(0, ax) in s.values for ax in range(3)
        )
    # the ramp: the mean field offset at t = 50 s exceeds the rms by the ramp 1e-8 x 50 = 5e-7 T
    late = [
        noisy.noise.sample_sequence(np.random.default_rng(k), [0.0, 50.0], device=noisy)[1].values[
            KEY_FIELD_OFFSET_T
        ]
        for k in range(200)
    ]
    assert np.mean(late) == pytest.approx(5e-7, abs=3e-8)
    # correlation of the 1 s field drift between 0 and 1 ms is ~1, between 0 and 50 s ~0
    pairs = [
        noisy.noise.sample_sequence(np.random.default_rng(k), [0.0, 1e-3, 50.0], device=noisy)
        for k in range(300)
    ]
    a = np.array([p[0].values[KEY_FIELD_OFFSET_T] for p in pairs])
    b = np.array([p[1].values[KEY_FIELD_OFFSET_T] for p in pairs])
    c = np.array([p[2].values[KEY_FIELD_OFFSET_T] - 5e-7 for p in pairs])
    assert np.corrcoef(a, b)[0, 1] > 0.99 and abs(np.corrcoef(a, c)[0, 1]) < 0.15
    assert not noisy.noise.is_quiet(noisy)


def test_sampled_bands_become_per_ion_trajectories_through_the_sensitivities_plus_the_mains() -> None:
    """S_B's tabulated band and the mains at the shot's trigger phase synthesize into delta nu_i(t) = d1 dB + d2 dB^2/2 per ion on a
    fixed grid; the mains phase is uniform per sample when free-running and zero when line-triggered."""
    dev = single_ion_raman_device()
    sp = dev.crystal.species[0]
    _f, d1, d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], dev.field.B_gauss)
    mains = Mains(60.0, {1: 2e-9}, {1: 0.0}, "free_running")
    noisy = _with(
        dev, S_B=ou_spectrum(1e-18, 1e-3, "T^2/(rad/s)", omega_max_rad_s=2.0 * math.pi * 2e4), mains=mains
    )
    s = noisy.noise.sample(np.random.default_rng(1), device=noisy, duration_s=2e-3)
    traj = s.trajectory(key_qubit_trajectory_hz(0))
    assert traj is not None and traj.times_s[-1] == pytest.approx(2e-3) and traj.times_s.size >= 65
    phase = s.values[KEY_MAINS_PHASE]
    # remove the sampled S_B part statistically: the mains component alone at t = 0 is d1 x 2e-9 T x cos(phase) (gauss inside d1)
    expected_mains = d1 * (2e-9 * GAUSS_PER_TESLA) * math.cos(phase)
    assert (
        abs(traj.values[0] - expected_mains)
        < 5.0 * d1 * math.sqrt(1e-18) * GAUSS_PER_TESLA + abs(0.5 * d2 * (2e-5) ** 2) + 1e-9
    )
    triggered = _with(dev, mains=dataclasses.replace(mains, trigger="line_triggered"))
    s2 = triggered.noise.sample(np.random.default_rng(1), device=triggered, duration_s=2e-3)
    assert s2.values[KEY_MAINS_PHASE] == 0.0
    # the device-less Appendix E call draws only the raw quantities
    raw = noisy.noise.sample(np.random.default_rng(1))
    assert KEY_MAINS_PHASE in raw.values and key_qubit_offset_hz(0) not in raw.values and not raw.ou_grids


def test_noise_model_requires_no_device_content_to_be_quiet_and_reports_white_intensity() -> None:
    nm = make_noise()
    assert nm.is_quiet() and nm.intensity_white_density() == 0.0 and nm.laser_frequency_drift is None
    nm2 = dataclasses.replace(nm, laser_intensity=white_spectrum(2e-9, "1/(rad/s)"))
    assert nm2.intensity_white_density() == 2e-9 and nm2.is_quiet(), (
        "a white level is a Lindblad channel, not a sample"
    )
    assert isinstance(nm, NoiseModel)
