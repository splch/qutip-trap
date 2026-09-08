"""The sampled beam-path phase reaches the builder (Section 6.3 route (d), Section 7.10; ledger conv.beam_phase_spectrum):
Delta phi(t) = phi_2(t) - phi_1(t) of the two Raman paths multiplies the beat note by e^{-i Delta phi(t)}, the same
convention as the quasi-static key_beam_phase_rad (PLAN.md:808), so constant trajectories reproduce the quasi-static build
exactly and a time-dependent one turns the sigma_+ coefficient by -Delta phi(t)."""

from __future__ import annotations

import numpy as np
import pytest

from qutip_trap.api import HilbertSpace, Pulse
from qutip_trap.dynamics.hamiltonian import build_hamiltonian
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.noise.processes import Trajectory
from qutip_trap.noise.sampling import (
    NoiseSample,
    key_beam_phase_rad,
    key_beam_phase_trajectory_rad,
    quiet_sample,
)
from tests.m2_fixtures import single_ion_raman_device

DURATION_S = 2e-6


def _built(dev, sample):  # type: ignore[no-untyped-def]
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    space = HilbertSpace((2,), (), None, tuple(range(len(dev.crystal.modes))))
    pulse = Pulse(square_drive(dd, include_stark=False), 0.0, DURATION_S, "p", ())
    return build_hamiltonian(dev, (pulse,), space, sample=sample), space


def _sigma_plus(built, space, t: float) -> complex:  # type: ignore[no-untyped-def]
    h = built.H(t)
    return complex(space.internal_ket([1]).dag() * h * space.internal_ket([0]))


def _grid(values0, values1):  # type: ignore[no-untyped-def]
    times = np.linspace(0.0, DURATION_S, 65)
    return {
        key_beam_phase_trajectory_rad(0): Trajectory(
            times, np.asarray(values0(times), dtype=float)
        ).as_grid(),
        key_beam_phase_trajectory_rad(1): Trajectory(
            times, np.asarray(values1(times), dtype=float)
        ).as_grid(),
    }


def test_constant_beam_phase_trajectories_reproduce_the_quasi_static_build() -> None:
    dev = single_ion_raman_device()
    static, space = _built(dev, NoiseSample(0, {key_beam_phase_rad(0): 0.3, key_beam_phase_rad(1): 0.1}, {}))
    sampled, _ = _built(dev, NoiseSample(0, {}, _grid(lambda t: 0.3 + 0.0 * t, lambda t: 0.1 + 0.0 * t)))
    quiet, _ = _built(dev, quiet_sample())
    for t in (0.0, 0.37e-6, 1.2e-6, 1.9e-6):
        a = _sigma_plus(static, space, t)
        b = _sigma_plus(sampled, space, t)
        q = _sigma_plus(quiet, space, t)
        assert b == pytest.approx(a, rel=1e-12, abs=1e-9 * abs(q))
        # Delta phi = phi_2 - phi_1 = -0.2, so the sigma_+ coefficient turns by e^{-i Delta phi} = e^{+0.2 i}
        assert np.angle(b / q) == pytest.approx(0.2, abs=1e-12)
    assert any("sampled beam-path phase trajectory" in a for a in sampled.approximations)
    assert not any("sampled beam-path phase trajectory" in a for a in static.approximations)


def test_a_time_dependent_beam_phase_turns_the_beat_note_by_minus_delta_phi_of_t() -> None:
    dev = single_ion_raman_device()
    phi1 = lambda t: 0.25 * np.sin(2.0 * np.pi * t / DURATION_S)  # noqa: E731
    phi2 = lambda t: -0.1 * np.cos(2.0 * np.pi * t / DURATION_S)  # noqa: E731
    grids = _grid(phi1, phi2)
    sampled, space = _built(dev, NoiseSample(0, {}, grids))
    quiet, _ = _built(dev, quiet_sample())
    # the builder reads the trajectories through Trajectory's linear interpolation, so the reference does too
    t1 = Trajectory.from_grid(grids[key_beam_phase_trajectory_rad(0)])
    t2 = Trajectory.from_grid(grids[key_beam_phase_trajectory_rad(1)])
    for t in np.linspace(0.05e-6, 1.95e-6, 7):
        expected = -(t2(float(t)) - t1(float(t)))
        assert np.angle(_sigma_plus(sampled, space, t) / _sigma_plus(quiet, space, t)) == pytest.approx(
            expected, abs=1e-9
        )
    # a common-mode path drift cancels in the differential phase
    common, _ = _built(dev, NoiseSample(0, {}, _grid(phi1, phi1)))
    for t in (0.3e-6, 1.1e-6):
        assert np.angle(_sigma_plus(common, space, t) / _sigma_plus(quiet, space, t)) == pytest.approx(
            0.0, abs=1e-12
        )


def test_one_beam_without_a_trajectory_contributes_zero_and_mismatched_grids_are_refused() -> None:
    dev = single_ion_raman_device()
    times = np.linspace(0.0, DURATION_S, 65)
    only_beam_1 = {key_beam_phase_trajectory_rad(1): Trajectory(times, 0.4 + 0.0 * times).as_grid()}
    sampled, space = _built(dev, NoiseSample(0, {}, only_beam_1))
    quiet, _ = _built(dev, quiet_sample())
    assert np.angle(_sigma_plus(sampled, space, 0.7e-6) / _sigma_plus(quiet, space, 0.7e-6)) == pytest.approx(
        -0.4, abs=1e-12
    )
    other = np.linspace(0.0, DURATION_S, 33)
    bad = {
        key_beam_phase_trajectory_rad(0): Trajectory(times, 0.0 * times).as_grid(),
        key_beam_phase_trajectory_rad(1): Trajectory(other, 0.0 * other).as_grid(),
    }
    with pytest.raises(ValueError, match="different time grids"):
        _built(dev, NoiseSample(0, {}, bad))
