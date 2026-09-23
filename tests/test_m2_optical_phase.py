"""The optical phase factor on sigma_+ (Section 13 row "Optical phase factor on sigma_+"; PLAN.md:808).

Two facts the builder must carry and did not:

* the factor is e^{+i(Delta k . X_i - Delta phi_i)} with Delta phi_i = phi_2 - phi_1 for a Raman pair with
  Delta k = k_1 - k_2 and E_j ~ cos(k_j . r - omega_j t + phi_j), so a sampled +0.4 rad on beam 1 plays the axis
  at +0.4 rad and not -0.4 (M2 audit E2);
* X_i carries the SAMPLED ion position, so a stray-field drift u_0 = Q E_dc/(m omega^2) (Section 4.1.1;
  Berkeland 1998 Eq. 16) enters the relative optical phase as Delta k . (u_0(j) - u_0(i)) and not only through the
  beams' intensity profile (M2 audit E1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.control.native import gpi2
from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import Schedule
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.noise.sampling import NoiseSample, key_beam_phase_rad, key_position_offset_m, quiet_sample
from tests.m2_fixtures import single_ion_raman_device, two_ion_raman_device

DOWN = np.array([1.0, 0.0], dtype=complex)


def _carrier_state(dev, dd, sample) -> np.ndarray:  # type: ignore[no-untyped-def]
    """|psi> after a pi/2 carrier pulse from |0> with every mode frozen at n = 0, normalized.

    The pulse length is read back from the builder's own effective Rabi frequency (carrier x Debye-Waller), so the
    rotation is exactly pi/2 whatever the frozen factors are and only the AXIS is under test.
    """
    space = HilbertSpace((2,), (), None, tuple(range(len(dev.crystal.modes))))
    drive = square_drive(dd, include_stark=False)
    probe = build_hamiltonian(dev, (Pulse(drive, 0.0, 1e-9, "p", ()),), space, sample=quiet_sample())
    om = probe.records[0].omega_peak_rad_s
    pulse = Pulse(drive, 0.0, 0.5 * math.pi / om, "p", ())
    eng = JointExactEngine(builder_options=BuilderOptions())
    tr = eng.run_pulses(
        dev,
        Schedule((pulse,), (), (), {0: 0.0}),
        space.initial_state([0]),
        space,
        sample,
        SeedSpec(0),
        SolverOptions(),
    )
    vec = np.asarray(tr.final.joint.full()).reshape(-1)
    vec = vec / np.linalg.norm(vec)
    # the frozen modes are one-dimensional factors, so the joint ket IS the two-level block
    return np.asarray(vec)


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>| for normalized states: the comparison up to the frame's own global phase."""
    return float(abs(np.vdot(a, b)))


def test_beam_path_phase_sign_is_the_plans_delta_phi() -> None:
    """A sampled phi_1 = +0.4 rad gives Delta phi = phi_2 - phi_1 = -0.4, so the factor is e^{+i0.4}: GPi2(+0.4).

    The WRONG sign is asserted to fail, which is what makes the pin sharp (Section 9.6's "a rule 'pinned' passes for
    either sign"). Before the M2 fix the builder played GPi2(-0.4) here.
    """
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    right = gpi2(+0.4) @ DOWN
    wrong = gpi2(-0.4) @ DOWN
    played = _carrier_state(dev, dd, NoiseSample(0, {key_beam_phase_rad(0): 0.4}, {}))
    assert _overlap(played, right) == pytest.approx(1.0, abs=1e-4)
    assert _overlap(played, wrong) < 1.0 - 1e-3
    # beam 2 enters with the opposite sign: phi_2 = +0.4 is Delta phi = +0.4, hence GPi2(-0.4)
    played2 = _carrier_state(dev, dd, NoiseSample(0, {key_beam_phase_rad(1): 0.4}, {}))
    assert _overlap(played2, wrong) == pytest.approx(1.0, abs=1e-4)
    assert _overlap(played2, right) < 1.0 - 1e-3
    # a common-mode path drift cancels: Delta phi depends on the DIFFERENCE only
    common = NoiseSample(0, {key_beam_phase_rad(0): 0.4, key_beam_phase_rad(1): 0.4}, {})
    assert _overlap(_carrier_state(dev, dd, common), gpi2(0.0) @ DOWN) == pytest.approx(1.0, abs=1e-6)


def test_beam_path_phase_is_reported_as_delta_phi() -> None:
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    space = HilbertSpace((2,), (), None, tuple(range(len(dev.crystal.modes))))
    pulse = Pulse(square_drive(dd, include_stark=False), 0.0, 1e-6, "p", ())
    built = build_hamiltonian(dev, (pulse,), space, sample=NoiseSample(0, {key_beam_phase_rad(0): 0.4}, {}))
    assert any("Delta phi = -0.4" in a for a in built.approximations)


def _sigma_plus_phase(built, space, ion: int) -> float:  # type: ignore[no-untyped-def]
    """arg <...1_ion...| H(0) |0, 0>: the optical phase on that ion's sigma_+ coefficient."""
    h0 = built.H(0.0)
    levels = [0] * space.n_ions
    lower = space.internal_ket(levels)
    levels[space.ion_labels.index(ion)] = 1
    upper = space.internal_ket(levels)
    return float(np.angle(complex(upper.dag() * h0 * lower)))


def test_ion_position_drift_enters_the_optical_phase() -> None:
    """Section 13: "beam-path AND ion-position drift stored as one scalar per ion per pulse".

    The two-ion fixture's crosstalk drive addresses ion 0 and spills onto ion 1, so the RELATIVE optical phase
    Delta k . (u_0(1) - u_0(0)) is observable in the builder's coefficient for ion 1. At 14.3 nm (a 1 V/m stray
    field on the 1 MHz axial mode of 171Yb+) on a 355 nm pair the phase is of order half a radian, six orders of
    magnitude above the 2.6e-7 the intensity profile sees; before the M2 fix it was dropped entirely.
    """
    dev = two_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    drive = square_drive(dd, include_stark=False, crosstalk={1: 0.1})
    delta_k = drive.delta_k(dev.beams)
    space = HilbertSpace((2, 2), (), None, tuple(range(len(dev.crystal.modes))))
    pulse = Pulse(drive, 0.0, 1e-6, "p", ())
    u0 = 14.3e-9  # Q E_dc/(m omega^2) at 1 V/m on the 1 MHz axial mode of 171Yb+ (Berkeland 1998 Eq. 16)
    axis = 0  # the 90-degree pair's Delta k = k(x - y) lies in the x-y plane
    drift = np.zeros(3)
    drift[axis] = u0
    expected = float(np.dot(delta_k, drift))
    assert abs(expected) > 0.1, "the fixture needs a Delta k component along the drift for this to be a test"

    quiet = build_hamiltonian(dev, (pulse,), space, sample=quiet_sample())
    moved = build_hamiltonian(
        dev, (pulse,), space, sample=NoiseSample(0, {key_position_offset_m(1, axis): u0}, {})
    )
    delta = _sigma_plus_phase(moved, space, 1) - _sigma_plus_phase(quiet, space, 1)
    assert math.remainder(delta, 2.0 * math.pi) == pytest.approx(
        math.remainder(expected, 2.0 * math.pi), abs=1e-9
    )
    assert any("ion-position drift phase" in a for a in moved.approximations)
    # the addressed ion's own drift is a per-ion constant the frame alignment of Section 7.5 step 3 absorbs, so a
    # COMMON drift leaves the relative phase alone
    both = NoiseSample(0, {key_position_offset_m(0, axis): u0, key_position_offset_m(1, axis): u0}, {})
    common = build_hamiltonian(dev, (pulse,), space, sample=both)
    assert _sigma_plus_phase(common, space, 1) == pytest.approx(_sigma_plus_phase(quiet, space, 1), abs=1e-12)


def test_ion_position_drift_dwarfs_the_intensity_effect() -> None:
    """The magnitude claim of the M2 audit: the dropped phase is 0.51 rad where the retained amplitude effect is 2.6e-7."""
    dev = two_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    drive = square_drive(dd, include_stark=False, crosstalk={1: 0.1})
    delta_k = drive.delta_k(dev.beams)
    u0 = 14.3e-9
    phase = abs(float(np.dot(delta_k, np.array([u0, 0.0, 0.0]))))
    waist = dev.beams[0].waist_m
    amplitude = abs(math.sqrt(math.exp(-2.0 * u0**2 / waist**2)) - 1.0)
    assert phase > 1e5 * amplitude
