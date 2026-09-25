"""Sympathetic cooling of a mixed crystal at level A: the coolant is the illuminated set, the shared modes take their
nbar from its stage and a mode it barely participates in is refused; the Home 2009 Be-Mg-Mg-Be axial spectrum."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from qutip_trap.device.presets import OBLIQUE
from qutip_trap.light.beams import Beam
from qutip_trap.prep.doppler import (
    PARTICIPATION_THRESHOLD,
    UncooledModeError,
    doppler_cooling,
    models_per_ion,
)
from qutip_trap.species.polarization import spherical_basis
from qutip_trap.trap.crystal import build_crystal
from qutip_trap.units import ATOMIC_MASS_KG as U_KG
from qutip_trap.units import C_M_PER_S, TWO_PI
from tests.fixtures import (
    TWO_LEVEL_EXCITED_PLUS,
    TWO_LEVEL_GROUND,
    gamma_rad_s,
    power_for_rabi,
    structure,
    two_level_atom,
)

# Home 2013 Table I, the NIST 9Be+/24Mg+ apparatus of Home et al. 2009
BE_MASS_U = 9.0121822
MG24_MASS_U = 23.985042
BE_SINGLE_ION_HZ = (12.26e6, 11.19e6, 2.69e6)
MG24_SINGLE_ION_HZ = (4.82e6, 3.72e6, 1.65e6)


COOLANT_QUBIT = (40.0, 171.0)
"""(coolant, qubit) masses in u of the mixed pair: a 40Ca+-like coolant next to a 171Yb+-like qubit."""
MIXED_TRAP_HZ = (30e6, 31e6, 3e6)
"""The coolant's single-ion frequencies: strong radial confinement, so the radial modes decouple by species."""


def _sigma_plus_along(st, k_hat, omega, delta, waist_m=2e-3):
    """A sigma+ beam along B = k_hat with Rabi frequency ``omega`` on the fixture's closed line (wide: one intensity)."""
    _m, _z, e_plus = spherical_basis(k_hat)
    pol = tuple(complex(x) for x in e_plus)
    power = power_for_rabi(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, omega, waist_m, pol, k_hat)
    omega_l = (
        TWO_PI * (st.state(TWO_LEVEL_EXCITED_PLUS).energy_hz - st.state(TWO_LEVEL_GROUND).energy_hz) + delta
    )
    return Beam(TWO_PI * C_M_PER_S / omega_l, k_hat, pol, waist_m, power, (0.0, 0.0, 0.0))


# ---- the mixed crystal's mode structure --------------------------------------------------------------------------


def test_home_2009_be_mg_mg_be_axial_spectrum_and_the_251_khz_mode_spacing() -> None:
    """From Home 2013's single-ion frequencies the Be-Mg-Mg-Be axial modes are 1.9488, 4.0854, 5.4862 and 5.7373 MHz
    (1e-3) with Jost 2009's mass-weighted mode vectors (0.005) and Home 2009's 251 kHz spacing of the top two
    (2 kHz)."""
    be = two_level_atom(mass_kg=BE_MASS_U * U_KG)
    mg = two_level_atom(mass_kg=MG24_MASS_U * U_KG)
    freqs = np.array([BE_SINGLE_ION_HZ, MG24_SINGLE_ION_HZ, MG24_SINGLE_ION_HZ, BE_SINGLE_ION_HZ])
    # the dc axial curvature is mass independent, so 1.65 MHz for 24Mg+ is 2.69 MHz sqrt(m_Be/m_Mg)
    assert BE_SINGLE_ION_HZ[2] * math.sqrt(BE_MASS_U / MG24_MASS_U) == pytest.approx(
        MG24_SINGLE_ION_HZ[2], rel=2e-3
    )
    crystal = build_crystal((be, mg, mg, be), TWO_PI * freqs)
    z = np.asarray(crystal.positions_m)[:, 2]
    assert list(np.argsort(z)) == [0, 1, 2, 3]  # the ions sit in the order they were given: Be, Mg, Mg, Be
    axial = sorted((m for m in crystal.modes if m.family == "axial"), key=lambda m: m.omega_hz)
    assert len(axial) == 4
    got = [m.omega_hz / 1e6 for m in axial]
    for value, printed in zip(got, (2.0, 4.1, 5.5, 5.7)):
        assert value == pytest.approx(printed, abs=0.06)  # Jost's table, printed to 0.1 MHz
    assert got == [
        pytest.approx(1.9488, abs=1e-3),
        pytest.approx(4.0854, abs=1e-3),
        pytest.approx(5.4862, abs=1e-3),
        pytest.approx(5.7373, abs=1e-3),
    ]
    # Home's 251 kHz spacing of the two highest axial modes, and its internal 3 delta = 3 x 83.6 kHz check
    spacing_hz = axial[3].omega_hz - axial[2].omega_hz
    assert spacing_hz == pytest.approx(251e3, abs=2e3)
    assert spacing_hz == pytest.approx(3.0 * 83.6e3, abs=2e3)
    # Jost's mode vectors are the mass-weighted components; the overall sign is a gauge
    for mode, printed in zip(
        axial,
        (
            (0.32, 0.63, 0.63, 0.32),
            (0.47, 0.53, 0.53, 0.47),
            (0.63, 0.32, 0.32, 0.63),
            (0.53, 0.47, 0.47, 0.53),
        ),
    ):
        pattern = mode.displacement_pattern()
        amps = [float(np.linalg.norm(pattern[i])) for i in range(4)]
        assert amps == [pytest.approx(p, abs=0.005) for p in printed]
        assert sum(a * a for a in amps) == pytest.approx(1.0, rel=1e-9)  # mass-weighted orthonormality
    # a mixed crystal has no exact centre-of-mass mode: the in-phase amplitudes are NOT all equal
    lowest = axial[0].displacement_pattern()
    assert float(np.linalg.norm(lowest[1])) / float(np.linalg.norm(lowest[0])) == pytest.approx(
        1.95, abs=0.02
    )


# ---- one stage call, a coolant and a transparent qubit species ----------------------------------------------------


def _mixed_frequencies(masses_u: Sequence[float], reference_hz: tuple[float, float, float]) -> np.ndarray:
    """Per-ion single-ion secular frequencies from ion 0's: the rf (radial) part of omega scales as 1/m and the
    static (axial) part as 1/sqrt(m) at a mass-independent dc curvature (Home 2013 Eq. 6)."""
    ref = float(masses_u[0])
    return np.array(
        [
            [reference_hz[0] * ref / m, reference_hz[1] * ref / m, reference_hz[2] * math.sqrt(ref / m)]
            for m in (float(x) for x in masses_u)
        ]
    )


def _coolant_weights(crystal) -> dict[int, float]:
    """c_{0,m}^2 per mode: the coolant ion's participation, which is the level-A objective's weight W_m."""
    out: dict[int, float] = {}
    for k, mode in enumerate(crystal.modes):
        out[k] = float(np.linalg.norm(mode.displacement_pattern()[0])) ** 2
    return out


def _mixed_pair(coolant_mass_u: float, qubit_mass_u: float, freqs_hz: tuple[float, float, float]):
    """(crystal, coolant structure, cooling beam) for a two-ion crystal whose ion 0 is the coolant."""
    coolant = two_level_atom(mass_kg=coolant_mass_u * U_KG)
    qubit = two_level_atom(mass_kg=qubit_mass_u * U_KG)
    freqs = _mixed_frequencies((coolant_mass_u, qubit_mass_u), freqs_hz)
    crystal = build_crystal((coolant, qubit), TWO_PI * freqs)
    st = structure(coolant, b_hat=OBLIQUE)
    beam = _sigma_plus_along(st, OBLIQUE, 0.05 * gamma_rad_s(), -0.5 * gamma_rad_s())
    return crystal, st, beam


def test_a_stage_illuminating_only_the_coolant_cools_the_shared_modes_through_its_participation() -> None:
    """Cooling only the coolant of a mixed pair gives each mode the single-ion nbar and the single-ion rate times its
    participation weight (3e-3), the axial weights being 0.07721 and 0.92279 (1e-5)."""
    crystal, st, beam = _mixed_pair(*COOLANT_QUBIT, MIXED_TRAP_HZ)
    weights = _coolant_weights(crystal)
    bright = [k for k, w in weights.items() if w >= PARTICIPATION_THRESHOLD]
    res = doppler_cooling(st, [beam], crystal, illuminated=[0], modes=bright)
    assert res.illuminated == (0,)
    # the mixed axial pair carries the coolant with weights 0.0772 and 0.9228
    axial = sorted((k for k in bright if crystal.modes[k].family == "axial"), key=lambda k: weights[k])
    assert [weights[k] for k in axial] == [
        pytest.approx(0.07721, abs=1e-5),
        pytest.approx(0.92279, abs=1e-5),
    ]
    coolant = two_level_atom(mass_kg=COOLANT_QUBIT[0] * U_KG)
    for m in res.modes:
        assert 0.0 < m.participation_weight <= 1.0
        # a SINGLE coolant ion whose own axial mode sits at the same frequency, seen by the same oblique beam:
        # identical nbar (the participation cancels) and a rate scaled by exactly the participation weight
        f = m.omega_rad_s / TWO_PI
        solo = build_crystal((coolant,), TWO_PI * np.array([[10.0 * f, 10.1 * f, f]]))
        one = doppler_cooling(st, [beam], solo, modes=[solo.mode_index("axial", 0)])
        assert m.nbar == pytest.approx(one.modes[0].nbar, rel=3e-3)
        assert m.rate_per_s == pytest.approx(m.participation_weight * one.modes[0].rate_per_s, rel=3e-3)
    # each axis family's participations complete to one
    for family in ("axial", "transverse_1", "transverse_2"):
        members = [k for k, mode in enumerate(crystal.modes) if mode.family == family]
        assert sum(weights[k] for k in members) == pytest.approx(1.0, rel=1e-9)


def test_models_per_ion_takes_a_per_ion_structure_and_refuses_an_illuminated_ion_without_one() -> None:
    """models_per_ion accepts a per-ion mapping of structures (the same rates to 1e-12) and refuses an illuminated ion
    without one."""
    crystal, st, beam = _mixed_pair(*COOLANT_QUBIT, MIXED_TRAP_HZ)
    shared = models_per_ion(st, [beam], crystal, (0,))
    mapped = models_per_ion({0: st}, [beam], crystal, (0,))
    assert set(shared) == set(mapped) == {0}
    assert mapped[0].scattering_rate_per_s() == pytest.approx(shared[0].scattering_rate_per_s(), rel=1e-12)
    with pytest.raises(KeyError, match="no AtomicStructure for illuminated ions"):
        models_per_ion({0: st}, [beam], crystal, (0, 1))
    # and the whole stage goes through the mapping
    bright = [k for k, w in _coolant_weights(crystal).items() if w >= PARTICIPATION_THRESHOLD]
    stage = doppler_cooling({0: st}, [beam], crystal, illuminated=[0], modes=bright)
    assert stage.illuminated == (0,)
    assert [m.mode for m in stage.modes] == bright


def test_a_mode_the_coolant_barely_participates_in_is_refused_instead_of_reported_as_a_steady_state() -> None:
    """Modes the coolant barely participates in are refused, and with participation_threshold = 0 they report relaxation
    rates below 1/s."""
    # a light coolant and a heavy qubit under strong radial confinement: the radial modes decouple by species
    crystal, st, beam = _mixed_pair(*COOLANT_QUBIT, MIXED_TRAP_HZ)
    coolant_weight = _coolant_weights(crystal)
    dark = [k for k, w in coolant_weight.items() if w < PARTICIPATION_THRESHOLD]
    assert dark, "the fixture must have at least one mode the coolant does not participate in"
    with pytest.raises(UncooledModeError, match="participation"):
        doppler_cooling(st, [beam], crystal, illuminated=[0])
    for k in dark:
        with pytest.raises(UncooledModeError, match="participation"):
            doppler_cooling(st, [beam], crystal, illuminated=[0], modes=[k])
    # lowering the threshold reports the steady state and shows why it is refused: a relaxation time of many seconds
    loose = doppler_cooling(st, [beam], crystal, illuminated=[0], modes=dark, participation_threshold=0.0)
    for m in loose.modes:
        assert 0.0 < m.rate_per_s < 1.0
    # the coolant-participating modes come back normally
    bright = [k for k, w in coolant_weight.items() if w >= PARTICIPATION_THRESHOLD]
    assert bright
    ok = doppler_cooling(st, [beam], crystal, illuminated=[0], modes=bright)
    assert all(m.nbar > 0.0 for m in ok.modes)


def test_a_stated_minimum_rate_refuses_a_mode_that_cannot_reach_its_steady_state_in_the_stage() -> None:
    """min_rate_per_s refuses the stage at twice its slowest mode's rate and passes it at half."""
    crystal, st, beam = _mixed_pair(*COOLANT_QUBIT, MIXED_TRAP_HZ)
    weights = _coolant_weights(crystal)
    bright = [k for k, w in weights.items() if w >= PARTICIPATION_THRESHOLD]
    res = doppler_cooling(st, [beam], crystal, illuminated=[0], modes=bright)
    slowest = min(m.rate_per_s for m in res.modes)
    with pytest.raises(UncooledModeError, match="min_rate_per_s"):
        doppler_cooling(st, [beam], crystal, illuminated=[0], modes=bright, min_rate_per_s=2.0 * slowest)
    assert doppler_cooling(
        st, [beam], crystal, illuminated=[0], modes=bright, min_rate_per_s=0.5 * slowest
    ).modes
