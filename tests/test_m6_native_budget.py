"""Section 9.6 row "Native-gate identities": the Section 11.1 pulse against the native MS matrix (M6 audit items P1-7, P1-8).

The row states that a pulse reproduces its native matrix "within each gate's intrinsic budget (off-resonant carrier
approx (Omega/nu)^2, Bessel force saturation, residual displacement, Debye-Waller), which is reported beside the result and
is 1.1 x 10^-4 for the Section 11.1 pulse at eta = 0.08 with no noise; to 1e-6 only with ``lamb_dicke_order`` and ``rwa``
on or for eta = 0 microwave carriers".

The Section 11.1 fixture is the two-ion 171Yb+ chain at (3.0, 2.9, 1.0) MHz with the Raman Delta k in the transverse
plane, its x-COM mode at 3.0000 MHz (eta = 0.078577 on both ions, the plan's "0.080"), a square bichromatic pulse with the
tones inside the sideband at eps/2pi = 10 kHz (a 100 us single loop) at the maximally entangling closure eta Omega/eps =
1/2, and d_m = 12 (joint dimension 48). The eta = 0 half of the row is ``tests/test_native_pulses.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.calibration.entangling import calibrate_entangling_angle, gate_space
from qutip_trap.control import native
from qutip_trap.control.schedule import PlayedGate, Schedule, entangling_pulses, ms_spin_phases
from qutip_trap.control.shaping import GateModes, symmetric_pulse
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.frames import PhaseFrame
from qutip_trap.dynamics.hamiltonian import BuilderOptions
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.run.job import intrinsic_budget, roos_bessel_saturation, sideband_lamb_dicke_deficit
from qutip_trap.run.space import select_space
from qutip_trap.units import TWO_PI
from qutip_trap.validation.two_qubit_closed_forms import roos_force_saturation
from tests.m4_fixtures import (
    derived_seeds,
    raman_gate_drives,
    table_with_waveform,
    two_ion_device,
    two_ion_modes,
)

GATE_MODE = 3
"""The x-COM mode of the M4 two-ion fixture at 3.0000 MHz: the Section 11.1 gate mode."""
EPSILON_HZ = 10e3
"""Section 11.1: the tones sit eps/2pi = 10 kHz inside the sideband, so one loop closes in 100 us."""
KET00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)


def _single_mode_fixture():  # type: ignore[no-untyped-def]
    """(device, drives, the one-mode GateModes of the x-COM, the closed-form pulse, the d_m = 12 space)."""
    device = two_ion_device()
    drives = raman_gate_drives(2)
    full = two_ion_modes(device)
    k = full.modes.index(GATE_MODE)
    modes = GateModes(
        ions=(0, 1),
        modes=(GATE_MODE,),
        omega_rad_s=(full.omega_rad_s[k],),
        eta={0: (full.eta[0][k],), 1: (full.eta[1][k],)},
        nbar=(0.0,),
    )
    waveform = symmetric_pulse(
        modes, gate_mode=GATE_MODE, loops=1, epsilon_hz=EPSILON_HZ, all_modes=False
    ).waveform
    space = gate_space(
        modes, 2, waveform=waveform, frozen=(0, 1, 2, 4, 5), n_modes_total=6, d_min=12, d_max=12
    )
    return device, drives, modes, waveform, space


def _played(device, drives, waveform, table, *, builder):  # type: ignore[no-untyped-def]
    """The waveform as one scheduled entangling gate, with the spin phases the pair scheduler would set."""
    spins, _ = ms_spin_phases(waveform, (0, 1), (0.0, 0.0), PhaseFrame())
    pulses = entangling_pulses(
        waveform,
        drives,
        spin_phases_rad=spins,
        t_start_s=0.0,
        table=table,
        gate_id="ms11",
        stark_compensation=False,
    )
    return Schedule(
        tuple(pulses),
        (),
        (),
        {0: 0.0, 1: 0.0},
        gates=(PlayedGate("ms11", "ms", (0, 1), waveform, drives[0].beams, 0.0, waveform.duration_s),),
    )


def test_the_section_11_1_fixture_is_the_plan_s_pulse() -> None:
    """d_m = 12 on one mode is joint dimension 48; the loop closes at 100 us with chi = pi/4 and eta Omega/eps = 1/2."""
    _device, _drives, modes, waveform, space = _single_mode_fixture()
    assert space.dims == [2, 2, 12] and space.dimension == 48
    assert waveform.duration_s == pytest.approx(1.0 / EPSILON_HZ, rel=1e-12)
    assert waveform.chi_total_rad == pytest.approx(math.pi / 4, rel=1e-9)
    eta = abs(modes.eta[0][0])
    assert eta == pytest.approx(0.078577, abs=1e-6), "the plan's eta = 0.080 fixture mode"
    assert waveform.segments is not None
    amp = float(waveform.segments[0].amplitude_hz[(0, "blue")])
    # the plan's eta Omega/eps = 1/2 is the leading-order closure; the exact chi = pi/4 solve lands at 0.499583, a
    # 8.3e-4 relative correction from the |alpha|^2 term of the Section 4.4.3 kernel
    assert eta * amp / EPSILON_HZ == pytest.approx(0.499583, rel=1e-5)
    assert eta * amp / EPSILON_HZ == pytest.approx(0.5, rel=1e-3), "the closure of Section 4.4.1"


@pytest.mark.slow
def test_the_section_11_1_pulse_reproduces_the_native_ms_matrix_inside_its_intrinsic_budget() -> None:
    """The exact play (no Lamb-Dicke expansion, no RWA, no noise) of the calibrated Section 11.1 pulse against
    MS(0, 0, pi/2), and the row's 1.1e-4.

    The row's number is the off-resonant carrier excitation probability of one tone, (Omega_tone/(2 nu))^2 = 1.143e-4 at
    the calibrated Omega_tone/2pi = 64.148 kHz and nu/2pi = 3.0000 MHz (1.123e-4 at the closed-form amplitude). The realized infidelity 4.66e-5 sits inside it,
    and is itself dominated by that term (two ions, each excited off resonance by both tones). NOTE the discrepancy
    recorded as ``anchor.m6.section_11_1_native_identity``: ``intrinsic_budget``'s ``carrier_scale`` key is (Omega/nu)^2 =
    4.49e-4, four times the row's number, so the row's own formula "approx (Omega/nu)^2" and its value 1.1e-4 disagree by
    a factor 4 and the value is the physically correct one."""
    device, drives, modes, waveform0, space = _single_mode_fixture()
    rabi, _stark = derived_seeds(device, drives)
    builder = BuilderOptions(include_stark=False)
    run = calibrate_entangling_angle(
        device,
        waveform0,
        (0, 1),
        drives,
        table_with_waveform((0, 1), waveform0, rabi_hz=rabi),
        space=space,
        tolerance_rad=1e-7,
        builder_options=builder,
        max_iterations=8,
    )
    assert run.converged
    waveform = run.waveform
    table = table_with_waveform((0, 1), waveform, rabi_hz=rabi)
    sched = _played(device, drives, waveform, table, builder=builder)
    traces = JointExactEngine(table=table, builder_options=builder).run_pulses(
        device,
        sched,
        space.initial_state([0, 0]),
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    target = qt.Qobj((native.ms(0.0, 0.0, math.pi / 2) @ KET00).reshape(-1, 1), dims=[[2, 2], [1, 1]])
    infidelity = 1.0 - float(np.real(qt.expect(traces.final.internal, target)))
    assert waveform.segments is not None
    omega_tone = TWO_PI * float(waveform.segments[0].amplitude_hz[(0, "blue")])
    nu = modes.omega_rad_s[0]
    off_resonant = (omega_tone / (2.0 * nu)) ** 2
    # the row's 1.1e-4, to the printed digits (recomputed value 1.1430e-4 at the calibrated amplitude; 1.1228e-4 at the
    # closed-form amplitude before the exact spot check corrected it by 0.9 %)
    assert off_resonant == pytest.approx(1.1430e-4, rel=2e-3)
    assert off_resonant == pytest.approx(1.1e-4, abs=5e-6), "the Section 9.6 row"
    # the identity holds inside it, and is not trivially small: it IS the off-resonant carrier, twice over (two ions)
    assert infidelity == pytest.approx(4.659e-5, rel=2e-2), infidelity
    assert infidelity < off_resonant
    assert infidelity > 0.25 * off_resonant
    # the repo's carrier_scale key is 4x the row's number: the recorded discrepancy
    selection = select_space(
        device, sched, SolverOptions(), nbar=dict.fromkeys(range(6), 0.0), caps={GATE_MODE: 12}
    )
    budget = intrinsic_budget(device, sched, selection)
    # (Omega_tone/nu_min)^2 with nu_min the x-rocking mode at 2.8284 MHz, so 4.5x rather than exactly 4x the row's number
    assert budget["ms11.carrier_scale"] == pytest.approx(5.144e-4, rel=2e-2)
    assert budget["ms11.carrier_scale"] > 4.0 * off_resonant
    assert infidelity < budget["ms11.carrier_scale"]
    # the loop closes, so the residual displacement is the sub-1e-3 term the row lists, and there is no thermal
    # Debye-Waller loss from the ground state
    assert budget["ms11.debye_waller"] == 0.0
    assert budget["ms11.residual_displacement"] < 1e-3
    # the sideband element's Lamb-Dicke deficit: 1 - <n+1|D(i eta)|n>/(eta sqrt(n+1)) at the cap's top n = 11, reported
    # beside the budget and NOT summed (its mean is the s^2 calibration's rescaling, its spread the Debye-Waller term)
    assert budget["ms11.sideband_lamb_dicke_deficit"] == pytest.approx(3.0554e-2, rel=1e-2)
    # Section 9.6's "Bessel force saturation" is Roos Eq. 17: f = 1 - (J_0 + J_2)(4 Omega/mu) at the tone amplitude and the
    # tone-to-carrier detuning, entered as the uncalibrated angle error's infidelity sin^2(pi f/2)
    gate = sched.gates[0]
    seg = gate.waveform.segments[0]
    omega_hz = max(abs(float(a)) for a in seg.amplitude_hz.values())
    mu_hz = min(abs(float(d)) for d in seg.detuning_hz.values())
    f = 1.0 - roos_force_saturation(TWO_PI * omega_hz, TWO_PI * mu_hz)
    assert 1e-4 < f < 2e-3, (
        f
    )  # x = 4 Omega/mu = 0.085 at Omega/2pi = 64 kHz against mu/2pi = 3.01 MHz: f = x^2/8
    assert budget["ms11.bessel_saturation"] == pytest.approx(math.sin(math.pi * f / 2.0) ** 2, rel=1e-9)
    assert budget["ms11.bessel_saturation"] == pytest.approx(roos_bessel_saturation(gate.waveform), rel=1e-12)
    assert budget["ms11.bessel_saturation"] < 1e-5, "far inside the row's 1.1e-4, as the row's sum requires"
    ms_terms = (
        budget["ms11.residual_displacement"]
        + budget["ms11.debye_waller"]
        + budget["ms11.carrier_scale"]
        + budget["ms11.bessel_saturation"]
        + budget["ms11.beat_phase_tilt"]
    )
    assert budget["total"] >= ms_terms * (1.0 - 1e-12), "the total carries the five summed MS terms"
    assert budget["total"] < budget["ms11.sideband_lamb_dicke_deficit"], (
        "the Lamb-Dicke deficit (3e-2 at the cap's top) is reported, not summed"
    )


@pytest.mark.slow
def test_the_same_identity_reaches_1e_6_with_lamb_dicke_order_and_rwa_on() -> None:
    """The row's companion clause: with ``lamb_dicke_order`` and ``rwa`` on, the simulated generator IS the ideal MS
    generator, so the same pulse reproduces the native matrix to 2.3e-10 - three orders inside the row's 1e-6. The
    contrast with the exact play (4.66e-5) is what the clause "to 1e-6 ONLY with lamb_dicke_order and rwa on" asserts."""
    device, drives, _modes, waveform0, space = _single_mode_fixture()
    rabi, _stark = derived_seeds(device, drives)
    builder = BuilderOptions(lamb_dicke_order=1, rwa=True, frame="interaction", include_stark=False)
    run = calibrate_entangling_angle(
        device,
        waveform0,
        (0, 1),
        drives,
        table_with_waveform((0, 1), waveform0, rabi_hz=rabi),
        space=space,
        tolerance_rad=1e-7,
        builder_options=builder,
        max_iterations=8,
    )
    assert run.converged
    table = table_with_waveform((0, 1), run.waveform, rabi_hz=rabi)
    sched = _played(device, drives, run.waveform, table, builder=builder)
    traces = JointExactEngine(table=table, builder_options=builder).run_pulses(
        device,
        sched,
        space.initial_state([0, 0]),
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    target = qt.Qobj((native.ms(0.0, 0.0, math.pi / 2) @ KET00).reshape(-1, 1), dims=[[2, 2], [1, 1]])
    infidelity = 1.0 - float(np.real(qt.expect(traces.final.internal, target)))
    assert infidelity < 1e-6, infidelity
    assert infidelity == pytest.approx(2.32e-10, rel=0.2), infidelity


def test_sideband_lamb_dicke_deficit_against_eta_sqrt_n_plus_one() -> None:
    """The reported (not summed) Lamb-Dicke deficit: f = 1 - |<n+1|D(i eta)|n>|/(eta sqrt(n+1)) (Wineland 1998 Eq. 18),
    the worst case over the gate's ions and its carried modes at the highest Fock index each carries; zero at eta -> 0 and
    growing with eta and n."""
    from qutip_trap.hilbert.operators import rabi_matrix_element
    from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
    from qutip_trap.run.space import SpaceSelection

    def _selection(d: int) -> SpaceSelection:
        space = HilbertSpace((2, 2), (ModeTruncation(GATE_MODE, d, (0, d - 1), 0.1),), None, ())
        return SpaceSelection(
            space=space,
            mode_class={GATE_MODE: "resolved"},
            contribution={},
            nbar={GATE_MODE: 0.0},
        )

    def _modes(eta: float) -> GateModes:
        return GateModes(
            ions=(0, 1),
            modes=(GATE_MODE,),
            omega_rad_s=(TWO_PI * 3.0e6,),
            eta={0: (eta,), 1: (eta,)},
            nbar=(0.0,),
        )

    # the closed form at the top of the carried range
    for eta, d in ((0.08, 12), (0.05, 8), (0.15, 6)):
        n = d - 1
        want = abs(1.0 - rabi_matrix_element(n + 1, n, eta) / (eta * math.sqrt(n + 1.0)))
        assert sideband_lamb_dicke_deficit(_modes(eta), _selection(d)) == pytest.approx(want, rel=1e-12)
    # monotone in eta at a fixed cap, and vanishing as eta -> 0 (the linear-in-eta limit of Section 4.3.1)
    values = [sideband_lamb_dicke_deficit(_modes(e), _selection(12)) for e in (1e-4, 0.01, 0.05, 0.08, 0.15)]
    assert values == sorted(values)
    assert values[0] < 1e-7 and values[-1] > 0.1
    # a mode with eta = 0 contributes nothing (no force to saturate)
    assert sideband_lamb_dicke_deficit(_modes(0.0), _selection(12)) == 0.0
