"""The control hardware chain (PLAN.md Section 7.10): quantized tone words, the modulator's first-order response with its
tail, per-train timing jitter, and the beat-phase reference the response demands."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.calibration.entangling import exact_gate_check, spot_check_space
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.hardware import TAIL_TIME_CONSTANTS, _trains, apply_hardware_chain
from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import Schedule, response_phase_rad
from qutip_trap.control.shaping import gate_modes
from qutip_trap.device.presets import ideal_hardware, yb171_chain
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.microwave import square_microwave_drive
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.options import Numerics
from tests.fixtures import REALISTIC_HARDWARE, single_ion_raman_device


def _device():
    """The single-ion fixture with the realistic 16-bit / 14-bit / 50 ns chain."""
    dev = single_ion_raman_device()
    return dataclasses.replace(dev, hardware=REALISTIC_HARDWARE)


def _pi_schedule(dev):
    der = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    drive = square_drive(der, include_stark=False)
    t_pi = der.pi_time_s()
    return Schedule((Pulse(drive, 0.0, t_pi, "pi", ()),), (), (), {0: 0.0}), t_pi


def test_hardware_record_validation_resolutions_and_description() -> None:
    hw = REALISTIC_HARDWARE
    assert ideal_hardware().aom_rise_s == 0.0 and ideal_hardware().phase_resolution_rad < 2e-9
    assert (
        hw.phase_resolution_rad == pytest.approx(2 * math.pi / 65536) and hw.frequency_resolution_hz is None
    )
    assert hw.amplitude_step_hz(1e5) == pytest.approx(1e5 / 16384) and not hw.ideal
    assert not dataclasses.replace(hw, aom_rise_s=0.0).ideal, (
        "a 100 MHz amplifier still shapes microwave envelopes"
    )
    ideal = dataclasses.replace(hw, aom_rise_s=0.0, amplifier_bandwidth_hz=math.inf)
    assert ideal.ideal and any("ideal modulator" in s for s in ideal.describe())
    assert any("infinite amplifier bandwidth" in s for s in ideal.describe())
    assert (
        hw.response_time_s("microwave") == pytest.approx(1.0 / (2 * math.pi * 1e8))
        and hw.response_time_s("raman") == 50e-9
    )
    with_clock = dataclasses.replace(hw, dds_clock_hz=1e9, dds_frequency_bits=32)
    assert with_clock.frequency_resolution_hz == pytest.approx(1e9 / 2**32) and any(
        "0.233 Hz" in s for s in with_clock.describe()
    )
    with pytest.raises(ValueError):
        dataclasses.replace(hw, dds_clock_hz=1e9)
    with pytest.raises(ValueError):
        dataclasses.replace(hw, timing_jitter_s=-1.0)


def test_response_filters_the_envelope_preserving_the_area_and_adds_the_tail() -> None:
    """The first-order modulator response makes a square envelope rise as 1 - e^{-t/tau} (to 1e-3) with a tail carrying the
    missing area, so the pi pulse still flips the ion to 2e-5."""
    dev = _device()
    sched, t_pi = _pi_schedule(dev)
    hw = dev.hardware
    played, notes = apply_hardware_chain(sched, hw)
    assert len(played.pulses) == 2 and played.pulses[1].gate_id == "pi/tail"
    assert played.pulses[1].t_end_s == pytest.approx(t_pi + TAIL_TIME_CONSTANTS * hw.aom_rise_s)
    env = played.pulses[0].drive.tones[0].envelope_hz
    assert isinstance(env, np.ndarray) and env[0] == 0.0
    grid = np.linspace(0.0, t_pi, env.size)
    k = int(env.size * 0.5)
    target = sched.pulses[0].drive.tones[0].envelope_hz
    assert env[k] == pytest.approx(target * (1 - math.exp(-grid[k] / hw.aom_rise_s)), rel=1e-3)
    tail = played.pulses[1].drive.tones[0].envelope_hz
    area = np.trapezoid(env, grid) + np.trapezoid(
        tail, np.linspace(0.0, TAIL_TIME_CONSTANTS * hw.aom_rise_s, tail.size)
    )
    assert area == pytest.approx(
        target * t_pi, rel=math.exp(-TAIL_TIME_CONSTANTS) * hw.aom_rise_s / t_pi + 2e-5
    )
    assert any("low-pass" in n for n in notes)
    space = HilbertSpace((2,), (ModeTruncation(1, 8, (0, 2), 0.2),), None, (0, 2))
    st = space.initial_state([0])
    p1 = {}
    for flag in (False, True):
        eng = JointExactEngine(hardware_chain=flag)
        tr = eng.run_pulses(dev, sched, st, space, quiet_sample(), SeedSpec(0), Numerics())
        p1[flag] = float(tr.expectations["P1[0]"][-1])
        rep = eng.last_report
        assert rep is not None and bool(rep.hardware_notes) == flag
    assert p1[True] == pytest.approx(p1[False], abs=2e-5)
    quant_only, _ = apply_hardware_chain(sched, dataclasses.replace(hw, aom_rise_s=0.0))
    assert len(quant_only.pulses) == 1 and isinstance(quant_only.pulses[0].drive.tones[0].envelope_hz, float)


def test_quantization_rounds_frequency_phase_and_amplitude_words() -> None:
    dev = _device()
    sched, _ = _pi_schedule(dev)
    tone = sched.pulses[0].drive.tones[0]
    drive = dataclasses.replace(
        sched.pulses[0].drive, tones=(dataclasses.replace(tone, detuning_hz=1234.5678, phase_rad=0.1234567),)
    )
    sched2 = Schedule((dataclasses.replace(sched.pulses[0], drive=drive),), (), (), {0: 0.0})
    hw = dataclasses.replace(
        dev.hardware,
        aom_rise_s=0.0,
        dds_clock_hz=1e9,
        dds_frequency_bits=32,
        dds_phase_bits=14,
        amplitude_full_scale_hz=2e5,
    )
    played, notes = apply_hardware_chain(sched2, hw)
    t = played.pulses[0].drive.tones[0]
    f_res = 1e9 / 2**32
    assert t.detuning_hz == pytest.approx(round(1234.5678 / f_res) * f_res) and t.detuning_hz != 1234.5678
    assert t.phase_rad == pytest.approx(round(0.1234567 / (2 * math.pi / 2**14)) * 2 * math.pi / 2**14)
    step = 2e5 / 2**14
    assert isinstance(t.envelope_hz, float) and t.envelope_hz == pytest.approx(
        round(tone.envelope_hz / step) * step
    )
    assert any("frequencies to" in n for n in notes)
    sat = dataclasses.replace(hw, amplifier_saturation_hz=tone.envelope_hz)
    played_sat, _ = apply_hardware_chain(sched2, sat)
    assert played_sat.pulses[0].drive.tones[0].envelope_hz == pytest.approx(
        round(tone.envelope_hz * math.tanh(1.0) / step) * step, rel=1e-3
    )


def test_trains_are_contiguous_same_ion_pulses_and_jitter_moves_a_train_rigidly() -> None:
    dev = _device()
    der = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    drive = square_drive(der, include_stark=False)
    pulses = (
        Pulse(drive, 0.0, 10e-6, "a/seg0", ()),
        Pulse(drive, 10e-6, 20e-6, "a/seg1", ()),
        Pulse(drive, 25e-6, 30e-6, "b", ()),
    )
    sched = Schedule(pulses, ((20e-6, 25e-6),), (), {0: 0.0})
    trains = _trains(pulses)
    assert [[p.gate_id for p in t] for t in trains] == [["a/seg0", "a/seg1"], ["b"]]
    hw = dataclasses.replace(dev.hardware, aom_rise_s=0.0, timing_jitter_s=1e-7)
    played, notes = apply_hardware_chain(sched, hw, rng=np.random.default_rng(3))
    a0, a1 = played.pulses[0], played.pulses[1]
    assert played.pulses[2].gate_id == "b"
    assert a1.t_start_s == pytest.approx(a0.t_end_s) and a0.t_start_s != 0.0
    assert (a0.t_start_s - 0.0) == pytest.approx(a1.t_start_s - 10e-6)
    assert any("jittered" in n for n in notes)
    with pytest.raises(ValueError):
        apply_hardware_chain(sched, hw)
    # the response state carries across the two segments of a train: no restart from zero at the boundary
    filt, _ = apply_hardware_chain(sched, dataclasses.replace(dev.hardware, aom_rise_s=1e-6))
    env1 = filt.pulses[1].drive.tones[0].envelope_hz
    assert isinstance(env1, np.ndarray) and env1[0] > 0.99 * der.carrier_rabi_hz


def test_response_phase_reference_is_the_first_order_filter_phase_at_the_beat_note() -> None:
    """The response phase reference is sgn(mu) arctan(2 pi |mu| tau_r) per leg, zero for an ideal modulator, with a callable
    (FM) beat note evaluated."""
    assert response_phase_rad(3.0e6, 0.0) == 0.0
    assert response_phase_rad(3.0e6, 50e-9) == pytest.approx(math.atan(2 * math.pi * 3e6 * 50e-9))
    assert response_phase_rad(-3.0e6, 50e-9) == pytest.approx(-math.atan(2 * math.pi * 3e6 * 50e-9))
    assert response_phase_rad(lambda tau: 1e6, 1e-9) == pytest.approx(math.atan(2 * math.pi * 1e-3))


@pytest.mark.slow
def test_calibrated_gate_survives_the_modulator_response_with_the_phase_reference() -> None:
    """With a 50 ns modulator rise the chain plus the Roos beat-phase reference reaches F = 0.999923 (leakage 3.315e-5)
    against 0.999868 (4.485e-5) for an ideal modulator, to 5e-6 and 0.5 %."""
    fx = yb171_chain(2)

    def check(hardware, chain: bool):
        dev = dataclasses.replace(fx.device, hardware=hardware)
        sur = surrogate_table(dev, pairs=[(0, 1)], detection_records=200, detection_windows_s=(20e-6,))
        wf = sur.table.waveform_for((0, 1))
        assert wf is not None
        nb = {m: e.value for m, e in sur.table.nbar.items()}
        modes = gate_modes(dev, (0, 1), (0, 1), nbar=nb)
        space = spot_check_space(dev, modes, wf, (0, 1), Numerics())[0]
        out, _ = exact_gate_check(
            dev,
            wf,
            (0, 1),
            fx.entangling_drives,
            sur.table,
            space=space,
            hardware_chain=chain,
        )
        return out

    realistic = check(REALISTIC_HARDWARE, True)
    ideal = check(ideal_hardware(phase_continuous=True), True)
    assert realistic.fidelity == pytest.approx(0.999923, abs=5e-6), realistic.fidelity
    assert ideal.fidelity == pytest.approx(0.999868, abs=5e-6), ideal.fidelity
    assert realistic.leakage == pytest.approx(3.315e-5, rel=5e-3), realistic.leakage
    assert ideal.leakage == pytest.approx(4.485e-5, rel=5e-3), ideal.leakage
    assert realistic.fidelity > ideal.fidelity, (
        "the arctan reference over-compensates the ideal modulator's (absent) delay slightly in the ion's favour"
    )
    # an ideal modulator's response is the identity, so the chain switch cannot matter there
    assert check(ideal_hardware(phase_continuous=True), False).fidelity == pytest.approx(
        ideal.fidelity, abs=1e-9
    )


def test_stark_shift_follows_the_played_light_into_the_tail() -> None:
    """Through a 100 MHz amplifier the played Stark shift follows the intensity (Omega_played/Omega)^2 through the rise and
    decays as e^{-2t/tau} in the tail (to 1e-3); an infinite bandwidth keeps the programmed shift, and a shaped envelope is
    quantized in place with its shift following the intensity."""
    hw = dataclasses.replace(REALISTIC_HARDWARE, aom_rise_s=0.0)
    tau = hw.response_time_s("microwave")
    stark, rabi, t_end = 250.0, 50e3, 1e-6
    drive = square_microwave_drive(0, rabi, stark_shift_hz=stark)
    sched = Schedule((Pulse(drive, 0.0, t_end, "mw", ()),), (), (), {0: 0.0})
    played, _ = apply_hardware_chain(sched, hw)
    assert len(played.pulses) == 2 and played.pulses[1].gate_id == "mw/tail"
    body, tail = played.pulses
    env = body.drive.tones[0].envelope_hz
    st_body = body.drive.stark_shift_hz
    assert isinstance(env, np.ndarray) and callable(st_body)
    grid = np.linspace(0.0, t_end, env.size)
    t = grid[int(np.searchsorted(grid, 2 * tau))]  # a grid point midway through the rise
    assert st_body(float(t)) == pytest.approx(stark * (float(np.interp(t, grid, env)) / rabi) ** 2, rel=1e-9)
    assert 0.5 < st_body(float(t)) / stark < 0.95 and st_body(t_end) == pytest.approx(stark, rel=1e-9)
    tail_env = tail.drive.tones[0].envelope_hz
    st_tail = tail.drive.stark_shift_hz
    assert isinstance(tail_env, np.ndarray) and callable(st_tail)
    assert tail_env[0] == pytest.approx(rabi, rel=1e-6) and st_tail(0.0) == pytest.approx(stark, rel=1e-6)
    assert st_tail(3 * tau) == pytest.approx(stark * math.exp(-6.0), rel=1e-3)
    # infinite bandwidth: no filtering, the programmed shift survives as a float (the amplitude word rounds 50 kHz to itself)
    quant = dataclasses.replace(hw, amplifier_bandwidth_hz=math.inf)
    played_q, _ = apply_hardware_chain(sched, quant)
    assert len(played_q.pulses) == 1 and played_q.pulses[0].drive.stark_shift_hz == stark
    assert isinstance(played_q.pulses[0].drive.stark_shift_hz, float)
    # a shaped (callable) envelope is quantized in place, not resampled, and the constant shift follows its intensity
    tone = dataclasses.replace(drive.tones[0], envelope_hz=lambda tau: rabi * math.sin(math.pi * tau / t_end))
    shaped = dataclasses.replace(drive, tones=(tone,))
    played_s, _ = apply_hardware_chain(
        Schedule((Pulse(shaped, 0.0, t_end, "mw", ()),), (), (), {0: 0.0}), quant
    )
    env_s = played_s.pulses[0].drive.tones[0].envelope_hz
    st_s = played_s.pulses[0].drive.stark_shift_hz
    assert callable(env_s) and callable(st_s)
    step = quant.amplitude_step_hz(rabi)
    assert env_s(t_end / 4) == pytest.approx(round(rabi * math.sin(math.pi / 4) / step) * step)
    assert st_s(t_end / 2) == pytest.approx(stark, rel=1e-6) and st_s(t_end / 6) == pytest.approx(
        stark / 4, rel=2e-4
    )
