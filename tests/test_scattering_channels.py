"""Photon-scattering collapse operators with recoil and leakage, and the D-level branching of the scattering budget
(PLAN.md Section 6.5)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import Schedule
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.raman import derive_raman_drive, lamb_dicke_parameters, scattering_budget, square_drive
from qutip_trap.light.recoil import angular_factor
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.noise.scattering import (
    ScatteringOptions,
    intensity_scale,
    internal_levels,
    nominal_rabi_hz,
    recoil_nodes,
    scattering_channels,
    scattering_estimates,
)
from qutip_trap.options import Numerics
from qutip_trap.readout.fluorescence import ReadoutScheme
from qutip_trap.species import species
from qutip_trap.trap.mathieu import UnstableMathieuError
from qutip_trap.trap.pseudopotential import RfDrive
from tests.fixtures import single_ion_raman_device


def _pulse(dev, duration_s=50e-6, scale=1.0):
    der = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    drive = square_drive(der, include_stark=False, rabi_scale=scale)
    return Pulse(drive, 0.0, duration_s, "p", ()), der


def test_recoil_nodes_are_exact_in_the_first_and_second_moments_for_every_pattern() -> None:
    b_hat = (0.3, 0.4, math.sqrt(1 - 0.25))
    for q in (-1, 0, 1):
        for recoil in ("minimal", "vector"):
            nodes = recoil_nodes(q, b_hat, ScatteringOptions(recoil=recoil))
            w = np.array([n.weight for n in nodes])
            k = np.array([n.k_hat for n in nodes])
            assert w.sum() == pytest.approx(1.0, abs=1e-12)
            assert np.allclose((w[:, None] * k).sum(axis=0), 0.0, atol=1e-12)
            for axis in (np.array(b_hat), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.6, 0.8])):
                cos_chi = float(np.dot(axis, b_hat))
                assert float(np.sum(w * (k @ axis) ** 2)) == pytest.approx(
                    angular_factor(q, cos_chi), abs=1e-12
                )
    off = recoil_nodes(1, b_hat, ScatteringOptions(recoil="off"))
    assert len(off) == 1 and off[0].weight == 1.0 and not np.any(off[0].k_hat)


def test_operator_rates_sum_to_the_amplitude_budget_at_the_played_intensity() -> None:
    """sum_L <a|L^dag L|a> is the atomic layer's Rayleigh + Raman rate out of each qubit level times the played
    intensity scale (1e-9), and at d = 2 the leakage is reported, not built."""
    dev = single_ion_raman_device()
    pulse, der = _pulse(dev, scale=3.0)
    space = HilbertSpace((2,), (ModeTruncation(1, 8, (0, 2), 0.25),), None, (0, 2))
    ops, notes = scattering_channels(dev, pulse, space)
    assert len(ops) > 0 and any("leakage" in n for n in notes)
    budget = scattering_budget(dev, 0, (0, 1))
    lower, upper = dev.crystal.species[0].qubit
    for level, label in ((0, lower), (1, upper)):
        st = space.initial_state([level])
        total = sum(float(qt.expect(o.op.dag() * o.op, st.joint).real) for o in ops)
        expected = 3.0 * (budget.rayleigh_per_s[label] + budget.raman_spin_flip_per_s[label])
        assert total == pytest.approx(expected, rel=1e-9)
    est = scattering_estimates(dev, pulse)
    assert est["ion0.P_raman"] == pytest.approx(
        3.0
        * pulse.duration_s
        * 0.5
        * (budget.raman_spin_flip_per_s[lower] + budget.raman_spin_flip_per_s[upper])
    )
    assert est["ion0.rayleigh_dephasing"] < 1e-9, (
        "clock states far from the fine structure: Gamma_el vanishes (Section 4.5.5)"
    )


def test_the_scattering_scale_raises_on_a_drive_the_light_layer_cannot_derive() -> None:
    """The played envelope scales the scattering rates by (|Omega|/Omega_nom)^p with Omega_nom the derived carrier Rabi
    frequency (sqrt 3 for a Raman pulse at three times it); an rf record the derived drive cannot evaluate (5 MHz under
    the 3 MHz radial frequency) raises instead of taking the rates at the beams' configured power."""
    dev = single_ion_raman_device(rf=RfDrive(100.0, 30e6))
    pulse, der = _pulse(dev, scale=3.0)
    assert nominal_rabi_hz(dev, 0, pulse) == der.carrier_rabi_hz > 0.0
    const, varying, note = intensity_scale(dev, pulse)
    assert const == pytest.approx(math.sqrt(3.0), rel=1e-12) and varying is None and note is None
    unstable = dataclasses.replace(dev, trap=dataclasses.replace(dev.trap, rf=RfDrive(100.0, 5e6)))
    with pytest.raises(UnstableMathieuError):
        intensity_scale(unstable, pulse)
    with pytest.raises(UnstableMathieuError):
        scattering_estimates(unstable, pulse)


def test_raman_flip_rate_and_recoil_heating_per_photon_in_mesolve() -> None:
    """Under the channels alone |0> flips at Gamma_{0 -> 1} (2 %) and the mode heats by eta_abs^2 + alpha eta_em^2
    quanta per scattered photon (25 %)."""
    dev = single_ion_raman_device()
    pulse, der = _pulse(dev, duration_s=1.0, scale=50.0)
    space = HilbertSpace((2,), (ModeTruncation(1, 10, (0, 3), 0.25),), None, (0, 2))
    ops, _ = scattering_channels(dev, pulse, space, options=ScatteringOptions(recoil="minimal"))
    budget = scattering_budget(dev, 0, (0, 1))
    lower = dev.crystal.species[0].qubit[0]
    rate_flip = 50.0 * budget.raman_spin_flip_per_s[lower]
    rate_tot = 50.0 * (budget.raman_spin_flip_per_s[lower] + budget.rayleigh_per_s[lower])
    t = 0.02 / rate_tot
    st = space.initial_state([0])
    h = 0.0 * space.identity()
    res = qt.mesolve(
        h, st.joint, [0.0, t], [o.op for o in ops], e_ops=[space.projector(0, 1), space.number(1)]
    )
    assert res.expect[0][-1] == pytest.approx(rate_flip * t, rel=0.02)
    # recoil: per photon, eta_abs^2 (single-beam absorption) + alpha eta_em^2 (emission), summed over the two beams' share of the rate
    b_hat = np.array(dev.field.direction)
    e_hat = np.array(dev.crystal.modes[1].e_hat)
    quanta = 0.0
    for b in (0, 1):
        beam = dev.beams[b]
        etas, _ = lamb_dicke_parameters(dev, 0, beam.k_vector())
        eta_em = (
            beam.k_rad_per_m
            * math.sqrt(1.054571817e-34 / (2 * dev.crystal.masses_kg[0] * dev.crystal.modes[1].omega_rad_s))
            * abs(dev.crystal.modes[1].eigenvector[0])
        )
        # the emission pattern mixes q' = -1, 0, +1 by the branching; alpha is bracketed by the pi and sigma factors along e_hat
        cos_chi = float(np.dot(e_hat, b_hat))
        share = 0.5  # equal beams
        quanta += share * (
            etas[1] ** 2 + 0.5 * (angular_factor(0, cos_chi) + angular_factor(1, cos_chi)) * eta_em**2
        )
    heated = res.expect[1][-1]
    photons = rate_tot * t
    assert heated / photons == pytest.approx(quanta, rel=0.25)
    assert heated > 0.0


def test_leakage_populates_the_sink_at_d_equals_3_and_is_read_dark() -> None:
    dev = single_ion_raman_device()
    pulse, _ = _pulse(dev, duration_s=1.0, scale=50.0)
    space = HilbertSpace((3,), (), None, (0, 1, 2))
    lev = internal_levels(dev.crystal.species[0], 3, dev.field.B_gauss, dev.field.direction)
    ops, notes = scattering_channels(
        dev, pulse, space, levels_by_ion={0: lev}, options=ScatteringOptions(recoil="off")
    )
    assert not any("leakage" in n and "not simulated" in n for n in notes)
    assert any("scatter_leak" in o.channel for o in ops)
    budget = scattering_budget(dev, 0, (0, 1))
    lower = dev.crystal.species[0].qubit[0]
    rate_leak = 50.0 * budget.leakage_per_s[lower]
    t = 0.02 / (50.0 * sum(budget.leakage_per_s.values()))
    st = space.initial_state([0])
    res = qt.mesolve(
        0.0 * space.identity(), st.joint, [0.0, t], [o.op for o in ops], e_ops=[space.projector(0, 2)]
    )
    assert res.expect[0][-1] == pytest.approx(rate_leak * t, rel=0.02)

    yb = species("171Yb+")
    scheme = ReadoutScheme.for_species(
        yb, ["S1/2 F=1 mF=0", "S1/2 F=1 mF=-1", "S1/2 F=1 mF=1"], labels=lev.labels
    )
    assert scheme.classes == ("dark", "bright", "dark") and scheme.n_levels == 3
    lev5 = internal_levels(yb, 5, 5.0, (1.0, 0.0, 0.0))
    scheme5 = ReadoutScheme.for_species(
        yb, ["S1/2 F=1 mF=0", "S1/2 F=1 mF=-1", "S1/2 F=1 mF=1"], labels=lev5.labels
    )
    assert scheme5.classes == ("dark", "bright", "bright", "bright", "dark"), (
        "leaked F = 1 sublevels are bright (Section 8.1)"
    )


def test_engine_builds_the_channels_per_segment_and_a_shaped_pulse_gets_a_time_dependent_operator() -> None:
    dev = single_ion_raman_device()
    pulse, der = _pulse(dev, duration_s=20e-6)
    tone = dataclasses.replace(
        pulse.drive.tones[0], envelope_hz=lambda tau: der.carrier_rabi_hz * math.sin(math.pi * tau / 20e-6)
    )
    shaped = Pulse(dataclasses.replace(pulse.drive, tones=(tone,)), 0.0, 20e-6, "shaped", ())
    space = HilbertSpace((2,), (ModeTruncation(1, 8, (0, 2), 0.25),), None, (0, 2))
    ops, _ = scattering_channels(dev, shaped, space, options=ScatteringOptions(recoil="off"))
    assert all(o.time_dependent for o in ops)
    mid = ops[0].op(10e-6)
    start = ops[0].op(0.0)
    assert mid.norm() > 100 * max(start.norm(), 1e-30)
    eng = JointExactEngine(
        device_channels=True, hardware_chain=False, scattering_channels=True, scattering_recoil="off"
    )
    st = space.initial_state([0])
    tr = eng.run_pulses(
        dev,
        Schedule((pulse,), (), (), {0: 0.0}),
        st,
        space,
        quiet_sample(),
        SeedSpec(0),
        Numerics(mesolve_dimension_max=4096),
    )
    rep = eng.last_report
    assert (
        rep is not None
        and rep.method == "mesolve"
        and set(rep.channel_names) >= {"scatter_rayleigh", "scatter_raman"}
    )
    assert rep.segments[0].n_collapse_ops > 0 and tr.final.joint is not None and tr.final.joint.isoper
