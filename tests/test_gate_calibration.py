"""The exact-spot-check calibration of the entangling angle, the thermal robustness curve and the ms/parity scans (PLAN.md
Section 7.5)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.calibration.entangling import (
    GateCheck,
    calibrate_entangling_angle,
    exact_gate_check,
    gate_space,
    ms_schedule,
    thermal_robustness,
)
from qutip_trap.control.shaping import (
    CHI_MAXIMAL_RAD,
    closure_duration_s,
    closure_rabi_rad_s,
    solve_amplitude_modulation,
    solve_fourier_amplitude_modulation,
    solve_frequency_modulation,
    solve_phase_modulation,
    waveform_integrals,
)
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.hamiltonian import BuilderOptions
from qutip_trap.experiments.entangling import ms_scan, parity_scan
from qutip_trap.machine import Machine
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.units import TWO_PI
from qutip_trap.validation.two_qubit_closed_forms import (
    ballance_thermal_error,
    thermal_debye_waller_infidelity,
)
from tests.m4_fixtures import (
    X_COM_TWO_IONS,
    derived_seeds,
    raman_gate_drives,
    table_with_waveform,
    two_ion_device,
    two_ion_modes,
)

RABI, STARK = derived_seeds(two_ion_device(), raman_gate_drives(2))
"""The derived carrier Rabi frequencies and Stark shifts a perfectly calibrated table carries (the played chain is then the
identity)."""


def test_surrogate_plus_exact_spot_check_converges_to_pi_over_four() -> None:
    """The closed-form waveform is 1 to 3% strong in chi (Debye-Waller, carrier); one or two exact checks correct it."""
    dev = two_ion_device()
    modes = two_ion_modes(dev)
    drives = raman_gate_drives(2)
    am = solve_amplitude_modulation(modes, mu_hz=2.914e6, duration_s=100e-6)
    ints = waveform_integrals(am.waveform, modes)
    assert (
        ints.chi_of(0, 1) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-9) and ints.residual_error(modes) < 1e-18
    )
    space = gate_space(modes, 2, waveform=am.waveform)
    table = table_with_waveform((0, 1), am.waveform, rabi_hz=RABI, stark_hz=STARK)
    run = calibrate_entangling_angle(dev, am.waveform, (0, 1), drives, table, space=space, tolerance_rad=1e-4)
    assert run.converged and len(run.checks) <= 3
    assert 0.005 < run.surrogate_error < 0.03
    assert run.checks[-1].chi_rad == pytest.approx(CHI_MAXIMAL_RAD, abs=1e-4)
    assert run.checks[-1].fidelity > 0.9995 and run.checks[-1].leakage < 2e-4
    assert run.waveform.phi_s.status == "calibrated" and run.waveform.phi_s.experiment == "exact_spot_check"
    assert 1.0 < run.factors[-1] < 1.02
    # the corrected waveform replays to the same angle
    again, _ = exact_gate_check(
        dev,
        run.waveform,
        (0, 1),
        drives,
        table_with_waveform((0, 1), run.waveform, rabi_hz=RABI, stark_hz=STARK),
        space=space,
    )
    assert again.chi_rad == pytest.approx(CHI_MAXIMAL_RAD, abs=2e-4)


def test_thermal_robustness_curve_follows_the_n0_referenced_debye_waller_law() -> None:
    """The thermal robustness curve on the gate mode with the exact sidebands. Ballance's eps_nbar = (pi^2/4) eta^4 nbar (2 nbar + 1)
    is the ENTANGLING-ANGLE error of an n = 0-calibrated gate, chi(n) = chi_0 [1 - eta^2 (2n + 1)]: read from the populations
    (P_11/(P_00 + P_11) = sin^2 chi_eff) it is recovered within 10%, and the re-optimized nbar^2 + nbar reference is excluded. The
    exact gate loses more: the n-dependent sideband coupling leaves residual spin-motion entanglement (leakage into |du>, |ud>) of
    the same order, a term linear in nbar, so the Bell-state infidelity lies between 1.1 and 1.6 times Ballance's form at nbar <= 2."""
    dev = two_ion_device()
    modes = two_ion_modes(dev).subset([X_COM_TWO_IONS])
    drives = raman_gate_drives(2)
    wf0 = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=20e3, kernel="rwa")
    opts = BuilderOptions(frame="interaction", rwa=True, frozen_debye_waller=False)
    space0 = gate_space(modes, 2, waveform=wf0)
    table = table_with_waveform((0, 1), wf0, rabi_hz=RABI, stark_hz=STARK)
    run = calibrate_entangling_angle(
        dev, wf0, (0, 1), drives, table, space=space0, builder_options=opts, tolerance_rad=1e-5
    )
    assert run.converged
    wf = run.waveform
    eta = abs(modes.eta[0][0])
    curve = thermal_robustness(
        dev,
        wf,
        (0, 1),
        drives,
        table_with_waveform((0, 1), wf, rabi_hz=RABI, stark_hz=STARK),
        modes=modes,
        gate_mode=X_COM_TWO_IONS,
        nbars=(0.0, 0.5, 1.0, 2.0),
        builder_options=opts,
    )
    base = 1.0 - curve[0][1].fidelity
    assert base < 2e-4, "the exact sidebands' n-dependence leaves a small residual even at n = 0"
    # Fock-resolved: chi_n from the populations of a pure |n> input (the leakage renormalized away) follows the Debye-Waller law, and the
    # leakage (residual spin-motion entanglement) grows linearly with n
    chis: dict[int, float] = {}
    leaks: dict[int, float] = {}
    for n in range(4):
        space_n = gate_space(modes, 2, waveform=wf, extra_levels=n + 2)
        state = space_n.initial_state([0, 0], fock={X_COM_TWO_IONS: n})
        check_n = _fock_check(dev, wf, drives, space_n, state, opts)
        p00, p11 = check_n.populations["P00"], check_n.populations["P11"]
        chis[n] = math.asin(math.sqrt(p11 / (p00 + p11)))
        leaks[n] = check_n.leakage
    for n in (1, 2, 3):
        assert (chis[n] - chis[0]) / (-(math.pi / 4) * 2.0 * eta**2 * n) == pytest.approx(1.0, abs=0.1), (
            n,
            chis,
        )
    assert leaks[2] / leaks[1] == pytest.approx(2.0, rel=0.3) and leaks[3] / leaks[2] == pytest.approx(
        1.5, rel=0.3
    )
    # the angle law thermally averaged: with the measured slope a = d chi/dn the loss sum_n P_n sin^2(a n) is Ballance's form
    slope = sum((chis[n] - chis[0]) * n for n in (1, 2, 3)) / sum(n * n for n in (1, 2, 3))
    ratios: list[float] = []
    for nb, check in curve[1:]:
        weights = np.array([nb**n / (1.0 + nb) ** (n + 1) for n in range(80)])
        angle_loss = float(np.sum(weights * np.sin(slope * np.arange(80)) ** 2))
        expected = ballance_thermal_error(eta, nb)
        assert angle_loss == pytest.approx(expected, rel=0.2), (nb, angle_loss, expected)
        assert abs(expected / thermal_debye_waller_infidelity(eta, nb, "mean") - 1.0) > 0.25
        total = (1.0 - check.fidelity) - base
        ratios.append(total / expected)
        assert 1.02 < total / expected < 1.7, (nb, total, expected)
    assert ratios == sorted(ratios, reverse=True), (
        "the residual-entanglement term is linear in n: its share shrinks with nbar"
    )
    losses = [(1.0 - c.fidelity) - base for _, c in curve]
    assert losses == sorted(losses)


def _fock_check(dev, wf, drives, space, state, opts):  # type: ignore[no-untyped-def]
    """The exact gate from an explicit joint state (a Fock input): populations, chi and leakage as exact_gate_check reads them."""
    sched = ms_schedule(wf, (0, 1), drives, table_with_waveform((0, 1), wf, rabi_hz=RABI, stark_hz=STARK))
    tr = JointExactEngine(builder_options=opts).run_pulses(
        dev, sched, state, space, quiet_sample(), SeedSpec(0), SolverOptions()
    )
    rho = tr.final.internal.full()
    pops = {f"P{a}{b}": float(np.real(rho[2 * a + b, 2 * a + b])) for a in range(2) for b in range(2)}
    chi = math.asin(math.sqrt(min(max(pops["P11"], 0.0), 1.0)))
    return GateCheck(pops, chi, pops["P01"] + pops["P10"], {}, 0.0, tr.final.internal, None)


@pytest.mark.slow
@pytest.mark.parametrize("family", ["fm", "fourier", "pm"])
def test_the_fm_fourier_and_pm_solutions_are_verified_by_exact_integration(family: str) -> None:
    """Every solver's solution verified by exact integration of the full Hamiltonian: exact chi/surrogate = 0.9847 (FM),
    0.9831 (Fourier AM) and 0.9528 (PM), leakage 4.7e-4, 4.7e-7 and 1.8e-2 on the two-ion fixture. The PM family's larger
    leakage is Roos's spin-axis tilt psi = (2 Omega/mu) sin phi_m: the motion phase IS its modulation parameter, so it cannot
    use the tilt-free phi_m = 0 convention."""
    dev = two_ion_device()
    modes = two_ion_modes(dev)
    drives = raman_gate_drives(2)
    if family == "fm":
        sp = solve_frequency_modulation(modes, duration_s=100e-6, n_vertices=9, mu0_hz=2.914e6)
        ratio, leak = 0.9847, 1e-3
    elif family == "fourier":
        sp = solve_fourier_amplitude_modulation(
            modes, mu_hz=2.914e6, duration_s=100e-6, n_basis=16, stabilization_order=1
        )
        ratio, leak = 0.9831, 1e-5
    else:
        sp = solve_phase_modulation(modes, mu_hz=2.914e6, duration_s=100e-6)
        ratio, leak = 0.9528, 3e-2
    space = gate_space(modes, 2, waveform=sp.waveform)
    check, tr = exact_gate_check(
        dev,
        sp.waveform,
        (0, 1),
        drives,
        table_with_waveform((0, 1), sp.waveform, rabi_hz=RABI, stark_hz=STARK),
        space=space,
    )
    assert check.chi_rad / sp.chi_rad == pytest.approx(ratio, rel=0.01), (family, check.chi_rad)
    assert check.leakage < leak, (family, check.leakage)
    assert max(tr.boundary_population.values()) < 1e-10, (family, tr.boundary_population)
    assert check.fidelity > 1.0 - 3.0 * leak - 0.05


def test_kirchmair_forty_calcium_consistency_anchors() -> None:
    """Kirchmair et al. 2009's 40Ca+ apparatus (nu/2pi = 1.232 MHz, eta = 0.044): t_g = 50 us is eps/2pi = 20 kHz at K = 1
    and needs Omega/2pi = 227.3 kHz, t_g = 25 us is eps/2pi = 40 kHz and needs 454.5 kHz - Omega/nu = 0.37, deep into
    Roos's carrier saturation. The intrinsic residual-displacement and Debye-Waller terms at nbar = 0 are below 1e-5,
    three orders under the paper's measured 1 - F = 7e-3 and 2.9e-2, which its technical terms dominate."""
    eta, nu = 0.044, TWO_PI * 1.232e6
    for t_g, eps_hz, omega_khz in ((50e-6, 20e3, 227.3), (25e-6, 40e3, 454.5)):
        eps = TWO_PI * eps_hz
        assert closure_duration_s(eps, 1) == pytest.approx(t_g, rel=1e-12)
        assert closure_rabi_rad_s(eta, eps, 1) / TWO_PI == pytest.approx(omega_khz * 1e3, rel=1e-3)
    assert closure_rabi_rad_s(eta, TWO_PI * 40e3, 1) / nu == pytest.approx(0.369, rel=1e-2)
    # the intrinsic terms at nbar = 0 are orders below the measured infidelities
    assert ballance_thermal_error(eta, 0.0) == 0.0
    assert thermal_debye_waller_infidelity(eta, 0.0, "minus_half") < 1e-5


def test_ms_scan_finds_the_closure_amplitude_and_parity_scan_the_contrast() -> None:
    """The population crossing P_00 = P_11 sits at the calibrated amplitude; the parity oscillates at 2 phi
    with a contrast near one, and F = (P_00 + P_11 + C)/2 bounds the Bell fidelity."""
    dev = two_ion_device()
    modes = two_ion_modes(dev)
    drives = raman_gate_drives(2)
    # the experiments read the roles
    dev = dataclasses.replace(dev, roles=dataclasses.replace(dev.roles, gate=drives))
    am = solve_amplitude_modulation(modes, mu_hz=2.914e6, duration_s=100e-6)
    space = gate_space(modes, 2, waveform=am.waveform)
    table0 = table_with_waveform((0, 1), am.waveform, rabi_hz=RABI, stark_hz=STARK)
    run = calibrate_entangling_angle(
        dev, am.waveform, (0, 1), drives, table0, space=space, tolerance_rad=2e-4
    )
    table = table_with_waveform((0, 1), run.waveform, rabi_hz=RABI, stark_hz=STARK)
    scan = ms_scan(
        Machine(dev), (0, 1), (0.9, 0.97, 1.03, 1.1), (0.0,), table=table, space=space, modes=modes
    )
    assert scan.data.shape == (4, 5) and scan.model == "ms_population_scan"
    assert scan.fitted["closure_scale"][0] == pytest.approx(1.0, abs=0.01)
    p11 = scan.data[:, 4]
    assert np.all(np.diff(p11) > 0), "P_11 = sin^2 chi rises with the amplitude below pi/4"
    par = parity_scan(
        Machine(dev),
        (0, 1),
        np.linspace(0.0, math.pi, 6, endpoint=False),
        table=table,
        space=space,
        modes=modes,
    )
    assert par.model == "parity_oscillation" and par.data.shape == (6, 4)
    assert par.fitted["contrast"][0] > 0.99
    assert par.fitted["bell_fidelity_bound"][0] > 0.99
    assert par.fitted["phi0_rad"][1] < 0.05
    with pytest.raises(ValueError, match="waveform"):
        ms_scan(Machine(dev), (0, 1), (1.0,), (0.0,), table=dataclasses.replace(table, ms={}))
