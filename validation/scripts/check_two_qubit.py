"""Two-qubit entangling gates (PLAN.md Sections 4.4.1, 4.4.3, 4.4.4, 4.4.7, 6.2, 9.4, 9.16, 9.17; milestone M4), through the package.

1. The check_ms_closure.py anchors through the builder: eta = 0.05, nu = 1 MHz, eps = 10 kHz, d = 16, one loop from |dd>:
   concurrence and populations at eta Omega/eps = 1/2 and 1/4 with equal tone phases, and the same pulse in Choi's sine
   beat-note convention (phi_b - phi_r = pi), whose carrier rotation tilts the spin axis by 2 Omega/mu (Roos 2008).
2. The two-mode 171Yb+ gate: the surrogate (closed-form) chi of the symmetric and the five-segment AM pulses against the exact
   two-body angle, the open spectator loop against sum |alpha|^2, and the exact spot-check calibration.
3. Ballance's motional-dephasing coefficient alpha_K from the block Liouvillians (Section 9.16 row 6-3).
4. Choi's five-ion closure counts and the AM residuals at 11 and 21 segments.
5. Baldwin's echo (Section 9.17): the printed Hamiltonian through R_x(pi) U R_-x(pi) U.
6. The Debye-Waller law and the three thermal references (Section 4.4.7 (1)); Kirchmair's 50 us / eta = 0.044 anchor.

Run: uv run python validation/scripts/check_two_qubit.py (about two minutes).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import qutip as qt

from qutip_trap.api import Beam
from qutip_trap.calibration.entangling import calibrate_entangling_angle, exact_gate_check, gate_space, ms_schedule
from qutip_trap.control.shaping import (
    SINE_MOTION_PHASE_RAD,
    gate_modes,
    solve_amplitude_modulation,
)
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.hamiltonian import BuilderOptions
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.units import TWO_PI
from qutip_trap.validation.two_qubit_closed_forms import (
    baldwin_echo_unitary,
    ballance_dephasing_coefficient,
    ballance_thermal_error,
    choi_segment_count,
    sideband_coupling_squared_difference,
    thermal_debye_waller_infidelity,
)
from tests.m4_fixtures import chain_device, raman_gate_drives, table_with_waveform, two_ion_device, two_ion_modes
from tests.test_two_qubit_gates import ANCHOR_MODE, EPS_ANCHOR_HZ, ETA_ANCHOR, anchor_device


def section(title: str) -> None:
    print(f"\n== {title}")


section("1. check_ms_closure.py anchors through the builder (eta = 0.05, nu = 1 MHz, eps = 10 kHz, d = 16)")
dev = anchor_device()
modes1 = gate_modes(dev, (0, 1), (0, 1)).subset([ANCHOR_MODE])
space1 = HilbertSpace((2, 2), (ModeTruncation(ANCHOR_MODE, 16, (0, 4), 0.1),), None, (0, 1, 2, 4, 5))
opts1 = BuilderOptions(frozen_debye_waller=False)
for ratio in (0.5, 0.25):
    for phi_m, label in ((0.0, "equal tone phases"), (SINE_MOTION_PHASE_RAD, "sine beat note, phi_b - phi_r = pi")):
        wf = Waveform.symmetric(modes1, gate_mode=ANCHOR_MODE, loops=1, epsilon_hz=EPS_ANCHOR_HZ, kernel="rwa", chi_target_rad=math.pi * ratio**2, phi_m_rad=phi_m)
        table = table_with_waveform((0, 1), wf)
        sched = ms_schedule(wf, (0, 1), raman_gate_drives(2), table)
        tr = JointExactEngine(builder_options=opts1).run_pulses(dev, sched, space1.initial_state([0, 0]), space1, quiet_sample(), SeedSpec(0), SolverOptions())
        rho = tr.final.internal
        pops = np.real(np.diag(rho.full()))
        print(f"  eta Omega/eps = {ratio} ({label}): Omega/2pi = {wf.segments[0].amplitude_hz[(0, 'blue')] / 1e3:.1f} kHz, concurrence = {qt.concurrence(rho):.4f}, populations (dd, du, ud, uu) = {np.round(pops, 4)}")
omega = TWO_PI * 0.5 * EPS_ANCHOR_HZ / ETA_ANCHOR
mu = TWO_PI * (1.0e6 - EPS_ANCHOR_HZ)
print(f"  Roos tilt for the sine convention: psi = 2 Omega/mu = {2 * omega / mu:.4f} rad, sin^2 psi = {math.sin(2 * omega / mu) ** 2:.4f} (the du + ud leakage above)")

section("2. two 171Yb+ ions, x modes 2.828 and 3.000 MHz, eta 0.081 / 0.079: surrogate against exact")
dev2 = two_ion_device()
modes2 = two_ion_modes(dev2)
drives = raman_gate_drives(2)
print(f"  modes {modes2.modes}, omega/2pi MHz {[round(w / TWO_PI / 1e6, 6) for w in modes2.omega_rad_s]}, eta {[[round(e, 6) for e in modes2.eta[i]] for i in (0, 1)]}")
wf_sym = Waveform.symmetric(modes2, gate_mode=3, loops=1, epsilon_hz=20e3)
space2 = gate_space(modes2, 2, waveform=wf_sym)
table2 = table_with_waveform((0, 1), wf_sym)
check, tr = exact_gate_check(dev2, wf_sym, (0, 1), drives, table2, space=space2)
print(f"  symmetric COM pulse (50 us, Omega/2pi = {wf_sym.segments[0].amplitude_hz[(0, 'blue')] / 1e3:.3f} kHz): surrogate chi = {wf_sym.chi_total_rad:.6f} (COM {wf_sym.chi_m[3]:.6f}, rocking {wf_sym.chi_m[2]:.6f}), exact chi = {check.chi_rad:.6f}")
print(f"    exact leakage P01 + P10 = {check.leakage:.4e}; residual quanta rocking = {check.residual_quanta[2]:.4e} against sum |alpha|^2 = {2 * abs(wf_sym.alpha_m[2]) ** 2:.4e}; COM = {check.residual_quanta[3]:.2e}; Bell fidelity {check.fidelity:.5f}")
am = solve_amplitude_modulation(modes2, mu_hz=2.914e6, duration_s=100e-6)
print(f"  five-segment AM pulse (mu = 2.914 MHz, 100 us): amplitudes/2pi kHz {[round(a / TWO_PI / 1e3, 3) for a in am.envelope.amplitude_rad_s[0]]}, chi by mode {dict((m, round(v, 6)) for (_a, _b, m), v in am.integrals.chi_by_mode.items())}, max |alpha| = {max(abs(v) for v in am.integrals.alpha.values()):.1e}")
space_am = gate_space(modes2, 2, waveform=am.waveform)
table_am = table_with_waveform((0, 1), am.waveform)
check_am, tr_am = exact_gate_check(dev2, am.waveform, (0, 1), drives, table_am, space=space_am)
print(f"    exact chi = {check_am.chi_rad:.6f} (surrogate {am.chi_rad:.6f}), leakage {check_am.leakage:.3e}, residual quanta {dict((m, f'{v:.2e}') for m, v in check_am.residual_quanta.items())}, fidelity {check_am.fidelity:.5f}, joint space dims {space_am.dims}")
run = calibrate_entangling_angle(dev2, am.waveform, (0, 1), drives, table_am, space=space_am, tolerance_rad=1e-5)
# the iteration count and the last digits of the factor depend on where the Newton step lands within the 1e-5 rad tolerance,
# which integrator round-off moves across platforms: print them at their reproducible precision
print(f"    calibration converged: {run.converged}; amplitude factor {run.factors[-1]:.4f}, exact chi {run.checks[-1].chi_rad:.5f}, fidelity {run.checks[-1].fidelity:.5f}, surrogate error {run.surrogate_error:.4f}")

section("3. Ballance's motional-dephasing coefficient alpha_K = (8K + 3)/(16 K^2) from the block Liouvillians")
d = 24
a_op = qt.destroy(d)
n_op = a_op.dag() * a_op
for loops in (1, 2, 4):
    eps = 1.0
    t_g = 2.0 * math.pi * loops / eps
    force = eps / (4.0 * math.sqrt(loops))
    tau = t_g / 1e-4
    lind = math.sqrt(2.0 / tau) * n_op
    rho0 = qt.basis(d, 0).proj()
    s_vals = [2, 0, 0, -2]
    pure = {}
    blocks = {}
    for i, s in enumerate(s_vals):
        h_s = eps * n_op + s * force * (a_op + a_op.dag())
        pure[i] = (-1j * h_s * t_g).expm() * qt.basis(d, 0)
        for j, sp in enumerate(s_vals):
            h_sp = eps * n_op + sp * force * (a_op + a_op.dag())
            lio = -1j * (qt.spre(h_s) - qt.spost(h_sp)) + qt.lindblad_dissipator(lind)
            blocks[(i, j)] = qt.vector_to_operator((lio * t_g).expm() * qt.operator_to_vector(rho0))
    th = np.angle([complex(qt.basis(d, 0).overlap(pure[i])) for i in range(4)])
    f_ent = float(np.real(sum(blocks[(i, j)].tr() * np.exp(-1j * (th[i] - th[j])) for i in range(4) for j in range(4)) / 16.0))
    print(f"  K = {loops}: (1 - F) tau/t_g at t_g/tau = 1e-4: {(1.0 - f_ent) * tau / t_g:.6f}; (8K + 3)/(16 K^2) = {ballance_dephasing_coefficient(loops):.6f}")

section("4. Choi's five-ion closure: 11 segments for one transverse family, 21 for both")
dev5 = chain_device(5, omega_hz=(3.045e6, 2.95e6, 0.55e6))
m5 = gate_modes(dev5, (1, 3), (0, 1))
print(f"  x modes/2pi MHz: {[round(w / TWO_PI / 1e6, 4) for w in m5.omega_rad_s]}; eta of ion 1: {[round(e, 4) for e in m5.eta[1]]}")
sp5 = solve_amplitude_modulation(m5, mu_hz=2.98e6, duration_s=190e-6)
print(f"  {choi_segment_count(5)} segments: chi = {sp5.chi_rad:.6f}, max |alpha| = {max(abs(v) for v in sp5.integrals.alpha.values()):.1e}, peak Omega/2pi = {sp5.diagnostics['peak_rabi_hz'] / 1e3:.1f} kHz")
c30, s30 = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
b1 = Beam(355e-9, (c30, s30, 0.0), (0.0, 0.0, 1.0), 200e-6, 10e-3, (0.0, 0.0, 0.0))
b2 = Beam(355e-9, (-c30, -s30, 0.0), (-s30, c30, 0.0), 200e-6, 10e-3, (0.0, 0.0, 0.0))
dev5b = dataclasses.replace(dev5, beams=(b1, b2), field=dataclasses.replace(dev5.field, direction=(c30, s30, 0.0)))
m5b = gate_modes(dev5b, (1, 3), (0, 1))
sp5b = solve_amplitude_modulation(m5b, mu_hz=2.98e6, duration_s=190e-6)
print(f"  both families ({m5b.n_modes} modes), {choi_segment_count(5, 2)} segments: chi = {sp5b.chi_rad:.6f}, max |alpha| = {max(abs(v) for v in sp5b.integrals.alpha.values()):.1e}, peak Omega/2pi = {sp5b.diagnostics['peak_rabi_hz'] / 1e3:.1f} kHz")
print(f"  segment counts 2N + 1 -> 4N + 1 for N = 5, 15, 17, 30: {[choi_segment_count(n) for n in (5, 15, 17, 30)]} -> {[choi_segment_count(n, 2) for n in (5, 15, 17, 30)]}")

section("5. Baldwin's echo: R_x(pi) U R_-x(pi) U of the printed H at 8 pi (eta Omega/delta)^2 = pi/4")
delta = TWO_PI * 20e3
u = baldwin_echo_unitary(0.1, delta / (0.1 * math.sqrt(32.0)), delta, d=24)
phase = u[0, 0] / abs(u[0, 0])
print(f"  diag(U)/U_00 = {np.round(np.diag(u) / phase, 6)}; max off-diagonal |U| = {np.max(np.abs(u - np.diag(np.diag(u)))):.1e}")

section("6. Debye-Waller law and the thermal references")
for n in (1, 2, 3):
    print(f"  (|M_{n}|^2 - |M_{n - 1}|^2)/(eta^2 [1 - eta^2 (2n + 1)]) at eta = 0.123: {sideband_coupling_squared_difference(0.123, n) / (0.123**2 * (1 - 0.123**2 * (2 * n + 1))):.4f}")
pref = (math.pi**2 / 4) * 0.1**4
print(f"  (pi^2/4) eta^4 <(n - n_ref)^2> at nbar = 1 in units of (pi^2/4) eta^4: mean {thermal_debye_waller_infidelity(0.1, 1.0, 'mean') / pref:.3f}, n0 {thermal_debye_waller_infidelity(0.1, 1.0, 'n0') / pref:.3f}, -1/2 {thermal_debye_waller_infidelity(0.1, 1.0, 'minus_half') / pref:.3f}")
print(f"  Kirchmair 40Ca+ anchor eta = 0.044: n = 0-referenced Debye-Waller loss at nbar = 20: {ballance_thermal_error(0.044, 20.0):.4e} (measured parity contrast 0.964 includes the laboratory noise)")
