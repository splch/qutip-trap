"""Noise and error channels (PLAN.md Section 6; Section 9.7; Section 9.16 rows 4.1-3, 4.4-8, 6-3, 6-4; Section 9.12 'Fang echo
identities'; milestone M7), through the package.

1. Heating and motional dephasing during a K-loop MS gate on the two-ion 171Yb+ fixture (Ballance eps_h = ndot t_g/(2K), Kirchmair
   Delta F = Gamma_h t_g/2, eps_d = alpha_K t_g/tau) through the JOINT_EXACT engine with mesolve; the single-mode closure.
2. The intensity-noise channel c_op = sqrt(2k) H_int in the force model: the two-term fit, its Gamma_I-independent ratio, and the
   carrier-contrast definition of Gamma_I (the factor 2 against the plan's derived A).
3. Dephasing correlation at N = 3: local against global sigma_z noise under the derived bounds.
4. Fang's crosstalk closed forms (fuzz), the printed halved forms, the echo identities of Section 9.12, Landsman's bound.
5. Photon scattering on the fixture: per-pulse probabilities, the operator sum rule, the recoil quanta per photon, leakage at d = 3.
6. Ornstein-Uhlenbeck field noise heating (Section 9.16 row 4.1-3): the exact double integral against e^2 S_E^(1)/(4 m hbar omega) and a
   classical Monte Carlo (MC: line).
7. The control hardware chain on the calibrated two-ion gate: the beat-phase reference the modulator response demands.
8. Filter-function numbers from the package machinery (gated CPMG against Biercuk, finite-pulse UDD collapse, dc floors).
9. Collisions and the Section 6.8 summaries.

Run: uv run python validation/scripts/check_noise.py (a few minutes). Lines prefixed MC: are Monte Carlo results.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import qutip as qt
from scipy.linalg import expm

from qutip_trap.api import (
    Collisions,
    HilbertSpace,
    ModeTruncation,
    ScatteringOptions,
    SolverOptions,
    Waveform,
    choi_from_unitary,
    collision_rate_per_ion,
    decoupling_sequence,
    depolarizing_choi,
    entanglement_infidelity,
    pauli_twirl,
    scattering_channels,
    scattering_estimates,
)
from qutip_trap.calibration.entangling import exact_gate_check, gate_space
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.composite import composite_pulse
from qutip_trap.control.schedule import response_phase_rad
from qutip_trap.control.shaping import gate_modes
from qutip_trap.dynamics.channels import heating_channels, motional_dephasing_channels
from qutip_trap.light.raman import derive_raman_drive, scattering_budget, square_drive
from qutip_trap.noise.decoupling import composite_segments, dc_floor
from qutip_trap.noise.levels import internal_levels
from qutip_trap.noise.summary import pauli_string
from qutip_trap.units import ATOMIC_MASS_KG
from qutip_trap.validation.two_qubit_closed_forms import (
    ballance_dephasing_error,
    ballance_heating_error,
    dephasing_error_derived,
    fang_bell_fidelity,
    fang_crosstalk_unitary,
    fang_printed_bell_fidelity,
    fang_printed_spectator_excitation,
    fang_spectator_excitation,
    landsman_parallel_gate_bound,
    ou_field_heating_finite_time,
    ou_field_heating_slope,
)
from tests.m2_fixtures import single_ion_raman_device
from tests.m4_fixtures import (
    X_COM_TWO_IONS,
    raman_gate_drives,
    table_with_waveform,
    two_ion_device,
    two_ion_modes,
)
from tests.m6_fixtures import circuit_fixture

TORR_PA = 133.32236842105263


def head(s: str) -> None:
    print(f"\n== {s}")


# ---------------------------------------------------------------------------------------------
head("1. heating and motional dephasing during the K-loop MS gate (Section 9.7 rows 1-2; exact engine, mesolve)")
dev = two_ion_device()
modes = two_ion_modes(dev)
FAST = SolverOptions(mesolve_dimension_max=4096)
for loops in (1, 2):
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, loops=loops, epsilon_hz=20e3 * loops, all_modes=False)
    space = HilbertSpace((2, 2), (ModeTruncation(X_COM_TWO_IONS, 12, (0, 4), 0.15),), None, (0, 1, 2, 4, 5))
    table = table_with_waveform((0, 1), wf, device=dev, drives=raman_gate_drives(2))
    base, _ = exact_gate_check(dev, wf, (0, 1), raman_gate_drives(2), table, space=space, options=FAST)
    t_g = wf.duration_s
    ndot = 400.0
    noisy, _ = exact_gate_check(
        dev, wf, (0, 1), raman_gate_drives(2), table, space=space, channels=heating_channels(space, {X_COM_TWO_IONS: ndot}), options=FAST
    )
    loss_h = base.fidelity - noisy.fidelity
    tau = t_g / 5e-3
    deph, _ = exact_gate_check(
        dev, wf, (0, 1), raman_gate_drives(2), table, space=space, channels=motional_dephasing_channels(space, {X_COM_TWO_IONS: tau}), options=FAST
    )
    loss_d = base.fidelity - deph.fidelity
    print(
        f"  K = {loops}: t_g = {t_g * 1e6:.1f} us, base fidelity {base.fidelity:.4f}; heating at 400 quanta/s: loss {loss_h:.3e} against "
        f"ndot t_g/(2K) = {ballance_heating_error(ndot, t_g, loops):.3e} (ratio {loss_h / ballance_heating_error(ndot, t_g, loops):.3f}); "
        f"motional dephasing tau = t_g/5e-3: loss {loss_d:.3e} against alpha_K t_g/tau = {ballance_dephasing_error(t_g, tau, loops):.3e} "
        f"(ratio {loss_d / ballance_dephasing_error(t_g, tau, loops):.3f})"
    )
print("  the two-mode closure played with the rocking mode frozen over-rotates the gate and gives 0.84 and 0.80 of the closed forms (the M7 diagnostic)")

# ---------------------------------------------------------------------------------------------
head("2. intensity-noise channel c_op = sqrt(2k) H_int in the force model (Section 9.16 row 4.4-8)")
d = 40
a = qt.destroy(d)
n_op = a.dag() * a
eps = 1.0
sx2 = qt.tensor(qt.sigmax(), qt.qeye(2)) + qt.tensor(qt.qeye(2), qt.sigmax())
idq = qt.tensor(qt.qeye(2), qt.qeye(2))
psi00 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
eta = 0.1
for loops in (1, 2):
    t_g = 2.0 * math.pi * loops / eps
    force = eps / (4.0 * math.sqrt(loops))
    h = eps * qt.tensor(idq, n_op) + force * qt.tensor(sx2, a + a.dag())
    h_int = force * qt.tensor(sx2, a + a.dag())
    omega = 2.0 * force / eta
    k = 2e-6
    gamma_i = k * omega**2
    losses = []
    for nb in (0.0, 0.5, 1.0):
        rho_m = qt.fock_dm(d, 0) if nb == 0.0 else qt.thermal_dm(d, nb)
        rho0 = qt.tensor(qt.ket2dm(psi00), rho_m)
        opts = {"atol": 1e-11, "rtol": 1e-9, "nsteps": 10**6}
        ideal = qt.mesolve(h, rho0, [0.0, t_g], [], options=opts).final_state.ptrace([0, 1])
        noisy = qt.mesolve(h, rho0, [0.0, t_g], [math.sqrt(2.0 * k) * h_int], options=opts).final_state.ptrace([0, 1])
        losses.append(1.0 - float(np.real((noisy * ideal).tr())))
    slope, intercept = np.polyfit([1.0, 2.0, 3.0], losses, 1)
    unit = gamma_i * t_g * eta**2
    print(
        f"  N = 2, K = {loops}: A/(Gamma_I t_g eta^2) = {slope / unit:.4f} (the plan's derived form has 1/2), B/A = {intercept / slope:.4f} "
        f"(derived 3(N-1)/(4K) = {3.0 / (4.0 * loops):.4f}; the printed (N-1)/4 term would give {(1.0 / 4.0) / (1.0 / 2.0):.4f})"
    )
rho = qt.basis(2, 0).proj()
lind = qt.lindblad_dissipator(math.sqrt(2.0 * 1e-3) * 0.5 * 5.0 * qt.sigmax())
drho = qt.vector_to_operator(lind * qt.operator_to_vector(rho))
print(f"  carrier-contrast definition: d<sigma_z>/dt / (k Omega^2 <sigma_z>) = {float(qt.expect(qt.sigmaz(), drho)) / (-1e-3 * 25.0):.6f} under sqrt(2k)(Omega/2) sigma_x")
print("  -> with Gamma_I the carrier-contrast rate the budget is Gamma_I t_g eta^2 (2 nbar + 1) + Gamma_I t_g eta^2 3(N-1)/(4K): twice the plan's derived A and B, the same ratio")

# ---------------------------------------------------------------------------------------------
head("3. dephasing correlation at N = 3 (Section 9.7 row 4): local against global sigma_z noise")
d3 = 12
a3 = qt.destroy(d3)
t_g = 2.0 * math.pi
force = 0.25
sx3 = sum(qt.tensor(*[qt.sigmax() if k == i else qt.qeye(2) for k in range(3)]) for i in range(3))
idq3 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2))
h3 = qt.tensor(idq3, a3.dag() * a3) + force * qt.tensor(sx3, a3 + a3.dag())
t2 = t_g / 2e-3
gamma = 1.0 / t2
zs = [qt.tensor(*[qt.sigmaz() if k == i else qt.qeye(2) for k in range(3)]) for i in range(3)]
local = [math.sqrt(gamma / 2.0) * qt.tensor(z, qt.qeye(d3)) for z in zs]
glob = [math.sqrt(gamma / 2.0) * qt.tensor(sum(zs[1:], zs[0]), qt.qeye(d3))]
rho0 = qt.tensor(qt.ket2dm(qt.tensor(*[qt.basis(2, 0)] * 3)), qt.fock_dm(d3, 0))
opts = {"atol": 1e-10, "rtol": 1e-8, "nsteps": 10**6}
ideal = qt.mesolve(h3, rho0, [0.0, t_g], [], options=opts).final_state.ptrace([0, 1, 2])
for label, c_ops, bound in (("local", local, dephasing_error_derived(3, t_g, t2, False)), ("global", glob, dephasing_error_derived(3, t_g, t2, True))):
    noisy = qt.mesolve(h3, rho0, [0.0, t_g], c_ops, options=opts).final_state.ptrace([0, 1, 2])
    loss = 1.0 - float(np.real((noisy * ideal).tr()))
    print(f"  {label:6s}: loss {loss:.4e} against the derived bound {bound:.4e} (ratio {loss / bound:.3f}); Bermudez's printed form is {8 if label == 'global' else 4}x the bound")

# ---------------------------------------------------------------------------------------------
head("4. Fang crosstalk forms, echo identities and Landsman's bound (Section 9.7 rows 6-7; 9.16 row 6-4; 9.12)")
rng = np.random.default_rng(0)
bell = np.zeros(8, complex)
bell[0], bell[6] = 1 / math.sqrt(2), -1j / math.sqrt(2)
psi0 = np.zeros(8, complex)
psi0[0] = 1.0
worst = 0.0
for _ in range(4000):
    phi = rng.uniform(0, 2 * math.pi)
    t13, t23 = rng.uniform(-0.5, 0.5, 2)
    psi = fang_crosstalk_unitary(math.pi / 4, t13, t23, phi) @ psi0
    p3 = sum(abs(psi[k]) ** 2 for k in range(8) if k % 2 == 1)
    worst = max(worst, abs(abs(np.vdot(bell, psi)) ** 2 - fang_bell_fidelity(t13, t23)), abs(p3 - fang_spectator_excitation(t13, t23)))
print(f"  4000 samples over (phi_beam, theta_13, theta_23): max |numeric - closed form| = {worst:.2e}")
print(
    f"  at (0.1644, -0.2763): F = {fang_bell_fidelity(0.1644, -0.2763):.6f}, P_ion3 = {fang_spectator_excitation(0.1644, -0.2763):.6f} "
    f"(printed {fang_printed_bell_fidelity(0.1644, -0.2763):.6f} / {fang_printed_spectator_excitation(0.1644, -0.2763):.6f}); at theta = pi/2: "
    f"P_ion3 = {fang_spectator_excitation(math.pi / 2, math.pi / 2):.1f} (printed {fang_printed_spectator_excitation(math.pi / 2, math.pi / 2):.1f})"
)
theta, t13, t23, phi = 0.7, 0.05, -0.03, 0.4
x1x2 = np.kron(np.kron(pauli_string("X"), pauli_string("X")), np.eye(2))
target = expm(-1j * theta * x1x2)
half = fang_crosstalk_unitary(theta / 2, t13 / 2, t23 / 2, phi)
z3 = np.kron(np.eye(4), pauli_string("Z"))
y12 = np.kron(np.kron(pauli_string("Y"), pauli_string("Y")), np.eye(2))
print(
    f"  echo identities at theta = 0.7, (theta_13, theta_23) = (0.05, -0.03): ||U U - XX(theta)|| = {np.linalg.norm(half @ half - target, 2):.3e}; "
    f"||(Z3 U)^2 - XX|| = {np.linalg.norm(z3 @ half @ z3 @ half - target, 2):.3e}; ||(Y1Y2 U)^2 - XX|| = {np.linalg.norm(y12 @ half @ y12 @ half - target, 2):.3e} (first-order cancellation)"
)
b, vac = landsman_parallel_gate_bound([0.1, -0.05, 0.02, 0.01])
print(f"  Landsman: (1/2)||E||_diamond <= sum |Theta| = {b:.2f} (vacuous: {vac}); with phases (0.6, 0.5, 0.1, 0.1) the bound {landsman_parallel_gate_bound([0.6, 0.5, 0.1, 0.1])[0]:.1f} > 1 is vacuous")

# ---------------------------------------------------------------------------------------------
head("5. photon scattering on the single-ion 171Yb+ fixture (Section 9.7 row 8; Sections 4.5.5, 6.5)")
sdev = single_ion_raman_device()
der = derive_raman_drive(sdev, 0, (0, 1), scattering=False)
budget = scattering_budget(sdev, 0, (0, 1))
lower, upper = sdev.crystal.species[0].qubit
print(
    f"  derived carrier Rabi {der.carrier_rabi_hz / 1e3:.3f} kHz at the beams' power; rates from |0>: Rayleigh {budget.rayleigh_per_s[lower]:.4e} s^-1, spin flip "
    f"{budget.raman_spin_flip_per_s[lower]:.4e}, leakage {budget.leakage_per_s[lower]:.4e}; Gamma_el = {budget.rayleigh_dephasing_per_s:.2e} s^-1 (clock states)"
)
from qutip_trap.control.pulses import Pulse

pulse = Pulse(square_drive(der, include_stark=False, rabi_scale=50.0), 0.0, 1.0, "p", ())
space1 = HilbertSpace((2,), (ModeTruncation(1, 10, (0, 3), 0.25),), None, (0, 2))
ops, notes = scattering_channels(sdev, pulse, space1, options=ScatteringOptions(recoil="minimal"))
st = space1.initial_state([0])
tot = sum(float(qt.expect(o.op.dag() * o.op, st.joint).real) for o in ops)  # type: ignore[union-attr]
print(f"  {len(ops)} operators at 50x the nominal intensity; sum <0|L^dag L|0> = {tot:.4e} against 50 x (Rayleigh + flip) = {50 * (budget.rayleigh_per_s[lower] + budget.raman_spin_flip_per_s[lower]):.4e}")
rate_tot = 50.0 * (budget.raman_spin_flip_per_s[lower] + budget.rayleigh_per_s[lower])
t = 0.02 / rate_tot
res = qt.mesolve(0.0 * space1.identity(), st.joint, [0.0, t], [o.op for o in ops], e_ops=[space1.projector(0, 1), space1.number(1)])
print(f"  mesolve over 0.02 scattering events: P(flip) = {res.expect[0][-1]:.4e} against Gamma_flip t = {50 * budget.raman_spin_flip_per_s[lower] * t:.4e}; recoil quanta per photon on the x mode {res.expect[1][-1] / (rate_tot * t):.4e}")
est = scattering_estimates(sdev, pulse)
print(f"  per-pulse estimates at 50x: {', '.join(f'{k} = {v:.3e}' for k, v in est.items())}")
lev = internal_levels(sdev.crystal.species[0], 3, sdev.field.B_gauss, sdev.field.direction)
space3 = HilbertSpace((3,), (), None, (0, 1, 2))
ops3, notes3 = scattering_channels(sdev, pulse, space3, levels_by_ion={0: lev}, options=ScatteringOptions(recoil="off"))
st3 = space3.initial_state([0])
t3 = 0.02 / (50.0 * sum(budget.leakage_per_s.values()))
res3 = qt.mesolve(0.0 * space3.identity(), st3.joint, [0.0, t3], [o.op for o in ops3], e_ops=[space3.projector(0, 2)])
print(f"  d = 3 with the SINK {lev.labels}: P(SINK) = {res3.expect[0][-1]:.4e} against Gamma_leak t = {50 * budget.leakage_per_s[lower] * t3:.4e}; {len(ops3)} operators, notes {[n for n in notes3 if 'leak' in n]}")

# ---------------------------------------------------------------------------------------------
head("6. Ornstein-Uhlenbeck field noise heating (Section 9.16 row 4.1-3; e = m = hbar = 1, omega = 1, sigma^2 = 0.02, tau_c = 0.05)")
slope = ou_field_heating_slope(0.02, 0.05, 1.0)
finite = ou_field_heating_finite_time(0.02, 0.05, 1.0, 200.0)
print(f"  e^2 S_E^(1)/(4 m hbar omega) = {slope:.6e}; exact double integral <n>(200)/200 = {finite / 200:.6e} (ratio {finite / 200 / slope:.5f}; the row prints 0.99991 for its estimator)")
print(f"  rival prefactors: (1/2) S^(1) gives ratio {2.0:.3f}, (1/4) S^(2) gives {0.25 * 2 * 0.02 * 0.05 / (1 + 0.05**2) / slope:.4f}")
rng = np.random.default_rng(7)
n_traj, t_end, dt = 4000, 200.0, 0.005
steps = int(round(t_end / dt))
x = np.zeros(n_traj)
p_mom = np.zeros(n_traj)
e_field = rng.normal(0.0, math.sqrt(0.02), n_traj)
rho_ou = math.exp(-dt / 0.05)
sig_ou = math.sqrt(0.02 * (1.0 - rho_ou**2))
for _ in range(steps):
    # velocity Verlet for x'' = -x + E(t) with E an exact OU chain held over each step; n = (x^2 + p^2)/2 classically
    acc0 = -x + e_field
    x = x + p_mom * dt + 0.5 * acc0 * dt * dt
    e_field = rho_ou * e_field + sig_ou * rng.standard_normal(n_traj)
    acc1 = -x + e_field
    p_mom = p_mom + 0.5 * (acc0 + acc1) * dt
n_mc = 0.5 * (x**2 + p_mom**2)
print(f"MC: <n>(200) over 4000 classical trajectories = {n_mc.mean():.5f} +- {n_mc.std() / math.sqrt(n_traj):.5f} against slope x 200 = {slope * 200:.5f}")

# ---------------------------------------------------------------------------------------------
head("7. the control hardware chain on the calibrated two-ion gate (Section 7.10; the M7 finding)")
fx0 = circuit_fixture(2)
from tests.fixtures import make_hardware

fx = dataclasses.replace(fx0, device=dataclasses.replace(fx0.device, hardware=dataclasses.replace(make_hardware(realistic=True), phase_continuous=False)))
sur = surrogate_table(fx.device, pairs=[(0, 1)], gate_drives=fx.gate_drives, entangling_drives=fx.entangling_drives, detection_records=200, detection_windows_s=(20e-6,))
wf2 = sur.table.waveform_for((0, 1))
assert wf2 is not None
nb = {m: e.value for m, e in sur.table.nbar.items()}
gm = gate_modes(fx.device, (0, 1), (0, 1), nbar=nb)
gspace = gate_space(gm, 2, waveform=wf2, nbar=nb)
mu = abs(float(wf2.segments[0].detuning_hz["blue"]))  # type: ignore[index, arg-type]
tau_r = fx.device.hardware.aom_rise_s
print(f"  beat note mu/2pi = {mu / 1e6:.4f} MHz, AOM tau_r = {tau_r * 1e9:.0f} ns: 2 pi mu tau_r = {2 * math.pi * mu * tau_r:.4f} rad, arctan = {math.atan(2 * math.pi * mu * tau_r):.4f} = the compensation {response_phase_rad(mu, tau_r):.4f}")
for label, hw in (("ideal modulator", False), ("chain with the arctan reference", True)):
    chk, _ = exact_gate_check(fx.device, wf2, (0, 1), fx.entangling_drives, sur.table, space=gspace, options=SolverOptions(hardware_chain=hw))
    print(f"  {label:32s}: fidelity {chk.fidelity:.4f}, leakage {chk.leakage:.2e}")
dev_nores = dataclasses.replace(fx.device, hardware=dataclasses.replace(fx.device.hardware, aom_rise_s=0.0))
chk, _ = exact_gate_check(dev_nores, wf2, (0, 1), fx.entangling_drives, sur.table, space=gspace, options=SolverOptions(hardware_chain=True))
print(f"  {'quantization only (zero rise)':32s}: fidelity {chk.fidelity:.4f}, leakage {chk.leakage:.2e}")
print("  (uncompensated, the response alone left 2.9e-3 leakage and 0.9970: Roos's tilt at the envelope's delayed arrival; the linear mu tau_r reference 2.2e-4)")

# ---------------------------------------------------------------------------------------------
head("8. filter functions from the package machinery (Section 6.9; check_composite.py sections 10-14)")
for n in (1, 2, 3, 4):
    s = decoupling_sequence("cpmg", n, 1.0, 0.02)
    g = float(s.filter_function(np.array([0.9]), gated=True)[0])
    print(f"  CPMG n = {n}, x = 0.9, delta_pi = 0.02: gated machinery {g:.12g} = Biercuk {float(s.biercuk_filter_function(np.array([0.9]))[0]):.12g}; noise on through the pulse {float(s.filter_function(np.array([0.9]))[0]):.6g}")
for n in (3, 4, 5):
    s = decoupling_sequence("udd", n, 1.0, 0.02)
    f = s.filter_function(np.array([1e-3, 1e-2]), gated=True)
    coeff = f[0] / (1e-3 * 0.02) ** 4 if n % 2 else f[0] / (1e-6 * (1e-3 * 0.02) ** 4)
    print(f"  UDD n = {n} at delta_pi = 0.02: low-frequency exponent {math.log(f[1] / f[0]) / math.log(10):.3f}, coefficient {coeff:.5f} (1/16 odd, 1/64 even)")
sk1 = composite_segments(composite_pulse("SK1", math.pi), 1.0)
print(f"  SK1 dc floor at the Kabytayev benchmark {dc_floor(22.8302557111, 1, 2.07e9 / math.pi, 1.5e6):.5e} (paper 5.86e-6); BB1 {dc_floor(9.388566343, 2, 2.07e9 / math.pi, 1.5e6):.5e} (paper 3.9e-9)")
print(f"  XY4 / XY8 / KDD / CDD(2) return to identity: {[bool(np.allclose(__import__('qutip_trap.noise.decoupling', fromlist=['final_adjoint']).final_adjoint(decoupling_sequence(t, n, 1.0, 0.01).segments()), np.eye(3), atol=1e-12)) for t, n in (('xy4', 4), ('xy8', 8), ('kdd', 20), ('cdd', 2))]}")

# ---------------------------------------------------------------------------------------------
head("9. collisions and the Section 6.8 summaries")
col = Collisions(1e-11 * TORR_PA, {"H2": 1.0}, {"heating_kick": 1.0})
for m in (9.012, 40.08, 171.0):
    r = collision_rate_per_ion(col, m * ATOMIC_MASS_KG)
    print(f"  {m:6.1f} u, H2 at 1e-11 torr, 300 K: Gamma_L = {r:.3e} s^-1 per ion (one per {1 / r / 60:.0f} min)")
for n_q in (1, 2):
    c = depolarizing_choi(3e-3, n_q)
    print(f"  Lambda_eps at eps = 3e-3, n = {n_q}: entanglement infidelity {entanglement_infidelity(c, choi_from_unitary(np.eye(2**n_q))):.6e}")
tw = pauli_twirl(choi_from_unitary(expm(-1j * 0.23 * pauli_string('XX'))), 2)
print(f"  twirl of exp(-i 0.23 XX): p_XX = {tw['XX']:.6f} = sin^2 0.23 = {math.sin(0.23) ** 2:.6f}, p_II = {tw['II']:.6f}")
print("\ndone.")
