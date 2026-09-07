"""Scaling I (PLAN.md milestone M9a; Sections 5.1, 5.1.1, 5.2, 5.4, 5.5, 9.8, 9.17, 11.3), through the package.

1. State-based process tomography on synthetic channels (Section 5.4 (a); Section 9.17 row "GATE_LOCAL tomography"): the Choi
   matrix of a unitary from its sixteen inputs, the Dykstra projection's residuals against CP and TP on a noisy depolarizing
   reconstruction, the Kraus operators and the Section 6.8 summary.
2. The contribution criterion (Section 11.3 item 2; Section 9.8 row 3): a mode with eta = 1e-3 two kilohertz from a tone against
   one with eta = 0.05 a megahertz away, and the freeze-against-drop rows of Section 9.17.
3. The frozen spectators' off-resonant excitation bound and the detuning guard (Section 5.2) on the two-ion fixture.
4. The ENR option (Section 5.1.1): the sum-generator exponential against the product of per-mode displacements inside and at the
   cap, the marginal by index sums against ptrace, the dims/shape rule, and the regrid of an ENR state to a larger cap.
5. The adaptive cap and margin policy (Section 5.5): a carrier pulse on a cap whose margin is below the Section 5.1.1 table.
6. GATE_LOCAL against JOINT_EXACT on the two-ion Bell circuit (Section 9.8): the register populations within the reported
   residual-displacement bound, the tracked occupations against the joint reduced state, the MS step's channel summary.

Run: uv run python validation/scripts/check_scaling.py (about twelve minutes). Lines prefixed MC: are Monte Carlo results.
"""

from __future__ import annotations

import math

import numpy as np
import qutip as qt

from qutip_trap.api import (
    Circuit,
    HilbertSpace,
    ModeTruncation,
    Operation,
    SolverOptions,
    last_record,
    register_fidelity,
    run,
)
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control import native
from qutip_trap.control.pulses import Drive, Pulse, Tone
from qutip_trap.control.schedule import Schedule, single_qubit_pulse
from qutip_trap.control.shaping import GateModes
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec
from qutip_trap.dynamics.tomography import (
    choi_least_squares,
    cp_residual,
    input_states,
    kraus_operators,
    project_cptp,
    tp_residual,
)
from qutip_trap.hilbert.operators import displacement_matrix_analytic, required_margin
from qutip_trap.hilbert.truncation import regrid_state
from qutip_trap.light.raman import lamb_dicke_parameters
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.noise.summary import apply_choi, choi_from_unitary, depolarizing_choi, entanglement_infidelity, pauli_twirl
from qutip_trap.run.space import classify, frozen_excitation_bounds, waveform_contributions
from qutip_trap.units import TWO_PI
from tests.m4_fixtures import chain_device, derived_seeds, raman_gate_drives
from tests.m6_fixtures import circuit_fixture


def section(title: str) -> None:
    print(f"\n== {title}")


OPTS = SolverOptions()

section("1. state-based process tomography on synthetic channels (Section 5.4 (a); Section 9.17)")
u = native.ms(0.3, -0.7, math.pi / 2.0)
inputs = input_states((2, 2))
rhos = [np.outer(k, k.conj()) for _l, k in inputs]
c = choi_least_squares(rhos, [u @ r @ u.conj().T for r in rhos])
cp, cpr, tpr, its = project_cptp(c)
ks = kraus_operators(cp)
phase = ks[0][0, 3] / u[0, 3]
print(
    f"  MS(0.3, -0.7, pi/2) from {len(inputs)} inputs: |C - C_U| = {np.max(np.abs(c - choi_from_unitary(u))):.1e}, "
    f"cp residual {cpr:.1e}, tp residual {tpr:.1e}, Dykstra iterations {its}, {len(ks)} Kraus operator, |K - e^(i phi) U| = "
    f"{np.max(np.abs(ks[0] - phase * u)):.1e}"
)
cd = depolarizing_choi(0.05, 2)
rng = np.random.default_rng(1)
outs = []
for r in rhos:
    o = apply_choi(cd, r) + 3e-4 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    outs.append(0.5 * (o + o.conj().T))
raw = choi_least_squares(rhos, outs)
cp2, cpr2, tpr2, its2 = project_cptp(raw)
eps = entanglement_infidelity(cp2, choi_from_unitary(np.eye(4, dtype=complex)))
tw = pauli_twirl(cp2, 2)
print(
    f"  depolarizing eps = 0.05 with 3e-4 noise on the outputs: raw residuals cp {cp_residual(raw):.2e}, tp {tp_residual(raw):.2e}; "
    f"projected cp {cpr2:.1e}, tp {tpr2:.1e} after {its2} iterations; |C - C_true| = {np.max(np.abs(cp2 - cd)):.2e}"
)
print(f"  Section 6.8 summary of the projected map: depolarizing rate (entanglement infidelity) {eps:.4f}, twirl p_II = {tw['II']:.4f}, mean p_P (P != I) = {np.mean([v for k, v in tw.items() if k != 'II']):.5f} against 0.05/15 = {0.05 / 15:.5f}")

section("2. the contribution criterion (Section 11.3 item 2; Section 9.8 row 3; Section 9.17 'Freeze against drop')")
omega_gate = TWO_PI * 3.0e6
probe = GateModes(ions=(0, 1), modes=(0, 1), omega_rad_s=(omega_gate, TWO_PI * 5.0e6), eta={0: (0.08, 1e-3), 1: (0.08, 1e-3)}, nbar=(0.0, 0.0))
wf0 = Waveform.symmetric(probe, gate_mode=0, epsilon_hz=20e3, duration_s=100e-6, all_modes=False)
assert wf0.segments is not None
mu_tone = abs(float(wf0.segments[0].detuning_hz["blue"]))  # type: ignore[arg-type]
for label, eta, gap in (("eta = 1e-3, 2 kHz from the tone", 1e-3, 2e3), ("eta = 0.05, 1 MHz from the tone", 0.05, 1e6), ("eta = 1e-3, 1 MHz from the tone", 1e-3, 1e6)):
    modes = GateModes(ions=(0, 1), modes=(0, 1), omega_rad_s=(omega_gate, TWO_PI * (mu_tone + gap)), eta={0: (0.08, eta), 1: (0.08, eta)}, nbar=(0.0, 0.0))
    wf = Waveform.symmetric(modes, gate_mode=0, epsilon_hz=20e3, duration_s=100e-6, all_modes=False)
    cc = waveform_contributions(wf, modes, (0, 1))[1]
    cls = classify(cc, coupled=True, freeze_alpha_max=OPTS.freeze_alpha_max, freeze_chi_max_rad=OPTS.freeze_chi_max_rad)
    cls_uncoupled = classify(cc, coupled=False, freeze_alpha_max=OPTS.freeze_alpha_max, freeze_chi_max_rad=OPTS.freeze_chi_max_rad)
    print(f"  {label}: |alpha|^2 (2n+1) = {cc.alpha2_weighted:.2e}, |chi| = {cc.chi_rad:.2e} rad, loop radius {cc.radius:.4f} -> {cls} (a mode no pulse couples to would be {cls_uncoupled})")
print(f"  tone at {mu_tone / 1e6:.6f} MHz for the x-COM at 3.0 MHz (inside detuning, eps = 20 kHz); amplitude {float(wf0.segments[0].amplitude_hz[(0, 'blue')]) / 1e3:.3f} kHz")  # type: ignore[arg-type]

section("3. the frozen spectators' off-resonant excitation bound and the detuning guard (Section 5.2)")
dev = chain_device(2)
rabi, stark = derived_seeds(dev, raman_gate_drives(2))
omega_hz = rabi[(0, 0)]
for mu_hz in (3.0e6 + 20e3, 3.0e6 + 100e3):
    drive = Drive("raman", (0,), (Tone(mu_hz, 0.0, omega_hz),), (0, 1), 0.0, {})
    pulse = Pulse(drive, 0.0, 50e-6, "p", ())
    bounds, guard = frozen_excitation_bounds(dev, [pulse], [2, 3], {2: 0.5, 3: 0.5})
    print(f"  tone at {mu_hz / 1e6:.3f} MHz, Omega/2pi = {omega_hz / 1e3:.2f} kHz, nbar = 0.5: bound on the rocking mode (2.828 MHz) {bounds[2]:.3e}, on the COM (3.0 MHz) {bounds[3]:.3e}; guard violations {len(guard)}")

section("4. the ENR option: sum-generator exponential, marginals, shape rule, regrid (Sections 5.1, 5.1.1; Section 9.17)")
space = HilbertSpace((2,), (), ((1, 2), 10), (0,))
eta1, eta2 = 0.1, 0.05
d_enr = space.enr_displacement({1: eta1, 2: eta2})
dims, n_exc = space._enr_dims()
_n, s2i, i2s = qt.enr_state_dictionaries(dims, n_exc)
dense = np.asarray(d_enr.full())
a1 = displacement_matrix_analytic(n_exc + 1, 1j * eta1)
a2 = displacement_matrix_analytic(n_exc + 1, 1j * eta2)
worst_inside = 0.0
worst_cap = 0.0
for (n1, n2), i in s2i.items():
    for (m1, m2), j in s2i.items():
        diff = abs(dense[j, i] - a1[m1, n1] * a2[m2, n2])
        if n1 + n2 <= n_exc // 2 and m1 + m2 <= n_exc // 2:
            worst_inside = max(worst_inside, diff)
        if n1 + n2 == n_exc or m1 + m2 == n_exc:
            worst_cap = max(worst_cap, diff)
print(f"  two modes at N_exc = 10, eta = (0.1, 0.05): unitarity defect {(d_enr.dag() * d_enr - space.enr_identity()).norm():.1e}; |expm(sum) - product| inside n1 + n2 <= 5: {worst_inside:.1e}, at the cap: {worst_cap:.2e}")
space6 = HilbertSpace((2,), (), ((1, 2), 6), (0,))
op = qt.tensor(qt.sigmap(), space6.enr_displacement({1: 0.1, 2: 0.05}))
print(f"  tensor(sigmap(), D_enr) at N_exc = 6: shape {op.shape[0]}, product of dims {int(np.prod(op.dims[0]))} (every dimension computation reads shape)")
rng = np.random.default_rng(3)
v = rng.normal(size=space6.dimension) + 1j * rng.normal(size=space6.dimension)
v /= np.linalg.norm(v)
ket = qt.Qobj(v.reshape(-1, 1), dims=[space6.dims, [1, 1]])
_n6, s2i6, i2s6 = qt.enr_state_dictionaries([7, 7], 6)
arr = np.asarray(ket.full()).reshape(space6.dims)
full = np.zeros((2, 7, 7), dtype=complex)
for idx in range(len(i2s6)):
    n1, n2 = i2s6[idx]
    full[:, n1, n2] = arr[:, idx]
full_ket = qt.Qobj(full.reshape(-1, 1), dims=[[2, 7, 7], [1, 1, 1]])
print(f"  ENR marginal by index sums against ptrace of the embedded product-space state: mode 1 {(space6.mode_marginal(ket, 1) - full_ket.ptrace(1)).norm():.1e}, mode 2 {(space6.mode_marginal(ket, 2) - full_ket.ptrace(2)).norm():.1e}, ion {(space6.internal_marginal(ket) - full_ket.ptrace(0)).norm():.1e}")
old = HilbertSpace((2,), (ModeTruncation(0, 5, (0, 1), 0.1),), ((1, 2), 4), ())
new = old.grown(1, 2)
st = old.initial_state([1], fock={0: 2, 1: 1, 2: 2})
assert st.joint is not None
big = regrid_state(st.joint, old, new)
print(f"  regrid of an ENR state from N_exc = 4 to 6: dims {old.dims} -> {new.dims}, norm {big.norm():.12f}, P(n_1 = 1) = {new.fock_populations(big, 1)[1]:.12f}, P(n_2 = 2) = {new.fock_populations(big, 2)[2]:.12f}")

section("5. the adaptive cap and margin policy (Section 5.5) on a carrier pulse")
drives = raman_gate_drives(2)
pulse = single_qubit_pulse(0, math.pi / 2.0, 0.0, drives[0], rabi[(0, 0)], 0.0, gate_id="gpi2", programmed=False)
sched = Schedule((pulse,), (), (), {0: 0.0, 1: 0.0})
etas0, _ = lamb_dicke_parameters(dev, 0, pulse.drive.delta_k(dev.beams))
need = required_margin(abs(etas0[3]))
small = HilbertSpace((2, 2), (ModeTruncation(3, 5, (0, 0), 0.1),), None, (0, 1, 2, 4, 5))
state = small.initial_state([0, 0], fock={3: 0})
eng = JointExactEngine()
tr = eng.run_pulses(dev, sched, state, small, quiet_sample(), SeedSpec(0), SolverOptions())
rep = eng.last_report
assert rep is not None
print(f"  eta_COM = {abs(etas0[3]):.4f} needs a margin of {need} levels; cap d = 5 above |0>: {rep.growth_retries} cap-raising retry, final dims {rep.space.dims}, populated n_max {rep.populated_n_max}, margin reached {rep.margin_reached}, P1 = {float(np.real(tr.expectations['P1[0]'][-1])):.6f}")
eng_off = JointExactEngine()
tr_off = eng_off.run_pulses(dev, sched, state, small, quiet_sample(), SeedSpec(0), SolverOptions(margin_check=False))
print(f"  margin_check=False keeps dims {eng_off.last_report.space.dims}: P1 = {float(np.real(tr_off.expectations['P1[0]'][-1])):.6f} (the carrier barely moves the mode; the policy guards the cases where it does)")

section("6. GATE_LOCAL against JOINT_EXACT on the two-ion Bell circuit (Sections 5.4, 9.8)")
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
fx = circuit_fixture(2)
sur = surrogate_table(fx.device, pairs=[(0, 1)], gate_drives=fx.gate_drives, entangling_drives=fx.entangling_drives, detection_records=2000, detection_windows_s=WINDOWS)
BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
kw = dict(table=sur.table, gate_drives=fx.gate_drives, entangling_drives=fx.entangling_drives, keep_final_state=True, options=SolverOptions(branch_weight_min=1e-3))
je = run(BELL, fx.device, 2000, level="JOINT_EXACT", **kw)  # type: ignore[arg-type]
gl = run(BELL, fx.device, 2000, level="GATE_LOCAL", **kw)  # type: ignore[arg-type]
assert je.final_state is not None and gl.final_state is not None and gl.diagnostics.gate_local is not None
p_je = np.real(np.diag(np.asarray(je.final_state.full())))
p_gl = np.real(np.diag(np.asarray(gl.final_state.full())))
rep_gl = gl.diagnostics.gate_local
print(f"  register populations (00, 01, 10, 11): JOINT_EXACT {np.round(p_je, 6).tolist()}, GATE_LOCAL {np.round(p_gl, 6).tolist()}; max difference {np.max(np.abs(p_je - p_gl)):.2e} against the reported bound {rep_gl.discrepancy_bound:.2e} (residual displacement {rep_gl.residual_bound_total:.2e}, frozen excitation {rep_gl.frozen_excitation_total:.2e}, dropped crosstalk {rep_gl.dropped_crosstalk_total:.2e})")
print(f"  register 1 - F against the compiled circuit: JOINT_EXACT {1.0 - register_fidelity(je):.3e}, GATE_LOCAL {1.0 - register_fidelity(gl):.3e}; |rho_JE - rho_GL| = {np.max(np.abs(np.asarray(je.final_state.full()) - np.asarray(gl.final_state.full()))):.2e}")
rec_je = last_record(je)
w_tot = sum(br.weight for br in rec_je.branches)
joint_final = {
    m: sum(br.weight * tr.final.motional.nbar[m] for br, tr in zip(rec_je.branches, rec_je.traces)) / w_tot for m in (2, 3)
}
for s in rep_gl.steps:
    if s.kind != "gate":
        continue
    summ = s.summary
    line = (
        f"  step {s.gate_id}: ions {s.ions}, dims {list(s.space_dims)}, resolved {s.resolved}, frozen coupled {s.frozen_coupled}, "
        f"{s.n_inputs} inputs x {s.n_branches} branches ({s.engine_runs} engine runs, {s.method}), cp/tp residual {s.cp_residual:.1e}/{s.tp_residual:.1e}"
    )
    if summ is not None:
        line += f", average gate infidelity {summ.average_gate_infidelity:.3e}, depolarizing rate {summ.depolarizing_rate:.3e}, twirl p_II {summ.pauli_twirled.get('II', float('nan')):.5f}"
    if s.resolved:
        line += f"; residual |alpha| per spin eigenstate {dict((m, f'{v:.2e}') for m, v in s.residual_displacement.items())}, bound {s.residual_bound:.2e}, purity deficit {dict((m, f'{v:.1e}') for m, v in s.purity_deficit.items())}"
        line += f"; tracked nbar after {dict((m, f'{s.nbar_after[m]:.5f}') for m in s.resolved)} against the joint run's branch-weighted final {dict((m, f'{joint_final[m]:.5f}') for m in s.resolved)}"
    print(line)
print(f"  engine runs {rep_gl.engine_runs}, cache hits {rep_gl.cache_hits}, largest gate-local dimension {rep_gl.largest_local_dimension}, register {rep_gl.register}")
print(f"MC: histograms over 2000 shots: JOINT_EXACT {dict(sorted(je.probabilities.items()))}, GATE_LOCAL {dict(sorted(gl.probabilities.items()))}")
print("\ndone")
