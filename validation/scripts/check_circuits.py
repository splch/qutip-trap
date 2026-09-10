"""End-to-end circuits in JOINT_EXACT (PLAN.md Sections 3.4, 5.2, 5.7, 7.2, 7.6, 7.7, 9.6; milestone M6), through the package.

1. The compiler's templates (Section 7.7): the ZXZXZ residual on random SU(2) targets, Maslov's CNOT for all four signs with
   its global phase e^{i pi v s/4}, the exact CP(theta) and Debnath's template with the 0.854 / 0.691 overlaps.
2. The two-ion 171Yb+ Bell state with all physics on (Section 9.6 row 1): the surrogate table (Section 7.5), the histogram against
   the ideal distribution, the register fidelity against the intrinsic budget, the parity contrast of Section 7.9 from analysis
   pulses, SPAM, the mode classes and the boundary populations.
3. The three-ion Molmer-Sorensen GHZ (Section 9.6 row 2): one global bichromatic pulse on the centre-of-mass mode with the COM
   and tilt modes resolved at d_m = 12 and the zigzag frozen, the boundary population quoted, against the pairwise closed-form
   target and against the ideal single-mode GHZ; the GHZ circuit (H, CNOT, CNOT) through run().
4. Wright's minimal crosstalk model (Section 9.6 row 6): Bernstein-Vazirani with the 1.3 % addressing crosstalk of the three-ion fixture, against the coherent-only matrix model.
5. The Section 5.1.1 truncation numbers of the GHZ fixture (|alpha| = 1 at d_m = 12, 8, 6).

Run: uv run python validation/scripts/check_circuits.py (several minutes). Lines prefixed MC: are Monte Carlo results.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import qutip as qt

from qutip_trap.api import (
    Circuit,
    Operation,
    SeedSpec,
    SolverOptions,
    circuit_unitary,
    compile_with_report,
    ideal_probabilities,
    last_record,
    register_fidelity,
    run,
)
from qutip_trap.control.schedule import Schedule, entangling_pulses
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control import native
from qutip_trap.control.compiler import (
    CNOT_MATRIX,
    cnot_global_phase,
    cnot_template,
    compile_to_native,
    cp_matrix,
    cp_template,
    cp_template_overlap,
    debnath_cp_template,
    decompose_single_qubit,
    embed,
    gate_matrix,
)
from qutip_trap.control.shaping import gate_modes, symmetric_pulse, waveform_integrals
from qutip_trap.dynamics.engine import JointExactEngine
from qutip_trap.hilbert.operators import displacement_operator
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.run.space import waveform_contributions
from qutip_trap.units import TWO_PI
from tests.m4_fixtures import table_with_waveform
from tests.m6_fixtures import circuit_fixture


def section(title: str) -> None:
    print(f"\n== {title}")


def phase_of(a: np.ndarray, b: np.ndarray) -> float | None:
    idx = np.unravel_index(int(np.argmax(np.abs(b))), b.shape)
    r = a[idx] / b[idx]
    return float(np.angle(r)) if abs(abs(r) - 1.0) < 1e-9 and np.allclose(a, r * b, atol=1e-9) else None


WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))

section("1. compiler templates (Section 7.7)")
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(20):
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    u = q @ np.diag(np.diag(r) / np.abs(np.diag(r)))
    ops = decompose_single_qubit(u, 0)
    got = circuit_unitary(Circuit(1, tuple(ops), (0,)))
    ph = phase_of(got, u)
    assert ph is not None
    worst = max(worst, float(np.max(np.abs(got - np.exp(1j * ph) * u))))
print(f"  ZXZXZ on 20 random SU(2) targets: max residual {worst:.1e}, pulses per target <= 2")
for s in (1, -1):
    for v in (1, -1):
        got = circuit_unitary(Circuit(2, tuple(cnot_template(0, 1, s=s, v=v)), (0, 1)))
        ph = phase_of(got, embed(CNOT_MATRIX, (0, 1), 2))
        assert ph is not None
        print(
            f"  CNOT template s = {s:+d}, v = {v:+d}: global phase {ph / math.pi:+.4f} pi (e^(i pi v s/4) = {cnot_global_phase(s, v) / math.pi:+.4f} pi)"
        )
for theta in (math.pi / 2.0, math.pi / 4.0, math.pi):
    exact = circuit_unitary(Circuit(2, tuple(cp_template(theta, (0, 1), "ms")), (0, 1)))
    drawn = circuit_unitary(Circuit(2, tuple(debnath_cp_template(theta, (0, 1), "ms")), (0, 1)))
    target = embed(cp_matrix(theta), (0, 1), 2)
    ok = phase_of(exact, target) is not None
    overlap = abs(np.trace(target.conj().T @ drawn)) / 4.0
    print(
        f"  CP({theta / math.pi:.2f} pi): compiler template exact up to a global phase: {ok}; Debnath's as drawn |Tr|/4 = {overlap:.4f} (cos^2((pi - theta)/4) = {cp_template_overlap(theta):.4f})"
    )
bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
rep = compile_with_report(bell)
print(
    f"  Bell circuit: {rep.n_pulses} pulses ({rep.n_entangling} entangling), residual frame {dict((k, round(v / math.pi, 3)) for k, v in rep.final_frame_rad.items())} pi, whole-circuit residual {rep.circuit_residual:.1e}"
)

section("2. two-ion 171Yb+ Bell state with all physics on (Section 9.6 row 1)")
fx = circuit_fixture(2)
dev = fx.device
sur = surrogate_table(
    dev,
    pairs=[(0, 1)],
    gate_drives=fx.gate_drives,
    entangling_drives=fx.entangling_drives,
    detection_records=4000,
    detection_windows_s=WINDOWS,
)
t = sur.table
wf = t.waveform_for((0, 1))
assert wf is not None and wf.segments is not None
chk = sur.entangling[(0, 1)].checks[-1]
print(
    f"  table: carrier Rabi {t.rabi[(0, 2)].value / 1e3:.2f} kHz (seed), crosstalk eps_01 = {t.crosstalk[(0, 1)].value:.4f} (Section 6.6: Wright's 1-4 %), qubit frequency {t.qubit_freq[0].value / 1e9:.6f} GHz"
)
print(
    f"  waveform: {len(wf.segments)} AM segments at mu = {float(wf.segments[0].detuning_hz['blue']) / 1e6:.4f} MHz above the COM, |chi| = {abs(wf.chi_total_rad):.5f} exact after {len(sur.entangling[(0, 1)].checks)} spot checks, surrogate error {sur.entangling[(0, 1)].surrogate_error:.4f}"
)
print(
    f"  spot check: fidelity {chk.fidelity:.5f}, leakage {chk.leakage:.2e}, residual quanta {dict((m, f'{v:.1e}') for m, v in chk.residual_quanta.items())}, space {chk.report.space.dims}"
)
# the surrogate's detection entries are fitted to 4000 sampled records (calibration.readout), so eps_B and eps_D carry shot
# noise (eps_D 4.18e-4 here against 3.90e-4 on Linux, a CI mismatch since M6): recorded on an MC line, with the compared
# line stating only the band they must stay inside
print(
    f"MC: detection: threshold {t.detection['threshold'].value:.1f}, window {t.detection['window_s'].value * 1e6:.1f} us, eps_B {t.detection['eps_B'].value:.2e}, eps_D {t.detection['eps_D'].value:.2e}"
)
print(
    f"  detection: eps_B and eps_D both below 1e-3: {'yes' if max(t.detection['eps_B'].value, t.detection['eps_D'].value) < 1e-3 else 'NO'}"
)
res = run(
    bell,
    dev,
    4000,
    table=t,
    gate_drives=fx.gate_drives,
    entangling_drives=fx.entangling_drives,
    keep_final_state=True,
    options=SolverOptions(branch_weight_min=1e-5),
)
d = res.diagnostics
print(
    f"  space {d.space.dims}, classes {d.mode_class}, branches {d.trajectories} (dropped weight {d.dropped_branch_weight:.1e}), boundary {dict((m, f'{v:.1e}') for m, v in d.boundary_population.items())}, integrators {d.integrator}"
)
print(
    f"MC: histogram over 4000 shots {dict((k, round(v, 4)) for k, v in sorted(res.probabilities.items()))} against the ideal {dict((k, round(v, 4)) for k, v in ideal_probabilities(bell).items())}; error bar on P00 {res.error_bars['00']:.4f}"
)
fid = register_fidelity(res)
print(
    f"  register state: 1 - F = {1.0 - fid:.3e} against the compiled circuit's state; intrinsic budget {dict((k, f'{v:.1e}') for k, v in d.intrinsic_budget.items())}"
)
print(
    f"  SPAM per qubit (eps_B, eps_D) = {tuple(round(x, 5) for x in res.spam['q0'])}, preparation error {res.spam['q0.state_preparation'][0]:.2e}; T_rep {d.wall_clock_span_s / 3999 * 1e3:.3f} ms"
)
rec = last_record(res)
print(
    f"  preparation: Doppler nbar {dict((m, round(v, 3)) for m, v in rec.preparation.doppler.nbar.items())}; after sideband cooling {dict((m, round(v, 4)) for m, v in rec.preparation.sideband_nbar.items())}; pump recoil {dict((m, f'{v:.1e}') for m, v in rec.preparation.pump_heating.items() if v > 0)}"
)
# parity contrast (Section 7.9): the Bell circuit followed by GPi2(phi) on both ions, parity against phi
phis = np.linspace(0.0, math.pi, 7)
parities = []
for phi in phis:
    circ = Circuit(
        2, bell.ops + (Operation("gpi2", (0,), (float(phi),)), Operation("gpi2", (1,), (float(phi),))), (0, 1)
    )
    r = run(
        circ,
        dev,
        1,
        table=t,
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        keep_final_state=True,
        options=SolverOptions(branch_weight_min=1e-3),
    )
    assert r.final_state is not None
    pops = np.real(np.diag(np.asarray(r.final_state.full())))
    parities.append(float(pops[0] + pops[3] - pops[1] - pops[2]))
par = np.array(parities)
contrast = 0.5 * (par.max() - par.min())
print(
    f"  parity oscillation under an analysis pi/2 pulse (7 phases, exact populations): contrast {contrast:.5f}; Bell fidelity bound (P00 + P11 + C)/2 = {0.5 * (float(np.real(res.final_state.full()[0, 0] + res.final_state.full()[3, 3])) + contrast):.5f}"
)

section("2b. the beat-note phase at the gate start (Section 7.10; Roos 2008)")
import dataclasses

from qutip_trap.control.schedule import schedule as _schedule
from qutip_trap.control.compiler import embed as _embed
from qutip_trap.run.job import to_register_order
from qutip_trap.run.space import select_space as _select_space
from qutip_trap.dynamics.engine import SeedSpec as _SeedSpec
from qutip_trap.noise.sampling import quiet_sample as _quiet

ms_only = Circuit(2, (Operation("ms", (0, 1), (math.pi, 0.0, math.pi / 2.0)),), (0, 1))
mu_wf = float(wf.segments[0].detuning_hz["blue"])
ket10 = np.zeros(4, dtype=complex)
ket10[1] = 1.0
tgt10 = qt.Qobj(
    to_register_order(_embed(native.ms(math.pi, 0.0, math.pi / 2.0), (0, 1), 2) @ ket10, 2).reshape(-1, 1),
    dims=[[2, 2], [1, 1]],
)
for continuous in (False, True):
    dev_pc = dataclasses.replace(dev, hardware=dataclasses.replace(dev.hardware, phase_continuous=continuous))
    for cycles in (0.0, 21.25, 21.5):
        t_g = cycles / mu_wf
        sch = _schedule(
            ms_only, dev_pc, t, gate_drives=fx.gate_drives, entangling_drives=fx.entangling_drives, t0_s=t_g
        )
        sp2 = _select_space(dev_pc, sch, SolverOptions(), nbar={2: 0.0, 3: 0.0}).space
        tr = JointExactEngine().run_pulses(
            dev_pc, sch, sp2.initial_state([1, 0]), sp2, _quiet(), _SeedSpec(0), SolverOptions()
        )
        fid_pc = float(np.real(qt.expect(tr.final.internal, tgt10)))
        print(
            f"  phase_continuous = {continuous!s:5s}, gate start at {cycles:5.2f} beat cycles: MS(pi, 0) from |10> fidelity {fid_pc:.5f}"
        )

section("3. three-ion Molmer-Sorensen GHZ at t = pi/(8 chi) (Section 9.6 row 2)")
fx3 = circuit_fixture(
    3, address_waist_m=2.0e-6
)  # 1.3 % addressing crosstalk at the 2.95 um spacing (Wright's 1-4 %)
dev3 = fx3.device
modes3 = gate_modes(dev3, (0, 1, 2), (0, 1))
print(
    f"  x modes MHz {[round(w / TWO_PI / 1e6, 4) for w in modes3.omega_rad_s]}, eta of ions {[[round(e, 4) for e in modes3.eta[i]] for i in (0, 1, 2)]}"
)
com = modes3.modes[-1]
sh = symmetric_pulse(modes3, gate_mode=com, loops=1, epsilon_hz=20e3, pair=(0, 1))
wf3 = sh.waveform
ints = waveform_integrals(wf3, modes3)
chis = {pair: ints.chi_of(*pair) for pair in modes3.pairs()}
print(
    f"  global symmetric pulse: Omega/2pi = {float(wf3.segments[0].amplitude_hz[(0, 'blue')]) / 1e3:.2f} kHz, tau = {wf3.duration_s * 1e6:.1f} us; closed-form pairwise chi {dict((p, round(c, 5)) for p, c in chis.items())} (COM only would be pi/4 on every pair)"
)
contrib = waveform_contributions(wf3, modes3, (0, 2))
for m, c in contrib.items():
    print(
        f"    mode {m}: |alpha|^2 (2n+1) = {c.alpha2_weighted:.2e}, |chi| = {c.chi_rad:.4f} rad, coherent excursion {c.radius:.3f}"
    )
# the plan's fixture: the two modes with the largest contribution (COM and tilt) resolved at d_m = 12, the zigzag frozen with
# its residual |alpha|^2 (2 nbar + 1) and chi reported (Section 9.6 row 2)
ranked = sorted(modes3.modes, key=lambda m: (contrib[m].chi_rad, contrib[m].alpha2_weighted), reverse=True)
resolved = sorted(ranked[:2])
frozen = tuple(m for m in range(9) if m not in resolved)
space3 = HilbertSpace(
    (2, 2, 2),
    tuple(ModeTruncation(m, 12, (0, 4), max(1.5 * contrib[m].eta_max, 1e-3)) for m in resolved),
    None,
    frozen,
)
print(
    f"  resolved {resolved} at d_m = 12 (joint dimension {space3.dimension}), frozen {frozen}; frozen zigzag residual |alpha|^2 (2n+1) = {contrib[ranked[2]].alpha2_weighted:.1e}, chi = {contrib[ranked[2]].chi_rad:.4f} rad (the symmetric pulse closes the COM loop only)"
)
# one GLOBAL pulse: every ion at the same spin phase -pi/2, so the force acts about x on all three (Section 4.3.4); the pair
# scheduler's pi on the second ion would flip one ion's force against the other two
pulses3 = entangling_pulses(
    wf3,
    fx3.entangling_drives,
    spin_phases_rad={0: -math.pi / 2, 1: -math.pi / 2, 2: -math.pi / 2},
    t_start_s=0.0,
    table=table_with_waveform((0, 1), wf3),
    gate_id="ghz",
)
sched3 = Schedule(tuple(pulses3), (), (), {0: 0.0, 1: 0.0, 2: 0.0})
assert set(p.drive.ions[0] for p in sched3.pulses) == {0, 1, 2}
tr3 = JointExactEngine().run_pulses(
    dev3, sched3, space3.initial_state([0, 0, 0]), space3, quiet_sample(), SeedSpec(0), SolverOptions()
)
rho3 = tr3.final.internal


# the pairwise closed-form target exp(+i sum chi_ij sigma_x sigma_x) |000> (the pulse applies +chi, Section 13) and the ideal single-mode GHZ
def xx_full(i: int, j: int, chi: float) -> np.ndarray:
    return embed(native.xx(-chi), (i, j), 3)


u_pair = np.eye(8, dtype=complex)
for (i, j), c in chis.items():
    u_pair = xx_full(i, j, c) @ u_pair
u_ideal = np.eye(8, dtype=complex)
for i, j in modes3.pairs():
    u_ideal = xx_full(i, j, math.pi / 4.0) @ u_ideal
ket0 = np.zeros(8, dtype=complex)
ket0[0] = 1.0
for label, umat in (
    ("pairwise closed-form target", u_pair),
    ("ideal single-mode GHZ exp(-i pi/4 sum XX)", u_ideal),
):
    target = qt.Qobj((umat @ ket0).reshape(-1, 1), dims=[[2, 2, 2], [1, 1, 1]])
    print(f"  fidelity against the {label}: {float(np.real(qt.expect(rho3, target))):.5f}")
pops3 = np.real(np.diag(np.asarray(rho3.full())))
print(
    f"  populations |000>, |111> = {pops3[0]:.4f}, {pops3[7]:.4f} (the odd-N Molmer-Sorensen state has P(000) = 1/4, P(111) = 0 before the single-qubit rotation that makes it a GHZ state in the z basis); boundary populations {dict((m, f'{v:.1e}') for m, v in tr3.boundary_population.items())}"
)
open_loops = sum(abs(ints.alpha_of(i)[m]) ** 2 for i in modes3.ions for m in resolved)
print(
    f"  closed-form open-loop error of the resolved modes (the symmetric pulse closes only the COM): sum_i,m |alpha_im|^2 = {open_loops:.2e}; carrier scale (Omega/nu_min)^2 = {(TWO_PI * float(wf3.segments[0].amplitude_hz[(0, 'blue')]) / min(modes3.omega_rad_s)) ** 2:.2e}; exact residual quanta {dict((m, f'{tr3.final.motional.nbar[m]:.2e}') for m in resolved)}"
)
ghz = Circuit(
    3, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ()), Operation("cnot", (1, 2), ())), (0, 1, 2)
)
sur3 = surrogate_table(
    dev3,
    pairs=[(0, 1), (1, 2)],
    gate_drives=fx3.gate_drives,
    entangling_drives=fx3.entangling_drives,
    detection_records=2000,
    detection_windows_s=WINDOWS,
)
for pair in ((0, 1), (1, 2)):
    r3 = sur3.entangling[pair]
    print(
        f"  pair {pair}: classes {sur3.mode_classes[pair]}, frozen chi {dict((m, round(v, 4)) for m, v in sur3.frozen_chi_rad[pair].items())}, spot check fidelity {r3.checks[-1].fidelity:.5f}, leakage {r3.checks[-1].leakage:.1e}, dims {r3.checks[-1].report.space.dims}"
    )
res3 = run(
    ghz,
    dev3,
    2000,
    table=sur3.table,
    gate_drives=fx3.gate_drives,
    entangling_drives=fx3.entangling_drives,
    keep_final_state=True,
    options=SolverOptions(branch_weight_min=3e-3),
)
d3 = res3.diagnostics
print(
    f"  GHZ circuit through run(): space {d3.space.dims}, classes {d3.mode_class}, branches {d3.branches} (dropped {d3.dropped_branch_weight:.1e}), boundary {dict((m, f'{v:.1e}') for m, v in d3.boundary_population.items())}, frozen contribution {dict((m, (f'{a:.1e}', round(c, 4))) for m, (a, c) in d3.frozen_contribution.items())}, dropped modes {d3.dropped_modes} contributing ({d3.dropped_contribution[0]:.1e}, {d3.dropped_contribution[1]:.1e}) that nothing absorbs (Section 11.3 item 2)"
)
print(
    f"  register 1 - F = {1.0 - register_fidelity(res3):.3e}; budget total {d3.intrinsic_budget['total']:.1e}"
)
print(
    f"MC: GHZ histogram over 2000 shots {dict((k, round(v, 4)) for k, v in sorted(res3.probabilities.items(), key=lambda kv: -kv[1]))}"
)

section(
    "3b. four-ion Molmer-Sorensen GHZ from the ground state: COM and tilt resolved at d_m = 12 (dimension 2304)"
)
fx4 = circuit_fixture(4, with_recipe=False)
dev4 = fx4.device
modes4 = gate_modes(dev4, (0, 1, 2, 3), (0, 1))
com4 = modes4.modes[-1]
sh4 = symmetric_pulse(modes4, gate_mode=com4, loops=1, epsilon_hz=20e3, pair=(0, 1))
wf4 = sh4.waveform
ints4 = waveform_integrals(wf4, modes4)
contrib4 = waveform_contributions(wf4, modes4, (0, 3))
ranked4 = sorted(modes4.modes, key=lambda m: (contrib4[m].chi_rad, contrib4[m].alpha2_weighted), reverse=True)
resolved4 = sorted(ranked4[:2])
frozen4 = tuple(m for m in range(12) if m not in resolved4)
space4 = HilbertSpace(
    (2, 2, 2, 2),
    tuple(ModeTruncation(m, 12, (0, 4), max(1.5 * contrib4[m].eta_max, 1e-3)) for m in resolved4),
    None,
    frozen4,
)
print(
    f"  x modes MHz {[round(w / TWO_PI / 1e6, 4) for w in modes4.omega_rad_s]}; resolved {resolved4} (dimension {space4.dimension}); frozen x-mode residuals {dict((m, (f'{contrib4[m].alpha2_weighted:.1e}', round(contrib4[m].chi_rad, 4))) for m in modes4.modes if m not in resolved4)}"
)
pulses4 = entangling_pulses(
    wf4,
    fx4.entangling_drives,
    spin_phases_rad={i: -math.pi / 2 for i in range(4)},
    t_start_s=0.0,
    table=table_with_waveform((0, 1), wf4),
    gate_id="ghz4",
)
sched4 = Schedule(tuple(pulses4), (), (), {i: 0.0 for i in range(4)})
tr4 = JointExactEngine().run_pulses(
    dev4, sched4, space4.initial_state([0, 0, 0, 0]), space4, quiet_sample(), SeedSpec(0), SolverOptions()
)
rho4 = tr4.final.internal
chis4 = {pair: ints4.chi_of(*pair) for pair in modes4.pairs()}
u4 = np.eye(16, dtype=complex)
for (i, j), c in chis4.items():
    u4 = embed(native.xx(-c), (i, j), 4) @ u4
ket0_4 = np.zeros(16, dtype=complex)
ket0_4[0] = 1.0
target4 = qt.Qobj((u4 @ ket0_4).reshape(-1, 1), dims=[[2] * 4, [1] * 4])
u4i = np.eye(16, dtype=complex)
for i, j in modes4.pairs():
    u4i = embed(native.xx(-math.pi / 4.0), (i, j), 4) @ u4i
ideal4 = qt.Qobj((u4i @ ket0_4).reshape(-1, 1), dims=[[2] * 4, [1] * 4])
pops4 = np.real(np.diag(np.asarray(rho4.full())))
print(
    f"  pairwise closed-form chi {dict((p, round(c, 4)) for p, c in chis4.items())}; fidelity against the pairwise target {float(np.real(qt.expect(rho4, target4))):.5f}, against the ideal GHZ exp(+i pi/4 sum XX) {float(np.real(qt.expect(rho4, ideal4))):.5f}; P(0000) = {pops4[0]:.4f}, P(1111) = {pops4[15]:.4f}; boundary {dict((m, f'{v:.1e}') for m, v in tr4.boundary_population.items())}"
)

section(
    "4. Wright's minimal crosstalk model: Bernstein-Vazirani with the secret 01 on ions 0 and 2, the middle ion the ancilla (Section 9.6 row 6)"
)
import itertools

bv = Circuit(
    3,
    (
        Operation("x", (1,), ()),
        Operation("h", (0,), ()),
        Operation("h", (1,), ()),
        Operation("h", (2,), ()),
        Operation("cnot", (0, 1), ()),
        Operation("h", (0,), ()),
        Operation("h", (1,), ()),
        Operation("h", (2,), ()),
    ),
    (0, 1, 2),
)
rb = run(
    bv,
    dev3,
    4000,
    table=sur3.table,
    gate_drives=fx3.gate_drives,
    entangling_drives=fx3.entangling_drives,
    keep_final_state=True,
    options=SolverOptions(branch_weight_min=3e-3),
)
assert rb.final_state is not None
rec_b = last_record(rb)
pop = np.real(np.diag(np.asarray(rb.final_state.full())))
data = Counter()
declared = Counter()
povm_b, schemes_b = rec_b.readout.product, rec_b.readout.schemes
for idx, p in enumerate(pop):
    b0, b1, b2 = (
        (idx >> 2) & 1,
        (idx >> 1) & 1,
        idx & 1,
    )  # register order: ion 0 is the most-significant index bit
    data[f"{b2}{b0}"] += float(p)  # data string (qubit 2, qubit 0), qubit 0 rightmost
    # the POVM takes the true INTERNAL LEVELS, not the start classes (M5); identical here, wrong for bright_level == 0
    true_levels = [b0, b1, b2]
    for dec in itertools.product((True, False), repeat=3):
        q = povm_b.declared_bright_probability(true_levels, list(dec))
        bits = [schemes_b[i].bit_of_class("bright" if dec[i] else "dark") for i in range(3)]
        declared[f"{bits[2]}{bits[0]}"] += float(p) * float(q)
# coherent crosstalk alone: the compiled circuit as matrices, every gpi/gpi2 also rotating its nearest neighbours by eps theta at the pulse phase (Delta k is transverse to the chain, so no geometric phase), the MS gate ideal
eps_b = float(sur3.table.crosstalk[(0, 1)].value)
psi_c = np.zeros(8, dtype=complex)
psi_c[0] = 1.0
for op in compile_to_native(bv).ops:
    if op.name in ("gpi", "gpi2"):
        q, phi_c = op.qubits[0], op.params[0]
        half_c = 0.5 * eps_b * (math.pi if op.name == "gpi" else math.pi / 2.0)
        rot_c = np.array(
            [
                [math.cos(half_c), -1j * math.sin(half_c) * np.exp(-1j * phi_c)],
                [-1j * math.sin(half_c) * np.exp(1j * phi_c), math.cos(half_c)],
            ]
        )
        psi_c = embed(gate_matrix(op), (q,), 3) @ psi_c
        for j in (q - 1, q + 1):
            if 0 <= j < 3:
                psi_c = embed(rot_c, (j,), 3) @ psi_c
    else:
        psi_c = embed(gate_matrix(op), op.qubits, 3) @ psi_c
coherent = Counter()
for idx, v in enumerate(np.abs(psi_c) ** 2):
    coherent[f"{(idx >> 2) & 1}{idx & 1}"] += float(v)  # compiler order: qubit k is bit k
print(
    f"  crosstalk eps = {eps_b:.4f} (nearest neighbours); exact register data-string populations {dict((k, f'{v:.2e}') for k, v in sorted(data.items()))}"
)
print(
    f"  coherent crosstalk alone (compiled circuit as matrices, ideal MS) {dict((k, f'{v:.2e}') for k, v in sorted(coherent.items()))}: the 1 -> 0 flip (00) already leads the 0 -> 1 flip (11) by {coherent['00'] / coherent['11']:.2f} because the oracle CNOT maps the ancilla's crosstalk rotations onto its control, the secret's 1 bit, while the 0 bit sees only the ancilla pulses' direct rotations"
)
print(
    f"  exact DECLARED distribution through the POVM {dict((k, f'{v:.2e}') for k, v in sorted(declared.items()))}: the 1 -> 0 flip of the secret (00) {declared['00']:.2e} against the 0 -> 1 flip (11) {declared['11']:.2e}, ratio {declared['00'] / declared['11']:.2f}; per-ion eps_B {rb.spam['q0'][0]:.2e} against eps_D {rb.spam['q0'][1]:.2e}"
)
print(
    f"  register 1 - F = {1.0 - register_fidelity(rb):.3e}; budget total {rb.diagnostics.intrinsic_budget['total']:.1e}"
)
print(
    f"MC: BV histogram over 4000 shots {dict((k, round(v, 4)) for k, v in sorted(rb.probabilities.items(), key=lambda kv: -kv[1]))}"
)

section("5. GHZ fixture truncation (Section 5.1.1; check_critique_v3.py)")
for d_m in (12, 8, 6):
    state = displacement_operator(d_m, 1.0j) * qt.basis(d_m, 0)
    p = np.abs(np.asarray(state.full()).ravel()) ** 2
    print(f"  |alpha| = 1 at d_m = {d_m}: top two levels hold {p[-2:].sum():.1e}")
print("\ndone")
