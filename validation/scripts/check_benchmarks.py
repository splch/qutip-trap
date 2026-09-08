"""Benchmark emulation (PLAN.md milestone M10; Sections 7.9, 9.12, 13 "RB error rate", 6.8, 9.6), through the package.

1. The Clifford groups behind randomized benchmarking: |C1| = 24 by closure, |C2| = 11520 by closure, the four entangling
   classes with their stabilizers (576, 64, 64, 576) and sizes (576, 5184, 5184, 576), 1.5 entangling gates per Clifford, the
   class recognition of random elements and the closure of random sequences by the inverse of their product.
2. The KAK decomposition of two-qubit unitaries (control/two_qubit.py): the named gates' canonical classes, Haar-random SU(4)
   in the Weyl chamber at three entangling gates, and the ideal heavy-output statistics of random two-qubit square circuits.
3. Single-qubit randomized benchmarking on the two-ion 171Yb+ example device (device/presets.py) with the budget alongside:
   the fitted decay, r = (1 - p)/2, the Section 6.8 channels of gpi2 and gpi reduced to the benchmarked qubit, the SPAM
   offsets; then simultaneous RB on both qubits.
4. Two-qubit randomized benchmarking on the pair, r = (3/4)(1 - p), the ms channel from GATE_LOCAL tomography.
5. GHZ fidelity on two and three ions: populations, the parity scan, the bound (P_0 + P_1 + C)/2, the exact register fidelity.
6. A quantum-volume style run at width two: heavy-output probabilities against the ideal ones and the depolarizing prediction.

Run: uv run python validation/scripts/check_benchmarks.py (about twenty minutes). Lines prefixed MC: carry shot noise.
"""

from __future__ import annotations

import math
import time

import numpy as np

from qutip_trap.api import (
    SolverOptions,
    decompose_two_qubit_clifford,
    ghz_fidelity,
    haar_random_unitary,
    kak_decomposition,
    quantum_volume,
    random_square_circuit,
    random_two_qubit_clifford,
    randomized_benchmarking,
    yb171_chain,
)
from qutip_trap.benchmarks.clifford import (
    CLASS_SIZES,
    CORES,
    ENTANGLING_COUNT,
    SINGLE_QUBIT_CLIFFORDS,
    TWO_QUBIT_GROUP_ORDER,
    operations_unitary,
    pauli_frame_of,
    stabilizer_size,
    two_qubit_clifford_group,
)
from qutip_trap.benchmarks.rb import two_qubit_sequence
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import CNOT_MATRIX, SWAP_MATRIX, cp_matrix
from qutip_trap.control.two_qubit import global_phase


def section(title: str) -> None:
    print(f"\n== {title}")


def fmt(v: tuple[float, float], digits: int = 4) -> str:
    return f"{v[0]:.{digits}f} +- {v[1]:.{digits}f}"


section("1. the Clifford groups (Section 7.9; Section 13 'RB error rate')")
group = two_qubit_clifford_group()
print(
    f"  |C1| = {len(SINGLE_QUBIT_CLIFFORDS)} by closure of {{H, S}}; |C2| = {len(group)} by closure of {{H, S, CNOT}} (11520)"
)
stab = {c: stabilizer_size(c) for c in CORES}
sizes = {c: 576 * 576 // stab[c] for c in CORES}
print(
    f"  stabilizers |L n g L g^-1| {stab}; class sizes 576^2/stabilizer {sizes} (sum {sum(sizes.values())})"
)
counts = {c: 0 for c in CORES}
for m in group:
    counts[pauli_frame_of(m)] += 1
print(f"  classes by the symplectic invariant over the whole group {counts}")
avg = sum(ENTANGLING_COUNT[c] * n for c, n in CLASS_SIZES.items()) / TWO_QUBIT_GROUP_ORDER
print(f"  entangling gates per Clifford: identity 0, cnot 1, iswap 2, swap 3; group average {avg:.4f}")
rng = np.random.default_rng(0)
ok = 0
for _ in range(200):
    c = random_two_qubit_clifford(rng)
    d = decompose_two_qubit_clifford(c.matrix)
    ok += int(d.core == c.core and global_phase(d.matrix, c.matrix) is not None)
print(f"  class recognition of 200 random elements: {ok} reproduced with their class")
closed = 0
for _ in range(50):
    seq = two_qubit_sequence(rng, (0, 1), 6, 2)
    closed += int(
        global_phase(operations_unitary(list(seq.circuit.ops), (0, 1)), np.eye(4), atol=1e-8) is not None
    )
print(
    f"  50 random six-Clifford sequences closed by the inverse of their product: {closed} return the identity"
)

section("2. the KAK decomposition and the ideal heavy outputs (control/two_qubit.py; Cross et al. 2019)")
iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
for name, u in (
    ("CNOT", CNOT_MATRIX),
    ("CZ", cp_matrix(math.pi)),
    ("iSWAP", iswap),
    ("SWAP", SWAP_MATRIX),
    ("CP(0.3)", cp_matrix(0.3)),
):
    k = kak_decomposition(u)
    a, b, c = k.coefficients
    print(
        f"  {name}: canonical (a, b, c)/pi = ({a / math.pi:.4f}, {b / math.pi:.4f}, {c / math.pi:.4f}), {k.entangling_count} entangling gate(s), residual {np.max(np.abs(k.matrix() - u)):.1e}"
    )
worst = 0.0
three = 0
for _ in range(200):
    u = haar_random_unitary(rng, 4)
    k = kak_decomposition(u)
    worst = max(worst, float(np.max(np.abs(k.matrix() - u))))
    three += int(k.entangling_count == 3)
print(f"  200 Haar-random SU(4): {three} at three entangling gates, worst residual {worst:.1e}")
hs = [random_square_circuit(rng, (0, 1), 2, 2).heavy_ideal_probability for _ in range(200)]
print(
    f"  ideal heavy-output probability of 200 random width-2 depth-2 circuits: mean {np.mean(hs):.4f}, std {np.std(hs):.4f} (the large-width asymptote (1 + ln 2)/2 = {(1 + math.log(2)) / 2:.4f})"
)

section("3. single-qubit randomized benchmarking on the two-ion example device (Sections 7.9, 9.12, 13)")
preset = yb171_chain(2)
dev = preset.device
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
t0 = time.perf_counter()
sur = surrogate_table(
    dev,
    pairs=[(0, 1)],
    gate_drives=preset.gate_drives,
    entangling_drives=preset.entangling_drives,
    detection_records=2000,
    detection_windows_s=WINDOWS,
)
print(f"MC: surrogate table in {time.perf_counter() - t0:.1f} s")
kw = dict(table=sur.table, options=SolverOptions(branch_weight_min=1e-3), **preset.run_kwargs())
t0 = time.perf_counter()
rb1 = randomized_benchmarking(
    dev, (0,), (1, 128, 512, 2048), n_sequences=3, shots=4000, budget=True, fix_offset=True, **kw
)
print(f"MC: wall {time.perf_counter() - t0:.1f} s for {len(rb1.results)} runs of 4000 shots")
print(
    f"MC: mean survival per length {dict(zip(rb1.lengths, [round(float(x), 5) for x in rb1.mean_survival]))} +- {[round(float(x), 5) for x in rb1.mean_sigma]}"
)
print(
    f"MC: fit A p^m + B: A {fmt(rb1.fit['A'], 5)}, p {fmt(rb1.fit['p'], 6)}, B {fmt(rb1.fit['B'], 5)}, chi2/dof {rb1.fit['chi2_per_dof'][0]:.3f}; {rb1.fidelity_form()}"
)
print(
    f"MC: error per Clifford r = (1 - p)/2 = {rb1.error_per_clifford[0]:.2e} +- {rb1.error_per_clifford[1]:.1e}; depolarizing entanglement infidelity (3/4)(1 - p) = {rb1.depolarizing_entanglement_infidelity[0]:.2e} (Section 13: not the RB error rate)"
)
b = rb1.budget
assert b is not None
print(
    f"  pulses per Clifford {rb1.pulses_per_clifford:.3f}; native pieces per Clifford {dict((k, round(v, 3)) for k, v in sorted(b.counts.items()))}"
)
print(
    f"  Section 6.8 channels reduced to qubit 0: {dict((k, f'{v:.3e}') for k, v in sorted(b.channel_infidelity.items()))}"
)
for k, ch in sorted(b.channels.items()):
    for s in ch.steps:
        print(
            f"    {k}: step ions {s.ions}, full-step average gate infidelity {s.summary.average_gate_infidelity:.3e}, depolarizing rate {s.summary.depolarizing_rate:.3e}, {s.engine_runs} engine runs on dimension {s.local_dimension}"
        )
print(
    f"  predicted r_channel {b.predicted['r_channel']:.3e} (first-order composition), r_intrinsic {b.predicted['r_intrinsic']:.3e} (Section 9.6 scales), F(0) from SPAM {b.predicted['F0_spam']:.5f}; SPAM {dict((k, (round(v[0], 6), round(v[1], 6))) for k, v in b.spam.items())}"
)
t0 = time.perf_counter()
rb_sim = randomized_benchmarking(
    dev, (0, 1), (1, 128, 512), n_sequences=2, shots=2000, budget=True, pair=False, fix_offset=True, **kw
)

print(
    f"MC: simultaneous single-qubit RB on both ions (survival = P(00)), wall {time.perf_counter() - t0:.1f} s: mean survival {dict(zip(rb_sim.lengths, [round(float(x), 5) for x in rb_sim.mean_survival]))}, p {fmt(rb_sim.fit['p'], 6)}, r = (3/4)(1 - p) {rb_sim.error_per_clifford[0]:.2e} +- {rb_sim.error_per_clifford[1]:.1e} (two Cliffords per unit)"
)
bs = rb_sim.budget
assert bs is not None
print(
    f"  simultaneous RB budget on the pair: channels {dict((k, f'{v:.3e}') for k, v in sorted(bs.channel_infidelity.items()))}; predicted r_channel {bs.predicted['r_channel']:.3e} (the neighbour's crosstalk rotation now counts, Section 6.6) against {bs.predicted['r_channel'] / 2:.1e} per Clifford; single-ion RB above saw {b.predicted['r_channel']:.1e}"
)

section("4. two-qubit randomized benchmarking on the pair (Section 7.9; Section 13)")
t0 = time.perf_counter()
rb2 = randomized_benchmarking(
    dev, (0, 1), (1, 6, 16), n_sequences=2, shots=400, budget=True, fix_offset=True, **kw
)
print(f"MC: wall {time.perf_counter() - t0:.1f} s for {len(rb2.results)} runs")
print(
    f"MC: mean survival per length {dict(zip(rb2.lengths, [round(float(x), 4) for x in rb2.mean_survival]))} +- {[round(float(x), 4) for x in rb2.mean_sigma]}"
)
print(
    f"MC: fit A {fmt(rb2.fit['A'])}, p {fmt(rb2.fit['p'], 5)}, B {fmt(rb2.fit['B'])}; r = (3/4)(1 - p) = {rb2.error_per_clifford[0]:.2e} +- {rb2.error_per_clifford[1]:.1e}; entanglement infidelity (15/16)(1 - p) = {rb2.depolarizing_entanglement_infidelity[0]:.2e}"
)
b2 = rb2.budget
assert b2 is not None
print(
    f"  pulses per Clifford {rb2.pulses_per_clifford:.3f}, entangling per Clifford {rb2.entangling_per_clifford:.3f}; pieces {dict((k, round(v, 3)) for k, v in sorted(b2.counts.items()))}"
)
print(f"  channels on the pair: {dict((k, f'{v:.3e}') for k, v in sorted(b2.channel_infidelity.items()))}")
ms_ch = b2.channels["ms[0,1]"].steps[0]
print(
    f"  ms[0,1] step: dimension {ms_ch.local_dimension}, {ms_ch.engine_runs} engine runs, average gate infidelity {ms_ch.summary.average_gate_infidelity:.3e}, depolarizing rate {ms_ch.summary.depolarizing_rate:.3e}, twirl p_II {ms_ch.summary.pauli_twirled['II']:.5f}"
)
print(
    f"  predicted r_channel {b2.predicted['r_channel']:.3e}, r_intrinsic {b2.predicted['r_intrinsic']:.3e}, F(0) from SPAM {b2.predicted['F0_spam']:.5f}"
)

section("5. GHZ fidelity (Section 7.9 'Entangling gate'; Section 9.6 rows 1 and 2)")
t0 = time.perf_counter()
g2 = ghz_fidelity(dev, (0, 1), shots=1000, budget=True, **kw)
print(
    f"MC: two ions, wall {time.perf_counter() - t0:.1f} s: P00 {fmt(g2.populations['P0'])}, P11 {fmt(g2.populations['P1'])}, parity contrast {fmt(g2.fit['contrast'])} (chi2/dof {g2.fit['chi2_per_dof'][0]:.2f}), fidelity bound (P0 + P1 + C)/2 = {fmt(g2.fidelity_bound)}"
)
print(
    f"  exact register fidelity {g2.register_fidelity:.5f}; predicted from the channels: F_gates {g2.budget.predicted['F_gates']:.5f}, bound {g2.budget.predicted['fidelity_bound']:.5f}; intrinsic scales {g2.budget.predicted['intrinsic_total']:.2e}"
)  # type: ignore[union-attr]
print(
    f"MC: parity points (phase/pi, parity): {[(round(float(r[0]) / math.pi, 3), round(float(r[1]), 3)) for r in g2.parity]}"
)
preset3 = yb171_chain(3, address_waist_m=2.0e-6)
dev3 = preset3.device
t0 = time.perf_counter()
sur3 = surrogate_table(
    dev3,
    pairs=[(0, 1), (1, 2)],
    gate_drives=preset3.gate_drives,
    entangling_drives=preset3.entangling_drives,
    detection_records=1500,
    detection_windows_s=WINDOWS,
)
kw3 = dict(table=sur3.table, options=SolverOptions(branch_weight_min=3e-3), **preset3.run_kwargs())
g3 = ghz_fidelity(
    dev3,
    (0, 1, 2),
    shots=400,
    analysis_phases_rad=np.linspace(0.0, 2.0 * math.pi / 3.0, 5, endpoint=False),
    budget=False,
    **kw3,
)
print(
    f"MC: three ions (dimension {int(np.prod(g3.results[0].diagnostics.space.dims))}), wall {time.perf_counter() - t0:.1f} s with the surrogate: P000 {fmt(g3.populations['P0'])}, P111 {fmt(g3.populations['P1'])}, contrast {fmt(g3.fit['contrast'])}, bound {fmt(g3.fidelity_bound)}"
)
print(
    f"  exact register fidelity {g3.register_fidelity:.5f}; intrinsic budget total {g3.results[0].diagnostics.intrinsic_budget['total']:.2e}; mode classes {g3.results[0].diagnostics.mode_class}"
)

section("6. quantum-volume style run at width two (Cross et al. 2019)")
t0 = time.perf_counter()
qv = quantum_volume(dev, (0, 1), n_circuits=4, shots=400, budget=True, **kw)
print(
    f"MC: wall {time.perf_counter() - t0:.1f} s for {qv.n_circuits} circuits of depth {qv.depth} ({qv.entangling_per_circuit:.1f} entangling gates and {qv.pulses_per_circuit:.1f} pulses per circuit)"
)
print(
    f"MC: heavy-output probabilities {[round(float(x), 4) for x in qv.heavy_output_probability]} +- {[round(float(x), 4) for x in qv.heavy_sigma]} against the ideal {[round(float(x), 4) for x in qv.ideal_heavy_probability]}"
)
print(
    f"MC: mean {qv.mean:.4f} +- {qv.sigma:.4f}; mean - 2 sigma > 2/3: {qv.passed} (protocol circuit count met: {qv.protocol_circuit_count_met})"
)
print(f"  exact register fidelities {[round(float(x), 5) for x in qv.register_fidelity]}")
print(
    f"  predicted: eps_gates {qv.budget.predicted['eps_gates']:.3e}, eps_readout {qv.budget.predicted['eps_readout']:.3e}, heavy-output probability {qv.budget.predicted['heavy_output_probability']:.4f} against the ideal mean {qv.budget.predicted['ideal_heavy_output_probability']:.4f}; intrinsic scales per circuit {qv.budget.predicted['intrinsic_total']:.2e}"
)  # type: ignore[union-attr]

print("\ndone")
