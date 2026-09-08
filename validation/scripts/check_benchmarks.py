"""Benchmark emulation (PLAN.md milestone M10; Sections 7.9, 9.12, 13 "RB error rate", 6.8, 9.6), through the package.

1. The Clifford groups behind randomized benchmarking: |C1| = 24 by closure, |C2| = 11520 by closure, the four entangling
   classes with their stabilizers (576, 64, 64, 576) and sizes (576, 5184, 5184, 576), 1.5 entangling gates per Clifford, the
   class recognition of random elements and the closure of random sequences by the inverse of their product.
2. The KAK decomposition of two-qubit unitaries (control/two_qubit.py): the named gates' canonical classes, Haar-random SU(4)
   in the Weyl chamber at three entangling gates, and the ideal heavy-output statistics of random two-qubit square circuits.
3. Single-qubit randomized benchmarking on the two-ion 171Yb+ example device (device/presets.py) with the budget alongside:
   the fitted decay, r = (1 - p)/2, the Section 6.8 channels of gpi2 and gpi reduced to the benchmarked qubit, the SPAM
   offsets; then simultaneous RB on both qubits (the per-qubit marginal fits of Gambetta et al. 2012 beside the isolated r,
   the joint decay as the correlation diagnostic); then the Knill-style variant of Section 7.9 fitted as B p^L + 1/2.
4. Two-qubit randomized benchmarking on the pair, r = (3/4)(1 - p), the ms channel from GATE_LOCAL tomography.
5. GHZ fidelity on two and three ions: populations, the parity scan, the bound (P_0 + P_1 + C)/2, and the two exact
   fidelities it sits between (max_theta <GHZ_theta| rho |GHZ_theta>, which it estimates, and the fixed-phase one).
6. A quantum-volume style run at width two: heavy-output probabilities against the ideal ones, Cross et al.'s Eq. (32)
   confidence beside the standard error of the mean, and the depolarizing prediction.

Run: uv run python validation/scripts/check_benchmarks.py (about twenty minutes). Lines prefixed MC: carry shot noise and
CI does not compare them; every headline Monte-Carlo number therefore also gets a compared ``pinned`` line, which states
only the band the number must stay inside, so that CI fails when an anchor moves out of it (``pinned`` below).
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


def pinned(label: str, value: float, low: float, high: float) -> None:
    """One compared (non-MC) line per headline Monte-Carlo number: ``run_checks.py`` skips every line containing ``MC:``
    and compares the numeric tokens of every other line at a relative tolerance of 1e-9, so this line prints only the
    band's two constants and a WORD that says whether the number is inside it. The word changes the line's skeleton, so a
    fitted p, r, GHZ bound or heavy-output probability that leaves its band no longer matches the committed line and CI
    fails -- which is how the anchors of ``anchor.m10.*`` are pinned in CI without comparing shot noise."""
    verdict = "inside" if low <= value <= high else "OUTSIDE"
    print(f"  pinned {label}: {verdict} the band [{low:g}, {high:g}]")


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
    f"  predicted r_channel {b.predicted['r_channel']:.3e} (first-order composition), r_intrinsic {b.predicted['r_intrinsic']:.3e} (Section 9.6 scales, summed over each kind's whole schedule entry and so over a larger error than r_channel), F(0) from SPAM {b.predicted['F0_spam']:.5f}; SPAM {dict((k, (round(v[0], 6), round(v[1], 6))) for k, v in b.spam.items())}"
)
r_alone = rb1.error_per_clifford[0]
pinned("single-qubit RB r per Clifford", r_alone, 1.0e-5, 4.0e-5)
pinned("single-qubit RB r_channel", b.predicted["r_channel"], 1.5e-5, 4.0e-5)
pinned("single-qubit RB r/r_channel", r_alone / b.predicted["r_channel"], 0.5, 1.2)
pinned(
    "single-qubit RB F(0) minus the SPAM prediction",
    rb1.mean_survival[0] - b.predicted["F0_spam"],
    -3e-4,
    3e-4,
)
pinned(
    "single-qubit RB r_intrinsic/r_channel", b.predicted["r_intrinsic"] / b.predicted["r_channel"], 8.0, 40.0
)
t0 = time.perf_counter()
rb_sim = randomized_benchmarking(
    dev, (0, 1), (1, 128, 512), n_sequences=2, shots=2000, budget=True, pair=False, fix_offset=True, **kw
)

print(
    f"MC: simultaneous single-qubit RB on both ions, wall {time.perf_counter() - t0:.1f} s: per-qubit MARGINAL survival"
    f" {[[round(float(x), 5) for x in rb_sim.marginal_survival[q].mean(axis=1)] for q in range(2)]} at m = {list(rb_sim.lengths)},"
    f" marginal p {[round(f['p'][0], 6) for f in rb_sim.marginal_fit]}, marginal r_q = (1 - p_q)/2"
    f" {[f'{v:.2e}' for v, _ in rb_sim.marginal_error_per_clifford]} against {r_alone:.2e} for the ion benchmarked alone"
    f" (Gambetta et al. 2012); r = mean_q r_q = {rb_sim.error_per_clifford[0]:.2e} +- {rb_sim.error_per_clifford[1]:.1e}"
)
assert rb_sim.joint_error_per_layer is not None
print(
    f"MC: the same run's JOINT survival P(00) {dict(zip(rb_sim.lengths, [round(float(x), 5) for x in rb_sim.mean_survival]))},"
    f" p {fmt(rb_sim.fit['p'], 6)}, (1 - p)(2^n - 1)/2^n = {rb_sim.joint_error_per_layer[0]:.2e} +-"
    f" {rb_sim.joint_error_per_layer[1]:.1e} per LAYER of two Cliffords (the correlation diagnostic, sum_q r_q to first order)"
)
bs = rb_sim.budget
assert bs is not None
print(
    f"  simultaneous RB budget: channels on the pair {dict((k, f'{v:.3e}') for k, v in sorted(bs.channel_infidelity.items()))};"
    f" predicted per-qubit r_channel {dict((k.split('.')[1], f'{v:.3e}') for k, v in sorted(bs.predicted.items()) if k.startswith('r_channel.'))}"
    f" (each kind's channel reduced to that ONE qubit, so the neighbour's crosstalk rotation counts, Section 6.6), mean"
    f" {bs.predicted['r_channel']:.3e} against the measured {rb_sim.error_per_clifford[0]:.3e}; r_channel_joint_layer"
    f" {bs.predicted['r_channel_joint_layer']:.3e} against the measured joint {rb_sim.joint_error_per_layer[0]:.3e};"
    f" single-ion RB above saw {b.predicted['r_channel']:.1e}"
)
pinned("simultaneous RB marginal r_q", rb_sim.error_per_clifford[0], 1.5e-4, 6.0e-4)
pinned(
    "simultaneous RB marginal r_q / r_channel",
    rb_sim.error_per_clifford[0] / bs.predicted["r_channel"],
    0.5,
    2.0,
)
pinned("simultaneous RB joint r per layer", rb_sim.joint_error_per_layer[0], 4.0e-4, 1.2e-3)
pinned(
    "simultaneous RB joint r / r_channel_joint_layer",
    rb_sim.joint_error_per_layer[0] / bs.predicted["r_channel_joint_layer"],
    0.5,
    2.0,
)
pinned("simultaneous RB marginal r_q / the isolated r", rb_sim.error_per_clifford[0] / r_alone, 5.0, 40.0)
t0 = time.perf_counter()
rb_knill = randomized_benchmarking(
    dev, (0,), (1, 128, 512), n_sequences=2, shots=2000, budget=False, variant="knill", **kw
)
print(
    f"MC: Knill-style RB on ion 0 (Section 7.9; a random Pauli then a random Clifford per computational gate, one final"
    f" pi/2), wall {time.perf_counter() - t0:.1f} s: survival of the sequence's own target"
    f" {dict(zip(rb_knill.lengths, [round(float(x), 5) for x in rb_knill.mean_survival]))}, {rb_knill.fidelity_form()},"
    f" p {fmt(rb_knill.fit['p'], 6)}, r = (1 - p)/2 = {rb_knill.error_per_clifford[0]:.2e} +-"
    f" {rb_knill.error_per_clifford[1]:.1e} per computational gate, {rb_knill.pulses_per_clifford:.3f} pulses per gate"
)
print(
    f"  Wright et al. 2019's operating point (p ~ 0.995, B + 1/2 ~ 0.993) is a hundred times this device's error and"
    f" cannot be configured from his paper (anchor.m10.knill_style_rb): here p = {rb_knill.fit['p'][0]:.6f} and"
    f" B + 1/2 = {rb_knill.fit['A'][0] + rb_knill.fit['B'][0]:.4f}"
)
pinned("Knill-style RB r per computational gate", rb_knill.error_per_clifford[0], 5.0e-6, 5.0e-4)
pinned("Knill-style RB pulses per computational gate", rb_knill.pulses_per_clifford, 1.3, 1.8)

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
pinned("two-qubit RB r per Clifford", rb2.error_per_clifford[0], 1.0e-3, 6.0e-3)
pinned("two-qubit RB r/r_channel", rb2.error_per_clifford[0] / b2.predicted["r_channel"], 0.5, 2.5)
pinned("two-qubit RB entangling gates per Clifford", rb2.entangling_per_clifford, 1.0, 2.0)
pinned("ms[0,1] average gate infidelity", ms_ch.summary.average_gate_infidelity, 5.0e-5, 3.0e-4)

section("5. GHZ fidelity (Section 7.9 'Entangling gate'; Section 9.6 rows 1 and 2)")
t0 = time.perf_counter()
g2 = ghz_fidelity(dev, (0, 1), shots=1000, budget=True, **kw)
print(
    f"MC: two ions, wall {time.perf_counter() - t0:.1f} s: P00 {fmt(g2.populations['P0'])}, P11 {fmt(g2.populations['P1'])}, parity contrast {fmt(g2.fit['contrast'])} (chi2/dof {g2.fit['chi2_per_dof'][0]:.2f}), fidelity bound (P0 + P1 + C)/2 = {fmt(g2.fidelity_bound)}"
)
print(
    f"  exact fidelities: max_theta <GHZ_theta| rho |GHZ_theta> {g2.register_fidelity_max_phase:.5f} (what the bound"
    f" estimates) and the fixed-phase <GHZ| rho |GHZ> {g2.register_fidelity:.5f} (which the bound therefore exceeds);"
    f" predicted from the channels: F_gates {g2.budget.predicted['F_gates']:.5f}, bound"
    f" {g2.budget.predicted['fidelity_bound']:.5f}; intrinsic scales {g2.budget.predicted['intrinsic_total']:.2e}"
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
    f"  exact fidelities: max_theta {g3.register_fidelity_max_phase:.5f}, fixed-phase {g3.register_fidelity:.5f};"
    f" intrinsic budget total {g3.results[0].diagnostics.intrinsic_budget['total']:.2e}; mode classes"
    f" {g3.results[0].diagnostics.mode_class}"
)
for tag, g in (("two-ion", g2), ("three-ion", g3)):
    pinned(f"{tag} GHZ bound (P0 + P1 + C)/2", g.fidelity_bound[0], 0.97, 1.01)
    pinned(f"{tag} GHZ exact max-phase fidelity", g.register_fidelity_max_phase, 0.97, 1.0)
    pinned(
        f"{tag} GHZ bound minus the exact max-phase fidelity (three sigma of the bound)",
        abs(g.fidelity_bound[0] - g.register_fidelity_max_phase) / (3.0 * g.fidelity_bound[1]),
        0.0,
        1.0,
    )
    pinned(
        f"{tag} GHZ bound minus the fixed-phase fidelity (the bound is above it)",
        g.fidelity_bound[0] - g.register_fidelity,
        -0.005,
        0.05,
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
    f"MC: mean {qv.mean:.4f}; Cross et al. Eq. (32) sigma = sqrt(h(1 - h)/n_c) = {qv.sigma:.4f} so mean - 2 sigma ="
    f" {qv.mean - 2.0 * qv.sigma:.4f} and threshold_cleared = {qv.threshold_cleared}; the standard error of the mean over"
    f" circuits {qv.standard_error_of_the_mean:.4f} (a diagnostic, {qv.sigma / qv.standard_error_of_the_mean:.2f} times"
    f" smaller, never the criterion); protocol circuit count met {qv.protocol_circuit_count_met}, passed {qv.passed},"
    f" log2 quantum volume {qv.log2_quantum_volume}"
)
print(f"  exact register fidelities {[round(float(x), 5) for x in qv.register_fidelity]}")
print(
    f"  predicted: eps_gates {qv.budget.predicted['eps_gates']:.3e}, eps_readout {qv.budget.predicted['eps_readout']:.3e}, heavy-output probability {qv.budget.predicted['heavy_output_probability']:.4f} against the ideal mean {qv.budget.predicted['ideal_heavy_output_probability']:.4f}; intrinsic scales per circuit {qv.budget.predicted['intrinsic_total']:.2e}"
)  # type: ignore[union-attr]
pinned("quantum-volume mean heavy-output probability", qv.mean, 0.62, 0.82)
pinned("quantum-volume Eq. (32) sigma", qv.sigma, 0.18, 0.26)
pinned(
    "quantum-volume mean - 2 sigma (below 2/3: no pass at four circuits)",
    qv.mean - 2.0 * qv.sigma,
    0.0,
    0.6666,
)
pinned(
    "quantum-volume Eq. (32) sigma / the standard error of the mean",
    qv.sigma / qv.standard_error_of_the_mean,
    2.5,
    7.0,
)
pinned(
    "quantum-volume measured mean minus the depolarizing prediction",
    qv.mean - qv.budget.predicted["heavy_output_probability"],  # type: ignore[union-attr]
    -0.05,
    0.05,
)
pinned("quantum-volume worst exact register fidelity", float(min(qv.register_fidelity)), 0.98, 1.0)

print("\ndone")
