"""Benchmark emulation on the simulated device (PLAN.md Section 10, milestone M10): randomized benchmarking, GHZ fidelity and a
quantum-volume style run, each through ``run`` (compile -> calibrate -> schedule -> prepare -> evolve -> readout, Section 3.4)
with the simulator's own error budget reported alongside (``budget``: the Section 9.6 closed-form scales, the Section 6.8
channel summaries of the native gate set by GATE_LOCAL tomography, the Section 8.4 SPAM errors).

The Clifford groups (``clifford``) and the SU(4) decomposition (``control.two_qubit``) supply the ideal matrices the protocols
are defined with; they are never applied to a simulated state (Section 3.1).
"""

from __future__ import annotations

from qutip_trap.benchmarks.budget import (
    BenchmarkBudget,
    GateChannel,
    StepChannel,
    clear_budget_cache,
    gate_channel,
)
from qutip_trap.benchmarks.clifford import (
    CLASS_SIZES,
    SINGLE_QUBIT_CLIFFORDS,
    TWO_QUBIT_GROUP_ORDER,
    TwoQubitClifford,
    decompose_two_qubit_clifford,
    random_two_qubit_clifford,
    two_qubit_clifford_group,
)
from qutip_trap.benchmarks.ghz import GHZResult, ghz_circuit, ghz_fidelity, parity_circuit
from qutip_trap.benchmarks.rb import RBResult, RBSequence, randomized_benchmarking
from qutip_trap.benchmarks.volume import QVCircuit, QVResult, quantum_volume, random_square_circuit

__all__ = [
    "CLASS_SIZES",
    "SINGLE_QUBIT_CLIFFORDS",
    "TWO_QUBIT_GROUP_ORDER",
    "BenchmarkBudget",
    "GHZResult",
    "GateChannel",
    "QVCircuit",
    "QVResult",
    "RBResult",
    "RBSequence",
    "StepChannel",
    "TwoQubitClifford",
    "clear_budget_cache",
    "decompose_two_qubit_clifford",
    "gate_channel",
    "ghz_circuit",
    "ghz_fidelity",
    "parity_circuit",
    "quantum_volume",
    "random_square_circuit",
    "random_two_qubit_clifford",
    "randomized_benchmarking",
    "two_qubit_clifford_group",
]
