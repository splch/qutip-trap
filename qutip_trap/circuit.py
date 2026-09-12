"""Rung 1 of the ladder, the circuit (docs/api_proposal.md Section 4.1; docs/api_implementation_plan.md 1.6; 0.2.0): what a
program is (``Circuit``, ``Operation``, the two gate tables), what the compiler does to it (``compile_with_report`` and its
``CompileReport``, ``compile_to_native``), what it should compute (``ideal_probabilities``, ``circuit_unitary``) and the
exact native gate matrices of ``control.native`` (radians; IonQ's turns through ``rad_from_turns`` and ``turns_from_rad``).
The same objects ``qutip_trap.api`` exports under the Appendix E names; the way down is ``Machine.schedule`` (rung 2).
"""

from __future__ import annotations

from qutip_trap.control.compiler import (
    EXPORTED_NATIVE,
    GATE_PARAMETERS,
    NATIVE_GATES,
    NON_UNITARY,
    STANDARD_GATES,
    Circuit,
    CompileError,
    CompileReport,
    Operation,
    circuit_unitary,
    compile_to_native,
    compile_with_report,
    gate_matrix,
    ideal_probabilities,
)
from qutip_trap.control.native import (
    equal_up_to_global_phase,
    gpi,
    gpi2,
    ms,
    r_phi,
    rad_from_turns,
    rz,
    turns_from_rad,
    xx,
    zz,
)

__all__ = [
    "EXPORTED_NATIVE",
    "GATE_PARAMETERS",
    "NATIVE_GATES",
    "NON_UNITARY",
    "STANDARD_GATES",
    "Circuit",
    "CompileError",
    "CompileReport",
    "Operation",
    "circuit_unitary",
    "compile_to_native",
    "compile_with_report",
    "equal_up_to_global_phase",
    "gate_matrix",
    "gpi",
    "gpi2",
    "ideal_probabilities",
    "ms",
    "r_phi",
    "rad_from_turns",
    "rz",
    "turns_from_rad",
    "xx",
    "zz",
]
