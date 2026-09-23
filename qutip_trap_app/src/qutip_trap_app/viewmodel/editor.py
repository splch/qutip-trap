"""The circuit editor's model: the preset circuits and the parser of the text, both formats by the core's importers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from qutip_trap_app import core

MAX_QUBITS = 4
"""Every ion adds motional modes and joint dimension; four keeps a full run within minutes on the example device."""
MAX_SHOTS = 100_000
"""The readout is sampled shot by shot in Python: this many take tens of seconds."""

BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'
)
GHZ_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\n'
    "measure q -> c;\n"
)


@dataclass(frozen=True)
class Preset:
    text: str
    shots: int
    device_kwargs: dict[str, float] = field(default_factory=dict)
    """Keyword arguments of the example device for this circuit."""


PRESETS: dict[str, Preset] = {
    "Bell state": Preset(BELL_QASM, 200),
    # three ions sit closer together: a tighter addressing beam keeps the light off the neighbours
    "GHZ state, three ions": Preset(GHZ_QASM, 200, {"address_waist_m": 2.0e-6}),
}


def parse_circuit(text: str) -> core.Circuit:
    """The circuit of an OpenQASM 2 program, or of an IonQ circuit JSON object when the text is one."""
    circuit = (
        core.load_ionq_json(json.loads(text)) if text.lstrip().startswith("{") else core.load_openqasm2(text)
    )
    if circuit.n_qubits > MAX_QUBITS:
        raise ValueError(f"the circuit has {circuit.n_qubits} qubits; at most {MAX_QUBITS} run here")
    return circuit
