"""OpenQASM 2 importer (subset) (PLAN.md Sections 1.4, 7.6; milestone M6).

The IonQ path is native JSON or OpenQASM 3; the client SDKs' OpenQASM 2 export declares gpi, gpi2, ms and zz
as custom gates built from u, rz, rxx and rzz, which this importer will accept (Section 7.6).
"""

from __future__ import annotations

from qutip_trap.control.compiler import Circuit

M6 = "milestone M6 (io/openqasm.py, PLAN.md Section 7.2)"


def load_openqasm2(text: str) -> Circuit:
    raise NotImplementedError(f"load_openqasm2 is {M6}")


__all__ = ["load_openqasm2"]
