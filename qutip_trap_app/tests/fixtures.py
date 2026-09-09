"""Constants shared by the application's tests (importable because the app's pytest config puts ``tests`` on the path)."""

from __future__ import annotations

from qutip_trap_app.core import Circuit, Operation, SolverOptions

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
FAST = SolverOptions(branch_weight_min=1e-3)
SHOTS = 200
SEED = 7

__all__ = ["BELL", "FAST", "SEED", "SHOTS"]
