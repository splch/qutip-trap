"""Orchestration: run, prepare, fidelity levels (JOINT_EXACT and GATE_LOCAL), results (PLAN.md Sections 3.4, 5.4, 8.6; M6, M9a)."""

from __future__ import annotations

from qutip_trap.run.gate_local import (
    GateLocalReport,
    GateLocalStep,
    GateStep,
    Register,
    gate_steps,
    step_space,
)
from qutip_trap.run.job import RunRecord, last_record, prepare, run
from qutip_trap.run.levels import FidelityLevel, resolve_level
from qutip_trap.run.results import Diagnostics, Result, RunState
from qutip_trap.run.space import SpaceSelection, frozen_excitation_bounds, select_space

__all__ = [
    "Diagnostics",
    "FidelityLevel",
    "GateLocalReport",
    "GateLocalStep",
    "GateStep",
    "Register",
    "Result",
    "RunRecord",
    "RunState",
    "SpaceSelection",
    "frozen_excitation_bounds",
    "gate_steps",
    "last_record",
    "prepare",
    "resolve_level",
    "run",
    "select_space",
    "step_space",
]
