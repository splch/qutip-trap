"""The core contract: every import of ``qutip_trap`` the application makes, in one module (PLAN.md Section 14.6).

Section 14.6: "The app calls only public core API that exists for the core's own reasons ... If a zoom view needs data the
core does not expose, the app records the gap and shows the view as unavailable rather than patching the core." The public
surface is ``qutip_trap.api`` (Appendix E). Everything the application uses from the core is re-exported from here, so
that a reader can audit the contract in one place and ``tests/test_provenance_coverage.py`` can assert that no other
module of the application imports ``qutip_trap`` directly.

Two names below come from outside ``qutip_trap.api``. Each is a core feature request (a re-export, nothing more) that the
application works around on its own side, exactly as Section 14.1 asks; they are listed in :data:`CORE_GAPS` and the run
record carries that list, so a reader of an exported record knows which core facilities the app leaned on.
"""

from __future__ import annotations

# ---- the public surface (Appendix E) -------------------------------------------------------------------------------------------
from qutip_trap import __version__ as core_version
from qutip_trap.api import (
    CalEntry,
    CalibrationTable,
    Circuit,
    CompileReport,
    Device,
    DevicePreset,
    Diagnostics,
    GateDrive,
    GateStep,
    GateTarget,
    HilbertSpace,
    MotionalModel,
    NoiseSample,
    Operation,
    PlayedGate,
    Pulse,
    Result,
    RunRecord,
    Schedule,
    SeedSpec,
    SolverOptions,
    State,
    Traces,
    Waveform,
    ca40_optical,
    calibrate,
    circuit_unitary,
    gate_steps,
    ideal_probabilities,
    last_record,
    prepare,
    register_fidelity,
    run,
    yb171_chain,
)

# ---- names the core does not re-export through qutip_trap.api (each a filed core feature request, Section 14.1) -----------
from qutip_trap.dynamics.engine import JointExactEngine
from qutip_trap.noise.sampling import KEY_BRANCH_WEIGHT, key_frozen_n

CORE_GAPS: tuple[str, ...] = (
    "qutip_trap.api exports the PulseEngine protocol but no concrete engine; the per-pulse evolution entry point Section 14.6 "
    "names (JointExactEngine.run_pulses) is imported from qutip_trap.dynamics.engine [core feature request: re-export "
    "JointExactEngine in qutip_trap.api]",
    "the noise-sample keys run() stamps on every branch (branch_weight, frozen_n[m]) are imported from qutip_trap.noise.sampling "
    "so that a re-simulation rebuilds the same NoiseSample [core feature request: re-export KEY_BRANCH_WEIGHT and key_frozen_n]",
    "Traces carries <n_m>(t) and the final reduced motional states only: a per-time Fock distribution inside a pulse is not "
    "exposed, so the Level 3 Fock heatmap has values at pulse boundaries and re-simulated sub-steps only [core feature request: "
    "an optional per-time mode_marginal store in Traces]",
    "run() returns no wall time per pulse and no per-gate register state for GATE_LOCAL runs; the app measures the run's wall "
    "time itself and marks the GATE_LOCAL per-gate register states unavailable until the channel replay of M11.2",
)
"""What the core does not expose (or does not re-export) that the application needs; Section 14.6's record of the gaps."""

__all__ = [
    "CORE_GAPS",
    "KEY_BRANCH_WEIGHT",
    "CalEntry",
    "CalibrationTable",
    "Circuit",
    "CompileReport",
    "Device",
    "DevicePreset",
    "Diagnostics",
    "GateDrive",
    "GateStep",
    "GateTarget",
    "HilbertSpace",
    "JointExactEngine",
    "MotionalModel",
    "NoiseSample",
    "Operation",
    "PlayedGate",
    "Pulse",
    "Result",
    "RunRecord",
    "Schedule",
    "SeedSpec",
    "SolverOptions",
    "State",
    "Traces",
    "Waveform",
    "ca40_optical",
    "calibrate",
    "circuit_unitary",
    "core_version",
    "gate_steps",
    "ideal_probabilities",
    "key_frozen_n",
    "last_record",
    "prepare",
    "register_fidelity",
    "run",
    "yb171_chain",
]
