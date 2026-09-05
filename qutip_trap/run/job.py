"""``run`` and ``prepare``: the orchestration of Section 3.4 (PLAN.md Appendix E; milestone M6).

Data flow (Appendix E): space = HilbertSpace.for_(device, schedule, options); sample = device.noise.sample(rng);
seeds = SeedSpec(seed) keyed by (sample, trajectory, shot, ion, channel); state = prepare(device, space, table,
sample, seeds); traces = engine.run_pulses(device, schedule, state, space, sample, seeds, options); readout on
traces -> Result. ``table=None`` calibrates (surrogate) at t0; a stale table is a legitimate input (Section 7.5).
``shot_period_s=None`` derives T_rep from the schedule plus cooling, detection and dead time, so shot k sees
drift and the mains phase at t0 + k T_rep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SeedSpec, SolverOptions, State
    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.noise.sampling import NoiseSample
    from qutip_trap.run.results import Result

M3 = "milestone M3 (prep/, PLAN.md Section 4.2)"
M6 = "milestone M6 (run/job.py, PLAN.md Section 3.4)"


def run(
    circuit: Circuit,
    device: Device,
    shots: int,
    *,
    table: CalibrationTable | None = None,
    t0_s: float = 0.0,
    shot_period_s: float | None = None,
    samples: int | None = None,
    level: Literal["JOINT_EXACT", "GATE_LOCAL", "auto"] = "auto",
    seed: int = 0,
    options: SolverOptions | None = None,
) -> Result:
    """Compile -> calibrate -> schedule -> prepare -> evolve -> readout -> Result (Section 3.4)."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    raise NotImplementedError(f"run is {M6}")


def prepare(
    device: Device, space: HilbertSpace, table: CalibrationTable, sample: NoiseSample, seeds: SeedSpec
) -> State:
    """Doppler -> sideband/EIT -> optical pump, in that order (Section 4.2.6); internal x motional state with provenance."""
    raise NotImplementedError(f"prepare is {M3}")


__all__ = ["prepare", "run"]
