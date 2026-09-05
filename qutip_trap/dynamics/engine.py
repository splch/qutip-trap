"""The pulse engine protocol and the run-time state records (PLAN.md Sections 3.4, 5.3, 5.4, 11; Appendix E).

``PulseEngine.run_pulses`` is the one entry point that ``run/``, ``calibration/`` and ``experiments/``
share. Randomness comes from one root ``SeedSequence`` per run, spawned deterministically by (sample,
trajectory, shot, ion, channel) so that every variate's key is independent of execution order and of
truncation retries (Section 3.4); reproducibility over 1 and 18 workers is a tolerance test (10^-12), not a
bitwise one, because ``MultiTrajResult`` accumulates in completion order (Section 3.4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from qutip import Qobj

    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.device.model import Device
    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.noise.sampling import NoiseSample

# QuTiP multistep integrators, never used (Section 5.3: the escalation ladder is dop853 then vern9)
MULTISTEP_INTEGRATORS: frozenset[str] = frozenset({"adams", "bdf", "lsoda", "vode", "zvode"})
ALLOWED_INTEGRATORS: frozenset[str] = frozenset(
    {"dop853", "vern7", "vern9", "tsit5", "explicit_rk", "krylov", "diag"}
)


@dataclass(frozen=True)
class MotionalModel:
    """Per-mode state carried between pulses (Section 5.4 b)."""

    reduced: dict[int, Qobj]
    """mode -> (n_max + 1)^2 reduced density matrix."""
    nbar: dict[int, float]
    frozen: tuple[int, ...]


@dataclass(frozen=True)
class State:
    """What ``prepare()`` returns and ``run_pulses()`` advances."""

    internal: Qobj
    """Register density matrix or ket on the ion factors of ``space``."""
    motional: MotionalModel
    joint: Qobj | None
    """The joint ket/density matrix when the run holds one (JOINT_EXACT)."""
    provenance: tuple[str, ...]
    """Ids of the preparation stages that produced it (Section 4.2.6 order)."""


@dataclass(frozen=True)
class SeedSpec:
    """One root SeedSequence per run, spawned by key (Section 3.4)."""

    root: int

    def __post_init__(self) -> None:
        if self.root < 0:
            raise ValueError("the root seed is a non-negative integer")

    @staticmethod
    def channel_key(channel: str) -> int:
        """A deterministic 32-bit key for a channel name (never Python's salted ``hash``)."""
        return int.from_bytes(hashlib.sha256(channel.encode("utf-8")).digest()[:4], "big")

    def child(
        self, sample: int, trajectory: int, shot: int, ion: int, channel: str
    ) -> np.random.SeedSequence:
        """The keyed child stream for one (sample, trajectory, shot, ion, channel), independent of execution order."""
        if min(sample, trajectory, shot, ion) < 0:
            raise ValueError("sample, trajectory, shot and ion indices are non-negative")
        return np.random.SeedSequence(
            self.root, spawn_key=(sample, trajectory, shot, ion, self.channel_key(channel))
        )


@dataclass(frozen=True)
class SolverOptions:
    atol: float = 1e-10
    rtol: float = 1e-8
    nsteps: int = 10**7
    integrators: tuple[str, ...] = ("dop853", "vern9")
    """The escalation ladder of Section 5.3; never a multistep method."""
    joint_dimension_max: int = 4096
    nnz_max: int = 2 * 10**7
    """The Section 11.5 guards that route to GATE_LOCAL."""
    boundary_population_max: float = 1e-6
    freeze_chi_max_rad: float = 0.05
    map: Literal["serial", "parallel", "loky"] = "parallel"
    """Coefficients are module-level functions or arrays, so they pickle."""
    e_ops_for_target_tol: bool = True
    """mcsolve needs e_ops to target a tolerance (Section 5.4)."""

    def __post_init__(self) -> None:
        if self.atol <= 0.0 or self.rtol <= 0.0 or self.nsteps <= 0:
            raise ValueError("tolerances and nsteps must be positive")
        if not self.integrators:
            raise ValueError("at least one integrator is required")
        bad = [name for name in self.integrators if name in MULTISTEP_INTEGRATORS]
        if bad:
            raise ValueError(f"multistep integrators are never used (Section 5.3): {bad}")
        unknown = [name for name in self.integrators if name not in ALLOWED_INTEGRATORS]
        if unknown:
            raise ValueError(f"unknown QuTiP integrators: {unknown}")
        if self.joint_dimension_max < 2 or self.nnz_max < 1:
            raise ValueError("the size guards must be positive")
        if not 0.0 < self.boundary_population_max < 1.0:
            raise ValueError("boundary_population_max is a population fraction in (0, 1)")


@dataclass(frozen=True)
class Traces:
    """What ``run_pulses()`` returns (Section 14.3)."""

    times_s: np.ndarray
    expectations: dict[str, np.ndarray]
    reduced_internal: tuple[Qobj, ...]
    mode_occupations: dict[int, np.ndarray]
    alpha_m: dict[int, np.ndarray]
    jumps: tuple[tuple[float, str], ...]
    final: State
    boundary_population: dict[int, float]


@dataclass(frozen=True)
class ChannelSummary:
    """Process tomography of one pulse (Section 5.4)."""

    choi: np.ndarray
    cp_tp_residual: tuple[float, float]
    """Residual against the completely-positive cone and the trace-preserving affine set after Dykstra projection."""
    n_traj: int
    average_gate_infidelity: float
    pauli_twirled: dict[str, float]
    depolarizing_rate: float
    """epsilon with Lambda_eps(rho) = (1 - eps) rho + eps/(4^n - 1) sum_{P != I} P rho P (Section 6.8)."""


class PulseEngine(Protocol):
    """The one entry point that run/, calibration/ and experiments/ share (Appendix E)."""

    def run_pulses(
        self,
        device: Device,
        schedule: Schedule,
        state: State,
        space: HilbertSpace,
        sample: NoiseSample,
        seeds: SeedSpec,
        options: SolverOptions,
    ) -> Traces: ...

    def process_tomography(
        self,
        device: Device,
        pulse: Pulse,
        space: HilbertSpace,
        motional_model: MotionalModel,
        sample: NoiseSample,
        seeds: SeedSpec,
    ) -> ChannelSummary: ...


__all__ = [
    "ALLOWED_INTEGRATORS",
    "MULTISTEP_INTEGRATORS",
    "ChannelSummary",
    "MotionalModel",
    "PulseEngine",
    "SeedSpec",
    "SolverOptions",
    "State",
    "Traces",
]
