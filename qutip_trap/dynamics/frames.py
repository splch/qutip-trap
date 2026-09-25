"""The virtual-Z phase frame (PLAN.md Section 7.6).

RZ(theta) on qubit i shifts the phase of every later pulse on i by phi -> phi - theta, the gates read in time order (the
+theta of the documentation holds in matrix order); the frame at the end of a schedule is ``Schedule.phase_frame``, which
the measurement discards.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhaseFrame:
    """The per-qubit virtual-Z frame: an offset theta_i (rad) that every later pulse phase on qubit i subtracts."""

    offsets_rad: dict[int, float] = field(default_factory=dict)

    def offset(self, qubit: int) -> float:
        return float(self.offsets_rad.get(qubit, 0.0))

    def rz(self, qubit: int, theta_rad: float) -> PhaseFrame:
        """RZ(theta) on ``qubit``: later pulses on it carry phi -> phi - theta."""
        new = dict(self.offsets_rad)
        new[qubit] = new.get(qubit, 0.0) + float(theta_rad)
        return PhaseFrame(new)

    def pulse_phase(self, qubit: int, phi_program_rad: float) -> float:
        """The phase the hardware plays for a gate programmed at phi on ``qubit``."""
        return float(phi_program_rad) - self.offset(qubit)

    def as_dict(self, n_qubits: int) -> dict[int, float]:
        return {q: self.offset(q) for q in range(n_qubits)}
