"""Transport budgets and scheduled transport operations (PLAN.md Section 4.6; Appendix E)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qutip_trap.transport.waveforms import VoltageWaveform

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

M12 = "milestone M12 (PLAN.md Section 4.6; not scheduled for the first release)"


@dataclass(frozen=True)
class TransportBudget:
    """What a transport, split or merge costs, per mode."""

    alpha: complex
    """Coherent amplitude in the co-moving frame, acceleration-form phase (Section 13)."""
    n_coherent: float
    """|alpha|^2, the Husimi-Kerner gamma."""
    n_thermal: float
    squeeze: complex
    reference_omega_hz: float
    """The frequency n_coherent is referenced to; required, since omega_f and omega_CP differ by 10x."""
    ermakov_state: tuple[float, float, float]
    per_ion: tuple[float, ...] | None
    heralded_failure: bool
    provenance_id: str


@dataclass(frozen=True)
class Transport:
    """One scheduled transport-family operation."""

    waveform: VoltageWaveform
    from_zone: str
    to_zone: str
    ions: tuple[int, ...]
    hold_s: float
    hold_offset_s: float
    reverse_of: Transport | None
    overhead_factor: float = 1.10

    def budget(self, device: Device) -> dict[int, TransportBudget]:
        raise NotImplementedError(f"Transport.budget is {M12}")


def transport_budget(
    device: Device, transport: Transport, *, level: str = "analytic"
) -> dict[int, TransportBudget]:
    """level="analytic" runs the Ermakov fast path of M12.2; level="exact" the QobjEvo solve of M12.3."""
    raise NotImplementedError(f"transport_budget is {M12}")


def design_waveform(
    device: Device,
    kind: str,
    from_zone: str,
    to_zone: str,
    *,
    duration_s: float,
    target_quanta: float = 1.0,
    shape: str = "erf",
) -> VoltageWaveform:
    """Design the voltage waveform of a transport, split or merge between two zones that reaches ``target_quanta`` of
    motional excitation within ``duration_s`` (Section 4.6). Owned by M12: raises ``NotImplementedError`` naming it."""
    raise NotImplementedError(f"design_waveform is {M12}")


def split_feasible(device: Device, waveform: VoltageWaveform) -> tuple[bool, str]:
    """False with a reason when |gamma| >= gamma_tilde = 1.06 (kappa^3 beta_CP^2)^(1/5)."""
    raise NotImplementedError(f"split_feasible is {M12}")
