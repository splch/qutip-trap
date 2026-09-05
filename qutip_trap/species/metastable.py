"""Metastable-level channels that default off (PLAN.md Section 4.5.7; Appendix E, Run 5 additions)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qutip_trap.species.model import Species

M0A = "milestone M0a (PLAN.md Section 4.5.7)"


@dataclass(frozen=True)
class MetastableChannels:
    """Blackbody D-D mixing, collisional quenching and reshelving: physical inputs, not fudge factors."""

    bbr_temperature_k: float | None = None
    """None disables blackbody D-D mixing entirely."""
    pressure_mbar: float | None = None
    """Background-gas total pressure; None disables collisions."""
    gas_fractions: dict[str, float] = field(default_factory=dict)
    """{"H2": 0.5, "N2": 0.5}, must sum to 1 when collisions are enabled."""
    reshelving_rate_hz: float = 0.0
    """R of p_D_dot = -Gamma p_D + R(1 - p_D); exposed, never fitted away."""

    def __post_init__(self) -> None:
        if self.pressure_mbar is not None:
            total = sum(self.gas_fractions.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"gas_fractions must sum to 1 when collisions are enabled, got {total}")
        if self.reshelving_rate_hz < 0.0:
            raise ValueError("reshelving_rate_hz must be non-negative")

    def bbr_rate_hz(self, transition: str) -> tuple[float, float]:
        """(downward A*n_bar, upward (g_u/g_l)*A*n_bar) with n_bar = 1/(exp(h nu/kT) - 1) MULTIPLIED (Section 13)."""
        raise NotImplementedError(f"MetastableChannels.bbr_rate_hz is {M0A}")

    def collision_rates_hz(self, species: Species) -> dict[str, float]:
        """{"quench": R_q, "j_mix": R_j} from R = sum_s Gamma_s p_s/(k_B T); Gamma_s in cm^3/s, never a rate."""
        raise NotImplementedError(f"MetastableChannels.collision_rates_hz is {M0A}")

    def reshelving_offset(self, tau_s: float) -> float:
        """R/(Gamma + R), the fitted-offset signature of reshelving (Section 4.5.7)."""
        if tau_s <= 0.0:
            raise ValueError("tau_s must be positive")
        gamma = 1.0 / tau_s
        return self.reshelving_rate_hz / (gamma + self.reshelving_rate_hz)


__all__ = ["MetastableChannels"]
