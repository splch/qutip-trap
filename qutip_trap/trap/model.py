"""The ``Trap`` record (PLAN.md Section 3.3; Appendix E; milestone M1 for the methods)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.trap.anharmonic import AnharmonicTerms
from qutip_trap.trap.pseudopotential import DcElectrodes, RfDrive
from qutip_trap.trap.surface import Electrodes

if TYPE_CHECKING:
    from qutip_trap.species.model import Species
    from qutip_trap.trap.mathieu import MathieuParameters
    from qutip_trap.trap.micromotion import MicromotionIndex

M1 = "milestone M1 (trap/, PLAN.md Section 4.1)"
M12 = "milestone M12 (PLAN.md Section 4.6; not scheduled for the first release)"


@dataclass(frozen=True)
class Trap:
    """Either explicit secular frequencies or voltages plus geometry (Section 3.3).

    The axis angle exists because the explicit-frequency path would otherwise pin the modes to the
    laboratory frame, and a mode with no projection on any cooling beam is uncoolable; the stray field and
    the shims exist because compensation is something a laboratory measures and re-nulls (2026-09-04
    experimentalist critique). The scheduler never sees ``stray_field_v_per_m`` (Section 7.3).
    """

    omega_hz: tuple[float, float, float] | None
    axis_angle_rad: float
    """Rotation of the radial principal axes about z (explicit-frequency path)."""
    rf: RfDrive | None
    dc: DcElectrodes | None
    geometry: Electrodes | None
    stray_field_v_per_m: tuple[float, float, float]
    """The TRUE stray field, hidden from the scheduler (Section 7.3)."""
    shim_voltages_v: dict[str, float]
    """Compensation applied; residual = stray + shim response."""
    dc_schedule: dict[str, np.ndarray] | None = None
    """M12: V_n(t) per electrode, sampled."""
    basis_potentials: dict[str, Callable[..., float]] | None = None
    """M12: phi_tilde_n(r) per electrode, plus "rf" for phi_tilde_rf (Section 4.1.6)."""

    def __post_init__(self) -> None:
        explicit = self.omega_hz is not None
        voltages = self.rf is not None and self.dc is not None and self.geometry is not None
        if not (explicit or voltages):
            raise ValueError("a Trap needs either omega_hz or (rf, dc, geometry)")
        if explicit and self.omega_hz is not None and any(w <= 0.0 for w in self.omega_hz):
            raise ValueError("secular frequencies must be positive (ordinary Hz)")

    def mathieu(self, species: Species) -> MathieuParameters:
        """a, q (matrices), beta, secular frequencies, C0 (Section 4.1.1)."""
        raise NotImplementedError(f"Trap.mathieu is {M1}")

    def micromotion_beta(self, species: Species, delta_k: np.ndarray) -> MicromotionIndex:
        """Residual beta = delta_k . u_1 for the FULL wavevector, as (in_phase, out_of_phase) (Section 4.1.1)."""
        raise NotImplementedError(f"Trap.micromotion_beta is {M1}")

    def anharmonic(self) -> AnharmonicTerms | None:
        raise NotImplementedError(f"Trap.anharmonic is {M1}")

    def pseudopotential_v(self, r_m: np.ndarray, species: Species) -> float:
        """M12: phi_ps in VOLTS (Section 13, "Junction pseudopotential")."""
        raise NotImplementedError(f"Trap.pseudopotential_v is {M12}")

    def split_coefficients(self, t_s: float) -> tuple[float, float, float]:
        """M12: (alpha, beta, gamma) of the volt potential beta x^4 + alpha x^2 + gamma x (Section 13)."""
        raise NotImplementedError(f"Trap.split_coefficients is {M12}")


__all__ = ["Trap"]
