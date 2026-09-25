"""The rf drive, the dc electrode voltages and their maps to the Mathieu (a, q) (PLAN.md Section 4.1.1).

a_i = 4 Q Phi''_dc,i/(m Omega^2) and q_i = 2 Q Phi''_rf,i/(m Omega^2) with Phi'' the curvatures of the electric potential
(Brownnutt 2015) and Q the ion charge; the rf amplitude is the PEAK of a cos(Omega t) drive, which is what puts the 4 in
the pseudopotential Q^2 |E_rf|^2/(4 m Omega^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap.units import E_C, TWO_PI


@dataclass(frozen=True)
class RfDrive:
    """The rf drive of a Paul trap: PEAK voltage (V), frequency Omega_rf/2pi (Hz), and Berkeland's phi_ac.

    ``phase_imbalance_rad`` is the rf phase difference between the two rf electrodes, which drives the irreducible
    out-of-phase micromotion (1/4) q_x R alpha phi_ac that no dc shim nulls; it needs the rod factor ``alpha``.
    """

    voltage_peak_v: float
    frequency_hz: float
    phase_imbalance_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("rf frequency must be positive")
        if self.voltage_peak_v < 0.0:
            raise ValueError("the rf amplitude is a magnitude (the sign is the phase)")

    @property
    def omega_rad_s(self) -> float:
        return TWO_PI * self.frequency_hz


@dataclass(frozen=True)
class DcElectrodes:
    """dc voltages by electrode name (V)."""

    voltages_v: dict[str, float]


def mathieu_matrices(
    hessian_dc_v_per_m2: np.ndarray,
    hessian_rf_v_per_m2: np.ndarray,
    mass_kg: float,
    omega_rf_rad_s: float,
    *,
    charge: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """(A, Q) = (4 Q H_dc, 2 Q H_rf)/(m Omega^2) in the standard sign; House 2008's own Q is the negative of this one."""
    pref = charge * E_C / (mass_kg * omega_rf_rad_s**2)
    return 4.0 * pref * np.asarray(hessian_dc_v_per_m2, dtype=float), 2.0 * pref * np.asarray(
        hessian_rf_v_per_m2, dtype=float
    )


def linear_trap_parameters(
    *,
    v_rf_peak_v: float,
    u_dc_v: float,
    omega_rf_rad_s: float,
    mass_kg: float,
    r_m: float,
    z0_m: float,
    kappa: float,
    charge: int = 1,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Berkeland 1998's rod or blade trap: a_x = a_y = -a_z/2 = -4 Q kappa U0/(m Z0^2 Omega^2), q_x = -q_y = 2 Q V0/(m R^2 Omega^2)."""
    if r_m <= 0.0 or z0_m <= 0.0 or kappa <= 0.0:
        raise ValueError("R, Z0 and kappa must be positive")
    qe = charge * E_C
    a_z = 8.0 * qe * kappa * u_dc_v / (mass_kg * z0_m**2 * omega_rf_rad_s**2)
    q_x = 2.0 * qe * v_rf_peak_v / (mass_kg * r_m**2 * omega_rf_rad_s**2)
    return (-a_z / 2.0, -a_z / 2.0, a_z), (q_x, -q_x, 0.0)
