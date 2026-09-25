"""rf drive, dc electrode voltages and the geometry-free maps to (a, q) (PLAN.md Sections 4.1.1, 4.1.6; 13; M1).

Conventions restated (Section 13):

- row "rf amplitude and pseudopotential": ``RfDrive.voltage_peak_v`` is the PEAK of a cos(Omega_rf t) drive,
  which is what puts the 4 in Psi = Q^2 |E_rf|^2/(4 m Omega^2); a stored peak read as rms doubles every depth
  and inflates every rf-set secular frequency by sqrt 2, one read as peak-to-peak quarters the depth and
  halves those frequencies (depth ~ f^2 and frequency ~ f always move together; derivation audit 2026-09-04).
- row "Mathieu equation": a_i = 4 Q Phi''_dc,i/(m Omega^2), q_i = 2 Q Phi''_rf,i/(m Omega^2) with Phi'' the second
  derivatives of the electric potential along the principal axes (Brownnutt 2015; Section 4.1.6), the
  geometry-free map every electrode model feeds; Q is the ion charge and q the Mathieu parameter (the two
  symbols collide in the sources and are ``Q_charge`` and ``q_mathieu`` in code).
- Berkeland 1998's linear-trap geometry: a_x = a_y = -a_z/2 = -4 Q kappa U0/(m Z0^2 Omega^2),
  q_x = -q_y = 2 Q V0/(m R'^2 Omega^2), q_z = 0, with R' ~ R (equal to R for infinitely long hyperbolic
  electrodes; the paper's own example uses R' = R = 1.0 mm and q_x = 2 sqrt 2 omega_x/Omega = 0.28).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qutip_trap.units import E_C, TWO_PI


@dataclass(frozen=True)
class RfDrive:
    """The rf drive of a Paul trap.

    ``voltage_peak_v`` is the PEAK of a cos(Omega_rf t) drive (Section 13, row "rf amplitude and
    pseudopotential"). ``frequency_hz`` is Omega_rf/2pi, an ordinary frequency (Section 5.6). ``phase_rad`` is
    the rf phase at t = 0 relative to the adopted Mathieu origin (Section 13, "Floquet function and rf phase
    origin"). ``phase_imbalance_rad`` is Berkeland's phi_ac, the rf phase difference between the two rf
    electrodes, which drives the irreducible out-of-phase excess micromotion (1/4) q_x R alpha phi_ac that no dc
    shim can null (Section 4.1.1); it needs the geometry factor ``alpha`` of the electrode record.
    """

    voltage_peak_v: float
    frequency_hz: float
    phase_rad: float = 0.0
    phase_imbalance_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("rf frequency must be positive")
        if self.voltage_peak_v < 0.0:
            raise ValueError("the rf amplitude is a magnitude (the sign is the phase)")

    @property
    def omega_rad_s(self) -> float:
        """Omega_rf = 2 pi frequency_hz."""
        return TWO_PI * self.frequency_hz


@dataclass(frozen=True)
class DcElectrodes:
    """dc voltages by electrode name (V)."""

    voltages_v: dict[str, float]

    def voltage(self, name: str) -> float:
        return float(self.voltages_v.get(name, 0.0))


# ---- geometry-free maps (Section 4.1.1) --------------------------------------------------------------------


def mathieu_a(
    phi_dc_second_derivative_v_per_m2: float, mass_kg: float, omega_rf_rad_s: float, *, charge: int = 1
) -> float:
    """a = 4 Q Phi''_dc/(m Omega^2) with Phi'' in V/m^2 (Brownnutt 2015; Section 4.1.1)."""
    return 4.0 * charge * E_C * phi_dc_second_derivative_v_per_m2 / (mass_kg * omega_rf_rad_s**2)


def mathieu_q(
    phi_rf_second_derivative_v_per_m2: float, mass_kg: float, omega_rf_rad_s: float, *, charge: int = 1
) -> float:
    """q = 2 Q Phi''_rf/(m Omega^2) with Phi''_rf the second derivative of the PEAK rf potential (Section 4.1.1)."""
    return 2.0 * charge * E_C * phi_rf_second_derivative_v_per_m2 / (mass_kg * omega_rf_rad_s**2)


def mathieu_matrices(
    hessian_dc_v_per_m2: np.ndarray,
    hessian_rf_v_per_m2: np.ndarray,
    mass_kg: float,
    omega_rf_rad_s: float,
    *,
    charge: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """(A, Q) matrices of House 2008 in the standard sign, A_ij = 4Q Phi_dc,ij/(m Omega^2), Q_ij = 2Q Phi_rf,ij/(m Omega^2).

    Both are traceless when the potentials satisfy Laplace's equation; House's own Q is the negative of this
    one (Section 13, "Mathieu sign in House 2008").
    """
    h_dc = np.asarray(hessian_dc_v_per_m2, dtype=float)
    h_rf = np.asarray(hessian_rf_v_per_m2, dtype=float)
    pref = charge * E_C / (mass_kg * omega_rf_rad_s**2)
    return 4.0 * pref * h_dc, 2.0 * pref * h_rf


def pseudopotential_j(
    e_rf_peak_v_per_m: np.ndarray | float, mass_kg: float, omega_rf_rad_s: float, *, charge: int = 1
) -> float:
    """Dehmelt pseudopotential Psi = Q^2 |E_rf|^2/(4 m Omega^2) in joules, E_rf the PEAK rf field (Section 4.1.1)."""
    e = np.asarray(e_rf_peak_v_per_m, dtype=float)
    e2 = float(np.dot(e.ravel(), e.ravel())) if e.ndim else float(e) ** 2
    return (charge * E_C) ** 2 * e2 / (4.0 * mass_kg * omega_rf_rad_s**2)


def pseudopotential_v(
    e_rf_peak_v_per_m: np.ndarray | float, mass_kg: float, omega_rf_rad_s: float, *, charge: int = 1
) -> float:
    """phi_ps = Q |E_rf|^2/(4 m Omega^2) in VOLTS (Section 13, "Junction pseudopotential"): Psi = Q phi_ps."""
    return pseudopotential_j(e_rf_peak_v_per_m, mass_kg, omega_rf_rad_s, charge=charge) / (charge * E_C)


def secular_from_hessian_rad_s(
    hessian_total_j_per_m2: np.ndarray, mass_kg: float
) -> tuple[np.ndarray, np.ndarray]:
    """omega_i^2 = eig(Hess U_eff)/m for the total static energy U_eff = Psi + Q Phi_dc (pseudopotential route).

    Returns (omega ascending in rad/s, principal directions as columns); the same equation as
    (Omega/2) sqrt(eig(A + Q^2/2)), implemented twice and cross-checked (Section 4.1.6).
    """
    h = np.asarray(hessian_total_j_per_m2, dtype=float)
    vals, vecs = np.linalg.eigh((h + h.T) / 2.0)
    if np.any(vals <= 0.0):
        raise ValueError(f"the effective potential is not confining: curvatures {vals}")
    return np.sqrt(vals / mass_kg), vecs


# ---- Berkeland's linear trap with rods or blades (Section 4.1.1) ------------------------------------------------


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
    """(a_x, a_y, a_z), (q_x, q_y, q_z) of Berkeland et al. 1998 for a linear trap of radial scale R' = R and endcap
    scale Z0 with geometric efficiency kappa: a_x = a_y = -a_z/2 = -4 Q kappa U0/(m Z0^2 Omega^2),
    q_x = -q_y = 2 Q V0/(m R^2 Omega^2), q_z = 0."""
    if r_m <= 0.0 or z0_m <= 0.0 or kappa <= 0.0:
        raise ValueError("R, Z0 and kappa must be positive")
    qe = charge * E_C
    a_z = 8.0 * qe * kappa * u_dc_v / (mass_kg * z0_m**2 * omega_rf_rad_s**2)
    a_x = -a_z / 2.0
    q_x = 2.0 * qe * v_rf_peak_v / (mass_kg * r_m**2 * omega_rf_rad_s**2)
    return (a_x, a_x, a_z), (q_x, -q_x, 0.0)


def trap_inversion_p_s(
    omega_rad_s: np.ndarray | tuple[float, float, float], mass_kg: float
) -> tuple[np.ndarray, np.ndarray]:
    """(P, S) of Home's trap inversion omega_i^2(m) = P_i/m^2 + S_i/m from ONE species' three secular frequencies.

    The pseudopotential splits the confinement into an rf part, whose curvature Psi'' ~ Q^2 |E_rf'|^2/(m Omega^2) makes
    omega^2 ~ 1/m^2, and a static part with omega^2 ~ 1/m (Home 2013 Eqs. 6-19; PLAN Section 4.1.7). The linear-trap
    structure of Section 4.1.1 (q_z = 0, q_y = -q_x, and Laplace on the dc potential) fixes P_x = P_y, P_z = 0 and
    sum_i S_i = 0, which closes the system: 2 P_x/m^2 = sum_i omega_i^2, S_z = m omega_z^2, S_{x,y} = m(omega_{x,y}^2 - P_x/m^2).

    ``P`` is in kg^2 (rad/s)^2 and ``S`` in kg (rad/s)^2. The identity this buys is that
    omega_x^2(m) - omega_y^2(m) = (m_ref/m)(omega_x,ref^2 - omega_y,ref^2), so the ORDER of the two radial axes is the
    same for every species (PLAN 9.13: "preserves sign(omega_x^2 - omega_y^2) across species").
    """
    w = np.asarray(omega_rad_s, dtype=float)
    if w.shape != (3,) or np.any(w <= 0.0) or mass_kg <= 0.0:
        raise ValueError("three positive secular frequencies (rad/s) and a positive mass")
    p_r = mass_kg**2 * float(np.dot(w, w)) / 2.0
    s = np.array(
        [mass_kg * w[0] ** 2 - p_r / mass_kg, mass_kg * w[1] ** 2 - p_r / mass_kg, mass_kg * w[2] ** 2]
    )
    return np.array([p_r, p_r, 0.0]), s


def pseudopotential_mass_scaling_rad_s(
    omega_ref_rad_s: np.ndarray | tuple[float, float, float], mass_ref_kg: float, mass_kg: float
) -> np.ndarray:
    """Another species' secular frequencies from one species' three, in the PSEUDOPOTENTIAL limit (Home's inversion).

    omega_i(m) = sqrt(P_i/m^2 + S_i/m) with (P, S) from ``trap_inversion_p_s``. Section 9.13's fixture: Be+ at
    [9.7, 12.9, 4.6] MHz gives Mg+ [1.52, 5.43, 2.82] MHz. This is the Omega_rf -> infinity limit of the exact-exponent
    route ``Trap.single_ion_frequencies_rad_s`` takes (which inverts the monodromy at the trap's own rf frequency and so
    carries the O(q^4) Floquet correction: 1.656/4.528 MHz at 40 MHz, 1.520/5.399 at 200 MHz, 1.523/5.430 at 1 GHz).
    """
    p, s = trap_inversion_p_s(omega_ref_rad_s, mass_ref_kg)
    if mass_kg <= 0.0:
        raise ValueError("a mass is positive")
    w2 = p / mass_kg**2 + s / mass_kg
    if np.any(w2 <= 0.0):
        raise ValueError(
            f"the pseudopotential does not confine mass {mass_kg} kg on every axis: omega^2 = {w2}"
        )
    return np.asarray(np.sqrt(w2))


def radial_frequency_from_voltage_rad_s(
    v_rf_peak_v: float, omega_rf_rad_s: float, mass_kg: float, r_m: float, *, charge: int = 1
) -> float:
    """omega_r = Q V0/(sqrt 2 Omega m R^2), Wineland 1998 Eq. 6 with Q the charge (Section 4.1.1), the a = 0
    pseudopotential frequency; identical to q Omega/(2 sqrt 2) with q = 2 Q V0/(m R^2 Omega^2)."""
    return charge * E_C * v_rf_peak_v / (math.sqrt(2.0) * omega_rf_rad_s * mass_kg * r_m**2)
