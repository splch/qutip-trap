"""The ``Trap`` record and its two paths to Mathieu parameters.

- Explicit: ``omega_hz`` are one species' measured secular frequencies; with an ``RfDrive`` the exact exponents are
  inverted for (a, q) (``mathieu_from_secular``), without one there are no Mathieu parameters (C0 = 1).
- Geometry: ``rf`` + ``geometry`` (+ ``dc``), a rod or blade trap (``linear_trap_parameters``) or a surface layout
  (``trap/surface.py``).

The residual field is stray + shim response; only the surface path has a shim -> field model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.trap.anharmonic import AnharmonicTerms
from qutip_trap.trap.mathieu import (
    MathieuParameters,
    mathieu_from_secular,
    mathieu_parameters,
    monodromy,
)
from qutip_trap.trap.micromotion import (
    MicromotionIndex,
    modulation_index,
    out_of_phase_amplitude_m,
)
from qutip_trap.trap.pseudopotential import DcElectrodes, RfDrive, linear_trap_parameters, mathieu_matrices
from qutip_trap.trap.surface import Electrodes, GaplessPlaneTrap
from qutip_trap.units import ATOMIC_MASS_KG, E_C, TWO_PI

if TYPE_CHECKING:
    from qutip_trap.species.model import Species

M12 = "milestone M12 (PLAN.md Section 4.6; not scheduled for the first release)"


def rotation_about_z(angle_rad: float) -> np.ndarray:
    """Columns: the principal axes (x', y', z) of a trap whose radial axes are rotated by ``angle_rad`` about z."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Trap:
    """A trap given by secular frequencies ``omega_hz`` (x', y', z; z the trap axis) or by voltages plus geometry."""

    omega_hz: tuple[float, float, float] | None
    axis_angle_rad: float
    """Rotation of the radial principal axes about z (explicit and rod/blade paths)."""
    rf: RfDrive | None
    dc: DcElectrodes | None
    geometry: Electrodes | None
    stray_field_v_per_m: tuple[float, float, float]
    """The true stray field (laboratory frame), hidden from the scheduler."""
    shim_voltages_v: dict[str, float]
    dc_schedule: dict[str, np.ndarray] | None = None
    """V_n(t) per electrode, sampled."""
    basis_potentials: dict[str, Callable[..., float]] | None = None
    """phi_tilde_n(r) per electrode, plus "rf" for phi_tilde_rf."""
    anharmonic_terms: AnharmonicTerms | None = None
    """Opt-in anharmonic couplings for H_anh/H_curv (``anharmonic.coulomb_anharmonic_terms``); None switches them off."""

    def __post_init__(self) -> None:
        explicit = self.omega_hz is not None
        geometry = self.rf is not None and self.geometry is not None
        if not (explicit or geometry):
            raise ValueError("a Trap needs either omega_hz or (rf, geometry[, dc])")
        if explicit and self.omega_hz is not None and any(w <= 0.0 for w in self.omega_hz):
            raise ValueError("secular frequencies must be positive (ordinary Hz)")
        if len(self.stray_field_v_per_m) != 3:
            raise ValueError("stray_field_v_per_m is a laboratory-frame 3-vector")
        # only the surface path has an electrode field model at the null; a shim elsewhere is refused rather than dropped
        surface = geometry and self.geometry is not None and self.geometry.is_surface
        if not surface and any(v != 0.0 for v in self.shim_voltages_v.values()):
            which = "the explicit-frequency path" if explicit and not geometry else "a rod or blade trap"
            raise ValueError(
                f"{which} has no electrode model to convert shim voltages into fields; "
                "fold the compensation into stray_field_v_per_m or give the geometry"
            )

    # ---- paths and frames -----------------------------------------------------------------------------------

    @property
    def path(self) -> str:
        """'explicit' (secular frequencies), 'linear' (rods or blades) or 'surface' (gapless-plane electrodes)."""
        if self.geometry is not None and self.rf is not None:
            return "surface" if self.geometry.is_surface else "linear"
        return "explicit"

    def surface_model(self) -> GaplessPlaneTrap:
        if self.path != "surface" or self.geometry is None:
            raise ValueError("only a surface-electrode trap has a gapless-plane model")
        return GaplessPlaneTrap(self.geometry)

    def dc_voltages(self) -> dict[str, float]:
        """dc plus shim voltages by electrode name (shims are dc electrodes used for compensation)."""
        volts = dict(self.dc.voltages_v) if self.dc is not None else {}
        for name, v in self.shim_voltages_v.items():
            volts[name] = volts.get(name, 0.0) + v
        return volts

    def rf_null_m(self) -> np.ndarray:
        """The rf null in the laboratory frame: the origin on the explicit and linear paths."""
        if self.path == "surface":
            return self.surface_model().rf_null()
        return np.zeros(3)

    def residual_field_v_per_m(self) -> np.ndarray:
        """stray + the dc and shim electrodes' field at the rf null on the surface path; stray alone otherwise."""
        e = np.asarray(self.stray_field_v_per_m, dtype=float)
        if self.path == "surface":
            model = self.surface_model()
            e = e + model.dc_field(model.rf_null(), self.dc_voltages())
        return e

    def principal_axes(self, species: Species | None = None) -> np.ndarray:
        """Columns (x', y', z) in the laboratory frame."""
        if self.path == "explicit":
            return rotation_about_z(self.axis_angle_rad)
        if self.path == "linear":
            return rotation_about_z(self.axis_angle_rad)
        if species is None:
            raise ValueError(
                "a surface trap's principal axes depend on the pseudopotential and need a species"
            )
        return self.mathieu(species).principal_axes

    # ---- Mathieu parameters, frequencies and micromotion ------------------------------------------------------

    def mathieu(self, species: Species) -> MathieuParameters:
        """a, q (matrices), beta, secular frequencies and C0 for ``species`` in this trap."""
        mass = species.mass_u * ATOMIC_MASS_KG
        if self.path == "explicit":
            if self.rf is None or self.omega_hz is None:
                raise ValueError(
                    "Mathieu parameters need the rf frequency: give Trap.rf (an RfDrive) alongside omega_hz, or run "
                    "with micromotion=None (pseudopotential, C0 = 1)"
                )
            a, q = mathieu_from_secular(self.omega_hz, self.rf.frequency_hz)
            return _with_mass(
                mathieu_parameters(a, q, self.rf.frequency_hz, axes=rotation_about_z(self.axis_angle_rad)),
                mass,
            )
        assert self.rf is not None and self.geometry is not None
        omega_rf = self.rf.omega_rad_s
        if self.path == "linear":
            p = self.geometry.parameters
            volts = self.dc_voltages()
            if len(volts) != 1:
                raise ValueError("a rod or blade trap carries one endcap voltage U0 in its dc record")
            u0 = next(iter(volts.values()))
            a_t, q_t = linear_trap_parameters(
                v_rf_peak_v=self.rf.voltage_peak_v,
                u_dc_v=u0,
                omega_rf_rad_s=omega_rf,
                mass_kg=mass,
                r_m=p["R_m"],
                z0_m=p["Z0_m"],
                kappa=p["kappa"],
            )
            return _with_mass(
                mathieu_parameters(
                    a_t, q_t, self.rf.frequency_hz, axes=rotation_about_z(self.axis_angle_rad)
                ),
                mass,
            )
        model = self.surface_model()
        null = model.rf_null()
        h_rf = self.rf.voltage_peak_v * model.rf_hessian_unit(null)
        h_dc = model.dc_hessian(null, self.dc_voltages())
        a_m, q_m = mathieu_matrices(h_dc, h_rf, mass, omega_rf)
        return _with_mass(mathieu_parameters(a_m, q_m, self.rf.frequency_hz), mass)

    def secular_hz(self, species: Species) -> tuple[float, float, float]:
        """(x', y', z) secular frequencies of ``species``: the explicit ones, or those of the Mathieu solution."""
        if self.path == "explicit" and self.omega_hz is not None and self.rf is None:
            return self.omega_hz
        return self.mathieu(species).secular_hz

    def single_ion_frequencies_rad_s(
        self, species: tuple[Species, ...], *, reference: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(omega (N, 3) in rad/s along the principal axes, axes, residual field) for the crystal solver; on the explicit
        path other species than ``species[reference]`` need the rf frequency, their (a, q) scaling as m_ref/m_i."""
        n = len(species)
        ref = species[reference]
        field = self.residual_field_v_per_m()
        if self.path == "explicit":
            assert self.omega_hz is not None
            axes = rotation_about_z(self.axis_angle_rad)
            omega = np.zeros((n, 3))
            same = [s.mass_u == ref.mass_u for s in species]
            if self.rf is None:
                if not all(same):
                    raise ValueError(
                        "a mixed-species crystal on the explicit-frequency path needs the rf frequency (Trap.rf) to "
                        "split the rf and static parts of the confinement (Section 4.1.7)"
                    )
                omega[:] = TWO_PI * np.asarray(self.omega_hz, dtype=float)
                return omega, axes, field
            a, q = mathieu_from_secular(self.omega_hz, self.rf.frequency_hz)
            omega_rf = self.rf.omega_rad_s
            for i, s in enumerate(species):
                if same[i]:
                    omega[i] = TWO_PI * np.asarray(self.omega_hz, dtype=float)
                    continue
                ratio = ref.mass_u / s.mass_u
                omega[i] = [monodromy(a[k] * ratio, q[k] * ratio).beta * omega_rf / 2.0 for k in range(3)]
            return omega, axes, field
        cache: dict[float, MathieuParameters] = {}
        omega = np.zeros((n, 3))
        for i, s in enumerate(species):
            if s.mass_u not in cache:
                cache[s.mass_u] = self.mathieu(s)
            omega[i] = TWO_PI * np.asarray(cache[s.mass_u].secular_hz, dtype=float)
        return omega, cache[ref.mass_u].principal_axes, field

    def micromotion_amplitude_m(self, species: Species) -> np.ndarray:
        """The signed in-phase excess-micromotion amplitude u_1 = -(1/2) Q u_0 (laboratory frame, peak), u_0 the residual
        field's static displacement against the pseudopotential spring. The minus sign follows the Mathieu origin
        a - 2q cos 2xi: the in-phase micromotion at the rf phase origin is a contraction."""
        params = self.mathieu(species)
        mass = species.mass_u * ATOMIC_MASS_KG
        axes = params.principal_axes
        e_res = axes.T @ self.residual_field_v_per_m()
        # a static force is balanced by the pseudopotential spring m (Omega/2)^2 (a + q^2/2), not by the exponent beta^2
        spring = params.pseudopotential_spring()
        confined = np.diag(spring) > 0.0
        u0 = np.zeros(3)
        if np.any(confined):
            idx = np.ix_(confined, confined)
            u0[confined] = np.linalg.solve(mass * spring[idx], E_C * e_res[confined])
        amp = -0.5 * np.asarray(params.q, dtype=float) @ u0
        return np.asarray(axes @ amp, dtype=float)

    def micromotion_beta(self, species: Species, delta_k: np.ndarray) -> MicromotionIndex:
        """Residual beta = delta_k . u_1 for the full wavevector as (signed in_phase, out_of_phase), peak convention:
        in_phase from the stray field (nullable by shims), out_of_phase Berkeland's (1/4) q_x R alpha phi_ac along x'
        (needs the rod geometry factors ``R_m`` and ``alpha``; not nullable)."""
        params = self.mathieu(species)
        axes = params.principal_axes
        amp_lab = self.micromotion_amplitude_m(species)
        q_axes = np.array(params.q_effective)
        dk = np.asarray(delta_k, dtype=float)
        in_phase = modulation_index(dk, amp_lab)
        out_of_phase = 0.0
        if self.rf is not None and self.rf.phase_imbalance_rad != 0.0:
            if (
                self.geometry is None
                or "alpha" not in self.geometry.parameters
                or "R_m" not in self.geometry.parameters
            ):
                raise ValueError(
                    "the out-of-phase micromotion (1/4) q_x R alpha phi_ac needs the rod geometry factors R_m and alpha"
                )
            p = self.geometry.parameters
            u_q = out_of_phase_amplitude_m(q_axes[0], p["R_m"], p["alpha"], self.rf.phase_imbalance_rad)
            out_of_phase = modulation_index(dk, axes[:, 0] * u_q)
        return MicromotionIndex(in_phase=in_phase, out_of_phase=out_of_phase, convention="peak")

    def anharmonic(self) -> AnharmonicTerms | None:
        """The opt-in anharmonic record; None means the term is off."""
        return self.anharmonic_terms

    def pseudopotential_v(self, r_m: np.ndarray, species: Species) -> float:
        """phi_ps in volts; not implemented."""
        raise NotImplementedError(f"Trap.pseudopotential_v is {M12}")

    def split_coefficients(self, t_s: float) -> tuple[float, float, float]:
        """(alpha, beta, gamma) of the volt potential beta x^4 + alpha x^2 + gamma x; not implemented."""
        raise NotImplementedError(f"Trap.split_coefficients is {M12}")


def _with_mass(params: MathieuParameters, mass_kg: float) -> MathieuParameters:
    return MathieuParameters(
        a=params.a,
        q=params.q,
        beta=params.beta,
        secular_hz=params.secular_hz,
        C0=params.C0,
        axes=params.axes,
        omega_rf_hz=params.omega_rf_hz,
        mass_kg=mass_kg,
    )
