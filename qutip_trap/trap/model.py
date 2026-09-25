"""The ``Trap`` record and its Mathieu parameters, principal axes, residual field and micromotion (PLAN.md Section 3.3).

Three paths lead to Mathieu parameters: EXPLICIT secular frequencies (x', y', z) of one species, rotated about z by
``axis_angle_rad``, inverted for (a, q) under the linear-trap structure when an ``RfDrive`` gives the rf frequency (without
one there is no Mathieu record and C0 = 1); a ROD or blade trap through Berkeland's map; a SURFACE layout through the
gapless-plane field solution of ``trap/surface.py`` (the Hessians at the rf null, the coupled Mathieu system, the axes of
the pseudopotential Hessian). The residual field is the stray field plus the shim electrodes' field at the null, which
only the surface path can compute, so non-zero shims are refused on the other two. The field displaces the ion against
the pseudopotential spring and drives the signed in-phase micromotion -(1/2) Q u_0.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.trap.anharmonic import AnharmonicTerms
from qutip_trap.trap.mathieu import MathieuParameters, mathieu_from_secular, mathieu_parameters, monodromy
from qutip_trap.trap.micromotion import MicromotionIndex, modulation_index, out_of_phase_amplitude_m
from qutip_trap.trap.pseudopotential import DcElectrodes, RfDrive, linear_trap_parameters, mathieu_matrices
from qutip_trap.trap.surface import Electrodes, GaplessPlaneTrap
from qutip_trap.units import ATOMIC_MASS_KG, E_C, TWO_PI

if TYPE_CHECKING:
    from qutip_trap.species.model import Species


def rotation_about_z(angle_rad: float) -> np.ndarray:
    """Columns: the principal axes (x', y', z) of a trap whose radial axes are rotated by ``angle_rad`` about z."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Trap:
    """Explicit secular frequencies (``omega_hz``, ordinary Hz, z the trap axis) or voltages plus geometry.

    ``stray_field_v_per_m`` is the TRUE stray field (the scheduler never sees it) and ``shim_voltages_v`` the applied
    compensation; ``anharmonic_terms`` is the opt-in cubic/quartic record (``anharmonic.coulomb_anharmonic_terms``).
    """

    omega_hz: tuple[float, float, float] | None
    axis_angle_rad: float
    """Rotation of the radial principal axes about z (explicit and rod paths)."""
    rf: RfDrive | None
    dc: DcElectrodes | None
    geometry: Electrodes | None
    stray_field_v_per_m: tuple[float, float, float]
    shim_voltages_v: dict[str, float]
    anharmonic_terms: AnharmonicTerms | None = None

    def __post_init__(self) -> None:
        explicit = self.omega_hz is not None
        geometry = self.rf is not None and self.geometry is not None
        if not (explicit or geometry):
            raise ValueError("a Trap needs either omega_hz or (rf, geometry[, dc])")
        if self.omega_hz is not None and any(w <= 0.0 for w in self.omega_hz):
            raise ValueError("secular frequencies must be positive (ordinary Hz)")
        if len(self.stray_field_v_per_m) != 3:
            raise ValueError("stray_field_v_per_m is a laboratory-frame 3-vector")
        surface = geometry and self.geometry is not None and self.geometry.is_surface
        if not surface and any(v != 0.0 for v in self.shim_voltages_v.values()):
            which = "the explicit-frequency path" if not geometry else "a rod or blade trap"
            raise ValueError(
                f"{which} has no electrode model to convert shim voltages into fields; "
                "fold the compensation into stray_field_v_per_m or give the geometry"
            )

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
        """The rf null in the laboratory frame: the origin on the explicit and rod paths."""
        if self.path == "surface":
            return self.surface_model().rf_null()
        return np.zeros(3)

    def residual_field_v_per_m(self) -> np.ndarray:
        """The stray field plus, on the surface path, the dc and shim electrodes' field at the rf null."""
        e = np.asarray(self.stray_field_v_per_m, dtype=float)
        if self.path == "surface":
            model = self.surface_model()
            e = e + model.dc_field(model.rf_null(), self.dc_voltages())
        return e

    def principal_axes(self, species: Species | None = None) -> np.ndarray:
        """Columns (x', y', z) in the laboratory frame; a surface trap's depend on the species' pseudopotential."""
        if self.path != "surface":
            return rotation_about_z(self.axis_angle_rad)
        if species is None:
            raise ValueError(
                "a surface trap's principal axes depend on the pseudopotential and need a species"
            )
        return self.mathieu(species).principal_axes

    def mathieu(self, species: Species) -> MathieuParameters:
        """a, q, beta, secular frequencies and C0 of ``species`` in this trap."""
        mass = species.mass_u * ATOMIC_MASS_KG
        if self.path == "surface":
            assert self.rf is not None
            model = self.surface_model()
            null = model.rf_null()
            h_rf = self.rf.voltage_peak_v * model.rf_hessian_unit(null)
            h_dc = model.dc_hessian(null, self.dc_voltages())
            a_m, q_m = mathieu_matrices(h_dc, h_rf, mass, self.rf.omega_rad_s)
            params = mathieu_parameters(a_m, q_m, self.rf.frequency_hz)
        elif self.path == "linear":
            assert self.rf is not None and self.geometry is not None
            volts = self.dc_voltages()
            if len(volts) != 1:
                raise ValueError("a rod or blade trap carries one endcap voltage U0 in its dc record")
            p = self.geometry.parameters
            a_t, q_t = linear_trap_parameters(
                v_rf_peak_v=self.rf.voltage_peak_v,
                u_dc_v=next(iter(volts.values())),
                omega_rf_rad_s=self.rf.omega_rad_s,
                mass_kg=mass,
                r_m=p["R_m"],
                z0_m=p["Z0_m"],
                kappa=p["kappa"],
            )
            params = mathieu_parameters(
                a_t, q_t, self.rf.frequency_hz, axes=rotation_about_z(self.axis_angle_rad)
            )
        else:
            if self.rf is None:
                raise ValueError(
                    "Mathieu parameters need the rf frequency: give Trap.rf (an RfDrive) alongside omega_hz, or run "
                    "with micromotion=None (pseudopotential, C0 = 1)"
                )
            assert self.omega_hz is not None
            a, q = mathieu_from_secular(self.omega_hz, self.rf.frequency_hz)
            params = mathieu_parameters(
                a, q, self.rf.frequency_hz, axes=rotation_about_z(self.axis_angle_rad)
            )
        return dataclasses.replace(params, mass_kg=mass)

    def secular_hz(self, species: Species) -> tuple[float, float, float]:
        """(x', y', z) secular frequencies of ``species``: the explicit ones, or those of the Mathieu solution."""
        if self.path == "explicit" and self.omega_hz is not None and self.rf is None:
            return self.omega_hz
        return self.mathieu(species).secular_hz

    def single_ion_frequencies_rad_s(
        self, species: tuple[Species, ...], *, reference: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(omega (N, 3) in rad/s along the principal axes, axes, residual field) for the crystal solver.

        Explicit path: the frequencies describe ``species[reference]``; another species needs the rf frequency, from which
        (a, q) scale as m_ref/m_i and the exact exponents give its frequencies. Geometry paths: each species from the
        field solution, the reference species' axes the crystal frame.
        """
        n = len(species)
        ref = species[reference]
        field = self.residual_field_v_per_m()
        omega = np.zeros((n, 3))
        if self.path == "explicit":
            assert self.omega_hz is not None
            axes = rotation_about_z(self.axis_angle_rad)
            same = [s.mass_u == ref.mass_u for s in species]
            if self.rf is None:
                if not all(same):
                    raise ValueError(
                        "a mixed-species crystal on the explicit-frequency path needs the rf frequency (Trap.rf) to "
                        "split the rf and static parts of the confinement"
                    )
                omega[:] = TWO_PI * np.asarray(self.omega_hz, dtype=float)
                return omega, axes, field
            a, q = mathieu_from_secular(self.omega_hz, self.rf.frequency_hz)
            for i, s in enumerate(species):
                if same[i]:
                    omega[i] = TWO_PI * np.asarray(self.omega_hz, dtype=float)
                    continue
                ratio = ref.mass_u / s.mass_u
                omega[i] = [
                    monodromy(a[k] * ratio, q[k] * ratio).beta * self.rf.omega_rad_s / 2.0 for k in range(3)
                ]
            return omega, axes, field
        cache: dict[float, MathieuParameters] = {}
        for i, s in enumerate(species):
            if s.mass_u not in cache:
                cache[s.mass_u] = self.mathieu(s)
            omega[i] = TWO_PI * np.asarray(cache[s.mass_u].secular_hz, dtype=float)
        return omega, cache[ref.mass_u].principal_axes, field

    def micromotion_amplitude_m(self, species: Species) -> np.ndarray:
        """The SIGNED in-phase excess-micromotion amplitude u_1 = -(1/2) Q u_0 (peak, laboratory frame).

        u_0 is the static displacement of the residual field against the PSEUDOPOTENTIAL spring m (Omega/2)^2 (a + q^2/2),
        the period average of the driven Mathieu equation, not the exact beta^2 (they differ by 0.39 q^2); Berkeland's
        per-axis -(1/2) q_i u_0i is its matrix form when the dc axes are rotated against the rf Hessian.
        """
        params = self.mathieu(species)
        axes = params.principal_axes
        e_res = axes.T @ self.residual_field_v_per_m()
        spring = params.pseudopotential_spring()
        confined = np.diag(spring) > 0.0
        u0 = np.zeros(3)
        if np.any(confined):
            idx = np.ix_(confined, confined)
            u0[confined] = np.linalg.solve(
                species.mass_u * ATOMIC_MASS_KG * spring[idx], E_C * e_res[confined]
            )
        amp = -0.5 * np.asarray(params.q, dtype=float) @ u0
        return np.asarray(axes @ amp, dtype=float)

    def micromotion_beta(self, species: Species, delta_k: np.ndarray) -> MicromotionIndex:
        """The micromotion index of a drive of full wavevector ``delta_k``: signed in-phase delta_k . u_1 and
        Berkeland's out-of-phase (1/4) q_x R alpha phi_ac along x' (needs the rod factors ``R_m`` and ``alpha``)."""
        params = self.mathieu(species)
        dk = np.asarray(delta_k, dtype=float)
        in_phase = modulation_index(dk, self.micromotion_amplitude_m(species))
        out_of_phase = 0.0
        if self.rf is not None and self.rf.phase_imbalance_rad != 0.0:
            p = {} if self.geometry is None else self.geometry.parameters
            if "alpha" not in p or "R_m" not in p:
                raise ValueError(
                    "the out-of-phase micromotion (1/4) q_x R alpha phi_ac needs the rod geometry factors R_m and alpha"
                )
            u_q = out_of_phase_amplitude_m(
                params.q_effective[0], p["R_m"], p["alpha"], self.rf.phase_imbalance_rad
            )
            out_of_phase = modulation_index(dk, params.principal_axes[:, 0] * u_q)
        return MicromotionIndex(in_phase=in_phase, out_of_phase=out_of_phase)

    def anharmonic(self) -> AnharmonicTerms | None:
        """The opt-in anharmonic record; None means the term is off."""
        return self.anharmonic_terms
