"""The ``Trap`` record (PLAN.md Section 3.3; Appendix E; Sections 4.1.1, 4.1.6; milestone M1).

Two paths lead from a ``Trap`` to Mathieu parameters (Section 3.3):

- the EXPLICIT path, ``omega_hz`` = measured secular frequencies (x', y', z) of one species with the radial
  principal axes rotated about z by ``axis_angle_rad``; with an ``RfDrive`` (its frequency) the exact exponents are
  inverted for (a, q) under the linear-trap structure q_z = 0, q_y = -q_x, a_x + a_y + a_z = 0
  (``mathieu_from_secular``), which is what gives C0 and the micromotion amplitudes; without an rf frequency there
  are no Mathieu parameters and ``Crystal.lamb_dicke`` runs with ``micromotion=None`` (C0 = 1);
- the GEOMETRY path, ``rf`` + ``geometry`` (+ ``dc``): a rod or blade trap through Berkeland's map
  (``linear_trap_parameters``), or a surface-electrode layout through the gapless-plane field solution of
  ``trap/surface.py``: rf null, static and rf Hessians there, A = 4Q H_dc/(m Omega^2), Q = 2Q H_rf/(m Omega^2), the
  coupled Mathieu system by monodromy, the principal axes from the pseudopotential Hessian (Section 4.1.6).

The stray field and the shims exist because compensation is something a laboratory measures and re-nulls
(2026-09-04 experimentalist critique); the residual field is stray + shim response, where the shim response is the
field of the named electrodes at the null on the SURFACE path and undefined on the explicit and rod/blade paths
(non-zero shim voltages on either is refused, not dropped: Berkeland's rod map carries no shim -> field response, and
PLAN 4.1.1 states that no source gives the geometry factors kappa, R', Z0 or Phi'' for any electrode design). The scheduler never sees ``stray_field_v_per_m`` (Section 7.3). The residual field
displaces the ion to u_0 = Q E/(m omega^2) per axis and the in-phase micromotion index of a drive is
-delta_k . sum_i (q_i/2) u_0i e_i (peak, signed; the minus sign is the adopted rf phase origin of Section 13, under which
the in-phase micromotion is a contraction); the out-of-phase index comes from Berkeland's (1/4) q_x R alpha phi_ac term
and needs the rod geometry factors (Section 4.1.1).
"""

from __future__ import annotations

import math
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


def rotation_about_z(angle_rad: float) -> np.ndarray:
    """Columns: the principal axes (x', y', z) of a trap whose radial axes are rotated by ``angle_rad`` about z."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Trap:
    """Either explicit secular frequencies or voltages plus geometry (Section 3.3).

    ``omega_hz`` are ordinary frequencies (x', y', z) with z the trap axis (Section 5.6); ``axis_angle_rad`` rotates
    the radial principal axes about z on the explicit path (a mode with no projection on any cooling beam is
    uncoolable, so the frame must be free). ``anharmonic_terms`` is the opt-in H_anh/H_curv record of Section 5.7
    (``anharmonic.coulomb_anharmonic_terms`` computes the Coulomb couplings of a crystal); None switches the term off.
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
    anharmonic_terms: AnharmonicTerms | None = None
    """Opt-in anharmonic couplings for H_anh/H_curv (Section 5.7); what ``anharmonic()`` returns."""

    def __post_init__(self) -> None:
        explicit = self.omega_hz is not None
        geometry = self.rf is not None and self.geometry is not None
        if not (explicit or geometry):
            raise ValueError("a Trap needs either omega_hz or (rf, geometry[, dc])")
        if explicit and self.omega_hz is not None and any(w <= 0.0 for w in self.omega_hz):
            raise ValueError("secular frequencies must be positive (ordinary Hz)")
        if len(self.stray_field_v_per_m) != 3:
            raise ValueError("stray_field_v_per_m is a laboratory-frame 3-vector")
        # Only the SURFACE path has an electrode field model at the null. The explicit path has no electrodes, and the
        # rod/blade map is Berkeland's (kappa, R', Z0) quadrupole map with no shim -> field response: PLAN 4.1.1, "No
        # source gives the geometry factors kappa, R', Z0 or Phi'' for any electrode design". A non-zero shim there
        # contributed nothing to residual_field_v_per_m and, when named like the endcap, silently changed a_z instead of
        # producing a field, so it is refused rather than dropped (audit item E.4).
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
        """stray + the dc and shim electrodes' field at the rf null (SURFACE path); stray alone otherwise.

        The explicit and rod/blade paths carry no shim -> field response (``__post_init__`` refuses non-zero shims there),
        so nothing is silently dropped here."""
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

    # ---- Appendix E methods -----------------------------------------------------------------------------------

    def mathieu(self, species: Species) -> MathieuParameters:
        """a, q (matrices), beta, secular frequencies, C0 (Section 4.1.1) for ``species`` in this trap."""
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
        """(omega (N, 3) in rad/s along the principal axes, axes, residual field) for the crystal solver (Section 4.1.7).

        Explicit path: the frequencies describe ``species[reference]``; other species need the rf frequency, from
        which (a, q) scale as m_ref/m_i and the exact exponents give their frequencies. Geometry path: per species
        from the field solution; the reference species' principal axes are the crystal frame.
        """
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
        """The SIGNED in-phase excess-micromotion amplitude vector u_1 in the laboratory frame, peak convention (Section 4.1.1):
        -(1/2) Q u_0 with u_0 the static displacement of the residual field against the pseudopotential spring. The minus
        sign is the adopted Mathieu origin a - 2q cos 2xi (Section 13, "Floquet function and rf phase origin"), under which
        the in-phase micromotion at the rf phase origin is a contraction, x_mu(t) = -(q_x/2) x_sec(t) cos(omega_rf t); beta
        along a wavevector k is the signed k . u_1 (``micromotion_beta``), and that sign is what an rf-photon-correlation
        signal crosses through at the compensated shim voltage (M8)."""
        params = self.mathieu(species)
        mass = species.mass_u * ATOMIC_MASS_KG
        axes = params.principal_axes
        e_res = axes.T @ self.residual_field_v_per_m()
        # a static force is balanced by the PSEUDOPOTENTIAL spring m (Omega/2)^2 (a + q^2/2) + O(q^4), the period average of
        # the driven Mathieu equation's particular solution, not by the exact exponent beta^2 (they differ by 0.39 q^2);
        # Berkeland's per-axis -(1/2) q_i u_0i is the matrix form -(1/2) Q u_0 when the dc axes are rotated against the rf Hessian
        spring = params.pseudopotential_spring()
        confined = np.diag(spring) > 0.0
        u0 = np.zeros(3)
        if np.any(confined):
            idx = np.ix_(confined, confined)
            u0[confined] = np.linalg.solve(mass * spring[idx], E_C * e_res[confined])
        amp = -0.5 * np.asarray(params.q, dtype=float) @ u0
        return np.asarray(axes @ amp, dtype=float)

    def micromotion_beta(self, species: Species, delta_k: np.ndarray) -> MicromotionIndex:
        """Residual beta = delta_k . u_1 for the FULL wavevector, as (SIGNED in_phase, out_of_phase), peak convention (Section 4.1.1).

        in_phase: the stray-field part, -delta_k . (1/2) Q u_0 with u_0 the static displacement against the pseudopotential
        spring m (Omega/2)^2 (a + q^2/2) (Berkeland's per-axis -(1/2) q_i u_0i when the dc and rf axes coincide), nullable by shims
        and SIGNED, so that it steps by pi across the compensated shim (Section 9.17);
        out_of_phase: Berkeland's (1/4) q_x R alpha phi_ac along x', from ``RfDrive.phase_imbalance_rad`` and the rod
        geometry factors, not nullable. The two never collapse into one number.
        """
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
        """The opt-in anharmonic record (Section 5.7); None means the term is off."""
        return self.anharmonic_terms


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
