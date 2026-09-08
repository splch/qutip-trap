"""Level-A rate coefficients per mode from the Bloch model, with the per-ion participation (PLAN.md Sections 4.2.1,
4.2.2, 4.2.8; milestone M3).

For every illuminated ion i and mode m the phonon heating and cooling rates are, with eta^2 INSIDE,

    heating_{i,m} = c_{i,m}^2 [S_i(+nu_m) + 2 D_i,m],   cooling_{i,m} = c_{i,m}^2 [S_i(-nu_m) + 2 D_i,m],

S_i the dipole-force fluctuation spectrum of ion i's internal steady state under all the beams it sees (the semi-analytic
path of Section 4.2.8 iii, exact in the saturation, QuTiP's spectrum(omega) = W(Delta - omega) pinned by the M3a
tests) and 2 D_i,m the emission recoil diffusion with one alpha per channel and one wavenumber per line (Section 4.2.8
ii); the participation enters through the ion's mass-weighted eigenvector component c_{i,m} of Section 4.1.3 folded
into the zero-point length, so the rates carry the weight W_k = sum_{i illuminated} c_{i,m}^2 (...) of Section 4.2
and the steady state nbar_m = sum_i heating/(sum_i cooling - sum_i heating) is independent of the participation
when every ion sees the same light (Section 4.2.2: the b cancels between eta~ and eta). The alternative ``closed_form``
method is the W(Delta -+ nu) form of Section 4.2.2 with each beam shifted in turn on the full multi-level model
(the repumpers stay in place, so a Lambda-type scheme keeps scattering), refused above Omega/Gamma = 0.1 unless the
caller allows the saturation error (Section 4.2.8 vii; the M3a finding that the saturated W is off by 1 + s).

A mode whose projection on every cooling beam and every illuminated ion is below the threshold is not cooled: the
module raises ``UncooledModeError`` rather than returning a steady state for it (Section 4.2, first paragraph). The
same refusal covers the mixed-species case of Section 4.2.5: a mode the ILLUMINATED ions barely participate in
(sum_i c_{i,m}^2 below ``PARTICIPATION_THRESHOLD``) carries a mathematically consistent nbar - the participation
cancels between eta~ and eta - at a relaxation rate of order 1e-2 s^-1, i.e. tens of seconds, so reporting it as the
prepared occupation of a microsecond stage is exactly what "such a mode is not cooled at all" forbids; ``min_rate_per_s``
adds the direct bound on W_m for a caller that knows its stage duration. The Section 4.2.8 (vii) conditions
eta^2 (2 nbar + 1) << 1 and W << nu, W << every internal rate are asserted here too (:mod:`qutip_trap.prep.validity`).

``stage_rates`` and ``models_per_ion`` take either one ``AtomicStructure`` for every illuminated ion or a per-ion
mapping, so one stage call can hold a coolant species and a qubit species that the cooling light does not address
(Section 4.2.5: the coolant is the illuminated set and the shared modes take their nbar from its stage).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from qutip_trap.dynamics.multilevel import ModeSpec, MultiLevelOptions
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import (
    WEAK_DRIVE_MAX,
    BlochModel,
    CoolingError,
    emission_diffusion_two_d,
    rate_coefficients_from_spectrum,
)
from qutip_trap.prep.validity import assert_adiabatic, assert_lamb_dicke
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.crystal import Crystal
from qutip_trap.units import HBAR_J_S

Method = Literal["spectrum", "closed_form"]

PROJECTION_THRESHOLD = 1e-2
"""(k_hat . e_m)^2 below which a beam does not address a mode (Section 4.2: an uncooled mode raises)."""

PARTICIPATION_THRESHOLD = 1e-4
"""sum_i c_{i,m}^2 over the ILLUMINATED ions below which the cooling light does not reach the mode (Section 4.2.5).

A mixed 171Yb+/40Ca+ pair cooled on the Ca+ leaves the two Yb-dominated radial modes at a coolant participation of
about 1e-5 and a relaxation rate of 3.4e-2 s^-1 (30 s); this threshold refuses them (the M3 finding)."""


class UncooledModeError(CoolingError):
    """No cooling beam projects on the mode at any illuminated ion: the mode is not cooled at all (Section 4.2)."""


@dataclass(frozen=True)
class IonModeRates:
    """One ion's contribution to one mode's rates (eta^2 and the participation inside; s^-1)."""

    ion: int
    mode: int
    heating_per_s: float
    cooling_per_s: float
    two_d_per_s: float
    """The emission-recoil diffusion share of both rates."""
    participation: float
    """|c_{i,m}|."""
    etas: dict[int, float]
    """eta_{i,b,m} = (k_b . e_{i,m}) c_{i,m} x0_{i,m} per beam (drive Lamb-Dicke parameters)."""
    projection_cos2: dict[int, float]
    """(k_hat_b . e_{i,m})^2 per beam."""


@dataclass(frozen=True)
class ModeRates:
    """The level-A description of one mode under the cooling light (Section 4.2, level A)."""

    mode: int
    omega_rad_s: float
    heating_per_s: float
    cooling_per_s: float
    participation_weight: float
    """sum_i c_{i,m}^2 over the illuminated ions (1 for a single ion, N for a COM mode of N equal ions)."""
    projection_cos2_max: float
    lamb_dicke_max: float
    """max over beams and illuminated ions of |eta_{i,b,m}|: the Lamb-Dicke guard's eta."""
    contributions: tuple[IonModeRates, ...]

    @property
    def rate_per_s(self) -> float:
        """W_m = cooling - heating: the exponential relaxation rate of <n_m> (Cirac Eqs. 29-32)."""
        return self.cooling_per_s - self.heating_per_s

    @property
    def nbar(self) -> float:
        """heating/(cooling - heating); raises CoolingError when the mode heats (Section 4.2.8)."""
        if self.cooling_per_s <= self.heating_per_s:
            raise CoolingError(
                f"mode {self.mode}: cooling {self.cooling_per_s:.4g} <= heating {self.heating_per_s:.4g} s^-1: no steady state"
            )
        return self.heating_per_s / (self.cooling_per_s - self.heating_per_s)

    def lamb_dicke_thermal(self) -> float:
        """eta sqrt(2 nbar + 1), the Lamb-Dicke expansion parameter at the steady state (Section 4.2.8 vii)."""
        return self.lamb_dicke_max * math.sqrt(2.0 * self.nbar + 1.0)


def ion_mode_spec(crystal: Crystal, ion: int, mode: int) -> ModeSpec | None:
    """A ModeSpec whose axis is ion i's displacement direction in mode m and whose zero-point length carries the participation,
    x0_eff = |c_{i,m}| sqrt(hbar/(2 m_i omega_m)) (through an effective mass m_i/c^2); None when the ion sits at a node."""
    m = crystal.modes[mode]
    pattern = np.asarray(m.displacement_pattern()[ion], dtype=float)
    c = float(np.linalg.norm(pattern))
    if c < 1e-12:
        return None
    axis = pattern / c
    mass = float(crystal.masses_kg[ion]) / c**2
    return ModeSpec(m.omega_rad_s, mass, (float(axis[0]), float(axis[1]), float(axis[2])), d=2)


def ion_mode_rates(
    model: BlochModel,
    crystal: Crystal,
    ion: int,
    mode: int,
    *,
    method: Method = "spectrum",
    allow_saturation: bool = False,
) -> IonModeRates | None:
    """heating and cooling rates of mode m from ion i (None at a node of the mode)."""
    spec = ion_mode_spec(crystal, ion, mode)
    if spec is None:
        return None
    etas = {b: spec.eta(beam.k_vector()) for b, beam in enumerate(model.beams)}
    cos2 = {
        b: float(np.dot(np.asarray(beam.k_hat, dtype=float), np.asarray(spec.axis))) ** 2
        for b, beam in enumerate(model.beams)
    }
    c = float(np.linalg.norm(np.asarray(crystal.modes[mode].displacement_pattern()[ion], dtype=float)))
    if all(eta == 0.0 for eta in etas.values()):
        # no beam projects on the mode at this ion: only the emission recoil heats it (no force spectrum to take)
        two_d = emission_diffusion_two_d(model, spec.axis, spec.x0_m)
        return IonModeRates(ion, mode, two_d, two_d, two_d, c, etas, cos2)
    if method == "spectrum":
        sc = rate_coefficients_from_spectrum(model, spec)
        return IonModeRates(
            ion, mode, sc.A_plus_eta2_per_s, sc.A_minus_eta2_per_s, sc.two_D_per_s, c, etas, cos2
        )
    if method != "closed_form":
        raise ValueError("method is 'spectrum' or 'closed_form'")
    sat = model.drive_saturation()
    if sat > WEAK_DRIVE_MAX and not allow_saturation:
        raise ValueError(
            f"Omega/Gamma = {sat:.3f} > {WEAK_DRIVE_MAX}: the W(Delta -+ nu) form needs Omega << Gamma (Section 4.2.8 vii); "
            "use method='spectrum' or allow_saturation=True"
        )
    two_d = emission_diffusion_two_d(model, spec.axis, spec.x0_m)
    heating = two_d
    cooling = two_d
    for b, eta in etas.items():
        if eta == 0.0:
            continue
        heating += eta**2 * model.shifted(b, -spec.omega_rad_s).scattering_rate_per_s()
        cooling += eta**2 * model.shifted(b, +spec.omega_rad_s).scattering_rate_per_s()
    return IonModeRates(ion, mode, heating, cooling, two_d, c, etas, cos2)


def mode_rates(
    models: Mapping[int, BlochModel],
    crystal: Crystal,
    mode: int,
    *,
    method: Method = "spectrum",
    cooling_beams: Sequence[int] | None = None,
    projection_threshold: float = PROJECTION_THRESHOLD,
    participation_threshold: float = PARTICIPATION_THRESHOLD,
    min_rate_per_s: float | None = None,
    allow_saturation: bool = False,
    allow_strong_coupling: bool = False,
    allow_fast_cooling: bool = False,
) -> ModeRates:
    """Sum the illuminated ions' contributions to mode m; raise UncooledModeError when the cooling light does not reach it.

    Three refusals, all of them Section 4.2's "such a mode is not cooled at all": no cooling beam projects on the
    mode at any illuminated ion (``projection_threshold`` on (k_hat . e_m)^2), the illuminated ions barely participate
    in it (``participation_threshold`` on sum_i c_{i,m}^2, the mixed-species case of Section 4.2.5) and - when the
    caller states one - the relaxation rate W_m = cooling - heating falls below ``min_rate_per_s``. The Section 4.2.8
    (vii) Lamb-Dicke and adiabatic conditions are asserted on the result unless their escapes are passed.
    """
    contributions: list[IonModeRates] = []
    for ion, model in models.items():
        r = ion_mode_rates(model, crystal, ion, mode, method=method, allow_saturation=allow_saturation)
        if r is not None:
            contributions.append(r)
    if not contributions:
        raise UncooledModeError(f"mode {mode}: every illuminated ion sits at a node")
    beams = (
        list(cooling_beams)
        if cooling_beams is not None
        else list(range(next(iter(models.values())).build.n_beams))
    )
    cos2_max = max(r.projection_cos2[b] for r in contributions for b in beams)
    if cos2_max < projection_threshold:
        raise UncooledModeError(
            f"mode {mode}: the largest cooling-beam projection (k_hat . e_m)^2 = {cos2_max:.2e} is below the threshold "
            f"{projection_threshold:.0e}; the mode is not addressed by the cooling light (Section 4.2)"
        )
    weight = sum(r.participation**2 for r in contributions)
    if weight < participation_threshold:
        raise UncooledModeError(
            f"mode {mode}: the illuminated ions' participation sum_i c_(i,m)^2 = {weight:.2e} is below the threshold "
            f"{participation_threshold:.0e}; the cooling light does not reach this mode (Section 4.2.5: a spectator "
            "species' mode of a mixed crystal). Cool it with a beam on the ions that carry it, drop it from `modes`, "
            "or lower participation_threshold to accept the reported steady state"
        )
    rates = ModeRates(
        mode=mode,
        omega_rad_s=crystal.modes[mode].omega_rad_s,
        heating_per_s=sum(r.heating_per_s for r in contributions),
        cooling_per_s=sum(r.cooling_per_s for r in contributions),
        participation_weight=weight,
        projection_cos2_max=cos2_max,
        lamb_dicke_max=max(abs(r.etas[b]) for r in contributions for b in beams),
        contributions=tuple(contributions),
    )
    if min_rate_per_s is not None and rates.rate_per_s < min_rate_per_s:
        raise UncooledModeError(
            f"mode {mode}: W_m = {rates.rate_per_s:.4g} s^-1 is below the stated min_rate_per_s = "
            f"{min_rate_per_s:.4g} s^-1 (time constant {1.0 / rates.rate_per_s if rates.rate_per_s > 0 else float('inf'):.4g} s); "
            "the mode does not reach its steady state within the stage (Section 4.2)"
        )
    if rates.cooling_per_s > rates.heating_per_s:
        assert_lamb_dicke(
            rates.lamb_dicke_max,
            rates.nbar,
            allow_strong_coupling=allow_strong_coupling,
            what=f"the level-A rate model of mode {mode}",
        )
        internal = {k: v for m in models.values() for k, v in m.build.level_rates_rad_s.items()}
        assert_adiabatic(
            rates.rate_per_s,
            rates.omega_rad_s,
            internal,
            allow_fast_cooling=allow_fast_cooling,
            what=f"the level-A rate model of mode {mode}",
        )
    return rates


@dataclass(frozen=True)
class StageRates:
    """The level-A description of one cooling stage: every mode's rates and the objective's weights (Section 4.2)."""

    modes: tuple[ModeRates, ...]
    weights: tuple[float, ...]
    """The participation weights of the objective, one per mode."""
    illuminated: tuple[int, ...]
    scattering_rate_per_s: dict[int, float]
    """W(Delta) at rest per illuminated ion: the photon rate of the cooling light."""
    method: Method
    approximations: tuple[str, ...]

    @property
    def nbar(self) -> dict[int, float]:
        return {m.mode: m.nbar for m in self.modes}

    @property
    def rates_per_s(self) -> dict[int, float]:
        return {m.mode: m.rate_per_s for m in self.modes}

    @property
    def weighted_nbar(self) -> float:
        """sum_m w_m nbar_m / sum_m w_m: the objective the beam parameters are chosen to minimize (Section 4.2.1)."""
        w = np.asarray(self.weights)
        n = np.array([m.nbar for m in self.modes])
        return float(np.dot(w, n) / np.sum(w))

    def lamb_dicke_guard(self) -> dict[int, float]:
        """eta sqrt(2 nbar + 1) per mode: Section 4.2.8 (vii) requires this << 1 for the level-A/B description."""
        return {m.mode: m.lamb_dicke_thermal() for m in self.modes}

    def mode(self, index: int) -> ModeRates:
        for m in self.modes:
            if m.mode == index:
                return m
        raise KeyError(f"mode {index} is not in this stage")


def stage_rates(
    structure: AtomicStructure | Mapping[int, AtomicStructure],
    beams: Sequence[Beam],
    crystal: Crystal,
    *,
    cooling_beams: Sequence[int] | None = None,
    illuminated: Sequence[int] | None = None,
    weights: Mapping[int, float] | None = None,
    modes: Sequence[int] | None = None,
    levels: Sequence[str] | None = None,
    states: Sequence[str] | None = None,
    options: MultiLevelOptions | None = None,
    method: Method = "spectrum",
    projection_threshold: float = PROJECTION_THRESHOLD,
    participation_threshold: float = PARTICIPATION_THRESHOLD,
    min_rate_per_s: float | None = None,
    allow_saturation: bool = False,
    allow_strong_coupling: bool = False,
    allow_fast_cooling: bool = False,
) -> tuple[StageRates, dict[int, BlochModel]]:
    """The rates of every mode (or of ``modes``) under ``beams`` at the illuminated ions, and the per-ion models used.

    ``beams`` are all the beams the ions see, repumpers included; ``cooling_beams`` restricts the projection guard to the
    beams meant to cool. ``weights`` (mode -> weight) is the objective's participation weighting, uniform when None.
    ``structure`` is one atomic structure for every illuminated ion, or a per-ion mapping for a mixed crystal
    (Section 4.2.5: pass ``illuminated=`` the coolant ions, so the qubit species need not be addressed by the light).
    """
    ions = tuple(illuminated) if illuminated is not None else illuminated_ions(beams, crystal)
    if not ions:
        raise ValueError("no ion is illuminated by the beams")
    models = models_per_ion(structure, beams, crystal, ions, levels=levels, states=states, options=options)
    cooling = list(cooling_beams) if cooling_beams is not None else list(range(len(beams)))
    which = list(modes) if modes is not None else list(range(len(crystal.modes)))
    rates = tuple(
        mode_rates(
            models,
            crystal,
            k,
            method=method,
            cooling_beams=cooling,
            projection_threshold=projection_threshold,
            participation_threshold=participation_threshold,
            min_rate_per_s=min_rate_per_s,
            allow_saturation=allow_saturation,
            allow_strong_coupling=allow_strong_coupling,
            allow_fast_cooling=allow_fast_cooling,
        )
        for k in which
    )
    w = tuple(float(weights.get(k, 0.0)) if weights is not None else 1.0 for k in which)
    if sum(w) <= 0.0:
        raise ValueError("the mode weights must not all vanish")
    approx = list(models[ions[0]].build.approximations)
    if method == "closed_form":
        approx.append(
            "W(Delta -+ nu) closed form: saturation error of order (Omega/Gamma)^2 (Section 4.2.8 vii) [background]"
        )
    return (
        StageRates(
            modes=rates,
            weights=w,
            illuminated=ions,
            scattering_rate_per_s={i: m.scattering_rate_per_s() for i, m in models.items()},
            method=method,
            approximations=tuple(approx),
        ),
        models,
    )


def illuminated_ions(
    beams: Sequence[Beam], crystal: Crystal, *, relative_intensity: float = 1e-3
) -> tuple[int, ...]:
    """Ions at whose position some beam's intensity exceeds ``relative_intensity`` of that beam's peak."""
    out: list[int] = []
    for i in range(crystal.n_ions):
        pos = np.asarray(crystal.positions_m[i], dtype=float)
        for beam in beams:
            peak = 2.0 * beam.power_w / (math.pi * beam.waist_m**2)
            if peak > 0.0 and beam.intensity_at(pos) >= relative_intensity * peak:
                out.append(i)
                break
    return tuple(out)


def models_per_ion(
    structure: AtomicStructure | Mapping[int, AtomicStructure],
    beams: Sequence[Beam],
    crystal: Crystal,
    ions: Sequence[int],
    *,
    levels: Sequence[str] | None = None,
    states: Sequence[str] | None = None,
    options: MultiLevelOptions | None = None,
) -> dict[int, BlochModel]:
    """One internal-only Bloch model per ion, the beams evaluated at that ion's equilibrium position (crosstalk-exact).

    ``structure`` is one ``AtomicStructure`` shared by every ion in ``ions``, or a mapping ion -> structure for a
    mixed crystal (Section 4.2.5). Every ion of ``ions`` must appear in the mapping: an ion the cooling light does
    not address belongs outside the illuminated set, not in it with a guessed structure.
    """
    opts = options or MultiLevelOptions()
    if opts.recoil != "off":
        raise ValueError(
            "the level-A rate models are internal-only builds: recoil enters through the diffusion term"
        )
    if isinstance(structure, Mapping):
        missing = [i for i in ions if i not in structure]
        if missing:
            raise KeyError(
                f"no AtomicStructure for illuminated ions {missing}: a mixed-crystal stage names one structure per "
                "illuminated ion (Section 4.2.5); ions the cooling light does not address stay out of `illuminated`"
            )
        by_ion = {i: structure[i] for i in ions}
    else:
        by_ion = dict.fromkeys(ions, structure)
    return {
        i: BlochModel(
            st,
            beams,
            levels=levels,
            states=states,
            position_m=tuple(float(x) for x in crystal.positions_m[i]),
            options=opts,
        )
        for i, st in by_ion.items()
    }


def thermal_energy_j(nbar: float, omega_rad_s: float) -> float:
    """hbar omega (nbar + 1/2)."""
    return HBAR_J_S * omega_rad_s * (nbar + 0.5)


__all__ = [
    "PARTICIPATION_THRESHOLD",
    "PROJECTION_THRESHOLD",
    "IonModeRates",
    "Method",
    "ModeRates",
    "StageRates",
    "UncooledModeError",
    "illuminated_ions",
    "ion_mode_rates",
    "ion_mode_spec",
    "mode_rates",
    "models_per_ion",
    "stage_rates",
    "thermal_energy_j",
]
