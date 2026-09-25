"""Doppler cooling at level A: per-mode heating and cooling rates from the multi-level Bloch model (PLAN.md Section 4.2).

For every illuminated ion i and mode m, with eta^2 inside,

    heating_{i,m} = c_{i,m}^2 [S_i(+nu_m) + 2 D_i,m],   cooling_{i,m} = c_{i,m}^2 [S_i(-nu_m) + 2 D_i,m],

S_i the dipole-force fluctuation spectrum of ion i's steady state under all its beams (exact in the saturation) and 2 D the
emission recoil diffusion with one alpha per polarization channel; the participation c_{i,m} (the ion's mass-weighted
eigenvector component) enters through the zero-point length, so nbar_m = sum_i heating/(sum_i cooling - sum_i heating)
is participation independent when every ion sees the same light. A Doppler beam has one detuning for all modes:
``optimize_detuning`` moves a beam group's common frequency to minimize the participation-weighted mean nbar. The RMP
force model is reported as a cross-check for the modes with nu/Gamma < 0.1. A mode the cooling light does not reach
(no beam projects on it, or the illuminated ions barely participate in it: a spectator species of a mixed crystal) raises
``UncooledModeError``; the Lamb-Dicke, adiabatic and Gamma > omega_R conditions are asserted (``prep.validity``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from qutip_trap.dynamics.multilevel import ModeSpec, MultiLevelOptions
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import (
    BlochModel,
    CoolingError,
    emission_diffusion_two_d,
    rate_coefficients_from_spectrum,
    shifted_beam,
)
from qutip_trap.prep.closed_forms import doppler_force_nbar
from qutip_trap.prep.validity import assert_adiabatic, assert_doppler_recoil_limit, assert_lamb_dicke
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.crystal import Crystal

PROJECTION_THRESHOLD = 1e-2
"""(k_hat . e_m)^2 below which a beam does not address a mode."""

PARTICIPATION_THRESHOLD = 1e-4
"""sum_i c_{i,m}^2 over the illuminated ions below which the cooling light does not reach the mode: a 171Yb+/40Ca+ pair
cooled on the Ca+ leaves the Yb-dominated radial modes at about 1e-5, relaxing at 3.4e-2 s^-1 (30 s)."""

FORCE_MODEL_MAX_NU_OVER_GAMMA = 0.1
"""The force model is reported only for modes with nu/Gamma below this."""


class UncooledModeError(CoolingError):
    """No cooling beam reaches the mode through the illuminated ions: the mode is not cooled at all."""


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
    """eta_{i,b,m} = (k_b . e_{i,m}) c_{i,m} x0_{i,m} per beam."""
    projection_cos2: dict[int, float]
    """(k_hat_b . e_{i,m})^2 per beam."""


@dataclass(frozen=True)
class ModeRates:
    """The level-A description of one mode under the cooling light."""

    mode: int
    omega_rad_s: float
    heating_per_s: float
    cooling_per_s: float
    participation_weight: float
    """sum_i c_{i,m}^2 over the illuminated ions (1 for a single ion, N for a COM mode of N equal ions)."""
    projection_cos2_max: float
    lamb_dicke_max: float
    """max over beams and illuminated ions of |eta_{i,b,m}|."""
    contributions: tuple[IonModeRates, ...]

    @property
    def rate_per_s(self) -> float:
        """W_m = cooling - heating: the exponential relaxation rate of <n_m>."""
        return self.cooling_per_s - self.heating_per_s

    @property
    def nbar(self) -> float:
        """heating/(cooling - heating); raises CoolingError when the mode heats."""
        if self.cooling_per_s <= self.heating_per_s:
            raise CoolingError(
                f"mode {self.mode}: cooling {self.cooling_per_s:.4g} <= heating {self.heating_per_s:.4g} s^-1: "
                "no steady state"
            )
        return self.heating_per_s / (self.cooling_per_s - self.heating_per_s)

    def lamb_dicke_thermal(self) -> float:
        """eta sqrt(2 nbar + 1), the Lamb-Dicke expansion parameter at the steady state."""
        return self.lamb_dicke_max * math.sqrt(2.0 * self.nbar + 1.0)


@dataclass(frozen=True)
class DopplerResult:
    """Every mode's rates, the objective's weights, the photon rate per illuminated ion and the force-model
    cross-check (nu/Gamma < 0.1 only)."""

    modes: tuple[ModeRates, ...]
    weights: tuple[float, ...]
    illuminated: tuple[int, ...]
    scattering_rate_per_s: dict[int, float]
    """W(Delta) at rest per illuminated ion."""
    approximations: tuple[str, ...]
    force_model_nbar: dict[int, float]

    @property
    def method(self) -> str:
        """The rate method: the dipole-force spectrum."""
        return "spectrum"

    @property
    def nbar(self) -> dict[int, float]:
        return {m.mode: m.nbar for m in self.modes}

    @property
    def weighted_nbar(self) -> float:
        """sum_m w_m nbar_m / sum_m w_m: the objective the beam detuning is chosen to minimize."""
        w = np.asarray(self.weights)
        n = np.array([m.nbar for m in self.modes])
        return float(np.dot(w, n) / np.sum(w))

    def lamb_dicke_guard(self) -> dict[int, float]:
        """eta sqrt(2 nbar + 1) per mode."""
        return {m.mode: m.lamb_dicke_thermal() for m in self.modes}

    def mode(self, index: int) -> ModeRates:
        for m in self.modes:
            if m.mode == index:
                return m
        raise KeyError(f"mode {index} is not in this stage")


def ion_mode_spec(crystal: Crystal, ion: int, mode: int) -> ModeSpec | None:
    """A ModeSpec along ion i's displacement in mode m whose zero-point length carries the participation,
    x0_eff = |c_{i,m}| sqrt(hbar/(2 m_i omega_m)) (an effective mass m_i/c^2); None when the ion sits at a node."""
    m = crystal.modes[mode]
    pattern = np.asarray(m.displacement_pattern()[ion], dtype=float)
    c = float(np.linalg.norm(pattern))
    if c < 1e-12:
        return None
    axis = pattern / c
    mass = float(crystal.masses_kg[ion]) / c**2
    return ModeSpec(m.omega_rad_s, mass, (float(axis[0]), float(axis[1]), float(axis[2])), d=2)


def ion_mode_rates(model: BlochModel, crystal: Crystal, ion: int, mode: int) -> IonModeRates | None:
    """Heating and cooling rates of mode m from ion i (None at a node of the mode)."""
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
        # no beam projects on the mode at this ion: only the emission recoil heats it
        two_d = emission_diffusion_two_d(model, spec.axis, spec.x0_m)
        return IonModeRates(ion, mode, two_d, two_d, two_d, c, etas, cos2)
    sc = rate_coefficients_from_spectrum(model, spec)
    return IonModeRates(ion, mode, sc.A_plus_eta2_per_s, sc.A_minus_eta2_per_s, sc.two_D_per_s, c, etas, cos2)


def mode_rates(
    models: Mapping[int, BlochModel],
    crystal: Crystal,
    mode: int,
    *,
    cooling_beams: Sequence[int] | None = None,
    projection_threshold: float = PROJECTION_THRESHOLD,
    participation_threshold: float = PARTICIPATION_THRESHOLD,
    min_rate_per_s: float | None = None,
    allow_strong_coupling: bool = False,
) -> ModeRates:
    """Sum the illuminated ions' contributions to mode m. Raises UncooledModeError when no cooling beam projects on the
    mode, when the illuminated ions barely participate in it, or when W_m falls below a stated ``min_rate_per_s``."""
    contributions: list[IonModeRates] = []
    for ion, model in models.items():
        r = ion_mode_rates(model, crystal, ion, mode)
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
            f"{projection_threshold:.0e}; the mode is not addressed by the cooling light"
        )
    weight = sum(r.participation**2 for r in contributions)
    if weight < participation_threshold:
        raise UncooledModeError(
            f"mode {mode}: the illuminated ions' participation sum_i c_(i,m)^2 = {weight:.2e} is below the threshold "
            f"{participation_threshold:.0e}; the cooling light does not reach this mode (a spectator species' mode "
            "of a mixed crystal). Cool it with a beam on the ions that carry it, drop it from `modes`, or lower "
            "participation_threshold to accept the reported steady state"
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
        time_constant = 1.0 / rates.rate_per_s if rates.rate_per_s > 0 else float("inf")
        raise UncooledModeError(
            f"mode {mode}: W_m = {rates.rate_per_s:.4g} s^-1 is below the stated min_rate_per_s = "
            f"{min_rate_per_s:.4g} s^-1 (time constant {time_constant:.4g} s); the mode does not reach its steady "
            "state within the stage"
        )
    if rates.cooling_per_s > rates.heating_per_s:
        what = f"the level-A rate model of mode {mode}"
        assert_lamb_dicke(
            rates.lamb_dicke_max, rates.nbar, allow_strong_coupling=allow_strong_coupling, what=what
        )
        internal = {k: v for m in models.values() for k, v in m.build.level_rates_rad_s.items()}
        assert_adiabatic(rates.rate_per_s, rates.omega_rad_s, internal, what=what)
    return rates


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
    options: MultiLevelOptions | None = None,
) -> dict[int, BlochModel]:
    """One internal-only Bloch model per ion, the beams evaluated at that ion's equilibrium position.

    ``structure`` is one ``AtomicStructure`` shared by every ion, or a mapping ion -> structure for a mixed crystal;
    every ion of ``ions`` must appear in the mapping (an ion the cooling light does not address stays out of the
    illuminated set).
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
                "illuminated ion; ions the cooling light does not address stay out of `illuminated`"
            )
        by_ion = {i: structure[i] for i in ions}
    else:
        by_ion = dict.fromkeys(ions, structure)
    return {
        i: BlochModel(
            st,
            beams,
            levels=levels,
            position_m=tuple(float(x) for x in crystal.positions_m[i]),
            options=opts,
        )
        for i, st in by_ion.items()
    }


def _force_model(model: BlochModel, ion: int, rates: ModeRates, cooling_beams: Sequence[int]) -> float | None:
    """The RMP force-model nbar of one mode when exactly one cooling beam addresses one resonant line (else None).

    s = 2 sum |Omega_qp|^2/Gamma^2 over the beam's couplings into the resonant upper level, Delta of the coupling nearest
    resonance, and the emission weight alpha/cos^2 = 2D/(eta^2 W) from the same steady state as the rate framework.
    """
    b = model.build
    if len(cooling_beams) != 1:
        return None
    beam = cooling_beams[0]
    couplings = [c for c in b.couplings if c.beam == beam]
    if not couplings:
        return None
    gamma = max(b.level_rates_rad_s.get(b.level_of(c.upper), 0.0) for c in couplings)
    if gamma <= 0.0 or rates.omega_rad_s / gamma >= FORCE_MODEL_MAX_NU_OVER_GAMMA:
        return None
    resonant = min(couplings, key=lambda c: abs(c.detuning_rad_s))
    if resonant.detuning_rad_s >= 0.0:
        return None
    s = 2.0 * sum(abs(c.omega_rad_s) ** 2 for c in couplings if c.upper == resonant.upper) / gamma**2
    contrib = next((r for r in rates.contributions if r.ion == ion), None)
    if contrib is None:
        return None
    cos2 = contrib.projection_cos2[beam]
    if cos2 < PROJECTION_THRESHOLD:
        return None
    w_rest = model.scattering_rate_per_s()
    eta2 = contrib.etas[beam] ** 2
    if w_rest <= 0.0 or eta2 <= 0.0:
        return None
    alpha_over_cos2 = contrib.two_d_per_s / (w_rest * eta2)
    return doppler_force_nbar(gamma, resonant.detuning_rad_s, s, rates.omega_rad_s, alpha_over_cos2)


def _assert_doppler_regime(
    model: BlochModel, crystal: Crystal, ion: int, cooling_beams: Sequence[int], *, allow_recoil_limited: bool
) -> None:
    """Gamma > omega_R on every cooling beam's resonant line."""
    b = model.build
    mass = float(crystal.masses_kg[ion])
    for beam in cooling_beams:
        couplings = [c for c in b.couplings if c.beam == beam]
        if not couplings:
            continue
        resonant = min(couplings, key=lambda c: abs(c.detuning_rad_s))
        gamma = b.level_rates_rad_s.get(b.level_of(resonant.upper), 0.0)
        if gamma <= 0.0:
            continue
        assert_doppler_recoil_limit(
            gamma, model.beams[beam].k_rad_per_m, mass, allow_recoil_limited=allow_recoil_limited
        )


def doppler_cooling(
    structure: AtomicStructure | Mapping[int, AtomicStructure],
    beams: Sequence[Beam],
    crystal: Crystal,
    *,
    cooling_beams: Sequence[int] | None = None,
    illuminated: Sequence[int] | None = None,
    weights: Mapping[int, float] | None = None,
    modes: Sequence[int] | None = None,
    levels: Sequence[str] | None = None,
    options: MultiLevelOptions | None = None,
    projection_threshold: float = PROJECTION_THRESHOLD,
    participation_threshold: float = PARTICIPATION_THRESHOLD,
    min_rate_per_s: float | None = None,
    allow_strong_coupling: bool = False,
    allow_recoil_limited: bool = False,
) -> DopplerResult:
    """nbar_D and the cooling rate of every mode (or of ``modes``) of ``crystal`` under ``beams``.

    ``beams`` are all the beams the ions see, repumpers included; ``cooling_beams`` (default: all) restricts the
    projection guard and the force-model cross-check to the beams meant to cool. ``weights`` (mode -> weight) is the
    objective's participation weighting, uniform when None; ``illuminated`` defaults to the ions the beams reach, and
    ``structure`` may be a per-ion mapping for a mixed crystal.
    """
    cooling = list(cooling_beams) if cooling_beams is not None else list(range(len(beams)))
    ions = tuple(illuminated) if illuminated is not None else illuminated_ions(beams, crystal)
    if not ions:
        raise ValueError("no ion is illuminated by the beams")
    models = models_per_ion(structure, beams, crystal, ions, levels=levels, options=options)
    # Gamma > omega_R is a property of the line and the mass alone: asserted before any rate is computed
    _assert_doppler_regime(
        models[ions[0]], crystal, ions[0], cooling, allow_recoil_limited=allow_recoil_limited
    )
    which = list(modes) if modes is not None else list(range(len(crystal.modes)))
    rates = tuple(
        mode_rates(
            models,
            crystal,
            k,
            cooling_beams=cooling,
            projection_threshold=projection_threshold,
            participation_threshold=participation_threshold,
            min_rate_per_s=min_rate_per_s,
            allow_strong_coupling=allow_strong_coupling,
        )
        for k in which
    )
    w = tuple(float(weights.get(k, 0.0)) if weights is not None else 1.0 for k in which)
    if sum(w) <= 0.0:
        raise ValueError("the mode weights must not all vanish")
    force: dict[int, float] = {}
    for r in rates:
        f = _force_model(models[ions[0]], ions[0], r, cooling)
        if f is not None:
            force[r.mode] = f
    return DopplerResult(
        modes=rates,
        weights=w,
        illuminated=ions,
        scattering_rate_per_s={i: m.scattering_rate_per_s() for i, m in models.items()},
        approximations=tuple(models[ions[0]].build.approximations),
        force_model_nbar=force,
    )


def with_detuning_offset(
    beams: Sequence[Beam], group: Sequence[int], offset_rad_s: float
) -> tuple[Beam, ...]:
    """The beam list with the frequencies of ``group`` moved by ``offset`` (a common detuning change for one group)."""
    out = list(beams)
    for b in group:
        out[b] = shifted_beam(out[b], offset_rad_s)
    return tuple(out)


def optimize_detuning(
    evaluate: Callable[[float], DopplerResult],
    bounds_rad_s: tuple[float, float],
    *,
    tolerance_rad_s: float = 1e3,
) -> tuple[float, DopplerResult]:
    """Minimize the participation-weighted mean nbar over a common detuning offset of one beam group.

    ``evaluate(offset)`` builds the DopplerResult with the group's beams shifted by ``offset``; a heating configuration
    (CoolingError) inside the bounds counts as an infinite objective. Returns (offset, result at the optimum).
    """
    cache: dict[float, DopplerResult] = {}

    def objective(x: float) -> float:
        try:
            res = evaluate(float(x))
            cache[float(x)] = res
            return res.weighted_nbar
        except CoolingError:
            return float("inf")

    sol = minimize_scalar(
        objective, bounds=bounds_rad_s, method="bounded", options={"xatol": tolerance_rad_s}
    )
    best = float(sol.x)
    result = cache.get(best) or evaluate(best)
    return best, result
