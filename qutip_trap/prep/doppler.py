"""Doppler cooling (PLAN.md Section 4.2.1; milestone M3): per-mode nbar_D from the rate framework, one (Delta, s, k_hat)
per beam over the whole mode set, the force model as the nu << Gamma cross-check only.

Level A of the Doppler stage is the rate framework of Section 4.2.2 with W from the multi-level Bloch steady state
(``prep.rates``: the dipole-force spectrum, exact in the saturation, or the W(Delta -+ nu) closed form for a weak drive),
valid at any nu/Gamma; the semiclassical force model of RMP Eqs. 96-107 linearizes in the velocity and is reported for
the modes with nu/Gamma < 0.1 only (Section 4.2.1, corrected after the 2026-09-04 critique). A Doppler beam has one
detuning, one saturation parameter and one direction for all 3N modes, so ``optimize_detuning`` moves a beam group's
common frequency to minimize the participation-weighted mean of the per-mode nbar (weights b_{i,m}^2 of the gate
modes the schedule uses next, uniform when none is given) and every mode's residual nbar is reported, never optimized
mode by mode. The two-level closed forms (k_B T_D = hbar Gamma/2 with its 1.43 fork, nbar ~ Gamma/(2 nu)) are
oracles in ``prep.closed_forms``, never the producer (Section 13, "Doppler limit" [corrected]).

Section 4.2.1 states the objective over three beam parameters, "Delta, s and k_hat are beam parameters chosen to
minimize the participation-weighted mean of the per-mode steady-state nbar"; only Delta is optimized here and s and
k_hat are inputs (``conv.doppler_optimizes_detuning_only``). The steady state is nearly s independent at the Doppler
limit, so the objective barely moves, but the RATE - which sets the stage duration - does depend on s and is not
optimized, and a genuine three-beam geometry choice is Itano and Wineland's apportionment
``closed_forms.three_beam_optimum`` rather than a search over k_hat.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from scipy.optimize import minimize_scalar

from qutip_trap.dynamics.multilevel import MultiLevelOptions
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import BlochModel, shifted_beam
from qutip_trap.prep.closed_forms import doppler_force_nbar
from qutip_trap.prep.rates import (
    PARTICIPATION_THRESHOLD,
    PROJECTION_THRESHOLD,
    Method,
    ModeRates,
    StageRates,
    illuminated_ions,
    models_per_ion,
    stage_rates,
)
from qutip_trap.prep.validity import assert_doppler_recoil_limit
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.crystal import Crystal

FORCE_MODEL_MAX_NU_OVER_GAMMA = 0.1
"""The force model is reported only for modes with nu/Gamma below this (Section 4.2.1)."""


@dataclass(frozen=True)
class DopplerResult(StageRates):
    """nbar_D per mode with the configured angular factor (Section 4.2.1), the rates, and the force-model cross-check."""

    force_model_nbar: dict[int, float] = field(default_factory=dict)
    """The RMP force-model nbar for the modes with nu/Gamma < 0.1 (cross-check only; empty otherwise)."""


def _force_model(model: BlochModel, ion: int, rates: ModeRates, cooling_beams: Sequence[int]) -> float | None:
    """The RMP force-model nbar for one mode when exactly one cooling beam addresses one resonant line (else None).

    The closed form needs a two-level saturation parameter and one detuning: s = 2 sum |Omega_qp|^2/Gamma^2 over the beam's
    couplings into the resonant upper level and Delta of the coupling nearest resonance; its emission weight alpha/cos^2 is
    read from the same steady state as 2D/(eta^2 W/cos^2), so the cross-check shares every input with the rate framework.
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
    states: Sequence[str] | None = None,
    options: MultiLevelOptions | None = None,
    method: Method = "spectrum",
    projection_threshold: float = PROJECTION_THRESHOLD,
    participation_threshold: float = PARTICIPATION_THRESHOLD,
    min_rate_per_s: float | None = None,
    allow_saturation: bool = False,
    allow_strong_coupling: bool = False,
    allow_fast_cooling: bool = False,
    allow_recoil_limited: bool = False,
) -> DopplerResult:
    """nbar_D and the cooling rate of every mode of ``crystal`` under ``beams`` (all of them cooling beams by default).

    ``beams`` are all the beams the ions see, repumpers included (they shape W); ``cooling_beams`` restricts the projection
    guard and the force-model cross-check to the beams meant to cool. ``weights`` (mode -> weight) is the objective's
    participation weighting, uniform when None; ``modes`` restricts the stage to a subset (default: every mode).
    ``structure`` may be a per-ion mapping for a mixed crystal (Section 4.2.5). Raises UncooledModeError for a mode no
    cooling beam addresses or the illuminated ions barely participate in, and ValidityError when a cooling beam's line
    fails Section 4.2.8 (vii)'s Gamma > omega_R.
    """
    cooling = list(cooling_beams) if cooling_beams is not None else list(range(len(beams)))
    ions = tuple(illuminated) if illuminated is not None else illuminated_ions(beams, crystal)
    if not ions:
        raise ValueError("no ion is illuminated by the beams")
    # Gamma > omega_R is a property of the line and the mass alone, so it is asserted BEFORE any rate is computed:
    # a recoil-limited line has no Doppler limit to report (Section 4.2.8 vii)
    _assert_doppler_regime(
        models_per_ion(structure, beams, crystal, ions[:1], levels=levels, states=states, options=options)[
            ions[0]
        ],
        crystal,
        ions[0],
        cooling,
        allow_recoil_limited=allow_recoil_limited,
    )
    stage, models = stage_rates(
        structure,
        beams,
        crystal,
        cooling_beams=cooling_beams,
        illuminated=ions,
        weights=weights,
        modes=modes,
        levels=levels,
        states=states,
        options=options,
        method=method,
        projection_threshold=projection_threshold,
        participation_threshold=participation_threshold,
        min_rate_per_s=min_rate_per_s,
        allow_saturation=allow_saturation,
        allow_strong_coupling=allow_strong_coupling,
        allow_fast_cooling=allow_fast_cooling,
    )
    first_ion = stage.illuminated[0]
    force: dict[int, float] = {}
    for r in stage.modes:
        f = _force_model(models[first_ion], first_ion, r, cooling)
        if f is not None:
            force[r.mode] = f
    return DopplerResult(
        modes=stage.modes,
        weights=stage.weights,
        illuminated=stage.illuminated,
        scattering_rate_per_s=stage.scattering_rate_per_s,
        method=stage.method,
        approximations=stage.approximations,
        force_model_nbar=force,
    )


def _assert_doppler_regime(
    model: BlochModel,
    crystal: Crystal,
    ion: int,
    cooling_beams: Sequence[int],
    *,
    allow_recoil_limited: bool,
) -> None:
    """Gamma > omega_R on every cooling beam's resonant line (Section 4.2.8 vii): below it the single-photon recoil
    exceeds the linewidth and the Doppler limit is replaced by the recoil limit."""
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


def with_detuning_offset(
    beams: Sequence[Beam], group: Sequence[int], offset_rad_s: float
) -> tuple[Beam, ...]:
    """The beam list with the frequencies of ``group`` moved by ``offset`` (a common detuning change for one beam group)."""
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
    """Minimize the participation-weighted mean nbar over a common detuning offset of one beam group (Section 4.2.1).

    ``evaluate(offset)`` builds the DopplerResult with the group's beams shifted by ``offset``; a heating configuration
    (CoolingError) inside the bounds is treated as an infinite objective. Returns (offset, result at the optimum).
    """
    from qutip_trap.light.bloch import CoolingError

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


__all__ = [
    "FORCE_MODEL_MAX_NU_OVER_GAMMA",
    "DopplerResult",
    "doppler_cooling",
    "optimize_detuning",
    "with_detuning_offset",
]
