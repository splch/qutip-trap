"""The preparation recipe of a device and its evaluation (PLAN.md Sections 3.4, 4.2.1, 4.2.2, 4.2.6, 4.2.7; milestone M6).

Section 3.4's ``prepare`` stage needs to know how the laboratory cools and pumps: which beams cool at which detunings, which
modes are sideband cooled with which pulse schedule, and which beams pump the qubit into its initial state. Those are
procedural device parameters (a cooling laser's detuning is set by an AOM, a sideband schedule is programmed), so they live
on the ``Device`` as a ``PreparationRecipe`` of ``Beam`` objects and pulse counts, and everything else is derived by the M3
stages: the Doppler occupations from the rate framework on the multi-level Bloch model of the cooling beams
(``prep.doppler``), the sideband-cooled occupations from the exact pulsed transfer matrices with the repump recoil kernel
(``prep.sideband``), and the pumped internal state with its preparation error, photon count and recoil heating from the
optical-pumping evolution (``prep.pumping``). ``standard_recipe`` builds the recipe a 171Yb+ laboratory runs from the
device's own detection and Raman beams and records every choice it makes; ``run_preparation`` evaluates a recipe once per
device (cached by the canonical digests) and ``prepare`` in ``run.job`` hands the result to the Hilbert space.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.dynamics.multilevel import LeakPolicy, MultiLevelOptions
from qutip_trap.hashing import canonical_digest
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import (
    BlochModel,
    CoolingError,
    beam_for_transition,
    emission_angular_factor,
    shifted_beam,
)
from qutip_trap.light.recoil import emission_lamb_dicke, minimal_quadrature
from qutip_trap.prep.doppler import DopplerResult, doppler_cooling, optimize_detuning, with_detuning_offset
from qutip_trap.prep.pumping import PumpingResult, optical_pumping
from qutip_trap.prep.sequence import (
    PreparationSequence,
    PreparationStage,
    doppler_stage,
    pulsed_sideband_stage,
    pump_stage,
)
from qutip_trap.prep.sideband import (
    SidebandPulse,
    apply_pulses,
    mean_occupation,
    optimize_per_order,
    pi_time_s,
    repump_kernel,
    thermal_distribution,
)
from qutip_trap.species.model import parse_state_label, parse_transition_label
from qutip_trap.species.polarization import linear_polarization
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.device.model import Device


@dataclass(frozen=True)
class SidebandCoolingSpec:
    """A pulsed Raman sideband-cooling stage (Section 4.2.2, Che 2017 / Rasmusson 2021 schedules) on the Raman pair."""

    beams: tuple[int, int]
    """The Raman pair (indices into Device.beams) that drives the red sidebands; its Delta k sets eta per mode."""
    modes: tuple[int, ...]
    """Crystal modes cooled, one after the other (positions in Crystal.modes)."""
    pulses_per_order: dict[int, int]
    """Sideband order -> number of pulses; higher orders are applied first (Section 4.2.2)."""
    repump_photons: float = 3.0
    """Mean photons scattered by the repump after every pulse (three for the 171Yb+ pump into |0>, Section 4.2.6)."""
    repump_time_s: float = 10e-6
    ion: int | None = None
    """The ion whose eta and Rabi frequency set the pulse times; None = the ion with the largest |eta| on the mode."""
    d_max: int = 400

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("a sideband-cooling stage names at least one mode")
        if any(k < 1 or n < 0 for k, n in self.pulses_per_order.items()):
            raise ValueError("sideband orders are positive and pulse counts non-negative")
        if self.repump_photons < 0.0 or self.repump_time_s < 0.0:
            raise ValueError("repump photons and time are non-negative")


@dataclass(frozen=True)
class PreparationRecipe:
    """How the device cools and pumps before every shot (Section 4.2.6 order: Doppler -> sideband -> pump)."""

    doppler_beams: tuple[Beam, ...]
    """The cooling beams at their cooling detunings (repumpers included); every mode must project on one of them."""
    doppler_duration_s: float
    pump_beams: tuple[Beam, ...]
    pump_duration_s: float
    sideband: SidebandCoolingSpec | None = None
    pump_target: tuple[str, ...] | None = None
    """State labels the pump prepares; None = the lower qubit level of the species."""
    levels: tuple[str, ...] | None = None
    """Fine-structure levels of the Bloch models; None = the cycling transition's two levels."""
    leak: LeakPolicy = "renormalize"
    pump_samples: int = 2001
    notes: tuple[str, ...] = field(default_factory=tuple)
    """What the recipe builder chose and why (standard_recipe records its assumptions here)."""

    def __post_init__(self) -> None:
        if not self.doppler_beams:
            raise ValueError("Doppler cooling comes first (Section 4.2.6): the recipe needs cooling beams")
        if not self.pump_beams:
            raise ValueError(
                "the sequence ends with an optical pump (Section 4.2.6): the recipe needs pump beams"
            )
        if self.doppler_duration_s <= 0.0 or self.pump_duration_s <= 0.0:
            raise ValueError("stage durations are positive")


def _structure(device: Device) -> AtomicStructure:
    species = {sp.name for sp in device.crystal.species}
    if len(species) != 1:
        raise NotImplementedError("the M6 preparation recipe covers single-species crystals")
    return AtomicStructure(device.crystal.species[0], device.field.B_gauss, device.field.direction)


def _cycling_levels(device: Device) -> tuple[str, str]:
    lower, upper = parse_transition_label(device.crystal.species[0].cycling)
    return lower, upper


def global_raman_pair(device: Device) -> tuple[int, int] | None:
    """The far-detuned equal-wavelength beam pair of the largest waist (the global pair a chain's sideband cooling uses), or None."""
    from qutip_trap.light.roles import gate_beams

    idx = gate_beams(device)
    best: tuple[int, int] | None = None
    best_waist = -1.0
    for a in idx:
        for b in idx:
            if b <= a:
                continue
            ba, bb = device.beams[a], device.beams[b]
            if abs(ba.wavelength_m - bb.wavelength_m) > 1e-3 * ba.wavelength_m:
                continue
            waist = min(ba.waist_m, bb.waist_m)
            if waist > best_waist:
                best, best_waist = (a, b), waist
    return best


def standard_recipe(
    device: Device,
    *,
    raman_pair: tuple[int, int] | None = None,
    doppler_detuning_rad_s: float | None = None,
    optimize_doppler_detuning: bool = True,
    doppler_duration_s: float = 1e-3,
    sideband_pulses: Mapping[int, int] | None = None,
    sideband_modes: Sequence[int] | None = None,
    pump_saturation: float = 0.5,
    pump_duration_s: float = 20e-6,
    eta_min: float = 1e-3,
) -> PreparationRecipe:
    """The recipe a hyperfine-clock-qubit laboratory runs, built from the device's own beams, every choice recorded.

    Doppler cooling: the detection beams (the resonant light of ``light.roles.detection_beams``) shifted to the red by
    Gamma/2, then, when ``optimize_doppler_detuning``, the common offset minimizing the participation-weighted mean
    occupation of the gate modes (Section 4.2.1). Sideband cooling: pulsed Raman cooling on the far-detuned Raman pair of
    every mode it couples to (|eta| > ``eta_min``), 10 second-order then 30 first-order pulses by default. Pump: the first
    detection beam retuned to the F = I + 1/2 -> F' = I + 1/2 line of the cycling transition (the 2.1 GHz sideband of the
    171Yb+ scheme, Section 4.2.6) at ``pump_saturation`` I_sat, which empties the upper hyperfine manifold into the F = 0
    clock state. Species whose lower qubit level is not an F = 0 clock state need an explicit recipe.
    """
    from qutip_trap.light.raman import derive_raman_drive
    from qutip_trap.light.roles import detection_beams

    species = device.crystal.species[0]
    st = _structure(device)
    lower, upper = _cycling_levels(device)
    line = species.transition(species.cycling)
    gamma = line.gamma_rad_s
    notes: list[str] = []
    det = [device.beams[k] for k in detection_beams(device, 0)]
    delta = -0.5 * gamma if doppler_detuning_rad_s is None else float(doppler_detuning_rad_s)
    cooling = tuple(shifted_beam(b, delta) for b in det)
    notes.append(
        f"Doppler beams: the {len(det)} detection beam(s) shifted by {delta / gamma:+.3f} Gamma "
        "(the cooling and detection light share one path, retuned by an AOM)"
    )
    # the gate modes: the ones the Raman pair couples to, which the sideband stage cools and the Doppler objective weights
    pair: tuple[int, int] | None
    if raman_pair is not None:
        pair = (int(raman_pair[0]), int(raman_pair[1]))
    else:
        pair = global_raman_pair(device)
    weights: dict[int, float] = {}
    sb_modes: list[int] = []
    if pair is not None:
        etas = [
            derive_raman_drive(device, i, pair, scattering=False).etas for i in range(device.crystal.n_ions)
        ]
        for m in range(len(device.crystal.modes)):
            e = max(abs(et[m]) for et in etas)
            if e > eta_min:
                weights[m] = e**2
                sb_modes.append(m)
    if optimize_doppler_detuning:
        opts = MultiLevelOptions(leak="renormalize")

        def evaluate(offset: float) -> DopplerResult:
            return doppler_cooling(
                st,
                with_detuning_offset(cooling, range(len(cooling)), offset),
                device.crystal,
                weights=weights or None,
                levels=(lower, upper),
                options=opts,
            )

        offset, _res = optimize_detuning(
            evaluate, (-0.9 * gamma - delta, 0.2 * gamma - delta), tolerance_rad_s=2e-3 * gamma
        )
        cooling = tuple(with_detuning_offset(cooling, range(len(cooling)), offset))
        delta += offset
        notes.append(
            f"Doppler detuning optimized over the gate modes' participation weights: {delta / gamma:+.4f} Gamma (Section 4.2.1)"
        )
    sideband: SidebandCoolingSpec | None = None
    if pair is not None and sb_modes:
        counts = dict(sideband_pulses) if sideband_pulses is not None else {2: 10, 1: 30}
        chosen = tuple(int(m) for m in (sideband_modes if sideband_modes is not None else sb_modes))
        sideband = SidebandCoolingSpec(beams=pair, modes=chosen, pulses_per_order=counts)
        notes.append(
            f"pulsed Raman sideband cooling of modes {chosen} on the Raman pair {pair}, pulses per order {counts}, "
            "durations optimized order by order (Section 4.2.2)"
        )
    else:
        notes.append(
            "no Raman pair couples to a mode: no sideband-cooling stage (Doppler occupations remain)"
        )
    # the pump: F = I + 1/2 -> F' = I + 1/2 of the cycling line, from the detection beam's path
    low_level, low_rest = parse_state_label(species.qubit[0])
    if low_level != lower or "F=0" not in low_rest.replace(" ", ""):
        raise NotImplementedError(
            f"no standard pump for a {species.name} qubit whose lower level is {species.qubit[0]!r}; give a PreparationRecipe"
        )
    f_up = int(round(species.nuclear_spin + 0.5))
    beam0 = det[0]
    power = pump_saturation * line.i_sat_w_m2 * math.pi * beam0.waist_m**2 / 2.0
    pump = beam_for_transition(
        st,
        f"{lower} F={f_up} mF=0",
        f"{upper} F={f_up} mF=0",
        0.0,
        beam0.k_hat,
        beam0.polarization,
        power_w=power,
        waist_m=beam0.waist_m,
        pointing_m=beam0.pointing_m,
    )
    notes.append(
        f"pump: the detection beam's path retuned to {lower} F={f_up} -> {upper} F={f_up} at {pump_saturation} I_sat "
        f"(the {species.name} scheme of Section 4.2.6), target {species.qubit[0]!r}"
    )
    return PreparationRecipe(
        doppler_beams=cooling,
        doppler_duration_s=doppler_duration_s,
        pump_beams=(pump,),
        pump_duration_s=pump_duration_s,
        sideband=sideband,
        pump_target=(species.qubit[0],),
        levels=(lower, upper),
        notes=tuple(notes),
    )


def magic_angle_polarization(k_hat: Sequence[float], b_hat: Sequence[float]) -> tuple[float, float, float]:
    """A linear polarization transverse to k at Berkeland's arccos(1/sqrt 3) to B (Section 8.1), for any k not parallel to B."""
    k = np.asarray(k_hat, dtype=float)
    b = np.asarray(b_hat, dtype=float)
    b_perp = b - np.dot(b, k) * k
    norm = float(np.linalg.norm(b_perp))
    target = 1.0 / math.sqrt(3.0)
    if norm < target:
        raise ValueError("k is too close to B for a transverse polarization at the magic angle")
    angle = math.acos(target / norm)
    pol = linear_polarization(tuple(k), angle, tuple(b))
    return (float(pol[0]), float(pol[1]), float(pol[2]))


@dataclass(frozen=True)
class PreparationRun:
    """What a recipe produced on a device (Section 4.2.7 hand-off), cached per (device, recipe)."""

    sequence: PreparationSequence
    doppler: DopplerResult
    sideband_nbar: dict[int, float]
    sideband_pulses: dict[int, tuple[SidebandPulse, ...]]
    pumps: dict[int, PumpingResult]
    pump_heating: dict[int, float]
    """Recoil heating of every mode by the pumps of all ions, in quanta (added to the cooled occupations)."""
    nbar: dict[int, float]
    """The final mean occupation per mode: the last cooling stage that addressed it plus the pump recoil."""
    duration_s: float
    notes: tuple[str, ...]

    def preparation_error(self, ion: int) -> float:
        return float(self.pumps[ion].preparation_error)

    @property
    def provenance(self) -> tuple[str, ...]:
        return self.sequence.provenance()


_CACHE: dict[tuple[str, str], PreparationRun] = {}


def run_preparation(device: Device, recipe: PreparationRecipe, *, cache: bool = True) -> PreparationRun:
    """Evaluate the recipe with the M3 stages: Doppler occupations, sideband-cooled occupations, the pumped states."""
    key = (device.hash(), canonical_digest(recipe))
    if cache and key in _CACHE:
        return _CACHE[key]
    from qutip_trap.light.raman import derive_raman_drive

    species = device.crystal.species[0]
    st = _structure(device)
    crystal = device.crystal
    levels = recipe.levels or _cycling_levels(device)
    opts = MultiLevelOptions(leak=recipe.leak)
    notes = list(recipe.notes)
    # 1. Doppler cooling of every mode
    doppler = doppler_cooling(st, recipe.doppler_beams, crystal, levels=levels, options=opts)
    stages: list[PreparationStage] = [doppler_stage(doppler)]
    nbar: dict[int, float] = dict(doppler.nbar)
    duration = recipe.doppler_duration_s
    # 2. pulsed sideband cooling of the gate modes
    sb_nbar: dict[int, float] = {}
    sb_pulses: dict[int, tuple[SidebandPulse, ...]] = {}
    if recipe.sideband is not None:
        spec = recipe.sideband
        lam = species.transition(species.cycling).wavelength_vac_m
        k_em = TWO_PI / lam
        for m in spec.modes:
            drives = {
                i: derive_raman_drive(device, i, spec.beams, scattering=False) for i in range(crystal.n_ions)
            }
            ion = spec.ion if spec.ion is not None else max(drives, key=lambda i: abs(drives[i].etas[m]))
            eta = abs(drives[ion].etas[m])
            omega0 = TWO_PI * drives[ion].carrier_rabi_hz
            if eta == 0.0 or omega0 == 0.0:
                raise CoolingError(
                    f"mode {m}: the Raman pair does not couple ion {ion} to it; it cannot be sideband cooled"
                )
            n0 = nbar[m]
            d = int(min(spec.d_max, max(40, math.ceil(30.0 * n0 + 20.0))))
            p0 = thermal_distribution(n0, d)
            # the repump's emitted photons carry one alpha PER POLARIZATION CHANNEL (Section 4.2.8 ii); the scalar the
            # one-dimensional Fock kernel needs is their photon-rate-weighted mean in the repump beams' own steady
            # state, read off the Bloch model this recipe already builds for the pump -- never a hard-coded 1/3
            alpha = emission_angular_factor(
                BlochModel(
                    st,
                    recipe.pump_beams,
                    levels=levels,
                    position_m=tuple(float(x) for x in crystal.positions_m[ion]),
                    options=opts,
                ),
                crystal.modes[m].e_hat,
            )
            eta_em = emission_lamb_dicke(crystal, ion, k_em, m)
            kernel = repump_kernel(d, eta_em, minimal_quadrature(alpha), spec.repump_photons)
            t_pi = pi_time_s(omega0, eta, 1, 1)
            pulses, n_final = optimize_per_order(
                p0, spec.pulses_per_order, eta, omega0, (0.05 * t_pi, 4.0 * t_pi), repump=kernel
            )
            if not pulses:
                continue
            check = mean_occupation(apply_pulses(p0, pulses, eta, omega0, repump=kernel))
            sb_nbar[m] = float(check)
            sb_pulses[m] = tuple(pulses)
            nbar[m] = float(check)
            duration += sum(p.duration_s for p in pulses) + spec.repump_time_s * len(pulses)
            notes.append(
                f"mode {m}: nbar {n0:.3f} -> {n_final:.4f} after {len(pulses)} pulses on ion {ion} (eta {eta:.4f}), "
                f"repump kernel alpha {alpha:.3f}"
            )
        if sb_nbar:
            stages.append(pulsed_sideband_stage(sb_nbar, tuple(range(crystal.n_ions))))
    # 3. the optical pump of every ion, at its own position
    target = tuple(recipe.pump_target) if recipe.pump_target is not None else (species.qubit[0],)
    pumps: dict[int, PumpingResult] = {}
    heating: dict[int, float] = {m: 0.0 for m in range(len(crystal.modes))}
    for i in range(crystal.n_ions):
        model = BlochModel(
            st,
            recipe.pump_beams,
            levels=levels,
            position_m=tuple(float(x) for x in crystal.positions_m[i]),
            options=opts,
        )
        res = optical_pumping(
            model,
            list(target),
            duration_s=recipe.pump_duration_s,
            samples=recipe.pump_samples,
            crystal=crystal,
            ion=i,
        )
        pumps[i] = res
        for m, dn in res.motional_heating_quanta.items():
            heating[m] += float(dn)
        stages.append(pump_stage(res, (i,), provenance=f"prep.pumping[{i}]"))
    duration += recipe.pump_duration_s
    final = {m: nbar[m] + heating.get(m, 0.0) for m in nbar}
    run = PreparationRun(
        sequence=PreparationSequence(tuple(stages)),
        doppler=doppler,
        sideband_nbar=sb_nbar,
        sideband_pulses=sb_pulses,
        pumps=pumps,
        pump_heating=heating,
        nbar=final,
        duration_s=duration,
        notes=tuple(notes),
    )
    if cache:
        _CACHE[key] = run
    return run


_STANDARD_CACHE: dict[tuple[str, tuple[int, int] | None], PreparationRecipe] = {}


def recipe_of(device: Device, *, raman_pair: tuple[int, int] | None = None) -> PreparationRecipe:
    """The device's recipe, or the standard one (built once per device and cooling pair, recorded as inferred)."""
    if device.preparation is not None:
        return device.preparation
    key = (device.hash(), raman_pair)
    if key not in _STANDARD_CACHE:
        _STANDARD_CACHE[key] = standard_recipe(device, raman_pair=raman_pair)
    return _STANDARD_CACHE[key]


def preparation_occupations(
    device: Device, recipe: PreparationRecipe | None = None, *, raman_pair: tuple[int, int] | None = None
) -> dict[int, float]:
    """The final mean occupation per mode the recipe leaves (what the mode selection and the calibration weights read)."""
    return dict(run_preparation(device, recipe or recipe_of(device, raman_pair=raman_pair)).nbar)


def clear_preparation_cache() -> None:
    _CACHE.clear()
    _STANDARD_CACHE.clear()


__all__ = [
    "PreparationRecipe",
    "PreparationRun",
    "SidebandCoolingSpec",
    "clear_preparation_cache",
    "global_raman_pair",
    "magic_angle_polarization",
    "preparation_occupations",
    "recipe_of",
    "run_preparation",
    "standard_recipe",
]
