"""Motional experiments: sideband-ratio thermometry, mode spectroscopy and the heating rate (PLAN.md Section 7.5).

- ``thermometry``: P_rsb/P_bsb = nbar/(nbar + 1) at equal durations on the first sidebands, exact in eta and in the
  duration because the same Omega_{n+1,n} appears on both sides (Turchette 2000 Eqs. 8-11), inverted as nbar = R/(1 - R);
  a ratio that changes with the duration flags a non-thermal state.
- ``mode_spectroscopy``: a coarse scan about the believed mode frequency, then fine scans of the blue sideband and the
  carrier fitted with the plan's lineshape; the mode frequency is their difference, so the frame error and the light shift
  cancel, |eta| follows from the sideband Rabi frequency (C0 inside) and nbar from the sideband ratio.
- ``heating_rate``: nbar by the sideband ratio after a delay with the device's heating channels on, against the delay,
  fitted by weighted linear regression; the interaction frame makes a long idle affordable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Unpack

import numpy as np
from scipy.optimize import brentq

from qutip_trap.experiments.fitting import (
    at_scan_edge,
    fit_lineshape,
    sigmas_or_none,
    thermal_n_max,
    weighted_fit,
)
from qutip_trap.experiments.result import (
    ExperimentResult,
    HeatingRateFit,
    ScanParameters,
    SidebandSpectrum,
    ThermometryResult,
)
from qutip_trap.experiments.single_ion import _WEIGHT_MIN, _Lab, _LabOptions, _Probe, _run, _Setup, _setup
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.machine import Machine
    from qutip_trap.options import Numerics


def _sideband_pi_time_s(rabi_hz: float, eta: float) -> float:
    """pi/(Omega eta e^{-eta^2/2}): the blue-sideband pi time out of |n = 0> (the exact n = 0 -> 1 matrix element)."""
    from qutip_trap.dynamics.operators import rabi_matrix_element

    return math.pi / (TWO_PI * rabi_hz * rabi_matrix_element(1, 0, eta))


def _excitation(
    lab: _Lab,
    ion: int,
    base: _Setup,
    probe: _Probe,
    detuning_hz: float,
    duration_s: float,
    weight_min: float = _WEIGHT_MIN,
) -> float:
    """P1 after one pulse of ``probe`` at ``detuning_hz`` on the space of ``base``."""
    from qutip_trap.control.pulses import Pulse

    setup = _setup(lab, ion, replace(probe, detuning_hz=detuning_hz), base.space)
    pulse = Pulse(setup.drive, 0.0, duration_s, "sideband_probe", ())
    return _run(lab, ion, [pulse], setup, weight_min=weight_min).final_p1(ion)


def _sideband_nbar(
    p_red: float, s_red: float | None, p_blue: float, s_blue: float | None
) -> tuple[float, float, float, float]:
    """(ratio, sigma, nbar, sigma) from the red and blue sideband excitations: nbar = R/(1 - R), R clamped to [0, 1)."""
    ratio = p_red / p_blue if p_blue > 0.0 else math.nan
    s_ratio = 0.0
    if s_red is not None and s_blue is not None and p_blue > 0.0:
        s_ratio = math.sqrt((s_red / p_blue) ** 2 + (p_red * s_blue / p_blue**2) ** 2)
    clamped = min(max(ratio, 0.0), 0.999999) if math.isfinite(ratio) else 0.0
    return ratio, s_ratio, clamped / (1.0 - clamped), s_ratio / (1.0 - clamped) ** 2


def thermometry(
    machine: Machine,
    ion: int,
    mode: int,
    *,
    check_durations_s: Sequence[float] = (),
    include_stark: bool = True,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """nbar of ``mode`` from the red/blue sideband ratio after equal pulses of the n = 0 blue pi time, probed at the
    crystal's mode frequency; ``check_durations_s`` adds durations for the thermality check. Data rows (detuning_hz,
    duration_s, P1); fitted nbar, ratio, P_rsb, P_bsb, duration_s and, with checks, ratio_spread."""
    lab = _Lab.of(machine, kw)
    probe = _Probe(include_stark=include_stark, mode=mode)
    base = _setup(lab, ion, probe)
    f_mode = float(lab.device.crystal.modes[mode].omega_hz)
    duration = _sideband_pi_time_s(base.rabi_hz, base.eta_driven)
    rows: list[tuple[float, float, float]] = []
    sig: list[float | None] = []
    ratios: list[tuple[float, float, float, float]] = []
    for k, t in enumerate([duration, *(float(x) for x in check_durations_s)]):
        p_r, s_r = lab.obs.p1(_excitation(lab, ion, base, probe, -f_mode, t, 1e-6), ion, "thermometry_red", k)
        p_b, s_b = lab.obs.p1(_excitation(lab, ion, base, probe, f_mode, t, 1e-6), ion, "thermometry_blue", k)
        rows += [(-f_mode, t, p_r), (f_mode, t, p_b)]
        sig += [s_r, s_b]
        if p_b > 0.0:
            ratios.append(_sideband_nbar(p_r, s_r, p_b, s_b))
    notes: list[str] = []
    fitted: dict[str, tuple[float, float]] = {}
    converged = bool(ratios) and rows[1][2] > 0.02
    if ratios:
        ratio, s_ratio, nbar, s_nbar = ratios[0]
        fitted = {
            "nbar": (nbar, s_nbar),
            "ratio": (ratio, s_ratio),
            "P_rsb": (rows[0][2], 0.0 if sig[0] is None else float(sig[0])),
            "P_bsb": (rows[1][2], 0.0 if sig[1] is None else float(sig[1])),
            "duration_s": (duration, 0.0),
        }
        if len(ratios) > 1:
            spread = max(abs(r[0] - ratio) for r in ratios[1:])
            tol = 3.0 * max(s_ratio, max(r[1] for r in ratios[1:]), 1e-3)
            fitted["ratio_spread"] = (spread, tol)
            if spread > tol:
                converged = False
                notes.append(
                    f"the sideband ratio changes with the pulse duration by {spread:.3g} (> {tol:.3g}): "
                    "the state is not thermal"
                )
        if not converged and rows[1][2] <= 0.02:
            notes.append("no blue-sideband excitation: the probe is off resonance or the coupling too weak")
    return ThermometryResult(
        data=np.array(rows),
        fitted=fitted,
        model="sideband_ratio_thermometry",
        provenance_id="anchor.m3.thermometry_exactness",
        converged=converged,
        notes=tuple(notes),
        sigma=sigmas_or_none(sig),
        requested=ScanParameters({"duration_s": (duration,)}) if fitted else None,
        subject={"ion": int(ion), "mode": int(mode)},
    )


def _solve_eta(ratio: float) -> float:
    """eta from eta e^{-eta^2/2} = ratio (the n = 0 blue-sideband element over the carrier), on (0, 1)."""
    if ratio <= 0.0:
        return 0.0
    if ratio >= math.exp(-0.5):  # the maximum of eta e^{-eta^2/2}, at eta = 1
        return 1.0
    return float(brentq(lambda e: e * math.exp(-0.5 * e * e) - ratio, 1e-12, 1.0))


def mode_spectroscopy(
    machine: Machine,
    ion: int,
    mode: int,
    *,
    seed_hz: float | None = None,
    span: float = 0.1,
    coarse_points: int | None = None,
    fine_points: int = 15,
    rabi_hz_belief: float | None = None,
    include_stark: bool = True,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """Section 7.5 item 2 for one mode: a coarse blue-sideband scan over ``seed_hz`` x (1 +- ``span``) (default the
    crystal's frequency; ``coarse_points``, default one point per linewidth), fine scans of ``fine_points`` of the blue
    sideband at full and half amplitude (the drive's own carrier light shift extrapolated to zero power) and of the carrier
    at a tenth, each fitted with the plan's lineshape; pulse durations from ``rabi_hz_belief`` (default the physical Rabi
    frequency). Fitted: mode_hz (blue minus carrier), |eta| from the sideband Rabi frequency, nbar from the sideband
    ratio, the line centres and Rabi frequencies; uncalibrated above 1 kHz of uncertainty. Data rows (detuning_hz, P1,
    kind) with kind 0 coarse, 1 fine blue, 2 carrier, 3 red, 4 thermometry."""
    lab = _Lab.of(machine, kw)
    probe = _Probe(include_stark=include_stark, mode=mode)
    base = _setup(lab, ion, probe)
    seed = float(lab.device.crystal.modes[mode].omega_hz if seed_hz is None else seed_hz)
    rabi_belief = float(rabi_hz_belief or base.rabi_hz)
    duration = _sideband_pi_time_s(rabi_belief, base.eta_driven)
    linewidth = 0.5 / duration  # Omega_bsb/2pi of a pi pulse: half depth at delta = Omega
    rows: list[tuple[float, float, int]] = []
    sig: list[float | None] = []
    notes: list[str] = []

    def measure(
        detuning: float, t: float, kind: int, index: int, key: str, scale: float = 1.0
    ) -> tuple[float, float | None]:
        exact = _excitation(lab, ion, base, replace(probe, amplitude_scale=scale), detuning, t)
        p, s = lab.obs.p1(exact, ion, key, index)
        rows.append((detuning, p, kind))
        sig.append(s)
        return p, s

    lo, hi = seed * (1.0 - span), seed * (1.0 + span)
    if coarse_points is not None:
        grid = np.linspace(lo, hi, int(coarse_points))
    else:
        grid = np.linspace(lo, hi, int(min(max(math.ceil((hi - lo) / linewidth), 3), 400)) + 1)
    coarse = [measure(float(mu), duration, 0, k, "mode_spectroscopy[0]")[0] for k, mu in enumerate(grid)]
    k_peak = int(np.argmax(coarse))
    centre = float(grid[k_peak])
    at_edge = k_peak in (0, len(grid) - 1)
    if at_edge:
        notes.append("the coarse scan's maximum sits at its edge: widen span")
    # a resonant sideband pulse light-shifts the qubit through its own off-resonant carrier coupling by Omega^2/(2 omega_m),
    # pulling the resonance toward the carrier; the shift scales as the intensity, so the fine scans at amplitude scales
    # 1 and 1/2 extrapolate the centre to zero power
    centres: list[tuple[float, float, float]] = []
    fine = grid
    fit_b = None
    for k_scale, s_amp in enumerate((1.0, 0.5)):
        lw = linewidth * s_amp
        t_scan = duration / s_amp
        grid = np.linspace(centre - 2.0 * lw, centre + 2.0 * lw, int(fine_points))
        key = f"mode_spectroscopy[1][{k_scale}]"
        pts = [measure(float(mu), t_scan, 1, k, key, s_amp) for k, mu in enumerate(grid)]
        fit_k = fit_lineshape(
            grid,
            np.array([p for p, _s in pts]),
            t_scan,
            sigma=sigmas_or_none([s for _p, s in pts]),
            guess=(centre if k_scale == 0 else centres[0][0], lw),
        )
        if fit_b is None:
            fit_b, fine = fit_k, grid
        centres.append((*fit_k.value(0), s_amp))
        if not fit_k.converged:
            notes.append(f"blue-sideband fit at amplitude scale {s_amp:g} did not converge")
    assert fit_b is not None
    (c1, e1, s1), (c2, e2, s2) = centres
    slope = (c1 - c2) / (s1**2 - s2**2)  # f(s) = f_0 + c s^2 through the two scales
    f_blue = c1 - slope * s1**2
    s_blue = math.hypot(e1 * (1.0 + s1**2 / (s1**2 - s2**2)), e2 * s1**2 / (s1**2 - s2**2))
    notes.append(
        f"blue sideband at scales {s1:g}, {s2:g}: {c1:.1f}, {c2:.1f} Hz; the drive's carrier light shift {slope * s1**2:+.1f} Hz "
        "removed by extrapolation to zero power"
    )
    omega_bsb, s_omega = fit_b.value(1)
    # the carrier at a tenth of the amplitude: a pi pulse ten times longer has a line as narrow as the sideband's
    t_car = 0.5 / (rabi_belief * 0.1)
    lw_car = 0.5 / t_car
    car = np.linspace(-2.0 * lw_car, 2.0 * lw_car, int(fine_points))
    carrier = [measure(float(mu), t_car, 2, k, "mode_spectroscopy[2]", 0.1) for k, mu in enumerate(car)]
    fit_c = fit_lineshape(
        car,
        np.array([p for p, _s in carrier]),
        t_car,
        sigma=sigmas_or_none([s for _p, s in carrier]),
        guess=(0.0, lw_car),
    )
    f_car, s_car = fit_c.value(0)
    f_blue_full = centres[0][0]  # the resonance at the full-power probe, where the thermometry probes sit
    # the red sideband on the mirrored grid (fitted when it carries a peak); its resonance is pulled toward the carrier too
    red_grid = 2.0 * f_car - fine
    red = [measure(float(mu), duration, 3, k, "mode_spectroscopy[3]") for k, mu in enumerate(red_grid)]
    red_p = np.array([p for p, _s in red])
    f_red: tuple[float, float] | None = None
    if float(red_p.max() - red_p.min()) > 0.1:
        fit_r = fit_lineshape(
            red_grid,
            red_p,
            duration,
            sigma=sigmas_or_none([s for _p, s in red]),
            guess=(2.0 * f_car - f_blue, linewidth),
        )
        if fit_r.converged:
            f_red = fit_r.value(0)
    # thermometry at the full-power resonances (the light shift pulls the red one toward the carrier by the same amount)
    p_b, s_b = measure(f_blue_full, duration, 4, 0, "mode_spectroscopy[4]")
    p_r, s_r = measure(2.0 * f_car - f_blue - (f_blue_full - f_blue), duration, 4, 1, "mode_spectroscopy[4]")
    ratio, s_ratio, nbar, s_nbar = _sideband_nbar(p_r, s_r, p_b, s_b)
    mode_hz = f_blue - f_car
    s_mode = math.hypot(s_blue, s_car)
    if f_red is not None:
        notes.append(
            f"blue - carrier gives {mode_hz:.3f} Hz, (blue - red)/2 at full power gives {0.5 * (f_blue_full - f_red[0]):.3f} Hz"
        )
    # eta solves g(eta) = eta e^{-eta^2/2} = Omega_bsb/(Omega sqrt(nbar + 1)) (the n = 0 element, thermally corrected), so
    # d eta = d ratio/g'(eta), with d ratio from Omega_bsb and from the thermometry's nbar added in quadrature
    denom = max(rabi_belief * math.sqrt(1.0 + max(nbar, 0.0)), 1e-300)
    ratio_omega = omega_bsb / denom
    eta_fit = _solve_eta(ratio_omega)
    g_prime = math.exp(-0.5 * eta_fit**2) * (1.0 - eta_fit**2)
    s_ratio_omega = math.hypot(s_omega / denom, ratio_omega * s_nbar / (2.0 * (1.0 + max(nbar, 0.0))))
    s_eta = s_ratio_omega / max(abs(g_prime), 1e-6)
    converged = (
        fit_b.converged
        and fit_c.converged
        and not at_edge
        and not at_scan_edge(f_blue, float(fine[0]), float(fine[-1]), 0.02)
        and s_mode < 1e3
        and p_b > 0.1
    )
    if s_mode >= 1e3:
        notes.append(f"mode-frequency uncertainty {s_mode:.3g} Hz exceeds 1 kHz")
    fitted = {
        "mode_hz": (mode_hz, s_mode),
        "blue_sideband_hz": (f_blue, s_blue),
        "blue_sideband_full_power_hz": (centres[0][0], centres[0][1]),
        "carrier_hz": (f_car, s_car),
        "omega_bsb_hz": (omega_bsb, s_omega),
        "omega_carrier_hz": fit_c.value(1),
        "eta": (eta_fit, s_eta),
        "nbar": (nbar, s_nbar),
        "ratio": (ratio, s_ratio),
        "duration_s": (duration, 0.0),
        "linewidth_hz": (linewidth, 0.0),
        "chi2_per_dof_blue": (fit_b.chi2_per_dof, 0.0),
        "chi2_per_dof_carrier": (fit_c.chi2_per_dof, 0.0),
    }
    if f_red is not None:
        fitted["red_sideband_hz"] = f_red
    data = np.array(rows)
    return SidebandSpectrum(
        data=data,
        fitted=fitted,
        model="sideband_lineshape_two_stage",
        provenance_id="conv.sideband_lineshape",
        converged=converged,
        notes=tuple(notes),
        sigma=sigmas_or_none(sig),
        requested=ScanParameters({"detunings_hz": data[:, 0]}),
        chi2=max(fit_b.chi2_per_dof, fit_c.chi2_per_dof),
        subject={"ion": int(ion), "mode": int(mode), "beam": base.gate_drive.table_key_beam},
        experiment="mode_spectroscopy",
    )


def _density_matrix_options(options: Numerics | None) -> Numerics:
    """``options`` (or the defaults) integrating by ``mesolve`` at any dimension: a thermal density matrix cannot take
    the trajectory route."""
    from qutip_trap.options import Numerics

    base = options if options is not None else Numerics()
    return replace(
        base, lindblad_method="mesolve", mesolve_dimension_max=max(int(base.mesolve_dimension_max), 1_000_000)
    )


def heating_rate(
    machine: Machine,
    mode: int,
    delays_s: Sequence[float],
    *,
    ion: int | None = None,
    nbar0: float | None = None,
    ndot_seed: float | None = None,
    mode_hz: float | None = None,
    rabi_hz_belief: float | None = None,
    include_stark: bool = True,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """n_dot of ``mode`` from a delay scan of the sideband asymmetry with the device's heating channels on during the
    delay (Section 4.1.5, Turchette Eqs. 8-11): nbar(delay) by the sideband ratio, the rate by weighted linear regression.
    ``ion`` probes (default the one with the largest participation), ``nbar0`` is the prepared occupation (default the
    device recipe's), ``ndot_seed`` sizes the truncation (default the noise model's rate), ``mode_hz`` and ``rabi_hz_belief``
    place and time the probes (default the crystal's and the physical values). The probe runs in the interaction frame
    whatever frame the builder options name, their other options kept. Data rows (delay_s, nbar, sigma, P_rsb, P_bsb);
    fitted ndot_per_s and nbar0 (the intercept)."""
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.dynamics.operators import required_margin
    from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
    from qutip_trap.prep.recipe import preparation_occupations

    lab = _Lab.of(machine, kw)
    device = lab.device
    crystal = device.crystal
    probe_ion = int(np.argmax(np.abs(crystal.modes[mode].eigenvector))) if ion is None else int(ion)
    nb0 = float(preparation_occupations(device).get(mode, 0.0) if nbar0 is None else nbar0)
    rate_seed = float(
        device.noise.heating_rates_quanta_per_s(device).get(mode, 0.0) if ndot_seed is None else ndot_seed
    )
    delays = sorted(float(x) for x in delays_s)
    if not delays or delays[0] < 0.0:
        raise ValueError("delays are non-negative")
    # the thermal probe is a density matrix, integrated by mesolve whatever options it was handed, in the interaction frame
    # (H_0 removed, so that the idle costs nothing) with the machine's or the call's other builder options
    builder = replace(lab.builder if lab.builder is not None else BuilderOptions(), frame="interaction")
    lab = replace(
        lab, nbar={**lab.nbar, mode: nb0}, builder=builder, options=_density_matrix_options(lab.options)
    )
    probe = _Probe(include_stark=include_stark, mode=mode)
    base = _setup(lab, probe_ion, probe)
    eta = base.eta_driven

    def space_for(delay: float) -> HilbertSpace:
        # the geometric tail of the hottest state this delay reaches (Section 5.5): a density matrix costs d^4 per step
        n_hi = thermal_n_max(nb0 + rate_seed * delay)
        d = n_hi + 1 + required_margin(eta)
        frozen = tuple(m for m in range(len(crystal.modes)) if m != mode)
        return HilbertSpace(
            base.space.ion_dims, (ModeTruncation(mode, d, (0, n_hi), eta * 1.5),), None, frozen
        )

    f_mode = float(crystal.modes[mode].omega_hz if mode_hz is None else mode_hz)
    duration = _sideband_pi_time_s(float(rabi_hz_belief or base.rabi_hz), eta)
    rows: list[tuple[float, float, float, float, float]] = []
    for k, delay in enumerate(delays):
        pops: list[float] = []
        space = space_for(delay)
        for sign in (-1.0, +1.0):
            setup = _setup(lab, probe_ion, replace(probe, detuning_hz=sign * f_mode), space)
            pulse = Pulse(setup.drive, delay, delay + duration, "heating_probe", ())
            idle = ((0.0, delay),) if delay > 0.0 else ()
            avg = _run(lab, probe_ion, [pulse], setup, idle=idle, fock_branches=False, device_channels=True)
            pops.append(avg.final_p1(probe_ion))
        p_r, s_r = lab.obs.p1(pops[0], probe_ion, "heating_red", k)
        p_b, s_b = lab.obs.p1(pops[1], probe_ion, "heating_blue", k)
        _ratio, _s_ratio, nbar, s_nbar = _sideband_nbar(p_r, s_r, p_b, s_b)
        rows.append((delay, nbar, s_nbar, p_r, p_b))
    data = np.array(rows)
    t, y = data[:, 0], data[:, 1]
    sigma = data[:, 2] if np.all(data[:, 2] > 0.0) else None
    fit = weighted_fit(lambda p, tt: p[0] + p[1] * tt, [nb0, max(rate_seed, 0.0)], t, y, sigma=sigma)
    return HeatingRateFit(
        data=data,
        fitted={"ndot_per_s": fit.value(1), "nbar0": fit.value(0), "duration_s": (duration, 0.0)},
        model="heating_rate_sideband_asymmetry",
        provenance_id="anchor.trap.heating_dynamics",
        converged=fit.converged and len(delays) >= 3,
        sigma=sigma,
        requested=ScanParameters({"delays_s": data[:, 0]}),
        chi2=fit.chi2_per_dof,
        subject={"mode": int(mode)},
    )
