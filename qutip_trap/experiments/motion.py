"""Motional experiments: sideband-ratio thermometry, mode spectroscopy and the heating-rate measurement (PLAN.md Sections
4.1.5, 4.2.7, 7.5 items 2 and 6, 7.9, 9.1, 9.3, 9.17; M8).

- ``thermometry``: P_rsb/P_bsb = nbar/(nbar + 1) at equal pulse durations on the first sidebands, exact in eta and in the
  pulse duration because the same Omega_{n+1,n} appears on both sides (Turchette 2000 Eqs. 8-11; Section 4.2.7), inverted as
  nbar = R/(1 - R); the constancy of the ratio under a second duration is the thermality test.
- ``mode_spectroscopy``: Section 7.5's coarse scan (+-``span`` about the seed frequency, stepped at the pulse's linewidth)
  followed by a fine scan of the blue sideband and of the carrier, each fitted with the plan's lineshape
  P = [Omega^2/(Omega^2 + delta^2)] sin^2((t/2) sqrt(Omega^2 + delta^2)) (never the half-Rabi form); the mode frequency is the
  blue centre minus the carrier centre, so the frame error and the light shift cancel, the sideband Rabi frequency gives
  |eta| = |Omega_bsb|/(sqrt(nbar + 1) Omega e^{-eta^2/2}) (Section 7.9, C0 inside), and the red and blue excitations at the fitted
  centres give nbar.
- ``heating_rate``: Section 4.1.5's procedure, a delay scan of the sideband asymmetry with the device's heating channels
  active during the delay (the engine's own collapse operators), nbar against delay by weighted linear regression; the
  interaction frame (H_0 removed, exact without k_max or rwa) makes a 100 ms idle affordable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import brentq

from qutip_trap.experiments.fitting import at_scan_edge, fit_lineshape, sigmas_or_none, weighted_fit
from qutip_trap.experiments.result import ExperimentResult
from qutip_trap.experiments.single_ion import _observation, _run, _setup, thermal_n_max
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.device.model import Device


def _sideband_pi_time_s(rabi_hz: float, eta: float) -> float:
    """pi/(Omega eta e^{-eta^2/2}): the blue-sideband pi time out of |n = 0> (the exact n = 0 -> 1 matrix element)."""
    from qutip_trap.hilbert.operators import rabi_matrix_element

    return math.pi / (TWO_PI * rabi_hz * rabi_matrix_element(1, 0, eta))


def _excitation(
    device: Device, ion: int, base: Any, detuning_hz: float, duration_s: float, kw: dict[str, Any]
) -> float:
    from qutip_trap.control.pulses import Pulse

    setup = _setup(device, ion, {**kw, "detuning_hz": detuning_hz, "space": base.space})
    pulse = Pulse(setup.drive, 0.0, duration_s, "sideband_probe", ())
    return _run(device, ion, [pulse], setup, kw).final_p1(ion)


def thermometry(device: Device, ion: int, mode: int, **kw: Any) -> ExperimentResult:
    """nbar of ``mode`` from the red/blue sideband ratio after equal pulses (Section 4.2.7; Turchette 2000).

    ``mode_hz`` and ``carrier_hz`` are the BELIEVED mode frequency and carrier offset the probes are placed at (default the
    crystal's frequency and 0); ``duration_s`` the probe duration (default the n = 0 blue pi time at ``rabi_hz_belief``, else
    the physical Rabi frequency); ``check_durations_s`` extra durations for the thermality check. Data rows
    (detuning_hz, duration_s, P1); fitted nbar, ratio, P_rsb, P_bsb.
    """
    kw2 = {**kw, "mode": mode, "detuning_hz": 0.0, "branch_weight_min": kw.get("branch_weight_min", 1e-6)}
    base = _setup(device, ion, kw2)
    obs = _observation(device, kw)
    eta = base.eta_driven
    f_mode = float(kw.get("mode_hz", device.crystal.modes[mode].omega_hz))
    carrier = float(kw.get("carrier_hz", 0.0))
    rabi_belief = float(kw.get("rabi_hz_belief") or base.rabi_hz)
    duration = float(kw.get("duration_s") or _sideband_pi_time_s(rabi_belief, eta))
    durations = [duration] + [float(x) for x in kw.get("check_durations_s", ())]
    rows: list[tuple[float, float, float]] = []
    sig: list[float | None] = []
    ratios: list[tuple[float, float]] = []
    for k, t in enumerate(durations):
        p_r_exact = _excitation(device, ion, base, carrier - f_mode, t, kw2)
        p_b_exact = _excitation(device, ion, base, carrier + f_mode, t, kw2)
        p_r, s_r = obs.p1(p_r_exact, ion, "thermometry_red", k)
        p_b, s_b = obs.p1(p_b_exact, ion, "thermometry_blue", k)
        rows.append((carrier - f_mode, t, p_r))
        rows.append((carrier + f_mode, t, p_b))
        sig.extend([s_r, s_b])
        if p_b > 0.0:
            ratio = p_r / p_b
            var = 0.0
            if s_r is not None and s_b is not None:
                var = (s_r / p_b) ** 2 + (p_r * s_b / p_b**2) ** 2
            ratios.append((ratio, math.sqrt(var)))
    notes: list[str] = []
    fitted: dict[str, tuple[float, float]] = {}
    converged = bool(ratios) and rows[1][2] > 0.02
    if ratios:
        ratio, s_ratio = ratios[0]
        ratio_c = min(max(ratio, 0.0), 0.999999)
        nbar = ratio_c / (1.0 - ratio_c)
        s_nbar = s_ratio / (1.0 - ratio_c) ** 2
        fitted = {
            "nbar": (nbar, s_nbar),
            "ratio": (ratio, s_ratio),
            "P_rsb": (rows[0][2], 0.0 if sig[0] is None else float(sig[0])),
            "P_bsb": (rows[1][2], 0.0 if sig[1] is None else float(sig[1])),
            "duration_s": (duration, 0.0),
        }
        if len(ratios) > 1:
            spread = max(abs(r - ratio) for r, _s in ratios[1:])
            tol = 3.0 * max(s_ratio, max(s for _r, s in ratios[1:]), 1e-3)
            fitted["ratio_spread"] = (spread, tol)
            if spread > tol:
                converged = False
                notes.append(
                    f"the sideband ratio changes with the pulse duration by {spread:.3g} (> {tol:.3g}): the state is not thermal"
                )
        if not converged and rows[1][2] <= 0.02:
            notes.append("no blue-sideband excitation: the probe is off resonance or the coupling too weak")
    return ExperimentResult(
        data=np.array(rows),
        fitted=fitted,
        model="sideband_ratio_thermometry",
        provenance_id="anchor.m3.thermometry_exactness",
        converged=converged,
        notes=tuple(notes),
        sigma=sigmas_or_none(sig),
    )


def _solve_eta(ratio: float) -> float:
    """eta from eta e^{-eta^2/2} = ratio (the n = 0 blue-sideband element over the carrier), on (0, 1)."""
    if ratio <= 0.0:
        return 0.0
    peak = math.exp(-0.5)  # the maximum of eta e^{-eta^2/2} at eta = 1
    if ratio >= peak:
        return 1.0
    return float(brentq(lambda e: e * math.exp(-0.5 * e * e) - ratio, 1e-12, 1.0))


def mode_spectroscopy(device: Device, ion: int, mode: int, **kw: Any) -> ExperimentResult:
    """Section 7.5 item 2 for one mode: coarse scan, fine scans of the blue sideband and the carrier fitted with the plan's
    lineshape, the mode frequency as their difference, |eta| from the sideband Rabi frequency and nbar from the sideband ratio.

    ``seed_hz`` the believed mode frequency (default the crystal's), ``span`` the coarse half-range (default 10 %),
    ``coarse_step_hz`` (default the pulse linewidth Omega_bsb/2pi) or ``coarse_points``, ``fine_points`` (15), ``fine_span_linewidths``
    (2), ``rabi_hz_belief`` the believed carrier Rabi frequency (default the physical one), ``carrier_hz`` the believed carrier
    offset (default 0), ``uncertainty_max_hz`` above which the entry is uncalibrated (1 kHz, the FM solvers' need). Data rows
    (detuning_hz, P1, kind) with kind 0 coarse blue, 1 fine blue, 2 carrier, 3 red, 4 thermometry.
    """
    kw2 = {**kw, "mode": mode, "detuning_hz": 0.0}
    base = _setup(device, ion, kw2)
    obs = _observation(device, kw)
    eta = base.eta_driven
    seed = float(kw.get("seed_hz", device.crystal.modes[mode].omega_hz))
    rabi_belief = float(kw.get("rabi_hz_belief") or base.rabi_hz)
    duration = float(kw.get("duration_s") or _sideband_pi_time_s(rabi_belief, eta))
    linewidth = 0.5 / duration  # Omega_bsb/2pi of a pi pulse: half depth at delta = Omega
    carrier0 = float(kw.get("carrier_hz", 0.0))
    span = float(kw.get("span", 0.1))
    rows: list[tuple[float, float, int]] = []
    sig: list[float | None] = []
    notes: list[str] = []

    def probe(detuning: float, t: float, kind: int, index: int) -> tuple[float, float | None]:
        exact = _excitation(device, ion, base, detuning, t, kw2)
        p, s = obs.p1(exact, ion, f"mode_spectroscopy[{kind}]", index)
        rows.append((detuning, p, kind))
        sig.append(s)
        return p, s

    # coarse scan of the blue sideband
    lo, hi = carrier0 + seed * (1.0 - span), carrier0 + seed * (1.0 + span)
    if kw.get("coarse_points") is not None:
        grid = np.linspace(lo, hi, int(kw["coarse_points"]))
    else:
        step = float(kw.get("coarse_step_hz", linewidth))
        n = int(min(max(math.ceil((hi - lo) / step), 3), 400))
        grid = np.linspace(lo, hi, n + 1)
    coarse = [probe(float(mu), duration, 0, k)[0] for k, mu in enumerate(grid)]
    k_peak = int(np.argmax(coarse))
    centre = float(grid[k_peak])
    at_edge = k_peak in (0, len(grid) - 1)
    if at_edge:
        notes.append("the coarse scan's maximum sits at its edge: widen span")
    # fine scans of the blue sideband at full and at reduced rf amplitude: a resonant sideband pulse light-shifts the qubit
    # through its own off-resonant CARRIER coupling by Omega^2/(2 omega_m) (1.7 kHz at Omega = 100 kHz, omega_m = 3 MHz), which
    # pulls the sideband resonance toward the carrier; a laboratory extrapolates the centre to zero power (the shift scales as the
    # intensity, the amplitude scale squared), and so does this scan (``power_scales``, default (1, 1/2))
    n_fine = int(kw.get("fine_points", 15))
    half = float(kw.get("fine_span_linewidths", 2.0)) * linewidth
    scales = tuple(float(x) for x in kw.get("power_scales", (1.0, 0.5)))
    centres: list[tuple[float, float, float]] = []
    fit_b = None
    for k_scale, s_amp in enumerate(scales):
        lw = linewidth * s_amp
        t_scan = duration / s_amp
        grid = np.linspace(
            centre - 2.0 * lw if k_scale else centre - half,
            centre + 2.0 * lw if k_scale else centre + half,
            n_fine,
        )
        kw_scale = {**kw2, "amplitude_scale": s_amp}
        pts = []
        for k, mu in enumerate(grid):
            exact = _excitation(device, ion, base, float(mu), t_scan, kw_scale)
            p_meas, s_meas = obs.p1(exact, ion, f"mode_spectroscopy[1][{k_scale}]", k)
            rows.append((float(mu), p_meas, 1))
            sig.append(s_meas)
            pts.append((p_meas, s_meas))
        fit_k = fit_lineshape(
            grid,
            np.array([p for p, _s in pts]),
            t_scan,
            sigma=sigmas_or_none([s for _p, s in pts]),
            guess=(centre if k_scale == 0 else centres[0][0], lw),
        )
        if fit_b is None:
            fit_b = fit_k
            fine = grid
        c_k, s_k = fit_k.value(0)
        centres.append((c_k, s_k, s_amp))
        if not fit_k.converged:
            notes.append(f"blue-sideband fit at amplitude scale {s_amp:g} did not converge")
    assert fit_b is not None
    if len(centres) >= 2:
        # f(s) = f_0 + c s^2 through the two scales
        (c1, e1, s1), (c2, e2, s2) = centres[0], centres[1]
        slope = (c1 - c2) / (s1**2 - s2**2)
        f_blue = c1 - slope * s1**2
        s_blue = math.hypot(e1 * (1.0 + s1**2 / (s1**2 - s2**2)), e2 * s1**2 / (s1**2 - s2**2))
        notes.append(
            f"blue sideband at scales {s1:g}, {s2:g}: {c1:.1f}, {c2:.1f} Hz; the drive's carrier light shift {slope * s1**2:+.1f} Hz "
            f"removed by extrapolation to zero power"
        )
    else:
        f_blue, s_blue = fit_b.value(0)
    omega_bsb, s_omega = fit_b.value(1)
    # the carrier at reduced rf amplitude (``carrier_amplitude_scale``, default 1/10): a pi pulse ten times longer has a line
    # ten times narrower, comparable to the sideband's, so the two centres carry similar uncertainties
    car_scale = float(kw.get("carrier_amplitude_scale", 0.1))
    t_car = 0.5 / (rabi_belief * car_scale)
    lw_car = 0.5 / t_car
    car = np.linspace(carrier0 - 2.0 * lw_car, carrier0 + 2.0 * lw_car, n_fine)
    kw_car = {**kw2, "amplitude_scale": car_scale}

    def probe_carrier(detuning: float, index: int) -> tuple[float, float | None]:
        exact = _excitation(device, ion, base, detuning, t_car, kw_car)
        p, s = obs.p1(exact, ion, "mode_spectroscopy[2]", index)
        rows.append((detuning, p, 2))
        sig.append(s)
        return p, s

    carrier = [probe_carrier(float(mu), k) for k, mu in enumerate(car)]
    fit_c = fit_lineshape(
        car,
        np.array([p for p, _s in carrier]),
        t_car,
        sigma=sigmas_or_none([s for _p, s in carrier]),
        guess=(carrier0, lw_car),
    )
    f_car, s_car = fit_c.value(0)
    omega_car, s_omega_car = fit_c.value(1)
    f_blue_full = centres[0][
        0
    ]  # the resonance at the full-power probe (where the thermometry probes must sit)
    # the red sideband on the mirrored grid (fitted when it carries a peak); its resonance is pulled toward the carrier too
    red_grid = 2.0 * f_car - fine
    red = [probe(float(mu), duration, 3, k) for k, mu in enumerate(red_grid)]
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
    shift = f_blue_full - f_blue
    p_b, s_b = probe(f_blue_full, duration, 4, 0)
    p_r, s_r = probe(2.0 * f_car - f_blue - shift, duration, 4, 1)
    ratio = p_r / p_b if p_b > 0.0 else math.nan
    s_ratio = 0.0
    if s_r is not None and s_b is not None and p_b > 0.0:
        s_ratio = math.sqrt((s_r / p_b) ** 2 + (p_r * s_b / p_b**2) ** 2)
    ratio_c = min(max(ratio, 0.0), 0.999999) if math.isfinite(ratio) else 0.0
    nbar = ratio_c / (1.0 - ratio_c)
    s_nbar = s_ratio / (1.0 - ratio_c) ** 2
    mode_hz = f_blue - f_car
    s_mode = math.hypot(s_blue, s_car)
    if f_red is not None:
        # the half-difference of the full-power resonances is the second estimate, low by the drive's carrier light shift
        mode_from_pair = 0.5 * (f_blue_full - f_red[0])
        notes.append(
            f"blue - carrier gives {mode_hz:.3f} Hz, (blue - red)/2 at full power gives {mode_from_pair:.3f} Hz"
        )
    # |eta| from the sideband and carrier Rabi frequencies (n = 0 element; the thermal correction sqrt(nbar + 1) for a hot mode)
    denom = max(rabi_belief * math.sqrt(1.0 + max(nbar, 0.0)), 1e-300)
    ratio_omega = omega_bsb / denom
    eta_fit = _solve_eta(ratio_omega)
    g_prime = math.exp(-0.5 * eta_fit**2) * (1.0 - eta_fit**2)
    # eta solves g(eta) = eta e^{-eta^2/2} = Omega_bsb/(Omega sqrt(nbar + 1)), so d eta = d ratio / g'(eta) with
    # d ratio/d Omega_bsb = 1/(Omega sqrt(nbar + 1)) - the sqrt(nbar + 1) the VALUE divides by was missing from the
    # uncertainty, which left s_eta high by that factor (1 % at nbar = 0.02, 22 % at nbar = 0.5) - and
    # d ratio/d nbar = -ratio/(2 (nbar + 1)), the thermometry's own uncertainty, which was ignored entirely. The two
    # terms are added in quadrature although both come from the same sideband probes, so the pair is mildly correlated.
    s_ratio_omega = math.hypot(s_omega / denom, ratio_omega * s_nbar / (2.0 * (1.0 + max(nbar, 0.0))))
    s_eta = s_ratio_omega / max(abs(g_prime), 1e-6)
    unc_max = float(kw.get("uncertainty_max_hz", 1e3))
    converged = (
        fit_b.converged
        and fit_c.converged
        and not at_edge
        and not at_scan_edge(f_blue, float(fine[0]), float(fine[-1]), 0.02)
        and s_mode < unc_max
        and p_b > 0.1
    )
    if s_mode >= unc_max:
        notes.append(f"mode-frequency uncertainty {s_mode:.3g} Hz exceeds {unc_max:.3g} Hz")
    fitted = {
        "mode_hz": (mode_hz, s_mode),
        "blue_sideband_hz": (f_blue, s_blue),
        "blue_sideband_full_power_hz": (centres[0][0], centres[0][1]),
        "carrier_hz": (f_car, s_car),
        "omega_bsb_hz": (omega_bsb, s_omega),
        "omega_carrier_hz": (omega_car, s_omega_car),
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
    return ExperimentResult(
        data=np.array(rows),
        fitted=fitted,
        model="sideband_lineshape_two_stage",
        provenance_id="conv.sideband_lineshape",
        converged=converged,
        notes=tuple(notes),
        sigma=sigmas_or_none(sig),
    )


def heating_rate(device: Device, mode: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """n_dot of ``mode`` from a delay scan of the sideband asymmetry with the device's heating channels active during the delay
    (Section 4.1.5, Turchette Eqs. 8-11; Section 7.5 item 6): nbar(delay) by the sideband ratio, the rate by weighted linear
    regression. ``ion`` the probe ion (default the one with the largest participation), ``nbar0`` the prepared occupation
    (default the device recipe's), ``ndot_seed`` sizes the truncation (default the noise model's rate). Data rows
    (delay_s, nbar, sigma, P_rsb, P_bsb); fitted ndot_per_s, nbar0 (the intercept), the linear fit's chi^2.
    """
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.dynamics.engine import SolverOptions
    from qutip_trap.hilbert.operators import required_margin
    from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation

    crystal = device.crystal
    ion = int(kw.get("ion", int(np.argmax(np.abs(crystal.modes[mode].eigenvector)))))
    if kw.get("nbar0") is not None:
        nbar0 = float(kw["nbar0"])
    else:
        from qutip_trap.prep.recipe import preparation_occupations

        nbar0 = float(preparation_occupations(device).get(mode, 0.0))
    ndot_seed = kw.get("ndot_seed")
    if ndot_seed is None:
        ndot_seed = float(device.noise.heating_rates_quanta_per_s(device).get(mode, 0.0))
    delays = sorted(float(x) for x in delays_s)
    if not delays or delays[0] < 0.0:
        raise ValueError("delays are non-negative")
    probe_kw = {
        **kw,
        "mode": mode,
        "detuning_hz": 0.0,
        "nbar": {**dict(kw.get("nbar", {})), mode: nbar0},
        "frame": kw.get("frame", "interaction"),
        "device_channels": True,
        "fock_branches": False,
        "options": kw.get("options")
        or SolverOptions(lindblad_method="mesolve", mesolve_dimension_max=1_000_000),
    }
    base = _setup(device, ion, probe_kw)
    eta = base.eta_driven

    def space_for(delay: float) -> HilbertSpace:
        # the geometric tail of the hottest state THIS delay reaches (Section 5.5): a density matrix costs d^4 per step, so the
        # early delays are not made to pay for the last one's cap
        n_hi = thermal_n_max(nbar0 + float(ndot_seed) * delay)
        d = int(kw.get("d", n_hi + 1 + required_margin(eta)))
        return HilbertSpace(
            base.space.ion_dims,
            (ModeTruncation(mode, d, (0, min(n_hi, d - 1)), eta * 1.5),),
            None,
            tuple(m for m in range(len(crystal.modes)) if m != mode),
        )

    obs = _observation(device, kw)
    f_mode = float(kw.get("mode_hz", crystal.modes[mode].omega_hz))
    carrier = float(kw.get("carrier_hz", 0.0))
    rabi_belief = float(kw.get("rabi_hz_belief") or base.rabi_hz)
    duration = float(kw.get("duration_s") or _sideband_pi_time_s(rabi_belief, eta))
    rows: list[tuple[float, float, float, float, float]] = []
    for k, delay in enumerate(delays):
        pops: list[float] = []
        probe_kw["space"] = space_for(delay)
        for sign in (-1.0, +1.0):
            setup = _setup(device, ion, {**probe_kw, "detuning_hz": carrier + sign * f_mode})
            pulse = Pulse(setup.drive, delay, delay + duration, "heating_probe", ())
            idle = ((0.0, delay),) if delay > 0.0 else ()
            avg = _run(device, ion, [pulse], setup, {**probe_kw, "idle": idle})
            pops.append(avg.final_p1(ion))
        p_r, s_r = obs.p1(pops[0], ion, "heating_red", k)
        p_b, s_b = obs.p1(pops[1], ion, "heating_blue", k)
        ratio = p_r / p_b if p_b > 0.0 else 0.0
        ratio_c = min(max(ratio, 0.0), 0.999999)
        nbar = ratio_c / (1.0 - ratio_c)
        var = 0.0
        if s_r is not None and s_b is not None and p_b > 0.0:
            var = (s_r / p_b) ** 2 + (p_r * s_b / p_b**2) ** 2
        s_nbar = math.sqrt(var) / (1.0 - ratio_c) ** 2 if var > 0.0 else 0.0
        rows.append((delay, nbar, s_nbar, p_r, p_b))
    data = np.array(rows)
    t, y = data[:, 0], data[:, 1]
    sigma = data[:, 2] if np.all(data[:, 2] > 0.0) else None
    fit = weighted_fit(lambda p, tt: p[0] + p[1] * tt, [nbar0, max(float(ndot_seed), 0.0)], t, y, sigma=sigma)
    ndot, s_ndot = fit.value(1)
    fitted = {
        "ndot_per_s": (ndot, s_ndot),
        "nbar0": fit.value(0),
        "chi2_per_dof": (fit.chi2_per_dof, 0.0),
        "duration_s": (duration, 0.0),
    }
    return ExperimentResult(
        data=data,
        fitted=fitted,
        model="heating_rate_sideband_asymmetry",
        provenance_id="anchor.trap.heating_dynamics",
        converged=fit.converged and len(delays) >= 3,
        sigma=sigma,
    )


__all__ = ["heating_rate", "mode_spectroscopy", "thermometry"]
