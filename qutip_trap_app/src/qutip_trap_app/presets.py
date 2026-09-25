"""The published-experiment presets (PLAN.md Section 14.5): each runner computes, through the core, the numbers a Section 9
entry pins and the curve that puts them in context, and returns a picklable ``PresetResult`` that the Learn view places
beside the published values (``viewmodel.presets.compare``). Every formula is the core's."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Literal

import numpy as np

from qutip_trap_app import core
from qutip_trap_app.record import Progress
from qutip_trap_app.viewmodel.presets import PRESETS, ChartRecord, PresetResult, PresetSpec, SeriesRecord

TWO_PI = core.TWO_PI


def _progress(progress: Progress | None, stage: str, fraction: float | None, message: str) -> None:
    if progress is not None:
        progress(stage, fraction, message)


# ---- Section 9.2: Harty 2014 -------------------------------------------------------------------------------------------------------


def harty_2014(progress: Progress | None = None, *, n_sets: int = 16, seed: int = 3) -> PresetResult:
    """Microwave randomized benchmarking with the paper's own error model (Section 4.3.3; ``qutip_trap.validation.harty_rb``)."""
    t0 = time.perf_counter()
    p = core.HartyParameters()
    _progress(
        progress, "benchmarking", 0.1, f"{n_sets} sets of 32 sequences of {p.n_gates} gates, both errors on"
    )
    both = core.simulate_epg_sets(n_sets, params=p, seed=seed)
    _progress(progress, "benchmarking", 0.6, "the pulse-area error alone")
    rabi_only = core.simulate_epg_sets(4, params=p, detuning_hz=0.0, seed=seed - 1)
    _progress(
        progress, "benchmarking", 0.8, "the detuning alone, with the -1.0 Hz ac Zeeman shift during pulses"
    )
    det_only = core.simulate_epg_sets(
        4, params=core.HartyParameters(ac_zeeman_hz=-1.0), rabi_error=0.0, seed=seed - 2
    )
    charts = (
        ChartRecord(
            title="error per gate of each set of 32 sequences",
            x_title="set",
            y_title="error per gate (1e-6)",
            series=(SeriesRecord("both errors", np.arange(n_sets, dtype=float), both * 1e6),),
            markers=(),
            bars=True,
        ),
    )
    return PresetResult(
        preset_id="harty_2014",
        simulated={
            "epg": float(both.mean()),
            "epg_rabi_only": float(rabi_only.mean()),
            "epg_detuning_only": float(det_only.mean()),
        },
        simulated_uncertainty={
            "epg": float(both.std(ddof=1)),
            "epg_rabi_only": float(rabi_only.std(ddof=1)),
            "epg_detuning_only": float(det_only.std(ddof=1)),
        },
        charts=charts,
        parameters={
            "t_pi2_s": p.t_pi2_s,
            "dead_time_s": p.dead_time_s,
            "detuning_hz": p.detuning_hz,
            "rabi_error": p.rabi_error,
            "n_gates": float(p.n_gates),
            "n_sets": float(n_sets),
        },
        notes=(
            "the spread over sets is the paper's Fig. 7 histogram width, so the uncertainty shown is that spread, not the error of the mean",
        ),
        wall_time_s=time.perf_counter() - t0,
    )


# ---- Section 9.1: James 1998 -------------------------------------------------------------------------------------------------------


def james_1998(progress: Progress | None = None, *, n_max: int = 10) -> PresetResult:
    """Equilibrium positions and axial mode eigenvalues of N equal ions (Sections 4.1.2, 4.1.3; ``trap.crystal``)."""
    t0 = time.perf_counter()
    simulated: dict[str, float] = {}
    series: list[SeriesRecord] = []
    positions: list[SeriesRecord] = []
    for n in range(2, n_max + 1):
        _progress(progress, "solving", (n - 1) / n_max, f"{n} ions: equilibrium and axial Hessian")
        u = core.equilibrium_dimensionless(n)
        mu, _vectors = core.axial_modes_dimensionless(u)
        if n == 2:
            simulated["u_n2"] = float(u[-1])
        if n == 3:
            simulated["u_n3"] = float(u[-1])
            simulated["mu3_n3"] = float(mu[2])
        if n == 5:
            simulated["mu2_n5"] = float(mu[1])
        if n == n_max:
            simulated["mu3_n10"] = float(mu[2])
        if n in (2, 3, 5, 10):
            series.append(SeriesRecord(f"{n} ions", np.arange(1, n + 1, dtype=float), np.asarray(mu)))
            positions.append(SeriesRecord(f"{n} ions", np.asarray(u), np.full(n, float(n))))
    charts = (
        ChartRecord(
            title="axial mode eigenvalues (nu_p/nu_z)^2 against the mode number",
            x_title="mode p",
            y_title="mu_p",
            series=tuple(series),
        ),
        ChartRecord(
            title="equilibrium positions in units of the Coulomb length l",
            x_title="u_i",
            y_title="N",
            series=tuple(positions),
        ),
    )
    return PresetResult(
        preset_id="james_1998",
        simulated=simulated,
        simulated_uncertainty={},
        charts=charts,
        parameters={"n_max": float(n_max)},
        notes=(
            "mu_1 = 1 and mu_2 = 3 for every N exactly; the positions are in units of l = (e^2/(4 pi eps0 m omega_z^2))^(1/3)",
        ),
        wall_time_s=time.perf_counter() - t0,
    )


# ---- Section 9.3: Monroe 1995 ------------------------------------------------------------------------------------------------------


def monroe_1995(progress: Progress | None = None) -> PresetResult:
    """9Be+ Doppler cooling at Monroe's parameters: the force model and the rate framework (Sections 4.2.1, 4.2.2)."""
    t0 = time.perf_counter()
    gamma = TWO_PI * 19.4e6
    delta = -TWO_PI * 30e6
    modes = {"112": TWO_PI * 11.2e6, "182": TWO_PI * 18.2e6, "298": TWO_PI * 29.8e6}
    alpha = 1.0 / 3.0
    _progress(progress, "cooling", 0.2, "the semiclassical force model and Stenholm's coefficients per mode")
    simulated: dict[str, float] = {}
    for key, nu in modes.items():
        simulated[f"rate_{key}"] = float(
            core.stenholm_coefficients(0.01 * gamma, gamma, nu, delta, alpha).nbar
        )
    # the force model is quoted for the 11.2 MHz mode only, where the paper quotes it; at nu/Gamma > 1 it has no meaning
    simulated["force_112"] = float(core.doppler_force_nbar(gamma, delta, 0.0, modes["112"], alpha))
    detunings = np.linspace(-60e6, -4e6, 113)
    force = np.array(
        [core.doppler_force_nbar(gamma, TWO_PI * d, 0.0, modes["112"], alpha) for d in detunings]
    )
    rate = np.array(
        [
            core.stenholm_coefficients(0.01 * gamma, gamma, modes["112"], TWO_PI * d, alpha).nbar
            for d in detunings
        ]
    )
    charts = (
        ChartRecord(
            title="steady-state occupation of the 11.2 MHz mode against the detuning",
            x_title="Delta/2pi (MHz)",
            y_title="nbar",
            series=(
                SeriesRecord("force model, isotropic emission", detunings / 1e6, force),
                SeriesRecord("rate framework, weak beam along the mode", detunings / 1e6, rate),
            ),
            log_y=True,
            markers=((-30.0, "Monroe: Delta = -30 MHz"),),
        ),
    )
    return PresetResult(
        preset_id="monroe_1995",
        simulated=simulated,
        simulated_uncertainty={},
        charts=charts,
        parameters={"gamma_hz": 19.4e6, "delta_hz": -30e6, "alpha": alpha, "s": 0.0},
        notes=(
            "nu/Gamma = 0.58 for the 11.2 MHz mode: the force model is quoted because the paper quotes it, the rate framework is the one valid there",
        ),
        wall_time_s=time.perf_counter() - t0,
    )


# ---- Section 9.3: Roos 2000 --------------------------------------------------------------------------------------------------------


def roos_2000(progress: Progress | None = None) -> PresetResult:
    """The Lamb-Dicke parameters of 40Ca+ on its 729 nm and 393 nm lines at 2 pi x 1 MHz (Section 4.1.7)."""
    t0 = time.perf_counter()
    ca = core.species("40Ca+")
    mass_kg = ca.mass_u * core.ATOMIC_MASS_KG
    omega = TWO_PI * 1e6
    _progress(progress, "computing", 0.5, "k x0 for the quadrupole and the dipole line")
    lines = {"eta_729": ca.transition("S1/2-D5/2"), "eta_393": ca.transition("S1/2-P3/2")}
    simulated = {
        key: float(core.lamb_dicke_parameter(TWO_PI / tr.wavelength_vac_m, mass_kg, omega))
        for key, tr in lines.items()
    }
    freqs = np.geomspace(0.2e6, 5e6, 60)
    charts = (
        ChartRecord(
            title="Lamb-Dicke parameter against the trap frequency",
            x_title="omega/2pi (Hz)",
            y_title="eta",
            series=tuple(
                SeriesRecord(
                    f"{tr.wavelength_vac_m * 1e9:.0f} nm",
                    freqs,
                    np.array(
                        [
                            core.lamb_dicke_parameter(TWO_PI / tr.wavelength_vac_m, mass_kg, TWO_PI * f)
                            for f in freqs
                        ]
                    ),
                )
                for tr in lines.values()
            ),
            log_x=True,
            log_y=True,
            markers=((1e6, "Roos: 1 MHz"),),
        ),
    )
    return PresetResult(
        preset_id="roos_2000",
        simulated=simulated,
        simulated_uncertainty={},
        charts=charts,
        parameters={
            "mass_u": float(ca.mass_u),
            "x0_m": float(core.x0_m(mass_kg, omega)),
            "omega_hz": 1e6,
        },
        notes=(f"x0 = {core.x0_m(mass_kg, omega) * 1e9:.3f} nm for {ca.mass_u:.3f} u at 1 MHz",),
        wall_time_s=time.perf_counter() - t0,
    )


# ---- Section 9.4: Kirchmair 2009 ---------------------------------------------------------------------------------------------------


def kirchmair_2009(progress: Progress | None = None) -> PresetResult:
    """Kirchmair's single-loop MS gate on a thermal mode: the populations of Eq. 14 and the Debye-Waller parity contrast
    (Sections 4.4.1, 4.4.7)."""
    t0 = time.perf_counter()
    eta, nu_hz, t_gate = 0.044, 1.232e6, 50e-6
    eps = TWO_PI / t_gate  # one loop closes at eps t_g = 2 pi
    omega = core.closure_rabi_rad_s(eta, eps, 1)
    _progress(progress, "computing", 0.3, "the exact single-loop populations for nbar = 0 and nbar = 20")
    times = np.linspace(0.0, t_gate, 201)
    charts: list[ChartRecord] = []
    for nbar in (0.0, 20.0):
        pops = np.array(
            [
                core.kirchmair_populations(
                    abs(core.ms_alpha(eta, omega, eps, float(t))),
                    core.ms_gamma(eta, omega, eps, float(t)),
                    nbar,
                )
                for t in times
            ]
        )
        charts.append(
            ChartRecord(
                title=f"populations with 0, 1 and 2 ions bright, nbar = {nbar:g}",
                x_title="t (us)",
                y_title="population",
                series=(
                    SeriesRecord("both bright (P00)", times * 1e6, pops[:, 2]),
                    SeriesRecord("one bright", times * 1e6, pops[:, 1]),
                    SeriesRecord("none bright (P11)", times * 1e6, pops[:, 0]),
                ),
            )
        )
    _progress(progress, "computing", 0.8, "the Debye-Waller parity contrast of a thermal mode at nbar = 20")
    angle = 4.0 * core.CHI_MAXIMAL_RAD * eta**2
    n = np.arange(4000)
    nbar20 = 20.0
    weights = np.exp(n * math.log(nbar20 / (1.0 + nbar20))) / (1.0 + nbar20)
    contrast = float(abs(np.sum(weights * np.exp(-1j * angle * n))))
    first_principles = core.ballance_thermal_error(eta, 0.0) + core.thermal_debye_waller_infidelity(
        eta, 0.0, "minus_half"
    )
    return PresetResult(
        preset_id="kirchmair_2009",
        simulated={
            "contrast_nbar20": contrast,
            "fidelity_first_principles": 1.0 - float(first_principles),
            "omega_hz": float(omega / TWO_PI),
        },
        simulated_uncertainty={},
        charts=tuple(charts),
        parameters={
            "eta": eta,
            "nu_hz": nu_hz,
            "t_gate_s": t_gate,
            "epsilon_hz": eps / TWO_PI,
            "omega_hz": omega / TWO_PI,
        },
        notes=(
            f"closing one loop in {t_gate * 1e6:.0f} us needs Omega/2pi = {omega / TWO_PI / 1e3:.1f} kHz at eta = {eta}",
            "the first-principles infidelity at nbar = 0 is the residual displacement plus the Debye-Waller term of Section 4.4.7",
        ),
        wall_time_s=time.perf_counter() - t0,
    )


# ---- Section 9.5: Myerson 2008 and Crain 2019 ---------------------------------------------------------------------------------------


def _threshold_scan(
    model: core.RecordModel,
    windows_s: np.ndarray,
    progress: Progress | None,
    what: str,
    *,
    dark_start: Literal["bright", "dark", "shelf"] = "dark",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = np.zeros(windows_s.size)
    n_c = np.zeros(windows_s.size)
    for k, w in enumerate(windows_s):
        _progress(progress, "scanning", 0.1 + 0.8 * k / windows_s.size, f"{what}: window {w * 1e6:.0f} us")
        opt = core.optimize_threshold(model, [float(w)], dark_start=dark_start)
        eps[k] = 0.5 * (opt.best.eps_B + opt.best.eps_D)
        n_c[k] = opt.best.n_c
    return windows_s, eps, n_c


def myerson_2008(progress: Progress | None = None) -> PresetResult:
    """Myerson's 40Ca+ PMT readout: the error at the paper's operating point and the exact chain's own optimum (Sections 8.2, 8.3)."""
    t0 = time.perf_counter()
    preset = core.MYERSON_CA40_PMT
    det = core.myerson_ca40_pmt_detector(window_s=420e-6)
    model = core.RecordModel.from_rates(preset.rates(), det)
    # the dark state is the shelved D5/2 level (lifetime 1.168 s), which decays to bright during the window: the "shelf" class
    pb = model.count_distribution("bright", 420e-6)
    pd = model.count_distribution("shelf", 420e-6)
    eps_b = 1.0 - pb.probability_above(5.5)
    eps_d = pd.probability_above(5.5)
    windows = np.geomspace(100e-6, 900e-6, 28)
    w, eps, n_c = _threshold_scan(model, windows, progress, "Myerson", dark_start="shelf")
    k = int(np.argmin(eps))
    charts = (
        ChartRecord(
            title="threshold readout error against the window (best threshold at each)",
            x_title="window (us)",
            y_title="(eps_B + eps_D)/2",
            series=(SeriesRecord("exact chain, Poisson counts", w * 1e6, eps),),
            log_y=True,
            markers=((420.0, "Myerson: 420 us, n_c = 5.5"),),
        ),
    )
    return PresetResult(
        preset_id="myerson_2008",
        simulated={
            "eps_at_point": 0.5 * (eps_b + eps_d),
            "eps_b_at_point": float(eps_b),
            "eps_d_at_point": float(eps_d),
            "eps_min": float(eps[k]),
            "window_opt": float(w[k]),
            "n_c_opt": float(n_c[k]),
        },
        simulated_uncertainty={},
        charts=charts,
        parameters={
            "detected_bright_per_s": float(preset.detected_bright_per_s or 0.0),
            "background_per_s": preset.background_per_s,
            "shelf_lifetime_s": float(preset.shelf_lifetime_s or 0.0),
        },
        notes=(f"optimum of the exact chain: {eps[k]:.3g} at n_c = {n_c[k]:.1f}, t_b = {w[k] * 1e6:.0f} us",),
        wall_time_s=time.perf_counter() - t0,
    )


def crain_2019(progress: Progress | None = None) -> PresetResult:
    """Crain's 171Yb+ SNSPD readout from the measured rates: the threshold error against the window (Section 8.3)."""
    t0 = time.perf_counter()
    preset = core.CRAIN_YB171_SNSPD
    det = core.crain_snspd_detector(window_s=22e-6)
    model = core.RecordModel.from_rates(preset.rates(), det)
    windows = np.geomspace(5e-6, 80e-6, 32)
    w, eps, n_c = _threshold_scan(model, windows, progress, "Crain")
    k = int(np.argmin(eps))
    charts = (
        ChartRecord(
            title="threshold readout error against the window (best threshold at each)",
            x_title="window (us)",
            y_title="(eps_B + eps_D)/2",
            series=(SeriesRecord("exact chain from the measured rates", w * 1e6, eps),),
            log_y=True,
            markers=((11.0, "Crain: 11 us average detection time"),),
        ),
    )
    return PresetResult(
        preset_id="crain_2019",
        simulated={"eps_min": float(eps[k]), "window_opt": float(w[k]), "n_c_opt": float(n_c[k])},
        simulated_uncertainty={},
        charts=charts,
        parameters={
            "detected_bright_per_s": float(preset.detected_bright_per_s or 0.0),
            "dark_pumping_per_s": preset.dark_pumping_per_s,
            "bright_pumping_per_s": preset.bright_pumping_per_s,
            "background_per_s": preset.background_per_s,
            "efficiency": preset.efficiency,
        },
        notes=(f"threshold optimum {eps[k]:.3g} at n_c = {n_c[k]:.1f}, window {w[k] * 1e6:.1f} us",),
        wall_time_s=time.perf_counter() - t0,
    )


RUNNERS: dict[str, Callable[[Progress | None], PresetResult]] = {
    "harty_2014": harty_2014,
    "james_1998": james_1998,
    "monroe_1995": monroe_1995,
    "roos_2000": roos_2000,
    "kirchmair_2009": kirchmair_2009,
    "myerson_2008": myerson_2008,
    "crain_2019": crain_2019,
}
"""One runner per experiment preset of ``viewmodel.presets.PRESETS`` (the circuit presets run through the machine)."""


def run_preset(preset_id: str, progress: Progress | None = None) -> PresetResult:
    spec: PresetSpec | None = PRESETS.get(preset_id)
    if spec is None:
        raise KeyError(f"unknown preset {preset_id!r}; known: {sorted(PRESETS)}")
    if spec.kind != "experiment":
        raise ValueError(
            f"{preset_id} is a circuit preset: it runs through Level 0's Run, not as an experiment"
        )
    return RUNNERS[preset_id](progress)
