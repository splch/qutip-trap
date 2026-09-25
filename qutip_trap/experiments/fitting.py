"""Fits, lineshapes and the observation model of the simulated experiments (PLAN.md Section 7.5).

The fit functions are the plan's conventions: the excitation lineshape [Omega^2/(Omega^2 + delta^2)] sin^2((t/2) sqrt(Omega^2 + delta^2))
(a pi pulse at Omega t = pi, half depth at delta = Omega), the carrier Rabi curve with the thermal Debye-Waller envelope, a
Ramsey fringe and the fringe C cos(k phi + phi_0) + B. The observation model reads a population through ``shots`` projective
measurements declared through the readout's (eps_B, eps_D), so every fit carries the uncertainty a laboratory would quote;
``shots=None`` gives the exact population. The draws are keyed by (sample, point, ion, outcome) through ``SeedSpec``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import eval_genlaguerre

from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.dynamics.operators import thermal_populations
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.light.bloch import BlochModel
    from qutip_trap.readout.fluorescence import FluorescenceRates, ReadoutScheme

LineshapeForm = Literal["plan", "half_rabi"]


# ---- lineshapes and models ------------------------------------------------------------------------------------------------------


def sideband_lineshape(delta_rad_s: np.ndarray | float, omega_rad_s: float, t_s: float) -> np.ndarray:
    """P = [Omega^2/(Omega^2 + delta^2)] sin^2((t/2) sqrt(Omega^2 + delta^2)) with Omega the resonant Rabi frequency of the
    driven line in the (hbar Omega/2) convention (Section 13, "Sideband excitation lineshape")."""
    d = np.asarray(delta_rad_s, dtype=float)
    w = abs(omega_rad_s)
    gen = np.sqrt(w * w + d * d)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(gen > 0.0, (w * w) / np.where(gen > 0.0, gen * gen, 1.0), 1.0)
    return np.asarray(weight * np.sin(0.5 * gen * t_s) ** 2)


def half_rabi_lineshape(delta_rad_s: np.ndarray | float, omega_rad_s: float, t_s: float) -> np.ndarray:
    """Omega^2/(Omega^2 + delta^2/4) sin^2(t sqrt(Omega^2 + delta^2/4)), the form of Wineland 1998 and Bluemel 2021 Eq. S28
    whose Omega is half the plan's: a negative control (fitting it returns Omega/2), never the fit function."""
    d = np.asarray(delta_rad_s, dtype=float)
    w = abs(omega_rad_s)
    gen2 = w * w + 0.25 * d * d
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(gen2 > 0.0, (w * w) / np.where(gen2 > 0.0, gen2, 1.0), 1.0)
    return np.asarray(weight * np.sin(np.sqrt(gen2) * t_s) ** 2)


def lineshape_model(
    p: np.ndarray, detunings_hz: np.ndarray, t_s: float, form: LineshapeForm = "plan"
) -> np.ndarray:
    """P(delta) = A L(2 pi (delta - delta_0), 2 pi Omega, t) + B with p = (delta_0_hz, omega_hz, A, B)."""
    delta0, omega, amp, off = float(p[0]), abs(float(p[1])), float(p[2]), float(p[3])
    d = TWO_PI * (np.asarray(detunings_hz, dtype=float) - delta0)
    shape = half_rabi_lineshape if form == "half_rabi" else sideband_lineshape
    return np.asarray(amp * shape(d, TWO_PI * omega, t_s) + off)


def thermal_n_max(nbar: float, tail: float = 1e-7) -> int:
    """The Fock cutoff of a thermal state: the smallest n with the geometric tail (nbar/(nbar + 1))^n below ``tail``, plus
    two (145 at nbar = 10, 41 at nbar = 2); 2 at nbar = 0."""
    if nbar <= 0.0:
        return 2
    return int(math.ceil(math.log(tail) / math.log(nbar / (nbar + 1.0)))) + 2


def thermal_rabi_model(p: np.ndarray, t: np.ndarray, eta: float) -> np.ndarray:
    """P_1(t) = A sum_n P_n(nbar) sin^2(Omega_n t/2), Omega_n = 2 pi f e^{-eta^2/2} L_n(eta^2), p = (f_hz, nbar, A)
    (Section 4.2.7), over the first 200 Fock levels."""
    f, nbar, amp = float(p[0]), max(float(p[1]), 0.0), float(p[2])
    n = np.arange(200)
    omegas = TWO_PI * f * math.exp(-(eta**2) / 2.0) * eval_genlaguerre(n, 0, eta**2)
    weights = thermal_populations(nbar, 200)
    out = amp * np.sum(weights[None, :] * np.sin(0.5 * omegas[None, :] * np.asarray(t)[:, None]) ** 2, axis=1)
    return np.asarray(out)


def debye_waller_branches(
    etas: Sequence[float], nbars: Sequence[float], weight_min: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """(weights, factors) of the joint thermal Fock distribution of the coupled modes, factor = prod_p e^{-eta_p^2/2}
    L_{n_p}(eta_p^2) (Wineland 1998 Eqs. 122-124): the carrier's Debye-Waller reduction by every mode the drive couples to;
    branches below ``weight_min`` are dropped and the weights renormalized."""
    weights = np.array([1.0])
    factors = np.array([1.0])
    for eta, nb in zip(etas, nbars):
        if eta == 0.0:
            continue
        if nb <= 0.0:
            factors = factors * math.exp(-(eta**2) / 2.0)
            continue
        d = thermal_n_max(nb, weight_min)
        f_n = math.exp(-(eta**2) / 2.0) * eval_genlaguerre(np.arange(d), 0, eta**2)
        weights = np.outer(weights, thermal_populations(nb, d)).ravel()
        factors = np.outer(factors, f_n).ravel()
        keep = weights >= weight_min
        weights, factors = weights[keep], factors[keep]
    return weights / weights.sum(), factors


def multimode_rabi_model(
    p: np.ndarray, t: np.ndarray, etas: Sequence[float], nbars: Sequence[float], weight_min: float = 1e-6
) -> np.ndarray:
    """P_1(t) = A sum_branches w sin^2(2 pi f F t/2) + B over ``debye_waller_branches`` with every mode's nbar fixed:
    p = (f_hz, A, B)."""
    weights, factors = debye_waller_branches(etas, nbars, weight_min)
    omegas = TWO_PI * float(p[0]) * factors
    out = float(p[1]) * np.sum(
        weights[None, :] * np.sin(0.5 * omegas[None, :] * np.asarray(t, dtype=float)[:, None]) ** 2, axis=1
    )
    return np.asarray(out + float(p[2]))


def ramsey_model(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P = A cos(2 pi f t + phi_0) + B with p = (A, f_hz, phi_0, B)."""
    return np.asarray(float(p[0]) * np.cos(TWO_PI * float(p[1]) * np.asarray(t) + float(p[2])) + float(p[3]))


def wrap_angle(angle: float) -> float:
    """``angle`` wrapped to [-pi, pi)."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


# ---- weighted least squares -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    """A weighted least-squares fit: the parameters, their one-sigma errors (from the covariance, scaled by max(1, chi^2/dof)
    when sigmas are given), the reduced chi-square, whether the optimiser converged, its message and the points fitted."""

    params: np.ndarray
    errors: np.ndarray
    chi2_per_dof: float
    converged: bool
    message: str
    n_points: int

    def value(self, k: int) -> tuple[float, float]:
        return float(self.params[k]), float(self.errors[k])


def weighted_fit(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
    *,
    sigma: np.ndarray | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
) -> FitResult:
    """Least squares of ``model(p, x)`` to ``y`` with optional per-point sigmas (Section 7.5: every fit reports an
    uncertainty). Without sigmas the covariance is (J^T J)^-1 s^2 with s^2 the residual variance per degree of freedom; with
    them it is (J^T W J)^-1 max(1, chi^2/dof), so an under-fitting model widens its own error bars."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    sg = None if sigma is None else np.asarray(sigma, dtype=float)
    if sg is not None and sg.shape != ys.shape:
        raise ValueError("one sigma per data point")
    if sg is not None and np.any(sg <= 0.0):
        raise ValueError("sigmas are positive")

    def resid(p: np.ndarray) -> np.ndarray:
        r = np.asarray(model(p, xs) - ys)
        return r if sg is None else r / sg

    kwargs: dict[str, Any] = {"x_scale": "jac"}
    if bounds is not None:
        kwargs["bounds"] = bounds
    res = least_squares(resid, np.asarray(x0, dtype=float), **kwargs)
    dof = max(len(ys) - len(x0), 1)
    chi2 = float(np.sum(res.fun**2))
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac)
        scale = chi2 / dof if sg is None else max(1.0, chi2 / dof)
        err = np.sqrt(np.maximum(np.diag(cov) * scale, 0.0))
        ok = bool(res.success) and bool(np.all(np.isfinite(err)))
    except np.linalg.LinAlgError:
        err = np.full(len(x0), np.nan)
        ok = False
    return FitResult(
        params=np.asarray(res.x, dtype=float),
        errors=np.asarray(err, dtype=float),
        chi2_per_dof=chi2 / dof,
        converged=ok,
        message=str(res.message),
        n_points=int(len(ys)),
    )


def at_scan_edge(value: float, lo: float, hi: float, fraction: float = 0.02) -> bool:
    """Whether ``value`` lies within ``fraction`` of the span of an edge of [lo, hi] (Section 7.5: such a fit leaves its
    entry ``uncalibrated``)."""
    span = hi - lo
    if span <= 0.0:
        return True
    return value <= lo + fraction * span or value >= hi - fraction * span


def fit_lineshape(
    detunings_hz: np.ndarray,
    p1: np.ndarray,
    t_s: float,
    *,
    sigma: np.ndarray | None = None,
    guess: tuple[float, float] | None = None,
    form: LineshapeForm = "plan",
) -> FitResult:
    """(delta_0_hz, Omega_hz, A, B) of the excitation lineshape fitted to a fine scan of one line; ``guess`` is
    (delta_0_hz, Omega_hz), by default the peak and 1/(2 t). ``form="half_rabi"`` fits the negative control (it returns
    Omega/2)."""
    x = np.asarray(detunings_hz, dtype=float)
    y = np.asarray(p1, dtype=float)
    if x.size < 5:
        raise ValueError("a lineshape fit needs at least five points")
    if guess is None:
        guess = (float(x[int(np.argmax(y))]), 0.5 / t_s)
    span = float(x.max() - x.min())
    x0 = [guess[0], guess[1], max(float(y.max() - y.min()), 1e-3), float(y.min())]
    lo = [x.min() - 0.5 * span, 1e-6 * guess[1], 0.0, -0.5]
    hi = [x.max() + 0.5 * span, 50.0 * guess[1], 1.5, 1.0]
    return weighted_fit(
        lambda p, xx: lineshape_model(p, xx, t_s, form), x0, x, y, sigma=sigma, bounds=(lo, hi)
    )


class Fringe(NamedTuple):
    """A fitted fringe C cos(k phi + phi_0) + B: (value, sigma) of C >= 0, of phi_0 in [-pi, pi) and of B."""

    contrast: tuple[float, float]
    phase_rad: tuple[float, float]
    offset: tuple[float, float]
    chi2_per_dof: float
    converged: bool


def fit_fringe(phases: np.ndarray, signal: np.ndarray, sigma: np.ndarray | None, k: float) -> Fringe:
    """C cos(k phi + phi_0) + B by weighted least squares from eight starting phases, the lowest chi^2 kept."""
    x = np.asarray(phases, dtype=float)
    y = np.asarray(signal, dtype=float)

    def model(p: np.ndarray, xx: np.ndarray) -> np.ndarray:
        return np.asarray(float(p[0]) * np.cos(k * np.asarray(xx) + float(p[1])) + float(p[2]))

    c0 = max(0.5 * float(y.max() - y.min()), 1e-3)
    starts = np.linspace(-math.pi, math.pi, 8, endpoint=False)
    best = min(
        (weighted_fit(model, [c0, g, float(y.mean())], x, y, sigma=sigma) for g in starts),
        key=lambda fit: fit.chi2_per_dof,
    )
    c = float(best.params[0])
    return Fringe(
        contrast=(abs(c), float(best.errors[0])),
        phase_rad=(wrap_angle(float(best.params[1]) + (math.pi if c < 0.0 else 0.0)), float(best.errors[1])),
        offset=best.value(2),
        chi2_per_dof=best.chi2_per_dof,
        converged=best.converged,
    )


# ---- observation model -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutErrors:
    """Per-ion (eps_B, eps_D) of the readout the experiments declare their populations through, and which level is bright."""

    eps_b: tuple[float, ...]
    eps_d: tuple[float, ...]
    bright_level: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.eps_b) == len(self.eps_d) == len(self.bright_level)):
            raise ValueError("one (eps_B, eps_D, bright level) triple per ion")

    def declared_bright(self, ion: int, p1: float) -> float:
        """P(declared bright) for a qubit with population ``p1`` in level 1."""
        p_bright = p1 if self.bright_level[ion] == 1 else 1.0 - p1
        return p_bright * (1.0 - self.eps_b[ion]) + (1.0 - p_bright) * self.eps_d[ion]

    def declared_p1(self, ion: int, p1: float) -> float:
        """The declared population of level 1."""
        db = self.declared_bright(ion, p1)
        return db if self.bright_level[ion] == 1 else 1.0 - db

    def declared_joint(self, ions: Sequence[int], populations: np.ndarray) -> np.ndarray:
        """The declared joint distribution over the computational strings of ``ions`` (product readout, no crosstalk)."""
        pops = np.asarray(populations, dtype=float)
        n = len(ions)
        out = np.zeros_like(pops)
        for idx in range(2**n):
            true_bits = [(idx >> (n - 1 - k)) & 1 for k in range(n)]
            for jdx in range(2**n):
                decl_bits = [(jdx >> (n - 1 - k)) & 1 for k in range(n)]
                prob = 1.0
                for k, ion in enumerate(ions):
                    p_decl1 = self.declared_p1(ion, float(true_bits[k]))
                    prob *= p_decl1 if decl_bits[k] == 1 else 1.0 - p_decl1
                out[jdx] += pops[idx] * prob
        return out


def _detection_rates(
    device: Device, ion: int, *, micromotion: bool = False
) -> tuple[FluorescenceRates, ReadoutScheme, BlochModel]:
    """``detection_rates_for_ion`` for ``ion`` under its detection beams; ``micromotion`` applies the J_0^2/J_1^2 factor of
    Section 8.8 that the readout stage of ``run`` applies (``run.job.detection_micromotion``)."""
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.fluorescence import detection_rates_for_ion
    from qutip_trap.run.job import detection_micromotion

    beams = [device.beams[k] for k in detection_beams(device, ion)]
    beta, omega_rf = detection_micromotion(device, ion, beams) if micromotion else (0.0, 0.0)
    return detection_rates_for_ion(
        device.crystal.species[ion],
        device.field.B_gauss,
        device.field.direction,
        beams,
        position_m=tuple(float(x) for x in device.crystal.positions_m[ion]),
        micromotion_beta=beta,
        omega_rf_rad_s=omega_rf,
    )


def readout_errors_for(device: Device, table: CalibrationTable | None = None) -> ReadoutErrors:
    """The readout the experiments see: per ion, the detection model's (eps_B, eps_D) at the table's threshold and window
    when calibrated, else at the detector's window and the true model's optimal threshold; (0, 0) for an ion no detection
    beam addresses."""
    from qutip_trap.control.table import usable
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.detection import RecordModel
    from qutip_trap.readout.discriminate import ThresholdDiscriminator, optimize_threshold

    eps_b: list[float] = []
    eps_d: list[float] = []
    bright: list[int] = []
    thr = table.detection.get("threshold") if table is not None else None
    win = table.detection.get("window_s") if table is not None else None
    for i in range(device.crystal.n_ions):
        try:
            detection_beams(device, i)
        except ValueError:
            eps_b.append(0.0)
            eps_d.append(0.0)
            bright.append(1)
            continue
        rates, scheme, _model = _detection_rates(device, i)
        record = RecordModel.from_rates(rates, device.detector)
        if usable(thr) and usable(win) and thr is not None and win is not None:
            disc = ThresholdDiscriminator(float(thr.value), float(win.value))
        else:
            disc = optimize_threshold(record, [float(device.detector.window_s)]).discriminator
        other = scheme.start_distribution(1 - scheme.bright_level)
        start = max(other, key=other.__getitem__)
        eb, ed = disc.error_rates(record, dark_start=start)
        eps_b.append(float(eb))
        eps_d.append(float(ed))
        bright.append(int(scheme.bright_level))
    return ReadoutErrors(tuple(eps_b), tuple(eps_d), tuple(bright))


@dataclass(frozen=True)
class Observation:
    """How an experiment reads populations: exact (``shots=None``) or ``shots`` declared measurements per point, through
    ``readout`` when given. ``stream`` labels the experiment (Section 3.4): the draws are keyed by (sample, point, ion,
    outcome), which repeat when a calibration runs an experiment twice or an experiment repeats a scan, so the label keeps
    every such run's shot noise independent (``single_ion.sub_stream``)."""

    shots: int | None = None
    readout: ReadoutErrors | None = None
    seed: int = 0
    sample_id: int = 0
    stream: str = ""

    def __post_init__(self) -> None:
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots is a positive count or None (exact populations)")

    def _rng(self, key: str, index: int, ion: int) -> np.random.Generator:
        channel = f"{self.stream}|{key}" if self.stream else key
        return np.random.default_rng(
            SeedSpec(int(self.seed)).child(self.sample_id, 0, int(index), int(ion), channel)
        )

    def p1(self, exact: float, ion: int, key: str, index: int) -> tuple[float, float | None]:
        """(measured P_1, sigma): the exact declared value with sigma None, or a binomial draw with its standard error."""
        p_true = min(max(float(exact), 0.0), 1.0)
        declared = self.readout.declared_p1(ion, p_true) if self.readout is not None else p_true
        if self.shots is None:
            return declared, None
        k = int(self._rng(key, index, ion).binomial(self.shots, min(max(declared, 0.0), 1.0)))
        frac = k / self.shots
        sigma = max(math.sqrt(frac * (1.0 - frac) / self.shots), 1.0 / self.shots)
        return frac, sigma

    def joint(
        self, exact: np.ndarray, ions: Sequence[int], key: str, index: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """(measured joint distribution over the computational strings of ``ions``, sigmas): a multinomial draw of ``shots``."""
        pops = np.clip(np.asarray(exact, dtype=float), 0.0, None)
        pops = pops / pops.sum() if pops.sum() > 0.0 else pops
        declared = self.readout.declared_joint(ions, pops) if self.readout is not None else pops
        if self.shots is None:
            return declared, None
        counts = self._rng(key, index, int(ions[0])).multinomial(self.shots, declared / declared.sum())
        frac = counts / self.shots
        sigma = np.maximum(np.sqrt(frac * (1.0 - frac) / self.shots), 1.0 / self.shots)
        return frac, sigma

    def noise(self, sigma: float, key: str, index: int, ion: int = 0) -> float:
        """A Gaussian error of standard deviation ``sigma`` on a measured signal (0 for exact observations)."""
        if self.shots is None or sigma <= 0.0:
            return 0.0
        return float(self._rng(key, index, ion).normal(0.0, sigma))

    def counts(self, mean: float, key: str, index: int, ion: int = 0) -> tuple[float, float]:
        """(measured count, sigma) for a Poisson signal of ``mean`` photons (exact with sigma sqrt(mean) when shots is None)."""
        if self.shots is None:
            return float(mean), math.sqrt(max(float(mean), 1.0))
        k = float(self._rng(key, index, ion).poisson(max(float(mean), 0.0)))
        return k, math.sqrt(max(k, 1.0))


def sigmas_or_none(values: Sequence[float | None]) -> np.ndarray | None:
    """An array of sigmas, or None when the observations were exact."""
    if any(v is None for v in values):
        return None
    return np.asarray([float(v) for v in values if v is not None], dtype=float)
