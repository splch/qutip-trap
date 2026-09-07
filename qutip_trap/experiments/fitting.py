"""Fits, lineshapes and the observation model shared by the simulated experiments (PLAN.md Sections 7.5, 7.9, 13; M8).

The fit functions are the plan's own conventions (Section 13): the sideband excitation lineshape
P = [Omega^2/(Omega^2 + delta^2)] sin^2((t/2) sqrt(Omega^2 + delta^2)) with a carrier pi pulse at Omega t = pi and half depth
at delta = Omega, the carrier Rabi curve with the thermal Debye-Waller envelope of Section 4.2.7, a Ramsey fringe, a parity
oscillation. The half-Rabi form Omega^2/(Omega^2 + delta^2/4) sin^2(t sqrt(Omega^2 + delta^2/4)) that Wineland and Bluemel print
is kept as the NEGATIVE CONTROL of Section 9.17 ("fitting the half-Rabi form returns Omega/2") and is never the fit function.

The observation model is how a laboratory reads a population: ``shots`` projective measurements per point declared through the
readout's (eps_B, eps_D), so that every fitted parameter carries the statistical uncertainty a laboratory would quote
(Section 7.5: "Every fit reports an uncertainty"); ``shots=None`` returns the exact population (the M2 to M7 behaviour). The
randomness is keyed by (sample, trajectory, shot, ion, channel) through ``SeedSpec`` (Section 3.4).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.special import eval_genlaguerre

from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device

LineshapeForm = Literal["plan", "half_rabi"]


# ---- lineshapes (Section 13) --------------------------------------------------------------------------------------------------------


def sideband_lineshape(delta_rad_s: np.ndarray | float, omega_rad_s: float, t_s: float) -> np.ndarray:
    """P = [Omega^2/(Omega^2 + delta^2)] sin^2((t/2) sqrt(Omega^2 + delta^2)): the plan's excitation lineshape (Section 13,
    "Sideband excitation lineshape"; the calibration fit function of Section 7.5). Omega is the resonant Rabi frequency of the
    driven transition (carrier or sideband) in the (hbar Omega/2) convention: a pi pulse at Omega t = pi, half depth at delta = Omega."""
    d = np.asarray(delta_rad_s, dtype=float)
    w = math.sqrt(omega_rad_s**2) if omega_rad_s >= 0.0 else -omega_rad_s
    gen = np.sqrt(w * w + d * d)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(gen > 0.0, (w * w) / np.where(gen > 0.0, gen * gen, 1.0), 1.0)
    return np.asarray(weight * np.sin(0.5 * gen * t_s) ** 2)


def half_rabi_lineshape(delta_rad_s: np.ndarray | float, omega_rad_s: float, t_s: float) -> np.ndarray:
    """Omega^2/(Omega^2 + delta^2/4) sin^2(t sqrt(Omega^2 + delta^2/4)): the half-Rabi form of Wineland 1998 and Bluemel 2021 Eq.
    S28, whose Omega is HALF the plan's; doubled on ingest, never fitted (Section 7.9; the negative control of Section 9.17)."""
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
    shape = (
        sideband_lineshape(d, TWO_PI * omega, t_s)
        if form == "plan"
        else half_rabi_lineshape(d, TWO_PI * omega, t_s)
    )
    return np.asarray(amp * shape + off)


def thermal_rabi_model(p: np.ndarray, t: np.ndarray, eta: float, n_max: int = 200) -> np.ndarray:
    """P_1(t) = A sum_n P_n(nbar) sin^2(Omega_n t/2), Omega_n = 2 pi f e^{-eta^2/2} L_n(eta^2): p = (f_hz, nbar, A) (Section 4.2.7)."""
    f, nbar, amp = float(p[0]), max(float(p[1]), 0.0), float(p[2])
    n = np.arange(n_max)
    if nbar == 0.0:
        weights = np.zeros(n_max)
        weights[0] = 1.0
    else:
        weights = np.exp(n * math.log(nbar) - (n + 1) * math.log1p(nbar))
    omegas = 2.0 * math.pi * f * math.exp(-(eta**2) / 2.0) * eval_genlaguerre(n, 0, eta**2)
    out = amp * np.sum(weights[None, :] * np.sin(0.5 * omegas[None, :] * np.asarray(t)[:, None]) ** 2, axis=1)
    return np.asarray(out)


def thermal_rabi_model_fixed_nbar(
    p: np.ndarray, t: np.ndarray, eta: float, nbar: float, n_max: int = 200
) -> np.ndarray:
    """The carrier curve with nbar FIXED from the sideband-ratio thermometry (Section 7.5, bootstrap): p = (f_hz, A, B)."""
    return np.asarray(thermal_rabi_model(np.array([p[0], nbar, p[1]]), t, eta, n_max) + float(p[2]))


def debye_waller_branches(
    etas: Sequence[float], nbars: Sequence[float], weight_min: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """(weights, factors) of the joint thermal Fock distribution over the coupled modes: factor = prod_p e^{-eta_p^2/2} L_{n_p}(eta_p^2)
    (Wineland 1998 Eqs. 122-124; Section 4.2.7 iii), the carrier Rabi frequency's Debye-Waller reduction by EVERY mode the drive
    couples to, spectators included; branches below ``weight_min`` are dropped and the weights renormalized."""
    weights = np.array([1.0])
    factors = np.array([1.0])
    for eta, nb in zip(etas, nbars):
        if eta == 0.0:
            continue
        if nb <= 0.0:
            factors = factors * math.exp(-(eta**2) / 2.0)
            continue
        d = int(math.ceil(math.log(weight_min) / math.log(nb / (1.0 + nb)))) + 2
        n = np.arange(max(d, 2))
        p_n = np.exp(n * math.log(nb) - (n + 1) * math.log1p(nb))
        f_n = math.exp(-(eta**2) / 2.0) * eval_genlaguerre(n, 0, eta**2)
        weights = np.outer(weights, p_n).ravel()
        factors = np.outer(factors, f_n).ravel()
        keep = weights >= weight_min
        weights, factors = weights[keep], factors[keep]
    return weights / weights.sum(), factors


def multimode_rabi_model(
    p: np.ndarray, t: np.ndarray, etas: Sequence[float], nbars: Sequence[float], weight_min: float = 1e-6
) -> np.ndarray:
    """P_1(t) = A sum_branches w sin^2(2 pi f F t/2) + B over the joint Fock branches of every coupled mode (the multi-mode
    Debye-Waller carrier curve of Section 4.2.7), nbar of every mode FIXED from the thermometry: p = (f_hz, A, B)."""
    weights, factors = debye_waller_branches(etas, nbars, weight_min)
    omegas = TWO_PI * float(p[0]) * factors
    out = float(p[1]) * np.sum(
        weights[None, :] * np.sin(0.5 * omegas[None, :] * np.asarray(t, dtype=float)[:, None]) ** 2, axis=1
    )
    return np.asarray(out + float(p[2]))


def ramsey_model(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P = A cos(2 pi f t + phi_0) + B with p = (A, f_hz, phi_0, B)."""
    return np.asarray(float(p[0]) * np.cos(TWO_PI * float(p[1]) * np.asarray(t) + float(p[2])) + float(p[3]))


def parity_model(p: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Pi(phi) = C cos(2 phi + phi_0) + B with p = (C, phi_0, B) (Section 7.9)."""
    return np.asarray(float(p[0]) * np.cos(2.0 * np.asarray(phi) + float(p[1])) + float(p[2]))


# ---- weighted least squares -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    params: np.ndarray
    errors: np.ndarray
    """One-sigma uncertainties from the covariance (scaled by max(1, chi^2/dof) when sigmas are given)."""
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
    max_nfev: int | None = None,
) -> FitResult:
    """Least squares of ``model(p, x)`` to ``y`` with optional per-point sigmas (Section 7.5: every fit reports an uncertainty).

    Without sigmas the covariance is (J^T J)^-1 s^2 with s^2 the residual variance per degree of freedom; with them it is
    (J^T W J)^-1 times max(1, chi^2/dof), so an under-fitting model widens its own error bars rather than hiding the misfit.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if sigma is not None:
        sg = np.asarray(sigma, dtype=float)
        if sg.shape != ys.shape:
            raise ValueError("one sigma per data point")
        if np.any(sg <= 0.0):
            raise ValueError("sigmas are positive")
    else:
        sg = None

    def resid(p: np.ndarray) -> np.ndarray:
        r = np.asarray(model(p, xs) - ys)
        return r if sg is None else r / sg

    kwargs: dict[str, Any] = {"x_scale": "jac"}
    if bounds is not None:
        kwargs["bounds"] = bounds
    if max_nfev is not None:
        kwargs["max_nfev"] = max_nfev
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
    """Section 7.5: a fit that lands at the edge of its scan range marks the entry ``uncalibrated``."""
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
    """Fit (delta_0_hz, Omega_hz, A, B) of the excitation lineshape to a fine scan of one line (Section 7.5).

    ``guess`` is (delta_0_hz, Omega_hz); the default takes the peak and 1/(2 t) (a pi pulse's Omega). The plan's form is the
    fit function; ``form="half_rabi"`` exists so that the Section 9.17 negative control (it returns Omega/2) can be exercised.
    """
    x = np.asarray(detunings_hz, dtype=float)
    y = np.asarray(p1, dtype=float)
    if x.size < 5:
        raise ValueError("a lineshape fit needs at least five points")
    if guess is None:
        k = int(np.argmax(y))
        guess = (float(x[k]), 0.5 / t_s)
    span = float(x.max() - x.min())
    x0 = [guess[0], guess[1], max(float(y.max() - y.min()), 1e-3), float(y.min())]
    lo = [x.min() - 0.5 * span, 1e-6 * guess[1], 0.0, -0.5]
    hi = [x.max() + 0.5 * span, 50.0 * guess[1], 1.5, 1.0]
    return weighted_fit(
        lambda p, xx: lineshape_model(p, xx, t_s, form), x0, x, y, sigma=sigma, bounds=(lo, hi)
    )


# ---- observation model -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutErrors:
    """Per-ion (eps_B, eps_D) of the readout the experiments declare their populations through, and which qubit level is bright."""

    eps_b: tuple[float, ...]
    eps_d: tuple[float, ...]
    bright_level: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.eps_b) == len(self.eps_d) == len(self.bright_level)):
            raise ValueError("one (eps_B, eps_D, bright level) triple per ion")

    @classmethod
    def ideal(cls, n_ions: int, bright_level: int = 1) -> ReadoutErrors:
        return cls(tuple([0.0] * n_ions), tuple([0.0] * n_ions), tuple([bright_level] * n_ions))

    def declared_bright(self, ion: int, p1: float) -> float:
        """P(declared bright) for a qubit with population ``p1`` in level 1."""
        p_bright = p1 if self.bright_level[ion] == 1 else 1.0 - p1
        return p_bright * (1.0 - self.eps_b[ion]) + (1.0 - p_bright) * self.eps_d[ion]

    def declared_p1(self, ion: int, p1: float) -> float:
        """The declared population of level 1 (a bright declaration read as level 1 when level 1 is the bright one)."""
        db = self.declared_bright(ion, p1)
        return db if self.bright_level[ion] == 1 else 1.0 - db

    def declared_joint(self, ions: Sequence[int], populations: np.ndarray) -> np.ndarray:
        """The declared joint distribution over the computational strings of ``ions`` (product readout at zero crosstalk)."""
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


def readout_errors_for(device: Device, table: CalibrationTable | None = None) -> ReadoutErrors:
    """The readout the experiments see: the device's detection model at the table's threshold and window when calibrated,
    else at the detector's window with the threshold optimum of the true model (a laboratory's quick histogram threshold);
    ideal when the device carries no detection beam."""
    from qutip_trap.control.table import usable
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.detection import RecordModel
    from qutip_trap.readout.discriminate import ThresholdDiscriminator, optimize_threshold
    from qutip_trap.readout.fluorescence import detection_rates_for_ion

    n = device.crystal.n_ions
    eps_b: list[float] = []
    eps_d: list[float] = []
    bright: list[int] = []
    for i in range(n):
        try:
            idx = detection_beams(device, i)
        except ValueError:
            eps_b.append(0.0)
            eps_d.append(0.0)
            bright.append(1)
            continue
        rates, scheme, _model = detection_rates_for_ion(
            device.crystal.species[i],
            device.field.B_gauss,
            device.field.direction,
            [device.beams[k] for k in idx],
            position_m=tuple(float(x) for x in device.crystal.positions_m[i]),
        )
        record = RecordModel.from_rates(rates, device.detector)
        thr = table.detection.get("threshold") if table is not None else None
        win = table.detection.get("window_s") if table is not None else None
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
    """How an experiment reads populations: exact (``shots=None``) or ``shots`` declared measurements per point."""

    shots: int | None = None
    readout: ReadoutErrors | None = None
    seed: int = 0
    sample_id: int = 0
    stream: str = ""
    """The experiment's stream label (Section 3.4): the draws are keyed by (sample, point index, ion, outcome), which repeat
    when a calibration runs the same experiment twice or an experiment repeats a scan (a Ramsey per beam, a Rabi scan per shim
    point); the label, appended to the outcome key, keeps every such run's shot noise independent (``sub_stream``)."""

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


__all__ = [
    "FitResult",
    "LineshapeForm",
    "Observation",
    "ReadoutErrors",
    "at_scan_edge",
    "debye_waller_branches",
    "fit_lineshape",
    "half_rabi_lineshape",
    "lineshape_model",
    "multimode_rabi_model",
    "parity_model",
    "ramsey_model",
    "readout_errors_for",
    "sideband_lineshape",
    "sigmas_or_none",
    "thermal_rabi_model",
    "thermal_rabi_model_fixed_nbar",
    "weighted_fit",
]
