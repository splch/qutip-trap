"""Level-C cooling solve (the master equation of the internal levels plus one mode) and the level-B Fock rate equation.

Level B is dP(n)/dt = eta^2 [A_-((n+1)P(n+1) - nP(n)) + A_+(nP(n-1) - (n+1)P(n))], bare A_+- with eta^2 outside; a
configuration with A_- <= A_+ raises rather than returning a negative occupation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import qutip as qt
from scipy.optimize import curve_fit

from qutip_trap.dynamics.multilevel import MultiLevelBuild
from qutip_trap.dynamics.steady import steady_state_direct
from qutip_trap.hilbert.operators import thermal_populations
from qutip_trap.light.bloch import CoolingError


@dataclass(frozen=True)
class LevelCSteadyState:
    nbar: float
    rho: qt.Qobj
    populations: dict[str, float]
    fock_populations: np.ndarray
    boundary_population: float
    """Population of the top two Fock levels (the truncation check)."""


@dataclass(frozen=True)
class LevelCTrace:
    times_s: np.ndarray
    nbar: np.ndarray
    excited_population: np.ndarray
    final: qt.Qobj


@dataclass(frozen=True)
class RelaxationFit:
    """<n>(t) = nbar + (n0 - nbar) exp(-W t) fitted to a level-C trace."""

    rate_per_s: float
    nbar: float
    n0: float
    rms_residual: float


def _joint_initial(build: MultiLevelBuild, internal: str, fock: int | None, thermal: float | None) -> qt.Qobj:
    if build.space is None:
        raise ValueError("the level-C solve needs a build with a mode")
    d = build.space.dims[1]
    ket = build.internal_state(internal)
    if thermal is not None:
        p = thermal_populations(thermal, d)
        motional = qt.Qobj(np.diag(p / p.sum()), dims=[[d], [d]])
        return qt.tensor(qt.ket2dm(ket), motional)
    return qt.tensor(qt.ket2dm(ket), qt.fock_dm(d, 0 if fock is None else fock))


def level_c_steady_state(build: MultiLevelBuild) -> LevelCSteadyState:
    """The joint steady state and its mean phonon number (direct sparse solve)."""
    if build.space is None:
        raise ValueError("the level-C solve needs a build with a mode")
    if not build.static:
        raise NotImplementedError("the level-C steady state is built for a consistent frame")
    assert isinstance(build.H, qt.Qobj)
    rho = steady_state_direct(build.H, build.c_ops)
    pn = build.space.fock_populations(rho, 0)
    return LevelCSteadyState(
        nbar=float(np.dot(np.arange(pn.size), pn)),
        rho=rho,
        populations=build.populations(rho),
        fock_populations=pn,
        boundary_population=float(pn[-2:].sum()),
    )


def level_c_trace(
    build: MultiLevelBuild,
    times_s: Sequence[float] | np.ndarray,
    *,
    internal: str,
    fock: int | None = None,
    thermal: float | None = None,
) -> LevelCTrace:
    """<n>(t) and the excited population from a Fock or thermal start (``mesolve``)."""
    times = np.asarray(times_s, dtype=float)
    rho0 = _joint_initial(build, internal, fock, thermal)
    excited = [lab for lab in build.labels if build.level_of(lab) in build.level_rates_rad_s]
    e_ops = {"n": build.number(), "P_e": build.manifold_projector(excited)}
    res = qt.mesolve(
        build.H,
        rho0,
        times,
        c_ops=list(build.c_ops),
        e_ops=e_ops,
        options={
            "store_final_state": True,
            "progress_bar": "",
            "method": "dop853",
            "atol": 1e-10,
            "rtol": 1e-8,
        },
    )
    return LevelCTrace(times, np.real(res.expect[0]), np.real(res.expect[1]), res.final_state)


def level_c_relaxation_rate(build: MultiLevelBuild, *, k: int = 6) -> float:
    """W = -Re of the slowest nonzero Liouvillian eigenvalue (s^-1, by shift-invert ARPACK): the phonon relaxation rate,
    valid when the motional relaxation is the slowest mode."""
    if build.space is None:
        raise ValueError("the level-C solve needs a build with a mode")
    if not build.static:
        raise NotImplementedError("the relaxation rate is built for a consistent frame")
    from scipy.sparse.linalg import eigs

    L = build.liouvillian().to("CSR").data.as_scipy().tocsc()
    # shift just below zero: at the stationary eigenvalue itself the shift-invert factorization is singular
    gamma_max = max(build.level_rates_rad_s.values())
    sigma = -1e-7 * gamma_max
    vals = eigs(L, k=k, sigma=sigma, return_eigenvectors=False)
    rates = np.sort(-np.real(vals))
    # drop the stationary state (|lambda| below the shift-invert resolution)
    keep = [r for r, v in zip(rates, vals[np.argsort(-np.real(vals))]) if r > 10.0 * abs(sigma)]
    if not keep:
        raise RuntimeError("no nonzero Liouvillian eigenvalue found near zero; increase k")
    return float(keep[0])


def fit_relaxation(times_s: np.ndarray, nbar: np.ndarray) -> RelaxationFit:
    """Least-squares fit of nbar + (n0 - nbar) exp(-W t)."""

    def model(t: np.ndarray, w: float, nb: float, n0: float) -> np.ndarray:
        return nb + (n0 - nb) * np.exp(-w * t)

    span = float(times_s[-1] - times_s[0])
    guess = (1.0 / max(span / 3.0, 1e-30), float(nbar[-1]), float(nbar[0]))
    popt, _ = curve_fit(model, times_s, nbar, p0=guess, maxfev=20000)
    resid = nbar - model(times_s, *popt)
    return RelaxationFit(float(popt[0]), float(popt[1]), float(popt[2]), float(np.sqrt(np.mean(resid**2))))


def phonon_generator(a_plus_per_s: float, a_minus_per_s: float, eta: float, d: int) -> np.ndarray:
    """The birth-death generator M on Fock populations (dP/dt = M P) with eta^2 outside A_+-."""
    if d < 2:
        raise ValueError("at least two Fock levels")
    up = eta**2 * a_plus_per_s
    down = eta**2 * a_minus_per_s
    m = np.zeros((d, d))
    for n in range(d):
        if n + 1 < d:
            m[n + 1, n] += (n + 1) * up
            m[n, n] -= (n + 1) * up
        if n > 0:
            m[n - 1, n] += n * down
            m[n, n] -= n * down
    return m


def phonon_steady_state(a_plus_per_s: float, a_minus_per_s: float) -> float:
    """nbar = A_+/(A_- - A_+), eta-independent; raises CoolingError when the configuration heats."""
    if a_minus_per_s <= a_plus_per_s:
        raise CoolingError(f"A_- = {a_minus_per_s:.4g} <= A_+ = {a_plus_per_s:.4g}: no cooling steady state")
    return a_plus_per_s / (a_minus_per_s - a_plus_per_s)


def phonon_mean_closed_form(
    a_plus_per_s: float, a_minus_per_s: float, eta: float, n0: float, times_s: np.ndarray
) -> np.ndarray:
    """<n>(t) = nbar + (n0 - nbar) exp(-W t), W = eta^2 (A_- - A_+) (Cirac et al. 1992 Eqs. 29-32)."""
    nb = phonon_steady_state(a_plus_per_s, a_minus_per_s)
    w = eta**2 * (a_minus_per_s - a_plus_per_s)
    return nb + (n0 - nb) * np.exp(-w * np.asarray(times_s, dtype=float))


def phonon_rate_equation_mesolve(
    a_plus_per_s: float, a_minus_per_s: float, eta: float, n0: int, d: int, times_s: np.ndarray
) -> np.ndarray:
    """<n>(t) from ``mesolve`` on the two collapse operators sqrt(eta^2 A_+) a^dagger and sqrt(eta^2 A_-) a (level B)."""
    a = qt.destroy(d)
    c_ops = [np.sqrt(eta**2 * a_plus_per_s) * a.dag(), np.sqrt(eta**2 * a_minus_per_s) * a]
    res = qt.mesolve(
        0.0 * a.dag() * a,
        qt.fock_dm(d, n0),
        np.asarray(times_s, dtype=float),
        c_ops=c_ops,
        e_ops=[a.dag() * a],
        options={"progress_bar": "", "atol": 1e-13, "rtol": 1e-11, "method": "dop853"},
    )
    return np.real(res.expect[0])


__all__ = [
    "LevelCSteadyState",
    "LevelCTrace",
    "RelaxationFit",
    "fit_relaxation",
    "level_c_relaxation_rate",
    "level_c_steady_state",
    "level_c_trace",
    "phonon_generator",
    "phonon_mean_closed_form",
    "phonon_rate_equation_mesolve",
    "phonon_steady_state",
]
