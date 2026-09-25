"""The level-C cooling solve: the internal levels plus one mode with exact displacement operators and the
recoil-resolved emission, the reference the level-A rates are checked against (PLAN.md Section 4.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt
from scipy.sparse.linalg import eigs

from qutip_trap.dynamics.multilevel import MultiLevelBuild
from qutip_trap.dynamics.steady import steady_state_direct


@dataclass(frozen=True)
class LevelCSteadyState:
    nbar: float
    rho: qt.Qobj
    populations: dict[str, float]
    fock_populations: np.ndarray
    boundary_population: float
    """Population of the top two Fock levels: the truncation check."""


def _static_mode_build(build: MultiLevelBuild, what: str) -> None:
    if build.space is None:
        raise ValueError("the level-C solve needs a build with a mode")
    if not build.static:
        raise NotImplementedError(f"the {what} is built for a consistent frame")


def level_c_steady_state(build: MultiLevelBuild) -> LevelCSteadyState:
    """The joint steady state (direct sparse solve) and its mean phonon number."""
    _static_mode_build(build, "level-C steady state")
    assert build.space is not None and isinstance(build.H, qt.Qobj)
    rho = steady_state_direct(build.H, build.c_ops)
    pn = build.space.fock_populations(rho, 0)
    return LevelCSteadyState(
        nbar=float(np.dot(np.arange(pn.size), pn)),
        rho=rho,
        populations=build.populations(rho),
        fock_populations=pn,
        boundary_population=float(pn[-2:].sum()),
    )


def level_c_relaxation_rate(build: MultiLevelBuild, *, k: int = 6) -> float:
    """W = -Re of the slowest nonzero Liouvillian eigenvalue (s^-1), by shift-invert ARPACK about zero; valid when the
    motional relaxation is the slowest mode (W << every internal rate)."""
    _static_mode_build(build, "relaxation rate")
    L = build.liouvillian().to("CSR").data.as_scipy().tocsc()
    # the shift sits just below zero: a shift AT the stationary eigenvalue makes the factorization singular
    sigma = -1e-7 * max(build.level_rates_rad_s.values())
    vals = eigs(L, k=k, sigma=sigma, return_eigenvectors=False)
    rates = np.sort(-np.real(vals))
    keep = [r for r in rates if r > 10.0 * abs(sigma)]
    if not keep:
        raise RuntimeError("no nonzero Liouvillian eigenvalue found near zero; increase k")
    return float(keep[0])
