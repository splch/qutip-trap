"""The integrator ladder and the tolerance-convergence check (PLAN.md Section 5.3).

Kets evolve with ``sesolve``, density matrices and any collapse operator with ``mesolve`` (the drive operators are CSR there,
so the Liouvillian stays sparse). The ladder is the configured integrators (``dop853`` then ``vern9``) and then the last one
at atol 1e-8 with a max_step of one fourteenth of the fastest period; never a multistep method, whose damping distorts the
oscillatory spectrum of -iH. The default atol relaxes from 1e-10 to 1e-8 above d_m = 100.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
import qutip as qt
from qutip.solver.integrator import IntegratorException

from qutip_trap.options import Numerics


class Rung(NamedTuple):
    """One rung of the integrator ladder: the QuTiP method, its atol, and a max_step (0 = the integrator's own)."""

    method: str
    atol: float
    max_step: float


def ladder(options: Numerics, omega_max_rad_s: float | None) -> tuple[Rung, ...]:
    """The escalation of Section 5.3: every configured integrator at ``options.atol``, then the last one again at an atol
    of at least 1e-8 with max_step a fourteenth of the fastest period ``2 pi/omega_max``. ``evolve`` and the engine's
    trajectory path climb it on the integrator's own ``IntegratorException``."""
    atol = options.atol
    max_step = (2.0 * math.pi / omega_max_rad_s / 14.0) if omega_max_rad_s else 0.0
    rungs = [Rung(m, atol, 0.0) for m in options.integrators]
    rungs.append(Rung(options.integrators[-1], max(atol, 1e-8), max_step))
    return tuple(rungs)


def retry_note(rung: Rung, exc: Exception) -> str:
    """The note a failed rung leaves in the retries a solve reports."""
    return f"{rung.method}@atol={rung.atol:g},max_step={rung.max_step:g}: {type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class Evolution:
    """One integration: the final state, the state and the expectation values at every requested time, and the rung that
    succeeded (integrator and atol) after the failed ones (``retries``)."""

    final: qt.Qobj
    states: tuple[qt.Qobj, ...]
    expect: dict[str, np.ndarray]
    integrator: str
    atol: float
    retries: tuple[str, ...]


def _solve(
    H: qt.QobjEvo | qt.Qobj,
    state0: qt.Qobj,
    times: np.ndarray,
    c_ops: Sequence[qt.Qobj],
    e_ops: Mapping[str, qt.Qobj] | None,
    method: str,
    atol: float,
    rtol: float,
    nsteps: int,
    max_step: float,
    propagator: bool,
) -> qt.solver.Result:
    options: dict[str, object] = {
        "method": method,
        "atol": atol,
        "rtol": rtol,
        "nsteps": nsteps,
        "store_final_state": True,
        "store_states": True,
        "normalize_output": False,
        "progress_bar": "",
    }
    if max_step > 0.0:
        options["max_step"] = max_step
    eops = dict(e_ops) if e_ops else None
    with warnings.catch_warnings():
        # scipy's dop853 warns before it raises on a too-small step; the ladder records the failure and escalates
        warnings.filterwarnings("ignore", message=".*step size becomes too small.*", category=UserWarning)
        if not c_ops and (state0.isket or propagator):
            # an operator-valued state under sesolve integrates the propagator U(t) itself
            return qt.sesolve(H, state0, times, e_ops=eops, options=options)
        return qt.MESolver(H, c_ops=list(c_ops), options=options).run(state0, times, e_ops=eops)


def evolve(
    H: qt.QobjEvo | qt.Qobj,
    state0: qt.Qobj,
    times_s: Sequence[float] | np.ndarray,
    *,
    c_ops: Sequence[qt.Qobj] = (),
    e_ops: Mapping[str, qt.Qobj] | None = None,
    options: Numerics | None = None,
    omega_max_rad_s: float | None = None,
    propagator: bool = False,
) -> Evolution:
    """Integrate from ``times_s[0]`` to ``times_s[-1]`` through the ladder, storing the state at every time.

    ``propagator=True`` integrates an
    operator-valued ``state0`` (the identity) under ``sesolve``, so the stored states are the propagators U(t, t_0). Only
    the integrator's own ``IntegratorException`` escalates; anything else propagates.
    """
    opts = options or Numerics()
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be an increasing array with at least two points")
    retries: list[str] = []
    result = None
    rungs = ladder(opts, omega_max_rad_s)
    used = rungs[-1]
    for rung in rungs:
        try:
            result = _solve(
                H,
                state0,
                times,
                c_ops,
                e_ops,
                rung.method,
                rung.atol,
                opts.rtol,
                opts.nsteps,
                rung.max_step,
                propagator,
            )
            used = rung
            break
        except IntegratorException as exc:
            retries.append(retry_note(rung, exc))
    if result is None:
        raise RuntimeError("every rung of the integrator ladder failed: " + " | ".join(retries))
    expect: dict[str, np.ndarray] = {}
    if e_ops:
        for key, arr in zip(e_ops.keys(), result.expect):
            expect[str(key)] = np.asarray(arr)
    return Evolution(
        final=result.final_state,
        states=tuple(result.states),
        expect=expect,
        integrator=used.method,
        atol=used.atol,
        retries=tuple(retries),
    )


def tightened(options: Numerics, factor: float = 10.0) -> Numerics:
    """``options`` with both atol and rtol divided by ``factor``: the Section 5.5 convergence companion."""
    return replace(options, atol=options.atol / factor, rtol=options.rtol / factor)


@dataclass(frozen=True)
class ConvergenceReport:
    """One tolerance-convergence comparison (Section 5.5): the change per observable under the tighter pair."""

    tolerances: tuple[float, float]
    tightened_tolerances: tuple[float, float]
    changes: dict[str, float]
    """Per observable, max_t |p_tight(t) - p(t)|."""
    tol: float
    """The threshold the comparison is judged against."""

    @property
    def max_change(self) -> float:
        return max(self.changes.values()) if self.changes else 0.0

    @property
    def converged(self) -> bool:
        return self.max_change < self.tol

    def summary(self) -> str:
        worst = max(self.changes, key=lambda k: self.changes[k]) if self.changes else "-"
        return (
            f"tolerance convergence: atol/rtol {self.tolerances[0]:g}/{self.tolerances[1]:g} against "
            f"{self.tightened_tolerances[0]:g}/{self.tightened_tolerances[1]:g} moves the probabilities by at most "
            f"{self.max_change:.3g} (on {worst!r}), threshold {self.tol:g}: "
            f"{'converged' if self.converged else 'NOT CONVERGED'}"
        )


def convergence_check(
    run: Callable[[Numerics], Mapping[str, np.ndarray]],
    options: Numerics | None = None,
    *,
    factor: float = 10.0,
    tol: float = 1e-6,
) -> ConvergenceReport:
    """Run ``run`` at ``options`` and at ``tightened(options, factor)`` and report the change per observable; ``run``
    returns the observable traces of one integration."""
    if factor <= 1.0:
        raise ValueError("the tightening factor exceeds one")
    opts = options or Numerics()
    tight = tightened(opts, factor)
    a = run(opts)
    b = run(tight)
    if set(a) != set(b):
        raise ValueError("the two runs returned different observables")
    changes = {
        key: float(np.max(np.abs(np.asarray(b[key], dtype=float) - np.asarray(a[key], dtype=float))))
        for key in a
    }
    return ConvergenceReport((opts.atol, opts.rtol), (tight.atol, tight.rtol), changes, tol)
