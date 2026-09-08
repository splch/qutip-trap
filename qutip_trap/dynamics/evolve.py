"""Solver selection (sesolve/mesolve), step control and the integrator ladder (PLAN.md Section 5.3; milestone M2).

Pure states evolve with ``sesolve`` (``dop853`` by default, atol 1e-10, rtol 1e-8, nsteps 1e7), density matrices or any
Lindblad channel with ``mesolve`` (the reference path for small spaces; the drive operators are CSR so ``liouvillian``
stays sparse, Section 5.3). The escalation ladder is ``dop853`` -> ``vern9`` -> ``vern9`` at atol 1e-8 with a max_step
budget of one fourteenth of the fastest period, never a multistep method (BDF damps the oscillatory spectrum of -iH);
atol is keyed to the measured points, 1e-10 up to d_m ~ 100 and 1e-8 above (2026-09-04 numerics critique). The
integrator actually used, the retries and the right-hand-side evaluation count are recorded.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import qutip as qt
from qutip.solver.integrator import IntegratorException

from qutip_trap.dynamics.engine import SolverOptions

M2 = "milestone M2 (dynamics/evolve.py, PLAN.md Section 5.3)"

LARGE_MODE_DIMENSION = 100
"""Above this per-mode dimension the DEFAULT atol relaxes to ``LARGE_MODE_ATOL`` (dop853 aborts as 'probably stiff' at
d_m = 121 with atol 1e-10; Section 5.3, keyed to the measured points)."""
LARGE_MODE_ATOL = 1e-8
"""The relaxed atol above ``LARGE_MODE_DIMENSION``, applied only when the caller left ``SolverOptions.atol`` at its default:
a deliberate tightening (the Section 5.5 convergence run) is never clamped back up."""
_DEFAULT_ATOL = SolverOptions().atol
"""1e-10. The keying is on 'the caller did not choose an atol', which is what this comparison expresses."""


@dataclass(frozen=True)
class Evolution:
    times_s: np.ndarray
    final: qt.Qobj
    states: tuple[qt.Qobj, ...] | None
    expect: dict[str, np.ndarray]
    integrator: str
    atol: float
    rtol: float
    retries: tuple[str, ...]
    rhs_evaluations: int | None
    """Right-hand-side evaluations from the builder's coefficient counter (calls over calls_per_rhs; None without one)."""


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
    store_states: bool,
    propagator: bool = False,
) -> qt.solver.Result:
    options: dict[str, object] = {
        "method": method,
        "atol": atol,
        "rtol": rtol,
        "nsteps": nsteps,
        "store_final_state": True,
        "store_states": store_states,
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
            # an operator-valued "state" under sesolve integrates the propagator U(t) itself (Section 11.3 item 5)
            return qt.sesolve(H, state0, times, e_ops=eops, options=options)
        return qt.mesolve(H, state0, times, c_ops=list(c_ops), e_ops=eops, options=options)


def evolve(
    H: qt.QobjEvo | qt.Qobj,
    state0: qt.Qobj,
    times_s: Sequence[float] | np.ndarray,
    *,
    c_ops: Sequence[qt.Qobj] = (),
    e_ops: Mapping[str, qt.Qobj] | None = None,
    options: SolverOptions | None = None,
    store_states: bool = False,
    omega_max_rad_s: float | None = None,
    largest_mode_dimension: int | None = None,
    counter_calls: Callable[[], int] | None = None,
    calls_per_rhs: int = 1,
    propagator: bool = False,
) -> Evolution:
    """Integrate from ``times_s[0]`` to ``times_s[-1]`` through the ladder of Section 5.3, storing at every time.

    ``propagator=True`` integrates an operator-valued ``state0`` (the identity) under ``sesolve``, so that the stored states
    are the propagators U(t, t_0) of an internal-state-only space (Section 11.3 item 5; M9b)."""
    opts = options or SolverOptions()
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be an increasing array with at least two points")
    atol = opts.atol
    if (
        largest_mode_dimension is not None
        and largest_mode_dimension > LARGE_MODE_DIMENSION
        and opts.atol == _DEFAULT_ATOL
    ):
        # the keying of Section 5.3 relaxes the DEFAULT atol above d_m ~ 100 (dop853 aborts as "probably stiff" at
        # d_m = 121 with 1e-10). It must not clamp a tolerance the caller chose deliberately: max(atol, 1e-8) turned a
        # Section 5.5 tolerance-convergence run at d_m > 100 into an rtol-only tightening, so the convergence check
        # reported a spuriously small change (M2 audit E9/E10).
        atol = LARGE_MODE_ATOL
    ladder: list[tuple[str, float, float]] = [(m, atol, 0.0) for m in opts.integrators]
    last = opts.integrators[-1]
    max_step = (2.0 * math.pi / omega_max_rad_s / 14.0) if omega_max_rad_s else 0.0
    ladder.append((last, max(atol, 1e-8), max_step))
    retries: list[str] = []
    calls0 = counter_calls() if counter_calls is not None else 0
    result = None
    used = ladder[-1]
    for method, a, ms in ladder:
        try:
            result = _solve(
                H,
                state0,
                times,
                c_ops,
                e_ops,
                method,
                a,
                opts.rtol,
                opts.nsteps,
                ms,
                store_states,
                propagator,
            )
            used = (method, a, ms)
            break
        except IntegratorException as exc:
            # the INTEGRATOR's own failure ("probably stiff", "excess work", "larger nsteps is needed"): escalate and
            # report. Anything else propagates: a bare `except Exception` here recorded a TypeError in a coefficient or a
            # KeyError from an unsupported solver option as an integrator failure, tried every remaining rung on the same
            # broken input and ended in "every rung of the integrator ladder failed", hiding the programming error the
            # plan never asked the ladder to rescue (M9b audit B8)
            retries.append(f"{method}@atol={a:g},max_step={ms:g}: {type(exc).__name__}: {exc}")
            continue
    if result is None:
        raise RuntimeError("every rung of the integrator ladder failed: " + " | ".join(retries))
    expect: dict[str, np.ndarray] = {}
    if e_ops:
        for key, arr in zip(e_ops.keys(), result.expect):
            expect[str(key)] = np.asarray(arr)
    states = tuple(result.states) if store_states else None
    evals = (counter_calls() - calls0) // max(calls_per_rhs, 1) if counter_calls is not None else None
    return Evolution(
        times_s=times,
        final=result.final_state,
        states=states,
        expect=expect,
        integrator=used[0],
        atol=used[1],
        rtol=opts.rtol,
        retries=tuple(retries),
        rhs_evaluations=evals,
    )


def tightened(options: SolverOptions, factor: float = 10.0) -> SolverOptions:
    """The Section 5.5 tolerance-convergence companion: atol and rtol divided by ``factor``.

    Both tolerances move, which is the point: above ``LARGE_MODE_DIMENSION`` the atol keying used to clamp the tightened
    atol back to 1e-8 and leave rtol alone, so the comparison measured an rtol-only change (M2 audit E9).
    """
    return replace(options, atol=options.atol / factor, rtol=options.rtol / factor)


@dataclass(frozen=True)
class ConvergenceReport:
    """The Section 5.5 / 9.9 tolerance-convergence comparison of one run (Section 14.5's convergence badge)."""

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
    run: Callable[[SolverOptions], Mapping[str, np.ndarray]],
    options: SolverOptions | None = None,
    *,
    factor: float = 10.0,
    tol: float = 1e-6,
) -> ConvergenceReport:
    """Run ``run`` at ``options`` and again at ``tightened(options, factor)``, and report the change per observable.

    Section 5.5 says the test suite does this for EVERY validation case; ``hilbert.truncation.halving_test`` is the
    single-array form behind the convergence badge and this is the named-observable form a run path can report
    (Section 9.9). ``run`` returns the observable traces of one integration, keyed as the engine keys them.
    """
    if factor <= 1.0:
        raise ValueError("the tightening factor exceeds one")
    opts = options or SolverOptions()
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


__all__ = [
    "LARGE_MODE_ATOL",
    "LARGE_MODE_DIMENSION",
    "ConvergenceReport",
    "Evolution",
    "convergence_check",
    "evolve",
    "tightened",
]
