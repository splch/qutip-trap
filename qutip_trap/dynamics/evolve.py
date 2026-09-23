"""Solver selection (sesolve/mesolve) and the integrator ladder: ``SolverOptions.integrators``, then the last one at
atol >= 1e-8 with max_step one fourteenth of the fastest period; never a multistep method (BDF damps the oscillatory
spectrum of -iH)."""

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
"""Above this per-mode dimension the default atol relaxes to ``LARGE_MODE_ATOL`` (dop853 aborts as 'probably stiff' at
d_m = 121 with atol 1e-10)."""
LARGE_MODE_ATOL = 1e-8
"""The relaxed atol above ``LARGE_MODE_DIMENSION``, applied only when ``SolverOptions.atol`` is at its default."""
_DEFAULT_ATOL = SolverOptions().atol


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
    """Right-hand-side evaluations read from the builder's coefficient counter (None without one)."""


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
) -> tuple[qt.solver.Result, int]:
    """The QuTiP result and the coefficient calls one right-hand-side evaluation makes per coefficient-bearing element of
    ``H`` (1 under ``sesolve``; 2 under ``mesolve``, whose Liouvillian holds each drive term as spre and spost)."""
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
            # an operator-valued "state" under sesolve integrates the propagator U(t) itself
            return qt.sesolve(H, state0, times, e_ops=eops, options=options), 1
        solver = qt.MESolver(H, c_ops=list(c_ops), options=options)
        return solver.run(state0, times, e_ops=eops), _calls_per_element(solver.rhs, H)


def _coefficient_elements(op: qt.QobjEvo | qt.Qobj) -> int:
    if not isinstance(op, qt.QobjEvo):
        return 0
    return sum(1 for el in op.to_list() if isinstance(el, list))


def _calls_per_element(rhs: qt.QobjEvo, H: qt.QobjEvo | qt.Qobj) -> int:
    """Coefficient-bearing elements of the solver's right-hand side per element of ``H`` (2 on the mesolve path)."""
    n_h = _coefficient_elements(H)
    if n_h == 0:
        return 1
    return max(1, round(_coefficient_elements(rhs) / n_h))


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
    """Integrate over ``times_s`` through the ladder; ``omega_max_rad_s`` sets the last rung's max_step and
    ``largest_mode_dimension`` keys the default atol. ``propagator=True`` evolves an operator ``state0`` (the identity)
    under ``sesolve``, so the stored states are the propagators U(t, t_0)."""
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
        # only the default: clamping a chosen atol would turn a convergence run into an rtol-only tightening
        atol = LARGE_MODE_ATOL
    ladder: list[tuple[str, float, float]] = [(m, atol, 0.0) for m in opts.integrators]
    last = opts.integrators[-1]
    max_step = (2.0 * math.pi / omega_max_rad_s / 14.0) if omega_max_rad_s else 0.0
    ladder.append((last, max(atol, 1e-8), max_step))
    retries: list[str] = []
    calls0 = counter_calls() if counter_calls is not None else 0
    result = None
    per_element = 1
    used = ladder[-1]
    for method, a, ms in ladder:
        try:
            result, per_element = _solve(
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
            # only the integrator's own failures escalate; any other exception is a bug and propagates
            retries.append(f"{method}@atol={a:g},max_step={ms:g}: {type(exc).__name__}: {exc}")
            continue
    if result is None:
        raise RuntimeError("every rung of the integrator ladder failed: " + " | ".join(retries))
    expect: dict[str, np.ndarray] = {}
    if e_ops:
        for key, arr in zip(e_ops.keys(), result.expect):
            expect[str(key)] = np.asarray(arr)
    states = tuple(result.states) if store_states else None
    evals = (
        (counter_calls() - calls0) // max(calls_per_rhs * per_element, 1)
        if counter_calls is not None
        else None
    )
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
    """The tolerance-convergence companion: atol and rtol both divided by ``factor``."""
    return replace(options, atol=options.atol / factor, rtol=options.rtol / factor)


@dataclass(frozen=True)
class ConvergenceReport:
    """The tolerance-convergence comparison of one run: (atol, rtol) against the tightened pair."""

    tolerances: tuple[float, float]
    tightened_tolerances: tuple[float, float]
    changes: dict[str, float]
    """Per observable, max_t |p_tight(t) - p(t)|."""
    tol: float
    """The threshold ``max_change`` must stay below."""

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
    """Run ``run`` (options -> observable traces by name) at ``options`` and at ``tightened(options, factor)``, and report
    the change per observable."""
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
