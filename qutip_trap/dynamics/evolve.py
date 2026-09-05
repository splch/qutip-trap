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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import qutip as qt

from qutip_trap.dynamics.engine import SolverOptions

M2 = "milestone M2 (dynamics/evolve.py, PLAN.md Section 5.3)"

LARGE_MODE_DIMENSION = 100
"""Above this per-mode dimension atol relaxes to 1e-8 (dop853 aborts as 'probably stiff' at d_m = 121 with 1e-10)."""


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
    if not c_ops and state0.isket:
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
) -> Evolution:
    """Integrate from ``times_s[0]`` to ``times_s[-1]`` through the ladder of Section 5.3, storing at every time."""
    opts = options or SolverOptions()
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be an increasing array with at least two points")
    atol = opts.atol
    if largest_mode_dimension is not None and largest_mode_dimension > LARGE_MODE_DIMENSION:
        atol = max(atol, 1e-8)
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
                H, state0, times, c_ops, e_ops, method, a, opts.rtol, opts.nsteps, ms, store_states
            )
            used = (method, a, ms)
            break
        except (
            Exception
        ) as exc:  # the integrator's own failure ("probably stiff", "excess work"): escalate, report
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
    """The Section 5.5 tolerance-convergence companion: atol and rtol divided by ``factor``."""
    return replace(options, atol=options.atol / factor, rtol=options.rtol / factor)


__all__ = ["LARGE_MODE_DIMENSION", "Evolution", "evolve", "tightened"]
