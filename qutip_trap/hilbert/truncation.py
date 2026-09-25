"""The boundary-population monitor, cap growth and the convergence regime (PLAN.md Section 5.5).

The monitor reads the population in the top two Fock levels of each resolved mode (the top shell of an ENR group); the
engine raises a cap and repeats the run when it exceeds the threshold, and ``regrid_state`` carries the state onto the
grown space.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import qutip as qt

from qutip_trap.dynamics.engine import SolverOptions, State
from qutip_trap.dynamics.evolve import ConvergenceReport, convergence_check
from qutip_trap.hilbert.space import HilbertSpace


class TruncationWarning(UserWarning):
    """A truncation the run could not make exact: a cap clamped by ``SolverOptions.mode_dimension_max``, or a boundary
    population above ``boundary_population_max`` after the cap-raising retries."""


def warn_cap_clamped(mode: int, d_wanted: int, n_hi_wanted: int, d: int, d_max: int) -> None:
    """The :class:`TruncationWarning` of a cap clamped by ``mode_dimension_max``, attributed to the selection's caller."""
    warnings.warn(
        TruncationWarning(
            f"mode {mode}: the cap rule asks for d = {d_wanted} (expected occupation up to n = {n_hi_wanted}) but "
            f"mode_dimension_max = {d_max} clamps it to d = {d}; the boundary monitor grows the cap only up to the engine's "
            "retry budget, and the Diagnostics report the clamped range"
        ),
        stacklevel=3,
    )


def warn_if_boundary_exceeds(boundary: Mapping[int, float], threshold: float) -> None:
    """One :class:`TruncationWarning` per mode whose boundary population exceeds ``threshold`` after the retries."""
    for mode, value in sorted(boundary.items()):
        if value > threshold:
            warnings.warn(
                TruncationWarning(
                    f"mode {mode}: boundary population {value:.3e} exceeds boundary_population_max = {threshold:.1e} after "
                    "the cap-raising retries (a branch of weight w is allowed boundary_population_max / w, so a "
                    "low-weight branch can report this legitimately); Diagnostics.boundary_population carries it"
                ),
                stacklevel=3,
            )


def boundary_population(state: State | qt.Qobj, space: HilbertSpace, mode: int) -> float:
    """Population in the top two Fock levels of a resolved mode, or in the top ENR shell."""
    cls = space.mode_class(mode)
    if cls == "resolved":
        p = space.fock_populations(state, mode)
        return float(np.sum(p[-2:]))
    if cls == "enr":
        assert space.enr_factor is not None
        sigma = np.real(np.diag(np.asarray(space.marginal(state, (space.enr_factor,)).full())))
        dims, n_exc = space._enr_dims()
        _n, _s2i, idx2state = qt.enr_state_dictionaries(dims, n_exc)
        top = [i for i in range(len(idx2state)) if sum(idx2state[i]) == n_exc]
        return float(np.sum(sigma[top]))
    raise KeyError(f"mode {mode} is frozen and has no boundary")


def boundary_populations(state: State | qt.Qobj, space: HilbertSpace) -> dict[int, float]:
    """``boundary_population`` of every resolved mode and ENR member (the members share the group's top shell)."""
    out = {m.mode: boundary_population(state, space, m.mode) for m in space.resolved}
    if space.enr_group is not None:
        top = boundary_population(state, space, space.enr_group[0][0])
        for m in space.enr_group[0]:
            out[m] = top
    return out


def grown_caps(space: HilbertSpace, add: int = 2, *, dimension_max: int | None = None) -> HilbertSpace:
    """Every resolved cap (and the ENR excitation cap) raised by ``add``; with ``dimension_max`` a mode whose growth would
    cross the ceiling is left at its cap."""
    if add <= 0:
        raise ValueError("grow by a positive number of levels")
    out = space
    for m in space.resolved:
        candidate = out.grown(m.mode, add)
        if dimension_max is not None and candidate.dimension > dimension_max:
            continue
        out = candidate
    if space.enr_group is not None:
        candidate = out.grown_enr(add)
        if dimension_max is None or candidate.dimension <= dimension_max:
            out = candidate
    return out


@dataclass(frozen=True)
class ConvergenceRegime:
    """The three convergence comparisons of a validation case: atol and rtol divided by ``factor`` (``tightened``),
    multiplied by it (``loosened``), and every cap raised by ``add`` at unchanged tolerances (``caps``)."""

    tightened: ConvergenceReport
    loosened: ConvergenceReport
    caps: ConvergenceReport
    grown_modes: tuple[int, ...]
    """The resolved modes whose caps were raised."""
    add: int

    @property
    def max_change(self) -> float:
        return max(self.tightened.max_change, self.loosened.max_change, self.caps.max_change)

    @property
    def converged(self) -> bool:
        return self.tightened.converged and self.loosened.converged and self.caps.converged

    def summary(self) -> str:
        return (
            f"convergence: tolerances tightened move the probabilities by "
            f"{self.tightened.max_change:.3g}, loosened by {self.loosened.max_change:.3g}, caps +{self.add} on modes "
            f"{list(self.grown_modes)} by {self.caps.max_change:.3g}; threshold {self.tightened.tol:g}: "
            f"{'converged' if self.converged else 'NOT CONVERGED'}"
        )


def convergence_report(
    run: Callable[[SolverOptions, HilbertSpace], Mapping[str, np.ndarray]],
    options: SolverOptions,
    space: HilbertSpace,
    *,
    factor: float = 10.0,
    add: int = 2,
    tol: float = 1e-6,
) -> ConvergenceRegime:
    """The convergence regime of one case: ``run(options, space)`` returns its probabilities by observable.

    The loosened arm is ``convergence_check`` started from ``options`` scaled up by ``factor``; the cap arm runs on
    ``grown_caps(space, add)`` within ``options.joint_dimension_max`` (a case with a fixed joint state regrids it itself).
    """
    if factor <= 1.0:
        raise ValueError("the tightening factor exceeds one")
    loose = replace(options, atol=options.atol * factor, rtol=options.rtol * factor)
    tightened_rep = convergence_check(lambda o: run(o, space), options, factor=factor, tol=tol)
    loosened_rep = convergence_check(lambda o: run(o, space), loose, factor=factor, tol=tol)
    grown = grown_caps(space, add, dimension_max=options.joint_dimension_max)
    a = run(options, space)
    b = a if grown == space else run(options, grown)
    if set(a) != set(b):
        raise ValueError("the two runs returned different observables")
    changes = {
        key: float(np.max(np.abs(np.asarray(b[key], dtype=float) - np.asarray(a[key], dtype=float))))
        for key in a
    }
    caps_rep = ConvergenceReport((options.atol, options.rtol), (options.atol, options.rtol), changes, tol)
    return ConvergenceRegime(
        tightened=tightened_rep,
        loosened=loosened_rep,
        caps=caps_rep,
        grown_modes=tuple(m.mode for m, g in zip(space.resolved, grown.resolved) if g.d > m.d),
        add=add,
    )


def regrid_state(joint: qt.Qobj, old: HilbertSpace, new: HilbertSpace) -> qt.Qobj:
    """A joint state of ``old`` embedded into ``new``, a copy of it with larger caps: zero-padded Fock factors, and for an
    ENR group every old Fock tuple mapped to its index in the larger group."""
    if old.ion_dims != new.ion_dims or old.frozen != new.frozen or old.ion_labels != new.ion_labels:
        raise ValueError("regrid_state changes the motional caps only")
    if [m.mode for m in old.resolved] != [m.mode for m in new.resolved]:
        raise ValueError("regrid_state keeps the resolved modes and their order")
    if (old.enr_group is None) != (new.enr_group is None) or (
        old.enr_group is not None and new.enr_group is not None and old.enr_group[0] != new.enr_group[0]
    ):
        raise ValueError("regrid_state keeps the ENR group's modes")
    old_dims, new_dims = old.dims, new.dims
    if any(n < o for o, n in zip(old_dims, new_dims)):
        raise ValueError("caps may only grow")
    if joint.shape[0] != old.dimension:
        raise ValueError("the state does not live on the old space")
    maps: list[np.ndarray] = [np.arange(d) for d in old_dims]
    if old.enr_group is not None and new.enr_group is not None:
        f = old.enr_factor
        assert f is not None
        dims_o, n_o = old._enr_dims()
        dims_n, n_n = new._enr_dims()
        _n_old, _s2i_old, i2s_old = qt.enr_state_dictionaries(dims_o, n_o)
        _n_new, s2i_new, _i2s_new = qt.enr_state_dictionaries(dims_n, n_n)
        maps[f] = np.array([s2i_new[tuple(i2s_old[i])] for i in range(len(i2s_old))], dtype=int)
    if joint.isket:
        arr = np.asarray(joint.full()).reshape(old_dims)
        out = np.zeros(new_dims, dtype=complex)
        out[np.ix_(*maps)] = arr
        return qt.Qobj(out.reshape(-1, 1), dims=[new_dims, [1] * len(new_dims)])
    arr = np.asarray(joint.full()).reshape(old_dims + old_dims)
    out = np.zeros(new_dims + new_dims, dtype=complex)
    out[np.ix_(*(maps + maps))] = arr
    size = int(np.prod(new_dims))
    return qt.Qobj(out.reshape(size, size), dims=[new_dims, new_dims])
