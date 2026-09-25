"""Truncation policy: n_max caps, the boundary-population monitor and adaptive growth (PLAN.md Section 5.5; M2, M9a).

The monitor records, per mode and per pulse, the population in the top two Fock levels (or the top ENR shell);
above the threshold (default 1e-6 of the population the pulse moves, with per-test overrides for hot modes) or when
the cap's margin above the populated range falls below the Section 5.1.1 margin for the pulse's eta, the run is
repeated with the cap raised, up to a configured limit, and both numbers are reported (Section 5.5). Tolerance
convergence repeats a run with tolerances tightened by a factor of 10 and reports the change in final
probabilities; the test suite does this for every validation case (Section 5.5).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import qutip as qt

from qutip_trap.dynamics.engine import SolverOptions, State
from qutip_trap.dynamics.evolve import ConvergenceReport
from qutip_trap.hilbert.space import HilbertSpace

M2 = "milestone M2 (hilbert/truncation.py, PLAN.md Section 5.5)"


class TruncationError(RuntimeError):
    """The boundary monitor tripped and the configured growth limit was reached (Section 5.5)."""


class TruncationWarning(UserWarning):
    """A truncation the run could not make exact: a cap the rule wanted larger than ``SolverOptions.mode_dimension_max``
    allowed, or a boundary population above ``boundary_population_max`` after the cap-raising retries. The warning names the
    mode and both numbers, so silence means the caps held."""


def warn_cap_clamped(mode: int, d_wanted: int, n_hi_wanted: int, d: int, d_max: int) -> None:
    """Issue the :class:`TruncationWarning` of a cap clamped by ``mode_dimension_max`` (``run.space.select_space`` and the
    GATE_LOCAL step spaces), attributed to the caller of the selection."""
    warnings.warn(
        TruncationWarning(
            f"mode {mode}: the cap rule asks for d = {d_wanted} (expected occupation up to n = {n_hi_wanted}) but "
            f"mode_dimension_max = {d_max} clamps it to d = {d}; the boundary monitor grows the cap only up to the engine's "
            "retry budget (Section 5.5), and the Diagnostics report the clamped range"
        ),
        stacklevel=3,
    )


def warn_if_boundary_exceeds(boundary: Mapping[int, float], threshold: float) -> None:
    """Issue one :class:`TruncationWarning` per mode whose reported boundary population exceeds ``threshold``
    (``SolverOptions.boundary_population_max``) after the run's cap-raising retries; nothing is issued when every mode is
    below it, so a silent run kept its caps."""
    for mode, value in sorted(boundary.items()):
        if value > threshold:
            warnings.warn(
                TruncationWarning(
                    f"mode {mode}: boundary population {value:.3e} exceeds boundary_population_max = {threshold:.1e} after "
                    "the cap-raising retries (Section 5.5; a branch of weight w is allowed boundary_population_max / w, so a "
                    "low-weight branch can report this legitimately); Diagnostics.boundary_population carries it"
                ),
                stacklevel=3,
            )


def boundary_population(state: State | qt.Qobj, space: HilbertSpace, mode: int, *, levels: int = 2) -> float:
    """Population in the top ``levels`` Fock levels of a resolved mode, or in the top ENR shell (n_total = N_exc)."""
    cls = space.mode_class(mode)
    if cls == "resolved":
        p = space.fock_populations(state, mode)
        return float(np.sum(p[-levels:]))
    if cls == "enr":
        assert space.enr_factor is not None
        sigma = np.real(np.diag(np.asarray(space.marginal(state, (space.enr_factor,)).full())))
        dims, n_exc = space._enr_dims()
        _n, _s2i, idx2state = qt.enr_state_dictionaries(dims, n_exc)
        top = [i for i in range(len(idx2state)) if sum(idx2state[i]) == n_exc]
        return float(np.sum(sigma[top]))
    raise KeyError(f"mode {mode} is frozen and has no boundary")


def boundary_populations(state: State | qt.Qobj, space: HilbertSpace, *, levels: int = 2) -> dict[int, float]:
    out = {m.mode: boundary_population(state, space, m.mode, levels=levels) for m in space.resolved}
    if space.enr_group is not None:
        top = boundary_population(state, space, space.enr_group[0][0], levels=levels)
        for m in space.enr_group[0]:
            out[m] = top
    return out


@dataclass(frozen=True)
class MarginReport:
    mode: int
    margin_levels: int
    required_levels: int

    @property
    def ok(self) -> bool:
        return self.margin_levels >= self.required_levels

    @property
    def deficit(self) -> int:
        return max(self.required_levels - self.margin_levels, 0)


def margin_reports(space: HilbertSpace, etas: Mapping[int, float] | None = None) -> tuple[MarginReport, ...]:
    """The Section 5.1.1 margin check per resolved mode for the given |eta| (default: the declared eta_max)."""
    deficits = space.margin_deficits(etas)
    return tuple(
        MarginReport(m.mode, m.margin_levels, m.margin_levels + deficits[m.mode]) for m in space.resolved
    )


def grow_for_margins(space: HilbertSpace, etas: Mapping[int, float] | None = None) -> HilbertSpace:
    """Raise every resolved cap whose margin is below the Section 5.1.1 requirement (no-op when all margins hold)."""
    out = space
    for rep in margin_reports(space, etas):
        if not rep.ok:
            out = out.grown(rep.mode, rep.deficit)
    return out


def grow_for_boundary(
    space: HilbertSpace, boundary: Mapping[int, float], threshold: float, *, add: int = 4
) -> HilbertSpace:
    """Raise the caps of the modes whose boundary population exceeds ``threshold`` by ``add`` levels each."""
    out = space
    for m in space.resolved:
        if boundary.get(m.mode, 0.0) > threshold:
            out = out.grown(m.mode, add)
    return out


def halving_test(
    run: Callable[[SolverOptions], np.ndarray],
    options: SolverOptions,
    *,
    factor: float = 10.0,
    tol: float = 1e-6,
) -> tuple[bool, float, SolverOptions]:
    """The tolerance-convergence test of Section 5.5 behind the convergence badge (Section 14.5).

    Runs ``run`` at ``options`` and again with atol and rtol tightened by ``factor`` (10 by default; the name keeps
    the plan's historical "halving" label for the step-density halving it generalizes) and returns (converged,
    max |delta probability|, the tightened options) with converged = the change is below ``tol``.
    """
    if factor <= 1.0:
        raise ValueError("the tightening factor exceeds one")
    tight = replace(options, atol=options.atol / factor, rtol=options.rtol / factor)
    p0 = np.asarray(run(options), dtype=float)
    p1 = np.asarray(run(tight), dtype=float)
    if p0.shape != p1.shape:
        raise ValueError("the two runs returned probabilities of different shapes")
    delta = float(np.max(np.abs(p0 - p1))) if p0.size else 0.0
    return delta < tol, delta, tight


def grown_caps(space: HilbertSpace, add: int = 2, *, dimension_max: int | None = None) -> HilbertSpace:
    """Every resolved cap of ``space`` (and the ENR group's excitation cap) raised by ``add``: the truncation half of Section
    9.9's convergence regime.

    Section 9.9 asks for the comparison "where the doubled cap stays within the dimension ceiling; otherwise the check runs
    on the reduced mode set", so with ``dimension_max`` the modes are grown one at a time in resolved order and a mode whose
    growth would cross the ceiling is left at its cap.
    """
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
    """The three comparisons Section 9.9 asks of every validation case, each one a ``dynamics.evolve.ConvergenceReport``.

    - ``tightened``: atol and rtol divided by ``factor`` (Section 5.5's second bullet, the run-level check
      ``SolverOptions.convergence_check`` reports on its own);
    - ``loosened``: both multiplied by ``factor``, which Section 9.9 asks for "for the integrator ladder";
    - ``caps``: every resolved cap and the ENR excitation cap raised by ``add`` at UNCHANGED tolerances, so its report's
      ``tolerances`` and ``tightened_tolerances`` are equal and ``grown_modes`` names what moved instead.
    """

    tightened: ConvergenceReport
    loosened: ConvergenceReport
    caps: ConvergenceReport
    grown_modes: tuple[int, ...]
    """The resolved modes whose caps were actually raised (a mode whose growth would cross the dimension ceiling is left)."""
    add: int

    @property
    def max_change(self) -> float:
        return max(self.tightened.max_change, self.loosened.max_change, self.caps.max_change)

    @property
    def converged(self) -> bool:
        return self.tightened.converged and self.loosened.converged and self.caps.converged

    def summary(self) -> str:
        return (
            f"Section 9.9 convergence: tolerances tightened move the probabilities by "
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
    """Section 9.9's convergence regime for one validation case, all three arms through
    :class:`~qutip_trap.dynamics.evolve.ConvergenceReport`.

    ``run(options, space)`` returns the case's reported probabilities keyed by observable, as the engine keys them. The
    tolerance arms are ``dynamics.evolve.convergence_check`` on a closure that holds the space fixed; the loosened arm is
    that same comparison started from ``options`` scaled UP by ``factor``, so its pair is (loose, options) and |delta p| is
    the same number the plan asks for. The cap arm raises the caps at fixed tolerances (``grown_caps``); a case that carries
    a fixed joint state regrids it inside its own closure (``regrid_state(state, space, grown)``), which is what makes that
    arm comparable at all.

    ``halving_test`` is the single-array form of the first comparison; this is the whole Section 9.9 row, and the tests that
    call it carry the ``convergence`` marker so the CI convergence-report artifact means something (Sections 5.5, 14.5).
    """
    from qutip_trap.dynamics.evolve import convergence_check

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
        grown_modes=tuple(
            m.mode
            for m, g in zip(space.resolved, grown.resolved)
            if g.d > m.d  # a mode left at its cap by the dimension ceiling is not part of the comparison
        ),
        add=add,
    )


def regrid_state(joint: qt.Qobj, old: HilbertSpace, new: HilbertSpace) -> qt.Qobj:
    """Embed a joint state of ``old`` into ``new``, a copy of it with larger caps: zero-padded Fock factors for the resolved
    modes, and for an ENR group a larger excitation cap, every old Fock tuple mapped to its index in the new group's
    dictionary (``enr_state_dictionaries``; M9a). The caps may only grow; ions, frozen modes and the ENR modes are unchanged.
    """
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
