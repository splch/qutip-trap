"""The boundary-population monitor, cap growth and the convergence regime (PLAN.md Section 5.5).

The monitor reads the population in the top two Fock levels of each resolved mode (the top shell of an ENR group); the
engine raises a cap and repeats the run when it exceeds the threshold, and ``regrid_state`` carries the state onto the
grown space.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import qutip as qt

from qutip_trap.dynamics.engine import State
from qutip_trap.dynamics.space import HilbertSpace


class TruncationWarning(UserWarning):
    """A truncation the run could not make exact: a cap clamped by ``Numerics.mode_dimension_max``, or a boundary
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
