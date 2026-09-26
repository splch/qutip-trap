"""Direct steady states and exponential-series spectra without QuTiP's options context (PLAN.md Section 8.1).

``qutip.steadystate(H, c_ops, method="direct")`` wraps its solve in ``CoreOptions(default_dtype_scope="creation")``, and
entering or leaving that context rebuilds the dispatch table of every data-layer function (20 to 80 ms with the two data types
this package registers). When the setting already is "creation" the context changes nothing, so the functions here make the
same calls to QuTiP's solvers without it; a process that changed the setting gets QuTiP's own path.

The direct solve needs a one-dimensional null space. States that no beam or decay leads out of (a closed class of the
basis states, ``closed_classes``) each carry a steady state of their own, and with two or more the solve is singular and
its answer is whatever the LAPACK build returns; ``steady_state_reached`` then takes the state the evolution from a given
initial state settles into.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip as qt
from qutip.core import data as _data
from qutip.solver.spectrum import _diagonal_evolution
from qutip.solver.steadystate import _steadystate_direct
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def steady_state_direct(H: qt.Qobj, c_ops: Sequence[qt.Qobj]) -> qt.Qobj:
    """``qutip.steadystate(H, c_ops, method="direct")`` for a Hamiltonian ``H`` and its collapse operators."""
    ops = list(c_ops)
    if qt.settings.core["default_dtype_scope"] != "creation":
        return qt.steadystate(H, ops, method="direct")
    if not ops:
        raise TypeError("Cannot calculate the steady state for a non-dissipative system.")
    rho: qt.Qobj = _steadystate_direct(qt.liouvillian(H, ops), 0, method=None)
    return rho


COUPLING_RTOL = 1e-12
"""An element of H or of a collapse operator below this fraction of that operator's largest element couples nothing."""
NULL_RTOL = 1e-12
"""Singular values of a Liouvillian below this fraction of the largest span its null space."""
REACHED_DIMENSION_MAX = 1600
"""The largest Liouvillian (m^2 for the m basis states reachable from the initial state) ``steady_state_reached``
decomposes densely."""


def coupling_graph(H: qt.Qobj, c_ops: Sequence[qt.Qobj]) -> np.ndarray:
    """edges[i, j]: H_ij or a collapse operator's <j|C|i> exceeds ``COUPLING_RTOL`` of that operator's largest element, so
    amplitude or population moves from basis state i to basis state j."""
    h = np.abs(np.asarray(H.full()))
    np.fill_diagonal(h, 0.0)
    edges = h > COUPLING_RTOL * float(h.max(initial=0.0))
    for c in c_ops:
        a = np.abs(np.asarray(c.full()))
        edges |= (a > COUPLING_RTOL * float(a.max(initial=0.0))).T
    np.fill_diagonal(edges, False)
    return edges


def closed_classes(H: qt.Qobj, c_ops: Sequence[qt.Qobj]) -> tuple[tuple[int, ...], ...]:
    """The closed classes of the basis states, in ascending order: the strongly connected sets of ``coupling_graph`` that
    no edge leaves. Each carries a steady state of its own, so two or more make the steady state depend on where the
    evolution starts."""
    edges = coupling_graph(H, c_ops)
    n_sets, labels = connected_components(csr_matrix(edges), directed=True, connection="strong")
    rows, cols = np.nonzero(edges)
    open_sets = set(labels[rows[labels[rows] != labels[cols]]].tolist())
    closed = (tuple(int(i) for i in np.flatnonzero(labels == k)) for k in range(n_sets) if k not in open_sets)
    return tuple(sorted(closed))


def steady_state_reached(H: qt.Qobj, c_ops: Sequence[qt.Qobj], rho0: qt.Qobj) -> qt.Qobj:
    """The steady state the evolution from ``rho0`` settles into, on the basis states ``coupling_graph`` reaches from
    rho0's (the others stay empty): rho0 projected onto the null space of their Liouvillian L along its range,
    V (U^dag V)^{-1} U^dag rho0 with V and U the right and left null vectors of a dense SVD, which is the long-time average
    of e^{L t} rho0 (the zero eigenvalue of a Lindblad generator is semisimple, so U^dag V is invertible). For a unique
    steady state it is the direct solve's; a Liouvillian above ``REACHED_DIMENSION_MAX`` is refused."""
    edges = coupling_graph(H, c_ops)
    r0 = np.asarray(rho0.full())
    reached = np.abs(r0).sum(axis=0) + np.abs(r0).sum(axis=1) > 0.0
    frontier = reached.copy()
    while frontier.any():
        frontier = edges[frontier].any(axis=0) & ~reached
        reached |= frontier
    keep = np.flatnonzero(reached)
    m = keep.size
    if m * m > REACHED_DIMENSION_MAX:
        raise NotImplementedError(
            f"the steady state reached from an initial state is decomposed densely, up to a Liouvillian of dimension "
            f"{REACHED_DIMENSION_MAX}; the {m} states the initial state reaches make one of {m * m}"
        )
    block = np.ix_(keep, keep)
    L = np.asarray(
        qt.liouvillian(
            qt.Qobj(np.asarray(H.full())[block]), [qt.Qobj(np.asarray(c.full())[block]) for c in c_ops]
        ).full()
    )
    u, s, vh = np.linalg.svd(L)
    k = int(np.count_nonzero(s <= NULL_RTOL * s[0]))
    right = vh[m * m - k :].conj().T
    left = u[:, m * m - k :]
    x0 = r0[block].reshape(-1, order="F")
    x = right @ np.linalg.solve(left.conj().T @ right, left.conj().T @ x0)
    sub = x.reshape(m, m, order="F")
    rho = np.zeros_like(r0, dtype=complex)
    rho[block] = 0.5 * (sub + sub.conj().T)
    return qt.Qobj(rho / np.trace(rho).real, dims=H.dims)


def spectrum_es(
    H: qt.Qobj,
    c_ops: Sequence[qt.Qobj],
    wlist: Sequence[float] | np.ndarray,
    a_op: qt.Qobj,
    b_op: qt.Qobj,
    rho_ss: qt.Qobj,
) -> np.ndarray:
    """``qutip.spectrum(H, wlist, c_ops, a_op, b_op, solver="es")`` with the steady state ``rho_ss`` of the same ``H`` and
    ``c_ops`` supplied instead of recomputed; the exponential series, the amplitude tidy-up at ``settings.core["atol"]`` and
    the Lorentzian sum are QuTiP's ``_spectrum_es``."""
    w = np.asarray(wlist, dtype=float)
    L = qt.liouvillian(H, list(c_ops))
    a_op_ss = qt.expect(a_op, rho_ss)
    b_op_ss = qt.expect(b_op, rho_ss)
    states, rates = _diagonal_evolution(L, b_op * rho_ss)
    ampls = [_data.expect(a_op.data, state) for state in states]
    ampls += [-a_op_ss * b_op_ss]
    rates += [0]
    atol = qt.settings.core["atol"]
    order = np.argsort(rates)
    clean_rates: list[complex] = []
    clean_ampls: list[complex] = []
    prev_rate = np.nan
    for idx in order:
        if np.abs(rates[idx] - prev_rate) < atol:
            clean_ampls[-1] += ampls[idx]
        else:
            clean_rates.append(rates[idx])
            clean_ampls.append(ampls[idx])
            prev_rate = rates[idx]
    kept = [(rate, ampl) for rate, ampl in zip(clean_rates, clean_ampls) if np.abs(ampl) > atol]
    rates_arr = np.array([r for r, _a in kept])
    ampls_arr = np.array([a for _r, a in kept])
    lw = np.subtract.outer(1j * w, rates_arr).T
    out: np.ndarray = (ampls_arr @ (2 / lw)).real
    return out
