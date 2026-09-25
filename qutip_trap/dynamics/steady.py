"""Direct steady states and exponential-series spectra without QuTiP's options context (PLAN.md Section 8.1).

``qutip.steadystate(H, c_ops, method="direct")`` wraps its solve in ``CoreOptions(default_dtype_scope="creation")``, and
entering or leaving that context rebuilds the dispatch table of every data-layer function (20 to 80 ms with the two data types
this package registers). When the setting already is "creation" the context changes nothing, so the functions here make the
same calls to QuTiP's solvers without it; a process that changed the setting gets QuTiP's own path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip as qt
from qutip.core import data as _data
from qutip.solver.spectrum import _diagonal_evolution
from qutip.solver.steadystate import _steadystate_direct


def steady_state_direct(H: qt.Qobj, c_ops: Sequence[qt.Qobj]) -> qt.Qobj:
    """``qutip.steadystate(H, c_ops, method="direct")`` for a Hamiltonian ``H`` and its collapse operators."""
    ops = list(c_ops)
    if qt.settings.core["default_dtype_scope"] != "creation":
        return qt.steadystate(H, ops, method="direct")
    if not ops:
        raise TypeError("Cannot calculate the steady state for a non-dissipative system.")
    rho: qt.Qobj = _steadystate_direct(qt.liouvillian(H, ops), 0, method=None)
    return rho


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
