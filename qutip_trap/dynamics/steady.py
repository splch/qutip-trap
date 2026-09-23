"""Direct steady states and exponential-series spectra without QuTiP's options context.

``qutip.steadystate(method="direct")`` wraps its solve in ``CoreOptions(default_dtype_scope="creation")``, whose entry and
exit rebuild every data-layer dispatch table (tens of ms, twice per call). When the setting already is "creation" (the
default) the context changes nothing, so these functions run QuTiP's own steps without it; otherwise they call QuTiP.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip as qt
from qutip.core import data as _data

try:  # the solver behind qutip.steadystate(method="direct"); private to QuTiP, pinned at 5.3.x by pyproject
    from qutip.solver.steadystate import _steadystate_direct
except ImportError:  # pragma: no cover - a QuTiP without it falls back to the public entry point
    _steadystate_direct = None
try:  # the eigen-decomposition behind qutip.spectrum(solver="es")
    from qutip.solver.spectrum import _diagonal_evolution
except ImportError:  # pragma: no cover
    _diagonal_evolution = None


def steady_state_direct(H: qt.Qobj, c_ops: Sequence[qt.Qobj]) -> qt.Qobj:
    """``qutip.steadystate(H, c_ops, method="direct")`` without the options context; ``H`` may be a Liouvillian."""
    ops = list(c_ops)
    if _steadystate_direct is None or qt.settings.core["default_dtype_scope"] != "creation":
        return qt.steadystate(H, ops, method="direct")
    if not H.issuper and not ops:
        raise TypeError("Cannot calculate the steady state for a non-dissipative system.")
    if not H.issuper:
        L = qt.liouvillian(H, ops)
    else:
        L = H
        for op in ops:
            L = L + qt.lindblad_dissipator(op)
    rho: qt.Qobj = _steadystate_direct(L, 0, method=None)
    return rho


def spectrum_es(
    H: qt.Qobj,
    c_ops: Sequence[qt.Qobj],
    wlist: Sequence[float] | np.ndarray,
    a_op: qt.Qobj,
    b_op: qt.Qobj,
    rho_ss: qt.Qobj | None = None,
) -> np.ndarray:
    """``qutip.spectrum(H, wlist, c_ops, a_op, b_op, solver="es")`` given the steady state ``rho_ss`` of the same ``H`` and
    ``c_ops`` (solved with :func:`steady_state_direct` when None)."""
    ops = list(c_ops)
    w = np.asarray(wlist, dtype=float)
    if _diagonal_evolution is None or _steadystate_direct is None:
        return np.asarray(qt.spectrum(H, w, ops, a_op, b_op, solver="es"))
    L = qt.liouvillian(H, ops) if not H.issuper else H + sum((qt.lindblad_dissipator(c) for c in ops), 0 * H)
    rho0 = rho_ss if rho_ss is not None else steady_state_direct(L, ())
    a_op_ss = qt.expect(a_op, rho0)
    b_op_ss = qt.expect(b_op, rho0)
    states, rates = _diagonal_evolution(L, b_op * rho0)
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
