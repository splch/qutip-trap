"""The 'QuTiP interface facts' row of PLAN.md Section 9.13 (milestone M0): every QuTiP 5.3.x interface the plan relies on
is present in the pinned toolchain, so a minor bump that removes one fails loudly rather than silently changing the
numerics. Introspected, never assumed (Appendix D)."""

from __future__ import annotations

import inspect

import numpy as np
import qutip as qt

from qutip_trap.units import TWO_PI


def test_propagator_has_the_piecewise_t_option() -> None:
    """`propagator(H, t, piecewise_t=...)` (Section 5.3: piecewise propagators on internal-state-only spaces)."""
    # QuTiP 5.3 exposes it through the solver options of `propagator`; the keyword must be accepted, not swallowed
    sig = inspect.signature(qt.propagator)
    accepts = "piecewise_t" in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    assert accepts, "propagator no longer accepts piecewise_t (Section 9.13 QuTiP interface facts)"
    if "piecewise_t" not in sig.parameters:
        # a **kwargs signature: the option must actually be honoured by the propagator machinery
        h = qt.sigmaz()
        u = qt.propagator(h, [0.0, 0.1, 0.2], options={"piecewise_t": True})
        assert isinstance(u, list) and len(u) == 3


def test_mcsolve_has_the_improved_sampling_option() -> None:
    """`mcsolve` option `improved_sampling` (Section 5.3: the no-jump trajectory as a weighted deterministic member)."""
    assert "improved_sampling" in qt.MCSolver.solver_options, (
        "mcsolve no longer has improved_sampling (Section 9.13 QuTiP interface facts)"
    )
    # a two-level decay with the no-jump member: the result carries the deterministic weight the plan draws shots from
    res = qt.mcsolve(
        qt.sigmaz(),
        qt.basis(2, 0),
        [0.0, 0.5, 1.0],
        c_ops=[np.sqrt(1.0) * qt.sigmam()],
        ntraj=4,
        options={"improved_sampling": True, "map": "serial", "progress_bar": False},
        seeds=[1, 2, 3, 4],
    )
    # the plan (Section 5.3) names this `deterministic_weight_info`; 5.3.1 exposes it as the pair below (ledger
    # conv.mcsolve_weight_attributes), and the shots must be drawn from the weighted mixture they describe
    assert hasattr(res, "deterministic_weights") and hasattr(res, "runs_weights"), (
        "McResult lost the no-jump / stochastic trajectory weights the weighted mixture of Section 5.3 needs"
    )
    assert len(res.deterministic_trajectories) == 1, (
        "improved_sampling evolves exactly one deterministic member"
    )
    w_det = float(res.deterministic_weights[0])
    assert 0.0 < w_det < 1.0 and abs(w_det + sum(res.runs_weights) - 1.0) < 1e-12


def test_coefficient_signatures_f_t_and_f_t_kwargs() -> None:
    """Coefficient callables `f(t)` and `f(t, **kwargs)` are both accepted by `QobjEvo` (Section 5.2 builder coefficients)."""

    def plain(t: float) -> complex:
        return complex(np.cos(TWO_PI * t))

    def with_args(t: float, omega: float = 1.0, **_: object) -> complex:
        return complex(np.cos(omega * t))

    a = qt.QobjEvo([qt.sigmax(), plain])
    b = qt.QobjEvo([qt.sigmax(), with_args], args={"omega": TWO_PI})
    assert abs((a(0.3) - b(0.3)).norm()) < 1e-12


def test_array_coefficients_with_tlist_and_order_zero() -> None:
    """Array coefficients with `tlist` and `order=0` (Section 5.5: noise realizations interpolated on a fixed grid)."""
    tlist = np.linspace(0.0, 1.0, 11)
    values = np.arange(11, dtype=float)
    coeff = qt.coefficient(values, tlist=tlist, order=0)
    assert coeff(0.55) == 5.0  # zero-order hold between grid points
    linear = qt.coefficient(values, tlist=tlist, order=1)
    assert abs(linear(0.55) - 5.5) < 1e-12


def test_integrator_option_nsteps_and_the_two_rungs_of_the_ladder() -> None:
    """The `nsteps` integrator option, and both rungs of the Section 5.3 escalation ladder, exist."""
    for method in ("dop853", "vern9"):
        opts = qt.SESolver(qt.sigmaz(), options={"method": method}).options
        assert opts["method"] == method
    integrator_options = qt.SESolver(qt.sigmaz(), options={"method": "dop853"}).options
    assert "nsteps" in integrator_options, "the dop853 integrator lost its nsteps option"
    res = qt.sesolve(qt.sigmaz(), qt.basis(2, 0), [0.0, 1.0], options={"method": "dop853", "nsteps": 10**7})
    assert res.final_state is not None
