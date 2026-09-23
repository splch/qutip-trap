"""The Section 5.5 / 9.9 tolerance-convergence run, applied to Section 9.2 fixtures.

"Every result is quoted with the integrator, the tolerances and the change under a tighter pair" (Section 5.5), and
"the test suite does this for every validation case". Two defects the M2 audit found:

* above d_m ~ 100 the atol keying of Section 5.3 was ``max(atol, 1e-8)``, which clamped a DELIBERATE tightening back
  up and left only rtol moving, so the comparison reported a spuriously small change (audit E9/B9);
* the machinery existed (``halving_test``, ``tightened``) and was exercised only on a synthetic closure - no
  validation case ran it (audit D).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import Schedule
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.evolve import (
    LARGE_MODE_ATOL,
    LARGE_MODE_DIMENSION,
    convergence_check,
    evolve,
    tightened,
)
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.hilbert.truncation import halving_test
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.units import TWO_PI
from qutip_trap.validation.spin_motion_closed_forms import sideband_rabi_rad_s
from tests.m2_fixtures import single_ion_raman_device

KX = 1
WX = TWO_PI * 3.0e6


def _runner(dev, drive, t_end, space, n0=0):  # type: ignore[no-untyped-def]
    """A closure that integrates one pulse under the SolverOptions it is handed and returns its traces."""

    def run(options: SolverOptions) -> dict[str, np.ndarray]:
        eng = JointExactEngine(store_per_segment=9)
        tr = eng.run_pulses(
            dev,
            Schedule((Pulse(drive, 0.0, t_end, "p", ()),), (), (), {0: 0.0}),
            space.initial_state([0], fock={KX: n0}),
            space,
            quiet_sample(),
            SeedSpec(0),
            options,
        )
        return {k: np.asarray(v) for k, v in tr.expectations.items() if k.startswith("P")}

    return run


@pytest.fixture(scope="module")
def raman():  # type: ignore[no-untyped-def]
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    space = HilbertSpace((2,), (ModeTruncation(KX, 12, (0, 3), 0.2),), None, (0, 2))
    return dev, dd, space


def test_tolerance_convergence_on_the_carrier_and_sideband_fixtures(raman) -> None:  # type: ignore[no-untyped-def]
    """Section 9.2's Rabi-flopping and sideband-flopping rows, each quoted with the change under atol/10, rtol/10."""
    dev, dd, space = raman
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]

    carrier = _runner(dev, square_drive(dd, include_stark=False), math.pi / om, space)
    rep = convergence_check(carrier)
    assert rep.tolerances == (1e-10, 1e-8)
    assert rep.tightened_tolerances == pytest.approx((1e-11, 1e-9), rel=1e-12)
    assert rep.converged, rep.summary()
    assert rep.max_change < 1e-6
    assert "tolerance convergence" in rep.summary() and "converged" in rep.summary()

    t_blue = math.pi / sideband_rabi_rad_s(om, eta, 0, 1)
    blue = _runner(dev, square_drive(dd, detuning_hz=3.0e6, include_stark=False), t_blue, space)
    rep_b = convergence_check(blue)
    assert rep_b.converged, rep_b.summary()
    assert rep_b.max_change < 1e-6
    # the single-array form behind the convergence badge agrees
    ok, delta, tight = halving_test(lambda o: np.concatenate(list(blue(o).values())), SolverOptions())
    assert ok and delta == pytest.approx(rep_b.max_change, abs=1e-12)
    assert (tight.atol, tight.rtol) == pytest.approx((1e-11, 1e-9), rel=1e-12)


def test_convergence_check_rejects_a_mismatched_observable_set_and_a_bad_factor(raman) -> None:  # type: ignore[no-untyped-def]
    dev, dd, space = raman
    om = TWO_PI * dd.carrier_rabi_hz
    run = _runner(dev, square_drive(dd, include_stark=False), math.pi / om, space)
    with pytest.raises(ValueError, match="tightening factor"):
        convergence_check(run, SolverOptions(), factor=1.0)

    def wobbly(options: SolverOptions) -> dict[str, np.ndarray]:
        out = run(options)
        if options.atol < 1e-10:
            out["extra"] = np.zeros(1)
        return out

    with pytest.raises(ValueError, match="different observables"):
        convergence_check(wobbly, SolverOptions())


def test_a_deliberate_tightening_survives_the_large_mode_atol_keying() -> None:
    """Section 5.3 relaxes the DEFAULT atol above d_m ~ 100; a caller who tightened it keeps what they asked for.

    The check is on the ``atol`` the integration actually reports, on a bare harmonic oscillator of d_m = 121 - the
    dimension at which the plan measured dop853's "probably stiff" abort.
    """
    import qutip as qt

    d = 121
    assert d > LARGE_MODE_DIMENSION
    h = WX * qt.num(d)
    psi0 = qt.basis(d, 1)
    times = np.linspace(0.0, 1e-6, 5)
    default = evolve(h, psi0, times, options=SolverOptions(), largest_mode_dimension=d)
    assert default.atol == LARGE_MODE_ATOL, "the default relaxes above d_m ~ 100 (the keying itself is right)"
    tight = tightened(SolverOptions())
    got = evolve(h, psi0, times, options=tight, largest_mode_dimension=d)
    assert got.atol == tight.atol, "a deliberate tightening is not clamped back to 1e-8"
    assert got.atol == pytest.approx(1e-11, rel=1e-12)
    assert got.atol < default.atol
    # and a deliberate LOOSENING is honoured too: the keying is not a floor on whatever the caller passed
    loose = evolve(h, psi0, times, options=SolverOptions(atol=1e-6), largest_mode_dimension=d)
    assert loose.atol == 1e-6


@pytest.mark.slow
def test_the_escalation_ladder_at_the_plans_three_large_caps() -> None:
    """Section 9.17's escalation-ladder row: d_m = 121, 151 and 201 on a Section 11.1-shaped pulse.

    Section 5.3 keys atol to the measured points, 1e-10 up to d_m ~ 100 and 1e-8 above, "validated at d_m = 121, 151
    and 201"; nothing ran those three (M2 audit E18). What is asserted here is what is true on this machine: all three
    integrate on the FIRST rung with no retry, the keying gives 1e-8 above the threshold and 1e-10 at it, and the norm
    survives. The plan's "probably stiff" abort at d_m = 121 with atol 1e-10 does NOT reproduce on a one-mode detuned
    spin-dependent force - it is specific to the Section 11.1 two-ion pulse at joint dimension 2048 - so the keying is
    exercised as the precaution it is and no failure is manufactured to make the ladder look used.
    """
    import qutip as qt

    eta, om_drive = 0.1, TWO_PI * 250e3
    delta = WX - TWO_PI * 20e3  # a blue-sideband-detuned SDF, the plan's own bench case
    for d in (101, 121, 151, 201):
        disp = qt.displace(d, 1j * eta)
        a = qt.destroy(d)
        h = qt.QobjEvo(
            [
                qt.tensor(qt.qeye(2), WX * a.dag() * a),
                [
                    qt.tensor(qt.sigmap(), disp),
                    lambda t, **kw: 0.5 * om_drive * np.exp(-1j * delta * t),
                ],
                [
                    qt.tensor(qt.sigmap(), disp).dag(),
                    lambda t, **kw: 0.5 * om_drive * np.exp(1j * delta * t),
                ],
            ]
        )
        psi0 = qt.tensor(qt.basis(2, 1), qt.basis(d, 0))
        ev = evolve(
            h,
            psi0,
            [0.0, 10e-6],
            options=SolverOptions(),
            largest_mode_dimension=d,
            omega_max_rad_s=WX,
        )
        assert ev.integrator == "dop853" and ev.retries == (), (d, ev.retries)
        assert ev.atol == LARGE_MODE_ATOL, d
        assert ev.final.norm() == pytest.approx(1.0, abs=1e-7), d
    # at the threshold itself the keying does not fire: "1e-10 up to d_m ~ 100"
    d = LARGE_MODE_DIMENSION
    h100 = WX * qt.num(d)
    at_threshold = evolve(
        h100, qt.basis(d, 1), np.linspace(0.0, 1e-6, 3), options=SolverOptions(), largest_mode_dimension=d
    )
    assert at_threshold.atol == 1e-10
