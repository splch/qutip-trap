"""Every exported discriminator is reachable from ``run`` (PLAN.md Section 8.3, "implemented as a strategy interface over
the same record"; Section 8.4's Monte-Carlo POVM).

Before the 2026-09-07 M5 fix ``readout_stage`` built the product POVM unconditionally with ``n_samples`` at 0, and
``per_ion_confusion`` raises for anything but a threshold discriminator, so ``run(..., discriminator=TimeResolvedML(...))``
raised whatever the readout mode was: Myerson's time-resolved ML, the adaptive protocol and the first-photon protocols were
library-only (audit B3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import Circuit, Operation, SolverOptions, last_record, run
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.options import Numerics, Physics, Readout
from qutip_trap.readout.discriminate import AdaptiveML, FirstPhoton, ThresholdDiscriminator, TimeResolvedML
from tests.m6_fixtures import circuit_fixture

ONE = Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 4))
FAST = SolverOptions(branch_weight_min=1e-2)


@pytest.fixture(scope="module")
def one_ion():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(1)
    sur = surrogate_table(fx.device, pairs=[], detection_records=400, detection_windows_s=WINDOWS)
    return fx, sur


def _kw(fx, sur):  # type: ignore[no-untyped-def]
    return {
        "table": sur.table,
        "numerics": Numerics.from_solver_options(FAST),
        "physics": Physics(noise=False),
    }


def test_time_resolved_ml_runs_end_to_end_on_the_full_record_path(one_ion) -> None:  # type: ignore[no-untyped-def]
    """The record path replaces the POVM (Section 5.7), so no Monte-Carlo confusion is built at all; the sub-bin record and
    the posterior confidence of Section 8.6 reach ``Result``."""
    fx, sur = one_ion
    disc = TimeResolvedML(5e-6, 20e-6)
    res = run(ONE, fx.device, 40, **_kw(fx, sur), readout=Readout(mode="full", discriminator=disc))
    rec = last_record(res)
    assert rec.readout.povm is None and rec.readout.product is None
    # the scheme states the dark class: 171Yb+ direct fluorescence, so "dark" and Myerson's 1/tau is R_b
    assert rec.readout.discriminator.dark_class == "dark"  # type: ignore[union-attr]
    assert res.photon_records is not None and res.photon_records.shape == (40, 1)
    assert res.sub_bin_records is not None and res.sub_bin_records.shape == (40, 1, 4)
    assert np.array_equal(res.sub_bin_records.sum(axis=2), res.photon_records)
    assert res.posteriors is not None and res.posteriors.shape == (40, 1)
    assert np.all(res.posteriors >= 0.0) and np.all(res.posteriors <= 1.0)
    assert any("SPAM definition" in s and "TimeResolvedML" in s for s in res.diagnostics.approximations)
    assert any("estimated from this run's own" in s for s in res.diagnostics.approximations)
    # a level the circuit never populates reports nan rather than a silent zero
    eps = res.spam["q0"]
    assert all(math.isnan(x) or 0.0 <= x <= 1.0 for x in eps)
    assert sum(res.probabilities.values()) == pytest.approx(1.0)


def test_a_monte_carlo_povm_runs_the_fast_path_and_declares_its_error_bar(one_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 8.4: a non-threshold discriminator's confusion has no closed form, so the fast path estimates it and reports
    the statistical uncertainty it earned (``POVM.uncertainty``)."""
    fx, sur = one_ion
    disc = TimeResolvedML(5e-6, 20e-6)
    res = run(ONE, fx.device, 40, **_kw(fx, sur), readout=Readout(discriminator=disc, povm_samples=400))
    rec = last_record(res)
    assert rec.readout.povm is not None and rec.readout.povm.per_ion is not None
    unc = rec.readout.povm.uncertainty
    assert unc == pytest.approx(math.sqrt(0.25 / 400), rel=1e-12)
    assert any(f"{unc:.2e}" in s for s in res.diagnostics.approximations)
    assert any("400 sampled" in s for s in res.diagnostics.approximations)
    eps_b, eps_d = res.spam["q0"]
    assert 0.0 <= eps_b <= 0.1 and 0.0 <= eps_d <= 0.1
    # the threshold discriminator's POVM stays exact and reports no uncertainty
    thr = run(
        ONE, fx.device, 40, **_kw(fx, sur), readout=Readout(discriminator=ThresholdDiscriminator(0.5, 20e-6))
    )
    assert last_record(thr).readout.povm.uncertainty == 0.0  # type: ignore[union-attr]
    assert not any("sampled records per level" in s for s in thr.diagnostics.approximations)


def test_the_adaptive_and_first_photon_protocols_also_run(one_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 8.3's adaptive Bayesian early termination and Noek's two-photon / Crain's stop-on-first-photon protocols;
    the first-photon record needs arrival times, which Section 8.6 asks ``Result`` to keep."""
    fx, sur = one_ion
    for disc in (
        AdaptiveML(5e-6, 40e-6, 1e-3),
        FirstPhoton(40e-6, cutoff_s=10e-6),
    ):
        res = run(ONE, fx.device, 30, **_kw(fx, sur), readout=Readout(mode="full", discriminator=disc))
        assert res.shots == 30 and res.photon_records is not None
        assert any("readout full path" in s for s in res.diagnostics.approximations)
        if disc.needs_arrivals:
            assert res.arrival_times_s is not None and len(res.arrival_times_s) == 30
            assert all(len(shot) == 1 for shot in res.arrival_times_s)
            assert all(
                len(shot[0]) == int(total)
                for shot, total in zip(res.arrival_times_s, res.photon_records[:, 0])
            )
        else:
            assert res.arrival_times_s is None
