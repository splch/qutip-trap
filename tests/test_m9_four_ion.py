"""The four-ion GATE_LOCAL-against-JOINT_EXACT circuit of PLAN.md Section 9.8 row 1 (M9a audit E5).

Section 9.8 row 1 asks for "three- and four-ion circuits of 3 to 5 gates at the resolved-mode sets of Section 9.6" whose
"final probabilities agree within the residual-displacement bound sum |alpha|^2 (2 nbar + 1) that GATE_LOCAL reports;
disagreement above it fails". M9a shipped the three-ion case only (``tests/test_gate_local.py``); the only four-ion case in the
tree was a single JOINT_EXACT Mølmer-Sørensen pulse in ``validation/scripts/check_circuits.py`` with no GATE_LOCAL
comparison, while README.md claimed the row covered both. This file is the four-ion half.

The fixture is check_circuits.py's four-ion chain (four 171Yb+ ions on the Section 11.1 trap, the global Raman pair along x),
run at a freeze tolerance that gives BOTH levels the SAME resolved-mode set - which is what makes the comparison measure
GATE_LOCAL's own approximation (tracing out the spin-motion and mode-mode correlations between steps) and nothing else.

Measured chi per Mølmer-Sørensen step on the GHZ4 circuit below (the four x modes, ground state):

    ms(0,1)   mode 4: 0.038   mode 5: 0.1078  mode 6: 0.1255  mode 7: 0.8056 rad
    ms(1,2)   mode 4: 0.1251  mode 5: 0.1123  mode 6: 0.0414  mode 7: 0.8395 rad
    ms(2,3)   mode 4: 0.038   mode 5: 0.1078  mode 6: 0.1255  mode 7: 0.8056 rad

At the default freeze tolerance of 0.05 rad all four are resolved at both levels, and the joint space is 2 x 9^3 x 12 x 16 =
139968 - far above the Section 11.5 guard, so JOINT_EXACT cannot run. At ``freeze_chi_max_rad = 0.3`` the gate mode alone is
resolved, at BOTH levels and for every step (0.8 rad against 0.13), so JOINT_EXACT runs at dimension 192 and each gate-local
space is 4 x 12 = 48; modes 4, 5 and 6 are frozen in both runs, both calibrated against the same surrogate table that absorbs
their chi, so their loss cancels in the difference.

**Deviation from the audit's suggested fixture, and why.** The audit and the task brief asked for check_circuits.py's
hand-built space (modes 6 and 7 resolved at d_m = 12, dimension 2304). That space cannot be used for this comparison: it is
not what the criterion selects for either level, and GATE_LOCAL's per-step ``step_space`` would resolve modes the hand-built
space freezes with |chi| = 0.11 to 0.13 rad each - a 14 to 16 % error on a pi/4 gate angle that ``discrepancy_bound`` does
not cover, so the test would measure the hand-built space's frozen-mode error instead of GATE_LOCAL's. There is no threshold
that makes the criterion select exactly {6, 7} at both levels: mode 6 sits at 0.1255 for two steps and 0.0414 for the third
while mode 4 sits at 0.1251 for that third one. Recorded in the ledger as ``conv.four_ion_gate_local_fixture``.
"""

from __future__ import annotations

import numpy as np
import pytest

from qutip_trap.api import (
    Circuit,
    Operation,
    SolverOptions,
    last_record,
    register_fidelity,
    run,
)
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import compile_to_native
from qutip_trap.control.schedule import schedule as make_schedule
from qutip_trap.options import Numerics
from qutip_trap.run.levels import within_budget
from qutip_trap.run.space import best_contributions, select_space
from tests.m6_fixtures import circuit_fixture

WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
GHZ4 = Circuit(
    4,
    (
        Operation("h", (0,), ()),
        Operation("cnot", (0, 1), ()),
        Operation("cnot", (1, 2), ()),
        Operation("cnot", (2, 3), ()),
    ),
    (0, 1, 2, 3),
)
"""Four gates, inside Section 9.8 row 1's "3 to 5 gates": one Hadamard and three CNOTs, so three Mølmer-Sørensen steps."""


@pytest.mark.slow
def test_four_ion_ghz_circuit_gate_local_against_joint_exact() -> None:
    """Section 9.8 row 1 on four ions: the final probabilities of the GHZ4 circuit at both levels agree within the bound
    GATE_LOCAL reports plus the weight of the initial-mixture branches JOINT_EXACT dropped, with no other slack, and the
    tracked occupation after each gate matches the joint run's reduced motional state to 5 %."""
    fx = circuit_fixture(4)
    sur = surrogate_table(
        fx.device, pairs=[(0, 1), (1, 2), (2, 3)], detection_records=200, detection_windows_s=WINDOWS
    )
    native = compile_to_native(GHZ4, fx.device)
    sched = make_schedule(native, fx.device, sur.table, t0_s=0.0)
    n_modes = len(fx.device.crystal.modes)
    nbar0 = {m: 0.0 for m in range(n_modes)}
    best = best_contributions(fx.device, sched.gates, nbar0)
    opts = SolverOptions(freeze_chi_max_rad=0.3)
    selection = select_space(fx.device, sched, opts, nbar=nbar0, caps={7: 12})
    resolved = selection.resolved_modes
    assert resolved == (7,), (resolved, {m: round(c.chi_rad, 4) for m, c in sorted(best.items())})
    assert selection.space.dims == [2, 2, 2, 2, 12] and selection.space.dimension == 192
    inside, dim, _nnz = within_budget(selection.space, opts)
    assert inside and dim == 192, "JOINT_EXACT must be affordable inside the Section 11.5 guard"
    # the three modes the freeze tolerance leaves out are frozen, not dropped: both runs carry their exact Debye-Waller
    # factors and Fock branches, so their chi loss is common to the two levels and cancels in the difference
    assert [selection.mode_class[m] for m in (4, 5, 6)] == ["frozen"] * 3, selection.mode_class
    kw = dict(
        table=sur.table,
        keep_final_state=True,
        numerics=Numerics.from_solver_options(opts, caps={7: 12}),
        seed=3,
    )
    a = run(GHZ4, fx.device, 200, level="JOINT_EXACT", **kw)  # type: ignore[arg-type]
    b = run(GHZ4, fx.device, 200, level="GATE_LOCAL", **kw)  # type: ignore[arg-type]
    assert a.diagnostics.space.dimension == 192
    assert tuple(a.diagnostics.mode_class[m] for m in resolved) == ("resolved",) * len(resolved)
    gl = b.diagnostics.gate_local
    assert gl is not None and gl.discrepancy_bound > 0.0
    ms_steps = [s for s in gl.steps if s.kind == "gate" and s.resolved]
    assert len(ms_steps) == 3, [s.gate_id for s in gl.steps]
    pa = np.real(np.diag(np.asarray(a.final_state.full())))
    pb = np.real(np.diag(np.asarray(b.final_state.full())))
    bound = gl.discrepancy_bound + a.diagnostics.dropped_branch_weight
    assert np.max(np.abs(pa - pb)) < bound, (np.max(np.abs(pa - pb)), gl.discrepancy_bound, bound)
    assert abs(register_fidelity(a) - register_fidelity(b)) < bound
    # Section 9.8 row 2: the tracked nbar after each gate against the joint run's reduced motional state at that time,
    # branch-weighted over the initial mixture (one trace per branch)
    rec_a = last_record(a)
    w_tot = sum(br.weight for br in rec_a.branches)
    for s in ms_steps:
        k = int(np.argmin(np.abs(rec_a.traces[0].times_s - s.t_end_s)))
        for m in s.resolved:
            joint_n = float(
                sum(
                    br.weight * np.real(tr.mode_occupations[m][k])
                    for br, tr in zip(rec_a.branches, rec_a.traces)
                )
                / w_tot
            )
            assert abs(s.nbar_after[m] - joint_n) <= 0.05 * max(joint_n, 1e-3) + 1e-4, (
                s.gate_id,
                m,
                s.nbar_after[m],
                joint_n,
            )
