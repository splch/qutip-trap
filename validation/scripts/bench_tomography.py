"""The three routes of the GATE_LOCAL tomography (dynamics/tomography.py; PLAN.md Section 5.4; performance pass 2026-09-09).

The state route of milestone M9a propagates every one of the prod_i d_i^2 input states per motional branch and fits the Choi
matrix by least squares. When the step is unitary the final state is linear in the initial ket, so the isometry route propagates
the prod_i d_i internal basis kets per branch and reads the channel off the Stinespring isometry (Kraus operators = its slices by
motional output index), and on an internal-state-only space the propagator route takes the cached segment propagator itself as the
branch's Kraus operator. This script runs the routes side by side on the fixtures of tests/m4_fixtures.py (two 171Yb+ ions):

  part 1: a pi/2 carrier pulse on the internal-state-only space [2, 2] with three Fock branches of the frozen x modes
          (propagator route against the state route);
  part 2: a 10 us single-loop entangling pulse with the x-COM resolved at d = 15 and the stretch mode a frozen coupled branch
          (isometry route against the state route), engine runs, wall time, the largest difference in the Choi matrices, the
          outputs, the reduced motional states and the residual displacements, the raw Choi matrix's TP residual and the Dykstra
          iteration counts;
  part 3: the register update sum_a K_a rho K_a^dag on a ten-qubit density matrix: the superoperator product of apply_kraus_dm
          against the per-operator einsum it replaced, and Register.marginal on the strided view against the copying form;
  part 4: the three declared relaxations keyed to the map accuracy (Tier 2 of the same pass) on the part 2 step: the branch
          tail rule (branches kept, dropped weight, the reported bound 2w), the keyed tolerance (right-hand sides, wall time,
          the reported ten-times-tighter change against the realized difference from the engine-tolerance extraction) and the
          derived cap margin (the cap, the measured interior element error and the leakage), each against the reference.

Wall times are never compared by run_checks.py (bench_ scripts are timed only); the agreement figures are the point.

Run:  uv run python validation/scripts/bench_tomography.py      (about half a minute on the plan's machine)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qutip_trap.api import HilbertSpace, ModeTruncation, SeedSpec, SolverOptions  # noqa: E402
from qutip_trap.calibration.entangling import ms_schedule  # noqa: E402
from qutip_trap.control.schedule import Schedule, single_qubit_pulse  # noqa: E402
from qutip_trap.control.table import Waveform  # noqa: E402
from qutip_trap.dynamics.engine import JointExactEngine, MotionalModel  # noqa: E402
from qutip_trap.dynamics.tomography import apply_kraus_dm, cp_residual, tp_residual  # noqa: E402
from qutip_trap.noise.sampling import quiet_sample  # noqa: E402
from qutip_trap.run.gate_local import Register  # noqa: E402
from tests.m4_fixtures import (  # noqa: E402
    X_COM_TWO_IONS,
    chain_device,
    derived_seeds,
    raman_gate_drives,
    table_with_waveform,
    two_ion_modes,
)


def maxdiff(a, b) -> float:  # type: ignore[no-untyped-def]
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def both_routes(dev, sched, space, model, weight_min, label):  # type: ignore[no-untyped-def]
    recs = {}
    for flag in (True, False):
        eng = JointExactEngine()
        opts = SolverOptions(branch_weight_min=weight_min, map="serial", tomography_isometry=flag)
        t0 = time.perf_counter()
        rec = eng.tomography(dev, sched, space, model, quiet_sample(), SeedSpec(0), opts)
        wall = time.perf_counter() - t0
        recs[flag] = rec
        solves = sum(r.propagator_solves for r in rec.reports)
        hits = sum(r.propagator_cache_hits for r in rec.reports)
        print(
            f"  {label} route={rec.route:<10} branches={rec.branches} engine_runs={rec.engine_runs:>3} "
            f"propagator solves/hits={solves}/{hits} dims={rec.space.dims} wall={wall:.2f} s "
            f"tp(choi_raw)={tp_residual(rec.choi_raw):.2e} cp(choi_raw)={cp_residual(rec.choi_raw):.2e} "
            f"dykstra={rec.dykstra_iterations}"
        )
    a, b = recs[True], recs[False]
    print(f"  agreement: choi {maxdiff(a.choi, b.choi):.2e}, choi_raw {maxdiff(a.choi_raw, b.choi_raw):.2e}, "
          f"outputs {max(maxdiff(x, y) for x, y in zip(a.outputs, b.outputs)):.2e}")
    for m in a.motional_out:
        print(f"  mode {m}: reduced motional outputs {max(maxdiff(x, y) for x, y in zip(a.motional_out[m], b.motional_out[m])):.2e}, "
              f"<a_m> {max(abs(x - y) for x, y in zip(a.alpha_out[m], b.alpha_out[m])):.2e}, "
              f"nbar {max(abs(x - y) for x, y in zip(a.nbar_out[m], b.nbar_out[m])):.2e}, "
              f"residual displacement {abs(a.residual_displacement()[m] - b.residual_displacement()[m]):.2e}")
    return recs


def part1(dev, drives, rabi):  # type: ignore[no-untyped-def]
    print("== Part 1: carrier pi/2 pulse on the internal-state-only space [2, 2], frozen x modes at nbar = 0.05 (three branches)")
    pulse = single_qubit_pulse(0, math.pi / 2.0, 0.3, drives[0], rabi[(0, 0)], 0.0, gate_id="gpi2", programmed=False)
    sched = Schedule((pulse,), (), (), {0: 0.0, 1: 0.0})
    space = HilbertSpace((2, 2), (), None, (0, 1, 2, 3, 4, 5))
    model = MotionalModel(reduced={}, nbar={0: 0.0, 1: 0.0, 2: 0.05, 3: 0.05, 4: 0.0, 5: 0.0}, frozen=tuple(range(6)))
    both_routes(dev, sched, space, model, 0.02, "carrier")


def part2(dev, drives, rabi, stark):  # type: ignore[no-untyped-def]
    print("== Part 2: 10 us single-loop entangling pulse, x-COM resolved at d = 15, stretch mode frozen and coupled (three branches)")
    modes = two_ion_modes(dev)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=100e3, all_modes=True)
    table = table_with_waveform((0, 1), wf, rabi_hz=rabi, stark_hz=stark)
    sched = ms_schedule(wf, (0, 1), drives, table)
    # d = 15 gives both routes headroom: at d = 13 the sixteen-input route's superpositions trip the Section 5.5 margin check one
    # level above the basis kets and the two routes end on different spaces (the difference the tomography's notes declare)
    space = HilbertSpace((2, 2), (ModeTruncation(2, 15, (0, 3), 0.13),), None, (0, 1, 3, 4, 5))
    model = MotionalModel(reduced={}, nbar={0: 0.0, 1: 0.0, 2: 0.05, 3: 0.05, 4: 0.0, 5: 0.0}, frozen=(0, 1, 3, 4, 5))
    both_routes(dev, sched, space, model, 0.02, "entangling")


def kraus_reference(rho, kraus, dims, factors):  # type: ignore[no-untyped-def]
    n = len(dims)
    total = int(np.prod(dims))
    fac = list(factors)
    rest = [f for f in range(n) if f not in fac]
    d_loc = int(np.prod([dims[f] for f in fac]))
    arr = rho.reshape(list(dims) + list(dims))
    perm = fac + rest + [n + f for f in fac] + [n + f for f in rest]
    a = np.transpose(arr, perm).reshape(d_loc, total // d_loc, d_loc, total // d_loc)
    out = np.zeros_like(a)
    for k in kraus:
        out += np.einsum("ab,bxcy,dc->axdy", k, a, k.conj())
    shaped = out.reshape([dims[f % n] for f in perm])
    return np.asarray(np.transpose(shaped, np.argsort(perm)).reshape(total, total))


def marginal_reference(dm, dims, factors):  # type: ignore[no-untyped-def]
    n = len(dims)
    fac = list(factors)
    rest = [f for f in range(n) if f not in fac]
    d_loc = int(np.prod([dims[f] for f in fac]))
    d_rest = int(np.prod(dims)) // d_loc
    arr = dm.reshape(list(dims) + list(dims))
    perm = fac + rest + [n + f for f in fac] + [n + f for f in rest]
    a = np.transpose(arr, perm).reshape(d_loc, d_rest, d_loc, d_rest)
    return np.asarray(np.einsum("axbx->ab", a))


def part3() -> None:
    print("== Part 3: the register update at ten qubits (four Kraus operators on factors (0, 5))")
    rng = np.random.default_rng(0)
    nq = 10
    dims = [2] * nq
    psi = rng.normal(size=2**nq) + 1j * rng.normal(size=2**nq)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    q, _ = np.linalg.qr(rng.normal(size=(16, 4)) + 1j * rng.normal(size=(16, 4)))
    kraus = [q[4 * a : 4 * a + 4, :] for a in range(4)]
    t0 = time.perf_counter()
    out = apply_kraus_dm(rho, kraus, dims, (0, 5))
    t_super = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = kraus_reference(rho, kraus, dims, (0, 5))
    t_einsum = time.perf_counter() - t0
    print(f"  apply_kraus_dm (superoperator product) {t_super:.3f} s, per-operator einsum {t_einsum:.3f} s, "
          f"max |difference| {maxdiff(out, ref):.2e}, trace {np.trace(out).real:.12f}")
    reg = Register.from_density_matrix(rho, dims)
    t0 = time.perf_counter()
    m = reg.marginal((5, 0))
    t_view = time.perf_counter() - t0
    t0 = time.perf_counter()
    m_ref = marginal_reference(rho, dims, (5, 0))
    t_copy = time.perf_counter() - t0
    print(f"  Register.marginal on the strided view {t_view * 1e3:.2f} ms, transpose-and-copy form {t_copy * 1e3:.2f} ms, "
          f"max |difference| {maxdiff(m, m_ref):.2e}")


def part4(dev, drives, rabi, stark):  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from qutip_trap.dynamics.engine import required_margin_under
    from qutip_trap.dynamics.tomography import motional_branches
    from qutip_trap.hilbert.operators import displacement_leakage, interior_element_error, required_margin

    print("== Part 4: the declared relaxations keyed to the map accuracy (the part 2 step, three branches at the reference)")
    modes = two_ion_modes(dev)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=100e3, all_modes=True)
    table = table_with_waveform((0, 1), wf, rabi_hz=rabi, stark_hz=stark)
    sched = ms_schedule(wf, (0, 1), drives, table)
    space = HilbertSpace((2, 2), (ModeTruncation(2, 15, (0, 3), 0.13),), None, (0, 1, 3, 4, 5))
    model = MotionalModel(reduced={}, nbar={0: 0.0, 1: 0.0, 2: 0.3, 3: 0.3, 4: 0.0, 5: 0.0}, frozen=(0, 1, 3, 4, 5))
    # (a) the tail rule on a warm input (nbar 0.3 on the resolved mode and on the frozen coupled one)
    full, dropped_full, _ = motional_branches(space, model, [3], 1e-6)
    for budget in (0.0, 1e-4, 2.5e-4, 1e-3):
        kept, dropped, _ = motional_branches(space, model, [3], 1e-6, dropped_weight_max=budget)
        print(f"  tail rule budget {budget:.1e}: {len(kept)} of {len(full)} branches kept, dropped weight {dropped:.3e}, bound 2w {2 * dropped:.3e}")
    # (b) the keyed tolerance against the engine tolerance, both without the tail rule
    reference = SolverOptions(branch_weight_min=0.02, map="serial", tomography_dropped_weight_max=0.0, tomography_tolerance_keyed=False)
    keyed = replace(reference, tomography_tolerance_keyed=True)
    recs = {}
    for label, o in (("engine tolerance", reference), ("keyed tolerance", keyed)):
        eng = JointExactEngine()
        t0 = time.perf_counter()
        rec = eng.tomography(dev, sched, space, model, quiet_sample(), SeedSpec(0), o)
        wall = time.perf_counter() - t0
        recs[label] = rec
        rhs = sum((s.rhs_evaluations or 0) for r in rec.reports[: 4 * rec.branches] for s in r.segments)
        print(f"  {label}: atol/rtol {rec.tolerances[0]:.0e}/{rec.tolerances[1]:.0e}, engine runs {rec.engine_runs}, right-hand sides {rhs}, "
              f"wall {wall:.2f} s, reported tolerance change {rec.tolerance_change}")
    a, b = recs["keyed tolerance"], recs["engine tolerance"]
    print(f"  realized difference keyed - engine: choi {maxdiff(a.choi, b.choi):.2e}, outputs {max(maxdiff(x, y) for x, y in zip(a.outputs, b.outputs)):.2e}, "
          f"d x trace norm {4 * float(np.sum(np.abs(np.linalg.eigvalsh(0.5 * ((a.choi - b.choi) + (a.choi - b.choi).conj().T))))):.2e}")
    # (c) the derived margin at the map accuracy's element tolerance, against the fixture
    for eta, n_hi in ((0.1, 2), (0.1, 5), (0.1, 10), (0.3, 5)):
        opts = SolverOptions(margin_element_tol=1e-8)
        m = required_margin_under(eta, opts, n_hi)
        print(f"  eta {eta}, top populated level {n_hi}: derived margin {m} (fixture {required_margin(eta)}), interior element error "
              f"{interior_element_error(n_hi + 1 + m, eta, n_hi):.1e} <= 1e-8, leakage past the cap {displacement_leakage(eta, n_hi, m):.1e} < 1e-7")


def main() -> None:
    dev = chain_device(2)
    drives = raman_gate_drives(2)
    rabi, stark = derived_seeds(dev, drives)
    part1(dev, drives, rabi)
    part2(dev, drives, rabi, stark)
    part3()
    part4(dev, drives, rabi, stark)


if __name__ == "__main__":
    main()
