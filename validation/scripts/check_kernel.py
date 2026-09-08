"""Scaling II (PLAN.md milestone M9b; Sections 3.4, 5.1.1, 5.3, 9.9, 11.1, 11.2, 11.3 items 4, 5 and 9), through the package.

1. The factorized drive operator against the assembled one (Section 11.3 item 4): sigma_+ (x) prod_m D_m held as its factors on
   the Bell fixture's space [2, 2, 10, 11] and on the Section 11.1 spaces, applied to random states, its adjoint, its factor-wise
   product, its trace; the QuTiP data-layer operations the solvers use (equality, Hermiticity, expectation values) and pickling.
2. The Section 11.2 cost model (constants of bench_ms_timing_v5.py) and the ``auto`` choice of BuilderOptions.kernel on the
   Section 11.1 spaces and the fixtures of the suite.
3. The engine with the kernel forced each way on a 20 us single-loop entangling pulse: final states, populations, segment reports.
4. Trajectories over 1 and N workers (Section 9.9): the reduced register to 1e-12, the per-trajectory final states and jump
   records identical under the same keyed seeds; the deterministic parts printed, the map's wall times prefixed MC:.
5. The propagator cache on an internal-state-only space (Section 11.3 item 5): one integration per distinct segment
   Hamiltonian, every other input a matrix product, against the per-state ODE path; the tomography of a carrier step.

Run: uv run python validation/scripts/check_kernel.py (about two minutes). Lines prefixed MC: carry wall times and are not compared.
"""

from __future__ import annotations

import dataclasses
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import qutip as qt
from qutip.settings import available_cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qutip_trap.api import HilbertSpace, ModeTruncation, SeedSpec, SolverOptions, white_spectrum  # noqa: E402
from qutip_trap.calibration.entangling import ms_schedule  # noqa: E402
from qutip_trap.control.schedule import Schedule, single_qubit_pulse  # noqa: E402
from qutip_trap.control.table import Waveform  # noqa: E402
from qutip_trap.dynamics.engine import JointExactEngine, MotionalModel  # noqa: E402
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian  # noqa: E402
from qutip_trap.dynamics.kernels import (
    FactorizedOperator,
    factorized_qobj,
    kernel_costs_us,
    prefer_factorized,
)  # noqa: E402
from qutip_trap.hilbert.operators import displacement_operator, qudit_sigma_plus  # noqa: E402
from qutip_trap.light.raman import lamb_dicke_parameters  # noqa: E402
from qutip_trap.noise.sampling import quiet_sample  # noqa: E402
from tests.m4_fixtures import (
    X_COM_TWO_IONS,
    chain_device,
    derived_seeds,
    raman_gate_drives,
    table_with_waveform,
    two_ion_modes,
)  # noqa: E402


def section(title: str) -> None:
    print(f"\n== {title}")


rng = np.random.default_rng(20260907)

section("1. the factorized drive operator against the assembled one (Section 11.3 item 4)")
dev = chain_device(2)
drives = raman_gate_drives(2)
rabi, stark = derived_seeds(dev, drives)
space = HilbertSpace(
    (2, 2), (ModeTruncation(2, 10, (0, 3), 0.13), ModeTruncation(3, 11, (0, 4), 0.13)), None, (0, 1, 4, 5)
)
dk = np.asarray(dev.beams[0].k_vector() - dev.beams[1].k_vector())
for ion in (0, 1):
    etas, _ = lamb_dicke_parameters(dev, ion, dk)
    active = {m: e for m, e in etas.items() if space.mode_class(m) != "frozen"}
    fact = space.drive_operator_factorized(ion, active)
    assembled = space.drive_operator(ion, active)
    v = rng.normal(size=space.dimension) + 1j * rng.normal(size=space.dimension)
    v = v / np.linalg.norm(v)
    ket = qt.Qobj(v.reshape(-1, 1), dims=[space.dims, [1] * len(space.dims)])
    diff_apply = (fact * ket - assembled * ket).norm()
    diff_full = np.max(np.abs(fact.full() - assembled.full()))
    diff_dag = np.max(np.abs(fact.dag().full() - assembled.dag().full()))
    prod = fact.dag() * fact
    diff_prod = np.max(np.abs(prod.full() - (assembled.dag() * assembled).full()))
    print(
        f"  ion {ion}: etas {dict((m, round(e, 4)) for m, e in active.items())}, dims {space.dims}, nnz assembled "
        f"{fact.data.nnz_assembled}; |F psi - A psi| = {diff_apply:.1e}, max|F - A| = {diff_full:.1e}, dagger {diff_dag:.1e}, "
        f"F^dag F factorized: {isinstance(prod.data, FactorizedOperator)} ({diff_prod:.1e}), trace {abs(fact.tr() - assembled.tr()):.1e}, "
        f"F == A (mixed representations): {fact == assembled}, F.isherm {fact.isherm}, "
        f"<psi|F|psi> agreement {abs(qt.expect(fact, ket) - qt.expect(assembled, ket)):.1e}, "
        f"pickle round trip {np.max(np.abs(pickle.loads(pickle.dumps(fact)).full() - fact.full())):.1e}"
    )
for dims, nmax in [((2, 2, 12), 12), ((2, 2, 8, 8), 8), ((2, 2, 6, 6, 6), 6), ((2, 2, 8, 8, 8), 8)]:
    nmodes = len(dims) - 2
    mats = [displacement_operator(nmax, 0.08j).full() for _ in range(nmodes)]
    fact = factorized_qobj(dims, {0: qudit_sigma_plus(2).full(), **{2 + m: mats[m] for m in range(nmodes)}})
    ref = qt.tensor(qt.Qobj(qudit_sigma_plus(2).full()), qt.qeye(2), *[qt.Qobj(m) for m in mats])
    v = rng.normal(size=int(np.prod(dims))) + 1j * rng.normal(size=int(np.prod(dims)))
    ket = qt.Qobj((v / np.linalg.norm(v)).reshape(-1, 1), dims=[list(dims), [1] * len(dims)])
    print(
        f"  Section 11.1 space {dims}: |F psi - A psi| = {(fact * ket - ref * ket).norm():.1e}, nnz assembled {fact.data.nnz_assembled} = {ref.to('CSR').data.as_scipy().nnz}"
    )

section("2. the cost model of Section 11.2 and the auto choice (constants of bench_ms_timing_v5.py)")
for dims, ion, modes in [
    ((2, 2, 12), 0, [2]),
    ((2, 2, 8, 8), 0, [2, 3]),
    ((2, 2, 10, 11), 0, [2, 3]),
    ((2, 2, 11, 13), 0, [2, 3]),
    ((2, 2, 6, 6, 6), 0, [2, 3, 4]),
    ((2, 2, 8, 8, 8), 0, [2, 3, 4]),
    ((2, 2, 2, 12, 12), 0, [3, 4]),
    ((2, 2, 2, 2, 8, 8, 8), 0, [4, 5, 6]),
]:
    a, f = kernel_costs_us(dims, ion, modes)
    print(
        f"  {str(dims):24s} D={int(np.prod(dims)):5d}: assembled {a:8.1f} us, factorized {f:6.1f} us per term -> {'factorized' if prefer_factorized(dims, ion, modes) else 'assembled'}"
    )

section("3. the engine with the kernel forced each way on a 20 us single-loop pulse (eps = 50 kHz, x-COM)")
modes = two_ion_modes(dev)
wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=50e3, all_modes=True)
table = table_with_waveform((0, 1), wf, rabi_hz=rabi, stark_hz=stark)
sched = ms_schedule(wf, (0, 1), drives, table)
built = {
    k: build_hamiltonian(
        dev, list(sched.pulses), space, sample=quiet_sample(), options=BuilderOptions(kernel=k)
    )
    for k in ("assembled", "factorized", "auto")
}
print(
    f"  builder: kernel labels {dict((k, b.kernel) for k, b in built.items())}, one fingerprint {len({b.fingerprint for b in built.values()}) == 1}"
)
state = space.initial_state([0, 0])
finals = {}
for kernel in ("assembled", "factorized"):
    eng = JointExactEngine(builder_options=BuilderOptions(kernel=kernel))
    t0 = time.perf_counter()
    tr = eng.run_pulses(dev, sched, state, space, quiet_sample(), SeedSpec(0), SolverOptions())
    wall = time.perf_counter() - t0
    rep = eng.last_report
    assert rep is not None and tr.final.joint is not None
    finals[kernel] = tr.final.joint
    pops = np.real(np.diag(tr.final.internal.full()))
    print(
        f"  {kernel:11s}: populations (00, 01, 10, 11) {np.round(pops, 8).tolist()}, nbar {dict((m, round(v, 6)) for m, v in tr.final.motional.nbar.items() if m in (2, 3))}, segments {[s.kernel for s in rep.segments]}, integrator {rep.segments[0].integrator}"
    )
    print(f"MC: {kernel} wall {wall:.2f} s, right-hand sides {rep.segments[0].rhs_evaluations}")
print(f"  |psi_assembled - psi_factorized| = {(finals['assembled'] - finals['factorized']).norm():.1e}")

section("4. trajectories over 1 and N workers (Section 9.9)")
noisy = dataclasses.replace(
    dev,
    noise=dataclasses.replace(
        dev.noise, S_E=white_spectrum(1e-9, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
    ),
)
rates = noisy.noise.heating_rates_quanta_per_s(noisy)
print(f"  heating {dict((m, round(v, 1)) for m, v in rates.items() if m in (2, 3))} quanta/s on the resolved x modes")
n_workers = int(available_cpu_count())
out = {}
for mp, workers in (("serial", 1), ("parallel", n_workers)):
    eng = JointExactEngine(device_channels=True)
    t0 = time.perf_counter()
    tr = eng.run_pulses(
        noisy,
        sched,
        state,
        space,
        quiet_sample(),
        SeedSpec(11),
        # plain (uniform-weight) trajectories: the Bell caps of this fixture were sized for that ensemble; under
        # improved_sampling the conditioned members of this seed reach mean occupations of 4 to 6 quanta and the
        # Section 5.1.1 margin rule asks for dimension 1156 (tests/test_parallel.py covers that path on a smaller space)
        SolverOptions(lindblad_method="mcsolve", ntraj=6, map=mp, workers=workers, improved_sampling=False),
    )  # type: ignore[arg-type]
    wall = time.perf_counter() - t0
    rep = eng.last_report
    assert rep is not None
    out[mp] = (tr, rep)
    print(f"MC: map {mp} over {rep.workers} worker(s): wall {wall:.2f} s")
tr_s, rep_s = out["serial"]
tr_p, rep_p = out["parallel"]
per_traj = max((a - b).norm() for a, b in zip(rep_s.trajectory_finals, rep_p.trajectory_finals))
print(
    f"  six keyed trajectories, kernel {rep_s.kernel}: |rho_serial - rho_parallel| = {(tr_s.final.internal - tr_p.final.internal).norm():.1e}, max per-trajectory |psi_s - psi_p| = {per_traj:.1e}, jump records identical {tr_s.jumps == tr_p.jumps} ({len(tr_s.jumps)} jumps), seeds {rep_s.trajectory_seeds == rep_p.trajectory_seeds}"
)
print(f"  register populations serial {np.round(np.real(np.diag(tr_s.final.internal.full())), 8).tolist()}")

section("5. the propagator cache on an internal-state-only space (Section 11.3 item 5)")
pulse = single_qubit_pulse(
    0, math.pi / 2.0, 0.3, drives[0], rabi[(0, 0)], 0.0, gate_id="gpi2", programmed=False
)
carrier = Schedule((pulse,), (), (), {0: 0.0, 1: 0.0})
frozen_space = HilbertSpace((2, 2), (), None, (0, 1, 2, 3, 4, 5))
eng = JointExactEngine()
ref = JointExactEngine()
labels = []
worst = 0.0
for levels in ([0, 0], [1, 0], [0, 1], [1, 1]):
    st = frozen_space.initial_state(levels)
    tr = eng.run_pulses(dev, carrier, st, frozen_space, quiet_sample(), SeedSpec(0), SolverOptions())
    tr_ref = ref.run_pulses(
        dev, carrier, st, frozen_space, quiet_sample(), SeedSpec(0), SolverOptions(propagator_cache=False)
    )
    rep = eng.last_report
    assert rep is not None and tr.final.joint is not None and tr_ref.final.joint is not None
    labels.append(
        (
            rep.propagator_solves,
            rep.propagator_cache_hits,
            [s.integrator for s in rep.segments if s.pulses][0],
        )
    )
    worst = max(worst, (tr.final.joint - tr_ref.final.joint).norm())
print(
    f"  four inputs through one GPi2 segment: (solves, hits, integrator) per run {labels}; max |psi_cache - psi_ode| = {worst:.1e}"
)
model = MotionalModel(
    reduced={}, nbar={0: 0.0, 1: 0.0, 2: 0.05, 3: 0.05, 4: 0.0, 5: 0.0}, frozen=tuple(range(6))
)
eng2 = JointExactEngine()
rec = eng2.tomography(
    dev, pulse, frozen_space, model, quiet_sample(), SeedSpec(0), SolverOptions(branch_weight_min=0.02)
)
solves = sum(r.propagator_solves for r in rec.reports)
hits = sum(r.propagator_cache_hits for r in rec.reports)
print(
    f"  tomography of the carrier step: {rec.branches} frozen-mode branches x 16 inputs = {rec.engine_runs} engine runs, {solves} propagator integrations, {hits} cache hits; tp residual {rec.tp_residual:.1e}, cp residual {rec.cp_residual:.1e}"
)
print("\ndone")
