"""The mode-factorized drive kernel (PLAN.md Section 11.3): ``FactorizedOperator``, sigma_+^i (x) prod_m D_m held as its
factors, against the assembled operator, the builder's choice between them, the parallel map and the timing rows."""

from __future__ import annotations

import dataclasses
import math
import pickle
import time

import numpy as np
import pytest
import qutip as qt

from qutip_trap.calibration.entangling import ms_schedule
from qutip_trap.control.hardware import apply_hardware_chain
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.dynamics.kernels import (
    FactorizedOperator,
    factorized_qobj,
    kernel_costs_us,
    prefer_factorized,
)
from qutip_trap.dynamics.operators import displacement_operator, qudit_projector, qudit_sigma_plus
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.raman import lamb_dicke_parameters
from qutip_trap.noise.sampling import KEY_INTENSITY_TRAJECTORY, NoiseSample, quiet_sample
from qutip_trap.noise.spectra import white_spectrum
from tests.fixtures import (
    REALISTIC_HARDWARE,
    X_COM_TWO_IONS,
    chain_device,
    derived_seeds,
    raman_gate_drives,
    table_with_waveform,
    two_ion_modes,
)

BELL_CAPS = (ModeTruncation(2, 10, (0, 3), 0.13), ModeTruncation(3, 11, (0, 4), 0.13))
"""The Bell fixture's resolved x modes (dimension 440): the ``auto`` rule factorizes here."""


def _held_factorized(h: qt.QobjEvo) -> bool:
    """Whether every coefficient-bearing element of ``h`` (the drive terms) carries factorized data."""
    kinds = [isinstance(el[0].data, FactorizedOperator) for el in h.to_list() if isinstance(el, list)]
    return bool(kinds) and all(kinds)


def _random_ket(dims: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=int(np.prod(dims))) + 1j * rng.normal(size=int(np.prod(dims)))
    return np.asarray(v / np.linalg.norm(v))


# ---- the data type against the assembled matrix --------------------------------------------------------------------------------


def test_factorized_operator_equals_its_assembled_matrix_on_kets_matrices_and_adjoints() -> None:
    """With a sigma_+, projector or dense ion factor and two displacement factors the operator is its Kronecker product to 1e-14,
    on kets and matrices to 1e-13, with that matrix's adjoint and largest element; misfit factors are refused."""
    dims = (2, 2, 5, 6)
    d1 = displacement_operator(5, 0.1j).full()
    d2 = displacement_operator(6, 0.07j).full()
    ion_factors = {
        "sigma_plus": qudit_sigma_plus(2).full(),
        "projector": (0.3 * qudit_projector(2, 0) + 2.3 * qudit_projector(2, 1)).full(),
        "dense": np.array([[0.2, 1.1j], [0.7, -0.4]]),
    }
    v = _random_ket(dims, 1)
    m = np.column_stack([_random_ket(dims, k) for k in (2, 3, 4)])
    for label, ion in ion_factors.items():
        op = FactorizedOperator(dims, {1: ion, 2: d1, 3: d2}, scale=0.5 - 0.25j)
        full = op.to_array()
        ref = np.kron(np.kron(np.kron(np.eye(2), ion), d1), d2) * (0.5 - 0.25j)
        assert np.max(np.abs(full - ref)) < 1e-14, label
        assert np.max(np.abs(op.apply(v) - full @ v)) < 1e-13, label
        assert np.max(np.abs(op.apply(m) - full @ m)) < 1e-13, label
        assert np.max(np.abs(op.adjoint().to_array() - full.conj().T)) < 1e-14
        assert abs(op.max_abs - np.max(np.abs(full))) < 1e-14
    # a factor with no non-zero element annihilates everything
    zero = FactorizedOperator(dims, {1: np.zeros((2, 2)), 2: d1})
    assert np.all(zero.apply(v) == 0.0) and zero.max_abs == 0.0
    with pytest.raises(ValueError):
        FactorizedOperator(dims, {2: np.eye(4)})
    with pytest.raises(IndexError):
        FactorizedOperator(dims, {7: np.eye(2)})


def test_qutip_data_layer_registration_keeps_the_type_through_the_operations_the_solvers_use() -> None:
    """The type survives Qobj construction, dag, scaling, factor-wise products and pickling, compares structurally, converts
    and multiplies Dense states to 1e-13, and a sum falls back to Dense."""
    dims = (2, 3, 4)
    d1 = displacement_operator(3, 0.2j).full()
    d2 = displacement_operator(4, 0.05j).full()
    q = factorized_qobj(dims, {0: qudit_sigma_plus(2).full(), 1: d1, 2: d2})
    assert isinstance(q.data, FactorizedOperator) and q.dtype is FactorizedOperator and q.shape == (24, 24)
    qd = q.dag()
    assert isinstance(qd.data, FactorizedOperator)
    assert np.max(np.abs(qd.full() - q.full().conj().T)) < 1e-14
    assert isinstance((2.5j * q).data, FactorizedOperator) and (2.5j * q).data.scale == 2.5j
    herm = factorized_qobj(dims, {0: qt.sigmax().full(), 1: 0.5 * (d1 + d1.conj().T)})
    # equality on one factor structure is exact: the same operator, a scale moved between factors, a different operator
    assert q == factorized_qobj(dims, {0: qudit_sigma_plus(2).full(), 1: d1, 2: d2})
    assert q == factorized_qobj(dims, {0: 2.0 * qudit_sigma_plus(2).full(), 1: 0.5 * d1, 2: d2})
    assert q != qd and q != herm
    assert np.max(np.abs(q.to("dense").full() - q.full())) < 1e-14
    ket = qt.Qobj(_random_ket(dims, 5).reshape(-1, 1), dims=[list(dims), [1, 1, 1]])
    prod = qd * q
    assert isinstance(prod.data, FactorizedOperator), (
        "c^dag c of a factorized collapse operator stays factorized"
    )
    assert np.max(np.abs(prod.full() - qd.full() @ q.full())) < 1e-13
    assert (q + qd).dtype is qt.data.Dense, (
        "a sum has no factorized form: QuTiP converts (the documented fallback)"
    )
    # pickling: what the parallel maps need
    again = pickle.loads(pickle.dumps(q))
    assert isinstance(again.data, FactorizedOperator) and np.max(np.abs(again.full() - q.full())) == 0.0
    # matmul on a Dense state through the dispatcher, kets and column stacks
    out = q.data @ ket.data
    assert isinstance(out, qt.data.Dense) and np.max(np.abs(out.to_array() - q.full() @ ket.full())) < 1e-13
    cols = qt.data.Dense(np.column_stack([_random_ket(dims, k) for k in (6, 7)]))
    assert np.max(np.abs((q.data @ cols).to_array() - q.full() @ cols.to_array())) < 1e-13


def test_hilbert_space_factorized_drive_operator_is_the_assembled_one_and_refuses_an_enr_group() -> None:
    dev = chain_device(2)
    drives = raman_gate_drives(2)
    space = HilbertSpace((2, 2), BELL_CAPS, None, (0, 1, 4, 5))
    dk = np.asarray(dev.beams[drives[1].beams[0]].k_vector() - dev.beams[drives[1].beams[1]].k_vector())
    etas, _ = lamb_dicke_parameters(dev, 1, dk)
    active = {m: e for m, e in etas.items() if space.mode_class(m) != "frozen"}
    fact = space.drive_operator_factorized(1, active)
    assembled = space.drive_operator(1, active)
    assert isinstance(fact.data, FactorizedOperator) and fact.dims == assembled.dims
    assert np.max(np.abs(fact.full() - assembled.full())) < 1e-14
    assert space.drive_operator_factorized(1, active) is fact, "cached per (ion, etas)"
    ket = space.initial_state([0, 1], fock={2: 1, 3: 2}).joint
    assert ket is not None
    assert (fact * ket - assembled * ket).norm() < 1e-13
    arr = np.asarray(ket.full())
    assert np.max(np.abs(fact.data.apply(arr) - (assembled.full() @ arr))) < 1e-13
    # a light-shift force operator on the ion factor (diagonal): the same equality
    force = 0.3 * qudit_projector(2, 0) + 2.3 * qudit_projector(2, 1)
    f2 = space.drive_operator_factorized(0, active, ion_op=force)
    assert np.max(np.abs(f2.full() - space.drive_operator(0, active, ion_op=force).full())) < 1e-14
    enr = HilbertSpace((2,), (), ((1, 2), 6), (0,))
    with pytest.raises(NotImplementedError, match="sum-generator"):
        enr.drive_operator_factorized(0, {1: 0.1, 2: 0.05})


# ---- the builder's choice -------------------------------------------------------------------------------------------------------


def test_cost_model_reproduces_the_measured_crossover_of_section_11_1() -> None:
    """The cost model assembles at dimensions 48 and 256 and factorizes from the 440-dimensional Bell space up, never without a
    mode factor; the assembled cost at 2048 is 0.5 + 0.57e-3 x nnz us."""
    assert not prefer_factorized((2, 2, 12), 0, [2])
    assert not prefer_factorized((2, 2, 8, 8), 0, [2, 3])
    assert prefer_factorized((2, 2, 10, 11), 0, [2, 3])
    assert prefer_factorized((2, 2, 6, 6, 6), 0, [2, 3, 4])
    assert prefer_factorized((2, 2, 8, 8, 8), 0, [2, 3, 4])
    assert prefer_factorized((2, 2, 2, 2, 8, 8, 8), 3, [4, 5, 6])
    assert not prefer_factorized((2, 2, 8, 8, 8), 0, []), (
        "a carrier on an all-frozen space has no mode factor"
    )
    a48, f48 = kernel_costs_us((2, 2, 12), 0, [2])
    a2048, f2048 = kernel_costs_us((2, 2, 8, 8, 8), 0, [2, 3, 4])
    assert a48 < f48 and f2048 < a2048 / 10.0
    assert a2048 == pytest.approx(0.5 + 0.57e-3 * 524288, rel=1e-12)


@pytest.fixture(scope="module")
def ms_fixture():  # type: ignore[no-untyped-def]
    """The two-ion 171Yb+ fixture with a 20 us single-loop symmetric pulse on the x-COM (eps = 50 kHz), the Bell caps."""
    dev = chain_device(2)
    drives = raman_gate_drives(2)
    rabi, stark = derived_seeds(dev, drives)
    modes = two_ion_modes(dev)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=50e3, all_modes=True)
    table = table_with_waveform((0, 1), wf, rabi_hz=rabi, stark_hz=stark)
    sched = ms_schedule(wf, (0, 1), drives, table)
    space = HilbertSpace((2, 2), BELL_CAPS, None, (0, 1, 4, 5))
    return dev, drives, sched, space, table


def test_builder_kernel_option_auto_rule_and_report(ms_fixture) -> None:  # type: ignore[no-untyped-def]
    dev, _drives, sched, space, _table = ms_fixture
    pulses = list(sched.pulses)
    built = {
        k: build_hamiltonian(dev, pulses, space, sample=quiet_sample(), options=BuilderOptions(kernel=k))
        for k in ("assembled", "factorized", "auto")
    }
    assert built["assembled"].kernel == "assembled" and not _held_factorized(built["assembled"].H)
    assert built["factorized"].kernel == "factorized" and _held_factorized(built["factorized"].H)
    assert built["auto"].kernel == "factorized", "dimension 440: the cost model factorizes"
    assert built["auto"].fingerprint == built["assembled"].fingerprint == built["factorized"].fingerprint, (
        "the same H(t) whichever way its operators are held"
    )
    psi = space.initial_state([0, 1], fock={2: 1, 3: 0}).joint
    assert psi is not None
    for t in (0.0, 3.3e-6, 9.9e-6):
        a = built["assembled"].H.matmul(t, psi)
        f = built["factorized"].H.matmul(t, psi)
        assert (a - f).norm() < 1e-12 * a.norm()
    # a one-ion space with one mode stays assembled under auto; the interaction picture and the Lamb-Dicke expansion never factorize
    small = HilbertSpace((2, 2), (ModeTruncation(3, 12, (0, 3), 0.1),), None, (0, 1, 2, 4, 5))
    assert build_hamiltonian(dev, pulses, small, sample=quiet_sample()).kernel == "assembled"
    assert (
        build_hamiltonian(
            dev, pulses, small, sample=quiet_sample(), options=BuilderOptions(kernel="factorized")
        ).kernel
        == "factorized"
    )
    assert (
        build_hamiltonian(
            dev,
            pulses,
            space,
            sample=quiet_sample(),
            options=BuilderOptions(frame="interaction", kernel="factorized"),
        ).kernel
        == "assembled"
    )
    assert (
        build_hamiltonian(
            dev,
            pulses,
            space,
            sample=quiet_sample(),
            options=BuilderOptions(lamb_dicke_order=1, kernel="factorized"),
        ).kernel
        == "assembled"
    )
    # a drive coupling to an ENR group is assembled even when factorization is forced (Section 5.1.1)
    enr = HilbertSpace((2, 2), (), ((2, 3), 6), (0, 1, 4, 5))
    assert (
        build_hamiltonian(
            dev, pulses, enr, sample=quiet_sample(), options=BuilderOptions(kernel="factorized")
        ).kernel
        == "assembled"
    )
    # no drive term at all
    assert build_hamiltonian(dev, [], space, sample=quiet_sample()).kernel == "none"


# ---- the engine ------------------------------------------------------------------------------------------------------------------


def test_engine_factorized_and_assembled_kernels_give_the_same_entangling_pulse(ms_fixture) -> None:  # type: ignore[no-untyped-def]
    """The 20 us MS pulse with the kernel forced each way gives final states equal to 1e-9 and reports that name the kernel;
    ``auto`` picks factorized."""
    dev, _drives, sched, space, _table = ms_fixture
    state = space.initial_state([0, 0])
    finals = {}
    for kernel in ("assembled", "factorized", "auto"):
        eng = JointExactEngine(builder_options=BuilderOptions(kernel=kernel))
        tr = eng.run_pulses(dev, sched, state, space, quiet_sample(), SeedSpec(0), SolverOptions())
        finals[kernel] = tr.final.joint
        rep = eng.last_report
        assert rep is not None
        expected = "assembled" if kernel == "assembled" else "factorized"
        assert rep.kernel == expected and all(s.kernel == expected for s in rep.segments if s.pulses)
        assert rep.propagator_solves == 0 and rep.propagator_cache_hits == 0
    assert (finals["assembled"] - finals["factorized"]).norm() < 1e-9
    assert (finals["auto"] - finals["factorized"]).norm() == 0.0
    p11 = abs(finals["factorized"].full()[3 * 110 : 4 * 110].ravel()) ** 2
    assert 0.05 < float(np.sum(p11)) < 0.95, (
        "the single loop entangles the pair partially (chi below pi/4 at this closure)"
    )


def test_mesolve_segments_assemble_while_trajectory_segments_factorize(ms_fixture) -> None:  # type: ignore[no-untyped-def]
    """With heating a forced-factorized mesolve segment assembles and mcsolve keeps the factorized kernel, whose three seeded
    trajectories match the assembled ones (jumps, states to 1e-8) on a 64-dimensional space with a loose boundary threshold."""
    dev, _drives, sched, _space, _table = ms_fixture
    noisy = dataclasses.replace(
        dev,
        noise=dataclasses.replace(
            dev.noise, S_E=white_spectrum(1e-13, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    space = HilbertSpace(
        (2, 2), (ModeTruncation(2, 4, (0, 0), 0.13), ModeTruncation(3, 4, (0, 0), 0.13)), None, (0, 1, 4, 5)
    )
    state = space.initial_state([0, 0])
    eng_me = JointExactEngine(device_channels=True, builder_options=BuilderOptions(kernel="factorized"))
    eng_me.run_pulses(
        noisy,
        sched,
        state,
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(lindblad_method="mesolve", margin_check=False, boundary_population_max=1e-2),
    )
    rep_me = eng_me.last_report
    assert rep_me is not None and rep_me.growth_retries == 0, rep_me.notes
    assert rep_me is not None and rep_me.method == "mesolve" and rep_me.kernel == "assembled"
    finals = {}
    for kernel in ("assembled", "factorized"):
        eng = JointExactEngine(device_channels=True, builder_options=BuilderOptions(kernel=kernel))
        tr = eng.run_pulses(
            noisy,
            sched,
            state,
            space,
            quiet_sample(),
            SeedSpec(0),
            # improved_sampling off: three plain trajectories per kernel
            SolverOptions(
                lindblad_method="mcsolve",
                ntraj=3,
                map="serial",
                margin_check=False,
                improved_sampling=False,
                boundary_population_max=1e-2,
            ),
        )
        rep = eng.last_report
        assert rep is not None and rep.method == "mcsolve" and rep.kernel == kernel and rep.trajectories == 3
        assert rep.growth_retries == 0, rep.notes
        finals[kernel] = (tr.final.joint, tr.jumps, tr.final.internal)
    assert (finals["assembled"][0] - finals["factorized"][0]).norm() < 1e-8
    assert [j[1] for j in finals["assembled"][1]] == [j[1] for j in finals["factorized"][1]]
    assert (finals["assembled"][2] - finals["factorized"][2]).norm() < 1e-8


def test_the_real_builders_qobjevo_pickles_and_mcsolve_runs_it_under_the_parallel_map(ms_fixture) -> None:  # type: ignore[no-untyped-def]
    """The real builder's factorized ``QobjEvo`` with filtered envelopes and an intensity trajectory evaluates identically after
    pickling, and ``mcsolve`` under ``map="parallel"`` reproduces the serial trajectories to 1e-14."""
    dev, _drives, sched, space, _table = ms_fixture
    played, _notes = apply_hardware_chain(sched, REALISTIC_HARDWARE, rng=np.random.default_rng(0))
    grid = np.linspace(0.0, 40e-6, 401)
    traj = np.vstack([grid, 1e-3 * np.sin(2.0 * math.pi * 1e5 * grid)])
    sample = NoiseSample(0, {}, {KEY_INTENSITY_TRAJECTORY: traj})
    active = [p for p in played.pulses if p.t_start_s <= 1e-9]
    built = build_hamiltonian(dev, active, space, sample=sample, options=BuilderOptions(kernel="factorized"))
    assert built.kernel == "factorized"
    assert any(isinstance(t.envelope_hz, np.ndarray) for p in active for t in p.drive.tones), (
        "the chain sampled and filtered the envelopes"
    )
    blob = pickle.dumps(built.H)
    again = pickle.loads(blob)
    psi = space.initial_state([0, 0]).joint
    assert psi is not None
    for t in (1e-7, 2.2e-6, 7.7e-6):
        assert (again.matmul(t, psi) - built.H.matmul(t, psi)).norm() == 0.0
    a2 = space.annihilation(2)
    c_ops = [math.sqrt(1e3) * a2, math.sqrt(1e3) * a2.dag()]
    seeds = [SeedSpec(3).child(0, k, 0, 0, "test") for k in range(4)]
    times = [float(active[0].t_start_s), float(active[0].t_start_s) + 4e-6]
    results = {}
    for mp, n_cpu in (("serial", 1), ("parallel", 2)):
        opts = {
            "method": "dop853",
            "atol": 1e-10,
            "rtol": 1e-8,
            "nsteps": 10**7,
            "keep_runs_results": True,
            "store_final_state": True,
            "progress_bar": "",
            "norm_t_tol": 1e-8 * 4e-6,
            "norm_tol": 1e-6,
            "norm_steps": 50,
            "map": mp,
            "num_cpus": n_cpu,
        }
        res = qt.MCSolver(built.H, c_ops, options=opts).run(psi, times, ntraj=4, seeds=seeds)
        results[mp] = {tuple(s.spawn_key): tr.final_state for s, tr in zip(res.seeds, res.trajectories)}
    assert results["serial"].keys() == results["parallel"].keys()
    for key, ket in results["serial"].items():
        assert (ket - results["parallel"][key]).norm() < 1e-14, "per-trajectory identity under the same seed"


# ---- the timing acceptance rows ---------------------------------------------------------------------------------------------------

OMEGA = (2 * np.pi * 3.0e6, 2 * np.pi * 2.8284e6, 2 * np.pi * 2.9e6)
ETA = ((0.080, 0.080), (0.0824, -0.0824), (0.047, 0.047))
EPS = 2 * np.pi * 10e3


COEFFICIENT_CALLS = [0]


def _counted_coefficient(t: float, Om: float, mu: float, tag: object = None) -> float:
    COEFFICIENT_CALLS[0] += 1
    return float(Om * np.cos(mu * t))


# The four rows under dop853 at atol 1e-10, rtol 1e-8: (right-hand-side evaluations, factorized us per evaluation, CSR us
# per evaluation, factorized wall time in seconds) on the reference machine. The evaluation counts are set by the
# integrator's arithmetic; the costs and the wall time are the machine's.
REFERENCE_ROWS = {
    (1, 12): (19283, 13.2, 1.9, 0.25),
    (2, 8): (21067, 21.5, 16.7, 0.35),
    (3, 6): (24366, 45.4, 195.2, 1.11),
    (3, 8): (35177, 89.5, 1207.9, 3.15),
}


def _ms_hamiltonian(nmodes: int, nmax: int, factorized: bool) -> qt.QobjEvo:
    """The benchmark fixture: two ions, the bichromatic force on every mode, one distinct coefficient per drive term so
    that ``QobjEvo.compress`` merges nothing (the structure the real builder produces)."""
    dims = [2, 2] + [nmax] * nmodes
    a1 = qt.destroy(nmax)
    h0 = 0.0 * qt.qeye(dims)
    for m in range(nmodes):
        ops = [qt.qeye(2), qt.qeye(2)] + [qt.qeye(nmax)] * nmodes
        ops[2 + m] = a1.dag() * a1
        h0 = h0 + OMEGA[m] * qt.tensor(*ops)
    terms: list[object] = [h0.to("CSR")]
    mu = OMEGA[0] - EPS
    om = EPS / (2 * ETA[0][0])
    for i in range(2):
        d_m = [(1j * ETA[m][i] * (a1 + a1.dag())).expm() for m in range(nmodes)]
        if factorized:
            v = factorized_qobj(
                dims, {i: qt.sigmap().full(), **{2 + m: d_m[m].full() for m in range(nmodes)}}
            )
        else:
            ops = [qt.qeye(2), qt.qeye(2)]
            ops[i] = qt.sigmap()
            v = qt.tensor(*ops, *[d.to("CSR") for d in d_m]).to("CSR")
        for j, op in enumerate((v, v.dag())):
            terms.append([op, qt.coefficient(_counted_coefficient, args={"Om": om, "mu": mu, "tag": (i, j)})])
    return qt.QobjEvo(terms)


@pytest.mark.slow
@pytest.mark.heavy  # wall times against the reference machine's: measured alone, never beside other workers
def test_section_11_1_rows_factorized_against_assembled_final_states_and_wall_time() -> None:
    """On the four Section 11.1 rows both kernels reach the same state (2e-7) in the reference evaluation counts (20%), with the
    cost ratio within a factor of two, the factorized wall time within a factor of eight, and the crossover."""
    walls: dict[tuple[int, int, bool], float] = {}
    per_eval: dict[tuple[int, int, bool], float] = {}
    for nmodes, nmax in [(1, 12), (2, 8), (3, 6), (3, 8)]:
        finals = {}
        evals = {}
        psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1), *[qt.basis(nmax, 0)] * nmodes)
        for factorized in (False, True):
            h = _ms_hamiltonian(nmodes, nmax, factorized)
            COEFFICIENT_CALLS[0] = 0
            t0 = time.perf_counter()
            res = qt.sesolve(
                h,
                psi0,
                [0.0, 20e-6],
                options={
                    "method": "dop853",
                    "atol": 1e-10,
                    "rtol": 1e-8,
                    "nsteps": 10**7,
                    "store_final_state": True,
                },
            )
            wall = time.perf_counter() - t0
            # four drive terms, each with its own coefficient object: four calls per right-hand side
            evals[factorized] = COEFFICIENT_CALLS[0] // 4
            walls[(nmodes, nmax, factorized)] = wall
            per_eval[(nmodes, nmax, factorized)] = 1e6 * wall / max(evals[factorized], 1)
            finals[factorized] = res.final_state
        assert (finals[True] - finals[False]).norm() < 2e-7, (nmodes, nmax)
        n_ref, us_fact, us_csr, wall_fact = REFERENCE_ROWS[(nmodes, nmax)]
        # the two kernels take the same steps here; on the Linux runner the integrator's step choice differs by 8%
        assert evals[True] == pytest.approx(evals[False], rel=0.1), (nmodes, nmax, evals)
        assert evals[True] == pytest.approx(n_ref, rel=0.2), (nmodes, nmax, evals[True], n_ref)
        ratio = per_eval[(nmodes, nmax, True)] / per_eval[(nmodes, nmax, False)]
        assert ratio == pytest.approx(us_fact / us_csr, rel=1.0), (nmodes, nmax, ratio, us_fact / us_csr)
        assert 0.125 * wall_fact < walls[(nmodes, nmax, True)] < 8.0 * wall_fact, (
            nmodes,
            nmax,
            walls[(nmodes, nmax, True)],
            wall_fact,
        )
    assert walls[(3, 8, True)] < 0.5 * walls[(3, 8, False)], walls
    assert walls[(1, 12, False)] < walls[(1, 12, True)], walls
