"""The exact rotating frame of the ket integrations (PLAN.md Sections 5.2, 5.3): psi = e^{-i H_0 t} phi is a change of
variable, so every test is an exactness test (the phased sum against the assembled matrix, V_I(t) against
Theta^dag [H(t) - H_0] Theta on the real builder's QobjEvo) plus the engine-level statement that both pictures give the same
state to the solver tolerance, on the sesolve and the mcsolve path.
"""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest
import qutip as qt

from qutip_trap.calibration.entangling import ms_schedule
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.dynamics.kernels import FactorizedOperator, factorized_qobj
from qutip_trap.dynamics.rotating import (
    FrameEnergies,
    PhasedSum,
    RotatingDrive,
    eigen_frequency,
    frame_energies_of,
    kronecker_energies,
    rotating_collapse,
    rotating_frame,
)
from qutip_trap.hilbert.operators import displacement_operator, qudit_sigma_plus, qudit_sigma_z
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.noise.spectra import white_spectrum
from tests.m4_fixtures import (
    X_COM_TWO_IONS,
    chain_device,
    derived_seeds,
    raman_gate_drives,
    table_with_waveform,
    two_ion_modes,
)

CAPS = (ModeTruncation(2, 10, (0, 3), 0.13), ModeTruncation(3, 11, (0, 4), 0.13))


def _constant_element(t: float, op: qt.Qobj) -> qt.Qobj:
    """A function element (the form the rotating frame refuses)."""
    return op


def _random_vector(n: int, seed: int, ncols: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (n,) if ncols is None else (n, ncols)
    return np.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))


# ---- the frame's energies ------------------------------------------------------------------------------------------------------


def test_kronecker_energies_decompose_a_separable_diagonal_and_refuse_a_coupled_one() -> None:
    dims = (2, 3, 4)
    parts = [np.array([-0.5, 0.5]), 2.0 * np.arange(3) + 0.25, 3.0 * np.arange(4)]
    grid = parts[0][:, None, None] + parts[1][None, :, None] + parts[2][None, None, :]
    out = kronecker_energies(grid.reshape(-1), dims)
    assert out is not None
    recon = out[0][:, None, None] + out[1][None, :, None] + out[2][None, None, :]
    assert np.allclose(recon, grid, atol=1e-12)
    fe = FrameEnergies(dims, out)
    assert np.allclose(fe.energies, grid.reshape(-1))
    t = 0.37
    assert np.allclose(fe.theta(t), np.exp(-1j * grid.reshape(-1) * t), atol=1e-13)
    coupled = grid + 0.1 * np.arange(24).reshape(dims) ** 2  # a term that is not a sum over the factors
    assert kronecker_energies(coupled.reshape(-1), dims) is None
    # the Qobj form: a diagonal Kronecker sum has a frame, a coupling (an anharmonic X X X) has none, zero has none
    n1 = qt.num(3)
    h = qt.tensor(qt.sigmaz(), qt.qeye(3), qt.qeye(4)) + 2.0 * qt.tensor(qt.qeye(2), n1, qt.qeye(4))
    frame = frame_energies_of(h, dims)
    assert frame is not None and np.allclose(frame.energies, np.real(h.diag()))
    x1, x2 = qt.destroy(3) + qt.create(3), qt.destroy(4) + qt.create(4)
    assert frame_energies_of(h + 0.01 * qt.tensor(qt.qeye(2), x1, x2), dims) is None
    assert frame_energies_of(0.0 * h, dims) is None
    again = pickle.loads(pickle.dumps(frame))
    assert np.allclose(again.theta(t), frame.theta(t))
    psi = qt.Qobj(_random_vector(24, 1).reshape(-1, 1), dims=[list(dims), [1, 1, 1]])
    back = frame.from_frame(frame.to_frame(psi, t), t)
    assert (back - psi).norm() < 1e-14
    rho = psi.proj()
    assert (frame.from_frame(frame.to_frame(rho, t), t) - rho).norm() < 1e-12


def test_eigen_frequencies_of_the_standard_collapse_operators() -> None:
    dims = (2, 4)
    omega, delta = 2.0e6, 3.0e3
    # the ENERGY sigma_z = |1><1| - |0><0| of hilbert/operators.py (the negative of QuTiP's sigmaz) puts the upper level at
    # +delta/2, as H_int = (Delta/2) sigma_z does in the builder
    h = 0.5 * delta * qt.tensor(qudit_sigma_z(2), qt.qeye(4)) + omega * qt.tensor(qt.qeye(2), qt.num(4))
    frame = frame_energies_of(h, dims)
    assert frame is not None
    a = qt.tensor(qt.qeye(2), qt.destroy(4))
    assert eigen_frequency(a, frame.energies) == pytest.approx(-omega)
    assert eigen_frequency(a.dag(), frame.energies) == pytest.approx(omega)
    assert eigen_frequency(a.dag() * a, frame.energies) == 0.0
    assert eigen_frequency(qt.tensor(qudit_sigma_z(2), qt.qeye(4)), frame.energies) == 0.0
    # so sigma_+ = |1><0| rotates at +delta (E_row - E_col), the sign the drive's e^{-i mu t} beat note is measured against
    sp = qt.tensor(qudit_sigma_plus(2), qt.qeye(4))
    assert eigen_frequency(sp, frame.energies) == pytest.approx(delta)
    kick = qt.tensor(qudit_sigma_plus(2), displacement_operator(4, 0.1j))
    assert eigen_frequency(kick, frame.energies) is None


# ---- the data type -------------------------------------------------------------------------------------------------------------


def test_phased_sum_equals_its_assembled_matrix_through_the_data_layer() -> None:
    """Applied to kets and column stacks, through ``matmul`` on a Dense state, converted to Dense and back, and pickled."""
    dims = (2, 3, 4)
    n = 24
    d1 = displacement_operator(3, 0.2j).full()
    d2 = displacement_operator(4, 0.05j).full()
    a = factorized_qobj(dims, {0: qudit_sigma_plus(2).full(), 1: d1, 2: d2})
    b = qt.tensor(qt.sigmaz(), qt.num(3), qt.qeye(4)).to("CSR")
    theta = np.exp(-1j * np.linspace(0.0, 5.0, n))
    c1, c2 = 0.3 - 0.7j, 1.1 + 0.2j
    ps = PhasedSum(dims, [(c1, a.data), (c2, b.data)], theta)
    dense = (np.conj(theta)[:, None] * (c1 * a.full() + c2 * b.full())) * theta[None, :]
    assert np.max(np.abs(ps.to_array() - dense)) < 1e-13
    v = _random_vector(n, 3)
    assert np.max(np.abs(ps.apply(v) - dense @ v)) < 1e-13
    m = _random_vector(n, 4, ncols=3)
    assert np.max(np.abs(ps.apply(m) - dense @ m)) < 1e-13
    q = qt.Qobj(ps, dims=[list(dims), list(dims)], copy=False)
    assert q.dtype is PhasedSum
    ket = qt.Qobj(v.reshape(-1, 1), dims=[list(dims), [1, 1, 1]])
    assert np.max(np.abs((q @ ket).full().ravel() - dense @ v)) < 1e-13
    assert np.max(np.abs(q.to("dense").full() - dense)) < 1e-13
    again = pickle.loads(pickle.dumps(q))
    assert again.dtype is PhasedSum and np.max(np.abs(again.full() - dense)) == 0.0
    # a Dense converts into the type (the registry needs both directions) and back
    round_trip = qt.Qobj(dense, dims=q.dims).to("phasedsum")
    assert round_trip.dtype is PhasedSum and np.max(np.abs(round_trip.full() - dense)) < 1e-13


# ---- the real builder's QobjEvo in the frame ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ms_fixture():  # type: ignore[no-untyped-def]
    dev = chain_device(2)
    drives = raman_gate_drives(2)
    rabi, stark = derived_seeds(dev, drives)
    modes = two_ion_modes(dev)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=50e3, all_modes=True)
    table = table_with_waveform((0, 1), wf, rabi_hz=rabi, stark_hz=stark)
    sched = ms_schedule(wf, (0, 1), drives, table)
    space = HilbertSpace((2, 2), CAPS, None, (0, 1, 4, 5))
    return dev, sched, space


def test_rotating_frame_of_the_real_hamiltonian_is_theta_dag_v_theta(ms_fixture) -> None:  # type: ignore[no-untyped-def]
    dev, sched, space = ms_fixture
    pulses = list(sched.pulses)
    for kernel in ("factorized", "assembled"):
        built = build_hamiltonian(
            dev, pulses, space, sample=quiet_sample(), options=BuilderOptions(kernel=kernel)
        )
        a2 = space.annihilation(2)
        rot = rotating_frame(built.H, space.dims, [a2, space.sigma_z(1)])
        assert rot is not None
        h0 = built.H(0.0) - sum(
            (el[0] * el[1](0.0) for el in built.H.to_list() if isinstance(el, list)), 0 * built.H(0.0)
        )
        assert np.allclose(rot.frame.energies, np.real(h0.diag()))
        elements = rot.H.to_list()
        assert len(elements) == 1 and isinstance(elements[0][0], RotatingDrive), (
            "one function element carries every drive term (to_list spells it [callable, args])"
        )
        psi = space.initial_state([0, 1], fock={2: 1, 3: 0}).joint
        assert psi is not None
        for t in (0.0, 1.7e-6, 9.9e-6):
            theta = rot.frame.theta(t)
            v = psi.full().ravel()
            expected = np.conj(theta) * (
                built.H.matmul(t, qt.Qobj((theta * v).reshape(-1, 1), dims=psi.dims)).full().ravel()
                - rot.frame.energies * theta * v
            )
            got = rot.H.matmul(t, psi).full().ravel()
            assert np.max(np.abs(got - expected)) < 1e-12 * max(1.0, float(np.max(np.abs(expected))))
        # collapse operators: the heating ladder carries the phase, sigma_z is unchanged
        c_a, c_z = rot.c_ops
        assert isinstance(c_a, qt.QobjEvo) and c_z is space.sigma_z(1)
        omega2 = built.mode_frequencies_rad_s[2]
        t = 2.3e-6
        assert (c_a(t) - a2 * np.exp(-1j * omega2 * t)).norm() < 1e-9 * a2.norm()
        assert any("phase of the rotating frame" in n for n in rot.notes)
        # a recoil kick is not an eigenoperator of H_0: the segment keeps the Schroedinger picture (module docstring)
        kick = space.embed_many({0: qudit_sigma_plus(2), 2: displacement_operator(10, 0.1j)})
        assert rotating_frame(built.H, space.dims, [a2, kick]) is None
        assert rotating_collapse(kick, rot.frame) is None
        # what the frame refuses: a function element, an interaction-picture build (no static part)
        assert rotating_frame(built.H, space.dims, [qt.QobjEvo(_constant_element, args={"op": a2})]) is None
    # an interaction-picture build keeps only its constant Stark shift as a static part (a frame of that alone, exact but
    # pointless, is still a frame); without the Stark term the static part is exactly zero and there is nothing to rotate
    inter = build_hamiltonian(
        dev,
        pulses,
        space,
        sample=quiet_sample(),
        options=BuilderOptions(frame="interaction", include_stark=False),
    )
    assert rotating_frame(inter.H, space.dims, []) is None
    idle = build_hamiltonian(dev, [], space, sample=quiet_sample())
    assert rotating_frame(idle.H, space.dims, []) is None, (
        "a constant Hamiltonian is the closed form's, not the frame's"
    )
    # the element pickles with its coefficients and operators (the parallel maps), and its Qobj carries the space's dims
    blob = pickle.loads(pickle.dumps(rot.H))
    assert (blob.matmul(1e-6, psi) - rot.H.matmul(1e-6, psi)).norm() == 0.0
    drive = rot.H.to_list()[0][0]
    assert isinstance(drive, RotatingDrive) and drive(0.0).dims == [space.dims, space.dims]
    assert isinstance(rotating_collapse(space.number(3), rot.frame), qt.Qobj)


# ---- the engine ----------------------------------------------------------------------------------------------------------------


def test_engine_rotating_frame_reproduces_the_schrodinger_picture(ms_fixture, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    dev, sched, space = ms_fixture
    # mode 3 starts displaced (alpha = 0.5) so that <a_3> is not zero by the symmetry of the spin-dependent force and the
    # e^{-i omega_3 t} the engine restores on the way back from the frame is actually tested
    state = space.initial_state([0, 0], states={3: qt.coherent(space.truncation(3).d, 0.5)})
    traces = {}
    reports = {}
    for flag in (False, True):
        # 10 stored points: 20/9 us apart, 6.67 periods of the 3 MHz mode, so the stored coherences sit at non-trivial phases
        # (9 points would sample it at whole and half periods, where e^{-i omega t} is +-1)
        eng = JointExactEngine(store_per_segment=10)
        with monkeypatch.context() as m:
            if (
                not flag
            ):  # the Schroedinger-picture reference: the frame builder declines and the engine falls back
                m.setattr("qutip_trap.dynamics.engine.rotating_frame", lambda *args, **kwargs: None)
            traces[flag] = eng.run_pulses(
                dev, sched, state, space, quiet_sample(), SeedSpec(0), SolverOptions()
            )
        rep = eng.last_report
        assert rep is not None
        reports[flag] = rep
    seg_s, seg_r = reports[False].segments[0], reports[True].segments[0]
    assert seg_s.frame == "schrodinger" and seg_r.frame == "rotating"
    assert seg_s.integrator == seg_r.integrator == "dop853" and seg_r.kernel == seg_s.kernel == "factorized"
    assert (traces[True].final.joint - traces[False].final.joint).norm() < 2e-6
    assert (traces[True].final.internal - traces[False].final.internal).norm() < 1e-6
    for key in traces[True].expectations:
        assert np.max(np.abs(traces[True].expectations[key] - traces[False].expectations[key])) < 1e-6, key
    for m in traces[True].alpha_m:
        # the coherence <a_m> is NOT invariant under the frame: the engine puts its e^{-i omega_m t} back
        assert np.max(np.abs(traces[True].alpha_m[m] - traces[False].alpha_m[m])) < 1e-6, m
    assert abs(traces[True].alpha_m[3][4]) > 0.3, (
        "the displaced mode keeps its coherence, so the check is not vacuous"
    )
    assert abs(traces[True].alpha_m[3][4] - traces[True].alpha_m[3][0]) > 0.1, (
        "and that coherence has rotated"
    )
    for m in traces[True].final.motional.nbar:
        assert traces[True].final.motional.nbar[m] == pytest.approx(
            traces[False].final.motional.nbar[m], abs=1e-6
        )
    assert traces[True].boundary_population.keys() == traces[False].boundary_population.keys()


def test_engine_rotating_frame_on_the_trajectory_path_matches_per_trajectory(ms_fixture, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Heating channels present, ``mcsolve`` forced: the collapse operators carry their e^{i lambda t} in the frame, and the
    trajectories (same seeds) make the same jumps at the same times and end in the same ensemble, to the solver tolerance."""
    dev, sched, _space = ms_fixture
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
    finals = {}
    for flag in (False, True):
        eng = JointExactEngine(device_channels=True)
        with monkeypatch.context() as m:
            if (
                not flag
            ):  # the Schroedinger-picture reference: the frame builder declines and the engine falls back
                m.setattr("qutip_trap.dynamics.engine.rotating_frame", lambda *args, **kwargs: None)
            tr = eng.run_pulses(
                noisy,
                sched,
                state,
                space,
                quiet_sample(),
                SeedSpec(0),
                SolverOptions(
                    lindblad_method="mcsolve",
                    ntraj=3,
                    map="serial",
                    margin_check=False,
                    improved_sampling=False,
                ),
            )
        rep = eng.last_report
        assert rep is not None and rep.method == "mcsolve" and rep.trajectories == 3
        assert all(s.frame == ("rotating" if flag else "schrodinger") for s in rep.segments if s.pulses)
        finals[flag] = (tr.final.joint, tr.jumps, tr.final.internal, tr.expectations)
    assert (finals[False][0] - finals[True][0]).norm() < 1e-6
    assert [j[1] for j in finals[False][1]] == [j[1] for j in finals[True][1]]
    for (ta, _), (tb, _) in zip(finals[False][1], finals[True][1]):
        assert ta == pytest.approx(tb, abs=1e-9)
    assert (finals[False][2] - finals[True][2]).norm() < 1e-6
    for key in finals[True][3]:
        assert np.max(np.abs(finals[True][3][key] - finals[False][3][key])) < 1e-6


def test_kernel_apply_into_accumulates_and_matches_apply() -> None:
    dims = (2, 2, 5, 6)
    n = int(np.prod(dims))
    d1 = displacement_operator(5, 0.2j).full()
    d2 = displacement_operator(6, 0.1j).full()
    op = FactorizedOperator(dims, {1: qudit_sigma_plus(2).full(), 2: d1, 3: d2}, 0.4 - 0.1j)
    x = _random_vector(n, 8, ncols=2)
    out = _random_vector(n, 9, ncols=2)
    expected = out + (0.7 + 0.2j) * (op.to_array() @ x)
    op._plan.apply_into(np.ascontiguousarray(x), 0.7 + 0.2j, out)
    assert np.max(np.abs(out - expected)) < 1e-13
    assert np.max(np.abs(op.apply(x) - op.to_array() @ x)) < 1e-13
    # a pure slice (no mode factor) still carries its prefactor
    bare = FactorizedOperator((2, 3), {0: qudit_sigma_plus(2).full()}, 2.0j)
    v = _random_vector(6, 10)
    assert np.max(np.abs(bare.apply(v) - bare.to_array() @ v)) < 1e-14
    acc = np.zeros((6, 1), dtype=complex)
    bare._plan.apply_into(np.ascontiguousarray(v.reshape(-1, 1)), 0.5, acc)
    assert np.max(np.abs(acc.ravel() - 0.5 * (bare.to_array() @ v))) < 1e-14
