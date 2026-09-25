"""State-based process tomography, the Choi reconstruction, the CP/TP projection and the Kraus application (PLAN.md
Sections 5.4, 6.8) on synthetic channels and against the ideal unitaries the scheduler records (``GateTarget``), and the
isometry routes against the state route on the real engine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.calibration.entangling import ms_schedule
from qutip_trap.control import native
from qutip_trap.control.schedule import GateTarget
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import JointExactEngine, MotionalModel, SeedSpec, SolverOptions
from qutip_trap.dynamics.tomography import (
    MotionalBranch,
    apply_kraus_dm,
    apply_kraus_ket,
    choi_from_isometry,
    choi_least_squares,
    computational_labels,
    cp_residual,
    expansion_coefficients,
    ideal_unitary_on,
    input_states,
    internal_basis,
    keyed_tolerances,
    kraus_operators,
    kraus_superoperator,
    motional_branches,
    project_cptp,
    regrid_reduced,
    single_qudit_inputs,
    tp_residual,
)
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.noise.summary import (
    apply_choi,
    choi_from_kraus,
    choi_from_unitary,
    depolarizing_choi,
    entanglement_infidelity,
    pauli_twirl,
)
from tests.m4_fixtures import (
    X_COM_TWO_IONS,
    chain_device,
    derived_seeds,
    raman_gate_drives,
    table_with_waveform,
    two_ion_modes,
)


def _dms(dims: tuple[int, ...]) -> list[np.ndarray]:
    return [np.outer(k, k.conj()) for _lab, k in input_states(dims)]


def test_input_states_span_the_operator_space() -> None:
    """d^2 pure inputs per qudit, linearly independent over the complex numbers (the least squares needs full rank)."""
    for dims in ((2,), (2, 2), (3,), (2, 3)):
        states = input_states(dims)
        d = int(np.prod(dims))
        assert len(states) == d * d
        r = np.array([np.outer(k, k.conj()).reshape(-1) for _l, k in states])
        assert np.linalg.matrix_rank(r) == d * d
        assert all(abs(np.linalg.norm(k) - 1.0) < 1e-12 for _l, k in states)
    assert [lab for lab, _k in single_qudit_inputs(2)] == ["0", "1", "+", "+i"]
    assert computational_labels((2, 2)) == ["0,0", "0,1", "1,0", "1,1"]
    with pytest.raises(ValueError):
        single_qudit_inputs(1)


def test_unitary_channel_is_reconstructed_exactly_and_returns_one_kraus_operator() -> None:
    """Sixteen inputs through MS(0.3, -0.7, pi/2): the Choi matrix of ``noise.summary.choi_from_unitary`` to round-off, both
    residuals at round-off after the projection, one Kraus operator equal to the unitary up to a phase."""
    u = native.ms(0.3, -0.7, math.pi / 2.0)
    rhos = _dms((2, 2))
    outs = [u @ r @ u.conj().T for r in rhos]
    c = choi_least_squares(rhos, outs)
    assert np.max(np.abs(c - choi_from_unitary(u))) < 1e-13
    cp, cpr, tpr = project_cptp(c)
    assert cpr < 1e-13 and tpr < 1e-13
    ks = kraus_operators(cp)
    assert len(ks) == 1
    phase = ks[0][0, 0] / u[0, 0] if abs(u[0, 0]) > 1e-9 else ks[0][0, 3] / u[0, 3]
    assert abs(abs(phase) - 1.0) < 1e-12 and np.max(np.abs(ks[0] - phase * u)) < 1e-12
    # E(rho) from the Choi matrix agrees with U rho U^dag on a state outside the input set
    psi = np.array([0.6, 0.0, 0.0, 0.8j])
    rho = np.outer(psi, psi.conj())
    assert np.max(np.abs(apply_choi(cp, rho) - u @ rho @ u.conj().T)) < 1e-12


def test_dykstra_projection_restores_trace_preservation_and_positivity_under_noise() -> None:
    """The projection onto CP and TP leaves ||Tr_out(Choi) - 1|| < 1e-10 and reports both residuals; a PSD
    projection alone would not (the noisy reconstruction violates TP at the noise level)."""
    cd = depolarizing_choi(0.05, 2)
    rhos = _dms((2, 2))
    rng = np.random.default_rng(1)
    outs = []
    for r in rhos:
        o = apply_choi(cd, r) + 3e-4 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        outs.append(0.5 * (o + o.conj().T))
    raw = choi_least_squares(rhos, outs)
    assert tp_residual(raw) > 1e-4, "the noise breaks trace preservation"
    assert cp_residual(raw) >= 0.0
    cp, cpr, tpr = project_cptp(raw)
    assert tpr < 1e-10 and cpr < 1e-10, (cpr, tpr)
    assert np.max(np.abs(cp - cd)) < 5e-3
    assert abs(np.trace(cp) - 1.0) < 1e-12
    w = np.linalg.eigvalsh(cp)
    assert w.min() > -1e-12
    # the summary numbers of Section 6.8 on the projected matrix: the depolarizing rate IS the entanglement infidelity
    eps = entanglement_infidelity(cp, choi_from_unitary(np.eye(4, dtype=complex)))
    assert eps == pytest.approx(0.05, abs=2e-3)
    twirl = pauli_twirl(cp, 2)
    assert twirl["II"] == pytest.approx(0.95, abs=2e-3)
    assert all(abs(v - 0.05 / 15.0) < 2e-3 for k, v in twirl.items() if k != "II")


def test_kraus_application_on_a_register_matches_the_embedded_unitary() -> None:
    """The map on factors (0, 2) of a three-qubit register, on a density matrix and by Kraus sampling on a ket (a unitary has
    one Kraus operator, so the sample is deterministic), against the embedded unitary."""
    u = native.ms(0.3, -0.7, math.pi / 2.0)
    ks = kraus_operators(choi_from_unitary(u))
    psi = np.zeros(8, dtype=complex)
    psi[0] = psi[7] = 1.0 / math.sqrt(2.0)
    rho = np.outer(psi, psi.conj())
    u3 = ideal_unitary_on([((0, 2), u)], (0, 1, 2), (2, 2, 2))
    assert np.max(np.abs(u3.conj().T @ u3 - np.eye(8))) < 1e-12
    out = apply_kraus_dm(rho, ks, (2, 2, 2), (0, 2))
    assert np.max(np.abs(out - u3 @ rho @ u3.conj().T)) < 1e-12
    assert np.trace(out).real == pytest.approx(1.0, abs=1e-12)
    v, a = apply_kraus_ket(psi, ks, (2, 2, 2), (0, 2), np.random.default_rng(0))
    assert a == 0
    ref = u3 @ psi
    phase = v[np.argmax(np.abs(ref))] / ref[np.argmax(np.abs(ref))]
    assert np.max(np.abs(v - phase * ref)) < 1e-12
    # the embedding on a subset in reversed order: the first listed ion is the first factor of the gate matrix
    u_rev = ideal_unitary_on([((2, 0), u)], (0, 1, 2), (2, 2, 2))
    swap = np.zeros((4, 4))
    swap[0, 0] = swap[3, 3] = swap[1, 2] = swap[2, 1] = 1.0
    assert np.max(np.abs(u_rev - ideal_unitary_on([((0, 2), swap @ u @ swap)], (0, 1, 2), (2, 2, 2)))) < 1e-12
    # a depolarizing channel is applied with its full Kraus set and stays trace one
    kd = kraus_operators(depolarizing_choi(0.1, 2))
    assert len(kd) == 16
    out_d = apply_kraus_dm(rho, kd, (2, 2, 2), (1, 2))
    assert np.trace(out_d).real == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.eigvalsh(out_d).min() > -1e-12


def test_expansion_coefficients_and_regridding() -> None:
    rhos = _dms((2, 2))
    target = 0.3 * rhos[3] + 0.7 * rhos[7]
    c = expansion_coefficients(target, rhos)
    assert c == pytest.approx(np.eye(16)[3] * 0.3 + np.eye(16)[7] * 0.7, abs=1e-12)
    # a generic Hermitian matrix: the coefficients are real and reproduce it
    rng = np.random.default_rng(2)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = 0.5 * (h + h.conj().T)
    c = expansion_coefficients(h, rhos)
    assert np.max(np.abs(sum(ck * r for ck, r in zip(c, rhos)) - h)) < 1e-10
    rho_m = np.diag([0.7, 0.2, 0.1]).astype(complex)
    padded, dropped = regrid_reduced(rho_m, 5)
    assert padded.shape == (5, 5) and dropped == 0.0 and padded[1, 1] == 0.2
    cut, dropped2 = regrid_reduced(rho_m, 2)
    assert dropped2 == pytest.approx(0.1) and np.trace(cut).real == pytest.approx(1.0)


def test_gate_target_unitary_follows_the_virtual_z_rule() -> None:
    """The ideal physical unitary of a played gate: the native gate at its frame-applied phase, then RZ(-theta) for the Stark
    frame the scheduler absorbed (Section 7.6: a virtual RZ(theta) leaves the state as RZ(-theta) times the ideal one)."""
    t = GateTarget("gpi2[0]", (1,), ("gpi2", (0.4,)), {1: 0.0}, ("gpi2[0]",), 0.0, 1e-6)
    assert np.max(np.abs(t.unitary() - native.gpi2(0.4))) < 1e-14
    t2 = GateTarget("gpi2[0]", (1,), ("gpi2", (0.4,)), {1: 0.25}, ("gpi2[0]",), 0.0, 1e-6)
    assert np.max(np.abs(t2.unitary() - native.rz(-0.25) @ native.gpi2(0.4))) < 1e-14
    ms = GateTarget(
        "ms[2]", (0, 1), ("ms", (0.1, -0.2, math.pi / 2)), {0: 0.05, 1: -0.03}, ("a", "b"), 0.0, 1e-4
    )
    expected = np.kron(native.rz(-0.05), native.rz(0.03)) @ native.ms(0.1, -0.2, math.pi / 2)
    assert np.max(np.abs(ms.unitary() - expected)) < 1e-14
    zz = GateTarget("zz[3]/loop1", (0, 1), ("zz", (0.3,)), {}, ("c",), 0.0, 1e-4)
    assert np.max(np.abs(zz.unitary() - native.zz(0.3))) < 1e-14


# ---- the isometry routes -----------------------------------------------------------------------------------------------------------


def _random_isometry(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _r = np.linalg.qr(rng.normal(size=(rows, cols)) + 1j * rng.normal(size=(rows, cols)))
    return np.asarray(q[:, :cols])


def _apply_kraus_reference(
    rho: np.ndarray, kraus: list[np.ndarray], dims: tuple[int, ...], factors: tuple[int, ...]
) -> np.ndarray:
    """One three-operand einsum per Kraus operator on the (local, rest, local, rest) reshaped register."""
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


def test_choi_from_isometry_is_the_kraus_construction_and_trace_preserving() -> None:
    """A (d_int d_mot) x d_int isometry sliced by motional output index gives d_mot Kraus operators; ``choi_from_isometry`` forms
    their trace-1 Choi matrix in ``choi_from_kraus``'s convention without materializing them, and a full-rank isometry is exactly
    trace preserving (Stinespring). The internal basis is the identity's columns in the computational order."""
    d_int, d_mot = 4, 3
    v = _random_isometry(d_int * d_mot, d_int, seed=3)
    kraus = [v.reshape(d_int, d_mot, d_int)[:, a, :] for a in range(d_mot)]
    assert np.max(np.abs(sum(k.conj().T @ k for k in kraus) - np.eye(d_int))) < 1e-13
    c = choi_from_isometry(v, d_int)
    assert np.max(np.abs(c - choi_from_kraus(kraus))) < 1e-14
    assert abs(np.trace(c) - 1.0) < 1e-13 and tp_residual(c) < 1e-12 and cp_residual(c) < 1e-13
    # d_mot = 1: a unitary's Choi state
    u = native.ms(0.3, -0.7, math.pi / 2.0)
    assert np.max(np.abs(choi_from_isometry(u, 4) - choi_from_unitary(u))) < 1e-14
    basis = internal_basis((2, 3))
    assert len(basis) == 6 and all(np.array_equal(b, np.eye(6)[:, j]) for j, b in enumerate(basis))
    assert len(basis) == len(computational_labels((2, 3)))
    with pytest.raises(ValueError):
        choi_from_isometry(v, 3)


def test_superoperator_kraus_application_matches_the_per_operator_sum_with_a_permuted_factor_order() -> None:
    """``apply_kraus_dm`` as one superoperator product against a per-operator einsum on a six-qubit register, the local
    factors in a non-trivial order (4, 1), for a random CPTP set of four Kraus operators and for a unitary; ``kraus_superoperator``
    of a unitary is U (x) conj(U) and its action on a vectorized state is the sandwich."""
    rng = np.random.default_rng(5)
    dims = (2,) * 6
    total = 2**6
    psi = rng.normal(size=total) + 1j * rng.normal(size=total)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    q = _random_isometry(16, 4, seed=7)
    kraus = [q[4 * a : 4 * a + 4, :] for a in range(4)]
    assert np.max(np.abs(sum(k.conj().T @ k for k in kraus) - np.eye(4))) < 1e-13
    for factors in ((4, 1), (1, 4), (0, 5), (2, 3)):
        out = apply_kraus_dm(rho, kraus, dims, factors)
        ref = _apply_kraus_reference(rho, kraus, dims, factors)
        assert np.max(np.abs(out - ref)) < 1e-14, factors
        assert abs(np.trace(out) - 1.0) < 1e-12
    u = native.ms(0.3, -0.7, math.pi / 2.0)
    s = kraus_superoperator([u])
    assert np.max(np.abs(s - np.kron(u, u.conj()))) < 1e-15
    r = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    assert np.max(np.abs((s @ r.reshape(-1)).reshape(4, 4) - u @ r @ u.conj().T)) < 1e-13
    with pytest.raises(ValueError):
        kraus_superoperator([])


@pytest.fixture(scope="module")
def one_mode_entangling():  # type: ignore[no-untyped-def]
    """A 10 us single-loop entangling pulse on the two-ion chain with the x-COM resolved at d = 15 and the stretch mode frozen but
    coupled, so the thermal input makes three motional branches at nbar = 0.05 (weight floor 0.02). The cap has headroom on
    both routes: at d = 13 the sixteen-input route's superpositions trip the Section 5.5 margin check one level above the basis
    kets (a superposition's Fock tail is up to d_int times a basis ket's), and the two routes would end on different spaces."""
    dev = chain_device(2)
    drives = raman_gate_drives(2)
    rabi, stark = derived_seeds(dev, drives)
    modes = two_ion_modes(dev)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=100e3, all_modes=True)
    table = table_with_waveform((0, 1), wf, rabi_hz=rabi, stark_hz=stark)
    sched = ms_schedule(wf, (0, 1), drives, table)
    space = HilbertSpace((2, 2), (ModeTruncation(2, 15, (0, 3), 0.13),), None, (0, 1, 3, 4, 5))
    model = MotionalModel(
        reduced={}, nbar={0: 0.0, 1: 0.0, 2: 0.05, 3: 0.05, 4: 0.0, 5: 0.0}, frozen=(0, 1, 3, 4, 5)
    )
    return dev, sched, space, model


def test_isometry_route_matches_the_state_route_on_a_resolved_space(one_mode_entangling) -> None:  # type: ignore[no-untyped-def]
    """The channel read off the four propagated basis kets per branch (Stinespring) against the sixteen-input least-squares
    reconstruction on the same space: the Choi matrices, the outputs, the reduced motional states and the residual displacements
    agree to the solver tolerance with four times fewer engine runs; the isometry's raw Choi matrix is completely positive by
    construction and trace preserving to the norm the columns lost, and the projection has almost nothing left to do."""
    dev, sched, space, model = one_mode_entangling
    recs = {}
    for flag in (True, False):
        eng = JointExactEngine()
        opts = SolverOptions(
            branch_weight_min=0.02,
            map="serial",
            tomography_isometry=flag,
            tomography_dropped_weight_max=0.0,
            tomography_tolerance_keyed=False,
        )
        recs[flag] = eng.tomography(dev, sched, space, model, quiet_sample(), SeedSpec(0), opts)
    iso, ref = recs[True], recs[False]
    assert iso.route == "isometry" and ref.route == "states"
    assert iso.tolerances == ref.tolerances == (1e-10, 1e-8) and iso.tolerance_change is None
    assert iso.branches == ref.branches == 3
    assert iso.engine_runs == 4 * iso.branches and ref.engine_runs == 16 * ref.branches
    assert iso.space == ref.space == space, "the same space: no growth on either route"
    assert iso.labels == ref.labels and len(iso.inputs) == len(ref.inputs) == 16
    assert np.max(np.abs(iso.choi - ref.choi)) < 1e-7
    assert np.max(np.abs(iso.choi_raw - ref.choi_raw)) < 1e-7
    assert max(np.max(np.abs(a - b)) for a, b in zip(iso.outputs, ref.outputs)) < 1e-7
    for m in (2,):
        assert max(np.max(np.abs(a - b)) for a, b in zip(iso.motional_out[m], ref.motional_out[m])) < 1e-7
        assert max(abs(a - b) for a, b in zip(iso.alpha_out[m], ref.alpha_out[m])) < 1e-7
    assert set(iso.motional_out) == set(ref.motional_out) == {2}
    assert iso.residual_displacement().keys() == ref.residual_displacement().keys()
    assert abs(iso.residual_displacement()[2] - ref.residual_displacement()[2]) < 1e-7
    # CP by construction, TP to the columns' norm loss; the least-squares fit needs the projection for both
    assert cp_residual(iso.choi_raw) < 1e-12 and tp_residual(iso.choi_raw) < 1e-6
    assert iso.cp_residual < 1e-12 and iso.tp_residual < 1e-12
    assert iso.n_traj == 1 and iso.method == "sesolve" and set(iso.boundary_population) == {2}
    assert iso.margin_reached.keys() == ref.margin_reached.keys()
    assert any("Stinespring" in n for n in iso.notes) and not any("Stinespring" in n for n in ref.notes)
    assert len(iso.reports) == iso.engine_runs and all(r.method == "sesolve" for r in iso.reports)
    # the record's derived quantities keep working on the isometry route
    ks = iso.kraus()
    assert 1 <= len(ks) <= 16 and np.max(np.abs(sum(k.conj().T @ k for k in ks) - np.eye(4))) < 1e-9
    rho_local = np.asarray(iso.inputs[5])
    mot_iso, alpha_iso = iso.motional_for(rho_local)
    mot_ref, alpha_ref = ref.motional_for(rho_local)
    assert np.max(np.abs(mot_iso[2] - mot_ref[2])) < 1e-7 and abs(alpha_iso[2] - alpha_ref[2]) < 1e-7


def test_a_dissipative_step_keeps_the_state_route_whatever_the_switch_says() -> None:
    """A trajectory is not linear in its initial ket: with a dephasing collapse operator on the space ``is_unitary`` is False,
    the tomography propagates every one of the sixteen inputs even though the switch is on, and ``propagator`` refuses the
    segment. Heating acts on modes, so the device's channels leave an internal-state-only space unitary and a space with a
    resolved mode not."""
    import dataclasses

    from qutip_trap.control.schedule import Schedule, single_qubit_pulse
    from qutip_trap.dynamics.channels import qubit_dephasing_channels
    from qutip_trap.noise.spectra import white_spectrum

    dev = chain_device(2)
    drives = raman_gate_drives(2)
    rabi, _stark = derived_seeds(dev, drives)
    pulse = single_qubit_pulse(
        0, math.pi / 2.0, 0.3, drives[0], rabi[(0, 0)], 0.0, gate_id="gpi2", programmed=False
    )
    sched = Schedule((pulse,), (), (), {0: 0.0, 1: 0.0})
    space = HilbertSpace((2, 2), (), None, (0, 1, 2, 3, 4, 5))
    model = MotionalModel(reduced={}, nbar={m: 0.0 for m in range(6)}, frozen=tuple(range(6)))
    opts = SolverOptions(lindblad_method="mesolve")
    assert JointExactEngine().is_unitary(dev, space, opts)
    dephasing = JointExactEngine(channels=qubit_dephasing_channels(space, {0: 1.0e3}))
    assert not dephasing.is_unitary(dev, space, opts)
    with pytest.raises(ValueError, match="collapse operators"):
        dephasing.propagator(dev, sched, space, quiet_sample(), SeedSpec(0), opts)
    rec = dephasing.tomography(dev, pulse, space, model, quiet_sample(), SeedSpec(0), opts)
    assert rec.route == "states" and rec.method == "mesolve" and rec.engine_runs == 16 * rec.branches == 16
    assert rec.tp_residual < 1e-10 and len(rec.kraus()) > 1, "dephasing: more than one Kraus operator"
    noisy = dataclasses.replace(
        dev,
        noise=dataclasses.replace(
            dev.noise, S_E=white_spectrum(1e-9, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    heating = JointExactEngine(device_channels=True)
    assert heating.is_unitary(noisy, space, opts), "heating acts on modes, and every mode is frozen here"
    resolved = HilbertSpace((2, 2), (ModeTruncation(2, 6, (0, 1), 0.13),), None, (0, 1, 3, 4, 5))
    assert not heating.is_unitary(noisy, resolved, opts)
    with pytest.raises(ValueError, match="internal-state-only"):
        heating.propagator(noisy, sched, resolved, quiet_sample(), SeedSpec(0), opts)


# ---- the declared relaxations: the tail rule and the keyed tolerance -------------------------------------------------------------


def test_the_tail_rule_drops_the_lightest_branches_inside_its_budget_and_reports_twice_the_weight() -> None:
    """``motional_branches`` with ``dropped_weight_max``: the lightest branches go one at a time while the total dropped weight
    stays inside the budget, the survivors are renormalized, the dropped weight is what the record turns into the bound 2w, a
    budget of zero changes nothing, and one branch always survives."""
    space = HilbertSpace((2, 2), (ModeTruncation(2, 8, (0, 3), 0.13),), None, (0, 1, 3))
    model = MotionalModel(
        reduced={}, nbar={0: 0.0, 1: 0.0, 2: 0.3, 3: 0.4}, frozen=(0, 1, 3)
    )  # a warm resolved mode and a warm frozen coupled mode: many branches
    full, dropped_full, _ = motional_branches(space, model, [3], 1e-6)
    same, dropped_same, _ = motional_branches(space, model, [3], 1e-6, dropped_weight_max=0.0)
    assert len(same) == len(full) and dropped_same == dropped_full
    tail, dropped_tail, notes = motional_branches(space, model, [3], 1e-6, dropped_weight_max=2.5e-4)
    assert len(tail) < len(full) and dropped_full <= dropped_tail <= 2.5e-4
    assert abs(sum(b.weight for b in tail) - 1.0) < 1e-12
    assert all(isinstance(b, MotionalBranch) for b in tail)
    # the survivors are the heaviest of the full set, renormalized by the kept weight
    kept = sum(b.weight for b in full[: len(tail)])
    for a, b in zip(tail, full[: len(tail)]):
        assert a.frozen_n == b.frozen_n and a.weight == pytest.approx(b.weight / kept, rel=1e-12)
    assert any("tail rule" in n and "2w" in n for n in notes)
    # the budget is honoured exactly: adding the next-lightest dropped branch would exceed it
    # the rule drops from the lightest up, so the branch it refused is the lightest survivor
    next_weight = full[len(tail) - 1].weight * (1.0 - dropped_full)
    assert dropped_tail + next_weight > 2.5e-4
    # an enormous budget still leaves one branch
    one, dropped_one, _ = motional_branches(space, model, [3], 1e-6, dropped_weight_max=0.999)
    assert len(one) == 1 and one[0].weight == 1.0 and dropped_one < 1.0


def test_keyed_tolerances_follow_the_map_accuracy_and_never_override_a_chosen_tolerance() -> None:
    assert keyed_tolerances(SolverOptions()) == pytest.approx((1e-8, 1e-6))
    assert keyed_tolerances(SolverOptions(map_accuracy=1e-4)) == pytest.approx((1e-9, 1e-7))
    assert keyed_tolerances(SolverOptions(tomography_tolerance_keyed=False)) is None
    # a caller's atol is kept, the default rtol still keyed; both chosen: nothing moves
    assert keyed_tolerances(SolverOptions(atol=1e-12)) == (1e-12, 1e-6)
    assert keyed_tolerances(SolverOptions(atol=1e-12, rtol=1e-9)) is None
    assert keyed_tolerances(SolverOptions(atol=1e-8, rtol=1e-6)) is None


def test_keyed_tolerance_is_reported_with_its_convergence_change_and_stays_inside_the_map_accuracy(
    one_mode_entangling,
) -> None:  # type: ignore[no-untyped-def]
    """The default extraction of a unitary step with a resolved mode integrates at the map-accuracy-keyed tolerance, reports the
    pair and the ten-times-tighter change of the dominant branch (a bound on the diamond-norm change) and agrees with the
    engine-tolerance extraction to well inside the map accuracy; the tail rule reports 2w and every term is in the record."""
    dev, sched, space, model = one_mode_entangling
    eng = JointExactEngine()
    keyed = eng.tomography(
        dev,
        sched,
        space,
        model,
        quiet_sample(),
        SeedSpec(0),
        # the default floor: the tail rule is what limits the branches here (a floor of 0.02 would drop more than the budget
        # on its own and the rule would then add nothing)
        SolverOptions(map="serial"),
    )
    ref = JointExactEngine().tomography(
        dev,
        sched,
        space,
        model,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(map="serial", tomography_tolerance_keyed=False),
    )
    assert keyed.route == ref.route == "isometry"
    assert keyed.tolerances == (1e-8, 1e-6) and ref.tolerances == (1e-10, 1e-8)
    assert keyed.tolerance_change is not None and 0.0 < keyed.tolerance_change < 1e-3 / 4
    assert ref.tolerance_change is None
    assert keyed.engine_runs == ref.engine_runs + 4, "the probe adds the dominant branch's four columns"
    assert len(keyed.reports) == keyed.engine_runs
    assert np.max(np.abs(keyed.choi - ref.choi)) < 1e-4
    assert max(np.max(np.abs(a - b)) for a, b in zip(keyed.outputs, ref.outputs)) < 1e-4
    assert any("keyed to the map accuracy" in n for n in keyed.notes)
    # the branch terms: the tail rule at the default budget (map_accuracy / 4) and the reported 2w
    assert keyed.branch_error_bound == pytest.approx(2.0 * keyed.dropped_branch_weight)
    assert keyed.dropped_branch_weight <= 1e-3 / 4
    assert any("channel error bound 2w" in n for n in keyed.notes)
    zero = JointExactEngine().tomography(
        dev,
        sched,
        space,
        model,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(map="serial", tomography_dropped_weight_max=0.0),
    )
    assert zero.branches > keyed.branches and zero.dropped_branch_weight < keyed.dropped_branch_weight
    assert (
        zero.branch_error_bound == pytest.approx(2.0 * zero.dropped_branch_weight)
        and zero.branch_error_bound < 1e-5
    )
    assert any("dropped by the tail rule" in n for n in keyed.notes)
    assert not any("dropped by the tail rule" in n for n in zero.notes)
