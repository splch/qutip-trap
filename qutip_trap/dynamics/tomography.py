"""State-based process tomography, Choi reconstruction, CP/TP projection and Kraus application (PLAN.md Section 5.4, GATE_LOCAL
item (a); Section 6.8; Section 9.17 row "GATE_LOCAL tomography"; milestone M9a).

The GATE_LOCAL level extracts the reduced operation of a pulse on the addressed ions' internal states as a completely positive
trace-preserving map, from the pulse's exact evolution on its own joint space (Section 5.4):

- ``input_states``: prod_i d_i^2 linearly independent pure internal inputs (for a qubit |0>, |1>, |+>, |+i>: the sixteen
  product states of a pair), each propagated as a state vector by the JOINT_EXACT engine from the tracked motional state,
  through ``mesolve`` or ``mcsolve`` where channels are active (n_traj = ceil(1/epsilon_map) per input on the trajectory
  path, the map-accuracy rule; the populations of every input are the engine's ``e_ops``, and the trajectory count is fixed
  from the keyed seed list rather than ``target_tol``, which ``mcsolve`` refuses without ``e_ops``);
- ``choi_least_squares``: the Choi matrix from the input-output pairs, E(rho) = d Tr_1[(rho^T (x) 1) C] in the trace-1
  convention of ``noise/summary.py`` (the first factor the input copy), exact when the inputs span the operator space and the
  outputs are deterministic, least squares under trajectory noise;
- ``project_cptp``: Dykstra's alternating projection onto the intersection of the positive-semidefinite cone and the
  trace-preserving affine set Tr_out C = 1/d, reporting the residual against BOTH constraints, because a PSD projection alone
  leaves Tr_out(Choi) != 1 and a map that changes the register's normalization on every application (Section 5.4
  [corrected: critique, 2026-09-04]); the plan's target ||Tr_out(Choi) - 1|| < 1e-10 is stated in the trace-d normalization
  and ``tp_residual`` reports it there;
- ``kraus_operators``: the eigen-decomposition of the projected Choi matrix, applied to a density-matrix register on its local
  factors (``apply_kraus_dm``) or by Kraus SAMPLING to one member of a pure-state ensemble (``apply_kraus_ket``), the two
  register representations of Section 5.4;
- ``expansion_coefficients``: the register's local marginal expanded in the input basis, so that the motional outputs of the
  tomography runs (linear in the input) give the reduced motional state the register actually leaves (item (b)).

M9b: the engine runs of a step with resolved modes are independent and are spread over the workers of ``SolverOptions.workers``
through QuTiP's map (Section 11.3 item 9), each worker running its own trajectories in-process; a step on an internal-state-only
space runs in-process (Section 11.3 item 5).

Performance pass 2026-09-09 (``SolverOptions.tomography_isometry``, the default): when the step's evolution is unitary (no
collapse operator on any segment, ``JointExactEngine.is_unitary``) the final state is linear in the initial ket, so the channel is
read off the propagated INTERNAL BASIS instead of the prod_i d_i^2 input states. Per motional branch b (weight w_b, motional ket
|m_b>) the prod_i d_i basis kets |j> (x) |m_b> are propagated and their finals stacked into the Stinespring isometry V_b, whose
slices by motional output index are the Kraus operators K_{b,a} = sqrt(w_b) (<a|_mot (x) 1) V_b (``choi_from_isometry``); the
outputs, reduced motional states and residual displacements of the d_int^2 inputs the map is stated on follow by linearity
(``_extract_from_columns``), so ``TomographyRecord`` keeps its meaning. On an internal-state-only space (a carrier step, an idle)
the segment propagator IS the branch's Kraus operator and ``JointExactEngine.propagator`` returns it without propagating a state:
one engine call per branch. The three routes are named in ``TomographyRecord.route`` ("states" is the M9a reference, which a
dissipative step always takes) and agree to the solver tolerance (``tests/test_tomography.py``). The Choi matrix of the isometry
routes is completely positive by construction; the CP/TP projection is still applied so that the register's normalization stays
exact, and ``tp_residual(choi_raw)`` is the norm the propagated columns lost (the motional truncation's leakage, the solver's drift).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import qutip as qt

from qutip_trap.control.pulses import fingerprint_pulse
from qutip_trap.dynamics.engine import (
    ChannelSummary,
    EngineReport,
    MotionalModel,
    SeedSpec,
    SolverOptions,
    Traces,
)
from qutip_trap.dynamics.parallel import map_tasks, worker_count
from qutip_trap.hilbert.operators import thermal_populations
from qutip_trap.hilbert.truncation import boundary_populations
from qutip_trap.noise.sampling import KEY_BRANCH_WEIGHT, NoiseSample, key_frozen_n
from qutip_trap.noise.summary import (
    average_gate_infidelity,
    choi_from_kraus,
    choi_from_unitary,
    entanglement_infidelity,
    pauli_twirl,
)

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import GateTarget, Schedule
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import JointExactEngine
    from qutip_trap.hilbert.space import HilbertSpace

M9A = "milestone M9a (dynamics/tomography.py, PLAN.md Section 5.4)"

TomographyRoute = Literal["states", "isometry", "propagator"]
"""How a channel was extracted: every input state propagated ("states", the M9a reference and the dissipative route), the internal
basis propagated per branch and the channel read off the Stinespring isometry ("isometry"), or the segment propagator of an
internal-state-only space taken as the branch's Kraus operator ("propagator")."""


# ---- input states ----------------------------------------------------------------------------------------------------------------


def single_qudit_inputs(d: int) -> list[tuple[str, np.ndarray]]:
    """d^2 pure states spanning the Hermitian d x d matrices over the reals: |j>, (|j> + |k>)/sqrt 2 and (|j> + i|k>)/sqrt 2 for
    j < k; for a qubit |0>, |1>, |+>, |+i> (the sixteen product inputs of a pair, Section 5.4)."""
    if d < 2:
        raise ValueError("a register factor has at least two levels")
    out: list[tuple[str, np.ndarray]] = []
    for j in range(d):
        v = np.zeros(d, dtype=complex)
        v[j] = 1.0
        out.append((str(j), v))
    for j in range(d):
        for k in range(j + 1, d):
            v = np.zeros(d, dtype=complex)
            v[j] = v[k] = 1.0 / math.sqrt(2.0)
            out.append((f"+{j}{k}" if d > 2 else "+", v))
            w = np.zeros(d, dtype=complex)
            w[j] = 1.0 / math.sqrt(2.0)
            w[k] = 1j / math.sqrt(2.0)
            out.append((f"+i{j}{k}" if d > 2 else "+i", w))
    return out


def input_states(ion_dims: Sequence[int]) -> list[tuple[str, np.ndarray]]:
    """The product inputs over the ion factors (first ion the first tensor factor), prod_i d_i^2 of them, as kets."""
    singles = [single_qudit_inputs(int(d)) for d in ion_dims]
    out: list[tuple[str, np.ndarray]] = []
    for combo in itertools.product(*singles):
        label = ",".join(lab for lab, _v in combo)
        ket = np.array([1.0 + 0.0j])
        for _lab, v in combo:
            ket = np.kron(ket, v)
        out.append((label, ket))
    return out


def internal_basis(ion_dims: Sequence[int]) -> list[np.ndarray]:
    """The prod_i d_i computational-basis kets of the internal space (first ion the first tensor factor), in the order of
    ``computational_labels``: the columns the isometry route propagates (``SolverOptions.tomography_isometry``)."""
    d = int(np.prod([int(x) for x in ion_dims]))
    eye = np.eye(d, dtype=complex)
    return [eye[:, j].copy() for j in range(d)]


def computational_labels(ion_dims: Sequence[int]) -> list[str]:
    """The labels of the computational-basis inputs among ``input_states`` (the spin eigenstates of Section 5.4's report)."""
    return [
        ",".join(str(j) for j in combo) for combo in itertools.product(*[range(int(d)) for d in ion_dims])
    ]


# ---- Choi reconstruction and projection ----------------------------------------------------------------------------------------


def choi_least_squares(inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray]) -> np.ndarray:
    """The trace-1 Choi matrix C with E(rho) = d Tr_1[(rho^T (x) 1) C] from input density matrices and their outputs, by least
    squares over the linear system S = R M, M[(i,j),(k,l)] = d C[(i,k),(j,l)]; Hermitized."""
    if len(inputs) != len(outputs) or not inputs:
        raise ValueError("one output per input, at least one pair")
    d = int(inputs[0].shape[0])
    r = np.array([np.asarray(x, dtype=complex).reshape(-1) for x in inputs])
    s = np.array([np.asarray(y, dtype=complex).reshape(-1) for y in outputs])
    m, _res, rank, _sv = np.linalg.lstsq(r, s, rcond=None)
    if rank < d * d:
        raise ValueError(f"the {len(inputs)} inputs span only {rank} of the {d * d} operator dimensions")
    c4 = m.reshape(d, d, d, d)  # (i, j, k, l)
    c = np.transpose(c4, (0, 2, 1, 3)).reshape(d * d, d * d) / d  # (i, k ; j, l)
    return np.asarray(0.5 * (c + c.conj().T))


def choi_from_isometry(v: np.ndarray, d_int: int) -> np.ndarray:
    """The trace-1 Choi matrix of rho -> Tr_mot[V rho V^dag] for the (d_int d_mot) x d_int isometry V (Stinespring).

    The columns of V are the propagated internal basis kets U(T)(|j> (x) |m_b>) of one motional branch (the joint basis is
    ordered ions first, so a column reshapes to (internal out, motional out)), the Kraus operators K_a = (<a|_mot (x) 1) V are
    its slices by motional output index a, and C = (1/d_int) sum_a |phi_a><phi_a| with phi_a = sum_i |i> (x) K_a|i> is the
    convention of ``noise.summary.choi_from_kraus``, formed as one product without materializing the d_mot Kraus operators.
    Completely positive by construction; trace preserving up to the norm the columns lost (``tp_residual``)."""
    arr = np.asarray(v, dtype=complex)
    if arr.ndim != 2 or arr.shape[1] != d_int or arr.shape[0] % d_int != 0:
        raise ValueError(
            f"an isometry of shape {arr.shape} does not embed a {d_int}-dimensional internal space"
        )
    d_mot = arr.shape[0] // d_int
    t = arr.reshape(
        d_int, d_mot, d_int
    )  # (internal out k, motional out a, internal in i): K_a[k, i] = t[k, a, i]
    m = np.transpose(t, (1, 2, 0)).reshape(d_mot, d_int * d_int)  # row a is phi_a in the (i, k) order
    return np.asarray(m.T @ m.conj() / d_int)


def _trace_out(c: np.ndarray, d: int) -> np.ndarray:
    """Tr over the output factor: T[i, j] = sum_k C[(i, k), (j, k)]."""
    c4 = c.reshape(d, d, d, d)
    return np.asarray(np.einsum("ikjk->ij", c4))


def cp_residual(choi: np.ndarray) -> float:
    """||C - P_PSD(C)||_F: the Frobenius distance to the positive-semidefinite cone (the negative spectrum's norm)."""
    w = np.linalg.eigvalsh(0.5 * (choi + choi.conj().T))
    return float(math.sqrt(float(np.sum(np.minimum(w, 0.0) ** 2))))


def tp_residual(choi: np.ndarray) -> float:
    """||d Tr_out(C) - 1||_F in the trace-d normalization of Section 5.4 (Tr_out of the trace-1 C is 1/d for a TP map)."""
    d = int(round(math.sqrt(choi.shape[0])))
    return float(np.linalg.norm(d * _trace_out(choi, d) - np.eye(d)))


def project_psd(c: np.ndarray) -> np.ndarray:
    h = 0.5 * (c + c.conj().T)
    w, v = np.linalg.eigh(h)
    w = np.maximum(w, 0.0)
    return np.asarray((v * w) @ v.conj().T)


def project_tp(c: np.ndarray) -> np.ndarray:
    """The orthogonal projection onto {C : Tr_out C = 1/d}: C - (Tr_out C - 1/d) (x) 1/d."""
    d = int(round(math.sqrt(c.shape[0])))
    t = _trace_out(c, d) - np.eye(d) / d
    corr = np.kron(t, np.eye(d)) / d  # (i, k ; j, l) with the identity on the output factor
    return np.asarray(c - corr)


def project_cptp(
    choi: np.ndarray, *, tol: float = 1e-13, max_iter: int = 20000
) -> tuple[np.ndarray, float, float, int]:
    """Dykstra's alternating projection of ``choi`` onto CP (PSD) and TP (affine); returns (C, cp_residual, tp_residual,
    iterations) with the residuals of the returned matrix against both constraints (Section 5.4; Section 9.17)."""
    x = np.asarray(choi, dtype=complex)
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        y = project_psd(x + p)
        p = x + p - y
        x_new = project_tp(y + q)
        q = y + q - x_new
        delta = float(np.linalg.norm(x_new - x))
        x = x_new
        if delta < tol:
            break
    # end on the TP set (the register's normalization is exact) and report what is left against the cone
    x = project_tp(x)
    x = 0.5 * (x + x.conj().T)
    return x, cp_residual(x), tp_residual(x), iterations


def kraus_operators(choi: np.ndarray, *, tol: float = 1e-14) -> list[np.ndarray]:
    """K_a = sqrt(d lambda_a) reshape(v_a)^T from the eigen-decomposition of the trace-1 Choi matrix (a unitary U returns [U])."""
    d = int(round(math.sqrt(choi.shape[0])))
    w, v = np.linalg.eigh(0.5 * (choi + choi.conj().T))
    out: list[np.ndarray] = []
    for lam, vec in sorted(zip(w, v.T), key=lambda t: -t[0]):
        if lam <= tol:
            continue
        out.append(np.asarray(math.sqrt(d * float(lam)) * vec.reshape(d, d).T))
    if not out:
        raise ValueError("the Choi matrix has no positive eigenvalue")
    return out


def kraus_superoperator(kraus: Sequence[np.ndarray]) -> np.ndarray:
    """S = sum_a K_a (x) conj(K_a): the d^2 x d^2 matrix of rho -> sum_a K_a rho K_a^dag on the row-major vectorization of rho,
    vec(K rho K^dag) = (K (x) conj(K)) vec(rho)."""
    if not kraus:
        raise ValueError("a channel has at least one Kraus operator")
    d = int(kraus[0].shape[0])
    s = np.zeros((d * d, d * d), dtype=complex)
    for k in kraus:
        k_arr = np.asarray(k, dtype=complex)
        if k_arr.shape != (d, d):
            raise ValueError("every Kraus operator of a channel has the same square shape")
        s += np.kron(k_arr, k_arr.conj())
    return s


def apply_kraus_dm(
    rho: np.ndarray, kraus: Sequence[np.ndarray], dims: Sequence[int], factors: Sequence[int]
) -> np.ndarray:
    """sum_a K_a rho K_a^dag with the Kraus operators acting on the tensor ``factors`` (in the Kraus operators' own order) of a
    register density matrix over ``dims``: ONE product of the d_loc^2 x d_loc^2 superoperator (``kraus_superoperator``) with the
    register reshaped to (local row, local column) x (rest row, rest column). The per-operator ``einsum`` this replaces
    (performance pass 2026-09-09) looped over all six indices without a matrix product, 19 times slower at ten qubits and
    the dominant cost of a twelve-qubit GATE_LOCAL walk; the two agree to round-off (``tests/test_tomography.py``)."""
    n = len(dims)
    total = int(np.prod(dims))
    fac = [int(f) for f in factors]
    rest = [f for f in range(n) if f not in fac]
    d_loc = int(np.prod([dims[f] for f in fac]))
    d_rest = total // d_loc
    arr = np.asarray(rho, dtype=complex).reshape(list(dims) + list(dims))
    perm = fac + [n + f for f in fac] + rest + [n + f for f in rest]
    a = np.transpose(arr, perm).reshape(d_loc * d_loc, d_rest * d_rest)
    out = kraus_superoperator(kraus) @ a
    shaped = out.reshape(
        [dims[f] for f in fac] + [dims[f] for f in fac] + [dims[f] for f in rest] + [dims[f] for f in rest]
    )
    return np.asarray(np.transpose(shaped, np.argsort(perm)).reshape(total, total))


def apply_kraus_ket(
    ket: np.ndarray,
    kraus: Sequence[np.ndarray],
    dims: Sequence[int],
    factors: Sequence[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Kraus SAMPLING on one pure register member: K_a drawn with probability ||K_a psi||^2, the result renormalized."""
    n = len(dims)
    total = int(np.prod(dims))
    fac = [int(f) for f in factors]
    rest = [f for f in range(n) if f not in fac]
    d_loc = int(np.prod([dims[f] for f in fac]))
    arr = np.asarray(ket, dtype=complex).reshape(list(dims))
    perm = fac + rest
    v = np.transpose(arr, perm).reshape(d_loc, -1)
    branches = [k @ v for k in kraus]
    probs = np.array([float(np.sum(np.abs(b) ** 2)) for b in branches])
    if probs.sum() <= 0.0:
        raise ValueError("the Kraus operators annihilate the state")
    probs = probs / probs.sum()
    a = int(rng.choice(len(branches), p=probs))
    w = branches[a] / math.sqrt(float(np.sum(np.abs(branches[a]) ** 2)))
    shaped = w.reshape([dims[f] for f in fac] + [dims[f] for f in rest])
    return np.asarray(np.transpose(shaped, np.argsort(perm)).reshape(total)), a


def expansion_coefficients(rho: np.ndarray, inputs: Sequence[np.ndarray]) -> np.ndarray:
    """The REAL coefficients c_k with rho = sum_k c_k rho_k over the input density matrices (a basis of the Hermitian matrices),
    by least squares on the stacked real and imaginary parts."""
    r = np.array([np.asarray(x, dtype=complex).reshape(-1) for x in inputs]).T  # (D^2, n_inputs)
    a = np.vstack([r.real, r.imag])
    b = np.concatenate(
        [np.asarray(rho, dtype=complex).reshape(-1).real, np.asarray(rho, dtype=complex).reshape(-1).imag]
    )
    c, _res, _rank, _sv = np.linalg.lstsq(a, b, rcond=None)
    return np.asarray(c, dtype=float)


# ---- the motional input as branches ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionalBranch:
    weight: float
    kets: dict[int, qt.Qobj]
    """Resolved mode -> the branch's ket in the mode's factor space."""
    frozen_n: dict[int, int]
    """Frozen coupled mode -> its Fock state."""


def regrid_reduced(rho: np.ndarray, d: int) -> tuple[np.ndarray, float]:
    """A mode's reduced density matrix in a space of ``d`` levels: zero-padded, or truncated with the dropped weight returned."""
    arr = np.asarray(rho, dtype=complex)
    d_old = arr.shape[0]
    if d_old == d:
        return arr, 0.0
    if d_old < d:
        out = np.zeros((d, d), dtype=complex)
        out[:d_old, :d_old] = arr
        return out, 0.0
    dropped = float(np.real(np.trace(arr)) - np.real(np.trace(arr[:d, :d])))
    out = arr[:d, :d].copy()
    tr = float(np.real(np.trace(out)))
    if tr > 0.0:
        out /= tr
    return out, max(dropped, 0.0)


def motional_branches(
    space: HilbertSpace,
    model: MotionalModel,
    coupled_frozen: Iterable[int],
    weight_min: float,
    *,
    dropped_weight_max: float | None = None,
) -> tuple[list[MotionalBranch], float, tuple[str, ...]]:
    """The motional input of a gate-local space as weighted pure branches (Section 5.3's Fock-sum path made general):
    every resolved mode's tracked reduced density matrix (thermal at its nbar when none is tracked) in its eigenbasis, every
    coupled frozen mode's Fock populations (the diagonal of its tracked state, else thermal); products of weight >= ``weight_min``
    kept and renormalized, the dropped weight reported. With ``dropped_weight_max`` the lightest of the kept branches are dropped
    too, one at a time, while the total dropped weight stays at or below it (``SolverOptions.tomography_dropped_weight_max``,
    the tail rule of the performance pass 2026-09-09): the channel is a convex mixture over the branches, so dropping weight w
    and renormalizing changes it by at most 2w in diamond norm, the bound the caller reports."""
    notes: list[str] = []
    per_mode: list[tuple[int, str, list[tuple[float, object]]]] = []
    for tr in space.resolved:
        m = tr.mode
        rho = model.reduced.get(m)
        if rho is not None:
            arr, dropped = regrid_reduced(np.asarray(rho.full()), tr.d)
            if dropped > 1e-12:
                notes.append(
                    f"mode {m}: the tracked state's weight above the local cap d = {tr.d}, {dropped:.2e}, was truncated"
                )
        else:
            p = thermal_populations(float(model.nbar.get(m, 0.0)), tr.d)
            arr = np.diag(p / p.sum()).astype(complex)
        h = 0.5 * (arr + arr.conj().T)
        evals, evecs = np.linalg.eigh(h)
        opts_m: list[tuple[float, object]] = []
        for k in np.argsort(-evals):
            if evals[k] < weight_min:
                continue
            opts_m.append((float(evals[k]), qt.Qobj(evecs[:, k].reshape(-1, 1), dims=[[tr.d], [1]])))
        if not opts_m:
            k = int(np.argmax(evals))
            opts_m.append((1.0, qt.Qobj(evecs[:, k].reshape(-1, 1), dims=[[tr.d], [1]])))
        per_mode.append((m, "resolved", opts_m))
    for m in sorted(set(int(x) for x in coupled_frozen)):
        rho = model.reduced.get(m)
        if rho is not None:
            pops = np.real(np.diag(np.asarray(rho.full())))
            off = float(np.linalg.norm(np.asarray(rho.full()) - np.diag(np.diag(np.asarray(rho.full())))))
            if off > 1e-9:
                notes.append(
                    f"mode {m}: frozen with off-diagonal weight {off:.2e} in its tracked state; the Debye-Waller factor reads "
                    "its Fock populations only (Section 5.2)"
                )
        else:
            nb = float(model.nbar.get(m, 0.0))
            pops = thermal_populations(nb, int(60 + 40 * nb)) if nb > 0.0 else np.array([1.0])
        opts_f: list[tuple[float, object]] = [
            (float(p), int(n)) for n, p in enumerate(pops) if p >= weight_min
        ]
        if not opts_f:
            opts_f = [(1.0, int(np.argmax(pops)))]
        per_mode.append((m, "frozen", opts_f))
    branches: list[MotionalBranch] = []
    total = 0.0
    for combo in itertools.product(*[opts for _m, _c, opts in per_mode]):
        weight = float(np.prod([p for p, _s in combo])) if combo else 1.0
        if weight < weight_min:
            continue
        kets: dict[int, qt.Qobj] = {}
        frozen_n: dict[int, int] = {}
        for (m, cls, _opts), (_p, st) in zip(per_mode, combo):
            if cls == "resolved":
                assert isinstance(st, qt.Qobj)
                kets[m] = st
            else:
                frozen_n[m] = int(st)  # type: ignore[call-overload]
        branches.append(MotionalBranch(weight, kets, frozen_n))
        total += weight
    if not branches:
        raise ValueError("no motional branch survives the weight threshold")
    branches.sort(key=lambda b: -b.weight)
    dropped = float(max(1.0 - total, 0.0))
    if dropped_weight_max is not None and dropped_weight_max > 0.0:
        # the tail rule: the lightest branches go while the total dropped weight stays inside the budget (always one branch left)
        n_tail = 0
        while len(branches) - n_tail > 1 and dropped + branches[-1 - n_tail].weight <= dropped_weight_max:
            dropped += branches[-1 - n_tail].weight
            n_tail += 1
        if n_tail:
            branches = branches[: len(branches) - n_tail]
            total = sum(b.weight for b in branches)
            notes.append(
                f"{n_tail} light motional branch(es) dropped by the tail rule (total dropped weight {dropped:.3e} within "
                f"tomography_dropped_weight_max = {dropped_weight_max:.3g}; channel error bound 2w = {2.0 * dropped:.3e})"
            )
    branches = [MotionalBranch(b.weight / total, b.kets, b.frozen_n) for b in branches]
    return branches, dropped, tuple(notes)


# ---- the record ------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TomographyRecord:
    """What the state-based process tomography of one pulse group produced (Section 5.4 (a) and (b))."""

    space: HilbertSpace
    """The gate-local space the runs were made on (grown by the truncation monitor where it tripped)."""
    labels: tuple[str, ...]
    inputs: tuple[np.ndarray, ...]
    """Input density matrices on the ion factors."""
    outputs: tuple[np.ndarray, ...]
    """The reduced internal outputs, one per input."""
    choi: np.ndarray
    """The trace-1 Choi matrix after the CP/TP projection."""
    choi_raw: np.ndarray
    cp_residual: float
    tp_residual: float
    dykstra_iterations: int
    n_traj: int
    method: str
    motional_out: dict[int, tuple[np.ndarray, ...]]
    """Per resolved mode, the reduced motional output per input (the branch average)."""
    alpha_out: dict[int, tuple[complex, ...]]
    """Per resolved mode, <a_m> at the end per input: the residual displacement of that input."""
    nbar_out: dict[int, tuple[float, ...]]
    boundary_population: dict[int, float]
    populated_n_max: dict[int, int]
    margin_reached: dict[int, int]
    branches: int
    dropped_branch_weight: float
    engine_runs: int
    """Engine calls: d_int^2 x branches state propagations on the "states" route, d_int x branches on the "isometry" route, one
    propagator per branch on the "propagator" route (``route``)."""
    notes: tuple[str, ...]
    approximations: tuple[str, ...]
    integrators: tuple[str, ...]
    reports: tuple[EngineReport, ...]
    workers: int = 1
    """Processes the runs actually used: the inputs spread over a parallel map, or the trajectories inside one engine run
    (M9b audit B10). 1 = everything in-process, which is every carrier step, whose local space has no resolved mode."""
    route: TomographyRoute = "states"
    """How the channel was extracted (performance pass 2026-09-09): "states" propagates every input (the M9a reference, and
    every dissipative step), "isometry" the internal basis per branch (Stinespring), "propagator" reads the branch's Kraus
    operator off the segment propagator of an internal-state-only space. ``choi_raw`` is the least-squares fit on the first
    route and the isometries' Choi matrix, completely positive by construction, on the other two."""
    branch_error_bound: float = 0.0
    """2 x ``dropped_branch_weight``: the diamond-norm bound on the channel error of dropping and renormalizing the motional
    branches (the floor ``branch_weight_min`` and the tail rule ``tomography_dropped_weight_max``); a term of
    ``GateLocalReport.discrepancy_bound``."""
    tolerances: tuple[float, float] = (SolverOptions.atol, SolverOptions.rtol)
    """(atol, rtol) the engine runs integrated at: the caller's, or the map-accuracy-keyed pair of
    ``SolverOptions.tomography_tolerance_keyed`` on a unitary step with resolved modes."""
    tolerance_change: float | None = None
    """The Section 5.5 convergence statement for the keyed tolerance: d times the trace norm of the change in the Choi matrix
    when the dominant branch's columns are re-integrated ten times tighter (a bound on the diamond-norm change), weighted by
    that branch's share of the mixture; None when the tolerance was not keyed (the caller's tolerance is the caller's contract)."""

    @property
    def dimension(self) -> int:
        return int(self.inputs[0].shape[0])

    def kraus(self) -> list[np.ndarray]:
        return kraus_operators(self.choi)

    def coefficients(self, rho_local: np.ndarray) -> np.ndarray:
        """The register's local marginal in the input basis (real coefficients)."""
        return expansion_coefficients(rho_local, self.inputs)

    def motional_for(self, rho_local: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, complex]]:
        """The reduced motional state and the residual displacement every resolved mode is left in when the register's local
        marginal is ``rho_local`` (Section 5.4 (b)): the tomography outputs are linear in the input, so the expansion of the
        marginal in the input basis carries over."""
        c = self.coefficients(rho_local)
        out: dict[int, np.ndarray] = {}
        alpha: dict[int, complex] = {}
        for m, mats in self.motional_out.items():
            acc = sum((ck * mk for ck, mk in zip(c, mats)), np.zeros_like(mats[0]))
            acc = 0.5 * (acc + acc.conj().T)
            tr = float(np.real(np.trace(acc)))
            if tr > 0.0:
                acc = acc / tr
            out[m] = acc
            alpha[m] = complex(sum(ck * ak for ck, ak in zip(c, self.alpha_out[m])))
        return out, alpha

    def residual_displacement(self) -> dict[int, float]:
        """Per resolved mode, max over the computational-basis inputs of |<a_m>| after the pulse: the residual displacement per
        spin eigenstate that Section 5.4 asks GATE_LOCAL to report."""
        comp = set(computational_labels(self.space.ion_dims))
        out: dict[int, float] = {}
        for m, alphas in self.alpha_out.items():
            vals = [abs(a) for lab, a in zip(self.labels, alphas) if lab in comp]
            out[m] = float(max(vals)) if vals else 0.0
        return out

    def summary(self, ideal: np.ndarray | None = None) -> ChannelSummary:
        """The Section 6.8 summary: the average gate infidelity, the Pauli twirl of the ERROR channel E o U^dag and the depolarizing
        rate (the entanglement infidelity) against ``ideal``; without an ideal the twirl is of the channel itself and the
        infidelities are NaN."""
        d = self.dimension
        n_qubits = len(self.space.ion_dims)
        all_qubits = all(int(x) == 2 for x in self.space.ion_dims)
        if ideal is None:
            twirl = pauli_twirl(self.choi, n_qubits) if all_qubits else {}
            return ChannelSummary(
                self.choi, (self.cp_residual, self.tp_residual), self.n_traj, math.nan, twirl, math.nan
            )
        u = np.asarray(ideal, dtype=complex)
        if u.shape != (d, d):
            raise ValueError(f"the ideal unitary has shape {u.shape}, the map acts on dimension {d}")
        eps = entanglement_infidelity(self.choi, choi_from_unitary(u))
        error_kraus = [k @ u.conj().T for k in self.kraus()]
        twirl = pauli_twirl(choi_from_kraus(error_kraus), n_qubits) if all_qubits else {}
        return ChannelSummary(
            self.choi,
            (self.cp_residual, self.tp_residual),
            self.n_traj,
            average_gate_infidelity(eps, d),
            twirl,
            float(eps),
        )


def _as_schedule(pulse: Pulse | Sequence[Pulse] | Schedule) -> Schedule:
    from qutip_trap.control.schedule import Schedule as _Schedule

    if isinstance(pulse, _Schedule):
        return pulse
    pulses = tuple(pulse) if isinstance(pulse, Sequence) else (pulse,)
    if not pulses:
        raise ValueError("process tomography needs at least one pulse or a Schedule")
    t0 = min(p.t_start_s for p in pulses)
    return _Schedule(tuple(pulses), (), (), {}, t0_s=t0)


def coupled_frozen_modes(device: Device, space: HilbertSpace, pulses: Sequence[Pulse]) -> list[int]:
    """The frozen modes of ``space`` that any pulse couples to on an ion the space carries (their Debye-Waller factors matter)."""
    from qutip_trap.light.raman import lamb_dicke_parameters

    out: set[int] = set()
    for pulse in pulses:
        dk = pulse.drive.delta_k(device.beams)
        if float(np.linalg.norm(dk)) == 0.0:
            continue
        ions = list(pulse.drive.ions) + [j for j in pulse.drive.crosstalk if space.has_ion(j)]
        for ion in ions:
            if not space.has_ion(ion):
                continue
            etas, _ = lamb_dicke_parameters(device, ion, dk)
            out.update(m for m in space.frozen if abs(etas[m]) > 1e-12)
    return sorted(out)


@dataclass(frozen=True)
class _TomographyTask:
    input_index: int
    """The input state ("states" route) or the internal basis column ("isometry" route) this run propagates."""
    branch_index: int
    state: object
    """The initial ``State`` on the gate-local space."""
    sample: NoiseSample


def _branch_sample(sample: NoiseSample, branch: MotionalBranch) -> NoiseSample:
    """The sample of one motional branch: the frozen coupled modes' Fock states and the branch weight (which scales the boundary
    threshold, Section 5.5) on top of the run's values."""
    values = dict(sample.values)
    values.update({key_frozen_n(m): float(n) for m, n in branch.frozen_n.items()})
    values[KEY_BRANCH_WEIGHT] = float(branch.weight)
    return NoiseSample(sample.sample_id, values, dict(sample.ou_grids), sample.t_s)


def _engine_run(
    payload: tuple[
        JointExactEngine, Device, Schedule, object, HilbertSpace, NoiseSample, SeedSpec, SolverOptions
    ],
) -> tuple[Traces, EngineReport]:
    """One engine run as a map task (module-level so that it pickles under ``map="parallel"``; Section 11.3 item 9)."""
    engine, device, sched, state, space, smp, seeds, opts = payload
    traces = engine.run_pulses(device, sched, state, space, smp, seeds, opts)  # type: ignore[arg-type]
    rep = engine.last_report
    assert rep is not None
    return traces, rep


def _run_tasks(
    engine: JointExactEngine,
    device: Device,
    sched: Schedule,
    space: HilbertSpace,
    payloads: Sequence[_TomographyTask],
    seeds: SeedSpec,
    opts: SolverOptions,
) -> tuple[list[tuple[Traces, EngineReport]], int]:
    """The tomography's engine runs and the number of PROCESSES they actually used (M9b audit B10: ``Diagnostics.workers``
    reported ``worker_count(options)`` for a GATE_LOCAL run whether or not anything ran in parallel).

    In-process on the shared engine (its propagator cache serves an internal-state-only space from one integration), or spread
    over the workers when the space has resolved modes and a parallel map is configured."""
    workers = worker_count(opts)
    if opts.map == "serial" or workers <= 1 or len(payloads) < 2 or not space.resolved:
        out: list[tuple[Traces, EngineReport]] = []
        for task in payloads:
            traces = engine.run_pulses(device, sched, task.state, space, task.sample, seeds, opts)  # type: ignore[arg-type]
            rep = engine.last_report
            assert rep is not None
            out.append((traces, rep))
        return out, 1
    inner = replace(opts, map="serial")  # a worker runs its trajectories in-process: no nested pool
    items = [(engine, device, sched, task.state, space, task.sample, seeds, inner) for task in payloads]
    used = min(workers, len(payloads))
    return map_tasks(_engine_run, items, map_kind=opts.map, workers=workers), used


def _grown_space(reports: Iterable[EngineReport], current: HilbertSpace) -> HilbertSpace | None:
    """The largest space the truncation monitor grew any run to (None when every run kept ``current``)."""
    grown: HilbertSpace | None = None
    for rep in reports:
        if rep.space != current and (grown is None or rep.space.dimension > grown.dimension):
            grown = rep.space
    return grown


@dataclass(frozen=True)
class _Extraction:
    """What one route produced on one space, before the record is assembled."""

    route: TomographyRoute
    outputs: list[np.ndarray]
    mot_out: dict[int, list[np.ndarray]]
    alpha_out: dict[int, list[complex]]
    nbar_out: dict[int, list[float]]
    boundary: dict[int, float]
    populated: dict[int, int]
    margins: dict[int, int]
    reports: list[EngineReport]
    n_traj: int
    method: str
    runs: int
    workers: int
    choi_raw: np.ndarray | None
    """The Choi matrix the route formed directly (the isometry routes); None when it is fit by least squares (the state route)."""
    grown: HilbertSpace | None
    """The space the truncation monitor grew a run to: the caller reruns everything on it."""
    notes: tuple[str, ...] = ()


def _extract_from_states(
    engine: JointExactEngine,
    device: Device,
    sched: Schedule,
    current: HilbertSpace,
    branches: Sequence[MotionalBranch],
    thermal_frozen: Mapping[int, float],
    labels_kets: Sequence[tuple[str, np.ndarray]],
    sample: NoiseSample,
    seeds: SeedSpec,
    opts: SolverOptions,
) -> _Extraction:
    """The "states" route (M9a): every input of ``labels_kets`` propagated for every branch, the outputs averaged with the branch
    weights; the Choi matrix is left to the least-squares fit."""
    d_int = int(np.prod(current.ion_dims))
    dims_int = [list(current.ion_dims), [1] * len(current.ion_dims)]
    outputs: list[np.ndarray] = []
    mot_out: dict[int, list[np.ndarray]] = {tr.mode: [] for tr in current.resolved}
    alpha_out: dict[int, list[complex]] = {tr.mode: [] for tr in current.resolved}
    nbar_out: dict[int, list[float]] = {tr.mode: [] for tr in current.resolved}
    boundary: dict[int, float] = {}
    populated: dict[int, int] = {}
    margins: dict[int, int] = {}
    n_traj = 1
    method = "sesolve"
    # every (input, branch) run is independent: prepare them all, run them through the map, then accumulate in order
    payloads: list[_TomographyTask] = []
    for lab_idx, (_lab, ket) in enumerate(labels_kets):
        internal = qt.Qobj(ket.reshape(-1, 1), dims=dims_int)
        for br_idx, br in enumerate(branches):
            state = current.initial_state(internal, states=br.kets, thermal=thermal_frozen)
            payloads.append(_TomographyTask(lab_idx, br_idx, state, _branch_sample(sample, br)))
    results, map_workers = _run_tasks(engine, device, sched, current, payloads, seeds, opts)
    reports = [rep for _t, rep in results]
    grown = _grown_space(reports, current)
    if grown is not None:
        return _Extraction(
            "states",
            [],
            {},
            {},
            {},
            {},
            {},
            {},
            reports,
            1,
            "sesolve",
            len(results),
            map_workers,
            None,
            grown,
        )
    for lab_idx in range(len(labels_kets)):
        rho_acc = np.zeros((d_int, d_int), dtype=complex)
        mot_acc = {
            m: np.zeros((current.truncation(m).d, current.truncation(m).d), dtype=complex) for m in mot_out
        }
        alpha_acc = {m: 0j for m in mot_out}
        for task, (traces, rep) in zip(payloads, results):
            if task.input_index != lab_idx:
                continue
            br = branches[task.branch_index]
            rho_acc += br.weight * np.asarray(traces.final.internal.full())
            for m in mot_out:
                mot_acc[m] += br.weight * np.asarray(traces.final.motional.reduced[m].full())
                alpha_acc[m] += br.weight * complex(traces.alpha_m[m][-1])
            for m, v in traces.boundary_population.items():
                boundary[m] = max(boundary.get(m, 0.0), float(v))
            for m, v in rep.populated_n_max.items():
                populated[m] = max(populated.get(m, 0), int(v))
            for m, v in rep.margin_reached.items():
                margins[m] = min(margins.get(m, int(v)), int(v))
            n_traj = max(n_traj, rep.trajectories if rep.method == "mcsolve" else 1)
            if rep.method != "sesolve":
                method = rep.method
        outputs.append(rho_acc)
        for m in mot_out:
            mot_out[m].append(mot_acc[m])
            alpha_out[m].append(alpha_acc[m])
            nbar_out[m].append(_nbar_of(mot_acc[m]))
    return _Extraction(
        "states",
        outputs,
        mot_out,
        alpha_out,
        nbar_out,
        boundary,
        populated,
        margins,
        reports,
        n_traj,
        method,
        len(results),
        map_workers,
        None,
        None,
    )


def _isometry_columns(
    engine: JointExactEngine,
    device: Device,
    sched: Schedule,
    current: HilbertSpace,
    branches: Sequence[MotionalBranch],
    thermal_frozen: Mapping[int, float],
    sample: NoiseSample,
    seeds: SeedSpec,
    opts: SolverOptions,
) -> tuple[list[np.ndarray], list[EngineReport], dict[int, float], int, HilbertSpace | None]:
    """The "isometry" route's runs: the d_int internal basis kets of every branch through the engine (the same map as the state
    route), their final joint kets stacked into one D x d_int isometry per branch. Returns (isometries, reports, the boundary
    populations the engine's monitor saw on those kets, processes used, the grown space or None)."""
    d_int = int(np.prod(current.ion_dims))
    dims_int = [list(current.ion_dims), [1] * len(current.ion_dims)]
    payloads: list[_TomographyTask] = []
    for j, ket in enumerate(internal_basis(current.ion_dims)):
        internal = qt.Qobj(ket.reshape(-1, 1), dims=dims_int)
        for br_idx, br in enumerate(branches):
            state = current.initial_state(internal, states=br.kets, thermal=thermal_frozen)
            payloads.append(_TomographyTask(j, br_idx, state, _branch_sample(sample, br)))
    results, map_workers = _run_tasks(engine, device, sched, current, payloads, seeds, opts)
    reports = [rep for _t, rep in results]
    grown = _grown_space(reports, current)
    if grown is not None:
        return [], reports, {}, map_workers, grown
    columns = [np.zeros((current.dimension, d_int), dtype=complex) for _ in branches]
    boundary: dict[int, float] = {}
    for task, (traces, _rep) in zip(payloads, results):
        final = traces.final.joint
        if final is None or not final.isket:
            raise RuntimeError(
                "the isometry route needs the final joint ket of a unitary run; the engine returned "
                f"{'no joint state' if final is None else 'a density matrix'} (is_unitary() gates this route)"
            )
        columns[task.branch_index][:, task.input_index] = np.asarray(final.full()).reshape(-1)
        for m, v in traces.boundary_population.items():
            boundary[m] = max(boundary.get(m, 0.0), float(v))
    return columns, reports, boundary, map_workers, None


def _propagator_columns(
    engine: JointExactEngine,
    device: Device,
    sched: Schedule,
    current: HilbertSpace,
    branches: Sequence[MotionalBranch],
    sample: NoiseSample,
    seeds: SeedSpec,
    opts: SolverOptions,
    motional_model: MotionalModel,
) -> tuple[list[np.ndarray], list[EngineReport]]:
    """The "propagator" route: on an internal-state-only space the segment propagator of each branch IS its isometry (d_mot = 1),
    read off ``JointExactEngine.propagator`` without propagating a state (one engine call per branch)."""
    columns: list[np.ndarray] = []
    reports: list[EngineReport] = []
    for br in branches:
        u, rep = engine.propagator(
            device, sched, current, _branch_sample(sample, br), seeds, opts, motional_model=motional_model
        )
        columns.append(u)
        reports.append(rep)
    return columns, reports


def _nbar_of(rho_m: np.ndarray) -> float:
    return float(np.real(np.trace(np.diag(np.arange(rho_m.shape[0])) @ rho_m)))


def _extract_from_columns(
    current: HilbertSpace,
    branches: Sequence[MotionalBranch],
    columns: Sequence[np.ndarray],
    reports: list[EngineReport],
    labels_kets: Sequence[tuple[str, np.ndarray]],
    boundary_seen: Mapping[int, float],
    workers: int,
    route: TomographyRoute,
) -> _Extraction:
    """The channel and the record's per-input quantities from one isometry V_b per branch, by linearity (Stinespring).

    The Choi matrix is sum_b w_b C(V_b) (``choi_from_isometry``); for every input ket psi_k of ``labels_kets`` the output joint
    ket of branch b is V_b psi_k, whose internal marginal, reduced motional states, <a_m> and boundary populations are the same
    functions the engine evaluates on its final states, averaged with the branch weights, so ``outputs``, ``motional_out``,
    ``alpha_out`` and ``nbar_out`` keep the meaning of the state route. ``boundary_seen`` is what the engine's monitor recorded on
    the propagated basis kets at every segment end; the d_int^2 inputs' final-time values are folded in, and a superposition's
    boundary population is bounded by d_int times the basis kets' maximum (Cauchy-Schwarz), which the record notes."""
    d_int = int(np.prod(current.ion_dims))
    dim = current.dimension
    d_mot = dim // d_int
    n_in = len(labels_kets)
    psi = np.array([k for _l, k in labels_kets], dtype=complex).T  # (d_int, n_in)
    out_acc = np.zeros((n_in, d_int, d_int), dtype=complex)
    choi_raw = np.zeros((d_int * d_int, d_int * d_int), dtype=complex)
    resolved = [tr.mode for tr in current.resolved]
    mot_acc: dict[int, list[np.ndarray]] = {
        m: [np.zeros((current.truncation(m).d, current.truncation(m).d), dtype=complex) for _ in range(n_in)]
        for m in resolved
    }
    alpha_acc: dict[int, list[complex]] = {m: [0j] * n_in for m in resolved}
    boundary = dict(boundary_seen)
    dims_ket = [list(current.dims), [1] * len(current.dims)]
    for br, v in zip(branches, columns):
        w = br.weight
        if v.shape != (dim, d_int):
            raise ValueError(f"branch isometry of shape {v.shape}, expected {(dim, d_int)}")
        choi_raw += w * choi_from_isometry(v, d_int)
        phi = v @ psi  # (dim, n_in): the output joint ket of every input
        phi3 = phi.reshape(d_int, d_mot, n_in)
        out_acc += w * np.einsum("iak,jak->kij", phi3, phi3.conj())
        if d_mot > 1:
            for k in range(n_in):
                ket = qt.Qobj(phi[:, k].reshape(-1, 1), dims=dims_ket)
                for m in resolved:
                    mot_acc[m][k] += w * np.asarray(current.mode_marginal(ket, m).full())
                    alpha_acc[m][k] += w * complex(qt.expect(current.annihilation(m), ket))
                for m, val in boundary_populations(ket, current).items():
                    boundary[m] = max(boundary.get(m, 0.0), float(val))
    populated: dict[int, int] = {}
    margins: dict[int, int] = {}
    for rep in reports:
        for m, v_pop in rep.populated_n_max.items():
            populated[m] = max(populated.get(m, 0), int(v_pop))
        for m, v_mar in rep.margin_reached.items():
            margins[m] = min(margins.get(m, int(v_mar)), int(v_mar))
    mot_out = {m: list(mats) for m, mats in mot_acc.items()}
    notes: tuple[str, ...]
    if route == "propagator":
        notes = (
            f"channel read off the segment propagator of the internal-state-only space, one per motional branch "
            f"({len(branches)}); no state propagated (Section 11.3 item 5)",
        )
    else:
        notes = (
            f"channel read off the propagated internal basis ({d_int} kets per motional branch, Stinespring isometry); the "
            f"boundary monitor ran on those kets, so a superposition input's boundary population is at most {d_int} times "
            "the maximum reported (Section 5.5)",
        )
    return _Extraction(
        route,
        [out_acc[k] for k in range(n_in)],
        mot_out,
        {m: list(vals) for m, vals in alpha_acc.items()},
        {m: [_nbar_of(mat) for mat in mats] for m, mats in mot_out.items()},
        boundary,
        populated,
        margins,
        reports,
        1,
        "sesolve",
        len(reports),
        workers,
        choi_raw,
        None,
        notes,
    )


def tomography(
    engine: JointExactEngine,
    device: Device,
    pulse: Pulse | Sequence[Pulse] | Schedule,
    space: HilbertSpace,
    motional_model: MotionalModel,
    sample: NoiseSample,
    seeds: SeedSpec,
    options: SolverOptions,
) -> TomographyRecord:
    """State-based process tomography of ``pulse`` on ``space`` from the motional state of ``motional_model`` (Section 5.4).

    For every motional branch of ``motional_branches`` (weights >= ``options.branch_weight_min``) the channel is extracted by one
    of three routes (``TomographyRecord.route``): when the evolution is unitary and ``options.tomography_isometry`` holds, the
    internal basis is propagated and the channel read off the Stinespring isometry (or, on an internal-state-only space, off the
    segment propagator itself, no state propagated); otherwise every input of ``input_states`` is propagated by
    ``engine.run_pulses`` and the Choi matrix fit by least squares. The outputs are averaged with the branch weights, the Choi
    matrix is projected onto CP and TP, and the motional outputs and residual displacements per input are kept for the
    register-weighted update. When the truncation monitor grows the space during a run, every run is repeated on the grown
    space so that all outputs share one space. On the trajectory path the engine's trajectory count is raised to
    ceil(1/epsilon_map) (``options.map_accuracy``).
    """
    sched = _as_schedule(pulse)
    dissipative = bool(engine.channels) or bool(engine.device_channels)
    opts = options
    if dissipative and (
        options.lindblad_method == "mcsolve"
        or (options.lindblad_method == "auto" and space.dimension > options.mesolve_dimension_max)
    ):
        # the map-accuracy rule of Section 5.4: n_traj = ceil(1/epsilon_map) per input replaces the general trajectory count
        opts = replace(options, ntraj=int(math.ceil(1.0 / options.map_accuracy)))
    labels_kets = input_states(space.ion_dims)
    labels = tuple(lab for lab, _k in labels_kets)
    inputs = tuple(np.outer(k, k.conj()) for _l, k in labels_kets)
    frozen_coupled = coupled_frozen_modes(device, space, sched.pulses)
    # the tail rule's budget (Section 5.4, performance pass 2026-09-09): the bound 2w on the dropped weight stays inside half the
    # map accuracy unless the caller set the budget (0.0 keeps every branch above the floor)
    tail_budget = (
        options.map_accuracy / 4.0
        if options.tomography_dropped_weight_max is None
        else float(options.tomography_dropped_weight_max)
    )
    current = space
    for _attempt in range(engine.max_growth_retries + 2):
        branches, dropped, branch_notes = motional_branches(
            current, motional_model, frozen_coupled, opts.branch_weight_min, dropped_weight_max=tail_budget
        )
        thermal_frozen = {m: float(motional_model.nbar.get(m, 0.0)) for m in current.frozen}
        route: TomographyRoute = "states"
        if opts.tomography_isometry and engine.is_unitary(device, current, opts):
            route = "propagator" if (not current.resolved and current.enr_group is None) else "isometry"
        # the map-accuracy-keyed tolerance of a unitary step with resolved modes (never a tolerance the caller chose, never a
        # propagator at dimension 4 to 16, never a dissipative step): the ten-times-tighter probe below reports its effect
        keyed = keyed_tolerances(options) if route == "isometry" else None
        run_opts = replace(opts, atol=keyed[0], rtol=keyed[1]) if keyed is not None else opts
        if route == "states":
            ext = _extract_from_states(
                engine, device, sched, current, branches, thermal_frozen, labels_kets, sample, seeds, opts
            )
        elif route == "propagator":
            columns, reports = _propagator_columns(
                engine, device, sched, current, branches, sample, seeds, opts, motional_model
            )
            ext = _extract_from_columns(current, branches, columns, reports, labels_kets, {}, 1, route)
        else:
            columns, reports, seen, map_workers, grown = _isometry_columns(
                engine, device, sched, current, branches, thermal_frozen, sample, seeds, run_opts
            )
            if grown is not None:
                current = grown
                continue
            ext = _extract_from_columns(
                current, branches, columns, reports, labels_kets, seen, map_workers, route
            )
        if ext.grown is not None:
            current = ext.grown
            continue
        tolerance_change: float | None = None
        probe_notes: list[str] = []
        if keyed is not None and route == "isometry":
            # the Section 5.5 statement for the keyed tolerance: the dominant branch's columns ten times tighter, the change in
            # the branch's Choi matrix (d times its trace norm bounds the diamond norm) weighted by the branch's share
            tight = replace(run_opts, atol=run_opts.atol / 10.0, rtol=run_opts.rtol / 10.0)
            probe_cols, probe_reports, _seen, _w, probe_grown = _isometry_columns(
                engine, device, sched, current, branches[:1], thermal_frozen, sample, seeds, tight
            )
            if probe_grown is None and probe_cols:
                d_int = int(np.prod(current.ion_dims))
                c_probe = choi_from_isometry(probe_cols[0], d_int)
                c_run = choi_from_isometry(columns[0], d_int)
                tolerance_change = float(branches[0].weight) * d_int * _trace_norm(c_probe - c_run)
                ext = replace(ext, reports=ext.reports + probe_reports, runs=ext.runs + len(probe_reports))
                probe_notes.append(
                    f"tolerances keyed to the map accuracy: atol {keyed[0]:.0e}, rtol {keyed[1]:.0e} (engine defaults "
                    f"{options.atol:.0e}, {options.rtol:.0e}); the dominant branch (weight {branches[0].weight:.3f}) "
                    f"re-integrated ten times tighter moves the channel by {tolerance_change:.2e} (d x trace norm of the Choi "
                    "change; Section 5.5)"
                )
        choi_raw = ext.choi_raw if ext.choi_raw is not None else choi_least_squares(inputs, ext.outputs)
        choi, cp_res, tp_res, its = project_cptp(choi_raw)
        approximations: list[str] = []
        integrators: list[str] = []
        for rep in ext.reports:
            for a in rep.approximations:
                if a not in approximations:
                    approximations.append(a)
            for seg in rep.segments:
                if seg.integrator not in integrators:
                    integrators.append(seg.integrator)
        notes = list(branch_notes) + list(ext.notes) + probe_notes
        if dropped > 0.0:
            notes.append(
                f"motional branches dropped (below branch_weight_min = {opts.branch_weight_min:g}, and the tail rule): weight "
                f"{dropped:.3e} (renormalized); channel error bound 2w = {2.0 * dropped:.3e} (Section 5.4)"
            )
        if current != space:
            notes.append(
                f"the truncation monitor grew the gate-local space from dims {space.dims} to {current.dims} (Section 5.5)"
            )
        return TomographyRecord(
            space=current,
            labels=labels,
            inputs=inputs,
            outputs=tuple(ext.outputs),
            choi=choi,
            choi_raw=choi_raw,
            cp_residual=cp_res,
            tp_residual=tp_res,
            dykstra_iterations=its,
            n_traj=ext.n_traj,
            method=ext.method,
            motional_out={m: tuple(v) for m, v in ext.mot_out.items()},
            alpha_out={m: tuple(v) for m, v in ext.alpha_out.items()},
            nbar_out={m: tuple(v) for m, v in ext.nbar_out.items()},
            boundary_population=ext.boundary,
            populated_n_max=ext.populated,
            margin_reached=ext.margins,
            branches=len(branches),
            dropped_branch_weight=dropped,
            engine_runs=ext.runs,
            notes=tuple(notes),
            approximations=tuple(approximations),
            integrators=tuple(integrators),
            reports=tuple(ext.reports),
            workers=max([ext.workers] + [rep.workers for rep in ext.reports]),
            route=ext.route,
            branch_error_bound=2.0 * dropped,
            tolerances=(float(run_opts.atol), float(run_opts.rtol)),
            tolerance_change=tolerance_change,
        )
    raise RuntimeError("the gate-local space kept growing beyond the engine's retry budget")


def keyed_tolerances(options: SolverOptions) -> tuple[float, float] | None:
    """(atol, rtol) the GATE_LOCAL tomography integrates a unitary step with resolved modes at under
    ``SolverOptions.tomography_tolerance_keyed``: 1e-5 and 1e-3 of the map accuracy where the caller left the engine defaults,
    the caller's own value where not (a chosen tolerance is never overridden, the Section 5.3 precedent); None when the switch
    is off or neither tolerance would move."""
    if not options.tomography_tolerance_keyed:
        return None
    defaults = SolverOptions()
    atol = options.map_accuracy * 1e-5 if options.atol == defaults.atol else options.atol
    rtol = options.map_accuracy * 1e-3 if options.rtol == defaults.rtol else options.rtol
    if atol == options.atol and rtol == options.rtol:
        return None
    return float(atol), float(rtol)


def _trace_norm(h: np.ndarray) -> float:
    """The trace norm of a Hermitian matrix (the sum of |eigenvalues|); the difference of two Choi matrices is Hermitian."""
    sym = 0.5 * (h + h.conj().T)
    return float(np.sum(np.abs(np.linalg.eigvalsh(sym))))


def ideal_unitary_on(
    targets: Sequence[tuple[Sequence[int], np.ndarray]], ions: Sequence[int], ion_dims: Sequence[int]
) -> np.ndarray:
    """The product (time order: first applied first) of gate unitaries given on subsets of ``ions`` (device indices), embedded
    on the full local register in the order of ``ions`` (identity on the ions a gate does not touch)."""
    dims = [int(d) for d in ion_dims]
    total = int(np.prod(dims))
    u = np.eye(total, dtype=complex)
    pos = {int(q): k for k, q in enumerate(ions)}
    for gate_ions, mat in targets:
        fac = [pos[int(q)] for q in gate_ions]
        d_loc = int(np.prod([dims[f] for f in fac]))
        m = np.asarray(mat, dtype=complex)
        if m.shape != (d_loc, d_loc):
            raise ValueError("a gate unitary does not fit its ions' dimensions")
        emb = np.zeros((total, total), dtype=complex)
        # act on the local factors of every basis state: reshape the identity's columns
        rest = [f for f in range(len(dims)) if f not in fac]
        perm = fac + rest
        arr = np.eye(total, dtype=complex).reshape(dims + [total])
        a = np.transpose(arr, perm + [len(dims)]).reshape(d_loc, -1, total)
        b = np.einsum("ab,bxc->axc", m, a)
        shaped = b.reshape([dims[f] for f in fac] + [dims[f] for f in rest] + [total])
        emb = np.transpose(shaped, list(np.argsort(perm)) + [len(dims)]).reshape(total, total)
        u = emb @ u
    return u


def local_ideal(
    space_ions: Sequence[int], ion_dims: Sequence[int], gate_targets: Sequence[GateTarget]
) -> np.ndarray | None:
    """The ideal unitary of a step's ``GateTarget`` records on the local register (None without any target)."""
    if not gate_targets:
        return None
    pairs: list[tuple[Sequence[int], np.ndarray]] = []
    for t in sorted(gate_targets, key=lambda g: (g.t_start_s, g.gate_id)):
        pairs.append((tuple(t.ions), t.unitary()))
    return ideal_unitary_on(pairs, space_ions, ion_dims)


def fingerprint_model(model: MotionalModel, modes: Iterable[int], *, digits: int = 9) -> tuple[object, ...]:
    """The motional model restricted to ``modes``: the tracked reduced states (rounded) or the occupations."""
    out: list[object] = []
    for m in sorted(set(int(x) for x in modes)):
        rho = model.reduced.get(m)
        if rho is not None:
            out.append((m, np.round(np.asarray(rho.full()), digits)))
        else:
            out.append((m, round(float(model.nbar.get(m, 0.0)), digits)))
    return tuple(out)


def fingerprint_sample(sample: NoiseSample) -> tuple[object, ...]:
    return (
        sample.sample_id,
        tuple(sorted(sample.values.items())),
        {k: v for k, v in sorted(sample.ou_grids.items())},
        sample.t_s,
    )


def fingerprint_options(options: SolverOptions) -> Mapping[str, object]:
    return {
        "atol": options.atol,
        "rtol": options.rtol,
        "integrators": options.integrators,
        "boundary_population_max": options.boundary_population_max,
        "branch_weight_min": options.branch_weight_min,
        "lindblad_method": options.lindblad_method,
        "mesolve_dimension_max": options.mesolve_dimension_max,
        "ntraj": options.ntraj,
        # the trajectory ENSEMBLE differs with these two, so an extraction cached under one must not serve the other:
        # improved_sampling adds the deterministic no-jump member and reweights the rest (Section 5.3), and
        # trajectory_target_tol changes how many trajectories phase two replays (Section 3.4)
        "improved_sampling": options.improved_sampling,
        "trajectory_target_tol": options.trajectory_target_tol,
        "map_accuracy": options.map_accuracy,
        "scattering_channels": options.scattering_channels,
        "scattering_recoil": options.scattering_recoil,
        "intensity_noise_channels": options.intensity_noise_channels,
        "hardware_chain": options.hardware_chain,
        "margin_check": options.margin_check,
        # the routes agree to the solver tolerance, but a record extracted by one must not answer for the other's report
        "tomography_isometry": options.tomography_isometry,
        # the Tier 2 relaxations change the branches kept, the tolerance integrated at and the caps built (Section 5.4)
        "tomography_dropped_weight_max": options.tomography_dropped_weight_max,
        "tomography_tolerance_keyed": options.tomography_tolerance_keyed,
        "margin_element_tol": options.margin_element_tol,
    }


__all__ = [
    "M9A",
    "MotionalBranch",
    "TomographyRecord",
    "TomographyRoute",
    "apply_kraus_dm",
    "apply_kraus_ket",
    "choi_from_isometry",
    "choi_least_squares",
    "computational_labels",
    "coupled_frozen_modes",
    "cp_residual",
    "expansion_coefficients",
    "fingerprint_model",
    "fingerprint_options",
    "fingerprint_pulse",
    "fingerprint_sample",
    "ideal_unitary_on",
    "input_states",
    "internal_basis",
    "keyed_tolerances",
    "kraus_operators",
    "kraus_superoperator",
    "local_ideal",
    "motional_branches",
    "project_cptp",
    "project_psd",
    "project_tp",
    "regrid_reduced",
    "single_qudit_inputs",
    "tomography",
    "tp_residual",
]
