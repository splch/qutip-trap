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

M9b: the prod_i d_i^2 x branches engine runs of a step with resolved modes are independent and are spread over the workers of
``SolverOptions.workers`` through QuTiP's map (Section 11.3 item 9), each worker running its own trajectories in-process; a step
on an internal-state-only space runs in-process, where the engine's propagator cache serves every input from one integration of
the segment propagator (Section 11.3 item 5).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

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


def apply_kraus_dm(
    rho: np.ndarray, kraus: Sequence[np.ndarray], dims: Sequence[int], factors: Sequence[int]
) -> np.ndarray:
    """sum_a K_a rho K_a^dag with the Kraus operators acting on the tensor ``factors`` (in the Kraus operators' own order) of a
    register density matrix over ``dims``."""
    n = len(dims)
    total = int(np.prod(dims))
    fac = [int(f) for f in factors]
    rest = [f for f in range(n) if f not in fac]
    d_loc = int(np.prod([dims[f] for f in fac]))
    d_rest = total // d_loc
    arr = np.asarray(rho, dtype=complex).reshape(list(dims) + list(dims))
    perm = fac + rest + [n + f for f in fac] + [n + f for f in rest]
    a = np.transpose(arr, perm).reshape(d_loc, d_rest, d_loc, d_rest)
    out = np.zeros_like(a)
    for k in kraus:
        out += np.einsum("ab,bxcy,dc->axdy", k, a, k.conj())
    inv = np.argsort(perm)
    shaped = out.reshape(
        [dims[f] for f in fac] + [dims[f] for f in rest] + [dims[f] for f in fac] + [dims[f] for f in rest]
    )
    return np.asarray(np.transpose(shaped, inv).reshape(total, total))


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
) -> tuple[list[MotionalBranch], float, tuple[str, ...]]:
    """The motional input of a gate-local space as weighted pure branches (Section 5.3's Fock-sum path made general):
    every resolved mode's tracked reduced density matrix (thermal at its nbar when none is tracked) in its eigenbasis, every
    coupled frozen mode's Fock populations (the diagonal of its tracked state, else thermal); products of weight >= ``weight_min``
    kept and renormalized, the dropped weight reported."""
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
    branches = [MotionalBranch(b.weight / total, b.kets, b.frozen_n) for b in branches]
    branches.sort(key=lambda b: -b.weight)
    return branches, float(max(1.0 - total, 0.0)), tuple(notes)


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
    notes: tuple[str, ...]
    approximations: tuple[str, ...]
    integrators: tuple[str, ...]
    reports: tuple[EngineReport, ...]

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
    branch_index: int
    state: object
    """The initial ``State`` on the gate-local space."""
    sample: NoiseSample


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
) -> list[tuple[Traces, EngineReport]]:
    """The tomography's engine runs: in-process on the shared engine (its propagator cache serves an internal-state-only space
    from one integration), or spread over the workers when the space has resolved modes and a parallel map is configured."""
    workers = worker_count(opts)
    if opts.map == "serial" or workers <= 1 or len(payloads) < 2 or not space.resolved:
        out: list[tuple[Traces, EngineReport]] = []
        for task in payloads:
            traces = engine.run_pulses(device, sched, task.state, space, task.sample, seeds, opts)  # type: ignore[arg-type]
            rep = engine.last_report
            assert rep is not None
            out.append((traces, rep))
        return out
    inner = replace(opts, map="serial")  # a worker runs its trajectories in-process: no nested pool
    items = [(engine, device, sched, task.state, space, task.sample, seeds, inner) for task in payloads]
    return map_tasks(_engine_run, items, map_kind=opts.map, workers=workers)


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

    Every input of ``input_states`` is propagated by ``engine.run_pulses`` for every motional branch of ``motional_branches``
    (weights >= ``options.branch_weight_min``), the outputs are averaged with the branch weights, the Choi matrix is reconstructed
    and projected, and the motional outputs and residual displacements per input are kept for the register-weighted update. When
    the truncation monitor grows the space during a run, every input is rerun on the grown space so that all outputs share one
    space. On the trajectory path the engine's trajectory count is raised to ceil(1/epsilon_map) (``options.map_accuracy``).
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
    dims_int = [list(space.ion_dims), [1] * len(space.ion_dims)]
    frozen_coupled = coupled_frozen_modes(device, space, sched.pulses)
    current = space
    for _attempt in range(engine.max_growth_retries + 2):
        branches, dropped, branch_notes = motional_branches(
            current, motional_model, frozen_coupled, opts.branch_weight_min
        )
        thermal_frozen = {m: float(motional_model.nbar.get(m, 0.0)) for m in current.frozen}
        d_int = int(np.prod(current.ion_dims))
        outputs: list[np.ndarray] = []
        mot_out: dict[int, list[np.ndarray]] = {tr.mode: [] for tr in current.resolved}
        alpha_out: dict[int, list[complex]] = {tr.mode: [] for tr in current.resolved}
        nbar_out: dict[int, list[float]] = {tr.mode: [] for tr in current.resolved}
        boundary: dict[int, float] = {}
        populated: dict[int, int] = {}
        margins: dict[int, int] = {}
        reports: list[EngineReport] = []
        n_traj = 1
        method = "sesolve"
        grown: HilbertSpace | None = None
        runs = 0
        # every (input, branch) run is independent: prepare them all, run them through the map, then accumulate in order
        payloads: list[_TomographyTask] = []
        for lab_idx, (_lab, ket) in enumerate(labels_kets):
            internal = qt.Qobj(ket.reshape(-1, 1), dims=dims_int)
            for br_idx, br in enumerate(branches):
                state = current.initial_state(internal, states=br.kets, thermal=thermal_frozen)
                values = dict(sample.values)
                values.update({key_frozen_n(m): float(n) for m, n in br.frozen_n.items()})
                values[KEY_BRANCH_WEIGHT] = float(br.weight)
                smp = NoiseSample(sample.sample_id, values, dict(sample.ou_grids), sample.t_s)
                payloads.append(_TomographyTask(lab_idx, br_idx, state, smp))
        results = _run_tasks(engine, device, sched, current, payloads, seeds, opts)
        runs = len(results)
        for rep in (r for _t, r in results):
            reports.append(rep)
            if rep.space != current and (grown is None or rep.space.dimension > grown.dimension):
                grown = rep.space
        if grown is None:
            for lab_idx in range(len(labels_kets)):
                rho_acc = np.zeros((d_int, d_int), dtype=complex)
                mot_acc = {
                    m: np.zeros((current.truncation(m).d, current.truncation(m).d), dtype=complex)
                    for m in mot_out
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
                    nbar_out[m].append(
                        float(np.real(np.trace(np.diag(np.arange(mot_acc[m].shape[0])) @ mot_acc[m])))
                    )
        if grown is not None:
            current = grown
            continue
        inputs = tuple(np.outer(k, k.conj()) for _l, k in labels_kets)
        choi_raw = choi_least_squares(inputs, outputs)
        choi, cp_res, tp_res, its = project_cptp(choi_raw)
        approximations: list[str] = []
        integrators: list[str] = []
        for rep in reports:
            for a in rep.approximations:
                if a not in approximations:
                    approximations.append(a)
            for seg in rep.segments:
                if seg.integrator not in integrators:
                    integrators.append(seg.integrator)
        notes = list(branch_notes)
        if dropped > 0.0:
            notes.append(
                f"motional branches below branch_weight_min = {opts.branch_weight_min:g} dropped: weight {dropped:.3e} (renormalized)"
            )
        if current != space:
            notes.append(
                f"the truncation monitor grew the gate-local space from dims {space.dims} to {current.dims} (Section 5.5)"
            )
        return TomographyRecord(
            space=current,
            labels=labels,
            inputs=inputs,
            outputs=tuple(outputs),
            choi=choi,
            choi_raw=choi_raw,
            cp_residual=cp_res,
            tp_residual=tp_res,
            dykstra_iterations=its,
            n_traj=n_traj,
            method=method,
            motional_out={m: tuple(v) for m, v in mot_out.items()},
            alpha_out={m: tuple(v) for m, v in alpha_out.items()},
            nbar_out={m: tuple(v) for m, v in nbar_out.items()},
            boundary_population=boundary,
            populated_n_max=populated,
            margin_reached=margins,
            branches=len(branches),
            dropped_branch_weight=dropped,
            engine_runs=runs,
            notes=tuple(notes),
            approximations=tuple(approximations),
            integrators=tuple(integrators),
            reports=tuple(reports),
        )
    raise RuntimeError("the gate-local space kept growing beyond the engine's retry budget")


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
        "map_accuracy": options.map_accuracy,
        "scattering_channels": options.scattering_channels,
        "scattering_recoil": options.scattering_recoil,
        "intensity_noise_channels": options.intensity_noise_channels,
        "hardware_chain": options.hardware_chain,
        "margin_check": options.margin_check,
    }


__all__ = [
    "M9A",
    "MotionalBranch",
    "TomographyRecord",
    "apply_kraus_dm",
    "apply_kraus_ket",
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
    "kraus_operators",
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
