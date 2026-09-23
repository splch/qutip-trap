"""The exact rotating frame of the joint-exact ket integration.

psi = Theta phi with Theta = e^{-i H_0 t}, H_0 the diagonal static part (H_mot + H_int), gives
i phi' = Theta^dag V(t) Theta phi. Nothing is expanded or truncated: the phases act on the state, the drive operators keep
their form, and the solver steps at the drive frequencies instead of the Fock energies. It is a change of integration
variable the engine makes and undoes, not a ``BuilderOptions.frame``. A collapse operator must be a constant eigenoperator of
ad_{H_0} (it gains e^{i lambda t}); any other keeps the segment in the Schroedinger picture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import qutip as qt
from qutip.core import data as _data
from qutip.core.data import Data, Dense
from qutip.core.dimensions import Dimensions

from qutip_trap.dynamics.kernels import FactorizedOperator, _prod

PERF_2026_09_09 = "performance pass 2026-09-09 (dynamics/rotating.py, PLAN.md Sections 5.2, 5.3, 11.2)"


# ---- the frame's energies -----------------------------------------------------------------------------------------------------


def kronecker_energies(energies: np.ndarray, dims: Sequence[int]) -> list[np.ndarray] | None:
    """Per-factor vectors e_f with E[i_0, i_1, ...] = sum_f e_f[i_f] for a real diagonal ``energies`` over ``dims``, or None
    when the diagonal is not such a Kronecker sum."""
    dims_t = tuple(int(d) for d in dims)
    e = np.asarray(energies, dtype=float)
    if e.size != _prod(dims_t):
        raise ValueError("energies do not match dims")
    grid = e.reshape(dims_t)
    n = len(dims_t)
    origin = float(grid[(0,) * n])
    parts: list[np.ndarray] = []
    for f in range(n):
        index: list[int | slice] = [0] * n
        index[f] = slice(None)
        parts.append(np.asarray(grid[tuple(index)], dtype=float) - origin)
    parts[0] = parts[0] + origin
    recon = np.zeros(dims_t)
    for f, part in enumerate(parts):
        shape = [1] * n
        shape[f] = -1
        recon = recon + part.reshape(shape)
    scale = max(1.0, float(np.max(np.abs(e))))
    if not np.allclose(recon, grid, rtol=0.0, atol=1e-9 * scale):
        return None
    return parts


class FrameEnergies:
    """The diagonal H_0 of a segment as its Kronecker-sum factors (``factors[f]``: rad/s per level of factor f), with the
    phase vector e^{-i E t} memoized at the last t."""

    __slots__ = ("_energies", "_head", "_t_last", "_tail", "_theta_last", "dims", "factors")

    def __init__(self, dims: Sequence[int], factors: Sequence[np.ndarray]) -> None:
        self.dims = tuple(int(d) for d in dims)
        self.factors = tuple(np.ascontiguousarray(np.asarray(f, dtype=float)) for f in factors)
        if len(self.factors) != len(self.dims) or any(
            f.shape != (d,) for f, d in zip(self.factors, self.dims)
        ):
            raise ValueError("one energy vector per tensor factor, of the factor's dimension")
        # e^{-i E t} = e^{-i E_head t} (x) e^{-i E_tail t}: two exponentials and one outer product per evaluation
        head = np.zeros(1)
        for f in self.factors[:-1]:
            head = (head[:, None] + f[None, :]).reshape(-1)
        self._head = np.ascontiguousarray(head)
        self._tail = self.factors[-1]
        self._energies: np.ndarray | None = None
        self._t_last = math.nan
        self._theta_last: np.ndarray | None = None

    @property
    def dimension(self) -> int:
        return _prod(self.dims)

    @property
    def energies(self) -> np.ndarray:
        """The joint diagonal E (rad/s), shape (D,)."""
        if self._energies is None:
            self._energies = np.ascontiguousarray((self._head[:, None] + self._tail[None, :]).reshape(-1))
        return self._energies

    def theta(self, t: float) -> np.ndarray:
        """e^{-i E t}, shape (D,), C order."""
        if t == self._t_last and self._theta_last is not None:
            return self._theta_last
        scale = -1j * t
        theta = (np.exp(scale * self._head)[:, None] * np.exp(scale * self._tail)[None, :]).reshape(-1)
        self._t_last, self._theta_last = t, theta
        return theta

    def to_frame(self, state: qt.Qobj, t: float) -> qt.Qobj:
        """phi = Theta(t)^dag psi for a ket, Theta^dag rho Theta for a density matrix."""
        theta = self.theta(t)
        arr = np.asarray(state.full())
        if state.isket:
            return qt.Qobj(np.conj(theta)[:, None] * arr, dims=state.dims, copy=False)
        return qt.Qobj((np.conj(theta)[:, None] * arr) * theta[None, :], dims=state.dims, copy=False)

    def from_frame(self, state: qt.Qobj, t: float) -> qt.Qobj:
        """psi = Theta(t) phi for a ket, Theta rho Theta^dag for a density matrix."""
        theta = self.theta(t)
        arr = np.asarray(state.full())
        if state.isket:
            return qt.Qobj(theta[:, None] * arr, dims=state.dims, copy=False)
        return qt.Qobj((theta[:, None] * arr) * np.conj(theta)[None, :], dims=state.dims, copy=False)

    def __reduce__(self) -> tuple[object, ...]:
        return (FrameEnergies, (self.dims, self.factors))


def eigen_frequency(op: qt.Qobj, energies: np.ndarray) -> float | None:
    """lambda with e^{i H_0 t} op e^{-i H_0 t} = e^{i lambda t} op for the diagonal H_0 of ``energies``; None when ``op`` is
    not an eigenoperator of ad_{H_0}."""
    coo = op.to("CSR").data.as_scipy().tocoo()
    if coo.nnz == 0:
        return 0.0
    lam = energies[coo.row] - energies[coo.col]
    tol = 1e-9 * max(float(np.max(np.abs(energies))), 1.0)
    if float(np.ptp(lam)) > tol:
        return None
    return float(lam[0])


def expectation_phase(op: qt.Qobj, frame: FrameEnergies) -> float | None:
    """The lambda with <psi|op|psi> = e^{i lambda t} <phi|op|phi> (0 for populations, -omega_m for a_m), or None when
    ``op`` is not an eigenoperator of ad_{H_0}."""
    return eigen_frequency(op, frame.energies)


# ---- the data-layer operator -----------------------------------------------------------------------------------------------------


class PhasedSum(Data):  # type: ignore[misc]
    """theta^* (sum_k c_k A_k) theta at one time, over (coefficient value, operator data) ``terms`` and the phase vector
    ``theta`` = e^{-i E t} of shape (D,) (None for the plain sum)."""

    def __init__(
        self,
        dims: Sequence[int],
        terms: Sequence[tuple[complex, Data]],
        theta: np.ndarray | None = None,
    ) -> None:
        self.dims = tuple(int(d) for d in dims)
        n = _prod(self.dims)
        checked: list[tuple[complex, Data]] = []
        for c, op in terms:
            if op.shape != (n, n):
                raise ValueError(f"term of shape {op.shape} on a space of dimension {n}")
            checked.append((complex(c), op))
        self.terms: tuple[tuple[complex, Data], ...] = tuple(checked)
        if theta is not None:
            theta = np.ascontiguousarray(theta, dtype=complex).reshape(-1)
            if theta.shape != (n,):
                raise ValueError("theta is one phase per joint basis state")
        self.theta = theta
        super().__init__((n, n))

    @classmethod
    def _unchecked(
        cls, dims: tuple[int, ...], terms: tuple[tuple[complex, Data], ...], theta: np.ndarray | None, n: int
    ) -> PhasedSum:
        """The constructor without the shape checks, for :class:`RotatingDrive`, which checked them once."""
        self = cls.__new__(cls)
        self.dims = dims
        self.terms = terms
        self.theta = theta
        Data.__init__(self, (n, n))
        return self

    def copy(self) -> PhasedSum:
        return PhasedSum(self.dims, self.terms, self.theta)

    def to_array(self) -> np.ndarray:
        """The assembled matrix (for conversions, never on the propagation path)."""
        n = self.shape[0]
        out = np.zeros((n, n), dtype=complex)
        for c, op in self.terms:
            out += c * np.asarray(op.to_array())
        if self.theta is not None:
            out = (np.conj(self.theta)[:, None] * out) * self.theta[None, :]
        return out

    def conj(self) -> PhasedSum:
        return PhasedSum(
            self.dims,
            [(np.conj(c), _data.conj(op)) for c, op in self.terms],
            None if self.theta is None else np.conj(self.theta),
        )

    def transpose(self) -> PhasedSum:
        # (theta^* M theta)^T = theta M^T theta^*: the phases swap roles, which is conj(theta) in this representation
        return PhasedSum(
            self.dims,
            [(c, _data.transpose(op)) for c, op in self.terms],
            None if self.theta is None else np.conj(self.theta),
        )

    def adjoint(self) -> PhasedSum:
        return PhasedSum(self.dims, [(np.conj(c), _data.adjoint(op)) for c, op in self.terms], self.theta)

    def trace(self) -> complex:
        return complex(sum(c * complex(_data.trace(op)) for c, op in self.terms))

    def __reduce__(self) -> tuple[object, ...]:
        return (PhasedSum, (self.dims, self.terms, self.theta))

    def __repr__(self) -> str:
        return f"PhasedSum(dims={self.dims}, {len(self.terms)} terms, phased={self.theta is not None}, shape={self.shape})"

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """(theta^* sum_k c_k A_k theta) @ arr for ``arr`` of shape (D,) or (D, ncols), in the same shape (C order)."""
        a = np.asarray(arr)
        vector = a.ndim == 1
        a2 = a.reshape(-1, 1) if vector else a
        if a2.shape[0] != self.shape[0]:
            raise ValueError(f"state of dimension {a2.shape[0]} does not live on dims {self.dims}")
        if a2.dtype != np.complex128:
            a2 = a2.astype(np.complex128)
        if self.theta is not None:
            x = self.theta[:, None] * a2
        else:
            x = np.ascontiguousarray(a2)
        out = np.zeros(x.shape, dtype=complex)
        x_dense: Dense | None = None
        out_dense: Dense | None = None
        for c, op in self.terms:
            if isinstance(op, FactorizedOperator):
                op._plan.apply_into(x, c, out)
            else:
                if x_dense is None:
                    x_dense = Dense(x, copy=False)
                    out_dense = Dense(out, copy=False)
                _data.iadd_dense(out_dense, _data.matmul(op, x_dense, dtype=Dense), c)
        if self.theta is not None:
            np.multiply(out, np.conj(self.theta)[:, None], out=out)
        return out.reshape(-1) if vector else out

    @property
    def max_abs_bound(self) -> float:
        """An upper bound on the largest element modulus: sum_k |c_k| max|A_k| (exact for a single term)."""
        total = 0.0
        for c, op in self.terms:
            if isinstance(op, FactorizedOperator):
                total += abs(c) * op.max_abs
            elif isinstance(op, PhasedSum):
                total += abs(c) * op.max_abs_bound
            else:
                arr = np.asarray(op.to_array())
                total += abs(c) * (float(np.max(np.abs(arr))) if arr.size else 0.0)
        return total


# ---- dispatcher specialisations ----------------------------------------------------------------------------------------------------


def _to_dense(matrix: PhasedSum) -> Dense:
    return Dense(matrix.to_array(), copy=False)


def _from_dense(matrix: Dense) -> PhasedSum:
    """A dense matrix as a one-term sum with no phase (the conversion QuTiP's registry requires in this direction)."""
    arr = np.array(matrix.as_ndarray(), dtype=complex, copy=True)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("a phased sum is square")
    return PhasedSum((arr.shape[0],), [(1.0, Dense(arr, copy=False))], None)


def _matmul_phased_dense(left: PhasedSum, right: Dense, scale: complex = 1) -> Dense:
    out = left.apply(right.as_ndarray())
    if scale != 1:
        out = out * scale
    return Dense(out, copy=False)


def _matmul_phased_phased(left: PhasedSum, right: PhasedSum, scale: complex = 1) -> PhasedSum:
    """(theta^* A theta)(theta^* B theta) = theta^* A B theta on a shared phase, as the pairwise products of the terms; with
    different phases the product is assembled."""
    same_phase = (left.theta is None and right.theta is None) or (
        left.theta is not None
        and right.theta is not None
        and left.theta.shape == right.theta.shape
        and bool(np.array_equal(left.theta, right.theta))
    )
    if left.dims != right.dims or not same_phase:
        if left.shape[1] != right.shape[0]:
            raise ValueError("operator shapes do not match")
        return _from_dense(Dense(scale * (left.to_array() @ right.to_array()), copy=False))
    terms: list[tuple[complex, Data]] = []
    for c1, a in left.terms:
        for c2, b in right.terms:
            terms.append((c1 * c2 * scale, _data.matmul(a, b)))
    return PhasedSum(left.dims, terms, left.theta)


def _mul(matrix: PhasedSum, value: complex) -> PhasedSum:
    return PhasedSum(matrix.dims, [(c * value, op) for c, op in matrix.terms], matrix.theta)


def _neg(matrix: PhasedSum) -> PhasedSum:
    return _mul(matrix, -1.0)


def _adjoint(matrix: PhasedSum) -> PhasedSum:
    return matrix.adjoint()


def _conj(matrix: PhasedSum) -> PhasedSum:
    return matrix.conj()


def _transpose(matrix: PhasedSum) -> PhasedSum:
    return matrix.transpose()


def _trace(matrix: PhasedSum) -> complex:
    return matrix.trace()


def _iszero(matrix: PhasedSum, tol: float = -1) -> bool:
    if tol < 0:
        tol = float(qt.settings.core["atol"])
    return matrix.max_abs_bound <= tol


def _probe(dimension: int, k: int) -> np.ndarray:
    rng = np.random.default_rng([20260909, dimension, k])
    return np.asarray(rng.normal(size=(dimension, 1)) + 1j * rng.normal(size=(dimension, 1)))


def _apply_any(matrix: Data, v: np.ndarray) -> np.ndarray:
    if isinstance(matrix, PhasedSum):
        return matrix.apply(v)
    if isinstance(matrix, FactorizedOperator):
        return matrix.apply(v)
    return np.asarray(_data.matmul(matrix, Dense(v, copy=False)).to_array())


def _isequal(a: Data, b: Data, atol: float = -1, rtol: float = -1) -> bool:
    """Equality within tolerance by two fixed probe vectors (a sum has no unique term structure to compare)."""
    if atol < 0:
        atol = float(qt.settings.core["atol"])
    if rtol < 0:
        rtol = float(qt.settings.core["rtol"])
    if a.shape != b.shape:
        return False
    dimension = a.shape[0]
    for k in range(2):
        v = _probe(dimension, k)
        bound = atol * math.sqrt(dimension) * float(np.max(np.abs(v)))
        if not np.allclose(_apply_any(a, v), _apply_any(b, v), atol=bound, rtol=rtol):
            return False
    return True


def _isherm(matrix: PhasedSum, tol: float = -1) -> bool:
    if tol < 0:
        tol = float(qt.settings.core["atol"])
    dimension = matrix.shape[0]
    adj = matrix.adjoint()
    for k in range(2):
        v = _probe(dimension, k)
        bound = tol * math.sqrt(dimension) * float(np.max(np.abs(v)))
        if not np.allclose(matrix.apply(v), adj.apply(v), atol=bound, rtol=0.0):
            return False
    return True


def _expect(op: PhasedSum, state: Dense) -> complex:
    arr = state.as_ndarray()
    if arr.shape[1] == 1:
        return complex(np.vdot(arr, op.apply(arr)))
    return complex(np.trace(op.apply(arr)))


_REGISTERED = False


def register_data_type() -> None:
    """Register :class:`PhasedSum` with QuTiP's conversion registry and dispatchers (idempotent; run at import)."""
    global _REGISTERED
    if _REGISTERED or PhasedSum in _data.to.dtypes:
        _REGISTERED = True
        return
    _data.to.add_conversions(
        [
            (Dense, PhasedSum, _to_dense, 1.0),
            (PhasedSum, Dense, _from_dense, 10.0),
        ]
    )
    _data.to.register_aliases(["phasedsum", "PhasedSum"], PhasedSum)
    _data.matmul.add_specialisations(
        [
            (PhasedSum, Dense, Dense, _matmul_phased_dense),
            (PhasedSum, PhasedSum, PhasedSum, _matmul_phased_phased),
        ]
    )
    _data.mul.add_specialisations([(PhasedSum, PhasedSum, _mul)])
    _data.imul.add_specialisations([(PhasedSum, PhasedSum, _mul)])
    _data.neg.add_specialisations([(PhasedSum, PhasedSum, _neg)])
    _data.adjoint.add_specialisations([(PhasedSum, PhasedSum, _adjoint)])
    _data.conj.add_specialisations([(PhasedSum, PhasedSum, _conj)])
    _data.transpose.add_specialisations([(PhasedSum, PhasedSum, _transpose)])
    _data.trace.add_specialisations([(PhasedSum, _trace)])
    _data.iszero.add_specialisations([(PhasedSum, _iszero)])
    _data.isherm.add_specialisations([(PhasedSum, _isherm)])
    _data.isequal.add_specialisations(
        [
            (PhasedSum, PhasedSum, _isequal),
            (PhasedSum, Data, _isequal),
            (Data, PhasedSum, _isequal),
        ]
    )
    _data.expect.add_specialisations([(PhasedSum, Dense, _expect)])
    _REGISTERED = True


register_data_type()


# ---- the function element and the frame change -----------------------------------------------------------------------------------


class RotatingDrive:
    """The ``QobjEvo`` function element t -> the :class:`PhasedSum` of ``[operator, coefficient]`` terms at t, each term
    times ``scale``."""

    def __init__(
        self,
        frame: FrameEnergies,
        operators: Sequence[Data],
        coefficients: Sequence[object],
        scale: complex = 1.0,
    ) -> None:
        if len(operators) != len(coefficients):
            raise ValueError("one coefficient per operator")
        self.frame = frame
        self.operators = tuple(operators)
        self.coefficients = tuple(coefficients)
        self.scale = complex(scale)
        n = frame.dimension
        for op in self.operators:
            if op.shape != (n, n):
                raise ValueError(f"operator of shape {op.shape} on a space of dimension {n}")
        d = list(frame.dims)
        # parsed once: Qobj(data, dims=[d, d]) would re-parse the nested lists at every call
        self._qdims = Dimensions([d, d])
        self._pairs = tuple(zip(self.coefficients, self.operators))
        self._qobj: qt.Qobj | None = None

    def __call__(self, t: float, **_: object) -> qt.Qobj:
        theta = self.frame.theta(t)
        scale = self.scale
        if scale == 1.0:
            terms = tuple((complex(c(t)), op) for c, op in self._pairs)  # type: ignore[operator]
        else:
            terms = tuple((scale * complex(c(t)), op) for c, op in self._pairs)  # type: ignore[operator]
        data = PhasedSum._unchecked(self.frame.dims, terms, theta, self.frame.dimension)
        # one Qobj per element, refilled through the data setter: QuTiP's element machinery reads its data before the next
        # call, so a caller that keeps the returned Qobj must copy it
        qobj = self._qobj
        if qobj is None:
            qobj = qt.Qobj(data, dims=self._qdims, copy=False)
            self._qobj = qobj
        else:
            qobj.data = data
        return qobj

    def __reduce__(self) -> tuple[object, ...]:
        return (RotatingDrive, (self.frame, self.operators, self.coefficients, self.scale))


@dataclass(frozen=True)
class RotatingSegment:
    """One segment's integration problem in the rotating frame: the ``QobjEvo`` of V_I(t), the collapse operators
    transformed the same way, and the frame that maps states in and out."""

    H: qt.QobjEvo
    c_ops: tuple[qt.Qobj | qt.QobjEvo, ...]
    frame: FrameEnergies
    notes: tuple[str, ...] = field(default_factory=tuple)


def _phase_coefficient(t: float, lam: float, **_: object) -> complex:
    """e^{i lambda t}: the rotating-frame coefficient of an eigenoperator of ad_{H_0} (module-level so it pickles)."""
    return complex(math.cos(lam * t), math.sin(lam * t))


def _split_terms(h: qt.QobjEvo) -> tuple[qt.Qobj | None, list[tuple[qt.Qobj, object]]] | None:
    """(constant part, [(Qobj, Coefficient)]) of a list-form ``QobjEvo``; None when it holds a function element."""
    const: qt.Qobj | None = None
    evo: list[tuple[qt.Qobj, object]] = []
    for el in h.to_list():
        # to_list() spells a [Qobj, Coefficient] pair and a function element ([callable, args]) both as two-item lists
        if isinstance(el, list) and len(el) == 2 and isinstance(el[0], qt.Qobj):
            evo.append((el[0], el[1]))
        elif isinstance(el, qt.Qobj):
            const = el if const is None else const + el
        else:
            return None
    return const, evo


def frame_energies_of(static: qt.Qobj, dims: Sequence[int]) -> FrameEnergies | None:
    """The :class:`FrameEnergies` of a constant Hamiltonian that is diagonal and a Kronecker sum over ``dims``; None otherwise."""
    coo = static.to("CSR").data.as_scipy().tocoo()
    if coo.nnz and bool(np.any(coo.row != coo.col)):
        return None
    energies = np.real(np.asarray(static.diag(), dtype=complex))
    if not np.any(energies):
        return (
            None  # nothing to rotate away (an interaction-picture build, an all-frozen space with no offsets)
        )
    parts = kronecker_energies(energies, dims)
    if parts is None:
        return None
    return FrameEnergies(dims, parts)


def rotating_collapse(op: qt.Qobj | qt.QobjEvo, frame: FrameEnergies) -> qt.Qobj | qt.QobjEvo | None:
    """A constant collapse operator in the rotating frame: unchanged when it commutes with H_0, ``[c, e^{i lambda t}]`` for
    any other eigenoperator of ad_{H_0}; None when it is not an eigenoperator or is time dependent."""
    if not isinstance(op, qt.Qobj):
        return None
    lam = eigen_frequency(op, frame.energies)
    if lam is None:
        return None
    if lam == 0.0:
        return op
    return qt.QobjEvo([[op, qt.coefficient(_phase_coefficient, args={"lam": lam})]])


def rotating_frame(
    H: qt.QobjEvo, dims: Sequence[int], c_ops: Sequence[qt.Qobj | qt.QobjEvo] = ()
) -> RotatingSegment | None:
    """The rotating-frame problem of a segment whose ``H`` is a diagonal constant part plus ``[operator, coefficient]``
    terms; None for a constant ``H``, a non-diagonal, non-separable or zero static part, a function element, or a
    collapse operator :func:`rotating_collapse` refuses."""
    if H.isconstant:
        return None
    parts = _split_terms(H)
    if parts is None:
        return None
    const, evo = parts
    if const is None or not evo:
        return None
    frame = frame_energies_of(const, dims)
    if frame is None:
        return None
    drive = RotatingDrive(frame, [q.data for q, _c in evo], [c for _q, c in evo])
    rotated: list[qt.Qobj | qt.QobjEvo] = []
    notes: list[str] = []
    for c in c_ops:
        r = rotating_collapse(c, frame)
        if r is None:
            return None
        rotated.append(r)
    n_phase = sum(1 for c, r in zip(c_ops, rotated) if isinstance(c, qt.Qobj) and isinstance(r, qt.QobjEvo))
    if n_phase:
        notes.append(f"{n_phase} collapse operator(s) carry the e^(i lambda t) phase of the rotating frame")
    return RotatingSegment(qt.QobjEvo(drive), tuple(rotated), frame, tuple(notes))


__all__ = [
    "PERF_2026_09_09",
    "FrameEnergies",
    "PhasedSum",
    "RotatingDrive",
    "RotatingSegment",
    "eigen_frequency",
    "expectation_phase",
    "frame_energies_of",
    "kronecker_energies",
    "register_data_type",
    "rotating_collapse",
    "rotating_frame",
]
