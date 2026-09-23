"""Mode-factorized drive operators sigma_+ (x) prod_m D_m, applied axis by axis without assembling the matrix.

:class:`FactorizedOperator` is a QuTiP data-layer type registered at import; an unsupported operation (a sum, a
Liouvillian) converts to Dense, so mesolve segments use the assembled CSR operator. A drive coupling to an ENR mode is
never factorized (an ENR displacement is not a product of per-mode factors).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import qutip as qt
from qutip.core import data as _data
from qutip.core.data import CSR, Data, Dense

M9B = "milestone M9b (dynamics/kernels.py, PLAN.md Section 11.3 item 4)"

KernelChoice = Literal["auto", "factorized", "assembled"]

# ---- cost model per drive term through QobjEvo.matmul, fit over spaces of dimension 48 to 8192 ----------------------------
COST_CSR_US = 0.5
"""Fixed cost of one assembled CSR term."""
COST_CSR_NS_PER_NNZ = 0.57
"""Cost per non-zero of an assembled CSR product (memory-bound)."""
COST_TERM_US = 2.0
"""Fixed cost per factorized term: dispatch, allocation and the zero fill of the output."""
COST_STEP_US = 2.5
"""Cost of the one batched matrix product call per factor."""
COST_NS_PER_MAC = 0.6
"""Cost per complex multiply-add of the batched products."""


def _prod(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def kernel_costs_us(dims: Sequence[int], ion_factor: int, mode_factors: Sequence[int]) -> tuple[float, float]:
    """(assembled CSR, factorized) microseconds per application of one drive term sigma_+^i (x) prod_m D_m on ``dims``:
    D prod_m d_m / d_ion non-zeros against each mode factor applied to the D/d_ion source-level amplitudes."""
    d = _prod(dims)
    d_ion = int(dims[ion_factor])
    block = d // d_ion
    nnz = block * _prod([dims[f] for f in mode_factors])
    assembled = COST_CSR_US + COST_CSR_NS_PER_NNZ * 1e-3 * nnz
    factorized = COST_TERM_US + sum(
        COST_STEP_US + COST_NS_PER_MAC * 1e-3 * block * dims[f] for f in mode_factors
    )
    return assembled, factorized


def prefer_factorized(dims: Sequence[int], ion_factor: int, mode_factors: Sequence[int]) -> bool:
    """The ``auto`` rule: factorize when the cost model says the factorized term is the cheaper one."""
    if not mode_factors:
        return False
    assembled, factorized = kernel_costs_us(dims, ion_factor, mode_factors)
    return factorized < assembled


# ---- the data-layer type -------------------------------------------------------------------------------------------------------


class FactorizedOperator(Data):  # type: ignore[misc]
    """scale x (x)_f A_f on the joint space of ``dims`` (ions, then resolved modes), held as its factors (A_f the identity
    where ``factors`` has no entry) and applied axis by axis."""

    def __init__(self, dims: Sequence[int], factors: Mapping[int, np.ndarray], scale: complex = 1.0) -> None:
        dims_t = tuple(int(d) for d in dims)
        if not dims_t or any(d < 1 for d in dims_t):
            raise ValueError("dims are positive tensor dimensions")
        facs: dict[int, np.ndarray] = {}
        for f, a in factors.items():
            k = int(f)
            if not 0 <= k < len(dims_t):
                raise IndexError(f"factor {k} out of range for {len(dims_t)} factors")
            arr = np.ascontiguousarray(a, dtype=complex)
            if arr.shape != (dims_t[k], dims_t[k]):
                raise ValueError(f"factor {k} has shape {arr.shape}, expected {(dims_t[k], dims_t[k])}")
            facs[k] = arr
        self.dims = dims_t
        self.factors = facs
        self.scale = complex(scale)
        n = _prod(dims_t)
        super().__init__((n, n))
        self._plan = _ApplicationPlan(dims_t, facs, self.scale)

    def copy(self) -> FactorizedOperator:
        return FactorizedOperator(self.dims, {k: v.copy() for k, v in self.factors.items()}, self.scale)

    def to_array(self) -> np.ndarray:
        """The assembled dense matrix scale x (x)_f A_f (for conversions, never on the propagation path)."""
        out = np.array([[self.scale]], dtype=complex)
        for f, d in enumerate(self.dims):
            a = self.factors.get(f)
            out = np.kron(out, np.eye(d, dtype=complex) if a is None else a)
        return out

    def conj(self) -> FactorizedOperator:
        return FactorizedOperator(
            self.dims, {k: v.conj() for k, v in self.factors.items()}, np.conj(self.scale)
        )

    def transpose(self) -> FactorizedOperator:
        return FactorizedOperator(self.dims, {k: v.T for k, v in self.factors.items()}, self.scale)

    def adjoint(self) -> FactorizedOperator:
        return FactorizedOperator(
            self.dims, {k: v.conj().T for k, v in self.factors.items()}, np.conj(self.scale)
        )

    def trace(self) -> complex:
        out = self.scale
        for f, d in enumerate(self.dims):
            a = self.factors.get(f)
            out *= d if a is None else complex(np.trace(a))
        return complex(out)

    def __reduce__(self) -> tuple[object, ...]:
        return (FactorizedOperator, (self.dims, self.factors, self.scale))

    def __repr__(self) -> str:
        return (
            f"FactorizedOperator(dims={self.dims}, factors on {sorted(self.factors)}, scale={self.scale}, "
            f"shape={self.shape})"
        )

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """(scale x (x)_f A_f) @ arr for ``arr`` of shape (D,) or (D, ncols), returned in the same shape (C order)."""
        return self._plan.apply(arr)

    @property
    def max_abs(self) -> float:
        """The largest element modulus of the represented matrix: |scale| prod_f max|A_f|."""
        out = abs(self.scale)
        for a in self.factors.values():
            out *= float(np.max(np.abs(a))) if a.size else 0.0
        return out

    @property
    def nnz_assembled(self) -> int:
        """Non-zeros the assembled operator would have."""
        out = 1
        for f, d in enumerate(self.dims):
            a = self.factors.get(f)
            out *= d if a is None else int(np.count_nonzero(a))
        return out


class _ApplicationPlan:
    """How to apply (x)_f A_f to an array reshaped to ``dims + (ncols,)``: a factor with one non-zero element is sliced, a
    diagonal one broadcast, any other one BLAS call where the layout allows (only a middle factor is batched); the scalar
    prefactor is folded into the first factor matrix."""

    __slots__ = (
        "block_dims",
        "d",
        "dims",
        "prefactor",
        "sliced",
        "slices_in",
        "slices_out",
        "steps",
        "tail_scale",
    )

    def __init__(self, dims: tuple[int, ...], factors: Mapping[int, np.ndarray], scale: complex) -> None:
        self.dims = dims
        self.d = _prod(dims)
        prefactor = complex(scale)
        slices_in: list[int | slice] = [slice(None)] * len(dims)
        slices_out: list[int | slice] = [slice(None)] * len(dims)
        sliced: list[int] = []
        remaining: list[tuple[int, str, np.ndarray]] = []
        for f in sorted(factors):
            a = factors[f]
            nz = np.argwhere(np.abs(a) > 0.0)
            if nz.shape[0] == 0:
                prefactor = 0.0 + 0.0j
                continue
            if nz.shape[0] == 1:
                i, j = int(nz[0][0]), int(nz[0][1])
                prefactor *= complex(a[i, j])
                slices_in[f] = j
                slices_out[f] = i
                sliced.append(f)
            elif np.count_nonzero(a - np.diag(np.diag(a))) == 0:
                remaining.append((f, "diag", np.ascontiguousarray(np.diag(a))))
            else:
                remaining.append((f, "dense", a))
        kept = [f for f in range(len(dims)) if f not in sliced]
        self.block_dims = tuple(dims[f] for f in kept)
        # (kind, lead, d, trail, matrix or diagonal, transposed matrix or None); the prefactor rides on the first step
        steps: list[tuple[str, int, int, int, np.ndarray, np.ndarray | None]] = []
        folded = prefactor != 0.0 and bool(remaining)
        for idx, (f, kind, mat) in enumerate(remaining):
            ax = kept.index(f)
            lead = _prod(self.block_dims[:ax])
            trail = _prod(self.block_dims[ax + 1 :])
            m = np.ascontiguousarray(mat * prefactor if (folded and idx == 0) else mat)
            m_t = np.ascontiguousarray(m.T) if kind == "dense" else None
            steps.append((kind, lead, int(dims[f]), trail, m, m_t))
        self.steps = tuple(steps)
        self.sliced = tuple(sliced)
        self.slices_in: tuple[int | slice, ...] = tuple(slices_in)
        self.slices_out: tuple[int | slice, ...] = tuple(slices_out)
        self.prefactor = prefactor
        self.tail_scale = 1.0 + 0.0j if folded else prefactor
        """1 once the prefactor is folded into a factor; the prefactor itself for a pure slice (no mode factor)."""

    def _transform(self, a: np.ndarray, ncols: int, alpha: complex = 1.0) -> np.ndarray:
        """alpha x (prefactor (x)_kept A_f) applied to the sliced-in block of ``a`` (shape (D, ncols), C order), returned with
        shape ``block_dims + (ncols,)``."""
        y = a.reshape(self.dims + (ncols,))[self.slices_in]
        first = True
        for kind, lead, d, trail, mat, mat_t in self.steps:
            if first and alpha != 1.0:
                # alpha rides on the first factor (d^2 multiplications) rather than on the output (D/d_ion of them)
                mat = mat * alpha
                mat_t = None if mat_t is None else mat_t * alpha
            first = False
            t = trail * ncols
            if kind == "dense":
                assert mat_t is not None
                if t == 1:
                    y = y.reshape(lead, d) @ mat_t
                elif lead == 1:
                    y = mat @ y.reshape(d, t)
                else:
                    y = np.matmul(mat, y.reshape(lead, d, t))
            else:
                y = y.reshape(lead, d, t) * mat[None, :, None]
        scale = self.tail_scale * (alpha if first else 1.0)
        if scale != 1.0:
            y = y * scale
        return y.reshape(self.block_dims + (ncols,))

    def apply(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.dtype != np.complex128:
            a = a.astype(np.complex128)
        vector = a.ndim == 1
        ncols = 1 if vector else a.shape[1]
        if a.shape[0] != self.d:
            raise ValueError(f"state of dimension {a.shape[0]} does not live on dims {self.dims}")
        if self.prefactor == 0.0:
            out = np.zeros(a.shape, dtype=complex)
            return out
        y = self._transform(a, ncols)
        if self.sliced:
            out = np.zeros(self.dims + (ncols,), dtype=complex)
            out[self.slices_out] = y
        else:
            out = y
        return out.reshape(-1) if vector else out.reshape(-1, ncols)

    def apply_into(self, a: np.ndarray, alpha: complex, out: np.ndarray) -> None:
        """``out += alpha x (self @ a)`` for C-contiguous complex ``a`` and ``out`` of one shape (D, ncols)."""
        if self.prefactor == 0.0 or alpha == 0.0:
            return
        y = self._transform(a, a.shape[1], complex(alpha))
        block = out.reshape(self.dims + (a.shape[1],))[self.slices_out]
        block += y


# ---- dispatcher specialisations ----------------------------------------------------------------------------------------------------


def _to_dense(matrix: FactorizedOperator) -> Dense:
    return Dense(matrix.to_array(), copy=False)


def _to_csr(matrix: FactorizedOperator) -> CSR:
    return _data.csr.from_dense(Dense(matrix.to_array(), copy=False))


def _from_dense(matrix: Dense) -> FactorizedOperator:
    """A dense matrix as a one-factor operator (the conversion QuTiP's registry requires in this direction)."""
    arr = np.array(matrix.as_ndarray(), dtype=complex, copy=True)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("a factorized operator is square")
    return FactorizedOperator((arr.shape[0],), {0: arr}, 1.0)


def _matmul_factorized_dense(left: FactorizedOperator, right: Dense, scale: complex = 1) -> Dense:
    arr = right.as_ndarray()
    if scale == 1:
        return Dense(left.apply(arr), copy=False)
    plan = left._plan
    if arr.dtype != np.complex128:
        arr = arr.astype(np.complex128)
    if plan.prefactor == 0.0:
        return Dense(np.zeros(arr.shape, dtype=complex), copy=False)
    y = plan._transform(np.ascontiguousarray(arr), arr.shape[1], complex(scale))
    if plan.sliced:
        out = np.zeros(plan.dims + (arr.shape[1],), dtype=complex)
        out[plan.slices_out] = y
    else:
        out = y
    return Dense(out.reshape(-1, arr.shape[1]), copy=False)


def _matmul_factorized_factorized(
    left: FactorizedOperator, right: FactorizedOperator, scale: complex = 1
) -> FactorizedOperator:
    """(x)_f A_f (x)_f B_f = (x)_f A_f B_f on a shared factor structure (the c^dag c of a factorized collapse operator)."""
    if left.dims != right.dims:
        if left.shape != right.shape:
            raise ValueError("operator shapes do not match")
        # different factor structures over the same dimension: assemble (the general fallback)
        return _from_dense(Dense(left.to_array() @ right.to_array(), copy=False))
    factors: dict[int, np.ndarray] = {}
    for f in range(len(left.dims)):
        a, b = left.factors.get(f), right.factors.get(f)
        if a is None and b is None:
            continue
        if a is None:
            assert b is not None
            factors[f] = b.copy()
        elif b is None:
            factors[f] = a.copy()
        else:
            factors[f] = a @ b
    return FactorizedOperator(left.dims, factors, left.scale * right.scale * scale)


def _mul(matrix: FactorizedOperator, value: complex) -> FactorizedOperator:
    return FactorizedOperator(matrix.dims, matrix.factors, matrix.scale * value)


def _neg(matrix: FactorizedOperator) -> FactorizedOperator:
    return FactorizedOperator(matrix.dims, matrix.factors, -matrix.scale)


def _adjoint(matrix: FactorizedOperator) -> FactorizedOperator:
    return matrix.adjoint()


def _conj(matrix: FactorizedOperator) -> FactorizedOperator:
    return matrix.conj()


def _transpose(matrix: FactorizedOperator) -> FactorizedOperator:
    return matrix.transpose()


def _trace(matrix: FactorizedOperator) -> complex:
    return matrix.trace()


def _iszero(matrix: FactorizedOperator, tol: float = -1) -> bool:
    if tol < 0:
        tol = float(qt.settings.core["atol"])
    return matrix.max_abs <= tol


def _probe(dimension: int, k: int) -> np.ndarray:
    """A fixed complex Gaussian probe vector, deterministic so that the probe tests are reproducible."""
    rng = np.random.default_rng([20260907, dimension, k])
    return np.asarray(rng.normal(size=(dimension, 1)) + 1j * rng.normal(size=(dimension, 1)))


def _apply_any(matrix: Data, v: np.ndarray) -> np.ndarray:
    if isinstance(matrix, FactorizedOperator):
        return matrix.apply(v)
    return np.asarray(_data.matmul(matrix, Dense(v, copy=False)).to_array())


def _scalar_ratio(x: np.ndarray, y: np.ndarray, atol: float, rtol: float) -> complex | None:
    """The c with ``x == c y`` elementwise, or None when ``x`` is not a scalar multiple of ``y``."""
    big = float(np.max(np.abs(y))) if y.size else 0.0
    if big == 0.0:
        return None
    i = np.unravel_index(int(np.argmax(np.abs(y))), y.shape)
    c = complex(x[i] / y[i])
    return c if np.allclose(x, c * y, atol=atol, rtol=rtol) else None


def _isequal(a: Data, b: Data, atol: float = -1, rtol: float = -1) -> bool:
    """Equality within tolerance without assembling (``QobjEvo.compress`` merges elements called equal): exact on shared
    dims (A_f = c_f B_f for every f with scale_a prod_f c_f = scale_b), else a probe test with two fixed vectors."""
    if atol < 0:
        atol = float(qt.settings.core["atol"])
    if rtol < 0:
        rtol = float(qt.settings.core["rtol"])
    if a.shape != b.shape:
        return False
    if isinstance(a, FactorizedOperator) and isinstance(b, FactorizedOperator) and a.dims == b.dims:
        zero_a, zero_b = _iszero(a, atol), _iszero(b, atol)
        if zero_a or zero_b:
            return zero_a and zero_b
        ratio = complex(1.0)
        for f in sorted(set(a.factors) | set(b.factors)):
            ident = np.eye(a.dims[f], dtype=complex)
            c = _scalar_ratio(a.factors.get(f, ident), b.factors.get(f, ident), atol, rtol)
            if c is None:
                return False
            ratio *= c
        return bool(abs(a.scale * ratio - b.scale) <= atol + rtol * abs(b.scale))
    dimension = a.shape[0]
    for k in range(2):
        v = _probe(dimension, k)
        va = _apply_any(a, v)
        vb = _apply_any(b, v)
        bound = atol * math.sqrt(dimension) * float(np.max(np.abs(v)))
        if not np.allclose(va, vb, atol=bound, rtol=rtol):
            return False
    return True


def _isherm(matrix: FactorizedOperator, tol: float = -1) -> bool:
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


def _expect(op: FactorizedOperator, state: Dense) -> complex:
    arr = state.as_ndarray()
    if arr.shape[1] == 1:
        return complex(np.vdot(arr, op.apply(arr)))
    return complex(np.trace(op.apply(arr)))


_REGISTERED = False


def register_data_type() -> None:
    """Register :class:`FactorizedOperator` with QuTiP's conversion registry and dispatchers (idempotent; run at import)."""
    global _REGISTERED
    if _REGISTERED or FactorizedOperator in _data.to.dtypes:
        _REGISTERED = True
        return
    _data.to.add_conversions(
        [
            (Dense, FactorizedOperator, _to_dense, 1.0),
            (CSR, FactorizedOperator, _to_csr, 1.5),
            (FactorizedOperator, Dense, _from_dense, 10.0),
        ]
    )
    _data.to.register_aliases(["factorized", "Factorized"], FactorizedOperator)
    _data.matmul.add_specialisations(
        [
            (FactorizedOperator, Dense, Dense, _matmul_factorized_dense),
            (FactorizedOperator, FactorizedOperator, FactorizedOperator, _matmul_factorized_factorized),
        ]
    )
    _data.mul.add_specialisations([(FactorizedOperator, FactorizedOperator, _mul)])
    _data.imul.add_specialisations([(FactorizedOperator, FactorizedOperator, _mul)])
    _data.neg.add_specialisations([(FactorizedOperator, FactorizedOperator, _neg)])
    _data.adjoint.add_specialisations([(FactorizedOperator, FactorizedOperator, _adjoint)])
    _data.conj.add_specialisations([(FactorizedOperator, FactorizedOperator, _conj)])
    _data.transpose.add_specialisations([(FactorizedOperator, FactorizedOperator, _transpose)])
    _data.trace.add_specialisations([(FactorizedOperator, _trace)])
    _data.iszero.add_specialisations([(FactorizedOperator, _iszero)])
    _data.isherm.add_specialisations([(FactorizedOperator, _isherm)])
    _data.isequal.add_specialisations(
        [
            (FactorizedOperator, FactorizedOperator, _isequal),
            (FactorizedOperator, Data, _isequal),
            (Data, FactorizedOperator, _isequal),
        ]
    )
    _data.expect.add_specialisations([(FactorizedOperator, Dense, _expect)])
    _REGISTERED = True


register_data_type()


# ---- constructors and the public application -------------------------------------------------------------------------------------


def factorized_qobj(
    dims: Sequence[int], factors: Mapping[int, np.ndarray | qt.Qobj], scale: complex = 1.0
) -> qt.Qobj:
    """A ``Qobj`` on the joint space of ``dims`` whose data is the factorized operator (identities on the factors not named)."""
    mats = {int(f): (a.full() if isinstance(a, qt.Qobj) else np.asarray(a)) for f, a in factors.items()}
    data = FactorizedOperator(dims, mats, scale)
    d = [int(x) for x in dims]
    return qt.Qobj(data, dims=[d, d], copy=False)


def is_factorized(op: qt.Qobj | qt.QobjEvo | Data) -> bool:
    """Whether a Qobj (or every element of a QobjEvo) carries factorized data."""
    if isinstance(op, Data):
        return isinstance(op, FactorizedOperator)
    if isinstance(op, qt.Qobj):
        return isinstance(op.data, FactorizedOperator)
    # a QobjEvo: every coefficient-bearing element (the drive terms) must be factorized; the constant part is CSR
    kinds = [
        isinstance(el[0].data, FactorizedOperator)
        for el in op.to_list()
        if isinstance(el, list) and isinstance(el[0], qt.Qobj)  # a function element is [callable, args]
    ]
    return bool(kinds) and all(kinds)


def apply_drive_kernel(op: qt.Qobj | FactorizedOperator, state: qt.Qobj | np.ndarray) -> qt.Qobj | np.ndarray:
    """Apply a factorized drive operator (a ``Qobj`` or its data) to a ket, a density matrix (column by column) or an array;
    a ``Qobj`` state returns a ``Qobj`` with its dims."""
    data = op.data if isinstance(op, qt.Qobj) else op
    if not isinstance(data, FactorizedOperator):
        raise TypeError(
            "apply_drive_kernel takes a factorized operator; build one with HilbertSpace.drive_operator_factorized"
        )
    if isinstance(state, qt.Qobj):
        arr = np.asarray(state.full())
        return qt.Qobj(data.apply(arr), dims=state.dims, copy=False)
    return data.apply(np.asarray(state))


__all__ = [
    "COST_CSR_NS_PER_NNZ",
    "COST_CSR_US",
    "COST_NS_PER_MAC",
    "COST_STEP_US",
    "COST_TERM_US",
    "M9B",
    "FactorizedOperator",
    "KernelChoice",
    "apply_drive_kernel",
    "factorized_qobj",
    "is_factorized",
    "kernel_costs_us",
    "prefer_factorized",
    "register_data_type",
]
