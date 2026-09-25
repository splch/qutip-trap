"""The mode-factorized drive operator sigma_+ (x) prod_m D_m as a QuTiP data type.

Applied factor by factor, a drive term costs sum_m d_m multiply-adds per amplitude instead of the prod_m d_m of the
assembled matrix. :class:`FactorizedOperator` holds the operator as its factors and registers with QuTiP's dispatchers
for what the solvers call on it: ``matmul`` on a Dense state (the right-hand side), ``matmul`` of two factorized operators
(the c^dag c of a factorized collapse operator), scalar ``mul``, ``adjoint``, ``iszero`` and a structural ``isequal``.
Anything else converts to Dense through QuTiP's conversion graph, which assembles the matrix (the reason every ``mesolve``
segment builds the assembled CSR operator). The factors are the per-mode exponentials the space oracle-checks; an ENR
group's sum-generator exponential is not a product, so a drive on an ENR mode is never factorized.

The ``auto`` choice of ``BuilderOptions.kernel`` is the cost model below, fitted through ``QobjEvo.matmul`` over spaces
of dimension 48 to 8192: an assembled term costs its non-zeros at 0.57 ns, a factorized one a fixed dispatch cost plus a
call and a batched product per factor; on the two-ion benchmark fixture the crossover lies between dimensions 256 and 440.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import qutip as qt
from qutip.core import data as _data
from qutip.core.data import Data, Dense

KernelChoice = Literal["auto", "factorized", "assembled"]

COST_CSR_US = 0.5
"""Fixed cost of one assembled CSR term (us)."""
COST_CSR_NS_PER_NNZ = 0.57
"""A CSR product per non-zero (ns), memory-bound."""
COST_TERM_US = 2.0
"""Dispatch, allocation and the zero fill of the output per factorized term (us)."""
COST_STEP_US = 2.5
"""One batched matrix-product call per factor (us)."""
COST_NS_PER_MAC = 0.6
"""Per complex multiply-add of the batched products (ns)."""


def _prod(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def kernel_costs_us(dims: Sequence[int], ion_factor: int, mode_factors: Sequence[int]) -> tuple[float, float]:
    """(assembled CSR, factorized) microseconds per application of one drive term sigma_+^i (x) prod_m D_m on ``dims``.

    The assembled term has D prod_m d_m / d_ion non-zeros; the factorized term applies each mode factor to the D/d_ion
    amplitudes of the source level.
    """
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
    """scale x (x)_f A_f on a product space of tensor ``dims`` (ions, then resolved modes), ``factors`` mapping a factor
    index to its square matrix (the identity on every factor not named), applied axis by axis."""

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
        """The assembled dense matrix (conversions only; never on the propagation path)."""
        out = np.array([[self.scale]], dtype=complex)
        for f, d in enumerate(self.dims):
            a = self.factors.get(f)
            out = np.kron(out, np.eye(d, dtype=complex) if a is None else a)
        return out

    def adjoint(self) -> FactorizedOperator:
        return FactorizedOperator(
            self.dims, {k: v.conj().T for k, v in self.factors.items()}, np.conj(self.scale)
        )

    def __reduce__(self) -> tuple[object, ...]:
        return (FactorizedOperator, (self.dims, self.factors, self.scale))

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """(scale x (x)_f A_f) @ arr for ``arr`` of shape (D,) or (D, ncols), returned in the same shape (C order)."""
        return self._plan.apply(arr)

    @property
    def max_abs(self) -> float:
        """The largest element modulus: |scale| prod_f max|A_f| (the identity contributes 1)."""
        out = abs(self.scale)
        for a in self.factors.values():
            out *= float(np.max(np.abs(a))) if a.size else 0.0
        return out


class _ApplicationPlan:
    """How to apply (x)_f A_f to an array reshaped to ``dims + (ncols,)``, precomputed once: a factor with one non-zero
    element is a slice of the source block into the target block, a diagonal factor a broadcast product, any other factor
    one BLAS call (a trailing factor as (lead, d) @ A^T, a leading one as A @ (d, trail), a middle one batched). The scalar
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
        # (kind, lead, d, trail, matrix or diagonal, transposed matrix or None)
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
        """What still multiplies the block after the steps: 1 once the prefactor is folded, else the prefactor itself (a
        bare sigma_+ with no mode factor)."""

    def _transform(self, a: np.ndarray, ncols: int, alpha: complex = 1.0) -> np.ndarray:
        """alpha x (prefactor (x)_kept A_f) on the sliced-in block of ``a`` (shape (D, ncols), C order), returned with
        shape ``block_dims + (ncols,)``."""
        y = a.reshape(self.dims + (ncols,))[self.slices_in]
        first = True
        for kind, lead, d, trail, mat, mat_t in self.steps:
            if first and alpha != 1.0:
                # alpha rides on the first factor (d^2 multiplications) rather than on the output
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
            return np.zeros(a.shape, dtype=complex)
        y = self._transform(a, ncols)
        if self.sliced:
            out = np.zeros(self.dims + (ncols,), dtype=complex)
            out[self.slices_out] = y
        else:
            out = y
        return out.reshape(-1) if vector else out.reshape(-1, ncols)

    def apply_into(self, a: np.ndarray, alpha: complex, out: np.ndarray) -> None:
        """``out += alpha x (self @ a)`` for C-contiguous complex ``a`` and ``out`` of one shape (D, ncols), with no zero
        fill of its own (the accumulation of ``dynamics.rotating.PhasedSum``)."""
        if self.prefactor == 0.0 or alpha == 0.0:
            return
        y = self._transform(a, a.shape[1], complex(alpha))
        block = out.reshape(self.dims + (a.shape[1],))[self.slices_out]
        block += y


# ---- dispatcher specialisations ----------------------------------------------------------------------------------------------------


def _to_dense(matrix: FactorizedOperator) -> Dense:
    return Dense(matrix.to_array(), copy=False)


def _from_dense(matrix: Dense) -> FactorizedOperator:
    """A dense matrix as a one-factor operator (the direction QuTiP's conversion registry requires)."""
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
    """(x)_f A_f (x)_f B_f = (x)_f A_f B_f on a shared factor structure; assembled when the structures differ."""
    if left.dims != right.dims:
        if left.shape != right.shape:
            raise ValueError("operator shapes do not match")
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


def _adjoint(matrix: FactorizedOperator) -> FactorizedOperator:
    return matrix.adjoint()


def _iszero(matrix: FactorizedOperator, tol: float = -1) -> bool:
    if tol < 0:
        tol = float(qt.settings.core["atol"])
    return matrix.max_abs <= tol


def _scalar_ratio(x: np.ndarray, y: np.ndarray, atol: float, rtol: float) -> complex | None:
    """The c with ``x == c y`` elementwise, or None when ``x`` is not a scalar multiple of ``y``."""
    big = float(np.max(np.abs(y))) if y.size else 0.0
    if big == 0.0:
        return None
    i = np.unravel_index(int(np.argmax(np.abs(y))), y.shape)
    c = complex(x[i] / y[i])
    return c if np.allclose(x, c * y, atol=atol, rtol=rtol) else None


def _isequal(a: FactorizedOperator, b: FactorizedOperator, atol: float = -1, rtol: float = -1) -> bool:
    """Equality within tolerance. On one factor structure it is exact and never assembles: over the union of the factor
    indices each pair must be a scalar multiple, and the ratios must carry one scale into the other (scale_a (x)_f A_f =
    scale_b (x)_f B_f iff A_f = c_f B_f with scale_a prod_f c_f = scale_b); ``QobjEvo.compress`` merges terms on a "yes".
    Two structures over one dimension are compared assembled."""
    if atol < 0:
        atol = float(qt.settings.core["atol"])
    if rtol < 0:
        rtol = float(qt.settings.core["rtol"])
    if a.shape != b.shape:
        return False
    if a.dims != b.dims:
        return bool(np.allclose(a.to_array(), b.to_array(), atol=atol, rtol=rtol))
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


def _register_data_type() -> None:
    """Register :class:`FactorizedOperator` with QuTiP's conversion registry and dispatchers (once, at import)."""
    if FactorizedOperator in _data.to.dtypes:
        return
    _data.to.add_conversions(
        [
            (Dense, FactorizedOperator, _to_dense, 1.0),
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
    _data.adjoint.add_specialisations([(FactorizedOperator, FactorizedOperator, _adjoint)])
    _data.iszero.add_specialisations([(FactorizedOperator, _iszero)])
    _data.isequal.add_specialisations([(FactorizedOperator, FactorizedOperator, _isequal)])


_register_data_type()


def factorized_qobj(
    dims: Sequence[int], factors: Mapping[int, np.ndarray | qt.Qobj], scale: complex = 1.0
) -> qt.Qobj:
    """A ``Qobj`` on the joint space of ``dims`` whose data is the factorized operator (identities on the factors not named)."""
    mats = {int(f): (a.full() if isinstance(a, qt.Qobj) else np.asarray(a)) for f, a in factors.items()}
    data = FactorizedOperator(dims, mats, scale)
    d = [int(x) for x in dims]
    return qt.Qobj(data, dims=[d, d], copy=False)
