"""Mode-factorized application of sigma_+ (x) prod_m D_m for matrix-free propagation (PLAN.md Section 11.3 item 4; Sections
5.1.1, 11.1, 11.2, 11.5; milestone M9b).

The drive operator of Section 5.2, sigma_+^i (x) prod_m D_m(i eta_im), is a tensor product, so applying it to a state costs
sum_m d_m multiply-adds per amplitude when it is applied mode by mode instead of prod_m d_m when it is assembled: 24 against
512 for three modes at d_m = 8 (Section 11.3). This module holds the operator as its factors and applies them one axis at a
time, as a QuTiP data-layer type, so that the one builder, ``sesolve``, ``mcsolve`` and the integrator ladder of Section 5.3
run unchanged on it:

- :class:`FactorizedOperator` is a :class:`qutip.core.data.Data` subclass: the tensor dims of the joint space, a small dense
  matrix per factor it acts on (identities elsewhere) and a scalar. Its application slices the ion factor when that factor has
  a single non-zero element (sigma_+, sigma_-: the block of the source level is mapped into the target level and the rest of the
  output is zero), multiplies a diagonal factor (a light-shift projector) by broadcasting, and applies every other factor as one
  batched matrix product along its axis, never forming a transpose or the assembled matrix.
- The type is registered with QuTiP's dispatchers at import: ``matmul`` on a Dense state (the right-hand side), ``matmul`` of two
  factorized operators (factor by factor: the c^dag c of a factorized collapse operator stays factorized), scalar ``mul`` and
  ``neg``, ``adjoint``, ``conj``, ``transpose``, ``trace``, ``iszero``, ``isherm``, ``isequal`` and ``expect``, plus conversions to
  Dense and CSR. Anything else (a sum of two factorized operators, a Liouvillian) converts to Dense through QuTiP's own
  shortest-path conversion, which assembles the matrix: that is the correct fallback and the reason the engine builds the
  assembled CSR operator on every ``mesolve`` path (Section 5.3: there the Liouvillian's memory decides, not the product).
  ``QobjEvo.compress`` compares elements with ``Qobj.__eq__``, so ``isequal`` is exact and cheap when the two operators share
  their factor structure and otherwise a deterministic probe test (two fixed random vectors), never an assembly.
- The cost model of Section 11.2 with the constants of ``bench_ms_timing_v5.py`` (Appendix D) decides the ``auto`` choice of
  ``BuilderOptions.kernel``: an assembled CSR term costs its non-zeros times a memory-bound 0.57 ns, a factorized term a
  fixed dispatch cost plus, per factor, a call cost and the block size times the factor dimension times 0.6 ns. On the
  Section 11.1 fixture that puts the crossover between the 256 and 440 rows: through ``QobjEvo.matmul`` one term costs 22 us
  factorized against 304 us CSR at dimension 2048 (the bare products 17 and 313 us, the dense product 691 us) and 3.3 against
  0.8 us at dimension 48, where the assembled operators win **[recomputed here]**.

The Section 5.1.1 rules hold unchanged: the factors ARE the per-mode matrix exponentials the space builds and oracle-checks
(``HilbertSpace.displacement_factor``), the product form is the physical operator in a product space, and an ENR group's
sum-generator exponential is not a product of per-mode factors, so a drive that couples to an ENR mode is never factorized.
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

# ---- the cost model of Section 11.2 with the constants of bench_ms_timing_v5.py [recomputed here] --------------------------
# Per drive term, through QobjEvo.matmul (dispatch included), least-squares fit over eleven spaces from dimension 48 to 8192:
# factorized ~ 2.9 us per factor step + 0.58 ns per complex multiply-add; CSR ~ 0.57 ns per non-zero. The fit reproduces the
# measured choice on every space; the crossover on the Section 11.1 fixture sits between the 256 and 440 rows.
COST_CSR_US = 0.5
"""Fixed cost of one assembled CSR term."""
COST_CSR_NS_PER_NNZ = 0.57
"""A CSR product costs 0.5 to 0.6 ns per non-zero, memory-bound (Section 11.2; 304 us for 524288 non-zeros at dimension 2048)."""
COST_TERM_US = 2.0
"""Dispatch, allocation and the zero fill of the output per factorized term (the QuTiP element loop's fixed cost)."""
COST_STEP_US = 2.5
"""One batched matrix product call per factor."""
COST_NS_PER_MAC = 0.6
"""Per complex multiply-add of the batched products (small matrices through the vendor BLAS)."""


def _prod(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def kernel_costs_us(dims: Sequence[int], ion_factor: int, mode_factors: Sequence[int]) -> tuple[float, float]:
    """(assembled CSR, factorized) microseconds per application of one drive term sigma_+^i (x) prod_m D_m on ``dims``.

    The assembled term has D prod_m d_m / d_ion non-zeros (2^{N-1} prod_m d_m^2 for two-level ions, Section 5.1.1); the
    factorized term applies each mode factor to the D/d_ion amplitudes of the source level.
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
    """A tensor-product operator on a product space, held as its factors and applied axis by axis (Section 11.3 item 4).

    ``dims`` are the tensor dimensions of the joint space (``HilbertSpace.dims``: ions, then resolved modes), ``factors`` map a
    factor index to its square matrix (the identity on every factor not named) and ``scale`` multiplies the whole operator.
    The matrix it represents is scale x (x)_f A_f with A_f the identity where absent.
    """

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

    # -- Data protocol -----------------------------------------------------------------------------------------------------

    def copy(self) -> FactorizedOperator:
        return FactorizedOperator(self.dims, {k: v.copy() for k, v in self.factors.items()}, self.scale)

    def to_array(self) -> np.ndarray:
        """The assembled dense matrix scale x (x)_f A_f (tests and conversions only; never on the propagation path)."""
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

    # -- the matrix-free product ---------------------------------------------------------------------------------------------

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """(scale x (x)_f A_f) @ arr for ``arr`` of shape (D,) or (D, ncols), returned in the same shape (C order)."""
        return self._plan.apply(arr)

    @property
    def max_abs(self) -> float:
        """The largest element modulus of the represented matrix: |scale| prod_f max|A_f| (the identity contributes 1)."""
        out = abs(self.scale)
        for a in self.factors.values():
            out *= float(np.max(np.abs(a))) if a.size else 0.0
        return out

    @property
    def nnz_assembled(self) -> int:
        """Non-zeros the assembled operator would have (the Section 11.5 memory the kernel avoids)."""
        out = 1
        for f, d in enumerate(self.dims):
            a = self.factors.get(f)
            out *= d if a is None else int(np.count_nonzero(a))
        return out


class _ApplicationPlan:
    """How to apply (x)_f A_f to an array reshaped to ``dims + (ncols,)``: which axes are sliced (a factor with one non-zero
    element), which are scaled by a diagonal (broadcast) and which take a batched matrix product, precomputed once."""

    __slots__ = ("block_dims", "d", "dims", "prefactor", "sliced", "slices_in", "slices_out", "steps")

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
        steps: list[tuple[str, int, int, int, np.ndarray]] = []
        for f, kind, mat in remaining:
            ax = kept.index(f)
            lead = _prod(self.block_dims[:ax])
            trail = _prod(self.block_dims[ax + 1 :])
            steps.append((kind, lead, int(dims[f]), trail, mat))
        self.steps = tuple(steps)
        self.sliced = tuple(sliced)
        self.slices_in: tuple[int | slice, ...] = tuple(slices_in)
        self.slices_out: tuple[int | slice, ...] = tuple(slices_out)
        self.prefactor = prefactor

    def apply(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=complex)
        vector = a.ndim == 1
        if vector:
            a = a.reshape(-1, 1)
        if a.shape[0] != self.d:
            raise ValueError(f"state of dimension {a.shape[0]} does not live on dims {self.dims}")
        ncols = a.shape[1]
        if self.prefactor == 0.0:
            out = np.zeros_like(a)
            return out.reshape(-1) if vector else out
        x = np.ascontiguousarray(a).reshape(self.dims + (ncols,))
        y = x[self.slices_in]
        for kind, lead, d, trail, mat in self.steps:
            y3 = y.reshape(lead, d, trail * ncols)
            if kind == "dense":
                y = np.matmul(mat, y3)
            else:
                y = mat[None, :, None] * y3
        y = y.reshape(self.block_dims + (ncols,))
        if self.sliced:
            out = np.zeros(self.dims + (ncols,), dtype=complex)
            out[self.slices_out] = y
        else:
            out = np.ascontiguousarray(y)
        out = out.reshape(-1, ncols)
        if self.prefactor != 1.0:
            out = out * self.prefactor
        return out.reshape(-1) if vector else out


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
    out = left.apply(right.as_ndarray())
    if scale != 1:
        out = out * scale
    return Dense(out, copy=False)


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
    """A fixed complex Gaussian probe vector (deterministic: the equality tests below are reproducible)."""
    rng = np.random.default_rng([20260907, dimension, k])
    return np.asarray(rng.normal(size=(dimension, 1)) + 1j * rng.normal(size=(dimension, 1)))


def _apply_any(matrix: Data, v: np.ndarray) -> np.ndarray:
    if isinstance(matrix, FactorizedOperator):
        return matrix.apply(v)
    return np.asarray(_data.matmul(matrix, Dense(v, copy=False)).to_array())


def _isequal(a: Data, b: Data, atol: float = -1, rtol: float = -1) -> bool:
    """Equality within tolerance: exact factor by factor when both operators share their factor structure, else a probe test
    with two fixed vectors (equal operators always pass; a difference above the tolerance on the probes fails), never an
    assembly of either operator."""
    if atol < 0:
        atol = float(qt.settings.core["atol"])
    if rtol < 0:
        rtol = float(qt.settings.core["rtol"])
    if a.shape != b.shape:
        return False
    if (
        isinstance(a, FactorizedOperator)
        and isinstance(b, FactorizedOperator)
        and a.dims == b.dims
        and set(a.factors) == set(b.factors)
    ):
        same = True
        for f, x in a.factors.items():
            if not np.allclose(a.scale * x, b.scale * b.factors[f], atol=atol, rtol=rtol):
                same = False
                break
        if same:
            return True
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
    # a QobjEvo: every coefficient-bearing element (the drive terms) is factorized; the constant part (H_mot + H_int) is CSR
    kinds = [isinstance(el[0].data, FactorizedOperator) for el in op.to_list() if isinstance(el, list)]
    return bool(kinds) and all(kinds)


def apply_drive_kernel(op: qt.Qobj | FactorizedOperator, state: qt.Qobj | np.ndarray) -> qt.Qobj | np.ndarray:
    """Apply a drive operator to a state mode by mode (Section 11.3 item 4): the matrix-free product the right-hand side uses.

    ``op`` is a factorized ``Qobj`` (``HilbertSpace.drive_operator_factorized``) or the data itself; ``state`` a ket, a density
    matrix (every column is multiplied) or a NumPy array. A ``Qobj`` in returns a ``Qobj`` with the state's dims.
    """
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
