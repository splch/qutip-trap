"""The composite space of a run: one qudit per ion, truncated modes, an optional ENR group (PLAN.md Section 5.1).

Each ion is a qudit (2 levels, or 2 + leakage levels); each resolved mode a truncated oscillator of d_m Fock levels; an
optional ENR group of cold undriven modes shares one excitation cap N_exc; frozen spectators enter only through their
Debye-Waller factors. The tensor order is ions, then the resolved modes in the order of ``resolved``, then
the ENR group as one factor: a ``factor`` is a position in that order, a ``mode`` a position in ``Crystal.modes``.
Operators are built once per space and cached.

A drive's displacement is the matrix exponential (per mode in a product space, of the sum generator in an
ENR group), asserted against the analytic oracle over the declared ``expected_n_range``; dimensions are read from shapes,
because QuTiP labels an ENR factor by per-mode dims whose product is not its size, and marginals over an ENR factor are
index sums over ``enr_state_dictionaries`` (``ptrace`` refuses them).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import comb, prod
from typing import TYPE_CHECKING, Literal

import numpy as np
import qutip as qt

from qutip_trap.hilbert.operators import (
    _element_error,
    displacement_operator,
    interior_tolerance,
    oracle_check,
    qudit_projector,
    qudit_sigma_minus,
    qudit_sigma_plus,
    qudit_sigma_z,
    thermal_populations,
)

if TYPE_CHECKING:
    from qutip_trap.dynamics.engine import State

ORACLE_MIN_MARGIN = 4
"""The smallest margin the Section 5.1.1 table covers (its d = 8 row); below it the oracle is reported, not asserted."""

IDENTITY_CHECK_MAX_DIMENSION = 1 << 16
"""Above this joint dimension ``check()`` asserts shape == prod(dims) from the factor shapes instead of forming the joint
identity, which would allocate O(D) before the size guards could refuse the space."""

ModeClass = Literal["resolved", "enr", "frozen"]


def enr_dimension(n_modes: int, n_exc: int) -> int:
    """Number of Fock tuples of ``n_modes`` modes with total excitation <= n_exc: C(n_modes + n_exc, n_exc)."""
    if n_modes <= 0 or n_exc < 0:
        raise ValueError("an ENR group has at least one mode and a non-negative cap")
    return comb(n_modes + n_exc, n_exc)


@dataclass(frozen=True)
class ModeTruncation:
    """The truncation of one resolved mode (Section 5.1.1): the mode, its Fock dimension d, the populated range the cap was
    derived for, the largest Lamb-Dicke parameter the drives put on it, and the interior-element tolerance the cap was
    derived at (None for a fixture cap)."""

    mode: int
    d: int
    expected_n_range: tuple[int, int]
    eta_max: float
    element_tol: float | None = None
    """With a declared tolerance the oracle asserts the exponential's elements to it at every margin; the fixture cap
    asserts the table's tolerance at margins of ``ORACLE_MIN_MARGIN`` and more."""

    def __post_init__(self) -> None:
        if self.d < 2:
            raise ValueError("a resolved mode needs at least two Fock levels")
        lo, hi = self.expected_n_range
        if lo < 0 or hi < lo:
            raise ValueError("expected_n_range must be ordered and non-negative")
        if hi > self.d - 1:
            raise ValueError(f"expected occupation up to n = {hi} exceeds n_max = {self.d - 1}")
        if self.eta_max < 0.0:
            raise ValueError("eta_max is a magnitude")
        if self.element_tol is not None and self.element_tol <= 0.0:
            raise ValueError("element_tol is a positive tolerance")

    @property
    def margin_levels(self) -> int:
        """Levels between the top of the expected range and the cap."""
        return self.d - 1 - self.expected_n_range[1]

    def grown(self, add: int) -> ModeTruncation:
        """The same declaration with ``add`` more Fock levels."""
        if add <= 0:
            raise ValueError("grow by a positive number of levels")
        return ModeTruncation(self.mode, self.d + add, self.expected_n_range, self.eta_max, self.element_tol)


@dataclass(frozen=True)
class HilbertSpace:
    """The composite space of a run (Sections 5.1, 5.4): one qudit factor per ion, one truncated oscillator per resolved
    mode, an optional ENR group as one factor, and the frozen spectators."""

    ion_dims: tuple[int, ...]
    """2, or 2 + leakage levels, per ion."""
    resolved: tuple[ModeTruncation, ...]
    enr_group: tuple[tuple[int, ...], int] | None
    """(modes, N_exc), optional."""
    frozen: tuple[int, ...]
    ions: tuple[int, ...] = ()
    """The device ion of each ion factor, in factor order, so that every ``ion`` argument is a device index (a GATE_LOCAL
    space carries a subset of the crystal); empty = (0, 1, ..., n_ions - 1)."""
    dropped: tuple[int, ...] = ()
    """The subset of ``frozen`` that Section 5.2 dropped: no tensor factor like a frozen mode, and also no Debye-Waller
    factor and no Fock branch (the mode is not modelled at all)."""

    def __post_init__(self) -> None:
        self.check()

    # ---- bookkeeping ---------------------------------------------------------------------------------------------

    def check(self) -> None:
        """The Section 5.1 invariants: one class per mode, and shape == prod(dims) with the ENR factor's C(M + N_exc, N_exc)
        read from the Qobj. Nothing joint is allocated above ``IDENTITY_CHECK_MAX_DIMENSION``."""
        if not self.ion_dims or any(d < 2 for d in self.ion_dims):
            raise ValueError("every ion is a qudit of dimension >= 2")
        if self.ions:
            if len(self.ions) != len(self.ion_dims):
                raise ValueError("ions names the device ion of every ion factor (one per factor)")
            if len(set(self.ions)) != len(self.ions) or any(i < 0 for i in self.ions):
                raise ValueError("ions are distinct non-negative positions in Crystal")
        resolved = [m.mode for m in self.resolved]
        enr_modes = list(self.enr_group[0]) if self.enr_group is not None else []
        frozen = list(self.frozen)
        all_modes = resolved + enr_modes + frozen
        if len(set(all_modes)) != len(all_modes):
            raise ValueError("a mode is resolved, in the ENR group or frozen, never in two classes")
        if not set(self.dropped) <= set(frozen):
            raise ValueError(
                "dropped names the subset of frozen that Section 5.2 dropped (no Debye-Waller factor, no Fock branch); "
                f"{sorted(set(self.dropped) - set(frozen))} are not frozen modes of this space"
            )
        if any(m < 0 for m in all_modes):
            raise ValueError("mode indices are positions in Crystal.modes and non-negative")
        if self.enr_group is not None and (not enr_modes or self.enr_group[1] < 0):
            raise ValueError("an ENR group names at least one mode and a non-negative excitation cap")
        if self.enr_group is not None:
            n_modes, n_exc = len(self.enr_group[0]), self.enr_group[1]
            d_enr = enr_dimension(n_modes, n_exc)
            if d_enr <= IDENTITY_CHECK_MAX_DIMENSION and self.enr_identity().shape[0] != d_enr:
                raise ValueError("ENR factor shape disagrees with C(M + N_exc, N_exc)")
        if self.dimension <= IDENTITY_CHECK_MAX_DIMENSION:
            ident = self.identity()
            if ident.shape[0] != self.dimension or ident.shape[1] != self.dimension:
                raise ValueError(f"joint identity has shape {ident.shape}, expected {self.dimension}")

    @property
    def n_ions(self) -> int:
        return len(self.ion_dims)

    @property
    def ion_labels(self) -> tuple[int, ...]:
        """The device ion of every ion factor, in factor order."""
        return self.ions if self.ions else tuple(range(self.n_ions))

    def has_ion(self, ion: int) -> bool:
        return ion in self.ion_labels

    def ion_dim(self, ion: int) -> int:
        """The qudit dimension of device ion ``ion``."""
        return self.ion_dims[self.ion_factor(ion)]

    @property
    def dims(self) -> list[int]:
        """Tensor factors in order: ions, resolved modes, then the ENR group as one factor."""
        d = list(self.ion_dims) + [m.d for m in self.resolved]
        if self.enr_group is not None:
            d.append(enr_dimension(len(self.enr_group[0]), self.enr_group[1]))
        return d

    @property
    def dimension(self) -> int:
        return prod(self.dims)

    @property
    def enr_factor(self) -> int | None:
        return None if self.enr_group is None else self.n_ions + len(self.resolved)

    def mode_class(self, mode: int) -> ModeClass:
        if any(m.mode == mode for m in self.resolved):
            return "resolved"
        if self.enr_group is not None and mode in self.enr_group[0]:
            return "enr"
        if mode in self.frozen:
            return "frozen"
        raise KeyError(f"mode {mode} is not carried by this space (resolved, ENR or frozen)")

    def truncation(self, mode: int) -> ModeTruncation:
        for m in self.resolved:
            if m.mode == mode:
                return m
        raise KeyError(f"mode {mode} is not a resolved mode of this space")

    def resolved_position(self, mode: int) -> int:
        for k, m in enumerate(self.resolved):
            if m.mode == mode:
                return k
        raise KeyError(f"mode {mode} is not a resolved mode of this space")

    def ion_factor(self, ion: int) -> int:
        """The tensor factor carrying device ion ``ion``."""
        if self.ions:
            try:
                return self.ions.index(ion)
            except ValueError:
                raise KeyError(f"ion {ion} is not carried by this space (ions {self.ions})") from None
        if not 0 <= ion < self.n_ions:
            raise IndexError(f"ion {ion} out of range for {self.n_ions} ions")
        return ion

    def mode_factor(self, mode: int) -> int:
        """The tensor factor carrying ``mode``: its resolved position, or the ENR factor for a group member."""
        cls = self.mode_class(mode)
        if cls == "resolved":
            return self.n_ions + self.resolved_position(mode)
        if cls == "enr":
            assert self.enr_factor is not None
            return self.enr_factor
        raise KeyError(f"mode {mode} is frozen and has no tensor factor")

    def enr_position(self, mode: int) -> int:
        if self.enr_group is None or mode not in self.enr_group[0]:
            raise KeyError(f"mode {mode} is not in the ENR group")
        return self.enr_group[0].index(mode)

    def _enr_dims(self) -> tuple[list[int], int]:
        assert self.enr_group is not None
        modes, n_exc = self.enr_group
        return [n_exc + 1] * len(modes), n_exc

    # ---- factor operators ---------------------------------------------------------------------------------------

    def enr_identity(self) -> qt.Qobj:
        dims, n_exc = self._enr_dims()
        return qt.enr_identity(dims, n_exc)

    def factor_identities(self) -> list[qt.Qobj]:
        ops: list[qt.Qobj] = [qt.qeye(d) for d in self.ion_dims] + [qt.qeye(m.d) for m in self.resolved]
        if self.enr_group is not None:
            ops.append(self.enr_identity())
        return ops

    def with_space_dims(self, op: qt.Qobj) -> qt.Qobj:
        """``op`` relabelled with this space's ``dims`` (the ENR group as one factor of dimension C(M + N_exc, N_exc)), the
        one labelling every joint operator and state of the space carries; the data is shared."""
        if self.enr_group is None:
            return op
        d = self.dims
        if op.isket:
            dims = [d, [1]]
        elif op.isbra:
            dims = [[1], d]
        else:
            dims = [d, d]
        if op.dims == dims:
            return op
        return qt.Qobj(op.data, dims=dims)

    def identity(self) -> qt.Qobj:
        return self.with_space_dims(qt.tensor(*self.factor_identities()))

    def embed(self, op: qt.Qobj, factor: int) -> qt.Qobj:
        """``op`` on one tensor factor, identities elsewhere (CSR)."""
        ops = self.factor_identities()
        if not 0 <= factor < len(ops):
            raise IndexError(f"factor {factor} out of range")
        if op.shape != ops[factor].shape:
            raise ValueError(
                f"operator shape {op.shape} does not fit factor {factor} of shape {ops[factor].shape}"
            )
        ops[factor] = op
        return self.with_space_dims(qt.tensor(*ops).to("CSR"))

    def embed_many(self, ops_by_factor: Mapping[int, qt.Qobj]) -> qt.Qobj:
        ops = self.factor_identities()
        for f, op in ops_by_factor.items():
            if op.shape != ops[f].shape:
                raise ValueError(f"operator shape {op.shape} does not fit factor {f}")
            ops[f] = op
        return self.with_space_dims(qt.tensor(*ops).to("CSR"))

    def sigma_plus(self, ion: int) -> qt.Qobj:
        """|1><0| on device ion ``ion``."""
        f = self.ion_factor(ion)
        return _cached(self, ("sigma_plus", f), lambda: self.embed(qudit_sigma_plus(self.ion_dims[f]), f))

    def sigma_minus(self, ion: int) -> qt.Qobj:
        f = self.ion_factor(ion)
        return _cached(self, ("sigma_minus", f), lambda: self.embed(qudit_sigma_minus(self.ion_dims[f]), f))

    def sigma_z(self, ion: int) -> qt.Qobj:
        """The energy sigma_z = |1><1| - |0><0| on ``ion``."""
        f = self.ion_factor(ion)
        return _cached(self, ("sigma_z", f), lambda: self.embed(qudit_sigma_z(self.ion_dims[f]), f))

    def projector(self, ion: int, level: int) -> qt.Qobj:
        f = self.ion_factor(ion)
        return _cached(
            self,
            ("projector", f, level),
            lambda: self.embed(qudit_projector(self.ion_dims[f], level), f),
        )

    def annihilation(self, mode: int) -> qt.Qobj:
        """a_m on the joint space: the truncated ladder operator of a resolved mode, ``enr_destroy`` of an ENR member."""

        def build() -> qt.Qobj:
            cls = self.mode_class(mode)
            if cls == "resolved":
                return self.embed(qt.destroy(self.truncation(mode).d), self.mode_factor(mode))
            if cls == "enr":
                dims, n_exc = self._enr_dims()
                ops = qt.enr_destroy(dims, n_exc)
                return self.embed(ops[self.enr_position(mode)], self.mode_factor(mode))
            raise KeyError(f"mode {mode} is frozen and has no ladder operator")

        return _cached(self, ("a", mode), build)

    def number(self, mode: int) -> qt.Qobj:
        a = self.annihilation(mode)
        return _cached(self, ("n", mode), lambda: (a.dag() * a).to("CSR"))

    def position(self, mode: int) -> qt.Qobj:
        """(a + a^dag) on the joint space; x = x0 (a + a^dag)."""
        a = self.annihilation(mode)
        return _cached(self, ("x", mode), lambda: (a + a.dag()).to("CSR"))

    def displacement_factor(self, mode: int, eta: float) -> qt.Qobj:
        """D(i eta) of one resolved mode in its own factor space, oracle-checked over the declared range."""
        tr = self.truncation(mode)
        if abs(eta) > tr.eta_max * (1.0 + 1e-12):
            raise ValueError(
                f"mode {mode}: |eta| = {abs(eta):.4g} exceeds the declared eta_max = {tr.eta_max:.4g} of this space"
            )

        def build() -> qt.Qobj:
            # rule (ii): a margin below the table's smallest row is left to the boundary monitor, unless the cap was
            # derived for a declared element tolerance
            if tr.element_tol is not None:
                oracle_check(tr.d, 1j * eta, tr.expected_n_range[1], tr.element_tol)
            elif tr.margin_levels >= ORACLE_MIN_MARGIN:
                oracle_check(
                    tr.d, 1j * eta, tr.expected_n_range[1], interior_tolerance(tr.margin_levels, eta)
                )
            return displacement_operator(tr.d, 1j * eta)

        return _cached(self, ("D", mode, _eta_key(eta)), build)

    def oracle_status(self, mode: int, eta: float) -> tuple[bool, float, float]:
        """(asserted, max interior element difference, tolerance) of the mode's displacement at eta (rule ii)."""
        tr = self.truncation(mode)
        diff = _element_error(tr.d, 1j * eta, tr.expected_n_range[1])
        if tr.element_tol is not None:
            return True, diff, tr.element_tol
        asserted = tr.margin_levels >= ORACLE_MIN_MARGIN
        return asserted, diff, interior_tolerance(max(tr.margin_levels, 1), eta)

    def enr_displacement(self, etas: Mapping[int, float]) -> qt.Qobj:
        """expm of the sum generator sum_m i eta_m (a_m + a_m^dag) in the ENR group's own space."""
        if self.enr_group is None:
            raise KeyError("this space has no ENR group")
        dims, n_exc = self._enr_dims()
        ops = qt.enr_destroy(dims, n_exc)
        key = tuple(sorted((m, _eta_key(e)) for m, e in etas.items()))

        def build() -> qt.Qobj:
            gen = 0.0 * qt.enr_identity(dims, n_exc)
            for m, e in etas.items():
                a = ops[self.enr_position(m)]
                gen = gen + 1j * e * (a + a.dag())
            return gen.expm().to("CSR")

        return _cached(self, ("D_enr", key), build)

    def drive_operator(
        self, ion: int, etas: Mapping[int, float], *, ion_op: qt.Qobj | None = None
    ) -> qt.Qobj:
        """sigma_+^ion (or ``ion_op`` on the ion's factor) (x) prod_m D_m(i eta_m) over the resolved modes, times the ENR
        group's sum-generator exponential (CSR). Frozen modes enter through the coefficient's Debye-Waller factor."""
        f_ion = self.ion_factor(ion)
        ops: dict[int, qt.Qobj] = {
            f_ion: qudit_sigma_plus(self.ion_dims[f_ion]) if ion_op is None else ion_op
        }
        enr_etas: dict[int, float] = {}
        for mode, eta in etas.items():
            if eta == 0.0:
                continue
            cls = self.mode_class(mode)
            if cls == "resolved":
                ops[self.mode_factor(mode)] = self.displacement_factor(mode, eta)
            elif cls == "enr":
                enr_etas[mode] = eta
        if enr_etas:
            assert self.enr_factor is not None
            ops[self.enr_factor] = self.enr_displacement(enr_etas)
        return self.embed_many(ops)

    def drive_operator_factorized(
        self, ion: int, etas: Mapping[int, float], *, ion_op: qt.Qobj | None = None
    ) -> qt.Qobj:
        """:meth:`drive_operator` held as a ``FactorizedOperator``; a product space only, since an ENR
        group's sum-generator exponential is not a product of per-mode factors (``NotImplementedError``)."""
        from qutip_trap.dynamics.kernels import factorized_qobj

        f_ion = self.ion_factor(ion)
        factors: dict[int, np.ndarray] = {
            f_ion: (qudit_sigma_plus(self.ion_dims[f_ion]) if ion_op is None else ion_op).full()
        }
        for mode, eta in etas.items():
            if eta == 0.0:
                continue
            cls = self.mode_class(mode)
            if cls == "resolved":
                factors[self.mode_factor(mode)] = self.displacement_factor(mode, eta).full()
            elif cls == "enr":
                raise NotImplementedError(
                    "a drive coupling to an ENR group is applied through its sum-generator exponential, which is not a "
                    "product of per-mode factors (Section 5.1.1): the factorized kernel does not apply"
                )
        if ion_op is None:
            key = ("D_fact", f_ion, tuple(sorted((m, _eta_key(e)) for m, e in etas.items() if e != 0.0)))
            return _cached(self, key, lambda: factorized_qobj(self.dims, factors))
        return factorized_qobj(self.dims, factors)

    # ---- states ------------------------------------------------------------------------------------------------

    def internal_ket(self, levels: Sequence[int]) -> qt.Qobj:
        """|l_0 l_1 ...> on the ion factors."""
        if len(levels) != self.n_ions:
            raise ValueError(f"one level per ion ({self.n_ions})")
        return qt.tensor(*[qt.basis(d, int(lv)) for d, lv in zip(self.ion_dims, levels)])

    def fock(self, mode: int, n: int) -> qt.Qobj:
        """|n> of a resolved mode in its own factor space."""
        d = self.truncation(mode).d
        if not 0 <= n < d:
            raise ValueError(f"Fock level {n} outside d = {d}")
        return qt.basis(d, n)

    def thermal(self, mode: int, nbar: float) -> qt.Qobj:
        """The thermal density matrix of a resolved mode at nbar, renormalized over the truncation."""
        d = self.truncation(mode).d
        p = thermal_populations(nbar, d)
        return qt.Qobj(np.diag(p / p.sum()), dims=[[d], [d]])

    def enr_state(
        self, fock: Mapping[int, int] | None = None, thermal: Mapping[int, float] | None = None
    ) -> qt.Qobj:
        """The ENR factor's state: a Fock tuple (ket) or a product thermal state (dm), exactly one of the two."""
        dims, n_exc = self._enr_dims()
        assert self.enr_group is not None
        if (fock is None) == (thermal is None):
            raise ValueError("give either Fock levels or thermal occupations for the ENR group")
        if fock is not None:
            ns = [int(fock.get(m, 0)) for m in self.enr_group[0]]
            if sum(ns) > n_exc:
                raise ValueError("ENR Fock tuple exceeds the excitation cap")
            return qt.enr_fock(dims, n_exc, ns)
        assert thermal is not None
        return qt.enr_thermal_dm(dims, n_exc, [float(thermal.get(m, 0.0)) for m in self.enr_group[0]])

    def product_state(self, internal: qt.Qobj | Sequence[int], motional: Mapping[int, qt.Qobj]) -> qt.Qobj:
        """The joint state: ``internal`` (ket or dm on the ion factors) x per-factor motional states.

        ``motional`` maps every resolved mode to its factor state and, with an ENR group, the key -1 to the group's; the
        result is a ket when every part is one, else the density matrix.
        """
        parts: list[qt.Qobj] = []
        if isinstance(internal, qt.Qobj):
            if internal.shape[0] != prod(self.ion_dims):
                raise ValueError("internal state does not match the ion factors")
            parts.append(internal)
        else:
            parts.append(self.internal_ket(internal))
        for m in self.resolved:
            if m.mode not in motional:
                raise ValueError(f"no motional state given for resolved mode {m.mode}")
            parts.append(motional[m.mode])
        if self.enr_group is not None:
            if -1 not in motional:
                raise ValueError("no state given for the ENR group (key -1)")
            parts.append(motional[-1])
        if all(p.isket for p in parts):
            return self.with_space_dims(qt.tensor(*parts))
        return self.with_space_dims(qt.tensor(*[p if p.isoper else qt.ket2dm(p) for p in parts]))

    def initial_state(
        self,
        internal: qt.Qobj | Sequence[int],
        *,
        fock: Mapping[int, int] | None = None,
        thermal: Mapping[int, float] | None = None,
        states: Mapping[int, qt.Qobj] | None = None,
        provenance: tuple[str, ...] = ("initial_state",),
    ) -> State:
        """A ``State`` from per-mode Fock levels, thermal occupations or explicit factor states.

        Resolved modes take ``states`` first, then ``fock``, then ``thermal`` (default |0>); ENR members take ``fock`` or
        ``thermal`` for the whole group; frozen modes carry only their nbar (``thermal``, default 0).
        """
        from qutip_trap.dynamics.engine import MotionalModel, State

        fock = dict(fock or {})
        thermal = dict(thermal or {})
        states = dict(states or {})
        motional: dict[int, qt.Qobj] = {}
        for m in self.resolved:
            if m.mode in states:
                motional[m.mode] = states[m.mode]
            elif m.mode in fock:
                motional[m.mode] = self.fock(m.mode, fock[m.mode])
            else:
                nb = thermal.get(m.mode, 0.0)
                motional[m.mode] = self.fock(m.mode, 0) if nb == 0.0 else self.thermal(m.mode, nb)
        if self.enr_group is not None:
            members = self.enr_group[0]
            if any(m in fock for m in members) and any(m in thermal for m in members):
                raise ValueError("an ENR group is either a Fock tuple or a product thermal state")
            if any(m in thermal for m in members):
                motional[-1] = self.enr_state(thermal={m: thermal.get(m, 0.0) for m in members})
            else:
                motional[-1] = self.enr_state(fock={m: fock.get(m, 0) for m in members})
        joint = self.product_state(internal, motional)
        reduced: dict[int, qt.Qobj] = {}
        nbar: dict[int, float] = {}
        for m in self.resolved:
            rho = self.mode_marginal(joint, m.mode)
            reduced[m.mode] = rho
            nbar[m.mode] = float(np.real(qt.expect(qt.num(m.d), rho)))
        if self.enr_group is not None:
            for em in self.enr_group[0]:
                rho_e = self.mode_marginal(joint, em)
                reduced[em] = rho_e
                nbar[em] = float(np.real(qt.expect(qt.num(rho_e.shape[0]), rho_e)))
        for fm in self.frozen:
            nbar[fm] = float(thermal.get(fm, 0.0))
        return State(
            internal=self.internal_marginal(joint),
            motional=MotionalModel(reduced=reduced, nbar=nbar, frozen=tuple(self.frozen)),
            joint=joint,
            provenance=tuple(provenance),
        )

    # ---- marginals ---------------------------------------------------------------------------------------------

    def _dense_dm(self, state: State | qt.Qobj) -> np.ndarray:
        obj = state if isinstance(state, qt.Qobj) else state.joint
        if obj is None:
            raise ValueError("the State carries no joint state to reduce")
        if obj.shape[0] != self.dimension:
            raise ValueError(
                f"state of dimension {obj.shape[0]} does not live on this space ({self.dimension})"
            )
        if obj.isket:
            v = np.asarray(obj.full()).reshape(-1)
            return np.outer(v, np.conj(v))
        return np.asarray(obj.full())

    def marginal(self, state: State | qt.Qobj, keep: tuple[int, ...]) -> qt.Qobj:
        """The reduced density matrix over the tensor factors ``keep`` (in factor order), by explicit index sums (an ENR
        group is kept or traced as one factor)."""
        dims = self.dims
        keep_sorted = tuple(sorted(set(keep)))
        if any(f < 0 or f >= len(dims) for f in keep_sorted):
            raise IndexError("keep lists factor indices")
        obj = state if isinstance(state, qt.Qobj) else state.joint
        if obj is not None and obj.isket and obj.shape[0] == self.dimension:
            # a ket: rho_keep = V V^dag with V the amplitudes reshaped to (kept, traced), never the D x D outer product
            arr = np.asarray(obj.full()).reshape(dims)
            traced = [f for f in range(len(dims)) if f not in keep_sorted]
            v = np.transpose(arr, list(keep_sorted) + traced)
            kept_dims_k = [dims[f] for f in keep_sorted]
            size_k = prod(kept_dims_k)
            mat = v.reshape(size_k, -1)
            return qt.Qobj(mat @ mat.conj().T, dims=[kept_dims_k, kept_dims_k])
        rho = self._dense_dm(state).reshape(dims + dims)
        traced = [f for f in range(len(dims)) if f not in keep_sorted]
        for f in reversed(traced):
            n_axes = rho.ndim // 2
            rho = np.trace(rho, axis1=f, axis2=f + n_axes)
        kept_dims = [dims[f] for f in keep_sorted]
        size = prod(kept_dims)
        return qt.Qobj(rho.reshape(size, size), dims=[kept_dims, kept_dims])

    def internal_marginal(self, state: State | qt.Qobj) -> qt.Qobj:
        return self.marginal(state, tuple(range(self.n_ions)))

    def mode_marginal(self, state: State | qt.Qobj, mode: int) -> qt.Qobj:
        """The reduced density matrix of one mode; for an ENR member the sum over the group's other modes."""
        cls = self.mode_class(mode)
        if cls == "resolved":
            return self.marginal(state, (self.mode_factor(mode),))
        if cls == "enr":
            assert self.enr_factor is not None
            sigma = np.asarray(self.marginal(state, (self.enr_factor,)).full())
            dims, n_exc = self._enr_dims()
            _nstates, _state2idx, idx2state = qt.enr_state_dictionaries(dims, n_exc)
            p = self.enr_position(mode)
            d = n_exc + 1
            out = np.zeros((d, d), dtype=complex)
            tuples = [tuple(idx2state[i]) for i in range(len(idx2state))]
            for i, ti in enumerate(tuples):
                rest_i = ti[:p] + ti[p + 1 :]
                for j, tj in enumerate(tuples):
                    if tj[:p] + tj[p + 1 :] == rest_i:
                        out[ti[p], tj[p]] += sigma[i, j]
            return qt.Qobj(out, dims=[[d], [d]])
        raise KeyError(f"mode {mode} is frozen and has no reduced state")

    def fock_populations(self, state: State | qt.Qobj, mode: int) -> np.ndarray:
        obj = state if isinstance(state, qt.Qobj) else state.joint
        if (
            obj is not None
            and obj.isket
            and self.mode_class(mode) == "resolved"
            and obj.shape[0] == self.dimension
        ):
            # a ket's Fock populations are |psi|^2 summed over the other factors, O(D)
            arr = np.abs(np.asarray(obj.full()).reshape(self.dims)) ** 2
            f = self.mode_factor(mode)
            return np.asarray(arr.sum(axis=tuple(i for i in range(arr.ndim) if i != f)), dtype=float)
        return np.real(np.diag(np.asarray(self.mode_marginal(state, mode).full())))

    # ---- growth ------------------------------------------------------------------------------------------------

    def grown(self, mode: int, add: int) -> HilbertSpace:
        """A copy with ``add`` more Fock levels on ``mode`` (Section 5.5); for an ENR member the group's excitation cap grows
        by ``add``."""
        if self.enr_group is not None and mode in self.enr_group[0]:
            return self.grown_enr(add)
        res = tuple(m.grown(add) if m.mode == mode else m for m in self.resolved)
        return HilbertSpace(self.ion_dims, res, self.enr_group, self.frozen, self.ions, self.dropped)

    def grown_enr(self, add: int) -> HilbertSpace:
        """A copy with the ENR group's excitation cap N_exc raised by ``add``."""
        if self.enr_group is None:
            raise KeyError("this space has no ENR group")
        if add <= 0:
            raise ValueError("grow by a positive number of excitations")
        modes, n_exc = self.enr_group
        return HilbertSpace(
            self.ion_dims, self.resolved, (modes, n_exc + add), self.frozen, self.ions, self.dropped
        )


# ---- module-level cache ----------------------------------------------------------------------------------------------------

_CACHE: dict[tuple[HilbertSpace, tuple[object, ...]], qt.Qobj] = {}
_CACHE_MAX = 4096


def _eta_key(eta: float) -> float:
    return float(np.round(eta, 15))


def _cached(space: HilbertSpace, key: tuple[object, ...], build: Callable[[], qt.Qobj]) -> qt.Qobj:
    k = (space, key)
    op = _CACHE.get(k)
    if op is None:
        op = build()
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[k] = op
    return op
