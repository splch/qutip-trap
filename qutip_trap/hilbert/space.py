"""The composite space: per-ion qudit x modes, with the ENR option (PLAN.md Sections 5.1, 5.4; Appendix E; M2).

Each ion is a qudit of dimension d (2 by default, 2 + leakage levels when scattering is modelled); each resolved
mode a truncated oscillator of d_m = n_max + 1 Fock levels; an optional ENR group of cold undriven modes shares one
excitation cap N_exc; frozen spectators contribute their Debye-Waller factors analytically (Section 5.2). The tensor
order is ions, then resolved modes in the order of ``resolved``, then the ENR group as ONE factor; every ``factor``
index below is a position in that order, every ``mode`` index a position in ``Crystal.modes`` (Section 13, row
"Mode index"). Operators are built once per space and cached (module-level cache keyed by the frozen space).

Rules from Section 5.1.1 and the 2026-09-04 numerics critique that this module encodes: the drive's displacement is
the matrix exponential (per mode in a product space, of the SUM generator in an ENR space, never a product of
per-mode exponentials there); the interior elements are asserted against the analytic Laguerre oracle over the
declared ``expected_n_range``; every dimension computation reads ``shape`` behind an assertion and never ``dims``,
because ``tensor(sigmap(), D_enr)`` reports dims whose product disagrees with its shape; ``ptrace`` raises on any
space with an ENR factor, so marginals are built here by explicit index sums over ``enr_state_dictionaries``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import comb, prod
from typing import TYPE_CHECKING, Literal

import numpy as np
import qutip as qt

from qutip_trap.hilbert.operators import (
    displacement_operator,
    interior_tolerance,
    oracle_check,
    qudit_projector,
    qudit_sigma_minus,
    qudit_sigma_plus,
    qudit_sigma_z,
    required_margin,
    thermal_populations,
)

if TYPE_CHECKING:
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions, State

M2 = "milestone M2 (hilbert/, PLAN.md Section 5.1)"
ORACLE_MIN_MARGIN = 4
"""The smallest margin the Section 5.1.1 table covers (its d = 8 row); below it the oracle is reported, not asserted."""
M9A = "milestone M9a (PLAN.md Sections 5.4, 11.3)"

ModeClass = Literal["resolved", "enr", "frozen"]


def enr_dimension(n_modes: int, n_exc: int) -> int:
    """Number of Fock tuples of ``n_modes`` modes with total excitation <= n_exc: C(n_modes + n_exc, n_exc)."""
    if n_modes <= 0 or n_exc < 0:
        raise ValueError("an ENR group has at least one mode and a non-negative cap")
    return comb(n_modes + n_exc, n_exc)


@dataclass(frozen=True)
class ModeTruncation:
    mode: int
    d: int
    """d = n_max + 1 Fock levels."""
    expected_n_range: tuple[int, int]
    eta_max: float

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

    @property
    def n_max(self) -> int:
        return self.d - 1

    @property
    def margin_levels(self) -> int:
        """Levels between the top of the expected range and the cap."""
        return self.d - 1 - self.expected_n_range[1]

    @property
    def required_margin(self) -> int:
        """The Section 5.1.1 margin for ``eta_max``."""
        return required_margin(self.eta_max)

    def grown(self, add: int) -> ModeTruncation:
        """The same declaration with ``add`` more Fock levels (Section 5.5 adaptive growth)."""
        if add <= 0:
            raise ValueError("grow by a positive number of levels")
        return ModeTruncation(self.mode, self.d + add, self.expected_n_range, self.eta_max)


@dataclass(frozen=True)
class CachedOperators:
    """sigma operators per ion, ladder operators per mode and displacement operators by (ion, mode) (M2)."""

    sigma_plus: tuple[qt.Qobj, ...]
    sigma_minus: tuple[qt.Qobj, ...]
    sigma_z: tuple[qt.Qobj, ...]
    a: tuple[qt.Qobj, ...]
    displacement: dict[tuple[int, int], qt.Qobj]


@dataclass(frozen=True)
class HilbertSpace:
    ion_dims: tuple[int, ...]
    """2, or 2 + leakage levels, per ion."""
    resolved: tuple[ModeTruncation, ...]
    """Product-space modes."""
    enr_group: tuple[tuple[int, ...], int] | None
    """(modes, N_exc), optional."""
    frozen: tuple[int, ...]
    """Frozen spectators."""

    def __post_init__(self) -> None:
        self.check()

    # ---- bookkeeping ---------------------------------------------------------------------------------------------

    def check(self) -> None:
        """The Section 5.1 invariants, including shape == prod(dims) on the product factors and the ENR shape rule."""
        if not self.ion_dims or any(d < 2 for d in self.ion_dims):
            raise ValueError("every ion is a qudit of dimension >= 2")
        resolved = [m.mode for m in self.resolved]
        enr_modes = list(self.enr_group[0]) if self.enr_group is not None else []
        frozen = list(self.frozen)
        all_modes = resolved + enr_modes + frozen
        if len(set(all_modes)) != len(all_modes):
            raise ValueError("a mode is resolved, in the ENR group or frozen, never in two classes")
        if any(m < 0 for m in all_modes):
            raise ValueError("mode indices are positions in Crystal.modes and non-negative")
        if self.enr_group is not None and (not enr_modes or self.enr_group[1] < 0):
            raise ValueError("an ENR group names at least one mode and a non-negative excitation cap")
        # the Qobj shape assertions (Section 5.1): product factors multiply, the ENR factor is one C(M + N_exc, N_exc) block
        ident = self.identity()
        if ident.shape[0] != self.dimension or ident.shape[1] != self.dimension:
            raise ValueError(f"joint identity has shape {ident.shape}, expected {self.dimension}")
        if self.enr_group is not None:
            n_modes, n_exc = len(self.enr_group[0]), self.enr_group[1]
            if self.enr_identity().shape[0] != enr_dimension(n_modes, n_exc):
                raise ValueError("ENR factor shape disagrees with C(M + N_exc, N_exc)")

    @property
    def n_ions(self) -> int:
        return len(self.ion_dims)

    @property
    def dims(self) -> list[int]:
        """Tensor factors in order: ions, resolved modes, then the ENR group as one factor."""
        d = list(self.ion_dims) + [m.d for m in self.resolved]
        if self.enr_group is not None:
            d.append(enr_dimension(len(self.enr_group[0]), self.enr_group[1]))
        return d

    @property
    def dimension(self) -> int:
        """The joint dimension that Section 11's cost model and the Section 5.4 budget guard."""
        return prod(self.dims)

    @property
    def n_factors(self) -> int:
        return len(self.dims)

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

    def identity(self) -> qt.Qobj:
        return qt.tensor(*self.factor_identities())

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
        return qt.tensor(*ops).to("CSR")

    def embed_many(self, ops_by_factor: Mapping[int, qt.Qobj]) -> qt.Qobj:
        ops = self.factor_identities()
        for f, op in ops_by_factor.items():
            if op.shape != ops[f].shape:
                raise ValueError(f"operator shape {op.shape} does not fit factor {f}")
            ops[f] = op
        return qt.tensor(*ops).to("CSR")

    def sigma_plus(self, ion: int) -> qt.Qobj:
        """|1><0| on ``ion`` (raising the lower qubit level to the upper), identities elsewhere."""
        return _cached(
            self, ("sigma_plus", ion), lambda: self.embed(qudit_sigma_plus(self.ion_dims[ion]), ion)
        )

    def sigma_minus(self, ion: int) -> qt.Qobj:
        return _cached(
            self, ("sigma_minus", ion), lambda: self.embed(qudit_sigma_minus(self.ion_dims[ion]), ion)
        )

    def sigma_z(self, ion: int) -> qt.Qobj:
        """The ENERGY sigma_z = |1><1| - |0><0| on ``ion`` (upper level positive; the negative of the computational Z)."""
        return _cached(self, ("sigma_z", ion), lambda: self.embed(qudit_sigma_z(self.ion_dims[ion]), ion))

    def projector(self, ion: int, level: int) -> qt.Qobj:
        return _cached(
            self,
            ("projector", ion, level),
            lambda: self.embed(qudit_projector(self.ion_dims[ion], level), ion),
        )

    def transition(self, ion: int, upper: int, lower: int) -> qt.Qobj:
        """|upper><lower| on ``ion`` for d > 2 (the transition operators of Section 5.7)."""
        d = self.ion_dims[ion]
        return self.embed(qt.basis(d, upper) * qt.basis(d, lower).dag(), ion)

    def annihilation(self, mode: int) -> qt.Qobj:
        """a_m on the joint space (resolved: truncated ladder operator; ENR member: enr_destroy of the group)."""

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
        """(a + a^dagger) on the joint space; x = x0 (a + a^dagger) (Section 13)."""
        a = self.annihilation(mode)
        return _cached(self, ("x", mode), lambda: (a + a.dag()).to("CSR"))

    def displacement_factor(self, mode: int, eta: float) -> qt.Qobj:
        """D(i eta) of ONE resolved mode in its own factor space (expm), oracle-checked over the declared range."""
        tr = self.truncation(mode)
        if abs(eta) > tr.eta_max * (1.0 + 1e-12):
            raise ValueError(
                f"mode {mode}: |eta| = {abs(eta):.4g} exceeds the declared eta_max = {tr.eta_max:.4g} of this space"
            )

        def build() -> qt.Qobj:
            # rule (ii) of Section 5.1.1: assert the interior elements against the analytic oracle to the table's tolerance;
            # a margin below the table's smallest row (4 levels) is not tabulated and is left to the boundary monitor
            if tr.margin_levels >= ORACLE_MIN_MARGIN:
                oracle_check(
                    tr.d, 1j * eta, tr.expected_n_range[1], interior_tolerance(tr.margin_levels, eta)
                )
            return displacement_operator(tr.d, 1j * eta)

        return _cached(self, ("D", mode, _eta_key(eta)), build)

    def oracle_status(self, mode: int, eta: float) -> tuple[bool, float, float]:
        """(asserted, max interior element difference, tolerance) of the mode's displacement at |eta| (Section 5.1.1 rule ii)."""
        tr = self.truncation(mode)
        exp_ = displacement_operator(tr.d, 1j * eta).full()
        from qutip_trap.hilbert.operators import displacement_matrix_analytic

        ana = displacement_matrix_analytic(tr.d, 1j * eta)
        hi = tr.expected_n_range[1] + 1
        diff = float(np.max(np.abs(exp_[:hi, :hi] - ana[:hi, :hi])))
        asserted = tr.margin_levels >= ORACLE_MIN_MARGIN
        return asserted, diff, interior_tolerance(max(tr.margin_levels, 1), eta)

    def displacement(self, ion: int, mode: int, eta: float) -> qt.Qobj:
        """D_m(i eta_{ion, mode}) embedded in the joint space (identities elsewhere); resolved modes only."""
        return _cached(
            self,
            ("D_full", ion, mode, _eta_key(eta)),
            lambda: self.embed(self.displacement_factor(mode, eta), self.mode_factor(mode)),
        )

    def enr_displacement(self, etas: Mapping[int, float]) -> qt.Qobj:
        """expm of the ENR SUM generator sum_m i eta_m (a_m + a_m^dagger) in the group's own space (Section 5.1.1)."""
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

    def drive_operator(self, ion: int, etas: Mapping[int, float]) -> qt.Qobj:
        """sigma_+^ion (x) prod_m D_m(i eta_{ion,m}) over the resolved modes, times the ENR group's sum-generator exponential.

        Frozen modes are absent here: their Debye-Waller factors multiply the coefficient (Section 5.2). Modes that
        the drive does not couple to (eta = 0) contribute the identity. CSR, with 2^{N-1} prod d_m^2 non-zeros per
        ion for two-level ions (Section 5.1.1).
        """
        ops: dict[int, qt.Qobj] = {self.ion_factor(ion): qudit_sigma_plus(self.ion_dims[ion])}
        enr_etas: dict[int, float] = {}
        for mode, eta in etas.items():
            if eta == 0.0:
                continue
            cls = self.mode_class(mode)
            if cls == "resolved":
                ops[self.mode_factor(mode)] = self.displacement_factor(mode, eta)
            elif cls == "enr":
                enr_etas[mode] = eta
            # frozen: handled by the caller through the Debye-Waller factor
        if enr_etas:
            assert self.enr_factor is not None
            ops[self.enr_factor] = self.enr_displacement(enr_etas)
        return self.embed_many(ops)

    def operators(self, etas: Mapping[tuple[int, int], float] | None = None) -> CachedOperators:
        """sigma_i, a_m and, for the given {(ion, mode): eta}, the embedded D_m(i eta) by expm with the oracle checks."""
        modes = [m.mode for m in self.resolved]
        if self.enr_group is not None:
            modes += list(self.enr_group[0])
        disp: dict[tuple[int, int], qt.Qobj] = {}
        if etas is not None:
            for (ion, mode), eta in etas.items():
                if self.mode_class(mode) == "enr":
                    disp[(ion, mode)] = self.embed(self.enr_displacement({mode: eta}), self.mode_factor(mode))
                else:
                    disp[(ion, mode)] = self.displacement(ion, mode, eta)
        return CachedOperators(
            sigma_plus=tuple(self.sigma_plus(i) for i in range(self.n_ions)),
            sigma_minus=tuple(self.sigma_minus(i) for i in range(self.n_ions)),
            sigma_z=tuple(self.sigma_z(i) for i in range(self.n_ions)),
            a=tuple(self.annihilation(m) for m in modes),
            displacement=disp,
        )

    # ---- states ------------------------------------------------------------------------------------------------

    def internal_ket(self, levels: Sequence[int]) -> qt.Qobj:
        """|l_0 l_1 ...> on the ion factors (computational ordering: 0 = lower qubit level)."""
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
        """The thermal density matrix of a resolved mode at nbar (truncated; the tail is the caller's boundary population)."""
        d = self.truncation(mode).d
        p = thermal_populations(nbar, d)
        return qt.Qobj(np.diag(p / p.sum()), dims=[[d], [d]])

    def coherent(self, mode: int, alpha: complex) -> qt.Qobj:
        return qt.coherent(self.truncation(mode).d, alpha)

    def enr_state(
        self, fock: Mapping[int, int] | None = None, thermal: Mapping[int, float] | None = None
    ) -> qt.Qobj:
        """The ENR factor's state: a Fock tuple (ket) or a product thermal state (dm); never both."""
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

        ``motional`` maps a resolved mode to its factor ket or dm and, for an ENR group, the key -1 to the group's
        factor state; every resolved mode and the group (if present) must be given. A ket results only if every part
        is a ket, otherwise the density matrix.
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
            return qt.tensor(*parts)
        return qt.tensor(*[p if p.isoper else qt.ket2dm(p) for p in parts])

    def initial_state(
        self,
        internal: qt.Qobj | Sequence[int],
        *,
        fock: Mapping[int, int] | None = None,
        thermal: Mapping[int, float] | None = None,
        states: Mapping[int, qt.Qobj] | None = None,
        provenance: tuple[str, ...] = ("m2.initial_state",),
    ) -> State:
        """A ``State`` (Appendix E) from per-mode Fock levels, thermal occupations or explicit factor states.

        Resolved modes take ``states`` first, then ``fock``, then ``thermal`` (default |0>); ENR members take
        ``fock`` or ``thermal`` (a Fock tuple or a product thermal state for the whole group); frozen modes carry
        only their nbar (``thermal``, default 0).
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
        """The reduced density matrix over the tensor factors ``keep`` (in factor order), by explicit index sums.

        Works with an ENR factor present, where ``ptrace`` raises (Section 5.1); the ENR group is kept or traced as
        one factor here, and ``mode_marginal`` reduces to one of its modes.
        """
        dims = self.dims
        keep_sorted = tuple(sorted(set(keep)))
        if any(f < 0 or f >= len(dims) for f in keep_sorted):
            raise IndexError("keep lists factor indices")
        rho = self._dense_dm(state).reshape(dims + dims)
        traced = [f for f in range(len(dims)) if f not in keep_sorted]
        for f in reversed(traced):
            # trace out factor f: pair axis f (row) with axis f + n_remaining (column)
            n_axes = rho.ndim // 2
            rho = np.trace(rho, axis1=f, axis2=f + n_axes)
        kept_dims = [dims[f] for f in keep_sorted]
        size = prod(kept_dims)
        return qt.Qobj(rho.reshape(size, size), dims=[kept_dims, kept_dims])

    def internal_marginal(self, state: State | qt.Qobj) -> qt.Qobj:
        return self.marginal(state, tuple(range(self.n_ions)))

    def mode_marginal(self, state: State | qt.Qobj, mode: int) -> qt.Qobj:
        """The (n_max + 1)^2 reduced density matrix of one mode; for an ENR member the sum over the group's other modes."""
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
        return np.real(np.diag(np.asarray(self.mode_marginal(state, mode).full())))

    # ---- margins -----------------------------------------------------------------------------------------------

    def margin_deficits(self, etas: Mapping[int, float] | None = None) -> dict[int, int]:
        """Per resolved mode, required margin minus available margin (positive = the cap is too low; Section 5.1.1)."""
        out: dict[int, int] = {}
        for m in self.resolved:
            eta = m.eta_max if etas is None else max(abs(etas.get(m.mode, 0.0)), 0.0)
            out[m.mode] = required_margin(eta) - m.margin_levels
        return out

    def grown(self, mode: int, add: int) -> HilbertSpace:
        """A copy with ``add`` more Fock levels on ``mode`` (Section 5.5 adaptive growth)."""
        res = tuple(m.grown(add) if m.mode == mode else m for m in self.resolved)
        return HilbertSpace(self.ion_dims, res, self.enr_group, self.frozen)

    @classmethod
    def for_(cls, device: Device, schedule: Schedule, options: SolverOptions) -> HilbertSpace:
        """The resolved-mode selection and truncation policy of Sections 5.5 and 11.3."""
        raise NotImplementedError(f"HilbertSpace.for_ is {M9A}")


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


def clear_operator_cache() -> None:
    _CACHE.clear()


__all__ = [
    "CachedOperators",
    "HilbertSpace",
    "ModeClass",
    "ModeTruncation",
    "clear_operator_cache",
    "enr_dimension",
]
