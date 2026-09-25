"""Hyperfine plus Zeeman structure of one fine-structure level at any field (PLAN.md Section 4.5.1).

On the uncoupled basis |m_I, m_J>, in Hz (Steck QAO Eq. 7.134):

    H/h = A I.J + B [3(I.J)^2 + (3/2) I.J - I(I+1)J(J+1)] / [2I(2I-1)J(2J-1)] + (mu_B/h) (g_J J_z + g_I I_z) B,

with g_I = -(mu_I/(I mu_N))(m_e/m_p) in Steck's sign convention (both g factors enter with one sign); the quadrupole
term needs I >= 1 and J >= 1. H commutes with F_z and is diagonalized exactly per m_F block.

Labels follow adiabatic continuation from B = 0: F is read off <F^2> at zero field, and inside one m_F block levels never
cross as B grows (no-crossing rule), so the energy order at B is the order at B = 0. A level with no hyperfine structure
is labelled by (m_J, m_I). The sign of A is a signed input; the ordering of the F manifolds is derived from it.

Sensitivities: dE/dB by Hellmann-Feynman, d^2E/dB^2 by second-order perturbation inside the block. Two curvature names
are kept: ``d2nu_dB2`` and ``taylor_c2`` = (1/2) d^2 nu/dB^2 (the 171Yb+ 310.87 Hz/G^2).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq

from qutip_trap.species.model import Level, level_j
from qutip_trap.species.wigner import angular_momentum_matrices, as_half_integer, m_values
from qutip_trap.units import M_E_OVER_M_P, MU_B_OVER_H_HZ_PER_T

MU_B_OVER_H_HZ_PER_G: float = MU_B_OVER_H_HZ_PER_T * 1e-4
"""mu_B/h in Hz per gauss (1.399624e6)."""

_QN = re.compile(r"(?P<key>F|mF|mJ|mI)=(?P<val>[+-]?\d+(?:/\d+)?(?:\.\d+)?)")


def g_I_steck(mu_I_nuclear_magnetons: float, nuclear_spin: float) -> float:
    """g_I = -(mu_I/(I mu_N)) (m_e/m_p): Steck's convention, electron-like sign (Section 13). Zero for I = 0."""
    if nuclear_spin == 0.0:
        if mu_I_nuclear_magnetons != 0.0:
            raise ValueError("a spin-zero nucleus has no magnetic moment")
        return 0.0
    return -(mu_I_nuclear_magnetons / nuclear_spin) * M_E_OVER_M_P


def format_half(x: Fraction) -> str:
    """'0', '1', '-1', '1/2', '-3/2' (no leading '+')."""
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def parse_quantum_numbers(text: str) -> dict[str, Fraction]:
    """Parse 'F=1 mF=0', 'mJ=-1/2' or 'mJ=+1/2 mI=-3/2' into exact quantum numbers."""
    found = {m.group("key"): as_half_integer(Fraction(m.group("val"))) for m in _QN.finditer(text)}
    if not found:
        raise ValueError(f"no quantum numbers found in {text!r}")
    return found


@dataclass(frozen=True)
class ZeemanSpectrum:
    """The hyperfine-Zeeman spectrum of one level at one field.

    ``energies_hz`` are ABSOLUTE (the level's cited energy plus the hyperfine-Zeeman eigenvalue), so that a
    transition frequency between states of different levels is a plain difference. Columns of ``eigenvectors``
    are the states in the uncoupled basis ``(basis_mI[i], basis_mJ[i])``. Derivatives are per gauss.
    """

    level: str
    B_gauss: float
    energies_hz: np.ndarray
    labels: tuple[str, ...]
    F: tuple[Fraction | None, ...]
    mF: tuple[Fraction, ...]
    eigenvectors: np.ndarray
    basis_mI: tuple[Fraction, ...]
    basis_mJ: tuple[Fraction, ...]
    dE_dB_hz_per_g: np.ndarray
    d2E_dB2_hz_per_g2: np.ndarray

    def index(self, label: str) -> int:
        """Index of a state by its quantum-number label ('F=1 mF=0' or 'mJ=-1/2 [mI=...]')."""
        want = parse_quantum_numbers(label)
        matches = [k for k, lab in enumerate(self.labels) if parse_quantum_numbers(lab) == want]
        if len(matches) != 1:
            raise KeyError(
                f"{self.level}: label {label!r} matches {len(matches)} states; known: {self.labels}"
            )
        return matches[0]

    def energy_hz(self, label: str) -> float:
        return float(self.energies_hz[self.index(label)])

    def state(self, label: str) -> np.ndarray:
        return np.asarray(self.eigenvectors[:, self.index(label)])


class HyperfineZeeman:
    """The hyperfine-Zeeman problem of one level for one isotope (nuclear spin I, moment mu_I)."""

    def __init__(self, level: Level, nuclear_spin: float, mu_I_nuclear_magnetons: float) -> None:
        self.level = level
        self.I = as_half_integer(nuclear_spin)
        self.J = level_j(level.name)
        self.g_I = g_I_steck(mu_I_nuclear_magnetons, float(self.I))
        self.mI = m_values(self.I)
        self.mJ = m_values(self.J)
        self.basis: tuple[tuple[Fraction, Fraction], ...] = tuple(
            (mi, mj) for mi in self.mI for mj in self.mJ
        )
        ix, iy, iz = angular_momentum_matrices(self.I)
        jx, jy, jz = angular_momentum_matrices(self.J)
        eye_i = np.eye(len(self.mI))
        eye_j = np.eye(len(self.mJ))
        i_dot_j = np.kron(ix, jx) + np.kron(iy, jy) + np.kron(iz, jz)
        h0 = level.A_hfs_hz * i_dot_j
        if level.B_hfs_hz != 0.0:
            if self.I < 1 or self.J < 1:
                raise ValueError(f"{level.name}: the quadrupole term needs I >= 1 and J >= 1")
            ii = float(self.I * (self.I + 1))
            jj = float(self.J * (self.J + 1))
            denom = float(2 * self.I * (2 * self.I - 1) * self.J * (2 * self.J - 1))
            h0 = (
                h0
                + level.B_hfs_hz
                * (3.0 * i_dot_j @ i_dot_j + 1.5 * i_dot_j - ii * jj * np.eye(len(self.basis)))
                / denom
            )
        self._h0_hz = np.real_if_close(h0).astype(float)
        self._dh_dB = MU_B_OVER_H_HZ_PER_G * np.real(
            level.g_J * np.kron(eye_i, jz) + self.g_I * np.kron(iz, eye_j)
        )
        self._f2 = np.real(
            np.kron(ix @ ix + iy @ iy + iz @ iz, eye_j)
            + np.kron(eye_i, jx @ jx + jy @ jy + jz @ jz)
            + 2.0 * i_dot_j
        )
        self._M = np.array([float(mi + mj) for mi, mj in self.basis])
        self.hyperfine_free = level.A_hfs_hz == 0.0 and level.B_hfs_hz == 0.0
        self._zero_field_order = self._label_zero_field()

    def hamiltonian_hz(self, B_gauss: float) -> np.ndarray:
        """H/h in Hz on the uncoupled basis, relative to the level's hyperfine centroid."""
        return self._h0_hz + B_gauss * self._dh_dB

    def _blocks(self) -> dict[float, np.ndarray]:
        return {m: np.flatnonzero(self._M == m) for m in sorted(set(self._M.tolist()))}

    def _label_zero_field(self) -> dict[float, list[Fraction | None]]:
        """Per m_F block: the F labels in ascending energy order at B = 0 (None when hyperfine-free)."""
        order: dict[float, list[Fraction | None]] = {}
        for m, idx in self._blocks().items():
            if self.hyperfine_free:
                order[m] = [None] * len(idx)
                continue
            w, v = np.linalg.eigh(self._h0_hz[np.ix_(idx, idx)])
            labels: list[Fraction | None] = []
            for k in range(len(idx)):
                vec = v[:, k]
                f2 = float(vec @ self._f2[np.ix_(idx, idx)] @ vec)
                f = Fraction(round(2.0 * (-0.5 + math.sqrt(0.25 + f2)))) / 2
                labels.append(f)
            if len(set(labels)) != len(labels):
                raise ValueError(f"{self.level.name}: degenerate F manifolds at zero field; labels ambiguous")
            order[m] = labels
        return order

    def spectrum(self, B_gauss: float) -> ZeemanSpectrum:
        dim = len(self.basis)
        energies = np.empty(dim)
        vectors = np.zeros((dim, dim))
        d1 = np.empty(dim)
        d2 = np.empty(dim)
        f_labels: list[Fraction | None] = [None] * dim
        m_labels: list[Fraction] = [Fraction(0)] * dim
        text: list[str] = [""] * dim
        h = self.hamiltonian_hz(B_gauss)
        col = 0
        for m, idx in self._blocks().items():
            sub = h[np.ix_(idx, idx)]
            dh = self._dh_dB[np.ix_(idx, idx)]
            if self.hyperfine_free:
                # exact: the uncoupled states are eigenstates, energies linear in B
                w = np.diag(sub)
                order = np.argsort(w, kind="stable")
                for k in order:
                    energies[col] = w[k]
                    vectors[idx[k], col] = 1.0
                    d1[col] = dh[k, k]
                    d2[col] = 0.0
                    mi, mj = self.basis[idx[k]]
                    m_labels[col] = mi + mj
                    text[col] = f"mJ={format_half(mj)}" + (f" mI={format_half(mi)}" if self.I != 0 else "")
                    col += 1
                continue
            w, v = np.linalg.eigh(sub)
            # deterministic sign gauge: largest component positive
            for k in range(v.shape[1]):
                j = int(np.argmax(np.abs(v[:, k])))
                if v[j, k] < 0:
                    v[:, k] *= -1.0
            hp = v.T @ dh @ v
            for k in range(len(idx)):
                energies[col] = w[k]
                vectors[idx, col] = v[:, k]
                d1[col] = hp[k, k]
                acc = 0.0
                for ell in range(len(idx)):
                    if ell != k:
                        acc += hp[ell, k] ** 2 / (w[k] - w[ell])
                d2[col] = 2.0 * acc
                f = self._zero_field_order[m][k]
                f_labels[col] = f
                m_labels[col] = Fraction(m).limit_denominator(2)
                text[col] = f"F={format_half(f)} mF={format_half(m_labels[col])}" if f is not None else ""
                col += 1
        return ZeemanSpectrum(
            level=self.level.name,
            B_gauss=B_gauss,
            energies_hz=energies + self.level.energy_hz,
            labels=tuple(text),
            F=tuple(f_labels),
            mF=tuple(m_labels),
            eigenvectors=vectors,
            basis_mI=tuple(mi for mi, _ in self.basis),
            basis_mJ=tuple(mj for _, mj in self.basis),
            dE_dB_hz_per_g=d1,
            d2E_dB2_hz_per_g2=d2,
        )


@lru_cache(maxsize=256)
def hyperfine_zeeman(level: Level, nuclear_spin: float, mu_I_nuclear_magnetons: float) -> HyperfineZeeman:
    """Cached constructor (Level is a frozen, hashable dataclass)."""
    return HyperfineZeeman(level, nuclear_spin, mu_I_nuclear_magnetons)


class TransitionSensitivity(NamedTuple):
    """nu = E_b - E_a (Hz) and its first and second field derivatives (Hz/G, Hz/G^2) at one field."""

    frequency_hz: float
    dnu_dB_hz_per_g: float
    d2nu_dB2_hz_per_g2: float


def transition_sensitivity(
    hz_a: HyperfineZeeman, label_a: str, hz_b: HyperfineZeeman, label_b: str, B_gauss: float
) -> TransitionSensitivity:
    sa = hz_a.spectrum(B_gauss)
    sb = hz_b.spectrum(B_gauss)
    ia, ib = sa.index(label_a), sb.index(label_b)
    return TransitionSensitivity(
        frequency_hz=float(sb.energies_hz[ib] - sa.energies_hz[ia]),
        dnu_dB_hz_per_g=float(sb.dE_dB_hz_per_g[ib] - sa.dE_dB_hz_per_g[ia]),
        d2nu_dB2_hz_per_g2=float(sb.d2E_dB2_hz_per_g2[ib] - sa.d2E_dB2_hz_per_g2[ia]),
    )


@dataclass(frozen=True)
class ClockPoint:
    """A field-insensitive point of a transition: the field B0 (G) where d nu/dB = 0, |nu| there (Hz) and the curvature
    d^2 nu/dB^2 (Hz/G^2) of |nu|, with ``taylor_c2_hz_per_g2`` half of it (Section 13 names both)."""

    B0_gauss: float
    frequency_hz: float
    d2nu_dB2_hz_per_g2: float

    @property
    def taylor_c2_hz_per_g2(self) -> float:
        return 0.5 * self.d2nu_dB2_hz_per_g2


def clock_points(
    hz_a: HyperfineZeeman,
    label_a: str,
    hz_b: HyperfineZeeman,
    label_b: str,
    B_lo_gauss: float,
    B_hi_gauss: float,
    *,
    n_grid: int = 400,
) -> tuple[ClockPoint, ...]:
    """Every field in [B_lo, B_hi] where d nu/dB = 0 for the pair, found on a grid and refined by brentq.

    The derivative is the analytic Hellmann-Feynman difference, so a root is located to the precision of the
    eigenvalues (the 9Be+ roots at 119.446 and 119.643 G resolve).
    """
    if B_lo_gauss <= 0.0 or B_hi_gauss <= B_lo_gauss:
        raise ValueError("the search interval must satisfy 0 < B_lo < B_hi")

    def slope(b: float) -> float:
        return transition_sensitivity(hz_a, label_a, hz_b, label_b, b).dnu_dB_hz_per_g

    grid = np.linspace(B_lo_gauss, B_hi_gauss, n_grid)
    values = np.array([slope(b) for b in grid])
    points: list[ClockPoint] = []
    for k in range(len(grid) - 1):
        if values[k] == 0.0:
            b0 = float(grid[k])
        elif values[k] * values[k + 1] < 0.0:
            b0 = float(brentq(slope, grid[k], grid[k + 1], xtol=1e-12, rtol=1e-14, maxiter=200))
        else:
            continue
        s = transition_sensitivity(hz_a, label_a, hz_b, label_b, b0)
        # report the curvature of |nu|: if the labelled pair is inverted (nu < 0) the sign flips
        curvature = s.d2nu_dB2_hz_per_g2 if s.frequency_hz >= 0.0 else -s.d2nu_dB2_hz_per_g2
        points.append(ClockPoint(B0_gauss=b0, frequency_hz=abs(s.frequency_hz), d2nu_dB2_hz_per_g2=curvature))
    return tuple(points)
