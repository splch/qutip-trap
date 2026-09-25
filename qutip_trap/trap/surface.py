"""Electrode geometry: gapless-plane surface-electrode traps and the rod/blade record (PLAN.md Section 4.1.6).

The substrate is the plane y = 0, y the height above it, z the trap axis along the rf rails. Every electrode is a set of
infinite STRIPS {x1 < x < x2} or finite RECTANGLES {x1 < x < x2, z1 < z < z2}; gaps have zero width, widths run from gap
centre to gap centre and the plane outside the electrodes is grounded (House 2008, Wesenberg 2008), so an electrode's
unit-voltage basis function Theta is dimensionless and a complete cover sums to 1 on the plane. Strip potentials come
from the complex potential f(w) = (i/pi) sum_k sigma_k ln(w - e_k), w = x + iy, sigma = +1 at a left edge and -1 at a
right edge, with phi = Re f, E_x - i E_y = -f'(w); rectangles from House's four-arctan solid angle (principal branch,
corners ordered, odd in y) with a complex-step Hessian of its analytic gradient. The rf null of a strip rf electrode is
the upper-half-plane zero of f'(w) and its escape point the saddle of |E|^2 at a zero of f''(w).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from qutip_trap.units import E_C

INF = math.inf
Kind = Literal["rod_quadrupole", "blade", "surface_five_wire", "surface_four_wire", "surface_general"]
_SURFACE_KINDS = ("surface_five_wire", "surface_four_wire", "surface_general")
_CURVATURE_KEYS = ("dc_curvature_xx_v_per_m2", "dc_curvature_yy_v_per_m2", "dc_curvature_zz_v_per_m2")


@dataclass(frozen=True)
class Electrodes:
    """Electrode geometry; ``parameters`` are the named lengths of the layout in metres.

    - ``rod_quadrupole`` and ``blade``: ``R_m`` (radial scale R' ~ R), ``Z0_m`` (endcap scale), ``kappa`` (endcap
      efficiency) and optionally ``alpha`` (the rf factor of Berkeland's phi_ac term); the dc record carries the endcap U0.
    - ``surface_five_wire``: ``a_m`` (centre electrode full width), ``b_m`` (right rail), optional ``c_m`` (left rail,
      default b): rf rails at -c < x < 0 and a < x < a + b, dc strips "centre", "left" and "right".
    - ``surface_four_wire``: ``a_m`` and ``c_m``: rails at -c < x < 0 and x > a, dc "centre" and "left".
    - ``surface_general``: the layout given by ``strips`` (x-intervals, ``math.inf`` allowed) and ``rectangles``
      ((x1, x2, z1, z2) patches), which the parametric kinds may add to.

    The rf electrode is ``rf_name``. Externally solved dc curvatures ``dc_curvature_xx/yy/zz_v_per_m2`` (the full
    traceless triple, e.g. from an FEM solution the strip model cannot represent) add to the dc Hessian at the null.
    """

    kind: Kind
    parameters: dict[str, float]
    strips: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)
    rectangles: dict[str, tuple[tuple[float, float, float, float], ...]] = field(default_factory=dict)
    rf_name: str = "rf"

    def __post_init__(self) -> None:
        if self.kind in ("rod_quadrupole", "blade"):
            for key in ("R_m", "Z0_m", "kappa"):
                if key not in self.parameters:
                    raise ValueError(f"{self.kind} geometry needs parameter {key!r}")
        elif self.kind == "surface_five_wire":
            if "a_m" not in self.parameters or "b_m" not in self.parameters:
                raise ValueError("a five-wire layout needs a_m and b_m (c_m optional)")
        elif self.kind == "surface_four_wire":
            if "a_m" not in self.parameters or "c_m" not in self.parameters:
                raise ValueError("a four-wire layout needs a_m and c_m")
        elif not self.strips and not self.rectangles:
            raise ValueError("surface_general needs strips or rectangles")
        present = [k for k in _CURVATURE_KEYS if k in self.parameters]
        if present and len(present) != 3:
            raise ValueError("external dc curvatures come as the full traceless triple xx, yy, zz")
        if present:
            tr = sum(self.parameters[k] for k in _CURVATURE_KEYS)
            scale = max(abs(self.parameters[k]) for k in _CURVATURE_KEYS)
            if abs(tr) > 1e-9 * max(scale, 1e-300):
                raise ValueError("external dc curvatures must be traceless (Laplace's equation)")
        for name, ivs in self.strips.items():
            for x1, x2 in ivs:
                if not x1 < x2:
                    raise ValueError(f"strip {name!r}: needs x1 < x2, got {x1}, {x2}")
        for name, rects in self.rectangles.items():
            for x1, x2, z1, z2 in rects:
                if not (x1 < x2 and z1 < z2):
                    raise ValueError(f"rectangle {name!r}: corners must be ordered x1 < x2, z1 < z2")

    @property
    def is_surface(self) -> bool:
        return self.kind in _SURFACE_KINDS

    def external_dc_hessian_v_per_m2(self) -> np.ndarray:
        if _CURVATURE_KEYS[0] not in self.parameters:
            return np.zeros((3, 3))
        return np.diag([self.parameters[k] for k in _CURVATURE_KEYS])

    def layout(
        self,
    ) -> tuple[
        dict[str, tuple[tuple[float, float], ...]], dict[str, tuple[tuple[float, float, float, float], ...]]
    ]:
        """The strips and rectangles by electrode name after expanding the parametric kinds."""
        strips: dict[str, list[tuple[float, float]]] = {k: list(v) for k, v in self.strips.items()}
        rects = {k: tuple(v) for k, v in self.rectangles.items()}
        p = self.parameters
        if self.kind == "surface_five_wire":
            a, b = p["a_m"], p["b_m"]
            c = p.get("c_m", b)
            if min(a, b, c) <= 0.0:
                raise ValueError("five-wire widths must be positive")
            strips.setdefault(self.rf_name, []).extend([(-c, 0.0), (a, a + b)])
            strips.setdefault("centre", []).append((0.0, a))
            strips.setdefault("left", []).append((-INF, -c))
            strips.setdefault("right", []).append((a + b, INF))
        elif self.kind == "surface_four_wire":
            a, c = p["a_m"], p["c_m"]
            if min(a, c) <= 0.0:
                raise ValueError("four-wire widths must be positive")
            strips.setdefault(self.rf_name, []).extend([(-c, 0.0), (a, INF)])
            strips.setdefault("centre", []).append((0.0, a))
            strips.setdefault("left", []).append((-INF, -c))
        return {k: tuple(v) for k, v in strips.items()}, rects


# ---- strips: complex potential ------------------------------------------------------------------------------------


def _edges(intervals: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    """The FINITE edges of a strip set with their signs (+1 left, -1 right); an infinite edge adds only a constant."""
    e: list[float] = []
    s: list[float] = []
    for x1, x2 in intervals:
        if math.isfinite(x1):
            e.append(x1)
            s.append(1.0)
        if math.isfinite(x2):
            e.append(x2)
            s.append(-1.0)
    return np.asarray(e, dtype=float), np.asarray(s, dtype=float)


def strip_potential(x: float, y: float, x1: float, x2: float) -> float:
    """phi = (1/pi)[arctan((x2 - x)/y) - arctan((x1 - x)/y)] at unit voltage; +-inf edges allowed."""
    if y <= 0.0:
        raise ValueError("the strip potential is evaluated above the plane, y > 0")
    hi = math.pi / 2.0 if x2 == INF else math.atan((x2 - x) / y)
    lo = -math.pi / 2.0 if x1 == -INF else math.atan((x1 - x) / y)
    return (hi - lo) / math.pi


def strip_complex_derivative(w: complex, intervals: tuple[tuple[float, float], ...], order: int) -> complex:
    """f^(order)(w) for f = (i/pi) sum_k sigma_k ln(w - e_k) at unit voltage (order >= 1)."""
    if order < 1:
        raise ValueError("use strip_potential for the potential itself")
    e, s = _edges(intervals)
    coef = (1j / math.pi) * (-1.0) ** (order - 1) * math.factorial(order - 1)
    return complex(coef * np.sum(s / (w - e) ** order))


# ---- rectangles: solid-angle potential and its analytic gradient -----------------------------------------------------------


def rectangle_potential(x: float, y: float, z: float, x1: float, x2: float, z1: float, z2: float) -> float:
    """phi = (1/2pi) sum_{ij} (-1)^{i+j} arctan[(x_i - x)(z_j - z)/(y R_ij)] at unit voltage (House 2008), odd in y."""
    tot = 0.0
    for i, xi in enumerate((x1, x2)):
        for j, zj in enumerate((z1, z2)):
            X, Z = xi - x, zj - z
            R = math.sqrt(X * X + Z * Z + y * y)
            tot += (-1.0) ** (i + j) * math.atan(X * Z / (y * R))
    return tot / (2.0 * math.pi)


def _rect_gradient(
    x: complex, y: complex, z: complex, x1: float, x2: float, z1: float, z2: float
) -> np.ndarray:
    """Analytic gradient of the rectangle potential (complex arguments allowed for the complex-step Hessian)."""
    gx = gy = gz = 0.0 + 0.0j
    for i, xi in enumerate((x1, x2)):
        for j, zj in enumerate((z1, z2)):
            sgn = (-1.0) ** (i + j)
            X, Z = xi - x, zj - z
            R = np.sqrt(X * X + Z * Z + y * y)
            gx += sgn * (-y * Z / (R * (X * X + y * y)))
            gz += sgn * (-y * X / (R * (Z * Z + y * y)))
            gy += sgn * (-X * Z * (R * R + y * y) / (R * (X * X + y * y) * (Z * Z + y * y)))
    return np.array([gx, gy, gz]) / (2.0 * math.pi)


def rectangle_gradient(
    x: float, y: float, z: float, x1: float, x2: float, z1: float, z2: float
) -> np.ndarray:
    return np.real(_rect_gradient(x, y, z, x1, x2, z1, z2))


def rectangle_hessian(x: float, y: float, z: float, x1: float, x2: float, z1: float, z2: float) -> np.ndarray:
    """Hessian by complex-step differentiation of the analytic gradient (exact to round-off)."""
    h = 1e-20 * max(abs(x2 - x1), abs(z2 - z1), abs(y))
    cols = []
    for axis in range(3):
        args = [complex(x), complex(y), complex(z)]
        args[axis] += 1j * h
        cols.append(np.imag(_rect_gradient(args[0], args[1], args[2], x1, x2, z1, z2)) / h)
    hess = np.column_stack(cols)
    return (hess + hess.T) / 2.0


# ---- the assembled electrode model ----------------------------------------------------------------------------------------


class GaplessPlaneTrap:
    """The unit-voltage basis functions of every electrode of a gapless-plane layout and the rf null, escape point and
    dc fields that follow; positions are laboratory (x, y, z) in metres, potentials per volt."""

    def __init__(self, electrodes: Electrodes) -> None:
        if not electrodes.is_surface:
            raise ValueError("GaplessPlaneTrap models the surface-electrode kinds only")
        self.electrodes = electrodes
        self.strips, self.rectangles = electrodes.layout()
        self.names: tuple[str, ...] = tuple(sorted(set(self.strips) | set(self.rectangles)))
        if electrodes.rf_name not in self.names:
            raise ValueError(f"no electrode named {electrodes.rf_name!r} in the layout")

    @property
    def translation_invariant(self) -> bool:
        """True when the rf electrode is made of strips only (the null is then a line along z)."""
        return self.electrodes.rf_name not in self.rectangles

    def potential(self, name: str, r: np.ndarray) -> float:
        x, y, z = (float(v) for v in r)
        tot = sum(strip_potential(x, y, x1, x2) for x1, x2 in self.strips.get(name, ()))
        for x1, x2, z1, z2 in self.rectangles.get(name, ()):
            tot += rectangle_potential(x, y, z, x1, x2, z1, z2)
        return tot

    def gradient(self, name: str, r: np.ndarray) -> np.ndarray:
        x, y, z = (float(v) for v in r)
        g = np.zeros(3)
        ivs = self.strips.get(name, ())
        if ivs:
            fp = strip_complex_derivative(complex(x, y), ivs, 1)  # f' = phi_x - i phi_y
            g[0] += fp.real
            g[1] += -fp.imag
        for x1, x2, z1, z2 in self.rectangles.get(name, ()):
            g += rectangle_gradient(x, y, z, x1, x2, z1, z2)
        return g

    def field(self, name: str, r: np.ndarray) -> np.ndarray:
        """E = -grad Theta (per volt)."""
        return -self.gradient(name, r)

    def hessian(self, name: str, r: np.ndarray) -> np.ndarray:
        x, y, z = (float(v) for v in r)
        h = np.zeros((3, 3))
        ivs = self.strips.get(name, ())
        if ivs:
            fpp = strip_complex_derivative(complex(x, y), ivs, 2)
            h[0, 0] += fpp.real
            h[1, 1] += -fpp.real
            h[0, 1] += -fpp.imag
            h[1, 0] += -fpp.imag
        for x1, x2, z1, z2 in self.rectangles.get(name, ()):
            h += rectangle_hessian(x, y, z, x1, x2, z1, z2)
        return h

    # -- rf --

    def rf_field_unit(self, r: np.ndarray) -> np.ndarray:
        return self.field(self.electrodes.rf_name, r)

    def rf_hessian_unit(self, r: np.ndarray) -> np.ndarray:
        return self.hessian(self.electrodes.rf_name, r)

    def rf_null(self, guess: np.ndarray | None = None) -> np.ndarray:
        """The zero of the rf field above the plane: algebraic for a strip rf electrode, Newton for rectangles."""
        name = self.electrodes.rf_name
        if self.translation_invariant:
            e, s = _edges(self.strips[name])
            # f'(w) = 0  <=>  sum_k s_k prod_{j != k} (w - e_j) = 0
            poly = np.zeros(1)
            for k in range(len(e)):
                poly = np.polyadd(poly, s[k] * np.poly(np.delete(e, k)))
            upper = [w for w in np.roots(poly) if w.imag > 1e-12 * max(1.0, float(np.max(np.abs(e))))]
            if not upper:
                raise ValueError("the rf field has no zero above the plane for this layout")
            best = min(upper, key=lambda w: abs(np.sum(s / (w - e))))
            return np.array([best.real, best.imag, 0.0])
        r = self._null_seed() if guess is None else np.array(guess, dtype=float)
        scale = self._scale()
        # Newton on E(r) = 0 with dE/dr = -H, least squares because a long finite rail leaves H_zz ~ 0
        for _ in range(100):
            step, *_ = np.linalg.lstsq(self.rf_hessian_unit(r), self.rf_field_unit(r), rcond=1e-12)
            r = r + step
            if r[1] <= 0.0:
                raise ValueError("the rf null search left the half-space above the electrodes")
            if np.max(np.abs(step)) < 1e-15 * scale:
                break
        else:
            raise ValueError("rf null search did not converge")
        if np.max(np.abs(self.rf_field_unit(r))) > 1e-9 / scale:
            raise ValueError("the rf field does not vanish at the point found; the layout may have no null")
        return np.asarray(r, dtype=float)

    def _scale(self) -> float:
        vals = [abs(v) for ivs in self.strips.values() for iv in ivs for v in iv if math.isfinite(v)]
        vals += [abs(v) for rs in self.rectangles.values() for rc in rs for v in rc]
        return max(vals) if vals else 1.0

    def _null_seed(self, n: int = 61) -> np.ndarray:
        """The best point of a BOUNDED (x, y) grid as a Newton seed: x across the rf electrode's finite edges, y log-spaced
        over 0.02 to 1 times that span at the rf patches' axial centre (|E_rf| also vanishes at infinity, so an unbounded
        search runs away upward)."""
        name = self.electrodes.rf_name
        xs = [v for iv in self.strips.get(name, ()) for v in iv if math.isfinite(v)]
        xs += [v for rc in self.rectangles.get(name, ()) for v in rc[:2]]
        zs = [v for rc in self.rectangles.get(name, ()) for v in rc[2:]]
        if not xs:
            raise ValueError(
                f"the rf electrode {name!r} has no finite transverse edge to bound the null search"
            )
        x_lo, x_hi, z_c = min(xs), max(xs), (0.5 * (min(zs) + max(zs)) if zs else 0.0)
        span = x_hi - x_lo
        if span <= 0.0:
            raise ValueError("the rf electrode has zero transverse extent")
        best_r, best = np.array([0.5 * (x_lo + x_hi), 0.25 * span, z_c]), math.inf
        for x in np.linspace(x_lo, x_hi, n):
            for y in np.geomspace(0.02 * span, span, n):
                r = np.array([x, y, z_c])
                e = self.rf_field_unit(r)
                value = float(np.dot(e, e))
                if value < best:
                    best, best_r = value, r
        return best_r

    def escape_point(self, null: np.ndarray | None = None) -> np.ndarray:
        """The saddle of |E_rf|^2 above the plane (the pseudopotential escape point).

        Strip rf electrodes: where E != 0 the critical points of |f'|^2 are the zeros of f'', a polynomial of degree
        2(K - 1) in the K finite edges, and the escape point is the upper-half-plane zero of lowest |f'|^2. Rectangles: a
        bracketed climb above the null plus Newton (``_escape_saddle_numeric``).
        """
        if not self.translation_invariant:
            return self._escape_saddle_numeric(null)
        e, s = _edges(self.strips[self.electrodes.rf_name])
        poly = np.zeros(1)
        for k in range(len(e)):
            others = np.delete(e, k)
            poly = np.polyadd(poly, s[k] * np.polymul(np.poly(others), np.poly(others)))
        scale = max(1.0, float(np.max(np.abs(e))))
        cands = [w for w in np.roots(poly) if w.imag > 1e-9 * scale]
        if not cands:
            raise ValueError("no saddle of the rf pseudopotential above the plane")
        w_best = cands[int(np.argmin([abs(np.sum(s / (w - e))) ** 2 for w in cands]))]
        return np.array([w_best.real, w_best.imag, 0.0])

    def _grad_psi_unit(self, r: np.ndarray) -> np.ndarray:
        """grad |E_rf|^2 = -2 H_rf E_rf (dE_i/dr_j = -H_ij, H symmetric)."""
        return np.asarray(-2.0 * (self.rf_hessian_unit(r) @ self.rf_field_unit(r)), dtype=float)

    def _escape_saddle_numeric(self, null: np.ndarray | None = None) -> np.ndarray:
        """The escape saddle of Psi ~ |E_rf|^2 for a layout with rectangles.

        d Psi/dy changes sign once on the vertical line above the null; that root seeds a Newton step on grad Psi = 0 with
        a finite-difference Jacobian and a least-squares solve (a long rail leaves its axial row near zero). The point is
        accepted only if the Hessian of Psi has exactly one negative eigenvalue: a saddle, not a maximum.
        """
        r0 = self.rf_null() if null is None else np.asarray(null, dtype=float)
        y0 = float(r0[1])
        if y0 <= 0.0:
            raise ValueError("the rf null must sit above the electrode plane")

        def dpsi_dy(y: float) -> float:
            return float(self._grad_psi_unit(np.array([r0[0], y, r0[2]]))[1])

        lo = 1.02 * y0
        if dpsi_dy(lo) <= 0.0:
            raise ValueError("Psi is already falling just above the rf null: no escape saddle above it")
        hi = 2.0 * y0
        for _ in range(40):
            if dpsi_dy(hi) < 0.0:
                break
            hi *= 1.5
        else:
            raise ValueError("Psi does not turn over above the rf null: the layout has no bounded depth")
        r = np.array([r0[0], float(brentq(dpsi_dy, lo, hi, xtol=1e-14 * y0, rtol=1e-15)), r0[2]])
        step_h = 1e-6 * y0

        def jacobian(at: np.ndarray) -> np.ndarray:
            jac = np.zeros((3, 3))
            for j in range(3):
                rp, rm = at.copy(), at.copy()
                rp[j] += step_h
                rm[j] -= step_h
                jac[:, j] = (self._grad_psi_unit(rp) - self._grad_psi_unit(rm)) / (2.0 * step_h)
            return jac

        for _ in range(50):
            delta, *_ = np.linalg.lstsq(jacobian(r), -self._grad_psi_unit(r), rcond=1e-10)
            r = r + delta
            if r[1] <= 0.0:
                raise ValueError("the escape-point search left the half-space above the electrodes")
            if float(np.max(np.abs(delta))) < 1e-13 * y0:
                break
        else:
            raise ValueError("the escape-point search did not converge")
        curvature = jacobian(r)
        eigs = np.linalg.eigvalsh((curvature + curvature.T) / 2.0)
        negative = int(np.sum(eigs < -1e-8 * float(np.max(np.abs(eigs)))))
        if negative != 1:
            raise ValueError(
                f"the point found is not a saddle of Psi: the curvature eigenvalues are {eigs} "
                f"({negative} negative, expected exactly one)"
            )
        return np.asarray(r, dtype=float)

    # -- dc --

    def dc_field(self, r: np.ndarray, voltages: dict[str, float]) -> np.ndarray:
        """The named dc electrodes' field at ``r`` plus the external curvature's, -H_ext (r - r_null)."""
        g = np.zeros(3)
        for name, v in voltages.items():
            if name == self.electrodes.rf_name or v == 0.0:
                continue
            g += v * self.field(name, r)
        ext = self.electrodes.external_dc_hessian_v_per_m2()
        if np.any(ext):
            g += -(ext @ (np.asarray(r, dtype=float) - self.rf_null()))
        return g

    def dc_hessian(self, r: np.ndarray, voltages: dict[str, float]) -> np.ndarray:
        h = np.zeros((3, 3))
        for name, v in voltages.items():
            if name == self.electrodes.rf_name or v == 0.0:
                continue
            h += v * self.hessian(name, r)
        return np.asarray(h + self.electrodes.external_dc_hessian_v_per_m2())

    # -- the ion --

    def pseudopotential_j(
        self, r: np.ndarray, v_rf_peak_v: float, mass_kg: float, omega_rf_rad_s: float, *, charge: int = 1
    ) -> float:
        """Q^2 |E_rf|^2/(4 m Omega^2) in joules with the PEAK rf field."""
        e = v_rf_peak_v * self.rf_field_unit(r)
        return (charge * E_C) ** 2 * float(np.dot(e, e)) / (4.0 * mass_kg * omega_rf_rad_s**2)

    def depth_j(self, v_rf_peak_v: float, mass_kg: float, omega_rf_rad_s: float, *, charge: int = 1) -> float:
        """The rf-only trap depth Psi(escape) - Psi(null) in joules."""
        return self.pseudopotential_j(
            self.escape_point(), v_rf_peak_v, mass_kg, omega_rf_rad_s, charge=charge
        )
