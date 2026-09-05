"""Electrode geometry: gapless-plane surface-electrode traps and the rod/blade map (PLAN.md Section 4.1.6; 13; M1).

Coordinates: the substrate is the plane y = 0, y is the height above it, z is the trap axis (along the rf
rails, the axial direction of the crystal module) and x the in-plane transverse coordinate; House 2008's
(x, y) transverse plane is this one. Every electrode is a set of INFINITE STRIPS along z, {x1 < x < x2}, or of
finite RECTANGLES {x1 < x < x2, z1 < z < z2}; gaps have zero width and the plane outside the electrodes is
grounded (House 2008, Wesenberg 2008), and finite gaps and thickness are outside the first release, which is
why ``Electrodes.parameters`` also accepts externally solved dc curvatures.

Conventions (Section 13, row "Surface-electrode geometry"): widths from gap centre to gap centre (second-order
accuracy in the gap); the ion at the minimum of the TOTAL potential, not at the rf null; the escape point a
saddle, found for the strip model from the algebraic condition f''(w) = 0 (a polynomial of degree 2(K - 1) in
the K finite edges, degree six for two rails) rather than by a search constrained to x = x0; the unit-voltage
basis functions Theta = phi/V stored dimensionless with sum_i Theta_i = 1 on the plane when every region has a
strip; House's four-arctan rectangle on the principal branch of arctan, corners ordered, odd in the height. The
five-wire null height is h = sqrt(a(a + 2b))/2 for FULL widths a (centre) and b (rails); the two-rail null is
x0 = ac/(b + c), y0 = sqrt(abc(a + b + c))/(b + c) in House's coordinates (rails at -c < x < 0 and
a < x < a + b), the null moving toward the narrower rail. The rf pseudopotential is exactly isotropic in the
transverse plane at a two-dimensional rf null, so the transverse principal axes are set by the dc potential
alone (Section 4.1.6).

Strip potentials use the complex potential f(w) = (i/pi) sum_k sigma_k V_k ln(w - e_k), w = x + i y, with
sigma = +1 at a left edge and -1 at a right edge (Wesenberg 2008); phi = Re f, E_x - i E_y = -f'(w), and
d^m_x d^n_y phi = Re[i^n f^(m+n)], so every derivative is analytic. Rectangles use the closed-form gradient of
the solid-angle potential and a complex-step derivative of that gradient for the Hessian (exact to round-off).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from qutip_trap.units import E_C

INF = math.inf
Kind = Literal["rod_quadrupole", "blade", "surface_five_wire", "surface_four_wire", "surface_general"]
_SURFACE_KINDS = ("surface_five_wire", "surface_four_wire", "surface_general")
_CURVATURE_KEYS = ("dc_curvature_xx_v_per_m2", "dc_curvature_yy_v_per_m2", "dc_curvature_zz_v_per_m2")


@dataclass(frozen=True)
class Electrodes:
    """Electrode geometry; ``parameters`` are the named lengths of the layout in metres (and, for surface kinds,
    optional externally solved dc curvatures in V/m^2).

    - ``rod_quadrupole`` and ``blade``: parameters ``R_m`` (radial scale R' ~ R), ``Z0_m`` (endcap scale), ``kappa``
      (endcap efficiency) and optionally ``alpha`` (the rf geometry factor of Berkeland's phi_ac term); the dc
      record carries the single endcap voltage U0 under any name.
    - ``surface_five_wire``: ``a_m`` (centre electrode FULL width), ``b_m`` (right rail width), optional ``c_m``
      (left rail width, default b): rf rails at -c < x < 0 and a < x < a + b, dc strips "centre" (0 < x < a),
      "left" (x < -c) and "right" (x > a + b), all infinite along z.
    - ``surface_four_wire``: ``a_m`` and ``c_m``: rails at -c < x < 0 and x > a (semi-infinite), dc "centre" and
      "left".
    - ``surface_general``: the layout is given explicitly by ``strips`` and ``rectangles``.

    ``strips`` maps an electrode name to its x-intervals (``math.inf`` allowed for a semi-infinite strip),
    ``rectangles`` maps a name to (x1, x2, z1, z2) patches; the parametric kinds may add either. The rf electrode
    is the one named ``rf_name``. Externally solved dc curvatures ``dc_curvature_xx/yy/zz_v_per_m2`` (one set,
    traceless, from an FEM solution of electrodes the strip model cannot represent, Section 4.1.6) add to the
    dc Hessian at the null.
    """

    kind: Kind
    parameters: dict[str, float]
    electrode_names: tuple[str, ...] = ()
    strips: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)
    rectangles: dict[str, tuple[tuple[float, float, float, float], ...]] = field(default_factory=dict)
    rf_name: str = "rf"

    def __post_init__(self) -> None:
        if self.kind in ("rod_quadrupole", "blade"):
            for key in ("R_m", "Z0_m", "kappa"):
                if key not in self.parameters:
                    raise ValueError(f"{self.kind} geometry needs parameter {key!r}")
        elif self.kind == "surface_five_wire":
            for key in ("a_m", "b_m"):
                if key not in self.parameters:
                    raise ValueError("a five-wire layout needs a_m and b_m (c_m optional)")
        elif self.kind == "surface_four_wire":
            for key in ("a_m", "c_m"):
                if key not in self.parameters:
                    raise ValueError("a four-wire layout needs a_m and c_m")
        elif self.kind == "surface_general":
            if not self.strips and not self.rectangles:
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


def _edges(intervals: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray, float]:
    """Finite edges with their signs (+1 left, -1 right) and the constant from infinite edges (in units of V)."""
    e: list[float] = []
    s: list[float] = []
    const = 0.0
    for x1, x2 in intervals:
        if math.isfinite(x1):
            e.append(x1)
            s.append(1.0)
        else:
            const += 0.5  # arctan((x1 - x)/y) -> -pi/2 contributes +1/2
        if math.isfinite(x2):
            e.append(x2)
            s.append(-1.0)
        else:
            const += 0.5
    return np.asarray(e, dtype=float), np.asarray(s, dtype=float), const


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
    e, s, _ = _edges(intervals)
    coef = (1j / math.pi) * (-1.0) ** (order - 1) * math.factorial(order - 1)
    return complex(coef * np.sum(s / (w - e) ** order))


def strip_derivative_xy(w: complex, intervals: tuple[tuple[float, float], ...], m: int, n: int) -> float:
    """d^m_x d^n_y phi = Re[i^n f^(m+n)] for a strip set at unit voltage (m + n >= 1)."""
    return float(np.real((1j) ** n * strip_complex_derivative(w, intervals, m + n)))


# ---- rectangles: solid-angle potential and its analytic gradient -----------------------------------------------------------


def rectangle_potential(x: float, y: float, z: float, x1: float, x2: float, z1: float, z2: float) -> float:
    """phi = (1/2pi) sum_{ij} (-1)^{i+j} arctan[(x_i - x)(z_j - z)/(y R_ij)] at unit voltage (House 2008), principal arctan.

    Needs x1 < x2 and z1 < z2 (one swap negates the potential); odd in y (returns -phi below the plane).
    """
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
    """Hessian by complex-step differentiation of the analytic gradient (no cancellation, exact to round-off)."""
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
    """Unit-voltage basis functions Theta_k(r) of every electrode of a gapless-plane layout and what follows from them.

    Positions are laboratory (x, y, z) triples in metres with y the height. All potentials are per volt.
    """

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

    # -- basis functions --

    def potential(self, name: str, r: np.ndarray) -> float:
        x, y, z = (float(v) for v in r)
        tot = 0.0
        for x1, x2 in self.strips.get(name, ()):
            tot += strip_potential(x, y, x1, x2)
        for x1, x2, z1, z2 in self.rectangles.get(name, ()):
            tot += rectangle_potential(x, y, z, x1, x2, z1, z2)
        return tot

    def gradient(self, name: str, r: np.ndarray) -> np.ndarray:
        x, y, z = (float(v) for v in r)
        g = np.zeros(3)
        w = complex(x, y)
        ivs = self.strips.get(name, ())
        if ivs:
            fp = strip_complex_derivative(w, ivs, 1)  # f' = phi_x - i phi_y
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

    def third_derivatives(self, name: str, r: np.ndarray, *, step_fraction: float = 1e-4) -> np.ndarray:
        """T_abc = d^3 Theta/dr_a dr_b dr_c: analytic for strips, central differences of the Hessian for rectangles."""
        x, y, z = (float(v) for v in r)
        t = np.zeros((3, 3, 3))
        ivs = self.strips.get(name, ())
        if ivs:
            w = complex(x, y)
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        n = (a == 1) + (b == 1) + (c == 1)
                        t[a, b, c] += strip_derivative_xy(w, ivs, 3 - n, n)
        rects = self.rectangles.get(name, ())
        if rects:
            h = step_fraction * max(abs(y), 1e-9)
            for axis in range(3):
                rp = np.array([x, y, z], dtype=float)
                rm = rp.copy()
                rp[axis] += h
                rm[axis] -= h
                hp = np.zeros((3, 3))
                hm = np.zeros((3, 3))
                for x1, x2, z1, z2 in rects:
                    hp += rectangle_hessian(rp[0], rp[1], rp[2], x1, x2, z1, z2)
                    hm += rectangle_hessian(rm[0], rm[1], rm[2], x1, x2, z1, z2)
                t[:, :, axis] += (hp - hm) / (2.0 * h)
            # symmetrize
            t = (
                t
                + t.transpose(0, 2, 1)
                + t.transpose(1, 0, 2)
                + t.transpose(1, 2, 0)
                + t.transpose(2, 0, 1)
                + t.transpose(2, 1, 0)
            ) / 6.0
        return t

    def sum_of_basis(self, r: np.ndarray) -> float:
        """sum_i Theta_i(r): 1 on the electrode plane when every region is assigned a strip (Section 13)."""
        return sum(self.potential(n, r) for n in self.names)

    # -- rf --

    def rf_field_unit(self, r: np.ndarray) -> np.ndarray:
        return self.field(self.electrodes.rf_name, r)

    def rf_hessian_unit(self, r: np.ndarray) -> np.ndarray:
        return self.hessian(self.electrodes.rf_name, r)

    def rf_null(self, guess: np.ndarray | None = None) -> np.ndarray:
        """The zero of the rf field above the plane: algebraic for a strip rf electrode, Newton for rectangles."""
        name = self.electrodes.rf_name
        if self.translation_invariant:
            e, s, _ = _edges(self.strips[name])
            # f'(w) = 0  <=>  sum_k s_k prod_{j != k} (w - e_j) = 0
            poly = np.zeros(1)
            for k in range(len(e)):
                others = np.delete(e, k)
                poly = np.polyadd(poly, s[k] * np.poly(others))
            roots = np.roots(poly)
            upper = [w for w in roots if w.imag > 1e-12 * max(1.0, float(np.max(np.abs(e))))]
            if not upper:
                raise ValueError("the rf field has no zero above the plane for this layout")
            best = min(upper, key=lambda w: abs(np.sum(s / (w - e))))
            return np.array([best.real, best.imag, 0.0])
        r = np.array([0.0, 1.0, 0.0]) * self._scale() if guess is None else np.array(guess, dtype=float)
        scale = self._scale()
        # Newton on E_u(r) = 0 with the analytic Jacobian dE/dr = -H (least squares: a long finite rail leaves H_zz ~ 0)
        for _ in range(100):
            e = self.rf_field_unit(r)
            h = self.rf_hessian_unit(r)
            step, *_ = np.linalg.lstsq(h, e, rcond=1e-12)
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

    def escape_point(self, null: np.ndarray | None = None) -> np.ndarray:
        """The saddle of |E_rf|^2 above the plane (the pseudopotential escape point) for a strip rf electrode.

        Where E != 0 the critical points of |f'|^2 are the zeros of f'' (Section 13, "escape point a saddle from
        G'(Z) = 0"), a polynomial of degree 2(K - 1) in the K finite edges; the escape point is the upper-half-plane
        zero of lowest pseudopotential.
        """
        if not self.translation_invariant:
            raise NotImplementedError(
                "the escape point is computed algebraically for strip rf electrodes only"
            )
        e, s, _ = _edges(self.strips[self.electrodes.rf_name])
        poly = np.zeros(1)
        for k in range(len(e)):
            others = np.delete(e, k)
            poly = np.polyadd(poly, s[k] * np.polymul(np.poly(others), np.poly(others)))
        roots = np.roots(poly)
        scale = max(1.0, float(np.max(np.abs(e))))
        cands = [w for w in roots if w.imag > 1e-9 * scale]
        if not cands:
            raise ValueError("no saddle of the rf pseudopotential above the plane")
        e2 = [abs(np.sum(s / (w - e))) ** 2 for w in cands]
        w_best = cands[int(np.argmin(e2))]
        return np.array([w_best.real, w_best.imag, 0.0])

    # -- dc --

    def dc_potential(self, r: np.ndarray, voltages: dict[str, float]) -> float:
        tot = 0.0
        for name, v in voltages.items():
            if name == self.electrodes.rf_name or v == 0.0:
                continue
            if name not in self.names:
                raise KeyError(f"dc voltage on unknown electrode {name!r}; layout has {self.names}")
            tot += v * self.potential(name, r)
        return tot

    def dc_field(self, r: np.ndarray, voltages: dict[str, float]) -> np.ndarray:
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
        e = v_rf_peak_v * self.rf_field_unit(r)
        return (charge * E_C) ** 2 * float(np.dot(e, e)) / (4.0 * mass_kg * omega_rf_rad_s**2)

    def total_energy_j(
        self,
        r: np.ndarray,
        v_rf_peak_v: float,
        mass_kg: float,
        omega_rf_rad_s: float,
        voltages: dict[str, float],
        *,
        stray_field_v_per_m: np.ndarray | None = None,
        charge: int = 1,
    ) -> float:
        """Psi + Q Phi_dc - Q E_stray . (r - r_null): the ion sits at its minimum, not at the rf null (Section 13)."""
        u = self.pseudopotential_j(r, v_rf_peak_v, mass_kg, omega_rf_rad_s, charge=charge)
        u += charge * E_C * self.dc_potential(r, voltages)
        ext = self.electrodes.external_dc_hessian_v_per_m2()
        d = np.asarray(r, dtype=float) - self.rf_null()
        if np.any(ext):
            u += charge * E_C * 0.5 * float(d @ ext @ d)
        if stray_field_v_per_m is not None:
            u += -charge * E_C * float(np.dot(np.asarray(stray_field_v_per_m, dtype=float), d))
        return u

    def total_gradient_j_per_m(
        self,
        r: np.ndarray,
        v_rf_peak_v: float,
        mass_kg: float,
        omega_rf_rad_s: float,
        voltages: dict[str, float],
        *,
        stray_field_v_per_m: np.ndarray | None = None,
        charge: int = 1,
    ) -> np.ndarray:
        """grad U = -(Q^2 V^2/(2 m Omega^2)) H_rf E_u + Q grad Phi_dc + Q ext (r - r_null) - Q E_stray, analytic."""
        r = np.asarray(r, dtype=float)
        qe = charge * E_C
        e_u = self.rf_field_unit(r)
        g = -((qe * v_rf_peak_v) ** 2) / (2.0 * mass_kg * omega_rf_rad_s**2) * (self.rf_hessian_unit(r) @ e_u)
        for name, v in voltages.items():
            if name == self.electrodes.rf_name or v == 0.0:
                continue
            g += qe * v * self.gradient(name, r)
        ext = self.electrodes.external_dc_hessian_v_per_m2()
        if np.any(ext):
            g += qe * (ext @ (r - self.rf_null()))
        if stray_field_v_per_m is not None:
            g += -qe * np.asarray(stray_field_v_per_m, dtype=float)
        return np.asarray(g)

    def total_hessian_j_per_m2(
        self,
        r: np.ndarray,
        v_rf_peak_v: float,
        mass_kg: float,
        omega_rf_rad_s: float,
        voltages: dict[str, float],
        *,
        charge: int = 1,
    ) -> np.ndarray:
        """Hess U = (Q^2 V^2/(2 m Omega^2)) [H_rf^2 - sum_c E_u,c T_abc] + Q H_dc + Q ext (T the third derivatives of Theta_rf);
        at the rf null the second term vanishes and this is the pseudopotential-route secular matrix (Section 4.1.6)."""
        r = np.asarray(r, dtype=float)
        qe = charge * E_C
        h_rf = self.rf_hessian_unit(r)
        e_u = self.rf_field_unit(r)
        h = h_rf @ h_rf
        if np.max(np.abs(e_u)) > 0.0:
            t = self.third_derivatives(self.electrodes.rf_name, r)
            h = h - np.einsum("c,abc->ab", e_u, t)
        h = (qe * v_rf_peak_v) ** 2 / (2.0 * mass_kg * omega_rf_rad_s**2) * h
        return np.asarray(h + qe * self.dc_hessian(r, voltages))

    def equilibrium(
        self,
        v_rf_peak_v: float,
        mass_kg: float,
        omega_rf_rad_s: float,
        voltages: dict[str, float],
        *,
        stray_field_v_per_m: np.ndarray | None = None,
        charge: int = 1,
    ) -> np.ndarray:
        """The stationary point of the total potential nearest the rf null (a single ion), by Newton iteration on the
        analytic gradient and Hessian; in the translation-invariant strip model the axial coordinate stays at the null's."""
        r = self.rf_null()
        free = [0, 1] if not self.rectangles else [0, 1, 2]
        scale = self._scale()
        for _ in range(100):
            g = self.total_gradient_j_per_m(
                r,
                v_rf_peak_v,
                mass_kg,
                omega_rf_rad_s,
                voltages,
                stray_field_v_per_m=stray_field_v_per_m,
                charge=charge,
            )[free]
            h = self.total_hessian_j_per_m2(r, v_rf_peak_v, mass_kg, omega_rf_rad_s, voltages, charge=charge)[
                np.ix_(free, free)
            ]
            step = np.linalg.solve(h, -g)
            r = r.copy()
            r[free] += step
            if r[1] <= 0.0:
                raise ValueError("the equilibrium search left the half-space above the electrodes")
            if np.max(np.abs(step)) < 1e-15 * scale:
                return np.asarray(r, dtype=float)
        raise ValueError("equilibrium search did not converge")

    def depth_j(self, v_rf_peak_v: float, mass_kg: float, omega_rf_rad_s: float, *, charge: int = 1) -> float:
        """The rf-only trap depth Psi(escape) - Psi(null) in joules (peak V_rf convention)."""
        return self.pseudopotential_j(
            self.escape_point(), v_rf_peak_v, mass_kg, omega_rf_rad_s, charge=charge
        )


def minimum_norm_dc_voltages(
    model: GaplessPlaneTrap,
    names: tuple[str, ...],
    *,
    curvature_zz_v_per_m2: float,
    at: np.ndarray | None = None,
) -> dict[str, float]:
    """House 2008 Eq. 30: the minimum-norm voltages that give a chosen axial curvature with zero field at the null.

    Minimize sum_i V_i^2 subject to sum_i V_i grad Theta_i(r) = 0 (no stray field, hence no excess micromotion) and
    sum_i V_i d^2 Theta_i/dz^2(r) = curvature; a linear system solved by the pseudo-inverse (equivalently Lagrange
    multipliers) that needs at least four independent electrodes.
    """
    r = model.rf_null() if at is None else np.asarray(at, dtype=float)
    if len(names) < 4:
        raise ValueError(
            "four independent dc electrodes are needed for three field components and one curvature"
        )
    rows = [[model.gradient(n, r)[axis] for n in names] for axis in range(3)]
    rows.append([model.hessian(n, r)[2, 2] for n in names])
    a = np.array(rows)
    rhs = np.array([0.0, 0.0, 0.0, curvature_zz_v_per_m2])
    v = np.linalg.pinv(a) @ rhs
    if np.max(np.abs(a @ v - rhs)) > 1e-9 * max(abs(curvature_zz_v_per_m2), 1e-300):
        raise ValueError("the electrodes cannot realize a pure axial curvature with zero field at this point")
    return {n: float(x) for n, x in zip(names, v)}


# ---- closed forms of House 2008, Nizamani 2012, Wesenberg 2008 (Section 4.1.6) ---------------------------------------------


def five_wire_null_height_m(a_m: float, b_m: float) -> float:
    """h = sqrt(a(a + 2b))/2 for FULL widths a (centre) and b (rails) (House 2008; PLAN.md M1 tests).

    The first version of the plan wrote sqrt(a(a + b)), the same formula with a as a half-width, unstated
    (caught by the 2026-09-04 critique); ``check_surface_mixed.py`` gives y0 = 0.921954 for a = 1, b = 1.2.
    """
    if a_m <= 0.0 or b_m <= 0.0:
        raise ValueError("widths must be positive")
    return math.sqrt(a_m * (a_m + 2.0 * b_m)) / 2.0


def two_rail_null(a_m: float, b_m: float, c_m: float) -> tuple[float, float]:
    """(x0, y0) = (ac/(b + c), sqrt(abc(a + b + c))/(b + c)) for rails at -c < x < 0 and a < x < a + b (House Eqs. 24-27).

    The null moves toward the narrower rail by a|b - c|/(2(b + c)); Nizamani and Hensinger's x0 = ac/(b + c) under a
    figure with mirrored rail labels reads ab/(b + c) in their own labelling (Section 4.1.6), the reason this module
    stores edges rather than widths.
    """
    if min(a_m, b_m, c_m) <= 0.0:
        raise ValueError("widths must be positive")
    return a_m * c_m / (b_m + c_m), math.sqrt(a_m * b_m * c_m * (a_m + b_m + c_m)) / (b_m + c_m)


def five_wire_escape_height_m(a_m: float, b_m: float) -> float:
    """y_E = [2ab + a^2 + 2(a + b) sqrt(2ab + a^2)]^{1/2}/2 at x_E = a/2 (House Eq. 28, the coefficient 2 restored)."""
    s = math.sqrt(2.0 * a_m * b_m + a_m * a_m)
    return math.sqrt(2.0 * a_m * b_m + a_m * a_m + 2.0 * (a_m + b_m) * s) / 2.0


def five_wire_depth_j(
    a_m: float, b_m: float, v_rf_peak_v: float, mass_kg: float, omega_rf_rad_s: float, *, charge: int = 1
) -> float:
    """psi_E = (Z^2 e^2 V^2/(pi^2 m Omega^2)) [b/((a + b)^2 + (a + b) sqrt(2ab + a^2))]^2 (House Eq. 29, equal rails only)."""
    s = math.sqrt(2.0 * a_m * b_m + a_m * a_m)
    g = b_m / ((a_m + b_m) ** 2 + (a_m + b_m) * s)
    return (charge * E_C * v_rf_peak_v) ** 2 * g * g / (math.pi**2 * mass_kg * omega_rf_rad_s**2)


def nizamani_kappa(a_m: float, b_m: float, c_m: float) -> float:
    """kappa = [2 sqrt(abc(a+b+c))/((2a+b+c)(2a+b+c+2 sqrt(a(a+b+c))))]^2 of Xi = e^2 V^2 kappa/(pi^2 m Omega^2 y0^2).

    Algebraically House's Eq. 29 for c = b (Section 4.1.6); for c != b it extrapolates a formula House derived for
    equal rails, so the module's numerical saddle is the reference there. Its maximum over b/a at fixed ion height
    is (5 sqrt 5 - 11)/8 = 0.0225425, the geometry-independent bound.
    """
    s = 2.0 * a_m + b_m + c_m
    return (
        2.0
        * math.sqrt(a_m * b_m * c_m * (a_m + b_m + c_m))
        / (s * (s + 2.0 * math.sqrt(a_m * (a_m + b_m + c_m))))
    ) ** 2


KAPPA_MAX: float = (5.0 * math.sqrt(5.0) - 11.0) / 8.0
"""The maximal depth coefficient at fixed ion height, House's four-wire 1/(2(11 + 5 sqrt 5)) = Wesenberg's optimal
quadrupole bound = the numerical maximum for every rail ratio (Section 4.1.6)."""


def depth_optimal_b_over_a() -> float:
    """b/a = 1.1914879 = cbrt(54 - 6 sqrt 33)/6 + cbrt(54 + 6 sqrt 33)/6, House's depth-optimal rail width at fixed a."""
    s = 6.0 * math.sqrt(33.0)
    return math.cbrt(54.0 - s) / 6.0 + math.cbrt(54.0 + s) / 6.0


def nizamani_q(
    v_rf_peak_v: float, mass_kg: float, omega_rf_rad_s: float, height_m: float, *, charge: int = 1
) -> float:
    """Nizamani and Hensinger's rf stability factor q_N = 2 e V_rf/(m Omega^2 h^2), normalized by the ion height.

    NOT the trap's Mathieu q (Section 4.1.6): it exceeds the principal-axis q = sqrt(Q11^2 + Q12^2) by a geometry-set
    factor (3.2002 for their own trap) and must never enter omega = q Omega/(2 sqrt 2) or a stability boundary.
    """
    return 2.0 * charge * E_C * v_rf_peak_v / (mass_kg * omega_rf_rad_s**2 * height_m**2)


__all__ = [
    "KAPPA_MAX",
    "Electrodes",
    "GaplessPlaneTrap",
    "depth_optimal_b_over_a",
    "five_wire_depth_j",
    "five_wire_escape_height_m",
    "five_wire_null_height_m",
    "minimum_norm_dc_voltages",
    "nizamani_kappa",
    "nizamani_q",
    "rectangle_gradient",
    "rectangle_hessian",
    "rectangle_potential",
    "strip_complex_derivative",
    "strip_derivative_xy",
    "strip_potential",
    "two_rail_null",
]
