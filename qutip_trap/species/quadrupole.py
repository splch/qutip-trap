"""Electric-quadrupole (E2) coupling for optical qubits (PLAN.md Section 4.5.7; Section 13 E2 row; M0a).

The E2 drive on |S, m> <-> |D, m'> has the dipole form with one beam's k (Section 4.3.1) and the scalar

    Omega_{mm'} = (e E_0 k/(2 hbar)) |<S||r^2 C^(2)||D>| |Lambda(m, m')| |g^(q)|,   q = m - m' = -Delta m,

with Lambda the 3j symbol (J 2 J'; -m q m'), C^(2)_q = sqrt(4 pi/5) Y_{2q} (Racah), and g^(q) = c^(q)_{ij} eps_i n_j
the contraction of the polarization and propagation direction with the rank-2 spherical basis tensors
c^(q) = (2/3) b^(q), normalized to sum_ij |c^(q)_ij|^2 = 2/3 (NOT 1) and defined by b^(q)_ij r_i r_j = r^2 C^(2)_q,
never the conjugate duals (derivation audit 2026-09-04: the duals share the norm but turn q = m - m' into
q = m' - m for circular light). The reduced element comes from the D-state lifetime,
A^(E2) = c alpha k^5 |<S||r^2 C^(2)||D>|^2 / (15 (2j' + 1)) with j' the UPPER level and k the VACUUM wavenumber;
Omega follows the plan's convention (carrier pi pulse at t = pi/Omega), twice the Boulder value.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

import numpy as np

from qutip_trap.species.polarization import CVec, Vec, to_atomic_frame
from qutip_trap.species.wigner import Half, as_half_integer, m_values, wigner_3j
from qutip_trap.units import A_0_M, ALPHA_FS, C_M_PER_S, E_C, HBAR_J_S, TWO_PI

C_ALPHA_M_PER_S: float = C_M_PER_S * ALPHA_FS
"""c alpha = e^2/(4 pi eps0 hbar) = 2.187691e6 m/s (Section 4.5.7)."""


def _b_tensors() -> dict[int, np.ndarray]:
    s38 = math.sqrt(3.0 / 8.0)
    x, y, z = np.eye(3)

    def sym(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.outer(a, b) + np.outer(b, a)

    b0 = np.diag([-0.5, -0.5, 1.0]).astype(complex)
    b_p1 = -s38 * (sym(x, z) + 1j * sym(y, z))
    b_m1 = +s38 * (sym(x, z) - 1j * sym(y, z))
    b_p2 = s38 * (np.outer(x, x) - np.outer(y, y) + 1j * sym(x, y))
    b_m2 = s38 * (np.outer(x, x) - np.outer(y, y) - 1j * sym(x, y))
    return {0: b0, 1: b_p1, -1: b_m1, 2: b_p2, -2: b_m2}


B_TENSORS: dict[int, np.ndarray] = _b_tensors()
"""b^(q)_ij with b^(q)_ij r_i r_j = r^2 C^(2)_q (James 1998 Appendix; Section 4.5.7)."""

C_TENSORS: dict[int, np.ndarray] = {q: (2.0 / 3.0) * b for q, b in B_TENSORS.items()}
"""c^(q) = (2/3) b^(q), normalized to sum_ij |c^(q)_ij|^2 = 2/3."""


def racah_c2(q: int, r: Sequence[float]) -> complex:
    """r^2 C^(2)_q(theta, phi) = sqrt(4 pi/5) r^2 Y_{2q} evaluated in Cartesian form, for the tensor identity test."""
    x, y, z = (float(v) for v in r)
    r2 = x * x + y * y + z * z
    if q == 0:
        return complex((3.0 * z * z - r2) / 2.0)
    if abs(q) == 1:
        return complex(-q * math.sqrt(3.0 / 2.0) * z * (x + 1j * q * y))
    if abs(q) == 2:
        return complex(math.sqrt(3.0 / 8.0) * (x + 1j * (q // 2) * y) ** 2)
    raise ValueError("q must lie in -2..2")


def geometric_factor(q: int, polarization_lab: CVec, k_hat_lab: Vec, b_hat: Vec) -> complex:
    """g^(q) = c^(q)_ij eps_i n_j with eps and n expressed in the B_hat frame (complex for elliptical light)."""
    eps = to_atomic_frame(polarization_lab, b_hat)
    n = to_atomic_frame(np.asarray(k_hat_lab, dtype=float), b_hat)
    return complex(eps @ C_TENSORS[q] @ n)


def geometric_factors(polarization_lab: CVec, k_hat_lab: Vec, b_hat: Vec) -> dict[int, complex]:
    return {q: geometric_factor(q, polarization_lab, k_hat_lab, b_hat) for q in (-2, -1, 0, 1, 2)}


def geometric_factor_closed_form(q: int, phi_rad: float, gamma_rad: float) -> float:
    """|g^(q)| in the frame B = z, k = (sin phi, 0, cos phi), eps = (cos gamma cos phi, sin gamma, -cos gamma sin phi) (Roos 2000; Section 4.5.7).

    g^(0) = (1/2)|cos gamma sin 2phi|; |g^(+-1)| = (1/sqrt 6)|cos gamma cos 2phi + i sin gamma cos phi|;
    |g^(+-2)| = (1/sqrt 6)|(1/2) cos gamma sin 2phi + i sin gamma sin phi|; gamma is the angle between the
    polarization and the k-B plane (arcsin|eps . y|).
    """
    cg, sg = math.cos(gamma_rad), math.sin(gamma_rad)
    if q == 0:
        return 0.5 * abs(cg * math.sin(2.0 * phi_rad))
    if abs(q) == 1:
        return abs(cg * math.cos(2.0 * phi_rad) + 1j * sg * math.cos(phi_rad)) / math.sqrt(6.0)
    if abs(q) == 2:
        return abs(0.5 * cg * math.sin(2.0 * phi_rad) + 1j * sg * math.sin(phi_rad)) / math.sqrt(6.0)
    raise ValueError("q must lie in -2..2")


def lambda_3j(J_lower: Half, m: Half, J_upper: Half, mp: Half) -> float:
    """Lambda(m, m') = (J 2 J'; -m q m') with q = m - m' (zero when |Delta m| > 2)."""
    jl, ml, ju, mu = (as_half_integer(x) for x in (J_lower, m, J_upper, mp))
    q = ml - mu
    if abs(q) > 2:
        return 0.0
    return wigner_3j(jl, 2, ju, -ml, q, mu)


def decay_weights(J_lower: Half, J_upper: Half) -> dict[tuple[Fraction, Fraction], float]:
    """w(m, m') = |Lambda(m, m')|^2 / sum_m |Lambda(m, m')|^2, the E2 decay branching of |D, m'> into |S, m>.

    Section 4.5.7 ("Dissipation, dephasing and crosstalk"): "the collapse operator is sqrt(A w(m, m'))
    |S, m><D, m'| with the same A that fixed the Rabi frequency, ... with the per-component weights w from the
    same 3-j squares (sum over m and q at fixed m' equals 1/6)". The raw sum is that 1/6 for every m' (a
    3-j orthogonality relation, pinned in ``tests/test_quadrupole.py``), so dividing by it is the same as
    multiplying by 6 and the weights sum to 1 over m at fixed m': sum_m A w(m, m') = A, the level's total rate.
    """
    jl, ju = as_half_integer(J_lower), as_half_integer(J_upper)
    raw: dict[tuple[Fraction, Fraction], float] = {}
    totals: dict[Fraction, float] = {}
    for mu in m_values(ju):
        for ml in m_values(jl):
            lam = lambda_3j(jl, ml, ju, mu)
            if lam != 0.0:
                raw[(ml, mu)] = lam * lam
                totals[mu] = totals.get(mu, 0.0) + lam * lam
    return {(ml, mu): sq / totals[mu] for (ml, mu), sq in raw.items()}


def quadrupole_collapse_operators(
    partial_rate_rad_s: float,
    J_lower: Half,
    J_upper: Half,
    lower_index: Callable[[Fraction], int],
    upper_index: Callable[[Fraction], int],
    dimension: int,
) -> list[tuple[tuple[Fraction, Fraction], float, np.ndarray]]:
    """The Section 4.5.7 E2 collapse operators sqrt(A w(m, m')) |S, m><D, m'| on an internal basis of ``dimension``.

    ``lower_index`` and ``upper_index`` map m and m' to the basis index of |S, m> and |D, m'>, so the caller
    owns the basis ordering; each entry is ((m, m'), rate = A w(m, m'), the operator matrix), and
    sum over m of the rates at fixed m' is A. ``partial_rate_rad_s`` is the ANGULAR partial rate of the E2
    line (``Transition.partial_rate_rad_s``), the same A that fixed the Rabi frequency.

    Not wired into any engine here: the noise layer of Section 4.5.7 decides between this per-component list
    and the plan's own escape hatch ("a lumped sqrt(A) sigma_- is adequate at A ~ 0.86 s^-1 for microsecond
    gates"), which is what ``readout/presets.py``'s ``shelf_lifetime_s`` uses today.
    """
    if partial_rate_rad_s < 0.0:
        raise ValueError("an E2 partial rate is non-negative")
    if dimension <= 0:
        raise ValueError("the internal basis must be non-empty")
    out: list[tuple[tuple[Fraction, Fraction], float, np.ndarray]] = []
    for (ml, mu), w in sorted(decay_weights(J_lower, J_upper).items()):
        i, j = lower_index(ml), upper_index(mu)
        if not (0 <= i < dimension and 0 <= j < dimension):
            raise IndexError(f"basis index out of range for (m, m') = ({ml}, {mu})")
        op = np.zeros((dimension, dimension), dtype=complex)
        op[i, j] = math.sqrt(partial_rate_rad_s * w)
        out.append(((ml, mu), partial_rate_rad_s * w, op))
    return out


def e2_stark_shift_rad_s(
    couplings_rad_s: Mapping[tuple[Fraction, Fraction], float],
    lower_energies_hz: Mapping[Fraction, float],
    upper_energies_hz: Mapping[Fraction, float],
    m_driven: Half,
    mp_driven: Half,
) -> float:
    """delta_St of the driven |S, m>-|D, m'> component, second order in the OTHER components' couplings.

    Section 4.5.7: "the ac Stark shift of a driven component by the nine off-resonant components inside the
    30 MHz Zeeman span is computed by second-order perturbation from the same Omega(m, m') table". With the
    laser on resonance with (m0, m0'), the detuning of component (m, m') is

        Delta(m, m') = 2 pi [(E_D(m0') - E_S(m0)) - (E_D(m') - E_S(m))],

    a pure Zeeman offset. In the (hbar Omega/2) convention a component shifts its LOWER level by
    +|Omega|^2/(4 Delta) and its UPPER level by -|Omega|^2/(4 Delta) (the same sign rule as
    ``AtomicStructure.light_shift_rad_s``: red light lowers the lower level), so the transition shift is

        delta_St = -sum_{m != m0} |Omega(m, m0')|^2/(4 Delta(m, m0'))
                   -sum_{m' != m0'} |Omega(m0, m')|^2/(4 Delta(m0, m')),

    the driven component itself excluded because the Rabi dynamics treats it exactly. There is no i gamma/2
    in these denominators (Section 4.5.6), so a component degenerate with the driven one raises rather than
    returning an infinity. Tagged UNVALIDATED in the ledger, as Section 12 tags the E2 dissipation block.
    """
    m0, mp0 = as_half_integer(m_driven), as_half_integer(mp_driven)
    if (m0, mp0) not in couplings_rad_s:
        raise KeyError(f"the driven component ({m0}, {mp0}) is not in the coupling table")
    resonance = upper_energies_hz[mp0] - lower_energies_hz[m0]
    total = 0.0
    for (ml, mu), omega in couplings_rad_s.items():
        if (ml, mu) == (m0, mp0) or omega == 0.0:
            continue
        if ml != m0 and mu != mp0:  # shares neither level with the driven pair: no second-order shift of it
            continue
        delta = TWO_PI * (resonance - (upper_energies_hz[mu] - lower_energies_hz[ml]))
        if delta == 0.0:
            raise ZeroDivisionError(
                f"E2 component ({ml}, {mu}) is degenerate with the driven ({m0}, {mp0}); the second-order sum "
                "of Section 4.5.6 has no i gamma/2 and is valid only away from degeneracy"
            )
        total -= abs(omega) ** 2 / (4.0 * delta)
    return total


def reduced_element_from_lifetime_m2(
    wavelength_vac_m: float, partial_rate_rad_s: float, J_upper: Half
) -> float:
    """|<S||r^2 C^(2)||D>| in m^2 from A = c alpha k^5 |<>|^2/(15 (2j'+1)) with the UPPER level's degeneracy."""
    k = 2.0 * math.pi / wavelength_vac_m
    gjp = 2.0 * float(as_half_integer(J_upper)) + 1.0
    return math.sqrt(15.0 * gjp * partial_rate_rad_s / (C_ALPHA_M_PER_S * k**5))


def reduced_element_a0_squared(wavelength_vac_m: float, partial_rate_rad_s: float, J_upper: Half) -> float:
    return reduced_element_from_lifetime_m2(wavelength_vac_m, partial_rate_rad_s, J_upper) / A_0_M**2


def rabi_frequency_e2_rad_s(
    e0_v_per_m: float,
    wavelength_vac_m: float,
    reduced_element_m2: float,
    J_lower: Half,
    m: Half,
    J_upper: Half,
    mp: Half,
    polarization_lab: CVec,
    k_hat_lab: Vec,
    b_hat: Vec,
) -> float:
    """Omega_{mm'} = (e E_0 k/(2 hbar)) |<S||r^2 C^(2)||D>| |Lambda(m, m')| |g^(q)|, q = m - m' (plan convention)."""
    k = 2.0 * math.pi / wavelength_vac_m
    ml, mu = as_half_integer(m), as_half_integer(mp)
    q = ml - mu
    if abs(q) > 2:
        return 0.0
    lam = lambda_3j(J_lower, ml, J_upper, mu)
    g = geometric_factor(int(q), polarization_lab, k_hat_lab, b_hat)
    return E_C * e0_v_per_m * k / (2.0 * HBAR_J_S) * reduced_element_m2 * abs(lam) * abs(g)


def stretched_closed_form_rad_s(
    e0_v_per_m: float, wavelength_vac_m: float, partial_rate_rad_s: float
) -> float:
    """Omega = (e E_0/hbar) sqrt(5 lambda_vac^3 A/(64 pi^3 c alpha)): the |Delta m| = 2 stretched line at (90, 90) degrees.

    90/(12^2 x 8) = 5/64 exactly (James Eqs. 5.11, 5.13; Roos p. 31).
    """
    return (
        E_C
        * e0_v_per_m
        / HBAR_J_S
        * math.sqrt(5.0 * wavelength_vac_m**3 * partial_rate_rad_s / (64.0 * math.pi**3 * C_ALPHA_M_PER_S))
    )


def e2_over_e1_amplitude_ratio(wavelength_vac_m: float) -> float:
    """k a_0 / 2, the E2/E1 amplitude suppression (about alpha/16 at optical wavelengths, not alpha/2)."""
    return 2.0 * math.pi / wavelength_vac_m * A_0_M / 2.0


__all__ = [
    "B_TENSORS",
    "C_ALPHA_M_PER_S",
    "C_TENSORS",
    "decay_weights",
    "e2_over_e1_amplitude_ratio",
    "e2_stark_shift_rad_s",
    "geometric_factor",
    "geometric_factor_closed_form",
    "geometric_factors",
    "lambda_3j",
    "quadrupole_collapse_operators",
    "rabi_frequency_e2_rad_s",
    "racah_c2",
    "reduced_element_a0_squared",
    "reduced_element_from_lifetime_m2",
    "stretched_closed_form_rad_s",
]
