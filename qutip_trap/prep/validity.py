"""The Section 4.2.8 (vii) validity conditions as run-time assertions (PLAN.md Section 4.2.8; milestone M3).

Section 4.2.8 (vii) lists the conditions "the module asserts at run time rather than documents":

    eta_ij^2 (2 n + 1) << 1 per transition including the emitted photon, Omega << Gamma for every closed form,
    the cooling rate W << nu and W << every internal rate for the adiabatic elimination, Gamma', gamma' << nu for
    resolved sidebands and Gamma > omega_R for Doppler cooling, s_aux << 1 and |delta_aux| << Gamma_10 + Gamma_12
    for the three-level reduction.

Every one of them lives here as an ``assert_*`` that raises :class:`ValidityError` (a ``ValueError``, so the
existing ``pytest.raises(ValueError)`` guards keep matching) with the measured ratio in the message, and every one
takes an escape hatch keyword so a caller that deliberately probes the boundary - a source's own published fixture
that violates its own regime, a saturation-error study, the algebraic normalization of a floor - says so explicitly
instead of the guard being weakened for everybody. ``<<`` is taken as "below one tenth" throughout, except for the
resolved-sideband condition, where :meth:`EffectiveTwoLevel.resolved` already fixed one fifth; the booleans the
plan's report functions return (``EffectiveTwoLevel.resolved``, ``effective_two_level_valid``,
``StageRates.lamb_dicke_guard``) are kept, so a caller can test a configuration without catching an exception.

Section 13 symbol note: ``Gamma`` is the FULL (angular) linewidth and ``Omega`` the (hbar Omega/2)-convention Rabi
frequency throughout, so ``Omega/Gamma`` here is sqrt(s/2) and not the saturation parameter itself.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from qutip_trap.light.bloch import WEAK_DRIVE_MAX
from qutip_trap.light.recoil import free_recoil_energy_j
from qutip_trap.units import HBAR_J_S

SMALL = 0.1
"""The plan's "<<": a ratio below one tenth (the same number as ``bloch.WEAK_DRIVE_MAX``)."""

LAMB_DICKE_MAX = 0.25
"""eta^2 (2 nbar + 1) above which the Lamb-Dicke expansion of Section 4.2.8 (i) is refused.

Equivalently eta sqrt(2 nbar + 1) < 1/2: the exact carrier element e^{-eta^2/2} L_n(eta^2) = 1 - eta^2(n + 1/2) + ...
then departs from its Lamb-Dicke value by under 12.5 %. This is the bound the plan's own anchors sit inside - the
171Yb+ polarization-gradient triple reports eta sqrt(2 nbar + 1) = 0.09023 sqrt(21) = 0.413 (Section 9.15, so
eta^2 (2 nbar + 1) = 0.171) and the nu = Gamma/20 Doppler fixtures 0.242 and 0.35 (Section 9.3, 0.059 and 0.123) -
while a genuinely non-Lamb-Dicke configuration (eta sqrt(2 nbar + 1) of order 1) is refused."""

ADIABATIC_MAX = SMALL
"""W/nu and W/Gamma above which the adiabatic elimination behind the rate equation is refused."""

RESOLVED_MAX = 0.2
"""max(Gamma', gamma')/nu above which the sidebands are not resolved (the bound ``EffectiveTwoLevel.resolved`` uses)."""


class ValidityError(ValueError):
    """A Section 4.2.8 (vii) validity condition of a closed form or rate model is violated."""


def assert_weak_drive(
    omega_rad_s: float,
    gamma_rad_s: float,
    *,
    allow_saturation: bool = False,
    what: str = "this closed form",
    maximum: float = WEAK_DRIVE_MAX,
) -> None:
    """Omega << Gamma for every closed form (Section 4.2.8 vii); the saturation error is of order (Omega/Gamma)^2."""
    if allow_saturation:
        return
    if gamma_rad_s <= 0.0:
        raise ValidityError("the linewidth is positive")
    ratio = abs(omega_rad_s) / gamma_rad_s
    if ratio > maximum:
        raise ValidityError(
            f"Omega/Gamma = {ratio:.4g} > {maximum:g}: {what} needs Omega << Gamma (Section 4.2.8 vii); use the "
            "level-C solve or the spectrum path, or pass allow_saturation=True to accept the saturation error"
        )


def assert_lamb_dicke(
    eta: float,
    nbar: float,
    *,
    allow_strong_coupling: bool = False,
    what: str = "the rate model",
    maximum: float = LAMB_DICKE_MAX,
) -> None:
    """eta^2 (2 nbar + 1) << 1 per transition, the emitted photon included (Section 4.2.8 i, vii)."""
    if allow_strong_coupling:
        return
    if nbar < 0.0:
        raise ValidityError("nbar is non-negative")
    value = eta**2 * (2.0 * nbar + 1.0)
    if value > maximum:
        raise ValidityError(
            f"eta^2 (2 nbar + 1) = {value:.4g} > {maximum:g} (eta = {eta:.4g}, nbar = {nbar:.4g}): {what} is a "
            "Lamb-Dicke expansion (Section 4.2.8 vii); use the level-C solve with the exact displacement operators, "
            "or pass allow_strong_coupling=True"
        )


def assert_adiabatic(
    rate_per_s: float,
    nu_rad_s: float,
    internal_rates_rad_s: Iterable[float] | Mapping[str, float],
    *,
    allow_fast_cooling: bool = False,
    what: str = "the rate equation",
    maximum: float = ADIABATIC_MAX,
) -> None:
    """W << nu and W << every internal rate: the adiabatic elimination behind the Fock rate equation (Section 4.2.8 vii).

    ``internal_rates_rad_s`` is the build's ``level_rates_rad_s`` (or any iterable of decay rates); an empty set
    checks W << nu alone.
    """
    if allow_fast_cooling:
        return
    if nu_rad_s <= 0.0:
        raise ValidityError("the mode frequency is positive")
    rates = (
        list(internal_rates_rad_s.values())
        if isinstance(internal_rates_rad_s, Mapping)
        else list(internal_rates_rad_s)
    )
    w = abs(rate_per_s)
    if w > maximum * nu_rad_s:
        raise ValidityError(
            f"W/nu = {w / nu_rad_s:.4g} > {maximum:g}: {what} eliminates the motion adiabatically and needs W << nu "
            "(Section 4.2.8 vii); pass allow_fast_cooling=True to accept it"
        )
    positive = [r for r in rates if r > 0.0]
    if positive and w > maximum * min(positive):
        raise ValidityError(
            f"W/Gamma_min = {w / min(positive):.4g} > {maximum:g}: {what} eliminates the internal state adiabatically "
            "and needs W << every internal rate (Section 4.2.8 vii); pass allow_fast_cooling=True to accept it"
        )


def assert_resolved(
    gamma_prime_rad_s: float,
    gamma_coherence_rad_s: float,
    nu_rad_s: float,
    *,
    allow_unresolved: bool = False,
    what: str = "the effective two-level system",
    maximum: float = RESOLVED_MAX,
) -> None:
    """Gamma' << nu and gamma' << nu separately: resolved sidebands after the adiabatic elimination (Section 4.2.8 v, vii)."""
    if allow_unresolved:
        return
    if nu_rad_s <= 0.0:
        raise ValidityError("the mode frequency is positive")
    worst = max(gamma_prime_rad_s, gamma_coherence_rad_s)
    if worst >= maximum * nu_rad_s:
        raise ValidityError(
            f"max(Gamma', gamma')/nu = {worst / nu_rad_s:.4g} >= {maximum:g}: the sidebands of {what} are not "
            "resolved (Section 4.2.8 v, vii); pass allow_unresolved=True to accept it"
        )


def assert_resolved_linewidth(
    gamma_rad_s: float, nu_rad_s: float, *, allow_unresolved: bool = False, maximum: float = SMALL
) -> None:
    """Gamma << nu: the resolved-sideband condition of the bare linewidth (Section 4.2.8 vii; Stenholm's floor regime)."""
    if allow_unresolved:
        return
    if nu_rad_s <= 0.0:
        raise ValidityError("the mode frequency is positive")
    if gamma_rad_s > maximum * nu_rad_s:
        raise ValidityError(
            f"Gamma/nu = {gamma_rad_s / nu_rad_s:.4g} > {maximum:g}: the sideband floor is the Gamma << nu limit "
            "(RMP Eq. 116; Section 4.2.8 vii); pass allow_unresolved=True to evaluate it outside that regime"
        )


def assert_three_level_valid(
    gamma_10_rad_s: float,
    gamma_12_rad_s: float,
    omega_aux_rad_s: float,
    delta_aux_rad_s: float,
    *,
    allow_invalid: bool = False,
) -> None:
    """s_aux << 1 and |delta_aux| << Gamma_10 + Gamma_12: the three-level reduction of Marzoli Eqs. 12, 15 (Section 4.2.8 vii)."""
    if allow_invalid:
        return
    total = gamma_10_rad_s + gamma_12_rad_s
    if total <= 0.0:
        raise ValidityError("the fast level's decay rates are positive")
    s_aux = (omega_aux_rad_s**2 / 2.0) / (delta_aux_rad_s**2 + (total / 2.0) ** 2)
    if not (s_aux < SMALL and abs(delta_aux_rad_s) < SMALL * total):
        raise ValidityError(
            f"s_aux = {s_aux:.4g} and |delta_aux|/(Gamma_10 + Gamma_12) = {abs(delta_aux_rad_s) / total:.4g}: the "
            "three-level reduction needs s_aux << 1 and |delta_aux| << Gamma_10 + Gamma_12 (Section 4.2.8 vii); "
            "pass allow_invalid=True to accept the reduction error"
        )


def recoil_frequency_rad_s(k_rad_per_m: float, mass_kg: float) -> float:
    """omega_R = hbar k^2/(2 m) = E_R/hbar: the single-photon recoil frequency (Section 4.2.1)."""
    return free_recoil_energy_j(k_rad_per_m, mass_kg) / HBAR_J_S


def assert_doppler_recoil_limit(
    gamma_rad_s: float,
    k_rad_per_m: float,
    mass_kg: float,
    *,
    allow_recoil_limited: bool = False,
    maximum: float = 1.0,
) -> None:
    """Gamma > omega_R for Doppler cooling (Section 4.2.8 vii): below it the single-photon recoil exceeds the linewidth
    and the Doppler limit is replaced by the recoil limit."""
    if allow_recoil_limited:
        return
    omega_r = recoil_frequency_rad_s(k_rad_per_m, mass_kg)
    if gamma_rad_s <= maximum * omega_r:
        raise ValidityError(
            f"Gamma/omega_R = {gamma_rad_s / omega_r:.4g} <= {maximum:g}: Doppler cooling needs Gamma > omega_R "
            f"(omega_R/2pi = {omega_r / (2.0 * math.pi):.4g} Hz; Section 4.2.8 vii); the line is recoil limited, so "
            "pass allow_recoil_limited=True to report the Doppler forms anyway"
        )


__all__ = [
    "ADIABATIC_MAX",
    "LAMB_DICKE_MAX",
    "RESOLVED_MAX",
    "SMALL",
    "ValidityError",
    "assert_adiabatic",
    "assert_doppler_recoil_limit",
    "assert_lamb_dicke",
    "assert_resolved",
    "assert_resolved_linewidth",
    "assert_three_level_valid",
    "assert_weak_drive",
]
