"""Validity conditions of the cooling closed forms and rate models, asserted at run time.

Each ``assert_*`` raises :class:`ValidityError` (a ``ValueError``) with the measured ratio unless its escape keyword is
passed; ``<<`` means below ``SMALL`` (one tenth) unless a constant below says otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from qutip_trap.light.bloch import WEAK_DRIVE_MAX
from qutip_trap.light.recoil import free_recoil_energy_j
from qutip_trap.units import HBAR_J_S

SMALL = 0.1
"""The "<<" threshold: a ratio below one tenth (the same number as ``bloch.WEAK_DRIVE_MAX``)."""

LAMB_DICKE_MAX = 0.25
"""eta^2 (2 nbar + 1) above which the Lamb-Dicke expansion is refused (the carrier element then errs by over 12.5 %)."""

ADIABATIC_MAX = SMALL
"""W/nu and W/Gamma above which the adiabatic elimination behind the rate equation is refused."""

RESOLVED_MAX = 0.2
"""max(Gamma', gamma')/nu above which the sidebands are not resolved (the bound ``EffectiveTwoLevel.resolved`` uses)."""


class ValidityError(ValueError):
    """A validity condition of a closed form or rate model is violated."""


def assert_weak_drive(
    omega_rad_s: float,
    gamma_rad_s: float,
    *,
    allow_saturation: bool = False,
    what: str = "this closed form",
    maximum: float = WEAK_DRIVE_MAX,
) -> None:
    """Omega << Gamma (Omega/Gamma = sqrt(s/2)) for every closed form; the saturation error is of order (Omega/Gamma)^2."""
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
    """eta^2 (2 nbar + 1) << 1 per transition, the emitted photon included."""
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
    """W << nu and W << every rate in ``internal_rates_rad_s``: the adiabatic elimination behind the Fock rate equation."""
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
    """Gamma' << nu and gamma' << nu separately: resolved sidebands after the adiabatic elimination."""
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
    """Gamma << nu: the resolved-sideband condition of the bare linewidth (the sideband-floor regime)."""
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
    """s_aux << 1 and |delta_aux| << Gamma_10 + Gamma_12: the three-level reduction of Marzoli et al. 1994 Eqs. 12, 15."""
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
    """omega_R = hbar k^2/(2 m) = E_R/hbar: the single-photon recoil frequency."""
    return free_recoil_energy_j(k_rad_per_m, mass_kg) / HBAR_J_S


def assert_doppler_recoil_limit(
    gamma_rad_s: float,
    k_rad_per_m: float,
    mass_kg: float,
    *,
    allow_recoil_limited: bool = False,
    maximum: float = 1.0,
) -> None:
    """Gamma > omega_R for Doppler cooling: below it the recoil limit replaces the Doppler limit."""
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
