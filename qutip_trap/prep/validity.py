"""The validity conditions of the cooling models as run-time assertions (PLAN.md Section 4.2.8).

Each ``assert_*`` raises :class:`ValidityError` with the measured ratio in the message; the escape keyword lets a caller
that deliberately probes the boundary say so. "<<" is "below one tenth" except where a constant says otherwise. Omega is
the (hbar Omega/2)-convention Rabi frequency and Gamma the full angular linewidth, so Omega/Gamma = sqrt(s/2).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from qutip_trap.light.recoil import free_recoil_energy_j
from qutip_trap.units import HBAR_J_S

SMALL = 0.1
"""The plan's "<<": a ratio below one tenth."""

LAMB_DICKE_MAX = 0.25
"""eta^2 (2 nbar + 1) above which the Lamb-Dicke expansion is refused: eta sqrt(2 nbar + 1) < 1/2, where the exact
carrier element e^{-eta^2/2} L_n(eta^2) departs from its Lamb-Dicke value by under 12.5 %."""


class ValidityError(ValueError):
    """A validity condition of a closed form or rate model is violated."""


def assert_weak_drive(
    omega_rad_s: float, gamma_rad_s: float, *, allow_saturation: bool = False, what: str = "this closed form"
) -> None:
    """Omega << Gamma; the saturation error is of order (Omega/Gamma)^2."""
    if allow_saturation:
        return
    if gamma_rad_s <= 0.0:
        raise ValidityError("the linewidth is positive")
    ratio = abs(omega_rad_s) / gamma_rad_s
    if ratio > SMALL:
        raise ValidityError(
            f"Omega/Gamma = {ratio:.4g} > {SMALL:g}: {what} needs Omega << Gamma; use the level-C solve or the "
            "spectrum path, or pass allow_saturation=True to accept the saturation error"
        )


def assert_lamb_dicke(
    eta: float, nbar: float, *, allow_strong_coupling: bool = False, what: str = "the rate model"
) -> None:
    """eta^2 (2 nbar + 1) << 1 per transition, the emitted photon included."""
    if allow_strong_coupling:
        return
    if nbar < 0.0:
        raise ValidityError("nbar is non-negative")
    value = eta**2 * (2.0 * nbar + 1.0)
    if value > LAMB_DICKE_MAX:
        raise ValidityError(
            f"eta^2 (2 nbar + 1) = {value:.4g} > {LAMB_DICKE_MAX:g} (eta = {eta:.4g}, nbar = {nbar:.4g}): {what} is a "
            "Lamb-Dicke expansion; use the level-C solve with the exact displacement operators, or pass "
            "allow_strong_coupling=True"
        )


def assert_adiabatic(
    rate_per_s: float,
    nu_rad_s: float,
    internal_rates_rad_s: Mapping[str, float],
    *,
    what: str = "the rate equation",
) -> None:
    """W << nu and W << every internal decay rate: the adiabatic elimination behind the Fock rate equation."""
    if nu_rad_s <= 0.0:
        raise ValidityError("the mode frequency is positive")
    w = abs(rate_per_s)
    if w > SMALL * nu_rad_s:
        raise ValidityError(
            f"W/nu = {w / nu_rad_s:.4g} > {SMALL:g}: {what} eliminates the motion adiabatically and needs W << nu"
        )
    positive = [r for r in internal_rates_rad_s.values() if r > 0.0]
    if positive and w > SMALL * min(positive):
        raise ValidityError(
            f"W/Gamma_min = {w / min(positive):.4g} > {SMALL:g}: {what} eliminates the internal state adiabatically "
            "and needs W << every internal rate"
        )


def recoil_frequency_rad_s(k_rad_per_m: float, mass_kg: float) -> float:
    """omega_R = hbar k^2/(2 m) = E_R/hbar: the single-photon recoil frequency."""
    return free_recoil_energy_j(k_rad_per_m, mass_kg) / HBAR_J_S


def assert_doppler_recoil_limit(
    gamma_rad_s: float, k_rad_per_m: float, mass_kg: float, *, allow_recoil_limited: bool = False
) -> None:
    """Gamma > omega_R for Doppler cooling: below it the single-photon recoil exceeds the linewidth and the Doppler
    limit is replaced by the recoil limit."""
    if allow_recoil_limited:
        return
    omega_r = recoil_frequency_rad_s(k_rad_per_m, mass_kg)
    if gamma_rad_s <= omega_r:
        raise ValidityError(
            f"Gamma/omega_R = {gamma_rad_s / omega_r:.4g} <= 1: Doppler cooling needs Gamma > omega_R "
            f"(omega_R/2pi = {omega_r / (2.0 * math.pi):.4g} Hz); the line is recoil limited, so pass "
            "allow_recoil_limited=True to report the Doppler forms anyway"
        )
