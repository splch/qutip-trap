"""Metastable D-level channels beyond spontaneous decay, all off by default: blackbody M1 mixing between the D
levels, collisional quenching and j-mixing (cited coefficients in cm^3/s; a partner without one raises rather than
defaulting), and reshelving."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qutip_trap.units import C_M_PER_S, H_J_S, K_B_J_PER_K

if TYPE_CHECKING:
    from qutip_trap.species.model import Species

MBAR_TO_PA = 100.0
CM3_TO_M3 = 1e-6
COLLISION_PROCESSES = ("quench", "j_mix")
"""The two Knoop et al. 1995 processes; the table ids are ``<species>.D.collision_{quench,jmix}_<gas>_cm3_s``."""
_TABLE_PROCESS = {"quench": "quench", "j_mix": "jmix"}


def bose_occupation(nu_hz: float, temperature_k: float) -> float:
    """n_bar = 1/(exp(h nu/k_B T) - 1), the blackbody photon occupation at frequency ``nu_hz``."""
    if nu_hz <= 0.0 or temperature_k <= 0.0:
        raise ValueError("the frequency and the temperature must be positive")
    return 1.0 / math.expm1(H_J_S * nu_hz / (K_B_J_PER_K * temperature_k))


def bbr_mixing_rates_hz(
    a_s: float, nu_hz: float, temperature_k: float, g_upper: float, g_lower: float
) -> tuple[float, float]:
    """(downward A n_bar, upward (g_u/g_l) A n_bar) in s^-1 for a line of Einstein A ``a_s`` (s^-1) at ``nu_hz``
    (Kreuter et al. 2005 Eq. 1)."""
    if a_s < 0.0:
        raise ValueError("an Einstein A coefficient is non-negative")
    if g_upper <= 0.0 or g_lower <= 0.0:
        raise ValueError("level degeneracies are positive")
    down = a_s * bose_occupation(nu_hz, temperature_k)
    return down, (g_upper / g_lower) * down


def _degeneracy(level_name: str) -> float:
    """2J + 1 from a fine-structure level name such as ``"D5/2"``."""
    from qutip_trap.species.model import level_j

    return 2.0 * float(level_j(level_name)) + 1.0


@dataclass(frozen=True)
class MetastableChannels:
    """Blackbody D-D mixing, collisional quenching and j-mixing, and reshelving of a metastable level."""

    bbr_temperature_k: float | None = None
    """None disables blackbody mixing."""
    pressure_mbar: float | None = None
    """Background-gas total pressure; None disables collisions."""
    gas_fractions: dict[str, float] = field(default_factory=dict)
    """Partner -> fraction of the pressure, e.g. {"H2": 0.5, "N2": 0.5}; sums to 1 when collisions are enabled."""
    reshelving_rate_hz: float = 0.0
    """R in p_D_dot = -Gamma p_D + R(1 - p_D), in s^-1."""
    gas_temperature_k: float = 300.0
    """The gas temperature in n_s = p_s/(k_B T); the collision rates are density driven."""

    def __post_init__(self) -> None:
        if self.pressure_mbar is not None:
            if self.pressure_mbar < 0.0:
                raise ValueError("pressure_mbar must be non-negative")
            total = sum(self.gas_fractions.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"gas_fractions must sum to 1 when collisions are enabled, got {total}")
            if min(self.gas_fractions.values(), default=0.0) < 0.0:
                raise ValueError("gas fractions are non-negative")
        if self.bbr_temperature_k is not None and self.bbr_temperature_k <= 0.0:
            raise ValueError("bbr_temperature_k must be positive")
        if self.gas_temperature_k <= 0.0:
            raise ValueError("gas_temperature_k must be positive")
        if self.reshelving_rate_hz < 0.0:
            raise ValueError("reshelving_rate_hz must be non-negative")

    # ---- blackbody mixing --------------------------------------------------------------------------------

    def bbr_rate_hz(self, transition: str, species: Species) -> tuple[float, float]:
        """(downward A n_bar, upward (g_u/g_l) A n_bar) in s^-1 for the tabulated line ``transition`` ("D3/2-D5/2") of
        ``species``, A being its partial rate; (0, 0) when disabled."""
        if self.bbr_temperature_k is None:
            return 0.0, 0.0
        tr = species.transition(transition)
        nu_hz = C_M_PER_S / tr.wavelength_vac_m
        return bbr_mixing_rates_hz(
            tr.partial_rate_rad_s,
            nu_hz,
            self.bbr_temperature_k,
            _degeneracy(tr.upper),
            _degeneracy(tr.lower),
        )

    # ---- collisions ---------------------------------------------------------------------------------------

    def partner_densities_m3(self) -> dict[str, float]:
        """n_s = f_s p/(k_B T) per partner, in m^-3."""
        if self.pressure_mbar is None:
            return {}
        n_total = self.pressure_mbar * MBAR_TO_PA / (K_B_J_PER_K * self.gas_temperature_k)
        return {gas: frac * n_total for gas, frac in self.gas_fractions.items()}

    def collision_coefficients_cm3_s(self, species: Species) -> dict[str, dict[str, float]]:
        """{process: {gas: Gamma_s in cm^3/s}} from the species table; ``LookupError`` names every uncited pair."""
        from qutip_trap.species import MODULES

        module = MODULES.get(species.name)
        table = getattr(module, "TABLE", {}) if module is not None else {}
        out: dict[str, dict[str, float]] = {p: {} for p in COLLISION_PROCESSES}
        missing: list[str] = []
        for process in COLLISION_PROCESSES:
            for gas in self.gas_fractions:
                suffix = f".D.collision_{_TABLE_PROCESS[process]}_{gas}_cm3_s"
                hits = [c for cid, c in table.items() if cid.endswith(suffix)]
                if not hits:
                    missing.append(f"{species.name} {process} by {gas} (table id *{suffix})")
                    continue
                c = hits[0]
                if c.unit != "cm^3/s":
                    raise ValueError(f"{c.key}: collision coefficients are cited in cm^3/s, never as rates")
                out[process][gas] = c.value
        if missing:
            raise LookupError(
                "no cited collision coefficient for: "
                + "; ".join(missing)
                + " (Knoop et al. 1995 tabulates 40Ca+ "
                "with H2 and N2; the plan never defaults a missing constant)"
            )
        return out

    def collision_rates_hz(self, species: Species) -> dict[str, float]:
        """{"quench": R_q, "j_mix": R_j} per ion in s^-1 from R = sum_s Gamma_s p_s/(k_B T); zero when disabled."""
        if self.pressure_mbar is None:
            return {p: 0.0 for p in COLLISION_PROCESSES}
        densities = self.partner_densities_m3()
        coefficients = self.collision_coefficients_cm3_s(species)
        return {
            process: sum(coefficients[process][gas] * CM3_TO_M3 * n for gas, n in densities.items())
            for process in COLLISION_PROCESSES
        }

    # ---- reshelving and the shelf's total loss rate -------------------------------------------------------

    def reshelving_offset(self, tau_s: float) -> float:
        """R/(Gamma + R) with Gamma = 1/``tau_s``: the long-delay offset that reshelving adds to a decay fit."""
        if tau_s <= 0.0:
            raise ValueError("tau_s must be positive")
        gamma = 1.0 / tau_s
        return self.reshelving_rate_hz / (gamma + self.reshelving_rate_hz)

    def shelf_loss_rates_hz(self, species: Species, shelf: str) -> dict[str, float]:
        """Every rate (s^-1) that empties the metastable level ``shelf``, by channel: decay, quenching, j-mixing and
        blackbody transfer along each tabulated M1 line out of it."""
        lifetime = species.level(shelf).lifetime_s
        if lifetime is None:
            raise ValueError(f"{species.name} {shelf}: no tabulated lifetime")
        rates: dict[str, float] = {"decay": 1.0 / lifetime}
        for process, rate in self.collision_rates_hz(species).items():
            rates[process] = rate
        if self.bbr_temperature_k is not None:
            for tr in species.transitions:
                if tr.multipole != "M1" or shelf not in (tr.lower, tr.upper):
                    continue
                down, up = self.bbr_rate_hz(tr.label, species)
                rates[f"bbr:{tr.label}"] = down if tr.upper == shelf else up
        return rates

    def effective_shelf_lifetime_s(self, species: Species, shelf: str) -> float:
        """1/(sum of :meth:`shelf_loss_rates_hz`): the shelf's lifetime in s with the channels on."""
        return 1.0 / sum(self.shelf_loss_rates_hz(species, shelf).values())
