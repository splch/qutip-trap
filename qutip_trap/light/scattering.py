"""Per-pi-pulse scattering probabilities with the D-level branch reported separately (PLAN.md Section 4.5.5).

eps_S = P_Raman and eps_D = f P_total per carrier pi pulse (t_pi = 1/(2 f_Rabi)), f the fraction of the driving beams'
excited-state decay that lands in a D level; the rates are the Kramers-Heisenberg sums of ``scattering_budget``.
"""

from __future__ import annotations

from collections.abc import Sequence

from qutip_trap.device.model import Device
from qutip_trap.light.raman import scattering_budget
from qutip_trap.units import C_M_PER_S, TWO_PI


def d_level_branching(device: Device, ion: int, beam_indices: Sequence[int]) -> float:
    """f: the tabulated and untabulated D-level branching of the excited levels the beams reach, weighted by their
    1/Delta_e^2 excitation at the first beam's frequency; zero for a species with no D manifold."""
    species = device.crystal.species[ion]
    by_upper: dict[str, float] = {}
    for tr in species.transitions:
        if tr.lower.startswith("D"):
            by_upper[tr.upper] = by_upper.get(tr.upper, 0.0) + tr.branching
    for lv in species.levels:
        for what, fraction in lv.untabulated_branching:
            if "D" in what and lv.name not in ("S1/2",):
                by_upper[lv.name] = by_upper.get(lv.name, 0.0) + fraction
    if not by_upper:
        return 0.0
    if not beam_indices:
        return float(max(by_upper.values()))
    omega_l = TWO_PI * C_M_PER_S / device.beams[beam_indices[0]].wavelength_m
    num = den = 0.0
    for tr in species.transitions:
        if tr.upper not in by_upper:
            continue
        detuning = TWO_PI * tr.frequency_hz - omega_l
        if detuning == 0.0:
            continue
        weight = 1.0 / detuning**2
        num += weight * by_upper[tr.upper]
        den += weight
    return float(num / den) if den > 0.0 else float(max(by_upper.values()))


def epsilon_s_and_d(
    device: Device, ion: int, beam_indices: Sequence[int], rabi_hz: float
) -> dict[str, float]:
    """eps_S = P_Raman and eps_D = f P_total per pi pulse, averaged over the qubit states, side by side.

    The elastic rate here excludes the D-level Raman channel (it counts as leakage, hence in eps_S), so
    ``ozeri_rayleigh_overstatement`` is eps_D: the amount by which Ozeri's P_Rayleigh closed form exceeds this one.
    """
    from qutip_trap.validation.noise_closed_forms import epsilon_d_from_p_total

    if rabi_hz <= 0.0:
        raise ValueError("rabi_hz must be positive")
    budget = scattering_budget(device, ion, beam_indices)
    t_pi = 0.5 / rabi_hz
    states = list(budget.rayleigh_per_s)
    raman = {a: (budget.raman_spin_flip_per_s[a] + budget.leakage_per_s[a]) * t_pi for a in states}
    rayleigh = {a: budget.rayleigh_per_s[a] * t_pi for a in states}
    p_total = sum(
        (budget.raman_spin_flip_per_s[a] + budget.leakage_per_s[a] + budget.rayleigh_per_s[a]) * t_pi
        for a in states
    ) / len(states)
    f_d = d_level_branching(device, ion, beam_indices)
    eps_d = epsilon_d_from_p_total(f_d, p_total)
    return {
        "epsilon_S": sum(raman.values()) / len(states),
        "epsilon_D": eps_d,
        "f_D": f_d,
        "P_total": p_total,
        "P_Rayleigh": sum(rayleigh.values()) / len(states),
        "ozeri_rayleigh_overstatement": eps_d,
    }
