"""Raman and Rayleigh scattering rates and amplitudes of a drive, differential-Rayleigh dephasing and leakage branching
from the angular algebra of species/ (PLAN.md Sections 3.2, 4.3.2, 4.5.5, 6.5; milestone M2 for the rates).

The rates are the Kramers-Heisenberg sums of the atomic layer (``AtomicStructure.scattering_rates``, sqrt(Gamma_e)
inside the coherent sum over intermediate states) at the ion's position under each beam, incoherent between beams of
different frequencies. At d = 2 they are an ESTIMATED per-pulse error (Section 4.3.2); the Lindblad operators on the
extended internal space (leakage, recoil) are milestone M7's ``NoiseModel.channels``. Ozeri's closed forms
(P_total = (pi gamma/omega_f)(2 Delta^2 + (Delta - omega_f)^2)/|Delta(Delta - omega_f)|, the minimum at
Delta = (sqrt2 - 1) omega_f, epsilon_S = P_Raman) are in ``qutip_trap.validation.atomic_closed_forms`` and are tested
against these sums in M0a's suite.
"""

from __future__ import annotations

from collections.abc import Sequence

from qutip_trap.device.model import Device
from qutip_trap.light.raman import ScatteringBudget, scattering_budget


def per_pulse_scattering_error(
    device: Device, ion: int, beam_indices: Sequence[int], duration_s: float
) -> float:
    """The d = 2 per-pulse Raman (spin-flip plus leakage) probability, averaged over the qubit states (Section 4.3.2)."""
    return scattering_budget(device, ion, beam_indices).per_pulse_error(duration_s)


def photons_per_pi_pulse(budget: ScatteringBudget, rabi_hz: float) -> dict[str, float]:
    """Scattering probabilities during a carrier pi pulse (t_pi = 1/(2 f_Rabi)) per qubit state: total, Raman, Rayleigh."""
    if rabi_hz <= 0.0:
        raise ValueError("rabi_hz must be positive")
    t_pi = 0.5 / rabi_hz
    out: dict[str, float] = {}
    for a in budget.rayleigh_per_s:
        raman = budget.raman_spin_flip_per_s[a] + budget.leakage_per_s[a]
        out[f"P_total[{a}]"] = (raman + budget.rayleigh_per_s[a]) * t_pi
        out[f"P_raman[{a}]"] = raman * t_pi
        out[f"P_rayleigh[{a}]"] = budget.rayleigh_per_s[a] * t_pi
    return out


def d_level_branching(device: Device, ion: int, beam_indices: Sequence[int]) -> float:
    """f, the fraction of the driving beams' excited-state decay that lands in a D level (Section 6.5's eps_D = f P_total).

    Averaged over the intermediate levels the beams reach, weighted by their 1/Delta_e^2 excitation: an excited level's
    D branching is the sum of its tabulated ``Transition.branching`` into levels whose name starts with "D", plus any
    ``untabulated_branching`` channel whose description names a D level. Zero for a species with no D manifold
    (9Be+, 25Mg+, 111Cd+), which is why Ozeri's Table II rows for those species need no eps_D.
    """
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
    # weight the reachable uppers by 1/Delta_e^2 (the far-detuned excitation), using the first beam's frequency
    beam = device.beams[beam_indices[0]] if beam_indices else None
    if beam is None:
        return float(max(by_upper.values()))
    from qutip_trap.units import C_M_PER_S, TWO_PI

    omega_l = TWO_PI * C_M_PER_S / beam.wavelength_m
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
    """eps_S = P_Raman and eps_D = f P_total per pi pulse, reported side by side (Section 9.7 row "Scattering").

    Section 4.5.5 records that Ozeri's own P_Rayleigh "silently includes the D-level Raman channel for Ca+, Sr+, Ba+ and
    Yb+, overstating the elastic rate by f P_total", so eps_D is a SEPARATE line. The module's own ``P_Rayleigh`` comes
    from the elastic amplitude sum and does NOT contain that channel (the D-level finals are counted in ``leakage_per_s``
    and hence in eps_S), so ``ozeri_rayleigh_overstatement`` is reported as the amount by which OZERI'S CLOSED FORM
    exceeds this one, not as a correction to subtract from it. ``P_total`` is the simulated total (Raman plus leakage
    plus Rayleigh) averaged over the qubit states; ``validation/noise_closed_forms.epsilon_d_from_p_total`` is the closed
    form the tests compare eps_D with.
    """
    from qutip_trap.validation.noise_closed_forms import epsilon_d_from_p_total

    budget = scattering_budget(device, ion, beam_indices)
    probs = photons_per_pi_pulse(budget, rabi_hz)
    states = list(budget.rayleigh_per_s)
    p_total = sum(probs[f"P_total[{a}]"] for a in states) / len(states)
    p_raman = sum(probs[f"P_raman[{a}]"] for a in states) / len(states)
    p_rayleigh = sum(probs[f"P_rayleigh[{a}]"] for a in states) / len(states)
    f_d = d_level_branching(device, ion, beam_indices)
    eps_d = epsilon_d_from_p_total(f_d, p_total)
    return {
        "epsilon_S": p_raman,
        "epsilon_D": eps_d,
        "f_D": f_d,
        "P_total": p_total,
        "P_Rayleigh": p_rayleigh,
        "ozeri_rayleigh_overstatement": eps_d,
    }
