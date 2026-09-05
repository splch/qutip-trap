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


__all__ = ["per_pulse_scattering_error", "photons_per_pi_pulse"]
