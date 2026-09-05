"""Differential light shifts for light-shift gates and error terms (PLAN.md Sections 3.2, 4.3.2, 4.5.4; milestone M2).

Each beam j shifts each qubit level by delta(g, j) = sum_e |Omega^{(j)}_{eg}|^2/(4 Delta_e^{(j)}) (Wineland 2003 Eq. 2.10 in
the plan's (hbar Omega/2) normalization, Section 4.5.4); the differential shift delta_St = delta(up) - delta(down)
enters the builder as (delta_St/2) sigma_z, proportional to the instantaneous intensity (Section 4.3.2). The closed
forms (Wineland's eight-term 9Be+ shift, the clock-qubit form with its three-index detuning, Ozeri 2005 Eq. 3) live
in ``qutip_trap.validation.atomic_closed_forms``; this module is the device-level entry point.
"""

from __future__ import annotations

from collections.abc import Sequence

from qutip_trap.device.model import Device
from qutip_trap.light.beams import Beam
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import TWO_PI


def level_shifts_hz(
    structure: AtomicStructure,
    labels: Sequence[str],
    beams: Sequence[Beam],
    position_m: Sequence[float] | None = None,
) -> dict[str, float]:
    """delta_g/2pi per state label under the beams at ``position_m`` (red light lowers a level)."""
    return {
        lab: structure.light_shift_rad_s(structure.state(lab), beams, position_m) / TWO_PI for lab in labels
    }


def differential_shift_hz(
    structure: AtomicStructure,
    lower: str,
    upper: str,
    beams: Sequence[Beam],
    position_m: Sequence[float] | None = None,
) -> float:
    """delta(upper) - delta(lower), the transition's Stark shift in Hz."""
    s = level_shifts_hz(structure, (lower, upper), beams, position_m)
    return s[upper] - s[lower]


def device_stark_shift_hz(device: Device, ion: int, beam_indices: Sequence[int]) -> float:
    species = device.crystal.species[ion]
    st = AtomicStructure(species, device.field.B_gauss, device.field.direction)
    lower, upper = species.qubit
    return differential_shift_hz(
        st,
        lower,
        upper,
        [device.beams[b] for b in beam_indices],
        [float(x) for x in device.crystal.positions_m[ion]],
    )


def stark_phase_rad(shift_hz: float, duration_s: float) -> float:
    """The phase 2 pi delta_St t a square pulse imprints on the qubit (the transition shift over the pulse)."""
    return TWO_PI * shift_hz * duration_s


def intensity_scaled(shift_peak_hz: float, envelope_fraction: float) -> float:
    """A Stark shift follows the INTENSITY: a fraction f of the peak field amplitude gives f^2 of the peak shift."""
    return shift_peak_hz * envelope_fraction**2


__all__ = [
    "device_stark_shift_hz",
    "differential_shift_hz",
    "intensity_scaled",
    "level_shifts_hz",
    "stark_phase_rad",
]
