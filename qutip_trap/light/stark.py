"""Differential light shifts: beam j shifts level g by delta(g, j) = sum_e |Omega^{(j)}_{eg}|^2/(4 Delta_e^{(j)})
(Wineland 2003 Eq. 2.10), entering the builder as (delta_St/2) sigma_z with delta_St = delta(up) - delta(down).
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


def intensity_scaled(shift_peak_hz: float, envelope_fraction: float, power: int = 2) -> float:
    """A fraction f of the peak played Rabi frequency gives f^power of the peak shift: ``power`` is 1 for a two-photon
    drive and 2 (the default) for a single-photon optical or microwave one."""
    return shift_peak_hz * envelope_fraction**power
