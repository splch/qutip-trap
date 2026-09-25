"""Differential light shifts for light-shift gates and error terms (PLAN.md Sections 3.2, 4.3.2, 4.5.4; milestone M2).

Each beam j shifts each qubit level by delta(g, j) = sum_e |Omega^{(j)}_{eg}|^2/(4 Delta_e^{(j)}) (Wineland 2003 Eq. 2.10 in
the plan's (hbar Omega/2) normalization, Section 4.5.4); the differential shift delta_St = delta(up) - delta(down)
enters the builder as (delta_St/2) sigma_z, proportional to the instantaneous intensity (Section 4.3.2).

``qutip_trap.validation.atomic_closed_forms`` carries the closed forms this explicit sum is tested against: Wineland's
clock-qubit shift with its shared prefactor and bracket (Eqs. 2.17-2.18, hence its polarization INDEPENDENCE and the
exact photons-per-Stark-radian ratio gamma/omega_0), Ozeri's per-pi-pulse scattering probabilities and the far-detuned
photons-per-radian saturation 0.9579 gamma/Delta_hf. Wineland Eq. 2.11's EIGHT-TERM |2,2> <-> |1,1> differential shift
and its Eq. 2.12 limit are NOT there: PLAN.md:1885 lists Eqs. 2.5-2.9 and 2.11-2.12 as untranscribed, so the coefficient
sets that pair the eight scattering channels with their detunings at first power do not exist in the plan and are not
invented here (M2 audit E9; ledger anchor.m2.wineland_eight_term_stark). What IS asserted against the derived sum is the
QUALITATIVE claim the plan makes for that line: it has a polarization null, which the clock line does not
(``tests/test_m2_stark_null.py``). This module is the device-level entry point.
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
    """A fraction f of the peak PLAYED Rabi frequency gives f^power of the peak shift (Section 4.3.2).

    ``power`` is ``control.schedule.stark_scaling_power(kind)``: 1 for a two-photon drive, whose Rabi frequency is
    itself proportional to the intensity, and 2 for a single-photon optical or microwave drive, where Omega goes with
    the field and the shift with the intensity. The default 2 is the single-photon case; passing the kind's own power is
    what the live scheduler path does, and hard-coding 2 here for every kind was a silent factor of f (M2 audit E19).
    """
    return shift_peak_hz * envelope_fraction**power
