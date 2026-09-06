"""Which beams of a device play which role (PLAN.md Sections 3.3, 7.3, 8.1; milestone M6).

A ``Device`` lists every beam the ions see (Section 3.1: device parameters in, everything else derived). The roles follow
from the physics, never from a label: a beam within ``RESONANT_WINDOW`` (relative wavelength) of a tabulated E1 line of the
crystal's species is a resonant beam (cooling, detection or repumping light: it scatters photons at the natural linewidth),
everything else is far-detuned gate light (a Raman pair, or a single beam on the optical qubit transition). The gate-drive
inference of the scheduler and the detection-beam selection of the readout both read these functions, so a device that
carries its detection beam next to its Raman pair schedules and reads out without any extra bookkeeping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qutip_trap.species.model import parse_transition_label

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

RESONANT_WINDOW = 2e-3
"""Relative wavelength window within which a beam counts as resonant with a tabulated line (the M5 detection rule)."""


def resonant_beams(device: Device, *, window: float = RESONANT_WINDOW) -> tuple[int, ...]:
    """Indices of the beams within ``window`` of any tabulated E1 transition of any species in the crystal."""
    out: list[int] = []
    for k, beam in enumerate(device.beams):
        for sp in device.crystal.species:
            if any(
                t.multipole == "E1"
                and abs(beam.wavelength_m - t.wavelength_vac_m) < window * t.wavelength_vac_m
                for t in sp.transitions
            ):
                out.append(k)
                break
    return tuple(out)


def gate_beams(device: Device, *, window: float = RESONANT_WINDOW) -> tuple[int, ...]:
    """The far-detuned beams: everything that is not resonant with a tabulated line."""
    res = set(resonant_beams(device, window=window))
    return tuple(k for k in range(len(device.beams)) if k not in res)


def detection_beams(device: Device, ion: int, *, window: float = RESONANT_WINDOW) -> tuple[int, ...]:
    """The beams near the species' cycling line and near its tabulated repump lines (the light that makes the ion fluoresce);
    a repump whose upper level the species table does not close (the 171Yb+ 935 nm line, M3a finding) is skipped."""
    species = device.crystal.species[ion]
    lower, upper = parse_transition_label(species.cycling)
    lines = [species.transition(species.cycling).wavelength_vac_m]
    for rep in species.repumps:
        if any(t.label == rep for t in species.transitions):
            lines.append(species.transition(rep).wavelength_vac_m)
    out: list[int] = []
    for k, beam in enumerate(device.beams):
        if any(abs(beam.wavelength_m - lam) < window * lam for lam in lines):
            out.append(k)
    if not out:
        raise ValueError(f"no beam of the device is near the {species.name} cycling line {lower}-{upper}")
    return tuple(out)


__all__ = ["RESONANT_WINDOW", "detection_beams", "gate_beams", "resonant_beams"]
