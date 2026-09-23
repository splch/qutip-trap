"""Which beams of a device play which role, decided by the physics: a beam within ``RESONANT_WINDOW`` of a tabulated E1
line is resonant (cooling, detection or repump light), every other beam is far-detuned gate light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qutip_trap.species.model import parse_transition_label

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

RESONANT_WINDOW = 2e-3
"""Relative wavelength window within which a beam counts as resonant with a tabulated line."""


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
    """The beams near the species' cycling line or its tabulated repump lines (the light that makes the ion fluoresce);
    raises when there is none."""
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


def infer_detection_beam(device: Device, *, window: float = RESONANT_WINDOW) -> int | None:
    """The beam nearest in relative wavelength to the first species' cycling line within ``window``, or None."""
    species = device.crystal.species[0]
    lam = species.transition(species.cycling).wavelength_vac_m
    best: tuple[float, int] | None = None
    for k, beam in enumerate(device.beams):
        rel = abs(beam.wavelength_m - lam) / lam
        if rel < window and (best is None or rel < best[0]):
            best = (rel, k)
    return None if best is None else best[1]
