"""``Device.specs()``: the derived quantities as a readable report (docs/api_implementation_plan.md 2.5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("qubits", ("qubit_freq_hz", "dnu_dB_hz_per_g", "d2nu_dB2_hz_per_g2", "field_gauss")),
    (
        "trap",
        (
            "secular_hz",
            "mathieu_q",
            "mathieu_beta",
            "C0",
            "ion_height_m",
            "trap_depth_ev",
            "micromotion_beta",
        ),
    ),
    ("modes", ("mode_hz", "heating_rate_per_s")),
    ("drives", ("rabi_hz", "stark_hz", "eta", "crosstalk")),
    ("detection", ("R_bright_per_s", "R_dark_pumping_per_s")),
)
"""The families ``specs`` groups the ``derived()`` keys by, in the order they are printed; a key of no family is listed last."""


def _family(key: str) -> str:
    stem = key.split("[", 1)[0]
    for name, stems in FAMILIES:
        if stem in stems:
            return name
    return "other"


def render_specs(device: Device) -> str:
    derived = device.derived()
    crystal = device.crystal
    species = ", ".join(sorted({sp.name for sp in crystal.species}))
    lines = [
        f"device {device.hash()[:12]}",
        f"  crystal: {crystal.n_ions} ion(s) of {species}, {len(crystal.modes)} mode(s)",
        f"  beams: {len(device.beams)} ({', '.join(f'{b.wavelength_m * 1e9:.1f} nm' for b in device.beams)})",
        f"  field: {device.field.B_gauss:.6g} G along {tuple(round(float(x), 4) for x in device.field.direction)}",
        f"  detector: {device.detector.kind}, efficiency {device.detector.efficiency:.4g}, window {device.detector.window_s:.4g} s",
        f"  electronics: {device.hardware.dds_phase_bits}-bit phase, {device.hardware.dds_amplitude_bits}-bit amplitude words, "
        f"dead time {device.hardware.dead_time_s:.4g} s",
    ]
    groups: dict[str, list[str]] = {}
    for key in derived.values:
        groups.setdefault(_family(key), []).append(key)
    order = [name for name, _stems in FAMILIES] + ["other"]
    for family in order:
        keys = groups.get(family)
        if not keys:
            continue
        lines.append("")
        lines.append(family)
        for key in sorted(keys):
            lines.append(f"  {key} = {derived.values[key]:.6g}  [{derived.provenance[key]}]")
    channels = device.noise.summary(device)
    lines.append("")
    lines.append("noise channels" if channels else "noise channels: none (the noise model is quiet)")
    for key, (value, unit) in channels.items():
        lines.append(f"  {key} = {value:.6g} {unit}")
    if derived.notes:
        lines.append("")
        lines.append("notes")
        lines.extend(f"  {n}" for n in derived.notes)
    return "\n".join(lines)


__all__ = ["FAMILIES", "render_specs"]
