"""Collapse operators from noise parameters (PLAN.md Sections 5.7, 6; Appendix E; milestone M7 for the full set).

Convention rows that bind here (Section 13): heating rates Gamma(N_bar + 1) on a and Gamma N_bar on a^dagger, both
Gamma_h in the electric-field-noise limit N_bar -> infinity (Section 4.1.5), with the quoted heating rate d<n>/dt at
n = 0; qubit dephasing L = sqrt(gamma_phi/2) sigma_z so that the coherence decays at gamma_phi = 1/T2; motional
dephasing L = a^dagger a sqrt(2/tau); the Rayleigh operator (1/2) sqrt(Gamma_el) sigma_z, whose dissipator prefactor
is Gamma_el/4 while the coherence decays at Gamma_el/2, both correct; recoil-resolved emission channels one operator
per (decay channel, recoil class), never summed coherently. M2 supplies the constructors the engine needs for the
motional and qubit channels; M7 assembles them from the device's spectra (``NoiseModel.channels``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import qutip as qt

from qutip_trap.trap.heating import (
    heating_rate_quanta_per_s,
    single_sided_from_two_sided,
    thermal_collapse_rates,
)
from qutip_trap.units import TWO_PI, hz_from_rad_s

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.hilbert.space import HilbertSpace


@dataclass(frozen=True)
class CollapseOp:
    op: qt.Qobj
    rate_hz: float
    """The rate the operator carries under its root, as an ordinary frequency for reporting."""
    channel: str
    ion: int | None
    mode: int | None


def heating_channels(
    space: HilbertSpace, rates_quanta_per_s: Mapping[int, float], *, n_bar_bath: float | None = None
) -> tuple[CollapseOp, ...]:
    """sqrt(Gamma(N_bar + 1)) a_m and sqrt(Gamma N_bar) a_m^dagger per carried mode from the quoted heating rates."""
    out: list[CollapseOp] = []
    for mode, ndot in rates_quanta_per_s.items():
        if space.mode_class(mode) == "frozen" or ndot <= 0.0:
            continue
        down, up = thermal_collapse_rates(float(ndot), n_bar_bath)
        a = space.annihilation(mode)
        out.append(CollapseOp(math.sqrt(down) * a, float(hz_from_rad_s(down)), "heating_down", None, mode))  # type: ignore[arg-type]
        out.append(CollapseOp(math.sqrt(up) * a.dag(), float(hz_from_rad_s(up)), "heating_up", None, mode))  # type: ignore[arg-type]
    return tuple(out)


def device_heating_rates(device: Device, space: HilbertSpace) -> dict[int, float]:
    """Gamma_h per carried mode from the device's two-sided S_E at the mode frequency (single-ion rate, Section 4.1.5).

    The correlated multi-ion generalization (Kielpinski, Brownnutt) needs the correlation length and lives in
    trap/heating.py; this is the per-mode single-ion rate at the MODE's frequency for one-ion spaces.
    """
    import numpy as np

    spec = device.noise.S_E
    out: dict[int, float] = {}
    for m, mode in enumerate(device.crystal.modes):
        if space.mode_class(m) == "frozen":
            continue
        w = mode.omega_rad_s
        s_two = float(np.interp(w, spec.omega_rad_s, spec.S, left=0.0, right=0.0))
        s_one = float(single_sided_from_two_sided(s_two))
        ion = int(np.argmax(np.abs(mode.eigenvector)))
        mass = float(device.crystal.masses_kg[ion])
        out[m] = heating_rate_quanta_per_s(s_one, mass, w)
    return out


def qubit_dephasing_channels(
    space: HilbertSpace, gamma_phi_hz: Mapping[int, float]
) -> tuple[CollapseOp, ...]:
    """L = sqrt(gamma_phi/2) sigma_z per ion, gamma_phi = 1/T2 the white dephasing rate (Section 13), gamma in s^-1."""
    out: list[CollapseOp] = []
    for ion, g in gamma_phi_hz.items():
        if g <= 0.0:
            continue
        out.append(
            CollapseOp(
                math.sqrt(g / 2.0) * space.sigma_z(ion), float(g / TWO_PI), "qubit_dephasing", ion, None
            )
        )
    return tuple(out)


def motional_dephasing_channels(space: HilbertSpace, tau_s: Mapping[int, float]) -> tuple[CollapseOp, ...]:
    """L = a^dagger a sqrt(2/tau) per mode (Ballance 2016; Section 13)."""
    out: list[CollapseOp] = []
    for mode, tau in tau_s.items():
        if tau <= 0.0 or space.mode_class(mode) == "frozen":
            continue
        out.append(
            CollapseOp(
                math.sqrt(2.0 / tau) * space.number(mode),
                float((2.0 / tau) / TWO_PI),
                "motional_dephasing",
                None,
                mode,
            )
        )
    return tuple(out)


def rayleigh_dephasing_channels(
    space: HilbertSpace, gamma_el_per_s: Mapping[int, float]
) -> tuple[CollapseOp, ...]:
    """(1/2) sqrt(Gamma_el) sigma_z per ion (Uys et al. 2010; Section 13, "Rayleigh dephasing dissipator")."""
    out: list[CollapseOp] = []
    for ion, g in gamma_el_per_s.items():
        if g <= 0.0:
            continue
        out.append(
            CollapseOp(
                0.5 * math.sqrt(g) * space.sigma_z(ion), float(g / TWO_PI), "rayleigh_dephasing", ion, None
            )
        )
    return tuple(out)


__all__ = [
    "CollapseOp",
    "device_heating_rates",
    "heating_channels",
    "motional_dephasing_channels",
    "qubit_dephasing_channels",
    "rayleigh_dephasing_channels",
]
