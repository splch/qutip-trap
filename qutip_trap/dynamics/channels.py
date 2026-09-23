"""Lindblad collapse operators for heating, dephasing, Rayleigh scattering and intensity noise; rates in s^-1."""

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

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.hilbert.space import HilbertSpace


@dataclass(frozen=True)
class CollapseOp:
    """One Lindblad collapse operator (rate inside the root; a ``QobjEvo`` when time dependent), its rate for reporting,
    its channel name and the ion or mode it acts on (None when neither applies)."""

    op: qt.Qobj | qt.QobjEvo
    rate_hz: float
    """The rate under the root in s^-1 (an ordinary rate, not divided by 2 pi)."""
    channel: str
    ion: int | None
    mode: int | None

    @property
    def time_dependent(self) -> bool:
        return isinstance(self.op, qt.QobjEvo)


def heating_channels(
    space: HilbertSpace, rates_quanta_per_s: Mapping[int, float], *, n_bar_bath: float | None = None
) -> tuple[CollapseOp, ...]:
    """sqrt(Gamma(N_bar + 1)) a_m and sqrt(Gamma N_bar) a_m^dagger per carried mode, Gamma N_bar the quoted heating rate;
    ``n_bar_bath=None`` is the N_bar -> infinity limit, both at Gamma_h."""
    out: list[CollapseOp] = []
    for mode, ndot in rates_quanta_per_s.items():
        if space.mode_class(mode) == "frozen" or ndot <= 0.0:
            continue
        down, up = thermal_collapse_rates(float(ndot), n_bar_bath)
        a = space.annihilation(mode)
        out.append(CollapseOp(math.sqrt(down) * a, float(down), "heating_down", None, mode))
        out.append(CollapseOp(math.sqrt(up) * a.dag(), float(up), "heating_up", None, mode))
    return tuple(out)


def device_heating_rates(device: Device, space: HilbertSpace) -> dict[int, float]:
    """Gamma_h in quanta/s per carried mode: the single-ion rate from the device's two-sided S_E at the mode frequency
    (the correlated multi-ion rate is in trap/heating.py)."""
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
    """L = sqrt(gamma_phi/2) sigma_z per ion, so the coherence decays at gamma_phi = 1/T2 (s^-1)."""
    out: list[CollapseOp] = []
    for ion, g in gamma_phi_hz.items():
        if g <= 0.0 or not space.has_ion(ion):
            continue  # a GATE_LOCAL space carries a subset of the ions
        out.append(
            CollapseOp(math.sqrt(g / 2.0) * space.sigma_z(ion), float(g), "qubit_dephasing", ion, None)
        )
    return tuple(out)


def motional_dephasing_channels(space: HilbertSpace, tau_s: Mapping[int, float]) -> tuple[CollapseOp, ...]:
    """L = a^dagger a sqrt(2/tau) per mode (Ballance 2016)."""
    out: list[CollapseOp] = []
    for mode, tau in tau_s.items():
        if tau <= 0.0 or space.mode_class(mode) == "frozen":
            continue
        out.append(
            CollapseOp(
                math.sqrt(2.0 / tau) * space.number(mode),
                float(2.0 / tau),
                "motional_dephasing",
                None,
                mode,
            )
        )
    return tuple(out)


def rayleigh_dephasing_channels(
    space: HilbertSpace, gamma_el_per_s: Mapping[int, float]
) -> tuple[CollapseOp, ...]:
    """(1/2) sqrt(Gamma_el) sigma_z per ion, so the coherence decays at Gamma_el/2 (Uys et al. 2010)."""
    out: list[CollapseOp] = []
    for ion, g in gamma_el_per_s.items():
        if g <= 0.0 or not space.has_ion(ion):
            continue
        out.append(
            CollapseOp(0.5 * math.sqrt(g) * space.sigma_z(ion), float(g), "rayleigh_dephasing", ion, None)
        )
    return tuple(out)


def intensity_noise_channels(
    drive_parts: Mapping[str, qt.QobjEvo], densities: Mapping[str, float]
) -> tuple[CollapseOp, ...]:
    """L = sqrt(D) H_drive(t) per pulse, rho_dot = -(D/2)[H_d, [H_d, rho]]: D the two-sided density of the fractional Rabi
    fluctuation in 1/(rad/s) (the intensity white level times 1 for a two-photon drive, 1/4 for a single-photon one)."""
    out: list[CollapseOp] = []
    for gate_id, part in drive_parts.items():
        dens = float(densities.get(gate_id, 0.0))
        if dens <= 0.0:
            continue
        out.append(CollapseOp(math.sqrt(dens) * part, float(dens), f"intensity_noise[{gate_id}]", None, None))
    return tuple(out)


def gamma_i_from_density(density_two_sided: float, omega_rad_s: float) -> float:
    """Gamma_I = D Omega^2/2: the zero-frequency intensity-noise rate from the fractional density D."""
    return 0.5 * density_two_sided * omega_rad_s**2
