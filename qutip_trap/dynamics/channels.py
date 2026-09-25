"""Collapse operators from noise parameters (PLAN.md Section 6).

Conventions: heating sqrt(Gamma_h) a and sqrt(Gamma_h) a^dag, the electric-field-noise limit N_bar -> infinity
with Gamma_h the quoted d<n>/dt at n = 0; qubit dephasing sqrt(gamma_phi/2) sigma_z, so a coherence decays at
gamma_phi = 1/T2; motional dephasing sqrt(2/tau) a^dag a; white intensity noise sqrt(D) H_drive(t).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import qutip as qt

from qutip_trap.trap.heating import thermal_collapse_rates

RecoilOption = Literal["off", "minimal", "vector"]
"""The photon-recoil model of the scattering channels: none, the minimal quadrature, or the full vector quadrature."""

if TYPE_CHECKING:
    from qutip_trap.hilbert.space import HilbertSpace


@dataclass(frozen=True)
class CollapseOp:
    """One Lindblad collapse operator of a run, the ion or mode it acts on (None when neither) and its channel name."""

    op: qt.Qobj | qt.QobjEvo
    """The operator with its rate under the root; a QobjEvo when the amplitude is time dependent."""
    rate_hz: float
    """Rate under the root, in s^-1."""
    channel: str
    ion: int | None
    mode: int | None

    @property
    def time_dependent(self) -> bool:
        return isinstance(self.op, qt.QobjEvo)


def heating_channels(space: HilbertSpace, rates_quanta_per_s: Mapping[int, float]) -> tuple[CollapseOp, ...]:
    """sqrt(Gamma_h) a_m and sqrt(Gamma_h) a_m^dag per carried mode from the quoted heating rates (quanta/s)."""
    out: list[CollapseOp] = []
    for mode, ndot in rates_quanta_per_s.items():
        if space.mode_class(mode) == "frozen" or ndot <= 0.0:
            continue
        down, up = thermal_collapse_rates(float(ndot))
        a = space.annihilation(mode)
        out.append(CollapseOp(math.sqrt(down) * a, float(down), "heating_down", None, mode))
        out.append(CollapseOp(math.sqrt(up) * a.dag(), float(up), "heating_up", None, mode))
    return tuple(out)


def qubit_dephasing_channels(
    space: HilbertSpace, gamma_phi_hz: Mapping[int, float]
) -> tuple[CollapseOp, ...]:
    """sqrt(gamma_phi/2) sigma_z per ion the space carries, gamma_phi = 1/T2 in s^-1."""
    out: list[CollapseOp] = []
    for ion, g in gamma_phi_hz.items():
        if g <= 0.0 or not space.has_ion(ion):
            continue
        out.append(
            CollapseOp(math.sqrt(g / 2.0) * space.sigma_z(ion), float(g), "qubit_dephasing", ion, None)
        )
    return tuple(out)


def motional_dephasing_channels(space: HilbertSpace, tau_s: Mapping[int, float]) -> tuple[CollapseOp, ...]:
    """sqrt(2/tau) a^dag a per carried mode (Ballance 2016)."""
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


def intensity_noise_channels(
    drive_parts: Mapping[str, qt.QobjEvo], densities: Mapping[str, float]
) -> tuple[CollapseOp, ...]:
    """sqrt(D) H_drive(t) per pulse: white multiplicative intensity noise on its drive term (Section 6.4).

    D is the flat two-sided density of the fractional Rabi fluctuation (1/(rad/s)): the laser-intensity ``white_level``
    for a two-photon drive, a quarter of it for a single-photon one. The dissipator is -(D/2)[H_d, [H_d, rho]], which is
    Bermudez's sqrt(2k) H_int with D = 2k, so Gamma_I = D Omega^2/2.
    """
    out: list[CollapseOp] = []
    for gate_id, part in drive_parts.items():
        dens = float(densities.get(gate_id, 0.0))
        if dens <= 0.0:
            continue
        out.append(CollapseOp(math.sqrt(dens) * part, float(dens), f"intensity_noise[{gate_id}]", None, None))
    return tuple(out)
