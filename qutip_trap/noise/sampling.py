"""The per-sample noise record ``NoiseSample`` and the keys the Hamiltonian builder and the engine read from it.

``values`` holds the quasi-static parameters and ``ou_grids`` the trajectories, as (2, N) arrays (times, values). A
missing key means the nominal value: no offset, unit scale, a frozen mode's n drawn from the seeds, no trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap.noise.processes import Trajectory

KEY_RABI_SCALE = "rabi_scale"
"""Multiplies every drive's Rabi frequency."""
KEY_RF_PHASE = "rf_phase_rad"
"""The trap rf phase at t = 0 for an unlocked drive's micromotion modulation (rad); absent = the J_0 average."""
KEY_MAINS_PHASE = "mains_phase_rad"
"""The line-trigger phase of this sample's mains pickup (rad): uniform when free-running, 0 when line-triggered."""
KEY_LASER_OFFSET_HZ = "laser_offset_hz"
"""The gate laser's frequency offset from nominal (an optical qubit's laser drift); a Raman beat note ignores it."""
KEY_FIELD_OFFSET_T = "field_offset_t"
"""The quasi-static magnetic-field offset of this sample (tesla)."""
KEY_RF_FRACTION = "rf_amplitude_fraction"
"""Common-mode fractional rf-amplitude offset dV/V of this sample: every rf-derived (transverse) mode moves by omega_m dV/V."""
KEY_BRANCH_WEIGHT = "branch_weight"
"""The weight w of the initial-mixture branch this engine call evolves (absent = 1): the truncation monitor divides its
threshold, a fraction of the whole mixture's population, by w."""

KEY_LASER_PHASE_TRAJECTORY = "laser_phase_rad"
"""ou_grids: phi_L(t) of the gate laser (rad), added to the phase of every single-photon optical drive."""
KEY_INTENSITY_TRAJECTORY = "intensity_fraction"
"""ou_grids: dI/I(t) of the gate light, scaling a two-photon Rabi frequency by 1 + dI/I, a one-photon by its root."""
KEY_RF_FRACTION_TRAJECTORY = "rf_amplitude_fraction_trajectory"
"""ou_grids: the sampled part of the rf-amplitude noise, dV/V(t), moving every transverse mode by omega_m dV/V(t)."""


def key_qubit_offset_hz(ion: int) -> str:
    """The quasi-static part of ``ion``'s transition frequency minus the frame frequency, in Hz."""
    return f"qubit_offset_hz[{ion}]"


def key_qubit_trajectory_hz(ion: int) -> str:
    """ou_grids: the sampled part of ``ion``'s transition offset, delta nu_i(t) in Hz (field noise and mains)."""
    return f"qubit_trajectory_hz[{ion}]"


def key_mode_offset_hz(mode: int) -> str:
    """The quasi-static frequency offset of ``mode`` in Hz."""
    return f"mode_offset_hz[{mode}]"


def key_frozen_n(mode: int) -> str:
    """The Fock state of the frozen spectator ``mode`` for this shot."""
    return f"frozen_n[{mode}]"


def key_beam_phase_rad(beam: int) -> str:
    """The optical path phase phi_j of ``beam`` (rad), E_j ~ cos(k_j . r - omega_j t + phi_j): a Raman pair's factor on
    sigma_+ is e^{+i(Delta k . X - Delta phi)}, Delta k = k_1 - k_2, Delta phi = phi_2 - phi_1 (one beam: -phi_1)."""
    return f"beam_phase_rad[{beam}]"


def key_beam_phase_trajectory_rad(beam: int) -> str:
    """ou_grids: the sampled part phi_j(t) of ``beam``'s optical path phase (rad), drawn independently per beam, so a
    Raman pair's differential phase has twice one beam's variance; the quasi-static part is ``key_beam_phase_rad``."""
    return f"beam_phase_trajectory_rad[{beam}]"


def key_beam_offset_m(beam: int, axis: int) -> str:
    """Pointing offset of ``beam`` along laboratory axis ``axis`` (m), moving its Rabi rates and crosstalk together."""
    return f"beam_offset_m[{beam}][{axis}]"


def key_position_offset_m(ion: int, axis: int) -> str:
    """Quasi-static displacement of ``ion`` along ``axis`` (m) by an uncompensated stray field, e E/(m omega^2)."""
    return f"position_offset_m[{ion}][{axis}]"


@dataclass(frozen=True)
class NoiseSample:
    """One draw of every quasi-static parameter plus this sample's fixed-grid trajectories."""

    sample_id: int
    values: dict[str, float]
    ou_grids: dict[str, np.ndarray]
    """(2, N) arrays of (times, values) per key."""
    t_s: float = 0.0
    """The shot-clock time this sample was drawn for."""

    def get(self, key: str, default: float) -> float:
        return float(self.values.get(key, default))

    def trajectory(self, key: str) -> Trajectory | None:
        grid = self.ou_grids.get(key)
        return None if grid is None else Trajectory.from_grid(grid)

    @property
    def is_quiet(self) -> bool:
        """No offsets, unit scales and no trajectories (frozen Fock states and the branch weight aside)."""
        for k, v in self.values.items():
            if k.startswith("frozen_n") or k == KEY_BRANCH_WEIGHT:
                continue
            nominal = 1.0 if k == KEY_RABI_SCALE else 0.0
            if v != nominal:
                return False
        return not self.ou_grids


def quiet_sample(sample_id: int = 0, t_s: float = 0.0) -> NoiseSample:
    """The nominal sample: no offsets, unit scales, nothing sampled."""
    return NoiseSample(sample_id=sample_id, values={}, ou_grids={}, t_s=t_s)
