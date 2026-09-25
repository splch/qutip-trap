"""One dynamical sample: quasi-static offsets under string keys and fixed-grid trajectories (PLAN.md Section 6.1).

A missing key means the nominal value (no offset, unit scale, no trajectory); ``ou_grids`` stores each trajectory as a
(2, N) array of (times, values).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap.noise.processes import Trajectory

KEY_RABI_SCALE = "rabi_scale"
"""Multiplies every drive's Rabi frequency."""
"""The trap rf phase at t = 0 for an unlocked drive's micromotion modulation; absent = the J_0 average."""
KEY_MAINS_PHASE = "mains_phase_rad"
"""The line-trigger phase of the mains pickup: uniform when free-running, 0 when line-triggered."""
KEY_LASER_OFFSET_HZ = "laser_offset_hz"
"""The gate laser's offset from its nominal frequency (an optical qubit's reference-cavity drift)."""
KEY_FIELD_OFFSET_T = "field_offset_t"
"""The quasi-static magnetic-field offset (T), the source of the qubit offsets."""
KEY_RF_FRACTION = "rf_amplitude_fraction"
"""The common-mode fractional rf-amplitude offset dV/V: every transverse mode moves by omega_m dV/V."""
KEY_BRANCH_WEIGHT = "branch_weight"
"""The weight of the initial-mixture branch an engine call evolves (the truncation monitor scales its threshold by it);
absent = 1."""

KEY_LASER_PHASE_TRAJECTORY = "laser_phase_rad"
"""ou_grids: the gate laser's phase phi_L(t) (rad), added to every single-photon optical drive."""
KEY_INTENSITY_TRAJECTORY = "intensity_fraction"
"""ou_grids: dI/I(t) of the gate light; a two-photon Rabi frequency scales by 1 + dI/I, a single-photon one by its root."""
KEY_RF_FRACTION_TRAJECTORY = "rf_amplitude_fraction_trajectory"
"""ou_grids: dV/V(t), moving every transverse mode by omega_m dV/V(t)."""


def key_qubit_offset_hz(ion: int) -> str:
    """The ion's true transition minus the frame frequency (Hz), quasi-static part."""
    return f"qubit_offset_hz[{ion}]"


def key_qubit_trajectory_hz(ion: int) -> str:
    """ou_grids: the sampled part of the ion's transition offset (Hz)."""
    return f"qubit_trajectory_hz[{ion}]"


def key_mode_offset_hz(mode: int) -> str:
    """The mode's frequency offset (Hz), quasi-static part."""
    return f"mode_offset_hz[{mode}]"


def key_frozen_n(mode: int) -> str:
    """The Fock state of a frozen spectator mode for this shot."""
    return f"frozen_n[{mode}]"


def key_beam_phase_rad(beam: int) -> str:
    """The optical path phase phi_j of a beam (rad), E_j ~ cos(k_j . r - omega_j t + phi_j).

    A Raman pair's beat note carries Delta phi = phi_2 - phi_1 and sigma_+ the factor e^{+i(Delta k . X - Delta phi)}, so
    beam 1's phase enters as e^{+i phi_1} and beam 2's as e^{-i phi_2}; a single-beam drive has Delta phi = -phi_1.
    """
    return f"beam_phase_rad[{beam}]"


def key_beam_phase_trajectory_rad(beam: int) -> str:
    """ou_grids: the sampled part of a beam's optical path phase (rad), independent per beam, so a Raman pair's beat note
    sees phi_2(t) - phi_1(t) with twice one beam's variance."""
    return f"beam_phase_trajectory_rad[{beam}]"


def key_beam_offset_m(beam: int, axis: int) -> str:
    """The beam's pointing offset along a laboratory axis (m): its intensity profile, hence every Rabi frequency and
    crosstalk ratio under it, moves with it."""
    return f"beam_offset_m[{beam}][{axis}]"


def key_position_offset_m(ion: int, axis: int) -> str:
    """The ion's quasi-static displacement along an axis (m) by a stray field, e E/(m omega^2)."""
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
        """No offsets, unit scales and no trajectories: the nominal sample."""
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
