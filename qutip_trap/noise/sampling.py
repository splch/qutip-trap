"""Quasi-static parameter sampling per shot and sampled processes for fast noise (PLAN.md Sections 5.5, 6.1; M7).

The ``NoiseSample`` record exists from M0; M2 fixed the KEYS the Hamiltonian builder and the engine read from
``values`` (Section 5.7, "Stochastic elements, sampled once per dynamical sample") and M7 adds the trajectory keys
of route (d), stored in ``ou_grids`` as (2, N) arrays (times, values) that ``trajectory`` turns back into a
:class:`~qutip_trap.noise.processes.Trajectory`. A missing key means the nominal value (no offset, unit scale,
n = drawn from the seeds, no trajectory).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap.noise.processes import Trajectory

KEY_RABI_SCALE = "rabi_scale"
"""Multiplies every drive's Rabi frequency (a per-sample scale factor, Section 6.4)."""
KEY_RF_PHASE = "rf_phase_rad"
"""The trap rf phase at t = 0 for an UNLOCKED drive's micromotion modulation (Section 4.3.6); absent = J_0 average."""
KEY_MAINS_PHASE = "mains_phase_rad"
"""The line-trigger phase of this sample's mains pickup (Section 6.3): uniform when free-running, 0 when line-triggered."""
KEY_LASER_OFFSET_HZ = "laser_offset_hz"
"""Frequency offset of the gate laser from its nominal frequency (an optical qubit's reference-cavity drift); a Raman
beat note is rf-referenced and does not see it (Section 6.3)."""
KEY_FIELD_OFFSET_T = "field_offset_t"
"""The quasi-static magnetic-field offset this sample carries (tesla), the source of the qubit offsets below."""
KEY_RF_FRACTION = "rf_amplitude_fraction"
"""Common-mode fractional rf-amplitude offset dV/V of this sample: every rf-derived (transverse) mode moves by omega_m dV/V."""
KEY_BRANCH_WEIGHT = "branch_weight"
"""The weight of the initial-mixture branch this engine call evolves (Section 5.3's Fock-sum path; M9a): the truncation
monitor's populated range is measured at the boundary threshold divided by it, since the threshold is a fraction of the
MIXTURE's population and a branch of weight w contributes w times its own (Section 5.5). Absent = 1."""

KEY_LASER_PHASE_TRAJECTORY = "laser_phase_rad"
"""ou_grids: phi_L(t) of the gate laser (rad), added to the phase of every single-photon optical drive (Section 6.3)."""
KEY_INTENSITY_TRAJECTORY = "intensity_fraction"
"""ou_grids: dI/I(t) of the gate light; a two-photon drive's Rabi frequency scales by (1 + dI/I), a single-photon one by
its square root (Section 6.4)."""
KEY_RF_FRACTION_TRAJECTORY = "rf_amplitude_fraction_trajectory"
"""ou_grids: the sampled part of the rf-amplitude noise, dV/V(t), moving every transverse mode by omega_m dV/V(t) (Section 6.2)."""


def key_qubit_offset_hz(ion: int) -> str:
    """The true transition of ``ion`` minus the frame frequency, in Hz (H_int of Section 5.7), quasi-static part."""
    return f"qubit_offset_hz[{ion}]"


def key_qubit_trajectory_hz(ion: int) -> str:
    """ou_grids: the sampled part of ion's transition offset, delta nu_i(t) in Hz (Section 6.3: S_B through the computed
    sensitivities plus the mains harmonics at the shot's trigger phase)."""
    return f"qubit_trajectory_hz[{ion}]"


def key_mode_offset_hz(mode: int) -> str:
    """Mode-frequency offset of ``mode`` in Hz (Section 6.2 quasi-static part)."""
    return f"mode_offset_hz[{mode}]"


def key_frozen_n(mode: int) -> str:
    """The Fock state of a frozen spectator for this shot (Section 5.2)."""
    return f"frozen_n[{mode}]"


def key_beam_phase_rad(beam: int) -> str:
    """The optical path phase of ``beam`` (rad); a Raman pair's difference phi_1 - phi_2 enters the drive as e^{-i Delta phi}
    (Section 13 row "Optical phase factor on sigma_+"; Section 6.6: phi_beam drifts on hundreds of milliseconds)."""
    return f"beam_phase_rad[{beam}]"


def key_beam_offset_m(beam: int, axis: int) -> str:
    """Pointing offset of ``beam`` along laboratory axis ``axis`` (m): the beam's intensity profile moves with it, so every
    ion's Rabi frequency and every crosstalk ratio under that beam change together (Sections 6.6, 7.10)."""
    return f"beam_offset_m[{beam}][{axis}]"


def key_position_offset_m(ion: int, axis: int) -> str:
    """Quasi-static displacement of ``ion`` along ``axis`` (m) by an uncompensated stray field, e E/(m omega^2)."""
    return f"position_offset_m[{ion}][{axis}]"


@dataclass(frozen=True)
class NoiseSample:
    """One draw of every quasi-static parameter (Section 6.1) plus this sample's fixed-grid trajectories."""

    sample_id: int
    values: dict[str, float]
    """Field offset, mode offsets, Rabi scale, beam phases, ..."""
    ou_grids: dict[str, np.ndarray]
    """Fixed-grid realizations of the fast processes (Section 5.5): (2, N) arrays of (times, values) per key."""
    t_s: float = 0.0
    """The shot-clock time this sample was drawn for (Section 7.5: shot k at t0 + k T_rep)."""

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
    """The nominal sample: no offsets, unit scales, nothing sampled (what M2's tests run under)."""
    return NoiseSample(sample_id=sample_id, values={}, ou_grids={}, t_s=t_s)


__all__ = [
    "KEY_BRANCH_WEIGHT",
    "KEY_FIELD_OFFSET_T",
    "KEY_INTENSITY_TRAJECTORY",
    "KEY_LASER_OFFSET_HZ",
    "KEY_LASER_PHASE_TRAJECTORY",
    "KEY_MAINS_PHASE",
    "KEY_RABI_SCALE",
    "KEY_RF_FRACTION",
    "KEY_RF_FRACTION_TRAJECTORY",
    "KEY_RF_PHASE",
    "NoiseSample",
    "key_beam_offset_m",
    "key_beam_phase_rad",
    "key_frozen_n",
    "key_mode_offset_hz",
    "key_position_offset_m",
    "key_qubit_offset_hz",
    "key_qubit_trajectory_hz",
    "quiet_sample",
]
