"""Direct hyperfine (magnetic-dipole) drives and their ac Zeeman shifts (PLAN.md Sections 3.2, 4.3.3; milestone M2).

A microwave field B(t) = Re[B_1 e^{-i omega t}] couples through H = mu_B (g_J J + g_I I) . B(t)/hbar (Section 13, the
Zeeman Hamiltonian with one sign for both g factors); in the rotating-wave approximation the resonant part of the
oscillation carries B_1/2, so the Rabi frequency of |down> -> |up> in the (hbar Omega/2) convention is
Omega = (mu_B/hbar) <up|(g_J J + g_I I) . B_1|down> with the DRESSED states of the atomic layer and B_1 decomposed
in the atomic frame about B_hat. eta ~ 0: no motional coupling without a field gradient (Section 4.3.3). The
second-order shift of level a from every other sublevel b of the manifold is
delta_a = sum_b [|m_ab|^2/(4(omega_a - omega_b + omega)) + |m~_ab|^2/(4(omega_a - omega_b - omega))]
with m_ab and m~_ab the couplings through B_1 and B_1^* (co- and counter-rotating components); the ac Zeeman shift of
the qubit transition is delta_up - delta_down over the SPECTATOR pairs (the driven pair's own dynamics excluded). With
the plan's detuning sign delta = omega_drive - omega_0, a negative transition shift increases the detuning:
delta_eff = delta - delta_ac (Harty 2014: +4.5 Hz - (-1.0 Hz) = +5.5 Hz, Section 9.2 "ac Zeeman sign").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qutip_trap.control.pulses import Drive, Tone
from qutip_trap.device.model import Device, Field
from qutip_trap.species.model import Species, level_j, parse_state_label
from qutip_trap.species.polarization import to_atomic_frame
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.species.wigner import angular_momentum_matrices, as_half_integer
from qutip_trap.species.zeeman import g_I_steck
from qutip_trap.units import HBAR_J_S, MU_B_J_PER_T, TWO_PI

M2 = "milestone M2 (light/microwave.py, PLAN.md Section 4.3.3)"


def magnetic_moment_operators(species: Species, level: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(G_x', G_y', G_z') = g_J J + g_I I on the |m_I, m_J> basis of ``level`` in the atomic frame (dimensionless)."""
    lv = species.level(level)
    nuc = as_half_integer(species.nuclear_spin)
    J = level_j(level)
    g_i = g_I_steck(species.mu_I_nuclear_magnetons, float(nuc))
    ix, iy, iz = angular_momentum_matrices(nuc)
    jx, jy, jz = angular_momentum_matrices(J)
    eye_i = np.eye(ix.shape[0])
    eye_j = np.eye(jx.shape[0])
    return tuple(  # type: ignore[return-value]
        lv.g_J * np.kron(eye_i, jq) + g_i * np.kron(iq, eye_j) for iq, jq in ((ix, jx), (iy, jy), (iz, jz))
    )


def coupling_rad_s(
    structure: AtomicStructure, a: str, b: str, b1_tesla_lab: Sequence[complex] | np.ndarray
) -> complex:
    """m_ba = (mu_B/hbar) <b|(g_J J + g_I I) . B_1|a> in rad/s for two dressed states of one level (B_1 complex, lab frame)."""
    sa, sb = structure.state(a), structure.state(b)
    if sa.level != sb.level:
        raise ValueError("a magnetic-dipole drive couples sublevels of one fine-structure level")
    gx, gy, gz = magnetic_moment_operators(structure.species, sa.level)
    b_atomic = to_atomic_frame(np.asarray(b1_tesla_lab, dtype=complex), structure.b_hat)
    op = gx * b_atomic[0] + gy * b_atomic[1] + gz * b_atomic[2]
    return complex(MU_B_J_PER_T / HBAR_J_S * np.vdot(sb.vector, op @ sa.vector))


def rabi_frequency_hz(
    species: Species, field: Field, b1_tesla_lab: Sequence[complex] | np.ndarray
) -> complex:
    """Omega/2pi of the qubit transition under the microwave amplitude B_1 (complex lab-frame vector, tesla)."""
    st = AtomicStructure(species, field.B_gauss, field.direction)
    lower, upper = species.qubit
    return complex(coupling_rad_s(st, lower, upper, b1_tesla_lab) / TWO_PI)


def ac_zeeman_shift_hz(
    species: Species, field: Field, b1_tesla_lab: Sequence[complex], drive_hz: float
) -> float:
    """delta_ac/2pi of the qubit transition from the off-resonant Zeeman spectator transitions (Section 4.3.3)."""
    st = AtomicStructure(species, field.B_gauss, field.direction)
    lower, upper = species.qubit
    level = parse_state_label(lower)[0]
    omega = TWO_PI * drive_hz
    b1 = np.asarray(b1_tesla_lab, dtype=complex)
    shifts: dict[str, float] = {}
    pair = {st.state(lower).full_label, st.state(upper).full_label}
    for a in (lower, upper):
        sa = st.state(a)
        total = 0.0
        for sb in st.states_of(level):
            if sb.full_label == sa.full_label or {sa.full_label, sb.full_label} == pair:
                continue
            m_co = coupling_rad_s(st, a, sb.full_label, b1)
            m_counter = coupling_rad_s(st, a, sb.full_label, np.conj(b1))
            w_ab = TWO_PI * (sa.energy_hz - sb.energy_hz)
            total += abs(m_co) ** 2 / (4.0 * (w_ab + omega)) + abs(m_counter) ** 2 / (4.0 * (w_ab - omega))
        shifts[a] = total
    return (shifts[upper] - shifts[lower]) / TWO_PI


def effective_detuning_hz(detuning_hz: float, ac_zeeman_shift_hz_: float) -> float:
    """delta_eff = delta - delta_ac (Section 9.2, "ac Zeeman sign"): a negative transition shift increases the detuning."""
    return detuning_hz - ac_zeeman_shift_hz_


@dataclass(frozen=True)
class MicrowaveDrive:
    ion: int
    rabi_hz: complex
    ac_zeeman_shift_hz: float
    b1_tesla_lab: tuple[complex, complex, complex]

    @property
    def carrier_rabi_hz(self) -> float:
        return abs(self.rabi_hz)


def derive_microwave_drive(
    device: Device, ion: int, b1_tesla_lab: Sequence[complex], *, drive_hz: float | None = None
) -> MicrowaveDrive:
    """Omega and the ac Zeeman shift of ``ion``'s qubit under the field amplitude B_1 (drive frequency default: the qubit's)."""
    species = device.crystal.species[ion]
    lower, upper = species.qubit
    f0 = species.transition_frequency_hz(lower, upper, device.field.B_gauss)[0]
    f_drive = abs(f0) if drive_hz is None else float(drive_hz)
    b1 = tuple(complex(x) for x in b1_tesla_lab)
    return MicrowaveDrive(
        ion=ion,
        rabi_hz=rabi_frequency_hz(species, device.field, b1),
        ac_zeeman_shift_hz=ac_zeeman_shift_hz(species, device.field, b1, f_drive),
        b1_tesla_lab=b1,  # type: ignore[arg-type]
    )


def square_microwave_drive(
    ion: int, rabi_hz: float, *, detuning_hz: float = 0.0, phase_rad: float = 0.0, stark_shift_hz: float = 0.0
) -> Drive:
    """A microwave Drive with no beams and eta = 0 (Section 4.3.3), Rabi frequency in Hz in the (hbar Omega/2) convention."""
    tone = Tone(detuning_hz=float(detuning_hz), phase_rad=float(phase_rad), envelope_hz=float(rabi_hz))
    return Drive(
        kind="microwave",
        ions=(ion,),
        tones=(tone,),
        beams=(),
        stark_shift_hz=float(stark_shift_hz),
        crosstalk={},
    )


__all__ = [
    "MicrowaveDrive",
    "ac_zeeman_shift_hz",
    "coupling_rad_s",
    "derive_microwave_drive",
    "effective_detuning_hz",
    "magnetic_moment_operators",
    "rabi_frequency_hz",
    "square_microwave_drive",
]
