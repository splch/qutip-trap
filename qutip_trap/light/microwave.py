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

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qutip_trap.control.pulses import Drive, Tone
from qutip_trap.device.model import Device, Field
from qutip_trap.species.model import Species, level_j, parse_state_label
from qutip_trap.species.polarization import to_atomic_frame
from qutip_trap.species.raman import AtomicStructure, structure_at
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
    st = structure_at(species, field.B_gauss, field.direction)
    lower, upper = species.qubit
    return complex(coupling_rad_s(st, lower, upper, b1_tesla_lab) / TWO_PI)


def ac_zeeman_shift_hz(
    species: Species, field: Field, b1_tesla_lab: Sequence[complex], drive_hz: float
) -> float:
    """delta_ac/2pi of the qubit transition from the off-resonant Zeeman spectator transitions (Section 4.3.3)."""
    st = structure_at(species, field.B_gauss, field.direction)
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


GAUSS_PER_TESLA = 1.0e4
"""1 T = 10^4 G: the species layer's field sensitivities are per gauss, the gradient record's per tesla."""


def field_sensitivity_rad_s_per_t(species: Species, field: Field) -> float:
    """d omega_0/dB of the qubit transition at the device's field, in rad/s per tesla (Section 4.4.5).

    The species layer returns d f_0/dB in Hz/G from the hyperfine-Zeeman diagonalization (never a hand-entered MHz/G,
    Section 4.5.1); this is that slope in the units Srinivas et al.'s Omega_g is written in. It VANISHES at a clock
    point, where a field gradient cannot couple to the spin at first order: a gradient gate needs a field-sensitive
    qubit, which is why Srinivas et al. drive |F=3, m_F=3> <-> |F=2, m_F=2> of 25Mg+ and not the 212.78 G clock line."""
    lower, upper = species.qubit
    slope_hz_per_gauss = species.transition_frequency_hz(lower, upper, field.B_gauss)[1]
    return float(TWO_PI * slope_hz_per_gauss * GAUSS_PER_TESLA)


@dataclass(frozen=True)
class GradientDrive:
    """The derived couplings of a near-field microwave-gradient drive on ``ions`` (Section 4.4.5; Srinivas et al. 2021).

    ``microwave_rabi_rad_s`` is Srinivas's Omega_mu = (B_x/2 hbar)<dn|mu_x|up>, HALF the Rabi frequency of the same
    field in the plan's (hbar Omega/2) convention: each of the two symmetrically detuned tones therefore carries the
    envelope ``tone_rabi_hz`` = 2 Omega_mu/2pi, and the two together drive 2 hbar Omega_mu cos(delta t) sigma_x, whose
    dressing weights the sigma_z force by J_2(4 Omega_mu/delta) and its dephasing by J_0(4 Omega_mu/delta).

    ``coupling_rad_s`` is the PHYSICAL force coefficient per (ion, mode): the term
    sum_{i,m} w_{i,m} sigma_z^i (a_m + a_m^dag) cos(omega_g t + phi_g) of the laboratory-frame Hamiltonian, with
    w_{i,m} = (1/2)(d omega_0/dB) grad(B) (r_hat . e_hat_m) c_{i,m} sqrt(hbar/(2 m_i omega_m)) built from the mode's
    mass-weighted displacement pattern exactly as a Lamb-Dicke parameter is. ``gradient_rabi_rad_s`` is the same
    physics in Srinivas's per-mode form Omega_g = (r_0/4)[grad(B_g . r_hat_q) . r_hat](d omega_0/dB) with r_0 =
    sqrt(hbar/(2 M_total omega_m)) carrying the TOTAL mass of the addressed ions (Section 13): for two equal-mass ions
    on a spatially antisymmetric mode w_{i,m} = +-2 Omega_g exactly, which is the (sigma_z1 - sigma_z2) normalization
    of the plan's Hamiltonian.
    """

    ions: tuple[int, ...]
    microwave_rabi_rad_s: float
    tone_rabi_hz: float
    ac_zeeman_shift_hz: float
    field_sensitivity_rad_s_per_t: float
    coupling_rad_s: dict[tuple[int, int], float]
    gradient_rabi_rad_s: dict[int, float]

    def bessel_argument(self, detuning_rad_s: float) -> float:
        """4 Omega_mu/delta, the argument of every Bessel factor of the dressed dynamics."""
        return 4.0 * self.microwave_rabi_rad_s / detuning_rad_s


def derive_gradient_drive(
    device: Device, ions: Sequence[int], *, drive_hz: float | None = None
) -> GradientDrive:
    """Omega_mu, Omega_g and the per-(ion, mode) sigma_z force of ``device.gradient`` on ``ions`` (Section 4.4.5).

    Everything is derived: Omega_mu from the magnetic-dipole matrix element of the configured microwave field amplitude
    (``coupling_rad_s``), d omega_0/dB from the species' hyperfine-Zeeman diagonalization, and the force coefficients
    from the crystal's mass-weighted mode patterns and the configured gradient. The simulator's "electronic drive"
    builds its Hamiltonian from these (Section 4.4.5's closing sentence)."""
    grad = device.gradient
    if grad is None:
        raise ValueError(
            "a microwave-gradient drive needs Device.gradient (the near-field electrodes' gradient amplitude, "
            "frequency and microwave field amplitude; Section 4.4.5)"
        )
    ions_ = tuple(int(i) for i in ions)
    if not ions_:
        raise ValueError("a gradient drive addresses at least one ion")
    crystal = device.crystal
    species_0 = crystal.species[ions_[0]]
    lower, upper = species_0.qubit
    f0 = species_0.transition_frequency_hz(lower, upper, device.field.B_gauss)[0]
    f_drive = abs(f0) if drive_hz is None else float(drive_hz)
    b1 = tuple(complex(x) for x in grad.b1_tesla_lab)
    omega_rabi = abs(complex(rabi_frequency_hz(species_0, device.field, b1))) * TWO_PI
    sensitivity = field_sensitivity_rad_s_per_t(species_0, device.field)
    axis = np.asarray(grad.axis, dtype=float)
    coupling: dict[tuple[int, int], float] = {}
    per_mode: dict[int, float] = {}
    mass_total = float(sum(crystal.masses_kg[i] for i in ions_))
    for m_index, mode in enumerate(crystal.modes):
        pattern = mode.displacement_pattern()
        w = mode.omega_rad_s
        for ion in ions_:
            projection = float(np.dot(axis, pattern[ion]))
            x0 = math.sqrt(HBAR_J_S / (2.0 * float(crystal.masses_kg[ion]) * w))
            coupling[(ion, m_index)] = 0.5 * sensitivity * grad.gradient_t_per_m * projection * x0
        r0_total = math.sqrt(HBAR_J_S / (2.0 * mass_total * w))
        per_mode[m_index] = 0.25 * r0_total * grad.gradient_t_per_m * sensitivity
    return GradientDrive(
        ions=ions_,
        microwave_rabi_rad_s=0.5 * omega_rabi,
        tone_rabi_hz=omega_rabi / TWO_PI,
        ac_zeeman_shift_hz=ac_zeeman_shift_hz(species_0, device.field, b1, f_drive),
        field_sensitivity_rad_s_per_t=sensitivity,
        coupling_rad_s=coupling,
        gradient_rabi_rad_s=per_mode,
    )


def gradient_drive(
    derived: GradientDrive, *, detuning_hz: float, phase_rad: float = 0.0, walsh_sign: int = 1
) -> Drive:
    """The two-tone microwave ``Drive`` of a gradient gate: tones at -/+ ``detuning_hz`` from the (ac-Zeeman-shifted)
    qubit frequency, each carrying ``derived.tone_rabi_hz`` = 2 Omega_mu/2pi (Section 4.4.5).

    The force sign is the relative phase of the gradient and the microwave dressing: the resonant dressed term carries
    e^{-i(2 phi_d + phi_g)} with phi_d = (phi_blue - phi_red)/2 the tones' half-difference, so ``walsh_sign = -1``
    (phi_d -> phi_d + pi/2) reverses it, which is how Srinivas et al.'s eight Walsh-modulated segments close the loop."""
    if walsh_sign not in (1, -1):
        raise ValueError(
            "walsh_sign is +1 or -1 (a pi/2 shift of the tones' half-difference reverses the force)"
        )
    half = 0.0 if walsh_sign == 1 else 0.5 * math.pi
    return Drive(
        kind="gradient",
        ions=derived.ions,
        tones=(
            Tone(-abs(float(detuning_hz)), phase_rad - half, derived.tone_rabi_hz),
            Tone(abs(float(detuning_hz)), phase_rad + half, derived.tone_rabi_hz),
        ),
        beams=(),
        stark_shift_hz=0.0,
        crosstalk={},
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
    "GAUSS_PER_TESLA",
    "GradientDrive",
    "MicrowaveDrive",
    "ac_zeeman_shift_hz",
    "coupling_rad_s",
    "derive_gradient_drive",
    "derive_microwave_drive",
    "effective_detuning_hz",
    "field_sensitivity_rad_s_per_t",
    "gradient_drive",
    "magnetic_moment_operators",
    "rabi_frequency_hz",
    "square_microwave_drive",
]
