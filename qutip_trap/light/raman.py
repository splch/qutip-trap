"""Derived drives: two-photon Rabi frequency, effective Delta k, per-mode eta, Stark shift and scattering from the beams
(PLAN.md Sections 3.3, 4.3.2, 4.5.4, 4.5.5; milestone M2).

Design principle (Section 3.1): device parameters in, everything else derived. A ``DerivedDrive`` is what ``light/``
computes for one ion under one physical drive; the Rabi frequency, the Stark shift and the scattering rates come from
the atomic layer of Section 4.5 (``AtomicStructure``) at the ion's position in the beams, the Lamb-Dicke parameters
from ``Crystal.lamb_dicke`` with the trap's Mathieu record (C0 applied there and nowhere else), and the excess
micromotion index from ``Trap.micromotion_beta``. Conventions (Section 13): Omega in the (hbar Omega/2) convention,
Omega_R = sum_e conj(Omega_2) Omega_1/(2 Delta_e) with beam 1 the higher-frequency beam absorbed from the lower qubit
level, so that the tone at beat note omega_1 - omega_2 = omega_0 + mu drives |down> -> |up> with momentum
Delta k = k_1 - k_2 (Wineland 2003 Eq. 2.3 read in the plan's normalization); a co-propagating pair has Delta k -> 0
and no motional coupling. The complex phase of Omega_R depends on the dressed-state sign gauge (M0a) and is recorded
but not applied: the frame alignment of Section 7.5 absorbs it once per ion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qutip_trap.control.pulses import Drive, DriveKind, LightShiftCouplings, Tone
from qutip_trap.device.model import Device
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.mathieu import MathieuParameters
from qutip_trap.trap.micromotion import MicromotionIndex
from qutip_trap.units import TWO_PI

M2 = "milestone M2 (light/raman.py, PLAN.md Section 4.3.2)"


@dataclass(frozen=True)
class ScatteringBudget:
    """Per second, at the ion's position, summed over the drive's beams (Section 4.5.5)."""

    rayleigh_per_s: dict[str, float]
    """Elastic rate from each qubit state (keyed by its label)."""
    raman_spin_flip_per_s: dict[str, float]
    """Rate from each qubit state into the OTHER qubit state."""
    leakage_per_s: dict[str, float]
    """Rate from each qubit state out of the qubit pair, summed over final states."""
    rayleigh_dephasing_per_s: float
    """Gamma_el of the pair (Uys et al. 2010): the elastic-amplitude DIFFERENCE squared."""
    residual_excited_population: float

    def per_pulse_error(self, duration_s: float) -> float:
        """The d = 2 estimate of Section 4.3.2: the Raman (spin-flip plus leakage) probability, averaged over the qubit states."""
        rates = [self.raman_spin_flip_per_s[k] + self.leakage_per_s[k] for k in self.raman_spin_flip_per_s]
        return float(np.mean(rates)) * duration_s if rates else 0.0


@dataclass(frozen=True)
class DerivedDrive:
    """What ``light/`` derives for one ion under one physical drive (Section 3.3, "derived from beams by light/")."""

    ion: int
    kind: DriveKind
    beams: tuple[int, ...]
    rabi_hz: complex
    """Omega/2pi, complex in the (hbar Omega/2) convention; |rabi_hz| is the carrier Rabi frequency, arg the gauge phase."""
    delta_k: np.ndarray
    etas: dict[int, float]
    """eta_{ion, m} per crystal mode, C0 inside when the trap carries an rf record."""
    c0_applied: bool
    stark_shift_hz: float
    """Differential light shift of the transition: shift of the upper level minus the lower one, Hz (Section 4.3.2)."""
    scattering: ScatteringBudget | None
    micromotion: MicromotionIndex | None
    provenance: tuple[str, ...]
    light_shift: LightShiftCouplings | None = None
    """For ``kind == "light_shift"``: the level weights and the off-resonant spin-flip weight of Section 4.4.4; ``rabi_hz`` is
    then Omega_LS/2pi = (Omega_upup - Omega_dndn)/(2 x 2pi), the coefficient of the spin-dependent force (Zhu 2006 Eq. 2)."""

    @property
    def carrier_rabi_hz(self) -> float:
        return abs(self.rabi_hz)

    def pi_time_s(self, area_rad: float = math.pi) -> float:
        """Omega t = area in the plan's convention (a carrier pi pulse at Omega t = pi)."""
        if self.carrier_rabi_hz == 0.0:
            raise ZeroDivisionError("the drive has no coupling at this ion")
        return area_rad / (TWO_PI * self.carrier_rabi_hz)

    def eta_along(self, mode: int) -> float:
        return self.etas[mode]


def _position(device: Device, ion: int) -> list[float]:
    return [float(x) for x in device.crystal.positions_m[ion]]


def _structure(device: Device, ion: int) -> AtomicStructure:
    species = device.crystal.species[ion]
    return AtomicStructure(species, device.field.B_gauss, device.field.direction)


def mathieu_or_none(device: Device, ion: int) -> MathieuParameters | None:
    """The trap's Mathieu record for the ion's species, or None when the trap has no rf record (C0 = 1, beta = 0)."""
    species = device.crystal.species[ion]
    try:
        return device.trap.mathieu(species)
    except ValueError:
        return None


def lamb_dicke_parameters(device: Device, ion: int, delta_k: np.ndarray) -> tuple[dict[int, float], bool]:
    """eta_{ion, m} for every crystal mode with C0 from the trap's Mathieu record when it exists; (etas, c0_applied).

    C0 is applied inside ``Crystal.lamb_dicke`` and nowhere else (Section 4.1.1); the builder, the pulse-shaping
    solvers and the light layer all read this one function.
    """
    params = mathieu_or_none(device, ion)
    etas = {
        m: float(device.crystal.lamb_dicke(ion, m, delta_k, micromotion=params))
        for m in range(len(device.crystal.modes))
    }
    return etas, params is not None


def _etas(device: Device, ion: int, delta_k: np.ndarray) -> tuple[dict[int, float], bool]:
    return lamb_dicke_parameters(device, ion, delta_k)


def _micromotion(device: Device, ion: int, delta_k: np.ndarray) -> MicromotionIndex | None:
    if float(np.linalg.norm(delta_k)) == 0.0:
        return None
    try:
        return device.trap.micromotion_beta(device.crystal.species[ion], delta_k)
    except ValueError:
        return None


def scattering_budget(device: Device, ion: int, beam_indices: Sequence[int]) -> ScatteringBudget:
    """Rayleigh, spin-flip and leakage rates of both qubit states under the beams, at the ion's position (Section 4.5.5)."""
    st = _structure(device, ion)
    species = device.crystal.species[ion]
    pos = _position(device, ion)
    lower, upper = species.qubit
    labels = {lower: st.state(lower).full_label, upper: st.state(upper).full_label}
    rayleigh: dict[str, float] = {}
    flip: dict[str, float] = {}
    leak: dict[str, float] = {}
    gamma_el = 0.0
    residual = 0.0
    for a, other in ((lower, upper), (upper, lower)):
        r = leak_rate = f = 0.0
        for b in beam_indices:
            beam = device.beams[b]
            rates = st.scattering_rates(st.state(a), beam, pos)
            for label, rate in rates.items():
                if label == labels[a]:
                    r += rate
                elif label == labels[other]:
                    f += rate
                else:
                    leak_rate += rate
            residual += st.residual_excited_population(st.state(a), beam, pos)
        rayleigh[a] = r
        flip[a] = f
        leak[a] = leak_rate
    for b in beam_indices:
        gamma_el += st.rayleigh_dephasing_rate(st.state(upper), st.state(lower), device.beams[b], pos)
    return ScatteringBudget(rayleigh, flip, leak, gamma_el, residual / 2.0)


def differential_stark_shift_hz(device: Device, ion: int, beam_indices: Sequence[int]) -> float:
    """sum_beams [delta(upper) - delta(lower)]/2pi at the ion's position: red light lowers a level (Section 4.5.4)."""
    st = _structure(device, ion)
    species = device.crystal.species[ion]
    pos = _position(device, ion)
    beams = [device.beams[b] for b in beam_indices]
    lower, upper = species.qubit
    return (
        st.light_shift_rad_s(st.state(upper), beams, pos) - st.light_shift_rad_s(st.state(lower), beams, pos)
    ) / TWO_PI


def derive_raman_drive(
    device: Device, ion: int, beams: tuple[int, int], *, scattering: bool = True
) -> DerivedDrive:
    """A stimulated-Raman drive of ``ion`` by (beam 1, beam 2): Omega_R, Delta k = k_1 - k_2, etas, Stark shift, scattering."""
    b1, b2 = beams
    st = _structure(device, ion)
    species = device.crystal.species[ion]
    pos = _position(device, ion)
    lower, upper = species.qubit
    omega = st.raman_coupling_rad_s(st.state(lower), st.state(upper), device.beams[b1], device.beams[b2], pos)
    delta_k = np.asarray(device.beams[b1].k_vector() - device.beams[b2].k_vector(), dtype=float)
    etas, c0 = _etas(device, ion, delta_k)
    return DerivedDrive(
        ion=ion,
        kind="raman",
        beams=(b1, b2),
        rabi_hz=complex(omega / TWO_PI),
        delta_k=delta_k,
        etas=etas,
        c0_applied=c0,
        stark_shift_hz=differential_stark_shift_hz(device, ion, beams),
        scattering=scattering_budget(device, ion, beams) if scattering else None,
        micromotion=_micromotion(device, ion, delta_k),
        provenance=(
            "conv.two_photon_rabi",
            "conv.effective_wavevector",
            "conv.lamb_dicke",
            "anchor.raman.wineland_2_3",
        ),
    )


def derive_optical_drive(device: Device, ion: int, beam: int, *, scattering: bool = True) -> DerivedDrive:
    """A single-photon optical drive (E1, or E2 for an I = 0 quadrupole qubit): Omega from Species.rabi_frequency_hz."""
    species = device.crystal.species[ion]
    lower, upper = species.qubit
    omega_hz = species.rabi_frequency_hz(lower, upper, device.beams[beam], device.field)
    lo_level = lower.split()[0]
    up_level = upper.split()[0]
    e2 = next(
        (
            t
            for t in species.transitions
            if t.multipole == "E2" and {t.lower, t.upper} == {lo_level, up_level}
        ),
        None,
    )
    kind: DriveKind = "optical_E2" if e2 is not None else "optical_E1"
    delta_k = np.asarray(device.beams[beam].k_vector(), dtype=float)
    etas, c0 = _etas(device, ion, delta_k)
    # the E2 Rabi frequency of Species.rabi_frequency_hz is evaluated at the beam's peak intensity; scale to the ion's position
    pos = _position(device, ion)
    peak = device.beams[beam].intensity_at(np.asarray(device.beams[beam].pointing_m, dtype=float))
    here = device.beams[beam].intensity_at(np.asarray(pos))
    if kind == "optical_E2" and peak > 0.0:
        omega_hz = omega_hz * math.sqrt(here / peak)
    budget = None
    if scattering and kind == "optical_E1":
        budget = scattering_budget(device, ion, (beam,))
    return DerivedDrive(
        ion=ion,
        kind=kind,
        beams=(beam,),
        rabi_hz=complex(omega_hz),
        delta_k=delta_k,
        etas=etas,
        c0_applied=c0,
        stark_shift_hz=0.0 if kind == "optical_E2" else differential_stark_shift_hz(device, ion, (beam,)),
        scattering=budget,
        micromotion=_micromotion(device, ion, delta_k),
        provenance=(
            "conv.quadrupole_coupling" if kind == "optical_E2" else "conv.rabi_from_intensity",
            "conv.lamb_dicke",
        ),
    )


def two_photon_self_couplings_hz(device: Device, ion: int, beams: tuple[int, int]) -> tuple[complex, complex]:
    """(Omega_{dn dn}, Omega_{up up})/2pi: the two-photon couplings of each qubit level to ITSELF under (beam 1, beam 2), the
    Raman formula sum_e conj(Omega^{(2)}_{e g}) Omega^{(1)}_{e g}/(2 Delta_e) with g' = g (Wineland 2003; Section 4.4.4).

    The beat note of the two beams modulates each level's light shift as Re[Omega_gg e^{-i(mu t - Delta k . x)}]; the
    differential part (Omega_upup - Omega_dndn)/2 is the state-dependent force of the light-shift gate and the common
    part a spin-independent force on the motion. Both vanish to leading order for a clock qubit under linearly polarized
    light, and the differential part for any qubit whose two levels see the same scalar and vector shifts.
    """
    b1, b2 = beams
    st = _structure(device, ion)
    species = device.crystal.species[ion]
    pos = _position(device, ion)
    lower, upper = species.qubit
    dn = st.raman_coupling_rad_s(st.state(lower), st.state(lower), device.beams[b1], device.beams[b2], pos)
    up = st.raman_coupling_rad_s(st.state(upper), st.state(upper), device.beams[b1], device.beams[b2], pos)
    return complex(dn / TWO_PI), complex(up / TWO_PI)


def derive_light_shift_drive(
    device: Device,
    ion: int,
    beams: tuple[int, int],
    *,
    scattering: bool = True,
    min_relative_force: float = 1e-2,
) -> DerivedDrive:
    """The light-shift (sigma_z sigma_z) gate drive of Section 4.4.4 by (beam 1, beam 2): the beat note is tuned near a MODE
    frequency, so the qubit is not flipped; the drive is the differential two-photon self-coupling Omega_LS = (Omega_upup -
    Omega_dndn)/2 (Zhu-Monroe-Duan 2006 Eq. 2, H = hbar Omega_j cos(Delta k . q_j + mu t) sigma_z^j), with the spin-independent
    part and the far-off-resonant spin-flip coupling Omega_R carried as weights relative to Omega_LS.

    Raises when the differential coupling is below ``min_relative_force`` (default 1%) of the common part: the spin-independent
    force then displaces the motion by more than a hundred loop radii before the differential one closes a loop. A clock qubit
    under linear polarization keeps only the hyperfine difference of the detunings, about 1e-3 of the scalar shift (Baldwin's
    D3/2 polarization-gradient construction is a different level scheme).
    """
    b1, b2 = beams
    dn, up = two_photon_self_couplings_hz(device, ion, beams)
    omega_ls = 0.5 * (up - dn)
    common = 0.5 * (up + dn)
    st = _structure(device, ion)
    species = device.crystal.species[ion]
    pos = _position(device, ion)
    lower, upper = species.qubit
    omega_r = (
        st.raman_coupling_rad_s(st.state(lower), st.state(upper), device.beams[b1], device.beams[b2], pos)
        / TWO_PI
    )
    # the force must dominate both the spin-independent part and the ordinary Raman coupling the same beams drive:
    # crossed linear polarizations make no intensity beat at all (both self-couplings vanish), a clock qubit under
    # parallel polarizations keeps only the hyperfine difference of the detunings
    if abs(omega_ls) <= min_relative_force * max(abs(common), abs(omega_r), 1e-300):
        raise ValueError(
            f"ion {ion}: the two qubit levels see the same two-photon light shift under beams {beams} "
            f"(Omega_dndn/2pi = {dn:.4g} Hz, Omega_upup/2pi = {up:.4g} Hz, Raman Omega_R/2pi = {omega_r:.4g} Hz): "
            "no state-dependent force (Section 4.4.4)"
        )
    f_qubit, _slope, _curv = species.transition_frequency_hz(lower, upper, device.field.B_gauss)
    couplings = LightShiftCouplings(
        level_weights=(complex(dn / omega_ls), complex(up / omega_ls)),
        spin_flip_weight=complex(omega_r / omega_ls),
        qubit_freq_hz=float(f_qubit),
    )
    delta_k = np.asarray(device.beams[b1].k_vector() - device.beams[b2].k_vector(), dtype=float)
    etas, c0 = _etas(device, ion, delta_k)
    return DerivedDrive(
        ion=ion,
        kind="light_shift",
        beams=(b1, b2),
        rabi_hz=complex(omega_ls),
        delta_k=delta_k,
        etas=etas,
        c0_applied=c0,
        stark_shift_hz=differential_stark_shift_hz(device, ion, beams),
        scattering=scattering_budget(device, ion, beams) if scattering else None,
        micromotion=_micromotion(device, ion, delta_k),
        provenance=(
            "conv.light_shift_force",
            "conv.two_photon_rabi",
            "conv.effective_wavevector",
            "conv.lamb_dicke",
        ),
        light_shift=couplings,
    )


def light_shift_drive(
    derived: DerivedDrive,
    *,
    beat_hz: float,
    phase_rad: float = 0.0,
    envelope_hz: float | None = None,
    crosstalk: dict[int, complex] | None = None,
    include_stark: bool = True,
) -> Drive:
    """A single-beat-note light-shift Drive: envelope |Omega_LS| (or ``envelope_hz``), beat note ``beat_hz`` near a mode."""
    if derived.kind != "light_shift" or derived.light_shift is None:
        raise ValueError("light_shift_drive takes the DerivedDrive of derive_light_shift_drive")
    tone = Tone(
        detuning_hz=float(beat_hz),
        phase_rad=float(phase_rad),
        envelope_hz=float(derived.carrier_rabi_hz if envelope_hz is None else envelope_hz),
    )
    return Drive(
        kind="light_shift",
        ions=(derived.ion,),
        tones=(tone,),
        beams=derived.beams,
        stark_shift_hz=float(derived.stark_shift_hz) if include_stark else 0.0,
        crosstalk=dict(crosstalk or {}),
        light_shift=derived.light_shift,
    )


def crosstalk_ratios(
    device: Device, ion: int, beams: Sequence[int], *, kind: DriveKind = "raman"
) -> dict[int, complex]:
    """eps_ij = |Omega_j|/|Omega_i| of the other ions under the same beams (Section 6.6, a Rabi amplitude ratio)."""
    if kind == "raman":
        ref = derive_raman_drive(device, ion, (beams[0], beams[1]), scattering=False)
        others = {
            j: derive_raman_drive(device, j, (beams[0], beams[1]), scattering=False)
            for j in range(device.crystal.n_ions)
            if j != ion
        }
    else:
        ref = derive_optical_drive(device, ion, beams[0], scattering=False)
        others = {
            j: derive_optical_drive(device, j, beams[0], scattering=False)
            for j in range(device.crystal.n_ions)
            if j != ion
        }
    if ref.carrier_rabi_hz == 0.0:
        raise ZeroDivisionError("the addressed ion has no coupling")
    return {
        j: complex(d.carrier_rabi_hz / ref.carrier_rabi_hz)
        for j, d in others.items()
        if d.carrier_rabi_hz > 0.0
    }


def square_drive(
    derived: DerivedDrive,
    *,
    detuning_hz: float = 0.0,
    phase_rad: float = 0.0,
    rabi_scale: float = 1.0,
    crosstalk: dict[int, complex] | None = None,
    include_stark: bool = True,
    rf_locked: bool = False,
    rf_phase_rad: float | None = None,
) -> Drive:
    """A single-tone square Drive on the derived drive's ion: envelope |Omega| x rabi_scale, detuning mu, phase phi."""
    tone = Tone(
        detuning_hz=float(detuning_hz),
        phase_rad=float(phase_rad),
        envelope_hz=float(derived.carrier_rabi_hz * rabi_scale),
    )
    return Drive(
        kind=derived.kind,
        ions=(derived.ion,),
        tones=(tone,),
        beams=derived.beams,
        stark_shift_hz=float(derived.stark_shift_hz) if include_stark else 0.0,
        crosstalk=dict(crosstalk or {}),
        rf_locked=rf_locked,
        rf_phase_rad=rf_phase_rad,
    )


__all__ = [
    "DerivedDrive",
    "ScatteringBudget",
    "crosstalk_ratios",
    "derive_light_shift_drive",
    "derive_optical_drive",
    "derive_raman_drive",
    "differential_stark_shift_hz",
    "lamb_dicke_parameters",
    "light_shift_drive",
    "mathieu_or_none",
    "scattering_budget",
    "square_drive",
    "two_photon_self_couplings_hz",
]
