"""Derived drives: two-photon Rabi frequency, effective Delta k, per-mode eta, Stark shift and scattering from the beams.

Omega_R = sum_e conj(Omega_2) Omega_1/(2 Delta_e) in the (hbar Omega/2) convention, beam 1 the higher-frequency beam, so
the beat note omega_1 - omega_2 drives |down> -> |up> with momentum Delta k = k_1 - k_2 (Wineland 2003 Eq. 2.3).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qutip_trap.control.pulses import Drive, DriveKind, LightShiftCouplings, Tone
from qutip_trap.device.model import Device
from qutip_trap.light.comb import CombSpec
from qutip_trap.species.raman import AtomicStructure, structure_at
from qutip_trap.trap.mathieu import MathieuParameters
from qutip_trap.trap.micromotion import MicromotionIndex
from qutip_trap.units import TWO_PI

M2 = "milestone M2 (light/raman.py, PLAN.md Section 4.3.2)"


@dataclass(frozen=True)
class ScatteringBudget:
    """Scattering rates per second at the ion's position, summed over the drive's beams."""

    rayleigh_per_s: dict[str, float]
    """Elastic rate from each qubit state (keyed by its label)."""
    raman_spin_flip_per_s: dict[str, float]
    """Rate from each qubit state into the other qubit state."""
    leakage_per_s: dict[str, float]
    """Rate from each qubit state out of the qubit pair, summed over final states."""
    rayleigh_dephasing_per_s: float
    """Gamma_el of the pair from the elastic-amplitude difference squared (Uys et al. 2010)."""
    residual_excited_population: float

    def per_pulse_error(self, duration_s: float) -> float:
        """The Raman (spin-flip plus leakage) probability over ``duration_s``, averaged over the qubit states."""
        rates = [self.raman_spin_flip_per_s[k] + self.leakage_per_s[k] for k in self.raman_spin_flip_per_s]
        return float(np.mean(rates)) * duration_s if rates else 0.0


@dataclass(frozen=True)
class DerivedDrive:
    """What ``light/`` derives for one ion under one physical drive."""

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
    """Differential light shift of the transition: shift of the upper level minus the lower one, Hz."""
    scattering: ScatteringBudget | None
    micromotion: MicromotionIndex | None
    provenance: tuple[str, ...]
    light_shift: LightShiftCouplings | None = None
    """For ``kind == "light_shift"``: the level and spin-flip weights; ``rabi_hz`` is then Omega_LS/2pi."""

    @property
    def carrier_rabi_hz(self) -> float:
        return abs(self.rabi_hz)

    def pi_time_s(self, area_rad: float = math.pi) -> float:
        """The duration with Omega t = ``area_rad`` (a carrier pi pulse at Omega t = pi)."""
        if self.carrier_rabi_hz == 0.0:
            raise ZeroDivisionError("the drive has no coupling at this ion")
        return area_rad / (TWO_PI * self.carrier_rabi_hz)

    def eta_along(self, mode: int) -> float:
        return self.etas[mode]


def _position(device: Device, ion: int) -> list[float]:
    return [float(x) for x in device.crystal.positions_m[ion]]


def _structure(device: Device, ion: int) -> AtomicStructure:
    species = device.crystal.species[ion]
    return structure_at(species, device.field.B_gauss, device.field.direction)


def mathieu_or_none(device: Device, ion: int) -> MathieuParameters | None:
    """The trap's Mathieu record for the ion's species, or None when the trap has no rf record (C0 = 1, beta = 0)."""
    species = device.crystal.species[ion]
    try:
        return device.trap.mathieu(species)
    except ValueError:
        return None


def lamb_dicke_parameters(device: Device, ion: int, delta_k: np.ndarray) -> tuple[dict[int, float], bool]:
    """eta_{ion, m} for every crystal mode with C0 from the trap's Mathieu record when it exists; (etas, c0_applied)."""
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
    """Rayleigh, spin-flip and leakage rates of both qubit states under the beams, at the ion's position."""
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
    """sum_beams [delta(upper) - delta(lower)]/2pi at the ion's position: red light lowers a level."""
    st = _structure(device, ion)
    species = device.crystal.species[ion]
    pos = _position(device, ion)
    beams = [device.beams[b] for b in beam_indices]
    lower, upper = species.qubit
    return (
        st.light_shift_rad_s(st.state(upper), beams, pos) - st.light_shift_rad_s(st.state(lower), beams, pos)
    ) / TWO_PI


def quadrupole_stark_shift_hz(device: Device, ion: int, beam: int) -> float:
    """delta_St/2pi of a driven E2 component from the other components inside the Zeeman span, to second order (I = 0
    only; no published number validates it)."""
    from qutip_trap.species.model import level_j
    from qutip_trap.species.quadrupole import (
        e2_stark_shift_rad_s,
        rabi_frequency_e2_rad_s,
        reduced_element_from_lifetime_m2,
    )
    from qutip_trap.species.zeeman import parse_quantum_numbers

    species = device.crystal.species[ion]
    if species.nuclear_spin != 0.0:
        raise NotImplementedError(
            "hyperfine-resolved E2 couplings are not specified (Section 4.5.7 treats I = 0)"
        )
    lower, upper = species.qubit
    lo_level, up_level = lower.split()[0], upper.split()[0]
    e2 = next(
        t for t in species.transitions if t.multipole == "E2" and {t.lower, t.upper} == {lo_level, up_level}
    )
    st = _structure(device, ion)
    b = device.beams[beam]
    pos = _position(device, ion)
    e0 = st.field_amplitude(b, pos)
    red = reduced_element_from_lifetime_m2(e2.wavelength_vac_m, e2.partial_rate_rad_s, level_j(e2.upper))
    j_lo, j_up = level_j(e2.lower), level_j(e2.upper)
    lower_states = {parse_quantum_numbers(s.label)["mJ"]: s.energy_hz for s in st.states_of(e2.lower)}
    upper_states = {parse_quantum_numbers(s.label)["mJ"]: s.energy_hz for s in st.states_of(e2.upper)}
    couplings = {
        (m, mp): rabi_frequency_e2_rad_s(
            e0, b.wavelength_m, red, j_lo, m, j_up, mp, b.polarization, b.k_hat, device.field.direction
        )
        for m in lower_states
        for mp in upper_states
    }
    m_driven = parse_quantum_numbers(lower.split(" ", 1)[1])["mJ"]
    mp_driven = parse_quantum_numbers(upper.split(" ", 1)[1])["mJ"]
    if e2.lower != lo_level:  # the qubit pair is written upper-first
        m_driven, mp_driven = mp_driven, m_driven
    return e2_stark_shift_rad_s(couplings, lower_states, upper_states, m_driven, mp_driven) / TWO_PI


_DERIVED: dict[tuple[object, ...], tuple[Device, DerivedDrive]] = {}
"""Derived drives keyed by (kind, device identity, ion, beams, scattering), the device kept alive so its id cannot be reused."""
_DERIVED_MAX = 4096


def _derived_memo(device: Device, key: tuple[object, ...]) -> DerivedDrive | None:
    entry = _DERIVED.get(key)
    if entry is not None and entry[0] is device:
        return entry[1]
    return None


def _remember(device: Device, key: tuple[object, ...], value: DerivedDrive) -> DerivedDrive:
    if len(_DERIVED) >= _DERIVED_MAX:
        _DERIVED.clear()
    _DERIVED[key] = (device, value)
    return value


def derive_raman_drive(
    device: Device, ion: int, beams: tuple[int, int], *, scattering: bool = True
) -> DerivedDrive:
    """A stimulated-Raman drive of ``ion`` by (beam 1, beam 2): Omega_R, Delta k = k_1 - k_2, etas, Stark shift and
    scattering, memoized per device instance."""
    key = ("raman", id(device), int(ion), (int(beams[0]), int(beams[1])), bool(scattering))
    hit = _derived_memo(device, key)
    if hit is not None:
        return hit
    return _remember(device, key, _derive_raman_drive(device, ion, beams, scattering=scattering))


def _derive_raman_drive(
    device: Device, ion: int, beams: tuple[int, int], *, scattering: bool = True
) -> DerivedDrive:
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
    """A single-photon optical drive (E1, or E2 for an I = 0 quadrupole qubit), memoized per device instance."""
    key = ("optical", id(device), int(ion), (int(beam),), bool(scattering))
    hit = _derived_memo(device, key)
    if hit is not None:
        return hit
    return _remember(device, key, _derive_optical_drive(device, ion, beam, scattering=scattering))


def _derive_optical_drive(device: Device, ion: int, beam: int, *, scattering: bool = True) -> DerivedDrive:
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
        # the E2 shift is second order in the other components, not zero
        stark_shift_hz=(
            quadrupole_stark_shift_hz(device, ion, beam)
            if kind == "optical_E2"
            else differential_stark_shift_hz(device, ion, (beam,))
        ),
        scattering=budget,
        micromotion=_micromotion(device, ion, delta_k),
        provenance=(
            "conv.quadrupole_coupling" if kind == "optical_E2" else "conv.rabi_from_intensity",
            *(("conv.e2_ac_stark_shift",) if kind == "optical_E2" else ()),
            "conv.lamb_dicke",
        ),
    )


def two_photon_self_couplings_hz(device: Device, ion: int, beams: tuple[int, int]) -> tuple[complex, complex]:
    """(Omega_dndn, Omega_upup)/2pi: each qubit level's Raman coupling to itself under (beam 1, beam 2); half their
    difference is the light-shift gate's state-dependent force."""
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
    """The light-shift (sigma_z sigma_z) gate drive Omega_LS = (Omega_upup - Omega_dndn)/2 by (beam 1, beam 2) (Zhu 2006
    Eq. 2); raises when |Omega_LS| is at most ``min_relative_force`` of the spin-independent part or of Omega_R."""
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
    # fails for crossed linear polarizations (no intensity beat) and a clock qubit under parallel ones (hyperfine only)
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
    """eps_ij = |Omega_j|/|Omega_i|: the Rabi amplitude ratio of each other coupled ion under the same beams."""
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


def comb_drive(
    derived: DerivedDrive,
    comb: CombSpec,
    *,
    omega_q_hz: float,
    gate_time_s: float,
    mode_hz: float | None = None,
    sideband: int = 0,
    phase_rad: float = 0.0,
    levels_hz: Sequence[float] | None = None,
    couplings_hz: Sequence[float] | None = None,
    n_index: int = 0,
    rabi_scale: float = 1.0,
    crosstalk: dict[int, complex] | None = None,
    nbar: float = 0.0,
    detuning_hz: float | None = None,
    fine_structure_hz: float | None = None,
    theta_per_pulse_rad: float | None = None,
) -> Drive:
    """A mode-locked (frequency-comb) Raman drive: tones from ``comb.tones()`` and, given ``levels_hz`` and
    ``couplings_hz`` (Hz), the static shift from ``comb.stark4_hz()``; failing or unevaluated validity guards warn."""
    if derived.kind != "raman":
        raise ValueError("a frequency comb generates a Raman drive (Section 4.3.7)")
    if gate_time_s <= 0.0:
        raise ValueError("gate_time_s is a positive duration")
    tones = comb.tones(
        omega_q_hz,
        mode_hz,
        omega0_hz=derived.carrier_rabi_hz * rabi_scale,
        gate_time_s=gate_time_s,
        phase_rad=phase_rad,
        sideband=sideband,
    )
    if not tones:
        raise ValueError(
            "the comb has no beat note inside the gate window: check aom_offset_hz against nu_q "
            f"({comb.resonance_target_hz(omega_q_hz, mode_hz, sideband):.6g} Hz) and the rep rate"
        )
    shift_hz = 0.0
    if levels_hz is not None or couplings_hz is not None:
        if levels_hz is None or couplings_hz is None:
            raise ValueError("the fourth-order shift needs both levels_hz and couplings_hz")
        shift_hz = float(
            comb.stark4_hz(
                levels_hz,
                couplings_hz=couplings_hz,
                n_index=n_index,
                gate_time_s=gate_time_s,
                resonance_hz=comb.resonance_target_hz(omega_q_hz, mode_hz, sideband),
            )[0]
        )
    # the comb's validity guards run here, the one place every input exists
    eta = max((abs(v) for v in derived.etas.values()), default=0.0)
    failing = [
        name
        for name, ok in comb.guards(
            omega_q_hz,
            eta,
            nbar,
            gate_time_s,
            abs(mode_hz) if mode_hz else 0.0,
            detuning_hz=detuning_hz,
            fine_structure_hz=fine_structure_hz,
            theta_per_pulse_rad=theta_per_pulse_rad,
        ).items()
        if not ok
    ]
    missing = comb.unevaluated_guards(detuning_hz=detuning_hz, theta_per_pulse_rad=theta_per_pulse_rad)
    if failing or missing:
        warnings.warn(
            "comb drive on ion "
            f"{derived.ion}: pulse-train-to-continuous-wave guards failing {failing or 'none'}, not evaluated "
            f"{list(missing) or 'none'} (Section 4.3.7's validity hierarchy; the builder records the clauses it can "
            "check in BuiltHamiltonian.approximations)",
            RuntimeWarning,
            stacklevel=2,
        )
    return Drive(
        kind="raman",
        ions=(derived.ion,),
        tones=tones,
        beams=derived.beams,
        stark_shift_hz=shift_hz,
        crosstalk=dict(crosstalk or {}),
        comb=comb,
    )
