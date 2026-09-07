"""Photon-scattering collapse operators of a pulse: Raman spin flips, leakage, Rayleigh, differential-Rayleigh dephasing and
recoil (PLAN.md Sections 4.2.8, 4.5.5, 6.5, 12; Section 13 rows "Rayleigh dephasing dissipator", "Recoil kernel
discretization"; milestone M7).

For every ion the beams of a pulse illuminate (the addressed ions and the crosstalk neighbours, at the intensity each sees
from the beam profile) and every beam of the drive, the atomic layer supplies the signed Kramers-Heisenberg amplitudes
r_{a -> b, q'} (units sqrt(1/s)) of scattering a photon of emitted polarization q' from register level a to level b
(``AtomicStructure.scattering_amplitudes``, sqrt(Gamma_e) inside the coherent sum). Three kinds of jump operator follow,
one per emitted (polarization q', direction k_hat) mode:

- Rayleigh (b = a for every a): the DIAGONAL operator sum_a r_{aa q'} |a><a| (x) K: Uys et al. 2010's form, whose
  (r_up + r_dn)/2 part heats only and whose (r_up - r_dn)/2 sigma_z part dephases the qubit at Gamma_el/2 with the
  Gamma_el/4 dissipator prefactor of Section 13, both automatically; the plan's separate 1/2 sqrt(Gamma_el) sigma_z operator
  is the same channel for two levels.
- Raman a -> b with b a resolved register level: r_{ab q'} |b><a| (x) K, one operator per (a, b) (photons of different
  frequency are distinguishable environment states, so different final energies never share an operator).
- Leakage a -> outside the resolved levels: sqrt(sum_b |r_{ab q'}|^2) |SINK><a| (x) K when the register factor carries a
  SINK (d > 2, ``noise/levels.py``); at d = 2 the rate is reported as an estimate and no operator is built (Section 12).

K is the recoil operator: the absorbed photon's e^{+i k_b . x} (the beam's single-photon wavevector, C0 inside its
Lamb-Dicke parameters) times the emitted photon's e^{-i k_em k_hat . x}, so K = prod_m D_m(i(eta_abs,m - eta_em,m(k_hat)))
over the resolved modes. The absorption factor belongs in an ELIMINATED jump operator (the excited state is not in the
space) and is the plan's own construction **[background]**: Section 13's rule "absorption recoil only through e^{i k x} in
the Hamiltonian, never both in a collapse operator" is stated for the explicit multi-level builder, where absorption is a
Hamiltonian process; here a scattering event absorbs one photon of the drive and the net kick is hbar(k_b - k_em), Ozeri
2007's recoil operator. The emission direction is discretized (Section 4.2.8): ``minimal`` uses the six axis directions
about B with weights alpha_par/2 (+-B) and alpha_perp/2 (the four perpendicular ones), exact in the first and second moments
of every dipole pattern for every mode axis (sum_k w_k = alpha_par + 2 alpha_perp = 1); ``vector`` the product quadrature of
``light/recoil.py`` with the pattern density as weight; ``off`` no motional factor (Uys's and Ozeri's internal-state-only
dissipators, an approximation recorded in the notes).

Rates scale with the played intensity: the amplitudes are computed at the beams' configured power, and the pulse plays
Omega(t) against the derived nominal Omega_nom of the addressed ion, so every amplitude carries sqrt(s(t)) with
s(t) = sum_tones (Omega_k(t)/Omega_nom)^p and p = ``stark_scaling_power(kind)`` (1 for a two-photon drive, whose Rabi
frequency is proportional to the intensity, 2 for a single-photon one), a QobjEvo coefficient for a shaped pulse.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import qutip as qt

from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import stark_scaling_power
from qutip_trap.device.model import Device
from qutip_trap.dynamics.channels import CollapseOp
from qutip_trap.hilbert.operators import displacement_operator
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.light.raman import (
    derive_light_shift_drive,
    derive_optical_drive,
    derive_raman_drive,
    lamb_dicke_parameters,
    scattering_budget,
)
from qutip_trap.light.recoil import angular_factor, direction_quadrature, pattern_density, recoil_projections
from qutip_trap.noise.levels import InternalLevels, internal_levels
from qutip_trap.species.polarization import atomic_frame
from qutip_trap.species.raman import AtomicStructure

RecoilOption = Literal["off", "minimal", "vector"]


@dataclass(frozen=True)
class ScatteringOptions:
    recoil: RecoilOption = "minimal"
    n_theta: int = 3
    n_phi: int = 5
    """The ``vector`` quadrature (exact for the dipole patterns' second moments at these minima)."""
    neighbours: bool = True
    """Crosstalk neighbours scatter the light they see (at their own position on the beam profile)."""
    rayleigh: bool = True
    raman: bool = True


@dataclass(frozen=True)
class RecoilNode:
    weight: float
    k_hat: np.ndarray
    """Unit emission direction; the zero vector for ``recoil="off"``."""


def recoil_nodes(q: int, b_hat: Sequence[float], options: ScatteringOptions) -> tuple[RecoilNode, ...]:
    """The emission-direction quadrature for polarization ``q``: weights sum to one (Section 4.2.8)."""
    if options.recoil == "off":
        return (RecoilNode(1.0, np.zeros(3)),)
    b = np.asarray(b_hat, dtype=float)
    if options.recoil == "minimal":
        x, y, z = atomic_frame(b)
        a_par = angular_factor(q, 1.0)
        a_perp = angular_factor(q, 0.0)
        return (
            RecoilNode(a_par / 2.0, z),
            RecoilNode(a_par / 2.0, -z),
            RecoilNode(a_perp / 2.0, x),
            RecoilNode(a_perp / 2.0, -x),
            RecoilNode(a_perp / 2.0, y),
            RecoilNode(a_perp / 2.0, -y),
        )
    quad = direction_quadrature(options.n_theta, options.n_phi, axis=b)
    out = []
    for k, w in zip(quad.directions, quad.weights):
        out.append(RecoilNode(float(w) * float(pattern_density(q, float(np.dot(k, b)))), np.asarray(k)))
    total = sum(n.weight for n in out)
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(f"recoil node weights sum to {total}, not 1 (quadrature bug)")
    return tuple(out)


def nominal_rabi_hz(device: Device, ion: int, pulse: Pulse) -> float | None:
    """The derived carrier Rabi frequency of the addressed ion under the drive's beams at their configured power."""
    drive = pulse.drive
    try:
        if drive.kind == "raman":
            return float(
                derive_raman_drive(
                    device, ion, (drive.beams[0], drive.beams[1]), scattering=False
                ).carrier_rabi_hz
            )
        if drive.kind == "light_shift":
            return float(
                derive_light_shift_drive(
                    device, ion, (drive.beams[0], drive.beams[1]), scattering=False
                ).carrier_rabi_hz
            )
        if drive.kind in ("optical_E1", "optical_E2"):
            return float(derive_optical_drive(device, ion, drive.beams[0], scattering=False).carrier_rabi_hz)
    except (ValueError, ZeroDivisionError):
        return None
    return None


@dataclass
class _IntensityScale:
    """sqrt(s(t)) with s(t) = sum_k (|Omega_k(t - t_start)|/Omega_nom)^p."""

    t_start: float
    duration: float
    envelopes: list[Callable[[float], float]]
    omega_nom: float
    power: int

    def __call__(self, t: float) -> float:
        tau = min(max(t - self.t_start, 0.0), self.duration)
        s = sum((abs(env(tau)) / self.omega_nom) ** self.power for env in self.envelopes)
        return math.sqrt(s)


def _scale_coef(t: float, scale: _IntensityScale, **_: object) -> float:
    return float(scale(t))


def _envelope_fn(value: Any, duration: float) -> Callable[[float], float]:
    if callable(value):
        fn = value
        return lambda tau: float(fn(tau))
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
        grid = np.linspace(0.0, duration, arr.size)
        return lambda tau: float(np.interp(tau, grid, arr))
    const = float(value)
    return lambda tau: const


def intensity_scale(device: Device, pulse: Pulse) -> tuple[float | None, _IntensityScale | None, str | None]:
    """(constant sqrt s, or the time-dependent scale, note): how the played envelope scales the nominal scattering rates."""
    drive = pulse.drive
    omega_nom = nominal_rabi_hz(device, drive.ions[0], pulse)
    if omega_nom is None or omega_nom <= 0.0:
        return (
            1.0,
            None,
            (
                f"pulse {pulse.gate_id!r}: no derived Rabi frequency to scale the scattering rates with; rates taken at the beams' "
                "configured power"
            ),
        )
    p = stark_scaling_power(drive.kind)
    envs = [tone.envelope_hz for tone in drive.tones]
    constants = [float(e) for e in envs if not callable(e) and not isinstance(e, np.ndarray)]
    if len(constants) == len(envs):
        s = sum((abs(c) / omega_nom) ** p for c in constants)
        return math.sqrt(s), None, None
    fns = [_envelope_fn(e, pulse.duration_s) for e in envs]
    return None, _IntensityScale(pulse.t_start_s, pulse.duration_s, fns, omega_nom, p), None


def _kick(
    space: HilbertSpace,
    device: Device,
    ion: int,
    k_rad_per_m: float,
    k_hat: np.ndarray,
    etas_abs: Mapping[int, float],
    notes: list[str],
) -> dict[int, qt.Qobj]:
    """Per resolved mode factor: D_m(i(eta_abs - eta_em)) for the absorbed beam and the emission direction."""
    out: dict[int, qt.Qobj] = {}
    em: dict[int, float] = {}
    if float(np.linalg.norm(k_hat)) > 0.0:
        em = recoil_projections(
            device.crystal, ion, k_rad_per_m, tuple(k_hat), modes=[t.mode for t in space.resolved]
        )
    for tr in space.resolved:
        eta = float(etas_abs.get(tr.mode, 0.0)) - float(em.get(tr.mode, 0.0))
        if abs(eta) < 1e-15:
            continue
        if abs(eta) <= tr.eta_max * (1.0 + 1e-12):
            out[space.mode_factor(tr.mode)] = space.displacement_factor(tr.mode, eta)
        else:
            note = f"recoil kick |eta| = {abs(eta):.3g} on mode {tr.mode} exceeds the space's eta_max {tr.eta_max:.3g}: margin not asserted"
            if note not in notes:
                notes.append(note)
            out[space.mode_factor(tr.mode)] = displacement_operator(tr.d, 1j * eta)
    return out


@dataclass(frozen=True)
class _OpContext:
    """Everything one scattering operator needs besides its internal matrix."""

    space: HilbertSpace
    ion: int
    d: int
    kick: dict[int, qt.Qobj]
    weight: float
    const_scale: float | None
    time_scale: _IntensityScale | None
    t_start: float


def _make_op(ctx: _OpContext, matrix: np.ndarray, name: str) -> CollapseOp:
    internal = qt.Qobj(matrix, dims=[[ctx.d], [ctx.d]])
    factors: dict[int, qt.Qobj] = {ctx.space.ion_factor(ctx.ion): internal}
    factors.update(ctx.kick)
    op = ctx.space.embed_many(factors).to("CSR")
    root_w = math.sqrt(ctx.weight)
    scale_rate = ctx.weight * float(np.max(np.linalg.eigvalsh(matrix.conj().T @ matrix)))
    if ctx.time_scale is None:
        assert ctx.const_scale is not None
        full: qt.Qobj | qt.QobjEvo = (root_w * ctx.const_scale) * op
        rate = scale_rate * ctx.const_scale**2
    else:
        full = qt.QobjEvo([root_w * op, qt.coefficient(_scale_coef, args={"scale": ctx.time_scale})])
        rate = scale_rate * float(ctx.time_scale(ctx.t_start)) ** 2
    return CollapseOp(full, float(rate / (2.0 * math.pi)), name, ctx.ion, None)


def scattering_channels(
    device: Device,
    pulse: Pulse,
    space: HilbertSpace,
    *,
    levels_by_ion: Mapping[int, InternalLevels] | None = None,
    options: ScatteringOptions | None = None,
) -> tuple[tuple[CollapseOp, ...], tuple[str, ...]]:
    """The scattering collapse operators of one pulse on ``space`` and the notes (dropped leakage, approximations)."""
    opts = options or ScatteringOptions()
    drive = pulse.drive
    if drive.kind in ("microwave", "gradient") or not drive.beams:
        return (), ()
    notes: list[str] = []
    const_scale, time_scale, note = intensity_scale(device, pulse)
    if note:
        notes.append(note)
    if opts.recoil == "off":
        notes.append(f"pulse {pulse.gate_id!r}: scattering operators carry no recoil (internal-state only)")
    ions = list(drive.ions)
    if opts.neighbours:
        ions += [j for j in drive.crosstalk if j not in ions]
    outside = [j for j in ions if not space.has_ion(j)]
    if outside:
        # a GATE_LOCAL space carries a subset of the ions (Section 5.4): neighbours outside it scatter in reality, not here
        notes.append(f"pulse {pulse.gate_id!r}: scattering of ions {outside} outside the space dropped")
        ions = [j for j in ions if space.has_ion(j)]
    b_hat = (
        float(device.field.direction[0]),
        float(device.field.direction[1]),
        float(device.field.direction[2]),
    )
    ops: list[CollapseOp] = []
    dropped_leak: dict[int, float] = {}
    for ion in ions:
        d = space.ion_dim(ion)
        species = device.crystal.species[ion]
        lev = (
            levels_by_ion[ion]
            if levels_by_ion is not None and ion in levels_by_ion
            else internal_levels(species, d, device.field.B_gauss, b_hat)
        )
        if lev.d != d:
            raise ValueError(f"ion {ion}: the level map has {lev.d} levels but the register factor has {d}")
        st = AtomicStructure(species, device.field.B_gauss, b_hat)
        pos = [float(x) for x in device.crystal.positions_m[ion]]
        for b in drive.beams:
            beam = device.beams[b]
            amps = {idx: st.scattering_amplitudes(st.state(lab), beam, pos) for idx, lab in lev.atomic}
            etas_abs, _c0 = lamb_dicke_parameters(device, ion, np.asarray(beam.k_vector(), dtype=float))
            for q in (-1, 0, 1):
                for node in recoil_nodes(q, b_hat, opts):
                    kick = _kick(space, device, ion, beam.k_rad_per_m, node.k_hat, etas_abs, notes)
                    ctx = _OpContext(
                        space, ion, d, kick, node.weight, const_scale, time_scale, pulse.t_start_s
                    )
                    if opts.rayleigh:
                        diag = np.zeros((d, d), dtype=complex)
                        for idx, lab in lev.atomic:
                            diag[idx, idx] = amps[idx].get((lab, q), 0.0 + 0.0j)
                        if np.any(diag != 0.0):
                            ops.append(_make_op(ctx, diag, f"scatter_rayleigh[beam {b}][q'={q:+d}]"))
                    if not opts.raman:
                        continue
                    for idx, lab in lev.atomic:
                        leak2 = 0.0
                        for (b_lab, qq), r in amps[idx].items():
                            if qq != q or b_lab == lab:
                                continue
                            b_idx = lev.index(b_lab)
                            if b_idx is None:
                                leak2 += abs(r) ** 2
                                continue
                            m = np.zeros((d, d), dtype=complex)
                            m[b_idx, idx] = r
                            ops.append(
                                _make_op(ctx, m, f"scatter_raman[beam {b}][{lab} -> {b_lab}][q'={q:+d}]")
                            )
                        if leak2 > 0.0:
                            if lev.has_sink:
                                sink = lev.sink_index
                                assert sink is not None
                                m = np.zeros((d, d), dtype=complex)
                                m[sink, idx] = math.sqrt(leak2)
                                ops.append(
                                    _make_op(ctx, m, f"scatter_leak[beam {b}][{lab} -> SINK][q'={q:+d}]")
                                )
                            else:
                                dropped_leak[ion] = dropped_leak.get(ion, 0.0) + node.weight * leak2
    for ion, rate in dropped_leak.items():
        s2 = (
            const_scale**2
            if const_scale is not None
            else (time_scale(pulse.t_start_s) ** 2 if time_scale else 1.0)
        )
        notes.append(
            f"pulse {pulse.gate_id!r}, ion {ion}: leakage out of the qubit pair at {rate * s2:.3g} s^-1 "
            f"({rate * s2 * pulse.duration_s:.3g} per pulse) has no register level at d = 2 and is reported, not simulated (Section 12)"
        )
    return tuple(ops), tuple(notes)


def scattering_estimates(device: Device, pulse: Pulse) -> dict[str, float]:
    """Per-pulse probabilities of the Section 9.7 row 'Scattering' for the addressed ion(s): P_raman (spin flip), P_leak (out
    of the pair, the D-level branching f P_total included when the species table resolves the D channel), P_rayleigh, and the
    differential-Rayleigh coherence loss Gamma_el t/2 (the coherence decays at Gamma_el/2); all at the played intensity."""
    drive = pulse.drive
    out: dict[str, float] = {}
    if drive.kind in ("microwave", "gradient") or not drive.beams:
        return out
    const_scale, time_scale, _note = intensity_scale(device, pulse)
    if time_scale is None:
        assert const_scale is not None
        s_int = const_scale**2 * pulse.duration_s
    else:
        grid = np.linspace(pulse.t_start_s, pulse.t_end_s, 401)
        s_int = float(np.trapezoid([time_scale(t) ** 2 for t in grid], grid))
    for ion in drive.ions:
        budget = scattering_budget(device, ion, drive.beams)
        keys = list(budget.raman_spin_flip_per_s)
        flip = float(np.mean([budget.raman_spin_flip_per_s[k] for k in keys])) * s_int
        leak = float(np.mean([budget.leakage_per_s[k] for k in keys])) * s_int
        ray = float(np.mean([budget.rayleigh_per_s[k] for k in keys])) * s_int
        out[f"ion{ion}.P_raman"] = flip
        out[f"ion{ion}.P_leak"] = leak
        out[f"ion{ion}.P_rayleigh"] = ray
        out[f"ion{ion}.rayleigh_dephasing"] = 0.5 * budget.rayleigh_dephasing_per_s * s_int
    return out


__all__ = [
    "RecoilNode",
    "RecoilOption",
    "ScatteringOptions",
    "intensity_scale",
    "nominal_rabi_hz",
    "recoil_nodes",
    "scattering_channels",
    "scattering_estimates",
]
