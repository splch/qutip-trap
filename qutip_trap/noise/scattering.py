"""Photon-scattering collapse operators of a pulse: Raman spin flips, leakage, Rayleigh dephasing and recoil (PLAN.md
Section 6.5).

For every ion a pulse illuminates (the addressed ions and the crosstalk neighbours, at the intensity each sees) and every
beam of the drive, the atomic layer supplies the Kramers-Heisenberg amplitudes r_{a -> b, q'} (sqrt(1/s)) of scattering a
photon of emitted polarization q' from register level a to b. One jump operator per emitted (q', k_hat) mode follows:

- Rayleigh: the diagonal sum_a r_{aa q'} |a><a| (x) K (Uys et al. 2010), whose (r_up - r_dn)/2 sigma_z part dephases the
  qubit at Gamma_el/2 and whose common part only heats;
- Raman a -> b to a resolved level: r_{ab q'} |b><a| (x) K, one operator per (a, b), photons of different frequency being
  distinguishable;
- leakage a -> outside the resolved levels: sqrt(sum_b |r_{ab q'}|^2) |SINK><a| (x) K when the register carries a SINK
  (d > 2); at d = 2 the rate is reported and no operator is built.

K = prod_m D_m(i(eta_abs,m - eta_em,m(k_hat))) is the recoil of the absorbed beam photon and the emitted one, hbar(k_b - k_em)
(Ozeri 2007). The emission direction is discretized: ``minimal`` uses the six axis directions about B with weights
alpha_par/2 and alpha_perp/2, exact in the first and second moments of every dipole pattern; ``vector`` a product
quadrature weighted by the pattern density; ``off`` no motional factor. Every amplitude carries sqrt(s(t)) with
s(t) = sum_tones (Omega_k(t)/Omega_nom)^p, p = 1 for a two-photon and 2 for a single-photon drive.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import qutip as qt

from qutip_trap.control.pulses import ConstantFn, InterpFn, Pulse, ScaledFn
from qutip_trap.control.schedule import stark_scaling_power
from qutip_trap.device.model import Device
from qutip_trap.dynamics.channels import CollapseOp, RecoilOption
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
from qutip_trap.species.raman import structure_at


@dataclass(frozen=True)
class ScatteringOptions:
    """The recoil discretization of the scattering operators: ``off``, ``minimal`` (the six axis directions about B) or
    ``vector`` (the 3 x 5 product quadrature, exact for the dipole patterns' second moments)."""

    recoil: RecoilOption = "minimal"


@dataclass(frozen=True)
class RecoilNode:
    weight: float
    k_hat: np.ndarray
    """Unit emission direction; the zero vector for ``recoil="off"``."""


def recoil_nodes(q: int, b_hat: Sequence[float], options: ScatteringOptions) -> tuple[RecoilNode, ...]:
    """The emission-direction quadrature for polarization ``q``; the weights sum to one."""
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
    quad = direction_quadrature(3, 5, axis=b)
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
    """A picklable callable of tau: the callable itself (as a float), a linear interpolation of a sampled array, a constant."""
    if callable(value):
        return ScaledFn(value, 1.0)
    if isinstance(value, np.ndarray):
        return InterpFn(np.asarray(value, dtype=float), duration)
    return ConstantFn(float(value))


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
    # a scattering rate is already s^-1: no 2 pi
    return CollapseOp(full, float(rate), name, ctx.ion, None)


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
    ions = list(drive.ions) + [j for j in drive.crosstalk if j not in drive.ions]
    outside = [j for j in ions if not space.has_ion(j)]
    if outside:
        # a gate-local space carries a subset of the ions: neighbours outside it scatter in reality, not here
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
        st = structure_at(species, device.field.B_gauss, b_hat)
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
                    diag = np.zeros((d, d), dtype=complex)
                    for idx, lab in lev.atomic:
                        diag[idx, idx] = amps[idx].get((lab, q), 0.0 + 0.0j)
                    if np.any(diag != 0.0):
                        ops.append(_make_op(ctx, diag, f"scatter_rayleigh[beam {b}][q'={q:+d}]"))
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
            f"({rate * s2 * pulse.duration_s:.3g} per pulse) has no register level at d = 2 and is reported, not simulated"
        )
    return tuple(ops), tuple(notes)


def scattering_estimates(device: Device, pulse: Pulse) -> dict[str, float]:
    """Per-pulse probabilities for the addressed ion(s) at the played intensity: P_raman (spin flip), P_leak (out of the
    pair), P_rayleigh, and the differential-Rayleigh coherence loss Gamma_el t/2."""
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
