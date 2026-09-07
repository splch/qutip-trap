"""Entangling-gate calibration by exact spot checks, and the thermal robustness curve (PLAN.md Sections 4.4.1, 4.4.7, 7.5, 9.4; M4).

Section 7.5: the default calibration is a surrogate, the closed-form alpha_m and chi_m integrals of Section 4.4.3 on the
mode structure (``control.shaping``), corrected by a small number of exact spot checks through the ``PulseEngine`` protocol.
This module is the spot check: it plays a Waveform on the pair through the JOINT_EXACT engine from |00>|n = 0> (or from the
thermal state, the ``reference`` the plan asks the device to record, Section 4.4.7 (1)), reads |chi| from P_11 = sin^2 chi
(an equatorial two-body rotation exp(+-i chi sigma_phi sigma_phi) takes |00> to cos chi |00> -/+ i e^{...} sin chi |11>,
so P_11 is the entangling angle and P_01 + P_10 the leakage from open loops and off-resonant excitation), rescales every
amplitude by sqrt(chi_target/chi) (chi ~ Omega^2, the s^2 law) and iterates. The residual displacement it reports is
exact: for the |00> input the final mean excitation of a mode is sum_j |alpha_jm|^2 (the spin-conditioned displacements
+-alpha_j add in quadrature, <a> alone averages to zero), which is eps_ent at nbar = 0.

This package sits above control/ and dynamics/ (Section 3.2) and never writes the CalibrationTable directly: it returns
the corrected Waveform and the checks behind it for calibrate() (M8) to store.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import qutip as qt

from qutip_trap.control.native import ms as native_ms
from qutip_trap.control.native import zz as native_zz
from qutip_trap.control.schedule import (
    GateDrive,
    Schedule,
    entangling_pulses,
    frame_after,
    ms_spin_phases,
    single_qubit_pulse,
)
from qutip_trap.control.shaping import CHI_MAXIMAL_RAD, GateModes, scaled, waveform_integrals
from qutip_trap.control.table import CalEntry, Waveform
from qutip_trap.dynamics.engine import EngineReport, JointExactEngine, SeedSpec, SolverOptions, Traces
from qutip_trap.dynamics.frames import PhaseFrame
from qutip_trap.hilbert.operators import required_margin
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.noise.sampling import NoiseSample, quiet_sample
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.hamiltonian import BuilderOptions

Reference = Literal["n0", "thermal"]


def gate_space(
    modes: GateModes,
    n_ions: int,
    *,
    nbar: Mapping[int, float] | None = None,
    waveform: Waveform | None = None,
    frozen: Sequence[int] = (),
    n_modes_total: int | None = None,
    d_min: int = 6,
    d_max: int = 64,
    extra_levels: int = 0,
) -> HilbertSpace:
    """A joint space resolving every mode of ``modes``: the cap follows the loop radius (the S = +-2 branch moves |alpha| = eta
    Omega/eps out from the origin, so n up to about (|alpha| + sqrt(nbar))^2 + 3 sqrt(nbar + 1) is populated) plus the Section
    5.1.1 margin for the mode's eta; every other crystal mode is frozen. ``waveform`` supplies the peak amplitude per mode."""
    nb = dict(nbar or {})
    resolved: list[ModeTruncation] = []
    for k, m in enumerate(modes.modes):
        eta_max = max(abs(modes.eta[i][k]) for i in modes.ions)
        radius = 0.0
        if waveform is not None and waveform.segments is not None:
            peak = 0.0
            for seg in waveform.segments:
                for amp in seg.amplitude_hz.values():
                    if callable(amp):
                        grid = np.linspace(0.0, seg.duration_s, 201)
                        peak = max(peak, float(np.max(np.abs([amp(x) for x in grid]))))
                    else:
                        peak = max(peak, abs(float(amp)))
                mu = seg.detuning_hz["blue"]
                mu_val = float(mu(0.0)) if callable(mu) else float(mu)
                eps = abs(modes.omega_rad_s[k] - TWO_PI * abs(mu_val))
                if eps > 0.0:
                    radius = max(radius, eta_max * TWO_PI * peak / eps)
        n_th = nb.get(m, 0.0)
        n_hi = int(
            math.ceil((radius + math.sqrt(n_th)) ** 2 + 2.0 * math.sqrt(radius**2 + n_th + 0.25) + 1.0)
        )
        d = min(max(n_hi + 1 + required_margin(eta_max) + extra_levels, d_min), d_max)
        resolved.append(ModeTruncation(m, d, (0, min(n_hi, d - 1)), max(eta_max * 1.5, 1e-3)))
    total = n_modes_total if n_modes_total is not None else 3 * n_ions
    frozen_all = tuple(sorted(set(frozen) | {m for m in range(total) if m not in modes.modes}))
    return HilbertSpace(tuple(2 for _ in range(n_ions)), tuple(resolved), None, frozen_all)


def ms_schedule(
    waveform: Waveform,
    pair: tuple[int, int],
    gate_drives: Mapping[int, GateDrive],
    table: CalibrationTable,
    *,
    phases_rad: tuple[float, float] = (0.0, 0.0),
    t0_s: float = 0.0,
    gate_id: str = "ms",
    response_delay_s: float = 0.0,
) -> Schedule:
    """The bare MS(phi_0, phi_1, .) pulse train of ``waveform`` on ``pair`` (no rescaling), as the scheduler would play it;
    ``response_delay_s`` is the modulator delay the tone phases compensate (``control.schedule.response_phase_rad``, M7)."""
    n = max(pair) + 1
    if waveform.kind == "ms":
        spins, _chi = ms_spin_phases(waveform, pair, phases_rad, PhaseFrame())
    else:
        spins = {pair[0]: 0.0, pair[1]: 0.0}
    pulses = entangling_pulses(
        waveform,
        dict(gate_drives),
        spin_phases_rad=spins,
        t_start_s=t0_s,
        table=table,
        gate_id=gate_id,
        response_delay_s=response_delay_s,
    )
    frame = frame_after(pulses, PhaseFrame())
    return Schedule(tuple(pulses), (), (), frame.as_dict(n))


def light_shift_echo_schedule(
    waveform: Waveform,
    pair: tuple[int, int],
    gate_drives: Mapping[int, GateDrive],
    single_qubit_drives: Mapping[int, GateDrive],
    table: CalibrationTable,
    *,
    dead_time_s: float,
    t0_s: float = 0.0,
    response_delay_s: float = 0.0,
    gate_id: str = "zz",
) -> Schedule:
    """Section 4.4.4's spin-echo form of the sigma_z sigma_z gate: the waveform, GPi(0) on both ions, the waveform again, GPi(pi) on
    both (R_x(pi) U R_{-x}(pi) U); the single-qubit sigma_z phases of the light-shift force (four times its two-body angle for a
    force on one level) cancel and the two-body angle doubles."""
    n = max(pair) + 1
    from qutip_trap.control.schedule import carrier_rabi_hz

    spins = {pair[0]: 0.0, pair[1]: 0.0}
    pulses = list(
        entangling_pulses(
            waveform,
            dict(gate_drives),
            spin_phases_rad=spins,
            t_start_s=t0_s,
            table=table,
            response_delay_s=response_delay_s,
            gate_id=f"{gate_id}/loop1",
        )
    )
    idle: list[tuple[float, float]] = []
    t = t0_s + waveform.duration_s
    idle.append((t, t + dead_time_s))
    t += dead_time_s
    ends = []
    for q in pair:
        p = single_qubit_pulse(
            q,
            math.pi,
            0.0,
            single_qubit_drives[q],
            carrier_rabi_hz(table, q, single_qubit_drives[q]),
            t,
            gate_id=f"{gate_id}/echo/ion{q}",
        )
        pulses.append(p)
        ends.append(p.t_end_s)
    t = max(ends)
    idle.append((t, t + dead_time_s))
    t += dead_time_s
    pulses.extend(
        entangling_pulses(
            waveform,
            dict(gate_drives),
            spin_phases_rad=spins,
            t_start_s=t,
            table=table,
            response_delay_s=response_delay_s,
            gate_id=f"{gate_id}/loop2",
        )
    )
    t += waveform.duration_s
    idle.append((t, t + dead_time_s))
    t += dead_time_s
    for q in pair:
        p = single_qubit_pulse(
            q,
            math.pi,
            math.pi,
            single_qubit_drives[q],
            carrier_rabi_hz(table, q, single_qubit_drives[q]),
            t,
            gate_id=f"{gate_id}/unecho/ion{q}",
        )
        pulses.append(p)
    return Schedule(tuple(pulses), tuple(idle), (), frame_after(pulses, PhaseFrame()).as_dict(n))


@dataclass(frozen=True)
class GateCheck:
    """One exact spot check of an entangling waveform (Section 7.5)."""

    populations: dict[str, float]
    """P_00, P_01, P_10, P_11 in the computational basis (0 = lower qubit level)."""
    chi_rad: float
    """|chi| = arcsin(sqrt(P_11)) of the |00> input."""
    leakage: float
    """P_01 + P_10: open loops and off-resonant excitation."""
    residual_quanta: dict[int, float]
    """Final mean excitation minus the initial one, per resolved mode: sum_j |alpha_jm|^2 at nbar = 0."""
    fidelity: float
    """Overlap of the reduced internal state with the ideal native gate's output for the same input."""
    internal: qt.Qobj
    report: EngineReport | None


def _internal_reference(n_ions: int, pair: tuple[int, int]) -> tuple[int, int]:
    return int(pair[0]), int(pair[1])


def _ideal_target(
    kind: str,
    chi_target_rad: float,
    phases_rad: tuple[float, float],
    n_ions: int,
    pair: tuple[int, int],
    internal: Sequence[int] | qt.Qobj,
) -> qt.Qobj:
    """The native gate's output for the input ``internal``: MS(phi_0, phi_1, 2 chi) or ZZ(2 chi) on the pair, identity elsewhere."""
    a, b = _internal_reference(n_ions, pair)
    if kind == "ms":
        mat = native_ms(phases_rad[0], phases_rad[1], 2.0 * chi_target_rad)
    else:
        mat = native_zz(4.0 * chi_target_rad)  # two echo loops of chi_target each
    u_pair = np.asarray(mat, dtype=complex).reshape(2, 2, 2, 2)  # (a', b', a, b)
    full = np.zeros([2] * (2 * n_ions), dtype=complex)
    others = [i for i in range(n_ions) if i not in (a, b)]
    for idx in np.ndindex(*([2] * n_ions)):
        for jdx in np.ndindex(*([2] * n_ions)):
            if any(idx[o] != jdx[o] for o in others):
                continue
            full[tuple(idx) + tuple(jdx)] = u_pair[idx[a], idx[b], jdx[a], jdx[b]]
    u_full = qt.Qobj(full.reshape(2**n_ions, 2**n_ions), dims=[[2] * n_ions, [2] * n_ions])
    ket = internal if isinstance(internal, qt.Qobj) else qt.tensor(*[qt.basis(2, int(lv)) for lv in internal])
    return u_full * ket


def exact_gate_check(
    device: Device,
    waveform: Waveform,
    pair: tuple[int, int],
    gate_drives: Mapping[int, GateDrive],
    table: CalibrationTable,
    *,
    space: HilbertSpace,
    phases_rad: tuple[float, float] = (0.0, 0.0),
    chi_target_rad: float = CHI_MAXIMAL_RAD,
    nbar: Mapping[int, float] | None = None,
    internal: Sequence[int] | qt.Qobj | None = None,
    sample: NoiseSample | None = None,
    options: SolverOptions | None = None,
    builder_options: BuilderOptions | None = None,
    qubit_shifts_hz: Mapping[int, float] | None = None,
    channels: Sequence[object] = (),
    single_qubit_drives: Mapping[int, GateDrive] | None = None,
) -> tuple[GateCheck, Traces]:
    """Play ``waveform`` on ``pair`` through the JOINT_EXACT engine and read chi, leakage, residual quanta and the fidelity.

    An MS waveform is played once from |00>: P_11 = sin^2 chi. A light-shift waveform is played in the spin-echo pair of Section
    4.4.4 (``single_qubit_drives`` supply the GPi echo pulses) from |+x +x> and read in the x basis: P_11 = sin^2(2 chi) with chi the
    two-body angle of ONE pulse, so the reported ``chi_rad`` is per pulse and the fidelity is against ZZ(4 chi_target).
    """
    n_ions = space.n_ions
    x_basis = waveform.kind == "light_shift"
    # the tone phases compensate the modulator's envelope delay exactly as the scheduler does (Section 7.10; M7)
    delay = float(device.hardware.aom_rise_s) if (options or SolverOptions()).hardware_chain else 0.0
    if x_basis:
        if single_qubit_drives is None:
            raise ValueError(
                "a light-shift waveform is checked in the spin-echo pair of Section 4.4.4: pass single_qubit_drives for the GPi pulses"
            )
        sched = light_shift_echo_schedule(
            waveform,
            pair,
            gate_drives,
            single_qubit_drives,
            table,
            dead_time_s=float(device.hardware.dead_time_s),
            response_delay_s=delay,
        )
    else:
        sched = ms_schedule(waveform, pair, gate_drives, table, phases_rad=phases_rad, response_delay_s=delay)
    if internal is None:
        # an equatorial force needs a sigma_z eigenstate input, a sigma_z force an equatorial one: |+x +x> on the pair
        if x_basis:
            plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
            internal = qt.tensor(*[plus if i in pair else qt.basis(2, 0) for i in range(n_ions)])
        else:
            internal = [0] * n_ions
    state = space.initial_state(
        list(internal) if not isinstance(internal, qt.Qobj) else internal, thermal=dict(nbar or {})
    )
    n0 = {m: float(state.motional.nbar.get(m, 0.0)) for m in [t.mode for t in space.resolved]}
    engine = JointExactEngine(
        builder_options=builder_options,
        store_per_segment=2,
        channels=tuple(channels),  # type: ignore[arg-type]
        qubit_shifts_hz=dict(qubit_shifts_hz or {}),
        table=table,
    )
    traces = engine.run_pulses(
        device, sched, state, space, sample or quiet_sample(), SeedSpec(0), options or SolverOptions()
    )
    rho = traces.final.internal
    a, b = pair
    # populations in the computational basis (MS) or the x basis (light shift): P_11 = sin^2 chi either way
    read = rho
    if x_basis:
        h = qt.Qobj(np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0))
        rot = qt.tensor(*[h if i in pair else qt.qeye(2) for i in range(n_ions)])
        read = rot * rho * rot.dag()
    pops: dict[str, float] = {}
    for sa in range(2):
        for sb in range(2):
            pops[f"P{sa}{sb}"] = float(np.real(qt.expect(_internal_projector(n_ions, a, sa, b, sb), read)))
    p11 = min(max(pops["P11"], 0.0), 1.0)
    chi = float(math.asin(math.sqrt(p11))) / (2.0 if x_basis else 1.0)
    target = frame_rotated(
        _ideal_target(waveform.kind, chi_target_rad, phases_rad, n_ions, pair, internal), sched.phase_frame
    )
    fid = float(np.real(qt.expect(rho, target))) if rho.isoper else float(abs(target.overlap(rho)) ** 2)
    residual = {m: float(traces.final.motional.nbar[m] - n0[m]) for m in n0}
    check = GateCheck(pops, chi, pops["P01"] + pops["P10"], residual, fid, rho, engine.last_report)
    return check, traces


def frame_rotated(target: qt.Qobj, phase_frame: Mapping[int, float]) -> qt.Qobj:
    """The ideal ``target`` (a register ket, ion 0 the first factor) as the physical state carries it after the scheduler absorbed a
    frame offset theta_q per ion: a virtual RZ(theta) leaves the state as RZ(-theta) times the ideal one (Section 7.6: the M6 pin
    GPi2(0) RZ(0.1) = RZ(0.1) GPi2(-0.1)), so the compensated light shifts' rotation e^{-i (2 pi delta t/2) sigma_z} with the
    builder's sigma_z = |1><1| - |0><0|, i.e. RZ(-2 pi delta t) in the native convention, is absorbed as the offset +2 pi delta t
    and reproduced here as RZ(-theta_q)."""
    dims = [int(d) for d in target.dims[0]]
    ops = []
    for q, d in enumerate(dims):
        theta = -float(phase_frame.get(q, 0.0))
        op = np.eye(d, dtype=complex)
        if theta != 0.0:
            op[0, 0], op[1, 1] = np.exp(-0.5j * theta), np.exp(0.5j * theta)
        ops.append(qt.Qobj(op))
    return qt.tensor(*ops) * target


def _internal_projector(n_ions: int, a: int, sa: int, b: int, sb: int) -> qt.Qobj:
    ops = [qt.qeye(2) for _ in range(n_ions)]
    ops[a] = qt.basis(2, sa).proj()
    ops[b] = qt.basis(2, sb).proj()
    return qt.tensor(*ops)


@dataclass(frozen=True)
class CalibrationRun:
    waveform: Waveform
    checks: tuple[GateCheck, ...]
    factors: tuple[float, ...]
    converged: bool
    reference: Reference

    surrogate_chi: float = 0.0
    """The closed-form angle of the waveform as handed in (before any rescaling)."""

    @property
    def surrogate_error(self) -> float:
        """|chi_exact - chi_surrogate|/|chi_surrogate| of the FIRST check: how far the closed forms were from the exact gate."""
        return abs(self.checks[0].chi_rad - abs(self.surrogate_chi)) / abs(self.surrogate_chi)


def calibrate_entangling_angle(
    device: Device,
    waveform: Waveform,
    pair: tuple[int, int],
    gate_drives: Mapping[int, GateDrive],
    table: CalibrationTable,
    *,
    space: HilbertSpace,
    chi_target_rad: float = CHI_MAXIMAL_RAD,
    reference: Reference = "n0",
    nbar: Mapping[int, float] | None = None,
    tolerance_rad: float = 1e-4,
    max_iterations: int = 6,
    options: SolverOptions | None = None,
    builder_options: BuilderOptions | None = None,
    sample: NoiseSample | None = None,
    experiment: str = "exact_spot_check",
    single_qubit_drives: Mapping[int, GateDrive] | None = None,
) -> CalibrationRun:
    """Correct the surrogate waveform's amplitude until the exact |chi| equals ``chi_target_rad``: Newton on the s^2 law,
    chi ~ Omega^2, from |00>|0> (``reference="n0"``, Ballance's convention) or from the thermal state of ``nbar`` (the
    power-optimized reference of Sorensen-Molmer); the phase entries are stamped with the experiment name."""
    current = waveform
    checks: list[GateCheck] = []
    factors: list[float] = []
    total = 1.0
    thermal = dict(nbar or {}) if reference == "thermal" else {}
    converged = False
    for _ in range(max_iterations):
        check, _traces = exact_gate_check(
            device,
            current,
            pair,
            gate_drives,
            table,
            space=space,
            chi_target_rad=chi_target_rad,
            nbar=thermal,
            options=options,
            builder_options=builder_options,
            sample=sample,
            single_qubit_drives=single_qubit_drives,
        )
        checks.append(check)
        if abs(check.chi_rad - chi_target_rad) < tolerance_rad:
            converged = True
            break
        if check.chi_rad <= 0.0:
            raise RuntimeError("the exact gate produced no entangling angle; the waveform is not a gate")
        factor = math.sqrt(chi_target_rad / check.chi_rad)
        current = scaled(current, factor)
        total *= factor
        factors.append(total)
    # the calibrated waveform carries the EXACT two-body angle (the sign from the surrogate, the per-mode split proportional to it),
    # so that the scheduler's s^2 rescaling starts from the measured angle and MS(., ., 2 chi_target) plays this amplitude unchanged
    surrogate_total = current.chi_total_rad
    exact_total = math.copysign(checks[-1].chi_rad, surrogate_total if surrogate_total != 0.0 else 1.0)
    ratio = exact_total / surrogate_total if surrogate_total != 0.0 else 1.0
    stamped = replace(
        current,
        chi_m={m: v * ratio for m, v in current.chi_m.items()},
        phi_s=replace(current.phi_s, status="calibrated", experiment=experiment),
        phi_m=replace(current.phi_m, status="calibrated", experiment=experiment),
    )
    return CalibrationRun(
        stamped, tuple(checks), tuple(factors), converged, reference, waveform.chi_total_rad
    )


def surrogate_check(waveform: Waveform, modes: GateModes) -> dict[str, float]:
    """The closed-form side of the spot check: |chi| and eps_ent of the waveform on the mode structure (the surrogate)."""
    ints = waveform_integrals(waveform, modes)
    a, b = modes.ions[0], modes.ions[1]
    return {"chi_rad": ints.chi_of(a, b), "residual_error": ints.residual_error(modes)}


def thermal_robustness(
    device: Device,
    waveform: Waveform,
    pair: tuple[int, int],
    gate_drives: Mapping[int, GateDrive],
    table: CalibrationTable,
    *,
    modes: GateModes,
    gate_mode: int,
    nbars: Sequence[float],
    chi_target_rad: float = CHI_MAXIMAL_RAD,
    options: SolverOptions | None = None,
    builder_options: BuilderOptions | None = None,
    d_max: int = 64,
    single_qubit_drives: Mapping[int, GateDrive] | None = None,
) -> list[tuple[float, GateCheck]]:
    """The gate's exact populations and fidelity against the mode's nbar (the thermal robustness curve of the M4 roadmap): a thermal
    initial state of the gate mode, the joint space grown with nbar, the Bell-state fidelity against the ideal gate."""
    out: list[tuple[float, GateCheck]] = []
    for nb in nbars:
        space = gate_space(modes, device.crystal.n_ions, nbar={gate_mode: nb}, waveform=waveform, d_max=d_max)
        check, _ = exact_gate_check(
            device,
            waveform,
            pair,
            gate_drives,
            table,
            space=space,
            chi_target_rad=chi_target_rad,
            nbar={gate_mode: nb},
            options=options,
            builder_options=builder_options,
            single_qubit_drives=single_qubit_drives,
        )
        out.append((float(nb), check))
    return out


def parity_after_analysis_pulse(
    device: Device,
    waveform: Waveform,
    pair: tuple[int, int],
    gate_drives: Mapping[int, GateDrive],
    table: CalibrationTable,
    *,
    space: HilbertSpace,
    analysis_phase_rad: float,
    analysis_rabi_hz: Mapping[int, float],
    nbar: Mapping[int, float] | None = None,
    options: SolverOptions | None = None,
    builder_options: BuilderOptions | None = None,
    sample: NoiseSample | None = None,
    analysis_drives: Mapping[int, GateDrive] | None = None,
    spin_phases_rad: tuple[float, float] = (0.0, 0.0),
    internal: Sequence[int] | None = None,
    analysis_stark_hz: Mapping[int, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Parity P_00 + P_11 - P_01 - P_10 after the gate and a pi/2 analysis pulse of phase ``analysis_phase_rad`` on both ions
    (Section 7.9); the analysis pulses use ``analysis_drives`` (default the gate drives' beams) at the given carrier Rabi
    frequencies, with the believed Stark shifts ``analysis_stark_hz`` compensated. ``spin_phases_rad`` are the MS gate's
    (phi_0, phi_1) and ``internal`` the register's initial levels (default |0...0>), for the phase scans of Section 7.5 (M8)."""
    n_ions = space.n_ions
    sched_ms = ms_schedule(waveform, pair, gate_drives, table, phases_rad=spin_phases_rad)
    dead = float(device.hardware.dead_time_s)
    t = sched_ms.duration_s + dead
    pulses = list(sched_ms.pulses)
    sq = dict(analysis_drives) if analysis_drives is not None else dict(gate_drives)
    frame = PhaseFrame(dict(sched_ms.phase_frame))
    for q in pair:
        pulses.append(
            single_qubit_pulse(
                q,
                math.pi / 2.0,
                frame.pulse_phase(q, analysis_phase_rad),
                sq[q],
                float(analysis_rabi_hz[q]),
                t,
                stark_shift_hz=float((analysis_stark_hz or {}).get(q, 0.0)),
                gate_id=f"analysis/ion{q}",
            )
        )
    sched = Schedule(
        tuple(pulses), ((sched_ms.duration_s, t),) if dead > 0 else (), (), {q: 0.0 for q in range(n_ions)}
    )
    levels = [0] * n_ions if internal is None else [int(x) for x in internal]
    state = space.initial_state(levels, thermal=dict(nbar or {}))
    engine = JointExactEngine(builder_options=builder_options, table=table)
    traces = engine.run_pulses(
        device, sched, state, space, sample or quiet_sample(), SeedSpec(0), options or SolverOptions()
    )
    rho = traces.final.internal
    a, b = pair
    pops = {
        f"P{sa}{sb}": float(np.real(qt.expect(_internal_projector(n_ions, a, sa, b, sb), rho)))
        for sa in range(2)
        for sb in range(2)
    }
    return pops["P00"] + pops["P11"] - pops["P01"] - pops["P10"], pops


def seed_entry(value: float, provenance_id: str, experiment: str = "surrogate") -> CalEntry:
    return CalEntry(float(value), 0.0, "seed", experiment, provenance_id, 0.0, 0)


__all__ = [
    "CalibrationRun",
    "GateCheck",
    "Reference",
    "calibrate_entangling_angle",
    "exact_gate_check",
    "frame_rotated",
    "gate_space",
    "light_shift_echo_schedule",
    "ms_schedule",
    "parity_after_analysis_pulse",
    "seed_entry",
    "surrogate_check",
    "thermal_robustness",
]
