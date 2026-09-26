"""Entangling-gate calibration by exact spot checks (PLAN.md Section 7.5).

The surrogate's closed-form waveform is played on the pair through the JOINT_EXACT engine from |00>|n = 0>, |chi| is read
from P_11 = sin^2 chi (an equatorial two-body rotation takes |00> to cos chi |00> -/+ i e^{...} sin chi |11>, so P_01 + P_10
is the leakage from open loops and off-resonant excitation), every amplitude is rescaled by sqrt(chi_target/chi) (the s^2
law) and the check repeats. Every check plays the scheduler's own schedule of the gate as a run plays it under the
machine's ``Physics`` (the Stark compensation, the crosstalk echo, the hardware chain and the builder options): an MS
waveform as MS(0, 0, 2|chi|), a sigma_z-force waveform (light shift, microwave gradient) in its spin echo, the sequence a
run plays for ZZ. For the |00> input the final mean excitation of a mode is sum_j |alpha_jm|^2, eps_ent at nbar = 0. The
corrected Waveform is returned, never written into a table.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import qutip as qt

from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.control.native import ms as native_ms
from qutip_trap.control.native import zz as native_zz
from qutip_trap.control.schedule import GateDrive, Schedule, schedule
from qutip_trap.control.shaping import CHI_MAXIMAL_RAD, GateModes, scaled
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import EngineReport, JointExactEngine, SeedSpec, Traces
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.noise.sampling import NoiseSample, quiet_sample
from qutip_trap.options import Numerics
from qutip_trap.run.space import ModeClass3, classify, mode_cap, waveform_contributions

if TYPE_CHECKING:
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.options import Physics


def spot_check_space(
    device: Device,
    modes: GateModes,
    waveform: Waveform,
    pair: tuple[int, int],
    options: Numerics,
    *,
    ions: Sequence[int] | None = None,
) -> tuple[HilbertSpace, dict[int, ModeClass3]]:
    """The joint space of a pair's spot check by the run's rule (``run.space.select_space``): every mode of ``modes`` classed
    by the closed-form contribution of ``waveform`` at the occupations ``modes`` carries, a resolved one truncated by the
    run's own rule (``run.space.mode_cap``: the cap rule at the numerics' boundary threshold and ceiling, ``options.caps``
    in its place), a dropped one out of the dynamics, every other crystal mode frozen; over every ion of the crystal
    (JOINT_EXACT), or over ``ions`` alone (the pair's GATE_LOCAL space). Returns the space and the class of every mode of
    ``modes``."""
    contributions = waveform_contributions(waveform, modes, pair)
    classes: dict[int, ModeClass3] = {}
    resolved: list[ModeTruncation] = []
    for k, m in enumerate(modes.modes):
        c = contributions[m]
        classes[m] = classify(
            c,
            coupled=True,
            freeze_alpha_max=options.freeze_alpha_max,
            freeze_chi_max_rad=options.freeze_chi_max_rad,
        )
        if classes[m] == "resolved":
            resolved.append(mode_cap(m, c, modes.nbar[k], options).truncation)
    held = {t.mode for t in resolved}
    frozen = tuple(m for m in range(len(device.crystal.modes)) if m not in held)
    dropped = tuple(m for m in modes.modes if classes[m] == "dropped")
    if ions is None:
        return HilbertSpace(
            tuple([2] * device.crystal.n_ions), tuple(resolved), None, frozen, (), dropped
        ), classes
    local = tuple(sorted(int(i) for i in ions))
    return HilbertSpace(tuple([2] * len(local)), tuple(resolved), None, frozen, local, dropped), classes


def _scheduled(
    device: Device,
    operations: Sequence[Operation],
    waveform: Waveform,
    pair: tuple[int, int],
    gate_drives: Mapping[int, GateDrive],
    table: CalibrationTable,
    *,
    physics: Physics,
    options: Numerics | None = None,
    single_qubit_drives: Mapping[int, GateDrive] | None = None,
) -> Schedule:
    """The native ``operations`` on the crystal as a run schedules them with ``waveform`` in the table for ``pair``
    (``run.pipeline.compile_calibrate_schedule``): under ``physics``'s Stark compensation, crosstalk echo and hardware chain
    and the numerics' addressing, with ``gate_drives`` for the entangling gate and ``single_qubit_drives`` (default the
    device's roles) for every carrier pulse, at the table's Stark shifts, crosstalk and frame and the chain's dead time
    between pulses. The schedule ends with the last pulse: the dead time after it belongs to what a run plays next."""
    sched = schedule(
        Circuit(device.crystal.n_ions, tuple(operations), ()),
        device,
        table.with_params(ms={pair: waveform}),
        gate_drives=None if single_qubit_drives is None else dict(single_qubit_drives),
        entangling_drives=dict(gate_drives),
        parallel=(options or Numerics()).addressing,
        crosstalk_suppression=physics.crosstalk_suppression,
        stark_compensation=physics.stark_compensation,
        hardware_chain=physics.hardware_chain,
    )
    end = max(p.t_end_s for p in sched.pulses)
    return replace(sched, idle=tuple((a, b) for a, b in sched.idle if a < end))


def ms_schedule(
    device: Device,
    waveform: Waveform,
    pair: tuple[int, int],
    gate_drives: Mapping[int, GateDrive],
    table: CalibrationTable,
    *,
    physics: Physics,
    options: Numerics | None = None,
    phases_rad: tuple[float, float] = (0.0, 0.0),
    single_qubit_drives: Mapping[int, GateDrive] | None = None,
) -> Schedule:
    """The scheduler's own MS(phi_0, phi_1, 2 |chi|) on ``pair`` with ``waveform`` in the table, the angle at which it plays
    the waveform unscaled, as a run plays it (``_scheduled``): the tones detuned by the believed light shift, the gate
    split around the crosstalk echo and the tone phases referenced to the chain's response as ``physics`` says."""
    theta = 2.0 * abs(waveform.chi_total_rad)
    gate = Operation("ms", pair, (float(phases_rad[0]), float(phases_rad[1]), theta))
    return _scheduled(
        device,
        (gate,),
        waveform,
        pair,
        gate_drives,
        table,
        physics=physics,
        options=options,
        single_qubit_drives=single_qubit_drives,
    )


def zz_echo_schedule(
    device: Device,
    waveform: Waveform,
    pair: tuple[int, int],
    gate_drives: Mapping[int, GateDrive],
    table: CalibrationTable,
    *,
    physics: Physics,
    options: Numerics | None = None,
    single_qubit_drives: Mapping[int, GateDrive] | None = None,
) -> Schedule:
    """The scheduler's own ZZ(-4 chi) on ``pair`` with ``waveform`` in the table, the angle at which it plays the waveform
    unscaled, as a run plays it (``_scheduled``): the sigma_z sigma_z spin echo of a light-shift or gradient waveform
    (the waveform, GPi(0) on both ions, the waveform again, GPi(pi) on both), whose single-qubit sigma_z phases cancel while
    the two-body angle doubles, the echo pulses ``single_qubit_drives``'s."""
    gate = Operation("zz", pair, (-4.0 * waveform.chi_total_rad,))
    return _scheduled(
        device,
        (gate,),
        waveform,
        pair,
        gate_drives,
        table,
        physics=physics,
        options=options,
        single_qubit_drives=single_qubit_drives,
    )


@dataclass(frozen=True)
class GateCheck:
    """One exact spot check of an entangling waveform."""

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


def _ideal_target(
    kind: str,
    chi_target_rad: float,
    phases_rad: tuple[float, float],
    n_ions: int,
    pair: tuple[int, int],
    internal: Sequence[int] | qt.Qobj,
) -> qt.Qobj:
    """The native gate's output for the input ``internal``: MS(phi_0, phi_1, 2 chi) or ZZ(4 chi) (two echo loops) on the pair,
    identity elsewhere; ``pair`` are FACTOR positions in the space's ion order."""
    a, b = pair
    if kind == "ms":
        mat = native_ms(phases_rad[0], phases_rad[1], 2.0 * chi_target_rad)
    else:
        mat = native_zz(4.0 * chi_target_rad)
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
    physics: Physics,
    phases_rad: tuple[float, float] = (0.0, 0.0),
    chi_target_rad: float = CHI_MAXIMAL_RAD,
    nbar: Mapping[int, float] | None = None,
    sample: NoiseSample | None = None,
    options: Numerics | None = None,
    qubit_shifts_hz: Mapping[int, float] | None = None,
    channels: Sequence[object] = (),
    single_qubit_drives: Mapping[int, GateDrive] | None = None,
) -> tuple[GateCheck, Traces]:
    """Play ``waveform`` on ``pair`` through the JOINT_EXACT engine as a run plays it under ``physics`` (the machine's: the
    scheduler's own schedule of the gate, the builder options and the hardware chain) and read chi, leakage, residual
    quanta and the fidelity.

    An MS waveform is played as MS(phi_0, phi_1, 2|chi|) (``ms_schedule``) from |00>: P_11 = sin^2 chi. A light-shift or
    gradient waveform (a sigma_z force) is played in its spin echo (``zz_echo_schedule``) from |+x +x> and read in the x
    basis of the frame the scheduler absorbed: P_11 = sin^2(2 chi) with chi the two-body angle of ONE pulse, so the reported
    ``chi_rad`` is per pulse and the fidelity is against ZZ(4 chi_target). The echo pulses, of the spin echo and of the
    crosstalk suppression, are ``single_qubit_drives``'s (default the device's roles).
    """
    n_ions = space.n_ions
    opts = options or Numerics()
    sigma_z = waveform.kind != "ms"
    if sigma_z:
        sched = zz_echo_schedule(
            device,
            waveform,
            pair,
            gate_drives,
            table,
            physics=physics,
            options=opts,
            single_qubit_drives=single_qubit_drives,
        )
    else:
        sched = ms_schedule(
            device,
            waveform,
            pair,
            gate_drives,
            table,
            physics=physics,
            options=opts,
            phases_rad=phases_rad,
            single_qubit_drives=single_qubit_drives,
        )
    # the pair's FACTOR positions: the device ions on a full space, their positions in ``space.ions`` on a GATE_LOCAL space
    fa, fb = space.ion_factor(pair[0]), space.ion_factor(pair[1])
    internal: list[int] | qt.Qobj
    if sigma_z:
        # a sigma_z force needs an equatorial input: |+x +x> on the pair
        plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        internal = qt.tensor(*[plus if i in (fa, fb) else qt.basis(2, 0) for i in range(n_ions)])
    else:
        internal = [0] * n_ions
    state = space.initial_state(internal, thermal=dict(nbar or {}))
    n0 = {m: float(state.motional.nbar.get(m, 0.0)) for m in [t.mode for t in space.resolved]}
    engine = JointExactEngine(
        builder_options=physics.builder,
        store_per_segment=2,
        channels=tuple(channels),  # type: ignore[arg-type]
        qubit_shifts_hz=dict(qubit_shifts_hz or {}),
        hardware_chain=physics.hardware_chain,
        table=table,
    )
    traces = engine.run_pulses(device, sched, state, space, sample or quiet_sample(), SeedSpec(0), opts)
    rho = traces.final.internal
    frame_local = {space.ion_factor(q): th for q, th in sched.phase_frame.items() if space.has_ion(q)}
    # populations in the frame the scheduler absorbed, in the computational basis (MS) or the x basis (sigma_z force):
    # P_11 = sin^2 chi either way
    frame = frame_operator([int(d) for d in rho.dims[0]], frame_local)
    read = frame.dag() * rho * frame
    if sigma_z:
        h = qt.Qobj(np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0))
        rot = qt.tensor(*[h if i in (fa, fb) else qt.qeye(2) for i in range(n_ions)])
        read = rot * read * rot.dag()
    pops = _populations(read, n_ions, fa, fb)
    p11 = min(max(pops["P11"], 0.0), 1.0)
    chi = float(math.asin(math.sqrt(p11))) / (2.0 if sigma_z else 1.0)
    target = frame_rotated(
        _ideal_target(waveform.kind, chi_target_rad, phases_rad, n_ions, (fa, fb), internal), frame_local
    )
    fid = float(np.real(qt.expect(rho, target))) if rho.isoper else float(abs(target.overlap(rho)) ** 2)
    residual = {m: float(traces.final.motional.nbar[m] - n0[m]) for m in n0}
    check = GateCheck(pops, chi, pops["P01"] + pops["P10"], residual, fid, rho, engine.last_report)
    return check, traces


def frame_operator(dims: Sequence[int], phase_frame: Mapping[int, float]) -> qt.Qobj:
    """(x)_q RZ(-theta_q) on a register of ``dims`` (ion 0 the first factor): what a frame offset theta_q the scheduler absorbed
    as a virtual RZ(theta_q) leaves on the physical state, which carries it times the ideal one."""
    ops = []
    for q, d in enumerate(dims):
        theta = -float(phase_frame.get(q, 0.0))
        op = np.eye(d, dtype=complex)
        if theta != 0.0:
            op[0, 0], op[1, 1] = np.exp(-0.5j * theta), np.exp(0.5j * theta)
        ops.append(qt.Qobj(op))
    return qt.tensor(*ops)


def frame_rotated(target: qt.Qobj, phase_frame: Mapping[int, float]) -> qt.Qobj:
    """The ideal ``target`` (a register ket, ion 0 the first factor) as the physical state carries it after the scheduler
    absorbed a frame offset theta_q per ion (``frame_operator``)."""
    return frame_operator([int(d) for d in target.dims[0]], phase_frame) * target


def _populations(rho: qt.Qobj, n_ions: int, a: int, b: int) -> dict[str, float]:
    """P_{sa sb} of the factors ``a`` and ``b``."""
    out: dict[str, float] = {}
    for sa in range(2):
        for sb in range(2):
            ops = [qt.qeye(2) for _ in range(n_ions)]
            ops[a] = qt.basis(2, sa).proj()
            ops[b] = qt.basis(2, sb).proj()
            out[f"P{sa}{sb}"] = float(np.real(qt.expect(qt.tensor(*ops), rho)))
    return out


@dataclass(frozen=True)
class CalibrationRun:
    """The record of an entangling-gate spot check: the corrected ``Waveform``, the exact ``GateCheck`` of every iteration,
    the amplitude factors applied, whether the angle converged, and the closed-form angle the waveform came in with."""

    waveform: Waveform
    checks: tuple[GateCheck, ...]
    factors: tuple[float, ...]
    converged: bool
    surrogate_chi: float

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
    physics: Physics,
    chi_target_rad: float = CHI_MAXIMAL_RAD,
    tolerance_rad: float = 1e-4,
    max_iterations: int = 6,
    options: Numerics | None = None,
    single_qubit_drives: Mapping[int, GateDrive] | None = None,
) -> CalibrationRun:
    """Correct the waveform's amplitude until the exact |chi| from |00>|0> (Ballance's reference) of the gate a run plays
    under ``physics`` (``exact_gate_check``) equals ``chi_target_rad``: Newton on the s^2 law, chi ~ Omega^2; the phase
    entries are stamped ``calibrated`` by ``exact_spot_check``."""
    current = waveform
    checks: list[GateCheck] = []
    factors: list[float] = []
    total = 1.0
    converged = False
    checked = current  # the waveform checks[-1] measured: never stamp an angle measured on another amplitude
    for _ in range(max_iterations):
        check, _traces = exact_gate_check(
            device,
            current,
            pair,
            gate_drives,
            table,
            space=space,
            chi_target_rad=chi_target_rad,
            options=options,
            physics=physics,
            single_qubit_drives=single_qubit_drives,
        )
        checks.append(check)
        checked = current
        if abs(check.chi_rad - chi_target_rad) < tolerance_rad:
            converged = True
            break
        if check.chi_rad <= 0.0:
            raise RuntimeError("the exact gate produced no entangling angle; the waveform is not a gate")
        factor = math.sqrt(chi_target_rad / check.chi_rad)
        current = scaled(current, factor)
        total *= factor
        factors.append(total)
    # the calibrated waveform carries the EXACT two-body angle (the sign from the surrogate, the per-mode split proportional
    # to it), so that the scheduler's s^2 rescaling starts from the measured angle
    current = checked
    surrogate_total = current.chi_total_rad
    exact_total = math.copysign(checks[-1].chi_rad, surrogate_total if surrogate_total != 0.0 else 1.0)
    ratio = exact_total / surrogate_total if surrogate_total != 0.0 else 1.0
    stamped = replace(
        current,
        chi_m={m: v * ratio for m, v in current.chi_m.items()},
        phi_s=replace(current.phi_s, status="calibrated", experiment="exact_spot_check"),
        phi_m=replace(current.phi_m, status="calibrated", experiment="exact_spot_check"),
    )
    return CalibrationRun(stamped, tuple(checks), tuple(factors), converged, waveform.chi_total_rad)


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
    physics: Physics,
    options: Numerics | None = None,
    single_qubit_drives: Mapping[int, GateDrive] | None = None,
) -> list[tuple[float, GateCheck]]:
    """The exact populations and fidelity of the gate a run plays under ``physics`` (``exact_gate_check``) against the gate
    mode's thermal occupation: per nbar, a thermal initial state of the gate mode (every other mode in its ground state) on
    the spot-check space sized for it."""
    opts = options or Numerics()
    out: list[tuple[float, GateCheck]] = []
    for nb in nbars:
        occupied = replace(modes, nbar=tuple(float(nb) if m == gate_mode else 0.0 for m in modes.modes))
        space, _classes = spot_check_space(device, occupied, waveform, pair, opts)
        check, _ = exact_gate_check(
            device,
            waveform,
            pair,
            gate_drives,
            table,
            space=space,
            nbar={gate_mode: nb},
            options=opts,
            physics=physics,
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
    physics: Physics,
    analysis_phase_rad: float,
    nbar: Mapping[int, float] | None = None,
    options: Numerics | None = None,
    sample: NoiseSample | None = None,
    single_qubit_drives: Mapping[int, GateDrive] | None = None,
    spin_phases_rad: tuple[float, float] = (0.0, 0.0),
    internal: Sequence[int] | None = None,
    qubit_shifts_hz: Mapping[int, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Parity P_00 + P_11 - P_01 - P_10 after the gate and a pi/2 analysis pulse of phase ``analysis_phase_rad`` on both ions
    (the parity scan): MS(phi_0, phi_1, 2 |chi|) and GPi2(analysis phase) on each ion of the pair as a run plays them under
    ``physics`` (``_scheduled``), the analysis pulses ``single_qubit_drives``'s (default the device's roles) at the
    table's Rabi frequencies with its Stark shifts and crosstalk, in the frame the gate left. ``spin_phases_rad`` are the MS
    gate's (phi_0, phi_1), ``internal`` the register's initial levels (default |0...0>), ``qubit_shifts_hz`` the true
    transition minus the table's frame per ion, so that a phase scan measures the axis in the frame the run applies."""
    n_ions = space.n_ions
    opts = options or Numerics()
    theta = 2.0 * abs(waveform.chi_total_rad)
    operations = (
        Operation("ms", pair, (float(spin_phases_rad[0]), float(spin_phases_rad[1]), theta)),
        *(Operation("gpi2", (q,), (float(analysis_phase_rad),)) for q in pair),
    )
    sched = _scheduled(
        device,
        operations,
        waveform,
        pair,
        gate_drives,
        table,
        physics=physics,
        options=opts,
        single_qubit_drives=single_qubit_drives,
    )
    levels = [0] * n_ions if internal is None else [int(x) for x in internal]
    state = space.initial_state(levels, thermal=dict(nbar or {}))
    engine = JointExactEngine(
        builder_options=physics.builder,
        hardware_chain=physics.hardware_chain,
        table=table,
        qubit_shifts_hz=dict(qubit_shifts_hz or {}),
    )
    traces = engine.run_pulses(device, sched, state, space, sample or quiet_sample(), SeedSpec(0), opts)
    pops = _populations(traces.final.internal, n_ions, space.ion_factor(pair[0]), space.ion_factor(pair[1]))
    return pops["P00"] + pops["P11"] - pops["P01"] - pops["P10"], pops
