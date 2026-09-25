"""Single-ion experiments on the JOINT_EXACT engine: Rabi flopping, Ramsey, the Ramsey-frequency experiment and sideband
spectroscopy, and the laboratory context every experiment runs in (PLAN.md Section 7.5).

An experiment drives one ion with the drive ``light/`` derives from its gate beams (Raman or single-photon optical; a
microwave drive from a given Rabi frequency) from |0> and the thermal occupation of every mode. The mode the drive couples
to most strongly (or the one asked for) is resolved and the other coupled modes are frozen; the thermal statistics of both
are summed exactly over their Fock states. Populations are read through ``fitting.Observation``.

Every experiment takes the keywords of ``_LabOptions`` beside its own, and an unknown keyword raises ``TypeError``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

import numpy as np

from qutip_trap.experiments.fitting import (
    Observation,
    fit_lineshape,
    multimode_rabi_model,
    ramsey_model,
    readout_errors_for,
    sigmas_or_none,
    thermal_n_max,
    thermal_rabi_model,
    weighted_fit,
)
from qutip_trap.experiments.result import (
    ExperimentResult,
    RabiScan,
    RamseyFringe,
    ScanParameters,
    SidebandSpectrum,
    realized_drive,
    requested_drive,
)

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Drive, Pulse
    from qutip_trap.control.schedule import GateDrive
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.dynamics.space import HilbertSpace
    from qutip_trap.machine import Machine
    from qutip_trap.noise.sampling import NoiseSample

_WEIGHT_MIN = 1e-3
"""The weight below which a branch of the thermal initial mixture is dropped (the dropped weight is noted)."""


# ---- the laboratory context -------------------------------------------------------------------------------------------------


class _LabOptions(TypedDict, total=False):
    """The keywords every experiment takes beside its own: the calibration ``table`` it reads (default the machine's),
    the solver ``options`` (default the machine's numerics; None the experiment's defaults), the ``builder_options``
    (default the machine's), the noise ``sample`` it runs under, ``shots`` per point (None: exact populations),
    ``readout`` (read through the device's readout errors), the ``seed`` and ``stream`` of the shot noise, the thermal
    occupation ``nbar`` per mode and ``qubit_shifts_hz`` per ion (the true transition minus the frame)."""

    table: CalibrationTable | None
    options: SolverOptions | None
    builder_options: BuilderOptions | None
    sample: NoiseSample | None
    shots: int | None
    readout: bool
    seed: int
    stream: str
    nbar: Mapping[int, float]
    qubit_shifts_hz: Mapping[int, float] | None


def _check_lab(kw: Mapping[str, Any]) -> None:
    """Refuse a keyword that is not one of ``_LabOptions``."""
    unknown = sorted(set(kw) - _LabOptions.__optional_keys__)
    if unknown:
        known = sorted(_LabOptions.__optional_keys__)
        raise TypeError(f"unexpected keyword arguments {unknown}; the laboratory keywords are {known}")


def sub_stream(kw: Mapping[str, Any], label: str) -> dict[str, Any]:
    """The keywords of a sub-experiment with its own shot-noise stream: ``label`` appended to the parent's (Section 3.4),
    for an experiment that repeats a scan (a Ramsey per beam of the Stark scan)."""
    parent = str(kw.get("stream", ""))
    return {**kw, "stream": f"{parent}/{label}" if parent else label}


@dataclass(frozen=True)
class _Lab:
    """The machine resolved for one experiment call: the device, the table, the solver and builder options, the noise
    sample, the observation model, the thermal occupations and the frame shifts."""

    device: Device
    table: CalibrationTable | None
    options: SolverOptions | None
    builder: BuilderOptions | None
    sample: NoiseSample | None
    obs: Observation
    nbar: dict[int, float]
    qubit_shifts_hz: dict[int, float]

    @classmethod
    def of(cls, machine: Machine, kw: _LabOptions) -> _Lab:
        _check_lab(kw)
        table = kw.get("table", machine.table)
        sample = kw.get("sample")
        shots = kw.get("shots")
        obs = Observation(
            shots=None if shots is None else int(shots),
            readout=readout_errors_for(machine.device, table) if kw.get("readout") else None,
            seed=int(kw.get("seed", 0)),
            sample_id=0 if sample is None else int(sample.sample_id),
            stream=str(kw.get("stream", "")),
        )
        return cls(
            device=machine.device,
            table=table,
            options=kw.get("options", machine.numerics.to_solver_options(machine.physics)),
            builder=kw.get("builder_options", machine.physics.builder),
            sample=sample,
            obs=obs,
            nbar={int(k): float(v) for k, v in kw.get("nbar", {}).items()},
            qubit_shifts_hz={int(k): float(v) for k, v in (kw.get("qubit_shifts_hz") or {}).items()},
        )


# ---- the drive and its evolution ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Probe:
    """The single-ion drive an experiment plays: its detuning, whether the drive's Stark shift is on, a microwave drive's
    Rabi frequency, the amplitude relative to the beams' power (a laboratory attenuates), the derived crosstalk onto the
    neighbours, the rf lock, and the mode to resolve (default the most strongly coupled)."""

    detuning_hz: float = 0.0
    include_stark: bool = True
    rabi_hz: float | None = None
    amplitude_scale: float = 1.0
    crosstalk: bool = False
    rf_locked: bool = False
    mode: int | None = None


@dataclass(frozen=True)
class _Setup:
    drive: Drive
    space: HilbertSpace
    eta_driven: float
    driven_mode: int | None
    rabi_hz: float
    """The physical carrier Rabi frequency the drive gives the ion (what the flopping measures)."""
    etas: dict[int, float]
    gate_drive: GateDrive
    """The single-qubit drive of the ion (the table keys the Rabi entry by its first beam)."""


def _setup(lab: _Lab, ion: int, probe: _Probe, space: HilbertSpace | None = None) -> _Setup:
    """The drive of ``probe`` on ``ion`` and, unless given, the space: the driven mode resolved at a Gaussian cutoff sized
    for the branches that carry weight (the boundary monitor of Section 5.5 guards it), every other mode frozen."""
    from qutip_trap.control.schedule import default_gate_drives
    from qutip_trap.dynamics.operators import required_margin
    from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
    from qutip_trap.light.microwave import square_microwave_drive
    from qutip_trap.light.raman import (
        crosstalk_ratios,
        derive_optical_drive,
        derive_raman_drive,
        square_drive,
    )

    device = lab.device
    gate_drive = default_gate_drives(device)[ion]
    n_modes = len(device.crystal.modes)
    if gate_drive.kind == "microwave":
        if probe.rabi_hz is None:
            raise ValueError("a microwave experiment needs rabi_hz (no beams to derive it from)")
        drive = square_microwave_drive(ion, float(probe.rabi_hz), detuning_hz=probe.detuning_hz)
        etas = {m: 0.0 for m in range(n_modes)}
        rabi_hz = float(probe.rabi_hz)
    else:
        if gate_drive.kind == "raman":
            derived = derive_raman_drive(
                device, ion, (gate_drive.beams[0], gate_drive.beams[1]), scattering=False
            )
        else:
            derived = derive_optical_drive(device, ion, gate_drive.beams[0], scattering=False)
        scale = probe.amplitude_scale
        drive = square_drive(
            derived,
            detuning_hz=probe.detuning_hz,
            rabi_scale=scale,
            include_stark=probe.include_stark,
            crosstalk=dict(crosstalk_ratios(device, ion, gate_drive.beams, kind=gate_drive.kind))
            if probe.crosstalk
            else None,
            rf_locked=probe.rf_locked,
            rf_phase_rad=0.0 if probe.rf_locked else None,
        )
        if scale != 1.0 and probe.include_stark:
            # a two-photon shift scales with the intensity, a single-photon one with its square (Section 4.3.2)
            power = 1 if derived.kind == "raman" else 2
            drive = replace(drive, stark_shift_hz=float(derived.stark_shift_hz) * scale**power)
        etas = derived.etas
        rabi_hz = derived.carrier_rabi_hz * scale
    driven: int | None
    if probe.mode is not None:
        driven = probe.mode
        if abs(etas[driven]) == 0.0:
            raise ValueError(f"the drive does not couple ion {ion} to mode {driven} (eta = 0)")
    else:
        driven = max(range(n_modes), key=lambda m: abs(etas[m])) if n_modes else None
    eta_driven = abs(etas[driven]) if driven is not None else 0.0
    if space is None:
        ion_dims = tuple(2 for _ in range(device.crystal.n_ions))
        if driven is None or eta_driven == 0.0:
            space = HilbertSpace(ion_dims, (), None, tuple(range(n_modes)))
            driven = None
        else:
            nb = lab.nbar.get(driven, 0.0)
            n_hi = int(math.ceil(nb + 5.0 * math.sqrt(nb * (nb + 1.0)) + 2.0))
            d = n_hi + 1 + required_margin(eta_driven)
            frozen = tuple(m for m in range(n_modes) if m != driven)
            space = HilbertSpace(
                ion_dims, (ModeTruncation(driven, d, (0, n_hi), eta_driven * 1.5),), None, frozen
            )
    return _Setup(drive, space, eta_driven, driven, rabi_hz, etas, gate_drive)


@dataclass(frozen=True)
class _Averaged:
    """Expectation values against time averaged over the branches of the initial mixture, and the weight dropped."""

    times_s: np.ndarray
    expectations: dict[str, np.ndarray]
    dropped_weight: float

    def p1(self, ion: int) -> np.ndarray:
        return np.asarray(np.real(self.expectations[f"P1[{ion}]"]), dtype=float)

    def final_p1(self, ion: int) -> float:
        return float(self.p1(ion)[-1])


def _branches(
    space: HilbertSpace,
    nbar: Mapping[int, float],
    etas: Mapping[int, float],
    weight_min: float,
    fock_resolved: bool,
) -> tuple[list[tuple[float, dict[int, int], dict[int, int]]], float]:
    """(weight, resolved Fock states, frozen Fock states) of the thermal initial mixture over the coupled modes: the frozen
    ones always (Wineland's shot-to-shot Debye-Waller statistics as a weighted sum, Section 5.2), the resolved ones when
    ``fock_resolved`` (the Fock sum of Section 5.3); returned with the weight dropped below ``weight_min``."""
    from qutip_trap.dynamics.operators import thermal_populations

    options: list[tuple[int, bool, list[tuple[int, float]]]] = []
    for m, nb in nbar.items():
        if nb <= 0.0 or abs(etas.get(m, 0.0)) <= 1e-12:
            continue
        cls = space.mode_class(m)
        if cls == "frozen":
            resolved_flag = False
        elif cls == "resolved" and fock_resolved:
            resolved_flag = True
        else:
            continue
        probs = thermal_populations(nb, thermal_n_max(nb, weight_min))
        opts = [(n, float(p)) for n, p in enumerate(probs) if p >= weight_min]
        options.append((m, resolved_flag, opts or [(0, 1.0)]))
    if not options:
        return [(1.0, {}, {})], 0.0
    out: list[tuple[float, dict[int, int], dict[int, int]]] = []
    for choice in itertools.product(*[o[2] for o in options]):
        w = float(np.prod([p for _n, p in choice]))
        if w < weight_min:
            continue
        res = {options[k][0]: n for k, (n, _p) in enumerate(choice) if options[k][1]}
        fro = {options[k][0]: n for k, (n, _p) in enumerate(choice) if not options[k][1]}
        out.append((w, res, fro))
    total = sum(w for w, _r, _f in out)
    return [(w / total, r, f) for w, r, f in out], float(max(1.0 - total, 0.0))


def _run(
    lab: _Lab,
    ion: int,
    pulses: Sequence[Pulse],
    setup: _Setup,
    *,
    store_times: Sequence[float] = (),
    idle: tuple[tuple[float, float], ...] = (),
    weight_min: float = _WEIGHT_MIN,
    fock_branches: bool = True,
    device_channels: bool = False,
) -> _Averaged:
    """Play ``pulses`` from |0...0> x thermal(nbar) on the setup's space, averaged over the branches of the initial mixture;
    ``device_channels`` assembles the device's collapse operators."""
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
    from qutip_trap.noise.sampling import NoiseSample, key_frozen_n, quiet_sample

    device = lab.device
    n = device.crystal.n_ions
    schedule = Schedule(tuple(pulses), idle, (), {q: 0.0 for q in range(n)})
    engine = JointExactEngine(
        builder_options=lab.builder,
        store_per_segment=2,
        store_times_s=tuple(store_times),
        qubit_shifts_hz=lab.qubit_shifts_hz,
        table=lab.table,
        device_channels=device_channels,
    )
    options = lab.options or SolverOptions()
    base_sample = lab.sample or quiet_sample()
    seeds = SeedSpec(lab.obs.seed)
    branches, dropped = _branches(setup.space, lab.nbar, setup.etas, weight_min, fock_branches)
    times: np.ndarray | None = None
    acc: dict[str, np.ndarray] = {}
    for weight, res_fock, frozen_fock in branches:
        thermal = {m: v for m, v in lab.nbar.items() if m not in res_fock}
        state = setup.space.initial_state([0] * n, fock=res_fock, thermal=thermal)
        values = dict(base_sample.values)
        values.update({key_frozen_n(m): float(k) for m, k in frozen_fock.items()})
        sample = NoiseSample(
            sample_id=base_sample.sample_id,
            values=values,
            ou_grids=dict(base_sample.ou_grids),
            t_s=base_sample.t_s,
        )
        tr = engine.run_pulses(device, schedule, state, setup.space, sample, seeds, options)
        if times is None:
            times = np.asarray(tr.times_s, dtype=float)
        for key, arr in tr.expectations.items():
            vals = np.asarray(arr)
            if key not in acc:
                acc[key] = weight * vals
            elif acc[key].shape == vals.shape:
                acc[key] = acc[key] + weight * vals
            else:
                # differing time grids (a boundary retry): interpolate onto the first grid
                acc[key] = acc[key] + weight * np.interp(times, np.asarray(tr.times_s), np.real(vals))
    assert times is not None
    return _Averaged(times, acc, dropped)


# ---- the experiments -----------------------------------------------------------------------------------------------------------


def rabi_scan(
    machine: Machine,
    ion: int,
    durations_s: Sequence[float],
    *,
    nbar_fixed: float | None = None,
    detuning_hz: float = 0.0,
    rabi_hz: float | None = None,
    include_stark: bool = True,
    rf_locked: bool = False,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """Carrier (or, at ``detuning_hz``, sideband) Rabi flopping of ``ion`` against the pulse duration; data columns (t, P1).

    Fitted: f_rabi_hz, nbar and contrast of the thermal Debye-Waller curve (Section 4.2.7); with ``nbar_fixed`` (the
    Section 7.5 bootstrap: the driven mode's occupation from the thermometry, every other coupled mode at ``nbar``)
    f_rabi_hz, contrast and offset, since a free nbar is degenerate over less than about 1/(eta^2 nbar) Rabi periods.
    ``rabi_hz`` is a microwave drive's Rabi frequency, ``rf_locked`` locks the drive's phase to the rf."""
    from qutip_trap.control.pulses import Pulse

    lab = _Lab.of(machine, kw)
    probe = _Probe(detuning_hz=detuning_hz, include_stark=include_stark, rabi_hz=rabi_hz, rf_locked=rf_locked)
    setup = _setup(lab, ion, probe)
    ts = np.array(sorted(float(t) for t in durations_s))
    if ts.size < 2 or ts[0] < 0.0:
        raise ValueError("durations_s: at least two non-negative durations")
    t_max = float(ts[-1]) if ts[-1] > 0.0 else 1e-9
    pulse = Pulse(setup.drive, 0.0, t_max, "rabi_scan", ())
    avg = _run(lab, ion, [pulse], setup, store_times=[t for t in ts if 0.0 < t < t_max])
    exact = np.interp(ts, avg.times_s, avg.p1(ion))
    measured = [lab.obs.p1(float(p), ion, "rabi_scan", k) for k, p in enumerate(exact)]
    p1 = np.array([m[0] for m in measured])
    sigma = sigmas_or_none([m[1] for m in measured])
    eta = setup.eta_driven if detuning_hz == 0.0 else 0.0
    fitted: dict[str, tuple[float, float]] = {}
    converged = True
    chi2: float | None = None
    notes: list[str] = []
    if avg.dropped_weight > 0.0:
        notes.append(
            f"initial-mixture branches below {_WEIGHT_MIN:g} dropped: weight {avg.dropped_weight:.2e}"
        )
    if ts.size >= 4:
        if eta == 0.0 and nbar_fixed is None:
            nbar_fixed = (
                0.0  # no Debye-Waller envelope to fit nbar from (a sideband scan, a microwave carrier)
            )
        if nbar_fixed is None:
            nb0 = lab.nbar.get(setup.driven_mode, 0.0) if setup.driven_mode is not None else 0.0
            fit = weighted_fit(
                lambda p, t: thermal_rabi_model(p, t, eta),
                [setup.rabi_hz, nb0, 1.0],
                ts,
                p1,
                sigma=sigma,
                bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, 1.0]),
            )
            fitted = {"f_rabi_hz": fit.value(0), "nbar": fit.value(1), "contrast": fit.value(2)}
        else:
            # every coupled mode's Debye-Waller factor (Section 4.2.7 iii), the driven one at nbar_fixed
            coupled = [m for m, e in sorted(setup.etas.items()) if e != 0.0] if eta != 0.0 else []
            etas = [abs(setup.etas[m]) for m in coupled]
            nbars = [float(nbar_fixed) if m == setup.driven_mode else lab.nbar.get(m, 0.0) for m in coupled]
            fit = weighted_fit(
                lambda p, t: multimode_rabi_model(p, t, etas, nbars),
                [setup.rabi_hz, 1.0, 0.0],
                ts,
                p1,
                sigma=sigma,
                bounds=([0.0, 0.0, -0.5], [np.inf, 1.5, 0.5]),
            )
            fitted = {
                "f_rabi_hz": fit.value(0),
                "contrast": fit.value(1),
                "offset": fit.value(2),
                "nbar": (float(nbar_fixed), 0.0),
            }
        chi2 = fit.chi2_per_dof
        converged = fit.converged and fitted["f_rabi_hz"][0] > 0.0
        if not fit.converged:
            notes.append(f"Rabi fit did not converge: {fit.message}")
    return RabiScan(
        data=np.column_stack([ts, p1]),
        fitted=fitted,
        model="thermal_debye_waller_rabi",
        provenance_id="conv.rabi_frequency",
        converged=converged,
        notes=tuple(notes),
        sigma=sigma,
        requested=ScanParameters({"durations_s": ts, **requested_drive(setup.drive, t_max)}),
        realized=ScanParameters({"durations_s": ts, **realized_drive(lab.device, setup.drive, t_max)}),
        chi2=chi2,
        subject={"ion": int(ion), "beam": setup.gate_drive.table_key_beam},
    )


def ramsey(
    machine: Machine,
    ion: int,
    delays_s: Sequence[float],
    *,
    detuning_hz: float = 0.0,
    rabi_hz_belief: float | None = None,
    delay_pulses: Callable[[float, float], list[Pulse]] | None = None,
    rabi_hz: float | None = None,
    include_stark: bool = True,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """pi/2 - delay - pi/2 on ``ion`` at drive detuning ``detuning_hz``; data columns (delay, P1); fitted: delta_hz,
    contrast, phi0_rad and offset of P = A cos(2 pi delta t + phi_0) + B.

    The pi/2 pulses last 1/(4 f) at ``rabi_hz_belief`` (the table's Rabi frequency; default the physical one);
    ``delay_pulses`` maps (start, end) to the pulses that fill the delay (the Stark scan's probe beams)."""
    from qutip_trap.control.pulses import Pulse

    lab = _Lab.of(machine, kw)
    setup = _setup(lab, ion, _Probe(detuning_hz=detuning_hz, include_stark=include_stark, rabi_hz=rabi_hz))
    t_half = 0.25 / (float(rabi_hz_belief) if rabi_hz_belief else setup.rabi_hz)
    rows: list[tuple[float, float]] = []
    sig: list[float | None] = []
    for k, delay in enumerate(sorted(float(x) for x in delays_s)):
        if delay < 0.0:
            raise ValueError("delays are non-negative")
        start2 = t_half + delay
        pulses = [
            Pulse(setup.drive, 0.0, t_half, "ramsey_1", ()),
            Pulse(setup.drive, start2, start2 + t_half, "ramsey_2", ()),
        ]
        idle: tuple[tuple[float, float], ...] = ()
        if delay > 0.0:
            if delay_pulses is not None:
                pulses.extend(delay_pulses(t_half, start2))
            else:
                idle = ((t_half, start2),)
        p, s = lab.obs.p1(_run(lab, ion, pulses, setup, idle=idle).final_p1(ion), ion, "ramsey", k)
        rows.append((delay, p))
        sig.append(s)
    data = np.array(rows)
    sigma = sigmas_or_none(sig)
    fitted: dict[str, tuple[float, float]] = {}
    converged = True
    chi2: float | None = None
    if data.shape[0] >= 4:
        t, y = data[:, 0], data[:, 1]
        guess = [0.5, detuning_hz if detuning_hz != 0.0 else 1.0 / max(float(t[-1]), 1e-9), 0.0, 0.5]
        fit = weighted_fit(ramsey_model, guess, t, y, sigma=sigma)
        fitted = {
            "contrast": (abs(float(fit.params[0])), float(fit.errors[0])),
            "delta_hz": fit.value(1),
            "phi0_rad": fit.value(2),
            "offset": fit.value(3),
        }
        converged, chi2 = fit.converged, fit.chi2_per_dof
    held = {"delays_s": data[:, 0], "pi_half_s": (t_half,)}
    return RamseyFringe(
        data=data,
        fitted=fitted,
        model="ramsey_fringe",
        provenance_id="conv.detuning_symbols",
        converged=converged,
        sigma=sigma,
        requested=ScanParameters({**held, **requested_drive(setup.drive, t_half)}),
        realized=ScanParameters({**held, **realized_drive(lab.device, setup.drive, t_half)}),
        chi2=chi2,
        subject={"ion": int(ion), "beam": setup.gate_drive.table_key_beam},
    )


def ramsey_frequency(
    machine: Machine,
    ion: int,
    delays_s: Sequence[float],
    *,
    probe_hz: float = 1e3,
    frame_hz: float = 0.0,
    rabi_hz_belief: float | None = None,
    rabi_hz: float | None = None,
    include_stark: bool = True,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """The qubit frequency for the table: ``frame_hz`` (the drive's reference) plus the offset x of the transition from it,
    from two Ramsey scans at drive detunings +-``probe_hz`` whose fringes |probe -+ x| resolve its sign (while |x| <
    probe). Fitted: qubit_freq_hz, qubit_offset_hz, fringe_plus_hz and fringe_minus_hz."""
    _check_lab(kw)
    probe = abs(float(probe_hz))
    plus, minus = (
        ramsey(
            machine,
            ion,
            delays_s,
            detuning_hz=sign * probe,
            rabi_hz_belief=rabi_hz_belief,
            rabi_hz=rabi_hz,
            include_stark=include_stark,
            **kw,
        )
        for sign in (1.0, -1.0)
    )

    def fringe(res: ExperimentResult) -> tuple[float, float]:
        return (abs(res.fitted["delta_hz"][0]), res.fitted["delta_hz"][1]) if res.fitted else (math.nan, 0.0)

    (fp, sp), (fm, sm) = fringe(plus), fringe(minus)
    x = 0.5 * (fm - fp)  # the fringe runs at |probe - x| for the + scan and at |probe + x| for the - scan
    unc = 0.5 * math.hypot(sp, sm)
    notes: list[str] = []
    if math.isfinite(x) and abs(x) >= probe:
        notes.append(
            f"the offset {x:.3g} Hz exceeds the probe {probe:.3g} Hz: "
            "the sign resolution is ambiguous; raise probe_hz"
        )
    chis = [c for c in (plus.chi2, minus.chi2) if c]
    return RamseyFringe(
        data=np.vstack(
            [
                np.column_stack([plus.data, np.full(len(plus.data), +probe)]),
                np.column_stack([minus.data, np.full(len(minus.data), -probe)]),
            ]
        ),
        fitted={
            "qubit_offset_hz": (x, unc),
            "qubit_freq_hz": (frame_hz + x, unc),
            "fringe_plus_hz": (fp, sp),
            "fringe_minus_hz": (fm, sm),
        },
        model="ramsey_two_probe",
        provenance_id="conv.detuning_symbols",
        converged=plus.converged and minus.converged and math.isfinite(x) and abs(x) < probe,
        notes=tuple(notes),
        requested=plus.requested,
        realized=plus.realized,
        chi2=max(chis) if chis else None,
        subject=dict(plus.subject),
        experiment="ramsey_frequency",
    )


def sideband_spectroscopy(
    machine: Machine,
    ion: int,
    detunings_hz: Sequence[float],
    *,
    duration_s: float | None = None,
    fit: bool = False,
    include_stark: bool = True,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """P1 after a pulse of ``duration_s`` (default the carrier pi time) against the drive detuning; fitted: the detunings of
    the local maxima nearest the carrier and the driven mode's first sidebands (Section 7.9) and their half-difference
    mode_hz. With ``fit`` the carrier and the blue sideband are fitted with the plan's lineshape where the scan has five
    points within a quarter of the mode frequency of them: carrier_fit_hz, omega_carrier_hz, blue_sideband_fit_hz and
    omega_bsb_hz."""
    from qutip_trap.control.pulses import Pulse

    lab = _Lab.of(machine, kw)
    probe = _Probe(include_stark=include_stark)
    base = _setup(lab, ion, probe)
    duration = 0.5 / base.rabi_hz if duration_s is None else float(duration_s)
    rows: list[tuple[float, float]] = []
    sig: list[float | None] = []
    realized_mus: list[float] = []
    for k, mu in enumerate(sorted(float(x) for x in detunings_hz)):
        setup = _setup(lab, ion, replace(probe, detuning_hz=mu), base.space)
        realized_mus.append(realized_drive(lab.device, setup.drive, duration)["detuning_hz"])
        avg = _run(lab, ion, [Pulse(setup.drive, 0.0, duration, "sideband_spectroscopy", ())], setup)
        p, s = lab.obs.p1(avg.final_p1(ion), ion, "sideband_spectroscopy", k)
        rows.append((mu, p))
        sig.append(s)
    data = np.array(rows)
    sigma = sigmas_or_none(sig)
    fitted: dict[str, tuple[float, float]] = {}
    mus, p1 = data[:, 0], data[:, 1]
    step = float(np.median(np.diff(mus))) if len(mus) > 1 else 0.0
    if base.driven_mode is not None:
        f_mode = float(lab.device.crystal.modes[base.driven_mode].omega_hz)
        for name, centre in (("carrier_hz", 0.0), ("blue_sideband_hz", f_mode), ("red_sideband_hz", -f_mode)):
            window = np.abs(mus - centre) <= max(0.25 * f_mode, 2 * step)
            if not np.any(window):
                continue
            idx = int(np.argmax(np.where(window, p1, -np.inf)))
            fitted[name] = (float(mus[idx]), step)
            if fit and int(np.sum(window)) >= 5 and name != "red_sideband_hz":
                sel = np.flatnonzero(window)
                line = fit_lineshape(
                    mus[sel],
                    p1[sel],
                    duration,
                    sigma=None if sigma is None else sigma[sel],
                    guess=(float(mus[idx]), 0.5 / duration),
                )
                carrier = name == "carrier_hz"
                fitted["carrier_fit_hz" if carrier else "blue_sideband_fit_hz"] = line.value(0)
                fitted["omega_carrier_hz" if carrier else "omega_bsb_hz"] = line.value(1)
        if "blue_sideband_hz" in fitted and "red_sideband_hz" in fitted:
            fitted["mode_hz"] = (0.5 * (fitted["blue_sideband_hz"][0] - fitted["red_sideband_hz"][0]), step)
    else:
        fitted["carrier_hz"] = (float(mus[int(np.argmax(p1))]), step)
    mode = {} if base.driven_mode is None else {"mode": int(base.driven_mode)}
    return SidebandSpectrum(
        data=data,
        fitted=fitted,
        model="sideband_spectrum_peaks",
        provenance_id="conv.detuning_symbols",
        sigma=sigma,
        requested=ScanParameters({"detunings_hz": mus, "duration_s": (duration,)}),
        realized=ScanParameters({"detunings_hz": realized_mus, "duration_s": (duration,)}),
        subject={"ion": int(ion), "beam": base.gate_drive.table_key_beam, **mode},
    )
