"""The full calibration by simulated experiments (PLAN.md Section 7.5).

``full_calibration`` starts from the surrogate table (every derived value a ``seed``, the closed-form waveforms, the
detection threshold) and replaces its entries with the results of the simulated experiments of ``qutip_trap.experiments``
in the dependency order ``ORDER``:

    crystal_image -> field_scan -> micromotion_scan -> sideband spectroscopy (modes, nbar, eta) -> rabi_scan, stark_scan,
    ramsey_frequency -> crosstalk_scan -> ms_scan (amplitude, detuning, phase), parity_scan -> detection_histogram,
    heating_rate

A fit refuses to run while an upstream entry it reads (``UPSTREAM``) is ``uncalibrated``, and every entry group it would
have written (``PRODUCES``) is then marked ``uncalibrated`` under its name, so that what the calibration could not
establish refuses to schedule instead of falling back to the derived seed; the refusal cascades. A group that is a
``seed`` because its experiment was not asked for stays a seed. Every experiment reads only the table's beliefs for what
the machine programs and starts from the derived values as initial guesses; every fit reports an uncertainty, and a fit
that fails to converge or lands at the edge of its scan marks its entry ``uncalibrated``. The calibration is fitted under
one dynamical sample of the noise model at ``t0_s``, whose id every entry carries. The scan sizes are ``CalibrationScans``
(the plan's defaults; the tests run reduced scans).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from qutip_trap.calibration.surrogate import SurrogateReport, surrogate_table
from qutip_trap.control.table import CalEntry, CalibrationTable, Waveform, usable
from qutip_trap.experiments.result import ExperimentResult
from qutip_trap.machine import Machine

if TYPE_CHECKING:
    from qutip_trap.control.schedule import GateDrive
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.noise.sampling import NoiseSample

ORDER: tuple[str, ...] = (
    "crystal_image",
    "field_scan",
    "micromotion_scan",
    "sideband_spectroscopy",
    "rabi_scan",
    "stark_scan",
    "ramsey_frequency",
    "crosstalk_scan",
    "ms_scan",
    "parity_scan",
    "detection_histogram",
    "heating_rate",
)
"""The dependency order (``ms_scan`` runs the amplitude, detuning and phase scans of the entangling gate)."""

UPSTREAM: dict[str, tuple[str, ...]] = {
    "crystal_image": (),
    "field_scan": (),
    "micromotion_scan": ("field",),
    "sideband_spectroscopy": ("field", "micromotion"),
    "rabi_scan": ("field", "micromotion", "nbar"),
    "stark_scan": ("field", "rabi"),
    "ramsey_frequency": ("field", "rabi"),
    "crosstalk_scan": ("rabi",),
    "ms_scan": ("modes", "nbar", "rabi", "stark", "qubit_freq"),
    "parity_scan": ("modes", "nbar", "rabi", "stark", "qubit_freq", "ms"),
    "detection_histogram": (),
    "heating_rate": ("modes", "nbar", "rabi"),
}
"""The table entry groups each experiment reads: a fit refuses to run while one of them is ``uncalibrated``."""

PRODUCES: dict[str, tuple[str, ...]] = {
    "crystal_image": (),
    "field_scan": ("field",),
    "micromotion_scan": ("micromotion",),
    "sideband_spectroscopy": ("modes", "nbar", "lamb_dicke"),
    "rabi_scan": ("rabi",),
    "stark_scan": ("stark",),
    "ramsey_frequency": ("qubit_freq",),
    "crosstalk_scan": ("crosstalk", "crosstalk_phase"),
    "ms_scan": ("ms",),
    "parity_scan": (),
    "detection_histogram": ("detection",),
    "heating_rate": ("heating",),
}
"""The entry groups each experiment writes: a REFUSED experiment marks exactly these ``uncalibrated``."""

ALIASES: dict[str, str] = {
    "mode_spectroscopy": "sideband_spectroscopy",
    "thermometry": "sideband_spectroscopy",
    "crosstalk_phase": "crosstalk_scan",
    "ms_phase_scan": "ms_scan",
}
"""Accepted names for PARTS of an experiment in ``ORDER``, mapped to the one that runs them."""

_PROBE_HZ = 1e3
"""The Ramsey probe detuning of the field, qubit-frequency and Stark scans (below the Nyquist frequency of their delay
grids)."""


class CalibrationError(RuntimeError):
    """An experiment could not run because a table entry it needs is absent."""


@dataclass(frozen=True)
class CalibrationScans:
    """The scan sizes of the full calibration (the plan's defaults; the tests reduce them)."""

    shots: int | None = 500
    """Shots per scan point (None: exact populations, no statistical uncertainty)."""
    rabi_points: int = 41
    """Points of the Rabi scan over ten pi times."""
    ramsey_delays_s: tuple[float, ...] = tuple(float(x) for x in np.linspace(0.0, 2e-3, 9))
    stark_delays_s: tuple[float, ...] = tuple(float(x) for x in np.linspace(0.0, 2e-3, 9))
    mode_span: float = 0.1
    mode_coarse_points: int | None = None
    mode_fine_points: int = 15
    crosstalk_points: int = 16
    crosstalk_phase_points: int = 6
    ms_amplitude_points: int = 5
    """Points of the entangling amplitude scan over +-30 %."""
    ms_detuning_offsets_hz: tuple[float, ...] = (-2e3, 0.0, 2e3)
    parity_points: int = 6
    phase_points: int = 6
    phase_inputs: tuple[str, ...] = ("00", "01")
    detection_records: int = 10_000
    detection_windows_s: tuple[float, ...] | None = None
    heating_delays: int = 6
    heating_span_over_ndot: float = 10.0
    micromotion_ranges: Mapping[str, tuple[float, float]] | None = None
    """Shim (or compensation-field) ranges to scan; None scans +-50 V/m on the explicit path, nothing on the geometry path."""


@dataclass(frozen=True)
class CalibrationReport:
    """The table with what every experiment measured, what was refused and why."""

    table: CalibrationTable
    surrogate: SurrogateReport
    results: dict[str, ExperimentResult]
    """Experiment results keyed as ``rabi_scan[0]``, ``mode_spectroscopy[3]``, ``ms_scan[(0, 1)]``, ..."""
    refused: dict[str, str]
    notes: tuple[str, ...]
    sample: NoiseSample
    experiments: tuple[str, ...]

    def surrogate_error(self) -> dict[str, float]:
        """Per entry, |calibrated - seed|/|seed| where both exist: the surrogate's error."""
        seeds = self.surrogate.table.entries()
        out: dict[str, float] = {}
        for key, entry in self.table.entries().items():
            seed = seeds.get(key)
            if seed is None or entry.status != "calibrated" or seed.value == 0.0:
                continue
            out[key] = abs(entry.value - seed.value) / abs(seed.value)
        return out


def _entry(
    res: ExperimentResult, key: str, experiment: str, provenance_id: str, t0_s: float, sample_id: int
) -> CalEntry:
    value, unc = res.fitted.get(key, (math.nan, math.nan))
    good = res.converged and math.isfinite(value) and math.isfinite(unc)
    return CalEntry(
        float(value) if math.isfinite(value) else 0.0,
        float(unc) if math.isfinite(unc) and unc >= 0.0 else 0.0,
        "calibrated" if good else "uncalibrated",
        experiment,
        provenance_id,
        float(t0_s),
        int(sample_id),
    )


def _uncalibrated(entry: CalEntry, experiment: str, t0_s: float, sample_id: int) -> CalEntry:
    return replace(
        entry, status="uncalibrated", experiment=experiment, fitted_at_s=float(t0_s), sample_id=int(sample_id)
    )


def refused_table(
    table: CalibrationTable, groups: Sequence[str], experiment: str, t0_s: float, sample_id: int
) -> CalibrationTable:
    """``table`` with every entry of the groups a REFUSED ``experiment`` would have written marked ``uncalibrated``."""
    for group in groups:
        if group == "field":
            table = replace(table, field=_uncalibrated(table.field, experiment, t0_s, sample_id))
        elif group == "ms":
            table = replace(
                table,
                ms={
                    pair: replace(
                        wf,
                        phi_s=_uncalibrated(wf.phi_s, experiment, t0_s, sample_id),
                        phi_m=_uncalibrated(wf.phi_m, experiment, t0_s, sample_id),
                    )
                    for pair, wf in table.ms.items()
                },
            )
        else:
            entries: dict[Any, CalEntry] = getattr(table, group)
            changes: dict[str, Any] = {
                group: {k: _uncalibrated(e, experiment, t0_s, sample_id) for k, e in entries.items()}
            }
            table = replace(table, **changes)
    return table


def upstream_status(table: CalibrationTable, needs: Sequence[str]) -> str | None:
    """The first uncalibrated upstream entry group an experiment reads, or None when every one is usable."""
    for name in needs:
        if name == "field":
            if not usable(table.field):
                return "field"
            continue
        if name == "ms":
            if any(not usable(w.phi_s) for w in table.ms.values()):
                return "ms"
            continue
        entries = getattr(table, name)
        if any(not usable(e) for e in entries.values()):
            return name
    return None


def frame_shifts(device: Device, table: CalibrationTable) -> dict[int, float]:
    """The true transition minus the table's frame per ion (what the engine adds to H_int)."""
    out: dict[int, float] = {}
    for i in range(device.crystal.n_ions):
        sp = device.crystal.species[i]
        f_true, _d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], device.field.B_gauss)
        entry = table.qubit_freq.get(i)
        out[i] = float(f_true - entry.value) if usable(entry) and entry is not None else 0.0
    return out


def full_calibration(
    device: Device,
    *,
    seed: int = 0,
    t0_s: float = 0.0,
    experiments: Sequence[str] = ("all",),
    pairs: Sequence[tuple[int, int]] | None = None,
    scans: CalibrationScans | None = None,
    options: SolverOptions | None = None,
    builder_options: BuilderOptions | None = None,
    surrogate: SurrogateReport | None = None,
    **surrogate_kwargs: Any,
) -> CalibrationReport:
    """Calibrate ``device`` by simulated experiments in the dependency order (module docstring), starting
    from ``surrogate`` (default: ``surrogate_table`` with ``surrogate_kwargs``) and restricted to ``experiments``."""
    from qutip_trap.control.schedule import resolve_drives
    from qutip_trap.control.shaping import phase_shifted, scaled
    from qutip_trap.experiments.entangling import _shift_detuning, ms_phase_scan, ms_scan, parity_scan
    from qutip_trap.experiments.imaging import crystal_image
    from qutip_trap.experiments.light import crosstalk_scan, field_scan, stark_scan
    from qutip_trap.experiments.micromotion import micromotion_scan
    from qutip_trap.experiments.motion import heating_rate, mode_spectroscopy
    from qutip_trap.experiments.readout import detection_histogram
    from qutip_trap.experiments.single_ion import rabi_scan, ramsey_frequency
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.noise.sampling import quiet_sample

    sc = scans or CalibrationScans()
    wanted = set(ORDER) if "all" in experiments else {ALIASES.get(str(e), str(e)) for e in experiments}
    unknown = wanted - set(ORDER)
    if unknown:
        raise ValueError(
            f"unknown calibration experiments {sorted(unknown)}; known: {ORDER} and the aliases {sorted(ALIASES)}"
        )
    drives, ent = resolve_drives(device)
    n = device.crystal.n_ions
    sur = surrogate or surrogate_table(
        device,
        seed=seed,
        t0_s=t0_s,
        pairs=pairs,
        options=options,
        builder_options=builder_options,
        **surrogate_kwargs,
    )
    table = sur.table
    notes: list[str] = list(sur.notes)
    results: dict[str, ExperimentResult] = {}
    refused: dict[str, str] = {}
    # the dynamical sample the calibration is fitted under
    if device.noise.is_quiet(device):
        sample = quiet_sample(0, t0_s)
    else:
        rng = np.random.default_rng([int(seed), 0x_CA1])
        sample = device.noise.sample(rng, device=device, t_s=t0_s, duration_s=1e-3, sample_id=0)
    sid = int(sample.sample_id)
    nbar_belief = {m: float(e.value) for m, e in table.nbar.items()}
    common: dict[str, Any] = {
        "shots": sc.shots,
        "readout": True,
        "sample": sample,
        "seed": seed,
        "options": options,
        "builder_options": builder_options,
        "nbar": nbar_belief,
    }

    def refuse(name: str, keys: Sequence[str], reason: str) -> None:
        nonlocal table
        refused[name] = reason
        notes.append(f"{name} refused: {reason}")
        table = refused_table(table, keys, name, t0_s, sid)

    def check(name: str) -> bool:
        bad = upstream_status(table, UPSTREAM[name])
        if bad is not None:
            refuse(name, PRODUCES[name], f"upstream entry group {bad!r} is uncalibrated")
            return False
        return True

    # 0. the crystal image: is the crystal what the device says it is?
    if "crystal_image" in wanted:
        img = crystal_image(Machine(device), **{**common, "stream": "crystal_image"})
        results["crystal_image"] = img
        if not img.converged:
            notes.append(
                "crystal_image: the imaged crystal differs from the nominal one; every later fit runs on the nominal crystal"
            )
    # 1. the field
    if "field_scan" in wanted and check("field_scan"):
        res = field_scan(
            Machine(device),
            0,
            sc.ramsey_delays_s,
            b_seed_gauss=float(table.field.value),
            probe_hz=_PROBE_HZ,
            table=table,
            rabi_hz_belief=_belief(table, 0, drives[0]),
            **{**common, "stream": "field_scan[0]"},
        )
        results["field_scan[0]"] = res
        table = replace(table, field=_entry(res, "B_gauss", "field_scan", "conv.curvature_naming", t0_s, sid))
        # the qubit frequencies follow the calibrated field until the Ramsey-frequency experiment measures them directly
        if table.field.status == "calibrated":
            qf: dict[int, CalEntry] = {}
            for i in range(n):
                sp = device.crystal.species[i]
                f_b, d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], table.field.value)
                qf[i] = CalEntry(
                    float(f_b),
                    abs(float(d1)) * table.field.uncertainty,
                    "calibrated",
                    "field_scan",
                    "conv.frequencies",
                    float(t0_s),
                    sid,
                )
            table = replace(table, qubit_freq=qf)
    # 2. micromotion compensation
    if "micromotion_scan" in wanted and check("micromotion_scan"):
        ranges = sc.micromotion_ranges
        if ranges is None:
            ranges = {"Ex": (-50.0, 50.0), "Ey": (-50.0, 50.0)} if device.trap.path == "explicit" else {}
        entries: dict[str, CalEntry] = {}
        if device.trap.rf is None:
            # no rf record: no excess micromotion (beta = 0, C0 = 1), a statement of the device model and not a
            # measurement, so the entries are usable seeds
            no_rf = CalEntry(
                0.0,
                0.0,
                "seed",
                "derived_no_rf_record",
                "anchor.trap.berkeland_excess_micromotion",
                t0_s,
                sid,
            )
            entries.update({f"shim[{name}]": no_rf for name in ranges})
            entries.update({f"beta[{drives[i].table_key_beam}]": no_rf for i in range(n)})
            notes.append(
                "micromotion_scan: no rf record on the trap; excess micromotion is not modelled (beta = 0) and the entries "
                "are seeds of the device model, not measurements"
            )
        else:
            res = micromotion_scan(
                Machine(device),
                0,
                int(detection_beams(device, 0)[0]),
                ranges,
                method="rf_photon_correlation",
                points=5,
                **{**common, "stream": "micromotion_scan[0]"},
            )
            results["micromotion_scan[0]"] = res
            for key, val in res.fitted.items():
                if key.startswith("shim[") or key.startswith("beta["):
                    entries[key] = CalEntry(
                        float(val[0]),
                        float(val[1]) if math.isfinite(val[1]) else 0.0,
                        "calibrated" if res.converged else "uncalibrated",
                        "micromotion_scan",
                        "anchor.trap.berkeland_excess_micromotion",
                        t0_s,
                        sid,
                    )
        table = replace(table, micromotion=entries)
    # 3. modes, occupations and Lamb-Dicke parameters
    if "sideband_spectroscopy" in wanted and check("sideband_spectroscopy"):
        modes_entries = dict(table.modes)
        nbar_entries = dict(table.nbar)
        eta_entries = dict(table.lamb_dicke)
        for m in _coupled_modes(device, drives):
            ion = _probe_ion(device, m)
            res = mode_spectroscopy(
                Machine(device),
                ion,
                m,
                seed_hz=float(table.modes[m].value),
                span=sc.mode_span,
                coarse_points=sc.mode_coarse_points,
                fine_points=sc.mode_fine_points,
                rabi_hz_belief=_belief(table, ion, drives[ion]),
                table=table,
                qubit_shifts_hz=frame_shifts(device, table),
                **{**common, "stream": f"mode_spectroscopy[{m}]"},
            )
            results[f"mode_spectroscopy[{m}]"] = res
            modes_entries[m] = _entry(
                res, "mode_hz", "sideband_spectroscopy", "conv.sideband_lineshape", t0_s, sid
            )
            nbar_entries[m] = _entry(
                res, "nbar", "sideband_spectroscopy", "anchor.m3.thermometry_exactness", t0_s, sid
            )
            eta_entries[(ion, m)] = _entry(res, "eta", "sideband_spectroscopy", "conv.lamb_dicke", t0_s, sid)
        table = replace(table, modes=modes_entries, nbar=nbar_entries, lamb_dicke=eta_entries)
        nbar_belief = {m: float(e.value) for m, e in table.nbar.items() if usable(e)}
        common["nbar"] = nbar_belief
    # 4. carrier Rabi frequencies (nbar from the thermometry), then Stark and the qubit frequencies
    if "rabi_scan" in wanted and check("rabi_scan"):
        rabi_entries = dict(table.rabi)
        for i in range(n):
            spec = drives[i]
            if spec.kind == "microwave":
                continue
            seed_rabi = _belief(table, i, spec)
            ts = [float(x) for x in np.linspace(0.0, 10.0 * 0.5 / seed_rabi, sc.rabi_points)]  # ten pi times
            driven = _driven_mode(device, i, spec)
            res = rabi_scan(
                Machine(device),
                i,
                ts,
                nbar_fixed=nbar_belief.get(driven, 0.0) if driven is not None else 0.0,
                table=table,
                qubit_shifts_hz=frame_shifts(device, table),
                **{**common, "stream": f"rabi_scan[{i}]"},
            )
            results[f"rabi_scan[{i}]"] = res
            rabi_entries[(i, spec.table_key_beam)] = _entry(
                res, "f_rabi_hz", "rabi_scan", "conv.rabi_frequency", t0_s, sid
            )
        table = replace(table, rabi=rabi_entries)
    if "stark_scan" in wanted and check("stark_scan"):
        stark_entries = dict(table.stark)
        for i in range(n):
            spec = drives[i]
            if spec.kind not in ("raman", "optical_E1", "optical_E2"):
                continue
            res = stark_scan(
                Machine(device),
                i,
                sc.stark_delays_s,
                probe_hz=_PROBE_HZ,
                rabi_hz_belief=_belief(table, i, spec),
                table=table,
                qubit_shifts_hz=frame_shifts(device, table),
                **{**common, "stream": f"stark_scan[{i}]"},
            )
            results[f"stark_scan[{i}]"] = res
            stark_entries[(i, spec.table_key_beam)] = _entry(
                res, "stark_shift_hz", "stark_scan", "conv.stark_scaling_with_amplitude", t0_s, sid
            )
        table = replace(table, stark=stark_entries)
    if "ramsey_frequency" in wanted and check("ramsey_frequency"):
        qf_entries = dict(table.qubit_freq)
        for i in range(n):
            spec = drives[i]
            frame_hz = float(table.qubit_freq[i].value) if usable(table.qubit_freq.get(i)) else 0.0
            res = ramsey_frequency(
                Machine(device),
                i,
                sc.ramsey_delays_s,
                probe_hz=_PROBE_HZ,
                frame_hz=frame_hz,
                rabi_hz_belief=_belief(table, i, spec),
                rabi_hz=_belief(table, i, spec) if spec.kind == "microwave" else None,
                table=table,
                qubit_shifts_hz=frame_shifts(device, table),
                **{**common, "stream": f"ramsey_frequency[{i}]"},
            )
            results[f"ramsey_frequency[{i}]"] = res
            qf_entries[i] = _entry(res, "qubit_freq_hz", "ramsey_frequency", "conv.frequencies", t0_s, sid)
        table = replace(table, qubit_freq=qf_entries)
    # 5. crosstalk
    if "crosstalk_scan" in wanted and check("crosstalk_scan"):
        xt_entries = dict(table.crosstalk)
        ph_entries = dict(table.crosstalk_phase)
        for i in range(n):
            spec = drives[i]
            if spec.kind not in ("raman", "optical_E1", "optical_E2"):
                continue
            neighbours = [j for (a, j) in table.crosstalk if a == i]
            if not neighbours:
                continue
            eps_seed = max(abs(table.crosstalk[(i, j)].value) for j in neighbours)
            f_i = _belief(table, i, spec)
            ts = [float(x) for x in np.linspace(0.0, 0.5 / max(eps_seed * f_i, 1e-9), sc.crosstalk_points)]
            res = crosstalk_scan(
                Machine(device),
                i,
                ts,
                rabi_hz_belief=f_i,
                analysis_phases_rad=np.linspace(
                    0.0, 2.0 * math.pi, sc.crosstalk_phase_points, endpoint=False
                ),
                table=table,
                qubit_shifts_hz=frame_shifts(device, table),
                **{**common, "stream": f"crosstalk_scan[{i}]"},
            )
            results[f"crosstalk_scan[{i}]"] = res
            for j in neighbours:
                if f"eps[{j}]" in res.fitted:
                    xt_entries[(i, j)] = _entry(
                        res, f"eps[{j}]", "crosstalk_scan", "conv.crosstalk_ratio", t0_s, sid
                    )
                if f"phase_rad[{j}]" in res.fitted:
                    ph_entries[(i, j)] = _entry(
                        res, f"phase_rad[{j}]", "crosstalk_scan", "conv.crosstalk_ratio", t0_s, sid
                    )
        table = replace(table, crosstalk=xt_entries, crosstalk_phase=ph_entries)
    # 6. the entangling gates: amplitude and detuning, the phase alignment, parity; each scan checked under its own name,
    # both checks run so that each marks the groups it would have written
    entangling_checks = [check(name) for name in ("ms_scan", "parity_scan") if name in wanted]
    if entangling_checks and all(entangling_checks):
        ms_entries: dict[tuple[int, int], Waveform] = dict(table.ms)
        for pair, wf in list(table.ms.items()):
            if ent[pair[0]].kind != "raman":
                notes.append(
                    f"pair {pair}: the entangling scans take a Raman drive; waveform kept as calibrated by the surrogate"
                )
                continue
            modes_hz = {m: float(e.value) for m, e in table.modes.items() if usable(e)}
            # the closure amplitude the pulse solver predicts AT THE CALIBRATED MODE FREQUENCIES, which
            # the scan then refines (only when the amplitude scan follows)
            if "ms_scan" in wanted:
                wf = _resolved_at_calibrated_modes(device, pair, wf, ent, modes_hz, nbar_belief, notes)
                ms_entries[pair] = wf
                table = replace(table, ms=ms_entries)
            gate_kw = {
                "table": table,
                "qubit_shifts_hz": frame_shifts(device, table),
                "mode_frequencies_hz": modes_hz,
                **{k: v for k, v in common.items() if k != "nbar"},
                "nbar": {m: nbar_belief.get(m, 0.0) for m in wf.chi_m},
            }
            current = wf
            if "ms_scan" in wanted:
                amps = [float(x) for x in np.linspace(0.7, 1.3, sc.ms_amplitude_points)]
                res = ms_scan(
                    Machine(device),
                    pair,
                    amps,
                    sc.ms_detuning_offsets_hz,
                    **{**gate_kw, "stream": f"ms_scan[{pair}]"},
                )
                results[f"ms_scan[{pair}]"] = res
                if res.converged and "closure_scale" in res.fitted:
                    s_cl, s_unc = res.fitted["closure_scale"]
                    # the offset the SCALE was measured at (the fitted closure offset when the parabola converged)
                    off = res.fitted.get("closure_offset_used_hz", (0.0, 0.0))[0]
                    current = scaled(_shift_detuning(wf, off), s_cl)
                    chi = math.copysign(math.pi / 4.0, wf.chi_total_rad)
                    current = replace(
                        current,
                        chi_m={m: v * (chi / current.chi_total_rad) for m, v in current.chi_m.items()},
                        phi_s=CalEntry(
                            current.phi_s.value, 0.0, "calibrated", "ms_scan", "conv.ms_closure", t0_s, sid
                        ),
                        phi_m=CalEntry(
                            current.phi_m.value, s_unc, "calibrated", "ms_scan", "conv.ms_closure", t0_s, sid
                        ),
                    )
                    notes.append(
                        f"pair {pair}: closure at scale {s_cl:.5f} +- {s_unc:.2g}, offset {off:+.1f} Hz"
                    )
                else:
                    current = replace(
                        current,
                        phi_s=_uncalibrated(current.phi_s, "ms_scan", t0_s, sid),
                        phi_m=_uncalibrated(current.phi_m, "ms_scan", t0_s, sid),
                    )
                    refuse(
                        f"ms_scan[{pair}]", (), "the closure fit did not converge or lies at the scan edge"
                    )
                ms_entries[pair] = current
                table = replace(table, ms=ms_entries)
                gate_kw["table"] = table
                if usable(current.phi_s):
                    phases = [float(x) for x in np.linspace(0.0, math.pi, sc.phase_points, endpoint=False)]
                    res_ph = ms_phase_scan(
                        Machine(device),
                        pair,
                        phases,
                        inputs=sc.phase_inputs,
                        **{**gate_kw, "stream": f"ms_phase_scan[{pair}]"},
                    )
                    results[f"ms_phase_scan[{pair}]"] = res_ph
                    if res_ph.converged:
                        corr = {
                            q: float(res_ph.fitted[f"correction_rad[{q}]"][0])
                            for q in pair
                            if f"correction_rad[{q}]" in res_ph.fitted
                        }
                        current = phase_shifted(current, corr)
                        unc_ph = max(res_ph.fitted[f"correction_rad[{q}]"][1] for q in pair)
                        current = replace(
                            current,
                            phi_s=CalEntry(
                                current.phi_s.value + float(np.mean(list(corr.values()))),
                                unc_ph,
                                "calibrated",
                                "ms_phase_scan",
                                "conv.spin_motion_phases",
                                t0_s,
                                sid,
                            ),
                        )
                        notes.append(f"pair {pair}: spin-phase corrections {corr} rad")
                    else:
                        notes.append(
                            f"pair {pair}: the phase scan did not converge; the surrogate's phases are kept"
                        )
                    ms_entries[pair] = current
                    table = replace(table, ms=ms_entries)
                    gate_kw["table"] = table
            if "parity_scan" in wanted and usable(current.phi_s):
                phases = [float(x) for x in np.linspace(0.0, math.pi, sc.parity_points, endpoint=False)]
                res_par = parity_scan(
                    Machine(device), pair, phases, **{**gate_kw, "stream": f"parity_scan[{pair}]"}
                )
                results[f"parity_scan[{pair}]"] = res_par
                if res_par.converged:
                    notes.append(
                        f"pair {pair}: parity contrast {res_par.fitted['contrast'][0]:.4f}, Bell fidelity bound "
                        f"{res_par.fitted['bell_fidelity_bound'][0]:.4f}"
                    )
    # 7. detection
    if "detection_histogram" in wanted:
        det_entries = dict(table.detection)
        res = detection_histogram(
            Machine(device),
            0,
            sc.detection_records,
            windows_s=sc.detection_windows_s,
            seed=seed,
            t0_s=t0_s,
            sample_id=sid,
        )
        results["detection_histogram[0]"] = res
        for key in (
            "threshold",
            "window_s",
            "eps_B",
            "eps_D",
            "R_bright_detected_per_s",
            "R_dark_pumping_per_s",
            "R_bright_pumping_per_s",
        ):
            if key in res.fitted:
                det_entries[key] = _entry(
                    res,
                    key,
                    "detection_histogram",
                    "conv.readout_figure_of_merit"
                    if key in ("threshold", "window_s", "eps_B", "eps_D")
                    else "conv.mean_count_curve",
                    t0_s,
                    sid,
                )
        table = replace(table, detection=det_entries)
    # 8. heating rates
    if "heating_rate" in wanted and check("heating_rate"):
        heat_entries = dict(table.heating)
        for m in _coupled_modes(device, drives):
            ndot_seed = float(table.heating[m].value) if m in table.heating else 0.0
            if ndot_seed <= 0.0:
                # the delay scan spans 10/ndot: at a derived rate of zero there is no scan and the entry stays the seed
                notes.append(
                    f"heating_rate[{m}]: the derived rate is zero (a quiet device); the delay scan (span 10/ndot) does not "
                    "exist and the entry is left as the surrogate's seed"
                )
                continue
            ion = _probe_ion(device, m)
            delays = [
                float(x) for x in np.linspace(0.0, sc.heating_span_over_ndot / ndot_seed, sc.heating_delays)
            ]
            res = heating_rate(
                Machine(device),
                m,
                delays,
                ion=ion,
                nbar0=nbar_belief.get(m, 0.0),
                ndot_seed=ndot_seed,
                mode_hz=float(table.modes[m].value),
                rabi_hz_belief=_belief(table, ion, drives[ion]),
                table=table,
                qubit_shifts_hz=frame_shifts(device, table),
                **{**{k: v for k, v in common.items() if k != "nbar"}, "stream": f"heating_rate[{m}]"},
            )
            results[f"heating_rate[{m}]"] = res
            heat_entries[m] = _entry(
                res, "ndot_per_s", "heating_rate", "conv.electric_field_noise", t0_s, sid
            )
        table = replace(table, heating=heat_entries)
    table = replace(table, surrogate=False, fitted_at_s=float(t0_s))
    return CalibrationReport(
        table=table,
        surrogate=sur,
        results=results,
        refused=refused,
        notes=tuple(notes),
        sample=sample,
        experiments=tuple(e for e in ORDER if e in wanted),
    )


# ---- helpers ------------------------------------------------------------------------------------------------------------------


def _resolved_at_calibrated_modes(
    device: Device,
    pair: tuple[int, int],
    wf: Waveform,
    ent: Mapping[int, GateDrive],
    modes_hz: Mapping[int, float],
    nbar_belief: Mapping[int, float],
    notes: list[str],
) -> Waveform:
    """``wf`` re-solved at the table's mode frequencies with its duration and beat-note rule, keeping its phase entries (the
    spot check only rescales amplitudes and the phase scan aligns the axis afterwards); a solver failure keeps ``wf`` with a
    note."""
    from qutip_trap.calibration.surrogate import surrogate_waveform
    from qutip_trap.control.shaping import ClosureError

    spec = ent[pair[0]]
    if wf.kind != "ms" or not modes_hz or len(spec.beams) != 2:
        return wf
    try:
        shaped, _gm = surrogate_waveform(
            device,
            (int(pair[0]), int(pair[1])),
            (int(spec.beams[0]), int(spec.beams[1])),
            nbar={m: float(nbar_belief.get(m, 0.0)) for m in wf.chi_m},
            duration_s=float(wf.duration_s),
            mode_frequencies_hz=modes_hz,
        )
    except (ClosureError, ValueError) as exc:
        notes.append(
            f"pair {pair}: no waveform at the calibrated mode frequencies ({exc}); the surrogate's solution is kept and "
            "the scan refines it by a rigid detuning shift"
        )
        return wf
    out = replace(shaped.waveform, phi_s=wf.phi_s, phi_m=wf.phi_m)
    shown = {m: round(float(modes_hz[m]), 1) for m in sorted(wf.chi_m) if m in modes_hz}
    notes.append(f"pair {pair}: waveform re-solved at the calibrated mode frequencies {shown}")
    return out


def _belief(table: CalibrationTable, ion: int, spec: GateDrive) -> float:
    entry = table.rabi.get((ion, spec.table_key_beam))
    if entry is None:
        raise CalibrationError(f"no Rabi entry for ion {ion} under drive {spec}")
    return float(entry.value)


def _etas(device: Device, ion: int, spec: GateDrive) -> dict[int, float] | None:
    """The Lamb-Dicke parameters of the drive's beams on ``ion``; None for a drive with no wavevector."""
    from qutip_trap.control.pulses import Drive, Tone
    from qutip_trap.light.raman import lamb_dicke_parameters

    if not spec.beams:
        return None
    probe = Drive(
        spec.kind, (ion,), (Tone(0.0, 0.0, 1.0),), spec.beams, 0.0, {}, light_shift=spec.light_shift
    )
    dk = probe.delta_k(device.beams)
    if float(np.linalg.norm(dk)) == 0.0:
        return None
    return lamb_dicke_parameters(device, ion, dk)[0]


def _coupled_modes(device: Device, drives: Mapping[int, GateDrive]) -> list[int]:
    """The modes any single-qubit drive couples to (|eta| > 1e-3), in index order."""
    out: set[int] = set()
    for ion, spec in drives.items():
        etas = _etas(device, ion, spec)
        if etas is not None:
            out.update(m for m, e in etas.items() if abs(e) > 1e-3)
    return sorted(out)


def _probe_ion(device: Device, mode: int) -> int:
    return int(np.argmax(np.abs(device.crystal.modes[mode].eigenvector)))


def _driven_mode(device: Device, ion: int, spec: GateDrive) -> int | None:
    etas = _etas(device, ion, spec)
    if etas is None:
        return None
    m = max(etas, key=lambda k: abs(etas[k]))
    return m if abs(etas[m]) > 0.0 else None
