"""The compile-calibrate-schedule prefix of ``run`` (PLAN.md Section 3.4), one function that ``run`` and ``Machine.schedule``
share (docs/api_implementation_plan.md 1.3; 0.3.0 moves the rest of the pipeline here).

    Device + Circuit
      -> compile: Circuit -> native gates (phase-tracked)                    [control.compiler]
      -> calibrate (surrogate, cached per Device): CalibrationTable         [calibration.surrogate]
      -> the calibrated micromotion shims programmed onto the device        [experiments.micromotion]
      -> schedule: native gates -> Pulses with absolute times, the measure event   [control.schedule]

The steps, their notes and their order are ``run``'s of 0.1.0, moved here unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from qutip_trap.control.compiler import CompileReport, compile_with_report
from qutip_trap.control.schedule import CrosstalkSuppression, GateDrive, Schedule, resolve_drives, schedule
from qutip_trap.dynamics.engine import SolverOptions

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.hamiltonian import BuilderOptions


@dataclass(frozen=True)
class Prefix:
    """What the prefix produced: the compile report and its native circuit, the table (built or given), the device the run
    evolves (the table's calibrated shims programmed onto it), the schedule, the resolved drive maps, the options after the
    ``internal_levels`` adjustment, and the notes so far, in the order ``run`` reports them."""

    report: CompileReport
    compiled: Circuit
    table: CalibrationTable
    device: Device
    schedule: Schedule
    gate_drives: dict[int, GateDrive]
    entangling_drives: dict[int, GateDrive]
    options: SolverOptions
    notes: tuple[str, ...]


def compile_calibrate_schedule(
    circuit: Circuit,
    device: Device,
    *,
    table: CalibrationTable | None = None,
    seed: int = 0,
    t0_s: float = 0.0,
    options: SolverOptions | None = None,
    gate_drives: Mapping[int, GateDrive] | None = None,
    entangling_drives: Mapping[int, GateDrive] | None = None,
    builder_options: BuilderOptions | None = None,
    caps: Mapping[int, int] | None = None,
    calibrate_kwargs: Mapping[str, Any] | None = None,
    entangler: Literal["ms", "zz"] = "ms",
    parallel: bool | None = None,
    crosstalk_suppression: CrosstalkSuppression = "none",
    stark_compensation: bool = True,
    internal_levels: int = 2,
) -> Prefix:
    """Compile, calibrate (the cached surrogate when ``table`` is None), program the calibrated shims and schedule: the
    prefix of Section 3.4's pipeline, with the keyword arguments of ``run`` it reads (module docstring)."""
    opts = options or SolverOptions()
    # the drive maps: the call's keyword arguments first, then the device's roles, then the inference (Device.roles, 0.2.0)
    drives, ent_drives = resolve_drives(device, gate_drives, entangling_drives)
    notes: list[str] = []
    # Section 4.5.5: "Leakage is therefore simulated, not estimated, whenever d > 2". A d > 2 register whose scattering
    # channels are off carries leakage levels that nothing can populate and reports no estimate for them either, so the
    # combination is refused rather than defaulted silently: the flag is turned on and the run says so. Keep
    # internal_levels = 2 for the d = 2 per-pulse estimate path of Section 4.3.2.
    if internal_levels > 2 and not opts.scattering_channels:
        # Section 4.5.5 asks for the LEAKAGE channel (and with it the spin-flip and Rayleigh operators on the register);
        # the recoil displacements D(i(eta_abs - eta_em)) of Section 4.5.5's photon-recoil term are a separate, expensive
        # choice (one displacement-dressed operator per emission direction, per pulse, per branch: 30 to 50x the cost of
        # the register-only operators on the two-ion fixture), so the automatic switch turns the channels on WITHOUT them
        # unless the caller asked for the vector quadrature; scattering_channels=True with scattering_recoil="minimal" or
        # "vector" is the explicit request for recoil (M7 fixer's E-10, consolidated 2026-09-08)
        recoil = opts.scattering_recoil if opts.scattering_recoil == "vector" else "off"
        opts = replace(opts, scattering_channels=True, scattering_recoil=recoil)
        notes.append(
            f"internal_levels = {internal_levels} > 2: scattering_channels turned ON with scattering_recoil={recoil!r} "
            "(Section 4.5.5, 'leakage is simulated, not estimated, whenever d > 2'; the recoil displacements are an "
            "explicit choice: pass scattering_channels=True with scattering_recoil='minimal' or 'vector'); pass "
            "internal_levels=2 for the d = 2 estimate path instead"
        )
    # 1. compile
    report = compile_with_report(circuit, device, entangler=entangler)
    compiled = report.circuit
    # 2. calibrate (surrogate, cached per device and seed, Section 7.5) when no table is given
    if table is None:
        from qutip_trap.calibration.cache import cached_surrogate

        kw = dict(calibrate_kwargs or {})
        kw.setdefault("pairs", compiled.entangling_pairs())
        sur = cached_surrogate(
            device,
            seed=seed,
            t0_s=t0_s,
            gate_drives=drives,
            entangling_drives=ent_drives,
            options=opts,
            builder_options=builder_options,
            caps=caps,
            **kw,
        )
        table = sur.table
        notes.extend(sur.notes)
    elif not table.is_current_for(device.hash()):
        notes.append(
            "calibration table fitted for another device configuration (hash mismatch): played as given, never regenerated "
            "silently (Section 7.5)"
        )
    # 2b. the calibrated micromotion compensation: the shim settings the table carries are what the machine has PROGRAMMED,
    # so the run evolves the compensated device and a stale calibration against a drifted stray field leaves the residual
    # excess micromotion a laboratory would have (Section 7.5: "stores shims and beta in the table, so that compensation is
    # calibrated, drifts with the stray field between calibrations and is re-nulled like a laboratory re-nulls it").
    # ``device_with_compensation`` re-solves the crystal, so the ions' displacement, the beams' intensity at the ions and
    # the Lamb-Dicke parameters move together. No device PARAMETER changed - only a programmed voltage - so the hash the
    # table is compared against above stays the uncompensated device's.
    shim_entries = {
        name: entry
        for name, entry in table.micromotion.items()
        if name.startswith("shim[") and name.endswith("]")
    }
    # only what the compensation experiment MEASURED is programmed: a seed shim is the device's own setting (already in the
    # device) and an uncalibrated one is a compensation the calibration could not establish, which leaves the device as it is
    shims = {
        name[len("shim[") : -1]: float(entry.value)
        for name, entry in shim_entries.items()
        if entry.status == "calibrated"
    }
    unresolved = sorted(name for name, entry in shim_entries.items() if entry.status == "uncalibrated")
    if unresolved:
        notes.append(
            f"micromotion compensation uncalibrated for {unresolved}: the run keeps the device's own shim settings and "
            "carries whatever excess micromotion they leave (Section 7.5)"
        )
    if shims:
        from qutip_trap.experiments.micromotion import device_with_compensation

        compensated = device_with_compensation(device, shims)
        if compensated.trap != device.trap:
            device = compensated
            notes.append(
                "calibrated micromotion compensation applied to the run: "
                + ", ".join(f"{k} = {v:.6g}" for k, v in sorted(shims.items()))
                + " (Section 7.5; the reported device hash is the uncompensated device's)"
            )
    # 3. schedule
    sched = schedule(
        compiled,
        device,
        table,
        gate_drives=drives,
        entangling_drives=ent_drives,
        t0_s=0.0,
        parallel=parallel,
        crosstalk_suppression=crosstalk_suppression,
        stark_compensation=stark_compensation,
    )
    return Prefix(
        report=report,
        compiled=compiled,
        table=table,
        device=device,
        schedule=sched,
        gate_drives=drives,
        entangling_drives=ent_drives,
        options=opts,
        notes=tuple(notes),
    )


__all__ = ["Prefix", "compile_calibrate_schedule"]

_ = Sequence  # re-exported typing name for the callers that annotate scan sequences
