"""``Device.derived()``: every computed number of a device with its provenance id (PLAN.md Sections 3.1, 3.3, 14.5).

The keys follow the calibration table's entries, so the surrogate's ``seed`` entries and a full calibration's
``calibrated`` ones compare entry by entry; every key carries the ledger id of the convention or anchor it follows. A
quantity whose input the device does not carry (a Mathieu record without an rf record, a microwave drive's Rabi frequency,
a detection rate without a detection beam) is absent and named in the notes; a record the device carries but that cannot
be evaluated (an rf drive outside the stability region, a layout without an rf null) raises.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.units import E_C

if TYPE_CHECKING:
    from qutip_trap.device.model import DerivedQuantities, Device


def _put_trap_geometry(device: Device, put: Callable[[str, float, str], None], notes: list[str]) -> None:
    """The Section 4.1 quantities reported with every device that uses the trap module: ``ion_height_m`` and
    ``trap_depth_ev`` on the surface path (4.1.6); ``second_order_doppler[i]``, <Delta nu/nu> = -<v^2>/(2 c^2) of the
    excess micromotion (4.1.1); and ``pseudopotential_error`` (4.1.7), |u_1|/s with u_1 = -(q/2) u_0 the signed micromotion
    amplitude and s the smallest inter-ion distance, the small parameter of Kaufmann's rf-modulated Coulomb curvature (a
    single ion gets a note instead)."""
    from qutip_trap.trap.micromotion import second_order_doppler_fraction

    trap = device.trap
    crystal = device.crystal
    if trap.path == "surface":
        assert trap.rf is not None  # the surface path is rf plus an electrode plane
        model = trap.surface_model()
        put("ion_height_m", float(model.rf_null()[1]), "conv.surface_electrode_geometry")
        depth = model.depth_j(trap.rf.voltage_peak_v, float(crystal.masses_kg[0]), trap.rf.omega_rad_s)
        put("trap_depth_ev", depth / E_C, "conv.surface_electrode_geometry")
    if trap.rf is None:
        notes.append(
            "second-order Doppler shift and the pseudopotential error estimate need the rf record (Trap.rf)"
        )
        return
    amplitudes: list[np.ndarray] = []
    for i, sp in enumerate(crystal.species):
        u1 = trap.micromotion_amplitude_m(sp)
        amplitudes.append(u1)
        put(
            f"second_order_doppler[{i}]",
            second_order_doppler_fraction(u1, trap.rf.omega_rad_s),
            "conv.micromotion_amplitude_convention",
        )
    if crystal.n_ions < 2:
        notes.append(
            "pseudopotential error estimate: a single ion has no Coulomb coupling for the off-null micromotion to "
            "modulate (Section 4.1.7); the |u_1|/s ratio needs at least two ions"
        )
        return
    pos = np.asarray(crystal.positions_m, dtype=float)
    gaps = [
        float(np.linalg.norm(pos[i] - pos[j]))
        for i in range(crystal.n_ions)
        for j in range(i + 1, crystal.n_ions)
    ]
    put(
        "pseudopotential_error",
        max(float(np.linalg.norm(u1)) for u1 in amplitudes) / min(gaps),
        "conv.pseudopotential_error_estimate",
    )


def derived_quantities(device: Device) -> DerivedQuantities:
    from qutip_trap.control.schedule import ScheduleError, default_gate_drives
    from qutip_trap.device.model import DerivedQuantities
    from qutip_trap.light.raman import crosstalk_ratios, derive_optical_drive, derive_raman_drive
    from qutip_trap.light.roles import NoDetectionBeamError, detection_beams
    from qutip_trap.readout.fluorescence import detection_rates_for_ion

    values: dict[str, float] = {}
    prov: dict[str, str] = {}
    notes: list[str] = []

    def put(key: str, value: float, pid: str) -> None:
        values[key] = float(value)
        prov[key] = pid

    crystal = device.crystal
    n = crystal.n_ions
    b = device.field.B_gauss
    put("field_gauss", b, "conv.curvature_naming")
    # the qubit transitions and their Zeeman sensitivities (Section 4.5.1)
    for i in range(n):
        sp = crystal.species[i]
        f0, d1, d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], b)
        put(f"qubit_freq_hz[{i}]", f0, "conv.frequencies")
        put(f"dnu_dB_hz_per_g[{i}]", d1, "conv.curvature_naming")
        put(f"d2nu_dB2_hz_per_g2[{i}]", d2, "conv.curvature_naming")
    # the trap: the first ion's secular frequencies, and its Mathieu parameters and C0 when the rf record exists (4.1.1)
    ref = crystal.species[0]
    for ax, f in zip(("x", "y", "z"), device.trap.secular_hz(ref)):
        put(f"secular_hz[{ax}]", f, "conv.rf_amplitude_pseudopotential")
    if device.trap.rf is not None:
        params = device.trap.mathieu(ref)
        for k, ax in enumerate(("x", "y", "z")):
            put(f"mathieu_q[{ax}]", float(params.q_effective[k]), "conv.mathieu_sign")
            put(f"mathieu_beta[{ax}]", float(params.beta[k]), "conv.mathieu_sign")
            put(f"C0[{ax}]", float(params.C0[k]), "conv.micromotion_correction")
    _put_trap_geometry(device, put, notes)
    # the modes and their heating rates (Sections 4.1.3, 4.1.5)
    for m, mode in enumerate(crystal.modes):
        put(f"mode_hz[{m}]", mode.omega_hz, "conv.mode_index")
    for m, rate in device.noise.heating_rates_quanta_per_s(device).items():
        put(f"heating_rate_per_s[{m}]", rate, "conv.electric_field_noise")
    # the single-qubit drives the beams identify (Section 4.3.2): Rabi frequencies, Stark shifts, crosstalk, Lamb-Dicke parameters
    try:
        drives = default_gate_drives(device)
    except ScheduleError as exc:
        drives = {}
        notes.append(f"gate drives: {exc}")
    for i, spec in drives.items():
        if spec.kind == "raman":
            dd = derive_raman_drive(device, i, (spec.beams[0], spec.beams[1]), scattering=False)
            pid = "conv.two_photon_rabi"
        elif spec.kind in ("optical_E1", "optical_E2"):
            dd = derive_optical_drive(device, i, spec.beams[0], scattering=False)
            pid = "conv.rabi_frequency"
        else:
            notes.append(f"ion {i}: a {spec.kind} drive has no derivable carrier Rabi frequency")
            continue
        key = spec.table_key_beam
        put(f"rabi_hz[({i}, {key})]", dd.carrier_rabi_hz, pid)
        put(f"stark_hz[({i}, {key})]", dd.stark_shift_hz, "conv.stark_scaling_with_amplitude")
        for m, eta in dd.etas.items():
            if eta != 0.0:
                put(f"eta[({i}, {m})]", eta, "conv.lamb_dicke")
        if dd.micromotion is not None:
            put(f"micromotion_beta[{i}]", dd.micromotion.total, "conv.micromotion_amplitude_convention")
        for j, eps in crosstalk_ratios(device, i, spec.beams, kind=spec.kind).items():
            put(f"crosstalk[({i}, {j})]", abs(eps), "conv.crosstalk_ratio")
    # the detection rates (Section 8.1)
    for i in range(n):
        try:
            idx = detection_beams(device, i)
        except NoDetectionBeamError as exc:
            notes.append(f"ion {i}: {exc}")
            continue
        rates, _scheme, _model = detection_rates_for_ion(
            crystal.species[i],
            b,
            device.field.direction,
            [device.beams[k] for k in idx],
            position_m=tuple(float(x) for x in crystal.positions_m[i]),
        )
        put(f"R_bright_per_s[{i}]", rates.R_bright_per_s, "conv.scattering_rate_object")
        put(f"R_dark_pumping_per_s[{i}]", rates.R_dark_pumping_per_s, "anchor.m3a.yb171_leakage_prefactors")
        put(
            f"R_bright_pumping_per_s[{i}]",
            rates.R_bright_pumping_per_s,
            "anchor.m3a.yb171_leakage_prefactors",
        )
    return DerivedQuantities(values=values, provenance=prov, notes=tuple(notes))
