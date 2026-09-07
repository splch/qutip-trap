"""``Device.derived()``: every computed number of a device with its provenance id (PLAN.md Sections 3.1, 3.3, 14.5; M8).

Design principle (Section 3.1): device parameters in, everything else derived. The dictionary this module returns is the
device's self-description in derived quantities, keyed like the calibration table's entries so that the surrogate's ``seed``
entries and a full calibration's ``calibrated`` ones can be compared entry by entry; every key carries the ledger id of the
convention or anchor it follows (``docs/provenance/ledger.yaml``). A quantity the device cannot derive (a microwave drive's Rabi
frequency, a detection rate without a detection beam) is absent and named in ``provenance["notes"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qutip_trap.device.model import DerivedQuantities, Device


def derived_quantities(device: Device) -> DerivedQuantities:
    from qutip_trap.control.schedule import ScheduleError, default_gate_drives
    from qutip_trap.device.model import DerivedQuantities
    from qutip_trap.light.raman import crosstalk_ratios, derive_optical_drive, derive_raman_drive
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.fluorescence import detection_rates_for_ion
    from qutip_trap.trap.heating import heating_rate_quanta_per_s, single_sided_from_two_sided

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
    # the trap: secular frequencies, Mathieu parameters and C0 when the rf record exists (Section 4.1.1)
    ref = crystal.species[0]
    try:
        sec = device.trap.secular_hz(ref)
        for ax, f in zip(("x", "y", "z"), sec):
            put(f"secular_hz[{ax}]", f, "conv.rf_amplitude_pseudopotential")
    except ValueError as exc:
        notes.append(f"secular frequencies: {exc}")
    if device.trap.rf is not None:
        try:
            params = device.trap.mathieu(ref)
            for k, ax in enumerate(("x", "y", "z")):
                put(f"mathieu_q[{ax}]", float(params.q_effective[k]), "conv.mathieu_sign")
                put(f"mathieu_beta[{ax}]", float(params.beta[k]), "conv.mathieu_sign")
                put(f"C0[{ax}]", float(params.C0[k]), "conv.micromotion_correction")
        except (ValueError, AttributeError, IndexError, TypeError) as exc:
            notes.append(f"Mathieu parameters: {exc}")
    # the modes and their heating rates (Sections 4.1.3, 4.1.5)
    for m, mode in enumerate(crystal.modes):
        put(f"mode_hz[{m}]", mode.omega_hz, "conv.mode_index")
    heating = (
        device.noise.heating_rates_quanta_per_s(device)
        if device.noise.correlation_length_m is not None
        else {}
    )
    for m, rate in heating.items():
        put(f"heating_rate_per_s[{m}]", rate, "conv.electric_field_noise")
    if not heating and not device.noise.S_E.is_zero():
        spec_e = device.noise.S_E
        for m, mode in enumerate(crystal.modes):
            w = mode.omega_rad_s
            s_two = float(np.interp(w, spec_e.omega_rad_s, spec_e.S, left=0.0, right=0.0))
            ion = int(np.argmax(np.abs(mode.eigenvector)))
            put(
                f"heating_rate_per_s[{m}]",
                heating_rate_quanta_per_s(
                    float(single_sided_from_two_sided(s_two)), float(crystal.masses_kg[ion]), w
                ),
                "conv.electric_field_noise",
            )
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
        except ValueError as exc:
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


__all__ = ["derived_quantities"]
