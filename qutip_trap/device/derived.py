"""``Device.derived()``: every computed number of a device with its provenance id (PLAN.md Sections 3.1, 3.3, 14.5; M8).

Design principle (Section 3.1): device parameters in, everything else derived. The dictionary this module returns is the
device's self-description in derived quantities, keyed like the calibration table's entries so that the surrogate's ``seed``
entries and a full calibration's ``calibrated`` ones can be compared entry by entry; every key carries the ledger id of the
convention or anchor it follows (``docs/provenance/ledger.yaml``). A quantity the device cannot derive (a microwave drive's Rabi
frequency, a detection rate without a detection beam) is absent and named in ``provenance["notes"]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.units import E_C

if TYPE_CHECKING:
    from qutip_trap.device.model import DerivedQuantities, Device


def _put_trap_geometry(device: Device, put: Callable[[str, float, str], None], notes: list[str]) -> None:
    """The Section 4.1 quantities that are "reported with every device that uses" the trap module.

    - 4.1.6: "the module reports h and the trap depth with every device that uses it" -> ``ion_height_m`` and
      ``trap_depth_ev`` on the surface path (the rf-null height above the electrode plane and the pseudopotential
      barrier at the escape point);
    - 4.1.1: "reports the implied second-order Doppler shift for completeness" -> ``second_order_doppler[i]``,
      <Delta nu/nu> = -<v^2>/(2 c^2) of the excess micromotion, negative by construction;
    - 4.1.7 close: "reports the pseudopotential error estimate (q/2 times the configured off-null displacement, in
      units of the mode spacing) with every device" -> ``pseudopotential_error``, |u_1|/s with u_1 = -(q/2) u_0 the
      signed micromotion amplitude and s the smallest inter-ion distance. That ratio is the small parameter of
      Kaufmann's mechanism: the Coulomb curvature ~ 1/r^3 is rf-modulated at fractional depth ~3 |u_1|/s wherever the
      ions sit off the rf null, and the pseudopotential modes of 4.1.3 are exact only at |u_1|/s = 0. "Mode spacing"
      is read as the ION spacing, the only length in the clause; a single ion has no Coulomb coupling to modulate and
      gets a note instead of a number.
    """
    from qutip_trap.trap.micromotion import second_order_doppler_fraction

    trap = device.trap
    crystal = device.crystal
    if trap.path == "surface":
        try:
            model = trap.surface_model()
            null = model.rf_null()
            put("ion_height_m", float(null[1]), "conv.surface_electrode_geometry")
        except (ValueError, NotImplementedError) as exc:
            notes.append(f"ion height: {exc}")
        if trap.rf is not None:
            try:
                depth = model.depth_j(
                    trap.rf.voltage_peak_v,
                    float(crystal.masses_kg[0]),
                    trap.rf.omega_rad_s,
                )
                put("trap_depth_ev", depth / E_C, "conv.surface_electrode_geometry")
            except (ValueError, NotImplementedError) as exc:
                notes.append(f"trap depth: {exc}")
    if trap.rf is None:
        notes.append(
            "second-order Doppler shift and the pseudopotential error estimate need the rf record (Trap.rf)"
        )
        return
    amplitudes: list[np.ndarray] = []
    for i, sp in enumerate(crystal.species):
        try:
            u1 = trap.micromotion_amplitude_m(sp)
        except (ValueError, NotImplementedError) as exc:
            notes.append(f"ion {i}: micromotion amplitude: {exc}")
            return
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
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.fluorescence import detection_rates_for_ion
    from qutip_trap.trap.heating import heating_rate_quanta_per_s, single_sided_from_spectrum

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
        # ValueError is the layer's own "this device has no rf record / lies outside the stability region" signal
        # (UnstableMathieuError subclasses it); an AttributeError, IndexError or TypeError would be a BUG in Trap.mathieu
        # and must surface as one rather than as a line on the device card (M8 audit B9)
        except ValueError as exc:
            notes.append(f"Mathieu parameters: {exc}")
    _put_trap_geometry(device, put, notes)
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
        # no correlation length, so no multi-ion projection: the single-ion rate per mode, with the SAME S_E adapter the
        # heating layer uses (the record's own folding, zero above the band, white level once; conv.electric_field_noise_adapter)
        s_e = single_sided_from_spectrum(device.noise.S_E)
        for m, mode in enumerate(crystal.modes):
            w = mode.omega_rad_s
            ion = int(np.argmax(np.abs(mode.eigenvector)))
            put(
                f"heating_rate_per_s[{m}]",
                heating_rate_quanta_per_s(s_e(w), float(crystal.masses_kg[ion]), w),
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
