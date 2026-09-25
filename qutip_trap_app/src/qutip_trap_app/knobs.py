"""The device knobs of Level 4 and how a change propagates downward (PLAN.md Section 14.4).

A knob is one named device parameter with its plain label, physics term, unit, section, governing ledger record and the
range the page offers; an override is ``{knob id: value}``. :func:`apply_overrides` rebuilds a preset with the overrides in a
fixed order, re-solving the crystal after a trap change (``Device`` does not re-derive on ``replace``) and re-deriving the
preparation recipe after a trap, field or beam change (its Doppler detuning is optimized over the mode structure, Section
4.2.1). The knobs act on a trap declared by its secular frequencies, where the rf amplitude scales the Mathieu q of both
radial axes at fixed a and beta follows from the Mathieu equation (Section 4.1.1), so the radial frequencies move as
nu = beta Omega_rf/2 rather than in proportion to V_rf; that needs the rf drive frequency declared (``trap.rf_frequency_hz``).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from qutip_trap_app import core

Page = Literal["species", "trap", "crystal", "light", "noise", "cooling", "readout", "hamiltonian"]

DEFAULT_RF_VOLTAGE_V = 100.0
"""The peak rf voltage a declared rf record gets when only the frequency was set (Section 4.1.1: on the explicit path the
voltage does not enter a, q or beta; it is recorded so that the rf record is complete)."""


class KnobError(ValueError):
    """An override the device cannot take: an unknown knob, a value outside its domain, or a path the trap does not have."""


@dataclass(frozen=True)
class Knob:
    id: str
    page: Page
    label: str
    """Plain language first (the ui-ux rule)."""
    term: str
    unit: str
    ledger_id: str
    section: str
    lo: float
    hi: float
    log: bool
    doc: str
    """What changes downstream when this knob moves (the learner reads it beside the slider)."""
    relative: bool = False
    """True for a scale factor that is not itself a device field (read back from the overrides, not the device)."""
    requires_rf: bool = False
    """True when the knob needs the trap's rf record (declared by ``trap.rf_frequency_hz`` on the public presets)."""


def _k(
    id: str,
    page: Page,
    label: str,
    term: str,
    unit: str,
    ledger_id: str,
    section: str,
    lo: float,
    hi: float,
    doc: str,
    *,
    log: bool = False,
    relative: bool = False,
    requires_rf: bool = False,
) -> Knob:
    return Knob(id, page, label, term, unit, ledger_id, section, lo, hi, log, doc, relative, requires_rf)


# fmt: off
_FIXED: tuple[Knob, ...] = (
    _k("trap.rf_amplitude_scale", "trap", "Radio-frequency drive strength, relative to the preset",
       "V_rf/V_rf,0 (q proportional to V_rf at fixed a; beta by the Mathieu equation)", "", "conv.rf_amplitude_pseudopotential", "4.1.1", 0.5, 1.5,
       "scales q of both radial axes at fixed a; beta and the radial secular frequencies follow from the Mathieu equation, the radial modes move, eta follows 1/sqrt(omega) times the micromotion factor C0 of the Mathieu record, the MS pulse must be re-solved and the table goes stale; needs the rf drive frequency", relative=True, requires_rf=True),
    _k("trap.omega_z_hz", "trap", "Axial trap frequency",
       "omega_z/2pi (the dc end-cap stiffness)", "Hz", "conv.rf_amplitude_pseudopotential", "4.1.2", 2e5, 3e6,
       "moves the ions closer or farther apart (spacing goes as omega_z^-2/3) and rescales every axial mode", log=True),
    _k("trap.rf_frequency_hz", "trap", "Radio-frequency drive frequency",
       "Omega_rf/2pi", "Hz", "conv.mathieu_sign", "4.1.1", 1e7, 1e8,
       "declares the rf record: a, q, beta and the micromotion factor C0 follow from the secular frequencies at this drive frequency", log=True),
    _k("trap.rf_voltage_peak_v", "trap", "Radio-frequency peak voltage",
       "V_rf (peak of a cos drive)", "V", "conv.rf_amplitude_pseudopotential", "4.1.1", 10.0, 1000.0,
       "recorded with the rf drive; on a trap declared by its secular frequencies it does not enter a, q or beta", requires_rf=True),
    _k("trap.stray_field_x_v_per_m", "trap", "Stray electric field along x",
       "E_dc,x", "V/m", "anchor.trap.berkeland_excess_micromotion", "4.1.1", -200.0, 200.0,
       "displaces the ion from the rf null and produces excess micromotion of amplitude (1/2) u_0 q along x"),
    _k("trap.stray_field_y_v_per_m", "trap", "Stray electric field along y",
       "E_dc,y", "V/m", "anchor.trap.berkeland_excess_micromotion", "4.1.1", -200.0, 200.0,
       "displaces the ion from the rf null and produces excess micromotion along y"),
    _k("trap.axis_angle_rad", "trap", "Rotation of the radial trap axes",
       "axis angle about z", "rad", "conv.mode_index", "4.1.3", -1.5708, 1.5708,
       "rotates the two radial mode families; a beam's projection on each family (and so every eta) follows"),
    _k("field.b_gauss", "species", "Magnetic field at the ions",
       "|B|", "G", "conv.curvature_naming", "4.5.1", 0.5, 50.0,
       "moves every Zeeman sublevel; the clock qubit shifts only quadratically (310.9 Hz/G^2 for 171Yb+); the detection dark states change"),
    _k("detector.window_s", "readout", "Detection window",
       "t_det", "s", "conv.readout_detected_line", "8.3", 2e-6, 2e-4,
       "more photons separate bright from dark, but the off-resonant pumping has longer to flip the ion: the error has an interior optimum", log=True),
    _k("detector.efficiency", "readout", "Fraction of scattered photons the detector counts",
       "eps_sys", "", "conv.detection_efficiency_once", "8.1", 0.002, 0.3,
       "scales the detected bright rate once (never the leakage rates), so the count histograms separate faster", log=True),
    _k("detector.background_cps", "readout", "Background counts per second",
       "R_bg", "1/s", "conv.detection_efficiency_once", "8.1", 0.0, 2000.0,
       "adds Poisson counts to a dark ion, raising the dark-read-as-bright error"),
    _k("noise.s_e_white", "noise", "Electric-field noise density at the ions",
       "S_E (two-sided, white)", "(V/m)^2/(rad/s)", "conv.electric_field_noise", "4.1.5", 1e-16, 1e-8,
       "heats every mode at n_dot = e^2 S_E/(4 m hbar omega): the heating collapse operators appear with these rates", log=True),
    _k("noise.correlation_length_m", "noise", "Distance over which the field noise is the same at two ions",
       "correlation length l_c", "m", "conv.electric_field_noise_adapter", "4.1.5", 0.0, 1e-3,
       "uniform noise heats only the centre-of-mass modes; uncorrelated noise heats every mode"),
    _k("noise.s_b_white", "noise", "Magnetic-field noise density",
       "S_B (two-sided, white)", "T^2/(rad/s)", "conv.qubit_dephasing_from_field_noise", "6.3", 1e-26, 1e-16,
       "dephases each qubit at gamma_phi = 2 pi^2 (dnu/dB)^2 S_B: nothing for a clock qubit at low field, everything for a Zeeman qubit", log=True),
    _k("noise.rabi_drift_rms", "noise", "Shot-to-shot spread of the laser power",
       "rms of dOmega/Omega (quasi-static)", "", "conv.noise_routing", "6.4", 0.0, 0.05,
       "every shot draws its own Rabi-frequency scale: pulse areas over- or under-rotate coherently"),
    _k("preparation.doppler_duration_s", "cooling", "How long Doppler cooling runs",
       "t_Doppler", "s", "conv.preparation_stage_order", "4.2.1", 1e-4, 1e-2,
       "the Doppler stage reaches its rate-equation steady state in a few 1/W_m; longer only costs time", log=True),
    _k("preparation.pump_duration_s", "cooling", "How long the optical pump runs",
       "t_pump", "s", "conv.preparation_stage_order", "4.2.6", 2e-6, 2e-4,
       "too short leaves population outside |0>: the preparation error and the pumping recoil follow from the master equation", log=True),
)
_BEAM_TEMPLATES: tuple[tuple[str, str, str, str, str, str, float, float, bool, str], ...] = (
    ("power_w", "Laser power in beam {k}", "P_{k}", "W",
     "conv.rabi_from_intensity", "4.5.4", 1e-6, 2.0, True,
     "the Rabi frequency goes as the field, sqrt(P_1 P_2) for a Raman pair, the light shift and the scattering rate as the intensity"),
    ("waist_m", "Beam {k} focus size", "w_0 of beam {k} (1/e^2 intensity radius)", "m",
     "conv.rabi_from_intensity", "4.5.4", 1e-6, 2e-4, True,
     "a tighter focus raises the intensity as 1/w_0^2 and, for an addressing beam, lowers the light that spills onto the neighbour"),
    ("wavelength_m", "Beam {k} wavelength", "lambda of beam {k} (vacuum)", "m",
     "conv.wavelengths_vacuum", "4.5.2", 3.0e-7, 1.2e-6, False,
     "sets the detuning from every atomic line: the Raman coupling falls as 1/Delta^2 and so does the scattering error, Section 4.3.2"),
)
# fmt: on

_BEAM_ID = re.compile(r"^beam\[(\d+)\]\.(power_w|waist_m|wavelength_m)$")


def beam_knob(k: int, field: str) -> Knob:
    for name, label, term, unit, ledger, section, lo, hi, log, doc in _BEAM_TEMPLATES:
        if name == field:
            return _k(
                f"beam[{k}].{name}",
                "light",
                label.format(k=k),
                term.format(k=k),
                unit,
                ledger,
                section,
                lo,
                hi,
                doc,
                log=log,
            )
    raise KnobError(f"no beam knob {field!r}")


def knobs_for(device: core.Device) -> dict[str, Knob]:
    """Every knob this device offers: the fixed ones plus one set per beam."""
    out = {k.id: k for k in _FIXED}
    for k in range(len(device.beams)):
        for name, *_rest in _BEAM_TEMPLATES:
            kb = beam_knob(k, name)
            out[kb.id] = kb
    return out


def knob(id: str) -> Knob:
    for k in _FIXED:
        if k.id == id:
            return k
    m = _BEAM_ID.match(id)
    if m:
        return beam_knob(int(m.group(1)), m.group(2))
    raise KnobError(f"unknown knob {id!r}")


def validate(overrides: Mapping[str, float], device: core.Device) -> dict[str, float]:
    """The overrides as floats, each a known knob of the device with a finite value inside the knob's declared range;
    anything else is a :class:`KnobError`."""
    known = knobs_for(device)
    out: dict[str, float] = {}
    for key in sorted(overrides):
        if key not in known:
            raise KnobError(f"unknown knob {key!r} for this device; known: {sorted(known)}")
        try:
            v = float(overrides[key])
        except (TypeError, ValueError) as exc:
            raise KnobError(f"{key}: the value {overrides[key]!r} is not a number") from exc
        if not math.isfinite(v):
            raise KnobError(f"{key}: the value must be finite")
        k = known[key]
        if not k.lo <= v <= k.hi:
            raise KnobError(
                f"{key}: {v:g} is outside the knob's range [{k.lo:g}, {k.hi:g}] {k.unit}".rstrip()
            )
        out[key] = v
    return out


def current_values(
    preset: core.DevicePreset, overrides: Mapping[str, float] | None = None
) -> dict[str, float]:
    """What every knob reads on this device (a relative knob reads its override, 1.0 when unset)."""
    dev = preset.device
    ov = dict(overrides or {})
    trap = dev.trap
    vals: dict[str, float] = {
        "trap.rf_amplitude_scale": float(ov.get("trap.rf_amplitude_scale", 1.0)),
        "trap.stray_field_x_v_per_m": float(trap.stray_field_v_per_m[0]),
        "trap.stray_field_y_v_per_m": float(trap.stray_field_v_per_m[1]),
        "trap.axis_angle_rad": float(trap.axis_angle_rad),
        "field.b_gauss": float(dev.field.B_gauss),
        "detector.window_s": float(dev.detector.window_s),
        "detector.efficiency": float(dev.detector.efficiency),
        "detector.background_cps": float(dev.detector.background_cps),
        "noise.s_e_white": float(dev.noise.S_E.white_level),
        "noise.correlation_length_m": float(dev.noise.correlation_length_m or 0.0),
        "noise.s_b_white": float(dev.noise.S_B.white_level) if dev.noise.S_B is not None else 0.0,
        "noise.rabi_drift_rms": float(dev.noise.rabi_drift.rms),
    }
    if trap.omega_hz is not None:
        vals["trap.omega_z_hz"] = float(trap.omega_hz[2])
    if trap.rf is not None:
        vals["trap.rf_frequency_hz"] = float(trap.rf.frequency_hz)
        vals["trap.rf_voltage_peak_v"] = float(trap.rf.voltage_peak_v)
    if dev.preparation is not None:
        vals["preparation.doppler_duration_s"] = float(dev.preparation.doppler_duration_s)
        vals["preparation.pump_duration_s"] = float(dev.preparation.pump_duration_s)
    for k, b in enumerate(dev.beams):
        vals[f"beam[{k}].power_w"] = float(b.power_w)
        vals[f"beam[{k}].waist_m"] = float(b.waist_m)
        vals[f"beam[{k}].wavelength_m"] = float(b.wavelength_m)
    return vals


def _scaled_radial_frequencies(
    trap: core.Trap, species: core.Species, scale: float
) -> tuple[float, float, float]:
    """The secular frequencies after the rf amplitude is scaled: q -> scale q on both radial axes at fixed a, beta by the
    core's monodromy (exact), nu = beta Omega_rf/2; the axial axis (q_z = 0) does not move (Section 4.1.1)."""
    assert trap.rf is not None and trap.omega_hz is not None
    mp = trap.mathieu(species)
    omega_rf = float(trap.rf.omega_rad_s)
    out: list[float] = []
    for k in range(3):
        a_k = float(mp.diagonal_a[k])
        q_k = float(mp.q_effective[k]) * scale
        if q_k == 0.0:
            out.append(float(trap.omega_hz[k]))
            continue
        mono = core.monodromy(a_k, q_k)
        if not mono.stable:
            raise KnobError(
                f"an rf amplitude scale of {scale:g} takes axis {k} to a = {a_k:.4g}, q = {q_k:.4g}, outside the stability region"
            )
        out.append(float(mono.beta) * omega_rf / 2.0 / (2.0 * math.pi))
    return (out[0], out[1], out[2])


def entangling_pair(preset: core.DevicePreset) -> tuple[int, int] | None:
    """The Raman pair of the preset's entangling drives (the pair the recipe's sideband cooling and the crystal page's
    Lamb-Dicke parameters refer to), or None when the device has none."""
    for drive in preset.entangling_drives.values():
        if len(drive.beams) == 2:
            return (int(drive.beams[0]), int(drive.beams[1]))
    return None


def apply_overrides(
    preset: core.DevicePreset, overrides: Mapping[str, float], *, rederive_recipe: bool = True
) -> core.DevicePreset:
    """The preset with the overrides applied (Section 14.4): a fixed order, the crystal re-solved after a trap change, the
    preparation recipe re-derived after a change of trap, field or beams unless ``rederive_recipe`` is False."""
    dev = preset.device
    ov = validate(overrides, dev)
    if not ov:
        return preset
    trap = dev.trap
    if trap.path != "explicit" or trap.omega_hz is None:
        raise KnobError("the knobs act on a trap declared by its secular frequencies")
    omega = [float(x) for x in trap.omega_hz]
    rf = trap.rf
    stray = [float(x) for x in trap.stray_field_v_per_m]
    axis = float(trap.axis_angle_rad)
    trap_keys = {k for k in ov if k.startswith("trap.")}
    if "trap.omega_z_hz" in ov:
        omega[2] = ov["trap.omega_z_hz"]
    if "trap.rf_frequency_hz" in ov or "trap.rf_voltage_peak_v" in ov:
        freq = ov.get("trap.rf_frequency_hz", None if rf is None else rf.frequency_hz)
        if freq is None:
            raise KnobError(
                "trap.rf_voltage_peak_v needs trap.rf_frequency_hz on a trap without an rf record"
            )
        volt = ov.get("trap.rf_voltage_peak_v", DEFAULT_RF_VOLTAGE_V if rf is None else rf.voltage_peak_v)
        rf = core.RfDrive(voltage_peak_v=float(volt), frequency_hz=float(freq))
    if "trap.stray_field_x_v_per_m" in ov:
        stray[0] = ov["trap.stray_field_x_v_per_m"]
    if "trap.stray_field_y_v_per_m" in ov:
        stray[1] = ov["trap.stray_field_y_v_per_m"]
    if "trap.axis_angle_rad" in ov:
        axis = ov["trap.axis_angle_rad"]
    if "trap.rf_amplitude_scale" in ov:
        if rf is None:
            raise KnobError(
                "trap.rf_amplitude_scale needs the rf drive frequency (trap.rf_frequency_hz): on a trap declared by its "
                "secular frequencies the Mathieu a is unknown without it, and omega_r proportional to V_rf holds only for a = 0"
            )
        probe = dataclasses.replace(
            trap,
            omega_hz=(omega[0], omega[1], omega[2]),
            rf=rf,
            stray_field_v_per_m=(stray[0], stray[1], stray[2]),
            axis_angle_rad=axis,
        )
        omega = list(_scaled_radial_frequencies(probe, dev.crystal.species[0], ov["trap.rf_amplitude_scale"]))
    if trap_keys:
        trap = dataclasses.replace(
            trap,
            omega_hz=(omega[0], omega[1], omega[2]),
            rf=rf,
            stray_field_v_per_m=(stray[0], stray[1], stray[2]),
            axis_angle_rad=axis,
        )
    field = dev.field
    if "field.b_gauss" in ov:
        field = dataclasses.replace(field, B_gauss=ov["field.b_gauss"])
    beams = list(dev.beams)
    beam_keys = [(m, v) for key, v in ov.items() if (m := _BEAM_ID.match(key))]
    for m, value in beam_keys:
        k, name = int(m.group(1)), m.group(2)
        if name == "power_w":
            beams[k] = dataclasses.replace(beams[k], power_w=value)
        elif name == "waist_m":
            beams[k] = dataclasses.replace(beams[k], waist_m=value)
        else:
            beams[k] = dataclasses.replace(beams[k], wavelength_m=value)
    detector = dev.detector
    if "detector.window_s" in ov:
        detector = dataclasses.replace(detector, window_s=ov["detector.window_s"])
    if "detector.efficiency" in ov:
        detector = dataclasses.replace(detector, efficiency=ov["detector.efficiency"])
    if "detector.background_cps" in ov:
        detector = dataclasses.replace(detector, background_cps=ov["detector.background_cps"])
    noise = dev.noise
    noise_changes: dict[str, object] = {}
    if "noise.s_e_white" in ov:
        noise_changes["S_E"] = core.white_spectrum(ov["noise.s_e_white"], noise.S_E.unit)
        if noise.correlation_length_m is None and "noise.correlation_length_m" not in ov:
            noise_changes["correlation_length_m"] = 0.0
    if "noise.correlation_length_m" in ov:
        noise_changes["correlation_length_m"] = ov["noise.correlation_length_m"]
    if "noise.s_b_white" in ov:
        unit = noise.S_B.unit if noise.S_B is not None else "T^2/(rad/s)"
        noise_changes["S_B"] = core.white_spectrum(ov["noise.s_b_white"], unit)
    if "noise.rabi_drift_rms" in ov:
        noise_changes["rabi_drift"] = dataclasses.replace(noise.rabi_drift, rms=ov["noise.rabi_drift_rms"])
    if noise_changes:
        noise = dataclasses.replace(noise, **noise_changes)  # type: ignore[arg-type]
    crystal = core.solve_crystal(trap, dev.crystal.species) if trap_keys else dev.crystal
    new_dev = dataclasses.replace(
        dev, trap=trap, crystal=crystal, field=field, beams=tuple(beams), detector=detector, noise=noise
    )
    recipe = dev.preparation
    if recipe is not None and rederive_recipe and (trap_keys or "field.b_gauss" in ov or beam_keys):
        sideband = recipe.sideband
        recipe = core.standard_recipe(
            dataclasses.replace(new_dev, preparation=None),
            raman_pair=entangling_pair(preset),
            doppler_duration_s=recipe.doppler_duration_s,
            pump_duration_s=recipe.pump_duration_s,
            sideband_pulses=None if sideband is None else dict(sideband.pulses_per_order),
        )
    if recipe is not None:
        if "preparation.doppler_duration_s" in ov:
            recipe = dataclasses.replace(recipe, doppler_duration_s=ov["preparation.doppler_duration_s"])
        if "preparation.pump_duration_s" in ov:
            recipe = dataclasses.replace(recipe, pump_duration_s=ov["preparation.pump_duration_s"])
    new_dev = dataclasses.replace(new_dev, preparation=recipe)
    note = "overrides: " + ", ".join(f"{k} = {v:.6g}" for k, v in ov.items())
    return dataclasses.replace(preset, device=new_dev, notes=tuple(preset.notes) + (note,))
