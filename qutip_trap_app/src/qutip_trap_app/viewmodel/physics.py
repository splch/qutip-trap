"""Level 4, the physics pages: view-models over the device layer (PLAN.md Section 14.2 row 4, Section 14.4; M11.3).

Every number a page shows is a :class:`Shown` with its catalogue id, so the provenance coverage test of Section 9.11 covers
these pages as it covers the levels above. The views are thin: the physics was computed in the worker into
:class:`~qutip_trap_app.device_layer.DeviceLayer`; here it is arranged for reading, plain label first, with the status a
learner must discriminate (device parameter, derived, estimate, calibrated, stale).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qutip_trap_app.device_layer import DeviceLayer
from qutip_trap_app.knobs import Knob, knob
from qutip_trap_app.record import HamiltonianRecord, Record, TableRecord
from qutip_trap_app.viewmodel.catalogue import Shown

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Row:
    label: str
    value: Shown
    status: str = ""
    """``device parameter`` (typed into the model), ``derived`` (closed form or small eigenproblem), ``estimate`` (closed-form
    error scale), ``calibrated`` (from the table), ``stale`` (calibrated for another device)."""


@dataclass(frozen=True)
class KnobRow:
    knob: Knob
    value: float
    is_override: bool


def knob_rows(layer: DeviceLayer, page: str) -> tuple[KnobRow, ...]:
    out: list[KnobRow] = []
    for kid, value in sorted(layer.knob_values.items()):
        k = knob(kid)
        if k.page != page:
            continue
        out.append(KnobRow(k, float(value), kid in layer.overrides))
    return tuple(out)


def stale_status(layer: DeviceLayer) -> str:
    if layer.table_hash is None:
        return "no calibration table yet for this device"
    if layer.stale:
        return "stale: the calibration table was made for another device (recalibrate to refresh it)"
    return "current: the calibration table belongs to this device"


# ---- species -------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeciesView:
    name: str
    identity: tuple[Row, ...]
    levels: tuple[tuple[str, tuple[Shown, ...]], ...]
    """Per level: (name, (energy, lifetime, linewidth, A, B, g_J))."""
    transitions: tuple[tuple[str, tuple[Shown, ...]], ...]
    """Per transition: (label, (wavelength, linewidth, branching, I_sat))."""
    qubit: tuple[Row, ...]
    clock_points: tuple[tuple[Shown, Shown, Shown], ...]
    clock_scan: tuple[float, float]
    sublevels: tuple[tuple[str, str, Shown, Shown, Shown], ...]
    """(level, label, energy, slope, curvature) at the operating field."""
    sweep_b_gauss: np.ndarray
    sweep_energies_hz: np.ndarray
    sweep_labels: tuple[str, ...]
    sweep_level: str
    notes: tuple[str, ...]


def species_view(layer: DeviceLayer) -> SpeciesView:
    sp = layer.species
    identity = (
        Row("species", Shown("species", sp.name), "device parameter"),
        Row("mass", Shown("mass", sp.mass_u), "table constant"),
        Row("nuclear spin", Shown("nuclear_spin", sp.nuclear_spin), "table constant"),
        Row("nuclear moment", Shown("nuclear_moment", sp.mu_i_nuclear_magnetons), "table constant"),
        Row("magnetic field", Shown("field", sp.field_gauss), "device parameter"),
    )
    levels = tuple(
        (
            lv.name,
            (
                Shown("level_energy", lv.energy_hz, lv.name),
                Shown("lifetime", lv.lifetime_s, lv.name),
                Shown("linewidth", lv.gamma_total_hz, f"Gamma/2pi of {lv.name}"),
                Shown("hyperfine_a", lv.A_hfs_hz, lv.name),
                Shown("hyperfine_b", lv.B_hfs_hz, lv.name),
                Shown("g_j", lv.g_J, lv.name),
            ),
        )
        for lv in sp.levels
    )
    transitions = tuple(
        (
            tr.label,
            (
                Shown("wavelength", tr.wavelength_m, f"{tr.label} ({tr.multipole})"),
                Shown("linewidth", tr.gamma_hz, f"total decay of {tr.upper}"),
                Shown("branching", tr.branching, f"{tr.upper} -> {tr.lower}"),
                Shown("saturation_intensity", tr.i_sat_w_m2, tr.label),
            ),
        )
        for tr in sp.transitions
    )
    qubit = (
        Row("lower state", Shown("species", sp.qubit[0]), "designated"),
        Row("upper state", Shown("species", sp.qubit[1]), "designated"),
        Row("qubit frequency at the field", Shown("qubit_frequency", sp.qubit_freq_hz), "derived"),
        Row(
            "first-order field sensitivity",
            Shown("field_slope", sp.dnu_db_hz_per_g, "dnu/dB of the qubit pair"),
            "derived",
        ),
        Row(
            "second-order field sensitivity",
            Shown("field_curvature", sp.d2nu_db2_hz_per_g2, "d2nu/dB2 (Taylor c2 is half of it)"),
            "derived",
        ),
        Row("cycling transition", Shown("species", sp.cycling), "designated"),
        Row("repumps", Shown("species", ", ".join(sp.repumps) or "none"), "designated"),
    )
    clocks = tuple(
        (
            Shown("clock_field", b0),
            Shown("qubit_frequency", f0, "at the clock point"),
            Shown("field_curvature", c2, "Taylor coefficient c2 at the clock point"),
        )
        for b0, f0, c2 in sp.clock_points
    )
    subs = tuple(
        (
            s.level,
            s.label,
            Shown("sublevel_energy", s.energy_hz, f"{s.level} {s.label}"),
            Shown("field_slope", s.dE_dB_hz_per_g, s.label),
            Shown("field_curvature", s.d2E_dB2_hz_per_g2, s.label),
        )
        for s in sp.sublevels
    )
    return SpeciesView(
        name=sp.name,
        identity=identity,
        levels=levels,
        transitions=transitions,
        qubit=qubit,
        clock_points=clocks,
        clock_scan=sp.clock_scan_gauss,
        sublevels=subs,
        sweep_b_gauss=sp.sweep.b_gauss,
        sweep_energies_hz=sp.sweep.energies_hz,
        sweep_labels=sp.sweep.labels,
        sweep_level=sp.sweep.level,
        notes=sp.notes,
    )


# ---- trap -------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrapView:
    path: str
    secular: tuple[Row, ...]
    rf: tuple[Row, ...]
    mathieu: tuple[tuple[str, Shown, Shown, Shown, Shown, Shown], ...]
    """Per axis: (axis, a, q, beta, nu from beta, C0)."""
    mathieu_note: str
    fields: tuple[Row, ...]
    stability_q: np.ndarray | None
    stability_lower: np.ndarray | None
    stability_upper: np.ndarray | None
    stability_edge: Shown | None
    operating_points: tuple[tuple[str, float, float], ...]
    notes: tuple[str, ...]


def trap_view(layer: DeviceLayer) -> TrapView:
    tr = layer.trap
    secular: list[Row] = []
    if tr.omega_hz is not None:
        for ax, w in zip(("x", "y", "z"), tr.omega_hz):
            secular.append(
                Row(
                    f"secular frequency {ax}", Shown("secular_frequency", w, f"axis {ax}"), "device parameter"
                )
            )
    secular.append(Row("radial axis rotation", Shown("axis_angle", tr.axis_angle_rad), "device parameter"))
    rf: list[Row] = []
    if tr.rf_frequency_hz is not None:
        rf.append(Row("rf drive frequency", Shown("rf_frequency", tr.rf_frequency_hz), "device parameter"))
        rf.append(Row("rf peak voltage", Shown("rf_voltage", tr.rf_voltage_peak_v), "device parameter"))
    mathieu: list[tuple[str, Shown, Shown, Shown, Shown, Shown]] = []
    if (
        tr.a is not None
        and tr.q is not None
        and tr.beta is not None
        and tr.secular_from_mathieu_hz is not None
        and tr.c0 is not None
    ):
        for k, ax in enumerate(("x", "y", "z")):
            mathieu.append(
                (
                    ax,
                    Shown("mathieu_a", tr.a[k], f"a_{ax}"),
                    Shown("mathieu_q", tr.q[k], f"q_{ax} (effective)"),
                    Shown("mathieu_beta", tr.beta[k], f"beta_{ax}"),
                    Shown(
                        "secular_frequency",
                        tr.secular_from_mathieu_hz[k],
                        f"beta_{ax} Omega_rf/2, the monodromy value",
                    ),
                    Shown("micromotion_c0", tr.c0[k], f"C0 along {ax}"),
                )
            )
    fields: list[Row] = []
    for k, ax in enumerate(("x", "y", "z")):
        fields.append(
            Row(
                f"stray field {ax}",
                Shown("stray_field", tr.stray_field_v_per_m[k], f"E_dc,{ax}"),
                "device parameter",
            )
        )
    if tr.displacement_m is not None:
        for k, ax in enumerate(("x", "y", "z")):
            fields.append(
                Row(
                    f"ion displacement {ax}",
                    Shown("ion_displacement", tr.displacement_m[k], f"u_0 along {ax}"),
                    "derived",
                )
            )
    if tr.micromotion_amplitude_m is not None:
        for k, ax in enumerate(("x", "y", "z")):
            fields.append(
                Row(
                    f"excess micromotion {ax}",
                    Shown("micromotion_amplitude", tr.micromotion_amplitude_m[k], f"u_1 along {ax}"),
                    "derived",
                )
            )
    if tr.residual_field_v_per_m is not None:
        for k, ax in enumerate(("x", "y", "z")):
            fields.append(
                Row(
                    f"residual field {ax}",
                    Shown("stray_field", tr.residual_field_v_per_m[k], "stray plus shims at the null"),
                    "derived",
                )
            )
    if tr.ion_height_m is not None:
        fields.append(Row("ion height", Shown("ion_height", tr.ion_height_m), "derived"))
    if tr.trap_depth_ev is not None:
        fields.append(Row("trap depth", Shown("trap_depth", tr.trap_depth_ev), "derived"))
    points: list[tuple[str, float, float]] = []
    if tr.a is not None and tr.q is not None:
        for k, ax in enumerate(("x", "y", "z")):
            points.append((ax, abs(tr.q[k]), tr.a[k]))
    st = tr.stability
    return TrapView(
        path=tr.path,
        secular=tuple(secular),
        rf=tuple(rf),
        mathieu=tuple(mathieu),
        mathieu_note=tr.mathieu_note,
        fields=tuple(fields),
        stability_q=None if st is None else st.q,
        stability_lower=None if st is None else st.a_lower,
        stability_upper=None if st is None else st.a_upper,
        stability_edge=None
        if st is None
        else Shown("stability_edge", st.edge_q_at_a0, f"{st.monodromy_calls} monodromy integrations"),
        operating_points=tuple(points),
        notes=tr.notes,
    )


# ---- crystal ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeRow:
    index: int
    family: str
    family_index: int
    frequency: Shown
    e_hat: tuple[float, float, float]
    eigenvector: np.ndarray
    components: tuple[Shown, ...]
    uniform_weight: Shown
    heating: Shown | None


@dataclass(frozen=True)
class CrystalView:
    n_ions: int
    species: tuple[str, ...]
    positions_m: np.ndarray
    positions: tuple[Shown, ...]
    modes: tuple[ModeRow, ...]
    eta: np.ndarray | None
    eta_rows: tuple[tuple[str, tuple[Shown, ...]], ...]
    """Per ion: (label, eta per mode) for the entangling pair's Delta k."""
    delta_k: Shown | None
    c0_applied: bool
    geometry: tuple[Row, ...]
    zigzag: tuple[Row, ...]
    notes: tuple[str, ...]


def crystal_view(layer: DeviceLayer) -> CrystalView:
    cr = layer.crystal
    modes = tuple(
        ModeRow(
            index=m.index,
            family=m.family,
            family_index=m.family_index,
            frequency=Shown("mode_frequency", m.omega_hz, f"mode {m.index}: {m.family} {m.family_index}"),
            e_hat=m.e_hat,
            eigenvector=m.eigenvector,
            components=tuple(
                Shown("eigenvector_component", float(c), f"ion {i}, mode {m.index}")
                for i, c in enumerate(m.eigenvector)
            ),
            uniform_weight=Shown("uniform_field_weight", m.uniform_field_weight, f"mode {m.index}"),
            heating=None
            if m.heating_quanta_per_s is None
            else Shown("heating_rate", m.heating_quanta_per_s, f"mode {m.index}"),
        )
        for m in cr.modes
    )
    eta_rows: list[tuple[str, tuple[Shown, ...]]] = []
    if cr.eta is not None:
        for i in range(cr.n_ions):
            eta_rows.append(
                (
                    f"ion {i}",
                    tuple(
                        Shown("lamb_dicke", float(cr.eta[i, m]), f"eta_{i},{m}")
                        for m in range(cr.eta.shape[1])
                    ),
                )
            )
    geometry: list[Row] = []
    if cr.length_scale_m is not None:
        geometry.append(Row("Coulomb length scale", Shown("length_scale", cr.length_scale_m), "derived"))
    if cr.spacing_m is not None:
        geometry.append(Row("nearest-neighbour spacing", Shown("ion_spacing", cr.spacing_m), "derived"))
    zig: list[Row] = []
    if cr.zigzag_ratio is not None:
        zig.append(Row("radial over axial stiffness", Shown("zigzag_ratio", cr.zigzag_ratio), "derived"))
    if cr.zigzag_critical is not None:
        zig.append(Row("buckling threshold", Shown("zigzag_critical", cr.zigzag_critical), "derived"))
    dk = None
    if cr.delta_k_rad_per_m is not None:
        dk = Shown("delta_k", float(np.linalg.norm(cr.delta_k_rad_per_m)), f"beams {cr.entangling_beams}")
    return CrystalView(
        n_ions=cr.n_ions,
        species=cr.species,
        positions_m=cr.positions_m,
        positions=tuple(
            Shown("ion_position", float(cr.positions_m[i, 2]), f"ion {i}, along the axis")
            for i in range(cr.n_ions)
        ),
        modes=modes,
        eta=cr.eta,
        eta_rows=tuple(eta_rows),
        delta_k=dk,
        c0_applied=cr.c0_applied,
        geometry=tuple(geometry),
        zigzag=tuple(zig),
        notes=cr.notes,
    )


# ---- light -------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BeamRow:
    index: int
    roles: tuple[str, ...]
    wavelength: Shown
    power: Shown
    waist: Shown
    direction: Shown
    polarization: Shown
    angle_to_field: Shown
    intensity_at_ions: tuple[Shown, ...]
    saturation_at_ions: tuple[Shown, ...]
    k_hat: tuple[float, float, float]
    pointing_m: tuple[float, float, float]


@dataclass(frozen=True)
class DriveRow:
    ion: int
    role: str
    kind: str
    beams: tuple[int, ...]
    rabi: Shown
    pi_time: Shown
    stark: Shown
    delta_k: Shown
    etas: tuple[Shown, ...]
    residual_excited: Shown
    raman_rate: Shown
    rayleigh_rate: Shown
    leakage_rate: Shown
    dephasing: Shown
    error_per_pi: Shown
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class LightView:
    beams: tuple[BeamRow, ...]
    drives: tuple[DriveRow, ...]
    crosstalk: tuple[Shown, ...]
    curve_wavelength_m: np.ndarray | None
    curve_detuning_hz: np.ndarray | None
    curve_error: np.ndarray | None
    curve_rabi_hz: np.ndarray | None
    curve_p12_m: float | None
    curve_p32_m: float | None
    current_wavelength_m: float | None
    notes: tuple[str, ...]


def _vec(v: tuple[float, float, float]) -> str:
    return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"


def light_view(layer: DeviceLayer) -> LightView:
    li = layer.light
    beams = tuple(
        BeamRow(
            index=b.index,
            roles=b.roles,
            wavelength=Shown("wavelength", b.wavelength_m, f"beam {b.index}"),
            power=Shown("beam_power", b.power_w, f"beam {b.index}"),
            waist=Shown("beam_waist", b.waist_m, f"beam {b.index}"),
            direction=Shown("beam_direction", _vec(b.k_hat), f"beam {b.index}"),
            polarization=Shown(
                "polarization",
                "(" + ", ".join(f"{z.real:+.2f}{z.imag:+.2f}i" for z in b.polarization) + ")",
                "laboratory-frame Jones vector",
            ),
            angle_to_field=Shown("beam_angle_to_field", b.angle_to_field_deg, f"beam {b.index}"),
            intensity_at_ions=tuple(
                Shown("intensity", float(x), f"beam {b.index} at ion {i}")
                for i, x in enumerate(b.intensity_at_ions_w_m2)
            ),
            saturation_at_ions=()
            if b.saturation_at_ions is None
            else tuple(
                Shown("saturation_parameter", float(x), f"beam {b.index} at ion {i}")
                for i, x in enumerate(b.saturation_at_ions)
            ),
            k_hat=b.k_hat,
            pointing_m=b.pointing_m,
        )
        for b in li.beams
    )
    drives = tuple(
        DriveRow(
            ion=d.ion,
            role=d.role,
            kind=d.kind,
            beams=d.beams,
            rabi=Shown("rabi_frequency", d.rabi_hz, f"{d.role}, ion {d.ion}"),
            pi_time=Shown("pi_time", d.pi_time_s),
            stark=Shown("stark_shift", d.stark_hz, "differential light shift of the qubit"),
            delta_k=Shown("delta_k", float(np.linalg.norm(d.delta_k_rad_per_m))),
            etas=tuple(
                Shown("lamb_dicke", e, f"eta_{d.ion},{m}") for m, e in sorted(d.etas.items()) if e != 0.0
            ),
            residual_excited=Shown("residual_excited", d.residual_excited_population),
            raman_rate=Shown(
                "scattering_rate",
                sum(d.raman_per_s.values()),
                "Raman spin flips within the qubit pair, summed over the two states",
            ),
            rayleigh_rate=Shown(
                "scattering_rate",
                sum(d.rayleigh_per_s.values()),
                "Rayleigh (elastic), summed over the two states",
            ),
            leakage_rate=Shown(
                "scattering_rate", sum(d.leakage_per_s.values()), "Raman leakage out of the qubit pair"
            ),
            dephasing=Shown("rayleigh_dephasing", d.rayleigh_dephasing_per_s),
            error_per_pi=Shown(
                "scattering_error", d.error_per_pi_pulse, "per pi pulse at this Rabi frequency"
            ),
            provenance=d.provenance,
        )
        for d in li.drives
    )
    xt = tuple(
        Shown("crosstalk", v, f"ion {k.split(',')[0]}'s light on ion {k.split(',')[1]}")
        for k, v in sorted(li.crosstalk.items())
    )
    sc = li.scattering_curve
    current = None
    if sc is not None:
        current = float(li.beams[sc.beams[0]].wavelength_m)
    return LightView(
        beams=beams,
        drives=drives,
        crosstalk=xt,
        curve_wavelength_m=None if sc is None else sc.wavelength_m,
        curve_detuning_hz=None if sc is None else sc.detuning_from_p12_hz,
        curve_error=None if sc is None else sc.error_per_pi_pulse,
        curve_rabi_hz=None if sc is None else sc.rabi_hz,
        curve_p12_m=None if sc is None else sc.p12_wavelength_m,
        curve_p32_m=None if sc is None else sc.p32_wavelength_m,
        current_wavelength_m=current,
        notes=li.notes,
    )


# ---- noise -------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrumView:
    name: str
    unit: str
    omega_rad_s: np.ndarray
    S: np.ndarray
    level: Shown
    apparatus: tuple[str, ...]
    is_zero: bool


@dataclass(frozen=True)
class NoiseView:
    quiet: bool
    spectra: tuple[SpectrumView, ...]
    correlation: Row | None
    heating: tuple[Shown, ...]
    motional_dephasing: tuple[Shown, ...]
    qubit_dephasing: tuple[Shown, ...]
    drifts: tuple[tuple[str, Shown, Shown, Shown | None], ...]
    mains: tuple[Shown, ...]
    collisions: tuple[Shown, ...]
    apparatus: tuple[str, ...]
    provenance_sentence: str
    undeclared_rates: int
    notes: tuple[str, ...]


def noise_view(layer: DeviceLayer) -> NoiseView:
    nz = layer.noise
    spectra: list[SpectrumView] = []
    for rec, qid in (
        [(nz.s_e, "noise_density_e")]
        + ([(nz.s_b, "noise_density_b")] if nz.s_b is not None else [])
        + [(o, "noise_density") for o in nz.other_spectra]
    ):
        spectra.append(
            SpectrumView(
                name=rec.name,
                unit=rec.unit,
                omega_rad_s=rec.omega_rad_s,
                S=rec.S,
                level=Shown(qid, rec.white_level, f"white level of {rec.name} above the tabulated band"),
                apparatus=rec.apparatus,
                is_zero=rec.is_zero,
            )
        )
    corr = (
        None
        if nz.correlation_length_m is None
        else Row(
            "correlation length", Shown("correlation_length", nz.correlation_length_m), "device parameter"
        )
    )
    drifts = tuple(
        (
            d.name,
            Shown("drift_rms", d.rms, f"{d.name} ({d.unit})"),
            Shown("drift_tau", d.tau_s, d.name),
            None if d.servo_bandwidth_hz is None else Shown("servo_bandwidth", d.servo_bandwidth_hz, d.name),
        )
        for d in nz.drifts
    )
    mains: list[Shown] = []
    if nz.mains is not None:
        line, amps = nz.mains
        for k, amp in sorted(amps.items()):
            mains.append(Shown("mains_amplitude", amp, f"harmonic {k} of {line:g} Hz"))
    coll: list[Shown] = []
    if nz.collisions is not None:
        coll.append(Shown("pressure", nz.collisions[0]))
        coll.append(Shown("collision_rate", nz.collisions[1], "Langevin rate per ion"))
    return NoiseView(
        quiet=nz.quiet,
        spectra=tuple(spectra),
        correlation=corr,
        heating=tuple(
            Shown("heating_rate", v, f"mode {m}") for m, v in sorted(nz.heating_quanta_per_s.items())
        ),
        motional_dephasing=tuple(
            Shown("motional_dephasing", v, f"mode {m}")
            for m, v in sorted(nz.motional_dephasing_tau_s.items())
        ),
        qubit_dephasing=tuple(
            Shown("qubit_dephasing", v, f"ion {i}") for i, v in sorted(nz.qubit_white_dephasing_per_s.items())
        ),
        drifts=drifts,
        mains=tuple(mains),
        collisions=tuple(coll),
        apparatus=nz.apparatus,
        provenance_sentence=nz.provenance_sentence,
        undeclared_rates=nz.undeclared_rates,
        notes=nz.notes,
    )


# ---- cooling -------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DopplerRow:
    mode: int
    frequency: Shown
    heating: Shown
    cooling: Shown
    rate: Shown
    nbar: Shown
    participation: Shown
    force_model: Shown | None


@dataclass(frozen=True)
class SidebandView:
    mode: int
    ion: int
    eta: Shown
    omega0: Shown
    orders: tuple[int, ...]
    durations: tuple[Shown, ...]
    durations_s: tuple[float, ...]
    times_s: np.ndarray
    nbar_after_pulse: np.ndarray
    nbar_start: Shown
    nbar_end: Shown
    nbar_final_run: Shown


@dataclass(frozen=True)
class PumpView:
    ion: int
    preparation_error: Shown
    photons: Shown
    time_to_reach: Shown | None
    heating: tuple[Shown, ...]
    populations_end: tuple[Shown, ...]
    trace_times_s: np.ndarray
    trace_populations: dict[str, np.ndarray]


@dataclass(frozen=True)
class CoolingView:
    recipe: tuple[Row, ...]
    stages: tuple[tuple[str, str, tuple[Shown, ...]], ...]
    """(kind, provenance, nbar per mode)."""
    doppler: tuple[DopplerRow, ...]
    doppler_method: str
    approximations: tuple[str, ...]
    sidebands: tuple[SidebandView, ...]
    pumps: tuple[PumpView, ...]
    final_nbar: tuple[Shown, ...]
    duration: Shown
    provenance: tuple[str, ...]
    notes: tuple[str, ...]
    error: str | None


def cooling_view(layer: DeviceLayer) -> CoolingView | None:
    co = layer.cooling
    if co is None:
        return None
    recipe = [
        Row(
            "Doppler stage duration",
            Shown("stage_duration", co.doppler_duration_s, "Doppler"),
            "device parameter",
        ),
        Row(
            "pump duration",
            Shown("stage_duration", co.pump_duration_s, "optical pumping"),
            "device parameter",
        ),
    ]
    if co.repump_time_s is not None:
        recipe.append(
            Row(
                "repump time per sideband pulse",
                Shown("stage_duration", co.repump_time_s, "repump"),
                "device parameter",
            )
        )
    if co.repump_photons is not None:
        recipe.append(
            Row(
                "repump photons per pulse",
                Shown("pump_photons", co.repump_photons, "repump"),
                "device parameter",
            )
        )
    stages = tuple(
        (
            st.kind,
            st.provenance,
            ()
            if st.nbar is None
            else tuple(
                Shown("mode_nbar", v, f"mode {m} after {st.kind}") for m, v in sorted(st.nbar.items())
            ),
        )
        for st in co.stages
    )
    doppler = tuple(
        DopplerRow(
            mode=d.mode,
            frequency=Shown("mode_frequency", d.omega_hz, f"mode {d.mode}"),
            heating=Shown("heating_coefficient", d.heating_per_s, f"mode {d.mode}"),
            cooling=Shown("cooling_coefficient", d.cooling_per_s, f"mode {d.mode}"),
            rate=Shown("cooling_rate", d.rate_per_s, f"mode {d.mode}"),
            nbar=Shown("doppler_limit", d.nbar, f"mode {d.mode}"),
            participation=Shown("participation", d.participation, f"mode {d.mode}"),
            force_model=None
            if d.force_model_nbar is None
            else Shown("doppler_limit", d.force_model_nbar, "RMP force model, the nu << Gamma cross-check"),
        )
        for d in co.doppler
    )
    sidebands = tuple(
        SidebandView(
            mode=sb.mode,
            ion=sb.ion,
            eta=Shown("lamb_dicke", sb.eta, f"eta of the cooled ion {sb.ion} on mode {sb.mode}"),
            omega0=Shown("rabi_frequency", sb.omega0_hz, "carrier Rabi frequency of the cooling pair"),
            orders=sb.orders,
            durations=tuple(
                Shown("cooling_pulse_duration", t, f"order {k}") for k, t in zip(sb.orders, sb.durations_s)
            ),
            durations_s=sb.durations_s,
            times_s=sb.times_s,
            nbar_after_pulse=sb.nbar_after_pulse,
            nbar_start=Shown("mode_nbar", sb.nbar_start, "after Doppler cooling"),
            nbar_end=Shown(
                "mode_nbar", float(sb.nbar_after_pulse[-1]), "after the last pulse, without the repump recoil"
            ),
            nbar_final_run=Shown(
                "mode_nbar", sb.nbar_final_run, "the run's own value, repump recoil included"
            ),
        )
        for sb in co.sidebands
    )
    pumps = tuple(
        PumpView(
            ion=p.ion,
            preparation_error=Shown("prep_error", p.preparation_error, f"ion {p.ion}"),
            photons=Shown("pump_photons", p.photons_scattered, f"ion {p.ion}"),
            time_to_reach=None
            if p.time_to_reach_s is None
            else Shown("stage_duration", p.time_to_reach_s, "time to reach the target within tolerance"),
            heating=tuple(
                Shown("recoil_heating", v, f"mode {m}") for m, v in sorted(p.motional_heating_quanta.items())
            ),
            populations_end=tuple(
                Shown("pump_population", v, k) for k, v in sorted(p.populations_end.items())
            ),
            trace_times_s=p.trace_times_s,
            trace_populations=p.trace_populations,
        )
        for p in co.pumps
    )
    return CoolingView(
        recipe=tuple(recipe),
        stages=stages,
        doppler=doppler,
        doppler_method=co.doppler_method,
        approximations=co.approximations,
        sidebands=sidebands,
        pumps=pumps,
        final_nbar=tuple(
            Shown("mode_nbar", v, f"mode {m}, handed to the circuit")
            for m, v in sorted(co.final_nbar.items())
        ),
        duration=Shown("stage_duration", co.duration_s, "the whole preparation"),
        provenance=co.provenance,
        notes=co.notes,
        error=layer.cooling_error,
    )


# ---- readout -------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutIonView:
    ion: int
    rates: tuple[Row, ...]
    window: Shown
    bright_pmf: np.ndarray
    dark_pmf: np.ndarray
    threshold_at_window: Shown
    eps_at_window: tuple[Shown, Shown]
    best: tuple[Shown, Shown, Shown, Shown]
    """(window, threshold, eps_B, eps_D) at the interior optimum of the window scan."""
    scan_windows_s: np.ndarray
    scan_eps: np.ndarray
    scan_eps_b: np.ndarray
    scan_eps_d: np.ndarray
    budget: tuple[tuple[str, Shown, Shown], ...]
    saturation_s: np.ndarray | None
    saturation_r_o: np.ndarray | None
    saturation_r_d: np.ndarray | None
    saturation_r_b: np.ndarray | None
    saturation_ceiling_per_s: float | None
    saturation_now: Shown | None
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class ReadoutView:
    detector: tuple[Row, ...]
    ions: tuple[ReadoutIonView, ...]
    table: tuple[Row, ...]
    notes: tuple[str, ...]


def readout_view(layer: DeviceLayer, table: TableRecord | None) -> ReadoutView:
    ro = layer.readout
    detector = [
        Row("detector", Shown("species", ro.detector_kind), "device parameter"),
        Row("detection efficiency", Shown("detector_efficiency", ro.efficiency), "device parameter"),
        Row("background", Shown("background_rate", ro.background_cps), "device parameter"),
        Row("detection window", Shown("detection_window", ro.window_s), "device parameter"),
    ]
    if ro.numerical_aperture is not None:
        detector.append(
            Row(
                "numerical aperture",
                Shown("detector_efficiency", ro.numerical_aperture, "NA of the objective"),
                "device parameter",
            )
        )
    ions: list[ReadoutIonView] = []
    for i in ro.ions:
        rates = (
            Row(
                "bright ion scatters",
                Shown("scatter_rate_bright", i.r_bright_per_s, f"ion {i.ion}, R_o"),
                "derived",
            ),
            Row(
                "detected from a bright ion",
                Shown("detected_rate", i.detected_bright_per_s, f"ion {i.ion}"),
                "derived",
            ),
            Row(
                "background counted",
                Shown("background_rate", i.background_per_s, f"ion {i.ion}"),
                "device parameter",
            ),
            Row(
                "bright pumped dark",
                Shown("pump_rate_dark", i.r_dark_pumping_per_s, f"ion {i.ion}, R_d"),
                "derived",
            ),
            Row(
                "dark pumped bright",
                Shown("pump_rate_bright", i.r_bright_pumping_per_s, f"ion {i.ion}, R_b"),
                "derived",
            ),
            Row(
                "saturation ceiling",
                Shown("saturation_ceiling", i.ceiling, "excited fraction the manifold allows"),
                "derived",
            ),
            Row("readout scheme", Shown("readout_polarity", f"{i.scheme_kind}, {i.polarity}"), "designated"),
        )
        sc = i.saturation_curve
        ions.append(
            ReadoutIonView(
                ion=i.ion,
                rates=rates,
                window=Shown("detection_window", i.window_s, "the device window"),
                bright_pmf=i.bright_pmf,
                dark_pmf=i.dark_pmf,
                threshold_at_window=Shown(
                    "threshold", i.at_window_n_c, "best threshold at the device window"
                ),
                eps_at_window=(
                    Shown("spam_eps_b", i.at_window_eps_b, "at the device window"),
                    Shown("spam_eps_d", i.at_window_eps_d, "at the device window"),
                ),
                best=(
                    Shown("detection_window", i.best_window_s, "interior optimum of the scan"),
                    Shown("threshold", i.best_n_c, "at the optimum window"),
                    Shown("spam_eps_b", i.best_eps_b, "at the optimum"),
                    Shown("spam_eps_d", i.best_eps_d, "at the optimum"),
                ),
                scan_windows_s=i.scan.windows_s,
                scan_eps=0.5 * (i.scan.eps_b + i.scan.eps_d),
                scan_eps_b=i.scan.eps_b,
                scan_eps_d=i.scan.eps_d,
                budget=tuple(
                    (
                        name,
                        Shown("readout_budget_line", eb, f"{name}: bright read as dark"),
                        Shown("readout_budget_line", ed, f"{name}: dark read as bright"),
                    )
                    for name, eb, ed in i.budget
                ),
                saturation_s=None if sc is None else sc.s,
                saturation_r_o=None if sc is None else sc.r_bright_per_s,
                saturation_r_d=None if sc is None else sc.r_dark_pumping_per_s,
                saturation_r_b=None if sc is None else sc.r_bright_pumping_per_s,
                saturation_ceiling_per_s=None if sc is None else sc.ceiling_per_s,
                saturation_now=None
                if i.saturation is None
                else Shown("saturation_parameter", i.saturation, "the detection beam at this ion"),
                provenance=i.provenance,
            )
        )
    trows: list[Row] = []
    if table is not None:
        stale = table.device_hash != layer.device_hash
        status = "stale" if stale else "calibrated"
        for key, label, qid in (
            ("threshold", "table threshold", "threshold"),
            ("window_s", "table window", "detection_window"),
            ("eps_B", "table eps_B", "spam_eps_b"),
            ("eps_D", "table eps_D", "spam_eps_d"),
        ):
            e = table.entries.get(f"detection[{key}]") or table.entries.get(key)
            if e is not None:
                trows.append(Row(label, Shown(qid, e.value, f"{e.status}, {e.experiment}"), status))
    return ReadoutView(detector=tuple(detector), ions=tuple(ions), table=tuple(trows), notes=ro.notes)


# ---- gate solutions ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateRow:
    pair: tuple[int, int]
    summary: tuple[Row, ...]
    chi_m: tuple[Shown, ...]
    alpha_m: tuple[Shown, ...]
    modes: tuple[tuple[int, Shown, tuple[Shown, ...]], ...]
    """Per coupled mode: (index, frequency, eta per ion of the pair)."""
    table: tuple[Row, ...]
    error: str | None


def gate_rows(layer: DeviceLayer, table: TableRecord | None) -> tuple[GateRow, ...]:
    out: list[GateRow] = []
    for g in layer.gates:
        summary: list[Row] = [
            Row("beat note from the carrier", Shown("tone_detuning", g.mu_hz, g.mu_rule), "derived"),
            Row("duration", Shown("gate_duration", g.duration_s), "derived"),
        ]
        if g.method:
            summary.append(Row("solver", Shown("solver_method", g.method), "derived"))
        if g.error is None:
            summary.append(
                Row(
                    "entangling angle, closed form",
                    Shown(
                        "entangling_angle",
                        g.waveform.chi_total_rad,
                        "signed sum over modes of the Lamb-Dicke closed form (Section 4.4.3), before any exact check",
                    ),
                    "derived",
                )
            )
        if g.table_chi_m is not None:
            summary.append(
                Row(
                    "entangling angle the table plays",
                    Shown(
                        "entangling_angle",
                        sum(g.table_chi_m.values()),
                        "the exact spot check's angle, within its tolerance of pi/4 (Section 7.8)",
                    ),
                    "calibrated",
                )
            )
        if g.spot_check_amplitude_ratio is not None:
            summary.append(
                Row(
                    "played over closed-form amplitude",
                    Shown(
                        "spot_check_amplitude_ratio",
                        g.spot_check_amplitude_ratio,
                        "the exact spot check rescaled every segment by this one factor so that the exact angle hits pi/4",
                    ),
                    "calibrated",
                )
            )
        if g.surrogate_error is not None:
            summary.append(
                Row(
                    "what the closed form missed",
                    Shown(
                        "surrogate_error",
                        g.surrogate_error,
                        "exact angle over the closed-form angle at the played amplitude, minus one: the Debye-Waller and "
                        "beyond-Lamb-Dicke terms of Section 4.4.3",
                    ),
                    "calibrated",
                )
            )
        if g.residual_error is not None:
            summary.append(
                Row(
                    "residual entanglement with the motion",
                    Shown("residual_error", g.residual_error),
                    "estimate",
                )
            )
        if g.peak_rabi_hz is not None:
            summary.append(Row("peak Rabi frequency needed", Shown("peak_rabi", g.peak_rabi_hz), "derived"))
        if g.power_integral_rad2_s is not None:
            summary.append(Row("power integral", Shown("power_integral", g.power_integral_rad2_s), "derived"))
        chi = tuple(Shown("chi_m", v, f"mode {m}") for m, v in sorted(g.waveform.chi_m.items()))
        alpha = tuple(
            Shown("closure_alpha", abs(v), f"|alpha_{m}| at closure")
            for m, v in sorted(g.waveform.alpha_m.items())
        )
        modes = tuple(
            (
                m,
                Shown("mode_frequency", w, f"mode {m}"),
                tuple(Shown("lamb_dicke", g.eta[i][k], f"eta_{i},{m}") for i in sorted(g.eta)),
            )
            for k, (m, w) in enumerate(zip(g.modes, g.omega_hz))
        )
        trows: list[Row] = []
        if table is not None:
            wf = table.waveforms.get(f"{g.pair[0]},{g.pair[1]}") or table.waveforms.get(
                f"{g.pair[1]},{g.pair[0]}"
            )
            if wf is not None:
                status = "stale" if table.device_hash != layer.device_hash else "calibrated"
                trows.append(
                    Row(
                        "table: entangling angle",
                        Shown("entangling_angle", wf.chi_total_rad, f"phi_s {wf.phi_s.status}"),
                        status,
                    )
                )
                trows.append(Row("table: duration", Shown("gate_duration", wf.duration_s), status))
                for m, v in sorted(wf.chi_m.items()):
                    trows.append(Row(f"table: chi mode {m}", Shown("chi_m", v, f"mode {m}"), status))
                if g.table_chi_closed_form_m is not None:
                    for m, v in sorted(g.table_chi_closed_form_m.items()):
                        trows.append(
                            Row(
                                f"table: closed-form chi mode {m} at the played amplitude",
                                Shown(
                                    "chi_closed_form",
                                    v,
                                    f"mode {m}: 2 Im int conj(alpha_a) d alpha_b of the played pulse",
                                ),
                                status,
                            )
                        )
        out.append(
            GateRow(
                pair=g.pair,
                summary=tuple(summary),
                chi_m=chi,
                alpha_m=alpha,
                modes=modes,
                table=tuple(trows),
                error=g.error,
            )
        )
    return tuple(out)


# ---- the Hamiltonian page ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TermView:
    index: int
    pulse: str
    ion: int
    primary_ion: int
    is_crosstalk: bool
    kind: str
    beams: tuple[int, ...]
    omega: Shown
    tones: tuple[tuple[Shown, Shown, Shown], ...]
    """Per tone: (detuning from the carrier, phase at the start, peak envelope)."""
    etas: tuple[Shown, ...]
    debye_waller: Shown
    frozen: tuple[Shown, ...]
    carrier_factor: Shown
    crosstalk_weight: Shown
    rabi_scale: Shown
    nnz: Shown
    matrix_elements: dict[int, np.ndarray]
    operator_text: str


@dataclass(frozen=True)
class CollapseView:
    channel: str
    rate: Shown
    ion: int | None
    mode: int | None
    time_dependent: bool
    nnz: int
    active: bool
    note: str
    operator_text: str


@dataclass(frozen=True)
class HamiltonianView:
    gate_id: str
    header: tuple[Row, ...]
    dims: tuple[int, ...]
    free_modes: tuple[tuple[int, str, Shown, Shown], ...]
    """Per mode: (index, class, frequency term, sample offset)."""
    qubit_offsets: tuple[Shown, ...]
    stark: tuple[Shown, ...]
    terms: tuple[TermView, ...]
    collapse: tuple[CollapseView, ...]
    segments: tuple[tuple[Shown, Shown, str, int, Shown], ...]
    approximations: tuple[Shown, ...]
    caps: tuple[Shown, ...]


_CHANNEL_OPERATORS = {
    "heating_down": "sqrt(Gamma (N + 1)) a_m",
    "heating_up": "sqrt(Gamma N) a_m^dag",
    "motional_dephasing": "sqrt(2/tau) a_m^dag a_m",
    "qubit_dephasing": "sqrt(gamma_phi/2) sigma_z",
    "rayleigh_dephasing": "(1/2) sqrt(Gamma_el) sigma_z",
}


def hamiltonian_view(record: Record, ham: HamiltonianRecord) -> HamiltonianView:
    header = (
        Row(
            "gate step",
            Shown("native_gate", ham.gate_id, f"{ham.t_start_s * 1e6:.2f} to {ham.t_end_s * 1e6:.2f} us"),
            "record",
        ),
        Row("frame", Shown("frame", ham.frame), "derived"),
        Row("joint dimension", Shown("dimension", ham.dimension, f"dims {list(ham.dims)}"), "derived"),
        Row("fastest frequency in the frame", Shown("omega_max", ham.omega_max_hz), "derived"),
        Row(
            "drive terms",
            Shown("h_operator_nnz", ham.n_drive_terms, "count of coefficient-bearing terms"),
            "derived",
        ),
        Row("kernel", Shown("kernel", ham.kernel), "derived"),
        Row("fingerprint", Shown("hamiltonian_fingerprint", ham.fingerprint[:16]), "derived"),
        Row(
            "free term non-zeros",
            Shown("h_operator_nnz", ham.free_term_nnz, "H_mot + H_int + static Stark"),
            "derived",
        ),
    )
    free = tuple(
        (
            m,
            ham.mode_classes.get(m, "?"),
            Shown("h_mode_term", w, f"mode {m}, {ham.mode_classes.get(m, '?')}"),
            Shown(
                "noise_sample_value",
                ham.mode_offsets_hz.get(m, 0.0),
                f"quasi-static offset of mode {m} in this sample",
            ),
        )
        for m, w in sorted(ham.mode_frequencies_hz.items())
    )
    offsets = tuple(Shown("h_qubit_offset", v, f"ion {i}") for i, v in sorted(ham.qubit_offsets_hz.items()))
    stark = tuple(
        Shown("stark_shift", v, f"ion {i}, the pulse's own beams at its start")
        for i, v in sorted(ham.stark_shifts_hz.items())
    )
    terms: list[TermView] = []
    resolved = [m for m, c in ham.mode_classes.items() if c == "resolved"]
    for k, d in enumerate(ham.drives):
        # no resolved mode (every mode frozen or dropped): the term is the bare spin operator, with no dangling product
        op_text = " (x) ".join(
            [f"sigma_+^({d.ion})"] + [f"D_{m}(i eta = i {d.etas.get(m, 0.0):+.4f})" for m in resolved]
        )
        terms.append(
            TermView(
                index=k,
                pulse=d.pulse,
                ion=d.ion,
                primary_ion=d.primary_ion,
                is_crosstalk=d.ion != d.primary_ion,
                kind=d.kind,
                beams=d.beams,
                omega=Shown("h_drive_omega", d.omega_peak_hz, f"peak Omega/2pi seen by ion {d.ion}"),
                tones=tuple(
                    (
                        Shown("tone_detuning", mu, f"tone {j}: mu from the carrier"),
                        Shown("tone_phase", ph, f"tone {j}: phase at the pulse start"),
                        Shown("tone_envelope", pk, f"tone {j}: peak Omega/2pi of the addressed ion"),
                    )
                    for j, (mu, ph, pk) in enumerate(
                        zip(d.tone_detunings_hz, d.tone_phases_rad, d.tone_peaks_hz)
                    )
                ),
                etas=tuple(
                    Shown("lamb_dicke", e, f"eta_{d.ion},{m} ({ham.mode_classes.get(m, '?')})")
                    for m, e in sorted(d.etas.items())
                    if e != 0.0
                ),
                debye_waller=Shown(
                    "h_debye_waller_frozen",
                    d.debye_waller,
                    "product over the frozen spectators at their sampled n",
                ),
                frozen=tuple(
                    Shown("h_debye_waller_frozen", v, f"frozen mode {m}, n = {d.frozen_n.get(m, 0)}")
                    for m, v in sorted(d.frozen_debye_waller.items())
                ),
                carrier_factor=Shown(
                    "h_carrier_factor", d.carrier_factor, f"beta_mm = {d.micromotion_beta:.3g}"
                ),
                crosstalk_weight=Shown("h_crosstalk_weight", abs(d.crosstalk), "1 for the addressed ion"),
                rabi_scale=Shown("noise_sample_value", d.rabi_scale, "the sample's Rabi-frequency scale"),
                nnz=Shown("h_operator_nnz", d.operator_nnz),
                matrix_elements=d.matrix_elements,
                operator_text=op_text,
            )
        )
    collapse = tuple(
        CollapseView(
            channel=c.channel,
            rate=Shown(
                "collapse_rate",
                c.rate_hz,
                c.channel
                + ("" if c.mode is None else f", mode {c.mode}")
                + ("" if c.ion is None else f", ion {c.ion}"),
            ),
            ion=c.ion,
            mode=c.mode,
            time_dependent=c.time_dependent,
            nnz=c.operator_nnz,
            active=c.active_in_run,
            note=c.note,
            operator_text=_CHANNEL_OPERATORS.get(c.channel.split("[")[0], c.channel),
        )
        for c in ham.collapse
    )
    segments = tuple(
        (
            Shown(
                "gate_duration",
                s.t_end_s - s.t_start_s,
                f"{s.t_start_s * 1e6:.2f} to {s.t_end_s * 1e6:.2f} us",
            ),
            Shown("omega_max", s.omega_max_hz),
            ", ".join(s.pulses),
            s.n_drive_terms,
            Shown("kernel", s.kernel),
        )
        for s in ham.segments
    )
    return HamiltonianView(
        gate_id=ham.gate_id,
        header=header,
        dims=ham.dims,
        free_modes=free,
        qubit_offsets=offsets,
        stark=stark,
        terms=tuple(terms),
        collapse=collapse,
        segments=segments,
        approximations=tuple(Shown("approximation", a) for a in ham.approximations),
        caps=tuple(Shown("truncation_cap", d, f"mode {m}") for m, d in sorted(ham.caps.items())),
    )


# ---- the device card of an edited device (Level 0's estimated columns, Section 14.4) -------------------------------------------------


@dataclass(frozen=True)
class LayerCardView:
    rows: tuple[Row, ...]
    modes: tuple[Shown, ...]
    spam: tuple[Row, ...]
    gate_errors: tuple[Row, ...]
    device_hash: Shown
    stale_note: str
    overrides: tuple[Row, ...]


def layer_card_view(layer: DeviceLayer, table: TableRecord | None) -> LayerCardView:
    card = layer.card
    rows: list[Row] = []
    if card.trap_omega_hz is not None:
        for ax, w in zip(("x", "y", "z"), card.trap_omega_hz):
            rows.append(Row(f"trap frequency {ax}", Shown("secular_frequency", w), "device parameter"))
    rows.append(Row("magnetic field", Shown("field", card.field_gauss), "device parameter"))
    for key, value in sorted(card.derived.values.items()):
        if key.startswith("qubit_freq_hz"):
            rows.append(
                Row(
                    f"qubit frequency {key[len('qubit_freq_hz') :]}",
                    Shown("qubit_frequency", value),
                    "derived",
                )
            )
    for m, rate in sorted(card.heating_quanta_per_s.items()):
        rows.append(Row(f"heating rate, mode {m}", Shown("heating_rate", rate), "derived"))
    rows.append(
        Row("detection window", Shown("detection_window", card.detector.window_s), "device parameter")
    )
    modes = tuple(Shown("mode_frequency", m.omega_hz, f"{m.family} {m.family_index}") for m in card.modes)
    spam: list[Row] = []
    stale = table is not None and table.device_hash != layer.device_hash
    for key in sorted(card.spam):
        eb, ed = card.spam[key]
        if key.endswith(".state_preparation"):
            spam.append(
                Row(f"{key.split('.')[0]} preparation error", Shown("prep_error", eb), "derived (recipe)")
            )
        else:
            spam.append(
                Row(
                    f"{key} bright read as dark",
                    Shown("spam_eps_b", eb),
                    "estimate (exact count distributions)",
                )
            )
            spam.append(
                Row(
                    f"{key} dark read as bright",
                    Shown("spam_eps_d", ed),
                    "estimate (exact count distributions)",
                )
            )
    if table is not None:
        status = "stale" if stale else "calibrated"
        for key, qid in (("eps_B", "spam_eps_b"), ("eps_D", "spam_eps_d")):
            e = table.entries.get(f"detection[{key}]") or table.entries.get(key)
            if e is not None:
                spam.append(Row(f"table {key}", Shown(qid, e.value, e.experiment), status))
    errors = tuple(
        Row(gate, Shown("gate_error_estimate", v), "estimate")
        for gate, v in sorted(card.gate_error_estimates.items())
    )
    overrides = tuple(
        Row(knob(kid).label, Shown(_knob_quantity(kid), v, knob(kid).term), "override")
        for kid, v in sorted(layer.overrides.items())
    )
    return LayerCardView(
        rows=tuple(rows),
        modes=modes,
        spam=tuple(spam),
        gate_errors=errors,
        device_hash=Shown("device_hash", layer.device_hash),
        stale_note=stale_status(layer),
        overrides=overrides,
    )


def _knob_quantity(knob_id: str) -> str:
    """The catalogue quantity a knob's current value is shown as (its own ledger id is on the knob)."""
    table = {
        "trap.rf_amplitude_scale": "rf_voltage",
        "trap.omega_z_hz": "secular_frequency",
        "trap.rf_frequency_hz": "rf_frequency",
        "trap.rf_voltage_peak_v": "rf_voltage",
        "trap.stray_field_x_v_per_m": "stray_field",
        "trap.stray_field_y_v_per_m": "stray_field",
        "trap.axis_angle_rad": "axis_angle",
        "field.b_gauss": "field",
        "detector.window_s": "detection_window",
        "detector.efficiency": "detector_efficiency",
        "detector.background_cps": "background_rate",
        "noise.s_e_white": "noise_density_e",
        "noise.correlation_length_m": "correlation_length",
        "noise.s_b_white": "noise_density_b",
        "noise.rabi_drift_rms": "drift_rms",
        "preparation.doppler_duration_s": "stage_duration",
        "preparation.pump_duration_s": "stage_duration",
    }
    if knob_id in table:
        return table[knob_id]
    if knob_id.endswith(".power_w"):
        return "beam_power"
    if knob_id.endswith(".waist_m"):
        return "beam_waist"
    if knob_id.endswith(".wavelength_m"):
        return "wavelength"
    return "noise_sample_value"


__all__ = [
    "BeamRow",
    "CollapseView",
    "CoolingView",
    "CrystalView",
    "DopplerRow",
    "DriveRow",
    "GateRow",
    "HamiltonianView",
    "KnobRow",
    "LayerCardView",
    "LightView",
    "ModeRow",
    "NoiseView",
    "PumpView",
    "ReadoutIonView",
    "ReadoutView",
    "Row",
    "SidebandView",
    "SpeciesView",
    "SpectrumView",
    "TermView",
    "TrapView",
    "cooling_view",
    "crystal_view",
    "gate_rows",
    "hamiltonian_view",
    "knob_rows",
    "layer_card_view",
    "light_view",
    "noise_view",
    "readout_view",
    "species_view",
    "stale_status",
    "trap_view",
]
