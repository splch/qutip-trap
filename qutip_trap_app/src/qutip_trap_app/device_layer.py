"""The Level 4 device layer: everything the physics pages show, derived from a built preset into frozen records (PLAN.md
Sections 14.2 row 4, 14.4).

The device model is the single source of truth (Section 14.4): the species, trap, crystal, light, noise, cooling and readout
pages, the pulse-solver solution of every entangling pair and the device card of Level 0 are computed here from the core
alone, in the worker. Every drawn position, arrow, curve and level comes from a core call; the one exception, a level
diagram's vertical spacing, is labelled by the page.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from qutip_trap_app import core, knobs
from qutip_trap_app.record import (
    CalEntryRecord,
    DeviceCard,
    Progress,
    TableRecord,
    WaveformRecord,
    device_card,
    vec3,
    waveform_record,
)

TWO_PI = core.TWO_PI

RESONANT_WINDOW_M = 1e-9
"""Beams within a nanometre of the species' cycling line are its near-resonant light (cooling, pumping, detection)."""

DEFAULT_MS_DURATION_S = 100e-6
"""The entangling-pulse duration the solver uses for a pair the table does not carry."""


# ---- species -------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelRecord:
    name: str
    energy_hz: float
    lifetime_s: float | None
    A_hfs_hz: float
    B_hfs_hz: float
    g_J: float
    gamma_total_hz: float | None


@dataclass(frozen=True)
class TransitionRecord:
    label: str
    lower: str
    upper: str
    wavelength_m: float
    gamma_hz: float
    branching: float
    multipole: str
    i_sat_w_m2: float | None


@dataclass(frozen=True)
class SublevelRecord:
    level: str
    label: str
    energy_hz: float
    """Relative to the level's zero-field energy (the hyperfine-plus-Zeeman eigenvalue)."""
    dE_dB_hz_per_g: float
    d2E_dB2_hz_per_g2: float


@dataclass(frozen=True)
class ZeemanSweep:
    level: str
    b_gauss: np.ndarray
    labels: tuple[str, ...]
    energies_hz: np.ndarray
    """(K, n) eigenvalues relative to the level's zero-field energy, columns in ``labels`` order (adiabatic labels)."""


@dataclass(frozen=True)
class SpeciesLayer:
    name: str
    mass_u: float
    nuclear_spin: float
    mu_i_nuclear_magnetons: float
    levels: tuple[LevelRecord, ...]
    transitions: tuple[TransitionRecord, ...]
    qubit: tuple[str, str]
    cycling: str
    repumps: tuple[str, ...]
    field_gauss: float
    qubit_freq_hz: float
    dnu_db_hz_per_g: float
    d2nu_db2_hz_per_g2: float
    clock_points: tuple[tuple[float, float, float], ...]
    """(B0 in G, |nu| in Hz, Taylor c2 in Hz/G^2) of the field-independent points of the qubit pair in the scanned range."""
    clock_scan_gauss: tuple[float, float]
    sublevels: tuple[SublevelRecord, ...]
    sweep: ZeemanSweep
    notes: tuple[str, ...]


def species_layer(device: core.Device) -> SpeciesLayer:
    crystal = device.crystal
    sp = crystal.species[0]
    b = float(device.field.B_gauss)
    levels = tuple(
        LevelRecord(
            name=str(lv.name),
            energy_hz=float(lv.energy_hz),
            lifetime_s=None if lv.lifetime_s is None else float(lv.lifetime_s),
            A_hfs_hz=float(lv.A_hfs_hz),
            B_hfs_hz=float(lv.B_hfs_hz),
            g_J=float(lv.g_J),
            gamma_total_hz=None if lv.gamma_total_hz is None else float(lv.gamma_total_hz),
        )
        for lv in sp.levels
    )
    transitions: list[TransitionRecord] = []
    for tr in sp.transitions:
        try:
            isat: float | None = float(tr.i_sat_w_m2)
        except (ValueError, ZeroDivisionError, AttributeError):
            isat = None
        transitions.append(
            TransitionRecord(
                label=str(tr.label),
                lower=str(tr.lower),
                upper=str(tr.upper),
                wavelength_m=float(tr.wavelength_vac_m),
                gamma_hz=float(tr.gamma_hz),
                branching=float(tr.branching),
                multipole=str(tr.multipole),
                i_sat_w_m2=isat,
            )
        )
    f0, d1, d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], b)
    b_hi = max(50.0, 4.0 * b)
    clock = tuple(
        (float(cp.B0_gauss), float(cp.frequency_hz), float(cp.taylor_c2_hz_per_g2))
        for cp in sp.clock_points(sp.qubit[0], sp.qubit[1], 0.01, b_hi)
    )
    qubit_level = sp.qubit[0].split(" ")[0]
    cycling_upper = sp.transition(sp.cycling).upper
    sublevels: list[SublevelRecord] = []
    for name in dict.fromkeys((qubit_level, cycling_upper)):
        z = sp.zeeman_spectrum(name, b)
        e0 = float(sp.level(name).energy_hz)
        for k, label in enumerate(z.labels):
            sublevels.append(
                SublevelRecord(
                    level=name,
                    label=str(label),
                    energy_hz=float(z.energies_hz[k]) - e0,
                    dE_dB_hz_per_g=float(z.dE_dB_hz_per_g[k]),
                    d2E_dB2_hz_per_g2=float(z.d2E_dB2_hz_per_g2[k]),
                )
            )
    grid = np.linspace(0.0, max(2.0 * b, 10.0), 41)
    ref = sp.zeeman_spectrum(qubit_level, b if b > 0 else 1.0)
    e0 = float(sp.level(qubit_level).energy_hz)
    energies = np.zeros((grid.size, len(ref.labels)))
    for k, bg in enumerate(grid):
        z = sp.zeeman_spectrum(qubit_level, float(bg))
        # adiabatic labels: the same label names the same state at every field
        for j, label in enumerate(ref.labels):
            energies[k, j] = float(z.energy_hz(str(label))) - e0
    notes = []
    if len({s.name for s in crystal.species}) > 1:
        notes.append(
            "mixed-species crystal: the page shows the first ion's species; the crystal page carries every mass"
        )
    return SpeciesLayer(
        name=str(sp.name),
        mass_u=float(sp.mass_u),
        nuclear_spin=float(sp.nuclear_spin),
        mu_i_nuclear_magnetons=float(sp.mu_I_nuclear_magnetons),
        levels=levels,
        transitions=tuple(transitions),
        qubit=(str(sp.qubit[0]), str(sp.qubit[1])),
        cycling=str(sp.cycling),
        repumps=tuple(str(r) for r in sp.repumps),
        field_gauss=b,
        qubit_freq_hz=float(f0),
        dnu_db_hz_per_g=float(d1),
        d2nu_db2_hz_per_g2=float(d2),
        clock_points=clock,
        clock_scan_gauss=(0.01, b_hi),
        sublevels=tuple(sublevels),
        sweep=ZeemanSweep(qubit_level, grid, tuple(str(x) for x in ref.labels), energies),
        notes=tuple(notes),
    )


# ---- trap ------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityMap:
    """The first stability region of x'' + (a - 2q cos 2xi) x = 0 (Section 4.1.1): for every q of the grid the lowest and
    highest stable a, found by the core's monodromy (``is_stable``) and bisected."""

    q: np.ndarray
    a_lower: np.ndarray
    a_upper: np.ndarray
    edge_q_at_a0: float
    """The q at which the region closes on the a = 0 axis (0.908 by the source, Section 4.1.1)."""
    monodromy_calls: int


@functools.cache
def stability_map() -> StabilityMap:
    """The stability boundary on q in [0, 1] (25 points, bisected to 2e-4), computed once per process: it is
    device-independent."""
    q_max, points, tolerance = 1.0, 25, 2e-4
    calls = 0

    def stable(a: float, q: float) -> bool:
        nonlocal calls
        calls += 1
        return bool(core.is_stable(a, q))

    def bisect(lo: float, hi: float, q: float, stable_at_lo: bool) -> float:
        # lo and hi bracket an edge: exactly one of them is stable
        while hi - lo > tolerance:
            mid = 0.5 * (lo + hi)
            if stable(mid, q) == stable_at_lo:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    qs = np.linspace(0.0, q_max, points)
    lower = np.full(points, np.nan)
    upper = np.full(points, np.nan)
    scan = np.arange(-0.7, 1.15, 0.05)
    for k, q in enumerate(qs):
        flags = [stable(float(a), float(q)) for a in scan]
        idx = [i for i, f in enumerate(flags) if f]
        if not idx:
            continue
        i0, i1 = idx[0], idx[-1]
        lo = float(scan[i0]) if i0 == 0 else bisect(float(scan[i0 - 1]), float(scan[i0]), float(q), False)
        hi = (
            float(scan[i1])
            if i1 == len(scan) - 1
            else bisect(float(scan[i1]), float(scan[i1 + 1]), float(q), True)
        )
        lower[k], upper[k] = lo, hi
    # the closing q on the a = 0 axis, by bisection on the boundary itself
    q_lo, q_hi = 0.5, q_max
    while q_hi - q_lo > tolerance and stable(0.0, q_lo):
        mid = 0.5 * (q_lo + q_hi)
        if stable(0.0, mid):
            q_lo = mid
        else:
            q_hi = mid
    return StabilityMap(
        q=qs,
        a_lower=lower,
        a_upper=upper,
        edge_q_at_a0=0.5 * (q_lo + q_hi),
        monodromy_calls=calls,
    )


@dataclass(frozen=True)
class TrapLayer:
    path: str
    omega_hz: tuple[float, float, float] | None
    axis_angle_rad: float
    rf_voltage_peak_v: float | None
    rf_frequency_hz: float | None
    a: tuple[float, float, float] | None
    q: tuple[float, float, float] | None
    beta: tuple[float, float, float] | None
    secular_from_mathieu_hz: tuple[float, float, float] | None
    c0: tuple[float, float, float] | None
    mathieu_note: str
    stray_field_v_per_m: tuple[float, float, float]
    residual_field_v_per_m: tuple[float, float, float]
    micromotion_amplitude_m: tuple[float, float, float] | None
    displacement_m: tuple[float, float, float]
    """Where the stray field parks the first ion: the crystal's linear response K^{-1} e E (Section 4.1.7), which is
    Q E/(m omega^2) along each principal axis of an equal-mass chain (Berkeland)."""
    stability: StabilityMap
    ion_height_m: float | None
    trap_depth_ev: float | None
    notes: tuple[str, ...]


def trap_layer(device: core.Device, derived: Mapping[str, float]) -> TrapLayer:
    """The trap page; ``derived`` is ``Device.derived().values``."""
    trap = device.trap
    sp = device.crystal.species[0]
    notes: list[str] = []
    a = q = beta = sec = c0 = None
    if trap.rf is None:
        note = (
            "this trap is declared by its secular frequencies with no rf record: a, q, beta and C0 need the rf drive frequency "
            "(set 'Radio-frequency drive frequency' below; Section 4.1.1)"
        )
    else:
        try:
            mp = trap.mathieu(sp)
            a = vec3(mp.diagonal_a)
            q = vec3(mp.q_effective)
            beta = vec3(mp.beta)
            sec = vec3(mp.secular_hz)
            c0 = vec3(mp.C0)
            note = f"a, q from the secular frequencies at Omega_rf/2pi = {trap.rf.frequency_hz:.4g} Hz (q_z = 0, q_y = -q_x, sum a = 0); beta by the monodromy method"
        except ValueError as exc:
            note = f"Mathieu parameters unavailable: {exc}"
    stray = vec3(trap.stray_field_v_per_m)
    residual = vec3(trap.residual_field_v_per_m())
    amp: tuple[float, float, float] | None = None
    if trap.rf is not None:
        try:
            amp = vec3(trap.micromotion_amplitude_m(sp))
        except (ValueError, NotImplementedError) as exc:
            notes.append(f"excess micromotion amplitude unavailable: {exc}")
    disp = vec3(device.crystal.field_displacement_m(np.asarray(stray, dtype=float))[0])
    height = derived.get("ion_height_m")
    depth = derived.get("trap_depth_ev")
    return TrapLayer(
        path=str(trap.path),
        omega_hz=None if trap.omega_hz is None else vec3(trap.omega_hz),
        axis_angle_rad=float(trap.axis_angle_rad),
        rf_voltage_peak_v=None if trap.rf is None else float(trap.rf.voltage_peak_v),
        rf_frequency_hz=None if trap.rf is None else float(trap.rf.frequency_hz),
        a=a,
        q=q,
        beta=beta,
        secular_from_mathieu_hz=sec,
        c0=c0,
        mathieu_note=note,
        stray_field_v_per_m=stray,
        residual_field_v_per_m=residual,
        micromotion_amplitude_m=amp,
        displacement_m=disp,
        stability=stability_map(),
        ion_height_m=None if height is None else float(height),
        trap_depth_ev=None if depth is None else float(depth),
        notes=tuple(notes),
    )


# ---- crystal ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeLayer:
    index: int
    family: str
    family_index: int
    omega_hz: float
    e_hat: tuple[float, float, float]
    eigenvector: np.ndarray
    uniform_field_weight: float
    heating_quanta_per_s: float | None


@dataclass(frozen=True)
class CrystalLayer:
    n_ions: int
    species: tuple[str, ...]
    positions_m: np.ndarray
    modes: tuple[ModeLayer, ...]
    entangling_beams: tuple[int, int] | None
    delta_k_rad_per_m: tuple[float, float, float] | None
    eta: np.ndarray | None
    """(N, 3N) Lamb-Dicke parameters for the entangling pair's Delta k, each ion with its own species' C0 when the trap has
    an rf record (``light.raman.lamb_dicke_parameters``); None when the trap record cannot be evaluated."""
    c0_applied: bool
    length_scale_m: float | None
    spacing_m: float | None
    zigzag_ratio: float | None
    """omega_r,min/omega_z,COM of this crystal."""
    zigzag_critical: float | None
    """sqrt((mu_N - 1)/2): Marquet's exact threshold from the axial spectrum (Section 4.1.2)."""
    notes: tuple[str, ...]


def crystal_layer(preset: core.DevicePreset) -> CrystalLayer:
    device = preset.device
    crystal = device.crystal
    heating = device.noise.heating_rates_quanta_per_s(device)
    modes = tuple(
        ModeLayer(
            index=k,
            family=str(m.family),
            family_index=int(m.index),
            omega_hz=float(m.omega_hz),
            e_hat=vec3(m.e_hat),
            eigenvector=np.asarray(m.eigenvector, dtype=float),
            uniform_field_weight=float(crystal.uniform_field_weight(k)),
            heating_quanta_per_s=None if k not in heating else float(heating[k]),
        )
        for k, m in enumerate(crystal.modes)
    )
    pair = knobs.entangling_pair(preset)
    eta = None
    dk: tuple[float, float, float] | None = None
    c0_applied = False
    notes: list[str] = []
    if pair is not None:
        dkv = np.asarray(device.beams[pair[0]].k_vector() - device.beams[pair[1]].k_vector(), dtype=float)
        dk = vec3(dkv)
        try:
            rows = [core.lamb_dicke_parameters(device, i, dkv) for i in range(crystal.n_ions)]
        except (
            ValueError
        ) as exc:  # a trap record the Mathieu map cannot evaluate: shown on the page, eta withheld
            notes.append(f"eta unavailable: {exc}")
        else:
            eta = np.array([[etas[m] for m in range(len(crystal.modes))] for etas, _ in rows], dtype=float)
            c0_applied = rows[0][1]
    else:
        notes.append(
            "no entangling Raman pair on this device: eta is shown per gate drive on the light page only"
        )
    axial = crystal.family("axial")
    ratio = crit = None
    if axial and device.trap.omega_hz is not None:
        w = sorted(m.omega_hz for m in axial)
        mu_n = (w[-1] / w[0]) ** 2
        crit = math.sqrt(max(mu_n - 1.0, 0.0) / 2.0)
        radial = [m.omega_hz for m in crystal.modes if m.family != "axial"]
        ratio = min(radial) / w[0] if radial else None
    length_scale = None
    if device.trap.omega_hz is not None:
        m_kg = float(crystal.masses_kg[0])
        wz = TWO_PI * float(device.trap.omega_hz[2])
        length_scale = (core.E_C**2 / (4.0 * math.pi * core.EPSILON_0_F_PER_M * m_kg * wz * wz)) ** (
            1.0 / 3.0
        )
    pos = np.asarray(crystal.positions_m, dtype=float)
    spacing = float(np.min(np.linalg.norm(pos[1:] - pos[:-1], axis=1))) if pos.shape[0] > 1 else None
    return CrystalLayer(
        n_ions=int(crystal.n_ions),
        species=tuple(str(s.name) for s in crystal.species),
        positions_m=pos,
        modes=modes,
        entangling_beams=pair,
        delta_k_rad_per_m=dk,
        eta=eta,
        c0_applied=c0_applied,
        length_scale_m=length_scale,
        spacing_m=spacing,
        zigzag_ratio=ratio,
        zigzag_critical=crit,
        notes=tuple(notes),
    )


# ---- light -----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BeamLayer:
    index: int
    wavelength_m: float
    k_hat: tuple[float, float, float]
    polarization: tuple[complex, complex, complex]
    waist_m: float
    power_w: float
    pointing_m: tuple[float, float, float]
    roles: tuple[str, ...]
    intensity_at_ions_w_m2: np.ndarray
    saturation_at_ions: np.ndarray | None
    """I/I_sat of the species' cycling line at every ion, for a beam near that line; None otherwise."""
    angle_to_field_deg: float


@dataclass(frozen=True)
class DriveLayer:
    ion: int
    role: str
    kind: str
    beams: tuple[int, ...]
    rabi_hz: float
    pi_time_s: float
    stark_hz: float
    delta_k_rad_per_m: tuple[float, float, float]
    etas: dict[int, float]
    residual_excited_population: float
    rayleigh_per_s: dict[str, float]
    raman_per_s: dict[str, float]
    leakage_per_s: dict[str, float]
    rayleigh_dephasing_per_s: float
    error_per_pi_pulse: float
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class ScatteringCurve:
    beams: tuple[int, int]
    wavelength_m: np.ndarray
    detuning_from_p12_hz: np.ndarray
    rabi_hz: np.ndarray
    error_per_pi_pulse: np.ndarray
    p12_wavelength_m: float
    p32_wavelength_m: float | None


@dataclass(frozen=True)
class LightLayer:
    beams: tuple[BeamLayer, ...]
    drives: tuple[DriveLayer, ...]
    crosstalk: dict[str, float]
    """``"i,j"`` -> |Omega_j/Omega_i| of ion i's single-qubit beams on ion j (``crosstalk_ratios``, Section 6.6)."""
    scattering_curve: ScatteringCurve | None
    notes: tuple[str, ...]


def _beam_roles(preset: core.DevicePreset) -> dict[int, list[str]]:
    device = preset.device
    roles: dict[int, list[str]] = {k: [] for k in range(len(device.beams))}
    if 0 <= preset.detection_beam < len(device.beams):
        roles[preset.detection_beam].append("detection")
    for ion, d in preset.gate_drives.items():
        for b in d.beams:
            roles[int(b)].append(f"single-qubit gates of ion {ion}")
    seen: set[int] = set()
    for d in preset.entangling_drives.values():
        for b in d.beams:
            if int(b) not in seen:
                roles[int(b)].append("entangling gates")
                seen.add(int(b))
    recipe = device.preparation
    if recipe is not None:
        for name, group in (
            ("Doppler cooling", recipe.doppler_beams),
            ("optical pumping", recipe.pump_beams),
        ):
            for rb in group:
                for k, beam in enumerate(device.beams):
                    same = (
                        abs(beam.wavelength_m - rb.wavelength_m) < RESONANT_WINDOW_M
                        and np.allclose(beam.k_hat, rb.k_hat)
                        and abs(beam.waist_m - rb.waist_m) <= 1e-12
                    )
                    if same and name not in roles[k]:
                        roles[k].append(name)
    return roles


def _drive_layer(device: core.Device, ion: int, beams: tuple[int, int], role: str) -> DriveLayer:
    dd = core.derive_raman_drive(device, ion, beams)
    sc = dd.scattering
    pi_time = float(dd.pi_time_s())
    return DriveLayer(
        ion=int(ion),
        role=role,
        kind=str(dd.kind),
        beams=tuple(int(b) for b in dd.beams),
        rabi_hz=float(dd.carrier_rabi_hz),
        pi_time_s=pi_time,
        stark_hz=float(dd.stark_shift_hz),
        delta_k_rad_per_m=vec3(dd.delta_k),
        etas={int(m): float(e) for m, e in dd.etas.items()},
        residual_excited_population=0.0 if sc is None else float(sc.residual_excited_population),
        rayleigh_per_s={} if sc is None else {str(k): float(v) for k, v in sc.rayleigh_per_s.items()},
        raman_per_s={} if sc is None else {str(k): float(v) for k, v in sc.raman_spin_flip_per_s.items()},
        leakage_per_s={} if sc is None else {str(k): float(v) for k, v in sc.leakage_per_s.items()},
        rayleigh_dephasing_per_s=0.0 if sc is None else float(sc.rayleigh_dephasing_per_s),
        error_per_pi_pulse=0.0 if sc is None else float(sc.per_pulse_error(pi_time)),
        provenance=tuple(str(p) for p in dd.provenance),
    )


def _scattering_curve(
    device: core.Device, ion: int, beams: tuple[int, int], *, points: int = 21
) -> ScatteringCurve | None:
    sp = device.crystal.species[ion]
    try:
        p12 = sp.transition(sp.cycling)
    except KeyError:
        return None
    lam_p12 = float(p12.wavelength_vac_m)
    lam_p32: float | None = None
    for tr in sp.transitions:
        if (
            tr.lower == p12.lower
            and tr.upper != p12.upper
            and tr.multipole == "E1"
            and tr.wavelength_vac_m < lam_p12
        ):
            lam_p32 = (
                float(tr.wavelength_vac_m) if lam_p32 is None else max(lam_p32, float(tr.wavelength_vac_m))
            )
    lo = lam_p32 * 1.01 if lam_p32 is not None else lam_p12 * 0.85
    hi = lam_p12 * 0.99
    out_rabi, out_err = [], []
    keep: list[float] = []
    for lam in np.linspace(lo, hi, points):
        new_beams = list(device.beams)
        for b in beams:
            new_beams[b] = dataclasses.replace(new_beams[b], wavelength_m=float(lam))
        try:
            dd = core.derive_raman_drive(dataclasses.replace(device, beams=tuple(new_beams)), ion, beams)
        except ZeroDivisionError:  # a tune-out wavelength: no two-photon coupling at the ion
            continue
        sc = dd.scattering
        if sc is None:
            continue
        keep.append(float(lam))
        out_rabi.append(float(dd.carrier_rabi_hz))
        out_err.append(float(sc.per_pulse_error(dd.pi_time_s())))
    if not keep:
        return None
    lam_arr = np.asarray(keep)
    return ScatteringCurve(
        beams=beams,
        wavelength_m=lam_arr,
        detuning_from_p12_hz=core.C_M_PER_S / lam_arr - core.C_M_PER_S / lam_p12,
        rabi_hz=np.asarray(out_rabi),
        error_per_pi_pulse=np.asarray(out_err),
        p12_wavelength_m=lam_p12,
        p32_wavelength_m=lam_p32,
    )


def light_layer(preset: core.DevicePreset, *, scattering_sweep: bool = True) -> LightLayer:
    device = preset.device
    pos = np.asarray(device.crystal.positions_m, dtype=float)
    sp = device.crystal.species[0]
    cycling = sp.transition(sp.cycling)
    b_hat = np.asarray(device.field.direction, dtype=float)
    roles = _beam_roles(preset)
    beams: list[BeamLayer] = []
    for k, b in enumerate(device.beams):
        intensity = np.asarray([float(b.intensity_at(p)) for p in pos])
        sat = None
        if abs(b.wavelength_m - cycling.wavelength_vac_m) < RESONANT_WINDOW_M:
            try:
                sat = intensity / float(cycling.i_sat_w_m2)
            except (ValueError, ZeroDivisionError):
                sat = None
        cosang = float(np.clip(abs(np.dot(np.asarray(b.k_hat, dtype=float), b_hat)), 0.0, 1.0))
        beams.append(
            BeamLayer(
                index=k,
                wavelength_m=float(b.wavelength_m),
                k_hat=vec3(b.k_hat),
                polarization=(
                    complex(b.polarization[0]),
                    complex(b.polarization[1]),
                    complex(b.polarization[2]),
                ),
                waist_m=float(b.waist_m),
                power_w=float(b.power_w),
                pointing_m=vec3(b.pointing_m),
                roles=tuple(roles[k]),
                intensity_at_ions_w_m2=intensity,
                saturation_at_ions=sat,
                angle_to_field_deg=math.degrees(math.acos(cosang)),
            )
        )
    drives: list[DriveLayer] = []
    notes: list[str] = []
    crosstalk: dict[str, float] = {}
    for ion, d in sorted(preset.gate_drives.items()):
        if d.kind == "raman" and len(d.beams) == 2:
            drives.append(
                _drive_layer(device, int(ion), (int(d.beams[0]), int(d.beams[1])), "single-qubit gates")
            )
            for j, eps in core.crosstalk_ratios(device, int(ion), d.beams).items():
                crosstalk[f"{ion},{j}"] = abs(eps)
        else:
            notes.append(
                f"ion {ion}: a {d.kind} gate drive is not derived here (the public API derives Raman pairs)"
            )
    for ion, d in sorted(preset.entangling_drives.items()):
        if d.kind == "raman" and len(d.beams) == 2:
            drives.append(
                _drive_layer(device, int(ion), (int(d.beams[0]), int(d.beams[1])), "entangling gates")
            )
    pair = knobs.entangling_pair(preset)
    return LightLayer(
        beams=tuple(beams),
        drives=tuple(drives),
        crosstalk=crosstalk,
        scattering_curve=_scattering_curve(device, 0, pair)
        if scattering_sweep and pair is not None
        else None,
        notes=tuple(notes),
    )


# ---- noise -----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrumRecord:
    name: str
    unit: str
    omega_rad_s: np.ndarray
    S: np.ndarray
    """Two-sided density on the plotting grid, ``NoiseSpectrum.value`` (tabulated plus the white level)."""
    white_level: float
    apparatus: tuple[str, ...]
    is_zero: bool


@dataclass(frozen=True)
class DriftRecord:
    name: str
    unit: str
    rms: float
    tau_s: float
    servo_bandwidth_hz: float | None


@dataclass(frozen=True)
class NoiseLayer:
    quiet: bool
    s_e: SpectrumRecord
    s_b: SpectrumRecord | None
    other_spectra: tuple[SpectrumRecord, ...]
    correlation_length_m: float
    heating_quanta_per_s: dict[int, float]
    motional_dephasing_tau_s: dict[int, float]
    qubit_white_dephasing_per_s: dict[int, float]
    drifts: tuple[DriftRecord, ...]
    mains: tuple[float, dict[int, float]] | None
    collisions: tuple[float, float] | None
    """(pressure in Pa, Langevin rate per ion in 1/s)."""
    apparatus: tuple[str, ...]
    provenance_sentence: str
    undeclared_rates: int
    notes: tuple[str, ...]


def _spectrum_record(name: str, spec: core.NoiseSpectrum) -> SpectrumRecord:
    grid = TWO_PI * np.logspace(2.0, 8.0, 121)
    return SpectrumRecord(
        name=name,
        unit=str(spec.unit),
        omega_rad_s=grid,
        S=np.asarray(spec.value(grid), dtype=float),
        white_level=float(spec.white_level),
        apparatus=tuple(str(p) for p in spec.provenance),
        is_zero=bool(spec.is_zero()),
    )


_DRIFT_UNITS = {
    "rf_amplitude_drift": "dV/V",
    "mode_drift_differential": "Hz",
    "rabi_drift": "dOmega/Omega",
    "beam_phase_drift": "rad",
    "field_drift": "T",
    "stray_field_drift": "V/m",
    "pointing_drift": "m",
    "laser_frequency_drift": "Hz",
}


def noise_layer(device: core.Device) -> NoiseLayer:
    noise = device.noise
    others = (
        ("laser_phase", noise.laser_phase),
        ("laser_intensity", noise.laser_intensity),
        ("rf_amplitude_noise", noise.rf_amplitude_noise),
        ("rf_phase_noise", noise.rf_phase_noise),
        ("rabi_amplitude", noise.rabi_amplitude),
        ("beam_phase_noise", noise.beam_phase_noise),
    )
    drifts = tuple(
        DriftRecord(
            name=str(name),
            unit=_DRIFT_UNITS.get(str(name), ""),
            rms=float(d.rms),
            tau_s=float(d.tau_s),
            servo_bandwidth_hz=None if d.servo_bandwidth_hz is None else float(d.servo_bandwidth_hz),
        )
        for name, d in sorted(noise.drifts.items())
    )
    mains = None
    if noise.mains is not None:
        mains = (float(noise.mains.line_hz), {int(k): float(v) for k, v in noise.mains.amplitudes_t.items()})
    collisions = None
    if noise.collisions is not None:
        rate = core.collision_rate_per_ion(noise.collisions, float(device.crystal.masses_kg[0]))
        collisions = (float(noise.collisions.pressure_pa), float(rate))
    heating = noise.heating_rates_quanta_per_s(device)
    notes: list[str] = []
    return NoiseLayer(
        quiet=bool(noise.is_quiet()),
        s_e=_spectrum_record("S_E", noise.S_E),
        s_b=None if noise.S_B is None else _spectrum_record("S_B", noise.S_B),
        other_spectra=tuple(_spectrum_record(name, spec) for name, spec in others if spec is not None),
        correlation_length_m=float(noise.correlation_length_m),
        heating_quanta_per_s={int(m): float(v) for m, v in heating.items()},
        motional_dephasing_tau_s={
            int(m): float(v) for m, v in noise.motional_dephasing_tau_s(device).items()
        },
        qubit_white_dephasing_per_s={
            int(i): float(v) for i, v in noise.qubit_white_dephasing_per_s(device).items()
        },
        drifts=drifts,
        mains=mains,
        collisions=collisions,
        apparatus=tuple(str(a) for a in noise.apparatus()),
        provenance_sentence=str(noise.provenance_sentence()),
        undeclared_rates=int(noise.undeclared_rate_count()),
        notes=tuple(notes),
    )


# ---- cooling -------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StageLayer:
    kind: str
    provenance: str
    nbar: dict[int, float] | None


@dataclass(frozen=True)
class DopplerModeLayer:
    mode: int
    omega_hz: float
    heating_per_s: float
    cooling_per_s: float
    rate_per_s: float
    nbar: float
    participation: float
    force_model_nbar: float | None


@dataclass(frozen=True)
class SidebandLayer:
    mode: int
    ion: int
    eta: float
    omega0_hz: float
    orders: tuple[int, ...]
    durations_s: tuple[float, ...]
    times_s: np.ndarray
    """(K + 1,) cumulative time at the end of each pulse-plus-repump, starting at 0."""
    nbar_after_pulse: np.ndarray
    """(K + 1,) <n> after each pulse, WITHOUT the repump recoil kernel; index 0 is the Doppler start."""
    nbar_start: float
    nbar_final_run: float
    """The run's own final n̄ for the mode, repump recoil included."""


@dataclass(frozen=True)
class PumpLayer:
    ion: int
    preparation_error: float
    photons_scattered: float
    time_to_reach_s: float | None
    motional_heating_quanta: dict[int, float]
    populations_end: dict[str, float]
    trace_times_s: np.ndarray
    trace_populations: dict[str, np.ndarray]


@dataclass(frozen=True)
class CoolingLayer:
    doppler_duration_s: float
    pump_duration_s: float
    repump_photons: float | None
    repump_time_s: float | None
    recipe_notes: tuple[str, ...]
    stages: tuple[StageLayer, ...]
    doppler: tuple[DopplerModeLayer, ...]
    doppler_method: str
    approximations: tuple[str, ...]
    sidebands: tuple[SidebandLayer, ...]
    pumps: tuple[PumpLayer, ...]
    final_nbar: dict[int, float]
    duration_s: float
    provenance: tuple[str, ...]
    notes: tuple[str, ...]


def _downsample(
    times: np.ndarray, values: Mapping[str, np.ndarray], n: int = 161
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t = np.asarray(times, dtype=float)
    if t.size <= n:
        return t, {k: np.asarray(v, dtype=float) for k, v in values.items()}
    idx = np.unique(np.linspace(0, t.size - 1, n).astype(int))
    return t[idx], {k: np.asarray(v, dtype=float)[idx] for k, v in values.items()}


def cooling_layer(preset: core.DevicePreset) -> CoolingLayer:
    device = preset.device
    recipe = device.preparation if device.preparation is not None else core.standard_recipe(device)
    prep = core.run_preparation(device, recipe)
    dop = prep.doppler
    doppler = tuple(
        DopplerModeLayer(
            mode=int(mr.mode),
            omega_hz=float(mr.omega_rad_s / TWO_PI),
            heating_per_s=float(mr.heating_per_s),
            cooling_per_s=float(mr.cooling_per_s),
            rate_per_s=float(mr.rate_per_s),
            nbar=float(mr.nbar),
            participation=float(mr.participation_weight),
            force_model_nbar=None
            if mr.mode not in dop.force_model_nbar
            else float(dop.force_model_nbar[mr.mode]),
        )
        for mr in dop.modes
    )
    sidebands: list[SidebandLayer] = []
    spec = recipe.sideband
    if spec is not None and prep.sideband_pulses:
        drives = {
            i: core.derive_raman_drive(device, i, spec.beams, scattering=False)
            for i in range(device.crystal.n_ions)
        }
        for m, pulses in sorted(prep.sideband_pulses.items()):
            ion = spec.ion if spec.ion is not None else max(drives, key=lambda i: abs(drives[i].etas[m]))
            eta = abs(float(drives[ion].etas[m]))
            omega0 = TWO_PI * float(drives[ion].carrier_rabi_hz)
            n0 = float(dop.nbar[m])
            d = int(min(spec.d_max, max(40, math.ceil(30.0 * n0 + 20.0))))
            p = core.thermal_distribution(n0, d)
            nbars = [float(core.mean_occupation(p))]
            times = [0.0]
            for pulse in pulses:
                p = core.apply_pulses(p, [pulse], eta, omega0)
                nbars.append(float(core.mean_occupation(p)))
                times.append(times[-1] + float(pulse.duration_s) + float(spec.repump_time_s))
            sidebands.append(
                SidebandLayer(
                    mode=int(m),
                    ion=int(ion),
                    eta=eta,
                    omega0_hz=omega0 / TWO_PI,
                    orders=tuple(int(pl.order) for pl in pulses),
                    durations_s=tuple(float(pl.duration_s) for pl in pulses),
                    times_s=np.asarray(times),
                    nbar_after_pulse=np.asarray(nbars),
                    nbar_start=n0,
                    nbar_final_run=float(prep.sideband_nbar.get(m, prep.nbar[m])),
                )
            )
    pumps: list[PumpLayer] = []
    for ion, pr in sorted(prep.pumps.items()):
        trace_t, pops = _downsample(
            np.asarray(pr.trace.times_s), {str(k): np.asarray(v) for k, v in pr.trace.populations.items()}
        )
        pumps.append(
            PumpLayer(
                ion=int(ion),
                preparation_error=float(pr.preparation_error),
                photons_scattered=float(pr.photons_scattered),
                time_to_reach_s=None if pr.time_to_reach_s is None else float(pr.time_to_reach_s),
                motional_heating_quanta={int(m): float(v) for m, v in pr.motional_heating_quanta.items()},
                populations_end={str(k): float(v) for k, v in pr.populations.items()},
                trace_times_s=trace_t,
                trace_populations=pops,
            )
        )
    return CoolingLayer(
        doppler_duration_s=float(recipe.doppler_duration_s),
        pump_duration_s=float(recipe.pump_duration_s),
        repump_photons=None if spec is None else float(spec.repump_photons),
        repump_time_s=None if spec is None else float(spec.repump_time_s),
        recipe_notes=tuple(str(n) for n in recipe.notes),
        stages=tuple(
            StageLayer(
                kind=str(st.kind),
                provenance=str(st.provenance),
                nbar=None if st.nbar is None else {int(m): float(v) for m, v in st.nbar.items()},
            )
            for st in prep.sequence.stages
        ),
        doppler=doppler,
        doppler_method=str(dop.method),
        approximations=tuple(str(a) for a in dop.approximations),
        sidebands=tuple(sidebands),
        pumps=tuple(pumps),
        final_nbar={int(m): float(v) for m, v in prep.nbar.items()},
        duration_s=float(prep.duration_s),
        provenance=tuple(str(p) for p in prep.provenance),
        notes=tuple(str(n) for n in prep.notes),
    )


# ---- readout -----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdScanLayer:
    windows_s: np.ndarray
    eps_b: np.ndarray
    eps_d: np.ndarray


@dataclass(frozen=True)
class SaturationCurve:
    s: np.ndarray
    r_bright_per_s: np.ndarray
    r_dark_pumping_per_s: np.ndarray
    r_bright_pumping_per_s: np.ndarray
    ceiling_per_s: float


@dataclass(frozen=True)
class ReadoutIonLayer:
    ion: int
    r_bright_per_s: float
    r_dark_pumping_per_s: float
    r_bright_pumping_per_s: float
    detected_bright_per_s: float
    background_per_s: float
    ceiling: float | None
    scheme_kind: str
    polarity: str
    saturation: float | None
    window_s: float
    bright_pmf: np.ndarray
    dark_pmf: np.ndarray
    scan: ThresholdScanLayer
    best_window_s: float
    best_n_c: float
    best_eps_b: float
    best_eps_d: float
    at_window_n_c: float
    at_window_eps_b: float
    at_window_eps_d: float
    budget: tuple[tuple[str, float, float], ...]
    """(channel, eps_B, eps_D) at the device window and its best threshold: discrimination alone, background, pumping."""
    saturation_curve: SaturationCurve | None
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class ReadoutLayer:
    detector_kind: str
    efficiency: float
    background_cps: float
    window_s: float
    numerical_aperture: float | None
    ions: tuple[ReadoutIonLayer, ...]
    notes: tuple[str, ...]


def readout_layer(preset: core.DevicePreset, *, saturation_sweep: bool = True) -> ReadoutLayer:
    device = preset.device
    det = device.detector
    pos = np.asarray(device.crystal.positions_m, dtype=float)
    ions: list[ReadoutIonLayer] = []
    notes: list[str] = []
    windows = np.unique(np.concatenate([np.geomspace(0.25, 2.5, 12) * det.window_s, [det.window_s]]))
    for i in range(device.crystal.n_ions):
        try:
            keys = core.detection_beams(device, i)
        except core.NoDetectionBeamError as exc:
            notes.append(f"ion {i}: {exc}; readout rates unavailable")
            continue
        sp = device.crystal.species[i]
        beams = [device.beams[k] for k in keys]
        rates, scheme, _model = core.detection_rates_for_ion(
            sp,
            device.field.B_gauss,
            device.field.direction,
            beams,
            position_m=tuple(float(x) for x in pos[i]),
        )
        rec_model = core.RecordModel.from_rates(rates, det)
        detected, background = rates.detected(det)
        pb = rec_model.count_distribution("bright", det.window_s)
        pd = rec_model.count_distribution("dark", det.window_s)
        opt = core.optimize_threshold(rec_model, [float(w) for w in windows])
        at_window = core.optimize_threshold(rec_model, [det.window_s]).best
        n_c_w, eb_w, ed_w = float(at_window.n_c), float(at_window.eps_B), float(at_window.eps_D)
        # budget: discrimination alone (no background, no pumping), then background, then pumping (Section 8.4)
        clean_rates = dataclasses.replace(rates, R_dark_pumping_per_s=0.0, R_bright_pumping_per_s=0.0)
        m_disc = core.RecordModel.from_rates(clean_rates, dataclasses.replace(det, background_cps=0.0))
        m_bg = core.RecordModel.from_rates(clean_rates, det)
        disc = core.ThresholdDiscriminator(n_c_w, det.window_s).error_rates(m_disc)
        bg = core.ThresholdDiscriminator(n_c_w, det.window_s).error_rates(m_bg)
        budget = (
            ("count overlap (discrimination)", float(disc[0]), float(disc[1])),
            ("background counts", float(max(bg[0] - disc[0], 0.0)), float(max(bg[1] - disc[1], 0.0))),
            ("off-resonant pumping", float(max(eb_w - bg[0], 0.0)), float(max(ed_w - bg[1], 0.0))),
        )
        cycling = sp.transition(sp.cycling)
        sat = None
        try:
            i_sat = float(cycling.i_sat_w_m2)
            main = min(beams, key=lambda b: abs(b.wavelength_m - cycling.wavelength_vac_m))
            sat = float(main.intensity_at(pos[i])) / i_sat
        except (ValueError, ZeroDivisionError):
            sat = None
        curve = None
        if saturation_sweep and sat is not None and sat > 0.0:
            grid = np.geomspace(0.05, 20.0, 13)
            ro, rd, rb = [], [], []
            for s_val in grid:
                scaled = [dataclasses.replace(b, power_w=b.power_w * float(s_val / sat)) for b in beams]
                r2, _s2, _m2 = core.detection_rates_for_ion(
                    sp,
                    device.field.B_gauss,
                    device.field.direction,
                    scaled,
                    position_m=tuple(float(x) for x in pos[i]),
                )
                ro.append(float(r2.R_bright_per_s))
                rd.append(float(r2.R_dark_pumping_per_s))
                rb.append(float(r2.R_bright_pumping_per_s))
            gamma = TWO_PI * float(cycling.gamma_hz) * float(cycling.branching)
            ceiling = float(rates.ceiling) if rates.ceiling is not None else 0.25
            curve = SaturationCurve(
                s=grid,
                r_bright_per_s=np.asarray(ro),
                r_dark_pumping_per_s=np.asarray(rd),
                r_bright_pumping_per_s=np.asarray(rb),
                ceiling_per_s=gamma * ceiling,
            )
        ions.append(
            ReadoutIonLayer(
                ion=i,
                r_bright_per_s=float(rates.R_bright_per_s),
                r_dark_pumping_per_s=float(rates.R_dark_pumping_per_s),
                r_bright_pumping_per_s=float(rates.R_bright_pumping_per_s),
                detected_bright_per_s=float(detected),
                background_per_s=float(background),
                ceiling=None if rates.ceiling is None else float(rates.ceiling),
                scheme_kind=str(scheme.kind),
                polarity=str(scheme.polarity),
                saturation=sat,
                window_s=float(det.window_s),
                bright_pmf=np.asarray(pb.pmf, dtype=float),
                dark_pmf=np.asarray(pd.pmf, dtype=float),
                scan=ThresholdScanLayer(
                    windows_s=np.asarray([p.window_s for p in opt.scan]),
                    eps_b=np.asarray([p.eps_B for p in opt.scan]),
                    eps_d=np.asarray([p.eps_D for p in opt.scan]),
                ),
                best_window_s=float(opt.best.window_s),
                best_n_c=float(opt.best.n_c),
                best_eps_b=float(opt.best.eps_B),
                best_eps_d=float(opt.best.eps_D),
                at_window_n_c=n_c_w,
                at_window_eps_b=eb_w,
                at_window_eps_d=ed_w,
                budget=budget,
                saturation_curve=curve,
                provenance=tuple(str(p) for p in rates.provenance),
            )
        )
    return ReadoutLayer(
        detector_kind=str(det.kind),
        efficiency=float(det.efficiency),
        background_cps=float(det.background_cps),
        window_s=float(det.window_s),
        numerical_aperture=None if det.numerical_aperture is None else float(det.numerical_aperture),
        ions=tuple(ions),
        notes=tuple(notes),
    )


# ---- pulse-solver solutions per pair (the analytic layer of Section 14.4) ----------------------------------------------------------


@dataclass(frozen=True)
class GateSolutionLayer:
    pair: tuple[int, int]
    modes: tuple[int, ...]
    omega_hz: tuple[float, ...]
    eta: dict[int, tuple[float, ...]]
    mu_hz: float
    duration_s: float
    method: str
    waveform: WaveformRecord
    residual_error: float | None
    """epsilon_ent = sum |alpha|^2 (2 nbar + 1) of the solution (Section 4.4.7 (8))."""
    peak_rabi_hz: float | None
    power_integral_rad2_s: float | None
    mu_rule: str
    error: str | None
    """Why no solution could be made, else None."""
    table_chi_m: dict[int, float] | None = None
    """The table's stored per-mode angles for this pair when the table was fitted for THIS device (the exact spot check's,
    Section 7.8); None for a stale layer, whose table belongs to another device."""
    table_chi_closed_form_m: dict[int, float] | None = None
    """The closed-form angles (Section 4.4.3) of the table's waveform at its played amplitude on this device's modes."""
    spot_check_amplitude_ratio: float | None = None
    """Omega_played / Omega_closed_form: how much the exact spot check rescaled the closed-form pulse (one factor for every
    segment: the check rescales, it does not reshape)."""
    surrogate_error: float | None = None
    """chi_table / chi_closed_form(at the played amplitude) - 1: what the closed form missed of the exact angle (Debye-Waller
    and beyond-Lamb-Dicke terms)."""


def _table_comparison(
    device: core.Device,
    pair: tuple[int, int],
    gm: core.GateModes,
    solved: core.Waveform,
    table: core.CalibrationTable | None,
    table_record: TableRecord | None,
) -> tuple[dict[int, float] | None, dict[int, float] | None, float | None, float | None]:
    """The layer's closed-form solution against the waveform the table plays for the same pair on the SAME device (Section
    7.8): the table's stored per-mode angles (the exact spot check's), the closed-form angles of the table's waveform at its
    played amplitude on this device's modes, the amplitude ratio played over closed-form, and the surrogate error
    chi_table / chi_closed(played) - 1. Nothing for a table fitted to another device, and only the stored angles when the
    app's table record is all there is."""
    stored: dict[int, float] | None = None
    closed: dict[int, float] | None = None
    ratio: float | None = None
    error: float | None = None
    h = device.hash()
    if table is not None and table.device_hash == h:
        wf_t = table.waveform_for(pair)
        if wf_t is None:
            return None, None, None, None
        stored = {int(m): float(v) for m, v in wf_t.chi_m.items()}
        env_t = core.envelope_of(wf_t, pair)
        env_s = core.envelope_of(solved, pair)
        if isinstance(env_t, core.SegmentedEnvelope) and isinstance(env_s, core.SegmentedEnvelope):
            by_mode = core.integrals_segmented(env_t, gm, "choi").chi_by_mode
            closed = {
                int(m): float(
                    by_mode.get((pair[0], pair[1], int(m)), by_mode.get((pair[1], pair[0], int(m)), 0.0))
                )
                for m in gm.modes
            }
            amp_t = np.concatenate([np.asarray(env_t.amplitude_rad_s[i], dtype=float) for i in pair])
            amp_s = np.concatenate([np.asarray(env_s.amplitude_rad_s[i], dtype=float) for i in pair])
            if float(np.dot(amp_s, amp_s)) > 0.0:
                ratio = math.sqrt(float(np.dot(amp_t, amp_t)) / float(np.dot(amp_s, amp_s)))
            total_closed = sum(closed.values())
            if total_closed != 0.0:
                error = sum(stored.values()) / total_closed - 1.0
    elif table_record is not None and table_record.device_hash == h:
        wf_r = table_record.waveforms.get(f"{pair[0]},{pair[1]}") or table_record.waveforms.get(
            f"{pair[1]},{pair[0]}"
        )
        if wf_r is not None:
            stored = {int(m): float(v) for m, v in wf_r.chi_m.items()}
    return stored, closed, ratio, error


def _pairs_for(preset: core.DevicePreset, table: core.CalibrationTable | None) -> list[tuple[int, int]]:
    if table is not None and table.ms:
        return sorted((int(a), int(b)) for a, b in table.ms)
    n = preset.device.crystal.n_ions
    return [(i, i + 1) for i in range(n - 1)]


def _unsolved(pair: tuple[int, int], duration_s: float, error: str) -> GateSolutionLayer:
    """The row of a pair without a solution: the reason, and an empty waveform."""
    blank = CalEntryRecord(0.0, "uncalibrated", "", "")
    waveform = WaveformRecord(0.0, {}, {}, 0.0, blank, blank, ())
    return GateSolutionLayer(pair, (), (), {}, 0.0, duration_s, "", waveform, None, None, None, "", error)


def gate_solutions(
    preset: core.DevicePreset,
    *,
    table: core.CalibrationTable | None,
    nbar: Mapping[int, float] | None,
    table_record: TableRecord | None = None,
) -> tuple[GateSolutionLayer, ...]:
    """The closed-form entangling pulse of every pair as the surrogate calibration solves it (``surrogate_waveform``,
    Section 7.8), at the table's duration for the pair, compared with the table's waveform when the table is this device's."""
    device = preset.device
    beams = knobs.entangling_pair(preset)
    if beams is None:
        return ()
    out: list[GateSolutionLayer] = []
    for pair in _pairs_for(preset, table):
        wf_t = None if table is None else table.waveform_for(pair)
        duration = DEFAULT_MS_DURATION_S if wf_t is None else float(wf_t.duration_s)
        try:
            shaped, gm = core.surrogate_waveform(device, pair, beams, nbar=nbar or {}, duration_s=duration)
        except ValueError as exc:  # no coupled mode, or loops the solver cannot close (ClosureError)
            out.append(_unsolved(pair, duration, f"the pulse solver could not close the loops: {exc}"))
            continue
        assert isinstance(shaped.envelope, core.SegmentedEnvelope), "the surrogate's pulses are segmented"
        diag = shaped.diagnostics
        mu_rule = (
            "single coupled mode: the symmetric square pulse detuned for one closed loop (Section 4.4.1)"
            if gm.n_modes == 1
            else f"beat note {core.MU_ABOVE_TOP_FRACTION:g} of the smallest coupled-mode gap above the highest "
            "coupled mode (Section 7.8)"
        )
        stored, closed, ratio, error = _table_comparison(
            device, pair, gm, shaped.waveform, table, table_record
        )
        out.append(
            GateSolutionLayer(
                pair=pair,
                modes=tuple(int(m) for m in gm.modes),
                omega_hz=tuple(float(w / TWO_PI) for w in gm.omega_rad_s),
                eta={int(i): tuple(float(e) for e in et) for i, et in gm.eta.items()},
                mu_hz=float(shaped.envelope.mu_rad_s / TWO_PI),
                duration_s=duration,
                method=str(shaped.method),
                waveform=waveform_record(shaped.waveform),
                residual_error=float(shaped.residual_error),
                peak_rabi_hz=None if "peak_rabi_hz" not in diag else float(diag["peak_rabi_hz"]),
                power_integral_rad2_s=None
                if "power_integral_rad2_s" not in diag
                else float(diag["power_integral_rad2_s"]),
                mu_rule=mu_rule,
                error=None,
                table_chi_m=stored,
                table_chi_closed_form_m=closed,
                spot_check_amplitude_ratio=ratio,
                surrogate_error=error,
            )
        )
    return tuple(out)


# ---- the layer -----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceLayer:
    """Everything the Level 4 pages read about one device, plus the Level 0 card's estimated columns (Section 14.4)."""

    device_hash: str
    overrides: dict[str, float]
    knob_values: dict[str, float]
    species: SpeciesLayer
    trap: TrapLayer
    crystal: CrystalLayer
    light: LightLayer
    noise: NoiseLayer
    cooling: CoolingLayer | None
    cooling_error: str | None
    readout: ReadoutLayer
    gates: tuple[GateSolutionLayer, ...]
    card: DeviceCard
    table_hash: str | None
    """The calibration table the layer was compared with (the current record's, or a recalibrated one), or None."""
    stale: bool
    """True when ``table_hash`` names a table made for another device hash: its calibrated numbers are stale (Section 14.4)."""
    notes: tuple[str, ...]
    wall_time_s: float


def derive_device_layer(
    preset: core.DevicePreset,
    *,
    overrides: Mapping[str, float] | None = None,
    table: core.CalibrationTable | None = None,
    table_record: TableRecord | None = None,
    sweeps: bool = True,
    progress: Progress | None = None,
) -> DeviceLayer:
    """The device layer of a built preset: the analytic re-derivation of Section 14.4 in one call."""
    t_all = time.perf_counter()
    device = preset.device
    notes: list[str] = []

    def stage(name: str, fraction: float) -> None:
        if progress is not None:
            progress("deriving", fraction, name)

    stage("species: levels, hyperfine and Zeeman structure", 0.05)
    species = species_layer(device)
    stage("trap: secular frequencies, Mathieu parameters, the stability boundary", 0.15)
    trap = trap_layer(device, device.derived().values)
    stage("crystal: equilibrium positions, normal modes, Lamb-Dicke parameters", 0.3)
    crystal = crystal_layer(preset)
    stage("light: Rabi frequencies, light shifts, scattering against detuning", 0.4)
    light = light_layer(preset, scattering_sweep=sweeps)
    stage("noise: spectra, heating and dephasing rates", 0.55)
    noise = noise_layer(device)
    stage("cooling: the preparation recipe by the rate and master equations", 0.65)
    cooling: CoolingLayer | None = None
    cooling_error: str | None = None
    try:
        cooling = cooling_layer(preset)
    except (
        Exception
    ) as exc:  # a recipe the device cannot run: shown as the reason, the other pages still derive
        cooling_error = f"{type(exc).__name__}: {exc}"
        notes.append(f"cooling: {cooling_error}")
    stage("readout: rates, count histograms, the threshold optimum", 0.8)
    readout = readout_layer(preset, saturation_sweep=sweeps)
    stage("pulse solver: the entangling waveforms of every pair", 0.9)
    gates = gate_solutions(
        preset, table=table, nbar=None if cooling is None else cooling.final_nbar, table_record=table_record
    )
    stage("device card", 0.95)
    spam: dict[str, tuple[float, float]] = {}
    for ion in readout.ions:
        spam[f"q{ion.ion}"] = (ion.at_window_eps_b, ion.at_window_eps_d)
    if cooling is not None:
        for pump in cooling.pumps:
            spam[f"q{pump.ion}.state_preparation"] = (pump.preparation_error, 0.0)
    # per gate, the closed-form error scales: an MS gate's residual displacement and its ions' scattering, a GPi2's scattering
    estimates: dict[str, float] = {}
    for g in gates:
        terms = [] if g.residual_error is None else [g.residual_error]
        terms += [
            d.error_per_pi_pulse * g.duration_s / d.pi_time_s if d.pi_time_s > 0 else 0.0
            for d in light.drives
            if d.role == "entangling gates" and d.ion in g.pair
        ]
        if terms:
            estimates[f"ms[{g.pair[0]},{g.pair[1]}]"] = float(sum(terms))
    for d in light.drives:
        if d.role == "single-qubit gates":
            estimates[f"gpi2[{d.ion}]"] = 0.5 * d.error_per_pi_pulse
    table_hash = (
        table.device_hash
        if table is not None
        else (table_record.device_hash if table_record is not None else None)
    )
    return DeviceLayer(
        device_hash=str(device.hash()),
        overrides={k: float(v) for k, v in (overrides or {}).items()},
        knob_values=knobs.current_values(preset, overrides),
        species=species,
        trap=trap,
        crystal=crystal,
        light=light,
        noise=noise,
        cooling=cooling,
        cooling_error=cooling_error,
        readout=readout,
        gates=gates,
        card=device_card(device, spam=spam, gate_estimates=estimates, tomography={}),
        table_hash=table_hash,
        stale=table_hash is not None and table_hash != device.hash(),
        notes=tuple(notes) + tuple(str(n) for n in preset.notes[-1:]) if overrides else tuple(notes),
        wall_time_s=time.perf_counter() - t_all,
    )
