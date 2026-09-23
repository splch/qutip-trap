"""Micromotion compensation by three of Berkeland et al. 1998's methods: scan the shims, null the excess micromotion.

``rf_photon_correlation`` reads the detection photon rate's first harmonic at the rf frequency (odd in beta),
``doppler_nulling`` its mean (even in beta) and ``sideband_ratio`` the first micromotion sideband against the carrier.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import qutip as qt
from scipy.optimize import brentq
from scipy.special import jv

from qutip_trap.experiments.fitting import at_scan_edge, weighted_fit
from qutip_trap.experiments.result import ExperimentResult, MicromotionScan, ScanParameters
from qutip_trap.experiments.single_ion import _observation
from qutip_trap.machine import Machine, laboratory_kwargs

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.light.bloch import BlochModel
    from qutip_trap.machine import Machine

METHODS = ("rf_photon_correlation", "doppler_nulling", "sideband_ratio")
FIELD_SHIMS = ("Ex", "Ey", "Ez")
_OPTIONS = {"method": "dop853", "atol": 1e-10, "rtol": 1e-8, "nsteps": 10**7, "progress_bar": ""}


def _sine(t: float, amplitude: float, omega: float, **_: object) -> float:
    return float(amplitude * math.sin(omega * t))


def periodic_scattering(
    model: BlochModel, beam_index: int, amplitude_rad_s: float, omega_rf_rad_s: float, *, n_points: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """(times over one rf period, total photon rate) in the periodic steady state of ``model`` with beam ``beam_index``'s
    frequency modulated as ``amplitude`` sin(Omega_rf t): the one-period propagator's fixed point, then one period."""
    b = model.build
    if not b.static or b.space is not None:
        raise NotImplementedError(
            "the rf-modulated periodic steady state needs a static internal-only Bloch model"
        )
    assert isinstance(b.H, qt.Qobj)
    h0 = b.H
    c_ops = list(b.c_ops)
    period = 2.0 * math.pi / omega_rf_rad_s
    times = np.linspace(0.0, period, n_points + 1)
    if amplitude_rad_s == 0.0:
        rho = qt.steadystate(h0, c_ops, method="direct")
        rate = float(np.sum(model.operator_rates(rho)))
        return times, np.full(times.shape, rate)
    shifted = model.shifted(beam_index, 1.0).build.H
    assert isinstance(shifted, qt.Qobj)
    d_op = shifted - h0  # the Hamiltonian change per rad/s of beam frequency offset (linear: a frame shift)
    l0 = qt.liouvillian(h0, c_ops)
    ld = qt.liouvillian(d_op)
    lt = qt.QobjEvo(
        [l0, [ld, qt.coefficient(_sine, args={"amplitude": amplitude_rad_s, "omega": omega_rf_rad_s})]]
    )
    prop = qt.propagator(lt, period, options=_OPTIONS)
    mat = np.asarray(prop.full())
    vals, vecs = np.linalg.eig(mat)
    k = int(np.argmin(np.abs(vals - 1.0)))
    if abs(vals[k] - 1.0) > 1e-5:
        raise RuntimeError(f"no fixed point of the rf-period propagator (closest eigenvalue {vals[k]})")
    n = b.n_internal
    rho0 = vecs[:, k].reshape(n, n, order="F")
    rho0 = 0.5 * (rho0 + rho0.conj().T)
    rho0 = rho0 / np.trace(rho0)
    res = qt.mesolve(lt, qt.Qobj(rho0, dims=h0.dims), times, options={**_OPTIONS, "store_states": True})
    rates = np.array([float(np.sum(model.operator_rates(st))) for st in res.states])
    return times, rates


def correlation_signal(times: np.ndarray, rates: np.ndarray, omega_rf_rad_s: float) -> tuple[complex, float]:
    """(S_1, mean rate): the complex first harmonic of the photon rate at the rf frequency, (2/T) int r(t) e^{-i Omega t} dt,
    normalized by the mean rate; its modulus is the modulation depth and its phase the atom's response lag behind the rf."""
    t = np.asarray(times)
    r = np.asarray(rates)
    period = float(t[-1] - t[0])
    mean = float(np.trapezoid(r, t) / period)
    s = complex(2.0 / period * np.trapezoid(r * np.exp(-1j * omega_rf_rad_s * t), t))
    return (s / mean if mean > 0.0 else 0j), mean


def device_with_compensation(device: Device, shims: Mapping[str, float]) -> Device:
    """The device with the shims set: voltages on the geometry path, compensation-field components (V/m) on the explicit path;
    the crystal is re-solved so that the ions' displacement follows."""
    from qutip_trap.trap.crystal import solve_crystal

    trap = device.trap
    if trap.path == "explicit":
        unknown = [k for k in shims if k not in FIELD_SHIMS]
        if unknown:
            raise ValueError(
                f"the explicit-frequency trap has no electrode model; its shims are the compensation-field components {FIELD_SHIMS}, "
                f"not {unknown}"
            )
        comp = np.array([float(shims.get(k, 0.0)) for k in FIELD_SHIMS])
        stray = np.asarray(trap.stray_field_v_per_m, dtype=float) + comp
        new_trap = replace(trap, stray_field_v_per_m=(float(stray[0]), float(stray[1]), float(stray[2])))
    else:
        new_trap = replace(
            trap, shim_voltages_v={**trap.shim_voltages_v, **{k: float(v) for k, v in shims.items()}}
        )
    crystal = solve_crystal(new_trap, device.crystal.species)
    return replace(device, trap=new_trap, crystal=crystal)


def signed_beta(device: Device, ion: int, k_vector: np.ndarray) -> float:
    """k . u_1, the signed peak modulation index along ``k_vector`` (0 without an rf record); with u_1 = -(1/2) Q u_0 under
    the Mathieu origin a - 2q cos 2 xi, sign(beta) = -sign(q_x E_x), so beta changes sign as a shim crosses the null."""
    if device.trap.rf is None:
        return 0.0
    amp = device.trap.micromotion_amplitude_m(device.crystal.species[ion])
    return float(np.dot(np.asarray(k_vector, dtype=float), amp))


class _Correlation:
    """The rf-photon-correlation and Doppler-nulling observables of the detection beam on one ion."""

    def __init__(self, device: Device, ion: int, beam: int, kw: dict[str, Any]) -> None:
        from qutip_trap.light.roles import detection_beams
        from qutip_trap.readout.fluorescence import detection_rates_for_ion

        self.device = device
        self.ion = ion
        self.beam = beam
        idx = detection_beams(device, ion)
        if beam not in idx:
            raise ValueError(f"beam {beam} is not a detection beam of ion {ion} (detection beams: {idx})")
        self.position = beam
        self.idx = idx
        self.kw = kw
        self.n_points = int(kw.get("rf_points", 32))
        _rates, _scheme, self.model = detection_rates_for_ion(
            device.crystal.species[ion],
            device.field.B_gauss,
            device.field.direction,
            [device.beams[k] for k in idx],
            position_m=tuple(float(x) for x in device.crystal.positions_m[ion]),
        )
        self.omega_rf = float(device.trap.rf.omega_rad_s) if device.trap.rf is not None else 0.0
        self.k_vec = np.asarray(device.beams[beam].k_vector(), dtype=float)
        # Berkeland's working point, half a linewidth to the red: on resonance the first-order response vanishes
        gamma = max(self.model.build.level_rates_rad_s.values())
        self.detuning_rad_s = float(kw.get("correlation_detuning_gammas", -0.5)) * gamma
        if self.detuning_rad_s != 0.0:
            self.model = self.model.shifted(self.idx.index(self.beam), self.detuning_rad_s)
        self._cache: dict[float, tuple[complex, float]] = {}
        self._reference_phase: float | None = None

    def harmonic(self, beta: float) -> tuple[complex, float]:
        """(complex first harmonic S_1, mean rate) at the signed modulation index ``beta`` (cached per beta)."""
        key = round(beta, 12)
        if key not in self._cache:
            times, rates = periodic_scattering(
                self.model,
                self.idx.index(self.beam),
                beta * self.omega_rf,
                self.omega_rf,
                n_points=self.n_points,
            )
            self._cache[key] = correlation_signal(times, rates, self.omega_rf)
        return self._cache[key]

    def reference_phase(self, beta_ref: float = 0.05) -> float:
        """The atom's response phase, arg S_1 at a positive reference beta."""
        if self._reference_phase is None:
            s_ref, _ = self.harmonic(beta_ref)
            self._reference_phase = float(np.angle(s_ref)) if abs(s_ref) > 0.0 else 0.0
        return self._reference_phase

    def observables(self, beta: float) -> tuple[float, float]:
        """(S, mean rate) at ``beta``: the first harmonic projected on the response phase, odd in beta with its full amplitude."""
        s1, mean = self.harmonic(beta)
        s = float(np.real(s1 * np.exp(-1j * self.reference_phase())))
        return s, mean

    def for_device(self, trial: Device) -> tuple[float, float, float]:
        """(S, mean rate, beta) of ``trial`` (the same beams, the ion at its new displacement)."""
        beta = signed_beta(trial, self.ion, self.k_vec)
        s, mean = self.observables(beta)
        return s, mean, beta

    def slope(self, beta_ref: float = 0.05) -> float:
        """dS/d beta near the null, from the model at beta_ref."""
        s_plus, _ = self.observables(beta_ref)
        s_minus, _ = self.observables(-beta_ref)
        return (s_plus - s_minus) / (2.0 * beta_ref)


def _invert_j1_over_j0(ratio: float) -> float:
    """|beta| from J_1(beta)/J_0(beta) = ratio on (0, 1.8), the first sideband's rate over the carrier's."""
    if ratio <= 0.0:
        return 0.0

    def f(b: float) -> float:
        return float(jv(1, b) / jv(0, b)) - ratio

    hi = 1.8
    if f(hi) < 0.0:
        return hi
    return float(brentq(f, 1e-12, hi))


def micromotion_scan(
    machine: Machine,
    ion: int,
    beam: int,
    shim_ranges_v: Mapping[str, tuple[float, float]],
    method: str = "rf_photon_correlation",
    **kw: Any,
) -> ExperimentResult:
    """Scan the shims over ``shim_ranges_v`` (name -> (low, high)) to null the excess micromotion along ``beam`` by
    ``method``; returns a ``MicromotionScan`` with shim[name] and the residual peak index beta[beam].

    ``points`` per shim (default 5), ``passes`` (default 1), ``acquisition_s`` of photons per point (default 1 s, sets
    the shot noise), ``gate_drive`` and ``sideband_duration_s`` for ``sideband_ratio`` (``beam`` then names the drive's
    table-key beam). Data rows (pass, shim index, setting, signal, sigma). Without an rf record: exact zeros, no scan.
    """
    device, kw = laboratory_kwargs(machine, kw)
    if method not in METHODS:
        raise ValueError(f"method is one of {METHODS}")
    obs = _observation(device, kw)
    n_points = int(kw.get("points", 5))
    passes = int(kw.get("passes", 1))
    names = list(shim_ranges_v)
    unit = "V/m (compensation field, explicit-frequency path)" if device.trap.path == "explicit" else "V"
    if device.trap.rf is None:
        fitted = {f"shim[{name}]": (0.0, 0.0) for name in names}
        fitted[f"beta[{beam}]"] = (0.0, 0.0)
        fitted["beta_before"] = (0.0, 0.0)
        return MicromotionScan(
            data=np.zeros((0, 5)),
            fitted=fitted,
            model=f"micromotion_{method}",
            provenance_id="anchor.trap.berkeland_excess_micromotion",
            notes=("no rf record on the trap: excess micromotion is not modelled (C0 = 1, beta = 0)",),
            requested=ScanParameters({f"shim[{n}]_v": tuple(shim_ranges_v[n]) for n in names}),
            subject={"ion": int(ion), "beam": int(beam)},
        )
    shims: dict[str, float] = {}
    if device.trap.path == "explicit":
        shims = {k: 0.0 for k in names}
    else:
        shims = {k: float(device.trap.shim_voltages_v.get(k, 0.0)) for k in names}
    rows: list[tuple[float, float, float, float, float]] = []
    notes: list[str] = [f"shim unit: {unit}"]
    converged = True

    if method in ("rf_photon_correlation", "doppler_nulling"):
        corr = _Correlation(device, ion, beam, kw)
        acquisition = float(kw.get("acquisition_s", 1.0))
        eff = float(device.detector.efficiency)

        def measure(trial: Device, index: int) -> tuple[float, float, float]:
            s, mean, beta = corr.for_device(trial)
            n_photons = max(mean * eff * acquisition, 1.0)
            if method == "rf_photon_correlation":
                sigma = math.sqrt(2.0 / n_photons)
                return s + obs.noise(sigma, "micromotion_correlation", index, ion), sigma, beta
            sigma = mean / math.sqrt(n_photons)
            return mean + obs.noise(sigma, "micromotion_doppler", index, ion), sigma, beta

        beta0 = signed_beta(device, ion, corr.k_vec)
        index = 0
        null_fits: dict[str, tuple[float, float]] = {}
        for p in range(passes):
            for j, name in enumerate(names):
                lo, hi = (float(x) for x in shim_ranges_v[name])
                grid = np.linspace(lo, hi, n_points)
                sig: list[float] = []
                sgs: list[float] = []
                for v in grid:
                    trial = device_with_compensation(device, {**shims, name: float(v)})
                    s, sg, _b = measure(trial, index)
                    index += 1
                    sig.append(s)
                    sgs.append(sg)
                    rows.append((float(p), float(j), float(v), s, sg))
                y = np.array(sig)
                sigma = np.array(sgs) if all(x > 0.0 for x in sgs) else None
                if method == "rf_photon_correlation":
                    fit = weighted_fit(
                        lambda q, x: float(q[0]) * (np.asarray(x) - float(q[1])),
                        [(y[-1] - y[0]) / max(hi - lo, 1e-300), 0.5 * (lo + hi)],
                        grid,
                        y,
                        sigma=sigma,
                    )
                    v0, s_v0 = fit.value(1)
                    ok = fit.converged and abs(float(fit.params[0])) > 0.0
                else:
                    fit = weighted_fit(
                        lambda q, x: float(q[0]) * (np.asarray(x) - float(q[1])) ** 2 + float(q[2]),
                        [
                            -(y.max() - y.min()) / max((hi - lo) ** 2 / 4.0, 1e-300),
                            float(grid[int(np.argmax(y))]),
                            float(y.max()),
                        ],
                        grid,
                        y,
                        sigma=sigma,
                    )
                    v0, s_v0 = fit.value(1)
                    ok = fit.converged and float(fit.params[0]) < 0.0
                if not ok or at_scan_edge(v0, lo, hi, 0.0):
                    converged = False
                    notes.append(f"shim {name!r}: the null fit failed or lies outside the scanned range")
                    v0 = float(min(max(v0, lo), hi))
                shims[name] = v0
                null_fits[name] = (v0, s_v0)
                if p == passes - 1:
                    notes.append(f"shim {name!r} null {v0:.6g} +- {s_v0:.3g} {unit}")
        final = device_with_compensation(device, shims)
        s_final, sg_final, _beta_true = measure(final, index)
        slope = corr.slope() if method == "rf_photon_correlation" else math.nan
        if method == "rf_photon_correlation" and abs(slope) > 0.0:
            beta_res, s_beta = s_final / slope, sg_final / abs(slope)
        else:
            # the Doppler method's mean rate is even in beta: it gives no signed residual
            beta_res, s_beta = 0.0, math.nan
            notes.append(
                "the Doppler-nulling method reports the null; the residual |beta| needs the correlation or sideband method"
            )
        fitted = {f"shim[{n}]": null_fits[n] for n in names}
        fitted[f"beta[{beam}]"] = (beta_res, s_beta)
        fitted["beta_before"] = (beta0, 0.0)
        return MicromotionScan(
            data=np.array(rows),
            fitted=fitted,
            model=f"micromotion_{method}",
            provenance_id="anchor.trap.berkeland_excess_micromotion",
            converged=converged,
            notes=tuple(notes),
            requested=ScanParameters({f"shim[{n}]_v": tuple(shim_ranges_v[n]) for n in names}),
            subject={"ion": int(ion), "beam": int(beam)},
        )

    # ---- the sideband-ratio method through the exact modulated builder --------------------------------------------------------
    from qutip_trap.control.schedule import default_gate_drives
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.experiments.single_ion import _run, _setup, rabi_scan
    from qutip_trap.light.raman import derive_optical_drive, derive_raman_drive

    gate_drive = default_gate_drives(device)[ion]
    if gate_drive.kind not in ("raman", "optical_E1", "optical_E2"):
        raise ValueError("the sideband-ratio method drives the ion with laser light")
    f_rf = float(device.trap.rf.frequency_hz)
    bopts = BuilderOptions(micromotion="modulated")
    base_kw = {
        **kw,
        "rf_locked": True,
        "rf_phase_rad": 0.0,
        "builder_options": bopts,
        "include_stark": False,
    }
    if gate_drive.kind == "raman":
        derived = derive_raman_drive(
            device, ion, (gate_drive.beams[0], gate_drive.beams[1]), scattering=False
        )
    else:
        derived = derive_optical_drive(device, ion, gate_drive.beams[0], scattering=False)
    omega = derived.carrier_rabi_hz
    k_vec = np.asarray(derived.delta_k, dtype=float)
    beta0 = signed_beta(device, ion, k_vec)
    t_sb = float(kw.get("sideband_duration_s") or 1.0 / (4.0 * float(jv(1, 0.2)) * omega))

    scan_kw = base_kw

    def with_role(trial: Device) -> Device:
        gate = dict(trial.roles.gate) if trial.roles.gate is not None else default_gate_drives(trial)
        return replace(trial, roles=replace(trial.roles, gate={**gate, ion: gate_drive}))

    def carrier_rate(trial: Device) -> tuple[float, float]:
        ts = [float(x) for x in np.linspace(0.0, 2.0 / omega, 9)]
        res = rabi_scan(Machine(with_role(trial)), ion, ts, **{**scan_kw, "detuning_hz": 0.0})
        return res.fitted["f_rabi_hz"]

    def sideband_excitation(trial: Device, index: int) -> tuple[float, float | None]:
        from qutip_trap.control.pulses import Pulse

        setup = _setup(with_role(trial), ion, {**base_kw, "detuning_hz": f_rf})
        pulse = Pulse(setup.drive, 0.0, t_sb, "micromotion_sideband", ())
        avg = _run(trial, ion, [pulse], setup, base_kw)
        return obs.p1(avg.final_p1(ion), ion, "micromotion_sideband", index)

    index = 0
    fits: dict[str, tuple[float, float]] = {}
    for p in range(passes):
        for j, name in enumerate(names):
            lo, hi = (float(x) for x in shim_ranges_v[name])
            grid = np.linspace(lo, hi, n_points)
            ys: list[float] = []
            sgs2: list[float | None] = []
            for v in grid:
                trial = device_with_compensation(device, {**shims, name: float(v)})
                pv, sv = sideband_excitation(trial, index)
                index += 1
                ys.append(pv)
                sgs2.append(sv)
                rows.append((float(p), float(j), float(v), pv, 0.0 if sv is None else sv))
            y = np.array(ys)
            sigma = None if any(s is None for s in sgs2) else np.array([s for s in sgs2 if s is not None])
            fit = weighted_fit(
                lambda q, x: float(q[0]) * (np.asarray(x) - float(q[1])) ** 2 + float(q[2]),
                [
                    (y.max() - y.min()) / max((hi - lo) ** 2 / 4.0, 1e-300),
                    float(grid[int(np.argmin(y))]),
                    float(y.min()),
                ],
                grid,
                y,
                sigma=sigma,
                bounds=([0.0, lo - (hi - lo), 0.0], [np.inf, hi + (hi - lo), 1.0]),
            )
            v0, s_v0 = fit.value(1)
            if not fit.converged or at_scan_edge(v0, lo, hi, 0.0):
                converged = False
                notes.append(
                    f"shim {name!r}: the sideband parabola did not converge or its minimum lies outside the range"
                )
                v0 = float(min(max(v0, lo), hi))
            shims[name] = v0
            fits[name] = (v0, s_v0)
    final = device_with_compensation(device, shims)
    f0, s_f0 = carrier_rate(final)
    p_sb, s_sb = sideband_excitation(final, index)
    # P = sin^2(pi f_1 t): the sideband rate, then |beta| from J_1/J_0 = f_1/f_0
    p_c = min(max(p_sb, 0.0), 1.0)
    f1 = math.asin(math.sqrt(p_c)) / (math.pi * t_sb)
    ratio = f1 / f0 if f0 > 0.0 else 0.0
    beta_res = _invert_j1_over_j0(ratio)
    # the uncertainty from the excitation's: d f1/d P = 1/(2 pi t sqrt(P (1 - P)))
    s_p = s_sb if s_sb is not None else 0.0
    dfdp = 1.0 / (2.0 * math.pi * t_sb * math.sqrt(max(p_c * (1.0 - p_c), 1e-12)))
    s_f1 = dfdp * s_p
    # d beta/d ratio near zero: J_1(b)/J_0(b) ~ b/2 -> 2
    s_beta = 2.0 * math.hypot(s_f1 / max(f0, 1e-300), f1 * s_f0 / max(f0, 1e-300) ** 2)
    fitted = {f"shim[{n}]": fits[n] for n in names}
    fitted[f"beta[{gate_drive.table_key_beam}]"] = (beta_res, s_beta)
    fitted["beta_before"] = (abs(beta0), 0.0)
    fitted["carrier_hz"] = (f0, s_f0)
    fitted["sideband_excitation"] = (p_sb, s_p)
    return MicromotionScan(
        data=np.array(rows),
        fitted=fitted,
        model="micromotion_sideband_ratio",
        provenance_id="anchor.trap.berkeland_excess_micromotion",
        converged=converged,
        notes=tuple(notes),
        requested=ScanParameters({f"shim[{n}]_v": tuple(shim_ranges_v[n]) for n in names}),
        subject={"ion": int(ion), "beam": int(beam)},
    )


__all__ = [
    "FIELD_SHIMS",
    "METHODS",
    "correlation_signal",
    "device_with_compensation",
    "micromotion_scan",
    "periodic_scattering",
    "signed_beta",
]
