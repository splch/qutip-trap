"""Micromotion compensation: scan the shims, null the excess micromotion, store shims and the residual beta (PLAN.md Sections
4.1.1, 4.3.6, 7.5; Berkeland et al. 1998; M8).

Three of Berkeland's methods, each computed from the physics the simulator already carries rather than from a signal model:

- ``rf_photon_correlation``: the detection beam's photon rate over one rf period. An ion whose excess micromotion has the
  signed amplitude u_1 along the beam sees the laser phase modulated as k . u(t) = beta cos(Omega t), i.e. the beam frequency
  shifted by beta Omega sin(Omega t); the multi-level Bloch model of the beam (``light/bloch.py``) is propagated with that
  time-periodic detuning to its periodic steady state. The measurement is made with the beam retuned to Berkeland's working
  point, half a linewidth to the red (``correlation_detuning_gammas``, default -0.5), where the rate's slope in detuning is
  largest: on resonance the first-order response vanishes and only the second harmonic (even in beta) survives, a thousand
  times smaller on the two-ion fixture (|S_1| = 3.2e-5 at beta = 0.045 against 3.4e-2 at -Gamma/2). The photon rate's first
  harmonic at the rf frequency, normalized by the mean rate, is a complex number whose phase is the atom's response lag
  (141 degrees at Omega_rf = 1.5 Gamma; a fixed sin projection would keep 63 % of it): the correlation signal S is its
  projection on the response phase the model gives at a reference beta, so S is odd in beta with the full amplitude, a shim
  scan crosses zero at the compensated setting (a line fit gives the null and its uncertainty), and the residual beta is read
  back from S through the same model. Photon shot noise on ``acquisition_s`` of counts sets the statistical uncertainty.
- ``doppler_nulling``: the same periodic steady state's MEAN photon rate, even in beta (the Doppler sidebands at
  +-Omega_rf take weight from the carrier), extremal at the null: a parabola fit per shim direction.
- ``sideband_ratio``: the gate drive rf-locked with the builder's exact modulation e^{i beta cos(Omega_rf t + delta)}
  (``BuilderOptions(micromotion="modulated")``): the first micromotion sideband at detuning Omega_rf/2 pi flops at
  J_1(beta) Omega against the carrier's J_0(beta) Omega, so the sideband excitation after a fixed pulse is even in beta and its
  ratio to the carrier rate inverts J_1/J_0 for |beta| (Section 4.1.1, "J_1^2/J_0^2 ~ (beta/2)^2").

Shims: on the geometry path the scan variables are the named shim electrodes' voltages (``Trap.shim_voltages_v``); on the
explicit-frequency path, which has no electrode model (``Trap`` refuses non-zero shim voltages there), the names ``Ex``,
``Ey``, ``Ez`` are the components of the compensation FIELD in V/m added to the stray field, the equivalent knob (Appendix E
lists the scan as ``shim_ranges_v``; the unit follows the path and is recorded in the notes). A trial device re-solves its
crystal at every setting, so the ions' displacement, the beams' intensity at the ions and the Lamb-Dicke parameters all move
together. A trap without an rf record has no micromotion to compensate (C0 = 1, beta = 0) and the scan returns exact zeros
with no measurement behind them, which ``calibration.experiments`` stores as ``seed`` entries, not ``calibrated`` ones
(Section 7.5: an entry carries the experiment that produced it, and none ran here).

The calibrated ``shim[...]`` entries are what the machine PROGRAMS: ``run()`` applies them to the device it evolves through
``device_with_compensation``, so a compensation fitted at one time against a stray field that has since drifted leaves the
residual excess micromotion a laboratory would have (the loop of Section 7.5). ``signed_beta`` and ``MicromotionIndex``
carry the sign of the index, so the correlation signal's pi step across the null is preserved; the fitted null itself is a
zero crossing and does not depend on it.
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
from qutip_trap.machine import laboratory_kwargs

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
    """(times over one rf period, total photon rate) of the periodic steady state of ``model`` with beam ``beam_index``'s
    frequency modulated as ``amplitude`` sin(Omega_rf t): the fixed point of the one-period propagator (as ``steadystate`` finds
    a Floquet fixed point) followed by one period of ``mesolve``."""
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
    """k . u_1: the signed modulation index along ``k_vector`` (peak), 0 without an rf record.

    The sign is the plan's, not a free choice: u_1 = -(1/2) Q u_0 under the adopted Mathieu origin a - 2q cos 2xi
    (Section 13, "Floquet function and rf phase origin"), so sign(beta) = -sign(q_x E_x) and beta steps by pi as a
    shim crosses the compensated value. The rf-photon correlation signal is odd in beta, so that pi step is what the
    servo reads (Section 9.17); the fitted null is a zero crossing and does not depend on it.
    """
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
        # Berkeland's working point: the beam retuned half a linewidth to the red of the detection transition for this
        # measurement (the lab's AOM), where the rate's slope in detuning is largest; on resonance the first-order response
        # vanishes and the first harmonic is a thousand times smaller (module docstring)
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
        """The atom's response phase: arg S_1 at a positive reference beta (the lab calibrates it with a deliberate offset)."""
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
    """|beta| from J_1(beta)/J_0(beta) = ratio on (0, 1.8) (the first sideband over the carrier, Section 4.1.1)."""
    if ratio <= 0.0:
        return 0.0

    def f(b: float) -> float:
        return float(jv(1, b) / jv(0, b)) - ratio

    hi = 1.8
    if f(hi) < 0.0:
        return hi
    return float(brentq(f, 1e-12, hi))


def micromotion_scan(
    machine: Machine | Device,
    ion: int,
    beam: int,
    shim_ranges_v: Mapping[str, tuple[float, float]],
    method: str = "rf_photon_correlation",
    **kw: Any,
) -> ExperimentResult:
    """Scan the shims over ``shim_ranges_v`` (name -> (low, high)), null the excess micromotion along ``beam`` by ``method`` and
    return the compensated shim settings and the residual beta (Section 7.5).

    ``points`` per shim (default 5), ``passes`` over the shim set (default 1), ``acquisition_s`` of photons per point for the
    correlation and Doppler methods (default 1 s, sets the shot-noise uncertainty), ``gate_drive`` and ``sideband_duration_s``
    for the sideband-ratio method (``beam`` then names the drive's table-key beam). Fitted: ``shim[name]`` (the null and its
    uncertainty), ``beta[beam]`` (the residual index along the beam at the null, peak convention), ``beta_before``; data rows
    (pass, shim index, setting, signal, sigma). A trap without an rf record returns exact zeros with no scan behind them
    (the caller stores them as seeds, not as a measurement).
    """
    device, kw = laboratory_kwargs(machine, kw, caller=micromotion_scan)
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
            # the Doppler method's mean rate is even in beta: the residual is read from the parabola's curvature
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

    gate_drive = kw.get("gate_drive") or default_gate_drives(device)[ion]
    if gate_drive.kind not in ("raman", "optical_E1", "optical_E2"):
        raise ValueError("the sideband-ratio method drives the ion with laser light")
    f_rf = float(device.trap.rf.frequency_hz)
    bopts = BuilderOptions(micromotion="modulated")
    base_kw = {
        **kw,
        "gate_drive": gate_drive,
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

    def carrier_rate(trial: Device) -> tuple[float, float]:
        ts = [float(x) for x in np.linspace(0.0, 2.0 / omega, 9)]
        res = rabi_scan(trial, ion, ts, **{**base_kw, "detuning_hz": 0.0})
        return res.fitted["f_rabi_hz"]

    def sideband_excitation(trial: Device, index: int) -> tuple[float, float | None]:
        from qutip_trap.control.pulses import Pulse

        setup = _setup(trial, ion, {**base_kw, "detuning_hz": f_rf})
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
