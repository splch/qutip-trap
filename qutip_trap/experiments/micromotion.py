"""Micromotion compensation: scan the shims, null the excess micromotion, report the shims and the residual beta (PLAN.md
Section 7.5; Berkeland et al. 1998).

Two of Berkeland's methods, each computed from the physics the simulator carries rather than from a signal model:

- ``rf_photon_correlation``: an ion whose excess micromotion has the signed amplitude u_1 along the detection beam sees
  the laser phase modulated as k . u(t) = beta cos(Omega t), i.e. the beam frequency shifted by beta Omega sin(Omega t);
  the beam's multi-level Bloch model (``light/bloch.py``) is propagated with that detuning to its periodic steady state,
  with the beam retuned half a linewidth to the red, where the rate's slope in detuning is largest (on resonance the
  first-order response vanishes). The rate's first harmonic at the rf frequency over the mean rate is complex, its phase
  the atom's response lag; projected on the response phase at a reference beta it is the signal S, odd in beta at full
  amplitude, so a shim scan crosses zero at the compensated setting (a line fit gives the null) and the residual beta is
  read back through the same model. Photon shot noise over one second per point sets the uncertainty.
- ``sideband_ratio``: the gate drive rf-locked with the builder's exact modulation e^{i beta cos(Omega_rf t + delta)}
  (``BuilderOptions(micromotion="modulated")``): the first micromotion sideband flops at J_1(beta) Omega against the
  carrier's J_0(beta) Omega, so the sideband excitation after a fixed pulse is even in beta (a parabola in the shim) and its
  ratio to the carrier rate inverts J_1/J_0 for |beta|.

Berkeland's Doppler nulling (the mean rate, even in beta and flat at the null) is not offered: it returns no residual beta,
and on the two-ion fixture with one second of photons per point and the six steady-state solves of a three-point
correlation scan it found no null in 3 of 12 seeded scans (the correlation scan in none), its other nulls scattering by
6.2 V/m against the correlation scan's 5.1.

The shims are the named electrode voltages on the geometry path and, on the explicit-frequency path (no electrode model),
the compensation-field components Ex, Ey, Ez in V/m added to the stray field; every trial device re-solves its crystal. A
trap without an rf record has no micromotion (C0 = 1, beta = 0): the scan returns exact zeros with no measurement behind
them. ``run`` programs calibrated shims onto the device it evolves through ``device_with_compensation``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Unpack

import numpy as np
import qutip as qt
from scipy.optimize import brentq
from scipy.special import jv

from qutip_trap.experiments.fitting import _detection_rates, at_scan_edge, sigmas_or_none, weighted_fit
from qutip_trap.experiments.result import ExperimentResult, MicromotionScan, ScanParameters
from qutip_trap.experiments.single_ion import _Lab, _LabOptions, _Probe, _run, _setup, rabi_scan

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.light.bloch import BlochModel
    from qutip_trap.machine import Machine

    _Signal = Callable[[Device, int], tuple[float, float | None]]
    """A nulling signal: (value, sigma or None) of a trial device at a measurement index."""
    _NullFit = Callable[[np.ndarray, np.ndarray, np.ndarray | None, float, float], tuple[float, float, bool]]
    """A null fit over one shim's grid: (null, sigma, usable) from (grid, signal, sigmas, low, high)."""

FIELD_SHIMS = ("Ex", "Ey", "Ez")
_OPTIONS = {"method": "dop853", "atol": 1e-10, "rtol": 1e-8, "nsteps": 10**7, "progress_bar": ""}
_BETA_REF = 0.05
"""The reference modulation index at which the correlation signal's response phase and slope are read."""


def _sine(t: float, amplitude: float, omega: float, **_: object) -> float:
    return float(amplitude * math.sin(omega * t))


def periodic_scattering(
    model: BlochModel, beam_index: int, amplitude_rad_s: float, omega_rf_rad_s: float, *, n_points: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """(times over one rf period, total photon rate) of the periodic steady state of ``model`` with beam ``beam_index``'s
    frequency modulated as ``amplitude`` sin(Omega_rf t): the fixed point of the one-period propagator, then one period of
    ``mesolve``."""
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
        return times, np.full(times.shape, float(np.sum(model.operator_rates(rho))))
    shifted = model.shifted(beam_index, 1.0).build.H
    assert isinstance(shifted, qt.Qobj)
    d_op = shifted - h0  # the Hamiltonian change per rad/s of beam frequency offset (linear: a frame shift)
    lt = qt.QobjEvo(
        [
            qt.liouvillian(h0, c_ops),
            [
                qt.liouvillian(d_op),
                qt.coefficient(_sine, args={"amplitude": amplitude_rad_s, "omega": omega_rf_rad_s}),
            ],
        ]
    )
    vals, vecs = np.linalg.eig(np.asarray(qt.propagator(lt, period, options=_OPTIONS).full()))
    k = int(np.argmin(np.abs(vals - 1.0)))
    if abs(vals[k] - 1.0) > 1e-5:
        raise RuntimeError(f"no fixed point of the rf-period propagator (closest eigenvalue {vals[k]})")
    n = b.n_internal
    rho0 = vecs[:, k].reshape(n, n, order="F")
    rho0 = 0.5 * (rho0 + rho0.conj().T)
    rho0 = rho0 / np.trace(rho0)
    res = qt.mesolve(lt, qt.Qobj(rho0, dims=h0.dims), times, options={**_OPTIONS, "store_states": True})
    return times, np.array([float(np.sum(model.operator_rates(st))) for st in res.states])


def correlation_signal(times: np.ndarray, rates: np.ndarray, omega_rf_rad_s: float) -> tuple[complex, float]:
    """(S_1, mean rate): the complex first harmonic of the photon rate at the rf frequency, (2/T) int r(t) e^{-i Omega t} dt,
    over the mean rate; its modulus is the modulation depth and its phase the atom's response lag behind the rf."""
    t = np.asarray(times)
    r = np.asarray(rates)
    period = float(t[-1] - t[0])
    mean = float(np.trapezoid(r, t) / period)
    s = complex(2.0 / period * np.trapezoid(r * np.exp(-1j * omega_rf_rad_s * t), t))
    return (s / mean if mean > 0.0 else 0j), mean


def device_with_compensation(device: Device, shims: Mapping[str, float]) -> Device:
    """The device with the shims set (voltages on the geometry path, compensation-field components in V/m on the explicit
    path) and its crystal re-solved, so that the ions' displacement follows."""
    from qutip_trap.trap.crystal import solve_crystal

    trap = device.trap
    if trap.path == "explicit":
        unknown = [k for k in shims if k not in FIELD_SHIMS]
        if unknown:
            raise ValueError(
                f"the explicit-frequency trap has no electrode model; its shims are the compensation-field components "
                f"{FIELD_SHIMS}, not {unknown}"
            )
        stray = np.asarray(trap.stray_field_v_per_m, dtype=float) + np.array(
            [float(shims.get(k, 0.0)) for k in FIELD_SHIMS]
        )
        new_trap = replace(trap, stray_field_v_per_m=(float(stray[0]), float(stray[1]), float(stray[2])))
    else:
        new_trap = replace(
            trap, shim_voltages_v={**trap.shim_voltages_v, **{k: float(v) for k, v in shims.items()}}
        )
    return replace(device, trap=new_trap, crystal=solve_crystal(new_trap, device.crystal.species))


def signed_beta(device: Device, ion: int, k_vector: np.ndarray) -> float:
    """k . u_1: the signed modulation index along ``k_vector`` (peak), 0 without an rf record. Under the adopted Mathieu
    origin u_1 = -(1/2) Q u_0 (Section 13, "Floquet function and rf phase origin"), so sign(beta) = -sign(q_x E_x) and the
    odd correlation signal steps by pi as a shim crosses the compensated value; the fitted null does not depend on it."""
    if device.trap.rf is None:
        return 0.0
    return float(
        np.dot(
            np.asarray(k_vector, dtype=float),
            device.trap.micromotion_amplitude_m(device.crystal.species[ion]),
        )
    )


class _Correlation:
    """The rf-photon-correlation signal of the detection beam ``beam`` on one ion, retuned half a linewidth to the red
    (Berkeland's working point), with ``n_points`` per rf period."""

    def __init__(self, device: Device, ion: int, beam: int, n_points: int = 32) -> None:
        from qutip_trap.light.roles import detection_beams

        rf = device.trap.rf
        if rf is None:
            raise ValueError("the rf-photon correlation needs the trap's rf record")
        idx = detection_beams(device, ion)
        if beam not in idx:
            raise ValueError(f"beam {beam} is not a detection beam of ion {ion} (detection beams: {idx})")
        self.ion = ion
        self.n_points = n_points
        self.omega_rf = float(rf.omega_rad_s)
        self.k_vec = np.asarray(device.beams[beam].k_vector(), dtype=float)
        self.beam_index = idx.index(beam)
        _rates, _scheme, model = _detection_rates(device, ion)
        gamma = max(model.build.level_rates_rad_s.values())
        self.model = model.shifted(self.beam_index, -0.5 * gamma)
        self._cache: dict[float, tuple[complex, float]] = {}

    def harmonic(self, beta: float) -> tuple[complex, float]:
        """(complex first harmonic S_1, mean rate) at the signed modulation index ``beta`` (cached per beta)."""
        key = round(beta, 12)
        if key not in self._cache:
            times, rates = periodic_scattering(
                self.model, self.beam_index, beta * self.omega_rf, self.omega_rf, n_points=self.n_points
            )
            self._cache[key] = correlation_signal(times, rates, self.omega_rf)
        return self._cache[key]

    def reference_phase(self) -> float:
        """The atom's response phase: arg S_1 at a positive reference beta (a laboratory calibrates it with an offset)."""
        s_ref, _ = self.harmonic(_BETA_REF)
        return float(np.angle(s_ref)) if abs(s_ref) > 0.0 else 0.0

    def observables(self, beta: float) -> tuple[float, float]:
        """(S, mean rate) at ``beta``: the first harmonic projected on the response phase, odd in beta."""
        s1, mean = self.harmonic(beta)
        return float(np.real(s1 * np.exp(-1j * self.reference_phase()))), mean

    def slope(self) -> float:
        """dS/d beta near the null, from the model at +-the reference beta."""
        return (self.observables(_BETA_REF)[0] - self.observables(-_BETA_REF)[0]) / (2.0 * _BETA_REF)


def _invert_j1_over_j0(ratio: float) -> float:
    """|beta| from J_1(beta)/J_0(beta) = ratio on (0, 1.8) (the first sideband over the carrier, Section 4.1.1)."""
    if ratio <= 0.0:
        return 0.0
    if float(jv(1, 1.8) / jv(0, 1.8)) < ratio:
        return 1.8
    return float(brentq(lambda b: float(jv(1, b) / jv(0, b)) - ratio, 1e-12, 1.8))


def _null_shims(
    device: Device,
    shims: dict[str, float],
    shim_ranges_v: Mapping[str, tuple[float, float]],
    points: int,
    signal: _Signal,
    fit: _NullFit,
    failure: str,
    notes: list[str],
) -> tuple[list[tuple[float, float, float, float, float]], dict[str, tuple[float, float]], bool]:
    """Null each shim in turn: scan it over its range with the others held, fit the null and hold it there (``shims`` is
    updated). Returns the data rows (pass, shim index, setting, signal, sigma), the nulls and whether every fit held."""
    rows: list[tuple[float, float, float, float, float]] = []
    nulls: dict[str, tuple[float, float]] = {}
    converged = True
    for j, name in enumerate(shim_ranges_v):
        lo, hi = (float(x) for x in shim_ranges_v[name])
        grid = np.linspace(lo, hi, points)
        measured: list[tuple[float, float | None]] = []
        for v in grid:
            value, s = signal(device_with_compensation(device, {**shims, name: float(v)}), len(rows))
            measured.append((value, s))
            rows.append((0.0, float(j), float(v), value, 0.0 if s is None else s))
        v0, s_v0, ok = fit(
            grid, np.array([x for x, _s in measured]), sigmas_or_none([s for _x, s in measured]), lo, hi
        )
        if not ok or at_scan_edge(v0, lo, hi, 0.0):
            converged = False
            notes.append(f"shim {name!r}: {failure}")
            v0 = float(min(max(v0, lo), hi))
        shims[name] = v0
        nulls[name] = (v0, s_v0)
    return rows, nulls, converged


def micromotion_scan(
    machine: Machine,
    ion: int,
    beam: int,
    shim_ranges_v: Mapping[str, tuple[float, float]],
    method: str = "rf_photon_correlation",
    *,
    points: int = 5,
    rf_points: int = 32,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """Null the excess micromotion along ``beam`` by ``method`` (``"rf_photon_correlation"`` or ``"sideband_ratio"``, whose
    ``beam`` names the gate drive's table-key beam), scanning each shim over ``shim_ranges_v`` (name -> (low, high)) at
    ``points`` settings; ``rf_points`` samples the rf period of the correlation signal. Fitted: shim[name] (the null and its
    uncertainty), beta[beam] (the residual index along the beam at the null, peak convention) and beta_before, plus
    carrier_hz and sideband_excitation for the sideband ratio. Data rows (pass, shim index, setting, signal, sigma)."""
    if method not in ("rf_photon_correlation", "sideband_ratio"):
        raise ValueError("method is 'rf_photon_correlation' or 'sideband_ratio'")
    lab = _Lab.of(machine, kw)
    device = lab.device
    requested = ScanParameters({f"shim[{n}]_v": tuple(shim_ranges_v[n]) for n in shim_ranges_v})
    subject = {"ion": int(ion), "beam": int(beam)}
    if device.trap.rf is None:
        fitted = {f"shim[{name}]": (0.0, 0.0) for name in shim_ranges_v}
        fitted[f"beta[{beam}]"] = (0.0, 0.0)
        fitted["beta_before"] = (0.0, 0.0)
        return MicromotionScan(
            data=np.zeros((0, 5)),
            fitted=fitted,
            model=f"micromotion_{method}",
            provenance_id="anchor.trap.berkeland_excess_micromotion",
            notes=("no rf record on the trap: excess micromotion is not modelled (C0 = 1, beta = 0)",),
            requested=requested,
            subject=subject,
        )
    explicit = device.trap.path == "explicit"
    shims = {k: 0.0 if explicit else float(device.trap.shim_voltages_v.get(k, 0.0)) for k in shim_ranges_v}
    unit = "V/m (compensation field, explicit-frequency path)" if explicit else "V"
    notes: list[str] = [f"shim unit: {unit}"]
    if method == "rf_photon_correlation":
        corr = _Correlation(device, ion, beam, rf_points)
        efficiency = float(device.detector.efficiency)

        def correlation(trial: Device, index: int) -> tuple[float, float | None]:
            s, mean = corr.observables(signed_beta(trial, ion, corr.k_vec))
            sigma = math.sqrt(2.0 / max(mean * efficiency * 1.0, 1.0))  # one second of photons per point
            return s + lab.obs.noise(sigma, "micromotion_correlation", index, ion), sigma

        def fit_line(
            grid: np.ndarray, y: np.ndarray, sigma: np.ndarray | None, lo: float, hi: float
        ) -> tuple[float, float, bool]:
            guess = [(y[-1] - y[0]) / max(hi - lo, 1e-300), 0.5 * (lo + hi)]
            fit = weighted_fit(
                lambda q, x: float(q[0]) * (np.asarray(x) - float(q[1])), guess, grid, y, sigma=sigma
            )
            return (*fit.value(1), fit.converged and abs(float(fit.params[0])) > 0.0)

        failure = "the null fit failed or lies outside the scanned range"
        rows, nulls, converged = _null_shims(
            device, shims, shim_ranges_v, points, correlation, fit_line, failure, notes
        )
        notes.extend(f"shim {n!r} null {v0:.6g} +- {s_v0:.3g} {unit}" for n, (v0, s_v0) in nulls.items())
        s_final, sg_final = correlation(device_with_compensation(device, shims), len(rows))
        slope = corr.slope()
        if abs(slope) > 0.0:
            beta_res, s_beta = s_final / slope, float(sg_final or 0.0) / abs(slope)
        else:
            beta_res, s_beta = 0.0, math.nan
            notes.append(
                "the correlation signal has no slope at the reference beta: the residual is not measured"
            )
        fitted = {f"shim[{n}]": nulls[n] for n in shim_ranges_v}
        fitted[f"beta[{beam}]"] = (beta_res, s_beta)
        fitted["beta_before"] = (signed_beta(device, ion, corr.k_vec), 0.0)
        return MicromotionScan(
            data=np.array(rows),
            fitted=fitted,
            model=f"micromotion_{method}",
            provenance_id="anchor.trap.berkeland_excess_micromotion",
            converged=converged,
            notes=tuple(notes),
            requested=requested,
            subject=subject,
        )
    return _sideband_ratio_scan(machine, lab, ion, beam, shim_ranges_v, shims, points, notes, requested, kw)


def _sideband_ratio_scan(
    machine: Machine,
    lab: _Lab,
    ion: int,
    beam: int,
    shim_ranges_v: Mapping[str, tuple[float, float]],
    shims: dict[str, float],
    points: int,
    notes: list[str],
    requested: ScanParameters,
    kw: _LabOptions,
) -> ExperimentResult:
    """``micromotion_scan(method="sideband_ratio")``: the first micromotion sideband's excitation under the exact modulated
    builder, a parabola per shim, and |beta| at the null from J_1/J_0 of the sideband and carrier rates."""
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import default_gate_drives
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.light.raman import derive_optical_drive, derive_raman_drive

    device = lab.device
    assert device.trap.rf is not None
    gate_drive = default_gate_drives(device)[ion]
    if gate_drive.kind == "raman":
        derived = derive_raman_drive(
            device, ion, (gate_drive.beams[0], gate_drive.beams[1]), scattering=False
        )
    elif gate_drive.kind in ("optical_E1", "optical_E2"):
        derived = derive_optical_drive(device, ion, gate_drive.beams[0], scattering=False)
    else:
        raise ValueError("the sideband-ratio method drives the ion with laser light")
    modulated = BuilderOptions(micromotion="modulated")
    lab = replace(lab, builder=modulated)
    omega = derived.carrier_rabi_hz
    probe = _Probe(detuning_hz=float(device.trap.rf.frequency_hz), include_stark=False, rf_locked=True)
    t_sb = 1.0 / (4.0 * float(jv(1, 0.2)) * omega)

    def excitation(trial: Device, index: int) -> tuple[float, float | None]:
        trial_lab = replace(lab, device=trial)
        setup = _setup(trial_lab, ion, probe)
        avg = _run(trial_lab, ion, [Pulse(setup.drive, 0.0, t_sb, "micromotion_sideband", ())], setup)
        return lab.obs.p1(avg.final_p1(ion), ion, "micromotion_sideband", index)

    def fit_parabola(
        grid: np.ndarray, y: np.ndarray, sigma: np.ndarray | None, lo: float, hi: float
    ) -> tuple[float, float, bool]:
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
        return (*fit.value(1), fit.converged)

    failure = "the sideband parabola did not converge or its minimum lies outside the range"
    rows, nulls, converged = _null_shims(
        device, shims, shim_ranges_v, points, excitation, fit_parabola, failure, notes
    )
    final = device_with_compensation(device, shims)
    carrier_times = [float(x) for x in np.linspace(0.0, 2.0 / omega, 9)]
    carrier_kw: _LabOptions = {**kw, "builder_options": modulated}
    carrier = rabi_scan(
        replace(machine, device=final), ion, carrier_times, rf_locked=True, include_stark=False, **carrier_kw
    )
    f0, s_f0 = carrier.fitted["f_rabi_hz"]
    p_sb, s_sb = excitation(final, len(rows))
    # P = sin^2(pi f_1 t): the sideband rate, then |beta| from J_1/J_0 = f_1/f_0
    p_c = min(max(p_sb, 0.0), 1.0)
    f1 = math.asin(math.sqrt(p_c)) / (math.pi * t_sb)
    beta_res = _invert_j1_over_j0(f1 / f0 if f0 > 0.0 else 0.0)
    # sigma from the excitation's (d f1/d P = 1/(2 pi t sqrt(P (1 - P)))) and the carrier's; d beta/d ratio ~ 2 near zero
    s_p = s_sb if s_sb is not None else 0.0
    s_f1 = 1.0 / (2.0 * math.pi * t_sb * math.sqrt(max(p_c * (1.0 - p_c), 1e-12))) * s_p
    s_beta = 2.0 * math.hypot(s_f1 / max(f0, 1e-300), f1 * s_f0 / max(f0, 1e-300) ** 2)
    fitted = {f"shim[{n}]": nulls[n] for n in shim_ranges_v}
    fitted[f"beta[{gate_drive.table_key_beam}]"] = (beta_res, s_beta)
    fitted["beta_before"] = (abs(signed_beta(device, ion, np.asarray(derived.delta_k, dtype=float))), 0.0)
    fitted["carrier_hz"] = (f0, s_f0)
    fitted["sideband_excitation"] = (p_sb, s_p)
    return MicromotionScan(
        data=np.array(rows),
        fitted=fitted,
        model="micromotion_sideband_ratio",
        provenance_id="anchor.trap.berkeland_excess_micromotion",
        converged=converged,
        notes=tuple(notes),
        requested=requested,
        subject={"ion": int(ion), "beam": int(beam)},
    )
