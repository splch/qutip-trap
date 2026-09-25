"""The multi-level optical-Bloch model: steady state, time evolution, slow manifolds and cooling coefficients.

One steady state of the multi-level master equation (built by :mod:`qutip_trap.dynamics.multilevel`) supplies every
scattering rate: the cooling rate W(Delta) = Gamma rho_ee, the readout rates, the light shifts and the dark states; its
evolution is optical pumping (PLAN.md Section 4.2.8). A static frame gives the direct steady state; an inconsistent frame
(two tones on one transition, a polarization modulation) makes the Liouvillian periodic and the fixed point of the
one-period propagator is period-averaged instead. Photon rates are asserted against the equal-population ceiling
n_e/(n_e + n_g) of the resonant manifold. The slowest connecting Liouvillian mode gives the bright <-> dark pumping rates
and the conditional bright state. The cooling coefficients are A_+- = 2 Re[S(-+ nu) + D] from the dipole-force spectrum
(QuTiP's spectrum(omega) = int e^{-i omega tau} <dF(tau) dF(0)> dtau, which is W(Delta - omega) for a weak drive) plus the
emission-recoil diffusion with one alpha per polarization channel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import qutip as qt
from scipy.linalg import expm

from qutip_trap.dynamics.multilevel import (
    SINK,
    EmissionChannel,
    ModeSpec,
    MultiLevelBuild,
    MultiLevelOptions,
    build_multilevel,
)
from qutip_trap.dynamics.steady import spectrum_es, steady_state_direct
from qutip_trap.light.beams import Beam, PolarizationModulation
from qutip_trap.light.recoil import angular_factor
from qutip_trap.readout.fluorescence import DarkStateReport, saturation_ceiling
from qutip_trap.species.raman import AtomicStructure, structure_at
from qutip_trap.units import C_M_PER_S, TWO_PI


class CeilingViolation(AssertionError):
    """A Bloch-solve photon rate exceeded the equal-population ceiling of its closed manifold."""


class CoolingError(ValueError):
    """A_- <= A_+: the configuration heats; no steady-state occupation exists."""


def beam_for_transition(
    structure: AtomicStructure,
    lower: str,
    upper: str,
    detuning_rad_s: float,
    k_hat: tuple[float, float, float],
    polarization: tuple[complex, complex, complex],
    *,
    power_w: float,
    waist_m: float,
    pointing_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    polarization_amplitudes: tuple[complex, complex, complex] | None = None,
    modulation: PolarizationModulation | None = None,
) -> Beam:
    """A beam ``detuning`` (rad/s, red negative) from the dressed transition lower -> upper at the field."""
    omega = TWO_PI * (structure.state(upper).energy_hz - structure.state(lower).energy_hz) + detuning_rad_s
    return Beam(
        TWO_PI * C_M_PER_S / omega,
        k_hat,
        polarization,
        waist_m,
        power_w,
        pointing_m,
        polarization_amplitudes,
        modulation,
    )


def shifted_beam(beam: Beam, offset_rad_s: float) -> Beam:
    """The same beam with its frequency moved by ``offset`` (rad/s)."""
    omega = TWO_PI * C_M_PER_S / beam.wavelength_m + offset_rad_s
    return replace(beam, wavelength_m=TWO_PI * C_M_PER_S / omega)


# ---- reports --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CeilingReport:
    """The resonant closed manifold and its equal-population ceiling (Berkeland 1998 Eqs. 13-14, 21)."""

    ground_labels: tuple[str, ...]
    excited_labels: tuple[str, ...]
    ceiling: float
    """n_e/(n_e + n_g); NaN when no coupling is near resonance (no closed resonant manifold to bound)."""
    excited_population: float


@dataclass(frozen=True)
class SteadyStateReport:
    """One steady state and every rate read from it; ``photon_rates_per_s`` is per decay line "lower<-upper" (the sink
    entry is the leak rate) and ``total_photon_rate_per_s`` is W(Delta), every scattered photon."""

    rho: qt.Qobj
    populations: dict[str, float]
    level_populations: dict[str, float]
    photon_rates_per_s: dict[str, float]
    total_photon_rate_per_s: float
    excited_population: float
    ceiling: CeilingReport
    method: Literal["steadystate", "floquet", "time-average"]
    period_s: float | None
    nbar: float | None
    """Mean phonon number of the mode, when the build has one."""


@dataclass(frozen=True)
class PumpingTrace:
    """Populations against time under the beams, the photon rate and its cumulative integral."""

    times_s: np.ndarray
    populations: dict[str, np.ndarray]
    level_populations: dict[str, np.ndarray]
    photon_rate_per_s: np.ndarray
    photons_scattered: np.ndarray
    photon_rates_per_operator: np.ndarray
    """(n_times, n_c_ops) rate Tr(C_k^dag C_k rho) per collapse operator: each emitted polarization's photons."""

    def photons_per_operator(self) -> np.ndarray:
        """Trapezoid integral of each collapse operator's rate over the trace."""
        r = self.photon_rates_per_operator
        return np.asarray(np.sum(0.5 * (r[1:] + r[:-1]) * np.diff(self.times_s)[:, None], axis=0))

    def population(self, labels: Sequence[str]) -> np.ndarray:
        return np.asarray(np.sum([self.populations[lab] for lab in labels], axis=0))

    def time_to_reach(self, labels: Sequence[str], target: float) -> float | None:
        """First sampled time at which the summed population of ``labels`` reaches ``target`` (None if never)."""
        hit = np.flatnonzero(self.population(labels) >= target)
        return None if hit.size == 0 else float(self.times_s[hit[0]])


@dataclass(frozen=True)
class ManifoldRates:
    """Two-manifold coarse graining of the slow Liouvillian dynamics: the rates between the manifolds, the stationary
    weight of A, the conditional state of A, and |Re lambda_next|/|Re lambda_slow| as the separation of the picture.

    ``intra_manifold_rates_per_s`` are the slower modes that live inside one manifold, skipped in choosing the connecting
    one (a 171Yb+ model with the D3/2 branch and its repump has one)."""

    rate_a_to_b_per_s: float
    rate_b_to_a_per_s: float
    weight_a: float
    conditional_a: qt.Qobj
    separation: float
    intra_manifold_rates_per_s: tuple[float, ...] = ()


@dataclass(frozen=True)
class DetectionRates:
    """R_o, R_d and R_b from one Liouvillian: bright-state photon rate, bright -> dark and dark -> bright pumping."""

    R_bright_per_s: float
    R_dark_pumping_per_s: float
    R_bright_pumping_per_s: float
    ceiling: CeilingReport
    conditional_bright: qt.Qobj
    separation: float


# ---- the model ------------------------------------------------------------------------------------------------------------


_PERIODIC_OPTIONS: dict[str, object] = {
    "method": "vern9",
    "nsteps": 10**7,
    "atol": 1e-10,
    "rtol": 1e-8,
    "progress_bar": "",
}
"""Integrator settings for the periodic Liouvillian (never a multistep method)."""


def _hermitize(rho: np.ndarray) -> np.ndarray:
    r = 0.5 * (rho + rho.conj().T)
    tr = np.trace(r)
    return np.asarray(r / tr) if abs(tr) > 0.0 else r


class BlochModel:
    """A species at a field under a set of beams: the internal levels (optionally times one mode) and their solves."""

    def __init__(
        self,
        structure: AtomicStructure,
        beams: Sequence[Beam],
        *,
        levels: Sequence[str] | None = None,
        states: Sequence[str] | None = None,
        position_m: Sequence[float] | None = None,
        mode: ModeSpec | None = None,
        options: MultiLevelOptions | None = None,
    ) -> None:
        self.structure = structure
        self.beams = tuple(beams)
        self.levels_requested = None if levels is None else tuple(levels)
        self.states_requested = None if states is None else tuple(states)
        self.position_m = None if position_m is None else tuple(float(x) for x in position_m)
        self.mode = mode
        self.options = options or MultiLevelOptions()
        self.build: MultiLevelBuild = build_multilevel(
            structure,
            self.beams,
            levels=levels,
            states=states,
            position_m=position_m,
            mode=mode,
            options=self.options,
        )

    def shifted(self, beam: int, offset_rad_s: float) -> BlochModel:
        """The same model with beam ``beam`` moved by ``offset`` in frequency."""
        beams = list(self.beams)
        beams[beam] = shifted_beam(beams[beam], offset_rad_s)
        return BlochModel(
            self.structure,
            beams,
            levels=self.levels_requested,
            states=self.states_requested,
            position_m=self.position_m,
            mode=self.mode,
            options=self.options,
        )

    # ---- rates from a state ----------------------------------------------------------------------------------

    def operator_rates(self, rho: qt.Qobj) -> np.ndarray:
        """Tr(C_k^dag C_k rho) for every collapse operator k, in s^-1."""
        r = rho if rho.isoper else qt.ket2dm(rho)
        return np.array([float(np.real(qt.expect(c.dag() * c, r))) for c in self.build.c_ops])

    def photon_rates(self, rho: qt.Qobj) -> dict[str, float]:
        """Photons per second on each decay line "lower<-upper"."""
        rates = self.operator_rates(rho)
        out: dict[str, float] = {}
        for ch in self.build.channels:
            start, stop = ch.operator_slice
            key = f"{ch.lower}<-{ch.upper}"
            out[key] = out.get(key, 0.0) + float(np.sum(rates[start:stop]))
        return out

    def resonant_manifold(
        self, *, window_gammas: float = 10.0, static_only: bool = False
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """(ground, excited) labels of the closed manifold the beams drive within ``window_gammas`` linewidths.

        The ground manifold is widened to every sublevel of the same level within that window of a resonant ground state
        (the Zeeman neighbours a differently polarized beam would drive), which is what the ceiling and Berkeland's
        dark-state count refer to.
        """
        b = self.build
        ground: list[str] = []
        excited: list[str] = []
        for c in b.couplings:
            gamma = b.level_rates_rad_s.get(b.level_of(c.upper), 0.0)
            if gamma <= 0.0 or (static_only and c.residual_rad_s != 0.0):
                continue
            if abs(c.detuning_rad_s) <= window_gammas * gamma:
                if c.lower not in ground:
                    ground.append(c.lower)
                if c.upper not in excited:
                    excited.append(c.upper)
        if not ground:
            return (), ()
        gamma_max = max(b.level_rates_rad_s.get(b.level_of(e), 0.0) for e in excited)
        for lab in b.labels:
            if lab == SINK or lab in ground:
                continue
            e_lab = TWO_PI * self.structure.state(lab).energy_hz
            for g_lab in list(ground):
                if b.level_of(g_lab) == b.level_of(lab):
                    if (
                        abs(e_lab - TWO_PI * self.structure.state(g_lab).energy_hz)
                        <= window_gammas * gamma_max
                    ):
                        ground.append(lab)
                        break
        order = {lab: k for k, lab in enumerate(b.labels)}
        return tuple(sorted(ground, key=order.__getitem__)), tuple(sorted(excited, key=order.__getitem__))

    def ceiling_report(self, rho: qt.Qobj, *, window_gammas: float = 10.0) -> CeilingReport:
        """The resonant manifold and the check of ``rho``'s excited population against its ceiling."""
        ground, excited = self.resonant_manifold(window_gammas=window_gammas)
        if not ground or not excited:
            return CeilingReport(
                (), (), math.nan, 0.0
            )  # NaN: no manifold to bound (1.0 would read as satisfied)
        pops = self.build.populations(rho)
        p_e = float(sum(pops[lab] for lab in excited))
        ceiling = saturation_ceiling(len(ground), len(excited))
        if p_e > ceiling * (1.0 + 1e-9) + 1e-12:
            raise CeilingViolation(
                f"resonant excited population {p_e:.6g} exceeds the equal-population ceiling {ceiling:.6g} of "
                f"{len(ground)} ground + {len(excited)} excited states"
            )
        return CeilingReport(tuple(ground), tuple(excited), ceiling, p_e)

    def _report(
        self, rho: qt.Qobj, method: Literal["steadystate", "floquet", "time-average"], period: float | None
    ) -> SteadyStateReport:
        rates = self.photon_rates(rho)
        pops = self.build.populations(rho)
        decaying = [
            lab for lab in self.build.labels if self.build.level_of(lab) in self.build.level_rates_rad_s
        ]
        return SteadyStateReport(
            rho=rho,
            populations=pops,
            level_populations=self.build.level_populations(rho),
            photon_rates_per_s=rates,
            total_photon_rate_per_s=float(sum(v for k, v in rates.items() if not k.startswith(SINK))),
            excited_population=float(sum(pops[lab] for lab in decaying)),
            ceiling=self.ceiling_report(rho),
            method=method,
            period_s=period,
            nbar=None if self.build.space is None else float(np.real(qt.expect(self.build.number(), rho))),
        )

    def steadystate(self, *, n_average: int = 64, settle_gammas: float = 1000.0) -> SteadyStateReport:
        """The direct steady state on a static frame, the period-averaged Floquet fixed point otherwise, and for
        incommensurate beats the average of the second half of ``settle_gammas`` lifetimes of propagation."""
        b = self.build
        if b.static:
            assert isinstance(b.H, qt.Qobj)
            return self._report(steady_state_direct(b.H, b.c_ops), "steadystate", None)
        if b.space is not None:
            raise NotImplementedError(
                "the periodic (Floquet) steady state is built for internal-only spaces; with a mode restrict the beams "
                "to a consistent frame"
            )
        L = b.liouvillian()
        period = b.frame.period_s
        if period is None:
            t_end = settle_gammas / min(b.level_rates_rad_s.values())
            times = np.linspace(0.0, t_end, 2 * n_average + 1)
            rho0 = qt.ket2dm(b.internal_state(b.labels[0]))
            res = qt.mesolve(L, rho0, times, options={**_PERIODIC_OPTIONS, "store_states": True})
            avg = _hermitize(
                np.mean(np.stack([np.asarray(s.full()) for s in res.states[n_average:]]), axis=0)
            )
            return self._report(qt.Qobj(avg, dims=b.H.dims), "time-average", None)
        vals, vecs = np.linalg.eig(np.asarray(qt.propagator(L, period, options=_PERIODIC_OPTIONS).full()))
        k = int(np.argmin(np.abs(vals - 1.0)))
        if abs(vals[k] - 1.0) > 1e-6:
            raise RuntimeError(f"no fixed point of the period propagator (closest eigenvalue {vals[k]})")
        n = b.n_internal
        rho_strob = _hermitize(vecs[:, k].reshape(n, n, order="F"))
        lowest = float(np.min(np.linalg.eigvalsh(rho_strob)))
        if lowest < -1e-6:
            raise RuntimeError(
                f"the period-propagator fixed point is not a density matrix (eigenvalue {lowest:.3g})"
            )
        times = np.linspace(0.0, period, n_average + 1)
        res = qt.mesolve(
            L, qt.Qobj(rho_strob, dims=b.H.dims), times, options={**_PERIODIC_OPTIONS, "store_states": True}
        )
        states = [np.asarray(s.full()) for s in res.states]
        # trapezoid average over one period (the endpoints coincide up to integration error)
        avg = _hermitize((sum(states[1:-1]) + 0.5 * (states[0] + states[-1])) / n_average)
        return self._report(qt.Qobj(avg, dims=b.H.dims), "floquet", period)

    def evolve(self, initial: str | qt.Qobj, times_s: Sequence[float] | np.ndarray) -> PumpingTrace:
        """Populations against time from ``initial`` (a state label or an internal ket/density matrix).

        Static internal-only builds are propagated exactly with the matrix exponential of the Liouvillian (the frame keeps
        GHz hyperfine detunings on the diagonal, which an adaptive integrator would have to resolve); builds with a mode
        or a periodic Liouvillian go through ``mesolve``.
        """
        b = self.build
        times = np.asarray(times_s, dtype=float)
        if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be increasing with at least two points")
        if isinstance(initial, str):
            rho0 = qt.ket2dm(b.internal_state(initial))
            if b.space is not None:
                rho0 = qt.tensor(rho0, qt.fock_dm(b.space.dims[1], 0))
        else:
            rho0 = initial if initial.isoper else qt.ket2dm(initial)
        states: list[qt.Qobj]
        if b.static and b.space is None:
            L = np.asarray(b.liouvillian().full())
            n = b.n_internal
            vec = np.asarray(rho0.full()).reshape(-1, order="F")
            dts = np.diff(times)
            if times[0] != 0.0:
                vec = expm(L * times[0]) @ vec
            states = [qt.Qobj(vec.reshape(n, n, order="F"), dims=b.H.dims)]
            step = expm(L * dts[0]) if bool(np.allclose(dts, dts[0], rtol=1e-9, atol=0.0)) else None
            for dt in dts:
                vec = (step if step is not None else expm(L * dt)) @ vec
                states.append(qt.Qobj(vec.reshape(n, n, order="F"), dims=b.H.dims))
        else:
            res = qt.mesolve(
                b.H, rho0, times, c_ops=list(b.c_ops), options={**_PERIODIC_OPTIONS, "store_states": True}
            )
            states = list(res.states)
        pops = {lab: np.zeros(times.size) for lab in b.labels}
        per_op = np.zeros((times.size, len(b.c_ops)))
        for i, s in enumerate(states):
            for lab, p in b.populations(s).items():
                pops[lab][i] = p
            per_op[i] = self.operator_rates(s)
        per_line: dict[str, np.ndarray] = {}
        for ch in b.channels:
            start, stop = ch.operator_slice
            key = f"{ch.lower}<-{ch.upper}"
            per_line[key] = per_line.get(key, 0.0) + np.sum(per_op[:, start:stop], axis=1)
        rate = np.zeros(times.size)
        for key, arr in per_line.items():
            if not key.startswith(SINK):
                rate += arr
        levels: dict[str, np.ndarray] = {}
        for lab, arr in pops.items():
            key = SINK if lab == SINK else b.level_of(lab)
            levels[key] = levels.get(key, 0.0) + arr
        photons = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(times))])
        return PumpingTrace(times, pops, levels, rate, photons, per_op)

    # ---- slow-manifold analysis ----------------------------------------------------------------------------

    def manifold_rates(
        self,
        a_labels: Sequence[str],
        b_labels: Sequence[str],
        *,
        connection_tol: float = 1e-9,
        closure_tol: float = 1e-3,
        weight_floor: float = 1e-9,
    ) -> ManifoldRates:
        """Coarse-grain the Liouvillian into two manifolds by its slowest mode that CONNECTS them.

        Each eigenvector is projected onto the two manifolds' populations and the slowest one with both projections above
        ``connection_tol`` of its trace norm is taken; the conditional states carry no population in the other manifold,
        and the rates follow from k = R_ab + R_ba and the stationary weights. Population outside the manifolds and the
        decaying levels (a D manifold the species table cannot repump, say) makes the rates meaningless and is refused
        (``closure_tol``), as is an absorbing manifold (``weight_floor``).
        """
        b = self.build
        if not b.static or b.space is not None:
            raise NotImplementedError("the slow-manifold analysis is for static internal-only builds")
        vals, vecs = np.linalg.eig(np.asarray(b.liouvillian().full()))
        order = np.argsort(-vals.real)
        n = b.n_internal
        rho_ss = _hermitize(vecs[:, order[0]].reshape(n, n, order="F"))
        p_a = b.manifold_projector(a_labels).full()
        p_b = b.manifold_projector(b_labels).full()
        trace_ss = float(np.real(np.trace(rho_ss)))
        if trace_ss == 0.0:
            raise ValueError(
                "the stationary Liouvillian eigenvector is traceless: no steady state to coarse-grain"
            )
        rho_ss = rho_ss / trace_ss
        pa_ss = float(np.real(np.trace(p_a @ rho_ss)))
        pb_ss = float(np.real(np.trace(p_b @ rho_ss)))
        decaying = [lab for lab in b.labels if b.level_of(lab) in b.level_rates_rad_s]
        accounted = (
            pa_ss
            + pb_ss
            + float(np.real(np.trace(b.manifold_projector(decaying).full() @ rho_ss)) if decaying else 0.0)
        )
        if lost := 1.0 - accounted - (rho_ss[b.index(SINK), b.index(SINK)].real if SINK in b.labels else 0.0):
            if lost > closure_tol:
                elsewhere = {
                    lab: float(np.real(rho_ss[b.index(lab), b.index(lab)]))
                    for lab in b.labels
                    if lab not in a_labels and lab not in b_labels and lab not in decaying and lab != SINK
                }
                worst = sorted(elsewhere.items(), key=lambda kv: -kv[1])[:4]
                raise ValueError(
                    f"the two manifolds and the decaying levels carry only {accounted:.4g} of the stationary "
                    f"population: {lost:.4g} sits elsewhere ({', '.join(f'{k} {v:.3g}' for k, v in worst)}), so the "
                    "two-manifold coarse graining is not a partition of the slow dynamics and its rates would come "
                    "out negative. Include that population in one of the manifolds, or close its decay path in the "
                    "species table"
                )
        chosen: int | None = None
        slow = np.zeros((n, n), dtype=complex)
        pa_v = pb_v = 0.0
        skipped: list[float] = []
        for position in range(1, order.size):
            candidate = vecs[:, order[position]].reshape(n, n, order="F")
            candidate = 0.5 * (candidate + candidate.conj().T)
            scale = float(np.sum(np.abs(np.diag(candidate))))
            if scale <= 0.0:
                continue
            a_weight = float(np.real(np.trace(p_a @ candidate)))
            b_weight = float(np.real(np.trace(p_b @ candidate)))
            if abs(a_weight) > connection_tol * scale and abs(b_weight) > connection_tol * scale:
                chosen, slow, pa_v, pb_v = position, candidate, a_weight, b_weight
                break
            skipped.append(-float(vals[order[position]].real))
        if chosen is None:
            raise ValueError(
                "no Liouvillian mode connects the two manifolds: the labels given do not exchange population; check "
                "that the manifolds are the bright and dark states of one pumping cycle"
            )
        k_slow = -float(vals[order[chosen]].real)
        third = -float(vals[order[chosen + 1]].real) if chosen + 1 < order.size else math.inf
        s = pa_ss / pa_v - pb_ss / pb_v
        w_a = (pa_ss / pa_v) / s
        w_b = 1.0 - w_a
        if not weight_floor <= w_a <= 1.0 - weight_floor:
            raise ValueError(
                f"the stationary weight of the first manifold is {w_a:.4g}, outside ({weight_floor:g}, "
                f"{1.0 - weight_floor:g}): one manifold is ABSORBING, so there is no two-way pumping cycle to "
                "coarse-grain and the conditional states diverge. This is what an open decay path looks like - a "
                "171Yb+ D3/2 branch whose 935 nm repump the species table cannot return to S1/2, say; close the path "
                "or drop the level from the model"
            )
        return ManifoldRates(
            rate_a_to_b_per_s=k_slow * w_b,
            rate_b_to_a_per_s=k_slow * w_a,
            weight_a=w_a,
            conditional_a=qt.Qobj(_hermitize(rho_ss + w_b * s * slow), dims=b.H.dims),
            separation=third / k_slow if k_slow > 0.0 else math.inf,
            intra_manifold_rates_per_s=tuple(skipped),
        )

    def detection_rates(
        self,
        bright_labels: Sequence[str],
        dark_labels: Sequence[str],
        line: str | None = None,
        *,
        connection_tol: float = 1e-9,
        weight_floor: float = 1e-9,
    ) -> DetectionRates:
        """R_o on ``line`` ("S1/2<-P1/2"; default every non-sink line) in the conditional bright state, with R_d and R_b."""
        mr = self.manifold_rates(
            bright_labels, dark_labels, connection_tol=connection_tol, weight_floor=weight_floor
        )
        rates = self.photon_rates(mr.conditional_a)
        r_bright = (
            float(sum(v for k, v in rates.items() if not k.startswith(SINK))) if line is None else rates[line]
        )
        return DetectionRates(
            R_bright_per_s=r_bright,
            R_dark_pumping_per_s=mr.rate_a_to_b_per_s,
            R_bright_pumping_per_s=mr.rate_b_to_a_per_s,
            ceiling=self.ceiling_report(mr.conditional_a),
            conditional_bright=mr.conditional_a,
            separation=mr.separation,
        )

    def dark_states(
        self, *, window_gammas: float = 10.0, tol: float = 1e-9, zero_field: bool = True
    ) -> DarkStateReport:
        """Dark superpositions of the resonant ground states: the null space of the summed coupling matrix.

        ``delta_over_omega`` is the resonant ground manifold's Zeeman span over sqrt(sum |Omega_qp|^2), ``theta_be_deg``
        the angle between the first beam's linear polarization and B (nan otherwise), ``raman_zero_margin_hz`` the
        smallest two-photon detuning between beams sharing an upper level from different lower levels (the S-P-D dark
        resonances). With ``zero_field`` the couplings are those of the pure |F m_F> states (Berkeland's Table I counts);
        at a finite field the hyperfine admixture turns a Table-I dark state into a small singular value.
        """
        b = self.build
        ground_t, excited_t = self.resonant_manifold(window_gammas=window_gammas, static_only=True)
        ground, excited = list(ground_t), list(excited_t)
        if not ground or not excited:
            raise ValueError("no resonant static coupling: nothing is driven")
        couplings = b.couplings
        if zero_field:
            st0 = structure_at(self.structure.species, 1e-6, tuple(float(x) for x in self.structure.b_hat))
            couplings = build_multilevel(
                st0,
                self.beams,
                levels=b.levels,
                states=[lab for lab in b.labels if lab != SINK],
                position_m=self.position_m,
                options=MultiLevelOptions(leak=self.options.leak, address_window=self.options.address_window),
            ).couplings
        m = np.zeros((len(excited), len(ground)), dtype=complex)
        for c in couplings:
            if c.lower in ground and c.upper in excited and c.residual_rad_s == 0.0:
                m[excited.index(c.upper), ground.index(c.lower)] += c.omega_rad_s
        _u, sv, vh = np.linalg.svd(m)
        rank = int(np.sum(sv > tol * (float(np.max(sv)) if sv.size else 1.0)))
        energies = np.array([TWO_PI * self.structure.state(lab).energy_hz for lab in ground])
        span = float(np.max(energies) - np.min(energies)) if len(ground) > 1 else 0.0
        omega_rms = float(np.sqrt(np.sum(np.abs(m) ** 2)))
        theta = math.nan
        if self.beams:
            pol = np.asarray(self.beams[0].polarization, dtype=complex)
            if np.allclose(pol.imag, 0.0):
                theta = math.degrees(math.acos(min(1.0, abs(float(np.dot(pol.real, self.structure.b_hat))))))
        margin: float | None = None
        for c1 in b.couplings:
            for c2 in b.couplings:
                if (
                    c1.beam < c2.beam
                    and c1.upper == c2.upper
                    and b.level_of(c1.lower) != b.level_of(c2.lower)
                ):
                    two_photon = abs(c1.detuning_rad_s - c2.detuning_rad_s) / TWO_PI
                    margin = two_photon if margin is None else min(margin, two_photon)
        return DarkStateReport(
            n_ground=len(ground),
            n_excited=len(excited),
            ceiling=saturation_ceiling(len(ground), len(excited)),
            dark_dimension=len(ground) - rank,
            dark_basis=vh[rank:].conj(),
            delta_over_omega=span / omega_rms if omega_rms > 0.0 else math.inf,
            theta_be_deg=theta,
            raman_zero_margin_hz=margin,
        )

    # ---- W(Delta) and the cooling coefficients ---------------------------------------------------------------

    def scattering_rate_per_s(self) -> float:
        """W(Delta): the total photon scattering rate of the steady state, s^-1."""
        return self.steadystate().total_photon_rate_per_s

    def force_operator(self, mode: ModeSpec) -> qt.Qobj:
        """F = sum_b eta_b (i/2)(O_b - O_b^dag): the first-order motional coupling, eta inside."""
        b = self.build
        if b.space is not None or not b.static:
            raise NotImplementedError("the force operator is built on the static internal-only model")
        n = b.n_internal
        f = np.zeros((n, n), dtype=complex)
        for beam_index, beam in enumerate(self.beams):
            eta = mode.eta(tuple(float(x) for x in beam.k_vector()))
            if eta == 0.0:
                continue
            o = np.zeros((n, n), dtype=complex)
            for c in b.couplings:
                if c.beam == beam_index:
                    o[b.index(c.upper), b.index(c.lower)] += c.omega_rad_s
            f += eta * 0.5j * (o - o.conj().T)
        return qt.Qobj(f, dims=b.H.dims)


@dataclass(frozen=True)
class SpectrumCoefficients:
    """A_+- = 2 Re[S(-+ nu) + D] with eta^2 INSIDE (the RMP form); ``two_D_per_s`` is the emission recoil diffusion."""

    A_plus_eta2_per_s: float
    A_minus_eta2_per_s: float
    two_D_per_s: float
    spectrum_plus_nu_per_s: float
    spectrum_minus_nu_per_s: float

    @property
    def nbar(self) -> float:
        if self.A_minus_eta2_per_s <= self.A_plus_eta2_per_s:
            raise CoolingError("A_- <= A_+: no cooling steady state")
        return self.A_plus_eta2_per_s / (self.A_minus_eta2_per_s - self.A_plus_eta2_per_s)

    @property
    def cooling_rate_per_s(self) -> float:
        return self.A_minus_eta2_per_s - self.A_plus_eta2_per_s


def rate_coefficients_from_spectrum(model: BlochModel, mode: ModeSpec) -> SpectrumCoefficients:
    """The steady state, the force-fluctuation spectrum at -+nu and the recoil D of a static internal-only model (recoil
    off): one Lamb-Dicke parameter per decay line from its own wavenumber, one alpha per channel and mode axis."""
    b = model.build
    if b.space is not None or not b.static:
        raise NotImplementedError("the spectrum path runs on the static internal-only model")
    assert isinstance(b.H, qt.Qobj)
    rho = steady_state_direct(b.H, b.c_ops)
    f = model.force_operator(mode)
    df = f - float(np.real(qt.expect(f, rho)))
    s = spectrum_es(b.H, b.c_ops, np.array([mode.omega_rad_s, -mode.omega_rad_s]), df, df, rho_ss=rho)
    s_plus, s_minus = float(np.real(s[0])), float(np.real(s[1]))
    two_d = emission_diffusion_two_d(model, mode.axis, mode.x0_m, rho)
    return SpectrumCoefficients(s_plus + two_d, s_minus + two_d, two_d, s_plus, s_minus)


VECTOR_FORM_Q = 9
"""``EmissionChannel.operator_qs`` sentinel of a vector-form (direction-resolved) channel, which mixes q."""


def operator_angular_factor(channel: EmissionChannel, index: int, cos_chi: float) -> float:
    """alpha of the collapse operator at offset ``index`` of ``channel``: alpha_q(chi) for a q-resolved operator, the
    channel's own tabulated alpha for a vector-form one with a single polarization index; a genuinely q-mixing channel has
    no scalar alpha and is refused rather than given the isotropic 1/3."""
    q = channel.operator_qs[index]
    if q != VECTOR_FORM_Q:
        return angular_factor(q, cos_chi)
    if len(channel.alpha) == 1:
        return float(next(iter(channel.alpha.values())))
    raise NotImplementedError(
        f"channel {channel.lower}<-{channel.upper} is direction-resolved over {sorted(channel.alpha)} polarization "
        "indices, so no scalar alpha describes one of its operators; its recoil is already exact in the operators "
        "themselves: use the level-C solve"
    )


def _static_steady_state(model: BlochModel, what: str) -> qt.Qobj:
    b = model.build
    if b.space is not None or not b.static:
        raise NotImplementedError(f"the {what} needs the static internal-only steady state")
    assert isinstance(b.H, qt.Qobj)
    return steady_state_direct(b.H, b.c_ops)


def emission_diffusion_two_d(
    model: BlochModel, axis: Sequence[float], x0_m: float, rho: qt.Qobj | None = None
) -> float:
    """2D = sum_channels alpha_q(chi) (k_em x0)^2 Gamma_q rho_ee: the emission recoil heating rate into a mode of axis
    ``axis`` and zero-point length ``x0`` (participation folded into x0), in the steady state by default."""
    b = model.build
    rho = _static_steady_state(model, "emission diffusion") if rho is None else rho
    cos_chi = float(np.dot(np.asarray(axis, dtype=float), model.structure.b_hat))
    rates = model.operator_rates(rho)
    two_d = 0.0
    for ch in b.channels:
        if ch.kind == "sink":
            continue
        eta_em = ch.wavenumber_rad_per_m * x0_m
        start, stop = ch.operator_slice
        for k in range(start, stop):
            two_d += operator_angular_factor(ch, k - start, cos_chi) * eta_em**2 * float(rates[k])
    return two_d


def emission_angular_factor(model: BlochModel, axis: Sequence[float]) -> float:
    """The photon-rate-weighted mean alpha over the emitted polarization channels of the steady state: the one scalar a
    Fock-space recoil kernel can carry. Raises CoolingError when nothing scatters."""
    b = model.build
    rho = _static_steady_state(model, "emission angular factor")
    cos_chi = float(np.dot(np.asarray(axis, dtype=float), model.structure.b_hat))
    rates = model.operator_rates(rho)
    total = 0.0
    weighted = 0.0
    for ch in b.channels:
        if ch.kind == "sink":
            continue
        start, stop = ch.operator_slice
        for k in range(start, stop):
            total += float(rates[k])
            weighted += float(rates[k]) * operator_angular_factor(ch, k - start, cos_chi)
    if total <= 0.0:
        raise CoolingError(
            "the decay lines scatter no photons in this steady state: no emission angular factor exists"
        )
    return weighted / total
