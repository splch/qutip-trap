"""Interaction pictures and phase bookkeeping between pulses.

- Virtual Z: RZ(theta) on qubit i shifts the phase of every later pulse on i by phi -> phi - theta, gates read in time
  order (+theta in matrix order); the frame left at the end of a schedule is discarded by the measurement.
- Tone phases: phi_s = (phi_b + phi_r)/2 and phi_m = (phi_b - phi_r)/2 (Haljan's sign); the force axis sits a further
  pi/2 from the carrier axis at the same beat-note phase.
- The interaction picture w.r.t. H_0 = sum_m omega_m a_m^dag a_m replaces sigma_+ (x) prod_m D_m by
  sum_k e^{i k . omega t} sigma_+ (x) A_k, A_k the elements of D with n' - n = k per mode.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import qutip as qt

from qutip_trap.hilbert.operators import sideband_operators
from qutip_trap.hilbert.space import HilbertSpace


@dataclass(frozen=True)
class PhaseFrame:
    """The per-qubit virtual-Z frame: an offset theta_i that every later pulse phase on qubit i subtracts."""

    offsets_rad: dict[int, float] = field(default_factory=dict)

    def offset(self, qubit: int) -> float:
        return float(self.offsets_rad.get(qubit, 0.0))

    def rz(self, qubit: int, theta_rad: float) -> PhaseFrame:
        """RZ(theta) on ``qubit``: later pulses on it carry phi -> phi - theta."""
        new = dict(self.offsets_rad)
        new[qubit] = new.get(qubit, 0.0) + float(theta_rad)
        return PhaseFrame(new)

    def pulse_phase(self, qubit: int, phi_program_rad: float) -> float:
        """The phase the hardware plays for a gate programmed at phi on ``qubit``."""
        return float(phi_program_rad) - self.offset(qubit)

    def as_dict(self, n_qubits: int) -> dict[int, float]:
        return {q: self.offset(q) for q in range(n_qubits)}


def spin_phase_rad(phi_blue_rad: float, phi_red_rad: float) -> float:
    """phi_s = (phi_b + phi_r)/2 (Haljan 2005)."""
    return 0.5 * (phi_blue_rad + phi_red_rad)


def motion_phase_rad(phi_blue_rad: float, phi_red_rad: float) -> float:
    """phi_m = (phi_b - phi_r)/2."""
    return 0.5 * (phi_blue_rad - phi_red_rad)


def force_axis_rad(
    phi_blue_rad: float, phi_red_rad: float, *, same_delta_k: bool = True, geometric_rad: float = 0.0
) -> float:
    """The azimuth of the spin-dependent force: phi_s + pi/2 + Delta k . X in the same-Delta k geometry, phi_s + pi/2 + pi
    in the crossed geometry (eta_r = -eta_b adds pi and cancels the position term)."""
    base = spin_phase_rad(phi_blue_rad, phi_red_rad) + 0.5 * math.pi
    if same_delta_k:
        return base + geometric_rad
    return base + math.pi


@dataclass(frozen=True)
class SidebandTerm:
    """One sideband combination of the interaction picture: operator sigma_+ (x) A_k, its k-vector and its weight."""

    op: qt.Qobj
    k: tuple[int, ...]
    modes: tuple[int, ...]
    weight: float


@dataclass(frozen=True)
class InteractionPicture:
    terms: tuple[SidebandTerm, ...]
    dropped_weight: float
    """Sum of the spectral norms of the dropped A_k."""


def interaction_picture(
    space: HilbertSpace,
    ion: int,
    etas: Mapping[int, float],
    *,
    k_max: int | None = None,
    matrices: Mapping[int, np.ndarray] | None = None,
    ion_op: qt.Qobj | None = None,
) -> InteractionPicture:
    """Decompose sigma_+^ion (or the embedded ``ion_op``) (x) prod_m D_m(i eta_m) over the resolved modes into sideband
    operators A_k, dropping combinations with some |k_m| > ``k_max`` (their weight is reported). ``matrices`` may replace
    the expm displacements; an ENR mode with eta != 0 raises NotImplementedError (its displacement does not factor)."""
    if space.enr_group is not None and any(space.mode_class(m) == "enr" for m in etas if etas[m] != 0.0):
        raise NotImplementedError("the interaction picture is defined on product spaces only (Section 5.1.1)")
    modes = [m.mode for m in space.resolved]
    per_mode: list[dict[int, np.ndarray]] = []
    norms: list[dict[int, float]] = []
    for m in modes:
        eta = float(etas.get(m, 0.0))
        d = space.truncation(m).d
        if eta == 0.0:
            parts = {0: np.eye(d, dtype=complex)}
        else:
            mat = (
                matrices[m]
                if matrices is not None and m in matrices
                else space.displacement_factor(m, eta).full()
            )
            parts = sideband_operators(np.asarray(mat))
        per_mode.append(parts)
        norms.append({k: float(np.linalg.norm(v, 2)) for k, v in parts.items()})
    kept: list[SidebandTerm] = []
    dropped = 0.0
    sp = space.sigma_plus(ion) if ion_op is None else ion_op
    for combo in itertools.product(*[sorted(p) for p in per_mode]):
        weight = math.prod(norms[i][k] for i, k in enumerate(combo)) if combo else 1.0
        if k_max is not None and any(abs(k) > k_max for k in combo):
            dropped += weight
            continue
        ops = {
            space.mode_factor(m): qt.Qobj(per_mode[i][combo[i]], dims=[[space.truncation(m).d]] * 2)
            for i, m in enumerate(modes)
        }
        op = space.embed_many(ops) * sp
        kept.append(SidebandTerm(op=op.to("CSR"), k=tuple(combo), modes=tuple(modes), weight=weight))
    return InteractionPicture(terms=tuple(kept), dropped_weight=dropped)


def free_evolution_phase(
    joint: qt.Qobj, space: HilbertSpace, omegas_rad_s: Mapping[int, float], t_s: float
) -> qt.Qobj:
    """exp(-i H_0 t) applied to a joint state: the map from the interaction picture back to the Schroedinger picture."""
    h0 = 0.0 * space.identity()
    for m, w in omegas_rad_s.items():
        if space.mode_class(m) != "frozen":
            h0 = h0 + w * space.number(m)
    u = (-1j * t_s * h0).expm()
    if joint.isket:
        return u * joint
    return u * joint * u.dag()


def sideband_weights(etas: Sequence[float], d: int, k_max: int) -> dict[tuple[int, ...], float]:
    """Diagnostic: the spectral-norm weights of the sideband combinations kept by ``k_max`` for the given etas."""
    out: dict[tuple[int, ...], float] = {}
    per = []
    for eta in etas:
        mat = qt.displace(d, 1j * eta).full()
        per.append({k: float(np.linalg.norm(v, 2)) for k, v in sideband_operators(mat).items()})
    for combo in itertools.product(*[sorted(p) for p in per]):
        if all(abs(k) <= k_max for k in combo):
            out[tuple(combo)] = math.prod(per[i][k] for i, k in enumerate(combo))
    return out


__all__ = [
    "InteractionPicture",
    "PhaseFrame",
    "SidebandTerm",
    "force_axis_rad",
    "free_evolution_phase",
    "interaction_picture",
    "motion_phase_rad",
    "sideband_weights",
    "spin_phase_rad",
]
