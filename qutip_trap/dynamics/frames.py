"""Interaction pictures and phase bookkeeping between pulses (PLAN.md Sections 4.3.4, 5.2, 7.6; milestone M2).

Sign rules restated (Section 13):

- "Virtual-Z propagation": RZ(theta) on qubit i shifts the phase of every LATER pulse on i by phi -> phi - theta with
  the gates read in time order (Section 7.6; the documentation's +theta holds in matrix order); the frame at the end
  of a schedule is what ``Schedule.phase_frame`` stores and what the measurement discards.
- "Spin and motion phases": phi_s = (phi_b + phi_r)/2 and phi_m = (phi_b - phi_r)/2 of the two tone phases (Haljan's
  sign; Lee writes the opposite), with the force axis a further pi/2 from the carrier axis at the same beat-note
  phase (the i of the sideband coupling), a constant the frame alignment of Section 7.5 absorbs.
- The interaction picture with respect to H_0 = sum_m omega_m a_m^dag a_m replaces sigma_+ (x) prod_m D_m by
  sum_k e^{i k . omega t} sigma_+ (x) A_k over the sideband operators A_k (the elements of D with n' - n = k per
  mode); it is the option ``frame="interaction"`` of the builder, whose useful form truncates |k_m| <= k_max and
  reports the dropped weight sum_{|k| > k_max} ||A_k|| (Section 5.2); the default stays the Schroedinger picture.
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
        """RZ(theta) on ``qubit``: later pulses on it carry phi -> phi - theta (time order, Section 7.6)."""
        new = dict(self.offsets_rad)
        new[qubit] = new.get(qubit, 0.0) + float(theta_rad)
        return PhaseFrame(new)

    def pulse_phase(self, qubit: int, phi_program_rad: float) -> float:
        """The phase the hardware plays for a gate programmed at phi on ``qubit``."""
        return float(phi_program_rad) - self.offset(qubit)

    def as_dict(self, n_qubits: int) -> dict[int, float]:
        return {q: self.offset(q) for q in range(n_qubits)}


def spin_phase_rad(phi_blue_rad: float, phi_red_rad: float) -> float:
    """phi_s = (phi_b + phi_r)/2 (Haljan 2005; Section 13)."""
    return 0.5 * (phi_blue_rad + phi_red_rad)


def motion_phase_rad(phi_blue_rad: float, phi_red_rad: float) -> float:
    """phi_m = (phi_b - phi_r)/2 (Section 13)."""
    return 0.5 * (phi_blue_rad - phi_red_rad)


def force_axis_rad(
    phi_blue_rad: float, phi_red_rad: float, *, same_delta_k: bool = True, geometric_rad: float = 0.0
) -> float:
    """The azimuth of the spin-dependent force: pi/2 + Delta k . X + phi_s in the same-Delta k geometry, phi_s + pi/2 + pi in
    the crossed geometry where eta_r = -eta_b adds a further pi and the position term cancels (Section 4.3.4)."""
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
    """sum over dropped combinations of the spectral norm of A_k (Section 5.2)."""


def interaction_picture(
    space: HilbertSpace,
    ion: int,
    etas: Mapping[int, float],
    *,
    k_max: int | None = None,
    matrices: Mapping[int, np.ndarray] | None = None,
    ion_op: qt.Qobj | None = None,
) -> InteractionPicture:
    """Decompose sigma_+^ion (x) prod_m D_m(i eta_m) over the RESOLVED modes into sideband operators A_k.

    ``matrices`` may supply the per-mode single-mode matrices (default: the space's expm displacement); a mode with
    eta = 0 contributes only k = 0 (the identity). The ENR group is not supported in this picture (Section 5.1.1: an
    ENR displacement cannot be applied mode by mode). ``ion_op`` (embedded in the joint space) replaces sigma_+^ion,
    for the level projectors of a light-shift drive (Section 4.4.4).
    """
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
