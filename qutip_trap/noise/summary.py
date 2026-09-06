"""Reporting a simulated channel to circuit-level clients (PLAN.md Section 6.8; Section 13 rows "Depolarizing normalization",
"RB error rate", "Rotation generators"; M7).

Phenomenological channels are never used inside the simulator; they are what it produces when asked to summarize a
simulated pulse. Three summaries of a channel given as a Choi matrix (or a unitary):

- the entanglement infidelity 1 - F_e with F_e = <Phi| (E (x) 1)(|Phi><Phi|) |Phi> for the maximally entangled state, and
  the average gate infidelity d/(d + 1) (1 - F_e) (Nielsen 2002);
- the Pauli twirl: the diagonal of the Pauli transfer matrix turned into the probabilities p_P of the twirled Pauli channel
  (exp(-i alpha XX) twirls to p_II = cos^2 alpha, p_XX = sin^2 alpha: Trout 2018's p_xx = sin^2 alpha, the two-qubit case
  only because XX(chi) carries chi with no 1/2, Section 13);
- the depolarizing rate epsilon with Lambda_eps(rho) = (1 - eps) rho + eps/(4^n - 1) sum_{P != I} P rho P, whose entanglement
  infidelity is exactly eps (Chen et al. 2023), so the reported rate IS the entanglement infidelity.

Also the randomized-benchmarking conversions of Section 13: r = (1 - p)(2^n - 1)/2^n is the average error per Clifford that
RB papers report, and (4^n - 1)(1 - p)/4^n the entanglement infidelity of the depolarizing channel with the same p. Bermudez
et al. 2017's three-variant mapping of epsilon onto Pauli channels is NOT transcribed in the plan and is not provided here.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_string(label: str) -> np.ndarray:
    """The tensor product of single-qubit Paulis named by ``label`` ('IX', 'XX', ...), qubit 0 the FIRST factor."""
    out = np.array([[1.0 + 0.0j]])
    for ch in label:
        out = np.kron(out, _PAULI[ch])
    return out


def pauli_labels(n_qubits: int) -> list[str]:
    return ["".join(p) for p in itertools.product("IXYZ", repeat=n_qubits)]


def choi_from_unitary(u: np.ndarray) -> np.ndarray:
    """The Choi state (trace 1) of the unitary channel: |Phi_U><Phi_U| with |Phi_U> = (1 (x) U)|Phi>, |Phi> = sum_i |ii>/sqrt d."""
    d = u.shape[0]
    phi = np.zeros(d * d, dtype=complex)
    for i in range(d):
        e = np.zeros(d, dtype=complex)
        e[i] = 1.0
        phi += np.kron(e, u @ e)
    phi /= math.sqrt(d)
    return np.asarray(np.outer(phi, phi.conj()))


def choi_from_kraus(kraus: list[np.ndarray]) -> np.ndarray:
    """The trace-1 Choi state of the channel with the given Kraus operators."""
    d = kraus[0].shape[0]
    out = np.zeros((d * d, d * d), dtype=complex)
    for k in kraus:
        phi = np.zeros(d * d, dtype=complex)
        for i in range(d):
            e = np.zeros(d, dtype=complex)
            e[i] = 1.0
            phi += np.kron(e, k @ e)
        out += np.outer(phi, phi.conj())
    return np.asarray(out / d)


def entanglement_fidelity(choi: np.ndarray, choi_ideal: np.ndarray) -> float:
    """F_e = Tr(C_ideal C) for a pure ideal Choi state (a unitary target)."""
    return float(np.real(np.trace(choi_ideal @ choi)))


def entanglement_infidelity(choi: np.ndarray, choi_ideal: np.ndarray) -> float:
    return 1.0 - entanglement_fidelity(choi, choi_ideal)


def average_gate_infidelity(eps_entanglement: float, d: int) -> float:
    """1 - F_avg = d/(d + 1) (1 - F_e) (Nielsen 2002); 4/5 for two qubits (Landsman's conversion, Section 4.4.7 (8))."""
    return d / (d + 1.0) * eps_entanglement


def apply_choi(choi: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """E(rho) from the trace-1 Choi state: E(rho) = d Tr_1[(rho^T (x) 1) C]."""
    d = rho.shape[0]
    c = choi.reshape(d, d, d, d)  # (i, k ; j, l) with the first factor the input copy
    return np.asarray(d * np.einsum("ij,ikjl->kl", rho, c))


def pauli_transfer_diagonal(choi: np.ndarray, n_qubits: int) -> dict[str, float]:
    """R_PP = Tr[P E(P)]/d: the diagonal of the Pauli transfer matrix."""
    d = 2**n_qubits
    out: dict[str, float] = {}
    for lab in pauli_labels(n_qubits):
        p = pauli_string(lab)
        out[lab] = float(np.real(np.trace(p @ apply_choi(choi, p))) / d)
    return out


def pauli_twirl(choi: np.ndarray, n_qubits: int) -> dict[str, float]:
    """Probabilities p_Q of the Pauli-twirled channel sum_Q p_Q Q rho Q from the transfer-matrix diagonal:
    p_Q = (1/4^n) sum_P R_PP chi(P, Q) with chi(P, Q) = +1 when P and Q commute and -1 when they anticommute."""
    diag = pauli_transfer_diagonal(choi, n_qubits)
    labels = pauli_labels(n_qubits)
    mats = {lab: pauli_string(lab) for lab in labels}
    out: dict[str, float] = {}
    for q in labels:
        total = 0.0
        for p in labels:
            comm = np.allclose(mats[p] @ mats[q], mats[q] @ mats[p])
            total += diag[p] * (1.0 if comm else -1.0)
        out[q] = float(total / 4**n_qubits)
    return out


def depolarizing_choi(eps: float, n_qubits: int) -> np.ndarray:
    """The Choi state of Lambda_eps(rho) = (1 - eps) rho + eps/(4^n - 1) sum_{P != I} P rho P (Chen 2023's normalization)."""
    d = 2**n_qubits
    labels = pauli_labels(n_qubits)
    kraus = [math.sqrt(1.0 - eps) * np.eye(d, dtype=complex)]
    kraus += [math.sqrt(eps / (4**n_qubits - 1)) * pauli_string(lab) for lab in labels if set(lab) != {"I"}]
    return choi_from_kraus(kraus)


def depolarizing_rate(choi: np.ndarray, choi_ideal: np.ndarray) -> float:
    """The depolarizing rate epsilon of Section 6.8: the entanglement infidelity, by Chen et al.'s normalization."""
    return entanglement_infidelity(choi, choi_ideal)


def rb_error_per_clifford(p: float, n_qubits: int) -> float:
    """r = (1 - p)(2^n - 1)/2^n, the average error per Clifford an RB fit reports (Magesan 2011; Section 13)."""
    return float((1.0 - p) * (2**n_qubits - 1) / 2**n_qubits)


def depolarizing_entanglement_infidelity(p: float, n_qubits: int) -> float:
    """(4^n - 1)(1 - p)/4^n: the entanglement infidelity of the depolarizing channel with the same p (Section 13)."""
    return float((4**n_qubits - 1) * (1.0 - p) / 4**n_qubits)


def over_rotation_twirl_probability(alpha_rad: float) -> float:
    """p_xx = sin^2 alpha for the over-rotation exp(-i alpha XX) (Trout 2018; Section 9.7)."""
    return math.sin(alpha_rad) ** 2


__all__ = [
    "apply_choi",
    "average_gate_infidelity",
    "choi_from_kraus",
    "choi_from_unitary",
    "depolarizing_choi",
    "depolarizing_entanglement_infidelity",
    "depolarizing_rate",
    "entanglement_fidelity",
    "entanglement_infidelity",
    "over_rotation_twirl_probability",
    "pauli_labels",
    "pauli_string",
    "pauli_transfer_diagonal",
    "pauli_twirl",
    "rb_error_per_clifford",
]
