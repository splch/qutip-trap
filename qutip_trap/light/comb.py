"""Frequency-comb (mode-locked) Raman drives (PLAN.md Section 4.3.7; Appendix E, Run 5 additions; milestone M2).

Conventions (Section 13, comb rows): every rf and laser input is an ORDINARY frequency (Hz) and every
Hamiltonian coefficient angular; ``tau_s`` is the sech FIELD-envelope parameter of sech(pi t/(2 tau)),
never a FWHM (field FWHM 1.6768 tau, intensity FWHM 1.1222 tau; ``check_critique_v3.py``); a single tooth
counted from the carrier carries sech(2 pi k nu_rep tau), a tooth PAIR indexed by separation l carries
sech(pi l nu_rep tau), and the comb Stark sum sech^2((j + k) pi nu_rep tau); ``carrier_order`` is a signed
tooth-index difference; ``aom_offset_hz`` is signed and its sign selects red versus blue.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Tone
    from qutip_trap.species.model import Level

M2 = "milestone M2 (light/comb.py, PLAN.md Section 4.3.7)"


@dataclass(frozen=True)
class CombSpec:
    """The mode-locked train that generates a Drive's tone set."""

    rep_rate_hz: float
    """nu_rep, ordinary Hz; the tooth spacing."""
    tau_s: float
    """sech ENVELOPE parameter, never a FWHM."""
    tau_convention: Literal["field_sech", "intensity_sech", "intensity_sech2"]
    carrier_order: int
    """j = round(omega_HF / omega_rep), signed."""
    aom_offset_hz: float
    """Delta_nu_M = nu_1 - nu_2, SIGNED; its sign selects red vs blue."""
    pair_order_max: int = 1200
    chain: Literal["I", "II"] = "I"
    """Omega = g^2/(2 Delta) (I) versus g^2/Delta (II)."""
    i_sat_w_m2: float | None = None
    """None = derive pi h c Gamma_partial/(3 lambda^3); a value here is the lumped D1 convention of chain II."""
    rep_rate_trajectory: Callable[[float], float] | None = None
    locked_tooth: int | None = None

    def __post_init__(self) -> None:
        if self.rep_rate_hz <= 0.0 or self.tau_s <= 0.0:
            raise ValueError("rep_rate_hz and tau_s must be positive")
        if self.pair_order_max < 1000:
            raise ValueError(
                "the C_{n,a} Stark sum needs pair_order_max >~ 1000 (Section 4.3.7); partial sums plateau falsely"
            )
        # chain and i_sat travel together: chain I with the derived pi h c Gamma/(3 lambda^3), chain II with the
        # source's lumped value; the cross-product counts the D1 line strength twice (Appendix E).
        if self.chain == "I" and self.i_sat_w_m2 is not None:
            raise ValueError("chain I derives I_sat from the transition; do not pass the lumped i_sat_w_m2")
        if self.chain == "II" and self.i_sat_w_m2 is None:
            raise ValueError("chain II needs the source's lumped i_sat_w_m2")

    def pair_weight(self, l: int) -> float:  # noqa: E741  (l is the tooth-separation index of Section 4.3.7)
        """sech(pi l nu_rep tau) for a tooth PAIR at separation l; NOT sech(2 pi l ...) (Section 13)."""
        return 1.0 / math.cosh(math.pi * l * self.rep_rate_hz * self.tau_s)

    def beat_note_hz(self, order: int) -> float:
        """|order * nu_rep + aom_offset_hz|."""
        return abs(order * self.rep_rate_hz + self.aom_offset_hz)

    def tooth_amplitudes(self, n_teeth: int) -> np.ndarray:
        """g_k/g_0, renormalized to the sum rule sum_k g_k^2 = g_0^2."""
        raise NotImplementedError(f"CombSpec.tooth_amplitudes is {M2}")

    def tones(self, omega_q_hz: float, mode_hz: float | None) -> tuple[Tone, ...]:
        raise NotImplementedError(f"CombSpec.tones is {M2}")

    def solve_offset(self, target_hz: float) -> tuple[int, float]:
        """Root-find (j, Delta_nu_M) against a carrier or sideband target over signed integer j."""
        raise NotImplementedError(f"CombSpec.solve_offset is {M2}")

    def stark4_hz(self, levels: Sequence[Level]) -> np.ndarray:
        """The single l-sum of Section 4.3.7; guards omega_a != l(2 pi nu_rep), never j != 0."""
        raise NotImplementedError(f"CombSpec.stark4_hz is {M2}")

    def guards(
        self, omega_q_hz: float, eta: float, nbar: float, t_train_s: float, mode_hz: float
    ) -> dict[str, bool]:
        """The validity hierarchy: teeth_resolved, bandwidth_straddles_qubit, adiabatic_elimination, ..."""
        raise NotImplementedError(f"CombSpec.guards is {M2}")


__all__ = ["CombSpec"]
