"""Laser cooling and state preparation (PLAN.md Section 4.2; milestone M3 on the M3a Bloch builder).

Order (Section 4.2.6, enforced by ``prep.sequence``): Doppler cooling first, then sideband or EIT cooling with its own
repump, then a final optical pump immediately before the circuit. Every stage offers three levels of description
sharing one parameter set: (A) closed-form rate equations (``closed_forms``, ``rates`` with W from ``light/bloch.py``),
(B) the Lamb-Dicke birth-death equation on Fock populations and the exact pulsed transfer matrices (``level_c``,
``sideband``), (C) the master equation of the internal levels plus one mode (``level_c`` on the M3a builder).

- ``doppler``: nbar_D per mode from the rate framework at any nu/Gamma, one (Delta, s, k_hat) per beam optimized over the
  whole mode set, the force model as the nu << Gamma cross-check.
- ``sideband``: resolved-sideband and Raman sideband cooling, the exact pulsed schedules with higher sidebands, the
  repump recoil, and the thermometry of Section 4.2.7.
- ``eit``: the closed forms of Section 4.2.3 in the plan's sign and the Zeeman-resolved level-C model.
- ``polarization_gradient``: Joshi's analytic j = 1/2 <-> 1/2 model and the twelve-operator Lindblad layer of Section 4.2.4.
- ``pumping``: optical pumping with the preparation error, duration, photon count and recoil heating.
- ``sequence``: the stage-order guard and the hand-off to the Appendix E ``State``.
"""

from __future__ import annotations

from qutip_trap.prep.doppler import DopplerResult, doppler_cooling, optimize_detuning, with_detuning_offset
from qutip_trap.prep.eit import (
    EitClosedForm,
    eit_rate_coefficients,
    eit_steady_state_nbar,
    lambda_level_c_model,
)
from qutip_trap.prep.pumping import PumpingResult, optical_pumping
from qutip_trap.prep.rates import ModeRates, StageRates, UncooledModeError, stage_rates
from qutip_trap.prep.sequence import PreparationSequence, PreparationStage, StageOrderError, prepare_state
from qutip_trap.prep.sideband import SidebandPulse, apply_pulses, transfer_matrix


def sideband_cooling(*args: object, **kwargs: object) -> StageRates:
    """Continuous resolved-sideband cooling at level A: the rate framework of ``prep.rates`` with the cooling beam on a red sideband
    (Delta = -k nu) and the quench or repump beams in the same model (Section 4.2.2); returns the per-mode ``StageRates``."""
    stage, _models = stage_rates(*args, **kwargs)  # type: ignore[arg-type]
    return stage


def eit_cooling(*args: object, **kwargs: object) -> float:
    """The EIT closed form <n>_S of Section 4.2.3 (``eit.eit_steady_state_nbar``)."""
    return eit_steady_state_nbar(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "DopplerResult",
    "EitClosedForm",
    "ModeRates",
    "PreparationSequence",
    "PreparationStage",
    "PumpingResult",
    "SidebandPulse",
    "StageOrderError",
    "StageRates",
    "UncooledModeError",
    "apply_pulses",
    "doppler_cooling",
    "eit_cooling",
    "eit_rate_coefficients",
    "eit_steady_state_nbar",
    "lambda_level_c_model",
    "optical_pumping",
    "optimize_detuning",
    "prepare_state",
    "sideband_cooling",
    "stage_rates",
    "transfer_matrix",
    "with_detuning_offset",
]
