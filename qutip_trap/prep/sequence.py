"""The preparation sequence and its hand-off to the initial ``State`` (PLAN.md Section 4.2).

Doppler cooling scrambles the internal state, so the stages run Doppler -> pulsed sideband -> a final optical pump of
every ion. The hand-off state is a product of per-mode thermal states at the mean occupation of the LAST stage that
addressed each mode, and each ion's pumped internal state; each stage's provenance id is recorded in ``State``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import qutip as qt

from qutip_trap.dynamics.engine import State
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.prep.doppler import DopplerResult
from qutip_trap.prep.pumping import PumpingResult

StageKind = Literal["doppler", "pulsed_sideband", "pump"]


@dataclass(frozen=True)
class PreparationStage:
    """One stage of the sequence with what it produced."""

    kind: StageKind
    provenance: str
    nbar: Mapping[int, float] | None = None
    """Per-mode mean occupations a cooling stage leaves (None for a pump)."""
    ions: tuple[int, ...] = ()
    """The ions the stage acted on."""
    pump: PumpingResult | None = None


def doppler_stage(result: DopplerResult, provenance: str = "prep.doppler") -> PreparationStage:
    return PreparationStage("doppler", provenance, nbar=result.nbar, ions=result.illuminated)


def pulsed_sideband_stage(
    nbar: Mapping[int, float], ions: Sequence[int], provenance: str = "prep.sideband.pulsed"
) -> PreparationStage:
    return PreparationStage("pulsed_sideband", provenance, nbar=dict(nbar), ions=tuple(ions))


def pump_stage(
    result: PumpingResult, ions: Sequence[int], provenance: str = "prep.pumping"
) -> PreparationStage:
    return PreparationStage("pump", provenance, ions=tuple(ions), pump=result)


@dataclass(frozen=True)
class PreparationSequence:
    """The ordered stages of one preparation."""

    stages: tuple[PreparationStage, ...]

    def final_nbar(self) -> dict[int, float]:
        """Per mode, the occupation of the LAST cooling stage that addressed it."""
        out: dict[int, float] = {}
        for stage in self.stages:
            if stage.nbar is not None:
                out.update({int(k): float(v) for k, v in stage.nbar.items()})
        return out

    def final_pump(self, ion: int) -> PumpingResult | None:
        """The last pump that acted on ``ion`` (None if never pumped)."""
        for stage in reversed(self.stages):
            if stage.kind == "pump" and ion in stage.ions:
                return stage.pump
        return None

    def provenance(self) -> tuple[str, ...]:
        return tuple(s.provenance for s in self.stages)


def prepare_state(
    space: HilbertSpace,
    sequence: PreparationSequence,
    *,
    qubit_labels: tuple[str, str],
    extra_nbar: Mapping[int, float] | None = None,
    levels: Mapping[int, Sequence[str]] | None = None,
) -> State:
    """The ``State`` after the sequence: per-mode thermal states at the final nbar plus ``extra_nbar`` (the pumps' recoil
    heating), and each ion's internal state from its last pump; ``levels`` gives the register labels of every ion whose
    factor has d > 2 (the pumped populations land on the resolved sublevels and the remainder in the SINK)."""
    nbar = sequence.final_nbar()
    for m, dn in (extra_nbar or {}).items():
        if int(m) in nbar:
            nbar[int(m)] = nbar[int(m)] + float(dn)
    missing = [m.mode for m in space.resolved if m.mode not in nbar] + [
        m for m in space.frozen if m not in nbar
    ]
    if space.enr_group is not None:
        missing += [m for m in space.enr_group[0] if m not in nbar]
    if missing:
        raise ValueError(
            f"no cooling stage addressed modes {sorted(set(missing))}: their occupation is undefined"
        )
    parts: list[qt.Qobj] = []
    for ion in space.ion_labels:  # device ions, in factor order
        d = space.ion_dim(ion)
        pump = sequence.final_pump(ion)
        if pump is None:
            raise ValueError(f"ion {ion} was never pumped: the sequence ends with a pump of every ion")
        if d == 2:
            parts.append(pump.qubit_density_matrix(qubit_labels))
            continue
        if levels is None or ion not in levels or len(levels[ion]) != d:
            raise ValueError(f"ion {ion} has a register factor of dimension {d}: pass its level labels")
        parts.append(pump.qudit_density_matrix(levels[ion]))
    rho_int = parts[0] if len(parts) == 1 else qt.tensor(*parts)
    return space.initial_state(rho_int, thermal=nbar, provenance=sequence.provenance())
