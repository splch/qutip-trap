"""The preparation sequence and its hand-off state.

Stage order is enforced: Doppler first (it scrambles the internal state), Doppler before every sub-Doppler stage, and a
pump that a later cooling stage scrambles must be followed by another pump.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import qutip as qt

from qutip_trap.dynamics.engine import State
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.prep.pumping import LeakToQubit, PumpingResult
from qutip_trap.prep.rates import StageRates

StageKind = Literal["doppler", "sideband", "eit", "polarization_gradient", "pulsed_sideband", "pump"]
COOLING_KINDS: frozenset[str] = frozenset(
    {"doppler", "sideband", "eit", "polarization_gradient", "pulsed_sideband"}
)
SUB_DOPPLER_KINDS: frozenset[str] = frozenset({"sideband", "eit", "polarization_gradient", "pulsed_sideband"})


class StageOrderError(ValueError):
    """The preparation sequence violates the stage order (Doppler, then sub-Doppler cooling, then the final pump)."""


@dataclass(frozen=True)
class PreparationStage:
    """One stage of the sequence with what it produced."""

    kind: StageKind
    provenance: str
    """The stage id recorded in ``State.provenance``."""
    nbar: Mapping[int, float] | None = None
    """Per-mode mean occupations a cooling stage leaves (None for a pump)."""
    ions: tuple[int, ...] = ()
    """The ions the stage acted on (the coolant for sympathetic cooling; the pumped ions for a pump)."""
    pump: PumpingResult | None = None
    rates: StageRates | None = None
    motional: Mapping[int, qt.Qobj] | None = None
    """Optional per-mode motional states (level-B Fock or level-C reduced) replacing the thermal state at ``nbar``."""

    def __post_init__(self) -> None:
        if self.kind == "pump" and self.pump is None:
            raise ValueError("a pump stage carries its PumpingResult")
        if self.kind in COOLING_KINDS and self.nbar is None:
            raise ValueError("a cooling stage carries its per-mode nbar")
        if self.motional is not None:
            if self.nbar is None:
                raise ValueError("a stage with explicit motional states also carries their nbar")
            missing = [m for m in self.motional if m not in self.nbar]
            if missing:
                raise ValueError(
                    f"modes {sorted(missing)} have an explicit motional state but no nbar: the hand-off needs both "
                    "(Section 4.2.7)"
                )


def doppler_stage(rates: StageRates, provenance: str = "prep.doppler") -> PreparationStage:
    return PreparationStage("doppler", provenance, nbar=rates.nbar, ions=rates.illuminated, rates=rates)


def sideband_stage(
    nbar: Mapping[int, float],
    ions: Sequence[int],
    provenance: str = "prep.sideband",
    rates: StageRates | None = None,
    motional: Mapping[int, qt.Qobj] | None = None,
) -> PreparationStage:
    return PreparationStage(
        "sideband",
        provenance,
        nbar=dict(nbar),
        ions=tuple(ions),
        rates=rates,
        motional=None if motional is None else dict(motional),
    )


def fock_distribution_state(populations: Sequence[float] | np.ndarray) -> qt.Qobj:
    """The normalized diagonal density matrix of a level-B Fock distribution."""
    p = np.asarray(populations, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("a Fock distribution is a one-dimensional array of at least two populations")
    if np.any(p < 0.0):
        raise ValueError("Fock populations are non-negative")
    total = float(p.sum())
    if total <= 0.0:
        raise ValueError("a Fock distribution has positive total probability")
    d = int(p.size)
    return qt.Qobj(np.diag(p / total), dims=[[d], [d]])


def eit_stage(
    nbar: Mapping[int, float], ions: Sequence[int], provenance: str = "prep.eit"
) -> PreparationStage:
    return PreparationStage("eit", provenance, nbar=dict(nbar), ions=tuple(ions))


def pulsed_sideband_stage(
    nbar: Mapping[int, float],
    ions: Sequence[int],
    provenance: str = "prep.sideband.pulsed",
    motional: Mapping[int, qt.Qobj] | None = None,
) -> PreparationStage:
    return PreparationStage(
        "pulsed_sideband",
        provenance,
        nbar=dict(nbar),
        ions=tuple(ions),
        motional=None if motional is None else dict(motional),
    )


def pump_stage(
    result: PumpingResult, ions: Sequence[int], provenance: str = "prep.pumping"
) -> PreparationStage:
    return PreparationStage("pump", provenance, ions=tuple(ions), pump=result)


@dataclass(frozen=True)
class PreparationSequence:
    """The ordered stages of one preparation; ``validate`` enforces the stage order."""

    stages: tuple[PreparationStage, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.stages:
            raise StageOrderError("a preparation sequence has at least one stage")
        kinds = [s.kind for s in self.stages]
        # three rules, each checked separately so that none is only implied by another
        if kinds[0] == "pump":
            raise StageOrderError(
                "the sequence starts with a pump: Doppler cooling comes first and scrambles the internal state, so a "
                "pump before it is erased (Section 4.2.6)"
            )
        for k, kind in enumerate(kinds):
            if kind in SUB_DOPPLER_KINDS and "doppler" not in kinds[:k]:
                raise StageOrderError(
                    f"stage {k} is the sub-Doppler stage {kind!r} with no Doppler cooling before it (Section 4.2.6): "
                    "a sub-Doppler stage starts from the Doppler-cooled occupations"
                )
            if kind == "pump" and any(later in COOLING_KINDS for later in kinds[k + 1 :]):
                if not any(later == "pump" for later in kinds[k + 1 :]):
                    raise StageOrderError(
                        f"stage {k} pumps before a later cooling stage scrambles the internal state again and no pump follows "
                        "(Section 4.2.6: the final optical pump comes immediately before the circuit)"
                    )

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(s.kind for s in self.stages)

    @property
    def ends_with_pump(self) -> bool:
        return self.stages[-1].kind == "pump"

    def final_nbar(self) -> dict[int, float]:
        """Per mode, the occupation of the last cooling stage that addressed it."""
        out: dict[int, float] = {}
        for stage in self.stages:
            if stage.nbar is not None:
                out.update({int(k): float(v) for k, v in stage.nbar.items()})
        return out

    def final_motional(self) -> dict[int, qt.Qobj]:
        """Per mode, the explicit motional state of the last stage that gave one, unless a later stage re-cooled it."""
        out: dict[int, qt.Qobj] = {}
        for stage in self.stages:
            if stage.nbar is None:
                continue
            for m in stage.nbar:
                out.pop(int(m), None)
            for m, rho in (stage.motional or {}).items():
                out[int(m)] = rho
        return out

    def final_pump(self, ion: int) -> PumpingResult | None:
        """The last pump that acted on ``ion`` (None if never pumped)."""
        for stage in reversed(self.stages):
            if stage.kind == "pump" and (not stage.ions or ion in stage.ions):
                return stage.pump
        return None

    def provenance(self) -> tuple[str, ...]:
        return tuple(s.provenance for s in self.stages)


def prepare_state(
    space: HilbertSpace,
    sequence: PreparationSequence,
    *,
    qubit_labels: tuple[str, str] | Mapping[int, tuple[str, str]],
    leak: LeakToQubit = "to_upper",
    internal: Mapping[int, qt.Qobj] | None = None,
    extra_nbar: Mapping[int, float] | None = None,
    levels: Mapping[int, Sequence[str]] | None = None,
) -> State:
    """The ``State`` after the sequence: modes at the final nbar (plus ``extra_nbar``) and each ion's last pumped state
    (``internal`` for ions never pumped, ``levels`` for register factors with d > 2)."""
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
    for (
        ion
    ) in space.ion_labels:  # device ions, in factor order (a space over a subset of the crystal names them)
        d = space.ion_dim(ion)
        pump = sequence.final_pump(ion)
        if pump is not None:
            labels = qubit_labels[ion] if isinstance(qubit_labels, Mapping) else qubit_labels
            if d != 2:
                if levels is None or ion not in levels or len(levels[ion]) != d:
                    raise ValueError(
                        f"ion {ion} has a register factor of dimension {d}: pass its level labels (noise/levels.py, M7)"
                    )
                parts.append(pump.qudit_density_matrix(levels[ion]))
            else:
                parts.append(pump.qubit_density_matrix(labels, leak=leak))
        elif internal is not None and ion in internal:
            rho = internal[ion]
            parts.append(rho if rho.isoper else qt.ket2dm(rho))
        else:
            raise ValueError(
                f"ion {ion} was never pumped and no explicit internal state was given (Section 4.2.6: the sequence ends with a pump)"
            )
    rho_int = parts[0] if len(parts) == 1 else qt.tensor(*parts)
    explicit = {
        m: rho for m, rho in sequence.final_motional().items() if m in {r.mode for r in space.resolved}
    }
    for m, rho in explicit.items():
        d = space.truncation(m).d
        if rho.shape != (d, d):
            raise ValueError(
                f"mode {m}'s explicit motional state has dimension {rho.shape[0]}, the space's factor {d} "
                "(Section 4.2.7: the hand-off state lives on the space's own truncation)"
            )
    return space.initial_state(
        rho_int, thermal=nbar, states=explicit or None, provenance=sequence.provenance()
    )
