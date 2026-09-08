"""The preparation sequence and its hand-off (PLAN.md Sections 4.2, 4.2.6, 4.2.7, 5.7; Appendix E ``State``; M3).

Stage order is a constraint, not prose (Section 4.2.6 [corrected: critique, 2026-09-04]): Doppler cooling runs on a
repumped cycling transition and scrambles the internal state, so a pump done before it is erased; sideband, EIT and
polarization-gradient cooling run with their own repump and scramble it too; the final optical pump comes immediately
before the circuit. ``PreparationSequence`` therefore refuses a pump placed before a cooling stage without a later pump,
a sub-Doppler stage with no Doppler precooling before it, and a sequence that does not start with Doppler cooling.
The hand-off state is a product of per-mode thermal states with the mean occupations of the LAST stage that addressed
each mode, and the pumped internal state of every ion (Section 4.2.7); ``prepare_state`` builds the Appendix E
``State`` through ``HilbertSpace.initial_state`` with the stages' provenance ids; a stage that produced a level-B Fock
distribution or a level-C reduced motional state carries it in ``PreparationStage.motional`` and it reaches
``initial_state(states=...)`` unchanged instead of being collapsed to its mean (Section 4.2, 4.2.7). Sympathetic cooling (Section 4.2.5)
is the same bookkeeping with the coolant as the illuminated set: the shared modes take their nbar from the coolant's
stage and the qubit ions' internal states are whatever the (separate) pump of those ions produced.
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
    """The preparation sequence violates the Section 4.2.6 order (Doppler -> sideband/EIT -> final pump)."""


@dataclass(frozen=True)
class PreparationStage:
    """One stage of the sequence with what it produced."""

    kind: StageKind
    provenance: str
    """The ledger id or stage id recorded in ``State.provenance`` (Section 4.2.6 order)."""
    nbar: Mapping[int, float] | None = None
    """Per-mode mean occupations a cooling stage leaves (None for a pump)."""
    ions: tuple[int, ...] = ()
    """The ions the stage acted on (the coolant for sympathetic cooling; the pumped ions for a pump)."""
    pump: PumpingResult | None = None
    rates: StageRates | None = None
    motional: Mapping[int, qt.Qobj] | None = None
    """Optional per-mode motional density matrix, overriding ``nbar`` on those modes in the hand-off.

    Section 4.2: "The output of every stage is a per-mode thermal density matrix (OR THE FOCK DISTRIBUTION FROM
    LEVEL B)"; Section 4.2.7: the hand-off state is a product of per-mode thermal states "OR THE LEVEL-C REDUCED
    MOTIONAL STATE". A level-B Fock distribution (a diagonal matrix built from ``prep.sideband``'s populations or
    ``prep.level_c.phonon_generator``'s stationary vector) or a level-C reduced motional state goes here and reaches
    ``HilbertSpace.initial_state(states=...)`` unchanged; ``nbar`` is still required, because the frozen modes and
    the provenance bookkeeping read it, and it is what a mode NOT in this mapping falls back to."""

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
    """The diagonal density matrix of a level-B Fock distribution (``prep.sideband``'s populations, or the stationary
    vector of ``prep.level_c.phonon_generator``), normalized: the "Fock distribution from level B" of Section 4.2."""
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
    """The ordered stages of one preparation; ``validate`` enforces Section 4.2.6."""

    stages: tuple[PreparationStage, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.stages:
            raise StageOrderError("a preparation sequence has at least one stage")
        kinds = [s.kind for s in self.stages]
        # the three rules of Section 4.2.6, each stated separately so that none is only implied by another
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
        """Per mode, the occupation of the LAST cooling stage that addressed it."""
        out: dict[int, float] = {}
        for stage in self.stages:
            if stage.nbar is not None:
                out.update({int(k): float(v) for k, v in stage.nbar.items()})
        return out

    def final_motional(self) -> dict[int, qt.Qobj]:
        """Per mode, the explicit motional state of the LAST stage that gave one (a level-B Fock distribution or a
        level-C reduced state, Section 4.2/4.2.7); a mode a later thermal stage re-cooled loses its explicit state."""
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
    """The Appendix E ``State`` after the sequence: per-mode thermal states at the final nbar, each ion's internal state from its
    last pump (or from ``internal`` for ions the sequence never pumped, e.g. an explicitly prepared qubit next to a coolant);
    ``extra_nbar`` adds quanta per mode on top of the last cooling stage (the pumps' recoil heating of Section 4.2.8, M6);
    ``levels`` gives the register labels of every ion whose factor has d > 2 (the pumped populations land on the resolved
    sublevels and the remainder in the SINK, M7)."""
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


__all__ = [
    "COOLING_KINDS",
    "SUB_DOPPLER_KINDS",
    "PreparationSequence",
    "PreparationStage",
    "StageKind",
    "StageOrderError",
    "doppler_stage",
    "eit_stage",
    "fock_distribution_state",
    "prepare_state",
    "pulsed_sideband_stage",
    "pump_stage",
    "sideband_stage",
]
