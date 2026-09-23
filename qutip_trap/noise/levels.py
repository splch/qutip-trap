"""The internal levels a register factor of dimension d > 2 carries.

Indices 0 and 1 are the qubit pair (index 0 the lower level); indices 2 .. d - 1 are leakage levels in a fixed order:
the remaining field-dressed sublevels of the qubit's own level(s), ascending in energy, then one ``SINK`` last that
collects every other final state. Readout treats the SINK as dark.
"""

from __future__ import annotations

from dataclasses import dataclass

from qutip_trap.species.model import Species, parse_state_label
from qutip_trap.species.raman import structure_at

SINK = "SINK"
"""The label of the collecting level: every scattering final state outside the resolved sublevels."""


@dataclass(frozen=True)
class InternalLevels:
    """The label of every register level of one ion, in index order."""

    labels: tuple[str, ...]

    @property
    def d(self) -> int:
        return len(self.labels)

    @property
    def has_sink(self) -> bool:
        return SINK in self.labels

    @property
    def sink_index(self) -> int | None:
        return self.labels.index(SINK) if self.has_sink else None

    def index(self, label: str) -> int | None:
        """The register index of a full atomic label, or None when the level is not resolved (it falls into the SINK)."""
        try:
            return self.labels.index(label)
        except ValueError:
            return None

    @property
    def atomic(self) -> tuple[tuple[int, str], ...]:
        """(index, label) of every resolved atomic level (the SINK excluded)."""
        return tuple((k, lab) for k, lab in enumerate(self.labels) if lab != SINK)


def internal_levels(
    species: Species, d: int, b_gauss: float, b_hat: tuple[float, float, float]
) -> InternalLevels:
    """The level map of one ion at dimension ``d`` (2 = the qubit alone; up to the qubit level(s)' sublevel count plus one)."""
    if d < 2:
        raise ValueError("a register factor carries at least the two qubit levels")
    st = structure_at(species, b_gauss, b_hat)
    lower, upper = species.qubit
    qubit_labels = (st.state(lower).full_label, st.state(upper).full_label)
    others: list[tuple[float, str]] = []
    seen_levels: list[str] = []
    for lab in (lower, upper):
        level, _ = parse_state_label(lab)
        if level in seen_levels:
            continue
        seen_levels.append(level)
        for state in st.states_of(level):
            if state.full_label not in qubit_labels:
                others.append((state.energy_hz, state.full_label))
    others.sort()
    n_max = 3 + len(others)
    if d > n_max:
        raise ValueError(
            f"{species.name}: a register factor holds at most {n_max} levels (the {2 + len(others)} sublevels of the qubit "
            f"level(s) plus the SINK), not {d}"
        )
    if d == 2:
        labels = list(qubit_labels)
    else:
        labels = list(qubit_labels) + [lab for _e, lab in others][: d - 3] + [SINK]
    return InternalLevels(tuple(labels))


__all__ = ["SINK", "InternalLevels", "internal_levels"]
