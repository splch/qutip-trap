"""Names outside the stability guarantee (docs/api_proposal.md Section 4.11; docs/api_implementation_plan.md 3.3; 0.4.0).

Everything importable from here may change or disappear in a minor release without the deprecation cycle of
docs/deprecations.md (Mitiq's rule for its own ``experimental`` namespace: "not covered by semantic versioning
guarantees"). The stability is gated by the import path, not by the object: the same objects stay importable from
``qutip_trap.api``, the frozen Appendix E surface, which never changes under them. Three groups live here until each has a
second consumer or a settled design:

- the transport records and functions of milestone M12 (PLAN.md Section 4.6), which are specification only: ``Zone``,
  ``VoltageWaveform``, ``FilterStage``, ``Transport``, ``TransportBudget``, ``design_waveform``, ``split_feasible``,
  ``transport_budget``; their methods raise ``NotImplementedError`` naming the milestone;
- the process-tomography internals the GATE_LOCAL walk uses and the application does not: ``choi_least_squares`` (the
  least-squares Choi reconstruction from input and output states) and ``project_cptp`` (Dykstra's projection onto the
  completely positive, trace-preserving set); ``qutip_trap.dynamics`` exported them in 0.2.0 and 0.3.0 and warns for them
  since 0.4.0;
- two oracles the run reports against rather than integrates: ``filter_function`` (the filter-function formalism of
  Section 6.9 for a decoupling sequence) and ``frozen_excitation_bounds`` (the Section 5.2 off-resonant excitation bound of
  the frozen spectator modes of a step).
"""

from __future__ import annotations

from qutip_trap.dynamics.tomography import choi_least_squares, project_cptp
from qutip_trap.noise.decoupling import filter_function
from qutip_trap.run.space import frozen_excitation_bounds
from qutip_trap.transport.budget import (
    Transport,
    TransportBudget,
    design_waveform,
    split_feasible,
    transport_budget,
)
from qutip_trap.transport.waveforms import FilterStage, VoltageWaveform
from qutip_trap.transport.zones import Zone

__all__ = [
    "FilterStage",
    "Transport",
    "TransportBudget",
    "VoltageWaveform",
    "Zone",
    "choi_least_squares",
    "design_waveform",
    "filter_function",
    "frozen_excitation_bounds",
    "project_cptp",
    "split_feasible",
    "transport_budget",
]
