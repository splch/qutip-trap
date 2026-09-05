"""``Field``, ``Device`` and ``DerivedQuantities`` (PLAN.md Section 3.3; Appendix E).

Design principle (Section 3.1): device parameters in, everything else derived. ``Device.derived()`` returns
every computed number with its provenance id from the ledger of Section 14.5; ``Device.hash()`` is the
canonical serialization of Appendix E (``qutip_trap.hashing``), the identity for the calibration cache and
the run record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qutip_trap.hashing import canonical_digest

if TYPE_CHECKING:
    from qutip_trap.control.hardware import HardwareChain
    from qutip_trap.light.beams import Beam
    from qutip_trap.noise.model import NoiseModel
    from qutip_trap.noise.spectra import NoiseSpectrum
    from qutip_trap.readout.detection import Detector
    from qutip_trap.transport.zones import Zone
    from qutip_trap.trap.crystal import Crystal
    from qutip_trap.trap.model import Trap


@dataclass(frozen=True)
class Field:
    """The static magnetic field at the ions: quantization axis (Section 4.5.3) and noise (Section 6.3).

    Its direction is the axis in which beam polarizations are decomposed into sigma+, pi and sigma-
    components; its magnitude feeds the Zeeman shifts of every level through the computed sensitivities,
    never through a hand-entered MHz/G (Section 4.5.1).
    """

    B_gauss: float
    direction: tuple[float, float, float]
    noise: NoiseSpectrum | None

    def __post_init__(self) -> None:
        if self.B_gauss < 0.0:
            raise ValueError("B_gauss is a magnitude; encode the sense in `direction`")
        norm = math.sqrt(sum(x * x for x in self.direction))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Field.direction must be a unit vector, got norm {norm}")


@dataclass(frozen=True)
class DerivedQuantities:
    """Every computed number with its provenance id (Section 14.5 ledger)."""

    values: dict[str, float]
    provenance: dict[str, str]

    def __post_init__(self) -> None:
        if set(self.values) != set(self.provenance):
            raise ValueError("every derived value needs exactly one provenance id and vice versa")


@dataclass(frozen=True)
class Device:
    """The aggregate device model (Section 3.3)."""

    crystal: Crystal
    trap: Trap
    field: Field
    beams: tuple[Beam, ...]
    noise: NoiseModel
    detector: Detector
    hardware: HardwareChain
    zones: tuple[Zone, ...] = ()
    """M12: empty for the single-zone first release."""

    def derived(self) -> DerivedQuantities:
        """Every computed number with its provenance id; filled in as milestones M1 to M8 add the modules."""
        raise NotImplementedError("Device.derived() gathers the derived quantities of milestones M1 to M8")

    def hash(self) -> str:
        """The canonical digest of Appendix E: declaration-order fields, 12-digit floats, sorted dicts, Qobj excluded."""
        return canonical_digest(self)


__all__ = ["DerivedQuantities", "Device", "Field"]
