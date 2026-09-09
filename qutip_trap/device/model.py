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
    from qutip_trap.prep.recipe import PreparationRecipe
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
class GradientField:
    """The near-field microwave-gradient drive's field configuration (PLAN.md Section 4.4.5; Srinivas et al. 2021).

    A near-field electrode pair carries an oscillating current at ``frequency_hz`` = omega_g/2pi whose magnetic field
    has a gradient d(B_g . B_hat)/dr = ``gradient_t_per_m`` along the unit direction ``axis``, so the qubit frequency of
    an ion displaced by x along that direction is modulated at omega_g by (d omega_0/dB) grad(B) x. Two microwave tones
    of field amplitude ``b1_tesla_lab`` (the complex lab-frame vector of the oscillating field, as
    ``light.microwave.rabi_frequency_hz`` takes it), symmetrically detuned by +-delta from the ac-Zeeman-shifted qubit
    frequency, dress the spin; the dressed sigma_z force on the motion carries J_2(4 Omega_mu/delta) and the gate is
    intrinsically dynamically decoupled where J_0(4 Omega_mu/delta) = 0 (Omega_mu/delta = 0.6012).

    ``phase_rad`` is the gradient's phase at t = 0 relative to the microwave reference (the laboratory's control over
    the sign of the spin-dependent force, Section 4.4.5); the gradient amplitude and its frequency are DEVICE
    parameters and Omega_g is derived from them and the species' field sensitivity
    (``light.microwave.derive_gradient_drive``), never entered as a coupling.
    """

    gradient_t_per_m: float
    frequency_hz: float
    axis: tuple[float, float, float]
    b1_tesla_lab: tuple[complex, complex, complex]
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("the gradient's oscillation frequency is positive")
        norm = math.sqrt(sum(x * x for x in self.axis))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"GradientField.axis must be a unit vector, got norm {norm}")


@dataclass(frozen=True)
class DerivedQuantities:
    """Every computed number with its provenance id (Section 14.5 ledger)."""

    values: dict[str, float]
    provenance: dict[str, str]
    notes: tuple[str, ...] = ()
    """What the device could not derive and why (a microwave drive's Rabi frequency, a species without a detection beam; M8)."""

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
    preparation: PreparationRecipe | None = None
    """How the device cools and pumps before every shot (Section 4.2.6; M6); None = ``prep.recipe.standard_recipe``."""
    gradient: GradientField | None = None
    """The near-field microwave-gradient electrodes (Section 4.4.5; M4): None on a device with no gradient drive, in which
    case a ``gradient`` ``Drive`` is refused by the builder."""

    def derived(self) -> DerivedQuantities:
        """Every computed number with its provenance id (Section 3.3; the ledger of Section 14.5): the qubit transition
        frequencies and their Zeeman sensitivities per ion (M0a), the secular frequencies, Mathieu parameters and C0 of the
        trap (M1), the mode frequencies and heating rates (M1, M7), the carrier Rabi frequencies, Stark shifts, crosstalk ratios
        and Lamb-Dicke parameters of the inferred single-qubit drives (M2), the detection rates (M5) and the prepared
        occupations (M6). These are the values the calibration of Section 7.5 starts from as ``seed`` entries (M8); a device
        whose beams do not identify a single-qubit drive reports what it can and says so in ``provenance``."""
        from qutip_trap.device.derived import derived_quantities

        return derived_quantities(self)

    def hash(self) -> str:
        """The canonical digest of Appendix E: declaration-order fields, 12-digit floats, sorted dicts, Qobj excluded.

        Memoized per instance (a frozen dataclass whose arrays and dicts the code never mutates in place; a changed device is a
        new instance through ``dataclasses.replace``): the builder fingerprints every segment Hamiltonian with it, and a
        GATE_LOCAL tomography builds tens of thousands of segments on one device (0.9 ms each at four ions, 80 s of a 1137 s
        four-qubit GHZ run; performance pass 2026-09-09).
        """
        entry = _HASHES.get(id(self))
        if entry is not None and entry[0] is self:
            return entry[1]
        digest = canonical_digest(self)
        if len(_HASHES) >= _HASHES_MAX:
            _HASHES.clear()
        _HASHES[id(self)] = (self, digest)
        return digest


_HASHES: dict[int, tuple[Device, str]] = {}
"""``Device.hash`` memo keyed by instance identity, the instance kept alive so that its id cannot be reused."""
_HASHES_MAX = 256


__all__ = ["DerivedQuantities", "Device", "Field", "GradientField"]
