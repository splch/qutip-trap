"""The device model (``Device``, ``Field``, ``GradientField``, ``BeamRoles``): device parameters in, everything else
derived (``Device.derived()``)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qutip_trap.hashing import canonical_digest

if TYPE_CHECKING:
    from qutip_trap.control.hardware import HardwareChain
    from qutip_trap.control.schedule import GateDrive
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
    """The static magnetic field at the ions: magnitude, unit direction (the quantization axis of the sigma+, pi and
    sigma- polarization components) and noise spectrum."""

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
    """The near-field microwave-gradient drive's fields (Srinivas et al. 2021): ``gradient_t_per_m`` = d(B_g . B_hat)/dr
    along the unit vector ``axis``, oscillating at ``frequency_hz``; ``b1_tesla_lab`` the complex lab-frame field of the
    dressing tones; ``phase_rad`` the gradient's phase at t = 0 relative to the microwave reference."""

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
    """Every computed number of a device by key, with its provenance id."""

    values: dict[str, float]
    provenance: dict[str, str]
    notes: tuple[str, ...] = ()
    """What the device could not derive, and why."""

    def __post_init__(self) -> None:
        if set(self.values) != set(self.provenance):
            raise ValueError("every derived value needs exactly one provenance id and vice versa")


@dataclass(frozen=True)
class ResolvedRoles:
    """What :meth:`BeamRoles.resolve` settled for one device; ``detection`` is None when no beam is near the species'
    cycling line, and ``inferred`` names the fields that were inferred rather than declared."""

    gate: dict[int, GateDrive]
    entangling: dict[int, GateDrive]
    detection: int | None
    inferred: tuple[str, ...] = ()


@dataclass(frozen=True)
class BeamRoles:
    """Which of a device's beams play which part: per ion the single-qubit gate drive and the entangling drive, and the
    detection beam. ``None`` on a field means "infer it": the gate drive from the far-detuned beams (none = microwave,
    one = optical, an equal-wavelength pair = Raman, anything else is ambiguous and refused), the entangling drives as the
    gate drives, the detection beam as the beam nearest the first species' cycling line. The roles are the operator's
    assignment, not the apparatus: they are left out of ``Device.hash()`` and carried by ``Machine.hash()``."""

    gate: Mapping[int, GateDrive] | None = None
    entangling: Mapping[int, GateDrive] | None = None
    """``{}`` declares that the device has no entangling drive."""
    detection: int | None = None
    """Index in ``Device.beams`` of the beam the detector collects from."""

    def __post_init__(self) -> None:
        for name in ("gate", "entangling"):
            value = getattr(self, name)
            if value is None:
                continue
            drives = {int(i): d for i, d in dict(value).items()}
            if any(i < 0 for i in drives):
                raise ValueError(f"BeamRoles.{name}: ion indices are non-negative")
            object.__setattr__(self, name, drives)
        if self.detection is not None and self.detection < 0:
            raise ValueError("BeamRoles.detection is a beam index (non-negative) or None")

    def resolve(
        self,
        device: Device,
        *,
        gate_drives: Mapping[int, GateDrive] | None = None,
        entangling_drives: Mapping[int, GateDrive] | None = None,
    ) -> ResolvedRoles:
        """Resolve the roles on ``device``: an explicit keyword argument first, then the declared role, then the inference;
        entangling drives fall back to the gate drives. Every ion and beam index is checked against ``device``; raises
        ``ScheduleError`` when a gate drive is needed and the beams do not identify one."""
        from qutip_trap.control.schedule import infer_gate_drives
        from qutip_trap.light.roles import infer_detection_beam

        inferred: list[str] = []
        if gate_drives is not None:
            gate = {int(i): d for i, d in dict(gate_drives).items()}
        elif self.gate is not None:
            gate = dict(self.gate)
        else:
            gate = infer_gate_drives(device)
            inferred.append("gate")
        if entangling_drives is not None:
            entangling = {int(i): d for i, d in dict(entangling_drives).items()}
        elif self.entangling is not None:
            entangling = dict(self.entangling)
        else:
            entangling = dict(gate)
            inferred.append("entangling")
        if self.detection is not None:
            detection: int | None = self.detection
        else:
            detection = infer_detection_beam(device)
            inferred.append("detection")
        n_ions = device.crystal.n_ions
        n_beams = len(device.beams)
        for name, drives in (("gate", gate), ("entangling", entangling)):
            for i, d in drives.items():
                if i >= n_ions:
                    raise ValueError(f"BeamRoles.{name} names ion {i} on a device of {n_ions} ion(s)")
                if any(b >= n_beams for b in d.beams):
                    raise ValueError(
                        f"BeamRoles.{name}[{i}] names beams {d.beams} on a device of {n_beams} beam(s)"
                    )
        if detection is not None and detection >= n_beams:
            raise ValueError(f"BeamRoles.detection = {detection} on a device of {n_beams} beam(s)")
        return ResolvedRoles(gate, entangling, detection, tuple(inferred))


@dataclass(frozen=True)
class Device:
    """The aggregate device model."""

    crystal: Crystal
    trap: Trap
    field: Field
    beams: tuple[Beam, ...]
    noise: NoiseModel
    detector: Detector
    hardware: HardwareChain
    zones: tuple[Zone, ...] = ()
    preparation: PreparationRecipe | None = None
    """How the device cools and pumps before every shot; None means ``prep.recipe.standard_recipe``."""
    gradient: GradientField | None = None
    """None on a device without gradient electrodes, where the builder refuses a ``gradient`` ``Drive``."""
    roles: BeamRoles = dataclasses.field(default_factory=BeamRoles, metadata={"hash": "exclude"})
    """Left out of :meth:`hash`, which identifies the apparatus; the default infers every role from the beams."""

    def derived(self) -> DerivedQuantities:
        """Every computed number with its provenance id (``qutip_trap.device.derived``); what the device cannot derive is
        named in ``notes``."""
        from qutip_trap.device.derived import derived_quantities

        return derived_quantities(self)

    def specs(self) -> str:
        """The derived quantities as a readable report, ``key = value  [provenance id]`` grouped by family."""
        from qutip_trap.device.specs import render_specs

        return render_specs(self)

    def to_dict(self) -> dict[str, Any]:
        """The device as plain JSON-able values (schema version 1, ``qutip_trap.device.serial``); ``from_dict`` reads it
        back to a device with the same hash."""
        from qutip_trap.device.serial import device_to_dict

        return device_to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Device:
        """The device of a ``to_dict`` record (schema version 1); an unknown field or version is refused."""
        from qutip_trap.device.serial import device_from_dict

        return device_from_dict(data)

    def hash(self) -> str:
        """The canonical digest (``qutip_trap.hashing``): declaration-order fields, 12-digit floats, sorted dicts, Qobj
        excluded. Memoized per instance, which is safe because a device is frozen and never mutated in place."""
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


__all__ = ["BeamRoles", "DerivedQuantities", "Device", "Field", "GradientField", "ResolvedRoles"]
