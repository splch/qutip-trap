"""The atomic-rate layer of readout: the photon rate of the bright state, the pumping rates and the readout scheme (PLAN.md
Section 8.1).

Conventions: bright/dark polarity is a property of the readout scheme (:class:`ReadoutScheme`), never of the qubit; Gamma is
an angular population decay rate in s^-1; the system detection efficiency epsilon_sys enters exactly once, at the
conversion from scattered to detected rate (:meth:`FluorescenceRates.detected`), with the background kept separate.
:func:`scattering_rate` builds the multi-level Bloch model of the detection beams and reads R_o (the photon rate of the
conditional bright state), R_d (bright -> dark pumping) and R_b (dark -> bright) from its slow-manifold coarse graining,
asserting R_o against the equal-population ceiling n_e/(n_e + n_g) of the closed manifold (Gamma/4 for the 171Yb+
F = 1 -> F' = 0 cycle). Measured apparatus rates enter through :func:`rates_from_detected`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.special import j0, j1

if TYPE_CHECKING:
    from qutip_trap.dynamics.multilevel import MultiLevelOptions
    from qutip_trap.light.beams import Beam
    from qutip_trap.light.bloch import BlochModel
    from qutip_trap.readout.detection import Detector
    from qutip_trap.species.model import Species
    from qutip_trap.species.raman import AtomicStructure

ReadoutClass = Literal["bright", "dark", "shelf"]
CLASSES: tuple[ReadoutClass, ...] = ("bright", "dark", "shelf")
"""The classes of the readout Markov chain: the fluorescing manifold, the dark ground manifold reached by off-resonant
pumping, and the metastable shelf that decays back to bright."""

ALL_LINES = "all"
"""``line=ALL_LINES`` sums the photon rate over every non-sink decay line: the total scattering rate, a diagnostic and not
a detected rate, because epsilon_sys carries one filter passing one wavelength."""


def saturation_ceiling(n_ground: int, n_excited: int) -> float:
    """P_f^max = n_e/(n_e + n_g), the equal-population ceiling of a closed manifold (Berkeland Eqs. 13-14, 21)."""
    if n_ground <= 0 or n_excited <= 0:
        raise ValueError("a closed manifold has at least one ground and one excited state")
    return n_excited / (n_excited + n_ground)


@dataclass(frozen=True)
class DarkStateReport:
    """What the CPT solve returns, so that the ceiling is never assumed."""

    n_ground: int
    n_excited: int
    ceiling: float
    """n_e/(n_e + n_g); the photon rate must satisfy Gamma*P_f <= Gamma*ceiling."""
    dark_dimension: int
    dark_basis: np.ndarray
    """(dark_dimension, n_ground) complex amplitudes c_m, straight pairing."""
    delta_over_omega: float
    theta_be_deg: float
    """Angle between the linear polarization and B; optimum 54.7356 degrees."""
    raman_zero_margin_hz: float | None

    def __post_init__(self) -> None:
        expected = saturation_ceiling(self.n_ground, self.n_excited)
        if abs(self.ceiling - expected) > 1e-12:
            raise ValueError(f"ceiling must be n_e/(n_e + n_g) = {expected}, got {self.ceiling}")


def neighbour_intensity_ratio(wavelength_m: float, spacing_m: float) -> float:
    """I_ion/I_sat = 3 lambda^2/(8 pi^2 x^2): a saturated bright neighbour's resonant light at distance x relative to
    I_sat = pi h c Gamma/(3 lambda^3) (PLAN.md Section 8.5)."""
    return 3.0 * wavelength_m**2 / (8.0 * math.pi**2 * spacing_m**2)


def neighbour_pumping_rates(
    rates: FluorescenceRates,
    s_beam: float,
    wavelength_m: float,
    spacing_m: float,
    *,
    polarization_purity: float = 1.0,
) -> tuple[float, float]:
    """(Delta R_d, Delta R_b) that one saturated bright neighbour at ``spacing_m`` adds to an ion's chain, the depumping half
    of Wineland's readout crosstalk (PLAN.md Section 8.5).

    R_d and R_b are linear in intensity, so the leaked light drives them at the ratio of its saturation parameter
    ``neighbour_intensity_ratio`` to the detection beam's ``s_beam``; ``polarization_purity`` = 1 is the plan's bound, a
    smaller value the share of the leaked light at the pumping polarization.
    """
    if s_beam <= 0.0:
        raise ValueError("the detection beam's saturation parameter is positive")
    if not 0.0 <= polarization_purity <= 1.0:
        raise ValueError("the polarization share lies in [0, 1]")
    if spacing_m <= 0.0 or wavelength_m <= 0.0:
        raise ValueError("the spacing and the wavelength are positive")
    scale = polarization_purity * neighbour_intensity_ratio(wavelength_m, spacing_m) / s_beam
    return rates.R_dark_pumping_per_s * scale, rates.R_bright_pumping_per_s * scale


def micromotion_detection_rate(
    rate_at_offset: Callable[[float], float], beta: float, omega_rf_rad_s: float
) -> float:
    """J_0(beta)^2 R(0) + J_1(beta)^2 [R(-Omega_rf) + R(+Omega_rf)], R the rate at a detuning offset: the carrier and
    first-sideband channels of a detection beam under excess micromotion of modulation index beta (PLAN.md Section 8.8)."""
    if beta < 0.0:
        raise ValueError("the modulation index is non-negative")
    carrier = float(j0(beta)) ** 2 * rate_at_offset(0.0)
    side = float(j1(beta)) ** 2 * (rate_at_offset(-omega_rf_rad_s) + rate_at_offset(omega_rf_rad_s))
    return carrier + side


def mean_count_curve(
    tau_s: np.ndarray | float,
    detected_bright_per_s: float,
    dark_pumping_per_s: float,
    bright_pumping_per_s: float,
) -> np.ndarray | float:
    """n(tau) = eps R_o [(R_b/k) tau + (R_d/k^2)(1 - e^{-k tau})], k = R_b + R_d: the mean detected count of a bright-prepared
    ion from the two-state rate equation (Noek 2013 Eq. 4)."""
    tau = np.asarray(tau_s, dtype=float)
    k = dark_pumping_per_s + bright_pumping_per_s
    if k <= 0.0:
        out = detected_bright_per_s * tau
    else:
        out = detected_bright_per_s * (
            bright_pumping_per_s / k * tau + dark_pumping_per_s / k**2 * (1.0 - np.exp(-k * tau))
        )
    return float(out) if np.ndim(tau_s) == 0 else out


def fit_mean_count_curve(
    tau_s: Sequence[float], mean_counts: Sequence[float], *, guess: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Fit (eps R_o, R_d, R_b) to mean counts n(tau) from the starting ``guess``; returns values and 1-sigma uncertainties."""
    tau = np.asarray(tau_s, dtype=float)
    n = np.asarray(mean_counts, dtype=float)
    if tau.shape != n.shape or tau.size < 4:
        raise ValueError("at least four (tau, nbar) samples")
    scale = np.array(guess, dtype=float)
    scale[scale == 0.0] = 1.0

    def residual(p: np.ndarray) -> np.ndarray:
        r0, rd, rb = p * scale
        model = np.asarray(mean_count_curve(tau, r0, abs(rd), abs(rb)))
        return np.asarray(model - n)

    sol = least_squares(residual, np.ones(3), method="lm")
    values = sol.x * scale
    dof = max(tau.size - 3, 1)
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * float(np.sum(sol.fun**2)) / dof
    sigma = np.sqrt(np.abs(np.diag(cov))) * scale
    return (float(values[0]), abs(float(values[1])), abs(float(values[2]))), tuple(float(s) for s in sigma)  # type: ignore[return-value]


@dataclass(frozen=True)
class ReadoutScheme:
    """Which internal level fluoresces, and how each level starts the photon record (PLAN.md Sections 8.1, 8.4).

    ``classes[l]`` is the ideal readout class of internal level l (levels 0 and 1 the qubit, the rest leakage levels).
    A shelving transfer is imperfect and only partly state-selective, so ``transfer`` carries the start distribution over
    classes per level, and the dark-outcome operator Pi_dark = sum_l p_l |l><l| is then not a rank-one projector.
    """

    kind: Literal["direct", "shelving"]
    classes: tuple[ReadoutClass, ...]
    transfer: tuple[tuple[tuple[ReadoutClass, float], ...], ...] | None = None
    """Per level, the (class, probability) pairs of the start distribution; None means the ideal ``classes``."""

    def __post_init__(self) -> None:
        if len(self.classes) < 2:
            raise ValueError("a scheme classifies at least the two qubit levels")
        if any(c not in CLASSES for c in self.classes):
            raise ValueError(f"classes are among {CLASSES}")
        if "bright" not in self.classes:
            raise ValueError("no level fluoresces: the scheme cannot discriminate anything")
        if self.kind == "direct" and "shelf" in self.classes:
            raise ValueError("a direct-fluorescence scheme has no shelf class")
        if self.transfer is not None:
            if len(self.transfer) != len(self.classes):
                raise ValueError("one start distribution per level")
            for dist in self.transfer:
                total = sum(p for _c, p in dist)
                if any(p < 0.0 for _c, p in dist) or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError("each start distribution has non-negative weights summing to 1")

    @property
    def n_levels(self) -> int:
        return len(self.classes)

    @property
    def bright_level(self) -> int:
        """The qubit level (0 or 1) whose ideal class is bright."""
        for lev in (0, 1):
            if self.classes[lev] == "bright":
                return lev
        raise ValueError("neither qubit level is bright in this scheme")

    @property
    def polarity(self) -> Literal["bright_is_0", "bright_is_1"]:
        return "bright_is_1" if self.bright_level == 1 else "bright_is_0"

    def start_distribution(self, level: int) -> dict[ReadoutClass, float]:
        """P(start class | internal level), ideal when ``transfer`` is None."""
        if self.transfer is None:
            return {self.classes[level]: 1.0}
        return {c: p for c, p in self.transfer[level] if p > 0.0}

    @property
    def dark_class(self) -> ReadoutClass:
        """The most likely non-bright start class of the other qubit level ("shelf" or "dark"), whose decay rate the
        time-resolved discriminators need."""
        dist = {c: p for c, p in self.start_distribution(1 - self.bright_level).items() if c != "bright"}
        if not dist:
            return "dark" if self.kind == "direct" else "shelf"
        return max(dist, key=dist.__getitem__)

    def bit_of_class(self, cls: ReadoutClass) -> int:
        """The bit a perfectly discriminated class reports: bright -> the bright level, anything else -> the other."""
        return self.bright_level if cls == "bright" else 1 - self.bright_level

    @classmethod
    def direct(cls, bright_level: int, *, leak_classes: Sequence[ReadoutClass] = ()) -> ReadoutScheme:
        """Direct hyperfine fluorescence: 171Yb+ has |1> = F = 1 bright, |0> = F = 0 dark (``bright_level = 1``)."""
        if bright_level not in (0, 1):
            raise ValueError("bright_level is a qubit level, 0 or 1")
        base: list[ReadoutClass] = ["dark", "dark"]
        base[bright_level] = "bright"
        return cls("direct", tuple(base) + tuple(leak_classes))

    @classmethod
    def shelving(
        cls,
        shelved_level: int,
        *,
        transfer_probability: float = 1.0,
        off_resonant_shelving: float = 0.0,
        leak_classes: Sequence[ReadoutClass] = (),
    ) -> ReadoutScheme:
        """Shelving readout: ``shelved_level`` goes to the metastable shelf with ``transfer_probability`` and the other qubit
        level with ``off_resonant_shelving``. For an optical qubit whose upper level is the shelf (40Ca+ S1/2-D5/2) the
        transfer is 1 and |0> = S1/2 is bright."""
        if shelved_level not in (0, 1):
            raise ValueError("shelved_level is a qubit level, 0 or 1")
        if not 0.0 <= transfer_probability <= 1.0 or not 0.0 <= off_resonant_shelving <= 1.0:
            raise ValueError("probabilities lie in [0, 1]")
        classes: list[ReadoutClass] = ["bright", "bright"]
        classes[shelved_level] = "shelf"
        classes.extend(leak_classes)
        transfer: list[tuple[tuple[ReadoutClass, float], ...]] = []
        for lev in (0, 1):
            p = transfer_probability if lev == shelved_level else off_resonant_shelving
            transfer.append((("shelf", p), ("bright", 1.0 - p)))
        transfer.extend(((c, 1.0),) for c in leak_classes)
        ideal = transfer_probability == 1.0 and off_resonant_shelving == 0.0
        return cls("shelving", tuple(classes), None if ideal else tuple(transfer))

    @classmethod
    def for_species(
        cls, species: Species, bright_labels: Sequence[str], *, labels: Sequence[str] | None = None
    ) -> ReadoutScheme:
        """The ideal scheme of a species: a label in the bright manifold is bright, one in a metastable D level the shelf,
        any other dark. ``labels`` extends the classes to every level of a d > 2 register factor (the SINK reads dark)."""
        from qutip_trap.noise.scattering import SINK
        from qutip_trap.species.model import parse_state_label

        wanted = tuple(species.qubit) if labels is None else tuple(labels)
        classes: list[ReadoutClass] = []
        for lab in wanted:
            if lab == SINK:
                classes.append("dark")
                continue
            level, _ = parse_state_label(lab)
            if lab in bright_labels:
                classes.append("bright")
            elif level.startswith("D"):
                classes.append("shelf")
            else:
                classes.append("dark")
        return cls("shelving" if "shelf" in classes else "direct", tuple(classes))


@dataclass(frozen=True)
class FluorescenceRates:
    """R_o, R_d, R_b and the shelf rates of one ion under its detection beams, all in s^-1 of SCATTERED photons.

    ``ceiling`` is the equal-population ceiling R_o was asserted against (None for rates ingested from a measurement).
    """

    R_bright_per_s: float
    """R_o: the photon scattering rate of the conditional bright state."""
    R_dark_pumping_per_s: float
    """R_d: bright -> dark off-resonant pumping."""
    R_bright_pumping_per_s: float
    """R_b: dark -> bright off-resonant pumping."""
    shelf_decay_per_s: float = 0.0
    """1/tau_D: the shelf decays back into the bright manifold."""
    shelf_pumping_per_s: float = 0.0
    """Bright -> shelf during detection."""
    ceiling: float | None = None
    excited_population: float | None = None
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rates = {
            "R_bright_per_s": self.R_bright_per_s,
            "R_dark_pumping_per_s": self.R_dark_pumping_per_s,
            "R_bright_pumping_per_s": self.R_bright_pumping_per_s,
            "shelf_decay_per_s": self.shelf_decay_per_s,
            "shelf_pumping_per_s": self.shelf_pumping_per_s,
        }
        for name, rate in rates.items():
            if rate < 0.0:
                raise ValueError(f"{name} is a rate and non-negative")
        if self.R_bright_per_s <= 0.0:
            raise ValueError("a bright state that scatters nothing cannot be read out")

    def transition_rates(self) -> dict[tuple[ReadoutClass, ReadoutClass], float]:
        """The readout Markov chain: (from, to) -> rate, zero rates omitted."""
        pairs: dict[tuple[ReadoutClass, ReadoutClass], float] = {
            ("bright", "dark"): self.R_dark_pumping_per_s,
            ("dark", "bright"): self.R_bright_pumping_per_s,
            ("shelf", "bright"): self.shelf_decay_per_s,
            ("bright", "shelf"): self.shelf_pumping_per_s,
        }
        return {k: v for k, v in pairs.items() if v > 0.0}

    def detected(self, detector: Detector) -> tuple[float, float]:
        """(epsilon_sys R_o, R_bg): the detected bright rate with the efficiency applied once and the background beside it."""
        return detector.efficiency * self.R_bright_per_s, detector.background_cps

    def with_shelf(self, lifetime_s: float) -> FluorescenceRates:
        if lifetime_s <= 0.0:
            raise ValueError("the shelf lifetime is positive")
        return replace(self, shelf_decay_per_s=1.0 / lifetime_s)


def rates_from_detected(
    detected_bright_per_s: float,
    efficiency: float,
    *,
    dark_pumping_per_s: float = 0.0,
    bright_pumping_per_s: float = 0.0,
    shelf_lifetime_s: float | None = None,
    provenance: tuple[str, ...] = (),
) -> FluorescenceRates:
    """Ingest an apparatus's measured detected rate by dividing the efficiency out once, so that the record layer applies it
    once again and the scattered rate can be compared with the Bloch solve."""
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency lies in (0, 1]")
    return FluorescenceRates(
        R_bright_per_s=detected_bright_per_s / efficiency,
        R_dark_pumping_per_s=dark_pumping_per_s,
        R_bright_pumping_per_s=bright_pumping_per_s,
        shelf_decay_per_s=0.0 if shelf_lifetime_s is None else 1.0 / shelf_lifetime_s,
        provenance=tuple(provenance) + ("apparatus: detected rate ingested, not solved",),
    )


def detected_line(model: BlochModel) -> str:
    """The one decay line whose photons the detector counts, when the model carries only one.

    The interference filter passes one wavelength, so photons on a repump line must not be counted in R_o; with several
    non-sink lines the choice is a device property and the caller must name it (or ask for :data:`ALL_LINES`).
    """
    from qutip_trap.dynamics.multilevel import SINK

    lines = []
    for ch in model.build.channels:
        key = f"{ch.lower}<-{ch.upper}"
        if not key.startswith(SINK) and key not in lines:
            lines.append(key)
    if not lines:
        raise ValueError("the build has no decay line: nothing is detected")
    if len(lines) > 1:
        raise ValueError(
            f"the detection model carries {len(lines)} decay lines {lines}: name the detected one "
            f"(R_det = eps_sys R_o counts one wavelength) or pass line={ALL_LINES!r} for the all-line diagnostic sum"
        )
    return lines[0]


def rates_from_bloch(
    model: BlochModel,
    bright_labels: Sequence[str],
    dark_labels: Sequence[str],
    *,
    line: str | None = None,
) -> FluorescenceRates:
    """R_o, R_d, R_b from one Liouvillian by the slow-manifold analysis, R_o asserted against the ceiling.

    With no dark labels (a shelving scheme whose bright manifold is everything the beams drive) the steady state is the
    bright state and the pumping rates vanish. ``line`` names the detected line "lower<-upper"; None derives it with
    :func:`detected_line`, :data:`ALL_LINES` sums every non-sink line.
    """
    if line == ALL_LINES:
        chosen: str | None = None
    else:
        chosen = line if line is not None else detected_line(model)
    if dark_labels:
        dr = model.detection_rates(bright_labels, dark_labels, line=chosen)
        r_o, r_d, r_b = dr.R_bright_per_s, dr.R_dark_pumping_per_s, dr.R_bright_pumping_per_s
        ceiling, p_e = dr.ceiling.ceiling, dr.ceiling.excited_population
    else:
        ss = model.steadystate()
        rates = ss.photon_rates_per_s
        r_o = float(rates[chosen]) if chosen is not None else ss.total_photon_rate_per_s
        r_d, r_b = 0.0, 0.0
        ceiling, p_e = ss.ceiling.ceiling, ss.ceiling.excited_population
    return FluorescenceRates(
        R_bright_per_s=r_o,
        R_dark_pumping_per_s=r_d,
        R_bright_pumping_per_s=r_b,
        ceiling=ceiling,
        excited_population=p_e,
        provenance=("conv.scattering_rate_object",),
    )


def scattering_rate(
    structure: AtomicStructure,
    beams: Sequence[Beam],
    *,
    line: str | None = None,
    levels: Sequence[str] | None = None,
    options: MultiLevelOptions | None = None,
    position_m: Sequence[float] | None = None,
) -> tuple[FluorescenceRates, BlochModel]:
    """The rate object from first principles: the multi-level Bloch model of ``beams`` on ``structure``.

    The bright manifold is the ground states the beams drive resonantly and the dark manifold the remaining sublevels of
    the same levels (171Yb+: F = 1 bright, F = 0 dark). Returns the rates and the model.
    """
    from qutip_trap.dynamics.multilevel import SINK, MultiLevelOptions
    from qutip_trap.light.bloch import BlochModel

    opts = options or MultiLevelOptions(leak="renormalize")
    model = BlochModel(structure, beams, levels=levels, position_m=position_m, options=opts)
    bright, _excited = model.resonant_manifold()
    if not bright:
        raise ValueError("the detection beams drive no ground state resonantly: nothing fluoresces")
    levels_of_bright = {model.build.level_of(lab) for lab in bright}
    dark = tuple(
        lab
        for lab in model.build.labels
        if lab != SINK and lab not in bright and model.build.level_of(lab) in levels_of_bright
    )
    return rates_from_bloch(model, tuple(bright), dark, line=line), model


def detection_rates_for_ion(
    species: Species,
    b_gauss: float,
    b_hat: tuple[float, float, float],
    beams: Sequence[Beam],
    *,
    position_m: Sequence[float] | None = None,
    levels: Sequence[str] | None = None,
    options: MultiLevelOptions | None = None,
    scheme: ReadoutScheme | None = None,
    micromotion_beta: float = 0.0,
    omega_rf_rad_s: float = 0.0,
) -> tuple[FluorescenceRates, ReadoutScheme, BlochModel]:
    """Rates and scheme for one ion of a device: the species' cycling line and the repump lines the beams drive, the shelf
    lifetime from the species' metastable level when the scheme shelves.

    R_o counts photons on the cycling line only (the filter passes one wavelength). ``micromotion_beta`` > 0 phase-modulates
    the detection drive at ``omega_rf_rad_s``, so every rate becomes J_0^2 R(Delta) + J_1^2 [R(Delta - Omega_rf) +
    R(Delta + Omega_rf)] over three solves.
    """
    from qutip_trap.light.bloch import shifted_beam
    from qutip_trap.light.roles import RESONANT_WINDOW
    from qutip_trap.species.model import parse_state_label, parse_transition_label
    from qutip_trap.species.raman import structure_at

    st = structure_at(species, b_gauss, b_hat)
    lower, upper = parse_transition_label(species.cycling)
    if levels is None:
        lv = [lower, upper]
        for rep in species.repumps:
            tr = next((t for t in species.transitions if t.label == rep and t.multipole == "E1"), None)
            if tr is None:
                continue
            # a repump manifold enters only when the device shines that light: without the beam the metastable level is a
            # dead end whose own lifetime becomes the slowest Liouvillian mode, and the bright/dark coarse graining fails
            if not any(
                abs(b.wavelength_m - tr.wavelength_vac_m) < RESONANT_WINDOW * tr.wavelength_vac_m
                for b in beams
            ):
                continue
            for end in parse_transition_label(rep):
                if end not in lv:
                    lv.append(end)
        levels = tuple(lv)
    cycling_line = f"{lower}<-{upper}"
    rates, model = scattering_rate(
        st, beams, levels=levels, options=options, position_m=position_m, line=cycling_line
    )
    if micromotion_beta > 0.0:
        if omega_rf_rad_s <= 0.0:
            raise ValueError(
                "a micromotion modulation index needs the trap's rf frequency Omega_rf for the sideband channels"
            )
        solved: dict[float, FluorescenceRates] = {0.0: rates}

        def at(offset: float) -> FluorescenceRates:
            if offset not in solved:
                solved[offset], _ = scattering_rate(
                    st,
                    [shifted_beam(b, offset) for b in beams],
                    levels=levels,
                    options=options,
                    position_m=position_m,
                    line=cycling_line,
                )
            return solved[offset]

        rates = replace(
            rates,
            R_bright_per_s=micromotion_detection_rate(
                lambda d: at(d).R_bright_per_s, micromotion_beta, omega_rf_rad_s
            ),
            R_dark_pumping_per_s=micromotion_detection_rate(
                lambda d: at(d).R_dark_pumping_per_s, micromotion_beta, omega_rf_rad_s
            ),
            R_bright_pumping_per_s=micromotion_detection_rate(
                lambda d: at(d).R_bright_pumping_per_s, micromotion_beta, omega_rf_rad_s
            ),
            provenance=rates.provenance
            + (
                f"micromotion on the detection beam (Section 8.8): beta = {micromotion_beta:.4g}, "
                f"Omega_rf/2pi = {omega_rf_rad_s / (2.0 * math.pi):.4g} Hz, "
                f"J_0^2 = {float(j0(micromotion_beta)) ** 2:.6f} on the carrier and "
                f"J_1^2 = {float(j1(micromotion_beta)) ** 2:.6f} on each first sideband",
            ),
        )
    ground, _ = model.resonant_manifold()
    chosen = scheme if scheme is not None else ReadoutScheme.for_species(species, ground)
    if "shelf" in chosen.classes and rates.shelf_decay_per_s == 0.0:
        shelf_level = None
        for lev_index, cls_name in enumerate(chosen.classes):
            if cls_name == "shelf" and lev_index < len(species.qubit):
                shelf_level, _ = parse_state_label(species.qubit[lev_index])
        if shelf_level is None and species.shelving is not None:
            _, shelf_level = parse_transition_label(species.shelving)
        if shelf_level is not None:
            tau = species.level(shelf_level).lifetime_s
            if tau is not None:
                rates = rates.with_shelf(tau)
    return rates, chosen, model
