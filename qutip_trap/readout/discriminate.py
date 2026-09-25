"""Discriminators, the readout POVM and the measurement (PLAN.md Section 8.3).

The discriminators are strategies over the same record: photon-count thresholding with the interior optimum over
(n_c, t_b); Myerson's time-resolved maximum likelihood by its O(N) recursion in the log domain (or the exact hidden-Markov
forward likelihood of the chain); the adaptive Bayesian early termination; and Noek's and Crain's first-photon protocols.

The POVM is the summary of the record and discriminator layers: the product form, exact only at zero readout crosstalk,
or the register-wide confusion factored by neighbour range at configured crosstalk. Its rows are indexed by the ion's
internal LEVEL with the scheme's transfer channel folded in once, so the fast path applies it to the projectively sampled
levels and never samples a start class of its own; the fast path replaces the record layer rather than preceding it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import numpy as np

from qutip_trap.readout.detection import (
    CLASS_INDEX,
    Depumping,
    PhotonRecord,
    RecordModel,
    log_poisson_pmf,
    neighbourhood_model,
    sample_register_records,
)
from qutip_trap.readout.fluorescence import CLASSES, ReadoutClass, ReadoutScheme

if TYPE_CHECKING:
    from qutip_trap.dynamics.engine import SeedSpec, State
    from qutip_trap.dynamics.space import HilbertSpace


@dataclass(frozen=True)
class Decision:
    bright: bool
    time_used_s: float
    posterior_error: float | None = None
    """The Bayesian error estimate of the declared outcome with equal priors, when the discriminator evaluates likelihoods."""


class Discriminator(Protocol):
    """A strategy over the record of one ion."""

    @property
    def window_s(self) -> float: ...

    @property
    def sub_bin_s(self) -> float | None: ...

    @property
    def needs_arrivals(self) -> bool: ...

    def decide(self, record: PhotonRecord, model: RecordModel) -> Decision: ...


def _posterior_error(log_pb: float, log_pd: float, bright: bool) -> float:
    """e_B = p_D/(p_B + p_D) when declared bright, e_D = p_B/(p_B + p_D) when declared dark (equal priors)."""
    m = max(log_pb, log_pd)
    pb, pd = math.exp(log_pb - m), math.exp(log_pd - m)
    return (pd if bright else pb) / (pb + pd)


# ---- threshold ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdDiscriminator:
    """Sum the counts over ``window_s`` and declare bright iff n > n_c, n_c half-integer."""

    n_c: float
    window_s: float
    sub_bin_s: ClassVar[float | None] = None
    needs_arrivals: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.window_s <= 0.0:
            raise ValueError("the window is positive")
        if abs(self.n_c - math.floor(self.n_c) - 0.5) > 1e-12:
            raise ValueError("the threshold is half-integer so that no count ties it")

    def decide(self, record: PhotonRecord, model: RecordModel) -> Decision:
        n = record.total if record.window_s <= self.window_s else record.counts_before(self.window_s)
        return Decision(bright=n > self.n_c, time_used_s=self.window_s)

    def error_rates(
        self, model: RecordModel, *, dark_start: ReadoutClass | Sequence[float] = "dark"
    ) -> tuple[float, float]:
        """The exact (epsilon_B, epsilon_D) from the chain's count distributions."""
        pb = model.count_distribution("bright", self.window_s)
        pd = model.count_distribution(dark_start, self.window_s)
        return 1.0 - pb.probability_above(self.n_c), pd.probability_above(self.n_c)


@dataclass(frozen=True)
class ThresholdScanPoint:
    window_s: float
    n_c: float
    eps_B: float
    eps_D: float

    @property
    def eps(self) -> float:
        return 0.5 * (self.eps_B + self.eps_D)


@dataclass(frozen=True)
class ThresholdOptimum:
    best: ThresholdScanPoint
    scan: tuple[ThresholdScanPoint, ...]
    """The best threshold at every window scanned, so the interior optimum in t_b is visible."""

    @property
    def discriminator(self) -> ThresholdDiscriminator:
        return ThresholdDiscriminator(self.best.n_c, self.best.window_s)


def optimize_threshold(
    model: RecordModel, windows_s: Sequence[float], *, dark_start: ReadoutClass | Sequence[float] = "dark"
) -> ThresholdOptimum:
    """Scan (n_c, t_b) for the minimum of the average error (eps_B + eps_D)/2, interior because the overlap error falls with
    t_b while pumping, shelf decay and background rise (Myerson: n_c = 5.5, t_b = 420 us)."""
    points: list[ThresholdScanPoint] = []
    for t in windows_s:
        pb = model.count_distribution("bright", t)
        pd = model.count_distribution(dark_start, t)
        n_max = min(pb.n_max, pd.n_max)
        best: ThresholdScanPoint | None = None
        for k in range(n_max):
            n_c = k + 0.5
            point = ThresholdScanPoint(
                float(t), n_c, 1.0 - pb.probability_above(n_c), pd.probability_above(n_c)
            )
            if best is None or point.eps < best.eps:
                best = point
        assert best is not None
        points.append(best)
    return ThresholdOptimum(min(points, key=lambda p: p.eps), tuple(points))


# ---- time-resolved maximum likelihood (Myerson 2008) --------------------------------------------------------------------------


def _decay_rate(model: RecordModel, dark_class: ReadoutClass) -> float:
    """The dark class's rate of becoming bright (1/tau: the shelf lifetime, or R_b for a hyperfine dark state)."""
    return model.rates.get((dark_class, "bright"), 0.0)


def myerson_log_likelihoods(
    counts: Sequence[int] | np.ndarray,
    sub_bin_s: float,
    model: RecordModel,
    *,
    dark_class: ReadoutClass = "dark",
    include_decay: bool = True,
) -> tuple[float, float]:
    """(ln p_B, ln p_D) of Myerson Eqs. 1-2 by the exact O(N) recursion in the log domain.

    p_B = prod_i B(n_i); p_D = (1 - t_b/tau) prod_i D(n_i) + (t_s/tau) sum_j prod_{i<j} D(n_i) prod_{i>=j} B(n_i) with
    M_k = M_{k-1} D(n_k), S_k = (S_{k-1} + M_{k-1}) B(n_k); B is Poisson at (eps R_o + R_bg) t_s, D at R_bg t_s. M_N
    underflows near N ~ 227 sub-bins at the paper's rates, hence the log domain.
    """
    n = np.asarray(counts, dtype=int)
    if n.ndim != 1 or n.size == 0:
        raise ValueError("a non-empty vector of sub-bin counts")
    lam_b = (model.detected_bright_per_s + model.background_per_s) * sub_bin_s
    lam_d = model.background_per_s * sub_bin_s
    log_b = np.asarray(log_poisson_pmf(n, lam_b))
    log_d = np.asarray(log_poisson_pmf(n, lam_d))
    log_pb = float(np.sum(log_b))
    rate = _decay_rate(model, dark_class) if include_decay else 0.0
    t_b = sub_bin_s * n.size
    if rate <= 0.0 or rate * t_b >= 1.0:
        if rate * t_b >= 1.0:
            raise ValueError("Myerson's first-order mixture needs t_b/tau < 1; use the exact likelihood")
        return log_pb, float(np.sum(log_d))
    log_m = 0.0
    log_s = -math.inf
    for k in range(n.size):
        log_s = np.logaddexp(log_s, log_m) + log_b[k]
        log_m = log_m + log_d[k]
    log_pd = float(np.logaddexp(math.log(1.0 - rate * t_b) + log_m, math.log(rate * sub_bin_s) + log_s))
    return log_pb, log_pd


def exact_log_likelihoods(
    counts: Sequence[int] | np.ndarray,
    sub_bin_s: float,
    model: RecordModel,
    *,
    dark_class: ReadoutClass = "dark",
) -> tuple[float, float]:
    """(ln p_B, ln p_D) by the hidden-Markov forward algorithm on the chain's exact sub-bin kernel: every jump order, both
    pumping directions and the shelf decay, with no first-order approximation."""
    n = np.asarray(counts, dtype=int)
    kernels = model.sub_bin_matrices(sub_bin_s, int(n.max()))
    out: list[float] = []
    for start in ("bright", dark_class):
        v = np.zeros(3)
        v[CLASS_INDEX[start]] = 1.0
        log_scale = 0.0
        for k in n:
            v = v @ kernels[k]
            s = float(v.sum())
            if s <= 0.0:
                log_scale = -math.inf
                break
            v /= s
            log_scale += math.log(s)
        out.append(log_scale)
    return out[0], out[1]


def _sub_bin_counts(record: PhotonRecord, sub_bin_s: float, n_bins: int) -> np.ndarray:
    """The record's counts on the discriminator's sub-bin grid: as stored, re-binned when the discriminator's sub-bin is an
    integer multiple of the record's, or histogrammed from the arrival times."""
    if record.sub_bins is not None and record.sub_bin_s is not None:
        ratio = sub_bin_s / record.sub_bin_s
        k = int(round(ratio))
        if k >= 1 and abs(ratio - k) < 1e-9:
            needed = n_bins * k
            if record.sub_bins.size < needed:
                raise ValueError("the record is shorter than the discriminator's window")
            return np.asarray(record.sub_bins[:needed].reshape(n_bins, k).sum(axis=1), dtype=int)
    if record.arrivals_s is not None:
        return np.asarray(
            np.histogram(record.arrivals_s, bins=n_bins, range=(0.0, n_bins * sub_bin_s))[0], dtype=int
        )
    raise ValueError("a time-resolved discriminator needs sub-bin counts or arrival times")


@dataclass(frozen=True)
class TimeResolvedML:
    """Myerson's maximum-likelihood discriminator on sub-bin counts, ties to bright; ``exact`` uses the hidden-Markov
    likelihood instead of the first-order recursion."""

    sub_bin_s: float
    window_s: float
    dark_class: ReadoutClass = "dark"
    include_decay: bool = True
    exact: bool = False
    needs_arrivals: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.sub_bin_s <= 0.0 or self.window_s <= 0.0:
            raise ValueError("sub-bin and window are positive")
        if abs(round(self.window_s / self.sub_bin_s) * self.sub_bin_s - self.window_s) > 1e-9 * self.window_s:
            raise ValueError("the window is an integer number of sub-bins")

    @property
    def n_sub_bins(self) -> int:
        return int(round(self.window_s / self.sub_bin_s))

    def decide(self, record: PhotonRecord, model: RecordModel) -> Decision:
        counts = _sub_bin_counts(record, self.sub_bin_s, self.n_sub_bins)
        if self.exact:
            log_pb, log_pd = exact_log_likelihoods(counts, self.sub_bin_s, model, dark_class=self.dark_class)
        else:
            log_pb, log_pd = myerson_log_likelihoods(
                counts, self.sub_bin_s, model, dark_class=self.dark_class, include_decay=self.include_decay
            )
        bright = log_pb >= log_pd
        return Decision(bright, self.window_s, _posterior_error(log_pb, log_pd, bright))


@dataclass(frozen=True)
class AdaptiveML:
    """Bayesian early termination: after each sub-bin compute the posterior error of the likelier outcome and stop when it
    falls below ``cutoff_error``, or at ``window_s`` (Myerson: 1.0e-4 at 145 us average). Myerson found faster detection at
    little cost without the decay term, hence ``include_decay=False``."""

    sub_bin_s: float
    window_s: float
    cutoff_error: float
    dark_class: ReadoutClass = "dark"
    include_decay: bool = False
    needs_arrivals: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.sub_bin_s <= 0.0 or self.window_s <= 0.0 or not 0.0 < self.cutoff_error < 0.5:
            raise ValueError("positive sub-bin and window; the cutoff is an error probability in (0, 0.5)")

    def decide(self, record: PhotonRecord, model: RecordModel) -> Decision:
        n_bins = int(round(self.window_s / self.sub_bin_s))
        counts = _sub_bin_counts(record, self.sub_bin_s, n_bins)
        bright = True
        err = 0.5
        used = self.window_s
        for k in range(1, n_bins + 1):
            log_pb, log_pd = myerson_log_likelihoods(
                counts[:k],
                self.sub_bin_s,
                model,
                dark_class=self.dark_class,
                include_decay=self.include_decay,
            )
            bright = log_pb >= log_pd
            err = _posterior_error(log_pb, log_pd, bright)
            if err < self.cutoff_error:
                used = k * self.sub_bin_s
                break
        return Decision(bright, used, err)


@dataclass(frozen=True)
class FirstPhoton:
    """Arrival-time protocols: Crain's stop on the first photon (``cutoff_s`` None: any photon before ``window_s`` is
    bright) and Noek's two-photon rule (none dark, a second arrival bright, a lone photon bright iff it came before the
    cutoff tau_c = ln(R_d/R_dc)/(eps R_o))."""

    window_s: float
    cutoff_s: float | None = None
    sub_bin_s: ClassVar[float | None] = None
    needs_arrivals: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.window_s <= 0.0 or (self.cutoff_s is not None and not 0.0 <= self.cutoff_s <= self.window_s):
            raise ValueError("positive window; the cutoff lies inside it")

    def decide(self, record: PhotonRecord, model: RecordModel) -> Decision:
        if record.arrivals_s is None:
            raise ValueError("first-photon protocols need arrival times")
        arr = record.arrivals_s[record.arrivals_s < self.window_s]
        if arr.size == 0:
            return Decision(False, self.window_s)
        if self.cutoff_s is None or arr[0] < self.cutoff_s:
            return Decision(True, float(arr[0]))
        if arr.size >= 2:
            return Decision(True, float(arr[1]))
        return Decision(False, self.window_s)


# ---- the POVM: per-ion confusion, product form, register confusion ---------------------------------------------------------------


@dataclass(frozen=True)
class RegisterConfusion:
    """P(declared bits | true internal levels) factored by neighbour range: per ion a conditional table over its
    neighbourhood, exact for per-ion decisions on additive neighbour light.

    The ion's own axis runs over its internal levels (the transfer channel folded in), a neighbour's over its start class
    (all its leaked light depends on); ``bright_probability`` marginalizes the neighbours' classes with ``bright_start``.
    """

    n_ions: int
    neighbour_range: int
    tables: tuple[np.ndarray, ...]
    """tables[i][(x_j for j in neighbourhood(i)) + (declared,)]: x_i the ion's level, x_j (j != i) the neighbour's class
    (0 = bright, 1 = non-bright), declared 0 = bright."""
    bright_start: tuple[np.ndarray, ...]
    """Per ion, P(start class == bright | internal level)."""

    def __post_init__(self) -> None:
        if len(self.tables) != self.n_ions or len(self.bright_start) != self.n_ions:
            raise ValueError("one conditional table and one bright-start vector per ion")

    def neighbourhood(self, ion: int) -> tuple[int, ...]:
        lo, hi = max(0, ion - self.neighbour_range), min(self.n_ions - 1, ion + self.neighbour_range)
        return tuple(range(lo, hi + 1))

    def bright_probability(self, ion: int, true_levels: Sequence[int]) -> float:
        """P(ion declared bright | the register's true levels), the neighbours' start classes marginalized."""
        neigh = self.neighbourhood(ion)
        others = tuple(j for j in neigh if j != ion)
        table = self.tables[ion]
        total = 0.0
        for pattern in product((0, 1), repeat=len(others)):
            weight = 1.0
            for j, c in zip(others, pattern):
                p_bright = float(self.bright_start[j][int(true_levels[j])])
                weight *= p_bright if c == 0 else 1.0 - p_bright
            if weight == 0.0:
                continue
            it = iter(pattern)
            idx = tuple(int(true_levels[ion]) if j == ion else next(it) for j in neigh)
            total += weight * float(table[idx + (0,)])
        return total

    def probability(self, true_levels: Sequence[int], declared_bright: Sequence[bool]) -> float:
        p = 1.0
        for i in range(self.n_ions):
            pb = self.bright_probability(i, true_levels)
            p *= pb if declared_bright[i] else 1.0 - pb
        return p

    def sample(self, true_levels: Sequence[int], rng: np.random.Generator) -> tuple[bool, ...]:
        return tuple(rng.random() < self.bright_probability(i, true_levels) for i in range(self.n_ions))


@dataclass(frozen=True)
class POVM:
    """The readout POVM: the per-ion product form (exact only at zero readout crosstalk) or the register-wide confusion
    factored by neighbour range, with the statistical uncertainty of Monte-Carlo entries (0 for exact ones)."""

    per_ion: tuple[np.ndarray, ...] | None
    """The product form: per ion an (n_levels, 2) matrix P(declared | internal level), columns (bright, dark), the scheme's
    transfer channel folded into the rows."""
    factored: RegisterConfusion | None = None
    """The register-wide form at configured crosstalk."""
    uncertainty: float = 0.0
    bright_levels: tuple[int, ...] = ()
    """Per ion of the product form, the qubit level whose ideal class is bright."""

    def __post_init__(self) -> None:
        if (self.per_ion is None) == (self.factored is None):
            raise ValueError("a POVM is either the product form or the register-wide confusion")
        if self.per_ion is not None:
            if len(self.bright_levels) != len(self.per_ion) or any(
                b not in (0, 1) for b in self.bright_levels
            ):
                raise ValueError("the product form carries one bright level (0 or 1) per ion")
            for m in self.per_ion:
                if (
                    m.ndim != 2
                    or m.shape[0] < 2
                    or m.shape[1] != 2
                    or np.any(m < -1e-12)
                    or not np.allclose(m.sum(axis=1), 1.0, atol=1e-9)
                ):
                    raise ValueError(
                        "each per-ion element is an (n_levels, 2) stochastic matrix over (level, declared)"
                    )

    @property
    def n_ions(self) -> int:
        if self.per_ion is not None:
            return len(self.per_ion)
        assert self.factored is not None
        return self.factored.n_ions

    def declared_bright_probability(
        self, true_levels: Sequence[int], declared_bright: Sequence[bool]
    ) -> float:
        """P(this declared-bright pattern | the ions' true internal levels)."""
        if self.per_ion is None:
            assert self.factored is not None
            return self.factored.probability(true_levels, declared_bright)
        p = 1.0
        for m, lev, d in zip(self.per_ion, true_levels, declared_bright):
            p *= float(m[int(lev), 0 if d else 1])
        return p

    def sample(self, true_levels: Sequence[int], rng: np.random.Generator) -> tuple[bool, ...]:
        """One declared-bright pattern given the projectively sampled internal levels."""
        if self.per_ion is None:
            assert self.factored is not None
            return self.factored.sample(true_levels, rng)
        return tuple(rng.random() < float(m[int(lev), 0]) for m, lev in zip(self.per_ion, true_levels))

    def per_ion_errors(self) -> tuple[tuple[float, float], ...]:
        """(epsilon_B, epsilon_D) per ion of the product form: P(declared dark | bright level), P(declared bright | the
        other qubit level), the transfer channel included."""
        if self.per_ion is None:
            raise ValueError(
                "per-ion errors are defined for the product form; marginalize the register form instead"
            )
        return tuple((float(m[b, 1]), float(m[1 - b, 0])) for m, b in zip(self.per_ion, self.bright_levels))


def level_start_vectors(scheme: ReadoutScheme) -> tuple[np.ndarray, ...]:
    """Per internal level, the start distribution over CLASSES: the transfer channel as a stochastic map level -> class."""
    out: list[np.ndarray] = []
    for lev in range(scheme.n_levels):
        v = np.zeros(3)
        for c, p in scheme.start_distribution(lev).items():
            v[CLASS_INDEX[c]] += p
        out.append(v)
    return tuple(out)


def bright_start_probabilities(scheme: ReadoutScheme) -> np.ndarray:
    """P(start class == bright | internal level) over the scheme's levels."""
    return np.array([v[CLASS_INDEX["bright"]] for v in level_start_vectors(scheme)])


def _declared_bright(
    model: RecordModel,
    start: np.ndarray,
    discriminator: Discriminator,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """(P(declared bright), its statistical uncertainty) from a start distribution over the classes: exact through the count
    distribution for a threshold discriminator, Monte Carlo over ``n_samples`` sampled records otherwise."""
    if isinstance(discriminator, ThresholdDiscriminator):
        dist = model.count_distribution(list(start), discriminator.window_s)
        return dist.probability_above(discriminator.n_c), 0.0
    if n_samples <= 0:
        raise ValueError("a non-threshold discriminator needs Monte Carlo samples (n_samples > 0)")
    hits = 0
    for _ in range(n_samples):
        cls = CLASSES[int(rng.choice(3, p=start))]
        rec = model.sample_record(
            cls,
            discriminator.window_s,
            rng,
            sub_bin_s=discriminator.sub_bin_s,
            arrivals=discriminator.needs_arrivals,
        )
        hits += int(discriminator.decide(rec, model).bright)
    return hits / n_samples, math.sqrt(0.25 / n_samples)


def per_ion_confusion(
    model: RecordModel,
    scheme: ReadoutScheme,
    discriminator: Discriminator,
    *,
    n_samples: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """P(declared | internal level) as an (n_levels, 2) matrix over level x (declared bright, declared dark), and the
    statistical uncertainty of its entries: every level gets its own row through its own start distribution."""
    gen = rng if rng is not None else np.random.default_rng(0)
    out = np.zeros((scheme.n_levels, 2))
    unc = 0.0
    for row, start in enumerate(level_start_vectors(scheme)):
        p, unc = _declared_bright(model, start, discriminator, n_samples, gen)
        out[row] = (p, 1.0 - p)
    return out, unc


def product_povm(
    models: Sequence[RecordModel],
    schemes: Sequence[ReadoutScheme],
    discriminator: Discriminator,
    *,
    n_samples: int = 0,
    rng: np.random.Generator | None = None,
) -> POVM:
    """The product POVM, exact only at zero readout crosstalk."""
    if len(schemes) != len(models):
        raise ValueError("one scheme per ion")
    mats: list[np.ndarray] = []
    unc = 0.0
    for m, s in zip(models, schemes):
        mat, u = per_ion_confusion(m, s, discriminator, n_samples=n_samples, rng=rng)
        mats.append(mat)
        unc = max(unc, u)
    return POVM(tuple(mats), None, unc, tuple(s.bright_level for s in schemes))


def register_confusion(
    models: Sequence[RecordModel],
    schemes: Sequence[ReadoutScheme],
    discriminator: Discriminator,
    leakage: Mapping[int, float],
    *,
    depumping: Depumping | None = None,
    n_samples: int = 0,
    rng: np.random.Generator | None = None,
) -> POVM:
    """The register-wide confusion at configured crosstalk: each ion's decision conditioned on its own level and on its
    neighbours' start classes, their leaked light entering as added counts and (with ``depumping``) extra pumping, frozen
    at the start class."""
    n = len(models)
    if len(schemes) != n:
        raise ValueError("one scheme per ion")
    leak = {int(k): float(v) for k, v in leakage.items() if v > 0.0}
    rng_range = max(leak) if leak else 0
    tables: list[np.ndarray] = []
    unc = 0.0
    for i in range(n):
        lo, hi = max(0, i - rng_range), min(n - 1, i + rng_range)
        neigh = tuple(range(lo, hi + 1))
        starts_i = level_start_vectors(schemes[i])
        axes = tuple(len(starts_i) if j == i else 2 for j in neigh)
        table = np.zeros(axes + (2,))
        for pattern in product(*(range(k) for k in axes)):
            classes: list[ReadoutClass] = ["bright"] * n
            for j, c in zip(neigh, pattern):
                if j != i:
                    classes[j] = "bright" if c == 0 else "dark"
            frozen = neighbourhood_model(models, i, classes, leak, depumping)
            gen = rng if rng is not None else np.random.default_rng(i)
            p_bright, u = _declared_bright(
                frozen, starts_i[pattern[neigh.index(i)]], discriminator, n_samples, gen
            )
            unc = max(unc, u)
            table[pattern + (0,)] = p_bright
            table[pattern + (1,)] = 1.0 - p_bright
        tables.append(table)
    factored = RegisterConfusion(
        n, rng_range, tuple(tables), tuple(bright_start_probabilities(s) for s in schemes)
    )
    return POVM(None, factored, unc)


def povm_for(
    models: Sequence[RecordModel],
    schemes: Sequence[ReadoutScheme],
    discriminator: Discriminator,
    leakage: Mapping[int, float] | None = None,
    *,
    depumping: Depumping | None = None,
    n_samples: int = 0,
    rng: np.random.Generator | None = None,
) -> POVM:
    """The POVM in the domain the device configures: product at zero crosstalk, register-wide otherwise."""
    leak = {int(k): float(v) for k, v in (leakage or {}).items() if v > 0.0}
    if not leak:
        return product_povm(models, schemes, discriminator, n_samples=n_samples, rng=rng)
    return register_confusion(
        models, schemes, discriminator, leak, depumping=depumping, n_samples=n_samples, rng=rng
    )


def max_confusion_discrepancy(register: POVM, product_form: POVM) -> float:
    """The largest difference, over ions, levels and neighbourhood configurations, between the register-wide confusion and
    the product POVM that cannot see the neighbours: the bounded discrepancy a run reports. The register form's per-ion
    bright probability is a convex combination of its table's entries, so this bounds every register state."""
    if register.factored is None:
        raise ValueError("the discrepancy is measured against the register-wide (factored) form")
    if product_form.per_ion is None:
        raise ValueError("the discrepancy is measured against the product form")
    fac = register.factored
    worst = 0.0
    for i, table in enumerate(fac.tables):
        own = fac.neighbourhood(i).index(i)
        rows = product_form.per_ion[i]
        for pattern in product(*(range(k) for k in table.shape[:-1])):
            level = pattern[own]
            if level >= rows.shape[0]:
                raise ValueError("the product POVM has no row for a level the register form carries")
            worst = max(worst, abs(float(table[pattern + (0,)]) - float(rows[level, 0])))
    return worst


# ---- the measurement of Section 5.7 ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutOutcome:
    """What the measurement returns for a batch of shots."""

    bits: np.ndarray
    """(shots, n_ions) declared bits, column j = qubit j."""
    levels: np.ndarray
    """(shots, n_ions) the projectively sampled internal level of each ion (the joint outcome, correlations intact)."""
    posteriors: np.ndarray | None
    """(shots, n_ions) posterior error estimates when the discriminator provides them."""
    time_used_s: np.ndarray
    records: tuple[tuple[PhotonRecord, ...], ...] | None
    mode: Literal["full", "fast"]

    @property
    def shots(self) -> int:
        return int(self.bits.shape[0])


def joint_level_probabilities(space: HilbertSpace, state: State) -> np.ndarray:
    """The diagonal of the register's reduced density matrix as an array over the ions' level multi-index."""
    rho = space.internal_marginal(state)
    diag = np.real(np.diag(np.asarray(rho.full())))
    diag = np.clip(diag, 0.0, None)
    total = float(diag.sum())
    if total <= 0.0:
        raise ValueError("the internal state has no population")
    return np.asarray((diag / total).reshape(tuple(space.ion_dims)))


def sample_joint_outcome(probabilities: np.ndarray, rng: np.random.Generator) -> tuple[int, ...]:
    """One projective outcome in the computational (level) basis of the register."""
    flat = probabilities.ravel()
    k = int(rng.choice(flat.size, p=flat / flat.sum()))
    return tuple(int(x) for x in np.unravel_index(k, probabilities.shape))


def measure(
    space: HilbertSpace,
    state: State,
    schemes: Sequence[ReadoutScheme],
    models: Sequence[RecordModel],
    discriminator: Discriminator,
    seeds: SeedSpec,
    *,
    shots: int,
    sample: int = 0,
    trajectory: int = 0,
    first_shot: int = 0,
    leakage: Mapping[int, float] | None = None,
    depumping: Depumping | None = None,
    mode: Literal["full", "fast"] = "full",
    povm: POVM | None = None,
    keep_records: bool = False,
) -> ReadoutOutcome:
    """Sample the joint internal outcome projectively, then either generate every ion's photon record conditioned on it
    and discriminate (``full``), or apply the POVM to the sampled levels (``fast``); never both, so the readout error is
    applied once. The full path samples each ion's start class from the scheme's transfer distribution; the fast path does
    not, because the POVM's rows carry that transfer. Randomness is keyed by (sample, trajectory, shot, ion, channel).
    """
    n = space.n_ions
    if len(schemes) != n or len(models) != n:
        raise ValueError("one scheme and one record model per ion")
    probs = joint_level_probabilities(space, state)
    bits = np.zeros((shots, n), dtype=np.uint8)
    levels = np.zeros((shots, n), dtype=np.uint8)
    times = np.zeros((shots, n))
    posts = np.full((shots, n), np.nan)
    records: list[tuple[PhotonRecord, ...]] = []
    if mode == "fast" and povm is None:
        povm = povm_for(models, schemes, discriminator, leakage, depumping=depumping)
    for s in range(shots):
        shot = first_shot + s
        rng_outcome = np.random.default_rng(seeds.child(sample, trajectory, shot, 0, "register_outcome"))
        outcome = sample_joint_outcome(probs, rng_outcome)
        levels[s] = outcome
        if mode == "fast":
            assert povm is not None
            rng_p = np.random.default_rng(seeds.child(sample, trajectory, shot, 0, "povm"))
            declared = povm.sample(outcome, rng_p)
            for i in range(n):
                bits[s, i] = schemes[i].bit_of_class("bright" if declared[i] else "dark")
                times[s, i] = discriminator.window_s
            continue
        starts: list[ReadoutClass] = []
        for i in range(n):
            rng_t = np.random.default_rng(seeds.child(sample, trajectory, shot, i, "readout_transfer"))
            dist = schemes[i].start_distribution(outcome[i])
            names = list(dist)
            starts.append(names[int(rng_t.choice(len(names), p=[dist[c] for c in names]))])
        rngs = [
            np.random.default_rng(seeds.child(sample, trajectory, shot, i, "photon_record")) for i in range(n)
        ]
        recs = sample_register_records(
            models,
            starts,
            discriminator.window_s,
            rngs,
            leakage=leakage,
            depumping=depumping,
            sub_bin_s=discriminator.sub_bin_s,
            arrivals=discriminator.needs_arrivals,
        )
        for i, rec in enumerate(recs):
            d = discriminator.decide(rec, models[i])
            bits[s, i] = schemes[i].bit_of_class("bright" if d.bright else "dark")
            times[s, i] = d.time_used_s
            if d.posterior_error is not None:
                posts[s, i] = d.posterior_error
        if keep_records:
            records.append(tuple(recs))
    return ReadoutOutcome(
        bits=bits,
        levels=levels,
        posteriors=None if np.all(np.isnan(posts)) else posts,
        time_used_s=times,
        records=tuple(records) if keep_records else None,
        mode=mode,
    )
