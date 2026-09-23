"""Discriminators, the readout POVM and the error budget.

Readout layers: rates (``fluorescence``) -> photon record (``detection``) -> discriminator -> POVM and budget (here).
The POVM summarizes the record and discriminator layers once per device, its rows indexed by internal level with the
transfer channel folded in, so the fast path replaces the record layer and the readout error is applied once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from qutip_trap.readout.detection import (
    CLASS_INDEX,
    CameraGeometry,
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
    from qutip_trap.hilbert.space import HilbertSpace

M5 = "milestone M5 (readout/discriminate.py, PLAN.md Section 8.3)"

Convention = Literal["mean", "min"]


@dataclass(frozen=True)
class Decision:
    bright: bool
    time_used_s: float
    log_likelihood_ratio: float | None = None
    """ln(p_B/p_D) when the discriminator evaluates likelihoods."""
    posterior_error: float | None = None
    """The Bayesian error estimate of the declared outcome with equal priors, a soft decision."""


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


@dataclass(frozen=True)
class ThresholdDiscriminator:
    """Sum counts over ``window_s`` and declare bright iff n > n_c, with n_c half-integer."""

    n_c: float
    window_s: float

    def __post_init__(self) -> None:
        if self.window_s <= 0.0:
            raise ValueError("the window is positive")
        if abs(self.n_c - math.floor(self.n_c) - 0.5) > 1e-12:
            raise ValueError("the threshold is half-integer so that no count ties it")

    @property
    def sub_bin_s(self) -> float | None:
        return None

    @property
    def needs_arrivals(self) -> bool:
        return False

    def decide(self, record: PhotonRecord, model: RecordModel) -> Decision:
        n = record.total if record.window_s <= self.window_s else record.counts_before(self.window_s)
        return Decision(bright=n > self.n_c, time_used_s=self.window_s)

    def error_rates(
        self,
        model: RecordModel,
        *,
        bright_start: ReadoutClass | Sequence[float] = "bright",
        dark_start: ReadoutClass | Sequence[float] = "dark",
        n_max: int | None = None,
    ) -> tuple[float, float]:
        """Exact (epsilon_B, epsilon_D) from the chain's count distributions: overlap plus every pumping and decay path."""
        pb = model.count_distribution(bright_start, self.window_s, n_max=n_max)
        pd = model.count_distribution(dark_start, self.window_s, n_max=n_max)
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
    """The minimum over n_c at every window scanned, so the interior optimum in t_b is visible."""

    @property
    def discriminator(self) -> ThresholdDiscriminator:
        return ThresholdDiscriminator(self.best.n_c, self.best.window_s)


def optimize_threshold(
    model: RecordModel,
    windows_s: Sequence[float],
    *,
    bright_start: ReadoutClass | Sequence[float] = "bright",
    dark_start: ReadoutClass | Sequence[float] = "dark",
    convention: Convention = "mean",
) -> ThresholdOptimum:
    """Scan (n_c, t_b) for the interior optimum of the mean (or worst-case) error."""
    points: list[ThresholdScanPoint] = []
    for t in windows_s:
        pb = model.count_distribution(bright_start, t)
        pd = model.count_distribution(dark_start, t)
        n_max = min(pb.n_max, pd.n_max)
        best: ThresholdScanPoint | None = None
        for k in range(n_max):
            n_c = k + 0.5
            eps_b = 1.0 - pb.probability_above(n_c)
            eps_d = pd.probability_above(n_c)
            point = ThresholdScanPoint(float(t), n_c, eps_b, eps_d)
            score = point.eps if convention == "mean" else max(eps_b, eps_d)
            if best is None or score < (best.eps if convention == "mean" else max(best.eps_B, best.eps_D)):
                best = point
        assert best is not None
        points.append(best)
    key = (lambda p: p.eps) if convention == "mean" else (lambda p: max(p.eps_B, p.eps_D))
    return ThresholdOptimum(min(points, key=key), tuple(points))


def _decay_rate(model: RecordModel, dark_class: ReadoutClass) -> float:
    """The dark class's rate of becoming bright, Myerson's 1/tau (shelf decay, or R_b for a hyperfine dark state)."""
    return model.rates.get((dark_class, "bright"), 0.0)


def myerson_log_likelihoods(
    counts: Sequence[int] | np.ndarray,
    sub_bin_s: float,
    model: RecordModel,
    *,
    dark_class: ReadoutClass = "dark",
    include_decay: bool = True,
) -> tuple[float, float]:
    """(ln p_B, ln p_D) of Myerson 2008 Eqs. 1-2 by the O(N) log-domain recursion, first order in t_b/tau (< 1):
    p_B = prod_i B(n_i), p_D = (1 - t_b/tau) prod_i D(n_i) + (t_s/tau) sum_j prod_{i<j} D(n_i) prod_{i>=j} B(n_i)."""
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


def myerson_brute_force(
    counts: Sequence[int] | np.ndarray,
    sub_bin_s: float,
    model: RecordModel,
    *,
    dark_class: ReadoutClass = "dark",
) -> tuple[float, float]:
    """The O(N^2) sum of Myerson Eq. 1 evaluated directly (linear domain), the oracle for the recursion."""
    n = np.asarray(counts, dtype=int)
    lam_b = (model.detected_bright_per_s + model.background_per_s) * sub_bin_s
    lam_d = model.background_per_s * sub_bin_s
    b = np.exp(np.asarray(log_poisson_pmf(n, lam_b)))
    d = np.exp(np.asarray(log_poisson_pmf(n, lam_d)))
    rate = _decay_rate(model, dark_class)
    t_b = sub_bin_s * n.size
    p_b = float(np.prod(b))
    p_d = (1.0 - rate * t_b) * float(np.prod(d))
    for j in range(n.size):
        p_d += rate * sub_bin_s * float(np.prod(d[:j])) * float(np.prod(b[j:]))
    return p_b, p_d


def exact_log_likelihoods(
    counts: Sequence[int] | np.ndarray,
    sub_bin_s: float,
    model: RecordModel,
    *,
    dark_class: ReadoutClass = "dark",
    n_max: int | None = None,
) -> tuple[float, float]:
    """(ln p_B, ln p_D) by the hidden-Markov forward algorithm on the chain's exact sub-bin kernel, no approximation."""
    n = np.asarray(counts, dtype=int)
    nmax = int(n.max()) if n_max is None else n_max
    kernels = model.sub_bin_matrices(sub_bin_s, max(nmax, int(n.max())))
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


@dataclass(frozen=True)
class TimeResolvedML:
    """Myerson's maximum likelihood on sub-bin counts, ties to bright; ``exact`` uses the hidden-Markov likelihood."""

    sub_bin: float
    window: float
    dark_class: ReadoutClass = "dark"
    include_decay: bool = True
    exact: bool = False

    def __post_init__(self) -> None:
        if self.sub_bin <= 0.0 or self.window <= 0.0:
            raise ValueError("sub-bin and window are positive")
        if abs(round(self.window / self.sub_bin) * self.sub_bin - self.window) > 1e-9 * self.window:
            raise ValueError("the window is an integer number of sub-bins")

    @property
    def window_s(self) -> float:
        return self.window

    @property
    def sub_bin_s(self) -> float | None:
        return self.sub_bin

    @property
    def needs_arrivals(self) -> bool:
        return False

    @property
    def n_sub_bins(self) -> int:
        return int(round(self.window / self.sub_bin))

    def log_likelihoods(self, record: PhotonRecord, model: RecordModel) -> tuple[float, float]:
        counts = _sub_bin_counts(record, self.sub_bin, self.n_sub_bins)
        if self.exact:
            return exact_log_likelihoods(counts, self.sub_bin, model, dark_class=self.dark_class)
        return myerson_log_likelihoods(
            counts, self.sub_bin, model, dark_class=self.dark_class, include_decay=self.include_decay
        )

    def decide(self, record: PhotonRecord, model: RecordModel) -> Decision:
        log_pb, log_pd = self.log_likelihoods(record, model)
        bright = log_pb >= log_pd
        return Decision(bright, self.window, log_pb - log_pd, _posterior_error(log_pb, log_pd, bright))


def _sub_bin_counts(record: PhotonRecord, sub_bin_s: float, n_bins: int) -> np.ndarray:
    """The record's counts on the discriminator's sub-bin grid: as stored, re-binned to a multiple, or from arrivals."""
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
class AdaptiveML:
    """Bayesian early termination: stop once the declared outcome's posterior error is below ``cutoff_error``."""

    sub_bin: float
    window: float
    cutoff_error: float
    dark_class: ReadoutClass = "dark"
    include_decay: bool = False
    """Myerson found faster detection at little cost by omitting the decay in the adaptive analysis."""

    def __post_init__(self) -> None:
        if self.sub_bin <= 0.0 or self.window <= 0.0 or not 0.0 < self.cutoff_error < 0.5:
            raise ValueError("positive sub-bin and window; the cutoff is an error probability in (0, 0.5)")

    @property
    def window_s(self) -> float:
        return self.window

    @property
    def sub_bin_s(self) -> float | None:
        return self.sub_bin

    @property
    def needs_arrivals(self) -> bool:
        return False

    def decide(self, record: PhotonRecord, model: RecordModel) -> Decision:
        n_bins = int(round(self.window / self.sub_bin))
        counts = _sub_bin_counts(record, self.sub_bin, n_bins)
        log_pb = log_pd = 0.0
        bright = True
        err = 0.5
        used = self.window
        for k in range(1, n_bins + 1):
            log_pb, log_pd = myerson_log_likelihoods(
                counts[:k], self.sub_bin, model, dark_class=self.dark_class, include_decay=self.include_decay
            )
            bright = log_pb >= log_pd
            err = _posterior_error(log_pb, log_pd, bright)
            if err < self.cutoff_error:
                used = k * self.sub_bin
                break
        return Decision(bright, used, log_pb - log_pd, err)


@dataclass(frozen=True)
class FirstPhoton:
    """Crain's stop on the first photon (``cutoff_s`` None) or Noek's rule: 0 counts dark, >= 2 bright at the second
    arrival, a lone photon bright iff it came before ``cutoff_s``."""

    window: float
    cutoff_s: float | None = None

    def __post_init__(self) -> None:
        if self.window <= 0.0 or (self.cutoff_s is not None and not 0.0 <= self.cutoff_s <= self.window):
            raise ValueError("positive window; the cutoff lies inside it")

    @property
    def window_s(self) -> float:
        return self.window

    @property
    def sub_bin_s(self) -> float | None:
        return None

    @property
    def needs_arrivals(self) -> bool:
        return True

    def decide(self, record: PhotonRecord, model: RecordModel) -> Decision:
        if record.arrivals_s is None:
            raise ValueError("first-photon protocols need arrival times")
        arr = record.arrivals_s[record.arrivals_s < self.window]
        if arr.size == 0:
            return Decision(False, self.window)
        if self.cutoff_s is None:
            return Decision(True, float(arr[0]))
        if arr[0] < self.cutoff_s:
            return Decision(True, float(arr[0]))
        if arr.size >= 2:
            return Decision(True, float(arr[1]))
        return Decision(False, self.window)


def average_detection_time_s(decisions: Sequence[Decision]) -> float:
    return float(np.mean([d.time_used_s for d in decisions])) if decisions else 0.0


def camera_pixel_means(
    geometry: CameraGeometry,
    models: Sequence[RecordModel],
    bright: Sequence[bool],
    exposure_s: float,
    *,
    read_noise_counts: float = 0.0,
) -> np.ndarray:
    """Expected counts per pixel for a register configuration (bright flags), the detector's background spread once."""
    background = max((m.background_per_s for m in models), default=0.0)
    mean = np.full(
        (geometry.n_rows, geometry.n_columns),
        read_noise_counts + background * exposure_s / geometry.n_pixels,
        dtype=float,
    )
    for i, (m, b) in enumerate(zip(models, bright)):
        if b:
            mean += geometry.weights(i) * m.detected_bright_per_s * exposure_s
    return mean


def camera_log_likelihoods(
    image: np.ndarray,
    geometry: CameraGeometry,
    models: Sequence[RecordModel],
    exposure_s: float,
    ion: int,
    roi: np.ndarray,
    neighbours_bright: Sequence[bool],
    *,
    read_noise_counts: float = 0.0,
) -> tuple[float, float]:
    """(ln p_B, ln p_D) of ion ``ion`` over its ROI pixels with the other ions in the given states, Burrell's per-pixel
    Poisson likelihoods (all-dark neighbours give the independent analysis)."""
    counts = np.asarray(image).ravel()[roi]
    out: list[float] = []
    for state in (True, False):
        flags = list(neighbours_bright)
        flags[ion] = state
        mean = camera_pixel_means(
            geometry, models, flags, exposure_s, read_noise_counts=read_noise_counts
        ).ravel()[roi]
        out.append(float(sum(float(log_poisson_pmf(int(c), float(mu))) for c, mu in zip(counts, mean))))
    return out[0], out[1]


@dataclass(frozen=True)
class CameraDecode:
    bright: tuple[bool, ...]
    log_ratios: tuple[float, ...]
    """R_k with sign: ln(p_B/p_D) per ion."""
    iterations: int

    @property
    def register_error_estimate(self) -> float:
        """~ sum_k e^{-R_k}, R_k = |ln(p_B/p_D)| (Burrell)."""
        return float(sum(math.exp(-abs(r)) for r in self.log_ratios))


def decode_camera_image(
    image: np.ndarray,
    geometry: CameraGeometry,
    models: Sequence[RecordModel],
    exposure_s: float,
    *,
    roi_pixels: int,
    neighbours: bool = True,
    max_iterations: int = 20,
    read_noise_counts: float = 0.0,
) -> CameraDecode:
    """Spatial maximum likelihood per ion in brightness order; with ``neighbours`` the likelihoods are conditioned on
    the neighbour states and decoded by iterated conditional modes seeded all-dark until the register is stable."""
    n = geometry.n_ions
    rois = [geometry.roi(k, roi_pixels) for k in range(n)]
    flags = [False] * n
    ratios = [0.0] * n
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        new_flags: list[bool] = []
        for k in range(n):
            context = flags if neighbours else [False] * n
            lb, ld = camera_log_likelihoods(
                image, geometry, models, exposure_s, k, rois[k], context, read_noise_counts=read_noise_counts
            )
            ratios[k] = lb - ld
            new_flags.append(lb >= ld)
        stable = new_flags == flags
        flags = new_flags
        if stable or not neighbours:
            break
    return CameraDecode(tuple(flags), tuple(ratios), iterations)


def camera_threshold_decode(
    image: np.ndarray, geometry: CameraGeometry, roi_pixels: int, thresholds: Sequence[float]
) -> tuple[bool, ...]:
    """Pixel-count thresholding over independent brightness-order ROIs, one threshold per ion (Burrell's method T)."""
    flat = np.asarray(image).ravel()
    return tuple(
        float(flat[geometry.roi(k, roi_pixels)].sum()) > thresholds[k] for k in range(geometry.n_ions)
    )


def spatio_temporal_log_likelihoods(
    per_exposure: Sequence[tuple[float, float]], exposure_s: float, decay_rate_per_s: float
) -> tuple[float, float]:
    """Burrell's spatio-temporal form: p_B = prod_j p_Bj, p_D = (1 - M t_s/tau) prod_j p_Dj + (t_s/tau) sum_j' prod_{j<j'} p_Dj
    prod_{j>=j'} p_Bj from the per-exposure (ln p_Bj, ln p_Dj), the Myerson recursion over exposures instead of sub-bins."""
    lb = np.array([x[0] for x in per_exposure])
    ld = np.array([x[1] for x in per_exposure])
    m = lb.size
    if decay_rate_per_s * exposure_s * m >= 1.0:
        raise ValueError("first-order decay weights need M t_s/tau < 1")
    log_pb = float(lb.sum())
    if decay_rate_per_s <= 0.0:
        return log_pb, float(ld.sum())
    log_m, log_s = 0.0, -math.inf
    for k in range(m):
        log_s = np.logaddexp(log_s, log_m) + lb[k]
        log_m = log_m + ld[k]
    log_pd = float(
        np.logaddexp(
            math.log(1.0 - decay_rate_per_s * exposure_s * m) + log_m,
            math.log(decay_rate_per_s * exposure_s) + log_s,
        )
    )
    return log_pb, log_pd


@dataclass(frozen=True)
class RegisterConfusion:
    """P(declared bits | true internal levels) factored by neighbour range: per ion a table over its neighbourhood, its
    own axis over its levels (transfer folded in) and each neighbour's over its start class (marginalized)."""

    n_ions: int
    neighbour_range: int
    tables: tuple[np.ndarray, ...]
    """tables[i][(x_j for j in neighbourhood(i)) + (declared,)]: x_i is ion i's LEVEL, x_j (j != i) the neighbour's class
    (0 = bright, 1 = non-bright), declared 0 = bright."""
    bright_start: tuple[np.ndarray, ...]
    """Per ion, P(start class == bright | internal level) over its levels: how a level gates the light it leaks."""

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

    def dense(self) -> np.ndarray:
        """The 2^N x 2^N tensor over (true LEVEL pattern, declared bright pattern), index bit i = ion i's qubit level for
        the rows and 1 = declared bright for the columns."""
        if self.n_ions > 12:
            raise ValueError(
                "the dense confusion tensor is guarded to N <= 12 (Section 5.7); use probability()/sample()"
            )
        size = 2**self.n_ions
        out = np.zeros((size, size))
        for a in range(size):
            levels = [(a >> i) & 1 for i in range(self.n_ions)]
            pb = np.array([self.bright_probability(i, levels) for i in range(self.n_ions)])
            for b in range(size):
                q = 1.0
                for i in range(self.n_ions):
                    q *= pb[i] if (b >> i) & 1 else 1.0 - pb[i]
                out[a, b] = q
        return out


@dataclass(frozen=True)
class POVM:
    """The readout POVM, computed once per device: a per-ion product form (exact only at zero readout crosstalk) or a
    register-wide confusion (dense to N = 12, factored beyond)."""

    per_ion: tuple[np.ndarray, ...] | None
    """Product form: per ion an (n_levels, 2) matrix P(declared | level), columns (bright, dark), transfer folded in."""
    confusion: np.ndarray | None
    """Register-wide 2^N x 2^N tensor, rows the true qubit-level pattern; None beyond N = 12."""
    crosstalk_domain: Literal["zero", "configured"]
    factored: RegisterConfusion | None = None
    """The neighbour-range factored form the dense tensor was built from (kept beyond N = 12, where ``confusion`` is None)."""
    uncertainty: float = 0.0
    """Statistical uncertainty of Monte-Carlo-estimated entries (0 for exact ones)."""
    bright_levels: tuple[int, ...] = ()
    """Per ion of the product form, the qubit level whose ideal class is bright."""

    def __post_init__(self) -> None:
        product_form = self.per_ion is not None
        register_form = self.confusion is not None or self.factored is not None
        if product_form == register_form:
            raise ValueError("a POVM is either the product form or the register-wide confusion tensor")
        if self.crosstalk_domain == "zero" and not product_form:
            raise ValueError("the zero-crosstalk domain is the product form")
        if self.crosstalk_domain == "configured" and product_form:
            raise ValueError(
                "the product form cannot carry configured crosstalk: Section 8.5 makes the records "
                "neighbour-coupled, so the configured domain is the register-wide confusion tensor"
            )
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
        if self.confusion is not None:
            n = self.confusion.shape[0]
            if self.confusion.shape != (n, n) or not np.allclose(self.confusion.sum(axis=1), 1.0, atol=1e-9):
                raise ValueError("the confusion tensor is square and row-stochastic (E_bright + E_dark = 1)")

    @property
    def n_ions(self) -> int:
        if self.per_ion is not None:
            return len(self.per_ion)
        if self.factored is not None:
            return self.factored.n_ions
        assert self.confusion is not None
        return int(round(math.log2(self.confusion.shape[0])))

    def declared_bright_probability(
        self, true_levels: Sequence[int], declared_bright: Sequence[bool]
    ) -> float:
        """P(this declared-bright pattern | the ions' true internal levels)."""
        if self.per_ion is not None:
            p = 1.0
            for m, lev, d in zip(self.per_ion, true_levels, declared_bright):
                p *= float(m[int(lev), 0 if d else 1])
            return p
        if self.factored is not None:
            return self.factored.probability(true_levels, declared_bright)
        assert self.confusion is not None
        a = sum(int(lev) << i for i, lev in enumerate(true_levels))
        b = sum(int(d) << i for i, d in enumerate(declared_bright))
        return float(self.confusion[a, b])

    def sample(self, true_levels: Sequence[int], rng: np.random.Generator) -> tuple[bool, ...]:
        """One declared-bright pattern drawn from the confusion, given the projectively sampled internal levels."""
        if self.per_ion is not None:
            return tuple(rng.random() < float(m[int(lev), 0]) for m, lev in zip(self.per_ion, true_levels))
        if self.factored is not None:
            return self.factored.sample(true_levels, rng)
        assert self.confusion is not None
        a = sum(int(lev) << i for i, lev in enumerate(true_levels))
        b = int(rng.choice(self.confusion.shape[1], p=self.confusion[a]))
        return tuple(bool((b >> i) & 1) for i in range(self.n_ions))

    def per_ion_errors(self) -> tuple[tuple[float, float], ...]:
        """(epsilon_B, epsilon_D) per ion of the product form, the scheme's transfer channel included."""
        if self.per_ion is None:
            raise ValueError(
                "per-ion errors are defined for the product form; marginalize the register form instead"
            )
        return tuple((float(m[b, 1]), float(m[1 - b, 0])) for m, b in zip(self.per_ion, self.bright_levels))


def level_start_vectors(scheme: ReadoutScheme) -> tuple[np.ndarray, ...]:
    """Per internal level, the start distribution over CLASSES: the transfer channel as a stochastic map."""
    out: list[np.ndarray] = []
    for lev in range(scheme.n_levels):
        v = np.zeros(3)
        for c, p in scheme.start_distribution(lev).items():
            v[CLASS_INDEX[c]] += p
        out.append(v)
    return tuple(out)


def bright_start_probabilities(scheme: ReadoutScheme) -> np.ndarray:
    """P(start class == bright | internal level) over the scheme's levels: how a level gates the light it leaks."""
    return np.array([v[CLASS_INDEX["bright"]] for v in level_start_vectors(scheme)])


def per_ion_confusion(
    model: RecordModel,
    scheme: ReadoutScheme,
    discriminator: Discriminator,
    *,
    n_samples: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """P(declared | internal level) as an (n_levels, 2) matrix, and its uncertainty: exact through the count
    distributions for a threshold discriminator, Monte Carlo over ``n_samples`` records otherwise."""
    starts = level_start_vectors(scheme)
    out = np.zeros((len(starts), 2))
    if isinstance(discriminator, ThresholdDiscriminator):
        for row, start in enumerate(starts):
            dist = model.count_distribution(list(start), discriminator.window_s)
            p_bright = dist.probability_above(discriminator.n_c)
            out[row] = (p_bright, 1.0 - p_bright)
        return out, 0.0
    if n_samples <= 0:
        raise ValueError("a non-threshold discriminator needs Monte Carlo samples (n_samples > 0)")
    gen = rng if rng is not None else np.random.default_rng(0)
    for row, start in enumerate(starts):
        hits = 0
        for _ in range(n_samples):
            cls = CLASSES[int(gen.choice(3, p=start))]
            rec = model.sample_record(
                cls,
                discriminator.window_s,
                gen,
                sub_bin_s=discriminator.sub_bin_s,
                arrivals=discriminator.needs_arrivals,
            )
            hits += int(discriminator.decide(rec, model).bright)
        p = hits / n_samples
        out[row] = (p, 1.0 - p)
    return out, math.sqrt(0.25 / n_samples)


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
    return POVM(tuple(mats), None, "zero", None, unc, tuple(s.bright_level for s in schemes))


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
    """The register-wide confusion at configured crosstalk: each ion's decision given its own level and its neighbours'
    start classes (their leaked light as added counts); dense 2^N x 2^N to N = 12, factored beyond."""
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
            start = starts_i[pattern[neigh.index(i)]]
            if isinstance(discriminator, ThresholdDiscriminator):
                dist = frozen.count_distribution(list(start), discriminator.window_s)
                p_bright = dist.probability_above(discriminator.n_c)
            else:
                if n_samples <= 0:
                    raise ValueError(
                        "a non-threshold discriminator needs Monte Carlo samples (n_samples > 0)"
                    )
                gen = rng if rng is not None else np.random.default_rng(i)
                hits = 0
                for _ in range(n_samples):
                    cls = CLASSES[int(gen.choice(3, p=start))]
                    rec = frozen.sample_record(
                        cls,
                        discriminator.window_s,
                        gen,
                        sub_bin_s=discriminator.sub_bin_s,
                        arrivals=discriminator.needs_arrivals,
                    )
                    hits += int(discriminator.decide(rec, frozen).bright)
                p_bright = hits / n_samples
                unc = max(unc, math.sqrt(0.25 / n_samples))
            table[pattern + (0,)] = p_bright
            table[pattern + (1,)] = 1.0 - p_bright
        tables.append(table)
    factored = RegisterConfusion(
        n, rng_range, tuple(tables), tuple(bright_start_probabilities(s) for s in schemes)
    )
    dense = factored.dense() if n <= 12 else None
    return POVM(None, dense, "configured", factored, unc)


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
    """The largest difference, over ions, internal levels and neighbourhood configurations, between the register-wide
    confusion and the product POVM; read off the factored tables, so it is defined beyond N = 12."""
    if register.factored is None:
        raise ValueError("the discrepancy is measured against the register-wide (factored) form")
    if product_form.per_ion is None:
        raise ValueError("the discrepancy is measured against the product form")
    fac = register.factored
    worst = 0.0
    for i, table in enumerate(fac.tables):
        neigh = fac.neighbourhood(i)
        own = neigh.index(i)
        rows = product_form.per_ion[i]
        for pattern in product(*(range(k) for k in table.shape[:-1])):
            level = pattern[own]
            if level >= rows.shape[0]:
                raise ValueError("the product POVM has no row for a level the register form carries")
            worst = max(worst, abs(float(table[pattern + (0,)]) - float(rows[level, 0])))
    return worst


@dataclass(frozen=True)
class ReadoutOutcome:
    """What the measurement stage returns for a batch of shots."""

    bits: np.ndarray
    """(shots, n_ions) declared bits, column j = qubit j."""
    levels: np.ndarray
    """(shots, n_ions) the internal level of each ion, sampled jointly from the register."""
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
    """Sample the joint internal outcome projectively, then either generate and discriminate every ion's photon record,
    neighbour-coupled by ``leakage`` and ``depumping`` (``full``), or apply the POVM to the sampled levels (``fast``);
    never both. Randomness is keyed by (sample, trajectory, shot, ion, channel)."""
    n = space.n_ions
    if len(schemes) != n or len(models) != n:
        raise ValueError("one scheme and one record model per ion")
    probs = joint_level_probabilities(space, state)
    bits = np.zeros((shots, n), dtype=np.uint8)
    levels = np.zeros((shots, n), dtype=np.uint8)
    times = np.zeros((shots, n))
    posts = np.full((shots, n), np.nan)
    records: list[tuple[PhotonRecord, ...]] = []
    if mode == "fast":
        if povm is None:
            povm = povm_for(models, schemes, discriminator, leakage, depumping=depumping)
    for s in range(shots):
        shot = first_shot + s
        rng_outcome = np.random.default_rng(seeds.child(sample, trajectory, shot, 0, "register_outcome"))
        outcome = sample_joint_outcome(probs, rng_outcome)
        levels[s] = outcome
        if mode == "fast":
            # the POVM rows already include the transfer channel: sampling a start class here would apply it twice
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


def confusion_from_outcomes(outcome: ReadoutOutcome, schemes: Sequence[ReadoutScheme]) -> np.ndarray:
    """The empirical per-ion confusion P(declared bit | true level) as an (n_ions, 2, 2) array over (level, declared bit)."""
    n = outcome.bits.shape[1]
    out = np.zeros((n, 2, 2))
    for i in range(n):
        for lev in (0, 1):
            mask = outcome.levels[:, i] == lev
            if np.any(mask):
                out[i, lev, 1] = float(np.mean(outcome.bits[mask, i]))
                out[i, lev, 0] = 1.0 - out[i, lev, 1]
    return out


def povm_confusion_over_levels(povm: POVM, schemes: Sequence[ReadoutScheme]) -> np.ndarray:
    """The product POVM as (n_ions, 2, 2) over (qubit level, declared bit), like :func:`confusion_from_outcomes`."""
    if povm.per_ion is None:
        raise ValueError("re-indexing by level is defined for the product form")
    out = np.zeros((len(schemes), 2, 2))
    for i, (m, s) in enumerate(zip(povm.per_ion, schemes)):
        for lev in (0, 1):
            p_bright = float(m[lev, 0])
            out[i, lev, s.bright_level] = p_bright
            out[i, lev, 1 - s.bright_level] = 1.0 - p_bright
    return out


@dataclass(frozen=True)
class BudgetLine:
    """A named error mechanism's contribution to epsilon_B (bright read dark) and epsilon_D (dark read bright)."""

    name: str
    eps_B: float
    eps_D: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.eps_B < 0.0 or self.eps_D < 0.0:
            raise ValueError("error contributions are non-negative")

    @classmethod
    def averaged(cls, name: str, eps: float, source: str = "") -> BudgetLine:
        """A row quoted state-averaged (Harty's, Christensen's tables), entered in both states with the quoted value."""
        return cls(name, eps, eps, source)


@dataclass(frozen=True)
class ReadoutBudget:
    """Named channels, never one scalar: preparation, transfer, shelf decay, discrimination, crosstalk."""

    lines: tuple[BudgetLine, ...]
    convention: Convention = "mean"
    """'mean': epsilon = (epsilon_B + epsilon_D)/2 (Myerson, Burrell, Harty, Christensen); 'min': Acton's worst case."""

    @property
    def eps_B(self) -> float:
        return float(sum(line.eps_B for line in self.lines))

    @property
    def eps_D(self) -> float:
        return float(sum(line.eps_D for line in self.lines))

    @property
    def error(self) -> float:
        return 0.5 * (self.eps_B + self.eps_D) if self.convention == "mean" else max(self.eps_B, self.eps_D)

    @property
    def fidelity(self) -> float:
        return 1.0 - self.error

    def register_fidelity(self, n_qubits: int) -> float:
        """F_s = (1 - epsilon_s)^N_q under i.i.d. errors (Christensen)."""
        return (1.0 - self.error) ** n_qubits

    def usable_qubits(self) -> float:
        """N_q < ln 2/epsilon_s, the register size at which F_s drops to 1/2."""
        return math.log(2.0) / self.error if self.error > 0.0 else math.inf

    def by_name(self) -> dict[str, tuple[float, float]]:
        return {line.name: (line.eps_B, line.eps_D) for line in self.lines}
