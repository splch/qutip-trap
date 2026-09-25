"""The photon-record layer: the bright/dark/shelf Markov chain, photon counts, background, detector non-idealities and
readout crosstalk (PLAN.md Section 8.2).

Given each ion's start class, the record over a window is a Poisson process at the piecewise rate of the continuous-time
Markov chain plus background. Records are sampled exactly (Gillespie holding times, Poisson counts per constant-rate cell);
the exact count distribution for any number of jumps is the Markov-modulated Poisson process solved by one matrix
exponential (:meth:`RecordModel.count_distribution`), which the threshold optimizer and the POVM integrate. epsilon_sys is
applied once, at scattered -> detected rate, and R_bg is separate; a bright neighbour's leaked light is added counts on its
neighbour plus extra pumping of the neighbour's chain (Wineland's mechanism), never a change of the efficiency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.linalg import expm
from scipy.sparse import block_diag, csr_matrix, diags, identity, kron
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaln, j1

from qutip_trap.readout.fluorescence import CLASSES, FluorescenceRates, ReadoutClass

CLASS_INDEX: dict[ReadoutClass, int] = {c: k for k, c in enumerate(CLASSES)}


@dataclass(frozen=True)
class Detector:
    """The photon detector of the device: its kind, the total system detection efficiency, the background rate (counts/s),
    the point-spread leakage onto neighbours by distance, the optional dead time (s) and afterpulse probability, the
    detection window (s) and, for a camera, the numerical aperture, the object-plane pixel pitch (m) and the read noise per
    pixel per readout (counts)."""

    kind: Literal["pmt", "camera", "snspd"]
    efficiency: float
    """Total system detection efficiency epsilon_sys."""
    background_cps: float
    psf_leakage: dict[int, float]
    """Neighbour distance -> the fraction of one ion's detected light landing on the neighbour's detector."""
    dead_time_s: float | None
    afterpulse_prob: float | None
    window_s: float
    numerical_aperture: float | None = None
    pixel_m: float | None = None
    read_noise_counts: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("detection efficiency lies in (0, 1]")
        if self.background_cps < 0.0 or self.window_s <= 0.0:
            raise ValueError("background is non-negative and the window positive")
        if any(not 0.0 <= f <= 1.0 for f in self.psf_leakage.values()):
            raise ValueError("PSF leakage fractions lie in [0, 1]")
        if any(d <= 0 for d in self.psf_leakage):
            raise ValueError("PSF leakage is keyed by a positive neighbour distance in ion indices")
        if self.dead_time_s is not None and self.dead_time_s < 0.0:
            raise ValueError("dead time is non-negative")
        if self.afterpulse_prob is not None and not 0.0 <= self.afterpulse_prob < 1.0:
            raise ValueError("afterpulse probability lies in [0, 1)")
        if self.numerical_aperture is not None and not 0.0 < self.numerical_aperture <= 1.0:
            raise ValueError("NA lies in (0, 1]")
        if self.pixel_m is not None and self.pixel_m <= 0.0:
            raise ValueError("pixel pitch is positive")
        if self.read_noise_counts < 0.0:
            raise ValueError("read noise is non-negative")


def log_poisson_pmf(n: np.ndarray | int, mean: float) -> np.ndarray | float:
    """ln P_p(n; mu) = n ln mu - mu - ln n! (mu = 0 gives ln delta_{n0})."""
    k = np.asarray(n, dtype=float)
    if mean <= 0.0:
        out = np.where(k == 0, 0.0, -np.inf)
    else:
        out = k * math.log(mean) - mean - gammaln(k + 1.0)
    return float(out) if np.ndim(n) == 0 else np.asarray(out)


# ---- the record model: the Markov chain and its exact count statistics ---------------------------------------------------


@dataclass(frozen=True)
class ClassPath:
    """One realization of the readout chain over a window: the start class and the jumps, in time order."""

    window_s: float
    start: ReadoutClass
    jump_times_s: tuple[float, ...]
    jump_targets: tuple[ReadoutClass, ...]

    def __post_init__(self) -> None:
        times = (0.0, *self.jump_times_s, self.window_s)
        if len(self.jump_times_s) != len(self.jump_targets) or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("one target per jump, and the jump times ordered inside the window")

    def segments(self) -> list[tuple[float, float, ReadoutClass]]:
        """[(t_start, t_end, class)] covering [0, window)."""
        out: list[tuple[float, float, ReadoutClass]] = []
        t0, cls = 0.0, self.start
        for t, target in zip(self.jump_times_s, self.jump_targets):
            out.append((t0, t, cls))
            t0, cls = t, target
        out.append((t0, self.window_s, cls))
        return out


@dataclass(frozen=True)
class PhotonRecord:
    """One ion's record in one shot: total, sub-bin counts, arrival times when kept, and the class path behind it."""

    ion: int
    window_s: float
    total: int
    sub_bin_s: float | None = None
    sub_bins: np.ndarray | None = None
    arrivals_s: np.ndarray | None = None
    path: ClassPath | None = None

    def __post_init__(self) -> None:
        if self.sub_bins is not None and int(np.sum(self.sub_bins)) != self.total:
            raise ValueError("sub-bin counts must sum to the total")
        if self.arrivals_s is not None and len(self.arrivals_s) != self.total:
            raise ValueError("one arrival time per detected photon")

    def counts_before(self, t_s: float) -> int:
        """Counts up to time t (needs arrival times, or sub-bins when t is a sub-bin edge)."""
        if self.arrivals_s is not None:
            return int(np.count_nonzero(self.arrivals_s < t_s))
        if self.sub_bins is not None and self.sub_bin_s is not None:
            k = int(round(t_s / self.sub_bin_s))
            return int(np.sum(self.sub_bins[:k]))
        if t_s >= self.window_s:
            return self.total
        raise ValueError("a time-resolved query needs arrival times or sub-bins")


@dataclass(frozen=True)
class CountDistribution:
    """P(n | start class) over a window, exact for the chain with any number of jumps, truncated at n_max."""

    pmf: np.ndarray
    overflow: float
    """Probability mass at n > n_max."""

    @property
    def n_max(self) -> int:
        return int(len(self.pmf) - 1)

    def probability_above(self, threshold: float) -> float:
        """P(n > threshold), the overflow mass included."""
        n = np.arange(len(self.pmf))
        return float(np.sum(self.pmf[n > threshold])) + self.overflow

    def mean(self) -> float:
        return float(np.dot(np.arange(len(self.pmf)), self.pmf))


@dataclass(frozen=True)
class RecordModel:
    """The chain and the detected rates of one ion: epsilon_sys R_o, R_bg and the (from, to) transition rates."""

    detected_bright_per_s: float
    background_per_s: float
    rates: dict[tuple[ReadoutClass, ReadoutClass], float]
    dead_time_s: float | None = None
    afterpulse_prob: float | None = None

    def __post_init__(self) -> None:
        if self.detected_bright_per_s <= 0.0 or self.background_per_s < 0.0:
            raise ValueError("the detected bright rate is positive and the background non-negative")
        for (a, b), r in self.rates.items():
            if a == b or a not in CLASSES or b not in CLASSES or r < 0.0:
                raise ValueError(f"malformed transition ({a}, {b}) = {r}")

    @classmethod
    def from_rates(cls, rates: FluorescenceRates, detector: Detector) -> RecordModel:
        """epsilon_sys applied once here; the detector's dead time and afterpulsing carried along."""
        detected, background = rates.detected(detector)
        return cls(
            detected_bright_per_s=detected,
            background_per_s=background,
            rates=rates.transition_rates(),
            dead_time_s=detector.dead_time_s,
            afterpulse_prob=detector.afterpulse_prob,
        )

    def generator(self) -> np.ndarray:
        """The 3 x 3 generator Q (rows sum to zero) in the CLASSES order."""
        q = np.zeros((3, 3))
        for (a, b), r in self.rates.items():
            q[CLASS_INDEX[a], CLASS_INDEX[b]] += r
        q -= np.diag(q.sum(axis=1))
        return q

    def class_count_rates(self) -> np.ndarray:
        """The detected rate in each class: eps R_o + R_bg in bright, R_bg elsewhere."""
        lam = np.full(3, self.background_per_s)
        lam[CLASS_INDEX["bright"]] += self.detected_bright_per_s
        return lam

    def _start_vector(self, start: ReadoutClass | Sequence[float]) -> np.ndarray:
        if isinstance(start, str):
            v = np.zeros(3)
            v[CLASS_INDEX[start]] = 1.0
            return v
        v = np.asarray(start, dtype=float)
        if v.shape != (3,) or np.any(v < 0.0) or not math.isclose(float(v.sum()), 1.0, abs_tol=1e-9):
            raise ValueError("a start distribution has three non-negative weights summing to 1")
        return v

    def bright_occupancy_mean_s(self, start: ReadoutClass | Sequence[float], window_s: float) -> float:
        """E[time spent bright] over the window, int_0^t P(bright at tau) d tau, by an augmented exponential."""
        q = self.generator()
        aug = np.zeros((6, 6))
        aug[:3, :3] = q
        aug[:3, 3:] = np.eye(3)
        block = expm(aug * window_s)[:3, 3:]
        return float(self._start_vector(start) @ block[:, CLASS_INDEX["bright"]])

    def mean_counts(self, start: ReadoutClass | Sequence[float], window_s: float) -> float:
        """E[n] = eps R_o E[bright time] + R_bg t, exact."""
        return (
            self.detected_bright_per_s * self.bright_occupancy_mean_s(start, window_s)
            + self.background_per_s * window_s
        )

    def suggested_n_max(self, window_s: float) -> int:
        """A count cutoff 12 sigma above the bright mean, beyond which the overflow mass is negligible."""
        mu = (self.detected_bright_per_s + self.background_per_s) * window_s
        return int(math.ceil(mu + 12.0 * math.sqrt(mu + 1.0))) + 5

    def _augmented_generator(self, n_max: int, lam: np.ndarray) -> csr_matrix:
        """The sparse generator of the Markov-modulated Poisson process on (count n, class): blocks Q - diag(lambda) on the
        diagonal, diag(lambda) one block up (a count increments n), and the bare Q on the last block, which absorbs n > n_max
        without counting further (its mass is the overflow). Its exponential is only ever applied to vectors."""
        q = np.asarray(self.generator(), dtype=float)
        counting = csr_matrix(q - np.diag(lam))
        diagonal = block_diag([kron(identity(n_max + 1, format="csr"), counting), csr_matrix(q)])
        shift = diags([np.ones(n_max + 1)], [1], shape=(n_max + 2, n_max + 2))
        return csr_matrix(diagonal + kron(shift, csr_matrix(np.diag(lam))))

    def count_distribution(
        self, start: ReadoutClass | Sequence[float], window_s: float, *, n_max: int | None = None
    ) -> CountDistribution:
        """The exact P(n | start) over the window, by the action of one matrix exponential of the augmented generator."""
        if window_s < 0.0:
            raise ValueError("the window is non-negative")
        nmax = self.suggested_n_max(window_s) if n_max is None else int(n_max)
        if nmax < 0:
            raise ValueError("n_max is non-negative")
        lam = self.class_count_rates()
        d = 3
        size = d * (nmax + 2)
        a = self._augmented_generator(nmax, lam)
        p0 = np.zeros(size)
        p0[:d] = self._start_vector(start)
        # p0 @ expm(A w) = expm(A^T w) @ p0
        p = np.asarray(expm_multiply((a.T * window_s).tocsc(), p0)).reshape(nmax + 2, d)
        pmf = p[: nmax + 1].sum(axis=1)
        overflow = float(p[nmax + 1].sum())
        return CountDistribution(pmf=np.clip(pmf, 0.0, None), overflow=max(overflow, 0.0))

    def sub_bin_matrices(self, sub_bin_s: float, n_max: int) -> np.ndarray:
        """M[n, i, j] = P(n counts in a sub-bin and class j at its end | class i at its start): the exact hidden-Markov
        emission and transition kernel of the time-resolved record."""
        lam = self.class_count_rates()
        d = 3
        size = d * (n_max + 2)
        a = self._augmented_generator(n_max, lam)
        # the first d rows of expm(A tau) are the columns of expm(A^T tau) applied to the unit vectors of the count-0 block
        starts = np.zeros((size, d))
        starts[:d, :d] = np.eye(d)
        rows = np.asarray(expm_multiply((a.T * sub_bin_s).tocsc(), starts)).T
        out = np.zeros((n_max + 2, d, d))
        for n in range(n_max + 2):
            out[n] = rows[:, n * d : (n + 1) * d]
        return out

    def sample_path(self, start: ReadoutClass, window_s: float, rng: np.random.Generator) -> ClassPath:
        """Gillespie sampling of the chain: exponential holding times at the total exit rate, targets by rate ratio."""
        t, cls = 0.0, start
        times: list[float] = []
        targets: list[ReadoutClass] = []
        exits = {c: [(b, r) for (a, b), r in self.rates.items() if a == c and r > 0.0] for c in CLASSES}
        while True:
            options = exits[cls]
            total = sum(r for _b, r in options)
            if total <= 0.0:
                break
            dt = rng.exponential(1.0 / total)
            if t + dt >= window_s:
                break
            t += dt
            u = rng.random() * total
            acc = 0.0
            nxt: ReadoutClass = options[-1][0]
            for b, r in options:
                acc += r
                if u < acc:
                    nxt = b
                    break
            times.append(t)
            targets.append(nxt)
            cls = nxt
        return ClassPath(window_s, start, tuple(times), tuple(targets))

    def sample_record(
        self,
        start: ReadoutClass,
        window_s: float,
        rng: np.random.Generator,
        *,
        ion: int = 0,
        sub_bin_s: float | None = None,
        arrivals: bool = False,
        path: ClassPath | None = None,
        extra_rate: Sequence[tuple[float, float, float]] = (),
    ) -> PhotonRecord:
        """One record: Poisson counts at the piecewise rate of the (given or sampled) path plus background and the
        ``extra_rate`` segments [(t_start, t_end, rate)] of neighbours' light; dead time and afterpulsing act on arrival
        times, which are then always generated."""
        p = path if path is not None else self.sample_path(start, window_s, rng)
        need_arrivals = arrivals or self.dead_time_s is not None or self.afterpulse_prob is not None
        cells = _rate_cells(
            p, self.detected_bright_per_s, self.background_per_s, extra_rate, sub_bin_s, window_s
        )
        if need_arrivals:
            times: list[np.ndarray] = []
            for a, b, rate in cells:
                k = rng.poisson(rate * (b - a))
                if k:
                    times.append(rng.uniform(a, b, size=k))
            arr = np.sort(np.concatenate(times)) if times else np.zeros(0)
            arr = apply_detector_nonidealities(arr, self.dead_time_s, self.afterpulse_prob, rng, window_s)
            sub = None
            if sub_bin_s is not None:
                n_bins = int(round(window_s / sub_bin_s))
                sub = np.histogram(arr, bins=n_bins, range=(0.0, window_s))[0]
            return PhotonRecord(ion, window_s, int(arr.size), sub_bin_s, sub, arr, p)
        # one vectorized Poisson draw over the cells, which draws the same numbers as one scalar draw per cell in order
        means = np.array([rate * (b - a) for a, b, rate in cells], dtype=float)
        counts = rng.poisson(means) if means.size else np.zeros(0, dtype=int)
        if sub_bin_s is None:
            return PhotonRecord(ion, window_s, int(counts.sum()), None, None, None, p)
        n_bins = int(round(window_s / sub_bin_s))
        sub = np.zeros(n_bins, dtype=int)
        index = np.minimum((np.array([a for a, _b, _r in cells]) / sub_bin_s + 1e-9).astype(int), n_bins - 1)
        np.add.at(sub, index, counts)
        return PhotonRecord(ion, window_s, int(sub.sum()), sub_bin_s, sub, None, p)


@lru_cache(maxsize=64)
def _sub_bin_edges(window_s: float, sub_bin_s: float) -> frozenset[float]:
    """The interior sub-bin edges k x sub_bin of a window."""
    n_bins = int(round(window_s / sub_bin_s))
    if abs(n_bins * sub_bin_s - window_s) > 1e-9 * window_s:
        raise ValueError("the window must be an integer number of sub-bins")
    return frozenset(k * sub_bin_s for k in range(1, n_bins))


def _rate_cells(
    path: ClassPath,
    detected_bright_per_s: float,
    background_per_s: float,
    extra_rate: Sequence[tuple[float, float, float]],
    sub_bin_s: float | None,
    window_s: float,
) -> list[tuple[float, float, float]]:
    """Refine the path's segments by the sub-bin grid and the extra-rate segments into cells of constant rate."""
    edges = {0.0, window_s}
    edges.update(path.jump_times_s)
    for a, b, _r in extra_rate:
        edges.update((max(a, 0.0), min(b, window_s)))
    if sub_bin_s is not None:
        edges.update(_sub_bin_edges(window_s, sub_bin_s))
    grid = np.asarray(sorted(e for e in edges if 0.0 <= e <= window_s), dtype=float)
    a_arr, b_arr = grid[:-1], grid[1:]
    keep = b_arr > a_arr
    a_arr, b_arr = a_arr[keep], b_arr[keep]
    mids = 0.5 * (a_arr + b_arr)
    # the class of each cell from its midpoint, located among the path's (ordered) segment starts
    segments = path.segments()
    starts = np.asarray([seg[0] for seg in segments], dtype=float)
    idx = np.clip(np.searchsorted(starts, mids, side="right") - 1, 0, len(segments) - 1)
    bright = np.array([seg[2] == "bright" for seg in segments], dtype=bool)[idx]
    rates = background_per_s + np.where(bright, detected_bright_per_s, 0.0)
    for ea, eb, er in extra_rate:
        rates = np.where((ea <= mids) & (mids < eb), rates + er, rates)
    return list(zip(a_arr.tolist(), b_arr.tolist(), rates.tolist()))


def apply_detector_nonidealities(
    arrivals_s: np.ndarray,
    dead_time_s: float | None,
    afterpulse_prob: float | None,
    rng: np.random.Generator,
    window_s: float,
) -> np.ndarray:
    """Non-paralyzable dead time (a photon within tau_dead of the last accepted one is lost) and afterpulsing (each accepted
    photon spawns a spurious count right after the dead time with probability p_ap), PLAN.md Section 8.8."""
    arr = np.sort(np.asarray(arrivals_s, dtype=float))
    if dead_time_s is None and afterpulse_prob is None:
        return arr
    dead = dead_time_s or 0.0
    kept: list[float] = []
    last = -math.inf
    pending: list[float] = []
    events = list(arr)
    while events or pending:
        if pending and (not events or pending[0] <= events[0]):
            t = pending.pop(0)
            spurious = True
        else:
            t = events.pop(0)
            spurious = False
        if t >= window_s:
            continue
        if t - last < dead:
            continue
        kept.append(t)
        last = t
        if afterpulse_prob and not spurious and rng.random() < afterpulse_prob:
            pending.append(t + dead + 1e-12)
            pending.sort()
    return np.asarray(kept)


def count_anomaly_band(
    model: RecordModel,
    window_s: float,
    starts: Sequence[ReadoutClass] = ("bright", "dark"),
    *,
    quantile: float = 1e-6,
) -> tuple[int, int]:
    """The [q, 1 - q] band of totals that any of the hypotheses in ``starts`` produces over the window: a total outside it
    is the count anomaly of PLAN.md Section 8.6 (a cosmic ray, an afterpulse burst, a stray-light flash)."""
    if not 0.0 < quantile < 0.5:
        raise ValueError("the quantile lies in (0, 0.5)")
    lo, hi = None, None
    for start in starts:
        dist = model.count_distribution(start, window_s)
        cdf = np.cumsum(dist.pmf)
        below = int(np.searchsorted(cdf, quantile, side="left"))
        above = int(np.searchsorted(cdf, 1.0 - quantile, side="left"))
        lo = below if lo is None else min(lo, below)
        hi = above if hi is None else max(hi, above)
    if lo is None or hi is None:
        raise ValueError("at least one start hypothesis")
    return lo, max(hi, lo)


# ---- registers: neighbour-coupled records (Section 8.5) --------------------------------------------------------------------

Depumping = Mapping[int, tuple[float, float]]
"""Neighbour distance -> (Delta R_d, Delta R_b) that one bright neighbour's leaked light adds to an ion's chain
(:func:`~qutip_trap.readout.fluorescence.neighbour_pumping_rates`)."""


def depumped_model(
    models: Sequence[RecordModel],
    ion: int,
    classes: Sequence[ReadoutClass],
    depumping: Depumping | None,
) -> RecordModel:
    """The ion's chain with the extra pumping every bright neighbour's leaked light drives, the neighbours frozen at
    ``classes``: R_d and R_b are linear in intensity, so several neighbours add, and R_o is untouched."""
    m = models[ion]
    if not depumping:
        return m
    d_d = 0.0
    d_b = 0.0
    for j, c in enumerate(classes):
        if j == ion or c != "bright":
            continue
        extra_d, extra_b = depumping.get(abs(ion - j), (0.0, 0.0))
        d_d += float(extra_d)
        d_b += float(extra_b)
    if d_d <= 0.0 and d_b <= 0.0:
        return m
    rates = dict(m.rates)
    rates[("bright", "dark")] = rates.get(("bright", "dark"), 0.0) + d_d
    rates[("dark", "bright")] = rates.get(("dark", "bright"), 0.0) + d_b
    return RecordModel(
        m.detected_bright_per_s,
        m.background_per_s,
        {k: v for k, v in rates.items() if v > 0.0},
        m.dead_time_s,
        m.afterpulse_prob,
    )


def sample_register_records(
    models: Sequence[RecordModel],
    starts: Sequence[ReadoutClass],
    window_s: float,
    rngs: Sequence[np.random.Generator],
    *,
    leakage: Mapping[int, float] | None = None,
    depumping: Depumping | None = None,
    sub_bin_s: float | None = None,
    arrivals: bool = False,
) -> list[PhotonRecord]:
    """The records of every ion in one shot: each chain sampled first (its pumping by the ``depumping`` of its bright
    neighbours frozen at their start classes), then each ion's counts at its own rate plus ``leakage[|i - j|]`` of every
    bright neighbour's detected rate while that neighbour is bright."""
    n = len(models)
    if len(starts) != n or len(rngs) != n:
        raise ValueError("one start class and one generator per ion")
    leak = dict(leakage or {})
    chains = [depumped_model(models, i, starts, depumping) for i in range(n)]
    paths = [m.sample_path(s, window_s, rng) for m, s, rng in zip(chains, starts, rngs)]
    records: list[PhotonRecord] = []
    for i, (m, rng) in enumerate(zip(chains, rngs)):
        extra: list[tuple[float, float, float]] = []
        for j, pj in enumerate(paths):
            f = leak.get(abs(i - j), 0.0) if j != i else 0.0
            if f <= 0.0:
                continue
            for a, b, c in pj.segments():
                if c == "bright":
                    extra.append((a, b, f * models[j].detected_bright_per_s))
        records.append(
            m.sample_record(
                starts[i],
                window_s,
                rng,
                ion=i,
                sub_bin_s=sub_bin_s,
                arrivals=arrivals,
                path=paths[i],
                extra_rate=extra,
            )
        )
    return records


def neighbourhood_model(
    models: Sequence[RecordModel],
    ion: int,
    classes: Sequence[ReadoutClass],
    leakage: Mapping[int, float],
    depumping: Depumping | None = None,
) -> RecordModel:
    """The ion's record model with its neighbours frozen in ``classes``: every bright neighbour's leaked light is a constant
    added background and (with ``depumping``) extra pumping of the ion's chain."""
    extra = 0.0
    for j, c in enumerate(classes):
        if j != ion and c == "bright":
            extra += leakage.get(abs(ion - j), 0.0) * models[j].detected_bright_per_s
    m = depumped_model(models, ion, classes, depumping)
    return RecordModel(
        m.detected_bright_per_s, m.background_per_s + extra, m.rates, m.dead_time_s, m.afterpulse_prob
    )


# ---- camera images (Section 8.3) -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraGeometry:
    """A pixel grid in the object plane with the ions' point-spread functions (Burrell 2010).

    ``n_rows x n_columns`` pixels of pitch ``pixel_m`` centred on the chain; the PSF is the Airy pattern of an objective of
    numerical aperture ``numerical_aperture`` at ``wavelength_m``, or a Gaussian of width ``psf_sigma_m``. ``weights(ion)``
    integrates the PSF over each pixel on a sub-grid, so leakage into a neighbour's region of interest emerges from the optics.
    """

    ion_positions_m: tuple[float, ...]
    pixel_m: float
    n_columns: int
    n_rows: int
    wavelength_m: float
    numerical_aperture: float | None = None
    psf_sigma_m: float | None = None
    subgrid: int = 5

    def __post_init__(self) -> None:
        if (self.numerical_aperture is None) == (self.psf_sigma_m is None):
            raise ValueError("give exactly one of numerical_aperture (Airy) or psf_sigma_m (Gaussian)")
        if self.pixel_m <= 0.0 or self.n_columns <= 0 or self.n_rows <= 0 or self.subgrid <= 0:
            raise ValueError("pixel pitch, grid and sub-grid are positive")

    @property
    def n_pixels(self) -> int:
        return self.n_rows * self.n_columns

    def pixel_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) centres of the pixels, shape (n_rows, n_columns), x along the chain."""
        centre = 0.5 * (min(self.ion_positions_m) + max(self.ion_positions_m))
        xs = centre + (np.arange(self.n_columns) - 0.5 * (self.n_columns - 1)) * self.pixel_m
        ys = (np.arange(self.n_rows) - 0.5 * (self.n_rows - 1)) * self.pixel_m
        x, y = np.meshgrid(xs, ys)
        return x, y

    def psf(self, r_m: np.ndarray) -> np.ndarray:
        """The normalized point-spread function (integrates to 1 over the plane)."""
        if self.numerical_aperture is not None:
            k_na = 2.0 * math.pi * self.numerical_aperture / self.wavelength_m
            v = k_na * np.asarray(r_m)
            with np.errstate(divide="ignore", invalid="ignore"):
                core = np.where(v > 1e-12, (2.0 * j1(v) / np.where(v > 1e-12, v, 1.0)) ** 2, 1.0)
            return np.asarray(k_na**2 / (4.0 * math.pi) * core)
        s = float(self.psf_sigma_m or 0.0)
        return np.asarray(np.exp(-0.5 * (np.asarray(r_m) / s) ** 2) / (2.0 * math.pi * s**2))

    def weights(self, ion: int) -> np.ndarray:
        """The fraction of the ion's detected photons landing on each pixel, shape (n_rows, n_columns)."""
        x, y = self.pixel_centres()
        offsets = (np.arange(self.subgrid) + 0.5) / self.subgrid - 0.5
        w = np.zeros_like(x)
        x0 = self.ion_positions_m[ion]
        for dx in offsets:
            for dy in offsets:
                r = np.hypot(x + dx * self.pixel_m - x0, y + dy * self.pixel_m)
                w += self.psf(r)
        return np.asarray(w * self.pixel_m**2 / self.subgrid**2)

    def roi(self, ion: int, n_pixels: int) -> np.ndarray:
        """The ion's region of interest: the flat indices of its ``n_pixels`` brightest pixels (Burrell)."""
        order = np.argsort(-self.weights(ion).ravel(), kind="stable")
        return order[: max(1, min(n_pixels, self.n_pixels))]
