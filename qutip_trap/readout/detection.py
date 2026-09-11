"""The photon-record layer: the bright/dark/shelf Markov chain, photon counts, background, detector non-idealities, camera
images and readout crosstalk (PLAN.md Sections 8.2, 8.5, 8.8; Appendix E ``Detector``; milestone M5).

Given each ion's start class (sampled from the joint internal outcome through the scheme of Section 8.1), the record over
a window t is a piecewise-constant-rate Poisson process driven by the bright-dark(-shelf) continuous-time Markov chain plus
background. The FAST path samples that chain exactly (alternating exponential holding times until the window closes) and
draws Poisson counts at the piecewise rates, so that windows with k tau not small carry as many transitions as the physics
gives them; the TRAJECTORY path runs ``mcsolve`` on the reduced class space with a photon-counting collapse operator and
records arrival times; both are asserted against the closed forms kept here as oracles: Crain's single-jump count
distribution with its missing no-jump branch restored, Acton's Poisson-exponential mixtures, the zero-threshold errors with
epsilon_sys restored in the final exponent, and Wineland's P_N(0) = (1 - eta_d)^N. The exact count distribution of the
chain with ANY number of jumps is the Markov-modulated Poisson process solved by one matrix exponential
(:meth:`RecordModel.count_distribution`), which is what the threshold optimizer and the POVM fast path integrate.

Conventions (Section 13): epsilon_sys is applied once, at scattered -> detected rate, and R_bg is separate; a bright
neighbour's leaked light is ADDED COUNTS on a dark neighbour PLUS extra off-resonant pumping on its chain (both halves of
Wineland's mechanism, Section 8.5: "added counts on a dark neighbour plus depumping-reduced N"), never a change of the
instrumental efficiency; the camera background is a detector property, spread over the pixels once per exposure rather
than once per ion; detector dead time, afterpulsing and SNSPD recovery are an optional model whose parameters the user
must supply (Section 8.8, a named gap of the sources).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import qutip as qt
from scipy.linalg import expm
from scipy.special import gammainc, gammaln, j1

from qutip_trap.readout.fluorescence import CLASSES, FluorescenceRates, ReadoutClass

M5 = "milestone M5 (readout/detection.py, PLAN.md Section 8.2)"

CLASS_INDEX: dict[ReadoutClass, int] = {c: k for k, c in enumerate(CLASSES)}


@dataclass(frozen=True)
class Detector:
    """The photon detector of the device (Sections 8.2, 8.5, 8.8): its kind, the total system detection efficiency, the
    background rate (counts/s), the point-spread leakage onto neighbours by distance, the optional dead time (s) and
    afterpulse probability, the detection window (s) and, for a camera, the numerical aperture, the object-plane pixel
    pitch (m) and the read noise per pixel per readout (counts)."""

    kind: Literal["pmt", "camera", "snspd"]
    efficiency: float
    """Total system detection efficiency epsilon_sys (Section 8.2)."""
    background_cps: float
    psf_leakage: dict[int, float]
    """Neighbour distance -> fraction of one ion's light landing on the neighbour's detector (Section 8.5)."""
    dead_time_s: float | None
    afterpulse_prob: float | None
    window_s: float
    numerical_aperture: float | None = None
    """Objective NA, for the Airy point-spread function of a camera and the geometric efficiency chain (M5 amendment)."""
    pixel_m: float | None = None
    """Object-plane pixel pitch of a camera (Burrell: 2.6 um per pixel)."""
    read_noise_counts: float = 0.0
    """Camera clock-induced-charge / read noise per pixel per readout, as an equivalent mean count (Section 8.3)."""

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

    @property
    def has_crosstalk(self) -> bool:
        return any(f > 0.0 for f in self.psf_leakage.values())

    def leakage(self, distance: int) -> float:
        return float(self.psf_leakage.get(abs(int(distance)), 0.0))


# ---- Poisson helpers -------------------------------------------------------------------------------------------------------


def poisson_pmf(n: np.ndarray | int, mean: float) -> np.ndarray | float:
    """P_p(n; mu) = e^{-mu} mu^n/n!, evaluated in the log domain (mu = 0 gives delta_{n0})."""
    k = np.asarray(n, dtype=float)
    if mean < 0.0:
        raise ValueError("a Poisson mean is non-negative")
    if mean == 0.0:
        out = np.where(k == 0, 1.0, 0.0)
    else:
        out = np.exp(k * math.log(mean) - mean - gammaln(k + 1.0))
    return float(out) if np.ndim(n) == 0 else np.asarray(out)


def log_poisson_pmf(n: np.ndarray | int, mean: float) -> np.ndarray | float:
    k = np.asarray(n, dtype=float)
    if mean <= 0.0:
        out = np.where(k == 0, 0.0, -np.inf)
    else:
        out = k * math.log(mean) - mean - gammaln(k + 1.0)
    return float(out) if np.ndim(n) == 0 else np.asarray(out)


def zero_photon_probability(n_photons: int, detection_efficiency: float) -> float:
    """Wineland 1998: P_N(0) = (1 - eta_d)^N for N emitted photons, ~ e^{-n_d} with n_d = N eta_d when eta_d << 1
    (e^{-10} = 4.5400e-5, e^{-100} = 3.7201e-44): a binomial over a fixed photon number (Section 8.2)."""
    if n_photons < 0 or not 0.0 <= detection_efficiency <= 1.0:
        raise ValueError("N is non-negative and eta_d lies in [0, 1]")
    return (1.0 - detection_efficiency) ** n_photons


# ---- closed forms of the sources (oracles for the exact chain) ---------------------------------------------------------------


def single_jump_count_distribution(
    n_max: int,
    window_s: float,
    rate_before_per_s: float,
    rate_after_per_s: float,
    jump_rate_per_s: float,
    *,
    include_no_jump: bool = True,
    nodes: int = 96,
) -> np.ndarray:
    """P(n; t) = e^{-R_T t} P_p(n; R_1 t) + int_0^t dtau R_T e^{-R_T tau} P_p(n; R_1 tau + R_2 (t - tau)), n = 0..n_max.

    Crain 2019 Eq. 1 with its missing no-jump branch restored (Section 8.2 [corrected]); the printed form
    (``include_no_jump=False``) sums to 1 - e^{-R_T t}. The sum over k of P_p(k; R_1 tau) P_p(n - k; R_2 (t - tau)) is the
    Poisson convolution P_p(n; R_1 tau + R_2 (t - tau)), so the integrand is smooth and Gauss-Legendre integrates it.
    """
    if n_max < 0 or window_s < 0.0 or min(rate_before_per_s, rate_after_per_s, jump_rate_per_s) < 0.0:
        raise ValueError("counts, window and rates are non-negative")
    n = np.arange(n_max + 1)
    out = np.zeros(n_max + 1)
    if include_no_jump:
        out += math.exp(-jump_rate_per_s * window_s) * np.asarray(
            poisson_pmf(n, rate_before_per_s * window_s)
        )
    if jump_rate_per_s > 0.0 and window_s > 0.0:
        x, w = np.polynomial.legendre.leggauss(nodes)
        tau = 0.5 * window_s * (x + 1.0)
        weight = 0.5 * window_s * w
        for t_k, w_k in zip(tau, weight):
            mean = rate_before_per_s * t_k + rate_after_per_s * (window_s - t_k)
            out += w_k * jump_rate_per_s * math.exp(-jump_rate_per_s * t_k) * np.asarray(poisson_pmf(n, mean))
    return out


def acton_dark_distribution(n_max: int, lambda0: float, alpha1_over_eta: float) -> np.ndarray:
    """Acton 2006 Eq. 5: p_dark(n) = e^{-a lambda_0}[delta_n0 + a/(1 - a)^{n+1} P(n + 1, (1 - a) lambda_0)], a = alpha_1/eta.

    The dark ion leaks to the bright cycle at rate a x (detected rate) and then scatters for the rest of the window; the
    regularized incomplete gamma P is scipy's ``gammainc``. Normalized to 1 exactly; requires a < 1.
    """
    a = alpha1_over_eta
    if not 0.0 <= a < 1.0 or lambda0 < 0.0:
        raise ValueError("alpha_1/eta lies in [0, 1) and lambda_0 is non-negative")
    n = np.arange(n_max + 1)
    out = np.zeros(n_max + 1)
    out[0] = 1.0
    if a > 0.0:
        out += a / (1.0 - a) ** (n + 1) * gammainc(n + 1.0, (1.0 - a) * lambda0)
    return np.asarray(math.exp(-a * lambda0) * out)


def acton_bright_distribution(n_max: int, lambda0: float, alpha2_over_eta: float) -> np.ndarray:
    """Acton 2006 Eq. 6: p_bright(n) = e^{-(1 + a) lambda_0} lambda_0^n/n! + a/(1 + a)^{n+1} P(n + 1, (1 + a) lambda_0),
    a = alpha_2/eta: the never-leaving Poisson term plus the smeared pumped-dark term (re-pumping neglected)."""
    a = alpha2_over_eta
    if a < 0.0 or lambda0 < 0.0:
        raise ValueError("alpha_2/eta and lambda_0 are non-negative")
    n = np.arange(n_max + 1)
    first = np.exp(
        -(1.0 + a) * lambda0 + n * (math.log(lambda0) if lambda0 > 0.0 else -np.inf) - gammaln(n + 1.0)
    )
    if lambda0 == 0.0:
        first = np.where(n == 0, math.exp(-(1.0 + a) * lambda0), 0.0)
    second = a / (1.0 + a) ** (n + 1) * gammainc(n + 1.0, (1.0 + a) * lambda0) if a > 0.0 else 0.0
    return np.asarray(first + second)


def zero_threshold_errors(
    window_s: float,
    detected_bright_per_s: float,
    dark_pumping_per_s: float,
    bright_pumping_per_s: float,
    background_per_s: float,
) -> tuple[float, float]:
    """(epsilon_B, epsilon_D) of the zero-threshold discriminator (any photon -> bright), Crain 2019 Eqs. 2-3 [corrected].

    epsilon_B = P(n = 0 | bright) = e^{-R_bg t}[e^{-(eps R_o + R_d) t} + R_d/(eps R_o + R_d) (1 - e^{-(eps R_o + R_d) t})];
    epsilon_D = 1 - P(n = 0 | dark), P(n = 0 | dark) = e^{-R_bg t}[e^{-R_b t} + R_b/(eps R_o - R_b)(e^{-R_b t} - e^{-eps R_o t})].
    The printed Eq. 3 carries e^{-(R_o + R_bg) t} with the SCATTERED rate in the no-jump term (see
    :func:`crain_printed_bright_error`), which changes the bright error at 11 us by a factor of about 9.
    """
    t = window_s
    r0, rd, rb, rbg = detected_bright_per_s, dark_pumping_per_s, bright_pumping_per_s, background_per_s
    if min(t, r0, rd, rb, rbg) < 0.0:
        raise ValueError("window and rates are non-negative")
    k = r0 + rd
    no_jump_bright = math.exp(-k * t)
    eps_b = math.exp(-rbg * t) * (no_jump_bright + (rd / k if k > 0.0 else 0.0) * (1.0 - no_jump_bright))
    if abs(r0 - rb) < 1e-12 * max(r0, 1.0):
        pumped = rb * t * math.exp(-rb * t)
    else:
        pumped = rb / (r0 - rb) * (math.exp(-rb * t) - math.exp(-r0 * t))
    p0_dark = math.exp(-rbg * t) * (math.exp(-rb * t) + pumped)
    return eps_b, 1.0 - p0_dark


def crain_printed_bright_error(
    window_s: float,
    detected_bright_per_s: float,
    scattered_bright_per_s: float,
    dark_pumping_per_s: float,
    background_per_s: float,
) -> float:
    """Crain Eq. 3 as printed: the no-jump term e^{-R_d t} e^{-(R_o + R_bg) t} with the scattered R_o, which is ~ 0 at
    any useful window and leaves only the pumped term R_d/(eps R_o + R_d) (the negative control of Section 9.5)."""
    t = window_s
    k = detected_bright_per_s + dark_pumping_per_s
    pumped = math.exp(-background_per_s * t) * dark_pumping_per_s / k * (1.0 - math.exp(-k * t))
    return pumped + math.exp(-dark_pumping_per_s * t) * math.exp(
        -(scattered_bright_per_s + background_per_s) * t
    )


def first_photon_cutoff_s(
    dark_pumping_per_s: float, dark_count_per_s: float, detected_bright_per_s: float
) -> float:
    """Noek 2013 Eq. 6: tau_c = ln(R_d/R_dc)/(eps R_o), before which a single detected photon more likely came from a
    bright ion that then pumped dark than from a background count on a dark ion."""
    if dark_pumping_per_s <= 0.0 or dark_count_per_s <= 0.0 or detected_bright_per_s <= 0.0:
        raise ValueError("rates are positive")
    return math.log(dark_pumping_per_s / dark_count_per_s) / detected_bright_per_s


# ---- the record model: the Markov chain and its exact count statistics ---------------------------------------------------------


@dataclass(frozen=True)
class ClassPath:
    """One realization of the readout chain over a window: the start class and the jumps (Section 8.2)."""

    window_s: float
    start: ReadoutClass
    jump_times_s: tuple[float, ...]
    jump_targets: tuple[ReadoutClass, ...]

    def segments(self) -> list[tuple[float, float, ReadoutClass]]:
        """[(t_start, t_end, class)] covering [0, window)."""
        out: list[tuple[float, float, ReadoutClass]] = []
        t0, cls = 0.0, self.start
        for t, target in zip(self.jump_times_s, self.jump_targets):
            out.append((t0, t, cls))
            t0, cls = t, target
        out.append((t0, self.window_s, cls))
        return out

    def occupancy_s(self, cls: ReadoutClass) -> float:
        return sum(b - a for a, b, c in self.segments() if c == cls)

    def class_at(self, t: float) -> ReadoutClass:
        for a, b, c in self.segments():
            if a <= t < b:
                return c
        return self.segments()[-1][2]

    @property
    def n_jumps(self) -> int:
        return len(self.jump_times_s)


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
    """Probability mass at n > n_max (must be negligible for the threshold optimizer)."""
    end_class: np.ndarray
    """P(class at the window's end) marginal over counts."""

    @property
    def n_max(self) -> int:
        return int(len(self.pmf) - 1)

    def probability_above(self, threshold: float) -> float:
        """P(n > threshold) including the overflow mass (threshold half-integer or integer)."""
        n = np.arange(len(self.pmf))
        return float(np.sum(self.pmf[n > threshold])) + self.overflow

    def mean(self) -> float:
        return float(np.dot(np.arange(len(self.pmf)), self.pmf))


@dataclass(frozen=True)
class RecordModel:
    """The chain and the detected rates of ONE ion: epsilon_sys R_o, R_bg and the (from, to) transition rates (Section 8.2)."""

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
        """epsilon_sys applied once here (Section 13); the detector's dead time and afterpulsing carried along."""
        detected, background = rates.detected(detector)
        return cls(
            detected_bright_per_s=detected,
            background_per_s=background,
            rates=rates.transition_rates(),
            dead_time_s=detector.dead_time_s,
            afterpulse_prob=detector.afterpulse_prob,
        )

    # ---- the chain -------------------------------------------------------------------------------------------------

    def generator(self) -> np.ndarray:
        """The 3 x 3 generator Q (rows sum to zero) in the CLASSES order."""
        q = np.zeros((3, 3))
        for (a, b), r in self.rates.items():
            q[CLASS_INDEX[a], CLASS_INDEX[b]] += r
        q -= np.diag(q.sum(axis=1))
        return q

    def class_count_rates(self, *, background: bool = True) -> np.ndarray:
        """Detected rate while in each class: eps R_o + R_bg in bright, R_bg elsewhere."""
        lam = np.full(3, self.background_per_s if background else 0.0)
        lam[CLASS_INDEX["bright"]] += self.detected_bright_per_s
        return lam

    def class_probabilities(self, start: ReadoutClass | Sequence[float], t_s: float) -> np.ndarray:
        """P(class at t | start) = e_start exp(Q t)."""
        p0 = self._start_vector(start)
        return np.asarray(p0 @ expm(self.generator() * t_s))

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
        """E[time spent bright] over the window: int_0^t P(bright at tau) d tau, by the generator's exponential."""
        # int_0^t e^{Q tau} d tau via the augmented exponential [[Q, I], [0, 0]]
        q = self.generator()
        aug = np.zeros((6, 6))
        aug[:3, :3] = q
        aug[:3, 3:] = np.eye(3)
        block = expm(aug * window_s)[:3, 3:]
        return float(self._start_vector(start) @ block[:, CLASS_INDEX["bright"]])

    def mean_counts(self, start: ReadoutClass | Sequence[float], window_s: float) -> float:
        """E[n] = eps R_o E[bright time] + R_bg t (exact; for a bright start it is Section 8.2's n̄(τ) when only R_d, R_b act)."""
        return (
            self.detected_bright_per_s * self.bright_occupancy_mean_s(start, window_s)
            + self.background_per_s * window_s
        )

    def suggested_n_max(self, window_s: float, *, sigmas: float = 12.0) -> int:
        mu = (self.detected_bright_per_s + self.background_per_s) * window_s
        return int(math.ceil(mu + sigmas * math.sqrt(mu + 1.0))) + 5

    def count_distribution(
        self,
        start: ReadoutClass | Sequence[float],
        window_s: float,
        *,
        n_max: int | None = None,
        background: bool = True,
    ) -> CountDistribution:
        """Exact P(n | start) of the Markov-modulated Poisson process over the window, by one matrix exponential.

        The augmented generator on (count n, class): blocks Q - diag(lambda) on the diagonal and diag(lambda) one block
        up (a count increments n); the last block absorbs n > n_max without counting further, so its mass is the reported
        overflow. Exact for any number of jumps (the single-jump closed forms are its small-k-tau oracles, Section 8.2).
        """
        if window_s < 0.0:
            raise ValueError("the window is non-negative")
        nmax = self.suggested_n_max(window_s) if n_max is None else int(n_max)
        if nmax < 0:
            raise ValueError("n_max is non-negative")
        q = self.generator()
        lam = self.class_count_rates(background=background)
        d = 3
        size = d * (nmax + 2)
        a = np.zeros((size, size))
        for n in range(nmax + 2):
            blk = slice(n * d, (n + 1) * d)
            if n <= nmax:
                a[blk, blk] = q - np.diag(lam)
                nxt = slice((n + 1) * d, (n + 2) * d)
                a[blk, nxt] += np.diag(lam)
            else:
                a[blk, blk] = q
        p0 = np.zeros(size)
        p0[:d] = self._start_vector(start)
        p = np.asarray(p0 @ expm(a * window_s)).reshape(nmax + 2, d)
        pmf = p[: nmax + 1].sum(axis=1)
        overflow = float(p[nmax + 1].sum())
        end_class = p.sum(axis=0)
        return CountDistribution(
            pmf=np.clip(pmf, 0.0, None), overflow=max(overflow, 0.0), end_class=end_class
        )

    def sub_bin_matrices(self, sub_bin_s: float, n_max: int) -> np.ndarray:
        """M[n, i, j] = P(n counts in a sub-bin and class j at its end | class i at its start): the exact HMM emission and
        transition kernel of the time-resolved record (the continuous-time likelihood no source gives, Section 8.7)."""
        q = self.generator()
        lam = self.class_count_rates()
        d = 3
        size = d * (n_max + 2)
        a = np.zeros((size, size))
        for n in range(n_max + 2):
            blk = slice(n * d, (n + 1) * d)
            if n <= n_max:
                a[blk, blk] = q - np.diag(lam)
                a[blk, slice((n + 1) * d, (n + 2) * d)] += np.diag(lam)
            else:
                a[blk, blk] = q
        e = np.asarray(expm(a * sub_bin_s))
        out = np.zeros((n_max + 2, d, d))
        for n in range(n_max + 2):
            out[n] = e[:d, n * d : (n + 1) * d]
        return out

    # ---- sampling ---------------------------------------------------------------------------------------------------

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
        """One record: Poisson counts at the piecewise rate of the (given or sampled) path, plus background and any
        ``extra_rate`` segments [(t_start, t_end, rate)] from neighbours (Section 8.5); dead time and afterpulsing act on
        arrival times, which are then always generated."""
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
        # one vectorized Poisson draw over the cells: NumPy's Generator draws the elements of an array in order, exactly as
        # the scalar draws per cell did, so the counts and the generator's state afterwards are unchanged (checked against
        # the scalar loop; performance pass 2026-09-09)
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
    """The interior sub-bin edges k x sub_bin of a window (the same floats every record of a calibration adds)."""
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
    grid = sorted(e for e in edges if 0.0 <= e <= window_s)
    # the class of each cell from ONE pass over the path's segments (contiguous, ordered; the midpoints increase): the same
    # class ``path.class_at(mid)`` returns and the same additions in the same order per cell, so the cells, and the Poisson
    # draws taken from them, are unchanged (performance pass 2026-09-09; the per-cell lookup rebuilt the segment list 200
    # times per record and was 60% of a detection calibration). A hand-built path whose jumps are not ordered takes the
    # general lookup.
    segments = path.segments()
    starts = [seg[0] for seg in segments]
    g = np.asarray(grid, dtype=float)
    a_arr, b_arr = g[:-1], g[1:]
    keep = b_arr > a_arr
    a_arr, b_arr = a_arr[keep], b_arr[keep]
    mids = 0.5 * (a_arr + b_arr)
    if all(x <= y for x, y in zip(starts, starts[1:])) and all(seg[0] <= seg[1] for seg in segments):
        idx = np.searchsorted(np.asarray(starts, dtype=float), mids, side="right") - 1
        idx = np.clip(idx, 0, len(segments) - 1)
        bright = np.array([seg[2] == "bright" for seg in segments], dtype=bool)[idx]
        # a midpoint past the last segment's end takes the last class, as class_at does
        last_end = segments[-1][1]
        bright = np.where(mids >= last_end, segments[-1][2] == "bright", bright)
    else:
        bright = np.array([path.class_at(float(m)) == "bright" for m in mids], dtype=bool)
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
    """Non-paralyzable dead time (a photon within tau_dead of the last ACCEPTED one is lost) and afterpulsing (each accepted
    photon spawns a spurious count right after the dead time with probability p_ap), Section 8.8's optional model."""
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


# ---- registers: neighbour-coupled records (Section 8.5) ---------------------------------------------------------------------


def count_anomaly_band(
    model: RecordModel,
    window_s: float,
    starts: Sequence[ReadoutClass] = ("bright", "dark"),
    *,
    quantile: float = 1e-6,
) -> tuple[int, int]:
    """The [q, 1 - q] band of totals that ANY of the hypotheses in ``starts`` can produce over the window.

    A record whose total lies outside it is the "count anomaly" of Section 8.6's herald list: neither the bright nor the
    dark hypothesis explains it, which in a laboratory is a cosmic ray, an afterpulse burst or a stray-light flash
    (Section 8.8 names detector non-idealities as a gap of the sources, and Myerson attributes about 20 % of his dark
    error to cosmic rays). Computed once per ion from the exact count distributions, so the per-shot test is a comparison.
    """
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


Depumping = Mapping[int, tuple[float, float]]
"""Neighbour distance -> (Delta R_d, Delta R_b) that ONE bright neighbour's leaked light adds to an ion's chain: the
depumping half of Wineland's crosstalk mechanism (Section 8.5), built by
:func:`~qutip_trap.readout.fluorescence.neighbour_pumping_rates`."""


def depumped_model(
    models: Sequence[RecordModel],
    ion: int,
    classes: Sequence[ReadoutClass],
    depumping: Depumping | None,
) -> RecordModel:
    """Ion ``ion``'s chain with the extra off-resonant pumping every BRIGHT neighbour's leaked light drives (Section 8.5's
    "depumping-reduced N"), the neighbours frozen at ``classes`` (the same first-order form as the added counts).

    R_d and R_b are linear in intensity with no saturation denominator (Section 8.1), so the contributions of several
    bright neighbours add; R_o is untouched, because the leaked light is far too weak to change the cycling rate (the
    bound is I_ion/I_sat ~ 1e-4) and Wineland's mechanism is a degradation of DISCRIMINATION, never of eta_d.
    """
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
    """Records of every ion in one shot: each chain sampled from its own generator first, then each ion's counts at its own
    rate plus the leaked light of every bright neighbour (fraction ``leakage[|i - j|]`` of the neighbour's detected rate),
    so the records are neighbour-coupled and the joint confusion is not a product (Section 5.7).

    ``depumping`` adds the other half of Wineland's mechanism (Section 8.5): the leaked light also pumps the neighbour
    between the bright and dark manifolds, so a bright ion beside a bright one loses photons as well (the
    "depumping-reduced N"). The pumping is frozen at the neighbours' START classes, the first-order form the factored
    confusion tensor uses, because the chain of one ion is sampled before its neighbours' paths are known.
    """
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
    """Ion ``ion``'s record model with its neighbours FROZEN in ``classes``: the leaked light of every bright neighbour is a
    constant added background and (with ``depumping``) extra pumping on the ion's own chain, the first-order form of the
    neighbour coupling used by the factored confusion tensor (Section 8.5, both halves of Wineland's mechanism)."""
    extra = 0.0
    for j, c in enumerate(classes):
        if j != ion and c == "bright":
            extra += leakage.get(abs(ion - j), 0.0) * models[j].detected_bright_per_s
    m = depumped_model(models, ion, classes, depumping)
    return RecordModel(
        m.detected_bright_per_s, m.background_per_s + extra, m.rates, m.dead_time_s, m.afterpulse_prob
    )


# ---- the trajectory path (mcsolve on the reduced class space) -----------------------------------------------------------------


def mcsolve_records(
    model: RecordModel,
    start: ReadoutClass,
    window_s: float,
    n_records: int,
    *,
    seed: int = 0,
    sub_bin_s: float | None = None,
) -> list[PhotonRecord]:
    """The trajectory path of Section 8.2: ``mcsolve`` on |bright>, |dark>, |shelf> with H = 0, a photon-counting collapse
    operator sqrt(eps R_o) |B><B|, a background operator sqrt(R_bg) 1 and the chain's transitions as jump operators;
    every jump time and type is read back from the trajectory, so the arrival record is the quantum-jump record.

    The solve runs in units of the window (rates x window, time in [0, 1]): mcsolve's collapse-time search has an ABSOLUTE
    time tolerance (``norm_t_tol``, 1e-6 by default) that a 20 us window in SI seconds cannot satisfy (QuTiP 5.3.1 raises
    "Could not find the collapse time"), while the scaled problem is well conditioned.
    """
    if n_records <= 0:
        raise ValueError("at least one trajectory")
    d = 3
    kets = [qt.basis(d, k) for k in range(d)]
    scale = window_s
    ops: list[qt.Qobj] = [math.sqrt(model.detected_bright_per_s * scale) * kets[CLASS_INDEX["bright"]].proj()]
    photon_indices = [0]
    if model.background_per_s > 0.0:
        ops.append(math.sqrt(model.background_per_s * scale) * qt.qeye(d))
        photon_indices.append(1)
    for (a, b), r in model.rates.items():
        ops.append(math.sqrt(r * scale) * kets[CLASS_INDEX[b]] * kets[CLASS_INDEX[a]].dag())
    result = qt.mcsolve(
        qt.qzero(d),
        kets[CLASS_INDEX[start]],
        [0.0, 1.0],
        ops,
        ntraj=n_records,
        options={
            "keep_runs_results": True,
            "map": "serial",
            "progress_bar": "",
            "store_states": False,
            "norm_t_tol": 1e-12,
            "norm_steps": 100,
        },
        seeds=seed,
    )
    records: list[PhotonRecord] = []
    n_bins = None if sub_bin_s is None else int(round(window_s / sub_bin_s))
    for times, which in zip(result.col_times, result.col_which):
        t = np.asarray(times, dtype=float) * scale
        w = np.asarray(which, dtype=int)
        arr = np.sort(t[np.isin(w, photon_indices)])
        sub = None if n_bins is None else np.histogram(arr, bins=n_bins, range=(0.0, window_s))[0]
        records.append(PhotonRecord(0, window_s, int(arr.size), sub_bin_s, sub, arr, None))
    return records


# ---- spectator dephasing during a neighbour's readout (Section 8.5) --------------------------------------------------------------


def spectator_coherence(tau_s: float, alpha_s: float) -> float:
    """exp(-tau^2/alpha^2): a data qubit's Ramsey contrast while a neighbour is detected decays as a GAUSSIAN, not an
    exponential (Crain 2019: alpha = 94(5) ms at 200 um and 814(77) ms at 370 um from the detected ion, 1716 ms baseline)."""
    if alpha_s <= 0.0 or tau_s < 0.0:
        raise ValueError("alpha is positive and tau non-negative")
    return math.exp(-((tau_s / alpha_s) ** 2))


def spectator_offset_sigma_rad_s(alpha_s: float) -> float:
    """The quasi-static frequency offset that reproduces the Gaussian decay: a per-shot draw delta ~ N(0, sigma) with
    sigma = sqrt(2)/alpha gives <cos(delta tau)> = exp(-sigma^2 tau^2/2) = exp(-tau^2/alpha^2); a constant-rate sigma_z
    Lindblad term would give an exponential lineshape instead (Section 8.5)."""
    if alpha_s <= 0.0:
        raise ValueError("alpha is positive")
    return math.sqrt(2.0) / alpha_s


def sample_spectator_offset_rad_s(alpha_s: float, rng: np.random.Generator) -> float:
    """One shot's quasi-static offset of a spectator qubit during its neighbour's detection (Section 5.7 stochastic elements)."""
    return float(rng.normal(0.0, spectator_offset_sigma_rad_s(alpha_s)))


CRAIN_SPECTATOR_ANCHORS_S: tuple[tuple[float, float], ...] = ((200e-6, 94e-3), (370e-6, 814e-3))
"""(distance, alpha) measured by Crain 2019 for the spectator's Gaussian coherence time, apparatus data."""


def crain_spectator_alpha_s(distance_m: float) -> float:
    """alpha(d) interpolated as a power law through Crain's two anchors (exponent ln(814/94)/ln(370/200) = 3.5): the wing of
    the detection beam at the data qubit falls steeply with distance; an interpolation of apparatus data, not physics."""
    (d1, a1), (d2, a2) = CRAIN_SPECTATOR_ANCHORS_S
    if distance_m <= 0.0:
        raise ValueError("distance is positive")
    p = math.log(a2 / a1) / math.log(d2 / d1)
    return float(a1 * (distance_m / d1) ** p)


# ---- camera images (Sections 8.3, 8.5) -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraGeometry:
    """A pixel grid in the object plane with the ions' point-spread functions (Burrell 2010; Section 8.3).

    Pixels are ``n_rows x n_columns`` of pitch ``pixel_m`` centred on the chain; the PSF is the normalized Airy pattern of
    an objective of numerical aperture ``numerical_aperture`` at ``wavelength_m``, or a Gaussian of width ``psf_sigma_m``
    (an aberrated system, Burrell's, sits well above the diffraction limit). ``weights(ion)`` integrates the PSF over each
    pixel with a sub-grid, so PSF leakage into a neighbour's region of interest emerges from the optics without a free
    parameter (Section 8.3 [verified]).
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
    def n_ions(self) -> int:
        return len(self.ion_positions_m)

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
        """Normalized point-spread function (integrates to 1 over the plane)."""
        if self.numerical_aperture is not None:
            k_na = 2.0 * math.pi * self.numerical_aperture / self.wavelength_m
            v = k_na * np.asarray(r_m)
            with np.errstate(divide="ignore", invalid="ignore"):
                core = np.where(v > 1e-12, (2.0 * j1(v) / np.where(v > 1e-12, v, 1.0)) ** 2, 1.0)
            return np.asarray(k_na**2 / (4.0 * math.pi) * core)
        s = float(self.psf_sigma_m or 0.0)
        return np.asarray(np.exp(-0.5 * (np.asarray(r_m) / s) ** 2) / (2.0 * math.pi * s**2))

    def weights(self, ion: int) -> np.ndarray:
        """Fraction of ion ``ion``'s detected photons landing on each pixel, shape (n_rows, n_columns)."""
        x, y = self.pixel_centres()
        offsets = (np.arange(self.subgrid) + 0.5) / self.subgrid - 0.5
        w = np.zeros_like(x)
        x0 = self.ion_positions_m[ion]
        for dx in offsets:
            for dy in offsets:
                r = np.hypot(x + dx * self.pixel_m - x0, y + dy * self.pixel_m)
                w += self.psf(r)
        return np.asarray(w * self.pixel_m**2 / self.subgrid**2)

    def brightness_order(self, ion: int) -> np.ndarray:
        """Flat pixel indices of ``ion`` in order of decreasing weight (Burrell's region-of-interest construction)."""
        return np.argsort(-self.weights(ion).ravel(), kind="stable")

    def roi(self, ion: int, n_pixels: int) -> np.ndarray:
        return self.brightness_order(ion)[: max(1, min(n_pixels, self.n_pixels))]

    def leakage_fraction(self, ion: int, other: int, n_pixels: int) -> float:
        """Signal of ``other`` inside ion ``ion``'s ROI RELATIVE TO THE ION'S OWN signal there (Burrell: 4.0 % nearest,
        0.9 % next-nearest at 14 um for an ROI of one spacing's diameter).

        This is the ratio the source quotes and is NOT :attr:`Detector.psf_leakage`, which is the ABSOLUTE fraction of the
        neighbour's total detected light that lands in the region of interest; the two differ by the ion's own ROI
        collection efficiency (0.757 in a Burrell-like geometry, so feeding 4.0 % straight in overstates the leak by 32 %).
        :meth:`absolute_leakage_fraction` and :func:`psf_leakage_from_geometry` do the conversion once.
        """
        roi = self.roi(ion, n_pixels)
        own = float(self.weights(ion).ravel()[roi].sum())
        return float(self.weights(other).ravel()[roi].sum()) / own if own > 0.0 else 0.0

    def own_roi_efficiency(self, ion: int, n_pixels: int) -> float:
        """Fraction of ion ``ion``'s own detected light that lands inside its own region of interest."""
        return float(self.weights(ion).ravel()[self.roi(ion, n_pixels)].sum())

    def absolute_leakage_fraction(self, ion: int, other: int, n_pixels: int) -> float:
        """Fraction of ``other``'s TOTAL detected light that lands in ion ``ion``'s region of interest: the convention
        :attr:`Detector.psf_leakage` and the added counts of Section 8.5 use."""
        roi = self.roi(ion, n_pixels)
        return float(self.weights(other).ravel()[roi].sum())


def psf_leakage_from_geometry(
    geometry: CameraGeometry, n_pixels: int, *, max_distance: int | None = None
) -> dict[int, float]:
    """:attr:`Detector.psf_leakage` derived from the optics: distance -> the absolute fraction of a neighbour's detected
    light that falls in an ion's region of interest, averaged over the pairs at that distance.

    Section 8.3: "crosstalk then emerges from PSF overlap without a free parameter". The values are ABSOLUTE fractions of
    the neighbour's total light (:meth:`CameraGeometry.absolute_leakage_fraction`), not the ROI-relative ratios the source
    quotes, so the conversion happens once, here.
    """
    if geometry.n_ions < 2:
        return {}
    limit = geometry.n_ions - 1 if max_distance is None else int(max_distance)
    out: dict[int, float] = {}
    for d in range(1, limit + 1):
        pairs = [geometry.absolute_leakage_fraction(i, i + d, n_pixels) for i in range(geometry.n_ions - d)]
        pairs += [geometry.absolute_leakage_fraction(i + d, i, n_pixels) for i in range(geometry.n_ions - d)]
        if not pairs:
            continue
        value = float(np.mean(pairs))
        if value > 0.0:
            out[d] = value
    return out


def sample_camera_image(
    geometry: CameraGeometry,
    models: Sequence[RecordModel],
    paths: Sequence[ClassPath],
    exposure_s: float,
    rng: np.random.Generator,
    *,
    read_noise_counts: float = 0.0,
    t_start_s: float = 0.0,
) -> np.ndarray:
    """One exposure: per pixel Poisson(sum_i w_i[p] eps R_o,i T_bright,i + background share + read noise), with T_bright,i
    the bright occupancy of ion i's path within the exposure (Section 8.3)."""
    if len(models) != geometry.n_ions or len(paths) != geometry.n_ions:
        raise ValueError("one record model and one class path per ion")
    # the background is a DETECTOR property, so it is spread over the pixels once per exposure and not once per ion (a
    # register of N ions used to collect N x R_bg; audit 2026-09-07 B12). Models may disagree on it only through their
    # detectors, so the maximum is the detector's rate.
    background = max((m.background_per_s for m in models), default=0.0)
    mean = np.full(
        (geometry.n_rows, geometry.n_columns),
        read_noise_counts + background * exposure_s / geometry.n_pixels,
        dtype=float,
    )
    for i, (m, p) in enumerate(zip(models, paths)):
        bright = sum(
            max(0.0, min(b, t_start_s + exposure_s) - max(a, t_start_s))
            for a, b, c in p.segments()
            if c == "bright"
        )
        mean += geometry.weights(i) * m.detected_bright_per_s * bright
    return np.asarray(rng.poisson(mean))


__all__ = [
    "Depumping",
    "CLASS_INDEX",
    "CameraGeometry",
    "ClassPath",
    "CountDistribution",
    "Detector",
    "PhotonRecord",
    "RecordModel",
    "acton_bright_distribution",
    "acton_dark_distribution",
    "apply_detector_nonidealities",
    "count_anomaly_band",
    "crain_printed_bright_error",
    "depumped_model",
    "first_photon_cutoff_s",
    "log_poisson_pmf",
    "mcsolve_records",
    "neighbourhood_model",
    "poisson_pmf",
    "psf_leakage_from_geometry",
    "sample_camera_image",
    "sample_spectator_offset_rad_s",
    "spectator_coherence",
    "spectator_offset_sigma_rad_s",
    "crain_spectator_alpha_s",
    "CRAIN_SPECTATOR_ANCHORS_S",
    "sample_register_records",
    "single_jump_count_distribution",
    "zero_photon_probability",
    "zero_threshold_errors",
]
