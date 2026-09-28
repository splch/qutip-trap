"""The heavy combinations of a product distribution: the branches of a diagonal mixture above a weight cut (PLAN.md
Sections 5.2 to 5.4).

A product distribution is a list of factors, each the probabilities of its options (an ion's pumped levels, a mode's
thermal Fock populations, a frozen mode's Fock states, the eigenvalues of a tracked motional state); a combination takes
one option of every factor and weighs the product of their probabilities. The Fock sum of a run's initial mixture, of an
experiment's thermal mixture and of a GATE_LOCAL step's motional input all keep the combinations of weight >=
``branch_weight_min`` and nothing else, and ``heavy_combinations`` is that enumeration.

It returns what walking the full ``itertools.product`` and testing every leaf returns: the same combinations, in the same
order, each weight the same float, because every product is formed left to right as ``np.prod`` forms it. The walk goes
depth first and prunes a prefix whose heaviest completion (its product times the largest probability of every factor still
to choose) falls below the cut by more than rounding can move a product, then tests every leaf exactly. The cost follows the
kept set, never the full product: at the default cut the two frozen radial modes of ``presets.ca40_optical(1)`` (nbar 8.9
and 9.1) keep 3 903 of their 1.2 x 10^4 products, and the four of ``presets.ca40_optical(2)`` (nbar 8.9 to 12.2) keep 193 483
branches where the full walk visits 2.3 x 10^8 leaves under each of the three choices of internal levels the cut keeps: ten
minutes leaf by leaf, 0.2 s pruned.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import NamedTuple

PRUNE_SLACK = 1e-9
"""The relative margin by which a prefix's heaviest completion must fall below the cut before the walk prunes it. A product
of n probabilities carries at most n roundings of 2^-53 each, so the margin keeps every leaf the exact test keeps for any
product of fewer than 10^6 factors; a cut below the smallest normal float, where the rounding is absolute rather than
relative, prunes nothing before the leaf test."""


class Combination(NamedTuple):
    """One combination of a product distribution: the option taken from every factor and its weight."""

    choice: tuple[int, ...]
    """The index of the option taken from each factor, in factor order."""
    weight: float
    """``scale`` times the product of the options' probabilities, the product formed left to right."""


def heavy_combinations(
    factors: Sequence[Sequence[float]], weight_min: float, *, scale: float = 1.0
) -> list[Combination]:
    """The combinations of ``factors`` (per factor, the probabilities of its options) whose weight reaches ``weight_min``, in
    the order ``itertools.product`` visits them.

    A weight is ``scale`` times the product of the chosen probabilities formed left to right: ``float(np.prod(ps))`` at the
    default ``scale`` of 1, and ``scale * float(np.prod(ps))`` for combinations enumerated under a choice of weight ``scale``
    already made, as ``run.job.enumerate_branches`` enumerates the modes under each kept choice of internal levels. No
    factor at all is one empty combination of weight ``scale``, and a factor without options leaves none. Probabilities and
    ``scale`` are finite and non-negative (``ValueError`` otherwise), the premise of the bound the walk prunes by."""
    probs = [[float(p) for p in factor] for factor in factors]
    if not (math.isfinite(scale) and scale >= 0.0):
        raise ValueError(
            f"the scale of a product distribution is a finite non-negative weight, got {scale!r}"
        )
    for j, factor in enumerate(probs):
        for k, p in enumerate(factor):
            if not (math.isfinite(p) and p >= 0.0):
                raise ValueError(
                    f"option {k} of factor {j} has probability {p!r}: probabilities are finite and non-negative"
                )
    n = len(probs)
    if n == 0:
        # the empty product, cut as every leaf is
        return [] if scale < weight_min else [Combination((), scale)]
    kept: list[Combination] = []
    if any(not factor for factor in probs):
        return kept
    # heaviest[j]: scale times the largest probability of every factor from j on, the heaviest completion of a prefix
    heaviest = [scale] * (n + 1)
    for j in range(n - 1, -1, -1):
        heaviest[j] = max(probs[j]) * heaviest[j + 1]
    floor = weight_min * (1.0 - PRUNE_SLACK) if weight_min >= sys.float_info.min else -math.inf
    # both tests are monotone in the option's probability, so the options that pass are a prefix of this order
    ranked = [sorted(range(len(factor)), key=factor.__getitem__, reverse=True) for factor in probs]
    choice = [0] * n
    last = n - 1

    def walk(j: int, prefix: float) -> None:
        """Every kept combination that extends ``choice[:j]``, whose product is ``prefix``, in ``itertools.product`` order."""
        factor = probs[j]
        passing: list[int] = []
        if j == last:
            for k in ranked[j]:
                if scale * (prefix * factor[k]) < weight_min:
                    break
                passing.append(k)
            passing.sort()
            for k in passing:
                choice[j] = k
                kept.append(Combination(tuple(choice), scale * (prefix * factor[k])))
            return
        rest = heaviest[j + 1]
        for k in ranked[j]:
            if prefix * factor[k] * rest < floor:
                break
            passing.append(k)
        passing.sort()
        for k in passing:
            choice[j] = k
            walk(j + 1, prefix * factor[k])

    walk(0, 1.0)
    return kept
