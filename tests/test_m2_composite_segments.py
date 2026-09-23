"""``composite_segments`` against ``CompositePulse``'s own derived quantities (Appendix E; PLAN.md:2800).

Two disagreements the M2 audit found between the two Appendix E entry points:

* the target azimuth phi_t was added a SECOND time, so every segment came out at phi_t too high (audit E4). It is
  already on every entry, the zeroth included: Section 13's row "Composite-pulse sequence order in time".
* ``n_rep`` was applied in ``composite_segments`` and ignored by ``total_rotation_rad``/``duration_s``, a factor
  n_rep on the pulse duration that Section 4.3.5 makes the scheduler's second reported quantity (audit E12/B5).
  Appendix E's ``n_rep`` is "repetitions of the cached T_{2j-2} block", 2^{2j-2} at order 2j and 1 elsewhere, and
  ``segments`` already carries the expansion, so it is a record and not a multiplier.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.control.composite import (
    composite_pulse,
    operator_distance_up_to_phase,
    rotation,
    suzuki_block_power,
)
from qutip_trap.noise.decoupling import composite_segments

PI = math.pi
FAMILIES = ("primitive", "SK1", "BB1", "NB1", "PB1", "CORPSE", "short_CORPSE", "SCROFULOUS", "CinSK", "CinBB")


@pytest.mark.parametrize("family", FAMILIES)
def test_composite_segments_does_not_add_the_target_azimuth_twice(family: str) -> None:
    """phi_rad = 0.7: the control segments' phases are the pulse's own, entry for entry."""
    pulse = composite_pulse(family, PI, 0.7)
    segs = composite_segments(pulse, 1.0)
    assert [s.phi_rad for s in segs] == [phase for _, phase in pulse.segments]
    assert [s.theta_rad for s in segs] == [area for area, _ in pulse.segments]
    # the zeroth entry carries phi_t too, which is what makes the double-add invisible at phi_t = 0
    at_zero = composite_pulse(family, PI, 0.0)
    assert [s.phi_rad for s in segs] == pytest.approx([p + 0.7 for _, p in at_zero.segments], abs=1e-15)


@pytest.mark.parametrize(
    ("family", "order"),
    [("P2j", 1), ("P2j", 2), ("P2j", 3), ("N2j", 1), ("N2j", 2), ("B2j", 1), ("B2j", 2), ("SK1", 1)],
)
def test_duration_agrees_with_the_control_segments(family: str, order: int) -> None:
    """sum of the segment areas over Omega, one number however it is reached (Section 4.3.5)."""
    pulse = composite_pulse(family, PI / 2.0, 0.7, order=order)
    segs = composite_segments(pulse, 2.0)
    assert sum(s.theta_rad for s in segs) == pytest.approx(pulse.total_rotation_rad(), rel=1e-15)
    assert sum(s.duration_s for s in segs) == pytest.approx(pulse.duration_s(2.0), rel=1e-15)


def test_n_rep_is_the_cached_block_power_and_is_already_expanded() -> None:
    """n_rep = 2^{2j-2} for P2j/N2j/B2j (1 at j = 1) and 1 elsewhere; passing it asserts, never repeats."""
    assert [suzuki_block_power(j) for j in (1, 2, 3, 4)] == [1, 4, 16, 64]
    for j in (1, 2, 3):
        p = composite_pulse("P2j", PI / 2.0, order=j)
        assert p.n_rep == suzuki_block_power(j)
        # the same object comes back when the caller asserts the derived value
        q = composite_pulse("P2j", PI / 2.0, order=j, n_rep=suzuki_block_power(j))
        assert q.segments == p.segments and q.n_rep == p.n_rep
        assert q.total_rotation_rad() == pytest.approx(
            sum(s.theta_rad for s in composite_segments(q, 1.0)), rel=1e-15
        )
    with pytest.raises(ValueError, match="block power"):
        composite_pulse("P2j", PI / 2.0, order=1, n_rep=3)
    with pytest.raises(ValueError, match="only value for SK1"):
        composite_pulse("SK1", PI / 2.0, n_rep=3)


def test_neither_alternative_reading_of_n_rep_is_order_preserving() -> None:
    """Why n_rep is a record: the two "active" readings are refuted here, not asserted away.

    Repeating the whole pulse multiplies the target angle (so ``target()`` would be wrong); repeating the corrector
    alone leaves the target but destroys the compensation, because the corrector cancels the TARGET pulse's error and
    a second copy has none to cancel.
    """
    p = composite_pulse("P2j", PI / 2.0, order=1)
    assert p.order_slope("amplitude", (1e-3, 1e-2))[0] == pytest.approx(6.0, abs=0.12)
    corrector = list(p.segments[1:])
    # the corrector is exactly the identity at eps = 0, so the doubled sequence is still a pi/2 pulse ...
    from qutip_trap.control.composite import fold

    doubled = tuple([p.segments[0]] + corrector * 2)
    assert operator_distance_up_to_phase(fold(doubled), rotation(PI / 2.0, 0.0)) < 1e-12
    # ... and its amplitude slope has fallen from 6 to 2
    eps = np.geomspace(1e-3, 1e-2, 7)
    vals = [
        1.0 - abs(np.trace(rotation(PI / 2.0, 0.0).conj().T @ fold(doubled, eps_a=e))) ** 2 / 4.0 for e in eps
    ]
    slope = float(np.polyfit(np.log(eps), np.log(vals), 1)[0])
    assert slope == pytest.approx(2.0, abs=0.12)
