"""The dc frozen-noise floor of Section 6.9 and its max rule (PLAN.md Section 6.9 "The dc floor and the max rule";
Section 9.15 rows "dc floor normalization" and "Leading coefficients at theta = pi"; M7 audit E-1, E-17).

Three defects these tests would have caught:

- **E-1**: ``filter_function`` divided an O(eps^2) infidelity by eps^{2(m+1)} whenever the scored quadrature was not one
  the composite pulse corrects, so BB1 reported a 70 % infidelity under 300 Hz dephasing noise. The whole
  ``CompositePulse`` branch of ``filter_function`` was uncalled by any test.
- The detuning moment was the variance of the sigma_z coefficient where the fitted c_hat wants Mount's eps_d, HALF the
  splitting: every reported detuning floor was low by 4^(m+1).
- **E-17**: a ``DecouplingSequence`` got no floor at all, so the module reported the first-order estimate alone for
  "precisely the band that dominates a real trap".

The arbiter throughout is the Gauss-Hermite average of the EXACT frozen-noise infidelity
1 - (1/4)|Tr(U_c^dag U)|^2, which owes nothing to the c_hat machinery.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import composite_pulse, decoupling_sequence, filter_function, gaussian_spectrum
from qutip_trap.control.composite import leading_coefficient
from qutip_trap.noise.decoupling import (
    composite_segments,
    dc_floor,
    frozen_noise_floor,
    frozen_noise_infidelity,
)
from tests.m2_fixtures import microwave_device

RABI_HZ = 50e3
RMS_HZ = 300.0


def _spectrum() -> object:
    """S_b of the sigma_z coefficient: 300 Hz rms, 300 Hz corner, so <beta^2>/Omega^2 = 3.6e-5 at Omega = 2 pi 50 kHz."""
    return gaussian_spectrum(2.0 * math.pi * RMS_HZ, 2.0 * math.pi * RMS_HZ, "(rad/s)^2/(rad/s)")


def test_dc_floor_order_is_channel_matched_not_the_pulses_own_order() -> None:
    """E-1: at m = 0 the floor is c_hat_1 <beta^2>_eps/Omega^2 with c_hat_1 = 1 (detuning, PLAN.md:1486's primitive 1.0)
    and pi^2/4 (amplitude), so the dephasing floors of the primitive, SK1 and BB1 COINCIDE and CORPSE's amplitude floor
    equals the primitive's. Before the fix SK1 read 108x and BB1 19400x too high (0.6998, a 70 % infidelity)."""
    dev, spec = microwave_device(), _spectrum()

    def floor(family: str, quadrature: str) -> float:
        r = filter_function(
            dev, composite_pulse(family, math.pi), quadrature=quadrature, spectrum=spec, rabi_hz=RABI_HZ
        )  # type: ignore[arg-type]
        return float(r.fitted["infidelity_dc"][0])

    # detuning: none of these three corrects it, so all three sit at the primitive's m = 0 floor
    prim_d = floor("primitive", "dephasing")
    assert floor("SK1", "dephasing") == pytest.approx(prim_d, rel=1e-5)
    assert floor("BB1", "dephasing") == pytest.approx(prim_d, rel=1e-5)
    assert prim_d < 1e-3, "the BB1 70 % absurdity is gone"
    # amplitude: CORPSE corrects only the detuning, so its amplitude floor is the primitive's
    prim_a = floor("primitive", "amplitude")
    assert floor("CORPSE", "amplitude") == pytest.approx(prim_a, rel=1e-5)
    # the two m = 0 leading coefficients the plan prints
    for fam in ("primitive", "SK1", "BB1"):
        assert leading_coefficient(composite_pulse(fam, math.pi), "detuning", 2) == pytest.approx(
            1.0, rel=1e-5
        )
    for fam in ("primitive", "CORPSE"):
        assert leading_coefficient(composite_pulse(fam, math.pi), "amplitude", 2) == pytest.approx(
            math.pi**2 / 4.0, rel=1e-5
        )
    # SCROFULOUS's eps_d coefficient is the plan's 4.0, so its dephasing floor is 4x the primitive's
    assert leading_coefficient(composite_pulse("SCROFULOUS", math.pi), "detuning", 2) == pytest.approx(
        4.0, rel=1e-5
    )
    assert floor("SCROFULOUS", "dephasing") == pytest.approx(4.0 * prim_d, rel=1e-4)


def test_frozen_noise_infidelity_is_the_exact_dc_limit_of_both_quadratures() -> None:
    """Section 6.9's H_0 = beta . sigma with NO 1/2 on the Pauli vector: a frozen sigma_z coefficient b on a primitive pi
    pulse costs 4(b/Omega)^2 (b being HALF the splitting), while a frozen amplitude beta_a costs sin^2(pi beta_a/2 Omega)
    -> (pi^2/4)(beta_a/Omega)^2. These two normalizations are the whole content of the factor-4 fix."""
    prim = composite_segments(composite_pulse("primitive", math.pi), 1.0)
    for rel in (1e-3, 2e-3, 4e-3):
        assert frozen_noise_infidelity(prim, rel, "dephasing") == pytest.approx(4.0 * rel**2, rel=1e-4)
        assert frozen_noise_infidelity(prim, rel, "amplitude") == pytest.approx(
            math.sin(math.pi * rel / 2.0) ** 2, rel=1e-9
        )
    assert frozen_noise_infidelity(prim, 0.0, "dephasing") == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        frozen_noise_infidelity(prim, 1e-3, "universal")


@pytest.mark.parametrize(
    ("family", "quadrature", "rel"),
    [
        ("primitive", "dephasing", 5e-4),
        ("primitive", "amplitude", 5e-4),
        ("SK1", "dephasing", 5e-4),
        ("SK1", "amplitude", 5e-4),
        ("BB1", "dephasing", 5e-4),
        ("BB1", "amplitude", 5e-4),
        ("CORPSE", "amplitude", 5e-4),
        # SCROFULOUS's three pi-ish segments make its O(beta^4) remainder the largest of the library at this band
        ("SCROFULOUS", "dephasing", 2e-3),
        ("SCROFULOUS", "amplitude", 5e-4),
    ],
)
def test_reported_dc_floor_matches_the_exact_gaussian_frozen_average(
    family: str, quadrature: str, rel: float
) -> None:
    """The c_hat x (2m+1)!! x rel^{m+1} floor the module reports is the leading term of <1 - F> over a Gaussian frozen
    beta; at <beta^2>/Omega^2 = 3.6e-5 the two agree to 5e-4 relative (2e-3 for SCROFULOUS). Before the detuning fix the
    dephasing rows were low by exactly 4 (m = 0) and the CORPSE detuning row by 16 (m = 1)."""
    dev, spec = microwave_device(), _spectrum()
    pulse = composite_pulse(family, math.pi)
    reported = float(
        filter_function(dev, pulse, quadrature=quadrature, spectrum=spec, rabi_hz=RABI_HZ).fitted[  # type: ignore[arg-type]
            "infidelity_dc"
        ][0]
    )
    omega = 2.0 * math.pi * RABI_HZ
    exact = frozen_noise_floor(composite_segments(pulse, omega), spec.variance(), quadrature)  # type: ignore[arg-type]
    assert reported == pytest.approx(exact, rel=rel), (family, quadrature, reported, exact)


def test_corpse_detuning_floor_needs_the_splitting_normalization_and_converges_as_four_to_the_m_plus_one() -> (
    None
):
    """CORPSE corrects the detuning at order m = 1, so its floor is c_hat_2 x 3!! x rel^2 and the eps_d-vs-beta_d factor
    enters as 4^2 = 16. Shrinking the noise drives the exact-to-leading ratio to 16 from above (33.4, 20.3, 17.1, 16.3 at
    <beta^2>/Omega^2 = 3.6e-5, 9.0e-6, 2.25e-6, 5.6e-7), which is what identifies the exponent as 4^(m+1) and not 4."""
    pulse = composite_pulse("CORPSE", math.pi)
    omega = 2.0 * math.pi * RABI_HZ
    segs = composite_segments(pulse, omega)
    c_hat = leading_coefficient(pulse, "detuning", 4)
    ratios = []
    for var_rel in (3.6e-5, 9.0e-6, 2.25e-6, 5.625e-7):
        var = var_rel * omega**2
        naive = dc_floor(c_hat, 1, var, omega)  # the un-normalized pairing the module used to report
        exact = frozen_noise_floor(segs, var, "dephasing", nodes=81)
        ratios.append(exact / naive)
    assert ratios[0] > ratios[-1] > 16.0, ratios
    assert ratios[-1] == pytest.approx(16.0, rel=0.03), ratios
    # and the module's own reported floor, which carries the 4 <beta^2>, is within 3 % of exact at the smallest noise
    var = 5.625e-7 * omega**2
    assert dc_floor(c_hat, 1, 4.0 * var, omega) == pytest.approx(
        frozen_noise_floor(segs, var, "dephasing", nodes=81), rel=0.03
    )


def test_decoupling_sequences_get_the_max_rule_too() -> None:
    """E-17: Section 6.9's "evaluate both estimates and report the larger at every point" applies to a sequence as much as
    to a composite pulse. On a 300 Hz Gaussian band with 4 us pi pulses over 200 us the reported floor is the exact
    Gaussian frozen average, and for CPMG-4 it EXCEEDS the first-order estimate (1.29e-6), so the max rule bites -
    without it the module reported 1.29e-6 for a sequence whose frozen-noise error is 8.9e-6."""
    dev = microwave_device()
    tau, tau_pi = 200e-6, 4e-6
    spec = gaussian_spectrum(
        2.0 * math.pi * RMS_HZ,
        2.0 * math.pi * RMS_HZ,
        "(rad/s)^2/(rad/s)",
        omega_max_rad_s=2.0 * math.pi * 3e3,
    )
    seqs = {
        "fid": decoupling_sequence("custom", 0, tau, 0.0, centres=[]),
        "hahn": decoupling_sequence("hahn", 1, tau, tau_pi),
        "cpmg2": decoupling_sequence("cpmg", 2, tau, tau_pi),
        "cpmg4": decoupling_sequence("cpmg", 4, tau, tau_pi),
    }
    out = {}
    for name, seq in seqs.items():
        r = filter_function(dev, seq, spectrum=spec)
        ff, floor_v = r.fitted["infidelity_ff"][0], r.fitted["infidelity_dc"][0]
        assert r.fitted["infidelity"][0] == pytest.approx(max(ff, floor_v))
        exact = frozen_noise_floor(seq.segments(), spec.variance(), "dephasing")
        assert floor_v == pytest.approx(exact, rel=1e-9), name
        out[name] = (ff, floor_v)
    # free induction: both estimates are the same O(xi^2) quantity, so they agree to a factor of order one
    assert 0.5 < out["fid"][1] / out["fid"][0] < 1.5
    # the echo suppresses the frozen offset far more than the band, so its floor collapses below its FF estimate
    assert out["hahn"][1] < 0.1 * out["hahn"][0]
    # CPMG-4's FF estimate is below its frozen floor: the max rule is load-bearing
    assert out["cpmg4"][1] > 5.0 * out["cpmg4"][0]
    assert out["cpmg4"][1] == pytest.approx(8.93e-6, rel=5e-3)


def test_white_noise_does_not_enter_the_frozen_floor_or_xi_squared() -> None:
    """The dc floor and xi^2 use the TABULATED band (``NoiseSpectrum.variance``) because a white level is not frozen over
    the sequence, while chi uses ``value`` (band plus white) because the filter function does score it. The retired
    ``+ spec.white_level * conv * 0.0`` said this by multiplying by zero (audit E-19)."""
    dev = microwave_device()
    tau = 200e-6
    band = gaussian_spectrum(
        2.0 * math.pi * RMS_HZ,
        2.0 * math.pi * RMS_HZ,
        "(rad/s)^2/(rad/s)",
        omega_max_rad_s=2.0 * math.pi * 3e3,
    )
    hot = gaussian_spectrum(
        2.0 * math.pi * RMS_HZ,
        2.0 * math.pi * RMS_HZ,
        "(rad/s)^2/(rad/s)",
        omega_max_rad_s=2.0 * math.pi * 3e3,
    )
    hot = type(hot)(hot.omega_rad_s, hot.S, hot.unit, white_level=1e3)
    seq = decoupling_sequence("cpmg", 2, tau, 4e-6)
    a = filter_function(dev, seq, spectrum=band).fitted
    b = filter_function(dev, seq, spectrum=hot).fitted
    assert b["xi2"][0] == pytest.approx(a["xi2"][0], rel=1e-12)
    assert b["infidelity_dc"][0] == pytest.approx(a["infidelity_dc"][0], rel=1e-12)
    assert b["chi"][0] > 1.05 * a["chi"][0], "the white level does raise chi"


def test_collapse_op_reports_ordinary_rates_not_rates_divided_by_two_pi() -> None:
    """Audit E-18: ``CollapseOp.rate_hz`` carries ndot, gamma_phi = 1/T_2, 2/tau and the intensity-noise and scattering
    densities, all of which are ordinary rates in s^-1 = Hz already. Section 5.6's 2 pi rule converts ANGULAR
    FREQUENCIES, so the six construction sites' ``/(2 pi)`` reported a number 6.28 times small. The audit offered a
    rename as an alternative, but Appendix E (PLAN.md:2561) declares the name ``rate_hz`` and the API freeze test
    enforces it, so the PLAN wins: the name stays and only the value is corrected."""
    from qutip_trap.api import HilbertSpace, ModeTruncation
    from qutip_trap.dynamics.channels import (
        heating_channels,
        motional_dephasing_channels,
        qubit_dephasing_channels,
    )
    from qutip_trap.trap.heating import thermal_collapse_rates

    space = HilbertSpace((2,), (ModeTruncation(0, 8, (0, 3), 0.1),), None, ())
    ndot = 400.0
    down, up = thermal_collapse_rates(ndot, None)
    ops = heating_channels(space, {0: ndot})
    by_channel = {o.channel: o for o in ops}
    assert by_channel["heating_down"].rate_hz == pytest.approx(down, rel=1e-12)
    assert by_channel["heating_up"].rate_hz == pytest.approx(up, rel=1e-12)
    assert by_channel["heating_up"].rate_hz == pytest.approx(ndot, rel=1e-12), (
        "the up rate IS ndot in quanta/s, not ndot/(2 pi)"
    )
    tau = 8e-3
    (md,) = motional_dephasing_channels(space, {0: tau})
    assert md.rate_hz == pytest.approx(2.0 / tau, rel=1e-12)
    t2 = 1.5
    (qd,) = qubit_dephasing_channels(space, {0: 1.0 / t2})
    assert qd.rate_hz == pytest.approx(1.0 / t2, rel=1e-12)
    # Appendix E's own name, kept: the freeze test (tests/test_api_freeze.py) requires it
    assert hasattr(md, "rate_hz") and not hasattr(md, "rate_per_s")


def test_frozen_noise_floor_guards() -> None:
    prim = composite_segments(composite_pulse("primitive", math.pi), 1.0)
    assert frozen_noise_floor(prim, 0.0, "dephasing") == 0.0
    with pytest.raises(ValueError):
        frozen_noise_floor(prim, -1.0, "dephasing")
    # the average of an even function of beta over a symmetric Gaussian: the odd nodes cancel exactly
    v = 1e-6
    assert frozen_noise_floor(prim, v, "dephasing", nodes=21) == pytest.approx(
        frozen_noise_floor(prim, v, "dephasing", nodes=61), rel=1e-8
    )
    assert frozen_noise_floor(prim, v, "dephasing") == pytest.approx(4.0 * v, rel=1e-4)
    assert np.isfinite(frozen_noise_floor(prim, v, "amplitude"))
