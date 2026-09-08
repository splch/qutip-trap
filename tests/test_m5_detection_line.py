"""The detected line and the micromotion factor on R_o (PLAN.md Sections 8.1, 8.8; Section 13 row "J_n(beta) on every drive").

Two defects of the 2026-09-07 M5 audit:

- B2: ``R_det = eps_sys R_o`` with eps_filter INSIDE eps_sys (Section 8.1, "Detected rate and background"), so R_o must
  count photons on the detected wavelength only. The rate object summed every non-sink decay line of the build, which for
  40Ca+ under 397 + 866 nm added the repump's 866 nm photons and made R_o 6.9 % high (6.4 % until the P1/2 -> D3/2
  branching became Ramm et al. 2013's 0.06435 on 2026-09-08).
- B5: Section 8.8 requires "R_o carries J_0(beta)^2 and a first-sideband channel J_1(beta)^2, like every other drive" and
  Section 13 requires "J_n(beta) on every drive: gates, cooling and detection". ``micromotion_detection_rate`` existed
  but had no caller, so the detection solve was unmodulated.
"""

from __future__ import annotations

import math

import pytest
from scipy.special import j0, j1

from qutip_trap.dynamics.multilevel import MultiLevelOptions
from qutip_trap.light.bloch import beam_for_transition, shifted_beam
from qutip_trap.readout.fluorescence import (
    ALL_LINES,
    detected_line,
    detection_rates_for_ion,
    micromotion_detection_rate,
    scattering_rate,
)
from qutip_trap.species import species
from qutip_trap.species.polarization import linear_polarization
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import TWO_PI
from tests.readout_fixtures import yb_detection_beam

CA = species("40Ca+")
YB = species("171Yb+")
MAGIC_ANGLE_RAD = math.acos(1.0 / math.sqrt(3.0))
WAIST_M = 20e-6
CA_LEVELS = ("S1/2", "P1/2", "D3/2")


def ca_beam(lower: str, upper: str, s_o: float, structure: AtomicStructure) -> object:
    """A beam on the dressed line at the magic angle to B = x, propagating along +y, with I/I_sat = s_o on axis."""
    line = structure.e1[(lower.split()[0], upper.split()[0])]
    power = s_o * line.i_sat_w_m2 * math.pi * WAIST_M**2 / 2.0
    pol = tuple(linear_polarization((0.0, 1.0, 0.0), MAGIC_ANGLE_RAD, (1.0, 0.0, 0.0)))
    return beam_for_transition(
        structure,
        lower,
        upper,
        0.0,
        (0.0, 1.0, 0.0),
        pol,  # type: ignore[arg-type]
        power_w=power,
        waist_m=WAIST_M,
    )


def ca_detection_beams() -> tuple[AtomicStructure, list[object]]:
    """40Ca+ under 397 nm (S1/2 <- P1/2, s_o = 1) and its 866 nm repump (D3/2 <- P1/2, s_o = 5)."""
    st = AtomicStructure(CA, 4.0, (1.0, 0.0, 0.0))
    beams = [
        ca_beam("S1/2 mJ=-1/2", "P1/2 mJ=1/2", 1.0, st),
        ca_beam("D3/2 mJ=-1/2", "P1/2 mJ=1/2", 5.0, st),
    ]
    return st, beams


def test_r_o_counts_the_cycling_line_only_and_is_6_percent_below_the_all_line_sum() -> None:
    """In steady state every photon leaves P1/2, so the all-line sum exceeds the detected 397 nm rate by exactly
    1 + Gamma(D3/2 <- P1/2)/Gamma(S1/2 <- P1/2) = 1/BR(S1/2): a closed identity, pinned at 1e-9 (Section 8.1)."""
    st, beams = ca_detection_beams()
    opts = MultiLevelOptions(leak="renormalize")
    detected, model = scattering_rate(st, beams, levels=CA_LEVELS, options=opts, line="S1/2<-P1/2")  # type: ignore[arg-type]
    all_lines, _ = scattering_rate(st, beams, levels=CA_LEVELS, options=opts, line=ALL_LINES)  # type: ignore[arg-type]
    repump, _ = scattering_rate(st, beams, levels=CA_LEVELS, options=opts, line="D3/2<-P1/2")  # type: ignore[arg-type]
    g_s = CA.transition("S1/2-P1/2").partial_rate_rad_s
    g_d = CA.transition("D3/2-P1/2").partial_rate_rad_s
    ratio = all_lines.R_bright_per_s / detected.R_bright_per_s
    assert ratio == pytest.approx(1.0 + g_d / g_s, rel=1e-9)
    assert all_lines.R_bright_per_s == pytest.approx(
        detected.R_bright_per_s + repump.R_bright_per_s, rel=1e-9
    )
    # the anchor, as the closed identity 1/BR(S1/2) it is: 2.2591e6/s on 397 nm against 2.4145e6 summed, a 6.9 %
    # inflation. It was 1.0638 (6.4 %) until 2026-09-08, when the P1/2 -> D3/2 branching became Ramm et al. 2013's
    # 0.06435 in place of Section 8.1's 0.06 (ledger conv.ca40_branching); 1/0.93565 = 1.068776
    assert ratio == pytest.approx(1.0688, rel=1e-4)
    assert ratio == pytest.approx(1.0 / (1.0 - 0.06435), rel=1e-9)
    assert detected.R_bright_per_s == pytest.approx(2.2591e6, rel=1e-3)
    # a multi-line model refuses to guess the detected line
    with pytest.raises(ValueError, match="name the detected one"):
        detected_line(model)


def test_detection_rates_for_ion_selects_the_species_cycling_line() -> None:
    """``detection_rates_for_ion`` is the only entry the run pipeline and the detection_histogram experiment use, and it
    never set the line: the 866 nm repump photons went into R_o, eps_B, eps_D and the whole budget."""
    st, beams = ca_detection_beams()
    opts = MultiLevelOptions(leak="renormalize")
    rates, scheme, _ = detection_rates_for_ion(
        CA,
        4.0,
        (1.0, 0.0, 0.0),
        beams,  # type: ignore[arg-type]
        levels=CA_LEVELS,
        options=opts,
    )
    explicit, _ = scattering_rate(st, beams, levels=CA_LEVELS, options=opts, line="S1/2<-P1/2")  # type: ignore[arg-type]
    assert rates.R_bright_per_s == pytest.approx(explicit.R_bright_per_s, rel=1e-9)
    summed, _ = scattering_rate(st, beams, levels=CA_LEVELS, options=opts, line=ALL_LINES)  # type: ignore[arg-type]
    assert rates.R_bright_per_s < summed.R_bright_per_s / 1.05
    assert "shelf" in scheme.classes or scheme.kind == "shelving"


def test_a_single_line_model_derives_its_detected_line() -> None:
    """171Yb+'s detection build carries one line (the 935 nm repump's upper level is not tabulated, M3a finding), so the
    detected line is unambiguous and line=None keeps deriving it."""
    beam = yb_detection_beam(0.5)
    opts = MultiLevelOptions(leak="renormalize")
    derived, model = scattering_rate(
        AtomicStructure(YB, 5.0, (1.0, 0.0, 0.0)), [beam], levels=("S1/2", "P1/2"), options=opts
    )
    assert detected_line(model) == "S1/2<-P1/2"
    explicit, _ = scattering_rate(
        AtomicStructure(YB, 5.0, (1.0, 0.0, 0.0)),
        [beam],
        levels=("S1/2", "P1/2"),
        options=opts,
        line="S1/2<-P1/2",
    )
    assert derived.R_bright_per_s == pytest.approx(explicit.R_bright_per_s, rel=1e-12)


def test_micromotion_puts_j0_squared_on_the_carrier_and_j1_squared_on_the_sidebands() -> None:
    """Section 13 row "J_n(beta) on every drive": R_o(beta) = J_0^2 R(Delta) + J_1^2 [R(Delta - Omega_rf) +
    R(Delta + Omega_rf)] and below R_o(0)."""
    beta = 0.4
    omega_rf = TWO_PI * 30e6
    beam = yb_detection_beam(0.5)
    st = AtomicStructure(YB, 5.0, (1.0, 0.0, 0.0))
    kw = {"levels": ("S1/2", "P1/2"), "options": MultiLevelOptions(leak="renormalize")}
    plain, _, _ = detection_rates_for_ion(YB, 5.0, (1.0, 0.0, 0.0), [beam], **kw)  # type: ignore[arg-type]
    modulated, _, _ = detection_rates_for_ion(
        YB,
        5.0,
        (1.0, 0.0, 0.0),
        [beam],
        micromotion_beta=beta,
        omega_rf_rad_s=omega_rf,
        **kw,  # type: ignore[arg-type]
    )

    def rate_at(offset: float) -> float:
        r, _ = scattering_rate(
            st,
            [shifted_beam(beam, offset)],
            line="S1/2<-P1/2",
            **kw,  # type: ignore[arg-type]
        )
        return r.R_bright_per_s

    expected = micromotion_detection_rate(rate_at, beta, omega_rf)
    assert modulated.R_bright_per_s == pytest.approx(expected, rel=1e-9)
    assert modulated.R_bright_per_s < plain.R_bright_per_s
    # J_0(0.4)^2 = 0.92237 on the carrier; the two sidebands 30 MHz off a 19.62 MHz line carry 2 J_1^2 = 0.0769 of the
    # drive but only ~0.097 of the resonant rate each, so they give back 0.0074: 0.9298 in total
    ratio = modulated.R_bright_per_s / plain.R_bright_per_s
    assert ratio == pytest.approx(0.930, abs=0.005)
    assert ratio > float(j0(beta)) ** 2, "the sidebands give a little back"
    assert ratio < float(j0(beta)) ** 2 + 2.0 * float(j1(beta)) ** 2
    # the pumping rates carry the same J-weighting, but they are linear in intensity with no saturation denominator
    # (Section 8.1), so sum_n J_n^2 = 1 leaves them almost unchanged (+0.5 % here, the sidebands' own R_d being 7 % above
    # the resonant one) while R_o loses 7 %: micromotion costs signal and RAISES the leakage ratio, an M5 finding
    assert modulated.R_dark_pumping_per_s == pytest.approx(plain.R_dark_pumping_per_s, rel=0.02)
    assert modulated.R_dark_pumping_per_s / modulated.R_bright_per_s > (
        plain.R_dark_pumping_per_s / plain.R_bright_per_s
    )
    assert any("micromotion" in p for p in modulated.provenance)
    # beta = 0 costs no extra solve and is the identity
    zero, _, _ = detection_rates_for_ion(
        YB, 5.0, (1.0, 0.0, 0.0), [beam], micromotion_beta=0.0, omega_rf_rad_s=omega_rf, **kw
    )  # type: ignore[arg-type]
    assert zero.R_bright_per_s == pytest.approx(plain.R_bright_per_s, rel=1e-12)
    with pytest.raises(ValueError, match="rf frequency"):
        detection_rates_for_ion(YB, 5.0, (1.0, 0.0, 0.0), [beam], micromotion_beta=0.3, **kw)  # type: ignore[arg-type]
