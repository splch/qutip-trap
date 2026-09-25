"""The atomic-rate layer of readout (PLAN.md Section 8.1): Bloch-solved rates against the closed forms and the ceiling, the
detected line, the micromotion factor, the efficiency entering once, the apparatus presets and the scheme's polarity."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt
from scipy.special import j0, j1

from qutip_trap.device.presets import crain_snspd_detector
from qutip_trap.dynamics.multilevel import MultiLevelOptions
from qutip_trap.light.bloch import CeilingViolation, beam_for_transition, shifted_beam
from qutip_trap.readout.detection import Detector, RecordModel
from qutip_trap.readout.fluorescence import (
    ALL_LINES,
    FluorescenceRates,
    ReadoutScheme,
    detected_line,
    detection_rates_for_ion,
    fit_mean_count_curve,
    mean_count_curve,
    neighbour_intensity_ratio,
    rates_from_bloch,
    rates_from_detected,
    saturation_ceiling,
    scattering_rate,
)
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD, MYERSON_CA40_PMT
from qutip_trap.species import species
from qutip_trap.species.polarization import linear_polarization
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.species.sources import SOURCES
from qutip_trap.units import TWO_PI
from tests.fixtures import (
    BRIGHT,
    D_HFP,
    D_HFS,
    DARK,
    GAMMA_S,
    MAGIC_ANGLE_RAD,
    YB,
    crain_record_model,
    detection_model,
    yb_detection_beam,
)
from tests.oracles import yb171_detection_rate

CA = species("40Ca+")
WAIST_M = 20e-6
CA_LEVELS = ("S1/2", "P1/2", "D3/2")


def _yb171_leakage_closed(s_o: float, gamma: float) -> tuple[float, float]:
    """(R_d, R_b) with Noek's prefactors and s = s_o/3: R_d = (2/3)(1/3)(Gamma/2) s (Gamma/(2 Delta_HFP))^2,
    R_b = (2/3)(Gamma/2) s (Gamma/(2(Delta_HFP + Delta_HFS)))^2."""
    s = s_o / 3.0
    r_d = (2.0 / 3.0) * (1.0 / 3.0) * (gamma / 2.0) * s * (gamma / (2.0 * D_HFP)) ** 2
    r_b = (2.0 / 3.0) * (gamma / 2.0) * s * (gamma / (2.0 * (D_HFP + D_HFS))) ** 2
    return r_d, r_b


# ---- 171Yb+ rates from the Bloch solve -------------------------------------------------------------------------------------


def test_yb171_bloch_rates_sit_below_the_closed_form_ceiling_and_obey_gamma_over_4() -> None:
    """R_o from the exact four-level solve is within 1 % below the (Gamma/18) form at s_o = 0.1, 1 G, the resonant manifold's
    ceiling is 1/4 and the excited population sits below it."""
    fl = rates_from_bloch(detection_model(0.1, 1.0), BRIGHT, DARK, line="S1/2<-P1/2")
    assert 0.99 < fl.R_bright_per_s / yb171_detection_rate(0.1, GAMMA_S) < 1.0
    assert fl.ceiling == 0.25
    assert fl.excited_population is not None and fl.excited_population < 0.25
    assert fl.R_bright_per_s < GAMMA_S / 4.0
    assert "conv.scattering_rate_object" in fl.provenance


def test_yb171_leakage_prefactors_and_the_3_over_49_ratio() -> None:
    """R_d and R_b from the slow-manifold analysis reproduce Noek's (2/3)(1/3) and (2/3) prefactors to 0.6 %;
    R_b/R_d = 3 (Delta_HFP/(Delta_HFP + Delta_HFS))^2 = 3/49 against 16.4/341 = 0.048 measured (30 % band)."""
    fl = rates_from_bloch(detection_model(0.1, 1.0), BRIGHT, DARK, line="S1/2<-P1/2")
    r_d, r_b = _yb171_leakage_closed(0.1, GAMMA_S)
    assert fl.R_dark_pumping_per_s == pytest.approx(r_d, rel=6e-3)
    assert fl.R_bright_pumping_per_s == pytest.approx(r_b, rel=6e-3)
    ratio = fl.R_bright_pumping_per_s / fl.R_dark_pumping_per_s
    closed_ratio = 3.0 * (D_HFP / (D_HFP + D_HFS)) ** 2
    assert ratio == pytest.approx(closed_ratio, rel=3e-3) and closed_ratio == pytest.approx(
        3.0 / 49.0, rel=2e-3
    )
    assert abs(ratio - 16.4 / 341.0) / (16.4 / 341.0) < 0.30


def test_scattering_rate_derives_the_bright_and_dark_manifolds_from_the_beams() -> None:
    """Without labels the bright manifold is the resonantly driven F = 1 triplet and the rates equal the explicit-label
    solve to 1e-9."""
    m = detection_model(0.1, 1.0)
    st = AtomicStructure(YB, 1.0, (0.0, 0.0, 1.0))
    fl, model = scattering_rate(st, m.beams, levels=("S1/2", "P1/2"))
    ground, excited = model.resonant_manifold()
    assert set(ground) == set(BRIGHT) and excited == ("P1/2 F=0 mF=0",)
    ref = rates_from_bloch(m, BRIGHT, DARK, line="S1/2<-P1/2")
    assert fl.R_bright_per_s == pytest.approx(ref.R_bright_per_s, rel=1e-9)
    assert fl.R_dark_pumping_per_s == pytest.approx(ref.R_dark_pumping_per_s, rel=1e-9)


def test_detection_rates_for_ion_builds_the_species_scheme_with_bright_is_1_polarity() -> None:
    beam = yb_detection_beam(0.5)
    fl, scheme, _model = detection_rates_for_ion(YB, 5.0, (1.0, 0.0, 0.0), [beam], levels=("S1/2", "P1/2"))
    assert (
        scheme.kind == "direct" and scheme.polarity == "bright_is_1" and scheme.classes == ("dark", "bright")
    )
    assert fl.R_bright_per_s > 0.0 and fl.shelf_decay_per_s == 0.0
    # the magic-angle beam along y at B = 5 G is destabilized: the rate is a sizeable fraction of the closed form
    assert fl.R_bright_per_s / yb171_detection_rate(0.5, GAMMA_S) > 0.5


def test_ceiling_violation_is_raised_not_assumed() -> None:
    """A rate object cannot be built from a state whose resonant excited population exceeds n_e/(n_e + n_g); the ceiling
    of the F = 1 -> F' = 0 manifold is 1/4, never the Lambda cycle's 1/3."""
    m = detection_model(0.1, 1.0)
    hot = qt.basis(m.build.n_internal, m.build.index("P1/2 F=0 mF=0")).proj()
    with pytest.raises(CeilingViolation):
        m.ceiling_report(hot)
    assert saturation_ceiling(3, 1) == 0.25 and saturation_ceiling(2, 1) == pytest.approx(1.0 / 3.0)
    assert saturation_ceiling(1, 1) == 0.5


# ---- the detected line and the micromotion factor ----------------------------------------------------------------------------


def _ca_beam(lower: str, upper: str, s_o: float, structure: AtomicStructure) -> object:
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
        pol,
        power_w=power,
        waist_m=WAIST_M,
    )


def _ca_detection_beams() -> tuple[AtomicStructure, list[object]]:
    """40Ca+ under 397 nm (S1/2 <- P1/2, s_o = 1) and its 866 nm repump (D3/2 <- P1/2, s_o = 5)."""
    st = AtomicStructure(CA, 4.0, (1.0, 0.0, 0.0))
    return st, [
        _ca_beam("S1/2 mJ=-1/2", "P1/2 mJ=1/2", 1.0, st),
        _ca_beam("D3/2 mJ=-1/2", "P1/2 mJ=1/2", 5.0, st),
    ]


def test_r_o_counts_the_cycling_line_only() -> None:
    """The all-line rate exceeds the detected 397 nm rate (2.2591e6/s) by 1/BR(S1/2) = 1.0688 to 1e-9, and a multi-line
    model refuses to guess the detected line."""
    st, beams = _ca_detection_beams()
    opts = MultiLevelOptions(leak="renormalize")
    detected, model = scattering_rate(st, beams, levels=CA_LEVELS, options=opts, line="S1/2<-P1/2")
    all_lines, _ = scattering_rate(st, beams, levels=CA_LEVELS, options=opts, line=ALL_LINES)
    repump, _ = scattering_rate(st, beams, levels=CA_LEVELS, options=opts, line="D3/2<-P1/2")
    g_s = CA.transition("S1/2-P1/2").partial_rate_rad_s
    g_d = CA.transition("D3/2-P1/2").partial_rate_rad_s
    ratio = all_lines.R_bright_per_s / detected.R_bright_per_s
    assert ratio == pytest.approx(1.0 + g_d / g_s, rel=1e-9)
    assert all_lines.R_bright_per_s == pytest.approx(
        detected.R_bright_per_s + repump.R_bright_per_s, rel=1e-9
    )
    assert ratio == pytest.approx(1.0 / (1.0 - 0.06435), rel=1e-9)
    assert detected.R_bright_per_s == pytest.approx(2.2591e6, rel=1e-3)
    with pytest.raises(ValueError, match="name the detected one"):
        detected_line(model)


def test_detection_rates_for_ion_selects_the_species_cycling_line() -> None:
    st, beams = _ca_detection_beams()
    opts = MultiLevelOptions(leak="renormalize")
    rates, scheme, _ = detection_rates_for_ion(
        CA,
        4.0,
        (1.0, 0.0, 0.0),
        beams,
        levels=CA_LEVELS,
        options=opts,
    )
    explicit, _ = scattering_rate(st, beams, levels=CA_LEVELS, options=opts, line="S1/2<-P1/2")
    assert rates.R_bright_per_s == pytest.approx(explicit.R_bright_per_s, rel=1e-9)
    summed, _ = scattering_rate(st, beams, levels=CA_LEVELS, options=opts, line=ALL_LINES)
    assert rates.R_bright_per_s < summed.R_bright_per_s / 1.05
    assert "shelf" in scheme.classes or scheme.kind == "shelving"


def test_a_single_line_model_derives_its_detected_line() -> None:
    """On 171Yb+'s one-line detection build ``line=None`` derives S1/2<-P1/2 and the explicit rate (1e-12)."""
    beam = yb_detection_beam(0.5)
    opts = MultiLevelOptions(leak="renormalize")
    st = AtomicStructure(YB, 5.0, (1.0, 0.0, 0.0))
    derived, model = scattering_rate(st, [beam], levels=("S1/2", "P1/2"), options=opts)
    assert detected_line(model) == "S1/2<-P1/2"
    explicit, _ = scattering_rate(st, [beam], levels=("S1/2", "P1/2"), options=opts, line="S1/2<-P1/2")
    assert derived.R_bright_per_s == pytest.approx(explicit.R_bright_per_s, rel=1e-12)


def test_micromotion_puts_j0_squared_on_the_carrier_and_j1_squared_on_the_sidebands() -> None:
    """R_o(beta) = J_0^2 R(Delta) + J_1^2 [R(Delta - Omega_rf) + R(Delta + Omega_rf)] to 1e-9 (0.930 of R_o(0) at beta = 0.4,
    30 MHz) with the pumping rates within 2 %; beta = 0 changes nothing and a missing rf frequency is refused."""
    beta = 0.4
    omega_rf = TWO_PI * 30e6
    beam = yb_detection_beam(0.5)
    st = AtomicStructure(YB, 5.0, (1.0, 0.0, 0.0))
    kw = {"levels": ("S1/2", "P1/2"), "options": MultiLevelOptions(leak="renormalize")}
    plain, _, _ = detection_rates_for_ion(YB, 5.0, (1.0, 0.0, 0.0), [beam], **kw)
    modulated, _, _ = detection_rates_for_ion(
        YB,
        5.0,
        (1.0, 0.0, 0.0),
        [beam],
        micromotion_beta=beta,
        omega_rf_rad_s=omega_rf,
        **kw,
    )

    def rate_at(offset: float) -> float:
        r, _ = scattering_rate(st, [shifted_beam(beam, offset)], line="S1/2<-P1/2", **kw)
        return r.R_bright_per_s

    expected = float(j0(beta)) ** 2 * rate_at(0.0) + float(j1(beta)) ** 2 * (
        rate_at(-omega_rf) + rate_at(omega_rf)
    )
    assert modulated.R_bright_per_s == pytest.approx(expected, rel=1e-9)
    ratio = modulated.R_bright_per_s / plain.R_bright_per_s
    assert ratio == pytest.approx(0.930, abs=0.005)
    assert float(j0(beta)) ** 2 < ratio < float(j0(beta)) ** 2 + 2.0 * float(j1(beta)) ** 2
    assert modulated.R_dark_pumping_per_s == pytest.approx(plain.R_dark_pumping_per_s, rel=0.02)
    assert modulated.R_dark_pumping_per_s / modulated.R_bright_per_s > (
        plain.R_dark_pumping_per_s / plain.R_bright_per_s
    )
    assert any("micromotion" in p for p in modulated.provenance)
    zero, _, _ = detection_rates_for_ion(
        YB, 5.0, (1.0, 0.0, 0.0), [beam], micromotion_beta=0.0, omega_rf_rad_s=omega_rf, **kw
    )
    assert zero.R_bright_per_s == pytest.approx(plain.R_bright_per_s, rel=1e-12)
    with pytest.raises(ValueError, match="rf frequency"):
        detection_rates_for_ion(YB, 5.0, (1.0, 0.0, 0.0), [beam], micromotion_beta=0.3, **kw)


# ---- the efficiency, the apparatus presets and the mean-count curve -------------------------------------------------------------


def test_the_efficiency_enters_once_and_linearly() -> None:
    """The detected rate is efficiency x R_o (1e-12) at every ``Detector.efficiency`` with the background separate, and
    ingested detected rates come back as measured."""
    rates = FluorescenceRates(
        R_bright_per_s=yb171_detection_rate(2.45, GAMMA_S),
        R_dark_pumping_per_s=341.0,
        R_bright_pumping_per_s=16.4,
    )
    for eff in (0.04356, 2.0 * 0.04356, 0.5):
        det = Detector("snspd", eff, 4.2, {}, None, None, 22e-6)
        assert rates.detected(det)[0] == pytest.approx(eff * rates.R_bright_per_s, rel=1e-12)
        model = RecordModel.from_rates(rates, det)
        assert model.detected_bright_per_s == pytest.approx(eff * rates.R_bright_per_s, rel=1e-12)
        assert model.background_per_s == 4.2
    ingested = rates_from_detected(472e3, 0.04356, dark_pumping_per_s=341.0, bright_pumping_per_s=16.4)
    detected, background = ingested.detected(crain_snspd_detector())
    assert detected == pytest.approx(472e3) and background == 4.2
    assert ingested.ceiling is None and "apparatus" in ingested.provenance[-1]


def test_the_apparatus_presets_ingest_their_measured_rates() -> None:
    """Myerson's 55800 s^-1 at 0.19 % and Crain's 472 kcps at 4.356 % with R_d = 341 Hz, R_b = 16.4 Hz; the scattered rate
    Crain's numbers imply is 0.088 Gamma, below the Gamma/4 ceiling."""
    myerson = MYERSON_CA40_PMT.rates()
    assert myerson.R_bright_per_s == pytest.approx(55_800.0 / 0.0019)
    assert myerson.shelf_decay_per_s == pytest.approx(1.0 / 1.168)
    assert myerson.ceiling is None and myerson.provenance[0] == "Myerson2008"
    crain = CRAIN_YB171_SNSPD.rates()
    assert crain.R_dark_pumping_per_s == 341.0 and crain.R_bright_pumping_per_s == 16.4
    assert crain.R_bright_per_s / GAMMA_S == pytest.approx(0.088, abs=1e-3)
    assert crain.R_bright_per_s < GAMMA_S / 4.0
    assert {MYERSON_CA40_PMT.source, CRAIN_YB171_SNSPD.source} <= set(SOURCES)


def test_neighbour_intensity_ratio() -> None:
    """I_ion/I_sat = 3 lambda^2/(8 pi^2 x^2) = 1.09e-4 for 111Cd+ at 4 um."""
    assert neighbour_intensity_ratio(214.5e-9, 4e-6) == pytest.approx(1.09e-4, abs=0.005e-4)


def test_mean_count_curve_is_the_two_state_rate_equation_and_its_fit_recovers_the_rates() -> None:
    """n(tau) = eps R_o [(R_b/k) tau + (R_d/k^2)(1 - e^{-k tau})] equals the Markov-chain mean to 1e-9, and the fit
    recovers (eps R_o, R_d, R_b) to 1e-5 from noiseless samples."""
    rm = crain_record_model()
    taus = np.linspace(1e-4, 60e-3, 12)
    curve = np.asarray(mean_count_curve(taus, 472e3, 341.0, 16.4))
    for t, n in zip(taus, curve):
        assert rm.mean_counts("bright", float(t)) - rm.background_per_s * t == pytest.approx(
            float(n), rel=1e-9
        )
    (r0, rd, rb), _sigma = fit_mean_count_curve(taus, curve, guess=(4e5, 300.0, 10.0))
    assert (
        r0 == pytest.approx(472e3, rel=1e-6)
        and rd == pytest.approx(341.0, rel=1e-5)
        and rb == pytest.approx(16.4, rel=1e-5)
    )


# ---- the scheme -----------------------------------------------------------------------------------------------------------------


def test_scheme_polarity_direct_vs_shelving_and_the_imperfect_transfer() -> None:
    yb = ReadoutScheme.direct(1)
    ca = ReadoutScheme.shelving(1)
    assert yb.polarity == "bright_is_1" and ca.polarity == "bright_is_0"
    assert yb.bit_of_class("bright") == 1 and ca.bit_of_class("bright") == 0
    assert yb.start_distribution(0) == {"dark": 1.0} and ca.start_distribution(1) == {"shelf": 1.0}
    harty = ReadoutScheme.shelving(0, transfer_probability=1.0 - 1.7e-4, off_resonant_shelving=3e-4)
    assert harty.transfer is not None
    assert harty.start_distribution(0)["shelf"] == pytest.approx(1.0 - 1.7e-4)
    assert harty.start_distribution(1)["bright"] == pytest.approx(1.0 - 3e-4)
    assert harty.dark_class == "shelf"
    for_species = ReadoutScheme.for_species(CA, ("S1/2 mJ=-1/2", "S1/2 mJ=+1/2"))
    assert for_species.kind == "shelving" and for_species.classes == ("bright", "shelf")
    with pytest.raises(ValueError):
        ReadoutScheme("direct", ("dark", "dark"))
    with pytest.raises(ValueError):
        FluorescenceRates(0.0, 1.0, 1.0)
