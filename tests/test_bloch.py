"""The multi-level optical-Bloch model: the two-level Lorentzian (RMP Eq. 96), the 171Yb+ detection rate
(Gamma/18) s_o/[1 + (2/9) s_o + (2 Delta/Gamma)^2] and its leakage rates, the Gamma/4 ceiling, Berkeland's dark-state
counts, optical pumping, the frame graph with its Floquet fallback, the 40Ca+ S-P-D dark resonance and the laser
linewidth on the optical coherences."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import qutip as qt
from scipy.optimize import brentq

from qutip_trap.dynamics.multilevel import (
    SINK,
    MultiLevelOptions,
    assign_frames,
    decay_sum_rule_residual,
)
from qutip_trap.light.beams import Beam, PolarizationModulation
from qutip_trap.light.bloch import BlochModel, CeilingViolation, beam_for_transition, shifted_beam
from qutip_trap.species import species
from qutip_trap.species.polarization import linear_polarization
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import C_M_PER_S, TWO_PI
from tests.bloch_fixtures import (
    TWO_LEVEL_EXCITED,
    TWO_LEVEL_EXCITED_PLUS,
    TWO_LEVEL_GROUND,
    gamma_rad_s,
    pi_beam,
    sigma_plus_beam,
    structure,
    two_level_atom,
    two_level_lorentzian,
)

YB = species("171Yb+")
YB_LINE = YB.transition("S1/2-P1/2")
GAMMA_S = YB_LINE.partial_rate_rad_s
"""The partial S1/2-P1/2 rate, 2 pi x 19.62 MHz, the Gamma of the detection-rate closed forms."""
D_HFP = TWO_PI * 2.105e9
D_HFS = TWO_PI * 12_642_812_118.5
BRIGHT = tuple(f"S1/2 F=1 mF={m}" for m in (-1, 0, 1))
DARK = ("S1/2 F=0 mF=0",)
MAGIC = tuple(linear_polarization((1.0, 0.0, 0.0), math.acos(1.0 / math.sqrt(3.0)), (0.0, 0.0, 1.0)))
WAIST = 20e-6


def detection_model(s0: float, b_gauss: float, delta_rad_s: float = 0.0, **kw: object) -> BlochModel:
    """171Yb+ S1/2 + P1/2 with the D3/2 branch folded back (the repump-level constants are not tabulated), the
    detection beam at the magic angle with I/I_sat = s0 on the partial-rate I_sat."""
    st = AtomicStructure(YB, b_gauss, (0.0, 0.0, 1.0))
    power = s0 * YB_LINE.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    beam = beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=0 mF=0",
        delta_rad_s,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=power,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    return BlochModel(
        st, [beam], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"), **kw
    )  # type: ignore[arg-type]


def closed_form_rate(s0: float, delta_rad_s: float = 0.0) -> float:
    return (GAMMA_S / 18.0) * s0 / (1.0 + (2.0 / 9.0) * s0 + (2.0 * delta_rad_s / GAMMA_S) ** 2)


# ---- the builder ---------------------------------------------------------------------------------------------------------


def test_two_level_steady_state_is_the_rmp_lorentzian_to_1e_9() -> None:
    """Gamma rho_ee = Gamma (s/2)/(1 + s + (2 Delta/Gamma)^2) with s = 2 Omega^2/Gamma^2 (RMP 2003 Eq. 96)."""
    st = structure(two_level_atom())
    g = gamma_rad_s()
    for om, dl in ((0.05 * g, -0.5 * g), (0.5 * g, 0.0), (2.0 * g, 0.3 * g)):
        m = BlochModel(st, [pi_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED, om, dl)])
        ss = m.steadystate()
        assert ss.total_photon_rate_per_s == pytest.approx(two_level_lorentzian(om, dl), rel=1e-9)
        assert ss.ceiling.ceiling == 0.5 and ss.excited_population < 0.5
        assert ss.method == "steadystate" and m.build.static


def test_intensity_to_rabi_chain_matches_the_plan_convention() -> None:
    """2 Omega^2/Gamma^2 = I/I_sat for the full-line Rabi frequency with the transition's two-level I_sat; the pi component
    drives |g> <-> |e, 0>."""
    st = structure(two_level_atom())
    g = gamma_rad_s()
    beam = pi_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED, 0.3 * g, 0.0)
    s0 = beam.intensity_at(np.asarray(beam.pointing_m)) / st.e1[("S0/2", "P2/2")].i_sat_w_m2
    # for J = 0 -> 1 each of the three components carries |<J||d||J'>|^2/3 with |<J||d||J'>|^2 = 9 pi eps0 hbar c^3 Gamma/omega^3,
    # so |g> <-> |e, 0> is a unit-strength closed two-level transition and 2 Omega_pi^2/Gamma^2 = I/I_sat exactly
    assert 2.0 * (0.3 * g) ** 2 / g**2 == pytest.approx(s0, rel=1e-9)


def test_dissipator_sum_rule_and_trace_preservation_for_171yb() -> None:
    m = detection_model(0.1, 1.0)
    b = m.build
    assert decay_sum_rule_residual(b) < 1e-12
    L = np.asarray(b.liouvillian().full())
    ident = np.eye(b.n_internal).reshape(-1, order="F")
    assert np.max(np.abs(ident @ L)) < 1e-6 * np.max(np.abs(L))
    assert len(b.c_ops) == 3, (
        "one collapse operator per polarization channel q, never one per excited sublevel"
    )
    assert b.labels == (
        "S1/2 F=1 mF=-1",
        "S1/2 F=0 mF=0",
        "S1/2 F=1 mF=0",
        "S1/2 F=1 mF=1",
        "P1/2 F=1 mF=-1",
        "P1/2 F=0 mF=0",
        "P1/2 F=1 mF=0",
        "P1/2 F=1 mF=1",
    )
    assert any("folded back" in a for a in b.approximations)


def test_frame_puts_every_detuning_on_the_diagonal() -> None:
    """The four F=1 -> F'=0, F=1 -> F'=1, F=0 -> F'=1 detunings of the detection beam appear as -Delta on the diagonal."""
    m = detection_model(0.1, 1.0, delta_rad_s=-0.5 * GAMMA_S)
    b = m.build
    assert isinstance(b.H, qt.Qobj)
    diag = np.real(np.diag(b.H.full()))
    by = {c.upper: c for c in b.couplings if c.lower == "S1/2 F=1 mF=0"}
    f0 = by["P1/2 F=0 mF=0"]
    assert f0.detuning_rad_s == pytest.approx(-0.5 * GAMMA_S, rel=1e-6)
    assert diag[b.index("P1/2 F=0 mF=0")] - diag[b.index("S1/2 F=1 mF=0")] == pytest.approx(
        0.5 * GAMMA_S, rel=1e-6
    )
    f1 = by["P1/2 F=1 mF=0"]
    assert f1.detuning_rad_s / TWO_PI == pytest.approx(-2.105e9 - 0.5 * GAMMA_S / TWO_PI, rel=1e-3)
    dark = next(c for c in b.couplings if c.lower == "S1/2 F=0 mF=0" and c.upper == "P1/2 F=1 mF=0")
    assert dark.detuning_rad_s / TWO_PI == pytest.approx(
        -(2.105e9 + 12.642812e9) - 0.5 * GAMMA_S / TWO_PI, rel=1e-3
    )


# ---- 171Yb+ detection rates -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("s0, b_gauss, tol", [(0.01, 0.3, 0.01), (0.1, 1.0, 0.01), (1.0, 2.6, 0.01)])
def test_yb171_detection_rate_recovers_gamma_over_18_and_two_ninths(
    s0: float, b_gauss: float, tol: float
) -> None:
    """R_o = (Gamma/18) s_o/[1 + (2/9) s_o] from the conditional bright state at pumping rate << Zeeman << Gamma; the 1/3 :
    2/3 branching makes the prefactor Gamma/18. The exact four-level rate sits BELOW the closed form, because the field that
    destabilizes the dark state also detunes the sigma transitions: 0.995 to 0.9986 of it at the best field."""
    dr = detection_model(s0, b_gauss).detection_rates(BRIGHT, DARK, line="S1/2<-P1/2")
    ratio = dr.R_bright_per_s / closed_form_rate(s0)
    assert 1.0 - tol < ratio <= 1.0 + 1e-6
    assert dr.ceiling.ceiling == 0.25
    assert (len(dr.ceiling.ground_labels), len(dr.ceiling.excited_labels)) == (3, 1)
    assert dr.R_bright_per_s <= GAMMA_S / 4.0
    assert dr.separation > 100.0


def test_yb171_detuning_dependence_is_the_lorentzian_bracket() -> None:
    dr0 = detection_model(0.1, 1.0).detection_rates(BRIGHT, DARK, line="S1/2<-P1/2")
    dr1 = detection_model(0.1, 1.0, delta_rad_s=-0.5 * GAMMA_S).detection_rates(
        BRIGHT, DARK, line="S1/2<-P1/2"
    )
    assert dr1.R_bright_per_s / dr0.R_bright_per_s == pytest.approx(
        closed_form_rate(0.1, -0.5 * GAMMA_S) / closed_form_rate(0.1), rel=6e-3
    )


def test_yb171_leakage_prefactors_settle_the_noek_crain_factor_in_noeks_favour() -> None:
    """R_d = (2/3)(1/3)(Gamma/2) s (Gamma/2 Delta_HFP)^2 and R_b = (2/3)(Gamma/2) s (Gamma/(2(Delta_HFP + Delta_HFS)))^2 with
    Noek's s = s_o/3, i.e. R_d = (2/27)(Gamma/2) s_o (Gamma/2 Delta_HFP)^2; R_b/R_d = 3/49."""
    s0 = 0.1
    dr = detection_model(s0, 1.0).detection_rates(BRIGHT, DARK, line="S1/2<-P1/2")
    rd_noek = (2.0 / 3.0) * (1.0 / 3.0) * (GAMMA_S / 2.0) * (s0 / 3.0) * (GAMMA_S / (2.0 * D_HFP)) ** 2
    rd_crain = (1.0 / 3.0) * (GAMMA_S / 2.0) * (s0 / 3.0) * (GAMMA_S / (2.0 * D_HFP)) ** 2
    rb_noek = (2.0 / 3.0) * (GAMMA_S / 2.0) * (s0 / 3.0) * (GAMMA_S / (2.0 * (D_HFP + D_HFS))) ** 2
    assert dr.R_dark_pumping_per_s == pytest.approx(rd_noek, rel=0.01)
    assert abs(dr.R_dark_pumping_per_s / rd_crain - 1.0) > 0.3
    assert dr.R_bright_pumping_per_s == pytest.approx(rb_noek, rel=0.01)
    assert dr.R_bright_pumping_per_s / dr.R_dark_pumping_per_s == pytest.approx(3.0 / 49.0, rel=0.01)


def test_yb171_crain_operating_point_numbers() -> None:
    """At s_o = 2.45 Noek's form predicts 243 Hz, Crain's 364 Hz for R_d and 14.9 Hz for R_b; the exact four-level solve
    at the field that maximizes the fluorescence sits near Crain's measured 341(13) and 16.4(5) Hz."""
    s0 = 2.45
    rd_noek = (2.0 / 27.0) * (GAMMA_S / 2.0) * s0 * (GAMMA_S / (2.0 * D_HFP)) ** 2
    rd_crain = (1.0 / 9.0) * (GAMMA_S / 2.0) * s0 * (GAMMA_S / (2.0 * D_HFP)) ** 2
    rb = (2.0 / 9.0) * (GAMMA_S / 2.0) * s0 * (GAMMA_S / (2.0 * (D_HFP + D_HFS))) ** 2
    assert rd_noek == pytest.approx(243.0, rel=0.01)
    assert rd_crain == pytest.approx(364.6, rel=0.01)
    assert rb == pytest.approx(14.86, rel=0.01)
    dr = detection_model(s0, 4.7).detection_rates(BRIGHT, DARK, line="S1/2<-P1/2")
    assert 0.99 < dr.R_bright_per_s / closed_form_rate(s0) <= 1.0 + 1e-6
    assert dr.R_dark_pumping_per_s == pytest.approx(rd_noek, rel=0.03)
    assert dr.R_bright_pumping_per_s == pytest.approx(rb, rel=0.01)
    # at a sub-optimal field (1 G) the bright manifold is redistributed by the dark-state coherence and R_d rises to about 347 Hz,
    # near Crain's measured 341(13) Hz; the measured triple is therefore not a test of the prefactor alone
    dr1 = detection_model(s0, 1.0).detection_rates(BRIGHT, DARK, line="S1/2<-P1/2")
    assert 320.0 < dr1.R_dark_pumping_per_s < 380.0
    assert dr1.R_bright_per_s / closed_form_rate(s0) < 0.3


def test_full_steady_state_is_mostly_dark_and_the_conditional_state_is_bright() -> None:
    """The long-time steady state of a detected ion sits in F = 0 with weight R_d/(R_d + R_b) ~ 0.95."""
    m = detection_model(0.1, 1.0)
    ss = m.steadystate()
    p_dark = ss.populations["S1/2 F=0 mF=0"]
    mr = m.manifold_rates(BRIGHT, DARK)
    assert p_dark == pytest.approx(1.0 - mr.weight_a, abs=2e-3)
    assert mr.weight_a == pytest.approx(mr.rate_b_to_a_per_s / (mr.rate_a_to_b_per_s + mr.rate_b_to_a_per_s))
    pops_b = m.build.populations(mr.conditional_a)
    assert pops_b["S1/2 F=0 mF=0"] == pytest.approx(0.0, abs=1e-9)
    assert sum(pops_b[lab] for lab in BRIGHT) == pytest.approx(
        1.0 - ss.excited_population / max(mr.weight_a, 1e-12), abs=0.05
    )


def _repump_935_model(s0: float, s935: float, b_gauss: float = 4.7) -> BlochModel:
    """171Yb+ S1/2 + P1/2 + D3/2 + 3D[3/2]1/2 with BOTH the 369.5 nm detection beam and the 935 nm repump present.

    The repump is tuned to the level centroid rather than to a sublevel pair, so its residual against the frame that
    the D3/2 <- P1/2 decay already fixes is zero and the build stays static.
    """
    st = AtomicStructure(YB, b_gauss, (0.0, 0.0, 1.0))
    line935 = YB.transition("D3/2-3D[3/2]1/2")
    power = s0 * YB_LINE.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    detect = beam_for_transition(
        st, "S1/2 F=1 mF=0", "P1/2 F=0 mF=0", 0.0, (1.0, 0.0, 0.0), MAGIC, power_w=power, waist_m=WAIST
    )  # type: ignore[arg-type]
    centroid = TWO_PI * (
        (YB.level("3D[3/2]1/2").energy_hz - YB.level("D3/2").energy_hz)
        - (st.state("3D[3/2]1/2 F=1 mF=0").energy_hz - st.state("D3/2 F=1 mF=0").energy_hz)
    )
    repump = beam_for_transition(
        st,
        "D3/2 F=1 mF=0",
        "3D[3/2]1/2 F=1 mF=0",
        centroid,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=s935 * line935.i_sat_w_m2 * math.pi * WAIST**2 / 2.0,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    return BlochModel(
        st,
        [detect, repump],
        levels=("S1/2", "P1/2", "D3/2", "3D[3/2]1/2"),
        options=MultiLevelOptions(leak="renormalize"),
    )


def test_the_slow_manifold_mode_is_chosen_by_its_manifold_weights_not_by_its_index() -> None:
    """With the 935 nm repump beam and the D manifold in the model, the SECOND-slowest Liouvillian mode is a D-manifold
    relaxation, not the bright <-> dark pumping, so the connecting mode is found by its projections on the two
    manifolds. 3D[3/2]1/2 -> S1/2 at 297 nm is untabulated branching, not a Transition, so nothing returns from the D
    branch to S1/2 and the D manifold is ABSORBING; both refusals name their cause instead of returning a negative rate.
    """
    m = _repump_935_model(2.45, 10.0)
    assert m.build.static and m.build.frame.beats_rad_s == ()
    assert m.build.n_internal == 20
    assert any(lab.startswith("3D[3/2]1/2") for lab in m.build.labels)
    # the D3/2 hyperfine manifold the repump does not address holds the whole steady state
    with pytest.raises(ValueError, match="not a partition of the slow dynamics"):
        m.manifold_rates(BRIGHT, DARK)
    # folding the whole D branch into the dark manifold makes the partition complete, and the mode selection then
    # skips two slower intra-manifold modes before it finds the connecting one
    dark_wide = ("S1/2 F=0 mF=0",) + tuple(lab for lab in m.build.labels if lab.startswith("D3/2"))
    with pytest.raises(ValueError, match="ABSORBING"):
        m.manifold_rates(BRIGHT, dark_wide)
    # the selection itself: the second- and third-slowest modes carry no weight on the bright manifold at all
    liouvillian = np.asarray(m.build.liouvillian().full())
    vals, vecs = np.linalg.eig(liouvillian)
    order = np.argsort(-vals.real)
    n = m.build.n_internal
    p_bright = m.build.manifold_projector(BRIGHT).full()
    p_dark = m.build.manifold_projector(dark_wide).full()
    connecting = []
    for position in range(1, 6):
        v = vecs[:, order[position]].reshape(n, n, order="F")
        v = 0.5 * (v + v.conj().T)
        scale = float(np.sum(np.abs(np.diag(v))))
        a = abs(float(np.real(np.trace(p_bright @ v)))) / scale
        b_w = abs(float(np.real(np.trace(p_dark @ v)))) / scale
        connecting.append(a > 1e-9 and b_w > 1e-9)
    assert connecting[0] is False and connecting[1] is False  # order[1] and order[2] are intra-manifold
    assert True in connecting  # a connecting mode exists further down the spectrum
    # the closed S1/2 + P1/2 model (the D branch folded back) is unaffected: no mode is skipped there
    closed = detection_model(2.45, 4.7).manifold_rates(BRIGHT, DARK)
    assert closed.intra_manifold_rates_per_s == ()
    assert closed.rate_a_to_b_per_s > 0.0 and closed.rate_b_to_a_per_s > 0.0
    assert 0.0 < closed.weight_a < 1.0


def test_ceiling_is_asserted_not_assumed() -> None:
    m = detection_model(0.1, 1.0)
    rho_bad = qt.Qobj(np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0]), dims=m.build.H.dims)
    with pytest.raises(CeilingViolation):
        m.ceiling_report(rho_bad)


# ---- dark states (Berkeland Table I) ------------------------------------------------------------------------------------------


def test_dark_state_counts_follow_berkelands_table_i() -> None:
    """J_i = 1 <-> J_f = 0 (171Yb+ F = 1 -> F' = 0): two dark states at any polarization; the magic angle is 54.7356 deg."""
    m = detection_model(0.1, 1.0)
    ds = m.dark_states()
    assert (ds.n_ground, ds.n_excited, ds.ceiling, ds.dark_dimension) == (3, 1, 0.25, 2)
    assert ds.theta_be_deg == pytest.approx(math.degrees(math.atan(math.sqrt(2.0))), abs=1e-6)
    assert ds.raman_zero_margin_hz is None


def test_sigma_plus_light_leaves_span_of_m0_and_m_plus1_dark() -> None:
    """The Zeeman-degenerate dark-state condition in the straight pairing, never the printed reversed one."""
    st = AtomicStructure(YB, 1.0, (0.0, 0.0, 1.0))
    sp = (-1.0 / math.sqrt(2.0) + 0j, -1j / math.sqrt(2.0), 0j)
    beam = beam_for_transition(
        st, "S1/2 F=1 mF=0", "P1/2 F=0 mF=0", 0.0, (0.0, 0.0, 1.0), sp, power_w=1e-6, waist_m=WAIST
    )
    m = BlochModel(st, [beam], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))
    ground, _excited = m.resonant_manifold()
    assert ground == ("S1/2 F=1 mF=-1", "S1/2 F=1 mF=0", "S1/2 F=1 mF=1")
    ds = m.dark_states()
    assert ds.dark_dimension == 2
    weights = np.sum(np.abs(ds.dark_basis) ** 2, axis=0)
    assert weights == pytest.approx([0.0, 1.0, 1.0], abs=1e-12)


def test_f1_to_f1_linear_light_has_one_dark_state_and_f0_to_f1_none() -> None:
    st = AtomicStructure(YB, 1.0, (0.0, 0.0, 1.0))
    pump = beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=1 mF=0",
        0.0,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=1e-6,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    m = BlochModel(st, [pump], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))
    ds = m.dark_states()
    assert (ds.n_ground, ds.n_excited, ds.dark_dimension) == (3, 3, 1)
    repump = beam_for_transition(
        st,
        "S1/2 F=0 mF=0",
        "P1/2 F=1 mF=0",
        0.0,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=1e-6,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    m2 = BlochModel(st, [repump], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))
    ds2 = m2.dark_states()
    assert (ds2.n_ground, ds2.n_excited, ds2.dark_dimension) == (1, 3, 0)


# ---- optical pumping -------------------------------------------------------------------------------------------------


def test_optical_pumping_into_f0_takes_three_photons_and_leaves_a_small_residual() -> None:
    """1/3 branching into |0> per excitation: three scattered photons on average; the residual F = 1
    population is the off-resonant F = 0 -> F' = 1 excitation at 12.6 GHz, of order 1e-6 at s_o = 0.5."""
    st = AtomicStructure(YB, 5.0, (0.0, 0.0, 1.0))
    power = 0.5 * YB_LINE.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    pump = beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=1 mF=0",
        0.0,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=power,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    m = BlochModel(st, [pump], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))
    times = np.linspace(0.0, 30e-6, 6001)  # 5 ns steps resolve the coherent oscillations of the photon rate
    trace = m.evolve("S1/2 F=1 mF=0", times)
    p0 = trace.population(DARK)
    assert p0[0] == pytest.approx(0.0, abs=1e-12)
    assert 0.5 < float(np.interp(1e-6, times, p0)) < 0.95
    assert p0[-1] > 1.0 - 1e-5
    assert trace.photons_scattered[-1] == pytest.approx(3.0, abs=0.05)
    t99 = trace.time_to_reach(DARK, 0.99)
    assert t99 is not None and 1e-6 < t99 < 5e-6
    ss = m.steadystate()
    residual = sum(ss.populations[lab] for lab in BRIGHT)
    assert 1e-7 < residual < 1e-5
    # the trace is the exact exponential of the Liouvillian: probability is conserved
    total = sum(trace.populations.values())
    assert np.allclose(total, 1.0, atol=1e-10)


def test_pumping_matches_the_weak_drive_rate_equations() -> None:
    """With Omega << Gamma and the Zeeman splitting >> pumping rate the evolution reduces to rate equations built from the
    Lorentzian excitation rates and the branching ratios."""
    st = AtomicStructure(YB, 5.0, (0.0, 0.0, 1.0))
    power = 0.02 * YB_LINE.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    pump = beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=1 mF=0",
        0.0,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=power,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    m = BlochModel(st, [pump], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))
    b = m.build
    ground = [lab for lab in b.labels if lab.startswith("S1/2")]
    gamma = b.level_rates_rad_s["P1/2"]
    # rate matrix: excitation p -> e at W = Omega^2 Gamma/(Gamma^2 + 4 Delta^2), then decay e -> p' with the channel branching
    branching = np.zeros((len(ground), b.n_internal))
    for c in b.c_ops:
        mat = np.abs(np.asarray(c.full())) ** 2
        for i, p in enumerate(ground):
            branching[i] += mat[b.index(p)]
    branching /= gamma
    rate = np.zeros((len(ground), len(ground)))
    for c in b.couplings:
        i = ground.index(c.lower)
        w = abs(c.omega_rad_s) ** 2 * gamma / (gamma**2 + 4.0 * c.detuning_rad_s**2)
        for j, _pp in enumerate(ground):
            rate[j, i] += w * branching[j, b.index(c.upper)]
    gen = rate - np.diag(rate.sum(axis=0))
    times = np.linspace(0.0, 60e-6, 61)
    trace = m.evolve("S1/2 F=1 mF=0", times)
    p0_init = np.zeros(len(ground))
    p0_init[ground.index("S1/2 F=1 mF=0")] = 1.0
    from scipy.linalg import expm

    for t, target in zip(times[::10], trace.population(DARK)[::10]):
        p = expm(gen * t) @ p0_init
        assert p[ground.index("S1/2 F=0 mF=0")] == pytest.approx(target, abs=0.01)


# ---- frame graph and the Floquet fallback ----------------------------------------------------------------------------


def test_frame_assignment_flags_two_tones_on_one_transition_as_a_beat() -> None:
    st = AtomicStructure(YB, 1.0, (0.0, 0.0, 1.0))
    main = beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=0 mF=0",
        -0.5 * GAMMA_S,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=1e-6,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    side = beam_for_transition(
        st,
        "S1/2 F=0 mF=0",
        "P1/2 F=1 mF=0",
        -0.5 * GAMMA_S,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=1e-6,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    fa = assign_frames(st, [main], ["S1/2", "P1/2"])
    assert fa.static and fa.period_s is None and len(fa.edges) == 1 and fa.edges[0].sets_frame
    fa2 = assign_frames(st, [main, side], ["S1/2", "P1/2"])
    assert not fa2.static
    assert fa2.beats_rad_s[0] / TWO_PI == pytest.approx(14.7478e9, rel=1e-3)
    assert fa2.period_s == pytest.approx(TWO_PI / fa2.beats_rad_s[0])
    # a decay-only level takes its frame from the transition frequency
    fa3 = assign_frames(st, [main], ["S1/2", "P1/2", "D3/2"])
    assert fa3.static
    assert fa3.frame_rad_s["P1/2"] - fa3.frame_rad_s["D3/2"] == pytest.approx(
        TWO_PI * C_M_PER_S / YB.transition("D3/2-P1/2").wavelength_vac_m, rel=1e-12
    )


def _secular_reference(m: BlochModel, window_gammas: float = 10.0) -> qt.Qobj:
    """The static Hamiltonian each tone would have in its own hyperfine frame: only near-resonant couplings kept, every
    excited sublevel referred to the tone that drives it (the 14.7 GHz cross terms averaged away)."""
    b = m.build
    kept = [
        c
        for c in b.couplings
        if abs(c.detuning_rad_s) <= window_gammas * b.level_rates_rad_s[b.level_of(c.upper)]
    ]
    tone_of_upper = {c.upper: c.beam for c in kept}
    omega_beam = {k: TWO_PI * C_M_PER_S / beam.wavelength_m for k, beam in enumerate(m.beams)}
    n = b.n_internal
    h = np.zeros((n, n), dtype=complex)
    ref = TWO_PI * m.structure.state("S1/2 F=1 mF=0").energy_hz
    for lab in b.labels:
        e = TWO_PI * m.structure.state(lab).energy_hz
        if lab.startswith("P1/2"):
            e -= omega_beam[tone_of_upper[lab]] + ref
        else:
            e -= ref
        h[b.index(lab), b.index(lab)] = e
    for c in kept:
        h[b.index(c.upper), b.index(c.lower)] += 0.5 * c.omega_rad_s
        h[b.index(c.lower), b.index(c.upper)] += 0.5 * np.conj(c.omega_rad_s)
    return qt.Qobj(h, dims=b.H.dims)


def test_floquet_fixed_point_matches_the_secular_static_model() -> None:
    """Two tones on one transition (369 nm on F=1 -> F'=0 plus its 14.7 GHz sideband on F=0 -> F'=1) make the frame graph
    inconsistent; the period-propagator fixed point agrees with the secular static model to O(Omega/14.7 GHz) and repumps F=0."""
    st = AtomicStructure(YB, 1.0, (0.0, 0.0, 1.0))
    power = 0.1 * YB_LINE.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    main = beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=0 mF=0",
        -0.5 * GAMMA_S,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=power,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    side = beam_for_transition(
        st,
        "S1/2 F=0 mF=0",
        "P1/2 F=1 mF=0",
        -0.5 * GAMMA_S,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=0.3 * power,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    m = BlochModel(st, [main, side], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))
    assert not m.build.static and isinstance(m.build.H, qt.QobjEvo)
    assert any("time periodic" in a for a in m.build.approximations)
    ss = m.steadystate()
    assert ss.method == "floquet" and ss.period_s == pytest.approx(TWO_PI / m.build.frame.beats_rad_s[0])
    rho_sec = qt.steadystate(_secular_reference(m), list(m.build.c_ops))
    for lab in m.build.labels:
        assert ss.populations[lab] == pytest.approx(
            float(np.real(rho_sec[m.build.index(lab), m.build.index(lab)])), abs=5e-4
        )
    rate_sec = sum(v for k, v in m.photon_rates(rho_sec).items() if not k.startswith(SINK))
    assert ss.total_photon_rate_per_s == pytest.approx(rate_sec, rel=2e-3)
    assert sum(ss.populations[lab] for lab in BRIGHT) > 0.9


@pytest.mark.slow
def test_polarization_modulation_makes_the_liouvillian_periodic_and_still_solves() -> None:
    st = AtomicStructure(YB, 0.0, (0.0, 0.0, 1.0))
    power = 0.1 * YB_LINE.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    mod = PolarizationModulation("aom", 3.0e6, 0.0)
    beam = beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=0 mF=0",
        0.0,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=power,
        waist_m=WAIST,
        modulation=mod,  # type: ignore[arg-type]
    )
    m = BlochModel(st, [beam], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))
    assert not m.build.static
    ss = m.steadystate()
    assert ss.method == "floquet" and ss.period_s == pytest.approx(1.0 / 3.0e6)
    # at zero field the unmodulated beam is trapped in the dark states; the modulation restores fluorescence
    static = BlochModel(
        st,
        [
            shifted_beam(
                Beam(
                    beam.wavelength_m,
                    beam.k_hat,
                    beam.polarization,
                    beam.waist_m,
                    beam.power_w,
                    beam.pointing_m,
                ),
                0.0,
            )
        ],
        levels=("S1/2", "P1/2"),
        options=MultiLevelOptions(leak="renormalize"),
    )
    dr_static = static.manifold_rates(BRIGHT, DARK)
    assert m.photon_rates(ss.rho)["S1/2<-P1/2"] > 10.0 * sum(
        v for k, v in static.photon_rates(dr_static.conditional_a).items() if not k.startswith(SINK)
    )


# ---- leak policies --------------------------------------------------------------------------------------------------


def test_sink_policy_preserves_the_trace_and_measures_the_leak() -> None:
    """171Yb+ S1/2 + P1/2 with the 0.501% D3/2 branch routed to the sink: the sink fills at Gamma_D P_e."""
    st = AtomicStructure(YB, 1.0, (0.0, 0.0, 1.0))
    power = 0.1 * YB_LINE.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    beam = beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=0 mF=0",
        0.0,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=power,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    m = BlochModel(st, [beam], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="sink"))
    assert m.build.labels[-1] == SINK and decay_sum_rule_residual(m.build) < 1e-12
    sink_channel = next(ch for ch in m.build.channels if ch.kind == "sink")
    assert sink_channel.rate_rad_s == pytest.approx(YB.transition("D3/2-P1/2").partial_rate_rad_s, rel=1e-9)
    times = np.linspace(0.0, 200e-6, 41)
    trace = m.evolve("S1/2 F=1 mF=0", times)
    leak = trace.populations[SINK]
    assert leak[-1] > leak[10] > 0.0
    # the leak rate follows Gamma_D P_e; P_e is the bright-state excited fraction while the ion is bright
    gamma_d = YB.transition("D3/2-P1/2").partial_rate_rad_s
    p_e = trace.level_populations["P1/2"]
    dleak = np.gradient(leak, times)
    assert np.allclose(dleak[5:-5], gamma_d * p_e[5:-5], rtol=0.05, atol=1e-3 * gamma_d * p_e.max())


def test_include_policy_pulls_the_decay_target_in_and_the_ion_goes_dark_without_a_repump() -> None:
    """Without the 935 nm repump the 0.501% branch collects the ion in D3/2 within a few hundred scattered photons; the
    eight D3/2 sublevels make the steady state non-unique, so the statement is one about the time evolution.

    MultiLevelOptions.address_window is 0.5 of the transition frequency, so the 369.5 nm beam counts the 329 nm S1/2-P3/2
    line as addressed and leak = "include" pulls in P3/2's D5/2 decay target; P3/2 is 26 THz off resonance and changes no
    rate here."""
    st = AtomicStructure(YB, 1.0, (0.0, 0.0, 1.0))
    power = 0.1 * YB_LINE.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    beam = beam_for_transition(
        st,
        "S1/2 F=1 mF=0",
        "P1/2 F=0 mF=0",
        0.0,
        (1.0, 0.0, 0.0),
        MAGIC,
        power_w=power,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    m = BlochModel(st, [beam])
    assert m.build.levels == ("S1/2", "D3/2", "D5/2", "P1/2", "P3/2") and m.build.n_internal == 36
    gamma_d = YB.transition("D3/2-P1/2").partial_rate_rad_s
    trace = m.evolve("S1/2 F=1 mF=0", np.linspace(0.0, 2.0e-3, 41))
    p_d = trace.level_populations["D3/2"]
    assert p_d[-1] > 0.99
    assert trace.photon_rate_per_s[-1] < 1e-2 * trace.photon_rate_per_s[1]
    # the D3/2 filling rate is Gamma_D P_e while the ion is bright: about 1/0.005 = 200 photons before it goes dark
    photons_when_dark = float(np.interp(0.9, p_d, trace.photons_scattered))
    assert 150.0 < photons_when_dark < 800.0
    assert gamma_d / YB_LINE.gamma_rad_s == pytest.approx(0.00501, rel=1e-3)


# ---- 40Ca+ Lambda system -------------------------------------------------------------------------------------------


def test_ca40_dark_resonance_at_the_two_photon_resonance() -> None:
    """S1/2-P1/2-D3/2 with 397 and 866 beams polarized perpendicular to B at 4 G: fluorescence vanishes when the two
    detunings coincide (the S(-1/2)-D(-1/2) dark state through P(+1/2) is exact at any field) and is restored on either
    side, with the Zeeman-shifted two-photon resonances as further dips."""
    ca = species("40Ca+")
    st = AtomicStructure(ca, 4.0, (0.0, 0.0, 1.0))
    t397 = ca.transition("S1/2-P1/2")
    t866 = ca.transition("D3/2-P1/2")
    g = t397.gamma_rad_s
    pol = tuple(linear_polarization((1.0, 0.0, 0.0), math.pi / 2.0, (0.0, 0.0, 1.0)))

    def model(d866: float) -> BlochModel:
        b1 = beam_for_transition(
            st,
            "S1/2 mJ=-1/2",
            "P1/2 mJ=-1/2",
            -0.5 * g,
            (1.0, 0.0, 0.0),
            pol,
            power_w=0.3 * t397.i_sat_w_m2 * math.pi * WAIST**2 / 2.0,
            waist_m=WAIST,  # type: ignore[arg-type]
        )
        b2 = beam_for_transition(
            st,
            "D3/2 mJ=-1/2",
            "P1/2 mJ=-1/2",
            d866,
            (1.0, 0.0, 0.0),
            pol,
            power_w=3.0 * t866.i_sat_w_m2 * math.pi * WAIST**2 / 2.0,
            waist_m=WAIST,  # type: ignore[arg-type]
        )
        return BlochModel(st, [b1, b2], levels=("S1/2", "P1/2", "D3/2"))

    on = model(-0.5 * g).steadystate()
    off = model(-0.5 * g + 0.5 * g).steadystate()
    far = model(-0.5 * g - 1.5 * g).steadystate()
    assert on.total_photon_rate_per_s < 1e-6 * off.total_photon_rate_per_s
    assert off.total_photon_rate_per_s > far.total_photon_rate_per_s > 1e4
    assert on.ceiling.ceiling == 0.25
    assert (len(on.ceiling.ground_labels), len(on.ceiling.excited_labels)) == (6, 2)
    ds = model(-0.5 * g).dark_states()
    assert ds.raman_zero_margin_hz is not None and ds.raman_zero_margin_hz < 1.0
    assert ds.dark_dimension == 4
    assert any("S1/2-P3/2" in a and "excluded" in a for a in model(-0.5 * g).build.approximations)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="two trap states leave the Liouvillian a two-dimensional null space, so the direct steady state is not unique: the "
    "Linux runner's LAPACK returns populations of +-7e14 where this machine returns a density matrix (a solver robustness gap)",
)
def test_ca40_pi_only_repump_leaves_the_m_three_halves_states_as_traps() -> None:
    ca = species("40Ca+")
    st = AtomicStructure(ca, 1e-6, (0.0, 0.0, 1.0))
    t397 = ca.transition("S1/2-P1/2")
    t866 = ca.transition("D3/2-P1/2")
    g = t397.gamma_rad_s
    poly = tuple(linear_polarization((1.0, 0.0, 0.0), math.pi / 2.0, (0.0, 0.0, 1.0)))
    b1 = beam_for_transition(
        st,
        "S1/2 mJ=-1/2",
        "P1/2 mJ=-1/2",
        -0.5 * g,
        (1.0, 0.0, 0.0),
        poly,
        power_w=0.5 * t397.i_sat_w_m2 * math.pi * WAIST**2 / 2.0,
        waist_m=WAIST,  # type: ignore[arg-type]
    )
    b2 = beam_for_transition(
        st,
        "D3/2 mJ=-1/2",
        "P1/2 mJ=-1/2",
        0.0,
        (0.0, 1.0, 0.0),
        (0j, 0j, 1 + 0j),
        power_w=2.0 * t866.i_sat_w_m2 * math.pi * WAIST**2 / 2.0,
        waist_m=WAIST,
    )
    ss = BlochModel(st, [b1, b2], levels=("S1/2", "P1/2", "D3/2")).steadystate()
    assert ss.populations["D3/2 mJ=-3/2"] + ss.populations["D3/2 mJ=3/2"] > 0.999


# ---- laser linewidth on the optical coherences ----------------------------------------------------------------------
# One phase-diffusion operator per beam, C_b = sqrt(delta omega_L/4)(P_upper - P_lower) (Berkeland and Boshier, Phys. Rev. A
# 65, 033413 (2002) Eq. 12): the driven coherence is damped at delta omega_L/2 and no population moves, so the excitation
# profile convolves to the full width Gamma + delta omega_L at constant area.

SP = two_level_atom()
ST = structure(SP)
G = gamma_rad_s()


def _beam(detuning_rad_s: float, omega_rad_s: float = 0.01 * G) -> Beam:
    return sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, omega_rad_s, detuning_rad_s)


def _profile_fwhm_over_gamma(width_rad_s: float) -> tuple[float, float]:
    """(peak W, full width at half maximum in units of Gamma) of the weak-drive excitation profile, the half width by
    root-finding on the symmetric single-peaked profile."""
    options = MultiLevelOptions(laser_linewidth_rad_s=(width_rad_s,))

    def rate(detuning: float) -> float:
        return BlochModel(ST, [_beam(detuning)], options=options).scattering_rate_per_s()

    peak = rate(0.0)
    half = brentq(lambda d: rate(d) - 0.5 * peak, 0.0, 20.0 * G, xtol=1e-4 * G)
    return peak, 2.0 * half / G


def test_the_excitation_profile_broadens_to_gamma_plus_the_laser_linewidth() -> None:
    """W(Delta) is a Lorentzian of full width Gamma + delta omega_L at constant area: the peak falls as
    Gamma/(Gamma + delta omega_L)."""
    peak0, fwhm0 = _profile_fwhm_over_gamma(0.0)
    assert fwhm0 == pytest.approx(1.0, abs=0.02)
    for factor in (0.5, 1.0, 2.0):
        peak, fwhm = _profile_fwhm_over_gamma(factor * G)
        assert fwhm == pytest.approx(1.0 + factor, abs=0.02)
        assert peak / peak0 == pytest.approx(1.0 / (1.0 + factor), rel=2e-3)
    width = 0.7 * G
    options = MultiLevelOptions(laser_linewidth_rad_s=(width,))
    on_resonance = BlochModel(ST, [_beam(0.0)], options=options).scattering_rate_per_s()
    for offset in (0.3, 0.9, 2.0):
        delta = offset * (G + width)
        got = BlochModel(ST, [_beam(delta)], options=options).scattering_rate_per_s()
        assert got / on_resonance == pytest.approx(1.0 / (1.0 + (2.0 * offset) ** 2), rel=3e-3)


def test_a_zero_linewidth_changes_no_rate_and_adds_no_operator() -> None:
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.3 * G, -0.5 * G)
    plain = BlochModel(ST, [beam])
    zero = BlochModel(ST, [beam], options=MultiLevelOptions(laser_linewidth_rad_s=(0.0,)))
    assert zero.scattering_rate_per_s() == pytest.approx(plain.scattering_rate_per_s(), rel=1e-12)
    assert zero.build.dephasing_slice == (len(zero.build.c_ops), len(zero.build.c_ops))
    assert len(zero.build.c_ops) == len(plain.build.c_ops)
    assert not any("linewidth" in a for a in zero.build.approximations)
    finite = BlochModel(ST, [beam], options=MultiLevelOptions(laser_linewidth_rad_s=(0.3 * G,)))
    start, stop = finite.build.dephasing_slice
    assert stop - start == 1 and stop == len(finite.build.c_ops)
    assert any("laser linewidth" in a for a in finite.build.approximations)
    assert finite.scattering_rate_per_s() < plain.scattering_rate_per_s()


def test_the_phase_diffusion_operator_touches_no_population_and_leaves_the_decay_sum_rule_alone() -> None:
    """C_b is diagonal, carries no photon and no recoil, and the decay sum rule sum_k C_k^dagger C_k = Gamma P_e
    excludes it."""
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.3 * G, 0.0)
    plain = BlochModel(ST, [beam])
    finite = BlochModel(ST, [beam], options=MultiLevelOptions(laser_linewidth_rad_s=(0.4 * G,)))
    assert decay_sum_rule_residual(finite.build) < 1e-12
    assert decay_sum_rule_residual(finite.build) == pytest.approx(
        decay_sum_rule_residual(plain.build), abs=1e-12
    )
    start, stop = finite.build.dephasing_slice
    matrix = np.asarray(finite.build.c_ops[start].full())
    assert np.allclose(matrix, np.diag(np.diag(matrix)))
    values = np.real(np.diag(matrix))
    assert np.max(values) == pytest.approx(np.sqrt(0.4 * G / 4.0), rel=1e-9)
    assert np.min(values) == pytest.approx(-np.sqrt(0.4 * G / 4.0), rel=1e-9)
    for channel in finite.build.channels:
        assert not channel.operator_slice[0] <= start < channel.operator_slice[1]
    assert sum(finite.photon_rates(finite.steadystate().rho).values()) == pytest.approx(
        finite.steadystate().total_photon_rate_per_s, rel=1e-12
    )


def test_the_optical_coherence_decays_at_gamma_over_two_plus_half_the_linewidth() -> None:
    """With the drive off, rho_eg decays at gamma/2 + delta omega_L/2 while the excited population decays at gamma."""
    width = 0.6 * G
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.0, 0.0)
    build = BlochModel(ST, [beam], options=MultiLevelOptions(laser_linewidth_rad_s=(width,))).build
    lower = build.index(TWO_LEVEL_GROUND)
    upper = build.index(TWO_LEVEL_EXCITED_PLUS)
    n = build.n_internal
    rho0 = np.zeros((n, n), dtype=complex)
    rho0[upper, upper] = 0.5
    rho0[lower, lower] = 0.5
    rho0[upper, lower] = 0.5
    rho0[lower, upper] = 0.5
    times = np.linspace(0.0, 3.0 / G, 61)
    res = qt.mesolve(
        build.H,
        qt.Qobj(rho0, dims=build.H.dims),
        times,
        c_ops=list(build.c_ops),
        options={"store_states": True, "progress_bar": "", "atol": 1e-13, "rtol": 1e-11},
    )
    coherence = np.array([abs(np.asarray(s.full())[upper, lower]) for s in res.states])
    population = np.array([np.real(np.asarray(s.full())[upper, upper]) for s in res.states])
    assert np.max(np.abs(coherence - 0.5 * np.exp(-(0.5 * G + 0.5 * width) * times))) < 1e-9
    assert np.max(np.abs(population - 0.5 * np.exp(-G * times))) < 1e-9


def test_one_linewidth_per_beam_is_required_and_non_negative() -> None:
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.1 * G, 0.0)
    with pytest.raises(ValueError, match="one entry per beam"):
        BlochModel(ST, [beam], options=MultiLevelOptions(laser_linewidth_rad_s=(1e3, 1e3)))
    with pytest.raises(ValueError, match="non-negative"):
        MultiLevelOptions(laser_linewidth_rad_s=(-1.0,))
