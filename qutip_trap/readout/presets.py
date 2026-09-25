"""Published readout operating points, as APPARATUS data (PLAN.md Sections 8.4, 8.8, 9.5; milestone M5).

Every number here is what a laboratory measured about its own optics, detectors and beams: detected rates, efficiencies,
backgrounds, windows and thresholds. They are validation anchors for the record and discriminator layers and calibration
inputs with provenance, never species constants (Section 8.8: the first version of the plan carried a fitted species scale
here, which contradicted Section 3.1). The species constants they combine with (linewidths, splittings, lifetimes) live in
the species tables and the atomic layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from qutip_trap.readout.fluorescence import FluorescenceRates, rates_from_detected


@dataclass(frozen=True)
class ApparatusPreset:
    """One published readout apparatus (Section 8.4 "Species presets")."""

    name: str
    species: str
    source: str
    detected_bright_per_s: float | None
    """The measured detected count rate of a bright ion (R_B, eps_sys R_o); None when the source does not quote one, in
    which case :meth:`rates` refuses rather than inventing a zero-scattering bright state."""
    efficiency: float
    """The quoted system detection efficiency epsilon_sys."""
    background_per_s: float
    """Detected background (dark counts plus scattered light), the R_D or R_bg of the source."""
    dark_pumping_per_s: float = 0.0
    bright_pumping_per_s: float = 0.0
    shelf_lifetime_s: float | None = None
    window_s: float | None = None
    threshold: float | None = None
    quoted_error: float | None = None
    """The state-averaged error the source quotes for its threshold or protocol at the window."""
    notes: str = ""

    @property
    def has_detected_rate(self) -> bool:
        """Whether the source quotes a detected bright rate at all (three of the seven presets do not)."""
        return self.detected_bright_per_s is not None

    def rates(self) -> FluorescenceRates:
        """The Section 8.1 rate object with the efficiency divided out once (the record layer applies it once again)."""
        if self.detected_bright_per_s is None:
            raise ValueError(
                f"{self.source} does not quote a detected bright rate ({self.notes.split(';')[-1].strip()}), so this "
                "preset has no rate object: use its efficiency, background and budget entries, or supply a rate from the "
                "Bloch solve of the apparatus's own intensity (Section 8.8)"
            )
        return rates_from_detected(
            self.detected_bright_per_s,
            self.efficiency,
            dark_pumping_per_s=self.dark_pumping_per_s,
            bright_pumping_per_s=self.bright_pumping_per_s,
            shelf_lifetime_s=self.shelf_lifetime_s,
            provenance=(self.source,),
        )


MYERSON_CA40_PMT = ApparatusPreset(
    name="Myerson 2008, 40Ca+ optical qubit, PMT",
    species="40Ca+",
    source="Myerson2008",
    detected_bright_per_s=55_800.0,
    efficiency=0.0019,
    background_per_s=442.0,
    shelf_lifetime_s=1.168,
    window_s=420e-6,
    threshold=5.5,
    quoted_error=1.8e-4,
    notes=(
        "R_B = 55800 s^-1, R_D = 442 s^-1 (PMT dark counts 8.2 s^-1 included), net efficiency 0.19(2) %; threshold optimum "
        "n_c = 5.5 at t_b = 420 us gives 1.8(1)e-4; maximum likelihood 0.87(11)e-4 asymptote (eps_D = 1.5(2)e-4), the "
        "Poisson simulation's asymptote 0.89e-4; adaptive 1.0(1)e-4 in 145 us average (72 us bright, 219 us dark); "
        "sub-bins t_s = 10 us; bright -> dark transfer < 1e-3 s^-1"
    ),
)
"""Myerson et al. 2008's 40Ca+ shelving detection through a PMT (Section 8.4): the collection and quantum efficiencies, the
dark counts and the detection window of the published apparatus, as an ``ApparatusPreset``."""

HARTY_CA43 = ApparatusPreset(
    name="Harty 2014, 43Ca+ hyperfine qubit shelved to D5/2, PMT",
    species="43Ca+",
    source="Harty2014",
    detected_bright_per_s=50_000.0,
    efficiency=0.003,
    background_per_s=0.0,
    shelf_lifetime_s=1.168,
    quoted_error=6.8e-4,
    notes=(
        "50 000 s^-1 detected at 0.3 % net efficiency; Table I: stretch-state preparation < 1e-4, transfer to qubit 1.8e-4, "
        "transfer from qubit 1.8e-4, shelving transfer 1.7e-4, time-resolved fluorescence detection 1.5e-4, measured SPAM "
        "6.8(5)e-4; B_0 = 146.094 G"
    ),
)

NOEK_YB171_PMT = ApparatusPreset(
    name="Noek 2013, 171Yb+ hyperfine qubit, 0.6 NA + PMT",
    species="171Yb+",
    source="Noek2013",
    detected_bright_per_s=None,
    efficiency=0.022,
    background_per_s=6.5,
    notes=(
        "eps = 2.2(1) % from 174Yb+; R_dc = 6.5 Hz PMT dark counts + 35 Hz per uW of detection power; Zeeman splitting "
        "fixed at 4.8 MHz; fidelities 99 % / 99.85(1) % / 99.915(7) % at 10.5 / 28.1 / 99.8 us average (worst case 17.0 / "
        "51.4 / 181.6 us; 36 and 8 mW/cm^2); R_d of order 1.5 kHz near 170 mW/cm^2 (Fig. 2b); the detected rate is "
        "intensity dependent and is not a single number in the source, hence 0 here"
    ),
)

CRAIN_YB171_SNSPD = ApparatusPreset(
    name="Crain 2019, 171Yb+ hyperfine qubit, 0.6 NA + SNSPD",
    species="171Yb+",
    source="Crain2019",
    detected_bright_per_s=472_000.0,
    efficiency=0.04356,
    background_per_s=4.2,
    dark_pumping_per_s=341.0,
    bright_pumping_per_s=16.4,
    window_s=500e-6,
    threshold=0.5,
    quoted_error=6.9e-4,
    notes=(
        "eps_sys R_o = 472(14) kcps at 56.2 mW/cm^2, R_d = 341(13) Hz, R_b = 16.4(5) Hz, R_bg = 4.2(1) cps "
        "(0.075 cps cm^2/mW), eps_sys = 4.356(6) % quoted, against 4.724 % for the printed chain "
        "10 % x 81.8 % x 73.1 % x 79 % (the residual 8 % is the apparatus discrepancy Section 8.8 anticipates: an "
        "intensity or transmission factor the paper rounds, not a missing physical factor); stop-on-first-photon: 99.931(6) % at "
        "11 us average, record window 500 us; zero-background limit 99.941 %; spectator coherence alpha = 1716 ms baseline, "
        "94(5) ms at 200 um, 814(77) ms at 370 um (Gaussian fringe decay exp(-tau^2/alpha^2))"
    ),
)
"""Crain et al. 2019's 171Yb+ state detection with a superconducting nanowire detector (Section 8.4): the published
apparatus as an ``ApparatusPreset``, the detector the example 171Yb+ machine carries."""

CHRISTENSEN_BA133 = ApparatusPreset(
    name="Christensen 2020, 133Ba+ hyperfine qubit shelved through P3/2",
    species="133Ba+",
    source="Christensen2020",
    detected_bright_per_s=39.0 / 4.5e-3,
    efficiency=0.01,
    background_per_s=1.0 / 4.5e-3,
    shelf_lifetime_s=30.0,
    window_s=4.5e-3,
    threshold=12.5,
    quoted_error=2.9e-4,
    notes=(
        "bright mean 39 counts and dark mean 1 count in 4.5 ms (0.28 NA), threshold n <= 12 dark; branching 0.74/0.23/0.03 "
        "to S1/2/D5/2/D3/2; Table I: initialization 0.1, CP Robust 180 0.5, spontaneous decay during readout 0.7 "
        "(1 - exp(-4.5e-3/30) = 1.5e-4 per |1> shot), shelving |1> 1.0, off-resonant shelving |0> 1.0, readout of the "
        "S1/2 manifold 0.1, total 3.4e-4 against measured 2.9(6)e-4 (eps_|0> = 1.9(4)e-4, eps_|1> = 3.8(5)e-4); the "
        "efficiency is not quoted and 1 % is a placeholder consistent with the count rate"
    ),
)

BURRELL_CA40_CAMERA = ApparatusPreset(
    name="Burrell 2010, 40Ca+ optical qubit, EMCCD",
    species="40Ca+",
    source="Burrell2010",
    detected_bright_per_s=None,
    efficiency=0.010,
    background_per_s=0.0,
    shelf_lifetime_s=1.168,
    window_s=400e-6,
    quoted_error=0.9e-4,
    notes=(
        "NA 0.25, net efficiency 1.0(1) %, camera QE 48 % at 397 nm with excess noise factor sqrt 2, 2.6 um/pixel, 50 x 10 "
        "pixel acquisition area; single 400 us exposure: threshold 0.9(3)e-4 (eps_B = 0) at N = 28 pixels, ML 1.0(4)e-4 at "
        "N = 10, adaptive N = 2.9; 18 exposures of 200 us: spatio-temporal ML 1.1(4)e-4 (no gain); four ions at 14 um: "
        "nearest-neighbour signal 4.0 %, next-nearest 0.9 %; per-qubit crosstalk error 6.8(8)e-4 threshold, 0.7(3)e-4 "
        "spatial ML, 0.1(1)e-4 neighbour-conditioned ML; the detected rate is not quoted in the source"
    ),
)

EGAN_YB171 = ApparatusPreset(
    name="Egan 2021, 171Yb+ hyperfine qubits, 100 us window",
    species="171Yb+",
    source="Egan2021",
    detected_bright_per_s=None,
    efficiency=0.01,
    background_per_s=7.0,
    window_s=100e-6,
    quoted_error=0.5 * (0.0071 + 0.0022),
    notes=(
        "single-ion SPAM budget (PLAN.md Section 6.7): bright-state error 0.71(4) % dominated by bright-to-dark pumping "
        "0.55 % in a 100 us window; dark-state error 0.22(2) % from dark-to-bright pumping 0.13 % and background 0.07 %; "
        "the two SPAM figures 99.3 % (randomized-benchmarking fits) and 99.80 % (one-sided microwave measurement) "
        "coexist in the paper; the arXiv text quotes a single-qubit Z-basis SPAM error of 0.46(2) %; the rate and "
        "efficiency entries are placeholders (not quoted)"
    ),
)

PRESETS: dict[str, ApparatusPreset] = {
    p.name: p
    for p in (
        MYERSON_CA40_PMT,
        HARTY_CA43,
        NOEK_YB171_PMT,
        CRAIN_YB171_SNSPD,
        CHRISTENSEN_BA133,
        BURRELL_CA40_CAMERA,
        EGAN_YB171,
    )
}
