"""Published readout apparatus, as apparatus data (PLAN.md Section 8.4): the detected rates, efficiencies and backgrounds a
laboratory measured about its own optics and detectors, never species constants."""

from __future__ import annotations

from dataclasses import dataclass

from qutip_trap.readout.fluorescence import FluorescenceRates, rates_from_detected


@dataclass(frozen=True)
class ApparatusPreset:
    """One published readout apparatus."""

    name: str
    species: str
    source: str
    detected_bright_per_s: float
    """The measured detected count rate of a bright ion, eps_sys R_o."""
    efficiency: float
    """The quoted system detection efficiency epsilon_sys."""
    background_per_s: float
    """The detected background (dark counts plus scattered light)."""
    dark_pumping_per_s: float = 0.0
    bright_pumping_per_s: float = 0.0
    shelf_lifetime_s: float | None = None

    def rates(self) -> FluorescenceRates:
        """The rate object with the efficiency divided out once (the record layer applies it once again)."""
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
)
"""Myerson et al. 2008's 40Ca+ shelving detection through a PMT: R_B = 55800 s^-1, R_D = 442 s^-1 (8.2 s^-1 PMT dark counts
included), net efficiency 0.19 %, shelf lifetime 1168 ms; the paper's threshold n_c = 5.5 at t_b = 420 us gives 1.8e-4."""

CRAIN_YB171_SNSPD = ApparatusPreset(
    name="Crain 2019, 171Yb+ hyperfine qubit, 0.6 NA + SNSPD",
    species="171Yb+",
    source="Crain2019",
    detected_bright_per_s=472_000.0,
    efficiency=0.04356,
    background_per_s=4.2,
    dark_pumping_per_s=341.0,
    bright_pumping_per_s=16.4,
)
"""Crain et al. 2019's 171Yb+ detection with a superconducting nanowire detector: eps_sys R_o = 472 kcps at 56.2 mW/cm^2,
R_d = 341 Hz, R_b = 16.4 Hz, R_bg = 4.2 cps, eps_sys = 4.356 %; stop-on-first-photon reads 99.931 % at 11 us average."""
