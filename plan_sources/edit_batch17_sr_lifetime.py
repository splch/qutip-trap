"""Edit batch 17 (2026-09-04): the modern 88Sr+ D5/2 lifetime (Letchumanan 2005 via thesis, NIST, Jiang 2009)."""

from pathlib import Path

from edit_batch13_errata import apply

HERE = Path(__file__).resolve().parent

EDITS = [
    (
        "p04e_atomic.md",
        "for 88Sr+ the only lifetime in the source set is the 1987 value 345(33) ms, giving 14.70 a₀², and a modern measured lifetime must be pinned before any 674 nm result is trusted (Section 12).",
        "for 88Sr+ the modern measured lifetime is 390.8(1.6) ms (Letchumanan, Wilson, Gill and Sinclair, Phys. Rev. A 72, 012509 (2005), read on 2026-09-04 as Chapter 5 of Letchumanan's 2004 Imperial College thesis because the paper is paywalled, quoted as 0.3908(16) s by Jiang et al. 2009 and adopted by the NIST Sr II compilation as A = 2.559(10) s⁻¹ at accuracy AA), giving 13.81(3) a₀², 6.06% below the 14.70 a₀² that the superseded 1987 ion-cloud value 345(33) ms of the first two revisions gave, so every 674 nm Rabi frequency falls by 6.06% and every π time lengthens by 6.45% (`check_sr_lifetime_quad.py`) **[verified]**, **[corrected]**, **[recomputed here]**. The measurement is the most direct of five (Gerz 1987, 345(33) ms from a pressure-extrapolated ion cloud; Madej and Sankey 1990, 372(25); Barwood 1993, 347(11) from 5120 decays; Biémont 2000, 408(22) from a storage ring with 10% loss corrections; Letchumanan 2005 from 160,000 single-ion shelving periods); its applied corrections total 0.11% of the rate (collisional deshelving 2.2(2.2) × 10⁻³ s⁻¹ from 8 observed collisions per hour and off-resonant 1033 nm quenching by the 1092 nm repumper 6(6) × 10⁻⁴ s⁻¹) and its residual error is statistical; the thesis prints 390.3 ms because its Table 5.2 repeats an uncorrected July-2002 rate, and the weighted mean of the corrected rates is the published 390.8 ms **[verified]**. Its blackbody correction is not a correction: the only blackbody deshelving channel is D₅/₂ → P₃/₂ at 1032.7 nm, whose rate (1 − b)[(2J_P + 1)/(2J_D + 1)]A_PD/(e^{ħω/k_BT} − 1) is 4 × 10⁻¹⁴ s⁻¹ at 300 K against the 2.559 s⁻¹ decay rate, eleven orders of magnitude below the measurement's uncertainty, and reaches the decay rate only near 1000 K, which is why a filament or an operating ion gauge in view of the ion matters (`check_sr_lifetime_bbr.py`); the blackbody frequency shift, a leading clock systematic, is a different quantity **[verified]**, **[recomputed here]**. Unit branching into S₁/₂ holds to 10⁻⁴ for Sr+, not 10⁻⁵: the M1 channel D₅/₂ → D₃/₂ is estimated at 2.4 × 10⁻⁴ s⁻¹, a 9 × 10⁻⁵ branch, 98 times Ca+'s because the D fine-structure splitting is 4.6 times larger and M1 rates scale as σ³, so the D₅/₂ collapse operators carry a D₃/₂ channel at that weight **[background: LS-coupling estimate, no source]**. Theory values bracket the measurement (Poirier 1993, 396 ms; Sahoo 2006, 357(12) ms; Jiang 2009, 394(3) ms); Nichol et al. 2022 quote about 400 ms citing Sahoo, a number that matches neither, and it must not be propagated. The NIST clock frequency 444 779 044 095 484.6 Hz gives λ_vac = 674.025591 nm, the plan's value to 1.3 ppb, while NIST's air Ritz wavelength 673.8392 nm for the same line is the 276 ppm trap of the previous paragraph **[verified]**.",
    ),
    (
        "p04e_atomic.md",
        "equal to 1/τ only at unit branching (true to 10⁻⁵ for the Ca+ and Sr+ D₅/₂ levels, false for Ba+ D₅/₂ or any P₁/₂ level with a D branch).",
        "equal to 1/τ only at unit branching (true to 10⁻⁵ for the Ca+ D₅/₂ level and to about 10⁻⁴ for Sr+, whose M1 branch to D₃/₂ is estimated below; false for Ba+ D₅/₂ or any P₁/₂ level with a D branch).",
    ),
    (
        "p04e_atomic.md",
        "88Sr+ (674.02559 nm, 0.345 s pending a modern value, 1.0, 1/2, 5/2), from which the module derives k, A, the reduced element (asserting 9.73 a₀² and 14.70 a₀²),",
        "88Sr+ (674.02559 nm, 0.3908 s, 1.0 less the 9 × 10⁻⁵ M1 branch, 1/2, 5/2), from which the module derives k, A, the reduced element (asserting 9.73 a₀² and 13.81 a₀², with 14.70 a₀² from the superseded 0.345 s as the negative control),",
    ),
    (
        "p09_validation.md",
        "88Sr+: 14.70 a₀² with τ = 0.345 s | James Eqs. 5.6-5.10; Roos Eq. 3.27 [corrected] |",
        "88Sr+: 13.81(3) a₀² with τ = 0.3908(16) s (Letchumanan 2005), A = 2.558854 s⁻¹, A/2π = 0.4073 Hz, and 14.70 a₀² with the superseded 0.345 s as the negative control; λ_vac from the NIST clock frequency 444 779 044 095 484.6 Hz is 674.025591 nm, the plan's value to 1.3 ppb, while NIST's air Ritz value 673.8392 nm is the 276 ppm trap; blackbody deshelving 3.7 × 10⁻¹⁴ s⁻¹ at 300 K (`check_sr_lifetime_quad.py`, `check_sr_lifetime_bbr.py`) | James Eqs. 5.6-5.10; Roos Eq. 3.27 [corrected]; Letchumanan 2005 through the thesis, Jiang 2009 and the NIST Sr II compilation [verified]; [recomputed here] |",
    ),
    (
        "p12_risks.md",
        "The quadrupole run (Section 4.5.7) leaves two named gaps: the only 88Sr+ D₅/₂ lifetime in the source set is the 1987 value 345(33) ms, so every 674 nm Rabi frequency is provisional until a modern measured lifetime (with its blackbody correction) is pinned, and the ac Stark shift",
        "The quadrupole run (Section 4.5.7) left two named gaps, one of which closed on 2026-09-04 when the 88Sr+ D₅/₂ lifetime was pinned at 390.8(1.6) ms with a negligible blackbody correction (Section 4.5.7; the 674 nm Rabi frequencies fell 6.06% against the superseded 1987 value); the remaining one is that the ac Stark shift",
    ),
]

SOURCES = (
    "\n\n**Metastable lifetimes (R5, single-agent follow-up of 2026-09-04)**\n"
    "- Letchumanan, Wilson, Gill, Sinclair, Lifetime measurement of the metastable 4d 2D5/2 state in 88Sr+ using a single trapped ion, Phys. Rev. A 72, 012509 (2005); read as Chapter 5 of V. Letchumanan, Coherent control and ground state cooling of a single 88Sr+ ion, PhD thesis, Imperial College London and NPL (2004), openly hosted by Imperial's ion-trapping group (the published paper is paywalled and has no preprint). R5.\n"
    "- Jiang, Arora, Safronova, Clark, Blackbody-radiation shift in a 88Sr+ ion optical frequency standard, J. Phys. B 42, 154020 (2009); arXiv:0904.2107 (Table 5 quotes 0.3908(16) s with the Letchumanan citation and the theory values). R5.\n"
    "- Sansonetti, Wavelengths, transition probabilities, and energy levels for the spectra of strontium ions, J. Phys. Chem. Ref. Data 41, 013102 (2012) (Tables 2-3: A = 2.559(10) s⁻¹, lifetime 390.8(16), the column header reading seconds where the entries are milliseconds; the 88Sr II clock frequency). R5.\n"
)

APPENDIX_D = " The one unavailable source was reached afterwards by a single agent through the first author's openly hosted PhD thesis (Letchumanan 2004, Chapter 5, which is the published measurement), corroborated by the NIST Sr II compilation and Jiang et al. 2009 and cross-checked against the thesis's own decay-rate table; Section 4.5.7 carries the result and `check_sr_lifetime_quad.py` and `check_sr_lifetime_bbr.py` the numbers."

if __name__ == "__main__":
    apply(EDITS)
    p = HERE / "p15_sources.md"
    t = p.read_text(encoding="utf-8")
    if "Metastable lifetimes (R5" not in t:
        p.write_text(t.rstrip("\n") + SOURCES, encoding="utf-8")
        print("appended lifetime sources to p15_sources.md")
    p = HERE / "p16_runs.md"
    t = p.read_text(encoding="utf-8")
    anchor = "`check_comb.py`, `check_composite.py`, `check_constants.py`, `check_transport.py`)."
    if APPENDIX_D not in t:
        assert t.count(anchor) == 1, t.count(anchor)
        p.write_text(t.replace(anchor, anchor + APPENDIX_D), encoding="utf-8")
        print("appended Sr+ sentence to the Run 5S paragraph")
