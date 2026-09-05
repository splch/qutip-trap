"""Edit batch 13 (2026-09-04): fold the errata-hunt results (wf_875da9f3-823) into the plan sources.

Idempotent exact-match edits: each (file, old, new) must match exactly once, or already be applied.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

EDITS = [
    (
        "p04a_trap.md",
        "Turchette et al. 2000 write the same physics with a two-sided density; their printed Eq. (3), Γ = q² S_E/(m ħ ω), does not follow from their Eq. (2), which yields q² S_E^{(2)}/(2 m ħ ω) **[corrected]**; with S_E^{(1)} = 2 S_E^{(2)} the two conventions agree at e² S_E^{(1)}/(4 m ħ ω).",
        "Turchette et al. 2000 write the same physics; the arXiv preprint (quant-ph/0002040) prints Eq. (3) as Γ = q² S_E/(m ħ ω), which does not follow from its Eq. (2), whereas the published Phys. Rev. A version prints q² S_E(ω_m)/(4 m ħ ω_m) with an explicitly single-sided S_E = 2∫dτ e^{iωτ}⟨ε(τ)ε(0)⟩ and is internally consistent (the errata check of 2026-09-04, Appendix D, re-derived its Eq. 3 from its Eq. 2 with zero residual and matched it to the first term of its Eq. 4), so the defect the research runs found is the preprint's and the version of record agrees with Brownnutt; with a two-sided density S_E^{(2)} = S_E^{(1)}/2 the same rate reads q² S_E^{(2)}/(2 m ħ ω) **[corrected: preprint only]**.",
    ),
    (
        "p04a_trap.md",
        "and Turchette's written S_E definition is two-sided while the 1/4 in its Eq. 4 is the single-sided form **[corrected]**.",
        "and the published Turchette S_E definition is single-sided, consistent with the 1/4 in its Eq. 4; the two-sided reading that the first revision of this plan attributed to the paper is the preprint's (errata check of 2026-09-04) **[corrected: preprint only]**.",
    ),
    (
        "p13_conventions.md",
        "Two-sided gives e²S_E/(2mħω); Turchette Eq. 3 inconsistent by 2 [corrected] |",
        "Two-sided gives e²S_E/(2mħω); the arXiv preprint of Turchette prints Eq. 3 inconsistent by 2, the published PRA version is consistent [corrected: preprint only] |",
    ),
    (
        "p13_conventions.md",
        "| Wineland Eq. 62 (typo fixed); Brownnutt Eqs. 15-17 | printed 2aρ†a† [corrected] |",
        "| Wineland Eq. 62 (typo fixed); Brownnutt Eqs. 15-17 | the journal prints 2aρ†a†, a typesetting artifact absent from both arXiv versions of the paper [corrected] |",
    ),
    (
        "p04a_trap.md",
        "(Wineland 1998 Eq. 62 with its printed typo 2aρ†a† corrected to 2aρa† **[corrected]**;",
        "(Wineland 1998 Eq. 62 with the journal's printed typo 2aρ†a† corrected to 2aρa†, the form both arXiv versions of the paper print, so the correction is the authors' own text (errata check of 2026-09-04) **[corrected]**;",
    ),
    (
        "p04e_atomic.md",
        "the Yb+ row does not, 0.29 × 10⁻⁴ and 2.9 mW against 0.2 and 2, and is excluded)",
        "the Yb+ row does not: it prints 0.2 × 10⁻⁴ and 2 mW where the paper's own Eq. 17 gives 0.29 × 10⁻⁴ and 2.9 mW, and it is excluded; the first revision of this plan had the printed and recomputed pairs transposed, caught by the errata check of 2026-09-04)",
    ),
    (
        "p09_validation.md",
        "the Yb+ row (0.29 × 10⁻⁴, 2.9 mW against 0.2, 2) is excluded",
        "the Yb+ row (printed 0.2 × 10⁻⁴ and 2 mW against 0.29 × 10⁻⁴ and 2.9 mW recomputed from Eq. 17) is excluded",
    ),
    (
        "p04c_hamiltonian.md",
        "Lee et al. 2005's printed ac Stark shift χ_{m,i} has denominator 2 where 2Δ is required, and is dimensionally a frequency squared as printed **[corrected]**.",
        "the arXiv v1 preprint of Lee et al. 2005 (quant-ph/0505203v1) prints the ac Stark shift χ_{m,i} with denominator 2 where 2Δ is required, dimensionally a frequency squared, and swaps the beam-to-level assignment in its Eq. 2, whereas the published J. Opt. B version prints both correctly; the errata check of 2026-09-04 traced the first revision's reading to the preprint (pdftotext also drops the Δ glyph from the journal PDF, which is the likely route in) **[corrected: preprint only]**.",
    ),
    (
        "p15_sources.md",
        "- Zhu, Monroe, Duan, Phys. Rev. Lett. 97, 050505 (2006); arXiv:quant-ph/0601185. R2.",
        "- Zhu, Monroe, Duan, Trapped ion quantum computation with transverse phonon modes, Phys. Rev. Lett. 97, 050505 (2006); arXiv:quant-ph/0601159 (the first revision of this plan cited quant-ph/0601185, a different paper; corrected by the errata check of 2026-09-04). R2.",
    ),
    (
        "p15_sources.md",
        "- Chen et al. (IonQ), Benchmarking a trapped-ion quantum computer with 30 qubits, arXiv:2308.05071 (2023). R2.",
        "- Chen et al. (IonQ), Benchmarking a trapped-ion quantum computer with 30 qubits, Quantum 8, 1516 (2024); arXiv:2308.05071 (v2 is the version of record; the 2023 v1 has no Appendix A and no Eq. 3). R2.",
    ),
    (
        "p04e_atomic.md",
        "its printed node and antinode positions are each a factor 2 too large (nodes at lλ/2, antinodes at (2l − 1)λ/4) **[corrected]**; it is not ported into the travelling-wave builder.",
        "its printed node and antinode positions in Sec. 4 (Eqs. 4.9 and 4.13; Sec. 5 and the Appendix contain none) are each a factor 2 too large (nodes at lλ/2, antinodes at (2l − 1)λ/4; the printed antinode (2l − 1)λ/2 is a node of the paper's own sin(kζ) field) **[corrected]**; it is not ported into the travelling-wave builder.",
    ),
    (
        "p12_risks.md",
        "and no published erratum could be checked for any of the roughly twenty printed source errors this plan corrects (publisher pages were unreachable during the run), so a later correction cannot be excluded for any of them.",
        "and the errata check of 2026-09-04 (Appendix D) found no published erratum, corrigendum or publisher's note for any of the twenty-five printed source errors it tested, so every one of them stands on this plan's recomputation: corroborated where a preprint exists (Wineland 1998's 2aρa† is printed correctly in both arXiv versions), and downgraded in two cases where the error turned out to be the preprint's alone while the journal version is correct (Turchette 2000, Lee 2005), with the Brownnutt Eq. 24 and ladder-operator sub-claims still resting on a single preprint because the published review was unreadable; APS abstract pages refused automated access, so the publishers' own erratum panels were not read and the negatives rest on Crossref, INSPIRE, OpenAlex, PubMed and the QIC index, each of which surfaced real errata for other papers in the same check.",
    ),
    (
        "p04a_trap.md",
        "No published erratum for House, Wesenberg, Nizamani or Madsen was found (publisher pages were unreachable during the run), so later corrections cannot be excluded.",
        "No published erratum for House, Wesenberg, Nizamani or Madsen was found in Run 4 or in the errata check of 2026-09-04 (Crossref, INSPIRE and OpenAlex; APS pages refused automated access), Madsen's 2400 against 2694 inconsistency was re-confirmed at its source, and the House Eq. 7 sign could be checked only against a mirrored PDF, so it carries medium confidence and later corrections cannot be excluded.",
    ),
]


def apply(edits):
    n_applied = n_already = 0
    for fn, old, new in edits:
        p = HERE / fn
        text = p.read_text(encoding="utf-8")
        if new in text and old not in text:
            n_already += 1
            continue
        c = text.count(old)
        if c != 1:
            sys.exit(f"{fn}: expected exactly one match, found {c} for: {old[:90]}...")
        p.write_text(text.replace(old, new), encoding="utf-8")
        n_applied += 1
    print(f"applied {n_applied}, already applied {n_already}, total {len(edits)}")


if __name__ == "__main__":
    apply(EDITS)
