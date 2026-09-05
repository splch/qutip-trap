"""Edit batch 13b (2026-09-04): remaining errata-hunt fold-ins and the Appendix D paragraph."""

from pathlib import Path

from edit_batch13_errata import apply

HERE = Path(__file__).resolve().parent

ERRATA_PARAGRAPH = (
    "\n\n**Errata check** (wf_875da9f3-823, 26 agents, 2.8 M subagent tokens, 1031 tool uses, 45 min, 2026-09-04): "
    "one agent per printed source error that this plan corrects on its own authority (25 items across 23 sources), each "
    "searching for a published erratum, corrigendum or publisher's note (Crossref update relations, INSPIRE, OpenAlex, "
    "PubMed and the journal indices), comparing arXiv versions where they exist, and re-checking the claimed error by "
    "recomputation or by the paper's own internal consistency, with one agent tabulating the outcomes. No item has a "
    "published erratum. Nineteen claims were confirmed outright, among them Leibfried 2003's Eqs. 10-11 and 67-68, "
    "Wineland 1998's Eq. 62 (whose 2aρa† both arXiv versions print correctly, so the journal's stray dagger is a "
    "typesetting artifact and the plan matches the authors' own text), Wineland 2003's Eq. 2.5, Berkeland 1998's Eq. 21 "
    "(sympy residual exactly zero), Brownnutt 2015's duplicated Eq. 14 dissipator, the three Zhu-Monroe-Duan forms in the "
    "published PRL, Leung 2018's Eq. 5 in all four arXiv versions, Blümel 2021's (S6), (S8) and (S9) by numerical "
    "evaluation (the printed S6 returns a complex infidelity), Home 2013's seven slips against the LaTeX source, House "
    "2008's Eq. 7 sign (medium confidence, from a mirrored PDF), Mount 2015's PD6 header, Acton 2006's ħ for h (with the "
    "Rinton Press errata index as the positive control), Crain 2019's Eq. 3, Landa 2012's undefined downward step, "
    "Steck's Eq. 7.132 (still printed in revision 0.16.10 of 27 June 2026), Marzoli 1994's Eq. 4 against Cirac 1992's own "
    "half-rate Eq. 41, James 1998's node positions in Sec. 4, Chen et al.'s singular matrix (introduced in the version of "
    "record, Quantum 8, 1516 (2024), absent from the 2023 preprint), Schindler 2013's crosstalk definition and Madsen "
    "2004's 2400 against 2694. Five were partly right and changed this plan's wording: Turchette 2000's and Lee 2005's "
    "errors are the arXiv preprints' alone, the journal versions being correct (Sections 4.1.5, 4.3.2 and 13 now say so "
    "and tag them [corrected: preprint only]); the Ozeri 2007 Table II Yb+ row prints 0.2 × 10⁻⁴ and 2 mW against "
    "0.29 × 10⁻⁴ and 2.9 mW recomputed, the reverse of what the first revision stated (Sections 4.5.5 and 9.13); the "
    "Kirchmair and James locators were narrowed; and Brownnutt's Eq. 24 and ladder-operator sub-claims remain "
    "unresolved against the published review, which was unreadable. One item refuted the check's own source list "
    "rather than the plan (the inverted bright/dark clause belongs to Baldwin 2021, as Section 4.4.4 says, not to Noek "
    "2013, whose text and a 6j check are correct, so Baldwin's clause still rests on the Run 3 verdict), one confirmed "
    "the plan's attribution against the check's (the 2400 against 2694 depth is Madsen's, not Chiaverini's), and the check "
    "found one bibliographic error, the Zhu-Monroe-Duan arXiv identifier in Section 15, now fixed. APS abstract pages "
    "refused every automated fetch, so no publisher erratum panel was read directly; the substitute databases surfaced "
    "real errata for other papers in the same queries, which is the evidence that the negatives are meaningful."
)

EDITS = [
    (
        "p13_conventions.md",
        "Landsman's k = k₂ − k₁; Lee's Eq. 2 inverts its own beam-to-level assignment [corrected] |",
        "Landsman's k = k₂ − k₁; Lee's arXiv v1 Eq. 2 inverts its own beam-to-level assignment, the journal version being correct [corrected: preprint only] |",
    ),
    (
        "p04e_atomic.md",
        "Steck's QAO Eq. 7.132 prints the hyperfine operator with half-value ħ exponents (use the dimensionless S = I·J/ħ² form),",
        "Steck's QAO Eq. 7.132 prints the hyperfine operator with half-value ħ exponents, still unfixed in revision 0.16.10 of 27 June 2026 (use the dimensionless S = I·J/ħ² form),",
    ),
    (
        "p13_conventions.md",
        "Steck QAO Eq. 7.132 as printed halves every ħ exponent, correct only for ħ = 1 [corrected] |",
        "Steck QAO Eq. 7.132 as printed halves every ħ exponent, correct only for ħ = 1, unchanged in the June 2026 revision [corrected] |",
    ),
    (
        "p07b_control_details.md",
        "whose printed matrix in Chen et al. is singular and whose right-hand side must be error rates r = (4ⁿ − 1)(1 − p)/4ⁿ rather than raw decay bases **[corrected]**.",
        "whose printed matrix in Chen et al. is singular (both rows [0.75, 0.25]; the equation entered with the November 2024 revision that is the version of record, Quantum 8, 1516, so here the newest text is the one not to copy) and whose right-hand side must be error rates r = (4ⁿ − 1)(1 − p)/4ⁿ rather than raw decay bases **[corrected]**.",
    ),
]

if __name__ == "__main__":
    apply(EDITS)
    p = HERE / "p16_runs.md"
    text = p.read_text(encoding="utf-8")
    if "**Errata check** (wf_875da9f3-823" not in text:
        p.write_text(text.rstrip("\n") + ERRATA_PARAGRAPH + "\n", encoding="utf-8")
        print("appended errata paragraph to p16_runs.md")
    else:
        print("errata paragraph already present")
