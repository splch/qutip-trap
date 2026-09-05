"""Edit batch 14 (2026-09-04): Appendix D and Section 15 header entries for the Run 5 source-extraction run."""

from pathlib import Path

from edit_batch13_errata import apply

HERE = Path(__file__).resolve().parent

RUN5S_PARAGRAPH = (
    "\n\n**Run 5S** (wf_9734a0eb-744, 360 agents, 30.8 M subagent tokens, 5736 tool uses, 180 min, 2026-09-04): the "
    "source-extraction run for the four topics the plan still left unsourced after Run 4, with the same extract-verify-"
    "synthesize-consolidate pipeline and escalation rule as Runs 2 to 4: frequency-comb (mode-locked) Raman drives "
    "(Hayes 2010, Islam 2014, Inlek 2014, Mizrahi 2013 and 2014, Campbell 2010, Lee 2016), composite pulses and "
    "dynamical decoupling with the filter-function formalism (Brown-Harrow-Chuang 2004, Low-Yoder-Chuang 2014, "
    "Merrill-Brown 2014, Cummins-Llewellyn-Jones 2003, Kabytayev 2014, Green 2013, Biercuk 2009), the named single-"
    "source items (Berkeland-Boshier 2002, Kreuter 2005, Uys 2010, Wineland 2003 Eqs. 2.13-2.18, Ejtemaee-Haljan 2017, "
    "Joshi 2020; Letchumanan 2005 was unavailable behind the publisher's access wall), and ion transport, splitting and "
    "junctions as a specification for a later milestone (Reichle 2006, Bowler 2012, Walther 2012, Kaufmann 2014, "
    "Blakestad 2011, Pino 2021, Sterk 2022). 27 of 28 sources were read (162 items), 308 verdicts gave 89 items verified "
    "with flags, 69 refuted as extracted and corrected, and 4 split; 19 chunk syntheses were assembled into four briefs "
    "(comb: 63 model entries, 158 parameters, 108 checks; composite: 63, 90, 114; constants: 44, 97, 74; transport: 65, "
    "123, 96) and one consolidation (38 glossary entries, 24 inconsistencies, 18 must-implement components, 34 validation "
    "rows, 26 open issues). As in Runs 3 and 4, most refutations fell on the extractors' glosses (locators, hedges "
    "promoted to assertions, swapped indices, an invented electrode count), but the run also established printed errors "
    "in the sources that Sections 4.3.5, 4.3.7, 4.6, 6.9 and 8.1 list where they matter: a pulse-count-to-Rabi map wrong "
    "by (2π)², a Kapitza-Dirac Bessel argument off by 2, a composite-pulse recursion factor and an SK1 sign that make the "
    "printed sequences worse than a bare pulse, a splitting equation of motion missing a factor e and a quintic ramp with "
    "a global sign error, a heating term whose prose scaling contradicts its own equation, a recoil dissipator with a "
    "coherent sum over recoil classes, and a filter-function resummation off by 2 in its exponent. Its consolidation "
    "resolved two long-standing factor questions by appeal to this plan's own convention rows (the comb two-photon Rabi "
    "frequency Ω = g²/2Δ, with the consequence that a printed π-pulse energy is 2× too small, recorded as derived rather "
    "than verified; and the recoil angular factors 1/3, 2/5 and 1/5 being three emission patterns under one definition), "
    "and it stated what it could not source (Section 12). The briefs and the consolidation were folded into the text by "
    "topic on 2026-09-04, each new number tagged [recomputed here] only where a script under `validation/scripts/` "
    "reproduces it (`check_comb.py`, `check_composite.py`, `check_constants.py`, `check_transport.py`)."
)

EDITS = [
    (
        "p15_sources.md",
        "R4 = the gap-filling run of 2026-09-04, wf_0495300f-081).",
        "R4 = the gap-filling run of 2026-09-04, wf_0495300f-081; R5 = the source run of 2026-09-04 on frequency-comb drives, composite pulses and decoupling, named single-source constants and transport physics, wf_9734a0eb-744).",
    ),
]

if __name__ == "__main__":
    apply(EDITS)
    p = HERE / "p16_runs.md"
    text = p.read_text(encoding="utf-8")
    if "**Run 5S** (wf_9734a0eb-744" not in text:
        p.write_text(text.rstrip("\n") + RUN5S_PARAGRAPH + "\n", encoding="utf-8")
        print("appended Run 5S paragraph to p16_runs.md")
    else:
        print("Run 5S paragraph already present")
