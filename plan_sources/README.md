# Plan sources

`PLAN.md` at the repository root is rendered from the `p*.md` section files in this directory:

    uv run --python 3.13 --with markdown python render_plan.py "$PWD" ../PLAN.md plan.html

`xref_check.py <rendered.md>` checks section cross-references (one known false positive, '45').

The `edit_batch13_*` to `edit_batch18_*` scripts are the idempotent exact-match edit batches of the third revision
(2026-09-04, afternoon and evening): the errata check (13, 13b), the Run 5S Appendix D entry (14), the settled
derivation-audit corrections (15), the corrected-closure timing benchmark (16), the 88Sr+ lifetime (17) and the
audit's final report (18). `integrate_topic.py` spliced the four Run 5S topic drafts (manifests alongside) and
`assemble_staged.py` built Section 9.15, the Section 12 Run 5 paragraph, the Section 13 amendments and the Appendix E
additions from the staged deliverables; those staged files lived in the session scratchpad and are not needed to
re-render. Earlier revisions edited the same section files directly (batches 1-12 of the morning of 2026-09-04 are
not preserved here; their results are in the text).
