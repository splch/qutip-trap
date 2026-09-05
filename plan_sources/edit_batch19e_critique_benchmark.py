"""Edit batch 19e (2026-09-04, critique v3 fold): the timing benchmark rebuilt (bench_ms_timing_v4.py) and the cost
model, budgets and Appendix D entries that rest on it."""

import sys
from pathlib import Path

from edit_batch13_errata import apply

HERE = Path(__file__).resolve().parent
TAG = "**[corrected: critique, 2026-09-04]**"

OLD_TABLE = """| System | Joint dimension | `dop853`, 20 µs | `vern9`, 20 µs | Full 100 µs gate (×5) |
|---|---|---|---|---|
| 2 ions × 1 mode, d_m = 12 | 48 | 0.09 s | 0.13 s | 0.5 s |
| 2 ions × 2 modes, d_m = 8 | 256 | 0.35 s | 0.47 s | 2 s |
| 2 ions × 3 modes, d_m = 6 | 864 | 12.3 s | 15.1 s | 1.0 to 1.3 min |
| 2 ions × 3 modes, d_m = 8 | 2048 | 124 s | 137 s | 10 to 11 min |"""

NEW_TABLE = """| System | Joint dimension | Merged dense, `dop853` / `vern9`, 20 µs | Merged CSR, `dop853` / `vern9` | Four-term dense (v3's structure), `dop853` / `vern9` | Evaluations per 20 µs (`dop853`) | Full 100 µs gate (×5, merged dense) |
|---|---|---|---|---|---|---|
| 2 ions × 1 mode, d_m = 12 | 48 | 0.03 / 0.04 s | 0.02 / 0.03 s | 0.09 / 0.12 s | 1.9 × 10⁴ | 0.15 s |
| 2 ions × 2 modes, d_m = 8 | 256 | 0.12 / 0.16 s | 0.34 / 0.48 s | 0.34 / 0.45 s | 2.1 × 10⁴ | 0.6 s |
| 2 ions × 3 modes, d_m = 6 | 864 | 1.23 / 1.08 s | 4.95 / 6.40 s | 11.6 / 14.1 s | 2.4 × 10⁴ | 6 s |
| 2 ions × 3 modes, d_m = 8 | 2048 | 24.8 / 28.9 s | 43.7 / 53.3 s | 114 / 130 s | 3.5 to 3.8 × 10⁴ | 2.1 to 2.4 min |"""

EDITS = [
    # ---- 11.1 intro: what v4 measures ----
    (
        "p11_performance.md",
        "and the re-measured times differ from the first version by at most a factor 1.8 at the largest case (`bench_ms_timing_v1.py` keeps the first fixture) **[recomputed here]**, **[corrected: derivation audit, 2026-09-04]**:",
        "and the re-measured times differ from the first version by at most a factor 1.8 at the largest case (`bench_ms_timing_v1.py` keeps the first fixture). The 2026-09-04 numerics critique then showed that v3 had timed a Dense operator (the exponential of the joint generator returns Dense on 5.3.1, and v3's non-zero probe printed −1) applied through four unmerged coefficient lambdas, one per drive term, so `bench_ms_timing_v4.py` re-measures the same fixture four ways: the drive operator as a tensor product of per-mode exponentials (CSR, non-zeros exactly 2^{N−1}Π_m d_m² per ion) and as the joint exponential (Dense), each with one coefficient function shared by all four drive terms, which `QobjEvo` merges into the single operator Σ_i(σ₊^i D_i + h.c.) with coefficient Ω cos(μt), and with four distinct coefficient functions as v3 had; it counts right-hand-side evaluations through the coefficient calls and compares final states across integrators and constructions. The v3 numbers reproduce on the idle machine (113.6 and 131.1 s at dimension 2048, `outputs/bench_ms_timing_v3_idle_rerun.out`), so they were not CPU contention: they are the four-term dense case, the fifth column below **[recomputed here]**, **[corrected: derivation audit, 2026-09-04]**, "
        + TAG
        + ":",
    ),
    ("p11_performance.md", OLD_TABLE, NEW_TABLE),
    # ---- 11.1 paragraph after the table ----
    (
        "p11_performance.md",
        "At the largest case the faster integrator changed from run to run (`dop853` 124 s against `vern9` 137 s at the maximally entangling closure; 179 against 132 s at the χ = π/16 closure of the withdrawn v2 run; `dop853` faster by 30% on the first fixture), which is the reason the escalation ladder of Section 5.3 tries both rather than fixing one; the numbers in Sections 11.2 to 11.4 and 14.4 use the slower of the two.",
        "`dop853` is faster than `vern9` in every v4 row but one (merged dense at 864, 1.23 against 1.08 s), by 10 to 25%, and the two integrators' final states agree to 1.3 × 10⁻⁷ (dimension 48) to 5.8 × 10⁻⁷ (2048) in norm at atol 10⁻¹⁰, rtol 10⁻⁸, which is what 'identical results' means throughout this plan, while the two constructions of the same operator agree to 10⁻¹³ to 2 × 10⁻⁸; the escalation ladder of Section 5.3 still tries both integrators because the margin is small and flipped between runs before (179 against 132 s in the withdrawn v2 run). The best representation at every measured size for `sesolve` is the merged dense operator: for two ions Σ_i(σ₊^i D_i + h.c.) fills N/2^N = one half of the matrix, and at that fill a CSR product (0.5 to 0.6 ns per non-zero, memory-bound at about 35 GB/s) loses to the vendor BLAS dense product (0.16 ns per element at dimension 2048, faster still while the matrix fits in cache), so 'sparse' is not the remedy for this operator and the mode-factorized kernel of Section 11.3 is; `mesolve` uses the CSR construction regardless, because there the Liouvillian's memory and not the product's speed decides (Section 5.3). The numbers in Sections 11.2 to 11.4 and 14.7 use the merged dense column "
        + TAG
        + ".",
    ),
    # ---- 11.2 cost model ----
    (
        "p11_performance.md",
        "- **Evaluations.** The adaptive step density measured in Section 5.2 is 14 to 45 steps per period of the highest mode frequency in the frame, growing slowly with the truncation cap, at 12 evaluations per DOP853 step: a 100 µs pulse on a 3 MHz mode needs 5 × 10⁴ to 1.6 × 10⁵ evaluations; a 10 µs single-qubit pulse 5 × 10³ to 1.6 × 10⁴.",
        "- **Evaluations.** Counted directly in v4 through the coefficient calls: 1.9 × 10⁴ (dimension 48), 2.1 × 10⁴ (256), 2.4 × 10⁴ (864) and 3.5 to 3.8 × 10⁴ (2048) right-hand-side evaluations per 20 µs with `dop853`, 1.2× more with `vern9`, that is 27 to 52 steps per period of the 3 MHz mode at 12 evaluations per step, growing with the cap as Section 5.2's 14 to 45 anticipated; a 100 µs pulse needs 1 to 2 × 10⁵ evaluations and a 10 µs single-qubit pulse 1 to 2 × 10⁴ (the second revision's 2 × 10⁴ per 20 µs came from a one-ion `solve_ivp` run in `bench_numerics.py` and was 1.8× low at 2048; 2026-09-04 numerics critique) "
        + TAG
        + ".",
    ),
    (
        "p11_performance.md",
        "- **Cost per evaluation.** Dominated by the drive operators: σ₊^i ⊗ Π_m D_m has 2^{N−1} Π_m d_m² non-zeros per ion (Section 5.1.1), and every right-hand-side evaluation applies both it and its Hermitian conjugate (the tones share the operator and enter through the coefficient), so one evaluation costs about N_ions × 2 × 2^{N−1} Π_m d_m² = N_ions 2^N Π_m d_m² complex multiply-adds. For 2 ions × 3 modes at d_m = 8 that is 2 × 4 × 8⁶ ≈ 2.1 × 10⁶ per evaluation, and the measured 124 to 137 s for about 2 × 10⁴ evaluations corresponds to about 3 to 4 ns per non-zero, a memory-bound sparse matrix-vector product; the benchmark script's term count is published with the table so the constant is reproducible.",
        "- **Cost per evaluation.** Dominated by the drive operator: σ₊^i ⊗ Π_m D_m has 2^{N−1} Π_m d_m² non-zeros per ion (Section 5.1.1), the merged operator Σ_i(σ₊^i D_i + h.c.) has N 2^N Π_m d_m², a fraction N/2^N of the (2^N Π_m d_m)² matrix (one half for two ions, 3/8 for three, 1/4 for four), and the tones share it through the coefficient. Measured constants (v4): a CSR product costs 0.5 to 0.6 ns per non-zero at every size from 256 to 2048, memory-bound at about 35 GB/s (20 bytes per non-zero), so the 2.1 × 10⁶ non-zeros at dimension 2048 cost 1.2 ms per evaluation; the dense product costs 0.16 ns per element at 2048 (67 MB, beyond cache, about 100 GB/s through the vendor BLAS) and 0.07 to 0.09 ns while the matrix fits in cache, so the same operator costs 0.65 ms per evaluation dense, and four unmerged dense terms 3.0 ms, which is where v3's 114 s came from. A 100 µs gate at dimension 2048 is therefore about 1.2 × 10⁵ × 0.65 ms ≈ 80 s merged dense (2.1 min measured), 2.5 min merged CSR and 10 min four-term dense; the second revision's '3 to 4 ns per non-zero' divided a four-term dense time by a CSR non-zero count and an evaluation count from another script (2026-09-04 numerics critique) "
        + TAG
        + ".",
    ),
    (
        "p11_performance.md",
        "- **Scaling.** Doubling d_m on three modes multiplies the cost by 64 × (a small factor from stiffness); adding a fourth mode at d_m = 8 multiplies it by 64; adding an ion doubles the dimension and adds a drive term. Four ions with three modes at d_m = 8 (dimension 8192, nnz ≈ 3 × 10⁷) would cost about an hour per 100 µs gate in this representation, and eight ions",
        "- **Scaling.** The cost is evaluations × elements touched per evaluation, dense D² = 4^N Π_m d_m² or CSR N 2^N Π_m d_m², so doubling d_m on three modes multiplies either by 64 and the evaluation count by about 1.4 (2.4 → 3.5 × 10⁴ from d_m = 6 to 8), adding a fourth mode at d_m = 8 multiplies by 64, and adding an ion multiplies the dense cost by 4 and the CSR cost by 2(N + 1)/N. Fitted to the four v4 rows, cost = a + b × (elements per evaluation) × evaluations with a ≈ 0.02 s and b = 0.16 ns (dense, beyond cache) or 0.58 ns (CSR) predicts every measured row within about 20% once the in-cache rows use their faster constant (the second revision's single-point rule mispredicted adjacent rows by up to 22×, the small rows being coefficient-call bound, four Python calls per evaluation being most of the 0.03 s at dimension 48). Four ions with three modes at d_m = 8 (dimension 8192; dense 6.7 × 10⁷ elements, 1.1 GB; CSR 1.7 × 10⁷ non-zeros, not the 3 × 10⁷ of the second revision) cost about 10 ms per evaluation either way and about 25 to 30 minutes per 100 µs gate at 1.5 × 10⁵ evaluations, the factorized kernel of Section 11.3 (24 against 512 amplitudes per mode product) being what brings that back to minutes "
        + TAG
        + "; and eight ions",
    ),
    # ---- 11.4 budgets ----
    (
        "p11_performance.md",
        "re-simulating one trajectory of a zoomed 100 µs entangling pulse with its recorded noise sample in under a second is possible at joint dimensions up to about 50 (one resolved mode) and in a few seconds up to about 256 (two modes at d_m = 8); above that,",
        "re-simulating one trajectory of a zoomed 100 µs entangling pulse with its recorded noise sample in under a second is possible at joint dimensions up to about 256 (two modes at d_m = 8: 0.6 s merged dense) and in a few seconds up to about 864 (three modes at d_m = 6: 6 s); above that (2 minutes at 2048),",
    ),
    # ---- 11.5 memory ----
    (
        "p11_performance.md",
        "An assembled drive operator at dimension 2048 with 2 × 10⁶ non-zeros is 32 MB in CSR form and grows as Π_m n_max,m²;",
        "An assembled drive operator at dimension 2048 with 2.1 × 10⁶ non-zeros is about 40 MB in CSR form (20 bytes per non-zero) and 67 MB dense, and grows as Π_m n_max,m²;",
    ),
    # ---- 5.3 integrator statement ----
    (
        "p05_numerics.md",
        "it was 20% faster than `vern9` on the Mølmer-Sørensen benchmark of Section 11 with identical results, while",
        "it was 10 to 25% faster than `vern9` on the Mølmer-Sørensen benchmark of Section 11 with final states agreeing to 6 × 10⁻⁷ in norm (`bench_ms_timing_v4.py`; the second revision said 'identical', which at these tolerances means 10⁻⁷ and not 10⁻¹⁵), while",
    ),
    # ---- 9.13 wall-time row ----
    (
        "p09_validation.md",
        "| Mølmer-Sørensen wall time | 0.09, 0.35, 12.3 and 124 s per 20 µs at joint dimensions 48, 256, 864 and 2048 with `dop853` and 0.13, 0.47, 15.1 and 137 s with `vern9`, pinned Python 3.13.14, on the realizable two-171Yb+ fixture at the maximally entangling closure ηΩ/ε = 1/2 (`bench_ms_timing_v3.py`); the v2 run at ηΩ/ε = 1/4 (a χ = π/16 pulse) gave 0.08, 0.43, 13.6, 179 s and 0.11, 0.43, 14.7, 132 s, and the first fixture (ηΩ/ε = 1/2; η 0.08, 0.06, 0.05) gave 0.09, 0.37, 10.1, 102 s; both are kept as controls (`outputs/bench_ms_timing.out`, `bench_ms_timing_v1.py`); results identical between integrators | Section 11.1 [recomputed here]; closure [corrected: derivation audit, 2026-09-04] |",
        "| Mølmer-Sørensen wall time | v4 (`bench_ms_timing_v4.py`), per 20 µs at joint dimensions 48, 256, 864 and 2048 with `dop853`: merged dense 0.03, 0.12, 1.23, 24.8 s; merged CSR 0.02, 0.34, 4.95, 43.7 s; four-term dense (v3's structure) 0.09, 0.34, 11.6, 114 s; `vern9` 1.1 to 1.3× slower; right-hand-side evaluations 1.9, 2.1, 2.4, 3.5 × 10⁴; CSR non-zeros per ion 288, 8192, 93312, 524288 = 2^{N−1}Π d_m² exactly; final states of the two integrators agree to 1.3 to 5.8 × 10⁻⁷ in norm and of the two constructions to 10⁻¹³ to 2 × 10⁻⁸; pinned Python 3.13.14, the realizable two-171Yb+ fixture at the maximally entangling closure ηΩ/ε = 1/2. The v3 run (0.09, 0.35, 12.3, 124 s and 0.13, 0.47, 15.1, 137 s) reproduces on the idle machine (113.6 and 131.1 s at 2048, `outputs/bench_ms_timing_v3_idle_rerun.out`) and is the four-term dense case; the v2 run at ηΩ/ε = 1/4 (a χ = π/16 pulse) gave 0.08, 0.43, 13.6, 179 s and the first fixture (η 0.08, 0.06, 0.05) 0.09, 0.37, 10.1, 102 s, both kept as controls (`outputs/bench_ms_timing.out`, `bench_ms_timing_v1.py`) | Section 11.1 [recomputed here]; closure [corrected: derivation audit, 2026-09-04]; construction [corrected: critique, 2026-09-04] |",
    ),
    # ---- 14.7 budgets ----
    (
        "p14_app.md",
        "| ≤ 1 s at joint dimension ≤ 50 (one resolved mode); ≤ 3 s at ≤ 256 (two modes, d_m = 8) | Section 11.1 measured 0.5 s and 2 s |",
        "| ≤ 1 s at joint dimension ≤ 256 (two modes, d_m = 8); ≤ 10 s at ≤ 864 (three modes, d_m = 6) | Section 11.1 (v4) measured 0.6 s and 6 s per 100 µs gate |",
    ),
    (
        "p14_app.md",
        "| Section 11.1 measured 1 min at dimension 864 and 10 min at 2048; Section 11.2 cost model |",
        "| Section 11.1 (v4) measured 6 s at dimension 864 and 2.1 min at 2048; Section 11.2 cost model |",
    ),
]

APPENDIX_D = """

**Run 5C, the four-lens critique of the revised plan (2026-09-04, evening; `wf_0f8efaf8-8c8`).** With the audit, the sources and the errata folded, the revised plan was frozen as a snapshot and read by twenty agents: for each of four lenses (a trapped-ion experimentalist, a light-matter theorist, a software architect, a QuTiP numerics expert) four scouts read the snapshot in quarters and listed candidate problems with line numbers, and one critic per lens re-read every cited passage, discarded candidates the plan already handled, merged duplicates and reported at most sixteen findings with a proposed fix and a confidence; 3.9 M tokens, 482 tool uses, 109 minutes. Three scouts (one theorist, two experimentalist) died on the 64 k output cap of the structured schema, so those quarters reached their critics through the remaining scouts only. Outcome: 64 findings, 10 blocking, all four critics judging the physics layer sound and the seams between sections, the interfaces of Appendix E and the numerics layer not yet fit to code from. Every finding was checked before adoption. The physics ones were recomputed in `check_critique_v3.py` (the sech transform pair behind the comb-tooth convention, the angular rate in I_sat, the dark-state destabilization optimum, the Fock tails of the GHZ fixtures, the zigzag pair, the composite-pulse durations, the Monroe 1995 beam geometry, the Morigi sign map, the Floquet function under the adopted Mathieu sign, the quadratic Zeeman coefficient for four g_J values, the infrared cutoff of the filter function, the order_slope round-off window, the 355 nm second-order weights, the d_m = 8 row of the Section 5.1.1 table, and QuTiP 5.3.1 probes for the Dense `expm`, the ENR `dims` against `shape`, `ptrace` on ENR spaces, `target_tol` without `e_ops` and `steadystate` on a singular kernel), and the numerics critic's benchmark finding was checked by rebuilding the timing benchmark (`bench_ms_timing_v4.py`, Section 11.1). That critic was right that v3 had timed a Dense operator with a non-zero probe printing −1, and the rebuilt measurement showed something the critique had not predicted: the 4.6× excess of v3 over the best construction is not the matrix representation but the coefficient structure, four distinct lambdas making `QobjEvo` apply four dense matrices per evaluation where one shared coefficient function merges them into one, and the CSR construction the critic asked for is 1.8× slower than the merged dense operator at this operator's 50% fill, which changes the cost model of Section 11.2 and confirms that the mode-factorized kernel, not sparsity, is the scaling remedy. Three of the critic's measurements (the `mesolve`/`sesolve` ratio 69.5, the `PicklingError` under `map="parallel"`, the ENR displacement's ⟨n₀⟩ = 2.2412) are adopted from the critique without a committed reproduction and say so in Section 12. Two proposed fixes were replaced rather than adopted: the (2π)³ infrared-cutoff scaling by the 10⁴ measured on the actual free-induction fixture, since the low-frequency estimate does not hold at ω_min T ≈ 1, and the CSR-everywhere recommendation by CSR for `mesolve` (memory) and merged dense for `sesolve` (speed). The fold is edit batches 19a to 19e beside the plan sources, each fix marked **[corrected: critique, 2026-09-04]** where it lands, with Section 9.17 carrying 46 regression targets and Section 12 what the critique left open. Two of the critique's findings overturned statements the derivation audit had just introduced (the u̇(0) = iν(1 − q) sign, written in the RMP's rf phase origin against Section 13's, and the 'η⁴ level' dismissal of C₀), which is the same lesson as the audit's: a correction is a claim like any other and gets its own check."""


def splice():
    p = HERE / "p16_runs.md"
    t = p.read_text(encoding="utf-8")
    anchor = "What the audit does not establish is stated in Section 12."
    if "**Run 5C, the four-lens critique of the revised plan" in t:
        print("Appendix D Run 5C paragraph already present")
        return
    if t.count(anchor) != 1:
        sys.exit(f"Appendix D anchor count {t.count(anchor)}")
    p.write_text(t.replace(anchor, anchor + APPENDIX_D), encoding="utf-8")
    print("appended the Run 5C paragraph to Appendix D")


if __name__ == "__main__":
    apply(EDITS)
    splice()
