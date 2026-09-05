"""Edit batch 16 (2026-09-04): the v3 Moelmer-Soerensen timing benchmark at the corrected closure."""

from edit_batch13_errata import apply

EDITS = [
    (
        "p11_performance.md",
        "| 2 ions × 1 mode, d_m = 12 | 48 | 0.08 s | 0.11 s | 0.4 s |\n| 2 ions × 2 modes, d_m = 8 | 256 | 0.43 s | 0.43 s | 2 s |\n| 2 ions × 3 modes, d_m = 6 | 864 | 13.6 s | 14.7 s | 1.1 min |\n| 2 ions × 3 modes, d_m = 8 | 2048 | 179 s | 132 s | 11 to 15 min |",
        "| 2 ions × 1 mode, d_m = 12 | 48 | 0.09 s | 0.13 s | 0.5 s |\n| 2 ions × 2 modes, d_m = 8 | 256 | 0.35 s | 0.47 s | 2 s |\n| 2 ions × 3 modes, d_m = 6 | 864 | 12.3 s | 15.1 s | 1.0 to 1.3 min |\n| 2 ions × 3 modes, d_m = 8 | 2048 | 124 s | 137 s | 10 to 11 min |",
    ),
    (
        "p11_performance.md",
        "At the largest case `vern9` is now the faster integrator (the first fixture had `dop853` faster by 30%), which is the reason the escalation ladder of Section 5.3 tries both rather than fixing one;",
        "At the largest case the faster integrator changed from run to run (`dop853` 124 s against `vern9` 137 s at the maximally entangling closure; 179 against 132 s at the χ = π/16 closure of the withdrawn v2 run; `dop853` faster by 30% on the first fixture), which is the reason the escalation ladder of Section 5.3 tries both rather than fixing one;",
    ),
    (
        "p11_performance.md",
        "the closure ratio ηΩ/ε = 1/4, which the derivation audit of 2026-09-04 later showed to be a χ = π/16 pulse in the plan's per-tone convention, the maximally entangling ratio being 1/2 (Section 4.4.1; `bench_ms_timing_v3.py` repeats the measurement at 1/2 and Section 9.13 reports both),",
        "the maximally entangling closure ηΩ/ε = 1/2 of Section 4.4.1 (`bench_ms_timing_v3.py`; the v2 run at 1/4, a χ = π/16 pulse in the plan's per-tone convention, gave times within 30% of these and is kept as `outputs/bench_ms_timing.out`),",
    ),
    (
        "p11_performance.md",
        "the measured 132 to 179 s for about 2 × 10⁴ evaluations",
        "the measured 124 to 137 s for about 2 × 10⁴ evaluations",
    ),
    (
        "p09_validation.md",
        "| Mølmer-Sørensen wall time | 0.08, 0.43, 13.6 and 179 s per 20 µs at joint dimensions 48, 256, 864 and 2048 with `dop853` and 0.11, 0.43, 14.7 and 132 s with `vern9`, pinned Python 3.13.14, on the realizable two-171Yb+ fixture at ηΩ/ε = 1/4; the first fixture (ηΩ/ε = 1/2; η 0.08, 0.06, 0.05) gave 0.09, 0.37, 10.1, 102 s and is kept as the negative control for the closure convention; results identical between integrators | Section 11.1 [recomputed here]; fixture [corrected] |",
        "| Mølmer-Sørensen wall time | 0.09, 0.35, 12.3 and 124 s per 20 µs at joint dimensions 48, 256, 864 and 2048 with `dop853` and 0.13, 0.47, 15.1 and 137 s with `vern9`, pinned Python 3.13.14, on the realizable two-171Yb+ fixture at the maximally entangling closure ηΩ/ε = 1/2 (`bench_ms_timing_v3.py`); the v2 run at ηΩ/ε = 1/4 (a χ = π/16 pulse) gave 0.08, 0.43, 13.6, 179 s and 0.11, 0.43, 14.7, 132 s, and the first fixture (ηΩ/ε = 1/2; η 0.08, 0.06, 0.05) gave 0.09, 0.37, 10.1, 102 s; both are kept as controls (`outputs/bench_ms_timing.out`, `bench_ms_timing_v1.py`); results identical between integrators | Section 11.1 [recomputed here]; closure [corrected: derivation audit, 2026-09-04] |",
    ),
]

if __name__ == "__main__":
    apply(EDITS)
