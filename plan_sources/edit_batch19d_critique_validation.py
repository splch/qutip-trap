"""Edit batch 19d (2026-09-04, critique v3 fold): validation rows, roadmap, performance rules, risks, conventions.

The benchmark-number edits (Sections 11.1, 11.2, 9.13's wall-time row, Appendix D) wait for bench_ms_timing_v4.py and
live in batch 19e.
"""

import sys
from pathlib import Path

from edit_batch13_errata import apply

HERE = Path(__file__).resolve().parent
TAG = "**[corrected: critique, 2026-09-04]**"

EDITS = [
    # ---- 9.1 zigzag row ----
    (
        "p09_validation.md",
        "| Zigzag threshold | pinned: α_crit = 2/(μ_N − 1), exactly 1 and √(12/5) at N = 2, 3; 9Be+ at 3 µm needs 7.806 MHz; the fit 0.73N^{0.86} is kept as a negative control (13 to 32% off, Section 9.12) | Marquet 2003; Wineland 1998 [verified], [corrected] |",
        "| Zigzag threshold | pinned: α_crit = 2/(μ_N − 1) = 1 and 5/12 at N = 2, 3, equivalently (ω_r/ω_z)_crit = 1/√α_crit = 1 and √(12/5) = 1.5492 (the second revision printed √(12/5) as α_crit, which coincides only at N = 2; `check_critique_v3.py`); the infinite-chain force balance ω_r² = (7ζ(3)/(8πε₀)) e²/(m s_c³) gives ω_r/2π = 7.806 MHz for 9Be+ at s_c = 3 µm spacing (Wineland 1998's example; a finite chain's threshold is the α_crit above); the fit 0.73N^{0.86} is kept as a negative control (13 to 32% off, Section 9.12) | Marquet 2003; Wineland 1998 [verified], [corrected]; [corrected: critique, 2026-09-04]; [recomputed here] |",
    ),
    # ---- 9.3 rate-framework row ----
    (
        "p09_validation.md",
        "with (η̃/η)² = α k_em²/(bΔk·ê)² (α for one on-axis beam, 1.376 for quenched 40Ca+)",
        "with (η̃/η)² = α k_em²/(Δk·ê)² (α for one on-axis beam, 1.376 for quenched 40Ca+; the participation b cancels between η̃ and η, so the floor is independent of N, the regression of Section 4.2.2)",
    ),
    # ---- 9.x EIT row ----
    (
        "p09_validation.md",
        "| EIT closed-form self-consistency | Morigi Fig. 3 parameters (ν = 2.0068, γ = 20, Ω₁ = Ω₂ = 17, Δ = −70 MHz, η = 0.02):",
        "| EIT closed-form self-consistency | Morigi Fig. 3 parameters (ν = 2.0068, γ = 20, Ω₁ = Ω₂ = 17, Δ_Morigi = −70 MHz, which is Δ = +70 MHz in the plan's sign with Ω_r = √(Ω₁² + Ω₂²) = 24.04 MHz in the Section 4.2.3 closed form; η = 0.02):",
    ),
    # ---- 9.x Baldwin row ----
    (
        "p09_validation.md",
        "| Light-shift gate | Ballance parameters and budget; Baldwin Φ = 8π(Ωη/δ)² from Eq. 1; diag(1, i, i, 1) echo | Ballance [verified]; Baldwin [corrected] |",
        "| Light-shift gate | Ballance parameters and budget; Baldwin Φ_loop = 2π(Ωη/δ)² on Ŝ² per loop, 8π(Ωη/δ)² on σ_zσ_z over the two echo loops (Eq. 1); the printed H integrated through the echo gives diag(1, i, i, 1) | Ballance [verified]; Baldwin [corrected]; [corrected: critique, 2026-09-04] |",
    ),
    # ---- 9.5 CPT and POVM rows ----
    (
        "p09_validation.md",
        "| CPT | F_CPT(η) = F_no-CPT(η/3) | Olmschenk 2007 [verified] |",
        "| CPT | the Bloch-solve rate obeys R_∘ ≤ Γ/4 (the F = 1 → F′ = 0 manifold's ceiling; at s_∘ = 0.815, R_∘ = 0.0880Γ, times ε_sys = 4.356% is Crain's 472 kcps); the lumped F_CPT(η) = F_no-CPT(η/3) is an alternative model, illegal beside the Bloch solve | Olmschenk 2007 [verified]; Section 8.1 [corrected: critique, 2026-09-04] |",
    ),
    (
        "p09_validation.md",
        "the POVM fast path and the full record path agree in confusion matrix and a Bell state's bit correlations survive both",
        "the POVM fast path and the full record path agree in confusion matrix at zero readout crosstalk, differ by a bounded and reported amount at the configured crosstalk (4.0% nearest-neighbour PSF leakage at 14 µm), and a Bell state's bit correlations survive both",
    ),
    # ---- 9.6 GHZ fixtures and the virtual-Z test ----
    (
        "p09_validation.md",
        "Mølmer-Sørensen GHZ at t = π/(8χ) reproduced with three ions and two resolved modes at d_m = 8 (joint dimension 512) and four ions with two resolved modes at d_m = 6 (dimension 576), the other modes frozen or dropped by the contribution criterion with their residual reported;",
        "Mølmer-Sørensen GHZ at t = π/(8χ) reproduced with three ions and two resolved modes at d_m = 12 (joint dimension 1152) and four ions with two resolved modes at d_m = 12 (dimension 2304), both inside the Section 5.4 ceiling, each fixture quoting the boundary population it reaches (a coherent displacement |α| = 1 leaves 5.9 × 10⁻⁴ in the top two levels at d_m = 8 and 1.9 × 10⁻² at d_m = 6, against the 10⁻⁶ threshold of Section 5.5 that d_m = 12 meets at 1.1 × 10⁻⁷; `check_critique_v3.py`; the second revision's d_m = 8 and 6 violated the plan's own truncation policy and sat below the smallest row of the Section 5.1.1 table, which now carries a d_m = 8 row; 2026-09-04 numerics critique), the other modes frozen or dropped by the contribution criterion with their residual reported;",
    ),
    (
        "p09_validation.md",
        "virtual-Z propagation rule pinned; GPi2(0) on ∣0⟩ gives the Section 7.6 state |",
        "virtual-Z propagation tested on a concrete sequence, RZ(0.1) then GPi2(0) on ∣0⟩ against the independently multiplied matrices, equal to GPi2(−0.1) followed by RZ(0.1) (φ → φ − θ, Section 7.6; a rule 'pinned' passes for either sign, 2026-09-04 architect critique); GPi2(0) on ∣0⟩ gives the Section 7.6 state |",
    ),
    # ---- 9.8 dropped-mode row gains the freeze class ----
    (
        "p09_validation.md",
        "| Dropped-mode policy | modes dropped by the contribution criterion (∣α_m(τ)∣²(2n̄_m + 1) < 10⁻⁶ and ∣χ_m(τ)∣ < 10⁻⁴ from the Section 4.4.3 integrals with the pulse's detuning schedule) change two-qubit fidelity by less than the summed dropped contribution the report states; a mode with η = 10⁻³ two kilohertz from a tone is kept |",
        "| Dropped-mode policy | modes dropped by the contribution criterion (∣α_m(τ)∣²(2n̄_m + 1) < 10⁻⁶ and ∣χ_m(τ)∣ < 10⁻⁴ from the Section 4.4.3 integrals with the pulse's detuning schedule) change two-qubit fidelity by less than the summed dropped contribution the report states; a mode with η = 10⁻³ two kilohertz from a tone is kept; a mode above the drop pair but with ∣χ_m∣ < 0.05 rad is classed frozen, never dropped, and its χ_m loss appears in the calibration target and in `Diagnostics.mode_class` (Section 5.2) |",
    ),
    (
        "p09_validation.md",
        "the same run over 1 and 18 workers gives bitwise-identical results;",
        "the same run over 1 and 18 workers agrees to 10⁻¹² with per-trajectory identity under the same seed (a bitwise assertion fails at 4.4 × 10⁻¹⁶ on 5.3.1 because trajectory sums accumulate in completion order, Section 3.4);",
    ),
    # ---- 9.12 anharmonic estimator ----
    (
        "p09_validation.md",
        "the phase over a gate is second order, g²t/Δ_res with virtual population (g/Δ_res)²: two modes with g(a + a†)²(b + b†) at g/2π = 1.419 kHz give 0.0041, 0.0034, 0.0032, 0.0032 rad on ∣1, 0⟩ over 100 µs at mismatches 1, 0.3, 0.1, 0.03 MHz",
        "the phase over a gate is the directly integrated value, which is the fixture: two modes with g(a + a†)²(b + b†) at g/2π = 1.419 kHz give 0.0041, 0.0034, 0.0032, 0.0032 rad on ∣1, 0⟩ over 100 µs at mismatches 1, 0.3, 0.1, 0.03 MHz, essentially flat, while the second-order label g²t/Δ_res with virtual population (g/Δ_res)² would give 0.0013, 0.0042, 0.0127, 0.0422 rad at the same mismatches (`check_anharmonic.out`), 3× low at 1 MHz and 13× high at 0.03 MHz, so that label is withdrawn as the scaling and the default-on estimator of Section 5.7 reports the integrated phase, with g²t/Δ_res kept only as an explicit upper bound valid for Δ_res ≫ the mode-frequency spread (2026-09-04 numerics critique) [corrected: critique, 2026-09-04]",
    ),
    # ---- 9.13 rows ----
    (
        "p09_validation.md",
        "| 171Yb+ clock quadratic shift | 310.85 Hz/G² numerically and from (g_J − g_I)²μ_B²/(2hA) with A = 12.642812118 GHz, g_J = 2.00254 | Olmschenk 2007 / Fisk: 310.8 Hz/G² [verified] |",
        "| 171Yb+ clock quadratic shift | 310.87 Hz/G² numerically and from (g_J − g_I)²μ_B²/(2h²A) with A = 12.642812118 GHz in Hz and the adopted g_J = 2.002615(70) (Han et al. 2025), tolerance ±0.02 Hz/G² from the g_J uncertainty; the earlier row's 310.85 used an uncited g_J = 2.00254 and printed 2hA (`check_critique_v3.py`) | Olmschenk 2007 / Fisk: 310.8 Hz/G²; Han et al. 2025: 31.0869(22) mHz/µT² [verified]; [corrected: critique, 2026-09-04]; [recomputed here] |",
    ),
    (
        "p09_validation.md",
        "171Yb+ 310.797 Hz/G² with g_J = 2.00292 and 310.85 with 2.00254 against the quoted 310.8, so g_J's fourth decimal is the tolerance; B in gauss |",
        "171Yb+ 310.797 Hz/G² with g_J = 2.00292 (g_I = 0) and 310.85 with 2.00254 against the quoted 310.8, both uncited values now superseded by the adopted g_J = 2.002615(70), which gives 310.87 (Section 4.5.1); g_J's fourth decimal is the tolerance; B in gauss |",
    ),
    (
        "p09_validation.md",
        "| Breit-Rabi curvature of the 171Yb+ clock line | taylor_c2 = (g_J − g_I)²μ_B²/(2h²ν₀) = 310.76 Hz/G² with g_J = 2.00225664 and ν₀ = 12.642812118466 GHz (310.59 with g_I = 0),",
        "| Breit-Rabi curvature of the 171Yb+ clock line | taylor_c2 = (g_J − g_I)²μ_B²/(2h²ν₀) = 310.76 Hz/G² with g_J = 2.00225664 (43Ca+'s value, superseded by the adopted 2.002615, which gives 310.87; Section 4.5.1) and ν₀ = 12.642812118466 GHz (310.59 with g_I = 0),",
    ),
    # ---- 9.x sech envelope widths row ----
    (
        "p09_validation.md",
        "| sech envelope widths | intensity FWHM 2 arccosh(2)/π = 0.8384014366 τ for a sech intensity envelope, 2 arccosh(√2)/π = 0.5611 τ for sech²;",
        "| sech envelope widths | for the tooth convention sech(2πkν_repτ), whose field envelope is sech(πt/(2τ)): field FWHM (4/π) arccosh(2) = 1.6768 τ, intensity (sech²) FWHM (4/π) arccosh(√2) = 1.1222 τ, with FFT[sech(πt/2τ)] ∝ sech(ωτ) checked numerically (`check_critique_v3.py`); the earlier row's 0.8384 τ and 0.5611 τ are the widths of sech(πt/τ), a different envelope whose teeth read sech(πkν_repτ) [corrected: critique, 2026-09-04];",
    ),
    # ---- 10: M0a anchors, M3a before M3, M4 exit criteria, M5 ownership ----
    (
        "p10_roadmap.md",
        "acceptance tests are the anchors of Section 9.13 (43Ca+ 146.0942 G, 9Be+ 119.446 G, 25Mg+ 212.78 G, 171Yb+ 310.85 Hz/G², the 171Yb+ 1/3 : 2/3 branching, I_sat = 50.77 mW/cm²).",
        "acceptance tests are the anchors of Section 9.13 (43Ca+ 146.0942 G, 9Be+ 119.446 G, 25Mg+ 212.78 G, 171Yb+ 310.87 ± 0.02 Hz/G² with the adopted g_J = 2.002615, the 171Yb+ 1/3 : 2/3 branching, I_sat = 50.83 mW/cm² from the partial 19.62 MHz rate; the second revision's 310.85 and 50.77 were the retired readings, and a CI check greps the tree for retired constants).",
    ),
    (
        "p10_roadmap.md",
        "optical pumping from first principles on the M3b builder.",
        "optical pumping from first principles on the M3a builder, which precedes this milestone.",
    ),
    (
        "p10_roadmap.md",
        "### M3b. Multi-level optical Bloch builder (2 weeks)",
        "### M3a. Multi-level optical Bloch builder (2 weeks; runs before M3, whose Doppler stage needs its W(Δ))",
    ),
    (
        "p10_roadmap.md",
        "- Tests: the level-A and level-B closed forms recovered from level C in their stated regimes; the 171Yb+ detection rate and leakage prefactors of Section 8.1 recovered from the angular algebra.",
        "- Tests: the level-A and level-B closed forms recovered from level C in their stated regimes; the 171Yb+ detection rate and leakage prefactors of Section 8.1 recovered from the angular algebra (this milestone owns that rate object; M5 consumes it).",
    ),
    (
        "p10_roadmap.md",
        "- Multi-mode: 3 to 5 ions, AM/FM/PM pulse solvers for closure over all modes.\n- Tests: Section 9.4.",
        "- Multi-mode: 3 to 5 ions, AM/FM/PM pulse solvers for closure over all modes.\n- Exit criteria: JOINT_EXACT at 2 to 3 ions with the resolved modes of Section 5.4, plus closed-form closure checks (the Section 4.4.3 integrals) at 4 to 5 ions; the exact multi-mode comparison at 4 to 5 ions is deferred to M9a, where mode selection, frozen spectators and the matrix-free kernel exist (the second revision required it here, where the plan's own rule made it unvalidatable; 2026-09-04 architect critique).\n- Tests: Section 9.4.",
    ),
    (
        "p10_roadmap.md",
        "- Fluorescence rate model; bright/dark pumping; shelving; photon-count sampling; threshold and time-resolved discrimination; crosstalk.",
        "- Fluorescence record model consuming M3a's rate object (R_∘, R_d, R_b from the Bloch solve); shelving; photon-count sampling; threshold and time-resolved discrimination; crosstalk; the product POVM at zero crosstalk and the register-wide confusion tensor otherwise (Section 5.7).",
    ),
    # ---- 11.3 reduction rules: freeze class, parallel map ----
    (
        "p11_performance.md",
        "not on η alone, because a mode with η = 10⁻³ two kilohertz from a tone contributes more than one with η = 0.05 a megahertz away; dropped modes evolve freely (their heating still applies) and the summed dropped contrib",
        "not on η alone, because a mode with η = 10⁻³ two kilohertz from a tone contributes more than one with η = 0.05 a megahertz away; a mode above this pair but with |χ_m| below the freeze tolerance of Section 5.2 (default 0.05 rad) is frozen rather than dropped, the third class; dropped modes evolve freely (their heating still applies) and the summed dropped contrib",
    ),
    (
        "p11_performance.md",
        "trajectories and quasi-static samples are spread over the 18 cores with the parallel `map`.",
        'trajectories and quasi-static samples are spread over the 18 cores with the parallel `map`, which requires every `QobjEvo` coefficient to be a module-level named function, a `Coefficient` or an array (a lambda or closure raises `PicklingError` under `map="parallel"` on 5.3.1, and only `map="loky"`, an optional dependency, survives it through cloudpickle), so the builder never emits closures and a CI test runs `mcsolve` with `map="parallel"` on a `QobjEvo` built by the real builder (2026-09-04 numerics critique) '
        + TAG
        + ".",
    ),
    # ---- 11.5 guard ----
    (
        "p11_performance.md",
        "The monitor refuses to build joint spaces above a configurable dimension (default 4 × 10⁶) and directs the run to GATE_LOCAL.",
        "The monitor refuses to build joint spaces above a configurable dimension (default 4096, from the measured cost model of Section 11.2) or above a configurable estimate of the drive operator's non-zeros (default 2 × 10⁷, from Π_m d_m² and the ion count), and directs the run to GATE_LOCAL; the second revision's default of 4 × 10⁶ was 2000 times the budget Section 5.4 states and could not have been honoured (2026-09-04 architect critique) "
        + TAG
        + ".",
    ),
    # ---- 12: the 8-ion figure ----
    (
        "p12_risks.md",
        "- **Exactness versus size.** Joint-exact simulation ends near 8 ions.",
        "- **Exactness versus size.** Joint-exact simulation ends at a joint dimension of about 2 × 10³ to 4 × 10³, two or three ions with two or three dynamically resolved modes (Section 5.4; the second revision's 'near 8 ions' rested on the withdrawn ENR estimate, and the guard of Section 11.5 refuses above 4096).",
    ),
    # ---- 13: the tau row ----
    (
        "p13_conventions.md",
        "| Pulse-duration parameter τ | the sech **field**-envelope parameter in sech(πt/τ), equivalently the tooth-formula parameter in sech(2πk ν_rep τ); never a FWHM. Intensity FWHM = 0.8384014366 τ for a sech intensity envelope, 0.5611 τ for sech². A `PulseShape` field records which envelope τ parameterizes |",
        "| Pulse-duration parameter τ | the sech **field**-envelope parameter in sech(πt/(2τ)), equivalently the tooth-formula parameter in sech(2πk ν_rep τ) (the transform pair; the second revision wrote sech(πt/τ), whose teeth would read sech(πkν_repτ)); never a FWHM. Field FWHM = 1.6768 τ, intensity (sech²) FWHM = 1.1222 τ (`check_critique_v3.py`; the earlier 0.8384 τ and 0.5611 τ are the widths of sech(πt/τ)). A `PulseShape` field records which envelope τ parameterizes [corrected: critique, 2026-09-04] |",
    ),
]

NEW_13_ROWS = """| Floquet function and rf phase origin | the adopted Mathieu sign d²x/dξ² + [a − 2q cos 2ξ]x = 0 fixes the rf phase origin: u(t) ≈ e^{iνt}[1 − (q/2)cos ω_rf t]/(1 − q/2), c_{±1}/c₀ = −q/((2 ± β)² − a), u̇(0) = iν(1 + q) at u(0) = 1; C₀ = 1 + 3q²/16 is even in q and applied once, in `Crystal.lamb_dicke` | Section 4.1.1; `check_critique_v3.py` | the RMP's quantum section (W(t) with +2q cos), Wineland's trajectory and the second revision's 4.1.1 use the opposite origin, [1 + (q/2)cos], iν(1 − q): the same physics half an rf period later, which left every micromotion sideband phase undetermined [corrected: critique, 2026-09-04] |
| EIT detuning sign | Δ = ω_drive − ω_transition (the plan's sign), so the EIT closed form of Section 4.2.3 requires Δ > 0 and Ω_r² = 4ν(ν + Δ); Ω_r = √(Ω₁² + Ω₂²) | Section 4.2.3 | Morigi's Δ is the negative of the plan's (her optimum 4ν(ν − Δ)); her Fig. 3 fixture is Δ = +70 MHz, Ω_r = 24.04 MHz in plan units [corrected: critique, 2026-09-04] |
| Sideband excitation lineshape | P = [Ω²/(Ω² + δ²)] sin²((t/2)√(Ω² + δ²)), half-depth at δ = Ω, π time π/Ω; the calibration fit function of Section 7.5 | Section 13 Rabi-frequency row; Section 7.5 | Wineland's and Blümel's Ω²/(Ω² + δ²/4) sin²(t√(Ω² + δ²/4)) is the half-Rabi form, doubled on ingest and never fitted [corrected: critique, 2026-09-04] |
| Virtual-Z propagation | RZ(θ) shifts every later pulse phase φ → φ − θ, gates read in time order | Section 7.6 (recomputed) | IonQ's printed +θ holds in matrix order; the second revision's 5.2 printed +θ in time order [corrected: critique, 2026-09-04] |
| Single-photon coupling symbol g | not a plan symbol: the code stores Ω (the ħΩ/2 convention) and two ingest maps, g = Ω/2 for Monroe 1995, Wineland 1998, Ozeri and Section 4.5.5's g_{b,r} = E⟨d⟩/2ħ, and g = Ω for the Lee comb chain (g₀ = γ√(Ī/2I_sat), Section 4.3.7); Monroe's Ω = g₁g₂/Δ is Ω₁Ω₂/(2Δ) in plan units | Sections 4.2.2, 4.3.7, 4.5.5 | one symbol meaning Ω/2 in two subsections and Ω in a third inside two-photon formulas that differ by exactly the factor 2 the plan hunts [corrected: critique, 2026-09-04] |
| Mode index | every `mode: int` is a position in `Crystal.modes`, ordered axial, transverse_1, transverse_2, ascending frequency within a family; eigenvectors unit-norm with the last component positive | Appendix E; Section 4.1.3 | eigensolver output order, frequency order across families, or (family, index) pairs, none declared in the second revision [corrected: critique, 2026-09-04] |
| Filter-function infrared cutoff | ω_min = 2π/T_total in rad/s, always reported with d ln χ/d ln ω_min beside χ | Section 6.9; Appendix E | 1/T for a rad/s field (the second revision); on an IR-divergent spectrum the two differ by 10⁴ on the free-induction fixture [corrected: critique, 2026-09-04] |
| Saturation-intensity rate | I_sat = πhcΓ_partial/(3λ³) with Γ_partial = 2π γ_hz × branching, ANGULAR: 171Yb+ 369.5 nm 50.83 mW/cm² | Section 9.13; Appendix E `Transition.partial_rate_rad_s` | `gamma_hz` fed in unconverted gives 8.09 mW/cm², a 2π in every saturation parameter, scattering rate and comb Rabi frequency [corrected: critique, 2026-09-04] |
| Composite-pulse durations | τ = Σ_l θ_l/Ω: SK1, BB1 4π + θ; CORPSE 4π + θ − 4k; CinSK, CinBB 8π + θ − 4k | Section 4.3.5; `check_composite.py` | the second revision's 8π + 2θ − 4k for the concatenations, 12% long at θ = π [corrected: critique, 2026-09-04] |
| 171Yb+ ground-state g_J | 2.002615(70) (Han et al. 2025, arXiv:2501.09973, MCDHF and MRCI), giving 310.87 Hz/G² with g_I from μ_I = +0.49367 μ_N | Section 4.5.1; `check_critique_v3.py` | 2.00225664 (43Ca+'s), 2.00254 and 2.00292, three uncited values in three rows of the second revision; the old spectroscopic 1.998 [corrected: critique, 2026-09-04] |"""

SECTION_9_17 = """### 9.17 Critique-fold targets (2026-09-04, four-lens critique)

The four-lens critique of 2026-09-04 (Appendix D) produced 64 findings; the rows below are the regression targets the fold introduced, one per finding that changes a number, a convention or an interface, written as the test suite must reproduce them. Numbers marked `check_critique_v3.py` are printed by that committed script (**[recomputed here]**); the rest are the implementer's targets.

| Test | Expected | Source, tag |
|---|---|---|
| Sideband-cooling floor independent of N | n̄_min of a fixed mode is identical for b_{i,m} = 1 and b_{i,m} = 1/√17 (the participation cancels between η̃ and η) | Section 4.2.2 [corrected: critique, 2026-09-04] |
| Morigi EIT fixture in the plan's sign | Δ = +70 MHz, Ω_r = √(17² + 17²) = 24.04 MHz → ⟨n⟩_S = 0.005102 = (γ/4Δ)²; Δ = −70 → −1.0051 and the A₋ − A₊ ≤ 0 guard fires; Ω_r = 17 → 0.1467, 29× the target | `check_critique_v3.py` [recomputed here] |
| Preparation stage order | the scheduler refuses optical pumping placed before Doppler cooling; the canonical sequence is Doppler → sideband/EIT → final pump | Section 4.2.6 [corrected: critique, 2026-09-04] |
| Two transverse families | the crystal solver returns 3N modes in three families with shared eigenvectors, Ω_p^{(x)} ≠ Ω_p^{(y)} whenever ω_x ≠ ω_y, in the canonical order of Appendix E | Section 4.1.3 [corrected: critique, 2026-09-04] |
| Waveform per (ion, leg) | `Waveform.symmetric(...)` equals the general form at equal envelopes; Ω_a ≠ Ω_b routes to the symmetrized χ kernel of Section 4.4.3 and the per-ion amplitude calibration entry | Appendix E [corrected: critique, 2026-09-04] |
| Drive Δk derived from the beams | Monroe 1995: ∣Δk∣ = √2 k gives η_x = 0.2009 against the quoted 0.21; 2k gives 0.284 (negative control); y, z anchors 0.111, 0.087 from a projection k | `check_critique_v3.py` [recomputed here] |
| Persistent run state | a dark-ion event at shot k changes N, positions, modes and the qubit-to-beam map for every shot after k until a reload event; `Result.run_state.events` records it; no i.i.d. herald | Section 6.7; Appendix E [corrected: critique, 2026-09-04] |
| Measure and reset in the IR | a circuit with a mid-circuit measure compiles and schedules; the first release's executor refuses it with the stated error; a terminal measurement is unchanged | Section 7.2 [corrected: critique, 2026-09-04] |
| Calibration dependency graph | `stark_scan`, `crosstalk_scan`, `field_scan` and `crystal_image` exist; a mode-frequency fit with `micromotion` uncalibrated refuses to run | Section 7.5 [corrected: critique, 2026-09-04] |
| Mathieu sign of the Floquet function | numerical Floquet solution under a − 2q cos 2ξ: c_{±1}/c₀ = −q/((2 ± β)² − a) (−0.02332, −0.02688 at q = 0.1) and u̇(0)/(iνu(0)) = 1.1036 against 1 + q; the modulation index of Section 4.3.6 changes sign across the compensated shim voltage | `check_critique_v3.py` [recomputed here] |
| C₀ applied once | η(q = 0.3)/η(0) = 1 + 3q²/16 + O(q⁴) = 1.017 to 10⁻³; a carrier-calibrated and a sideband-calibrated gate differ in two-body phase by 2(C₀ − 1) = 3.4% at q = 0.3, the sideband-calibrated one by 0 | Sections 4.1.1, 4.3.6 [corrected: critique, 2026-09-04] |
| I_sat constructor assertion | 171Yb+ 369.5 nm with the partial 19.62 MHz rate, angular: 50.83 mW/cm²; `gamma_hz` unconverted 8.09 (negative control) | `check_critique_v3.py` [recomputed here] |
| Micromotion index | `Trap.micromotion_beta` returns (in-phase, out-of-phase) with a peak/rms tag; β scales with ∣Δk∣ and not with k̂ alone | Appendix E [corrected: critique, 2026-09-04] |
| Doppler detuning per beam | one (Δ, s, k̂) for all modes; the per-mode n̄ vector for Monroe's 11.2, 18.2, 29.8 MHz modes from the single Δ = −30 MHz within the Section 9.3 tolerance | Section 4.2.1 [corrected: critique, 2026-09-04] |
| Composite-pulse durations | Σ_l θ_l at θ = π: CORPSE 4.3333π (780°), CinSK and CinBB 8.3333π (1500°), SK1 and BB1 5π; the printed 8π + 2θ − 4k = 9.3333π is the negative control | `check_critique_v3.py`; `check_composite.py` [recomputed here] |
| Shot clock | shot k is evaluated at t0 + k T_rep; a `Drift` with `rate_per_s` advances linearly across a run; `Diagnostics.wall_clock_span_s` is reported | Appendix E [corrected: critique, 2026-09-04] |
| Sideband lineshape fit | P = [Ω²/(Ω² + δ²)] sin²((t/2)√(Ω² + δ²)); the fitted π time on a noiseless carrier is π/Ω and the half-depth sits at δ = Ω; fitting the half-Rabi form returns Ω/2 (negative control) | Section 7.5 [corrected: critique, 2026-09-04] |
| Virtual-Z concrete sequence | RZ(0.1) then GPi2(0) on ∣0⟩ equals GPi2(−0.1) followed by RZ(0.1) as multiplied matrices (φ → φ − θ) | Sections 5.2, 7.6, 9.6 [corrected: critique, 2026-09-04] |
| CPT ceiling | the Bloch-solve R_∘ never exceeds Γ/4; at s_∘ = 0.815, R_∘ = 0.0880Γ, and times ε_sys = 4.356% it is Crain's 472 kcps | Section 8.1 [corrected: critique, 2026-09-04] |
| Dark-state destabilization optimum | B_opt = 2.33 G at Ω = γ/3 (δ_B/2π = 3.267 MHz); 5 G is 2.14× the optimum | `check_critique_v3.py` [recomputed here] |
| 171Yb+ quadratic Zeeman coefficient | (g_J − g_I)²μ_B²/(2h²A) = 310.87 Hz/G² with g_J = 2.002615, tolerance ±0.02; the Breit-Rabi terms are all in Hz | `check_critique_v3.py` [recomputed here] |
| sech envelope widths | FWHM_field = 1.6768 τ, FWHM_intensity = 1.1222 τ for the tooth convention; FFT[sech(πt/2τ)] = sech(ωτ) to 10⁻⁶ at ωτ = 0.5 to 3 | `check_critique_v3.py` [recomputed here] |
| Symbol g at ingest | Monroe's Ω = g₁g₂/Δ enters as Ω₁Ω₂/(2Δ); Lee's g₀ enters as Ω | Section 13 [corrected: critique, 2026-09-04] |
| Zigzag pair | α_crit = 1, 5/12 and (ω_r/ω_z)_crit = 1, 1.5492 at N = 2, 3 | `check_critique_v3.py` [recomputed here] |
| PGC static gradient | an ion at φ = π/4 raises the uncooled-mode error; the moving-gradient minimum 0.8693 is returned only inside W < δ < ω_z | Section 4.2.4 [corrected: critique, 2026-09-04] |
| Ramsey coherence normalization | C(T) = exp(−χ) with χ from Section 6.9's `filter_function`, S_b = S_δ/4; reading the splitting PSD in place of S_b gives 4× the exponent (negative control) | Section 4.5.7 [corrected: critique, 2026-09-04] |
| Diabatic-splitting pairing | the printed π²/8 with ξ² = 0.1 is the one pairing; π²/32 (ξ² 0.4) and π²/264 (ξ² 3.3) are refused when mixed | Section 4.6 [corrected: critique, 2026-09-04] |
| Baldwin echo | the printed H integrated through two loops gives diag(1, i, i, 1); Φ_loop = 2π(ηΩ/δ)² on Ŝ² per loop | Section 4.4.4 [corrected: critique, 2026-09-04] |
| Device hash | identical across processes for the same device file; unchanged by a float perturbation at 10⁻¹³ relative, changed at 10⁻¹¹ | Appendix E [corrected: critique, 2026-09-04] |
| Size guard | a joint dimension above 4096 or an estimated drive-operator non-zero count above 2 × 10⁷ routes to GATE_LOCAL | Section 11.5 [corrected: critique, 2026-09-04] |
| Seeds and reproducibility | two shots on one (sample, trajectory) draw different photon records; 1 and 18 workers agree to 10⁻¹²; an OU realization is identical under two integrator step sequences | Sections 3.4, 5.5 [corrected: critique, 2026-09-04] |
| GATE_LOCAL tomography | n_traj ≈ 1/ε_map per input with population `e_ops`; the Dykstra projection leaves ∣∣Tr_out(Choi) − 𝟙∣∣ < 10⁻¹⁰ and reports both residuals | Section 5.4 [corrected: critique, 2026-09-04] |
| Steady-state kernel guard | a three-level system with an uncoupled level: `direct` raises singular, `power`, `eigen` and `svd` return three different states (kernel dimension 2); the module raises on dim(kernel) > 1 and restricts the solve to the driven block | `check_critique_v3.py` [recomputed here] |
| Picklable coefficients | `mcsolve` with `map="parallel"` runs on the real builder's `QobjEvo` | Section 11.3 [corrected: critique, 2026-09-04] |
| Freeze against drop | a spectator with ∣χ_m∣ = 0.03 rad and ∣α_m∣²(2n̄+1) = 10⁻⁵ is classed frozen, not dropped, and its χ_m loss appears in the calibration target; one with 10⁻⁷ and 10⁻⁵ rad is dropped | Section 5.2 [corrected: critique, 2026-09-04] |
| ENR marginal | `HilbertSpace.marginal` equals `ptrace` of the same state embedded in the product space; `shape` == 56 where `dims` multiply to 98 for two modes at N_exc = 6 | `check_critique_v3.py` [recomputed here] |
| Escalation ladder | d_m = 121, 151 and 201 on the Section 11.1 pulse succeed with atol 10⁻⁸ on `dop853` or `vern9`; no multistep rung exists | Section 5.3 [corrected: critique, 2026-09-04] |
| Comb tone cut | moving the explicit-tone cut from 10/t_g to 20/t_g changes the gate phase by less than 10⁻⁴ rad | Section 4.3.7 [corrected: critique, 2026-09-04] |
| GHZ fixture truncation | at ∣α∣ = 1 the top two levels hold 1.1 × 10⁻⁷ at d_m = 12, 5.9 × 10⁻⁴ at d_m = 8 and 1.9 × 10⁻² at d_m = 6 | `check_critique_v3.py` [recomputed here] |
| Anharmonic phase estimator | the integrated phases 0.0041, 0.0034, 0.0032, 0.0032 rad are the fixture; the g²t/Δ_res label is an upper bound only | Section 9.12 [corrected: critique, 2026-09-04] |
| order_slope window | BB1 returns slope 6.0 ± 0.1 on (10⁻², 10⁻¹) with a residual below 10⁻²; on (10⁻⁶, 10⁻⁵) the fit is noise | `check_critique_v3.py` [recomputed here] |
| Infrared cutoff | `omega_min_rad_s` defaults to 2π/T_total; on the T = 1 free-induction fixture χ(1/T)/χ(2π/T) = 1.06 × 10⁴, so the reported χ carries its cutoff and sensitivity | `check_critique_v3.py` [recomputed here] |
| Section 5.1.1 d_m = 8 row | max element difference 2 × 10⁻¹³ / 2 × 10⁻⁶ / 1 × 10⁻³ and analytic norm loss 3 × 10⁻⁸ / 6 × 10⁻³ / 0.3 at η = 0.1 / 0.5 / 1.0 | `check_critique_v3.py` [recomputed here] |
| POVM fast-path domain | product POVM equals the full record path at zero crosstalk; at 4.0% PSF leakage the discrepancy is bounded and reported | Section 5.7 [corrected: critique, 2026-09-04] |
| 355 nm second-order weights | signed 1/Δ_{1/2} + 2/Δ_{3/2} = 4.5 × 10⁻⁴ THz⁻¹ against the double-counted 6.0 × 10⁻² THz⁻¹, ratio 133, both positive | `check_critique_v3.py` [recomputed here] |
"""

SECTION_12_PARA = """**What the 2026-09-04 four-lens critique changed and what it left open.** The critique of Appendix D (four lenses, 64 findings) was folded on 2026-09-04 as edit batches 19a to 19e; every fix is marked **[corrected: critique, 2026-09-04]** where it lands and Section 9.17 carries its regression targets. What it did not settle is recorded here so that none of it is mistaken for done: the theorist's overall paragraph names a q-for-Q and a b-for-β slip in displayed equations of Section 4.1 without a line, and neither was located; three of the numerics critic's measurements are quoted from the critique and not yet reproduced by a committed script (the `mesolve`/`sesolve` wall-time ratio 69.5 at dimension 64, the `PicklingError` under `map="parallel"`, and the ENR displacement's ⟨n₀⟩ = 2.2412 against 2.25), so they carry no **[recomputed here]** tag; the Monroe 1995 beam geometry behind the Lamb-Dicke anchor is inferred from the quoted numbers (a 90° crossing reproduces them, a counter-propagating pair does not) and is to be confirmed against the paper's figure; the Doppler acceptance test of Section 4.2.1 (three n̄ from one Δ) is stated as the design and not yet computed; the 171Yb+ g_J the plan now adopts is a theoretical determination (Han et al. 2025) with no modern measurement behind it; and the cost model of Section 11 was rebuilt on the v4 benchmark, whose findings Section 11.2 and Appendix D record, including that the v3 table had timed a dense operator through four unmerged coefficient terms."""


def splice():
    # Section 13 rows
    p = HERE / "p13_conventions.md"
    t = p.read_text(encoding="utf-8")
    anchor = "\n\n**Run 5 amendments to the rows above**"
    if t.count(anchor) != 1:
        sys.exit(f"p13 anchor count {t.count(anchor)}")
    if "| Floquet function and rf phase origin |" not in t:
        t = t.replace(anchor, "\n" + NEW_13_ROWS + anchor)
        p.write_text(t, encoding="utf-8")
        print("appended 10 Section 13 rows")
    else:
        print("Section 13 rows already present")
    # Section 9.17
    p = HERE / "p09_validation.md"
    t = p.read_text(encoding="utf-8")
    if "### 9.17 " not in t:
        p.write_text(t.rstrip("\n") + "\n\n" + SECTION_9_17, encoding="utf-8")
        print("appended Section 9.17")
    else:
        print("Section 9.17 already present")
    # Section 12 paragraph
    p = HERE / "p12_risks.md"
    t = p.read_text(encoding="utf-8")
    if "**What the 2026-09-04 four-lens critique changed" not in t:
        p.write_text(t.rstrip("\n") + "\n\n" + SECTION_12_PARA + "\n", encoding="utf-8")
        print("appended Section 12 critique paragraph")
    else:
        print("Section 12 critique paragraph already present")
    # Section 5.1.1 table: the d_m = 8 row
    p = HERE / "p05_numerics.md"
    t = p.read_text(encoding="utf-8")
    row10 = "| 10 | 0.1 / 0.5 / 1.0 | 3 × 10⁻¹⁶ / 7 × 10⁻⁸ / 2 × 10⁻⁴ | 2 × 10⁻¹⁰ / 1 × 10⁻³ / 0.18 |"
    row8 = "| 8 | 0.1 / 0.5 / 1.0 | 2 × 10⁻¹³ / 2 × 10⁻⁶ / 1 × 10⁻³ | 3 × 10⁻⁸ / 6 × 10⁻³ / 0.3 |"
    if row8 not in t:
        if t.count(row10) != 1:
            sys.exit("5.1.1 table row for d_m = 10 not found once")
        t = t.replace(row10, row8 + "\n" + row10)
        p.write_text(t, encoding="utf-8")
        print("inserted the d_m = 8 row (check_critique_v3.py section O)")
    else:
        print("d_m = 8 row already present")
    # M3a block moves before M3
    p = HERE / "p10_roadmap.md"
    t = p.read_text(encoding="utf-8")
    i3 = t.find("### M3. Cooling and state preparation")
    i3a = t.find("### M3a. Multi-level optical Bloch builder")
    i4 = t.find("### M4. Two-ion entangling gates")
    if i3 == -1 or i3a == -1 or i4 == -1:
        sys.exit("roadmap anchors missing")
    if i3a > i3:
        block = t[i3a:i4]
        t = t[:i3a] + t[i4:]
        t = t.replace(
            "### M3. Cooling and state preparation",
            block.rstrip("\n") + "\n\n### M3. Cooling and state preparation",
            1,
        )
        p.write_text(t, encoding="utf-8")
        print("moved M3a before M3")
    else:
        print("M3a already precedes M3")


if __name__ == "__main__":
    apply(EDITS)
    splice()
