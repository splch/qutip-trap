"""Edit batch 18 (2026-09-04): the derivation audit's final report - the 171Yb+ bright-rate prefactor, the Bermudez
budget, the annotations, Section 9.16, the Appendix D paragraph and the Section 12 note."""

import json
from pathlib import Path

from edit_batch13_errata import apply

HERE = Path(__file__).resolve().parent
AUD = "**[corrected: derivation audit, 2026-09-04]**"
REPORT = json.load(
    open(
        "/private/tmp/claude-501/-Users-churchill-Repositories-qutip-trap/b2f1cadd-ba53-4734-9dbe-b340274f9083/scratchpad/run5/audit_digest/audit_report.json",
        encoding="utf-8",
    )
)

EDITS = [
    # ---- 8-1: 171Yb+ bright-state rate ----
    (
        "p08_readout.md",
        "R_∘ = (Γ/6) s_∘ / [1 + (2/3) s_∘ + (2Δ/Γ)²],  s_∘ = 2Ω²/Γ²,",
        "R_∘ = (Γ/18) s_∘ / [1 + (2/9) s_∘ + (2Δ/Γ)²],  s_∘ = 2Ω²/Γ² = I/I_sat,  equivalently R_∘ = (Γ/4) s̃/[1 + s̃ + (2Δ/Γ)²] with s̃ = (2/9)s_∘ = I/(229 mW/cm²),",
    ),
    (
        "p08_readout.md",
        "with Γ = 2π × 19.6 MHz, saturating at Γ/4 (three ground plus one excited level), not Γ/2 **[verified]**. The Clebsch-Gordan factor of the F=1 → F′=0 line is folded into the Γ/6 prefactor, so a simulator that computes Ω from intensity and a dipole matrix element must not apply it twice **[verified, contested normalization]**; Section 4.5.2 derives that prefactor (relative strength 1/3 per Zeeman component of a J = J′ line gives an excited fraction s_∘/6 at low intensity with s_∘ = 2Ω²/Γ² for the full-line Ω, and Γ/4 at saturation), which settles the normalization in Noek's favour and makes the (Γ/6)s_∘ form a computed result of the level-C Bloch solve rather than an input.",
        "with Γ = 2π × 19.6 MHz, Ω the full J = 1/2 → J′ = 1/2 line Rabi frequency Ω = |⟨J‖d‖J′⟩|E₀/ħ with no angular factor, I_sat = πhcΓ/(3λ³) = 50.8 mW/cm² at 369.5 nm, saturating at Γ/4 = 30.79 µs⁻¹ (three ground plus one excited level), not Γ/2, with half-maximum at s_∘ = 4.5 **[verified]**, "
        + AUD
        + ". Noek's printed form (Γ/6)s/[1 + (2/3)s + (2Δ/Γ)²] is the same four-level result written for s = s_∘/3, so a simulator that computes Ω from intensity and a dipole matrix element must apply the angular factors exactly once: the prefactor Γ/18 = Γ × ½ × (1/3) × (1/3) carries two independent factors of 1/3, the hyperfine reduced element |⟨F = 1‖d‖F′ = 0⟩|² = |⟨J‖d‖J′⟩|²/3 and the polarization share |ε_q|² = 1/3 of each ground sublevel's matching component at the magic angle 54.74° to B, which is what 'optimally polarized' means here (pure σ± light, or a polarization parallel or perpendicular to B, leaves a dark state and R_∘ = 0). The second revision of this plan wrote the prefactor as Γ/6 with (2/3)s_∘, folding one factor of 1/3 and not the other, which the derivation audit of 2026-09-04 caught: the exact four-level Lindblad steady state (sixteen-dimensional Liouvillian, magic-angle polarization, pumping rate ≪ Zeeman splitting ≪ Γ) sits at 0.3335, 0.3332 and 0.331 of the second revision's form and at 0.99999, 0.9993 and 0.992 of the corrected one over three decades of s_∘, and maximizing that steady state over polarization, field and detuning at fixed s_∘ reproduces (Γ/18, 2/9) to nine digits and never reaches the old value; Section 4.5.2 derives the prefactor and the level-C Bloch solve computes it.",
    ),
    (
        "p04e_atomic.md",
        "each ground sublevel couples with Ω/√3; in a rate model with the three ground states equally populated the excited fraction is 3 × (1/3) × (Ω²/3)/Γ² × [1 + (2Δ/Γ)²]⁻¹ = s_∘/6 at low intensity, which is Noek's (Γ/6)s_∘ prefactor of Section 8.1, and the saturation ceiling of Γ/4 follows from four equally populated levels. Noek's prefactor is therefore derived, not fitted, his s_∘ = 2Ω²/Γ² is I/I_sat with the two-level I_sat, and the factor 1.35 between Crain's quoted I/I_sat and the s_∘ inferred from his measured rate (Section 8.8) must be intensity calibration, detuning or polarization, all of which the level-C Bloch solve of Section 8.1 takes as inputs.",
        "each ground sublevel |1, m⟩ couples to |0, 0⟩ through one polarization component with |⟨0,0|d_q|1,m⟩|² = |⟨J‖d‖J′⟩|²/3, so its Rabi frequency is Ω_m² = |ε_q|²Ω²/3, and since Σ_q|ε_q|² = 1 for any single beam the three sublevels share Σ_m Ω_m² = Ω²/3 however the beam is polarized, the magic-angle choice |ε_q|² = 1/3 giving each Ω_m² = Ω²/9; in a rate model with the three ground states equally populated the excited fraction is Σ_m (1/3)(Ω_m²/Γ²)[1 + (2Δ/Γ)²]⁻¹ = (Ω²/9Γ²)[1 + (2Δ/Γ)²]⁻¹ = s_∘/18 at low intensity with s_∘ = 2Ω²/Γ² for the full-line Ω, which is the Γ/18 prefactor of Section 8.1 (the second revision wrote Ω/√3 per sublevel and s_∘/6, counting the hyperfine factor but not the polarization share; derivation audit of 2026-09-04), and the saturation ceiling of Γ/4 follows from four equally populated levels. Noek's prefactor is therefore derived, not fitted, his s = 2Ω²/Γ² in the (Γ/6, 2/3) form being s_∘/3, and the factor between Crain's quoted I/I_sat and the saturation inferred from his measured rate (Section 8.8) must be intensity calibration, detuning or polarization, all of which the level-C Bloch solve of Section 8.1 takes as inputs "
        + AUD
        + ".",
    ),
    (
        "p08_readout.md",
        "- **Two saturation parameters, never aliased.** Noek's s_∘ = 2Ω²/Γ² is not I/I_sat: pinning s_∘ from Crain's measured detected rate (472 kcps at ε_sys = 4.356%) gives s_∘ = 0.815 against the quoted I/I_sat = 1.10, a factor 1.35 apart. Acton's s is I/I_sat. The device model stores both and the intensity-to-Ω chain is marked uncalibrated until the optical-Bloch check (Section 8.1) closes the gap **[corrected]**; Section 4.5.2 shows that s_∘ is I/I_sat with the two-level I_sat and the 1/3 angular factor applied per component, so the residual 1.35 is an apparatus quantity (intensity calibration, detuning or polarization impurity), not a missing factor in the physics.",
        "- **Two saturation parameters, never aliased.** Noek's s = 2Ω²/Γ² in his (Γ/6, 2/3) form is s_∘/3 = I/(3I_sat) in the plan's full-line convention (Section 8.1): pinning it from Crain's measured detected rate (472 kcps at ε_sys = 4.356%) gives s = 0.815, that is s_∘ = 2.45, against the quoted I/I_sat = 1.10 at 56.2 mW/cm² (the two-level I_sat = 51 mW/cm² convention), a factor 2.2 in intensity (the second revision, with a prefactor one factor of 1/3 too large, made it 1.35 in the other direction). Acton's s is I/I_sat. Because the corrected R_∘ is the maximum of the exact four-level steady state over polarization, field and detuning at fixed intensity, a measured rate above it can only mean more light at the ion or a higher ε_sys than stated, so the residual is an apparatus quantity (intensity calibration of a focused beam, polarization impurity, detection efficiency), not a missing factor in the physics; the device model stores both parameters and the intensity-to-Ω chain is marked uncalibrated until the optical-Bloch check (Section 8.1) closes the gap "
        + AUD
        + ".",
    ),
    (
        "p13_conventions.md",
        "171Yb+ detection saturates at Γ/4 with ρ_ee → s_∘/6 at low intensity [corrected] |",
        "171Yb+ detection saturates at Γ/4 with ρ_ee → s_∘/18 at low intensity for s_∘ = I/I_sat (s_∘/6 in the second revision, one polarization factor of 1/3 short; derivation audit 2026-09-04) [corrected] |",
    ),
    (
        "p13_conventions.md",
        "Noek's s_∘ = 2Ω²/Γ² is not I/I_sat off the cycling line (Section 8.8) [corrected] |",
        "Noek's s in the (Γ/6, 2/3) detection form is I/(3I_sat), the exact four-level result with one polarization share per sublevel (Sections 8.1, 8.8; derivation audit 2026-09-04) [corrected] |",
    ),
    # ---- 4.4-8: Bermudez budget ----
    (
        "p04d_gates.md",
        "Bermudez et al. 2017 give the closed forms the simulator's noise module reproduces for an N-ion MS gate **[verified]**, with the caveat",
        "Bermudez et al. 2017 give the closed forms that the second revision of this plan took as the noise module's targets for an N-ion MS gate **[verified as transcribed]**, with the caveat",
    ),
    (
        "p04d_gates.md",
        "and the recalibration rule t_g(n̄_f) = t_g(0)(1 + η²(2n̄ + 1)/N) after heating **[verified]**, together with the mapping of the total ε onto Pauli depolarizing channels for circuit-level simulation **[verified]**.",
        "and the recalibration rule t_g(n̄_f) = t_g(0)(1 + η²(2n̄ + 1)/N) after heating **[verified as transcribed]**, together with the mapping of the total ε onto Pauli depolarizing channels for circuit-level simulation **[verified]**. The derivation audit of 2026-09-04 re-derived these budgets from the exact gate dynamics (two blind derivations, a comparer and two adjudicators, one of whom held out for the printed forms) and found them to be scale estimates with fitted coefficients rather than derivable results, so the plan's targets are the derived forms below and Bermudez's are kept as the source's statements: the spectator-loop error is ε_loop = Σ_{m≠g}(2n_m + 1)Σ_i|α_{i,m}(t_g)|² = π²NK Σ_{m≠g}(2n_m + 1)(ω_z/ω_m) sin²(δ_m t_g/2)/(δ_m t_g)² with δ_m = μ − ω_m (Bermudez's πN(δ − ω_z)/(2ω_z²t_g) = π²NK/(ω_z t_g)² is the right scale with the mode sum collapsed into the constant 0.8, and the thermal weight is 2n_m + 1, not 0.8(n̄ + 1)); the Debye-Waller error is (π²/8)N(N − 1)(η⁴/N²)(n̄² + n̄), the bracket being Var(n) rather than 1.2n̄² + 1.4n̄ (28 to 34% high); the dephasing error is ε_d = (Γ_d t_g/2)Σ_ij C_ij Cov_t(σ_z^i, σ_z^j) ≤ (Γ_d t_g/2)Σ_ij C_ij, so Nt_g/(2T₂) for uncorrelated and N²t_g/(4T₂) for global noise with T₂ the single-ion 1/e coherence time, 4 and 8 times below the printed 2t_gN/T₂ and 2t_gN²/T₂ (no single convention for Γ_d rescues both printed forms at once); the intensity-noise spin term is Γ_I t_g η² · 3(N − 1)/(8K), not (N − 1)/4, because the phase accumulates as (1 − cos δt) and ∫(1 − cos δt)² dt = (3/2)t_g, a Γ_I-independent ratio of the two terms (fitted intercept-to-slope ratios 1.003 to 1.010 of the derived form against 1.5/K of the printed one); and the recalibration rule keeps η²(2n̄ + 1) but not the 1/N. Section 9.16 pins the derived forms against exact integration "
        + AUD
        + ".",
    ),
    (
        "p04d_gates.md",
        "Section 6 turns these into collapse operators and sampled parameters, and Section 9.4 requires the simulated gate to reproduce each closed form in its regime of validity.",
        "Section 6 turns these into collapse operators and sampled parameters, and Sections 9.4 and 9.16 require the simulated gate to reproduce each derived closed form in its regime of validity.",
    ),
    (
        "p04d_gates.md",
        "(3) Trout's circuit-level dephasing rate r_d = 15 s⁻¹ and the 0.5 s T₂ it cites are reconciled by Bermudez's global-noise branch, T₂ = 2N²/r_d = 0.53 s at N = 2, not by the naive 1/(2r_d) = 33 ms **[corrected]**.",
        "(3) Trout's circuit-level dephasing rate r_d = 15 s⁻¹ and the 0.5 s T₂ it cites were reconciled by the second revision through Bermudez's global-noise branch, T₂ = 2N²/r_d = 0.53 s at N = 2; with the derived global coefficient N²t_g/(4T₂) of Section 4.4.7 the same pair gives T₂ = N²/(4r_d) = 0.067 s, so the reconciliation no longer closes and the pair (15 s⁻¹, 0.5 s) is recorded as unreconciled, the naive 1/(2r_d) = 33 ms being no better "
        + AUD
        + ".",
    ),
    (
        "p09_validation.md",
        "naive 0.0333 s | Trout 2018; Bermudez 2017 [corrected] |",
        "naive 0.0333 s; with the derived global coefficient N²t_g/(4T₂) the pair gives 0.067 s and is unreconciled (Section 4.4.7) | Trout 2018; Bermudez 2017 [corrected]; [corrected: derivation audit, 2026-09-04] |",
    ),
    (
        "p06_noise.md",
        "so that the gate error goes as 2t_gN/T₂ (uncorrelated) or 2t_gN²/T₂ (global) (Bermudez 2017) **[verified]**;",
        "so that the gate error goes as Nt_g/(2T₂) (uncorrelated) or N²t_g/(4T₂) (global), with T₂ the single-ion 1/e coherence time (Bermudez 2017 prints 2t_gN/T₂ and 2t_gN²/T₂, 4 and 8 times high; Section 4.4.7) "
        + AUD
        + ";",
    ),
    (
        "p06_noise.md",
        "fast intensity noise with zero-frequency density Γ_I gives Bermudez's ε_I ≈ Γ_I t_g η²(n̄ + ½) + Γ_I t_g η²(N − 1)/4 **[verified]**.",
        "fast intensity noise with zero-frequency density Γ_I gives ε_I ≈ Γ_I t_g η²(n̄ + ½) + Γ_I t_g η² · 3(N − 1)/(8K) (Bermudez 2017 prints (N − 1)/4 for the second term; Section 4.4.7) "
        + AUD
        + ".",
    ),
    (
        "p09_validation.md",
        "| Dephasing correlation | 2t_gN/T₂ local; 2t_gN²/T₂ global | Bermudez 2017 [verified] |",
        "| Dephasing correlation | Nt_g/(2T₂) local; N²t_g/(4T₂) global (Bermudez's printed 2t_gN/T₂ and 2t_gN²/T₂ are the labelled source forms) | Bermudez 2017 [corrected: derivation audit, 2026-09-04] |",
    ),
    (
        "p09_validation.md",
        "| Intensity noise | Bermudez ε_I two terms | Bermudez 2017 [verified] |",
        "| Intensity noise | ε_I = Γ_I t_g η²(n̄ + ½) + Γ_I t_g η² · 3(N − 1)/(8K); the fitted intercept-to-slope ratio is Γ_I-independent | Bermudez 2017 [corrected: derivation audit, 2026-09-04] |",
    ),
    # ---- annotations ----
    (
        "p04d_gates.md",
        "thermal ε_n̄ = (1/4)π²η⁴n̄(2n̄ + 1) for n̄ < 1;",
        "thermal ε_n̄ = (1/4)π²η⁴n̄(2n̄ + 1) = (π²/4)η⁴⟨n²⟩ for n̄ < 1, referenced to an entangling angle calibrated at n = 0 (calibrating the power on the thermal ensemble instead leaves the variance form (π²/4)η⁴n̄(n̄ + 1); the two agree to O(n̄), differing by 2% at n̄ = 0.02 and 50% at n̄ = 1, and the Debye-Waller law χ(n) = χ₀[1 − η²(2n + 1)] behind both was confirmed by the derivation audit of 2026-09-04);",
    ),
    (
        "p04d_gates.md",
        "The XX gate on ions a, b requires χ_{ab}(τ_g) = π/4 and α_{a,m}(τ_g) = 0 for all N modes",
        "The XX gate on ions a, b requires |χ_{ab}(τ_g)| = π/4 modulo π/2, the sign s being recorded by the compiler so that a solver landing on −π/4 is not scored as a failure (derivation audit of 2026-09-04), and α_{a,m}(τ_g) = 0 for all N modes",
    ),
    (
        "p04e_atomic.md",
        "the D₃/₂ lifetime 52.7 ms with its 935.2 nm repump and 3.07 GHz sideband,",
        "the D₃/₂ lifetime 52.7 ms with its 935.2 nm repump and 3.07 GHz sideband (the P₁/₂ → D₃/₂ decay channel itself lies at 2.438 µm, the 935.2 nm line being D₃/₂ → ³D[3/2]₁/₂, a conflation that changes the inferred P₁/₂ → D₃/₂ reduced element by 4.2 times and that one derivation of the 2026-09-04 audit fell into, so the species table stores the channel wavelength next to its element),",
    ),
    (
        "p04e_atomic.md",
        "(87Rb D2: 2.989311 against 4.227524 e a₀, so substituting one for the other moves every rate by 2)",
        "(87Rb D2: 2.989311 against 4.227524 e a₀ at τ = 26.2348(77) ns, the elements being derived from the stored lifetime with its uncertainty and never typed in, since the rounded 26.24 ns already gives 4.227104 e a₀; so substituting one element for the other moves every rate by 2)",
    ),
]

COMMITTED = {
    "4.1-7": "`check_c0_floquet.py`",
    "4.4-4": "`check_ms_closure.py`",
    "13-8": "`check_ms_closure.py`",
}


def clean(s):
    return str(s).replace("|", "∣").replace("\n", " ").strip()


def section_9_16():
    rows = []
    for t in REPORT["newTests"]:
        eq = t["equationId"]
        src = "derivation audit 2026-09-04"
        src += (
            (", re-derived in " + COMMITTED[eq])
            if eq in COMMITTED
            else " (the audit agents' scripts live in the run transcript and are not committed; the row is the implementer's target)"
        )
        rows.append(f"| {eq}: {clean(t['test'])} | {clean(t['expected'])} | {src} |")
    return (
        "### 9.16 Derivation-audit targets (Appendix D, 2026-09-04)\n\n"
        "The blind re-derivation audit of Appendix D confirmed 37 of 58 load-bearing equations and corrected 16; the rows below are the numerical checks its derivers, comparers and adjudicators ran to discriminate the plan's forms from the corrected ones, written as the test suite must reproduce them. Each row names the equation by its audit id (section-index of the harvest). Two of them are re-derived by committed scripts (`check_c0_floquet.py`, `check_ms_closure.py`); the rest were computed by the audit's agents in the run's own scripts and are recorded here as targets an implementer reproduces, not as **[recomputed here]** numbers, since Appendix D's rule is that the tag requires a committed script.\n\n"
        "| Test | Expected | Source, tag |\n|---|---|---|\n" + "\n".join(rows) + "\n"
    )


APPENDIX_D = "\n\n**Derivation audit** (wf_42fdefe6-fb4, 236 agents, 23.9 M subagent tokens, 3275 tool uses, 249 min, 2026-09-04): the check that no earlier run had made, and the reason this revision exists. Nine harvesters read Sections 4.1 to 4.5, 5, 6, 8 and 13 and listed 126 load-bearing equations with self-contained setups that state the model and conventions but not the result; the 60 highest-ranked (quotas of 4 to 8 per section) went to two derivers each, one analytic and one numerical, forbidden to read the plan or the web and told to derive from the stated conventions and first principles, the numerical one verifying its result in QuTiP, sympy or mpmath; a comparer then judged the plan's form against both derivations (agree, plan wrong, derivations wrong, convention mismatch, undecidable), settling every disagreement by computation, and every non-agree verdict went to two adversarial adjudicators instructed to overturn the comparer, with the majority of three deciding; one agent wrote the report. Eight agents died on the output-length caps of the schemas (five derivers, one adjudicator, two comparers), so two equations, the transverse Hessian of Section 4.1.3 and the emission angular factors of Section 4.2.8, lost their comparison and 58 were audited; four rows rest on a single blind derivation and one on a single adjudicator, and about fifteen derivers reported their setup insufficient, so a few rows rest on a guessed rather than a stated convention. Outcome: 37 equations confirmed as written (re-derivations, not consensus), 16 wrong, 4 convention mismatches (the Ballance thermal infidelity's calibration reference, the R_d and R_b naming, the φ_tone sign convention, the ħ in the displayed master equation) and 1 undecidable (which triangle of the displacement matrix is 'above' the diagonal). Twelve of the sixteen change a simulator result: the micromotion factor on the Lamb-Dicke parameter (Section 4.1.1; the rf-clock artifact behind (1 + q/2)⁻¹), the carrier weight in the sideband-cooling rate coefficients (4.2.2), the modulus in the Laguerre Rabi element (4.3.1), the π/2 between the force axis and the carrier axis (4.3.4), one factor 2 shared by the Mølmer-Sørensen force prefactor, its closure ratio and Roos's Ω_c (4.4.1 and 13, entered when the second revision doubled the per-tone coupling ηΩ/2 into ηΩ and then 'halved on ingest'), the symmetrized two-body kernel (4.4.3), the Steck partial-versus-total rate clause (4.5.6), √Γ_e inside the Kramers-Heisenberg sum (4.5.5), the rank-2 tensor conjugation behind q = m − m′ (4.5.7) and the 171Yb+ bright-state prefactor (8.1, a factor 3 from one missed polarization share); four change a stated number or test target: the Bermudez budget coefficients (4.4.7, up to 8×), the 43Ca+ quadratic Zeeman convention (6.3), α₁ = 11/16 for motional dephasing (6.2) and the pseudopotential amplitude failure note (13). Three of the corrections overturned statements that the 2026-09-04 critique had itself introduced or endorsed (the MS closure, the timing-benchmark fixture and the C₀ regression numbers), and every one of the twelve had carried a **[verified]** or **[corrected]** tag from the source-fidelity runs, which is the empirical content of Appendix D's warning that fidelity to a source is not correctness. The author re-derived two of them independently before adopting any (`check_ms_closure.py`, `check_c0_floquet.py`), re-ran the timing benchmark at the corrected closure (`bench_ms_timing_v3.py`), and applied the rest as the edit batches recorded beside the plan sources, each marked **[corrected: derivation audit, 2026-09-04]** in the text; Section 9.16 carries the audit's thirty discriminating tests. What the audit does not establish is stated in Section 12."

SECTION_12 = "\n\n**What the derivation audit does not establish (2026-09-04).** The audit re-derived equations from the plan's stated conventions and confirmed or corrected them numerically; it did not check citations against the source PDFs, so the provenance claims about printed errors (Wineland's Eq. 62, Brownnutt's Eq. 14, Turchette's Eq. 2) rest on the errata check and on internal consistency; it left every empirical species input unverified (ε_sys presets, background rates, measured branching ratios, reduced-element values are ingest data, not derivable quantities, and both derivers of the readout rows said so); it audited equations, not the code that will implement them; four rows rest on one blind derivation and one on one adjudicator, and about fifteen setups were judged insufficient, so a handful of confirmations rest on a guessed convention that happened to match. Five items it left open are recorded here so that they are not mistaken for settled: which triangle of the displacement matrix ⟨n′|D|n⟩ a plan sentence calls 'above' the diagonal (now stated explicitly in Section 5.1.1, but the modulus is transpose-symmetric and nothing computable depends on it); the 171Yb+ P₁/₂ → D₃/₂ reduced element, whose value hinges on using the 2.438 µm channel wavelength rather than the 935 nm repump (Section 4.5.6 now pins it); the number of digits at which the 87Rb reference elements are quoted, which follow the stored lifetime (Section 4.5.6); the sign branch of the entangling angle, which Section 4.4.3 now states as |χ| = π/4 modulo π/2 with the compiler's sign; and the apparatus quantities of Section 8 (detection efficiencies, background rates, the D₃/₂ branching), which no derivation can supply. The two equations that lost their comparer to the output caps, the transverse Hessian of Section 4.1.3 and the emission angular factors of Section 4.2.8, keep their earlier tags and the committed check scripts behind them (`check_recoil.py`, `check_surface_mixed.py`), and are the first candidates for a second audit pass."

if __name__ == "__main__":
    apply(EDITS)
    p = HERE / "p09_validation.md"
    t = p.read_text(encoding="utf-8")
    if "### 9.16 " not in t:
        p.write_text(t.rstrip("\n") + "\n\n" + section_9_16(), encoding="utf-8")
        print("Section 9.16 written with", len(REPORT["newTests"]), "rows")
    p = HERE / "p16_runs.md"
    t = p.read_text(encoding="utf-8")
    if "**Derivation audit** (wf_42fdefe6-fb4" not in t:
        p.write_text(t.rstrip("\n") + APPENDIX_D + "\n", encoding="utf-8")
        print("Appendix D derivation-audit paragraph appended")
    p = HERE / "p12_risks.md"
    t = p.read_text(encoding="utf-8")
    if "**What the derivation audit does not establish" not in t:
        p.write_text(t.rstrip("\n") + SECTION_12 + "\n", encoding="utf-8")
        print("Section 12 audit paragraph appended")
