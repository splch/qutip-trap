"""Edit batch 19a (2026-09-04, critique v3 fold): Part II physics sections (4.1 to 4.6, 6.9).

Every edit answers a finding of the four-lens critique of 2026-09-04 (run5/critique_v3/findings.md); numbers marked
[recomputed here] are printed by validation/scripts/check_critique_v3.py.
"""

from edit_batch13_errata import apply

TAG = "**[corrected: critique, 2026-09-04]**"

EDITS = [
    # ---- 4.1.1: Mathieu sign of the Floquet function, u'(0), C0 ownership and size ----
    (
        "p04a_trap.md",
        "The secular motion is multiplied, not added to, by a micromotion modulation of fractional amplitude q_i/2 at Ω_T **[verified]**;",
        "The secular motion is multiplied, not added to, by a micromotion modulation of fractional amplitude q_i/2 at Ω_T **[verified]** (Wineland's rf phase origin, a_i + 2q_i cos Ω_T t; under the plan's adopted Mathieu sign, Section 13, every Ω_T t in that trajectory reads Ω_T t + π, the only change);",
    ),
    (
        "p04a_trap.md",
        "H = p²/2m + (m/2) W(t) x² with W(t) = (ω_rf²/4)[a_x + 2 q_x cos(ω_rf t)] is solved by",
        "H = p²/2m + (m/2) W(t) x² with W(t) = (ω_rf²/4)[a_x + 2 q_x cos(ω_rf t)] in the RMP's rf phase origin (the plan's adopted Mathieu sign has −2q_x cos(ω_rf t); see below) is solved by",
    ),
    (
        "p04a_trap.md",
        "To lowest order u(t) ≈ e^{iνt}[1 + (q_x/2) cos(ω_rf t)]/(1 + q_x/2). The first two revisions of this plan read the normalization (1 + q_x/2)^{−1} of that form",
        "To lowest order, in that rf phase origin, u(t) ≈ e^{iνt}[1 + (q_x/2) cos(ω_rf t)]/(1 + q_x/2); in the sign the plan adopts for the Mathieu equation (Section 13: d²x/dξ² + [a − 2q cos 2ξ]x = 0, so W(t) = (ω_rf²/4)[a_x − 2q_x cos(ω_rf t)]) the same solution reads u(t) ≈ e^{iνt}[1 − (q_x/2) cos(ω_rf t)]/(1 − q_x/2), the two differing by half an rf period in the time origin and in nothing else, and the simulator's rf phase reference is the adopted sign: the Floquet recursion gives c_{±1}/c₀ = −q_x/((2 ± β)² − a) under it (`check_critique_v3.py` integrates the Mathieu equation under both signs), so the in-phase micromotion at the rf phase origin is a contraction, x_μ(t) = −(q_x/2) x_sec(t) cos(ω_rf t), and the phase of every micromotion sideband and the sign with which a dc shim adds to Berkeland's irreducible φ_ac term (below) follow from that one statement; the regression is that the modulation index of Section 4.3.6 changes sign, a phase step of π in the rf-photon correlation signal, as the shim voltage crosses the compensated value (the 2026-09-04 experimentalist critique found this paragraph in the RMP's sign against Section 13's, which left every micromotion phase undetermined) "
        + TAG
        + ", **[recomputed here]**. The first two revisions of this plan read the normalization (1 + q_x/2)^{−1} of that form",
    ),
    (
        "p04a_trap.md",
        "(the exact u cannot satisfy both: scaled to u(0) = 1 it has u̇(0) = iν(1 − q))",
        "(the exact u cannot satisfy both: scaled to u(0) = 1 it has u̇(0) = iν(1 + q) in the adopted sign and iν(1 − q) in the RMP's rf phase origin, to first order in q; `check_critique_v3.py`)",
    ),
    (
        "p04a_trap.md",
        "The simulator's C₀ is therefore the Wronskian-normalized rf-period Fourier coefficient from the monodromy module, applied to η (Section 4.3), with 1 + 3q²/16 as its O(q²) check; the effect sits at the η⁴ level of gate errors, and the 5 to 13% reductions of η that the earlier revisions carried were spurious.",
        "The simulator's C₀ is therefore the Wronskian-normalized rf-period Fourier coefficient from the monodromy module, with 1 + 3q²/16 as its O(q²) check, and it multiplies η in exactly one place, `Crystal.lamb_dicke` (Appendix E), which records it in the provenance of every η; the drive builder of Section 4.3.6 and the Hamiltonian of Section 5.7 receive η with C₀ already inside and never reapply it, and a test compares η at q = 0.3 with η at q = 0 against 1 + 3q²/16 to O(q⁴) (the 2026-09-04 architect critique found three sections naming the multiplication and none naming its owner, so that η could have carried C₀² or C₀⁰). Its size is not negligible: C₀ shifts η by 1.8% at q = 0.3, first order in every sideband Rabi frequency and 3.7% in the Mølmer-Sørensen two-body phase (∝ η²Ω²), an uncalibrated infidelity near 10⁻³; what makes it harmless is that the sideband amplitude calibration of Section 7.5 absorbs it, so the defensible statement is that C₀ is absorbed by calibration, not that it sits at the η⁴ level as the second revision said, and the case in which it matters is a calibration made on the carrier rather than on the sideband, for which Section 9.17 pins the pseudopotential-versus-Floquet η difference as a validation case instead of dismissing it (2026-09-04 experimentalist critique) "
        + TAG
        + "; the 5 to 13% reductions of η that the earlier revisions carried were spurious.",
    ),
    # ---- 4.1.3: two transverse families ----
    (
        "p04a_trap.md",
        "which shares the eigenvectors of A and has eigenvalues γ_p = 1/α + 1/2 − μ_p/2 **[verified]**. The transverse frequencies are Ω_p = ω₃ √γ_p, so their ordering is reversed:",
        "which shares the eigenvectors of A and has eigenvalues γ_p = 1/α + 1/2 − μ_p/2 **[verified]**. A linear Paul trap has two transverse families, not one: with the two radial secular frequencies ω_x and ω_y along the principal axes (rotated about the trap axis by `Trap.axis_angle_rad`, Appendix E) the anisotropies are α_x = (ω₃/ω_x)² and α_y = (ω₃/ω_y)², the Hessians B^{(x)} and B^{(y)} carry 1/α_x and 1/α_y in the diagonal term and share A's eigenvectors, so γ_p^{(x)} = 1/α_x + 1/2 − μ_p/2 and γ_p^{(y)} = 1/α_y + 1/2 − μ_p/2 give two generally non-degenerate radial families of N modes each, identical eigenvectors and different frequencies Ω_p^{(x)} = ω₃√γ_p^{(x)} and Ω_p^{(y)} = ω₃√γ_p^{(y)}, which is why Section 4.4.7's reconciliation needs 2N radial modes, why `Crystal` declares 3N modes, and what the crystal module's unit test asserts (3N modes in three families; the 2026-09-04 experimentalist critique found only the single-α form written down, from which an implementer builds half the radial spectrum and leaves the other family's phase-space loops open) "
        + TAG
        + ". Within each family the transverse frequencies are Ω_p = ω₃ √γ_p, so their ordering is reversed:",
    ),
    # ---- 4.2.1: Doppler detuning is a beam parameter ----
    (
        "p04b_cooling.md",
        "with W from the multi-level Bloch steady state, valid at any ν/Γ, and optimizes Δ numerically; the force model is the ν ≪ Γ cross-check",
        "with W from the multi-level Bloch steady state, valid at any ν/Γ, and optimizes Δ numerically, once per beam over the whole mode set: a Doppler beam has one detuning, one saturation parameter and one direction for all 3N modes, so Δ, s and k̂ are beam parameters chosen to minimize the participation-weighted mean of the per-mode steady-state n̄ (weights b_{i,m}² of the gate modes the schedule uses next, uniform when none is scheduled), and the per-mode residual n̄ is reported, never optimized mode by mode, since a per-mode optimum is a state no single beam can prepare (the optimum Δ for a 0.5 MHz axial and a 4 MHz radial mode differ by more than Γ); the acceptance test is Monroe's three modes from his single Δ = −30 MHz beam, whose three measured n̄ one shared detuning has to reproduce together (2026-09-04 experimentalist critique) "
        + TAG
        + "; the force model is the ν ≪ Γ cross-check",
    ),
    # ---- 4.2.2: the A_pm carrier weight, g symbols, the Monroe anchor geometry ----
    (
        "p04b_cooling.md",
        "η̃²/η² = α k_em²/(b_{i,m} Δk·ê_m)²,",
        "η̃²/η² = α k_em²/(Δk·ê_m)²,",
    ),
    (
        "p04b_cooling.md",
        "the two coinciding only for one resonant beam along the mode axis with b_{i,m} = 1, where the weight reduces to α",
        "the two coinciding only for one resonant beam along the mode axis, where the weight reduces to α (the participation b_{i,m} multiplies both η̃ and η and cancels from the ratio, so the sideband-cooling floor of a given mode is independent of N, which is the regression; the second revision carried b_{i,m} in the denominator, inflating the floor by 1/b² for every multi-ion mode while every single-ion test passed at b = 1; 2026-09-04 experimentalist critique) "
        + TAG,
    ),
    (
        "p04b_cooling.md",
        "with two-photon Rabi frequency Ω = g₁g₂/Δ (in Monroe's convention P = sin²(Ωτ), so a π pulse is Ωτ = π/2, half the usual sin²(Ωt/2) convention **[verified flag]**)",
        "with two-photon Rabi frequency Ω = g₁g₂/Δ, g₁ and g₂ the single-photon couplings in the half convention g = Ω_single/2 (in Monroe's convention P = sin²(Ωτ), so a π pulse is Ωτ = π/2, half the usual sin²(Ωt/2) convention, and in plan units the same quantity is Ω₁Ω₂/(2Δ), Section 13's two-photon row and its g-symbol row **[verified flag]**)",
    ),
    (
        "p04b_cooling.md",
        "Lamb-Dicke parameters η_ν = δk_ν r_ν with δk = 2k for counter-propagating beams and r_ν = √(ħ/2mω_ν) were (0.21, 0.12, 0.09)",
        "Lamb-Dicke parameters η_ν = δk_ν r_ν with r_ν = √(ħ/2mω_ν) were (0.21, 0.12, 0.09), and the geometry that reproduces them is not counter-propagating: at 313 nm and 11.2 MHz, r_x = 7.08 nm, so δk = 2k gives 0.284 while a 90° crossing with δk = √2 k along x gives 0.2008, the quoted 0.21, and the y and z values follow to 8% and 3% from a projection k on each axis, a second 90° pair whose δk bisects y and z (`check_critique_v3.py`; the second revision's '2k for counter-propagating beams' does not reproduce the anchor and left the test unwritable, so the fixture records |Δk| = 2k sin(θ/2) and its direction cosines per axis; 2026-09-04 experimentalist critique) "
        + TAG
        + ", **[recomputed here]**",
    ),
    # ---- 4.2.4: PGC phase average, U_trap units ----
    (
        "p04b_cooling.md",
        "the two ground Zeeman states acquire U_± = U_trap + (1/3)Δs ∓ (1/3)Δs sin(2kz + 2φ), with U_trap = ½mω_z²z², Δ > 0 the detuning,",
        "the two ground Zeeman states acquire U_± = U_trap + (ħ/3)Δs ∓ (ħ/3)Δs sin(2kz + 2φ), with U_trap = ½mω_z²z² (ħ restored: the light shifts are frequencies and the trap term an energy, which the second revision's ħ = 1 form mixed; 2026-09-04 theorist critique), Δ > 0 the detuning,",
    ),
    (
        "p04b_cooling.md",
        "and phase-averaged, over many ions sampling the gradient or a moving one, ⟨n⟩ = (3/4)ξ + 5/(8ξ) − ½ with minimum √(15/8) − ½ = 0.8693064 at ξ = √(5/6) = 0.9128709 **[recomputed here]**.",
        "and phase-averaged, ⟨n⟩ = ⟨H⟩_φ/⟨W⟩_φ − ½ = (3/4)ξ + 5/(8ξ) − ½ with minimum √(15/8) − ½ = 0.8693064 at ξ = √(5/6) = 0.9128709 **[recomputed here]**, a form valid only for a gradient moving fast against W (the W < δ < ω_z window below), where every ion sees the averaged coefficients; for a static gradient sampled by many ions at fixed distinct φ each ion has its own steady state H(φ)/W(φ) − ½, W(φ) ∝ cos²2φ vanishes at φ = π/4, so ions near a node are not cooled at all and the ensemble mean diverges rather than reading 0.87, and the module averages H/W − ½ over the ions' actual φ and raises when any |cos 2φ| falls below the cooling threshold, the uncooled-mode failure Section 4.2 promises to raise on (the second revision applied the moving-gradient average to 'many ions sampling the gradient'; 2026-09-04 theorist critique) "
        + TAG
        + ".",
    ),
    # ---- 4.2.6: laboratory stage order ----
    (
        "p04b_cooling.md",
        "The procedure precedes Doppler cooling in the canonical sequence (pump to |↓⟩, then cool to about 1 mK leaving ⟨n⟩ ≥ 1, then sideband cool) **[corrected]** (Wineland 1998 prints ≥, not ≳).",
        "The laboratory order is Doppler cooling first, then sideband or EIT cooling with its own repump, then a final optical pump immediately before the circuit, with a re-pump after any interleaved recooling: Doppler cooling runs on a repumped cycling transition and scrambles the internal state (171Yb+ over S₁/₂ F = 1 through the 14.7 GHz sideband, 40Ca+ over both S₁/₂ Zeeman states), so a pump done first is erased, which is the order Section 2 and the 9Be+ anchor of Section 4.2.2 (50 µs Doppler precooling, then 7 µs of optical pumping, then Raman sideband cycles) already state; the second revision printed the reverse order here (pump to |↓⟩, cool to about 1 mK leaving ⟨n⟩ ≥ 1, sideband cool), which an implementer following it would have turned into a scrambled initial state and a preparation error orders of magnitude too large, so the stage order is a scheduler constraint (Section 7.3) and not prose (2026-09-04 experimentalist critique) "
        + TAG
        + " (Wineland 1998 prints ⟨n⟩ ≥ 1 after Doppler cooling, not ≳).",
    ),
    # ---- 4.2.8: stale Doppler producer, item (iii), item (iv), the EIT fixture ----
    (
        "p04b_cooling.md",
        "the simulator computes the Doppler stage from the force model with the actual α of the configured geometry, as Section 4.2.1 requires.",
        "n̄_D comes from the Section 4.2.2 rate coefficients evaluated with the configured α and cos²θ_L, as Section 4.2.1 requires, and the force-model value is reported only as the ν/Γ < 0.1 cross-check (the second revision's sentence here still named the force model as the producer, against 4.2.1; 2026-09-04 critique) "
        + TAG
        + ".",
    ),
    (
        "p04b_cooling.md",
        "and A_± = (Ω²/Γ)η²[cos²θ_L W(Δ ∓ ν) + αW(Δ)] below saturation, with η² factored out and the beam projection explicit (Eschner et al. 2003 Eq. 6): this is Section 4.2.2's form,",
        "and, below saturation, W(Δ) → Ω²Γ/(4Δ² + Γ²) = (Ω²/Γ)L(Δ) with the dimensionless Lorentzian L(Δ) = Γ²/(4Δ² + Γ²), so that A_± = (Ω²/Γ)[L(Δ ∓ ν) + (η̃²/η²)L(Δ)], which is Eschner et al. 2003 Eq. 6 once their unprojected η² and explicit cos²θ_L are folded into the plan's η_{i,m} (Δk·ê_m inside η, Section 4.1.3) and their α into (η̃/η)²; the code stores the bare A_± in s⁻¹ with W(Δ) = Γρ_ee from `light/bloch.py`, which returns a rate for both of its consumers, and η² stays outside A_± in the rate equation Ṗ(n) of Section 4.2.3 (the second revision printed (Ω²/Γ)η²[cos²θ_L W(Δ ∓ ν) + αW(Δ)] here, η² inside and the projection twice, which fed into 4.2.3 would have multiplied every cooling rate by a second η² while leaving every n̄ test unchanged; the test is that η²(A₋ − A₊) reproduces Lechner's 19 × 10³ s⁻¹; 2026-09-04 critique) "
        + TAG
        + ": this is Section 4.2.2's form,",
    ),
    (
        "p04b_cooling.md",
        "n̄_SB = (Γ/2ν)²[α/cos²θ_L + 1/4], which is Roos's Eq. 3.20 with (η̃/η)² = α(k_em/k_L)²/cos²θ_L, and α → 0 leaves (Γ/4ν)² from off-resonant blue-sideband excitation, not zero; at α = 2/5 the recoil term is 62% of n̄ and the blue sideband 38% **[verified]**.",
        "n̄_SB = (Γ/2ν)²[α(k_em/k_L)²/cos²θ_L + 1/4] = (Γ/2ν)²[(η̃/η)² + 1/4], which is Roos's Eq. 3.20 with (η̃/η)² = α(k_em/k_L)²/cos²θ_L, and α → 0 leaves (Γ/4ν)² from off-resonant blue-sideband excitation, not zero; at α = 2/5 with k_em = k_L and one on-axis beam the recoil term is 62% of n̄ and the blue sideband 38%, the closed-cycle special case, while the quenched 40Ca+ scheme of Section 4.2.2 (729 nm drive, 393 nm recoil photon, (k_em/k_L)² = 3.44, (η̃/η)² = 1.37) has bracket 1.62, the regression (the second revision's bracket here dropped the (k_em/k_L)² its own sentence names and returned 0.65 for that case, the very number 4.2.2 marks as the uncorrected value; 2026-09-04 theorist critique) **[verified]**, "
        + TAG
        + ".",
    ),
    (
        "p04b_cooling.md",
        "at her Fig. 3 parameters (ν = 2.0068, γ = 20, Ω₁ = Ω₂ = 17, Δ = −70 MHz) the rate coefficients give n̄ = 0.005102 against (γ/4|Δ|)² = 0.0051020 and her caption's 0.005, while the caption's printed Δ = +70 MHz returns 1.005, so the module is parameterized from her body text; her optimum Ω² = 4ν(ν − Δ) is leading order in γ/|Δ|",
        "at her Fig. 3 parameters (ν = 2.0068, γ = 20, Ω₁ = Ω₂ = 17 MHz, Δ_Morigi = −70 MHz, which is Δ = +70 MHz in the plan's sign, Section 13, because her optimum reads Ω² = 4ν(ν − Δ_Morigi) where Section 4.2.3's reads Ω_r² = 4ν(ν + Δ), so Δ = −Δ_Morigi) the rate coefficients give n̄ = 0.005102 against (γ/4|Δ|)² = 0.0051020 and her caption's 0.005, while the caption's printed Δ_Morigi = +70 MHz (Δ = −70 in the plan's sign) returns −1.005 from the Section 4.2.3 closed form, a negative occupation that trips its own A₋ − A₊ ≤ 0 guard, so the module is parameterized from her body text; the two Rabi frequencies enter that closed form as the quadrature sum Ω_r = √(Ω₁² + Ω₂²) = 24.04 MHz, the value that satisfies Ω_r² = 4ν(ν + Δ) = 578 at Δ = +70 (a single 17 in its place returns 0.147, 29 times the target; `check_critique_v3.py`), and since Ω₁ = Ω₂ sits outside the weak-probe regime Ω_g ≪ Ω_r of Section 4.2.3, level C rather than the closed form validates the figure (the second revision quoted the fixture in Morigi's sign with neither the sign map nor the composition rule; 2026-09-04 experimentalist critique) "
        + TAG
        + ", **[recomputed here]**; her optimum Ω² = 4ν(ν − Δ_Morigi) is leading order in γ/|Δ|",
    ),
    # ---- 4.3.1: the resonant map is Wineland's half convention ----
    (
        "p04c_hamiltonian.md",
        "The resonant map is |↓⟩|n⟩ → cos(Ω_{n,n′}t)|↓⟩|n⟩ − i e^{iφ} sin(Ω_{n,n′}t)|↑⟩|n′⟩ (Wineland 2003 Eq. 2.4);",
        "The resonant map is |↓⟩|n⟩ → cos(Ω_{n,n′}t)|↓⟩|n⟩ − i e^{iφ} sin(Ω_{n,n′}t)|↑⟩|n′⟩ in Wineland's half-Rabi convention (Wineland 2003 Eq. 2.4), which reads cos(Ω_{n,n′}t/2) and sin(Ω_{n,n′}t/2) in the plan's convention of the next paragraph, the conversion made once at ingest (2026-09-04 critique);",
    ),
    # ---- 4.3.5: concatenated durations ----
    (
        "p04c_hamiltonian.md",
        "so τ_SK1 = τ_BB1 = (4π + θ)/Ω and τ_CinSK = τ_CinBB = (8π + 2θ − 4k)/Ω,",
        "so τ_SK1 = τ_BB1 = (4π + θ)/Ω, τ_CORPSE = (4π + θ − 4k)/Ω and τ_CinSK = τ_CinBB = (8π + θ − 4k)/Ω (the CORPSE list sums to 4π + θ − 4k and each concatenation appends 4π of correctors, so at θ = π, k = π/6 the pulse list 420° + 300° + 60° + 360° + 360° = 1500° = 8π + π/3; the second revision printed 8π + 2θ − 4k, 12% long at θ = π, and `check_composite.py` now asserts every family's Σ_l θ_l against these forms; 2026-09-04 experimentalist critique) "
        + TAG
        + ", **[recomputed here]**,",
    ),
    # ---- 4.3.6: C0 not reapplied; rf phase reference ----
    (
        "p04c_hamiltonian.md",
        "The builder applies this as an optional multiplicative modulation of the drive coefficient and the intrinsic-micromotion C₀ rescaling of η (Section 4.1.1).",
        "The builder applies this as an optional multiplicative modulation of the drive coefficient; the intrinsic-micromotion factor C₀ (Section 4.1.1) is already inside the η the builder receives from `Crystal.lamb_dicke` and is never reapplied here (2026-09-04 architect critique). The modulation e^{iβ cos(ω_rf t + δ)} also needs the pulse start time relative to the trap rf: a drive flagged `rf_locked` carries an rf phase reference and keeps the coherent sideband, an unlocked one averages δ per shot so that only J₀ and the J_n² weights survive (`Drive.rf_phase`, Appendix E), and the default without a declared reference is the unlocked average, because a result that depended on the integrator's time origin would be an artifact (Section 12 had listed the rule as absent; 2026-09-04 experimentalist critique) "
        + TAG
        + ".",
    ),
    # ---- 4.3.7: sech widths, the explicit tone cut, the 355 nm weights ----
    (
        "p04c_hamiltonian.md",
        "τ is the sech **field**-envelope parameter, never a width: the intensity FWHM is 2 arccosh(2)/π = 0.8384014366 τ for a sech envelope and 0.5611 τ for sech² **[recomputed here]**.",
        "τ is the sech **field**-envelope parameter, never a width, and its envelope is sech(πt/(2τ)): the tooth formula sech(2πk ν_rep τ) is the Fourier transform of a field sech(πt/(2τ)) (FT[sech(at)] ∝ sech(πω/2a), and a = π/2τ gives sech(ωτ) = sech(2πkν_repτ) at ω = 2πkν_rep), so the field FWHM is (4/π) arccosh(2) τ = 1.6768 τ and the intensity (sech²) FWHM is (4/π) arccosh(√2) τ = 1.1222 τ; the second revision printed 0.8384 τ and 0.5611 τ, the widths of sech(πt/τ), whose teeth would read sech(πkν_repτ), so a τ inferred from a measured pulse width was off by 2 and with it every sech(ω_qτ/2) suppression and the 27% pair-versus-single test (`check_critique_v3.py`, which also checks the transform pair numerically; 2026-09-04 theorist critique) "
        + TAG
        + ", **[recomputed here]**.",
    ),
    (
        "p04c_hamiltonian.md",
        "Near-resonant tones stay explicit as `QobjEvo` coefficients while only far-detuned ones fold into the static fourth-order shift below, never both **[verified]**.",
        "Near-resonant tones stay explicit as `QobjEvo` coefficients while only far-detuned ones fold into the static fourth-order shift below, never both **[verified]**; the cut is explicit: a tone pair stays explicit when |μ_j − ω_res| ≲ 10/t_g (about 100 kHz for a 100 µs gate) and everything beyond folds into the l-sum shift, with a convergence test that moves the cut by 2× and reports the change in the gate phase, because the off-resonant beat notes sit at multiples of ω_rep = 2π × 120 MHz and retaining even the nearest one raises the highest frequency in the frame from about 3 MHz to 120 MHz, a 40× step-count multiplier that turns the Section 11.1 cost of a 100 µs gate at dimension 2048 into hours (2026-09-04 numerics critique) "
        + TAG
        + ".",
    ),
    (
        "p04c_hamiltonian.md",
        "and pairing −2 with one's own signed Δ_{3/2} double-counts it, flipping E^{(2)}_{00} from about +0.1 to −2 MHz at the near-total D1/D2 cancellation of 355 nm **[corrected]**.",
        "and pairing −2 with one's own signed Δ_{3/2} double-counts it, inflating E^{(2)}_{00} by a factor 133 at the near-total D1/D2 cancellation of 355 nm (1/33 + 2/(−67) = 0.00045 THz⁻¹ against 1/33 + 2/67 = 0.0602 THz⁻¹, both positive; the second revision printed this as a sign flip from about +0.1 to −2 MHz, which neither reading gives; `check_critique_v3.py`) **[corrected]**, "
        + TAG
        + ", **[recomputed here]**.",
    ),
    # ---- 4.4.3: exact integration through the PulseEngine ----
    (
        "p04d_gates.md",
        "and verifies every solution by exact integration of the full Hamiltonian, which also captures what the Lamb-Dicke solvers omit (carrier, Debye-Waller, off-resonant terms).",
        "and verifies every solution by exact integration of the full Hamiltonian through the injected `PulseEngine` protocol rather than by importing `dynamics` (control/ never imports dynamics/, which consumes its Pulse objects; the cycle the 2026-09-04 architect critique found), which also captures what the Lamb-Dicke solvers omit (carrier, Debye-Waller, off-resonant terms).",
    ),
    # ---- 4.4.4: Baldwin's geometric phase per loop ----
    (
        "p04d_gates.md",
        "The printed geometric phase Φ = 2π(Ωη/δ)² is inconsistent with the paper's own Eq. 1, which gives 8π(Ωη/δ)² **[corrected]**;",
        "The printed geometric phase Φ = 2π(Ωη/δ)² is the per-loop coefficient of Ŝ² in Φ_loop Ŝ², with Ŝ = σ_z² − σ_z¹ the force operator of the H two clauses above and Ŝ² = 2 − 2σ_z¹σ_z²: one closed loop of duration 2π/δ gives 2π(ηΩ/δ)² on Ŝ², hence 4π(ηΩ/δ)² on σ_z¹σ_z², and the 8π(Ωη/δ)² of the paper's Eq. 1 is the σ_zσ_z coefficient over the K = 2 loops of the spin echo; the two are consistent once the operator normalization and the loop count are named, which the second revision's bare 'inconsistent, gives 8π' did not do, leaving the calibrated Ω for the two U(π/4) segments ambiguous by 2 or 4, so the fixture integrates the printed H through the echo numerically and asserts diag(1, i, i, 1) (Section 9.17; 2026-09-04 theorist critique) **[corrected]**, "
        + TAG
        + ";",
    ),
    # ---- 4.4.7: the Roos table factor ----
    (
        "p04d_gates.md",
        "which gives (8/3)η, a factor of about 900 at η = 0.1,",
        "which gives (8/3)η, a factor 1/η² = 100 at η = 0.1 (the second revision printed 900; 2026-09-04 theorist critique),",
    ),
    # ---- 4.5.1: Breit-Rabi units, the quadratic shift, one g_J ----
    (
        "p04e_atomic.md",
        "E(F = I ± ½, m_F) = −ΔE_hfs/(2(2I+1)) + g_I μ_B m_F B ± (ΔE_hfs/2)[1 + 4m_F x/(2I+1) + x²]^{1/2},  x = (g_J − g_I) μ_B B/ΔE_hfs,  ΔE_hfs = A(I + ½),",
        "E(F = I ± ½, m_F)/h = −ΔE_hfs/(2(2I+1)) + (g_I μ_B/h) m_F B ± (ΔE_hfs/2)[1 + 4m_F x/(2I+1) + x²]^{1/2},  x = (g_J − g_I) μ_B B/(h ΔE_hfs),  ΔE_hfs = A(I + ½) with A in Hz (every term in Hz; the second revision added a joule to two Hz terms; 2026-09-04 theorist critique),",
    ),
    (
        "p04e_atomic.md",
        "and the 171Yb+ clock transition's quadratic shift is 310.85 Hz/G² both numerically and from the closed form (g_J − g_I)²μ_B²/(2hA), against the 310.8 Hz/G² of Section 6.3.",
        "and the 171Yb+ clock transition's quadratic shift is 310.9 Hz/G² both numerically and from the closed form (g_J − g_I)²μ_B²/(2h²A) with A = 12.642812118 GHz in Hz (the second revision printed 2hA, one factor of h short; 2026-09-04 theorist critique), against the 310.8 Hz/G² of Section 6.3. The coefficient moves by 0.03% across the three g_J values the second revision carried in different rows (2.00225664, which is 43Ca+'s; 2.00254; 2.00292, giving 310.76, 310.85 and 310.97 Hz/G²), so the plan adopts one cited value for the 171Yb+ ground state, g_J = 2.002615(70) (Han et al., arXiv:2501.09973, 2025, a multiconfiguration Dirac-Hartree-Fock and multireference configuration-interaction determination whose own second-order coefficient is 31.0869(22) mHz/µT² = 310.869 Hz/G²; the only other value in the literature is a spectroscopic 1.998 of larger uncertainty), which with g_I from μ_I = +0.49367 μ_N gives 310.87 Hz/G² and sets the test tolerance at ±0.02 Hz/G² from the g_J uncertainty, with the quoted 310.8 inside it (`check_critique_v3.py`) **[verified]**, "
        + TAG
        + ", **[recomputed here]**.",
    ),
    # ---- 4.5.5: the estimator that is identically zero ----
    (
        "p04e_atomic.md",
        "under-predicts the measured decoherence by a factor of five at the detuning where the two rates are equal **[corrected]**.",
        "under-predicts the measured decoherence at the detuning where the two rates are equal, where the estimator vanishes identically while the measured rate does not (the second revision quantified this as 'a factor of five', a ratio to zero; 2026-09-04 theorist critique) **[corrected]**.",
    ),
    # ---- 4.5.7: geometric-factor maxima, the Ramsey filter function ----
    (
        "p04e_atomic.md",
        "and g^{(±1)} = g^{(±2)} = 1/√6 at φ = 0 and at (90°, 90°), where only |Δm| = 2 is driven, a clean two-level system by construction,",
        "g^{(±1)} = 1/√6 at φ = 0 for every γ, where only |Δm| = 1 is driven, and g^{(±2)} = 1/√6 at (90°, 90°), where only |Δm| = 2 is driven, a clean two-level system by construction (the second revision's sentence gave both maxima at both geometries, which its own sum rule forbids, 2/3 against 1/3; 2026-09-04 theorist critique),",
    ),
    (
        "p04e_atomic.md",
        "but uses the Ramsey filter function C(T) = exp[−∫dω A(ω)² sin²(ωT/2)/ω²] with A(ω) the amplitude spectral density of the laser frequency noise, or a sampled phase trajectory with that spectrum (Section 6.3).",
        "but routes the coherence through Section 6.9's `filter_function` with the free-induction filter F = 4 sin²(ωT/2) and S_b = S_δ/4, the two-sided PSD of the σ_z coefficient (half the laser-frequency fluctuation), so that C(T) = exp[−(2/π)∫₀^∞ dω S_b(ω) F(ω)/ω²] in Section 6.9's normalization and under no second convention (the second revision printed a local form exp[−∫dω A(ω)² sin²(ωT/2)/ω²] with no prefactor, no sidedness and A undefined, which differs from the hyperfine path by 2/π or by 4 depending on how A² is read, the factor-4 trap Section 6.9 warns against, on the only default dephasing model of the 729 and 674 nm qubits; 2026-09-04 theorist critique) "
        + TAG
        + ", or a sampled phase trajectory with that spectrum (Section 6.3).",
    ),
    # ---- 4.6: the diabatic-splitting prefactor pairing ----
    (
        "p04e_atomic.md",
        "Eq. (23) also drops the exact prefactor 16^{1/5}/5 = 0.34822 and the minus sign of ∂d/∂α, harmless for the quadratic δE but not for directional use of δd_CP **[recomputed here]**.",
        "Eq. (23) also drops the exact prefactor 16^{1/5}/5 = 0.34822 and the minus sign of ∂d/∂α; the sign is harmless for the quadratic δE and matters only for directional use of δd_CP, but the prefactor is not, since δE ∝ (∂d/∂α)² and restoring it multiplies δE by 0.34822² = 0.1213, turning the chained π²/32 into π²/264, so the plan states one pairing end to end, the printed Eq. (25) with its π²/8 and the fitted ξ² ≈ 0.1 exactly as the source calibrated them (the Section 13 row), and never the chained π²/32 or the prefactor-restored π²/264, each of which needs its own refitted ξ² (about 0.4 and about 3.3, the latter no longer readable as a fractional diabatic impulse); the three pairings are mutually exclusive and the code refuses to mix them (the second revision called the prefactor harmless; 2026-09-04 theorist critique) **[recomputed here]**, "
        + TAG
        + ".",
    ),
    # ---- 6.9: the bang-bang exponent ----
    (
        "p06_noise.md",
        "In the bang-bang limit F ∝ (ωτ_π)^{2(α+1)}, that is 6(α + 1) dB/octave on the power convention:",
        "In the bang-bang limit F ∝ (ωτ)^{2(α+1)} with τ the sequence time (τ_π → 0 there, so τ_π cannot appear; the second revision printed ωτ_π; 2026-09-04 theorist critique), that is 6(α + 1) dB/octave on the power convention:",
    ),
]

if __name__ == "__main__":
    apply(EDITS)
