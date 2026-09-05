"""Edit batch 15 (2026-09-04): derivation-audit corrections with two concurring adjudicators or confirmed by
check_ms_closure.py / check_c0_floquet.py. Idempotent exact-match edits (see edit_batch13_errata.apply)."""

from edit_batch13_errata import apply

AUD = "**[corrected: derivation audit, 2026-09-04]**"

EDITS = [
    # ---- A. micromotion factor C0 (4.1.1, 9.2, 9.10, 12, 13) ----
    (
        "p04a_trap.md",
        "To lowest order u(t) ≈ e^{iνt}[1 + (q_x/2) cos(ω_rf t)]/(1 + q_x/2), whose normalization C₀ = (1 + q_x/2)^{−1} is the factor that rescales the effective Lamb-Dicke parameter, η_eff = η C₀ **[corrected]** (Section 4.3 explains why it belongs on η, not on the carrier Rabi frequency). The simulator's C₀ is the rf-period average of the exact Floquet function u(t) normalized to u(0) = 1, computed by the monodromy module; (1 + q/2)⁻¹ is its lowest-order check, and the two differ by 0.05%, 0.26% and 0.78% at q = 0.1, 0.2 and 0.3 (Section 9.10), which matters at the η⁴ level of gate errors.",
        "To lowest order u(t) ≈ e^{iνt}[1 + (q_x/2) cos(ω_rf t)]/(1 + q_x/2). The first two revisions of this plan read the normalization (1 + q_x/2)^{−1} of that form as a factor C₀ rescaling the effective Lamb-Dicke parameter, η_eff = ηC₀, following the RMP; the derivation audit of 2026-09-04 (Appendix D) showed that this is an artifact of normalizing u to u(0) = 1 at one rf phase. With x₀ = √(ħ/2mν) and [â, â†] = 1 the canonical commutator [x̂, p̂] = iħ fixes the normalization of the exact Floquet function through its Wronskian, Im(u* u̇) = ν, not through u(0) = 1 (the exact u cannot satisfy both: scaled to u(0) = 1 it has u̇(0) = iν(1 − q)), and under that normalization the e^{iνt} Fourier component of u, which is what the secular sideband couples to, is C₀ = [Σ_n c_n²(1 + nω_rf/ν)]^{−1/2} = 1 + 3q_x²/16 + O(q⁴), slightly greater than one, even in q and with no first-order term: 1.001890, 1.007741 and 1.018161 at q = 0.1, 0.2 and 0.3 against 0.952, 0.909 and 0.870 for (1 + q/2)⁻¹ (`check_c0_floquet.py`; the u(0) = 1 average depends on the rf phase assigned to t = 0 and runs from 0.86 to 1.18 at q = 0.3 as that phase moves by half an rf period, while a measured sideband Rabi frequency cannot depend on the rf clock) "
        + AUD
        + ", **[recomputed here]**. The simulator's C₀ is therefore the Wronskian-normalized rf-period Fourier coefficient from the monodromy module, applied to η (Section 4.3), with 1 + 3q²/16 as its O(q²) check; the effect sits at the η⁴ level of gate errors, and the 5 to 13% reductions of η that the earlier revisions carried were spurious.",
    ),
    (
        "p12_risks.md",
        "with the C0 normalization correction [corrected] applied to effective Lamb-Dicke parameters.",
        "with the Wronskian-normalized micromotion factor C₀ = 1 + 3q²/16 + O(q⁴) applied to effective Lamb-Dicke parameters (the (1 + q/2)⁻¹ of the earlier revisions was an rf-clock artifact) [corrected: derivation audit, 2026-09-04].",
    ),
    (
        "p13_conventions.md",
        '| Micromotion correction | η_eff = ηC₀ with C₀ the rf-period average of the Floquet function u(t) from the monodromy module; (1 + q/2)⁻¹ is the O(q) check (0.906701 against 0.909091 at q = 0.2); excess micromotion as phase modulation with index φ_Ω | RMP Eq. 37; Wineland Eqs. 71-72; Berkeland Eq. 21 (misprint corrected) | RMP attaches C₀ to Ω₀; "Debye-Waller" is a different factor [corrected]; pdftotext renders φ_Ω as φ_V [corrected] |',
        '| Micromotion correction | η_eff = ηC₀ with C₀ the e^{iνt} Fourier coefficient of the exact Floquet function normalized by its Wronskian, Im(u* u̇) = ν, from the monodromy module: C₀ = 1 + 3q²/16 + O(q⁴) (1.007741 at q = 0.2), no first-order term; excess micromotion as phase modulation with index φ_Ω | derivation audit of 2026-09-04 (`check_c0_floquet.py`); Wineland Eqs. 71-72; Berkeland Eq. 21 (misprint corrected) | the RMP\'s (1 + q/2)⁻¹, whether attached to Ω₀ or to η, is the u(0) = 1 normalization at one rf phase, an rf-clock artifact [corrected: derivation audit, 2026-09-04]; "Debye-Waller" is a different factor [corrected]; pdftotext renders φ_Ω as φ_V [corrected] |',
    ),
    (
        "p09_validation.md",
        "| Micromotion factors | η_eff = ηC₀; J₀(φ_Ω) carrier reduction | RMP Eq. 37; Wineland Eqs. 71-72 [corrected] |",
        "| Micromotion factors | η_eff = ηC₀ with C₀ = 1 + 3q²/16 + O(q⁴) (Section 4.1.1); J₀(φ_Ω) carrier reduction | derivation audit 2026-09-04; Wineland Eqs. 71-72 [corrected] |",
    ),
    (
        "p09_validation.md",
        "| Micromotion carrier and comb | exact rf-averaged carrier element 1.000000 at q = 0.2, η → 0, against the RMP's printed Ω₀/(1 + q/2) reading (0.909091; the Floquet value at q = 0.2 is 0.906701); the simulator's C₀, the rf-period average of the Floquet function from the monodromy module, is 0.951917, 0.906701, 0.862823 at q = 0.1, 0.2, 0.3 (tolerance 10⁻⁶), and (1 + q/2)⁻¹ = 0.952381, 0.909091, 0.869565 is asserted to agree to O(q²); comb phase per rf period 0.222580 against βω_rfT/2 = 0.222144 (the printed βω_rfT is exactly 2×) | RMP Eqs. 37, 67-68 [corrected] |",
        "| Micromotion carrier and comb | exact rf-averaged carrier element 1.000000 at q = 0.2, η → 0; the simulator's C₀, the Wronskian-normalized e^{iνt} coefficient of the exact Floquet function, is 1.001890, 1.007741, 1.018161 at q = 0.1, 0.2, 0.3 (tolerance 10⁻⁶) and agrees with 1 + 3q²/16 to O(q⁴); the negative controls are the u(0) = 1 average, which depends on the rf phase assigned to t = 0 (0.906701 or 1.111511 at q = 0.2 for two phases half a period apart), and (1 + q/2)⁻¹ = 0.952381, 0.909091, 0.869565; comb phase per rf period 0.222580 against βω_rfT/2 = 0.222144 (the printed βω_rfT is exactly 2×) | RMP Eqs. 37, 67-68 [corrected]; derivation audit 2026-09-04, `check_c0_floquet.py` [recomputed here] |",
    ),
    # ---- B. carrier-term weight in A_± (4.2.1, 4.2.2, 9.3, 13) ----
    (
        "p04b_cooling.md",
        "R_{n→n+1} = (n+1) η² A₊,  R_{n→n−1} = n η² A₋,  A_± = W(Δ∓ν) + α W(Δ),",
        "R_{n→n+1} = (n+1) η² A₊,  R_{n→n−1} = n η² A₋,  A_± = W(Δ∓ν) + (η̃²/η²) W(Δ),  η̃²/η² = α k_em²/(b_{i,m} Δk·ê_m)²,",
    ),
    (
        "p04b_cooling.md",
        "with the intermediate state |e, n±1⟩ paired with detuning Δ∓ν (anti-correlated) **[corrected]**; this is level B's engine, with W(Δ) = Γρ_ee(Δ) from the Bloch steady state and α the recoil angular factor of Section 4.2.8, which is exactly Stenholm's A_± = P(Δ ∓ ν) + αP(Δ) (Stenholm 1986 Eqs. 5.49-5.54).",
        "with the intermediate state |e, n±1⟩ paired with detuning Δ∓ν (anti-correlated) **[corrected]**; this is level B's engine, with W(Δ) = Γρ_ee(Δ) from the Bloch steady state, α the recoil angular factor of Section 4.2.8 and k_em the emitted photon's wavenumber, so that the carrier term carries the emitted photon's Lamb-Dicke parameter η̃ while the sideband terms carry the drive's η, the two coinciding only for one resonant beam along the mode axis with b_{i,m} = 1, where the weight reduces to α and the expression to Stenholm's A_± = P(Δ ∓ ν) + αP(Δ) (Stenholm 1986 Eqs. 5.49-5.54); the first two revisions of this plan wrote the weight as α in general while computing (η̃/η)² = 1.376 for quenched 40Ca+ a few lines above, an inconsistency the derivation audit of 2026-09-04 caught "
        + AUD
        + ".",
    ),
    (
        "p04b_cooling.md",
        "and the carrier term W(Δ) is kept, weighted by α, because it is the recoil heating that accompanies carrier scattering and produces the (η̃/η)² = α piece of the floor above.",
        "and the carrier term W(Δ) is kept, weighted by η̃²/η², because it is the recoil heating that accompanies carrier scattering and produces the (η̃/η)² piece of the floor above (equal to α only in the single-beam, on-axis case).",
    ),
    (
        "p04b_cooling.md",
        "⟨n⟩ = A₊/(A₋ − A₊) = [W(Δ−ν) + αW(Δ)]/[W(Δ+ν) − W(Δ−ν)],",
        "⟨n⟩ = A₊/(A₋ − A₊) = [W(Δ−ν) + (η̃²/η²)W(Δ)]/[W(Δ+ν) − W(Δ−ν)],",
    ),
    (
        "p04b_cooling.md",
        "uses the rate framework of Section 4.2.2, A_± = W(Δ ∓ ν) + αW(Δ) with W from the multi-level Bloch steady state,",
        "uses the rate framework of Section 4.2.2, A_± = W(Δ ∓ ν) + (η̃²/η²)W(Δ) with W from the multi-level Bloch steady state,",
    ),
    (
        "p13_conventions.md",
        "| Cooling coefficients A_± | bare coefficients with units of rate, A_± = W(Δ ∓ ν) + αW(Δ), the Fock rate equation carrying η²; the carrier term is kept with the recoil factor α and vanishes only for EIT at the dark resonance |",
        "| Cooling coefficients A_± | bare coefficients with units of rate, A_± = W(Δ ∓ ν) + (η̃²/η²)W(Δ) with η̃² = α(k_em x₀)² the emitted photon's Lamb-Dicke parameter, the Fock rate equation carrying the drive's η²; the carrier weight reduces to α for one resonant beam along the mode axis (Stenholm's case) and vanishes only for EIT at the dark resonance [corrected: derivation audit, 2026-09-04] |",
    ),
    (
        "p09_validation.md",
        "| Lamb-Dicke rate framework | bare A_± = W(Δ ∓ ν) + αW(Δ) with anti-correlated pairing and η² in the rate equation; the level-B steady state equals (Γ̃/2ν)²[(η̃/η)² + 1/4] with (η̃/η)² = α and matches the level-C master equation where W(Δ) is not negligible;",
        "| Lamb-Dicke rate framework | bare A_± = W(Δ ∓ ν) + (η̃²/η²)W(Δ) with anti-correlated pairing and η² in the rate equation; the level-B steady state equals (Γ̃/2ν)²[(η̃/η)² + 1/4] with (η̃/η)² = α k_em²/(bΔk·ê)² (α for one on-axis beam, 1.376 for quenched 40Ca+) and matches the level-C master equation where W(Δ) is not negligible;",
    ),
    # ---- C. Laguerre matrix element as a modulus (4.3.1) ----
    (
        "p04c_hamiltonian.md",
        "Ω_{n′,n} = Ω |⟨n′| e^{iη(a+a†)} |n⟩| = Ω e^{−η²/2} (n_<!/n_>!)^{1/2} η^{|n′−n|} L^{|n′−n|}_{n_<}(η²),\n\nexact in η for one running-wave drive and one mode.",
        "⟨n′| e^{iη(a+a†)} |n⟩ = ⟨n′|D(iη)|n⟩ = e^{−η²/2} (iη)^{|n′−n|} (n_<!/n_>!)^{1/2} L^{|n′−n|}_{n_<}(η²),  Ω_{n′,n} = Ω |⟨n′|D(iη)|n⟩| = Ω e^{−η²/2} (n_<!/n_>!)^{1/2} |η|^{|n′−n|} |L^{|n′−n|}_{n_<}(η²)|,\n\nexact in η for one running-wave drive and one mode, the signed element being what the builder uses and the modulus what a Rabi frequency means (the first two revisions equated the modulus to the signed Laguerre expression, which is negative for odd |n′ − n| at negative η and beyond the polynomial's first zero, already at n = 6 to 8 for η = 0.5; caught by the derivation audit of 2026-09-04) "
        + AUD
        + ".",
    ),
    # ---- D. MS closure and the force prefactor (4.4.1, 9.4, 11.1, 13, Appendix D) ----
    (
        "p04d_gates.md",
        "H(t) = −ħηΩ(a† e^{iεt} + a e^{−iεt}) S_y,  U(t) = D̂(α(t) S_y) exp[i(λt − χ sin(εt)) S_y²],  α(t) = (ηΩ/ε)(e^{iεt} − 1),\n\nso each eigenstate of S_y is displaced along a circle of radius ηΩ/ε (Haljan writes α₀ = Ω_sb/2δ with Ω_sb the sideband Rabi frequency) that closes at εt = 2πm **[verified]**.",
        "H(t) = −(ħηΩ/2)(a† e^{iεt} + a e^{−iεt}) S_y,  U(t) = D̂(α(t) S_y) exp[i(λt − χ sin(εt)) S_y²],  α(t) = (ηΩ/2ε)(e^{iεt} − 1),  λ = η²Ω²/(4ε),  χ = η²Ω²/(4ε²),\n\nwith Ω the per-tone Rabi frequency of the plan's convention (H_tone = (ħΩ/2)σ₊e^{i(...)} + h.c.; to first order in η the blue tone alone feeds a†e^{iεt} and the red tone alone a e^{−iεt}, each with amplitude ηΩ/2), so each eigenstate m of S_y is displaced along a circle of radius |m|ηΩ/(2ε) that closes at εt = 2πm (Haljan's α₀ = Ω_sb/(2ε) with Ω_sb = ηΩ the sideband Rabi frequency) **[verified]**. The first two revisions of this plan displayed −ħηΩ(...)S_y with α(t) = (ηΩ/ε)(e^{iεt} − 1), a form that is correct only for a Rabi frequency defined without the ½ (Ω′ = Ω/2, the reading under which Roos's Ω_c = |ε|/(4η) becomes |ε|/(2η) in the plan's Ω; whether Kirchmair's and Roos's own Ω is that one was not re-verified here, and the ingest converter carries the factor), a doubled force that the derivation audit of 2026-09-04 caught and direct integration confirmed (`check_ms_closure.py`) "
        + AUD
        + ".",
    ),
    (
        "p04d_gates.md",
        "in the S_y = Σσ_y normalization of the displayed propagator, which is also that of the P₀, P_{±2} decomposition and of Kirchmair's thermal envelopes below, the same condition reads ηΩ/ε = 1/(4√K) and τ = π√K/(2ηΩ), which is Roos's critical Rabi frequency Ω_c = |ε|/(4η) for two ions at |λt*| = π/8 **[verified]**. The two are one condition in two spin conventions and must never be mixed: integrating the displayed Hamiltonian for one loop from |↓↓⟩ gives a Bell state (populations 0.5, 0, 0, 0.5, concurrence 1.000) at ηΩ/ε = 1/4 and a product state (concurrence 0.000) at ηΩ/ε = 1/2 **[corrected; the first version of this plan quoted both forms as one, which was caught by the 2026-09-04 critique and confirmed by direct integration]**; Section 13 carries the row, and the simulator's S_α = Σ_i σ_α^i is the convention of every MS formula in this plan.",
        "and the ratio ηΩ/ε is a ratio of physical quantities that cannot depend on whether the spin operator is written J_y or S_y = 2J_y: in the plan's per-tone convention the maximally entangling closure is ηΩ/ε = 1/(2√K) with τ = 2πK/ε = π√K/(ηΩ) whichever generator is written, and only the two-body phase target rescales as the square of the spin normalization, A(τ) = −π/2 on J_y², −π/8 on S_y² and χ = π/4 on σ_yσ_y, so that |λt*| = π/8 on S_y² gives Ω = ε/(2η√K) and Roos's Ω_c = |ε|/(4η) is the same condition for a Rabi frequency defined without the ½ "
        + AUD
        + ". The second revision of this plan stated the S_y form as ηΩ/ε = 1/(4√K), τ = π√K/(2ηΩ) and Ω_c = |ε|/(4η) in the plan's own Ω and reported that integrating 'the displayed Hamiltonian' gives a Bell state at 1/4 and a product state at 1/2; both statements were true of the doubled Hamiltonian −ħηΩ(...)S_y and false of the plan's drive. Integrating the exact bichromatic Hamiltonian with (ħΩ/2)σ₊ per tone for one loop from |↓↓⟩ (η = 0.05, ν = 2π × 1 MHz, ε = 2π × 10 kHz, d_m = 16) gives concurrence 0.9999 with populations (0.493, 0, 0, 0.507) at ηΩ/ε = 1/2 and concurrence 0.38 with populations (0.038, 0, 0, 0.962), a χ = π/16 pulse, at 1/4, and the effective Hamiltonian −(ħηΩ/2)(...)S_y reproduces both (`check_ms_closure.py`) **[recomputed here]**; the same error had reached the timing-benchmark fixture of Section 11.1, whose 'v2' closure was a χ = π/16 pulse while the withdrawn 'v1' closure at 1/2 was the maximally entangling one. Section 13 carries the row, and the simulator's S_α = Σ_i σ_α^i is the convention of every MS formula in this plan.",
    ),
    (
        "p09_validation.md",
        "| Exact MS propagator | D̂(αS_y)exp[i(λt − χ sin εt)S_y²]; α(t) = (ηΩ/ε)(e^{iεt} − 1); loop closes at εt = 2πm | Kirchmair 2009; Haljan 2005 [verified] |\n| Closure and entanglement, S = Σσ convention | ηΩ/ε = 1/(4√K), τ = π√K/(2ηΩ), Ω_c = ∣ε∣/(4η); direct integration of the displayed Hamiltonian from ∣↓↓⟩ gives concurrence 1.000 at ηΩ/ε = 1/4 and 0.000 at 1/2 | Roos 2008; Kirchmair 2009 [verified], [corrected] |\n| Closure and entanglement, J = ½Σσ convention | ηΩ/(ν − δ) = 1/(2√K), τ = √Kπ/(ηΩ) in Sørensen-Mølmer's normalization; the conversion to the row above is asserted | Sørensen-Mølmer 2000 [verified] |",
        "| Exact MS propagator | D̂(αS_y)exp[i(λt − χ sin εt)S_y²]; α(t) = (ηΩ/2ε)(e^{iεt} − 1) with Ω the per-tone Rabi frequency; loop closes at εt = 2πm | Kirchmair 2009; Haljan 2005 [verified]; prefactor [corrected: derivation audit, 2026-09-04] |\n| Closure and entanglement, one condition in both spin normalizations | ηΩ/ε = 1/(2√K), τ = 2πK/ε = π√K/(ηΩ); phase targets A = −π/2 on J_y², −π/8 on S_y², χ = π/4 on σ_yσ_y; exact per-tone integration from ∣↓↓⟩ (η = 0.05, ν = 2π × 1 MHz, ε = 2π × 10 kHz, d_m = 16) gives concurrence 0.9999 at ηΩ/ε = 1/2 and 0.38 (χ = π/16) at 1/4, and the effective −(ħηΩ/2)S_y(...) form reproduces both (`check_ms_closure.py`) | Sørensen-Mølmer 2000; Roos 2008 (Ω_c = ∣ε∣/(4η) for a Rabi frequency without the ½) [verified]; [corrected: derivation audit, 2026-09-04]; [recomputed here] |",
    ),
    (
        "p13_conventions.md",
        "| Spin operator in MS formulas | S_α = Σ_i σ_α^i (eigenvalues 0, ±2 for two ions), the normalization of the displayed propagator, the P₀, P_{±2} decomposition, Kirchmair's envelopes and Roos's Ω_c = ∣ε∣/(4η); closure ηΩ/ε = 1/(4√K) | Kirchmair 2009; Roos 2008 | Sørensen-Mølmer's J_α = ½Σσ_α with closure ηΩ/(ν − δ) = 1/(2√K) and A(τ) = −π/2, halved on ingest [corrected] |",
        "| Spin operator in MS formulas | S_α = Σ_i σ_α^i (eigenvalues 0, ±2 for two ions) for the displayed propagator, the P₀, P_{±2} decomposition and Kirchmair's envelopes; the force on S_y is −(ħηΩ/2)(a†e^{iεt} + h.c.) for the plan's per-tone Ω; closure ηΩ/ε = 1/(2√K) in every spin normalization, with the phase target −π/2 on J_y², −π/8 on S_y², π/4 on σ_yσ_y | Sørensen-Mølmer 2000; Kirchmair 2009; Roos 2008; derivation audit 2026-09-04 (`check_ms_closure.py`) | Sørensen-Mølmer's J_α = ½Σσ_α with closure 1/(2√K) is ingested unchanged (the second revision halved it to 1/(4√K), a factor-2 force error caught by the derivation audit); Roos's Ω_c = ∣ε∣/(4η) reads ∣ε∣/(2η) in the plan's Ω [corrected: derivation audit, 2026-09-04] |",
    ),
    (
        "p11_performance.md",
        "the maximally entangling closure ηΩ/ε = 1/4 of Section 13,",
        "the closure ratio ηΩ/ε = 1/4, which the derivation audit of 2026-09-04 later showed to be a χ = π/16 pulse in the plan's per-tone convention, the maximally entangling ratio being 1/2 (Section 4.4.1; `bench_ms_timing_v3.py` repeats the measurement at 1/2 and Section 9.13 reports both),",
    ),
    (
        "p11_performance.md",
        "The first version of this table used ηΩ/ε = 1/2, which in this convention is a product state with a displacement excursion of 2 rather than 1, and an η triple (0.08, 0.06, 0.05) that no two-ion crystal produces from one Δk, since the two modes of a family share |b| = 1/√2 and their η differ only by (ω_COM/ω_rock)^{1/2}; the 2026-09-04 experimentalist critique caught both, and the re-measured times differ from the first version by at most a factor 1.8 at the largest case (`bench_ms_timing_v1.py` keeps the first fixture) **[recomputed here]**, **[corrected]**:",
        "The first version of this table used ηΩ/ε = 1/2 and an η triple (0.08, 0.06, 0.05) that no two-ion crystal produces from one Δk, since the two modes of a family share |b| = 1/√2 and their η differ only by (ω_COM/ω_rock)^{1/2}; the 2026-09-04 experimentalist critique caught the fixture and, wrongly, the closure, which it called a product state on the strength of the doubled effective Hamiltonian of the second revision's Section 4.4.1: in the plan's per-tone convention 1/2 is the maximally entangling ratio and 1/4 a χ = π/16 pulse (`check_ms_closure.py`), so the v2 timings below were taken at half the gate's Rabi frequency, `bench_ms_timing_v3.py` repeats them at 1/2 (Section 9.13), and the re-measured times differ from the first version by at most a factor 1.8 at the largest case (`bench_ms_timing_v1.py` keeps the first fixture) **[recomputed here]**, **[corrected: derivation audit, 2026-09-04]**:",
    ),
    (
        "p16_runs.md",
        "(the Mølmer-Sørensen closure normalization by integrating the displayed Hamiltonian, the heating-coherence rate",
        "(the Mølmer-Sørensen closure normalization by integrating the displayed Hamiltonian, a check the derivation audit later showed to have integrated the doubled effective form of the second revision's Section 4.4.1 rather than the plan's per-tone drive, the heating-coherence rate",
    ),
    # ---- E. chi_ij symmetrization (4.4.3, 13) ----
    (
        "p04d_gates.md",
        "α_{i,m}(τ) = iη_{i,m}∫₀^τ Ω_i(t) sin(μt) e^{iω_m t} dt,  χ_{ij}(τ) = 2Σ_m η_{i,m}η_{j,m} ∫₀^τ∫₀^{t′} Ω_i(t)Ω_j(t′) sin(μt) sin(μt′) sin[ω_m(t′ − t)] dt dt′,",
        "α_{i,m}(τ) = iη_{i,m}∫₀^τ Ω_i(t) sin(μt) e^{iω_m t} dt,  χ_{ij}(τ) = Σ_m η_{i,m}η_{j,m} ∫₀^τ dt′∫₀^{t′} dt [Ω_i(t)Ω_j(t′) + Ω_j(t)Ω_i(t′)] sin(μt) sin(μt′) sin[ω_m(t′ − t)],",
    ),
    (
        "p04d_gates.md",
        "The factor 2 in χ is a pair-counting convention: Blümel et al. 2021 use a factor-2-free kernel and therefore a maximally entangling target of π/8, so a simulator using the π/4 convention must double their kernel, not halve it **[corrected]**. The simulator adopts χ = π/4 with the factor-2 kernel and tests the conversion.",
        "The symmetrized envelope product replaces the factor 2 of Choi's printed form 2Ω_i(t)Ω_j(t′): the second-order Magnus term is Σ_{i,j}σ_x^iσ_x^j K_ij with a time-ordered kernel that is odd under t ↔ t′, so grouping (a, b) with (b, a) gives χ_ab = K_ab + K_ba, which equals 2K_ab only when Ω_a(t) ∝ Ω_b(t); the printed form is exact for the equal-envelope gate solver, wrong by ±18% for independently shaped per-ion envelopes (±30% in a three-ion example), and as printed gives χ_ij ≠ χ_ji although σ_x^iσ_x^j is symmetric (derivation audit of 2026-09-04) "
        + AUD
        + ". Blümel et al. 2021 use a kernel without the pair sum and therefore a maximally entangling target of π/8, so a simulator using the π/4 convention must double their kernel, not halve it **[corrected]**. The simulator adopts χ = π/4 with the symmetrized kernel and tests both conversions.",
    ),
    (
        "p13_conventions.md",
        "| Entangling angle | χ_ij = 2Σ_m η_imη_jm∫∫… with the pair sum over i < j, maximally entangling at χ = π/4; XX(χ) = exp(−iχσ_xσ_x) |",
        "| Entangling angle | χ_ij = Σ_m η_imη_jm∫∫[Ω_i(t)Ω_j(t′) + Ω_j(t)Ω_i(t′)]… with the pair sum over i < j (Choi's printed factor 2 holds only for proportional envelopes; derivation audit 2026-09-04), maximally entangling at χ = π/4; XX(χ) = exp(−iχσ_xσ_x) |",
    ),
    # ---- F. dipole normalization stack: no second factor 2 (4.5.6, 13) ----
    (
        "p04e_atomic.md",
        "one level up, Steck's Γ_{J_gJ_e} is a partial rate carrying the degeneracy ratio (2J_g + 1)/(2J_e + 1) while Ozeri's, Uys's, Wineland's and Olmschenk's γ is the total 1/τ, a second factor 2 on a D2 line, so every stored element and rate carries a convention tag and is converted once at ingest, and the degeneracy ratio is never applied on top of a 3j-normalized element.",
        "one level up, Steck's Γ_{J_gJ_e} = ω³(2J_g + 1)|⟨J_g‖d‖J_e⟩|²/(3πε₀ħc³(2J_e + 1)) is the total, m_F-independent decay rate out of any sublevel of J_e into the fine-structure level J_g and equals b(J_e → J_g)/τ, so on a unit-branching alkali D2 line it is 1/τ, Ozeri's, Uys's, Wineland's and Olmschenk's γ exactly, with no second factor 2 (the second revision of this plan claimed one; the degeneracy ratio inside the formula is the same factor already spent in converting the reduced element to the stretched one, and applying it again is a factor 4 on a D2 line; caught by the derivation audit of 2026-09-04) "
        + AUD
        + "; every stored element and rate carries a convention tag and is converted once at ingest, and the degeneracy ratio is never applied on top of a 3j-normalized element.",
    ),
    (
        "p13_conventions.md",
        "Steck's Γ_{J_gJ_e} is a partial rate with the degeneracy ratio, others' γ is the total 1/τ; every stored element and rate carries its convention tag |",
        "Steck's Γ_{J_gJ_e} is the total rate out of J_e into the level J_g, b/τ, equal to 1/τ on a unit-branching D2 line (no second factor 2); every stored element and rate carries its convention tag |",
    ),
    (
        "p13_conventions.md",
        "stacking a reduced element into a stretched-element formula and a total rate into a partial-rate formula are each a factor 2 on a D2 line [corrected] |",
        "stacking a reduced element into a stretched-element formula is a factor 2 on a D2 line; the 'total rate into a partial-rate formula' factor that the second revision listed does not exist [corrected: derivation audit, 2026-09-04] |",
    ),
    # ---- G. clock-qubit curvature convention (6.3) ----
    (
        "p06_noise.md",
        "and clock qubits only quadratically (310.8 Hz/G² for 171Yb+; 2.4 mHz/mG² for 43Ca+ at 146 G) **[verified]**; both coefficients are now outputs of the hyperfine-Zeeman diagonalization of Section 4.5.1 (310.85 Hz/G² and 2.415 mHz/mG² recomputed), so a field sample is converted to per-level frequency offsets through computed sensitivities.",
        "and clock qubits only quadratically: written as δν = c₂(δB)², c₂ = 310.8 Hz/G² for 171Yb+ at zero field and 1.21 mHz/mG² for the 43Ca+ |4,0⟩ ↔ |3,+1⟩ transition at 146.094 G, whose second derivative d²ν/dB² = 2c₂ is Harty's printed 2.4 mHz/mG² (the second revision paired the two numbers as if they shared a convention, which the derivation audit of 2026-09-04 caught; the Section 13 'curvature naming' row is the rule) "
        + AUD
        + "; both are outputs of the hyperfine-Zeeman diagonalization of Section 4.5.1 (310.85 Hz/G² and c₂ = 1.2077 mHz/mG², d²ν/dB² = 2.415 mHz/mG², recomputed), so a field sample is converted to per-level frequency offsets through computed sensitivities.",
    ),
    # ---- H. alpha_K closed form (4.4.4, 6.2) ----
    (
        "p04d_gates.md",
        "motional dephasing with Lindblad operator L = a†a√(2/τ) giving ε_d = α_K t_g/τ, α_K = {0.686, 0.297, 0.137} for K = {1, 2, 4} **[verified]** (0.297 × 100 µs/200 ms = 0.15 × 10⁻³ reproduces the paper's line;",
        "motional dephasing with Lindblad operator L = a†a√(2/τ) giving ε_d = α_K t_g/τ, α_K = 1/(2K) + 3/(16K²) = (8K + 3)/(16K²) = {11/16, 19/64, 35/256} = {0.6875, 0.2969, 0.1367} for K = {1, 2, 4}, the 1/(2K) being residual spin-motion entanglement and the 3/(16K²) the randomized geometric phase; Ballance prints {0.686, 0.297, 0.137}, the first being the value at finite t_g/τ = 10⁻³ rather than the asymptotic coefficient (derivation audit of 2026-09-04) **[verified]**, "
        + AUD
        + " (0.297 × 100 µs/200 ms = 0.15 × 10⁻³ reproduces the paper's line;",
    ),
    (
        "p06_noise.md",
        "gives a gate error ε_d = α_K t_g/τ with α_K = {0.686, 0.297, 0.137} for K = {1, 2, 4} (Ballance 2016) **[verified]**;",
        "gives a gate error ε_d = α_K t_g/τ with α_K = (8K + 3)/(16K²) = {0.6875, 0.2969, 0.1367} for K = {1, 2, 4} (Ballance 2016 prints 0.686, 0.297, 0.137; Section 4.4.4) **[verified]**, "
        + AUD
        + ";",
    ),
    # ---- I. diagonal labels (5.1.1) ----
    (
        "p05_numerics.md",
        "for a general complex α the two branches are α^{n′−n} above the diagonal and (−α*)^{n−n′} below it; the first version of this plan printed n′ − n without the modulus, which is wrong below the diagonal by (iη)^{−2|n′−n|}",
        "for a general complex α the two branches are α^{n′−n} where n′ > n, below the diagonal of the matrix ⟨n′|·|n⟩ with row n′ and column n, and (−α*)^{n−n′} where n′ < n, above it (the second revision had the two labels interchanged; derivation audit of 2026-09-04); the first version of this plan printed n′ − n without the modulus, which is wrong above the diagonal by (iη)^{−2|n′−n|}",
    ),
    # ---- J. master equation with hbar-divided H (5.7) ----
    (
        "p05_numerics.md",
        "ρ̇ = −(i/ħ)[H(t), ρ] + Σ_k 𝒟[L_k]ρ,  𝒟[L]ρ = LρL† − ½{L†L, ρ},\n\nrun as",
        "ρ̇ = −i[H(t), ρ] + Σ_k 𝒟[L_k]ρ,  𝒟[L]ρ = LρL† − ½{L†L, ρ},  H(t) = H_phys(t)/ħ in rad/s (Section 5.6), L_k = √γ_k A_k in s^{−1/2},\n\n(the terms listed below are written as H_phys with their ħ; the second revision wrote −(i/ħ)[H, ρ] one paragraph after declaring H already divided by ħ, a units-layer slip caught by the derivation audit of 2026-09-04) run as",
    ),
    # ---- K. sign convention of the optical phase (5.2) ----
    (
        "p05_numerics.md",
        "which is the e^{+i(Δk·X̄_i − Δφ_i)} of Section 13 and the e^{i(δkX̄_i − μ_i t − δφ_i)} of Section 4.3.1 written as one symbol.",
        "which is the e^{+i(Δk·X̄_i − Δφ_i)} of Section 13 and the e^{i(δkX̄_i − μ_i t − δφ_i)} of Section 4.3.1 written as one symbol; Δφ_i is defined as the phase of the beat note written E ∝ cos(ω_L t − Δk·r + Δφ_i), so that for a Raman pair with Δk = k₁ − k₂ and beam fields E_j ∝ cos(k_j·r − ω_j t + φ_j) it is Δφ_i = φ₂ − φ₁ (the relative sign of Δk·X̄_i and the optical phase is physical, the field depending only on their sum in the other ordering, and a virtual RZ(θ) adds +θ to every later φ_tone; the derivation audit of 2026-09-04 found the definition missing) "
        + AUD
        + ".",
    ),
    # ---- L. 9/20 against 5/12 (4.5.5) ----
    (
        "p04e_atomic.md",
        "the dipole-weighted value for fixed polarization being 9/20 (16% more),",
        "the dipole-weighted value for fixed polarization being 9/20 (8% more, 27/25; the second revision wrote 16%),",
    ),
    # ---- M. Kramers-Heisenberg rate with sqrt(Gamma_e) inside the coherent sum (4.5.5) ----
    (
        "p04e_atomic.md",
        "Γ^{(j)}_{a→b} = Σ_{q′} Γ_e |Σ_e (Ω^{(j)}_{ea}/(2Δ_e^{(j)})) c_{e→b,q′}|², with c_{e→b,q′} the normalized decay amplitude of Section 4.5.2 (Σ_{b,q′}|c_{e→b,q′}|² = 1).",
        "Γ^{(j)}_{a→b} = Σ_{q′} |Σ_e √Γ_e c_{e→b,q′} Ω^{(j)}_{ea}/(2Δ_e^{(j)})|², with c_{e→b,q′} the normalized decay amplitude of Section 4.5.2 (Σ_{b,q′}|c_{e→b,q′}|² = 1) and √Γ_e inside the coherent sum, one factor per intermediate path (strictly √(Γ_e(ω_j/ω_e)³), a 10⁻⁴ fine-structure correction), because the emission element ⟨b|d_{q′}|e⟩ = √(3πε₀ħc³Γ_e/ω_e³) c_{e→b,q′} carries it; the second revision placed Γ_e outside the sum, which is exact only when every intermediate level has the same total rate, true for 9Be+ under one radial integral and false for the species whose P₃/₂ branches to D levels, by up to 11× per channel (derivation audit of 2026-09-04) "
        + AUD
        + ".",
    ),
    # ---- N. pseudopotential amplitude failure note (4.1.6, 13) ----
    (
        "p13_conventions.md",
        "rms halves the depth and inflates every secular frequency by √2, peak-to-peak quarters it; the charge applied twice [corrected] |",
        "reading the stored peak amplitude as an rms value (implied peak × √2) doubles the depth and inflates every rf-set secular frequency by √2, reading it as peak-to-peak (implied peak/2) quarters the depth and halves those frequencies, depth ∝ f² and frequency ∝ f always moving together (the second revision paired a halved depth with a √2-inflated frequency, which no single field factor gives; derivation audit 2026-09-04); the charge applied twice [corrected] |",
    ),
    (
        "p04a_trap.md",
        "so that an rms amplitude halves every depth and inflates every secular frequency by √2 and a peak-to-peak amplitude quarters the depth",
        "so that a stored peak read as an rms amplitude (implied peak × √2) doubles every depth and inflates every rf-set secular frequency by √2, and one read as peak-to-peak (implied peak/2) quarters the depth and halves those frequencies, depth ∝ f² and frequency ∝ f always moving together (the second revision paired a halved depth with a √2-inflated frequency; derivation audit of 2026-09-04)",
    ),
    # ---- O. R_d and R_b named at first use (8.1) ----
    (
        "p08_readout.md",
        "**Leakage rates.** Off-resonant pumping between the bright and dark manifolds is the dominant error channel for direct hyperfine readout. For 171Yb+ **[corrected]**:",
        "**Leakage rates.** Off-resonant pumping between the bright and dark manifolds is the dominant error channel for direct hyperfine readout; R_d is the rate at which a bright ion (F = 1) is pumped dark and R_b the rate at which a dark ion (F = 0) is pumped bright, Noek's and Crain's names, so R_d carries the bright state's off-resonant F = 1 → F′ = 1 channel at Δ_HFP with its 1/3 branching into F = 0 and R_b the dark state's only dipole-allowed channel, F = 0 → F′ = 1 at Δ_HFP + Δ_HFS (a reader who takes R_d for the dark-to-bright rate finds the two detunings interchanged, which one comparer of the 2026-09-04 derivation audit did). For 171Yb+ **[corrected]**:",
    ),
    # ---- P. the pi/2 between force axis and carrier axis (4.3.4, 13) ----
    (
        "p04c_hamiltonian.md",
        "The simulator's convention is Haljan's, φ_s = +(φ_b + φ_r)/2 (Sections 4.4.2 and 13); Lee writes φ_S = −(half-sum), whose minus sign the extracted prose had dropped **[corrected]**, and it is converted at ingest (the first version of this plan adopted Lee's sign here and Haljan's in Section 4.4.2, a contradiction caught by the 2026-09-04 critique; the sign flips the entangling axis of every MS gate relative to the GPi frame and is pinned by the Section 9.4 test that an MS pulse with φ_b = φ_r = φ entangles about the azimuth +φ of GPi(φ)).",
        "The simulator's convention is Haljan's, φ_s = +(φ_b + φ_r)/2 for the half-sum (Sections 4.4.2 and 13); Lee writes φ_S = −(half-sum), whose minus sign the extracted prose had dropped **[corrected]**, and it is converted at ingest (the first version of this plan adopted Lee's sign here and Haljan's in Section 4.4.2, a contradiction caught by the 2026-09-04 critique). The axis of the force is not the half-sum itself: expanding the plan's own drive factor e^{i(Δk·x̂ − μt + φ)} to first order gives the sideband coupling an explicit i (iη(a + a†)), and i[σ₊e^{iψ} − h.c.] = σ_{ψ+π/2}, so in the same-Δk (phase-sensitive) geometry the force axis sits at π/2 + Δk·X̄_i + (φ_b + φ_r)/2, a quarter turn from the carrier axis at the same beat-note phase, while in the crossed-Δk (phase-insensitive) geometry η_r = −η_b adds a further π and Δk·X̄_i cancels; the constant offset is what the phase-frame alignment of Section 7.5 step 3 measures and the native MS(φ₀, φ₁) definition absorbs, so it changes no gate, but the Section 9.4 pin must carry it: an MS pulse with φ_b = φ_r = φ on ions at the frame origin entangles about the azimuth φ + π/2 in the same-Δk geometry (the second revision's '+φ' would have certified a quarter-turn error; derivation audit of 2026-09-04) "
        + AUD
        + ".",
    ),
    (
        "p13_conventions.md",
        "| Spin and motion phases | φ_s = (φ_b + φ_r)/2, φ_m = (φ_b − φ_r)/2 with σ_φ = e^{−iφ_s}σ₊ + e^{iφ_s}σ₋ | Haljan 2005 | Lee 2005 writes φ_S with the opposite sign [corrected] |",
        "| Spin and motion phases | half-sum φ_s = (φ_b + φ_r)/2 and half-difference φ_m = (φ_b − φ_r)/2 of the tone phases, with σ_φ = e^{−iφ_s}σ₊ + e^{iφ_s}σ₋; the force axis carries a further π/2 from the i of the sideband coupling (plus Δk·X̄_i in the same-Δk geometry), a constant the frame alignment absorbs (Section 4.3.4) | Haljan 2005; derivation audit 2026-09-04 | Lee 2005 writes φ_S with the opposite sign [corrected]; the second revision's Section 9.4 pin omitted the π/2 [corrected: derivation audit, 2026-09-04] |",
    ),
    # ---- Q. rank-2 tensor conjugation (4.5.7) ----
    (
        "p04e_atomic.md",
        "the rank-2 tensors are normalized to Σ_ij |c^{(q)}_{ij}|² = 2/3, not 1, so a library that unit-normalizes them inflates every quadrupole Rabi frequency by √(3/2); and Ω follows the plan's convention",
        "the rank-2 tensors are normalized to Σ_ij |c^{(q)}_{ij}|² = 2/3, not 1, so a library that unit-normalizes them inflates every quadrupole Rabi frequency by √(3/2), and they are c^{(q)} = (2/3)b^{(q)} with b^{(q)}_{ij}r_ir_j = r²C^{(2)}_q (b^{(0)} = diag(−½, −½, 1), b^{(±1)} = ∓√(3/8)[x̂ẑ + ẑx̂ ± i(ŷẑ + ẑŷ)], b^{(±2)} = √(3/8)[x̂x̂ − ŷŷ ± i(x̂ŷ + ŷx̂)]), not the conjugate duals (2/3)b^{(q)*}, which share the 2/3 norm but turn q = m − m′ into q = m′ − m for circular light, so the module asserts c^{(q)}_{ij}r_ir_j = (2/3)r²C^{(2)}_q at construction, which is what fixes the sign of q for circular polarization (derivation audit of 2026-09-04) "
        + AUD
        + "; and Ω follows the plan's convention",
    ),
    # ---- R. emission diffusion coefficient and the dissipator it multiplies (4.2.8) ----
    (
        "p04b_cooling.md",
        "(ii) The emission diffusion coefficient in the adiabatically eliminated motional master equation is D = (α/2) Σ_{i→j} ρ_ii^SS Γ_{i→j} η_ij², one Lamb-Dicke parameter per decay channel built from that channel's own wavenumber, whose contribution to the phonon heating rate is 2D = α η_em² R_sc;",
        "(ii) The emission diffusion coefficient in the adiabatically eliminated motional master equation is D = ½ Σ_{i→j} ρ_ii^SS Γ_{i→j} α_{ij,m} η_ij², one Lamb-Dicke parameter per decay channel built from that channel's own wavenumber and one angular factor per channel and mode axis, and it multiplies Cirac's half-rate dissipator L_em μ = D(2XμX − X²μ − μX²) = 2D·𝒟[X] with X = a + a†, so its contribution to the phonon heating rate is 2D = Σ ρ_ii Γ_ij α_ij η_ij², which collapses to α η_em² R_sc only when every channel shares one pattern and one wavenumber (an implementer who feeds D to the plan's global 𝒟[L] convention halves the heating, and one mean α is up to 37% wrong for mixed π and σ channels; derivation audit of 2026-09-04) "
        + AUD
        + ";",
    ),
]

if __name__ == "__main__":
    apply(EDITS)
