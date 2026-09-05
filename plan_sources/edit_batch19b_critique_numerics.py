"""Edit batch 19b (2026-09-04, critique v3 fold): architecture, numerics, control and readout sections (3, 5, 7, 8)."""

from edit_batch13_errata import apply

TAG = "**[corrected: critique, 2026-09-04]**"

EDITS = [
    # ---- 3.2 module tree: one Hamiltonian builder, the level-C owner, the Doppler producer, the engine signature ----
    (
        "p03_architecture.md",
        "# shared multi-level optical-Bloch steady state: W(Δ), R_∘, ρ_ee, pumping rates; one object for cooling, readout, scattering",
        "# multi-level matrix elements and optical-Bloch steady state: W(Δ) = Γρ_ee in s⁻¹, R_∘, pumping rates, dark-state kernel check (Section 5.3); one supplier for cooling, readout, scattering. Level-C time evolution (cooling with one mode, pumping) runs through dynamics/hamiltonian.py's multi-level mode, owned by prep/",
    ),
    (
        "p03_architecture.md",
        "    hamiltonian.py    # assemble H(t) = H0 + sum drive terms from pulses and device; frames",
        "    hamiltonian.py    # the ONE Hamiltonian builder: H(t) = H0 + drive terms from pulses and device, plus the multi-level mode of Section 4.2.8 (manifold-by-manifold frames, Floquet fallback) that cooling, pumping and detection share; frames",
    ),
    (
        "p03_architecture.md",
        "# Doppler cooling model -> thermal n_bar per mode (rate model + optional ME)",
        "# Doppler cooling: Section 4.2.2 rate coefficients with W from light/bloch.py, one (Delta, s, k_hat) per beam optimized over all modes -> n_bar per mode; force model and two-level closed form as guarded cross-checks only (Section 4.2.1)",
    ),
    (
        "p03_architecture.md",
        "# PulseEngine protocol: run_pulses(device, schedule, state, seeds) -> traces; the one entry point",
        "# PulseEngine protocol: run_pulses(device, schedule, state, space, sample, seeds, options) -> Traces, Appendix E's signature verbatim; the one entry point",
    ),
    # ---- 3.4 pipeline and randomness ----
    (
        "p03_architecture.md",
        "  -> readout: fluorescence model on final state -> photon counts -> bits [readout.*]",
        "  -> readout: fluorescence model -> photon counts -> bits; measure, reset and recool are IR\n       operations, so the loop is prep -> [gates | measure | reset | recool]* -> results (7.2) [readout.*, prep.*]",
    ),
    (
        "p03_architecture.md",
        "chooses trajectories per sample by `mcsolve`'s `target_tol` on the final populations, draws shots from the weighted trajectory mixture (Section 5.3),",
        "chooses the trajectory count per sample in two phases, an estimate from `mcsolve`'s `target_tol` on the final populations (which needs `e_ops`, Section 5.4) followed by a replay of a fixed seed list of that length, draws shots from the weighted trajectory mixture (Section 5.3),",
    ),
    (
        "p03_architecture.md",
        "spawned deterministically by (sample, trajectory, ion, channel) so that every variate's key is independent of execution order and of truncation retries; the per-trajectory children are passed to `mcsolve` as an explicit seed list, and a test asserts bitwise-identical results over 1 and 18 workers.",
        "spawned deterministically by (sample, trajectory, shot, ion, channel), the shot slot added because photon-count sampling is per shot per ion and two shots on one (sample, trajectory) would otherwise share a record and the error bars would lose the within-trajectory detection variance, so that every variate's key is independent of execution order and of truncation retries; the per-trajectory children are passed to `mcsolve` as an explicit seed list, and the reproducibility test is a tolerance test (agreement to 10⁻¹² over 1 and 18 workers plus per-trajectory identity under the same seed) rather than a bitwise one, because `MultiTrajResult` accumulates running sums in completion order and floating-point addition is not associative (4.4 × 10⁻¹⁶ measured on 5.3.1 with identical seeds), and because `target_tol` stops on a check whose firing point depends on scheduling (200 trajectories serial against 5 parallel in one measured call; 2026-09-04 numerics critique) "
        + TAG
        + ".",
    ),
    # ---- 5.1 ENR caveats ----
    (
        "p05_numerics.md",
        "ENR is an option for a group of cold, undriven modes that must nevertheless be carried dynamically, with the caveats of Section 5.1.1:",
        "ENR is an option for a group of cold, undriven modes that must nevertheless be carried dynamically, with the caveats of Section 5.1.1 and three more measured on 5.3.1: `ptrace` on any space containing an ENR factor raises `NotImplementedError`, so an ENR run cannot produce the reduced qubit state that `Result.spam`, the channel summary and GATE_LOCAL tomography need, or a per-mode reduced density matrix, unless the module builds the marginal itself from `enr_state_dictionaries`, which it does; `tensor(sigmap(), D_enr)` reports `dims` whose product (98 for two modes at N_exc = 6) disagrees with its `shape` (56), so every dimension computation reads `shape` behind an assertion and never `dims`; and the ENR displacement is unitary to 8 × 10⁻¹⁵ while silently wrong near the cap (⟨n₀⟩ = 2.2412 against 2.25 after a displacement α = 1.5 at N_exc = 6), which is why its cost is quoted at N_exc ≈ 2n_phys with the top-shell population as the only convergence diagnostic (2026-09-04 numerics critique) "
        + TAG
        + ":",
    ),
    # ---- 5.2 virtual-Z sign, freeze class ----
    (
        "p05_numerics.md",
        "and a virtual RZ(θ) adds +θ to every later φ_tone; the derivation audit of 2026-09-04 found the definition missing)",
        "and a virtual RZ(θ) shifts every later φ_tone by φ → φ − θ, the time-order rule of Section 7.6 and the one pinned sign, tested on a concrete sequence (RZ(0.1) then GPi2(0) on |0⟩ against independently multiplied matrices, Section 9.6) rather than by asserting that a rule is pinned; the second revision printed +θ here against 7.6's −θ, which the 2026-09-04 architect critique caught, and the derivation audit of 2026-09-04 had found the definition missing)",
    ),
    (
        "p05_numerics.md",
        "|α_m(τ)|²(2n̄_m + 1) below the residual-displacement tolerance and |χ_m(τ)| below the angle tolerance, the same criterion as Section 11.3's dropped-mode rule;",
        "|α_m(τ)|²(2n̄_m + 1) below the residual-displacement tolerance and |χ_m(τ)| below a freeze tolerance that is looser than Section 11.3's dropped-mode rule and distinct from it: every mode is assigned to exactly one of three classes, dropped when |α_m|²(2n̄_m + 1) < 10⁻⁶ and |χ_m| < 10⁻⁴ (nothing absorbs a dropped mode's loss), frozen when above that pair but with |χ_m| below the miscalibration the entangling-gate calibration of Section 7.5 absorbs (default 0.05 rad, the χ_m loss folded into the calibration target and reported per mode), and resolved otherwise, the class recorded per mode in `Diagnostics.mode_class`; the second revision said freezing shared the dropped-mode criterion while describing a frozen spectator that loses 0.03 rad of χ, 300 times the angle tolerance, so that freezing was unreachable as written and with it the default of Section 5.4 (2026-09-04 architect and numerics critiques) "
        + TAG
        + ";",
    ),
    # ---- 5.3 escalation ladder, mesolve boundary ----
    (
        "p05_numerics.md",
        "so the solver applies an escalation ladder `dop853` → `vern9` → `bdf`, scales atol with the mode dimension (atol ≈ 10⁻¹⁰ × d_m/50, the boundary monitor governing the tail), records the integrator used in the diagnostics, and reports a tightened convergence run that aborts rather than losing it.",
        "so the solver applies an escalation ladder `dop853` → `vern9` → a larger `max_step` budget at atol 10⁻⁸, and never a multistep method: QuTiP's `bdf` is scipy's zvode with a dense finite-difference Jacobian (D extra right-hand sides per Jacobian and an O(D³) factorization, 67 MB at D = 2048), and the failure is not stiffness anyway, since the spectrum of −iH lies on the imaginary axis, where BDF of order 3 and above is unstable and BDF2 strongly damped, so a damping multistep rung would destroy unitarity on exactly the oscillatory problem it is asked to rescue; atol is keyed to the measured points, 10⁻¹⁰ up to d_m ≈ 100 and 10⁻⁸ above, validated at d_m = 121, 151 and 201 (the second revision's rule atol ≈ 10⁻¹⁰ × d_m/50 returned 2.4 × 10⁻¹⁰ at the very case that motivated it, 40 times tighter than the 10⁻⁸ measured to clear the abort, so the ladder as written still aborted there; 2026-09-04 numerics critique) "
        + TAG
        + "; the solver records the integrator used in the diagnostics and reports a tightened convergence run that aborts rather than losing it.",
    ),
    (
        "p05_numerics.md",
        "- **Density-matrix evolution** (`mesolve`) for small spaces (dimension below about 300, where the Liouvillian is below 10⁵ × 10⁵). This is the reference mode for validating heating, dephasing and scattering channels, and the default whenever the joint dimension is below about 300;",
        '- **Density-matrix evolution** (`mesolve`) for small spaces (dimension below about 50 to 100, the Liouvillian below 10⁴ × 10⁴). This is the reference mode for validating heating, dephasing and scattering channels and a reference-only path, not a default: one Liouvillian application costs about 2D times one H·ψ, the measured `mesolve`/`sesolve` wall-time ratio at dimension 64 on the Section 11.1 Hamiltonian is 69.5, and with the operator construction the v3 benchmark used (the exponential of the joint generator, which returns a Dense `Qobj`) `liouvillian` returns Dense, 130 GB at D = 300, so the drive operator is built as a tensor product of per-mode exponentials (CSR with exactly the predicted non-zeros, `bench_ms_timing_v4.py`) with an explicit `.to("CSR")` guard before any `liouvillian` call (the second revision put the boundary at 300 and made `mesolve` the default there; 2026-09-04 numerics critique) '
        + TAG
        + ";",
    ),
    # ---- 5.4 GATE_LOCAL tomography ----
    (
        "p05_numerics.md",
        "with the trajectory count set by the map's target accuracy through `target_tol`, and `mesolve` as the deterministic cross-check where the gate-local dimension is below 300), the Choi matrix is reconstructed by least squares and projected onto the completely positive cone with the projection residual reported,",
        "with the trajectory count fixed from a stated map-accuracy rule, n_traj ≈ 1/ε_map per input state so that the multinomial error of each output's populations sits below the target ε_map, the sixteen population `e_ops` passed explicitly because `mcsolve` raises `ValueError` on `target_tol` without `e_ops`, and `mesolve` as the deterministic cross-check where the gate-local dimension is below about 100), the Choi matrix is reconstructed by least squares and projected onto the intersection of the completely positive cone and the trace-preserving affine set by alternating (Dykstra) projection with the residual against both constraints reported, because a positive-semidefinite projection alone leaves Tr_out(Choi) ≠ 𝟙 and the extracted map then changes the register's normalization on every application (2026-09-04 numerics critique) "
        + TAG
        + ",",
    ),
    (
        "p05_numerics.md",
        "so that GATE_LOCAL costs 16 state evolutions of the gate-local space rather than density-matrix evolutions of it;",
        "so that GATE_LOCAL costs 16 × n_traj state evolutions of the gate-local space, a few hundred trajectories per input at ε_map = 10⁻³ and therefore more than one `mesolve` of the same space where `mesolve` fits (the second revision counted 16; 2026-09-04 numerics critique), which is what Section 11.2 charges;",
    ),
    # ---- 5.5 noise grid ----
    (
        "p05_numerics.md",
        "noise realizations are generated by the exact Ornstein-Uhlenbeck update on the solver's own grid with Δt ≤ τ_c/10 and interpolated with `order=1`,",
        "noise realizations are generated by the exact Ornstein-Uhlenbeck update on an independent fixed grid with Δt ≤ τ_c/10, stored in the run record and interpolated with `order=1` (never on the solver's accepted steps, which `dop853` chooses adaptively, so that a realization generated there could not be reproduced; 2026-09-04 architect critique),",
    ),
    # ---- 5.7 C0 ownership, POVM fast-path domain ----
    (
        "p05_numerics.md",
        "the micromotion factor C₀ on every η (4.1.1), and",
        "the micromotion factor C₀ already inside every η as `Crystal.lamb_dicke` delivers it (4.1.1; never reapplied here), and",
    ),
    (
        "p05_numerics.md",
        "so the readout error is never applied twice (the first version of this plan listed the POVM before the record layer; caught by the 2026-09-04 critique).",
        "so the readout error is never applied twice (the first version of this plan listed the POVM before the record layer; caught by the 2026-09-04 critique). The fast path's domain is stated: the product POVM is exact only at zero readout crosstalk, because Section 8.5 makes the records neighbour-coupled (added counts on a dark neighbour, camera PSF leakage, neighbour-conditioned likelihoods), so the full path's joint confusion matrix is not a tensor product of per-ion POVMs; at non-zero crosstalk the fast path carries a register-wide 2^N × 2^N confusion tensor with its own size guard (dense to N = 12, factored by neighbour range beyond), and Section 9.5's test is agreement at zero crosstalk plus a bounded, reported discrepancy at the configured crosstalk (2026-09-04 architect critique) "
        + TAG
        + ".",
    ),
    # ---- 7.2 rule 4: measure and reset in the IR ----
    (
        "p07_control.md",
        "4. Measurement operations are collected at the end (mid-circuit measurement is out of scope for the first release and is rejected with a clear error).",
        "4. Measure, reset and recool are schedulable operations of the IR and of `Schedule` from the first release, so the pipeline shape is prep → [gates | measure | reset | recool]* → results; the first release executes measurement only as the terminal stage and refuses a mid-circuit measure or reset with a clear error, but the IR, the scheduler and `Result` already carry the positions, so that adding the physics of Section 8.5 (detection recoil, neighbour Stark shift, depumping, recooling and re-preparation of the measured ion) is a stage implementation and not a pipeline change (Section 1.3 names a QEC stack as a client, and that is the client that needs it; 2026-09-04 experimentalist critique) "
        + TAG
        + ".",
    ),
    # ---- 7.5 experiments: Stark, crosstalk, field, and the dependency graph ----
    (
        "p07_control.md",
        "6. **Heating rate**: delay-scan of sideband asymmetry; store measured n̄-dot per mode (used to schedule recooling where the device has it).\n\nCalibrations are cached per device configuration",
        "6. **Heating rate**: delay-scan of sideband asymmetry; store measured n̄-dot per mode (used to schedule recooling where the device has it).\n7. **Stark shift**: a Ramsey experiment with the gate beams on and detuned, per (ion, beam), fitted for the differential light shift that sets the MS phase; stored in `stark`.\n8. **Crosstalk**: drive ion i, fit the Rabi rate and phase on every neighbour j; stored in `crosstalk` as ε_ij.\n9. **Field**: the clock-point or Zeeman-splitting scan that fixes B and its drift, stored in `field`, run first because every other fit reads the Zeeman sensitivities.\n\nThe experiments form a dependency graph that `calibrate()` follows and never reorders: field, then micromotion compensation, then mode frequencies, then carrier Rabi and Stark, then crosstalk, then the entangling-gate scans, then detection and heating, and a downstream fit refuses to run while an upstream entry it reads is `uncalibrated` (a mode-frequency fit on an uncompensated ion carries J₁ micromotion sidebands; the second revision declared the experiments as an unordered list, and its `CalibrationTable` named three entries, `stark`, `crosstalk` and `field`, for which no experiment existed; 2026-09-04 experimentalist critique) "
        + TAG
        + ".\n\nCalibrations are cached per device configuration",
    ),
    (
        "p07_control.md",
        "followed by a fine scan fitted with the plan's own excitation lineshape Ω²/(Ω² + δ²/4) sin²(t√(Ω² + δ²/4)) rather than a Lorentzian,",
        "followed by a fine scan fitted with the excitation lineshape in the plan's own convention, P = [Ω²/(Ω² + δ²)] sin²((t/2)√(Ω² + δ²)) (Section 13: carrier π pulse at Ωt = π, half-depth at δ = Ω), rather than a Lorentzian, with a CI assertion that the fitted π time on a noiseless carrier equals π/Ω and that the fitted Ω reproduces the device's derived Ω within the reported uncertainty (the second revision printed here the Boulder half-Rabi form Ω²/(Ω² + δ²/4) sin²(t√(Ω² + δ²/4)) that Section 7.9 flags, unflagged and called the plan's own, which would have written Ω at half the truth into the table and into every π time, η and pulse amplitude derived from it; 2026-09-04 critique, all four lenses) "
        + TAG
        + ",",
    ),
    # ---- 7.9: the half-Rabi copy is an ingest form ----
    (
        "p07b_control_details.md",
        "remembering that this Ω is half the usual Rabi frequency (Blümel 2021 Eq. S28) **[verified]**;",
        "remembering that this Ω is half the usual Rabi frequency (Blümel 2021 Eq. S28), a half-Rabi form doubled on ingest and never the fit function of Section 7.5 **[verified]**;",
    ),
    # ---- 8.1: the retired 19.6 MHz, the CPT ceiling, the field optimum, the derived prefactors ----
    (
        "p08_readout.md",
        "with Γ = 2π × 19.6 MHz, Ω the full J = 1/2 → J′ = 1/2 line Rabi frequency",
        "with Γ the partial rate of the S₁/₂ ↔ P₁/₂ line, Γ_S = 2π × 19.62 MHz (0.99499 of the 19.72 MHz total from the 8.07(9) ns lifetime, Section 9.12; the 19.6 MHz reading the second revision carried here and in the 8.4 preset is retired everywhere, and a CI check greps the tree for retired constants), Ω the full J = 1/2 → J′ = 1/2 line Rabi frequency",
    ),
    (
        "p08_readout.md",
        "Coherent population trapping in the F = 1 → F′ = 0 detection cycle cuts the 171Yb+ scattering rate to about one third of the two-level saturated rate while leaving leakage unchanged, equivalent to η → η/3 **[verified]** (Olmschenk et al. 2007); the ≈ 5 G field is applied to destabilize the dark states (Berkeland and Boshier 2002, extracted and verified in Run 5), so the factor 1/3 is a ceiling reached under optimal destabilization and B → 0 does not recover the full rate but collapses it as two dark states become exact;",
        "Coherent population trapping in the F = 1 → F′ = 0 detection cycle caps the 171Yb+ scattering rate at Γ/4, the saturation ceiling n_e/(n_e + n_g) of the three-ground-plus-one-excited manifold, one half of the two-level Γ/2, while leaving leakage unchanged (Olmschenk et al. 2007 quote one third, which is the Λ repump cycle's ceiling and not this manifold's; the lumped 'η → η/3' model of the earlier revisions under-predicts detected counts by 3 and is illegal whenever R_∘ comes from the Bloch solve, whose assertion is R_∘ ≤ Γ/4; at Crain's fitted s_∘ = 0.815 the corrected R_∘ alone gives 0.0880Γ, which times ε_sys = 4.356% is his 472 kcps) "
        + TAG
        + "; the ≈ 5 G field is applied to destabilize the dark states (Berkeland and Boshier 2002, extracted and verified in Run 5), so the ceiling is reached under optimal destabilization and B → 0 does not recover the full rate but collapses it as two dark states become exact;",
    ),
    (
        "p08_readout.md",
        "and this section's ≈ 5 G operating point sits about two decades above the optimum, where P_f falls as 1/B².",
        "and this section's ≈ 5 G operating point sits about a factor 2 above that optimum (δ_B/2π = 3.27 MHz at Ω = γ/3, that is B ≈ 2.3 G at μ_B/h = 1.3996 MHz/G; `check_critique_v3.py`), where P_f begins to fall as 1/B²; the second revision printed 'two decades', which at 1/B² would have been a factor 10³ in fluorescence and contradicted the same paragraph's claim that the ceiling is reached at this field, so the optimum is a computed output of the CPT solve at the configured Ω and polarization and the 5 G rate is asserted against the Γ/4 ceiling rather than against a scaling (2026-09-04 theorist critique) "
        + TAG
        + ", **[recomputed here]**.",
    ),
    (
        "p08_readout.md",
        "the simulator therefore exposes the prefactor as a species parameter pinned to a measured value (Crain: R_d = 341(13) Hz at 56.2 mW/cm²; Noek: about 1.5 kHz at about 170 mW/cm²) **[contested]**.",
        "the atomic-rate layer therefore computes the (R_d, R_b) prefactors from the angular algebra of Section 4.5.2, which settles the bookkeeping in Noek's favour, and reproduces the measured values (Crain: R_d = 341(13) Hz at 56.2 mW/cm²; Noek: about 1.5 kHz at about 170 mW/cm²) through the apparatus entries of the calibration table (intensity scale, detuning, polarization purity), never through a species constant (Section 8.8; the second revision's sentence here still carried the retracted fitted species parameter; 2026-09-04 architect critique) **[contested]**, "
        + TAG
        + ".",
    ),
    (
        "p08_readout.md",
        "171Yb+ (Γ = 2π × 19.6 MHz, Δ_HFP = 2.1 GHz,",
        "171Yb+ (Γ_S = 2π × 19.62 MHz partial, 19.72 total, Δ_HFP = 2.1 GHz,",
    ),
    (
        "p08_readout.md",
        "- **Leakage-rate normalization is fitted in the first release.** With that s_∘,",
        "- **Leakage-rate prefactors are derived, and the apparatus residual is calibrated.** With that s_∘,",
    ),
    (
        "p08_readout.md",
        "the physics for it (the recoil channel, optical pumping, sympathetic cooling) is what Sections 4.2 and 4.5 supply;",
        "the physics for it (the recoil channel, optical pumping, sympathetic cooling) is what Sections 4.2 and 4.5 supply, and the IR and the scheduler carry measure, reset and recool as operations from the first release (Section 7.2), so implementing the stage changes no pipeline shape;",
    ),
]

if __name__ == "__main__":
    apply(EDITS)
