# Physics notes: what qutip-trap integrates, and where each equation comes from

qutip-trap simulates a trapped-ion quantum computer by integrating its physics: the ions' internal states and motional
modes evolve under the Hamiltonian the laser and microwave fields produce, with the dissipation the environment and the
scattered photons cause, from the cooled and pumped initial state to the detected photons. Circuits are compiled to native
gates and scheduled as pulses; the pulses are what the simulator integrates. No gate matrix is ever applied to the
simulated state (the compiler and the validation suite are the only places ideal unitaries appear).

These notes list the equations as implemented, each with the section of [`PLAN.md`](../PLAN.md) that specifies and tags
it, the module that implements it, and the provenance record of [`docs/provenance/ledger.yaml`](provenance/ledger.yaml)
(`conv.*` for a convention row, `anchor.*` for a number a test pins) that a reader can follow to the source and to the
check that established it. Tags follow the plan's Appendix D: **verified** (checked against the primary source by
independent verifiers), **corrected** (a printed or extracted error found and the corrected form stated), **recomputed
here** (a committed script under `validation/scripts/` reproduces the number), **derived** (computed by the package from
cited inputs), **background** (textbook physics the author supplies), **contested**. A tag records what checking was done,
never that a value is final: the tests of Section 9 define correctness.

Notation: internal frequencies are angular (rad/s), public ones ordinary (Hz), converted with one explicit 2π; ħ is kept in
the displayed Hamiltonians and divided out once when the master equation is integrated (Section 5.6; `conv.frequencies`).
The full convention table is in [conventions.md](conventions.md).

## 1. One master equation

Everything the simulator integrates is one master equation on the joint space of the ions' internal levels and the
resolved motional modes (Section 5.7):

    ρ̇ = −i[H(t), ρ] + Σ_k D[L_k] ρ,      D[L] ρ = L ρ L† − ½{L†L, ρ},      H(t) = H_phys(t)/ħ,

run as `sesolve` on the pure branches of the initial mixture, `mcsolve` with keyed quantum-jump trajectories, or `mesolve`
on the density matrix where the Liouvillian fits (Section 5.3; `dynamics/evolve.py`, the `dop853` → `vern9` escalation
ladder of `conv.solver_integrators`). The one Hamiltonian builder is `dynamics/hamiltonian.py` (Section 3.1: every pulse,
whether a calibration scan, a single-qubit gate, a Mølmer-Sørensen gate or a cooling pulse, is a set of drive terms on a
shared space assembled by the same builder); the collapse operators come from `dynamics/channels.py`,
`noise/scattering.py` and the readout layer. Every term below is switchable and every switch is recorded in
`Result.diagnostics.approximations`.

## 2. The Hamiltonian

H(t) = H_mot + H_int + Σ_i H_drive,i(t) + Σ_i H_Stark,i(t) + H_anh + H_curv (Section 5.7).

- **Motion.** H_mot = Σ_m ħ ω_m a_m† a_m over the resolved modes, the zero-point energy dropped (`conv.zero_point_energy`),
  the frequencies the calibrated (noise-shifted) secular frequencies of the pseudopotential frame (Sections 4.1.3, 4.1.5)
  with per-sample offsets from the quasi-static noise. Modes are indexed by position in `Crystal.modes`, ordered axial,
  transverse_1, transverse_2, ascending in frequency within a family, eigenvectors unit norm with the last component positive
  (`conv.mode_index`, `conv.mode_eigenvector_gauge`).
- **Internal offsets.** H_int = Σ_i (ħ Δ_i/2) σ_z^i (Σ_p ħ Δ_{i,p} |p⟩⟨p| when the register factor has d > 2), the offset of
  ion i's true transition from the frame frequency: the magnetic-field sample through the exact Zeeman diagonalization at
  the shifted field (Section 6.3), the ac Zeeman shift of a microwave drive (Section 4.3.3), the qubit frequency the table
  believes against the one the atom has (Section 7.3). σ_z = |1⟩⟨1| − |0⟩⟨0| in the computational ordering
  (`conv.computational_ordering`).
- **Drives.** For every tone of every pulse on ion i (Sections 4.3.1, 5.2; `conv.rabi_frequency`, `conv.detuning`,
  `conv.drive_coefficient_and_phase_continuity`)

      H_drive,i(t) = (ħ/2) Σ_tones Ω_{i,tone}(t) e^{−i(μ_tone t − φ_tone(t))} σ₊^i ⊗ Π_m D_m(i η_{i,m}) + h.c.,

  with the displacement operators D(iη) = exp[iη(a + a†)] built as exact matrix exponentials and asserted against the
  analytic Laguerre elements ⟨n′|D|n⟩ over the populated range (Section 5.1.1; `anchor.m2.displacement_table`,
  `conv.oracle_margin_rule`), or held as tensor factors and applied mode by mode by the matrix-free kernel of
  `dynamics/kernels.py` (Section 11.3 item 4; `anchor.m9b.factorized_kernel_exactness`). Ω comes from the beam intensities
  and the dipole matrix elements (Sections 4.3.2, 4.5; `light/raman.py`, `light/microwave.py`), never from a table: the
  scheduler requests Ω_table and the played chain of Section 7.10 delivers Ω_phys = Ω_req Ω_derived/Ω_table
  (`conv.played_chain`). The same term on a neighbour j with Ω → ε_{ij} Ω is addressing crosstalk (Section 6.6; the ratio is a
  Rabi ratio, `conv.crosstalk_ratio` in Section 13); the coefficient multiplied by e^{iβ cos(Ω_rf t + δ)} is excess
  micromotion (Section 4.3.6); Ω_i → Ω_i(1 + (Ω″/2Ω) x̂_i²) is Cetina's beam-curvature coupling (`anchor.m2.cetina_curvature`);
  a sampled φ(t) or Ω(t) is laser phase or intensity noise (Sections 6.3, 6.4). Frozen spectator modes leave the space and
  enter through per-shot Debye-Waller factors e^{−η²(n + ½)} drawn from their thermal distribution
  (`conv.frozen_spectator_shot_sample`, `anchor.m2.debye_waller_identity`).
- **Spin and motion phases.** A bichromatic pulse's two tones at phases φ_b, φ_r define the spin phase φ_s = (φ_b + φ_r)/2
  and the motion phase φ_m = (φ_b − φ_r)/2; the force axis carries a further π/2 from the i of the sideband coupling
  (Section 4.3.4; `conv.spin_motion_phases`, `conv.motion_phase_default`). The Lamb-Dicke parameter is η_{i,m} =
  (Δk · ê_m) c_{i,m} √(ħ/(2 m_i ω_m)) with the ion's own mass and the mass-weighted eigenvector component, C₀ = 1 + 3q²/16
  applied once (Sections 4.1.1, 4.1.7; `conv.lamb_dicke`, `conv.micromotion_correction`, `anchor.trap.c0_wronskian`).
- **Light shifts.** H_Stark,i(t) = (ħ/2) δ_St,i(t) σ_z^i, proportional to the instantaneous intensity, the two-photon shift
  scaling linearly and a single-photon shift quadratically with the amplitude (Section 4.3.2; `conv.stark_scaling_with_amplitude`);
  the scheduler compensates the believed shift by detuning every tone and absorbs the accumulated z-rotation into the
  virtual-Z frame (Section 7.5 item 7; `conv.stark_compensation`).
- **Anharmonicity.** H_anh is the cubic Coulomb coupling of Section 4.1.4, opt-in because it couples the three mode
  families; the resonance checker and the directly integrated phase estimate are on by default (`anchor.trap.nonlinearity_epsilon`).

## 3. The collapse operators

- **Heating.** √(Γ(N̄ + 1)) a_m and √(Γ N̄) a_m† per mode, with Γ_h = Γ N̄ the quoted heating rate d⟨n⟩/dt at n = 0 and
  Γ_h = e² S_E/(4 m ħ ω) from the single-sided electric-field noise density, projected on the mode with the configured
  correlation length for a multi-ion crystal (Sections 4.1.5, 6.2; `conv.heating_master_equation`,
  `conv.electric_field_noise`, `conv.heating_rate_meaning`; `trap/heating.py`, `noise/model.py`).
- **Motional dephasing.** √(2/τ_m) a_m† a_m from the white rf-amplitude noise on the rf-derived modes (Section 6.2;
  `conv.motional_dephasing_from_rf_noise`).
- **Qubit dephasing.** √(γ_φ/2) σ_z^i with γ_φ = 2π² (dν/dB)² S_B,white from the computed Zeeman sensitivity, so a Ramsey
  coherence decays as e^{−γ_φ t} and γ_φ = 1/T₂ for the white part; the slow part is a sampled Δ_i in H_int
  (Section 6.3; `conv.qubit_dephasing_operator`, `conv.qubit_dephasing_from_field_noise`).
- **Photon scattering.** One Raman operator √Γ_{p→q} |q⟩⟨p| ⊗ R per channel from the signed amplitudes of the atomic layer,
  including channels out of the qubit pair when d > 2 (leakage into a sink level, `noise/levels.py`), the elastic Rayleigh
  channel as c = ½√Γ_el σ_z (the dissipator prefactor Γ_el/4, the coherence decaying at Γ_el/2; `conv.rayleigh_dissipator`),
  and R the recoil operator D(i(η_abs − η_em)) with the emission direction on a moment-exact quadrature about B
  (Sections 4.2.8, 4.5.5, 6.5; `conv.scattering_channels`, `anchor.m3.recoil_energy_identity`). At d = 2 the run reports the
  scattering probabilities as estimates in the intrinsic budget instead.
- **Laser intensity noise.** √D H_drive(t), the white part of the intensity spectrum (Section 6.4; `conv.intensity_noise_channel`).
- **Readout.** The bright/dark/shelf continuous-time Markov chain of Section 8.2 with its photon-counting operator, and the
  shelf decay √(1/τ_D) |S⟩⟨D| of Section 8.1.

Every spectrum is stored two-sided in angular frequency with the e^{−iωt} kernel and a separately declared white level, the
white level the Lindblad route and the tabulated band the sampled-trajectory route (Sections 6.1, 9.17; `conv.noise_routing`).

## 4. Trap, crystal and modes

- **Mathieu equation** d²x/dξ² + [a − 2q cos 2ξ] x = 0, ξ = ω_rf t/2, the stability edge and the characteristic exponent from
  the monodromy matrix, the Floquet function's Wronskian normalization Im(u* u̇) = ν giving C₀ (Section 4.1.1;
  `conv.mathieu_sign`, `conv.floquet_phase_origin`, `conv.floquet_branch_normalization`; `trap/mathieu.py`).
- **Pseudopotential** ψ = Q²|E|²/(4 m Ω²) with V_rf the peak of the cos drive (Section 4.1.6; `conv.rf_amplitude_pseudopotential`;
  `trap/pseudopotential.py`, `trap/surface.py` for gapless-plane electrodes with the five-wire height √(a(a + 2b))/2,
  `anchor.trap.house_five_wire`).
- **Equilibrium positions and modes.** The N-ion equilibrium from the Coulomb-plus-trap potential, the mass-weighted Hessian
  and its eigenproblem on all three axes, the ordering checks (axial COM lowest, transverse COM highest, the zigzag
  threshold), the per-species eigenvector components c_{i,m} = √m_i b_i (Sections 4.1.2, 4.1.3, 4.1.7;
  `conv.mass_weighted_eigenvectors_parity`; `anchor.trap.james_spectrum`, `anchor.trap.zigzag_threshold`,
  `anchor.trap.mixed_two_ion_axial`; `trap/crystal.py`).
- **Heating from S_E(ω)** as an input spectrum (Section 12: no first-principles model exists), with the multi-ion
  projection and the S_E round trip (`anchor.trap.heating_round_trip`).

## 5. Atomic structure and the light-atom coupling chain

- **Hyperfine and Zeeman structure** at the operating field by exact diagonalization of H_hfs + H_Z on every fine-structure
  level, with A_hfs signed, the measured g_J, g_I = −(μ_I/(I μ_N))(m_e/m_p), the Breit-Rabi formula as the J = 1/2 check,
  labels connected adiabatically through avoided crossings, the clock points found from dν/dB = 0 (Section 4.5.1;
  `conv.sign_A_hfs_and_g_factors`, `conv.nuclear_g_sign`; `anchor.ca43.clock_point`, `anchor.yb171.quadratic_zeeman`;
  `species/zeeman.py`).
- **Dipole matrix elements** by Wigner-Eckart, ⟨F m_F|d_q|F′ m_F′⟩ = ⟨J‖d‖J′⟩ × (6j and 3j factors), the reduced element from
  the measured lifetime through Γ = ω³ (2J + 1)|⟨J‖d‖J′⟩|²/(3π ε₀ ħ c³ (2J′ + 1)), the saturation intensity I_sat = π h c Γ/(3λ³)
  with the angular partial rate (Sections 4.5.2, 9.13; `conv.hyperfine_dipole_element`, `conv.dipole_normalization_stack`,
  `conv.saturation_intensity`; `anchor.yb171.i_sat`, `anchor.yb171.branching_third`; `species/dipole.py`).
- **Polarization** decomposed about B into σ⁺, π, σ⁻ components in the spherical basis, the helicity index of absorption
  against the operator index q_op = m_lower − m_upper (Section 4.5.3; `conv.polarization_components`; `species/polarization.py`).
- **Raman couplings** Ω_R = Σ_e Ω₂* Ω₁/(2Δ_e) with the single-photon Rabi frequencies in the ħΩ/2 convention, light shifts
  δ_g = Σ|Ω|²/(4Δ_e), the differential shift of the qubit pair, and the scattering amplitudes (Raman, Rayleigh, the
  differential-Rayleigh dephasing) from the explicit intermediate-state sums (Sections 4.5.4, 4.5.5; `conv.raman_beam_labels`,
  `anchor.be9.ozeri_2005_amplitudes`, `anchor.be9.wineland_clock_light_shift`; `species/raman.py`, `light/raman.py`,
  `light/scattering.py`).
- **Electric-quadrupole coupling** for optical qubits, Ω = (e E₀ k/(2ħ)) |⟨S‖r² C^{(2)}‖D⟩| |Λ(m, m′)| |g^{(q)}| with the
  Racah-normalized rank-2 tensor and the upper-level degeneracy in the lifetime formula (Section 4.5.7;
  `conv.quadrupole_element_normalization`; `anchor.ca40.reduced_element_e2`, `anchor.sr88.reduced_element_e2`;
  `species/quadrupole.py`).
- **The multi-level optical Bloch builder** of Section 4.2.8 (`dynamics/multilevel.py`, `light/bloch.py`): every dressed
  sublevel of every included level, one frame frequency per manifold with the inconsistency detector and the Floquet
  fallback, one collapse operator per decay line and emitted polarization from the dipole operator itself (Σ C†C = Γ P_e),
  the steady state as the single scattering-rate object W(Δ) = Σ Tr(C†C ρ_ss) that cooling, detection and pumping share
  (`conv.scattering_rate_object`, `conv.multilevel_frame_assignment`; `anchor.m3a.yb171_detection_rate`,
  `anchor.m3a.dark_states`).

## 6. Cooling and state preparation

- **Doppler cooling** from the rate framework of Section 4.2.2: per illuminated ion and mode the heating and cooling rates
  c²_{i,m}[S(±ν) + 2D] from the dipole-force spectrum of the ion's steady state under all its beams and the emission-recoil
  diffusion, one detuning per beam group optimized over the participation-weighted modes, the RMP force model and the
  Doppler limit (Γ/4ν)(1 + α) as guarded cross-checks where ν ≪ Γ (Section 4.2.1; `conv.level_a_rates_participation`,
  `conv.doppler_stage_producer`; `anchor.m3.doppler_rate_framework`, `anchor.m3a.doppler_limit`; `prep/doppler.py`,
  `prep/rates.py`).
- **Resolved-sideband and Raman sideband cooling**: the rate coefficients A_± = W(Δ ∓ ν) + (η̃²/η²) W(Δ) with the carrier
  weight, Stenholm's floor (Γ/2ν)² [α/cos²θ_L + 1/4], the exact pulsed transfer matrices with higher sidebands and the
  Laguerre nodes where a single order strands population, the repump recoil kernel, and the sideband-ratio thermometry
  R_k = (n̄/(1 + n̄))^k exact for every pulse duration (Sections 4.2.2, 4.2.7; `conv.pulsed_transfer_matrix`;
  `anchor.m3.laguerre_nodes_and_che_schedules`, `anchor.m3.thermometry_exactness`; `prep/sideband.py`).
- **EIT cooling** with Δ > 0 and Ω_r² = 4ν(ν + Δ) in the plan's sign (Section 4.2.3; `conv.eit_plan_sign_and_composition`;
  `anchor.m3.eit_fixture_and_lechner`; `prep/eit.py`); **polarization-gradient cooling** in Joshi's analytic model and the
  Lindblad layer (Section 4.2.4; `conv.pgc_two_models`; `prep/polarization_gradient.py`).
- **Optical pumping** by the exact exponential of the multi-level Liouvillian from the scrambled manifold, returning the
  preparation error, the photon count and the per-mode recoil heating (Section 4.2.6; `anchor.m3.optical_pumping_recoil`;
  `prep/pumping.py`); the stages run Doppler → sideband/EIT → final pump (`conv.preparation_stage_order`,
  `conv.preparation_recipe`; `prep/sequence.py`, `prep/recipe.py`).
- **Recoil.** The emission kernel over a direction quadrature that derives every angular factor (α = 2/5 for the σ pattern
  along the beam, 1/5 for π perpendicular, 1/3 isotropic), the per-ion participation and the joint multi-mode kick, and the
  recoil-energy identity (Section 4.2.8; `anchor.m3.recoil_energy_identity`; `light/recoil.py`).

## 7. Gates

- **Single-qubit gates.** A native GPi(φ) or GPi2(φ) is a resonant carrier pulse of area π or π/2 at phase φ relative to
  the ion's frame, R(θ, φ) = exp[−iθ σ_φ/2] (Section 4.3.5); the exact dynamics carry the off-resonant sideband coupling
  (ηΩ/ω_m)², the Debye-Waller factors and the addressing crosstalk sin²(ε_{ij} θ/2) on the neighbours, which the intrinsic
  budget reports (Section 9.6). Virtual RZ(θ) is a frame update, φ → φ − θ on every later pulse (Section 7.6;
  `conv.virtual_z_propagation`, `conv.virtual_z_concrete_sequence`, `conv.compiler_frame_absorption`). The single-qubit
  generator carries θ/2 while the two-qubit XX(χ) carries χ with no ½, and no helper computes both (Section 13;
  `conv.rotation_generators`). The composite-pulse library of Section 4.3.5 (SK1, BB1, CORPSE, SCROFULOUS, the
  Low-Yoder-Chuang families) is `control/composite.py`, whose internal figure of merit is F_K = ¼|Tr(U_id†U)|² with F_C and
  F_avg as conversions (Section 13; `conv.gate_fidelity_measure`; `anchor.m2.composite_order_ladder`,
  `anchor.m2.mount_pd6`).
- **The Mølmer-Sørensen gate.** A bichromatic pulse with tones at ±μ from the carrier on both ions; in the plan's per-tone
  convention the spin-dependent force on S_y is −(ħηΩ/2)(a† e^{iεt} + h.c.) and the maximally entangling closure of a
  single-mode square pulse is ηΩ/ε = 1/(2√K) (Section 4.4.1; `conv.ms_closure`; `anchor.m4.ms_closure_through_the_builder`).
  The closed-form displacement α_m(t) and entangling angle χ_ij = Σ_m η_im η_jm ∫∫[Ω_i(t)Ω_j(t′) + Ω_j(t)Ω_i(t′)] … of Section
  4.4.3 (Choi's form with sin(μt) in the integrand, the symmetrized two-body kernel) drive the AM, FM and Fourier pulse
  solvers that close every resolved mode's loop (`conv.entangling_angle`; `anchor.m4.symmetrized_kernel`,
  `anchor.m4.choi_closure_counts`; `control/shaping.py`); the played waveform's χ_m and α_m per mode decide the mode classes
  (`conv.mode_classes_and_tolerances`). The gate's error model (residual displacement ε_ent = Σ|α|²(2n̄ + 1), the thermal
  Debye-Waller error (π²/4)η⁴⟨(n − n_ref)²⟩ with its calibration reference, spectator loops, dephasing, intensity noise) is
  Section 4.4.7 (`anchor.m4.debye_waller_references`, `anchor.m4.residual_displacement_conversions`, `anchor.m4.ballance_alpha_k`).
- **The light-shift (σ_z σ_z) gate** from the two-photon self-couplings of the two qubit levels, and the ZZ native gate as the
  inferred GPi2-wrapper construction on an MS device (Sections 4.4.4, 7.1; `conv.light_shift_force`,
  `conv.zz_wrapper_construction`; `anchor.m4.light_shift_gate_ca40`).
- **Native matrices and the compiler.** GPi, GPi2, MS(φ₀, φ₁, θ), ZZ(θ) and RZ as `control/native.py` defines them
  (`conv.native_ms_matrix`, `conv.gate_parameters`), the ZXZXZ decomposition of any SU(2), Maslov's one-XX CNOT template with
  its sign e^{iπvs/4}, the exact CP(θ) = [RZ(θ/2) ⊗ RZ(θ/2)] ZZ(−θ/2), every block verified against its target up to a global
  phase (Sections 7.2, 7.7; `conv.compiler_single_qubit_decomposition`, `conv.cnot_template_sign`, `conv.cp_template_exact`;
  `control/compiler.py`); an arbitrary SU(4) through the KAK decomposition in the magic basis, (a, b, c) in the Weyl
  chamber π/4 ≥ a ≥ b ≥ |c|, at most three entangling gates (`control/two_qubit.py`; `conv.kak_weyl_chamber`).
- **Calibration.** The scheduler reads only the calibration table (Section 7.3): the surrogate of Section 7.5 (closed-form
  waveforms corrected by exact spot checks, derived values as `seed` entries) or the table fitted by simulated experiments in
  the dependency order field → micromotion → modes → Rabi, Stark, qubit frequency → crosstalk → entangling scans → detection,
  heating, every entry with its uncertainty, age and noise sample (`conv.calibration_graph`, `conv.sideband_lineshape`,
  `conv.servo_residual`; `anchor.m8.*`; `calibration/`, `experiments/`).

## 8. Readout

- **The rate object.** R_∘ (the photon rate of the conditional bright state), R_d and R_b (the off-resonant pumping rates out
  of the bright and dark manifolds) from the slow-manifold analysis of the detection beams' Bloch steady state, asserted
  against the equal-population ceiling n_e/(n_e + n_g) (Γ/4 for 171Yb+'s F = 1 → F′ = 0 cycle) and reproducing the (Γ/18)
  s_∘/[1 + (2/9) s_∘] form and Noek's leakage prefactors (Sections 8.1, 8.8; `conv.saturation_ceiling`; `anchor.m5.yb171_rate_object`;
  `readout/fluorescence.py`).
- **Photon records.** The bright/dark/shelf continuous-time Markov chain: the exact count distribution over a window as one
  matrix exponential of the augmented generator, the sampled chain with Poisson counts at the piecewise rates, dead time,
  afterpulsing, neighbour crosstalk as added counts, a camera point-spread function (Section 8.2; `conv.readout_chain_exact`,
  `conv.readout_crosstalk_added_counts`, `conv.detection_efficiency_once`; `readout/detection.py`).
- **Discriminators.** Thresholding with the interior optimum over (n_c, t_b), Myerson's time-resolved maximum likelihood,
  adaptive early termination, first-photon protocols, camera decoders (Section 8.3; `conv.readout_likelihood_first_order`;
  `anchor.m5.myerson_optimum_and_recursion`; `readout/discriminate.py`).
- **The measurement operator.** The joint internal outcome is sampled projectively from the final state so that entangled
  correlations survive; the product POVM at zero crosstalk and the register-wide confusion tensor otherwise; ε = ½(ε_B + ε_D)
  reported per ion (Sections 8.4, 5.7; `conv.readout_figure_of_merit`, `conv.readout_polarity_scheme`; `anchor.m5.povm_domains`).

## 9. Noise as physical processes

Section 6.1's rule routes every noise source as physics, never as a phenomenological channel: (b) a white spectral part as a
Lindblad operator, (c) a slow part as a quasi-static parameter drawn per dynamical sample at the shot clock (an
Ornstein-Uhlenbeck chain with its correlation time, high-passed by an optional servo), (d) a band as a sampled time series
the drive coefficients see, (e) rare events (collisions) as heralded shots (`conv.noise_routing`, `conv.seed_keys`;
`noise/`). The control hardware chain of Section 7.10 quantizes phases and amplitudes to the DDS words, applies the
modulator's first-order response and the amplifier bandwidth, and carries the beat-note phase reference at the gate start
(`conv.hardware_chain`, `conv.response_phase_reference`, `conv.beat_phase_at_gate_start`; `control/hardware.py`). The
filter-function layer of Section 6.9 (`noise/decoupling.py`) computes χ = (2/π)∫(dω/ω²) S_b(ω) F(ωτ) with the full
toggling-frame control matrix, the dc frozen-noise coefficients and the decoupling sequences (`conv.filter_function_machinery`).
The circuit-level summaries of Section 6.8 (`noise/summary.py`) are what the simulator exports, never what it integrates:
the entanglement and average gate infidelity of a simulated channel, its Pauli twirl, the depolarizing rate ε with
Λ_ε(ρ) = (1 − ε)ρ + ε/(4ⁿ − 1) Σ_{P≠I} PρP whose ε is the entanglement infidelity (`conv.depolarizing_normalization`,
the same channel as Trout 2018's uniform p/3 and p/15 Kraus weights, with Qiskit's λ = p 4ⁿ/(4ⁿ − 1) as a conversion — the
Forte medians 2.0 × 10⁻⁴ and 46.4 × 10⁻⁴ enter Qiskit as 2.67 × 10⁻⁴ and 49.5 × 10⁻⁴), and the RB conversions of Section 13
(`conv.rb_error_rate`). Section 6.8 also lists Bermudez et al.'s three-variant mapping of ε onto Pauli channels as
provided, but the plan writes no formulas for the three variants, so this release omits it and records the omission
(`conv.bermudez_pauli_mapping`).

## 10. Fidelity levels and numerics

- **JOINT_EXACT** evolves the full state of all ions and all resolved modes through every pulse; the only approximation is
  the truncation (Section 5.4), and the truncation policy is enforced: caps from the pulse's closed-form coherent excursion at
  the 10⁻⁶ tail plus the Section 5.1.1 margin, the boundary monitor after every pulse, cap growth with a repeated run
  (Section 5.5; `anchor.m9a.cap_rule_populated_range`; `hilbert/truncation.py`, `run/space.py`).
- **GATE_LOCAL** above the guards (joint dimension 4096, 2 × 10⁷ drive non-zeros): every gate step exactly on its own space,
  its channel by state-based process tomography (the Π d_i² product inputs, the Choi matrix by least squares, Dykstra's
  projection onto the CP cone and the TP affine set with both residuals reported), applied to the register as Kraus
  operators, the motional model tracked between steps, the residual displacement and purity deficit reported and held to a
  JOINT_EXACT comparison (Sections 5.4, 9.8; `anchor.m9a.tomography_projection`, `anchor.m9a.gate_local_bell`;
  `dynamics/tomography.py`, `run/gate_local.py`).
- **The initial mixture as branches** (Section 5.3; `conv.fock_sum_branches`), **the matrix-free kernel** and its measured
  cost model (Sections 11.1 to 11.3; `anchor.m9b.cost_model_constants`), **parallelism** over trajectories, branches and
  samples through QuTiP's maps with keyed seeds, agreement over 1 and 18 workers to 10⁻¹² (Sections 3.4, 9.9;
  `anchor.m9b.*`; `dynamics/parallel.py`), **the propagator cache** on internal-state-only spaces (Section 11.3 item 5).
- **Results.** Shots drawn from the dynamical samples with the effective sample size behind every error bar (Section 3.4;
  `run/job.py`), the bitstring order of `conv.result_bit_order`, the IonQ formats (`conv.ionq_json_encoding`;
  `anchor.ionq.*`), and the diagnostics block that records the level, the space, the mode classes, the boundary populations,
  the integrators, the seeds, the calibration and every approximation.

## 11. Benchmarks and their budget (milestone M10)

- **Randomized benchmarking** (`benchmarks/rb.py`, `benchmarks/clifford.py`): m uniform Cliffords and the inverse of their
  product, compiled and played as pulses; the mean survival fitted to F(m) = A p^m + B; r = (1 − p)(2ⁿ − 1)/2ⁿ the average
  error per Clifford (Section 13; `conv.rb_error_rate`), beside the entanglement infidelity (4ⁿ − 1)(1 − p)/4ⁿ of the
  depolarizing channel with the same p. The 24 single-qubit Cliffords are the closure of {H, S}, of which 20 cost exactly
  one pulse (16 GPi2, 4 GPi) and four — the Z-rotation subgroup, one sixth — cost none, a group mean of 0.8333 pulses per
  Clifford; the 11520 two-qubit ones are sampled uniformly from the four double cosets L g L, g ∈ {1, CNOT, iSWAP, SWAP},
  of the local group L = C1 ⊗ C1, with sizes 576, 5184, 5184, 576 from the stabilizers 576, 64, 64, 576
  (`conv.two_qubit_clifford_classes`). Simultaneous RB on several ions fits each ion's OWN marginal survival and reports
  r_q = (1 − p_q)/2 per ion beside the r that ion measures alone (Gambetta et al. 2012), keeping the joint decay's
  (1 − p)(2ⁿ − 1)/2ⁿ as the per-layer correlation diagnostic and never as an n-qubit Clifford rate
  (`conv.simultaneous_rb_units`); `variant="knill"` runs Section 7.9's Knill-style sequences instead — π/2 pulses with an
  interleaved π Pauli or an identity and one final π/2 into the computational basis — fitted in the published form
  B p^L + ½ (`anchor.m10.knill_style_rb`).
- **GHZ fidelity** (`benchmarks/ghz.py`): P_0 = P(0…0), P_1 = P(1…1) and the parity ⟨Π Z⟩ after GPi2(φ) on every qubit,
  oscillating as C cos(Nφ + φ₀) with C = 2|ρ_{0…0,1…1}|. The laboratory's (P_0 + P_1 + C)/2 is then exactly
  max_θ ⟨GHZ_θ|ρ|GHZ_θ⟩ and therefore an UPPER bound on the fidelity against a fixed-phase GHZ state, not a lower one
  (Section 7.9; `conv.ghz_parity_bound`); the simulator reports both exact numbers beside it, the phase-optimised fidelity
  the bound estimates and the fixed-phase one it exceeds.
- **Quantum volume** (`benchmarks/volume.py`): square circuits of Haar-random SU(4) on random pairings, heavy outputs above
  the median of the ideal distribution, and the pass criterion of Cross et al. 2019 Appendix C Eq. (32) — mean − 2σ > 2/3
  with the per-circuit binomial σ = √(h(1 − h)/n_c), over at least the protocol's 100 circuits, the standard error of the
  mean reported beside it as a diagnostic and never as the criterion (`conv.heavy_output_criterion`).
- **The budget alongside** (`benchmarks/budget.py`): the Section 9.6 closed-form scales per unit; the Section 6.8 channels of
  the native gate kinds by GATE_LOCAL tomography, reduced to the benchmarked qubits with the crosstalk neighbours traced out
  in |0⟩, composed to first order (r_channel = Σ counts × ε_avg; F_gates = Π(1 − ε)); the SPAM offsets. The composition is an
  estimate, not a bound in either direction: coherent errors of one Clifford's pulses partly cancel, so the measured r sits
  at about 0.8 of r_channel on the example device, while the coherent crosstalk of a GHZ circuit's five consecutive carrier
  pulses on one neighbour adds up faster than the product (`anchor.m10.*`, `conv.benchmark_budget_composition`). The
  Section 9.6 scales are summed over each native kind's whole schedule entry, the crosstalk on ions OUTSIDE the benchmarked
  set included, so r_intrinsic bounds a larger error than r_channel measures and the gap between them is partly a
  difference of scope.
