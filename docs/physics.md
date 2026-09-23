# Physics and conventions

What qutip-trap integrates, where each piece lives, and the one convention it fixes for each quantity. Frequencies are ordinary (Hz) in the public API and angular (rad/s) inside, converted with one explicit 2 pi; hbar is divided out once, so the integrated H is in rad/s.

## The master equation

    d rho/dt = -i [H(t), rho] + sum_k D[L_k] rho,    D[L] rho = L rho L^dag - (1/2){L^dag L, rho}

on the joint space of the ions' internal levels and the resolved motional modes. It is run as `sesolve` on the pure branches of the initial mixture, `mcsolve` with keyed quantum-jump trajectories, or `mesolve` where the Liouvillian fits (`dynamics/evolve.py`, `dynamics/engine.py`). One builder assembles every term (`dynamics/hamiltonian.py`), and every switch it applies is recorded in `Result.diagnostics.approximations`.

## The Hamiltonian

H(t) = H_mot + H_int + sum_i H_drive,i(t) + sum_i H_Stark,i(t) + H_anh

- **Motion.** H_mot = sum_m hbar omega_m a_m^dag a_m over the resolved modes, zero-point energy dropped, with per-sample frequency offsets from the quasi-static noise. Modes are indexed by their position in `Crystal.modes`: axial, transverse_1, transverse_2, ascending in frequency within a family; eigenvectors are unit-norm, mass-weighted, with a positive last component.
- **Internal offsets.** H_int = sum_i (hbar Delta_i / 2) sigma_z^i, where Delta_i is the offset of ion i's transition from its frame: the sampled magnetic field through the exact Zeeman diagonalization, the ac Zeeman shift of a microwave drive, and the difference between the qubit frequency the table believes and the one the atom has. sigma_z = |1><1| - |0><0|.
- **Drives.** For every tone of every pulse on ion i

      H_drive,i(t) = (hbar/2) sum_tones Omega_i(t) exp[-i(mu t - phi(t))] sigma_+^i (x) prod_m D_m(i eta_im) + h.c.

  with D(i eta) = exp[i eta (a + a^dag)] the exact displacement operator: built as a matrix exponential and checked against the analytic Laguerre elements over the populated range, or applied mode by mode (`dynamics/kernels.py`). Omega comes from the beam intensities and dipole or quadrupole matrix elements (`light/raman.py`, `light/microwave.py`), never from the table: the scheduler requests Omega_table and the played chain delivers Omega_req Omega_derived / Omega_table. The same term on a neighbour with Omega -> eps_ij Omega is addressing crosstalk. The coefficient times exp[i beta cos(Omega_rf t + delta)] is excess micromotion. A sampled phi(t) or Omega(t) is laser phase or intensity noise. Frozen spectator modes leave the space and enter through per-shot Debye-Waller factors drawn from their thermal distribution.
- **Phases.** A bichromatic pulse with tone phases phi_b and phi_r has spin phase (phi_b + phi_r)/2 and motion phase (phi_b - phi_r)/2. The Lamb-Dicke parameter is eta_im = (dk . e_m) c_im sqrt(hbar / (2 m_i omega_m)) C0, using the ion's own mass and the mass-weighted eigenvector component, with C0 = 1 + 3 q^2/16 applied once, in `Crystal.lamb_dicke`.
- **Light shifts.** H_Stark,i = (hbar/2) delta_St,i(t) sigma_z^i, proportional to the instantaneous intensity. The scheduler detunes every tone by the believed shift and absorbs the accumulated rotation into the virtual-Z frame.
- **Anharmonicity.** The cubic Coulomb mode coupling is opt-in (`trap/anharmonic.py`).

## Collapse operators

- **Heating.** sqrt(Gamma (N + 1)) a_m and sqrt(Gamma N) a_m^dag per mode. Gamma N is the quoted heating rate d<n>/dt at n = 0, with Gamma_h = e^2 S_E / (4 m hbar omega) from the single-sided field-noise density (`trap/heating.py`, `noise/model.py`).
- **Motional dephasing.** sqrt(2/tau_m) a_m^dag a_m from white rf-amplitude noise.
- **Qubit dephasing.** sqrt(gamma_phi / 2) sigma_z with gamma_phi = 2 pi^2 (d nu/dB)^2 S_B, so a Ramsey coherence decays as exp(-gamma_phi t) and gamma_phi = 1/T2 for the white part; the slow part is a sampled Delta_i.
- **Photon scattering.** One Raman operator sqrt(Gamma_pq) |q><p| (x) R per channel, from the signed amplitudes of the atomic structure, including leakage out of the qubit pair when `internal_levels > 2`. The elastic Rayleigh channel is (1/2) sqrt(Gamma_el) sigma_z. R is the recoil displacement averaged over the emission pattern. With two-level ions the scattering probabilities are reported as estimates in the intrinsic budget instead (`noise/scattering.py`, `light/recoil.py`).
- **Intensity noise.** sqrt(D) H_drive(t) for the white part of the intensity spectrum.

Every spectrum is stored two-sided in angular frequency with the exp(-i omega t) kernel and a separately declared white level. The white level goes to the Lindblad operators; a tabulated band is sampled as a time series.

## Trap and crystal

- **Mathieu equation** d^2x/dxi^2 + [a - 2 q cos 2 xi] x = 0 with xi = Omega_rf t / 2. The stability and the characteristic exponent come from the monodromy matrix, and the Floquet function's Wronskian normalization Im(u* u') = nu gives C0 (`trap/mathieu.py`).
- **Pseudopotential** psi = Q^2 |E|^2 / (4 m Omega^2), with V_rf the peak of the cosine drive. Gapless-plane electrode potentials are in `trap/surface.py`.
- **Equilibrium and modes.** Newton iteration on the trap-plus-Coulomb energy from James 1998's linear chain gives the equilibrium. The mass-weighted Hessian's eigenproblem on all three axes gives the modes. A non-positive transverse eigenvalue means the chain has buckled (zigzag) and is refused (`trap/crystal.py`).

## Atomic structure and couplings

- **Hyperfine and Zeeman structure** by exact diagonalization of H_hfs + H_Z on every fine-structure level at the operating field. A_hfs is signed and g_I = -(mu_I / (I mu_N))(m_e/m_p). Breit-Rabi is the J = 1/2 check. Labels follow avoided crossings adiabatically, and clock points solve d nu/dB = 0 (`species/zeeman.py`).
- **Dipole elements** by Wigner-Eckart, with the reduced element from the measured lifetime. The saturation intensity is I_sat = pi h c Gamma / (3 lambda^3) with the angular partial rate (`species/dipole.py`). Polarization is decomposed about B into sigma+, pi and sigma- (`species/polarization.py`).
- **Raman couplings** Omega_R = sum_e Omega_2* Omega_1 / (2 Delta_e) in the hbar Omega/2 convention, light shifts delta = sum |Omega|^2 / (4 Delta_e), and the Raman, Rayleigh and differential-Rayleigh scattering amplitudes from explicit intermediate-state sums (`species/raman.py`, `light/raman.py`).
- **Electric-quadrupole coupling** for optical qubits uses the Racah-normalized rank-2 tensor (`species/quadrupole.py`).
- **Optical Bloch steady state** (`dynamics/multilevel.py`, `light/bloch.py`): every dressed sublevel of every included level, one frame per manifold with a Floquet fallback, and one collapse operator per decay line and polarization. The scattering rate W(Delta) = sum Tr(C^dag C rho_ss) is the single object that cooling, pumping and detection share.

## Cooling and preparation

- **Doppler** (`prep/doppler.py`, `prep/rates.py`): per ion and mode, the heating and cooling rates c_im^2 [S(+-nu) + 2D] from the dipole-force spectrum of the ion's Bloch steady state and the emission-recoil diffusion. The detuning of each beam group is optimized over the participation-weighted modes.
- **Resolved sideband** (`prep/sideband.py`): exact pulsed transfer matrices with higher sidebands and the Laguerre nodes where a single order strands population, the repump recoil kernel, and sideband-ratio thermometry R = nbar/(1 + nbar). EIT (`prep/eit.py`) and polarization-gradient cooling (`prep/polarization_gradient.py`) are alternative stages.
- **Optical pumping** (`prep/pumping.py`) takes the exact exponential of the multi-level Liouvillian and returns the preparation error, the photon count and the recoil heating. The recipe runs Doppler, then sideband or EIT, then the final pump (`prep/recipe.py`).

## Gates

- **Single-qubit.** GPi(phi) and GPi2(phi) are resonant carrier pulses of area pi and pi/2 at phase phi: R(theta, phi) = exp(-i theta sigma_phi / 2). Virtual RZ(theta) shifts every later pulse phase, phi -> phi - theta. Composite pulses (SK1, BB1, CORPSE, ...) are in `control/composite.py`.
- **Molmer-Sorensen.** Tones at +-mu from the carrier drive both ions. The spin-dependent force on S_y is -(hbar eta Omega / 2)(a^dag exp(i eps t) + h.c.). A single-mode square pulse is maximally entangling at eta Omega / eps = 1/(2 sqrt K). The closed-form displacement alpha_m(t) and entangling angle chi_ij drive the AM, FM and Fourier solvers that close every resolved mode's loop (`control/shaping.py`). The gate's error model is the residual displacement sum |alpha|^2 (2 nbar + 1), the thermal Debye-Waller error (pi^2/4) eta^4 <(n - n_ref)^2>, spectator loops, dephasing and intensity noise.
- **Light-shift ZZ gate** from the two-photon self-couplings of the qubit levels.
- **Native matrices** are defined in `control/native.py`: GPi, GPi2, MS(phi0, phi1, theta) with theta = pi/2 fully entangling, ZZ(theta) and RZ. The single-qubit generator carries theta/2, while XX(chi) = exp(-i chi sigma_x sigma_x) carries chi, maximally entangling at chi = pi/4.

## Readout

- **Rates**: the bright-state photon rate and the off-resonant pumping rates out of the bright and dark manifolds, from the detection beams' Bloch steady state (`readout/fluorescence.py`).
- **Records**: the exact count distribution over a window from one matrix exponential of the augmented Markov generator, or sampled records with dead time, afterpulsing and neighbour crosstalk as added counts (`readout/detection.py`).
- **Discriminators and POVM**: a threshold with its optimum, time-resolved maximum likelihood, adaptive and first-photon protocols, camera decoders; the product POVM at zero crosstalk and the register-wide confusion tensor otherwise (`readout/discriminate.py`). The figure of merit is eps = (eps_B + eps_D)/2, both reported; for 171Yb+ |0> (F = 0) is dark.

## Conventions

- **Units.** Public numbers are ordinary frequencies in Hz (`units.Hz`) and internal ones angular in rad/s (`units.RadPerS`), converted by `rad_s_from_hz` and `hz_from_rad_s`. A linewidth `Transition.gamma_hz` is the upper level's total decay rate over 2 pi. Public floats carry unit suffixes: `_hz`, `_s`, `_m`, `_w`, `_gauss`, `_rad`, `_v`, `_pa`, `_cps`, and `_turns` only at the IonQ boundary.
- **Rabi frequency.** H = (hbar Omega/2) sigma_+ e^{...} + h.c., so a carrier pi pulse takes Omega t = pi.
- **Detuning.** delta = omega_drive - omega_transition, so red is negative. A bichromatic gate's detuning mu is measured from the carrier, and delta_im = mu_i - omega_m.
- **Gate parameters** are radians internally and IonQ turns at the API boundary (1 turn = 2 pi).
- **Result bit order.** In every bitstring key of a `Result`, qubit 0 is the least-significant bit, the rightmost character: "101" on three qubits is qubit 0 = 1, qubit 1 = 0, qubit 2 = 1, and the IonQ v1 decimal key "5". IonQ's v2 result strings run the other way, q[0] first. `Result.final_state` is in QuTiP's tensor order, with ion 0 the first factor.
- **Tensor order of the native gate matrices**: the gate's first qubit is the left Kronecker factor (stated in `control/native.py`).
- **Fidelity measures, kept apart.**
  - A channel's average gate infidelity is 1 - F_avg = d/(d + 1) (1 - F_e), where F_e is the entanglement fidelity.
  - The depolarizing rate eps is the entanglement infidelity of Lambda(rho) = (1 - eps) rho + eps/(4^n - 1) sum_{P != I} P rho P. Qiskit's lambda = p 4^n / (4^n - 1) is a conversion, not the same number.
  - Randomized benchmarking reports r = (1 - p)(2^n - 1)/2^n per Clifford. Simultaneous RB fits each qubit's own marginal decay.
  - For a single-qubit unitary error, the composite-pulse and filter-function layers use F_K = |Tr(U_id^dag U)|^2 / 4.
- **Noise spectra** are two-sided in angular frequency. The single-sided S_E of the heating literature is converted at ingest.
- **Seeds.** One root `SeedSequence` per run is spawned by (sample, trajectory, shot, ion, channel), and shots are dealt to the dynamical samples in contiguous blocks. The same (shots, samples, seed, machine hash) reproduces every shot, whether the run went through `Machine.run` or a `Job`. A different `shots` changes the blocks and therefore the shots.
- **Identity.** `Device.hash()` is a canonical serialization: declaration-order fields, floats rounded to 12 significant digits, dicts by sorted key, `Qobj` fields excluded. It keys the calibration cache and every cached tomography.

## Approximations a run reports

- **Fock truncation.** Caps come from the pulse's coherent excursion at the 1e-6 tail plus a margin. The boundary population and margin are checked after every pulse, and a cap is raised and the pulse repeated when either fails (`Diagnostics.boundary_population`, `margin_reached`, `cap_growth`). A cap clamped at `mode_dimension_max`, or a boundary left above threshold, is a `TruncationWarning`.
- **Mode classes.**
  - Resolved: carried in the space.
  - Frozen: per-shot Debye-Waller factors, with the off-resonant excitation bound and the entangling-angle loss reported.
  - Dropped: contribution below 1e-6 in |alpha|^2 (2 nbar + 1) and 1e-4 rad in |chi|, with the summed contribution reported.
- **Branches.** The initial mixture is evolved as weighted pure branches. Branches below `branch_weight_min` are dropped and their weight is reported.
- **GATE_LOCAL.** Correlations between spin and motion, or between modes, left after a step are traced out. Every step reports its residual displacement and bound, the purity deficit of the motional state, the frozen excitation bound and the dropped crosstalk. Two relaxations tied to the map accuracy add their own terms to the step's `discrepancy_bound`: the branch-tail rule and the keyed tolerance.
- **Micromotion.** Runs use the pseudopotential frame with C0 in every Lamb-Dicke parameter and the J0(beta) carrier factor. Explicit rf dynamics is not on the default path.
- **Calibration.** The default table is the surrogate. Drift enters through the samples and an optional servo; recalibration inside a run is not modelled.
- **Inputs the physics cannot supply.** Anomalous heating (S_E is a device input), the noise spectra, apparatus readout quantities (detection efficiency, background) and the cited empirical species constants.
