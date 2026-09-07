# qutip-trap

A first-principles trapped-ion quantum computer simulator built on QuTiP. The specification is
[`PLAN.md`](PLAN.md); this repository implements it milestone by milestone (Section 10 of the plan).

**Status: milestones M0 (scaffolding and public interfaces), M0a (atomic structure layer), M1 (trap and
crystal), M2 (single ion, spin-motion coupling, single-qubit gates), M3a (multi-level optical-Bloch builder), M3 (cooling
and state preparation), M4 (two-ion entangling gates), M5 (readout), M6 (end-to-end circuits in JOINT_EXACT), M7 (noise
and error channels) and M8 (calibration emulation).** The
simulator now evolves one ion with its motional modes through the one Hamiltonian builder of Section 4.3: exact displacement
operators by matrix exponential asserted against the analytic Laguerre elements over the populated range (Section 5.1.1),
cached operators and marginals (ENR included), the boundary monitor with cap-raising retries (Section 5.5), Raman,
single-photon optical (E1 and E2) and microwave drives whose Rabi frequencies, Stark shifts and scattering rates are derived
from the beams and the atomic layer rather than entered (`light/`), frames (Schroedinger-motion default, the
sideband-decomposed interaction picture with `k_max` and `rwa` as declared approximations), the `dop853` -> `vern9` ladder
with the step-density record (Section 5.3), a JOINT_EXACT pulse engine (`JointExactEngine.run_pulses`) that segments a
schedule at pulse boundaries, applies frozen-spectator Debye-Waller factors drawn per shot from the keyed seeds, micromotion
J_0 factors or the rf-locked modulation, crosstalk terms, Stark and anharmonic terms and Cetina's beam-curvature coupling,
single-qubit native gates (GPi, GPi2) as pulses with virtual-RZ frame tracking through the scheduler subset of Section 7.3,
the composite-pulse library of Section 4.3.5, the frequency-comb tone set of Section 4.3.7, Harty's microwave
randomized-benchmarking model, and the Rabi, Ramsey, Ramsey-frequency and sideband-spectroscopy experiments on the engine.

M3a adds the shared multi-level optical-Bloch builder of Section 4.2.8 (`dynamics/multilevel.py`, re-exported by the one
builder module): every fine-structure level's dressed sublevels from the atomic layer, one frame frequency per manifold
assigned along the beams with the inconsistency detector (two tones on one transition, closed beam loops) and the
time-periodic fallback, one collapse operator per decay line and emitted polarization from the dipole operator itself
(the sum rule sum C^dag C = Gamma P_e holds to 1e-16), the recoil kernel of `light/recoil.py` (the vector form over a
direction quadrature that DERIVES every angular factor, its per-q marginal reduction and the second-moment-exact minimal
quadrature), and the leak policies include / sink / renormalize for decay targets outside the included manifold. On top of
it `light/bloch.py` is the single scattering-rate object of Section 13: `steadystate` (direct solve, or the fixed point of
the one-period propagator with period averaging when the frame graph is inconsistent or a polarization is modulated), the
photon rate per line asserted against the equal-population ceiling n_e/(n_e + n_g), the slow-manifold coarse graining that
yields R_o in the conditional bright state with the pumping rates R_d and R_b, dark-state counts from the coupling matrix,
optical pumping by the exact exponential of the Liouvillian, W(Delta) as a function of detuning, the level-A/B coefficients
A_+- = W(Delta -+ nu) + (eta~^2/eta^2) W(Delta) with a weak-drive guard, and the semi-analytic A_+- = 2 Re[S(-+ nu) + D]
from the dipole-force spectrum. `prep/level_c.py` is the one-mode level-C cooling solve (steady state, relaxation rate
from the Liouvillian's slowest eigenvalue, `mesolve` traces) and the level-B Fock rate equation. Acceptance tests
(`tests/test_bloch.py`, `test_recoil.py`, `test_level_c.py`): the RMP Eq. 96 Lorentzian to 1e-9; the 171Yb+ detection
rate (Gamma/18) s_o/[1 + (2/9) s_o] recovered to 0.5 % at its optimum field with the Gamma/4 ceiling; the leakage
prefactors from the angular algebra settling the Noek-Crain factor in Noek's favour, R_d = (2/27)(Gamma/2) s_o
(Gamma/2 Delta_HFP)^2, and R_b/R_d = 3/49; Berkeland's dark-state counts with the sigma+ unit test of Section 13; three
photons per optical pump into |0>; the Floquet fixed point against the secular static model; the 40Ca+ S-P-D dark
resonance; Stenholm's sideband floor (Gamma/2 nu)^2 [alpha/cos^2 theta_L + 1/4] from level C with all three recoil
discretizations (alpha -> 0 leaving (Gamma/4 nu)^2), the mixed pi + sigma channels needing one alpha per channel, the
Doppler limit (Gamma/4 nu)(1 + alpha) at Delta = -Gamma/2, the phonon rate equation to 1e-12, and Morigi's EIT figure.
M3 adds the cooling and preparation stages of Section 4.2 on top of the M3a builder (`prep/`). `light/recoil.py` gained the
per-ion participation and the joint multi-mode kick: the emitted photon's Lamb-Dicke parameter eta_em,{i,m} = k |c_{i,m}| x0_{i,m}
with the ion's own mass, the product of displacements over the modes for one sampled direction, and the recoil-energy identity
(the modes of one axis family share exactly alpha_axis (hbar k)^2/(2 m_i), to 1e-16, for single and mixed crystals). `prep/rates.py`
is the level-A engine every stage shares: per illuminated ion and mode the phonon heating and cooling rates c^2 [S(+-nu) + 2D]
from the dipole-force spectrum of the ion's internal steady state under all its beams (exact in the saturation) with the
emission-recoil diffusion per channel, the participation folded into the zero-point length so the rates carry W_k = sum_i c^2
and nbar is participation independent, and the guard that raises for a mode no cooling beam addresses. `prep/doppler.py`
produces nbar_D per mode at any nu/Gamma, optimizes one detuning per beam group over the participation-weighted mode set and
reports the RMP force model only where nu/Gamma < 0.1. `prep/sideband.py` holds the resolved-sideband and Raman closed forms
(Stenholm's floor with Marzoli's effective two-level parameters, the saturating rate), the exact pulsed transfer matrices with
higher sidebands and the three schedule optimizations, the Laguerre nodes where single-order cooling strands population, the
repump recoil kernel, and the thermometry of Section 4.2.7 (the exact sideband ratio, Rasmusson's tail sums, blue-sideband
flopping inversion, the double-thermal fit). `prep/eit.py` is Section 4.2.3 in the plan's sign with the Zeeman-resolved level-C
model; `prep/polarization_gradient.py` carries Joshi's analytic j = 1/2 <-> 1/2 model (with the static-gradient raise) and builds
the Lindblad layer through the M3a builder (`PolGradientBeams.xi/limits/moving_gradient_ok` now work); `prep/pumping.py`
evolves the pump from the scrambled manifold and returns the preparation error, duration, photon count and per-mode recoil
heating; `prep/sequence.py` enforces the Doppler -> sideband/EIT -> final-pump order and hands off the Appendix E `State`.
Acceptance tests (`tests/test_recoil_modes.py`, `test_cooling_closed_forms.py`, `test_doppler.py`, `test_sideband_cooling.py`,
`test_eit.py`, `test_polarization_gradient.py`, `test_pumping.py`, `test_prep_sequence.py`): the Doppler fork 0.480/0.336 mK,
the force model's 349.5 against 350, Stenholm's 7/12 and 13/20, the Morigi-Walther steady-state list, Marzoli's 0.55/5.5 and
-0.058, Cirac's 0.554 and 2.66, Che's 0.031117 repump recoil, the Laguerre nodes 39.79/71.77/112.29/202.01, Che's accumulation
centres and Rasmusson's fixed-pulse prediction, the Turchette ratio to 1e-13, the EIT fixture 0.005102 with its negative controls
and the RMP tuning 0.00361702, the PGC minima 1/2 and 0.8693 with alpha = 1/3, the 171Yb+ pump's three photons and 8.5e-3 quanta
of recoil, and the level-C checks of the Doppler limit, the PGC limit (0.756 against 0.75) and the multi-ion participation
weights.

M4 adds the two-qubit entangling gates of Section 4.4 on the same builder: nothing is a separate gate implementation, an MS
pulse is one `Drive` per ion carrying the red and blue tones (the coefficient already summed tones), and the exact dynamics
(carrier, every sideband, Debye-Waller factors, spectator modes) come with it. `control/shaping.py` holds the Section 4.4.3
integrals with the symmetrized two-body kernel in two forms that are never mixed (the exact first-order Lamb-Dicke kernel
cos(mu t - phi_m) e^{i omega t}, Choi's sine form being its phi_m = pi/2 case, and the slow-envelope kernel), analytic for
segmented pulses (F_k, the triangle T_k and Im F_l F_k^*, checked against scipy quadrature to 1e-9) and by Simpson's rule for
sampled ones, and the three solvers in the plan's pi/4 convention: Zhu/Choi's segmented AM (2N + 1 segments close N modes,
the power-optimal null-space direction), Leung's vertex-parameterized FM with the time-averaged-trajectory robustness cost,
and Blumel's Fourier-sine AM with K-th-order frequency-derivative stabilization and the extended-null-space relaxation.
`Waveform.symmetric` is the square bichromatic pulse (eta Omega/eps = 1/(2 sqrt K) on one mode, spectators included in chi);
`Waveform.chi_m` is signed; the scheduler plays `ms` and `zz` from the table (MS(phi_0, phi_1, theta) sets the spin phases
phi_i - pi/2 because the force acts about phi_s + pi/2, a positive kernel sign is a pi on the second ion, partial angles
rescale by the s^2 law; ZZ on an MS waveform is the inferred GPi2-wrapper construction, on a light-shift waveform the
spin-echo pair of Section 4.4.4). The `light_shift` drive kind derives Zhu's sigma_z force from the atomic layer (the
two-photon self-couplings of the two qubit levels, weights differing by exactly 2, the spin-independent force and the
far-off-resonant Raman coupling kept). `calibration/entangling.py` is Section 7.5's spot check: the closed-form waveform
played through the JOINT_EXACT engine, chi read from P_11 = sin^2 chi (or in the x basis through the echo pair for a sigma_z
force), the residual displacement read exactly as the final mode energy (sum_j |alpha_j|^2), the amplitude corrected by
sqrt(chi_target/chi) until |chi| = pi/4, and the thermal robustness curve; `ms_scan` and `parity_scan` are the laboratory's
population and parity scans on the engine. Acceptance tests (`tests/test_shaping.py`, `test_two_qubit_gates.py`,
`test_entangling_schedule.py`, `test_light_shift_gate.py`, `test_gate_calibration.py`; `validation/scripts/check_two_qubit.py`):
the check_ms_closure.py anchors through the builder (concurrence 0.9999 with populations (0.5074, 0, 0, 0.4926) at eta Omega/eps
= 1/2, 0.3825 at 1/4), the exact propagator D(alpha S_y) exp[i gamma S_y^2] and Kirchmair's thermal envelopes at first order in
eta, the Debye-Waller law and the three thermal references (n0 recovered, the re-optimized one excluded), the symmetrized kernel
against block-diagonal exact integration (the printed factor 2 wrong by tens of percent for non-proportional envelopes), the
residual-displacement conversions (eps_ent = final mode energy, 1 - F_ent, Landsman's 4/5), the two-mode 171Yb+ gate (the
symmetric pulse's open rocking-mode loop against sum |alpha|^2, the five-segment AM closure, exact chi 0.772 against the
surrogate 0.785 corrected in two checks to 0.785398 with Bell fidelity 0.99977), Ballance's alpha_K = 11/16, 19/64, 35/256 from
block Liouvillians, Choi's 11 and 21 segments for five ions with one and two transverse families, Leung's FM robustness and
Blumel's stabilization, Baldwin's echo diag(1, i, i, 1), the 40Ca+ optical-qubit light-shift ZZ gate from first principles
(fidelity above 0.99 after calibration), and the native MS(phi_0, phi_1, theta), partial angles, virtual-Z frames and the
IonQ JSON path through `schedule`. `run.prepare` arrived with M6 (the Device carries the preparation recipe); the
microwave-gradient drive of Section 4.4.5 exists as closed forms (Srinivas's J_2 factor and the
IDD ratio 0.6012) but not as a device-level drive.

M5 adds the readout of Section 8 in the four layers the plan keeps separate (`readout/`). The atomic-rate layer
(`fluorescence.py`) consumes the M3a scattering-rate object: `scattering_rate` builds the multi-level Bloch model of the
device's detection beams, takes the bright manifold from what the beams drive resonantly and the dark manifold from the
remaining sublevels, and reads R_o (the photon rate of the conditional bright state), R_d and R_b from the slow-manifold
analysis with the equal-population ceiling asserted; bright/dark polarity is a property of the `ReadoutScheme`, whose
shelving transfer is a start distribution per level (Pi_dark is not rank one); epsilon_sys enters once. The closed forms of
the sources (the (Gamma/18, 2/9) form, Noek's and Crain's saturation parameters, Acton's lambda_0, alpha_1, alpha_2 and
angular factors, the 4/9 clock-state ceiling, the corrected I_sat and neighbour-intensity ratio, the efficiency chain,
micromotion J_0^2/J_1^2 factors, Doppler widths, shelf decay and branching) are oracles, and published operating points are
`presets.py` apparatus data, never species constants. The record layer (`detection.py`) is the bright/dark/shelf
continuous-time Markov chain: its exact count distribution over a window is one matrix exponential of the augmented
generator (any number of jumps), the fast path samples the chain exactly with Poisson counts at the piecewise rates (sub-bins,
arrival times, dead time and afterpulsing on request), the trajectory path runs `mcsolve` on the class space with a
photon-counting collapse operator, neighbour crosstalk adds a bright neighbour's leaked light to its neighbours' records, and
a camera model integrates an Airy or Gaussian point-spread function over a pixel grid. The discriminator layer
(`discriminate.py`) has thresholding with the interior optimum over (n_c, t_b), Myerson's time-resolved maximum likelihood
(the O(N) recursion in the log domain, plus the exact hidden-Markov forward likelihood), adaptive early termination, Noek's
two-photon and Crain's first-photon protocols, and Burrell's camera decoders (brightness-ordered pixel likelihoods,
neighbour-conditioned iterated conditional modes, the register error estimate sum e^-R_k). The POVM of Section 5.7 is the
product form at zero crosstalk and a register-wide confusion factored by neighbour range otherwise (dense to N = 12);
`measure` samples the joint internal outcome projectively and then either generates every ion's record (full path) or applies
the POVM (fast path), never both, with seeds keyed per (sample, trajectory, shot, ion, channel). `calibration/readout.py`
is Section 7.5 item 5 (histograms, the mean-count fit over long windows, the threshold and window choice) and
`experiments.detection_histogram` runs it on a device. Acceptance tests (`tests/test_readout_rates.py`,
`test_photon_records.py`, `test_discriminators.py`, `test_readout_povm.py`, `test_readout_budgets.py`;
`validation/scripts/check_readout.py`): the Bloch rates within 0.6 % of the closed forms with R_b/R_d = 3/49, Acton's
M_1 = 2/9 .. 12/49 and 99.9375 %, 1.669 mW/cm^2 and 1.09e-4, Crain's Eq. 1 normalization and the 6.3e-3 against 7.2e-4 bright
error, the zero-threshold optimum 5.84e-4 near 22 us, the exact chain against the single-jump forms to 1e-13 and against
Acton's mixtures to 1e-15, Myerson's recursion to 1e-14 with the ideal-Poisson optimum 1.24e-4 at (3.5, 320 us) and 1.37e-4
at his (5.5, 420 us) against the measured 1.8(1)e-4, the ML asymptote and the adaptive times, Burrell's eps_D floor
t_exp/(2 tau) = 1.7e-4 and the PSF leakage, the product POVM against the full path at zero crosstalk with the Bell state's
correlations surviving both and the 0.34 discrepancy at 4 % leakage, Harty's 6.8e-4 and Christensen's 3.4e-4 budgets, the
Gaussian spectator dephasing as a quasi-static offset, and the calibration recovering (eps R_o, R_d, R_b).

M6 closes the loop of Section 3.4: `run(circuit, device, shots)` compiles, calibrates, schedules, prepares, evolves and reads
out, and returns the `Result` of Section 8.6. The compiler (`control/compiler.py`) is the one place ideal gate matrices are
used: single-qubit gates become one virtual rz, one GPi/GPi2 pulse or the ZXZXZ pair (20 random SU(2) targets to 1.5e-15),
CNOT is Maslov's one-XX template (equal to e^(i pi v s/4) CNOT for all four signs), CP(theta) = RZ(theta/2)^x2 ZZ(-theta/2)
exactly (Debnath's fixed-rotation template is kept as the plan's defective one, overlaps 0.854 and 0.691), ZZ is the native
gate or the M4 wrapper, every block and the whole circuit are verified up to a global phase, and the virtual-Z frames are
absorbed into the later pulse phases so the compiled circuit carries only gpi, gpi2, ms and zz plus the residual frame the
measurement discards (the IonQ JSON round trip holds). The scheduler makes the terminal measure a `ScheduledEvent` of the
table's detection window, refuses mid-circuit measure/reset/recool with the Section 8.5 message, records every entangling
gate as played, scales a two-photon drive's Stark shift linearly with its amplitude (the M4 quadratic rule was the
single-photon case) and resets the bichromatic beat note at each gate start when the hardware is not phase-continuous.
`prep/recipe.py` carries the laboratory's procedure on the `Device` (`PreparationRecipe`; `standard_recipe` derives the
171Yb+ one from the device's detection and Raman beams: Doppler light at the detuning that minimizes the gate modes'
occupation, pulsed Raman sideband cooling of the coupled modes with the exact transfer matrices and the repump recoil
kernel, the pump on F=1 -> F'=1) and `run_preparation` evaluates it with the M3 stages. `run/space.py` assigns every mode
its Section 5.2 class from the Section 4.4.3 integrals of the played waveforms (dropped, frozen with its chi and residual
reported, resolved with a cap from its loop radius) and is `HilbertSpace.for_`. `calibration/surrogate.py` is the Section
7.5 default: derived seeds, the AM or symmetric waveform per pair with the beat note above the highest coupled mode, the
exact spot check on the reduced space when it fits the 4096-dimension guard, the detection threshold and window. The
initial state (pumped populations x thermal modes, a mixture diagonal in the computational and Fock bases) is evolved as
weighted pure branches through the JOINT_EXACT engine (the Fock-sum path of Section 5.3, branches below
`SolverOptions.branch_weight_min` dropped and reported), the recombined register state is measured with the M5 POVM
(fast) or the photon-record path (full) with the Section 3.4 seed keys, and `Diagnostics` reports the level, space,
mode classes, boundary populations, branch count, SPAM with its definition and the intrinsic budget of Section 9.6
(residual displacement, Debye-Waller, the (Omega/nu)^2 carrier scale, the frozen chi, Roos's beat-phase tilt, the
single-qubit pulses' sideband scale and addressing crosstalk). The
OpenQASM 2 importer (`io/openqasm.py`) inlines custom gate declarations (the client SDKs' gpi/gpi2/ms/zz), broadcasts
registers and evaluates parameter expressions. Acceptance (`tests/test_compiler.py`, `test_openqasm.py`,
`test_run_circuits.py`; `validation/scripts/check_circuits.py`): on the two-ion 171Yb+ fixture with a global 355 nm pair,
per-ion addressing pairs (2.2 % Rabi crosstalk from the 2.5 um waist) and an oblique cooling/detection beam, the Bell
circuit's 4000-shot histogram is 00: 0.5015, 11: 0.4965, 01: 0.0013, 10: 0.0008 against the ideal 0.5/0.5, the register
infidelity 2.4e-3 sits inside the reported budget 8.2e-3 (the carrier scale dominates the bound; the gate's own exact
check gives 1.3e-4), SPAM (eps_B, eps_D) = (6.1e-4, 3.5e-4) with a 2.5e-6 preparation error, Doppler occupations 3.5 to
11.6 fall to 0.015 to 0.019 on the gate modes after 40 sideband pulses, the six initial-mixture branches carry all but
3e-5 of the weight, and the same root seed reproduces every shot.
On the three-ion fixture (2.0 um addressing waist, 1.3 % crosstalk) the global symmetric Molmer-Sorensen pulse at t =
pi/(8 chi) reaches 0.9823 against its pairwise closed-form target (0.9735 against the ideal single-mode GHZ unitary)
with the top two x modes resolved at d_m = 12 and the zigzag frozen with its 1.1e-3 residual reported, the four-ion
pulse 0.9672 (0.9373) at dimension 2304, and the H, CNOT, CNOT GHZ circuit through `run` gives 000: 0.5155, 111: 0.4775
over 2000 shots with a register infidelity 1.5e-2 inside its 3.0e-2 budget (the seven-segment adjacent-pair gates check
at 0.9953 with 4.4e-3 leakage). Bernstein-Vazirani with the secret 01 on the outer ions and the middle ion as the
ancilla (Section 9.6 row 6) declares the 1 -> 0 flip of the secret 3.3 times as often as the 0 -> 1 flip (2.4e-3 against
7.3e-4), the dominance of Wright's minimal model; here it comes mostly from the coherent circuit (the matrix model with
eps theta rotations on the neighbours and an ideal MS gate gives 5.4, because the oracle CNOT maps the ancilla's
crosstalk rotations onto its control, the secret's 1 bit, while the 0 bit sees only the ancilla pulses' direct
rotations) and the readout's eps_B > eps_D adds to both flips.

M7 gives the simulator its noise and error channels (Section 6), every one a physical process routed by Section 6.1's rule
rather than a phenomenological channel. `noise/spectra.py` keeps every spectrum two-sided in angular frequency with the
e^{-i omega t} kernel and declares its white part as a separate field (`NoiseSpectrum.white_level`, the Lindblad route),
the tabulated band being the sampled-trajectory route: `noise/processes.py` synthesizes Gaussian realizations on a fixed
grid with (1/pi) int S d omega as the variance, so an integrator's step sequence never changes what the drive sees
(Section 9.17). `NoiseModel.channels` turns S_E through the mode-projected multi-ion formula with the configured
correlation length (never the uniform limit by default) into heating operators, the white rf-amplitude density into the
motional dephasing operator a^dag a sqrt(2/tau) on the rf-derived modes, and the white field density through the computed
Zeeman sensitivities into sqrt(gamma/2) sigma_z with gamma = 2 pi^2 (d nu/dB)^2 S_B (a Ramsey coherence decays as
e^{-gamma t}); `NoiseModel.sample_sequence` draws every Drift as an Ornstein-Uhlenbeck chain over the shot clock with its
ramp, converts the field offset through the exact diagonalization at the shifted field, moves every transverse mode by the
common rf fraction plus its differential drift, draws beam phases and pointing offsets, and synthesizes the per-ion
transition-frequency trajectories (S_B through d nu/dB and d^2 nu/dB^2 plus the mains at a per-shot trigger phase), the laser
phase and intensity trajectories and the rf-amplitude trajectory. The Hamiltonian builder consumes all of it: a time-dependent
(delta nu_i(t)/2) sigma_z^i and omega_m dV/V(t) a^dag a term, beam-path phases as e^{-i Delta phi} on sigma_+, pointing and
stray-field displacements as intensity factors on Omega and the crosstalk ratios together, the laser phase on single-photon
drives and (1 + dI/I)^p on every laser drive, and it exposes each pulse's drive term for the white intensity-noise channel
sqrt(D) H_drive(t). `noise/scattering.py` builds the photon-scattering operators of every pulse from the atomic layer's
signed amplitudes: the diagonal Rayleigh operator over the register levels (Uys's Gamma_el/4 dissipator automatically), one
Raman operator per (a -> b), leakage into a SINK level when the register factor has d > 2 (`noise/levels.py`; the leaked F = 1
sublevels of 171Yb+ read bright, the SINK dark) and a reported estimate at d = 2, each with the recoil D(i(eta_abs -
eta_em)) of the absorbed beam and a moment-exact six-direction emission quadrature about B, at the played intensity. The
engine integrates a segment with collapse operators through `mesolve` up to `SolverOptions.mesolve_dimension_max` and through
`SolverOptions.ntraj` keyed quantum-jump trajectories above it (each trajectory identical under its own seed whatever the
worker count, the jumps returned in `Traces.jumps`), and passes every schedule through the control hardware chain of Section
7.10 (`control/hardware.py`: DDS words, the modulator's first-order response with an 8 tau tail continuous across a pulse
train, the amplifier bandwidth for microwaves, saturation, rigid per-train timing jitter from the keyed seeds; the drive's
Stark shift follows the played light with the drive kind's scaling power, so a tail carries the decaying shift and not the
programmed one).
`noise/decoupling.py` is the filter-function layer of Section 6.9 with the full 3 x 3 toggling machinery (the closed-form
segment integrals with removable poles, CPMG, UDD, XY4, XY8, KDD, CDD, custom timings; the amplitude quadrature and its dc
polygon; chi = (2/pi) int S_b F/omega^2 with omega_min and d ln chi/d ln omega_min reported; the dc floor and the max rule;
a Monte Carlo over sampled trajectories through the builder as the route (c)/(d) closure; `DecouplingSequence.moments`
counts the pulses from j = 1 as Biercuk's sum does, so A_1 = (-1)^n/2 is its first-order cancellation condition, and
`feasible()` is a query on the record while `decoupling_sequence` refuses infeasible timings), `noise/collisions.py` the
Langevin collision process, `noise/summary.py` the Section 6.8 reporting (entanglement and average infidelity, the Pauli
twirl, Chen's depolarizing normalization). `run()` distributes shots round-robin over dynamical samples at the shot clock
(default min(shots, 64) unless the model is quiet), evolves every initial-mixture branch of every sample, reads each sample's
register state out with the seeds keyed by (sample, trajectory, shot, ion, channel), derives the histogram's error bars from
the between/within-sample effective sample size, applies the collision process per shot (heralds, discarded shots, a
permuted ion order, dark and lost ions read dark for every later shot), carries leakage levels (`internal_levels`) and the
Section 6.6 echo schemes (`crosstalk_suppression="local"|"neighbour"`), and adds the per-pulse scattering probabilities to
the intrinsic budget. Acceptance (`tests/test_noise_*.py`, `test_decoupling.py`, `test_hardware_chain.py`,
`test_scattering_channels.py`, `test_collisions.py`, `test_run_noise.py`; `validation/scripts/check_noise.py`): heating and
motional dephasing during the two-ion gate reproduce Ballance's ndot t_g/(2K) and alpha_K t_g/tau to 3-4 % at K = 1 and 2
(alpha_K = 11/16, 19/64); Fang's exact crosstalk forms (0.900790, 0.097217) and the echo identities to 1e-15; the scattering
operators' sum rule to 1e-9 with the flip and leakage rates in `mesolve`; the filter-function machinery against every
`check_composite.py` number (gated CPMG equals Biercuk to 1e-15, the finite-pulse UDD collapse to 4 and 6 with the
universal 1/16 and 1/64, the SK1/BB1 dc floors 5.87365e-6 and 3.53675e-9); the OU heating slope 9.97506e-4 with its rival
prefactors excluded; the Langevin rates of `check_collisions.py`; and the two-ion Bell circuit through `run()` with a quiet
model reproducing M6 (register infidelity 2.35e-3 to 2.36e-3 with the branch threshold, M6's 2.353e-3) and with heating, a
field drift and a Rabi drift on the trajectory path.

M8 makes the machine act on what it believes and the ions on what they are (Sections 7.3, 7.5). Every pulse the scheduler emits
now carries the CalibrationTable's beliefs and is marked `Drive.programmed`; the engine's played chain (`control/played.py`)
converts a requested Rabi frequency through the device's derived rf-power-to-Omega map divided by the calibrated one
(Omega_phys = Omega_req Omega_derived/Omega_table, Section 7.10), the light shift to the derived shift at the played intensity and
the crosstalk to the derived ratios of the listed neighbours, so that a miscalibrated entry is the over-rotation, detuning or
crosstalk error a laboratory's would be while a surrogate table (seeds = derived values) leaves M2 to M7 unchanged. The
scheduler compensates the believed differential light shift the way a laboratory does, by detuning every tone of a pulse by it
(both legs of a bichromatic segment together, per segment amplitude), and absorbs the z-rotation 2 pi int delta dt the shifted qubit
accumulates during the pulse into the ion's virtual-Z frame (Section 7.5 item 7, "the light shift that sets the MS phase"): at the
two-ion fixture's gate amplitude (two legs at 195 kHz) the shift is -149 Hz and the 100 us gate accumulates 0.094 rad per ion, which
uncompensated costs 8e-3 of Bell fidelity and which M6's tables never carried, because the surrogate seeded no entry for the
entangling drive's beams; it now seeds every drive, and `register_fidelity`, `exact_gate_check` and the M4 schedule tests rotate
their targets by the schedule's final frame (`frame_rotated`). The tones are phase-continuous in absolute time, so a compensation
detuning also needs a phase reference: every compensated tone carries 2 pi delta_s t_start (`compensation_phase_rad`), minus the
frame the earlier segments of the same gate accumulated; without it the axis of a compensated GPi2 one millisecond into a schedule
was off by 0.24 rad (1.5e-2 of fidelity, exact at t = 0), the Bell circuit through `run` read 4.2e-3 against M6's 2.35e-3 and the
three-ion GHZ 5.3e-2 against 1.5e-2; with it both return to M6's numbers. The experiments package holds the simulated experiments of Section 7.5
on the JOINT_EXACT engine, every one reading its populations through an observation model (`shots` declared through the readout's
(eps_B, eps_D), keyed by the Section 3.4 seeds with one stream per experiment and per sub-scan) and fitting by weighted least squares so that every parameter carries the
uncertainty a laboratory would quote: `field_scan` (the Ramsey-frequency experiment inverted through the exact nu(B) of the atomic
layer, 3.1 kHz/G on the clock transition at 5 G), `micromotion_scan` (Berkeland's three methods from the physics itself: the exact
e^{i beta cos Omega_rf t} modulation of an rf-locked drive for the sideband ratio, and the periodic steady state of the detection
beam's Bloch model under the Doppler-modulated detuning for the rf-photon correlation and the Doppler nulling; the shims scanned,
the null fitted, shims and residual beta stored), `mode_spectroscopy` (the coarse scan, fine scans fitted with the plan's lineshape
P = [Omega^2/(Omega^2 + delta^2)] sin^2((t/2) sqrt(Omega^2 + delta^2)) rather than the half-Rabi form, the mode frequency as blue
centre minus a weak long carrier's centre, |eta| from the sideband Rabi frequency with C0 inside, nbar from the sideband ratio),
`thermometry` (Turchette's exact ratio with the thermality check), `rabi_scan` with nbar fixed from the thermometry and the
Debye-Waller factor of EVERY coupled mode, `stark_scan` per beam (one beam on during the Ramsey delay), `crosstalk_scan` (the
neighbours' Rabi rates and the crosstalk axis from a pi/2 - pi - pi/2(phi) sequence), `ms_scan` with the closure detuning from the
leakage parabola and the entangling amplitude from cos(2 chi_1 s^2), `ms_phase_scan` (Section 7.5 step 3: the entangling axis
against the single-qubit frame from |00> and |01>, giving both ions' corrections), `parity_scan`, `heating_rate` (the delay scan of
the sideband asymmetry with the device's heating channels active, in the interaction frame) and `crystal_image` (dark, lost and,
for mixed species, reordered ions). `calibrate(surrogate=False)` (`calibration/experiments.py`) runs them in the dependency order
field -> micromotion -> modes -> rabi, stark, qubit frequency -> crosstalk -> entangling scans -> detection, heating, refuses a
downstream fit whose upstream entry group is `uncalibrated` and marks a fit that fails or lands at its scan edge `uncalibrated`,
fits under one dynamical sample of the noise model whose id every entry carries, stores the surrogate's error beside every entry it
replaces, and is cached per device hash and seed (`calibration/cache.py`: a changed device parameter invalidates, never regenerates
silently). `Drift.servo_bandwidth_hz` is applied: a drift with a servo is high-passed into its residual band over the shot clock
(`servo_residual`). `Device.derived()` gathers the derived quantities with their ledger ids. Acceptance (`tests/test_calibration_layer.py`,
`test_calibration_experiments.py`, `test_full_calibration.py`; `validation/scripts/check_calibration.py`): the Section 9.17 lineshape
row (the fitted pi time pi/Omega, the half depth at delta = Omega, the half-Rabi form returning Omega/2), the C0 row (the
sideband-calibrated eta of a single ion at q = 0.3 carries C0 = 1.018 and a gate built from the carrier-derived one misses 2(C0 - 1)
of two-body phase), thermometry exact to 2e-3 with the thermality flag, the heating rate recovered to 3 % from the engine's own
channels, the field to 1 mG from a 20 mG-wrong seed, the two-ion fixture's light shift (-42 +- 11 Hz against -38.4), crosstalk
(0.02218 +- 0.00017 against 0.02202) and crosstalk axis (0.00 +- 0.02 rad), the mode frequency to 275 Hz after the extrapolation of
the sideband's own 1.5 kHz carrier light shift to zero power, the micromotion nulls at -30.00 +- 0.46 V/m (sideband ratio, residual |beta| 2.5e-4) and
-19.3 +- 0.8 V/m against -20 (rf-photon correlation with the beam retuned to -Gamma/2, Berkeland's working point, and the first
harmonic projected on the atom's response phase; 10 s of photon shot noise), and the full calibration of the two-ion fixture by simulated experiments (`check_calibration.py` section 5, reduced scans, 400 shots per point): no refusal and no uncalibrated entry; the field 4.99997 +- 0.00129 G against 5; the qubit frequencies within 7 Hz of the truth; the carrier Rabi frequencies 100886 +- 55 and 100966 +- 51 Hz against 100928 (0.8 sigma); the light shifts -42.1 +- 7.4 and -45.8 +- 7.2 Hz against -38.4; the crosstalk ratios 0.02199 +- 0.00029 and 0.02189 +- 0.00028 against 0.02202; the modes 2.82848 +- 0.00024 and 2.99986 +- 0.00021 MHz against 2.82843 and 3.00000; the entangling closure at scale 1.0019 +- 0.0084 with spin-phase corrections -0.031 +- 0.039 and -0.014 +- 0.039 rad, parity contrast 0.9863 and a Bell fidelity bound 0.9932; the surrogate's error 4e-4 (Rabi), 5e-5 (modes), 6e-12 (qubit frequencies), 0.19 (Stark) and 6e-3 (crosstalk); and the Bell circuit run from the fitted table at 1 - F = 3.5e-3 against 2.3e-3 from the surrogate table and an intrinsic budget of 9.0e-3.

The M8 test suite exposed where the engine's wall time went, and three numerics changes (none an approximation beyond the
declared branch threshold) bring the calibration from hours to minutes (`tests/test_engine_numerics.py` pins each against the path
it replaces). (i) A segment whose Hamiltonian is constant, an idle interval or a zero-envelope pulse carrying only its light shift,
is propagated by its exact phases (H_mot + H_int + H_Stark are diagonal in the Fock x computational basis) instead of the ODE
ladder, which spent 10^6 right-hand sides resolving the 3 MHz rotation of a 2 ms Ramsey delay; with the device's collapse operators
present the idle is integrated in the frame rotating with H_mot, where the heating, motional-dephasing and qubit-dephasing operators
are eigenoperators of ad_H and their dissipators unchanged; the segment reports integrator `exact` (`JointExactEngine.closed_form_constant`).
The Ramsey-type scans of Section 7.5 (field, Stark, qubit frequency) fell from 33, 271 and 73 s to 0.6, 1.3 and 1.0 s on the two-ion
fixture. (ii) A density matrix evolved without collapse operators is decomposed into the weighted pure branches of its
eigen-decomposition and each goes through `sesolve` (the Fock-sum path of Section 5.3, branches below `branch_weight_min` dropped
and reported): the entangling scans start from the thermal motional state, which the first M8 build integrated as a 400^2
Liouvillian at 58 ms per right-hand side, about two hours per gate check and thirty checks per calibration
(`JointExactEngine.pure_branches`); the entangling experiments take `branch_weight_min` (default 1e-3, the single-ion experiments'
default) so a gate check is three branches at 7 s each. (iii) The plain and conjugate drive terms, and every sideband term of the
interaction picture, share one memoized tone-sum evaluation per time, square tones skip their callables, and a drive whose tones are
identically zero adds no operator term. The heating-rate scan sizes its truncation per delay rather than for the hottest delay of the
scan. The same run surfaced an M8 bug the surrogate tables had hidden: two simultaneous pulses of different lengths (the parity
scan's analysis pulses at two ions' FITTED Rabi frequencies, whose pi/2 times differ) made the builder refuse the segment, because it
demanded coinciding pulses where the engine's contract only asks that every pulse span the segment; the builder now takes the common
interval and gives every pulse its own clock. What remains is the gate segments themselves: a 100 us Mølmer-Sørensen pulse at
dimension 400 costs 7 s per branch (29 000 right-hand sides per 20 us segment at 68 us each, four CSR drive terms of 20 000
non-zeros), and the merged dense operator of Section 11.1 does not pay in this QuTiP build (two dense elements in a `QobjEvo`
cost 42 us against 33 us for the four CSR terms at this size), so the mode-factorized kernel and the branch and scan-point
parallelism of M9b are the next factor.

Plan inconsistencies surfaced by M8 (ledger `conv.*` records of the calibration layer, `anchor.m8.*`): a resonant sideband
pulse light-shifts the qubit through its own off-resonant carrier coupling by Omega^2/(2 omega_m) (1.70 kHz at Omega/2pi =
100.9 kHz, omega_m/2pi = 3 MHz), pulling both sideband resonances toward the carrier, so Section 7.5's single fine scan at full
power would write the mode frequency 1.5 kHz low into the table, above the FM solvers' sub-kilohertz need (the scan fits the
blue centre at two rf amplitudes and extrapolates to zero power); the both-beams-on Stark measurement of item 7 carries the odd
coupling shift Omega^2/(2 Delta) (1.27 kHz at Delta = 4 MHz), whose fringe aliases with the plan's Ramsey delays, so the shift is
measured per beam with the other blocked; a pi/2 crosstalk rotation about an axis parallel to the preparation pulse's gives no
fringe (the axis is measured with a pi rotation, the fringe phase twice the axis angle); the carrier Rabi fit must carry the
Debye-Waller factor of every coupled mode (the frozen COM mode's e^{-eta^2/2} = 0.9969 alone biases the two-ion fixture's Omega by
400 Hz, nine standard errors); the differential light shift at a gate's amplitude is the pair's shift times sum_legs
Omega_leg/Omega_cal (3.9 x on the fixture), and its accumulated phase is a virtual-Z the plan's item 7 names but no section
schedules; M4's two-ion fixture drove 100 kHz tables with 10 mW, 200 um beams that physically give 303 Hz, which a machine playing
the physical Rabi frequency cannot (the fixture now carries 0.3 W in 60 um); and the 40Ca+ light-shift fixture's 729 nm E2
Rabi frequency is not derived for its beam geometry (0), so its table entry is a supplied value the chain plays as physical
and says so.

Plan inconsistencies surfaced by M7 (ledger `conv.*` records of the noise layer, `anchor.m7.*`): the control hardware
chain reintroduces Roos's spin-axis tilt after the per-gate beat-note reset, because the modulator's first-order response
delays the envelope while the beat note advances 2 pi mu tau_r = 0.96 rad on the two-ion fixture (fidelity 0.99700, leakage
2.9e-3 uncompensated); the tone phases must carry the phase of the response at the beat frequency, arctan(2 pi mu tau_r)
(0.99992 / 3.3e-5 against 0.99987 / 4.5e-5 for an ideal modulator; the linear mu tau_r leaves 2.2e-4), a reference no source
states and one M8's phase scans will absorb; the Section 9.16 row 4.4-8 intensity-noise budget reproduces its two-term
structure and the Gamma_I-independent ratio B/A = 3(N - 1)/(4K) exactly, but with c_op = sqrt(2k) H_int and Gamma_I = k
Omega^2 defined by the carrier-contrast rate (checked analytically) both terms are twice the plan's derived A = Gamma_I t_g
eta^2/2 and B, so the plan's A pairs with Gamma_I equal to half the contrast rate or with another fidelity measure; the
derived dephasing forms N t_g/(2 T2) and N^2 t_g/(4 T2) are bounds (Cov_t <= 1) and the N = 3 force model sits at 0.67 and
0.68 of them with the global > local ordering; Ballance's gate budgets hold for the gate as calibrated, and a symmetric pulse
whose two-mode closure is played with one mode frozen returns 0.84 and 0.80 of them; the row 4.1-3 estimator ratio 0.99991
is not reproduced (the exact double integral over 200 gives 0.99975 of the slope, the slope itself and the rival prefactors
are); Biercuk's finite-pulse filter function assumes the noise gated off during the pulses, so the toggling machinery
carries a `gated` option, and with the noise on through a finite pulse every odd-n sequence keeps a first-order transverse
term (F proportional to omega^2); the KDD sequence needs 20 pulses (an XY4 cycle of Knill blocks) to return to identity,
two blocks being a pi rotation about z; and Bermudez's three-variant Pauli mapping of Section 6.8 is not transcribed in the
plan and is not provided.

Plan inconsistencies surfaced by M6 (ledger `conv.*` records of the compiler, scheduler and run path, `anchor.m6.*`):
Section 7.7's CP overlaps 0.854 (pi/2) and 0.691 (pi/4) belong to Debnath's template with fixed RZ(pi/2) rotations,
while CP(theta) = RZ(theta/2)^x2 ZZ(-theta/2) is exact; the same section's s = sgn(chi) is a free convention here
because the scheduler realizes either MS sign from the waveform's signed chi; the M4 Stark scaling (Omega/Omega_cal)^2
is the single-photon rule and turned -0.1 Hz into -70 kHz for a Raman waveform played at 550x a weak pair's derived Rabi
frequency (a two-photon shift scales linearly); with phase-continuous tones the MS spin axis tilts by Roos's psi = (4
Omega/mu) sin(zeta), zeta the beat phase at the gate start (fidelity 0.99995 at zeta = 0 and 0.98886 at pi/2), so an
exact spot check at t = 0 never sees the gate such hardware plays, and the builder's per-segment phase reset is not the
fix (leakage 0.31; the per-gate reset through the leg phases is); the 2N+1-segment AM solution at the exact mode
midpoint flips its null-space direction within +-20 Hz and leaves 1e-2 leakage (M4's beat note sat 213 Hz off it), so
the surrogate places the beat note 0.35 gap above the top mode; the outer pair (0, 2) of a three-ion chain couples three
modes (dimension 8000 above the 4096 guard) and waits for GATE_LOCAL (M9a); the plan's three-ion GHZ fixture with "two
resolved modes at d_m = 12" holds only with the zigzag frozen and its 1.1e-3 residual reported; a Raman crosstalk ratio
is the intensity ratio (a 3 um waist at 3.45 um spacing gives 7 %, 2.5 um 2.2 %, 2.0 um at 2.95 um 1.3 %); and Section
9.6 row 6's 1 -> 0 dominance emerges from the coherent circuit through the oracle CNOT, with the readout asymmetry
adding to it rather than producing it.

Plan inconsistencies surfaced by M5 (ledger `anchor.m5.*`): the Section 9.5 CPT row's "s_o = 0.815" is Noek's s (s_o = 2.45)
for R_o = 0.0880 Gamma; Myerson's measured optimum (5.5, 420 us, 1.8e-4) sits above the ideal-Poisson one because his PMT dark
counts are non-Poissonian; Burrell's 0.9 % next-nearest leakage is a PSF wing neither an Airy pattern nor a Gaussian gives;
Crain's 11 us is the average detection time of the first-photon protocol, not the window; the plan's four Doppler widths do
not share one mode set; the mean-count calibration fit is degenerate at bin-time windows and needs Noek's tens of
milliseconds; and the arXiv text of Egan carries 0.46 % for the single-qubit SPAM error rather than the 0.71 %/0.22 % split.

Plan inconsistencies surfaced by M4 (recorded in the ledger as `conv.motion_phase_default` and `anchor.m4.*`): Section
4.4.3's sine beat-note convention (tone phases differing by pi, force zero at t = 0) is a convention for the closed forms only:
played literally, the carrier's frame rotation acquires the mean 2 Omega/mu and tilts the entangling axis by Roos 2008's
psi = (4 Omega_Roos/delta) sin zeta = 0.202 rad on the check_ms_closure.py fixture, leaking 3.8% into |du>, |ud> (concurrence
0.9597 against 0.9999 with equal tone phases), so the played waveforms default to equal tone phases; Zhu's printed thermal
infidelity sum beta (|alpha_j|^2 + |alpha_n|^2)/4 is one quarter of what direct integration gives for the |dd> input with
Choi's alpha (the plan's "a factor 1/4 below eps_ent for the same displacements" needs Zhu's alpha to be twice Choi's);
Hughes's printed theta_g ~ int (Omega_g^2 + alpha_dot^2)/delta dt is twice the angle of the displayed Hamiltonian
(int Omega_g^2/(2 delta) dt = pi K (Omega_g/delta)^2 for the square pulse); Baldwin's diag(1, i, i, 1) is one detuning side
(the other gives its conjugate); Choi's 190 us / 9 segments for five ions closes four modes, not five; Ballance's 0.686 is
alpha_1 at t_g/tau = 1e-3, not the coefficient 0.6875; and Ballance's thermal error (pi^2/4) eta^4 nbar (2 nbar + 1) is the
entangling-angle part only: the exact n-dependent sideband coupling also leaves residual spin-motion entanglement of the same order
in eta^4 and linear in n (the Fock-state loss is close to (pi^2/4) eta^4 n(n + 1)), so the exact thermal curve of an n = 0-calibrated
gate lies 1.05 to 1.5 times above the printed form at nbar <= 2, the excess shrinking with nbar.

Plan inconsistencies surfaced by M3 (recorded in the ledger as `anchor.m3.*`): Monroe 1995's 'theoretical 0.484' is the
semiclassical force model with isotropic emission at Delta = -30 MHz, outside that model's nu << Gamma regime, and one oblique
beam does not reproduce his measured triple (0.47, 0.30, 0.18) at any saturation, so the Section 4.2.1 acceptance test needs
the D1-D3 beam geometry; the 40Ca+ 'two-level estimate fails because of the multi-level structure' is not reproduced (the
eight-state model is within 10 % of the S-P model at the plan's parameters); Section 9.13 row 123 prints half the A_+- of
Section 4.2.3's own formula for the EIT fixture (nbar unaffected, W halved); Section 9.12's Lechner rate ratio is inverted
(the higher mode cools faster, 3.2 against the measured 3.4); Rasmusson's 0.06 after 50 optimized pulses is not reached
(0.11-0.12 with independent durations) and his 'about 0.3 quanta stranded above n = 112' is 0.1 for the thermal tail; Joshi's
twelve-operator kernel is nine in the Wigner-Eckart form (one operator per emitted polarization and recoil class, the two pi
decays summed) with identical second moments; the analytic polarization-gradient W at Joshi's operating point is 9.9e4 s^-1
against the quoted 6.6e4 (the window W < delta < omega_z holds either way).


Plan inconsistencies surfaced by M3a (recorded in the ledger as `anchor.m3a.*`): the four-level 171Yb+ steady state
reproduces (Gamma/18, 2/9) as a CEILING to 0.15-0.5 % at the best destabilizing field, not "to nine digits"; the
W(Delta -+ nu) closed form must be fed the unsaturated scattering rate (at Omega = Gamma/2 it is off by exactly 1 + s), which
Section 4.2.8 (vii) states and the code now guards; the "Morigi Fig. 3" fixture of Sections 4.2.8 and 13 (nu = 2.0068,
Omega_1 = Omega_2 = 17 MHz, Delta = 70 MHz) is not the caption of Morigi, Eschner and Keitel 2000, which reads
Omega_r = gamma, Omega_g = gamma/20, nu = gamma/10, eta = 0.145, Delta = 2.5 gamma with 99 % ground-state occupation
(reproduced); the 935 nm repump upper level (3D[3/2]1/2) has no tabulated lifetime or branching (Olmschenk 2007 cites
Bell et al. 1991 for it), so the 171Yb+ detection model folds the 0.501 % D3/2 branch back or routes it to a sink until
that constant is cited.

Plan inconsistencies surfaced by M1 and M2 are recorded in the ledger rather than absorbed (`anchor.trap.*`,
`anchor.m2.*`): the "4.077 MHz Floquet" of Section 9.13 is the preprint closed form, not the exact exponent (4.080 MHz);
the 7.806 MHz 9Be+ zigzag example uses m = 9 u; the leading sign of the cubic Coulomb term printed in Section 4.1.4
does not match its own D_222 = -1.1225; the distance-to-infidelity conversion "E^2/2" of Sections 4.3.5 and 13 is the
1 - F_C form (1 - F_K needs E^2); Section 9.12's Debye-Waller "n = 1 start gives 0.4662967" belongs to its second case
(eta^2 = 0.09, nbar = 1.7) and the thermal mean is exactly exp[-eta^2(nbar + 1/2)]; Section 9.10's target-first versus
target-last composite-pulse numbers under detuning (4.33e-3 vs 1.84e-3) are not reproduced; Harty's identity-gate
timing is ambiguous in the source and the one-delay-per-replaced-pulse reading is the one that reproduces the budget.

## Layout

| Path | What it is |
|---|---|
| `PLAN.md`, `plan_sources/` | the specification and the section files it is rendered from |
| `qutip_trap/` | the core package (Section 3.2 layout); `qutip_trap/api.py` re-exports the Appendix E surface |
| `qutip_trap/units.py` | CODATA constants and the `Hz` / `RadPerS` and `Gauss` / `Tesla` types (Sections 5.6, 13) |
| `qutip_trap/species/` | one table of cited constants per isotope, the `Species` builder, and the atomic layer (`wigner`, `zeeman`, `dipole`, `polarization`, `raman`, `quadrupole`; `atomic.py` is the facade) |
| `qutip_trap/trap/` | the trap layer of Section 4.1: `mathieu` (monodromy, Floquet function, C0), `pseudopotential` (rf drive, geometry-free maps), `surface` (gapless-plane electrodes), `crystal` (equilibrium, mass-weighted modes, Lamb-Dicke), `micromotion`, `heating`, `anharmonic`; `model.py` is the `Trap` record |
| `qutip_trap/hilbert/` | the composite space of Section 5.1: `operators` (analytic Laguerre elements, expm displacement, the Section 5.1.1 tolerance and margin fixture, Debye-Waller factors, qubit operators in the computational ordering), `space` (cached operators, marginals with the ENR index sums, state constructors), `truncation` (boundary monitor, cap growth, the tolerance-tightening test) |
| `qutip_trap/dynamics/` | the ONE builder `hamiltonian.build_hamiltonian` (H_mot + H_int + drives with exact D + Stark + anharmonic + curvature, frames, micromotion, crosstalk, frozen Debye-Waller), `frames` (virtual-Z `PhaseFrame`, the sideband decomposition), `evolve` (sesolve/mesolve with the dop853 -> vern9 ladder), `engine` (`JointExactEngine`, the Appendix E protocol), `channels` (heating, dephasing, Rayleigh collapse operators), `multilevel` (the multi-level mode of Section 4.2.8: manifold frames with the inconsistency detector, per-polarization collapse operators, recoil kernels, leak policies; M3a); the engine's mesolve and keyed-trajectory paths, per-segment channels, jump records and the hardware chain (M7); the builder's sampled trajectories, beam phases, pointing factors and drive parts (M7) |
| `qutip_trap/light/` | drives derived from beams: `raman` (two-photon Rabi frequency, Delta k, eta per mode, Stark shift, scattering budget, crosstalk ratios), `microwave` (magnetic-dipole Rabi frequency, ac Zeeman shift), `stark`, `scattering`, `comb` (Section 4.3.7 tone set, comb factor, guards), `beams`, `bloch` (the scattering-rate object of Section 13: steady state, Floquet fixed point, slow-manifold rates, dark states, pumping evolution, A_+- suppliers; M3a), `recoil` (the emission kernel of Section 4.2.8; M3a), `roles` (which beams are resonant cooling/detection light and which are far-detuned gate light; M6) |
| `qutip_trap/noise/` | the noise layer of Section 6 (M7): `spectra` (two-sided spectra with a declared white level, Drift, Mains, Collisions), `processes` (fixed-grid Gaussian and OU trajectories, mains), `sampling` (the NoiseSample keys), `model` (`NoiseModel.channels` and `sample_sequence`), `scattering` (Raman, Rayleigh, leakage and recoil operators per pulse), `levels` (register level maps with the SINK), `collisions` (Langevin events), `decoupling` (the Section 6.9 filter-function machinery and sequences), `summary` (Section 6.8 reporting) |
| `qutip_trap/prep/` | the cooling and preparation stages of Section 4.2 (M3): `closed_forms` (the level-A oracles), `rates` (per-ion, per-mode level-A rates with participation), `doppler`, `sideband` (continuous and pulsed schedules, thermometry), `eit`, `polarization_gradient` (Joshi's analytic model and the Lindblad layer), `pumping`, `sequence` (stage order and the `State` hand-off), `level_c` (the one-mode level-C solve and the level-B Fock rate equation; M3a), `recipe` (the device's PreparationRecipe, `standard_recipe`, `run_preparation`; M6) |
| `qutip_trap/control/` | native gates, the circuit IR and the compiler (`compiler`: decompositions, templates, frame propagation, verification; M6), `pulses` (Tone/Drive/Pulse, the `light_shift` kind and its couplings), `schedule` (single-qubit gates as pulses with virtual-RZ tracking; `ms`/`zz` from the table's waveforms with the wrapper and echo constructions, M4; the terminal measurement event, mid-circuit refusal, played-gate records and the per-gate beat-phase reset, M6), `shaping` (the Section 4.4.3 integrals and the AM/FM/Fourier solvers, M4), `table` (`Waveform.symmetric`, segments with callable amplitudes and detunings), `composite` (Section 4.3.5 library), `hardware` (the Section 7.10 chain and `apply_hardware_chain`; the beat-phase reference `response_phase_rad` and the `crosstalk_suppression` echoes in `schedule`, M7); `played` (the requested -> physical chain of Section 7.3 the engine applies to programmed drives, M8); the Stark compensation and its frame update (`stark_phase_rad`, `frame_after`) in `schedule`, M8 |
| `qutip_trap/calibration/` | `entangling` (the exact spot-check calibration of the entangling angle, the thermal robustness curve, the light-shift echo schedule, `frame_rotated`; M4, M8); `readout` (the detection threshold and window, M5); `surrogate` and `calibrate(surrogate=True)` (the Section 7.5 surrogate table: derived seeds for every drive, spot-checked waveforms on the resolved-mode space, detection; M6, M8); `experiments` (`full_calibration`: the simulated-experiment path `calibrate(surrogate=False)` on the dependency graph, `CalibrationScans`, `CalibrationReport` with the surrogate's error; M8); `cache` (tables per device hash and seed; M8) |
| `qutip_trap/experiments/` | the simulated experiments of Section 7.5 on the engine: `single_ion` (`rabi_scan`, `ramsey`, `ramsey_frequency`, `sideband_spectroscopy`; M2, extended), `motion` (`thermometry`, `mode_spectroscopy`, `heating_rate`), `light` (`stark_scan`, `crosstalk_scan`, `field_scan`), `entangling` (`ms_scan`, `parity_scan`, `ms_phase_scan`; M4, extended), `micromotion` (`micromotion_scan` by three of Berkeland's methods), `imaging` (`crystal_image`), `readout` (`detection_histogram`; M5), `fitting` (the plan's lineshapes, weighted fits, the observation model of shots and readout errors); M8 |
| `qutip_trap/run/` | `job` (`run`, `prepare`, the initial-mixture branches, the readout stage, `register_fidelity`; M6), `space` (the resolved/frozen/dropped mode classes and the joint space, `HilbertSpace.for_`; M6), `levels` (the Section 11.5 budget), `results` (`Result`, `Diagnostics` with the intrinsic budget, `RunState`); samples at the shot clock, the effective sample size, collisions and heralds, leakage levels (M7) |
| `qutip_trap/device/` | `model` (`Device`, `Field`, `DerivedQuantities`), `derived` (`Device.derived()`: every computed number with its ledger id, the calibration's seeds; M8) |
| `qutip_trap/io/` | `ionq` (IonQ circuit JSON, both ways), `openqasm` (the OpenQASM 2 subset importer with custom-gate inlining; M6) |
| `qutip_trap/validation/` | closed forms used as test oracles (`atomic_closed_forms`, `spin_motion_closed_forms`, `two_qubit_closed_forms`, Harty's RB model `harty_rb`); the Fang, Landsman and OU-heating forms of M7 |
| `docs/provenance/ledger.yaml` | the provenance ledger of Section 14.5 (one record per quantity) |
| `validation/scripts/` | the check and benchmark scripts of Appendix D with their committed outputs; `run_checks.py` re-runs and compares them |
| `tests/` | pytest suite (API freeze against Appendix E, units, species tables, hashing, seeds, IonQ formats, the atomic anchors of Sections 9.13/9.14/9.16, the trap and crystal anchors of Sections 9.1/9.10/9.12/9.13/9.17, the M2 spin-motion, composite-pulse, comb, native-pulse, Harty RB and experiment tests, the M3a Bloch and recoil tests, the M3 cooling and preparation tests, the M4 shaping, two-qubit gate, scheduler, light-shift and calibration tests, the M8 calibration-layer, experiment and end-to-end calibration tests) |
| `qutip_trap_app/` | the separate Flet application package of Section 14 (scaffold only until M11) |
| `.github/workflows/ci.yml` | CI: validation scripts first, then lint, type-check, tests, `flet doctor`, convergence-report artifact |

## Commands

    uv sync --all-extras                       # Python 3.13 (see .python-version), pinned in uv.lock
    uv run ruff check . && uv run ruff format --check .
    uv run mypy
    uv run pytest
    uv run python validation/scripts/run_checks.py --report    # re-run the Appendix D check scripts, write validation/report/
    uv run flet doctor                          # the gui extra (Flet) is installed

Ruff 0.16 formats fenced Python blocks inside Markdown, which would rewrite Appendix E of `PLAN.md`; `*.md` is
excluded in `pyproject.toml` so the plan stays byte-identical to its render from `plan_sources/`.

## Conventions that matter from the first line of code

Internal frequencies are angular (rad/s); the public API is in Hz and converts at the boundary with an
explicit 2π (`qutip_trap.units`). Wavelengths are vacuum wavelengths. `Transition.gamma_hz` is the upper
level's total decay rate over 2π, and I_sat uses the angular partial rate. `A_hfs` is signed. Section 13 of
the plan is the full convention table; each module restates the rows it touches in its docstring.

## Species tables

`171Yb+`, `40Ca+` and `88Sr+` build a `Species`; `43Ca+`, `133Ba+`, `137Ba+`, `9Be+` and `25Mg+` carry every
constant the plan or NIST cites and raise `IncompleteSpeciesTable` listing what is still missing. Every
number is a `Cited` value with its source and provenance tag; hyperfine-resolved quantities are never typed in
(they are derived in milestone M0a).
