"""The catalogue of displayed quantities: every number a screen shows is a :class:`Shown` naming one of them, so it carries a
provenance chip (PLAN.md Section 14.5).

A :class:`Quantity` is a plain-language label, the physics term and symbol, the unit, the level it first appears at, the
section the explain drawer opens and the ledger id its chip resolves to.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quantity:
    id: str
    label: str
    term: str
    unit: str
    level: int
    section: str
    ledger_id: str


@dataclass(frozen=True)
class Shown:
    """A displayed value with the quantity it is an instance of (and therefore its chip)."""

    quantity: str
    value: float | int | str | bool | None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.quantity not in CATALOGUE:
            raise KeyError(
                f"{self.quantity!r} is not in the catalogue: add it with its ledger id before showing it"
            )


def vector_text(v: tuple[float, float, float]) -> str:
    """A unit vector's components as a displayed value: (+0.707, +0.000, -0.707)."""
    return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"


@dataclass(frozen=True)
class Row:
    """A labelled value with its status: ``device parameter`` (typed into the model), ``derived`` (closed form or a small
    eigenproblem), ``estimate`` (closed-form error scale), ``calibrated`` (from the table), ``measured`` (from the run),
    ``stale`` (calibrated for another device)."""

    label: str
    value: Shown
    status: str = ""


# fmt: off
_ROWS: tuple[tuple[str, str, str, str, int, str, str], ...] = (
    # Level 0, the machine
    ("shots", "Repetitions of the experiment", "shots N", "", 0, "3.4", "conv.shot_blocks_per_sample"),
    ("count", "How many times this outcome came out", "count n_x", "", 0, "8.6", "conv.result_bit_order"),
    ("probability", "Fraction of repetitions with this outcome", "estimated probability p_x = n_x/N", "", 0, "8.6", "conv.result_bit_order"),
    ("error_bar", "Statistical uncertainty of the fraction", "sqrt(p (1 - p)/n_eff)", "", 0, "3.4", "conv.effective_sample_size"),
    ("target_probability", "What an ideal machine would give", "|<x|U|0>|^2 of the compiled circuit", "", 0, "7.2", "conv.compiler_frame_absorption"),
    ("bitstring", "The outcome as bits, qubit 0 rightmost", "bitstring, qubit0_lsb", "", 0, "13", "conv.result_bit_order"),
    ("spam_eps_b", "Chance a bright ion is read as dark", "eps_B", "", 0, "8.4", "conv.readout_figure_of_merit"),
    ("spam_eps_d", "Chance a dark ion is read as bright", "eps_D", "", 0, "8.4", "conv.readout_figure_of_merit"),
    ("prep_error", "Chance the ion did not start in |0>", "preparation error", "", 0, "4.2.6", "conv.preparation_stage_order"),
    ("photon_count", "Photons the detector counted from this ion", "n per ion per shot", "counts", 0, "8.2", "conv.readout_chain_exact"),
    ("detection_window", "How long the detector looked", "t_det", "s", 0, "8.3", "conv.readout_detected_line"),
    ("threshold", "Counts above which the ion is called bright", "n_c (half-integer)", "counts", 0, "8.3", "anchor.m5.myerson_optimum_and_recursion"),
    ("detected_rate", "Photons per second from a bright ion", "eps_sys R_o", "1/s", 0, "8.1", "conv.detection_efficiency_once"),
    ("background_rate", "Stray counts per second", "R_bg", "1/s", 0, "8.1", "conv.detection_efficiency_once"),
    ("species", "The atom", "ion species and mass", "", 0, "4.5", "conv.mass_ion"),
    ("mode_frequency", "How fast the ions vibrate together in this mode", "omega_m/2pi", "Hz", 0, "4.1.3", "conv.mode_index"),
    ("secular_frequency", "Trap stiffness along one axis", "secular frequency omega/2pi", "Hz", 0, "4.1.1", "conv.rf_amplitude_pseudopotential"),
    ("qubit_frequency", "The qubit's transition frequency", "omega_0/2pi", "Hz", 0, "4.5.1", "conv.frequencies"),
    ("field", "Magnetic field at the ions", "B", "G", 0, "4.5.1", "conv.curvature_naming"),
    ("gate_error_estimate", "Closed-form estimate of this gate's error", "Section 9.6 scales, summed", "", 0, "9.6", "anchor.m6.bell_state"),
    ("gate_error_tomography", "Measured error of this gate (process tomography)", "1 - F_avg", "", 0, "6.8", "conv.gate_fidelity_measure"),
    ("native_gate_set", "Operations the machine can do directly", "GPi, GPi2, MS, ZZ, virtual RZ", "", 0, "7.1", "conv.gate_parameters"),
    ("device_hash", "Fingerprint of every device parameter", "Device.hash()", "", 0, "3.3", "conv.device_hash"),
    ("seed", "The random seed the run used", "root SeedSequence", "", 0, "3.4", "conv.seed_keys"),
    ("fidelity_level", "How exactly the machine was simulated", "JOINT_EXACT or GATE_LOCAL", "", 0, "5.4", "anchor.m9a.gate_local_bell"),
    ("heating_rate", "How fast a mode heats up", "n_dot", "quanta/s", 0, "4.1.5", "conv.heating_rate_meaning"),
    ("motional_dephasing", "How long the motion keeps its phase", "tau_mot", "s", 0, "6.2", "conv.motional_dephasing_from_rf_noise"),
    ("beam_power", "Laser power in this beam", "P", "W", 0, "4.5.4", "conv.rabi_from_intensity"),
    ("beam_waist", "How tightly the beam is focused", "w_0", "m", 0, "4.5.4", "conv.rabi_from_intensity"),
    ("wavelength", "The laser's colour", "vacuum wavelength lambda", "m", 0, "4.5.2", "conv.wavelengths_vacuum"),
    ("detector_efficiency", "Fraction of photons the detector catches", "eps_sys", "", 0, "8.1", "conv.detection_efficiency_once"),
    # Level 1, the circuit
    ("native_gate", "One operation the machine plays", "GPi(phi), GPi2(phi), MS(phi0, phi1, theta), ZZ(theta)", "rad", 1, "7.6", "conv.gate_parameters"),
    ("gate_unitary", "The matrix this gate is meant to apply", "target unitary U", "", 1, "7.6", "conv.native_ms_matrix"),
    ("gate_duration", "How long the gate takes", "t_gate", "s", 1, "7.3", "conv.drive_coefficient_and_phase_continuity"),
    ("rabi_frequency", "How fast the laser flips the qubit", "Omega/2pi", "Hz", 1, "4.3.2", "conv.rabi_frequency"),
    ("entangling_angle", "How much the two qubits were entangled", "chi", "rad", 1, "4.4.1", "conv.entangling_angle"),
    ("phase_frame", "A rotation the machine bookkeeps instead of playing", "virtual-Z frame theta, phi -> phi - theta", "rad", 1, "7.6", "conv.virtual_z_propagation"),
    ("stark_frame", "Frame update that cancels the laser's own level shift", "compensated light-shift frame", "rad", 1, "7.5", "conv.stark_compensation"),
    ("population", "Probability of each computational state", "diag(rho)", "", 1, "5.7", "conv.computational_ordering"),
    ("bloch_vector", "The qubit's state as an arrow", "(<X>, <Y>, <Z>)", "", 1, "4.3.4", "conv.computational_ordering"),
    ("pauli_expectation", "Average of a Pauli measurement", "<P>", "", 1, "5.7", "conv.computational_ordering"),
    ("purity", "How much of the state is a single pure state", "Tr rho^2", "", 1, "5.4", "conv.computational_ordering"),
    ("state_fidelity", "Overlap of the register with its target after this gate", "<target| rho |target>", "", 1, "9.6", "conv.gate_fidelity_measure"),
    ("compile_residual", "How closely the native gates reproduce the requested one", "max |U_block - e^{i a} U_target|", "", 1, "7.2", "conv.compiler_single_qubit_decomposition"),
    ("channel_infidelity", "Average error of the gate as a channel", "1 - F_avg", "", 1, "6.8", "conv.gate_fidelity_measure"),
    ("depolarizing_rate", "The gate's error as a depolarizing rate", "eps", "", 1, "6.8", "conv.depolarizing_normalization"),
    ("pauli_twirl", "The gate's error as Pauli flip probabilities", "p_P", "", 1, "6.8", "conv.depolarizing_normalization"),
    # Level 2, the schedule
    ("tone_detuning", "How far the laser tone is from the qubit frequency", "mu (from the carrier)", "Hz", 2, "4.3.1", "conv.detuning_symbols"),
    ("tone_envelope", "The tone's strength over time", "Omega(t)", "Hz", 2, "7.4", "conv.rabi_frequency"),
    ("tone_phase", "The tone's phase", "phi_tone(t)", "rad", 2, "5.2", "conv.spin_motion_phases"),
    ("sideband_detuning", "How far the tone is from talking to this mode", "delta_{i,m} = mu_i - omega_m for a blue tone, mu_i + omega_m for a red one: the distance to its own sideband", "Hz", 2, "4.4.1", "conv.detuning_symbols"),
    ("waveform_segment", "One piece of the gate's pulse shape", "segment (duration, Omega, phi, mu)", "", 2, "7.4", "conv.drive_coefficient_and_phase_continuity"),
    ("closure_alpha", "Leftover motion the gate did not undo", "alpha_m(tau)", "", 2, "4.4.3", "conv.ms_closure"),
    ("chi_m", "This mode's share of the entangling angle", "chi_m", "rad", 2, "4.4.3", "conv.entangling_sign"),
    ("crosstalk", "Light meant for one ion that hits its neighbour", "eps_ij = Omega_j/Omega_i", "", 2, "6.6", "conv.crosstalk_ratio"),
    ("beam_direction", "Which way the light travels", "k_hat; Delta k = k_1 - k_2", "", 2, "4.3.2", "conv.effective_wavevector"),
    ("lamb_dicke", "How strongly light couples spin to motion", "eta_{i,m}", "", 2, "4.1.7", "conv.lamb_dicke"),
    ("stark_shift", "How the laser itself shifts the qubit frequency", "light shift", "Hz", 2, "4.5.4", "conv.stark_compensation"),
    ("idle", "A gap where nothing is played and noise acts", "idle interval", "s", 2, "6.7", "conv.noise_routing"),
    ("measurement_event", "When the detection light is on", "measure event", "s", 2, "7.2", "conv.terminal_measurement_event"),
    ("beat_phase", "Where the two-tone beat starts", "beat-note phase at the gate start", "rad", 2, "7.10", "conv.beat_phase_at_gate_start"),
    # Level 3, the dynamics
    ("P1", "Probability the qubit is in |1>", "P1(t) = <|1><1|>", "", 3, "5.7", "conv.computational_ordering"),
    ("coherence", "How much superposition the qubit holds", "|rho_01|(t)", "", 3, "5.7", "conv.computational_ordering"),
    ("mode_nbar", "Average number of vibration quanta in this mode", "<n_m>(t)", "quanta", 3, "5.1", "conv.zero_point_energy"),
    ("alpha_m", "Where the mode is in phase space", "<a_m>(t)", "", 3, "4.4.1", "conv.ms_closure"),
    ("branch_alpha", "Where one spin branch takes the mode in phase space", "alpha_im(t)", "", 3, "4.4.1", "conv.ms_closure"),
    ("branch_closure", "How far a spin branch's loop failed to return", "|alpha_im(tau) - alpha_im(0)|", "", 3, "4.4.3", "conv.ms_closure"),
    ("mean_alpha_excursion", "How far the spin-averaged motion strayed from rest", "max |<a_m>(t)|", "", 3, "4.4.1", "conv.ms_closure"),
    ("chi_closed_form", "Entangling angle the loops sweep, before the exact check", "2 Im int conj(alpha_a) d alpha_b", "rad", 3, "4.4.3", "conv.entangling_angle"),
    ("spot_check_amplitude_ratio", "How much the exact check rescaled the pulse", "Omega_played / Omega_closed", "", 4, "7.8", "conv.entangling_angle"),
    ("surrogate_error", "What the closed form missed of the exact angle", "chi_exact / chi_closed - 1", "", 4, "7.8", "conv.entangling_angle"),
    ("fock_population", "Probability of each vibration number", "P(n_m)", "", 3, "5.3", "conv.fock_sum_branches"),
    ("jump", "A random event the environment caused", "quantum jump (channel, time)", "s", 3, "6.1", "conv.collapse_op_rate_units"),
    ("noise_sample_value", "A slowly drifting parameter's value for this repetition", "quasi-static sample", "", 3, "6.1", "conv.noise_provenance"),
    ("debye_waller", "How thermal motion weakens the laser coupling", "Debye-Waller factor", "", 3, "5.2", "conv.frozen_spectator_shot_sample"),
    ("concurrence", "How entangled the two qubits are", "concurrence C(rho)", "", 3, "4.4.1", "conv.entangling_angle"),
    # Level 3, the process matrix and the Hamiltonian record
    ("entanglement_infidelity", "How far the gate's channel is from the ideal one", "1 - F_e (entanglement infidelity)", "", 3, "6.8", "conv.gate_fidelity_measure"),
    ("cp_tp_residual", "How far the measured map had to be moved to be physical", "residual against CP and TP after Dykstra projection", "", 3, "5.4", "anchor.m9a.tomography_projection"),
    # Level 4, the Hamiltonian page
    ("h_mode_term", "Energy of one vibration quantum in the equation", "hbar omega_m a_m^dag a_m (omega_m/2pi shown)", "Hz", 4, "5.7", "conv.zero_point_energy"),
    ("h_qubit_offset", "How far the qubit's true frequency sits from the frame", "Delta_i in H_int = (hbar Delta_i/2) sigma_z", "Hz", 4, "5.7", "conv.frequencies"),
    ("h_drive_omega", "Strength of the drive term", "Omega_peak/2pi of (hbar Omega/2) e^{-i(mu t - phi)} sigma_+ (x) prod D", "Hz", 4, "5.2", "conv.rabi_frequency"),
    ("h_matrix_element", "How strongly the light connects two vibration numbers", "Omega_{n',n}/Omega = |<n'|D(i eta)|n>|", "", 4, "4.3.1", "anchor.m2.rabi_matrix_elements"),
    ("h_debye_waller_frozen", "Weakening of the drive by a frozen spectator mode", "e^{-eta^2/2} L_n(eta^2) at the sample's n", "", 4, "5.2", "anchor.m2.debye_waller_identity"),
    ("h_carrier_factor", "Weakening of the drive by micromotion", "J_0(beta_mm)", "", 4, "4.3.6", "anchor.m2.micromotion_carrier_factor"),
    ("h_crosstalk_weight", "Share of the addressed ion's light this ion sees", "eps_ij (1 for the addressed ion)", "", 4, "6.6", "conv.crosstalk_ratio"),
    ("h_operator_nnz", "Non-zero entries of the drive operator", "nnz of sigma_+ (x) prod_m D_m", "", 4, "5.1.1", "anchor.m9b.cost_model_constants"),
    ("collapse_rate", "How often this random event happens", "rate under the square root of L_k = sqrt(rate) A_k", "1/s", 4, "5.7", "conv.collapse_op_rate_units"),
    ("omega_max", "Fastest oscillation the solver must follow", "omega_max/2pi (the highest mode or beat note in the frame)", "Hz", 4, "5.2", "conv.solver_integrators"),
    ("hamiltonian_fingerprint", "Digest of everything that determined the equation", "fingerprint of H(t)", "", 4, "11.3", "conv.hamiltonian_fingerprint_carries_the_device"),
    ("kernel", "How the drive operator is applied", "assembled (CSR) or factorized (matrix-free, mode by mode)", "", 4, "11.3", "anchor.m9b.factorized_kernel_exactness"),
    ("frame", "The picture the equation is written in", "Schroedinger picture for the motion, qubit frame at omega_0", "", 4, "5.2", "anchor.m2.picture_equivalence_and_step_density"),
    ("approximation", "A simplification the builder made and recorded", "approximation note", "", 4, "5.7", "conv.validity_assertions"),
    # Level 4, the species page
    ("level_energy", "Energy of a level above the ground state", "E/h", "Hz", 4, "4.5.1", "conv.frequencies"),
    ("lifetime", "How long an excited level lives", "tau (Gamma = 1/tau)", "s", 4, "4.5.2", "conv.linewidth_gamma"),
    ("linewidth", "Natural width of the transition", "Gamma/2pi of the upper level", "Hz", 4, "4.5.2", "conv.linewidth_gamma"),
    ("branching", "Share of decays that take this path", "fine-structure branching ratio", "", 4, "4.5.2", "conv.branching_deficit"),
    ("hyperfine_a", "Magnetic hyperfine constant of the level", "A_hfs (signed)", "Hz", 4, "4.5.1", "conv.sign_A_hfs_and_g_factors"),
    ("hyperfine_b", "Electric-quadrupole hyperfine constant", "B_hfs", "Hz", 4, "4.5.1", "conv.hyperfine_factors"),
    ("g_j", "Landé factor of the level", "g_J", "", 4, "4.5.1", "conv.sign_A_hfs_and_g_factors"),
    ("nuclear_spin", "Spin of the nucleus", "I", "", 4, "4.5.1", "conv.nuclear_g_sign"),
    ("nuclear_moment", "Magnetic moment of the nucleus", "mu_I", "mu_N", 4, "4.5.1", "conv.mu_i_shielding_convention"),
    ("mass", "Mass of the ion", "m (atomic mass units)", "u", 4, "4.5", "conv.mass_ion"),
    ("sublevel_energy", "Energy of a magnetic sublevel within its level", "E(F, m_F) - E_level at the field", "Hz", 4, "4.5.1", "conv.curvature_naming"),
    ("field_slope", "How fast the level moves with the field", "dE/dB", "Hz/G", 4, "4.5.1", "conv.curvature_naming"),
    ("field_curvature", "How the level bends with the field", "d^2E/dB^2", "Hz/G^2", 4, "4.5.1", "conv.curvature_naming"),
    ("clock_field", "Field at which the qubit stops moving to first order", "B_0 (clock point)", "G", 4, "4.5.1", "conv.curvature_naming"),
    ("saturation_intensity", "Light level that half-saturates the transition", "I_sat = pi h c Gamma/(3 lambda^3)", "W/m^2", 4, "4.5.2", "conv.saturation_intensity"),
    # Level 4, the trap page
    ("mathieu_a", "Static part of the trap's equation of motion", "a (Mathieu)", "", 4, "4.1.1", "conv.mathieu_sign"),
    ("mathieu_q", "Oscillating part of the trap's equation of motion", "q (Mathieu)", "", 4, "4.1.1", "conv.mathieu_sign"),
    ("mathieu_beta", "Characteristic exponent: how fast the ion swings", "beta (nu = beta Omega_rf/2)", "", 4, "4.1.1", "anchor.trap.stability_edge"),
    ("micromotion_c0", "Correction of the coupling by intrinsic micromotion", "C0 = 1 + 3q^2/16 + O(q^4)", "", 4, "4.1.1", "conv.micromotion_correction"),
    ("rf_frequency", "How fast the trap voltage oscillates", "Omega_rf/2pi", "Hz", 4, "4.1.1", "conv.mathieu_sign"),
    ("rf_voltage", "Peak of the oscillating trap voltage", "V_rf (peak of a cos drive)", "V", 4, "4.1.1", "conv.rf_amplitude_pseudopotential"),
    ("stray_field", "Uncompensated electric field at the ion", "E_dc", "V/m", 4, "4.1.1", "anchor.trap.berkeland_excess_micromotion"),
    ("micromotion_amplitude", "How far the drive drags the ion each cycle", "u_1 = (1/2) q u_0 (signed, peak)", "m", 4, "4.1.1", "conv.micromotion_amplitude_convention"),
    ("ion_displacement", "How far the stray field pushes the ion", "u_0 = Q E/(m omega^2)", "m", 4, "4.1.1", "anchor.trap.berkeland_excess_micromotion"),
    ("stability_edge", "Largest q that still traps at a = 0", "q_edge (0.908 by the source)", "", 4, "4.1.1", "anchor.trap.stability_edge"),
    ("axis_angle", "Rotation of the radial trap axes about the chain", "axis angle", "rad", 4, "4.1.3", "conv.mode_index"),
    # Level 4, the crystal page
    ("eigenvector_component", "How much this ion moves in the mode", "c_{i,m} (mass-weighted, unit norm, last component positive)", "", 4, "4.1.3", "conv.mode_eigenvector_gauge"),
    ("uniform_field_weight", "How strongly a uniform field noise drives this mode", "(sum_i c_i/sqrt(m_i))^2", "1/kg", 4, "4.1.5", "anchor.trap.kielpinski_three_ion"),
    ("zigzag_ratio", "Radial against axial stiffness of this crystal", "omega_r,min/omega_z", "", 4, "4.1.2", "anchor.trap.zigzag_threshold"),
    ("zigzag_critical", "Below this ratio the chain buckles into a zigzag", "(omega_r/omega_z)_crit = sqrt((mu_N - 1)/2)", "", 4, "4.1.2", "anchor.trap.zigzag_threshold"),
    ("length_scale", "Natural length of the Coulomb chain", "l = (e^2/(4 pi eps0 m omega_z^2))^{1/3}", "m", 4, "4.1.2", "anchor.trap.james_spectrum"),
    ("ion_spacing", "Distance between neighbouring ions", "min |x_i - x_j|", "m", 4, "4.1.2", "anchor.trap.james_spectrum"),
    ("delta_k", "Momentum kick of the two-beam drive", "|Delta k| = |k_1 - k_2|", "rad/m", 4, "4.3.2", "conv.effective_wavevector"),
    # Level 4, the light page
    ("intensity", "Light intensity at the ion", "I(r) = (2P/(pi w0^2)) e^{-2 r^2/w0^2}", "W/m^2", 4, "4.5.2", "conv.rabi_from_intensity"),
    ("saturation_parameter", "Intensity in units of the saturating one", "s = I/I_sat", "", 4, "8.1", "conv.saturation_ceiling"),
    ("pi_time", "How long a full flip takes", "t_pi = pi/Omega", "s", 4, "4.3.1", "conv.rabi_frequency"),
    ("scattering_error", "Chance a pi pulse scatters a photon", "P_total per pi pulse", "", 4, "4.3.2", "conv.scattering_channels"),
    ("rayleigh_dephasing", "Dephasing from unequal elastic scattering of the two qubit states", "Gamma_el", "1/s", 4, "4.5.5", "conv.rayleigh_dissipator"),
    ("residual_excited", "Population left in the far-detuned upper level", "sum_e |Omega_e|^2/(4 Delta_e^2)", "", 4, "4.5.4", "conv.two_photon_rabi"),
    ("beam_angle_to_field", "Angle between the beam and the magnetic field", "angle(k, B)", "deg", 4, "4.5.3", "conv.polarization_components"),
    # Level 4, the noise page
    ("noise_density_e", "Electric-field noise density", "S_E(omega), two-sided", "(V/m)^2/(rad/s)", 4, "4.1.5", "conv.electric_field_noise"),
    ("noise_density_b", "Magnetic-field noise density", "S_B(omega), two-sided", "T^2/(rad/s)", 4, "6.3", "conv.qubit_dephasing_from_field_noise"),
    ("noise_density", "Noise density of a control parameter", "S(omega), two-sided", "", 4, "6.1", "conv.noise_routing"),
    ("correlation_length", "Distance over which field noise is the same at two ions", "l_c", "m", 4, "4.1.5", "conv.electric_field_noise_adapter"),
    ("qubit_dephasing", "How fast a qubit loses its phase to white field noise", "gamma_phi = 2 pi^2 (dnu/dB)^2 S_B", "1/s", 4, "6.3", "conv.qubit_dephasing_operator"),
    ("drift_rms", "Size of a slow parameter drift", "rms of the quasi-static draw", "", 4, "6.1", "conv.noise_routing"),
    ("drift_tau", "How long a drift stays correlated", "tau_c", "s", 4, "6.1", "conv.trajectory_grid_tau_c"),
    ("servo_bandwidth", "How fast a servo removes a drift", "servo bandwidth", "Hz", 4, "7.5", "conv.servo_residual"),
    ("mains_amplitude", "Field wobble at a mains harmonic", "B_k", "T", 4, "6.3", "conv.noise_routing"),
    ("collision_rate", "How often a background-gas atom hits an ion", "Gamma_L = n_bg k_L", "1/s", 4, "6.7", "conv.collision_outcomes"),
    ("pressure", "Background gas pressure", "p", "Pa", 4, "6.7", "conv.collision_outcomes"),
    # Level 4, the cooling page
    ("cooling_rate", "How fast the mode cools", "W_m = A_- - A_+ (times eta^2)", "1/s", 4, "4.2.2", "anchor.m3a.phonon_rate_equation"),
    ("heating_coefficient", "Photon-recoil heating coefficient of the stage", "A_+", "1/s", 4, "4.2.2", "anchor.m3.doppler_rate_framework"),
    ("cooling_coefficient", "Cooling coefficient of the stage", "A_-", "1/s", 4, "4.2.2", "anchor.m3.doppler_rate_framework"),
    ("doppler_limit", "Where the stage's cooling stops", "nbar = A_+/(A_- - A_+)", "quanta", 4, "4.2.1", "anchor.m3a.doppler_limit"),
    ("stage_duration", "How long a preparation stage lasts", "t_stage", "s", 4, "4.2.6", "conv.preparation_stage_order"),
    ("pump_photons", "Photons scattered while pumping the ion into |0>", "N_photons", "", 4, "4.2.6", "anchor.m3.optical_pumping_recoil"),
    ("recoil_heating", "Vibration quanta added by the pumping photons", "Delta n", "quanta", 4, "4.2.8", "anchor.m3.optical_pumping_recoil"),
    ("participation", "Share of the mode's motion the cooled ions carry", "W_k = sum_j c_{j,k}^2", "", 4, "4.1.7", "conv.level_a_rates_participation"),
    # Level 4, the readout page
    ("scatter_rate_bright", "Photons per second a bright ion scatters", "R_o", "1/s", 4, "8.1", "conv.saturation_ceiling"),
    ("pump_rate_dark", "How fast a bright ion is pumped dark by the same light", "R_d", "1/s", 4, "8.1", "anchor.m3a.yb171_leakage_prefactors"),
    ("pump_rate_bright", "How fast a dark ion is pumped bright", "R_b", "1/s", 4, "8.1", "anchor.m3a.yb171_leakage_prefactors"),
    ("saturation_ceiling", "Largest excited fraction the level manifold allows", "n_e/(n_e + n_g)", "", 4, "8.1", "conv.saturation_ceiling"),
    ("readout_budget_line", "One named cause of readout error", "eps_B, eps_D of a channel", "", 4, "8.4", "conv.readout_figure_of_merit"),
    ("readout_polarity", "Which qubit state is the bright one", "bright_is_0 or bright_is_1", "", 4, "8", "conv.readout_polarity_scheme"),
    # Level 4, the pulse-solver solutions
    ("residual_error", "Entanglement the solved pulse leaves with the motion", "eps_ent = sum_{i,m} |alpha_im|^2 (2 nbar_m + 1)", "", 4, "4.4.7", "anchor.m4.residual_displacement_conversions"),
    ("peak_rabi", "Largest Rabi frequency the pulse needs", "Omega_peak/2pi", "Hz", 4, "4.4.3", "conv.rabi_frequency"),
    ("power_integral", "Total drive power the pulse spends", "sum Omega^2 dt", "rad^2/s", 4, "4.4.3", "anchor.m4.fm_and_fourier_robustness"),
    ("solver_method", "Which pulse-shaping method closed the loops", "am_segmented / symmetric / fm", "", 4, "4.4.3", "anchor.m4.solvers_verified_by_exact_integration"),
    # the channel replay
    ("derivation_residual", "How much this derived run can be off by", "channel-derivation residual: sum |alpha_m|^2 (2 nbar_m + 1) + frozen + dropped + projection + covariance", "", 0, "9.8", "conv.gate_local_residual_bound_pre_step"),
    # the numerics panel
    ("dimension", "Size of the simulated state space", "joint dimension prod dims", "", 3, "5.1", "conv.space_declaration_before_allocation"),
    ("truncation_cap", "Highest vibration number the simulation keeps", "d_m = n_max + 1", "", 3, "5.5", "conv.mode_dimension_max"),
    ("boundary_population", "Population that reached the cut-off", "top two Fock levels' population", "", 3, "5.5", "conv.boundary_threshold_is_branch_scaled_on_both_sides"),
    ("margin", "Spare vibration levels above what was used", "cap margin (levels)", "levels", 3, "5.1.1", "conv.oracle_margin_rule"),
    ("mode_class", "How each mode was treated", "resolved / frozen / dropped / enr", "", 3, "5.2", "conv.mode_classes_and_tolerances"),
    ("integrator", "The differential-equation solver used", "dop853 -> vern9 ladder", "", 3, "5.3", "conv.solver_integrators"),
    ("tolerance", "The solver's error tolerances", "(atol, rtol)", "", 3, "5.3", "conv.integrator_tolerances"),
    ("trajectories", "Random histories averaged", "trajectories", "", 3, "3.4", "conv.trajectory_count_two_phases"),
    ("branches", "Initial states summed exactly", "Fock-sum branches", "", 3, "5.3", "conv.fock_sum_branches"),
    ("samples", "Draws of the slowly drifting parameters", "quasi-static samples", "", 3, "3.4", "conv.shot_blocks_per_sample"),
    ("wall_time", "How long the computer took", "wall time", "s", 3, "11.1", "anchor.m9b.cost_model_constants"),
    ("norm_deficit", "How much total probability the solver lost or gained", "1 - Tr rho (the integrator runs unnormalized, Section 5.3)", "", 3, "5.3", "conv.solver_integrators"),
    ("target_distance", "Distance of the histogram from the ideal one", "total variation (1/2) sum_x |p_x - t_x|", "", 0, "7.2", "conv.compiler_frame_absorption"),
    ("largest_deviation", "Largest gap to the ideal, in error bars", "max_x |p_x - t_x| / sigma_x", "", 0, "3.4", "conv.effective_sample_size"),
    # the published-experiment presets: each published number and its simulated twin
    ("harty_epg_published", "Error per gate, published", "EPG (Harty 2014)", "", 1, "9.2", "published.harty_2014_epg"),
    ("harty_epg_simulated", "Error per gate, this model", "EPG from the propagator product of Section 4.3.3", "", 1, "9.2", "anchor.m2.harty_randomized_benchmarking"),
    ("james_u_published", "Ion position, published", "u_i (James 1998)", "", 4, "9.1", "published.james_1998_positions"),
    ("james_u_simulated", "Ion position, this solver", "u_i from the equilibrium solver", "", 4, "4.1.2", "anchor.trap.james_spectrum"),
    ("james_mu_published", "Axial mode eigenvalue, published", "mu_p (James 1998)", "", 4, "9.1", "published.james_1998_axial_modes"),
    ("james_mu_simulated", "Axial mode eigenvalue, this solver", "mu_p = (nu_p/nu_z)^2 from the Hessian", "", 4, "4.1.3", "anchor.trap.james_spectrum"),
    ("monroe_nbar_published", "Doppler-cooled occupation, published", "nbar (Monroe 1995)", "", 4, "9.3", "published.monroe_1995_doppler"),
    ("monroe_nbar_simulated", "Doppler-cooled occupation, this model", "nbar from the force model or the A_+- rates", "", 4, "4.2.1", "anchor.m3.monroe_doppler_triple"),
    ("roos_eta_published", "Lamb-Dicke parameter, published", "eta (Roos 2000)", "", 4, "9.3", "published.roos_2000_lamb_dicke"),
    ("roos_eta_simulated", "Lamb-Dicke parameter, this model", "eta = k x0", "", 4, "4.1.7", "anchor.ca40.lamb_dicke_729"),
    ("kirchmair_contrast_published", "Parity contrast, published", "C at nbar = 20 (Kirchmair 2009)", "", 3, "9.4", "published.kirchmair_2009_thermal_ms"),
    ("kirchmair_contrast_simulated", "Parity contrast, closed form", "|sum_n P_n exp(-i 4 chi eta^2 n)|", "", 3, "4.4.7", "anchor.m4.kirchmair_ca40"),
    ("kirchmair_fidelity_published", "Bell-state fidelity, published", "F (Kirchmair 2009)", "", 3, "9.4", "published.kirchmair_2009_thermal_ms"),
    ("kirchmair_fidelity_simulated", "Bell-state fidelity, first-principles terms only", "1 - eps_ent - eps_DW at nbar = 0", "", 3, "4.4.7", "anchor.m4.kirchmair_ca40"),
    ("myerson_eps_published", "Readout error, published", "eps (Myerson 2008)", "", 4, "9.5", "published.myerson_2008_readout"),
    ("myerson_eps_simulated", "Readout error, exact chain", "(eps_B + eps_D)/2", "", 4, "8.3", "anchor.m5.myerson_optimum_and_recursion"),
    ("myerson_window_published", "Detection window, published", "t_b (Myerson 2008)", "s", 4, "9.5", "published.myerson_2008_readout"),
    ("myerson_window_simulated", "Detection window at the exact chain's optimum", "t_b", "s", 4, "8.3", "anchor.m5.myerson_optimum_and_recursion"),
    ("crain_eps_published", "Readout error, published", "1 - F (Crain 2019)", "", 4, "9.5", "published.crain_2019_snspd"),
    ("crain_eps_simulated", "Readout error, exact chain", "(eps_B + eps_D)/2 at the best window", "", 4, "8.3", "anchor.m5.crain_corrections_and_operating_point"),
    ("crain_window_published", "Detection time, published", "average detection time (Crain 2019)", "s", 4, "9.5", "published.crain_2019_snspd"),
    ("crain_window_simulated", "Detection window at the optimum", "t_b", "s", 4, "8.3", "anchor.m5.crain_corrections_and_operating_point"),
    ("bell_check_probability_published", "Outcome probability, the reference run", "P_x (the Section 9.6 reference run, 4000 shots)", "", 0, "9.6", "anchor.m6.bell_state"),
    ("bell_check_probability_simulated", "Outcome probability, this run", "P_x = n_x/N", "", 0, "8.6", "conv.result_bit_order"),
    ("bell_check_infidelity_published", "Register infidelity, the reference run", "1 - F (the Section 9.6 reference run)", "", 0, "9.6", "anchor.m6.bell_state"),
    ("bell_check_infidelity_simulated", "Register infidelity, this run", "1 - <ideal|rho|ideal>", "", 0, "7.2", "conv.gate_fidelity_measure"),
    ("ghz_check_probability_published", "Outcome probability, the reference run", "P_x (the Section 9.6 reference run, 400 shots)", "", 0, "9.6", "anchor.m10.ghz_fidelity"),
    ("ghz_check_probability_simulated", "Outcome probability, this run", "P_x = n_x/N", "", 0, "8.6", "conv.result_bit_order"),
)
# fmt: on

CATALOGUE: dict[str, Quantity] = {row[0]: Quantity(*row) for row in _ROWS}


def ledger_ids() -> frozenset[str]:
    """Every ledger id the catalogue resolves to."""
    return frozenset(q.ledger_id for q in CATALOGUE.values())
