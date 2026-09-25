"""The learning layer (DESIGN.md Section 3): concepts on the zoom ladder with explanations at three depths, each ending in a
retrieval prompt; the prior-knowledge plan (assist a newcomer, collapse for a physicist, assist when unknown); the
declining-share spacing rule; the mastery log, which keeps session accuracy apart from delayed unaided accuracy (only the
latter is evidence of learning); and the scorers. Every concept names its ledger records and section, so an explanation
opens the chips of the number it explains.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

PriorKnowledge = Literal["newcomer", "circuits", "physicist", "unknown"]
"""Three self-descriptions the app asks for once, plus 'unknown' (assist when in doubt)."""

Depth = Literal["sentence", "picture", "equation"]
DEPTHS: tuple[Depth, ...] = ("sentence", "picture", "equation")
PromptKind = Literal[
    "predict_histogram", "choose", "predict_direction", "predict_closure", "free_text", "locate"
]


@dataclass(frozen=True)
class Explanation:
    sentence: str
    """One sentence, plain words, no symbols."""
    picture: str
    """What to look at on the screen and what it shows."""
    equation: str
    """The equation as the ledger states it, symbols named."""

    def at(self, depth: Depth) -> str:
        return {"sentence": self.sentence, "picture": self.picture, "equation": self.equation}[depth]


@dataclass(frozen=True)
class Prompt:
    """A retrieval prompt, asked before the view reveals the answer and checked against the record."""

    id: str
    kind: PromptKind
    question: str
    options: tuple[str, ...] = ()
    answer: str | None = None
    """For ``choose``: the correct option. For ``locate``: the route the learner must reach."""
    rubric: str = ""
    """For free text: what a complete answer contains."""
    where: str = ""
    """For ``predict_*`` and ``locate``: where on the screen to look, in plain words."""


@dataclass(frozen=True)
class Concept:
    id: str
    level: int
    title: str
    """Plain-language name."""
    term: str
    """The physics term."""
    ledger_ids: tuple[str, ...]
    section: str
    explain: Explanation
    prompts: tuple[Prompt, ...]
    can_do: str
    """What the learner will be able to do (the delayed, unaided task)."""


def _c(
    id: str,
    level: int,
    title: str,
    term: str,
    ledger_ids: tuple[str, ...],
    section: str,
    sentence: str,
    picture: str,
    equation: str,
    can_do: str,
    *prompts: Prompt,
) -> Concept:
    return Concept(
        id, level, title, term, ledger_ids, section, Explanation(sentence, picture, equation), prompts, can_do
    )


# fmt: off
CONCEPTS: dict[str, Concept] = {c.id: c for c in (
    _c("shot", 0, "One repetition", "shot", ("conv.shot_blocks_per_sample",), "3.4",
       "A shot is one complete run of the machine: cool the ions, prepare them, play the circuit, read them out once.",
       "Each bar of the histogram is a pile of shots; click a bar to see the shots stacked inside it.",
       "N shots split into S samples of slow drifts; shot k belongs to sample floor(k S/N) (conv.shot_blocks_per_sample).",
       "say what one shot is and why a result needs many of them",
       Prompt("shot.q1", "choose", "You ask for 200 shots. How many times does the machine read the ions?",
              ("once", "200 times", "as many as there are qubits"),
              "200 times")),
    _c("histogram", 0, "Reading the histogram", "measured probability distribution", ("conv.result_bit_order", "conv.effective_sample_size"), "8.6",
       "The histogram counts how often each outcome came out; a probability is a count divided by the number of shots.",
       "Bar height is the fraction of shots; the thin line on top is the statistical uncertainty of that fraction.",
       "p_x = n_x/N with error sqrt(p_x (1 - p_x)/n_eff), n_eff the effective sample size (conv.effective_sample_size).",
       "read a histogram and judge whether two bars differ by more than statistics",
       Prompt("histogram.q1", "predict_histogram", "Before running the Bell circuit: sketch the histogram you expect.",
              where="the bars of the Results card once the run is in"),
       Prompt("histogram.q2", "free_text", "The 01 and 10 bars are not exactly zero. Name one physical reason.",
              rubric="readout error (a bright ion read dark or the reverse), imperfect entangling angle, residual motion, crosstalk; the simulator's error budget lists them with numbers")),
    _c("bitstring_order", 0, "Which bit is which qubit", "bit order", ("conv.result_bit_order",), "13",
       "In a key like 01, the rightmost bit is qubit 0.",
       "Hover a bar: the key is spelled with qubit 0 rightmost; the IonQ decimal key is the same number.",
       "key = sum_j b_j 2^j read right to left (qubit0_lsb); '101' on three qubits is qubit0 = 1, qubit1 = 0, qubit2 = 1.",
       "translate a bitstring key into per-qubit outcomes without looking it up",
       Prompt("bitstring_order.q1", "choose", "On two qubits, the key '10' means",
              ("qubit 0 = 1, qubit 1 = 0", "qubit 0 = 0, qubit 1 = 1"),
              "qubit 0 = 0, qubit 1 = 1")),
    _c("target_vs_simulated", 0, "Ideal beside real", "target distribution against the simulated one", ("conv.compiler_frame_absorption", "anchor.m6.bell_state"), "7.2",
       "The faint bars are what a perfect machine would give; the solid bars are what this simulated machine gave.",
       "The two sets of bars sit side by side and never replace each other; the gap between them is the physics of errors.",
       "target: |<x|U|0>|^2 from the compiled unitary; simulated: shots read from the evolved, measured register.",
       "tell an ideal prediction from a simulated result and name what fills the gap",
       Prompt("target_vs_simulated.q1", "choose", "The solid 11 bar is 0.495 and the faint one 0.5. Is that a problem?",
              ("yes, the machine failed", "not if 0.005 is within the error bar", "the faint bar is wrong"),
              "not if 0.005 is within the error bar")),
    _c("spam", 0, "Reading errors", "state preparation and measurement error", ("conv.readout_figure_of_merit", "conv.readout_chain_exact"), "8.4",
       "Sometimes a bright ion gives too few photons and is called dark, or a dark ion scatters a photon and is called bright.",
       "The device card lists the two chances separately; clicking one opens the photon-count histograms they come from.",
       "eps_B and eps_D reported separately; eps = (eps_B + eps_D)/2 only at the end (conv.readout_figure_of_merit).",
       "explain why a readout error has two numbers and where each comes from",
       Prompt("spam.q1", "choose", "Which error puts counts into the 01 bar of a Bell state?",
              ("a bright ion read as dark", "a dark ion read as bright", "either one"),
              "either one")),
    _c("bloch_vector", 1, "A qubit as an arrow", "Bloch vector", ("conv.computational_ordering",), "4.3.4",
       "A qubit's state is an arrow: pointing up is 0, down is 1, sideways is an equal superposition.",
       "After each gate the arrow of every ion is drawn from the recorded state; a shrunken arrow means mixture or entanglement.",
       "r = (<X>, <Y>, <Z>) = (Tr rho X, Tr rho Y, Tr rho Z); |r| = 1 for a pure single-qubit state.",
       "predict the arrow after a single-qubit gate before seeing it",
       Prompt("bloch_vector.q1", "predict_direction", "Where does ion 0's arrow point after the first GPi2 pulse?",
              where="ion 0's Bloch arrow in the Register card, with the first gpi2 gate selected on Level 1")),
    _c("native_gate", 1, "What the machine can actually do", "native gate set", ("conv.gate_parameters", "conv.native_ms_matrix"), "7.6",
       "The machine has a few native moves: quarter turns and half turns of one arrow, and one two-ion entangling move; everything else is built from them.",
       "The timeline shows only native gates; the H and CNOT you wrote were compiled into them and verified.",
       "GPi2(phi) = R_phi(pi/2), GPi(phi) = R_phi(pi), MS(phi0, phi1, theta), ZZ(theta); RZ is a frame update (conv.gate_parameters).",
       "name the native gates and say which ones move the arrows and which entangle",
       Prompt("native_gate.q1", "choose", "How many entangling pulses does H then CNOT need on this machine?", ("0", "1", "2"), "1")),
    _c("compile", 1, "From your circuit to native moves", "compilation", ("conv.compiler_single_qubit_decomposition", "conv.compiler_frame_absorption"), "7.2",
       "The compiler rewrites your gates as native ones and checks, with a matrix multiplication, that the result equals what you asked for.",
       "The compile report shows the residual of every block: how far the native sequence is from the requested gate.",
       "max |U_block - e^{i alpha} U_target| < 1e-9 per block (conv.compiler_single_qubit_decomposition).",
       "read a compile report and say what a nonzero residual would mean",
       Prompt("compile.q1", "choose", "A block residual of 1e-15 means",
              ("the gate is slightly wrong", "the native sequence equals the target up to a global phase", "the machine has an error"),
              "the native sequence equals the target up to a global phase")),
    _c("virtual_z", 1, "Rotations for free", "virtual Z and the phase frame", ("conv.virtual_z_propagation",), "7.6",
       "A rotation about the vertical axis is never played: the machine remembers it and shifts the phase of every later laser pulse instead.",
       "The phase register shows each ion's remembered angle; hover a gate to see its phase already shifted.",
       "phi -> phi - theta for every later pulse after RZ(theta) (conv.virtual_z_propagation).",
       "explain why a Z rotation costs no time and no error",
       Prompt("virtual_z.q1", "choose", "RZ appears in your circuit. On the pulse schedule you will see",
              ("a short pulse", "nothing: later pulses have shifted phases", "a longer MS pulse"),
              "nothing: later pulses have shifted phases")),
    _c("entanglement_by_ms", 1, "Two arrows that vanish together", "Mølmer-Sørensen gate and entanglement", ("conv.entangling_angle", "conv.native_ms_matrix"), "4.4.1",
       "After the entangling gate each ion's arrow shrinks to nothing, yet the pair is in a definite state: the information sits in the correlation.",
       "Watch both Bloch arrows collapse to the centre after the MS gate while the concurrence rises to one.",
       "XX(chi) = exp(-i chi sigma_x sigma_x); chi = pi/4 is maximally entangling (conv.entangling_angle).",
       "predict what the Bloch arrows do during an entangling gate and say why",
       Prompt("entanglement_by_ms.q1", "predict_direction", "Where do the arrows point right after the MS gate?",
              where="both Bloch arrows in the Register card, with the MS gate selected on Level 1"),
       Prompt("entanglement_by_ms.q2", "free_text", "The arrows vanished but the state is pure. How can both be true?",
              rubric="the two-qubit state is entangled: the reduced single-qubit states are maximally mixed while the joint state has purity one; concurrence near 1")),
    _c("pulse", 2, "A gate is light for a while", "pulse", ("conv.rabi_frequency", "conv.drive_coefficient_and_phase_continuity"), "7.4",
       "Every gate is a laser beam on an ion for a set time; how fast the arrow turns is set by the light's strength.",
       "The schedule shows each pulse as a bar on the ion's lane; its height is the Rabi frequency, its length the duration.",
       "theta = Omega t for a carrier pulse: a quarter turn needs Omega t = pi/2 (conv.rabi_frequency).",
       "estimate a pulse duration from a Rabi frequency",
       Prompt("pulse.q1", "choose", "Doubling the laser power (Rabi frequency x sqrt 2) makes a GPi2 pulse",
              ("longer", "shorter", "the same"),
              "shorter")),
    _c("mode", 2, "How the ions vibrate together", "normal mode", ("conv.mode_index", "conv.mode_eigenvector_gauge"), "4.1.3",
       "Two ions in a trap vibrate in patterns: both together, or against each other, each pattern at its own frequency.",
       "The spectrum under the schedule marks each mode's frequency; hover one to see the ions' motion pattern drawn from the eigenvector.",
       "modes ordered axial, transverse_1, transverse_2, ascending frequency; eigenvectors mass-weighted, unit norm (conv.mode_index).",
       "identify a mode by its frequency and describe how the ions move in it",
       Prompt("mode.q1", "choose", "Two ions have how many vibration modes?", ("2", "3", "6"), "6")),
    _c("tone_and_sideband", 2, "Talking to the motion", "sideband detuning", ("conv.detuning_symbols",), "4.4.1",
       "A laser tuned exactly to the qubit flips it; tuned a mode frequency above or below, it flips the qubit and adds or removes a quantum of vibration.",
       "The MS pulse has two tones drawn just outside the mode lines: one red of the mode, one blue of it.",
       "delta_{i,m} = mu_i - omega_m, always two-indexed; red: mu ~ -omega_m, blue: mu ~ +omega_m (conv.detuning_symbols).",
       "read a tone's position against the mode spectrum and name what it drives",
       Prompt("tone_and_sideband.q1", "choose", "The MS tones sit a few kHz outside which mode?",
              ("the one with the largest chi_m", "the lowest mode", "the qubit frequency"),
              "the one with the largest chi_m")),
    _c("loop_closure", 2, "Borrow the motion, give it back", "loop closure", ("conv.ms_closure", "conv.entangling_sign"), "4.4.3",
       "The entangling gate pushes the ions' motion in a loop that must come back to where it started, or the qubits stay tangled with the motion.",
       "The closure indicators show, per ion and mode, how far each spin branch's loop missed closing; a tiny number means the motion was returned.",
       "alpha_m(tau) ~ 0 for every mode while sum_m chi_m = chi_target (conv.ms_closure).",
       "explain what fails when a loop does not close and where the error shows up",
       Prompt("loop_closure.q1", "choose", "If alpha_m(tau) is not zero after the gate, the qubits are",
              ("still entangled with the motion, which looks like an error", "perfectly fine", "unentangled"),
              "still entangled with the motion, which looks like an error")),
    _c("crosstalk", 2, "Light that spills", "addressing crosstalk", ("conv.crosstalk_ratio",), "6.6",
       "A beam aimed at one ion also lights its neighbour a little, so the neighbour turns a little too.",
       "Each single-ion pulse shows a faint bar on the neighbour's lane with the spill ratio.",
       "eps_ij = Omega_j/Omega_i, a Rabi (amplitude) ratio, not an intensity ratio (conv.crosstalk_ratio).",
       "estimate a neighbour's unwanted rotation from the crosstalk ratio",
       Prompt("crosstalk.q1", "choose", "A 2 % Rabi crosstalk on a pi/2 pulse rotates the neighbour by about",
              ("2 % of pi/2", "0.04 % of pi/2", "pi/2"),
              "2 % of pi/2")),
    _c("spin_dependent_force", 3, "Pushed one way or the other", "spin-dependent force", ("conv.ms_closure", "conv.spin_motion_phases"), "4.4.1",
       "The two tones push the motion in a direction that depends on the qubits' state; the loop's area becomes a phase that entangles them.",
       "The phase-space plot traces alpha_im(t), where one spin branch takes each mode as the pulse runs; the area the pair's two loops sweep together, 2 Im of the integral of conj(alpha_a) d alpha_b, is the entangling angle. The average over both branches, <a>(t), cancels and shows nothing.",
       "H = (hbar Omega/2) sum e^{-i(mu t - phi)} sigma_+ D(i eta) + h.c.; chi from the enclosed phase-space area (Section 4.4.1).",
       "explain how a force on the motion can entangle two qubits",
       Prompt("spin_dependent_force.q0", "predict_closure", "Before the loops are drawn: does the motion of every mode return to where it started by the end of this pulse?",
              ("every loop closes", "at least one loop stays open", "cannot be known before running"),
              where="the phase-space loops of the Dynamics card once revealed, with the closure distance under each"),
       Prompt("spin_dependent_force.q1", "free_text", "Why does the loop have to be closed for the phase to be useful?",
              rubric="an open loop leaves spin-motion entanglement; tracing out the motion then decoheres the qubits; closed loop leaves only the geometric phase")),
    _c("fock_states", 3, "Quanta of vibration", "Fock states and thermal occupation", ("conv.fock_sum_branches", "conv.zero_point_energy"), "5.3",
       "Vibration comes in whole quanta; after cooling a mode holds mostly zero quanta with a small chance of one or more.",
       "The Fock bars at the pulse's start and end show the quanta distribution; the initial mixture is summed exactly over its branches.",
       "rho_th = sum_n p_n |n><n|, p_n = nbar^n/(1 + nbar)^{n+1}; branches below branch_weight_min are dropped and reported (conv.fock_sum_branches).",
       "read a Fock distribution and estimate the mean occupation",
       Prompt("fock_states.q1", "choose", "A cooled mode with nbar = 0.02 is in n = 0 with probability about", ("98 %", "50 %", "2 %"), "98 %")),
    _c("debye_waller", 3, "Thermal motion blurs the laser", "Debye-Waller factor", ("conv.frozen_spectator_shot_sample", "conv.drop_test_debye_waller_spread"), "5.2",
       "An ion that is moving sees the laser's phase smeared, so the same pulse turns it a little less; hotter motion, weaker turn.",
       "The frozen spectator modes list their factor and its shot-to-shot spread; the gate's estimate includes the loss.",
       "e^{-eta^2 (2 n + 1)/2} per frozen mode, n drawn per shot (conv.frozen_spectator_shot_sample).",
       "predict whether heating a spectator mode raises or lowers a gate's fidelity",
       Prompt("debye_waller.q1", "choose", "Heating a spectator mode makes the carrier Rabi frequency",
              ("larger", "smaller and more variable", "unchanged"),
              "smaller and more variable")),
    _c("quantum_jumps", 3, "When the environment acts", "quantum jumps and collapse operators", ("conv.collapse_op_rate_units", "conv.heating_master_equation"), "6.1",
       "Heating, dephasing and scattered photons are random events; the simulation draws them and marks where they happened.",
       "Jump markers on the time axis name the channel; a run without noise has none.",
       "d rho/dt = -i[H, rho] + sum_k D[C_k] rho with C_k = sqrt(rate) op (conv.collapse_op_rate_units).",
       "name three physical channels and the operator each corresponds to",
       Prompt("quantum_jumps.q1", "choose", "A heating jump on a mode applies",
              ("a^dagger, one quantum added", "sigma_z", "a photon-number measurement"),
              "a^dagger, one quantum added")),
    _c("truncation_and_convergence", 3, "How the simulator checks itself", "truncation and convergence", ("conv.boundary_threshold_is_branch_scaled_on_both_sides", "conv.section_9_9_convergence_regime"), "5.5",
       "The simulation keeps only so many vibration levels; it watches how much population reaches the top and re-runs with tighter settings to see if the answer moves.",
       "The numerics panel's badge is red when population hit the cut-off or when tightening the solver changed the answer.",
       "boundary population < 1e-6 of what the pulse moves; tolerances x0.1 and caps +2 change probabilities by < 1e-6 (Sections 5.5, 9.9).",
       "judge whether a simulated result can be trusted from its numerics panel",
       Prompt("truncation_and_convergence.q1", "choose", "The badge says 'not checked'. The result is",
              ("wrong", "converged", "of unknown convergence until the check runs"),
              "of unknown convergence until the check runs")),
    _c("hamiltonian", 4, "The one equation being solved", "the spin-motion Hamiltonian", ("conv.rabi_frequency", "conv.lamb_dicke", "conv.spin_motion_phases"), "4.3.1",
       "Every pulse, every gate and every error in this app comes from integrating one equation with the actual numbers of this device.",
       "The Hamiltonian builder page lists the terms assembled for the pulse you zoomed into, with their matrix elements.",
       "H/hbar = sum_m omega_m a_m^dag a_m + sum_i (omega_0/2) sigma_z + drives (hbar Omega/2) e^{-i(mu t - phi)} sigma_+ prod_m D_m(i eta_im) + h.c. (Section 5.7).",
       "point to the term of H that a chosen knob changes",
       Prompt("hamiltonian.q1", "locate", "Find the matrix element of the drive term that couples |0, n=0> to |1, n=1>.",
              answer="/device/hamiltonian",
              where="the Hamiltonian builder page of Level 4 (Physics in the rail)")),
    _c("lamb_dicke", 4, "How strongly light shakes the ion", "Lamb-Dicke parameter", ("conv.lamb_dicke", "conv.ladder_operators"), "4.1.7",
       "The kick a photon gives, compared with how far the ion already wobbles, sets how strongly light couples spin to motion.",
       "The crystal page lists eta per ion and mode; stiffen the trap and watch eta fall.",
       "eta_{i,m} = (Delta k . e_m) c_{i,m} sqrt(hbar/(2 m_i omega_m)) (conv.lamb_dicke).",
       "predict how eta changes when the trap frequency doubles",
       Prompt("lamb_dicke.q1", "choose", "Raising the mode frequency by a factor 4 changes eta by", ("x2", "x1/2", "x1/4"), "x1/2")),
    _c("trap_and_mathieu", 4, "Why the ion stays put", "Paul trap and the Mathieu equation", ("conv.mathieu_sign", "conv.rf_amplitude_pseudopotential"), "4.1.1",
       "An oscillating voltage cannot hold a charge still, but its average effect is a bowl the ion rolls in.",
       "The trap page draws the stability diagram with this device's operating point and the bowl's frequencies.",
       "x'' + [a - 2q cos 2xi] x = 0, beta ~ sqrt(a + q^2/2), omega_sec = beta Omega_rf/2 (conv.mathieu_sign).",
       "explain what the a and q parameters control and where the operating point sits",
       Prompt("trap_and_mathieu.q1", "choose", "Raising the rf amplitude raises",
              ("q and the radial secular frequencies", "the axial frequency only", "nothing"),
              "q and the radial secular frequencies")),
    _c("atomic_structure", 4, "Which two levels are the qubit", "hyperfine qubit and the level structure", ("conv.frequencies", "conv.hyperfine_dipole_element"), "4.5.1",
       "The qubit is two energy levels of the atom chosen so that magnetic noise barely moves their splitting.",
       "The species page draws the levels, the driven transitions and the qubit pair; the spacing is labelled not to scale.",
       "171Yb+: S1/2 F=0 <-> F=1 at 12.642812118 GHz, first-order field-insensitive at the clock point (Section 4.5.1).",
       "say why a clock-state qubit is chosen and what it costs",
       Prompt("atomic_structure.q1", "choose", "A clock-state qubit is insensitive, to first order, to",
              ("laser power", "magnetic field", "trap voltage"),
              "magnetic field")),
    _c("noise_as_physics", 4, "Errors are not bolted on", "noise channels derived from device parameters", ("conv.electric_field_noise", "conv.noise_provenance"), "6.1",
       "Every error rate here is computed from a physical cause: electric-field noise heats the motion, laser phase noise dephases, scattered photons leak.",
       "The noise page shows the spectra and the rates they imply; change a density and every rate above recomputes.",
       "n_dot = (q^2/(4 m hbar omega)) S_E(omega) (conv.electric_field_noise).",
       "trace a gate error back to the device parameter that causes it",
       Prompt("noise_as_physics.q1", "choose", "Doubling the electric-field noise density S_E doubles",
              ("the heating rate", "the qubit frequency", "the Rabi frequency"),
              "the heating rate")),
    _c("light_coupling", 4, "How much light does what", "Rabi frequency, light shift and scattering from intensity", ("conv.rabi_from_intensity", "conv.two_photon_rabi", "conv.scattering_channels"), "4.5.4",
       "Brighter light turns the qubit faster, but also shifts its frequency and scatters photons; how far the light is tuned from the atomic lines sets the trade-off.",
       "The light page shows the Rabi frequency, the light shift and the scattering error per pulse for each beam pair; the scattering curve falls as the detuning grows.",
       "Omega = sum_e Omega_e^(1) Omega_e^(2)*/(2 Delta_e); P_scatter per pi pulse ~ (pi gamma/omega_f)(2Delta^2 + (Delta - omega_f)^2)/|Delta(Delta - omega_f)| (Sections 4.3.2, 4.5.4).",
       "predict how the scattering error per pulse changes when the laser is detuned further",
       Prompt("light_coupling.q1", "choose", "Doubling the laser power of a Raman pair (both beams) changes the two-photon Rabi frequency by",
              ("x2", "x sqrt 2", "x4"),
              "x2")),
    _c("cooling_ladder", 4, "Cold, then colder", "Doppler cooling, then sideband cooling, then optical pumping", ("conv.preparation_stage_order", "anchor.m3a.doppler_limit", "anchor.m3.rasmusson_pulsed_cooling"), "4.2",
       "Every shot starts by cooling the motion with light: a fast stage takes the ions to a few quanta, a slow stage on the sidebands takes the gate modes to almost none, and a final pump puts the qubits in zero.",
       "The cooling page shows each stage's occupation per mode, the sideband pulses one by one and the pumping populations against time.",
       "Doppler: nbar_D = A_+/(A_- - A_+) from the rate coefficients; pulsed sideband: p -> W_k(t) p per pulse with exact Omega_{n,n-k}; pump: the multi-level master equation (Sections 4.2.1, 4.2.2, 4.2.6).",
       "say why the pump comes last and what a sideband pulse does to the vibration numbers",
       Prompt("cooling_ladder.q1", "choose", "Optical pumping is done last because",
              ("Doppler cooling scrambles the internal state, so a pump done first would be erased", "the pump needs a cold ion to work", "it takes the longest"),
              "Doppler cooling scrambles the internal state, so a pump done first would be erased")),
    _c("readout_rates", 4, "Why the detector cannot look forever", "R_o, R_d, R_b and the window optimum", ("conv.saturation_ceiling", "anchor.m3a.yb171_leakage_prefactors", "conv.readout_figure_of_merit"), "8.1",
       "A bright ion scatters photons at a rate that saturates, while the same light slowly pumps it dark and pumps a dark ion bright; a longer window collects more photons but gives the pumping more time.",
       "The readout page shows the two count histograms, the error against the window length with its minimum, and the three rates against the light level.",
       "R_o = (Gamma/18) s/[1 + (2/9) s + (2 Delta/Gamma)^2] capped at Gamma/4; R_d, R_b linear in s with no saturation, so s stays near 1 (Section 8.1).",
       "explain why the readout error has an interior optimum in the window length",
       Prompt("readout_rates.q1", "choose", "Raising the detection light far above saturation makes",
              ("the bright rate saturate while the pumping rates keep growing: worse discrimination", "everything faster and better", "no difference"),
              "the bright rate saturate while the pumping rates keep growing: worse discrimination")),
    _c("provenance_tags", 0, "What the chips say", "provenance tag", ("conv.noise_provenance", "conv.frequencies"), "14.5",
       "Every number wears a small chip that says what checking was done on it: checked against its source, corrected, taken from a source unchecked, or computed here.",
       "Hover any chip: the glyph and word are the tag, the lines under it the source, the equation and the corrected form; click it to read the section.",
       "The provenance tags: verified, corrected, extracted, background, recomputed here, derived, contested; a tag records what checking was done, never that a value is final.",
       "tell a verified number from an extracted or derived one and say what each tag promises",
       Prompt("provenance_tags.q1", "choose", "A chip reads '❝ extracted'. The number was",
              ("taken from a source with a quote, not independently checked", "checked against its source by verifier agents", "computed by this machine"),
              "taken from a source with a quote, not independently checked")),
    _c("calibration", 4, "The machine measures itself", "calibration emulation", ("conv.calibration_graph", "conv.played_chain"), "7.5",
       "The simulated machine does not know its own parameters; it measures them with the same experiments a laboratory runs, and schedules from those beliefs.",
       "Every table entry says whether it is a seed, a fit, or uncalibrated; a device change turns the table stale until it is re-measured.",
       "requested -> physical through the device's derived values; the scheduler reads only the table (conv.played_chain).",
       "explain why changing a trap voltage makes the calibration stale",
       Prompt("calibration.q1", "choose", "After you change the rf amplitude at Level 4, the gate durations are",
              ("still calibrated", "stale until re-measured", "unchanged forever"),
              "stale until re-measured")),
)}
# fmt: on


def ledger_ids() -> frozenset[str]:
    return frozenset(i for c in CONCEPTS.values() for i in c.ledger_ids)


def level_concepts(level: int) -> tuple[str, ...]:
    """The concepts of one level in teaching order: the explain drawer's chips."""
    return tuple(c.id for c in CONCEPTS.values() if c.level == level)


# ---- the level plan and the explain cards ------------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelPlan:
    """How a level presents itself for one learner."""

    explain_open: bool
    explain_depth: Depth
    expanded: bool
    """Details disclosures and the numerics strip start open."""
    plain_labels_first: bool


def plan_for(knowledge: PriorKnowledge, level: int) -> LevelPlan:
    """Assistance tracks prior knowledge (d = +0.5 for novices, -0.4 for experts); when unknown, assist."""
    if knowledge in ("newcomer", "unknown"):
        return LevelPlan(True, "sentence", False, True)
    if knowledge == "circuits":
        return LevelPlan(level >= 2, "picture", level >= 3, True)
    return LevelPlan(False, "equation", True, False)


@dataclass(frozen=True)
class ExplainCard:
    concept: Concept
    text: str
    deeper: Depth | None
    prompt: Prompt


def explain(id: str, depth: Depth) -> ExplainCard:
    c = CONCEPTS[id]
    k = DEPTHS.index(depth)
    return ExplainCard(c, c.explain.at(depth), DEPTHS[k + 1] if k + 1 < len(DEPTHS) else None, c.prompts[0])


# ---- the spacing rule and the mastery log ------------------------------------------------------------------------------

SPACING_ANCHORS: tuple[tuple[float, float, float], ...] = ((7.0, 0.20, 0.40), (365.0, 0.05, 0.10))
"""(retention days, low share, high share): the declining-share rule (Cepeda et al. 2008; IES 2007)."""

DEFAULT_RETENTION_DAYS = 90.0
"""The default retention target ("explain it to a colleague in three months"); the learner sets it in Learn."""


def review_gap_days(retention_days: float) -> tuple[float, float]:
    """The optimal first gap as a range of days: 20 to 40 % of the target at one week, 5 to 10 % at one year, log-space
    interpolation between, held at the anchors outside. Err toward the longer gap: overshooting costs little."""
    if retention_days <= 0.0:
        raise ValueError("the retention target is positive")
    (d0, lo0, hi0), (d1, lo1, hi1) = SPACING_ANCHORS
    if retention_days <= d0:
        lo, hi = lo0, hi0
    elif retention_days >= d1:
        lo, hi = lo1, hi1
    else:
        t = (math.log(retention_days) - math.log(d0)) / (math.log(d1) - math.log(d0))
        lo = math.exp(math.log(lo0) + t * (math.log(lo1) - math.log(lo0)))
        hi = math.exp(math.log(hi0) + t * (math.log(hi1) - math.log(hi0)))
    return retention_days * lo, retention_days * hi


def now_days() -> float:
    """The learner's clock, in days."""
    return time.time() / 86400.0


@dataclass(frozen=True)
class Attempt:
    concept_id: str
    prompt_id: str
    t_days: float
    """Time of the attempt on the learner's clock."""
    correct: bool | None
    """None for a skip or an unscored answer: an exposure, never a wrong answer."""
    unaided: bool
    """True when the explain panel was closed and no hint was shown."""


@dataclass
class MasteryLog:
    """Attempts per concept; session accuracy and delayed unaided accuracy kept apart (the latter is the evidence)."""

    attempts: list[Attempt] = field(default_factory=list)
    retention_days: float = DEFAULT_RETENTION_DAYS

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)

    def session_accuracy(self, concept_id: str, *, now_days: float) -> float | None:
        """Accuracy within the last day: shown as 'in session', never called learning."""
        recent = [
            a
            for a in self.attempts
            if a.concept_id == concept_id and a.correct is not None and now_days - a.t_days <= 1.0
        ]
        if not recent:
            return None
        return sum(1 for a in recent if a.correct) / len(recent)

    def delayed_unaided_accuracy(self, concept_id: str) -> float | None:
        """Accuracy over unaided attempts made at least the review gap after the previous exposure of the concept."""
        gap_lo, _ = review_gap_days(self.retention_days)
        ordered = sorted((a for a in self.attempts if a.concept_id == concept_id), key=lambda a: a.t_days)
        judged = [
            bool(a.correct)
            for k, a in enumerate(ordered)
            if k > 0 and a.correct is not None and a.unaided and a.t_days - ordered[k - 1].t_days >= gap_lo
        ]
        if not judged:
            return None
        return sum(judged) / len(judged)

    def due(self, now_days: float) -> tuple[str, ...]:
        """Concepts whose last exposure is at least the review gap ago (the long end of the range)."""
        _, gap_hi = review_gap_days(self.retention_days)
        last: dict[str, float] = {}
        for a in self.attempts:
            last[a.concept_id] = max(a.t_days, last.get(a.concept_id, a.t_days))
        return tuple(cid for cid in CONCEPTS if cid in last and now_days - last[cid] >= gap_hi)


# ---- scoring against the record ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionScore:
    total_variation: float
    within_error_bars: bool
    bars_off: tuple[str, ...]
    feedback: str
    """About the task, never about the person (Kluger & DeNisi 1996)."""


def score_histogram_prediction(
    predicted: Mapping[str, float], simulated: Mapping[str, float], error_bars: Mapping[str, float]
) -> PredictionScore:
    """Total variation between a sketched distribution and the simulated one, and which bars miss by more than two
    error bars."""
    keys = sorted(set(predicted) | set(simulated))
    total = sum(max(float(v), 0.0) for v in predicted.values())
    norm = {k: max(float(predicted.get(k, 0.0)), 0.0) / total if total > 0 else 0.0 for k in keys}
    tv = 0.5 * sum(abs(norm[k] - float(simulated.get(k, 0.0))) for k in keys)
    off = tuple(
        k
        for k in keys
        if abs(norm[k] - float(simulated.get(k, 0.0))) > 2.0 * max(float(error_bars.get(k, 0.0)), 1e-12)
    )
    if not off:
        fb = "every bar of the sketch sits within the statistical uncertainty of the simulated one"
    else:
        fb = f"the sketch differs beyond the uncertainty on {', '.join(off)}; open those bars to see the shots and the error budget"
    return PredictionScore(tv, not off, off, fb)


def sketch_distribution(kind: str, target: Mapping[str, float], n_qubits: int) -> dict[str, float]:
    """The histogram a prediction sketch stands for: "ideal" (the ideal machine's distribution ``target``), "uniform"
    (every outcome equally likely) or "zero" (everything in |0...0>)."""
    if kind == "uniform":
        return {format(k, f"0{n_qubits}b"): 1.0 / 2**n_qubits for k in range(2**n_qubits)}
    if kind == "zero":
        return {"0" * n_qubits: 1.0}
    return dict(target)


def score_choice(prompt: Prompt, answer: str) -> bool:
    if prompt.kind != "choose" or prompt.answer is None:
        raise ValueError("score_choice takes a 'choose' prompt with an answer")
    return answer == prompt.answer


LoopKey = tuple[int, int]
"""(ion, mode) of one spin-branch loop of a played entangling waveform."""

CLOSURE_TOLERANCE = 0.05
"""A loop counts as closed when its end-to-start distance is below this fraction of its largest excursion: an open loop at
5 % of its radius leaves 0.25 % of a quantum's worth of spin-motion entanglement per unit (2 nbar + 1) (Section 4.4.3)."""


def score_closure(answer: str, closes: Mapping[LoopKey, float], excursions: Mapping[LoopKey, float]) -> bool:
    """Whether a closure prediction matches the played waveform's spin-branch loops (the spin-averaged <a_m>(t) is never
    scored: the branches cancel in it). A step without loops cannot score either answer."""
    if not closes:
        raise ValueError("no loops to score: the step plays no entangling waveform")
    all_closed = all(closes[m] <= CLOSURE_TOLERANCE * max(excursions[m], 1e-12) for m in closes)
    if answer == "every loop closes":
        return all_closed
    if answer == "at least one loop stays open":
        return not all_closed
    return False


# ---- the exercises -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TourStop:
    route: str
    """A route with ``{id}``, ``{gate}``, ``{pulse}`` and ``{sample}`` to fill."""
    title: str
    concept_id: str
    look_for: str


# fmt: off
BELL_TOUR: tuple[TourStop, ...] = (
    TourStop("/job/{id}", "Two bars, not four", "target_vs_simulated",
             "the 00 and 11 bars near one half, the faint target bars beside them, the tiny 01 and 10"),
    TourStop("/job/{id}/circuit/{gate}", "The gate that made them agree", "entanglement_by_ms",
             "both Bloch arrows collapse after the MS gate while the concurrence reaches one"),
    TourStop("/job/{id}/schedule/{pulse}", "Two tones on either side of a mode", "tone_and_sideband",
             "the red and blue tones straddling the mode with the largest chi_m; the closure indicators"),
    TourStop("/job/{id}/dynamics/{pulse}/{sample}", "The loop that closes", "spin_dependent_force",
             "<a>(t) tracing a loop and returning; P1 of both ions; the concurrence rising"),
    TourStop("/device/hamiltonian", "The equation behind the loop", "hamiltonian",
             "the drive term for the zoomed pulse with its Rabi frequency, detuning and Lamb-Dicke parameter"),
    TourStop("/device/hamiltonian", "One matrix element", "lamb_dicke",
             "the <1, n+1| drive |0, n> element proportional to eta sqrt(n + 1) and the Debye-Waller factor beside it"),
)
# fmt: on
"""The worked example: a Bell-state job followed from a histogram bar to a matrix element in at most six clicks. The
three-ion exercise walks the same stops on a GHZ job with the annotations (``look_for``) withheld until asked."""

FREE_EXERCISE: tuple[str, ...] = (
    "Build a circuit of your own on Level 0: place gates on the wires, or import OpenQASM 2 or IonQ JSON.",
    "Predict its histogram before you run it.",
    "Run it. Find the bar that sits farthest from its target, in error bars.",
    "Follow that bar down: the gate on Level 1, the pulse on Level 2, the loop on Level 3, the term of H on Level 4.",
    "Name the device parameter that would move the bar, then check your sentence against the device card's error budget.",
)
"""The free version of the exercise: build, predict, run, explain one off-target bar."""
