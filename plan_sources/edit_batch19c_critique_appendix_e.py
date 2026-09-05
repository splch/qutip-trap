"""Edit batch 19c (2026-09-04, critique v3 fold): Appendix E public interfaces (p17_interfaces.md)."""

from edit_batch13_errata import apply

TAG = "**[corrected: critique, 2026-09-04]**"

MISSING_TYPES = """
# ---- types the second revision referenced without declaring (2026-09-04 architect critique) ------------
@dataclass(frozen=True)
class State:                                      # what prepare() returns and run_pulses() advances
    internal: Qobj                                  # register density matrix or ket on the ion factors of `space`
    motional: "MotionalModel"                       # per-mode state: Fock-basis density matrix for resolved modes, n_bar for frozen
    joint: Qobj | None                              # the joint ket/density matrix when the run holds one (JOINT_EXACT)
    provenance: tuple[str, ...]                     # ids of the preparation stages that produced it (Section 4.2.6 order)

@dataclass(frozen=True)
class MotionalModel:
    reduced: dict[int, Qobj]                        # mode -> (n_max + 1)^2 reduced density matrix (Section 5.4 b)
    nbar: dict[int, float]; frozen: tuple[int, ...]

@dataclass(frozen=True)
class NoiseSample:                                  # one draw of every quasi-static parameter (Section 6.1)
    sample_id: int; values: dict[str, float]        # field offset, mode offsets, Rabi scale, beam phases, ...
    ou_grids: dict[str, np.ndarray]                 # fixed-grid realizations of the fast processes (Section 5.5)

@dataclass(frozen=True)
class SeedSpec:                                     # one root SeedSequence per run, spawned by key (Section 3.4)
    root: int
    def child(self, sample: int, trajectory: int, shot: int, ion: int, channel: str) -> np.random.SeedSequence: ...

@dataclass(frozen=True)
class SolverOptions:
    atol: float = 1e-10; rtol: float = 1e-8; nsteps: int = 10**7
    integrators: tuple[str, ...] = ("dop853", "vern9")   # the escalation ladder of Section 5.3; never a multistep method
    joint_dimension_max: int = 4096; nnz_max: int = 2 * 10**7   # the Section 11.5 guards that route to GATE_LOCAL
    boundary_population_max: float = 1e-6; freeze_chi_max_rad: float = 0.05
    map: Literal["serial", "parallel", "loky"] = "parallel"     # coefficients are module-level functions or arrays, so they pickle
    e_ops_for_target_tol: bool = True               # mcsolve needs e_ops to target a tolerance (Section 5.4)

@dataclass(frozen=True)
class Traces:                                       # what run_pulses() returns (Section 14.3)
    times_s: np.ndarray; expectations: dict[str, np.ndarray]
    reduced_internal: tuple[Qobj, ...]; mode_occupations: dict[int, np.ndarray]
    alpha_m: dict[int, np.ndarray]; jumps: tuple[tuple[float, str], ...]
    final: State; boundary_population: dict[int, float]

@dataclass(frozen=True)
class CollapseOp:
    op: Qobj; rate_hz: float; channel: str; ion: int | None; mode: int | None

@dataclass(frozen=True)
class ChannelSummary:                               # process tomography of one pulse (Section 5.4)
    choi: np.ndarray; cp_tp_residual: tuple[float, float]; n_traj: int
    average_gate_infidelity: float; pauli_twirled: dict[str, float]; depolarizing_rate: float

@dataclass(frozen=True)
class NoiseSpectrum:                                # ALWAYS two-sided, angular frequency, exp(-i omega t) kernel (Section 13)
    omega_rad_s: np.ndarray; S: np.ndarray; unit: str; sidedness: Literal["two-sided"] = "two-sided"

@dataclass(frozen=True)
class MathieuParameters:
    a: np.ndarray; q: np.ndarray; beta: tuple[float, float, float]; secular_hz: tuple[float, float, float]
    C0: tuple[float, float, float]                  # Wronskian-normalized rf-period Fourier coefficient per axis (Section 4.1.1)
    sign_convention: Literal["a - 2q cos 2xi"] = "a - 2q cos 2xi"   # the plan's rf phase origin (Section 13)

@dataclass(frozen=True)
class MicromotionIndex:
    in_phase: float; out_of_phase: float; convention: Literal["peak", "rms"]

@dataclass(frozen=True)
class Detector:
    kind: Literal["pmt", "camera", "snspd"]; efficiency: float; background_cps: float
    psf_leakage: dict[int, float]                   # neighbour-distance -> fraction (Section 8.5)
    dead_time_s: float | None; afterpulse_prob: float | None; window_s: float

@dataclass(frozen=True)
class POVM:                                         # the readout fast path (Section 5.7)
    per_ion: tuple[np.ndarray, ...] | None          # product form, valid only at zero readout crosstalk
    confusion: np.ndarray | None                    # register-wide 2^N x 2^N tensor otherwise (guarded, Section 5.7)
    crosstalk_domain: Literal["zero", "configured"]

@dataclass(frozen=True)
class HardwareChain:                                # DDS/AOM/amplifier response (Section 7.10)
    dds_phase_bits: int; dds_amplitude_bits: int; aom_rise_s: float; amplifier_bandwidth_hz: float
    dead_time_s: float; phase_continuous: bool

@dataclass(frozen=True)
class DerivedQuantities:                            # every computed number with its provenance id
    values: dict[str, float]; provenance: dict[str, str]

@dataclass(frozen=True)
class ExperimentResult:
    data: np.ndarray; fitted: dict[str, tuple[float, float]]; model: str; provenance_id: str

@dataclass(frozen=True)
class ScheduledEvent:                               # measure / reset / recool with absolute times (Section 7.2)
    kind: Literal["measure", "reset", "recool"]; ions: tuple[int, ...]; t_start_s: float; t_end_s: float

@dataclass(frozen=True)
class RunState:                                     # machine state that PERSISTS across shots (Sections 6.7, 8.5)
    order: tuple[int, ...]                          # physical position -> qubit label; a reorder permutes it for every later shot
    dark: frozenset[int]; lost: frozenset[int]      # dark-ion and loss flags, survival minutes to hours / until reload
    events: tuple[tuple[int, str], ...]             # (shot index, event) log; crystal_image is the calibration experiment that detects them
    def recrystallize(self) -> "RunState": ...
    def reload(self, device: Device) -> "RunState": ...
"""

EDITS = [
    # ---- preamble: hashability is a canonical serialization ----
    (
        "p17_interfaces.md",
        "every object is immutable after construction and hashable, because device hashes key the calibration cache and the run record.",
        "every object is immutable after construction, and identity for the calibration cache and the run record is `Device.hash()`, a canonical serialization rather than Python's `hash()` (which `frozen=True` does not supply for the ndarray, dict and `Qobj` fields these objects carry): enumerated fields in declaration order, floats rounded to 12 significant digits, ndarrays as dtype plus C-order bytes, dicts by sorted key, callables by qualified name plus a source digest, `Qobj` fields excluded, and a cross-process test asserts that the same device file yields the same hash (2026-09-04 architect critique) "
        + TAG
        + ".",
    ),
    # ---- Transition: angular and partial rates ----
    (
        "p17_interfaces.md",
        "    wavelength_vac_m: float; gamma_hz: float       # Gamma/2pi of the upper level's total decay\n    branching: float                                # fine-structure branching into `lower`",
        "    wavelength_vac_m: float; gamma_hz: float       # Gamma/2pi of the upper level's TOTAL decay (an ordinary frequency)\n    branching: float                                # fine-structure branching into `lower`\n    @property\n    def gamma_rad_s(self) -> float: ...             # 2 pi gamma_hz, the angular total rate\n    @property\n    def partial_rate_rad_s(self) -> float: ...      # 2 pi gamma_hz branching: what I_sat = pi h c Gamma_partial/(3 lambda^3) takes\n                                                    #   (171Yb+ 369.5 nm: 50.83 mW/cm^2, asserted at construction; gamma_hz fed in\n                                                    #   unconverted gives 8.09; Section 9.13, check_critique_v3.py)",
    ),
    # ---- Species.dipole_element takes the field ----
    (
        "p17_interfaces.md",
        "    def dipole_element(self, a: str, b: str, q: int) -> complex: ...     # <b|d_q|a> in C m",
        '    def dipole_element(self, a: str, b: str, q: int, field: "Field") -> complex: ...   # <b|d_q|a> in C m in the field-dressed\n                                                    #   eigenbasis (Section 4.5.6: at 146 G the mixing IS the clock point); the zero-field\n                                                    #   reduced element is reduced_element(transition)',
    ),
    # ---- Trap.micromotion_beta ----
    (
        "p17_interfaces.md",
        "    def micromotion_beta(self, species: Species, k_hat: tuple[float, float, float]) -> float: ...  # residual, per beam",
        "    def micromotion_beta(self, species: Species, delta_k: np.ndarray) -> MicromotionIndex: ...\n                                                    # residual beta = delta_k . u_1 for the FULL wavevector (single-photon k or Raman\n                                                    #   Delta k), returned as (in_phase, out_of_phase) with the peak/rms tag of\n                                                    #   Zone.micromotion_convention: in-phase (stray field, nullable by shims) and\n                                                    #   out-of-phase (rf quadrature, Berkeland's phi_ac term, not nullable) never\n                                                    #   collapse into one number (Section 4.1.1)",
    ),
    # ---- Mode declared, canonical ordering, C0 applied in lamb_dicke ----
    (
        "p17_interfaces.md",
        "@dataclass(frozen=True)\nclass Crystal:\n    species: tuple[Species, ...]                    # one entry per ion (mixed species allowed)\n    positions_m: np.ndarray                         # (N, 3) equilibrium positions\n    modes: tuple[Mode, ...]                         # 3N modes: family, index, omega_hz, direction e_hat, eigenvector c (mass-weighted)\n    def lamb_dicke(self, ion: int, mode: int, delta_k: np.ndarray) -> float: ...",
        '@dataclass(frozen=True)\nclass Mode:\n    family: Literal["axial", "transverse_1", "transverse_2"]\n    index: int                                      # position within the family, ascending frequency\n    omega_hz: float; e_hat: tuple[float, float, float]\n    eigenvector: np.ndarray                         # mass-weighted c_{i,m}, unit norm, last component positive (Section 4.1.3)\n\n@dataclass(frozen=True)\nclass Crystal:\n    species: tuple[Species, ...]                    # one entry per ion (mixed species allowed)\n    positions_m: np.ndarray                         # (N, 3) equilibrium positions\n    modes: tuple[Mode, ...]                         # 3N modes in ONE canonical order: axial, transverse_1, transverse_2, ascending\n                                                    #   frequency within a family; every `mode: int` in this appendix is a position in\n                                                    #   this tuple, asserted by the crystal solver\'s tests (Section 4.1.3)\n    def lamb_dicke(self, ion: int, mode: int, delta_k: np.ndarray, *, micromotion: "MathieuParameters | None") -> float: ...\n                                                    # eta_{i,m} = (delta_k . e_hat) c_{i,m} sqrt(hbar/(2 m_i omega_m)) times C0 from\n                                                    #   `micromotion` (Section 4.1.1), applied HERE and nowhere else, recorded in provenance',
    ),
    # ---- Drive: beams-derived delta_k, rf phase reference (first block) ----
    (
        "p17_interfaces.md",
        "    ions: tuple[int, ...]; tones: tuple[Tone, ...]\n    delta_k: np.ndarray; stark_shift_hz: Callable[[float], float] | float\n    crosstalk: dict[int, complex]                   # epsilon_ij onto neighbours\n\n@dataclass(frozen=True)\nclass Pulse:",
        "    ions: tuple[int, ...]; tones: tuple[Tone, ...]\n    beams: tuple[int, ...]                          # indices into Device.beams: one for optical_E1/E2, two for raman (k_1 - k_2)\n    stark_shift_hz: Callable[[float], float] | float\n    crosstalk: dict[int, complex]                   # epsilon_ij onto neighbours\n    rf_locked: bool = False; rf_phase_rad: float | None = None   # pulse start relative to the trap rf (Section 4.3.6); unlocked = averaged\n    @property\n    def delta_k(self) -> np.ndarray: ...            # DERIVED from the beams' wavelengths and k_hat, never a free field: |delta_k| =\n                                                    #   2k sin(theta/2) for a Raman pair at crossing angle theta, k k_hat for one beam,\n                                                    #   0 for microwave (the Monroe anchor needs sqrt2 k, Section 4.2.2)\n\n@dataclass(frozen=True)\nclass Pulse:",
    ),
    # ---- Drive: second (Run 5 comb) block ----
    (
        "p17_interfaces.md",
        "    ions: tuple[int, ...]; tones: tuple[Tone, ...]\n    delta_k: np.ndarray; stark_shift_hz: Callable[[float], float] | float\n    crosstalk: dict[int, complex]                   # epsilon_ij onto neighbours\n    comb:",
        "    ions: tuple[int, ...]; tones: tuple[Tone, ...]\n    beams: tuple[int, ...]; stark_shift_hz: Callable[[float], float] | float   # delta_k DERIVED from `beams`, as above\n    crosstalk: dict[int, complex]                   # epsilon_ij onto neighbours\n    comb:",
    ),
    # ---- Schedule carries events; phase rule ----
    (
        "p17_interfaces.md",
        "class Schedule:\n    pulses: tuple[Pulse, ...]; idle: tuple[tuple[float, float], ...]\n    phase_frame: dict[int, float]                   # per-qubit virtual-Z frame at the end",
        'class Schedule:\n    pulses: tuple[Pulse, ...]; idle: tuple[tuple[float, float], ...]\n    events: tuple["ScheduledEvent", ...]           # measure / reset / recool with absolute times, interleaved with pulses (Section 7.2)\n    phase_frame: dict[int, float]                   # per-qubit virtual-Z frame at the end; the rule is phi -> phi - theta (Section 7.6)',
    ),
    # ---- HilbertSpace: ENR marginal and shape check ----
    (
        "p17_interfaces.md",
        "    def operators(self) -> CachedOperators: ...     # sigma_i, a_m, D_i(delta_k) by expm; analytic oracle checks",
        '    def operators(self) -> CachedOperators: ...     # sigma_i, a_m, D_i(delta_k) by expm; analytic oracle checks\n    def marginal(self, state: "State | Qobj", keep: tuple[int, ...]) -> Qobj: ...   # reduced state over ions/modes, built from\n                                                    #   enr_state_dictionaries when an ENR group is present, since ptrace raises there (5.1)\n    def check(self) -> None: ...                    # asserts shape == prod(dims) on product factors and the ENR shape rule (Section 5.1)',
    ),
    # ---- the missing types, before the noise/device block ----
    (
        "p17_interfaces.md",
        "\n# ---- noise, device (Sections 6, 3.3) ------------------------------------------------------\n@dataclass(frozen=True)\nclass Drift:            # a slow parameter: rms amplitude, correlation time, optional servo bandwidth (Section 7.5)\n    rms: float; tau_s: float; servo_bandwidth_hz: float | None",
        MISSING_TYPES
        + "\n# ---- noise, device (Sections 6, 3.3) ------------------------------------------------------\n@dataclass(frozen=True)\nclass Drift:            # a slow parameter: rms amplitude, correlation time, optional servo bandwidth, optional ramp (Section 7.5)\n    rms: float; tau_s: float; servo_bandwidth_hz: float | None\n    rate_per_s: float = 0.0                         # deterministic ramp (a reference cavity in Hz/s is the dominant optical-qubit drift)",
    ),
    # ---- NoiseModel: pointing drift and rabi_amplitude live here ----
    (
        "p17_interfaces.md",
        "    rabi_drift: Drift; beam_phase_drift: Drift; field_drift: Drift; stray_field_drift: Drift",
        '    rabi_drift: Drift; beam_phase_drift: Drift; field_drift: Drift; stray_field_drift: Drift\n    pointing_drift: Drift                           # beam pointing, which moves crosstalk and Rabi rate together (Section 6.6)\n    rabi_amplitude: "NoiseSpectrum | None"          # two-sided S_a(omega) of the ADDITIVE amplitude noise in rad/s (Section 6.9); Omega is\n                                                    #   taken from the pulse at use and never folded as Omega^2 into a device PSD',
    ),
    # ---- Operation / Circuit: measure, reset, recool in the IR ----
    (
        "p17_interfaces.md",
        "class Operation:\n    name: str; qubits: tuple[int, ...]; params: tuple[float, ...]     # radians; turns at the IonQ boundary\n\n@dataclass(frozen=True)\nclass Circuit:\n    n_qubits: int; ops: tuple[Operation, ...]; measure: tuple[int, ...]",
        'class Operation:\n    name: str; qubits: tuple[int, ...]; params: tuple[float, ...]     # radians; turns at the IonQ boundary; names include the\n                                                    #   non-unitary "measure", "reset" and "recool" (Section 7.2), positioned in `ops`\n\n@dataclass(frozen=True)\nclass Circuit:\n    n_qubits: int; ops: tuple[Operation, ...]; measure: tuple[int, ...]   # `measure` = the terminal targets; mid-circuit measure\n                                                    #   and reset live in `ops`, and the first release\'s scheduler refuses them with an error',
    ),
    # ---- Waveform: per-(ion, leg) segments ----
    (
        "p17_interfaces.md",
        "    segments: tuple[tuple[float, float, float, float], ...] | None   # (duration_s, amplitude_hz, detuning_hz, phase_rad)",
        '    segments: tuple["Segment", ...] | None         # per segment: duration_s, and amplitude_hz and phase_rad indexed by (ion, leg) with\n                                                    #   leg in {red, blue}, detuning_hz per leg; the equal-envelope symmetric-detuning case\n                                                    #   is the constructor shortcut Waveform.symmetric(...); per-ion amplitude imbalance is\n                                                    #   a CalibrationTable entry, and the chi kernel is the symmetrized one whenever\n                                                    #   Omega_a != Omega_b (Section 4.4.3; 2026-09-04 experimentalist critique)',
    ),
    # ---- Diagnostics: mode class, run state, wall clock ----
    (
        "p17_interfaces.md",
        'class Diagnostics:\n    level: Literal["JOINT_EXACT", "GATE_LOCAL"]; space: HilbertSpace',
        'class Diagnostics:\n    level: Literal["JOINT_EXACT", "GATE_LOCAL"]; space: HilbertSpace   # the level actually run; run(level="auto") resolves\n                                                    #   through resolve_level(device, circuit, options) with the Section 5.4 budget\n    mode_class: dict[int, Literal["resolved", "frozen", "dropped"]]   # Section 5.2\'s three classes, per mode\n    run_state: "RunState"                          # ion order, dark/lost flags and event times at the end of the run (Section 6.7)\n    wall_clock_span_s: float                        # t0 to the last shot: shot k is evaluated at t0 + k T_rep (Section 7.5)',
    ),
    # ---- Result: persistent run state beside the heralds ----
    (
        "p17_interfaces.md",
        "    heralds: np.ndarray; discarded_shots: int      # per-shot flags (collision, all-dark, count anomaly); Section 6.7",
        '    heralds: np.ndarray; discarded_shots: int      # per-shot flags (collision, all-dark, count anomaly); Section 6.7\n    run_state: "RunState"                          # persistent machine state threaded through the shots: a dark ion, a reorder or a\n                                                    #   loss changes N, the positions, the modes and the qubit-to-beam map for every LATER\n                                                    #   shot until a recrystallize/reload event, never an i.i.d. per-shot herald',
    ),
    # ---- entry points: run, prepare, resolve_level, calibrate ----
    (
        "p17_interfaces.md",
        'def run(circuit: Circuit, device: Device, shots: int, *, table: CalibrationTable | None = None, t0_s: float = 0.0,\n        samples: int | None = None, level: str = "auto", seed: int = 0, options: SolverOptions | None = None) -> Result: ...\n        # table=None calibrates (surrogate) at t0; a stale table is a legitimate input (Section 7.5)',
        'def run(circuit: Circuit, device: Device, shots: int, *, table: CalibrationTable | None = None, t0_s: float = 0.0,\n        shot_period_s: float | None = None, samples: int | None = None,\n        level: Literal["JOINT_EXACT", "GATE_LOCAL", "auto"] = "auto", seed: int = 0,\n        options: SolverOptions | None = None) -> Result: ...\n        # table=None calibrates (surrogate) at t0; a stale table is a legitimate input (Section 7.5). shot_period_s=None derives\n        # T_rep from the schedule plus cooling, detection and dead time, so shot k sees drift and the mains phase at t0 + k T_rep.\n        # Data flow: space = HilbertSpace.for_(device, schedule, options); sample = device.noise.sample(rng); seeds = SeedSpec(seed)\n        # keyed by (sample, trajectory, shot, ion, channel); state = prepare(device, space, table, sample, seeds);\n        # traces = engine.run_pulses(device, schedule, state, space, sample, seeds, options); readout on traces -> Result.\ndef prepare(device: Device, space: HilbertSpace, table: CalibrationTable, sample: NoiseSample, seeds: SeedSpec) -> State: ...\n        # Doppler -> sideband/EIT -> optical pump, in that order (Section 4.2.6); internal x motional state with provenance\ndef resolve_level(device: Device, circuit: Circuit, options: SolverOptions) -> Literal["JOINT_EXACT", "GATE_LOCAL"]: ...\n        # the Section 5.4 budget: joint dimension <= options.joint_dimension_max (default 4096) and the estimated drive-operator\n        # non-zeros <= options.nnz_max, else GATE_LOCAL (Section 11.5)',
    ),
    (
        "p17_interfaces.md",
        'def calibrate(device: Device, *, seed: int = 0, experiments: tuple[str, ...] = ("all",), surrogate: bool = True,\n              t0_s: float = 0.0) -> CalibrationTable: ...',
        'def calibrate(device: Device, *, seed: int = 0, experiments: tuple[str, ...] = ("all",), surrogate: bool = True,\n              t0_s: float = 0.0) -> CalibrationTable: ...\n        # follows the dependency graph of Section 7.5 (field -> micromotion -> modes -> rabi, stark -> crosstalk -> ms -> detection,\n        # heating) and refuses a downstream fit whose upstream entry is `uncalibrated`; the experiment set includes stark_scan,\n        # crosstalk_scan, field_scan and crystal_image (Sections 7.5, 6.7)',
    ),
    # ---- CombSpec: which Gamma ----
    (
        "p17_interfaces.md",
        "    i_sat_w_m2: float | None = None                 # None = derive pi h c Gamma/(3 lambda^3);\n                                                    #   a value here is the lumped D1 convention",
        "    i_sat_w_m2: float | None = None                 # None = derive pi h c Gamma_partial/(3 lambda^3) with Gamma_partial =\n                                                    #   Transition.partial_rate_rad_s = 2 pi gamma_hz branching (ANGULAR; gamma_hz\n                                                    #   unconverted gives 8.09 instead of 50.83 mW/cm^2 for 171Yb+, Section 9.13);\n                                                    #   a value here is the lumped D1 convention",
    ),
    # ---- CompositePulse: order_slope window, tolerances ----
    (
        "p17_interfaces.md",
        '    def order_slope(self, channel: Literal["amplitude", "detuning", "simultaneous"],\n                    eps_range: tuple[float, float] = (1e-6, 1e-5)) -> float: ...\n    def dc_polygon(self) -> np.ndarray:                 # sum_l theta_l rho-tilde^(l), (3,); 0 iff dc-cancelling\n        ...\n    def certificate(self) -> dict[int, complex]:        # {j: Phi_L^j - f_L^j(gamma)}, equal-area families only\n        ...   # raises for non-uniform areas (SCROFULOUS); order n iff |.| ~ 0 for j <= n and != 0 at n+1',
        '    def order_slope(self, channel: Literal["amplitude", "detuning", "simultaneous"],\n                    eps_range: tuple[float, float] | None = None) -> tuple[float, float]: ...\n        # (slope, fit residual); eps_range=None picks an order-aware window eps >> 10^(-8/(n+1)), about (1e-2, 1e-1) for\n        # n <= 2, because 1 - F ~ eps^{2(n+1)} sits below the 4e-16 round-off floor at the old default (1e-6, 1e-5) for every\n        # compensating family (SK1 2e-19, BB1 9e-30; check_critique_v3.py; 2026-09-04 numerics critique)\n    def dc_polygon(self, rtol: float = 1e-10) -> tuple[np.ndarray, bool]:\n        ...   # sum_l theta_l rho-tilde^(l), (3,), and the boolean |.| < rtol sum_l theta_l\n    def certificate(self, rtol: float = 1e-10) -> dict[int, tuple[complex, bool]]:\n        ...   # {j: (Phi_L^j - f_L^j(gamma), |.| < rtol L)}, equal-area families only; raises for non-uniform areas\n              # (SCROFULOUS); order n iff the boolean holds for j <= n and fails at n+1',
    ),
    # ---- filter_function: the cutoff in rad/s, the misplaced NoiseModel fields ----
    (
        "p17_interfaces.md",
        '    # amplitude one. omega_min_rad_s defaults to 1/(total experiment duration) and is ALWAYS reported,\n    # because chi is IR-divergent for free induction, the Hahn echo and every odd-n sequence on a\n    # 1/omega^4 spectrum. monte_carlo_samples > 0 additionally runs the sampled-trajectory path of\n    # Section 6.1(d) through the Section 4.3.1 Hamiltonian and returns both, so the two must agree\n    # inside xi^2 << 1; the result records xi^2 and flags the comparison when it is not small.\n    laser_phase: "NoiseSpectrum | None"; laser_intensity: "NoiseSpectrum | None"\n    rabi_amplitude: "NoiseSpectrum | None"   # ADDED: two-sided S_a(omega) of the ADDITIVE amplitude\n                                             # noise beta_a in rad/s (multiply a relative-amplitude\n                                             # spectrum by Omega^2 on ingest); Section 6.9',
        "    # amplitude one. omega_min_rad_s defaults to 2 pi/(total experiment duration), in rad/s as its name says (the second\n    # revision wrote 1/T for a rad/s field), and is ALWAYS reported together with the local sensitivity d ln chi/d ln omega_min,\n    # because chi is IR-divergent for free induction, the Hahn echo and every odd-n sequence on a 1/omega^4 spectrum and is then\n    # a function of the declared cutoff: on the T = 1 free-induction fixture chi(1/T)/chi(2 pi/T) = 1.06e4 (check_critique_v3.py;\n    # the (2 pi)^3 = 248 of the low-frequency estimate holds only for omega_min T << 1). monte_carlo_samples > 0 additionally\n    # runs the sampled-trajectory path of Section 6.1(d) through the Section 4.3.1 Hamiltonian and returns both, so the two\n    # must agree inside xi^2 << 1; the result records xi^2 and flags the comparison when it is not small. (The second revision\n    # pasted three NoiseModel field declarations here; they live on NoiseModel above, where rabi_amplitude is declared.)",
    ),
]

if __name__ == "__main__":
    apply(EDITS)
