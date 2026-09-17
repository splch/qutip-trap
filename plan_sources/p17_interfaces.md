## Appendix E. Public interfaces

The 2026-09-04 critique observed that a document calling itself the specification an implementer codes from contained no function signature, field list or protocol. This appendix fixes the public surface that milestone M0 freezes and that the application of Section 14 is allowed to depend on. Types are written as Python dataclasses and protocols; frequencies in the public API are in Hz and are converted to rad/s at the boundary (Section 5.6); every object is immutable after construction, and identity for the calibration cache and the run record is `Device.hash()`, a canonical serialization rather than Python's `hash()` (which `frozen=True` does not supply for the ndarray, dict and `Qobj` fields these objects carry): enumerated fields in declaration order, floats rounded to 12 significant digits, ndarrays as dtype plus C-order bytes, dicts by sorted key, callables by qualified name plus a source digest, `Qobj` fields excluded, and a cross-process test asserts that the same device file yields the same hash (2026-09-04 architect critique) **[corrected: critique, 2026-09-04]**.

```python
# ---- species and fields (Section 4.5) --------------------------------------------------------
@dataclass(frozen=True)
class Level:            # one fine-structure level
    name: str           # "S1/2", "P1/2", "D5/2", ...
    energy_hz: float    # from the ground level
    lifetime_s: float | None
    A_hfs_hz: float; B_hfs_hz: float; g_J: float
    citations: tuple[str, ...]

@dataclass(frozen=True)
class Transition:       # one fine-structure transition
    lower: str; upper: str
    wavelength_vac_m: float; gamma_hz: float       # Gamma/2pi of the upper level's TOTAL decay (an ordinary frequency)
    branching: float                                # fine-structure branching into `lower`
    @property
    def gamma_rad_s(self) -> float: ...             # 2 pi gamma_hz, the angular total rate
    @property
    def partial_rate_rad_s(self) -> float: ...      # 2 pi gamma_hz branching: what I_sat = pi h c Gamma_partial/(3 lambda^3) takes
                                                    #   (171Yb+ 369.5 nm: 50.83 mW/cm^2, asserted at construction; gamma_hz fed in
                                                    #   unconverted gives 8.09; Section 9.13, check_critique_v3.py)
    multipole: Literal["E1", "E2", "M1"]
    citations: tuple[str, ...]

@dataclass(frozen=True)
class Species:
    name: str; mass_u: float; nuclear_spin: float; mu_I_nuclear_magnetons: float
    levels: tuple[Level, ...]; transitions: tuple[Transition, ...]
    qubit: tuple[str, str]                          # state labels "S1/2 F=0 mF=0" or "S1/2 mJ=-1/2"
    cycling: str; repumps: tuple[str, ...]; shelving: str | None
    # derived (Section 4.5), computed lazily and cached, all carrying provenance:
    def zeeman_spectrum(self, level: str, B_gauss: float) -> ZeemanSpectrum: ...
    def transition_frequency_hz(self, a: str, b: str, B_gauss: float) -> tuple[float, float, float]:
        ...   # value, d/dB (Hz/G), d2/dB2 (Hz/G^2)
    def dipole_element(self, a: str, b: str, q: int, field: "Field") -> complex: ...   # <b|d_q|a> in C m in the field-dressed
                                                    #   eigenbasis (Section 4.5.6: at 146 G the mixing IS the clock point); the zero-field
                                                    #   reduced element is reduced_element(transition)
    def rabi_frequency_hz(self, a: str, b: str, beam: "Beam", field: "Field") -> complex: ...
    def raman_coupling_hz(self, g1: str, g2: str, beam1: "Beam", beam2: "Beam", field: "Field") -> complex: ...
    def light_shift_hz(self, g: str, beam: "Beam", field: "Field") -> float: ...
    def scattering_rates_hz(self, a: str, beam: "Beam", field: "Field") -> dict[str, float]: ...  # a -> b, incl. Rayleigh
    def rayleigh_dephasing_hz(self, beam: "Beam", field: "Field") -> float: ...

@dataclass(frozen=True)
class Field:            # static magnetic field: quantization axis and noise
    B_gauss: float; direction: tuple[float, float, float]
    noise: "NoiseSpectrum | None"

# ---- trap, crystal, beams (Sections 4.1, 4.5.3) --------------------------------------------
@dataclass(frozen=True)
class Trap:             # either secular frequencies or voltages plus geometry
    omega_hz: tuple[float, float, float] | None
    axis_angle_rad: float                           # rotation of the radial principal axes about z (explicit path)
    rf: "RfDrive | None"; dc: "DcElectrodes | None"; geometry: "Electrodes | None"
    stray_field_v_per_m: tuple[float, float, float] # true stray field (hidden from the scheduler, Section 7.3)
    shim_voltages_v: dict[str, float]               # compensation applied; residual = stray + shim response
    def mathieu(self, species: Species) -> MathieuParameters: ...   # a, q (matrices), beta, secular, C0
    def micromotion_beta(self, species: Species, delta_k: np.ndarray) -> MicromotionIndex: ...
                                                    # residual beta = delta_k . u_1 for the FULL wavevector (single-photon k or Raman
                                                    #   Delta k), returned as (in_phase, out_of_phase) with the peak/rms tag of
                                                    #   Zone.micromotion_convention: in-phase (stray field, nullable by shims) and
                                                    #   out-of-phase (rf quadrature, Berkeland's phi_ac term, not nullable) never
                                                    #   collapse into one number (Section 4.1.1)
    def anharmonic(self) -> AnharmonicTerms | None: ...

@dataclass(frozen=True)
class Mode:
    family: Literal["axial", "transverse_1", "transverse_2"]
    index: int                                      # position within the family, ascending frequency
    omega_hz: float; e_hat: tuple[float, float, float]
    eigenvector: np.ndarray                         # mass-weighted c_{i,m}, unit norm, last component positive (Section 4.1.3)

@dataclass(frozen=True)
class Crystal:
    species: tuple[Species, ...]                    # one entry per ion (mixed species allowed)
    positions_m: np.ndarray                         # (N, 3) equilibrium positions
    modes: tuple[Mode, ...]                         # 3N modes in ONE canonical order: axial, transverse_1, transverse_2, ascending
                                                    #   frequency within a family; every `mode: int` in this appendix is a position in
                                                    #   this tuple, asserted by the crystal solver's tests (Section 4.1.3)
    def lamb_dicke(self, ion: int, mode: int, delta_k: np.ndarray, *, micromotion: "MathieuParameters | None") -> float: ...
                                                    # eta_{i,m} = (delta_k . e_hat) c_{i,m} sqrt(hbar/(2 m_i omega_m)) times C0 from
                                                    #   `micromotion` (Section 4.1.1), applied HERE and nowhere else, recorded in provenance

@dataclass(frozen=True)
class Beam:
    wavelength_m: float; k_hat: tuple[float, float, float]; polarization: tuple[complex, complex, complex]
    waist_m: float; power_w: float; pointing_m: tuple[float, float, float]
    def intensity_at(self, position_m: np.ndarray) -> float: ...

# ---- drives, pulses, schedules (Sections 4.3, 5.2, 7.4) -------------------------------------
@dataclass(frozen=True)
class Tone:
    detuning_hz: Callable[[float], float] | float   # mu(t) from the carrier
    phase_rad: Callable[[float], float] | float     # phi_tone(t) in the ion frame (Section 5.2)
    envelope_hz: Callable[[float], float] | np.ndarray   # Omega(t)

@dataclass(frozen=True)
class Drive:
    kind: Literal["raman", "optical_E1", "optical_E2", "microwave", "gradient"]
    ions: tuple[int, ...]; tones: tuple[Tone, ...]
    beams: tuple[int, ...]                          # indices into Device.beams: one for optical_E1/E2, two for raman (k_1 - k_2)
    stark_shift_hz: Callable[[float], float] | float
    crosstalk: dict[int, complex]                   # epsilon_ij onto neighbours
    rf_locked: bool = False; rf_phase_rad: float | None = None   # pulse start relative to the trap rf (Section 4.3.6); unlocked = averaged
    @property
    def delta_k(self) -> np.ndarray: ...            # DERIVED from the beams' wavelengths and k_hat, never a free field: |delta_k| =
                                                    #   2k sin(theta/2) for a Raman pair at crossing angle theta, k k_hat for one beam,
                                                    #   0 for microwave (the Monroe anchor needs sqrt2 k, Section 4.2.2)

@dataclass(frozen=True)
class Pulse:
    drive: Drive; t_start_s: float; t_end_s: float; gate_id: str | None
    closes_modes: tuple[int, ...]

@dataclass(frozen=True)
class Schedule:
    pulses: tuple[Pulse, ...]; idle: tuple[tuple[float, float], ...]
    events: tuple["ScheduledEvent", ...]           # measure / reset / recool with absolute times, interleaved with pulses (Section 7.2)
    phase_frame: dict[int, float]                   # per-qubit virtual-Z frame at the end; the rule is phi -> phi - theta (Section 7.6)

# ---- Hilbert space and engine (Sections 5, 11) ---------------------------------------------
@dataclass(frozen=True)
class ModeTruncation:
    mode: int; d: int                               # d = n_max + 1 Fock levels
    expected_n_range: tuple[int, int]; eta_max: float

@dataclass(frozen=True)
class HilbertSpace:
    ion_dims: tuple[int, ...]                       # 2, or 2 + leakage levels
    resolved: tuple[ModeTruncation, ...]            # product space
    enr_group: tuple[tuple[int, ...], int] | None   # (modes, N_exc), optional
    frozen: tuple[int, ...]                         # frozen spectators
    def operators(self) -> CachedOperators: ...     # sigma_i, a_m, D_i(delta_k) by expm; analytic oracle checks
    def marginal(self, state: "State | Qobj", keep: tuple[int, ...]) -> Qobj: ...   # reduced state over ions/modes, built from
                                                    #   enr_state_dictionaries when an ENR group is present, since ptrace raises there (5.1)
    def check(self) -> None: ...                    # asserts shape == prod(dims) on product factors and the ENR shape rule (Section 5.1)

class PulseEngine(Protocol):
    def run_pulses(self, device: "Device", schedule: Schedule, state: State, space: HilbertSpace,
                   sample: NoiseSample, seeds: SeedSpec, options: SolverOptions) -> Traces: ...
    def process_tomography(self, device: "Device", pulse: Pulse, space: HilbertSpace,
                           motional_model: MotionalModel, sample: NoiseSample, seeds: SeedSpec) -> ChannelSummary: ...

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

# ---- noise, device (Sections 6, 3.3) ------------------------------------------------------
@dataclass(frozen=True)
class Drift:            # a slow parameter: rms amplitude, correlation time, optional servo bandwidth, optional ramp (Section 7.5)
    rms: float; tau_s: float; servo_bandwidth_hz: float | None
    rate_per_s: float = 0.0                         # deterministic ramp (a reference cavity in Hz/s is the dominant optical-qubit drift)

@dataclass(frozen=True)
class NoiseModel:
    S_E: "NoiseSpectrum"; correlation_length_m: float | None
    S_B: "NoiseSpectrum | None"; mains: "Mains | None"      # amplitude per harmonic, line-trigger policy (Section 6.3)
    laser_phase: "NoiseSpectrum | None"; laser_intensity: "NoiseSpectrum | None"
    rf_amplitude_noise: "NoiseSpectrum | None"; rf_phase_noise: "NoiseSpectrum | None"   # Section 4.1.5 sidebands, 2 omega_sec
    rf_amplitude_drift: Drift                       # common-mode fractional drift of every rf-derived mode of a family
    mode_drift_differential: Drift                  # per-mode dc-derived drift
    rabi_drift: Drift; beam_phase_drift: Drift; field_drift: Drift; stray_field_drift: Drift
    pointing_drift: Drift                           # beam pointing, which moves crosstalk and Rabi rate together (Section 6.6)
    rabi_amplitude: "NoiseSpectrum | None"          # two-sided S_a(omega) of the ADDITIVE amplitude noise in rad/s (Section 6.9); Omega is
                                                    #   taken from the pulse at use and never folded as Omega^2 into a device PSD
    collisions: "Collisions | None"                  # pressure_pa, gas, outcome probabilities (Section 6.7)
    def channels(self, device: "Device", space: HilbertSpace) -> tuple[CollapseOp, ...]: ...
    def sample(self, rng: np.random.Generator) -> NoiseSample: ...

@dataclass(frozen=True)
class Device:
    crystal: Crystal; trap: Trap; field: Field; beams: tuple[Beam, ...]
    noise: NoiseModel; detector: Detector; hardware: HardwareChain
    def derived(self) -> DerivedQuantities: ...     # every computed number with its provenance id
    def hash(self) -> str: ...

# ---- circuits, calibration, results (Sections 7, 8) ----------------------------------------
@dataclass(frozen=True)
class Operation:
    name: str; qubits: tuple[int, ...]; params: tuple[float, ...]     # radians; turns at the IonQ boundary; names include the
                                                    #   non-unitary "measure", "reset" and "recool" (Section 7.2), positioned in `ops`

@dataclass(frozen=True)
class Circuit:
    n_qubits: int; ops: tuple[Operation, ...]; measure: tuple[int, ...]   # `measure` = the terminal targets; mid-circuit measure
                                                    #   and reset live in `ops`, and the first release's scheduler refuses them with an error

@dataclass(frozen=True)
class CalEntry:
    value: float; uncertainty: float; status: Literal["seed", "calibrated", "uncalibrated"]
    experiment: str; provenance_id: str
    fitted_at_s: float; sample_id: int              # calibration age and the noise sample it was fitted under (Section 7.5)

@dataclass(frozen=True)
class Waveform:         # a calibrated entangling pulse: what the scheduler plays
    segments: tuple["Segment", ...] | None         # per segment: duration_s, and amplitude_hz and phase_rad indexed by (ion, leg) with
                                                    #   leg in {red, blue}, detuning_hz per leg; the equal-envelope symmetric-detuning case
                                                    #   is the constructor shortcut Waveform.symmetric(...); per-ion amplitude imbalance is
                                                    #   a CalibrationTable entry, and the chi kernel is the symmetrized one whenever
                                                    #   Omega_a != Omega_b (Section 4.4.3; 2026-09-04 experimentalist critique)
    fourier: tuple[complex, ...] | None             # or Fourier coefficients of the modulation (Section 4.4.3)
    duration_s: float; phi_s: CalEntry; phi_m: CalEntry
    chi_m: dict[int, float]; alpha_m: dict[int, complex]   # per-mode entangling angle and residual displacement at closure

@dataclass(frozen=True)
class CalibrationTable:                           # plain data; control.schedule reads it, never writes it
    device_hash: str; seed: int; surrogate: bool     # closed-form surrogate with spot checks, or full simulated experiments
    qubit_freq: dict[int, CalEntry]                 # from the Ramsey-frequency experiment; the scheduler never reads the true value
    rabi: dict[tuple[int, int], CalEntry]           # (ion, beam)
    stark: dict[tuple[int, int], CalEntry]; crosstalk: dict[tuple[int, int], CalEntry]
    modes: dict[int, CalEntry]; nbar: dict[int, CalEntry]
    ms: dict[tuple[int, int], Waveform]
    field: CalEntry; micromotion: dict[str, CalEntry]   # shim voltages and residual beta per beam direction
    detection: dict[str, CalEntry]; heating: dict[int, CalEntry]

@dataclass(frozen=True)
class Diagnostics:
    level: Literal["JOINT_EXACT", "GATE_LOCAL"]; space: HilbertSpace   # the level actually run; run(level="auto") resolves
                                                    #   through resolve_level(device, circuit, options) with the Section 5.4 budget
    mode_class: dict[int, Literal["resolved", "frozen", "dropped"]]   # Section 5.2's three classes, per mode
    run_state: "RunState"                          # ion order, dark/lost flags and event times at the end of the run (Section 6.7)
    wall_clock_span_s: float                        # t0 to the last shot: shot k is evaluated at t0 + k T_rep (Section 7.5)
    boundary_population: dict[int, float]; margin_levels: dict[int, int]
    dropped_modes: tuple[int, ...]; frozen_contribution: dict[int, tuple[float, float]]   # (|alpha|^2(2n+1), chi)
    integrator: str; tolerances: tuple[float, float]; samples: int; trajectories: int; shots_per_sample: int
    effective_sample_size: float; root_seed: int; calibration: CalibrationTable; approximations: tuple[str, ...]

@dataclass(frozen=True)
class Result:
    bitstrings: np.ndarray; bit_order: Literal["qubit0_lsb"]
    counts: dict[str, int]; probabilities: dict[str, float]; error_bars: dict[str, float]
    photon_records: np.ndarray | None; posteriors: np.ndarray | None; noise_samples: tuple[NoiseSample, ...]
    heralds: np.ndarray; discarded_shots: int      # per-shot flags (collision, all-dark, count anomaly); Section 6.7
    run_state: "RunState"                          # persistent machine state threaded through the shots: a dark ion, a reorder or a
                                                    #   loss changes N, the positions, the modes and the qubit-to-beam map for every LATER
                                                    #   shot until a recrystallize/reload event, never an i.i.d. per-shot herald
    spam: dict[str, tuple[float, float]]           # per qubit (eps_B, eps_D) with the definition used
    final_state: Qobj | None; diagnostics: Diagnostics
    def to_ionq_json(self) -> dict: ...            # decimal-integer keys, qubit 0 least significant

# ---- entry points -------------------------------------------------------------------------
def run(circuit: Circuit, device: Device, shots: int, *, table: CalibrationTable | None = None, t0_s: float = 0.0,
        shot_period_s: float | None = None, samples: int | None = None,
        level: Literal["JOINT_EXACT", "GATE_LOCAL", "auto"] = "auto", seed: int = 0,
        options: SolverOptions | None = None) -> Result: ...
        # table=None calibrates (surrogate) at t0; a stale table is a legitimate input (Section 7.5). shot_period_s=None derives
        # T_rep from the schedule plus cooling, detection and dead time, so shot k sees drift and the mains phase at t0 + k T_rep.
        # Data flow: space = HilbertSpace.for_(device, schedule, options); sample = device.noise.sample(rng); seeds = SeedSpec(seed)
        # keyed by (sample, trajectory, shot, ion, channel); state = prepare(device, space, table, sample, seeds);
        # traces = engine.run_pulses(device, schedule, state, space, sample, seeds, options); readout on traces -> Result.
def prepare(device: Device, space: HilbertSpace, table: CalibrationTable, sample: NoiseSample, seeds: SeedSpec) -> State: ...
        # Doppler -> sideband/EIT -> optical pump, in that order (Section 4.2.6); internal x motional state with provenance
def resolve_level(device: Device, circuit: Circuit, options: SolverOptions) -> Literal["JOINT_EXACT", "GATE_LOCAL"]: ...
        # the Section 5.4 budget: joint dimension <= options.joint_dimension_max (default 4096) and the estimated drive-operator
        # non-zeros <= options.nnz_max, else GATE_LOCAL (Section 11.5)
def calibrate(device: Device, *, seed: int = 0, experiments: tuple[str, ...] = ("all",), surrogate: bool = True,
              t0_s: float = 0.0) -> CalibrationTable: ...
        # follows the dependency graph of Section 7.5 (field -> micromotion -> modes -> rabi, stark -> crosstalk -> ms -> detection,
        # heating) and refuses a downstream fit whose upstream entry is `uncalibrated`; the experiment set includes stark_scan,
        # crosstalk_scan, field_scan and crystal_image (Sections 7.5, 6.7)
def compile_to_native(circuit: Circuit, device: Device) -> Circuit: ...
def schedule(circuit: Circuit, device: Device, table: CalibrationTable) -> Schedule: ...

# experiments (Sections 7.5, 7.9); each returns data plus the fitted parameters with uncertainties
def rabi_scan(device, ion, durations_s, **kw) -> ExperimentResult: ...
def ramsey(device, ion, delays_s, **kw) -> ExperimentResult: ...
def ramsey_frequency(device, ion, delays_s, **kw) -> ExperimentResult: ...       # qubit frequency for the table
def micromotion_scan(device, ion, beam, shim_ranges_v, method="rf_photon_correlation", **kw) -> ExperimentResult: ...
def sideband_spectroscopy(device, ion, detunings_hz, **kw) -> ExperimentResult: ...
def ms_scan(device, pair, amplitudes, detunings_hz, **kw) -> ExperimentResult: ...
def parity_scan(device, pair, analysis_phases_rad, **kw) -> ExperimentResult: ...
def heating_rate(device, mode, delays_s, **kw) -> ExperimentResult: ...
def detection_histogram(device, ion, n_records, **kw) -> ExperimentResult: ...
def load_ionq_json(obj: dict) -> Circuit: ...
def load_openqasm2(text: str) -> Circuit: ...
```

**Run 5 additions (2026-09-04).** The topics folded in on 2026-09-04 add the objects below; they follow the same rules (frozen dataclasses, Hz in the public API, provenance on every derived number) and the amendments to existing declarations are listed after the code.

```python
# ---- Run 5: Frequency-comb Raman drives (Section 4.3.7) ----
# ---- comb drives (Section 4.3.7) ------------------------------------------------------------
@dataclass(frozen=True)
class CombSpec:            # the mode-locked train that generates a Drive's tone set
    rep_rate_hz: float                              # nu_rep, ordinary Hz; the tooth spacing
    tau_s: float                                    # sech ENVELOPE parameter, never a FWHM
    tau_convention: Literal["field_sech", "intensity_sech", "intensity_sech2"]
    carrier_order: int                              # j = round(omega_HF / omega_rep), signed
    aom_offset_hz: float                            # Delta_nu_M = nu_1 - nu_2, SIGNED; its sign
                                                    #   selects red vs blue (Section 4.3.7)
    pair_order_max: int = 1200                       # |l - j| retained; the C_{n,a} sum needs >~ 1000
    chain: Literal["I", "II"] = "I"                 # Omega = g^2/(2 Delta) vs g^2/Delta
    i_sat_w_m2: float | None = None                 # None = derive pi h c Gamma_partial/(3 lambda^3) with Gamma_partial =
                                                    #   Transition.partial_rate_rad_s = 2 pi gamma_hz branching (ANGULAR; gamma_hz
                                                    #   unconverted gives 8.09 instead of 50.83 mW/cm^2 for 171Yb+, Section 9.13);
                                                    #   a value here is the lumped D1 convention
    rep_rate_trajectory: Callable[[float], float] | None = None   # delta_r(t) in Hz (Section 6.1(d))
    locked_tooth: int | None = None                 # n of the feed-forward lock; None = unlocked.
                                                    #   Residual per tooth m: (m-n) lower sideband,
                                                    #   (m+n) upper, m bare (Section 4.3.7)

    def tooth_amplitudes(self, n_teeth: int) -> np.ndarray: ...   # g_k/g_0, renormalized to the sum rule
    def pair_weight(self, l: int) -> float: ...                   # sech(pi l nu_rep tau); NOT sech(2 pi l ...)
    def tones(self, omega_q_hz: float, mode_hz: float | None) -> tuple[Tone, ...]: ...
    def beat_note_hz(self, order: int) -> float: ...              # |order*nu_rep + aom_offset_hz|
    def solve_offset(self, target_hz: float) -> tuple[int, float]: ...  # root-find (j, Delta_nu_M)
    def stark4_hz(self, levels: "Sequence[Level]") -> np.ndarray: ...   # the l-sum of Section 4.3.7
    def guards(self, omega_q_hz: float, eta: float, nbar: float,
               t_train_s: float, mode_hz: float) -> dict[str, bool]: ...  # the validity hierarchy
@dataclass(frozen=True)
class Drive:
    kind: Literal["raman", "optical_E1", "optical_E2", "microwave", "gradient"]
    ions: tuple[int, ...]; tones: tuple[Tone, ...]
    beams: tuple[int, ...]; stark_shift_hz: Callable[[float], float] | float   # delta_k DERIVED from `beams`, as above
    crosstalk: dict[int, complex]                   # epsilon_ij onto neighbours
    comb: "CombSpec | None" = None                  # set for a mode-locked Raman drive; when
                                                    #   present, `tones` is generated by comb.tones()
                                                    #   and stark_shift_hz by comb.stark4_hz()

# ---- Run 5: Composite pulses, dynamical decoupling and filter functions (Sections 4.3.5, 6.9) ----
@dataclass(frozen=True)
class CompositePulse:            # a static-error-compensating single-qubit sequence (Section 4.3.5)
    family: Literal["primitive", "SK1", "SKn", "BB1", "NB1", "PB1", "P2j", "N2j", "B2j",
                    "CORPSE", "short_CORPSE", "SCROFULOUS", "B2CORPSE",
                    "CinSK", "CinBB", "PDn", "APn", "ToPn", "BBn"]
    theta_rad: float                          # target rotation angle
    phi_rad: float                            # target axis azimuth; ADDED to every segment phase
    order: int                                # n; residual O(eps^{n+1}), infidelity slope 2(n+1)
    corrects: frozenset[Literal["amplitude", "pulse_length", "addressing", "detuning"]]
    segments: tuple[tuple[float, float], ...] # (area_rad, phase_rad) in TIME order, areas > 0
    n_rep: int = 1                            # P2j/N2j/B2j: repetitions of the cached T_{2j-2} block
    provenance_id: str = ""                   # ledger id; carries the [verified]/[corrected] tag
    # ---- derived, all pure functions of the fields ----
    def total_rotation_rad(self) -> float: ...          # sum of areas; duration is this over Omega
    def propagator(self, eps_a: float = 0.0, eps_d: float = 0.0, eps_N: float | None = None) -> np.ndarray:
        ...   # 2x2, folded right-to-left; eps_N selects the addressing model with target = I
    def infidelity(self, eps_a: float = 0.0, eps_d: float = 0.0, measure: str = "F_K") -> float: ...
    def order_slope(self, channel: Literal["amplitude", "detuning", "simultaneous"],
                    eps_range: tuple[float, float] | None = None) -> tuple[float, float]: ...
        # (slope, fit residual); eps_range=None picks an order-aware window eps >> 10^(-8/(n+1)), about (1e-2, 1e-1) for
        # n <= 2, because 1 - F ~ eps^{2(n+1)} sits below the 4e-16 round-off floor at the old default (1e-6, 1e-5) for every
        # compensating family (SK1 2e-19, BB1 9e-30; check_critique_v3.py; 2026-09-04 numerics critique)
    def dc_polygon(self, rtol: float = 1e-10) -> tuple[np.ndarray, bool]:
        ...   # sum_l theta_l rho-tilde^(l), (3,), and the boolean |.| < rtol sum_l theta_l
    def certificate(self, rtol: float = 1e-10) -> dict[int, tuple[complex, bool]]:
        ...   # {j: (Phi_L^j - f_L^j(gamma), |.| < rtol L)}, equal-area families only; raises for non-uniform areas
              # (SCROFULOUS); order n iff the boolean holds for j <= n and fails at n+1
    def filter_function_amplitude(self, omega_rad_s: np.ndarray, omega_rabi_rad_s: float) -> np.ndarray:
        ...   # exact A_l/B_l segment sum, no quadrature (Section 6.9)
    def dc_floor(self, moments: dict[str, float], omega_rabi_rad_s: float) -> float: ...
        # c-hat_{m+1} <(beta/Omega)^{2(m+1)}>; c-hat fitted numerically and cached per (family, theta)

def composite_pulse(family: str, theta_rad: float, phi_rad: float = 0.0, *, order: int = 1,
                    n_rep: int = 1) -> CompositePulse: ...
    # phases from the verified closed forms (Section 4.3.5); raises on |theta| outside the arccos domain
    # and on the even-n-only families at odd n, never returning NaN
@dataclass(frozen=True)
class DecouplingSequence:        # a net-identity pulse train, or a decoupled idle (Section 6.9)
    timing: Literal["hahn", "cpmg", "udd", "xy4", "xy8", "kdd", "cdd", "custom"]
    n_pulses: int
    tau_s: float                              # TOTAL duration, INCLUSIVE of the pi-pulse widths
    tau_pi_s: float                           # one pulse duration; delta_pi = tau_pi_s/tau_s
    deltas: tuple[float, ...]                 # fractional pulse CENTRES in [0, 1]; len == n_pulses
    axes_rad: tuple[float, ...]               # per-pulse axis azimuth; all zero for single-axis families
    inner: CompositePulse | None = None       # a composite pi pulse in place of a primitive one
    provenance_id: str = ""
    def is_single_axis(self) -> bool: ...     # if False the scalar (-1)^l bookkeeping is INVALID
    def feasible(self) -> bool: ...           # delta_pi <= 2 sin^2[pi/(2n+2)] and no pulse overlap
    def moments(self, k_max: int = 2) -> tuple[float, ...]: ...      # A_k = sum_j (-1)^j delta_j^k
    def control_matrix(self, omega_rad_s: np.ndarray) -> np.ndarray:
        ...   # (len(omega), 3, 3) complex R_ij(omega) from the full toggling machinery; the closed
              # (-1)^{n+1} form is used only when is_single_axis() and inner is None
    def filter_function(self, omega_rad_s: np.ndarray,
                        quadrature: Literal["dephasing", "amplitude", "universal"] = "dephasing"
                        ) -> np.ndarray: ...
    def rounded_to_clock(self, clock_s: float) -> "DecouplingSequence": ...
        # snap pulse centres to a grid while preserving A_1; reports the residual moment error

def decoupling_sequence(timing: str, n_pulses: int, tau_s: float, tau_pi_s: float, *,
                        inner: CompositePulse | None = None) -> DecouplingSequence: ...
def filter_function(device: Device, control: "CompositePulse | DecouplingSequence | Schedule",
                    *, quadrature: Literal["dephasing", "amplitude", "universal"] = "dephasing",
                    omega_rad_s: np.ndarray | None = None,
                    spectrum: "NoiseSpectrum | None" = None,
                    omega_min_rad_s: float | None = None,
                    dc_floor: bool = True,
                    monte_carlo_samples: int = 0) -> ExperimentResult: ...
    # Returns F(omega) on the grid, the first-order infidelity 1 - F_av = (1/pi) int dw/w^2 S F,
    # chi = 2 x that, W = exp(-chi), the fitted low-frequency roll-off order alpha, the dc floor,
    # and the reported max[FF, dc] (Section 6.9). spectrum=None uses device.noise.S_B converted
    # through the computed Zeeman sensitivities to S_b (the PSD of the sigma_z coefficient, HALF the
    # splitting fluctuation) for the dephasing quadrature and device.noise.rabi_amplitude for the
    # amplitude one. omega_min_rad_s defaults to 2 pi/(total experiment duration), in rad/s as its name says (the second
    # revision wrote 1/T for a rad/s field), and is ALWAYS reported together with the local sensitivity d ln chi/d ln omega_min,
    # because chi is IR-divergent for free induction, the Hahn echo and every odd-n sequence on a 1/omega^4 spectrum and is then
    # a function of the declared cutoff: on the T = 1 free-induction fixture chi(1/T)/chi(2 pi/T) = 1.06e4 (check_critique_v3.py;
    # the (2 pi)^3 = 248 of the low-frequency estimate holds only for omega_min T << 1). monte_carlo_samples > 0 additionally
    # runs the sampled-trajectory path of Section 6.1(d) through the Section 4.3.1 Hamiltonian and returns both, so the two
    # must agree inside xi^2 << 1; the result records xi^2 and flags the comparison when it is not small. (The second revision
    # pasted three NoiseModel field declarations here; they live on NoiseModel above, where rabi_amplitude is declared.)

# ---- Run 5: Detection dark states, metastable lifetimes, laser scattering, polarization-gradient cooling (Sections 8.1, 4.5.5, 4.5.7, 4.2.4) ----
# ---- species and fields (Section 4.5) --------------------------------------------------------

@dataclass(frozen=True)
class MetastableChannels:           # Section 4.5.7, all default off: physical inputs, not fudge factors
    bbr_temperature_k: float | None = None       # None disables blackbody D-D mixing entirely
    pressure_mbar: float | None = None           # background-gas total pressure; None disables collisions
    gas_fractions: dict[str, float] = field(default_factory=dict)   # {"H2": 0.5, "N2": 0.5}, must sum to 1
    reshelving_rate_hz: float = 0.0              # R of p_D_dot = -Gamma p_D + R(1 - p_D); exposed, never fitted away
    # derived, each carrying provenance:
    def bbr_rate_hz(self, transition: str) -> tuple[float, float]: ...
        # (downward A*n_bar, upward (g_u/g_l)*A*n_bar); asserts n_bar = 1/(exp(h nu/kT) - 1) is MULTIPLIED
    def collision_rates_hz(self, species: Species) -> dict[str, float]: ...
        # {"quench": R_q, "j_mix": R_j} from R = sum_s Gamma_s p_s/(k_B T); Gamma_s in cm^3/s, never a rate
    def reshelving_offset(self, tau_s: float) -> float: ...          # R/(Gamma + R), the fitted-offset signature

@dataclass(frozen=True)
class DarkStateReport:              # Section 8.1, returned by the CPT solve so the ceiling is never assumed
    n_ground: int; n_excited: int
    ceiling: float                  # n_e/(n_e + n_g); the photon rate must satisfy Gamma*P_f <= Gamma*ceiling
    dark_dimension: int             # dim ker M for the configured polarization and (J_i, J_f)
    dark_basis: np.ndarray          # (dark_dimension, n_ground) complex amplitudes c_m, straight pairing
    delta_over_omega: float         # dark-state evolution rate / rms Rabi frequency; optimum ~ 1/2
    theta_be_deg: float             # angle between the linear polarization and B; optimum 54.7356
    raman_zero_margin_hz: float | None   # distance to the nearest species dark resonance, None if unmapped

@dataclass(frozen=True)
class PolGradientBeams:             # Section 4.2.4; a lin-perp-lin pair, not two independent Beams
    beam_a: Beam; beam_b: Beam      # counter-propagating, orthogonal linear polarizations
    detuning_hz: float              # must be > 0 (blue) on an inverted j_e <= j_g line; asserted, not warned
    beat_hz: float = 0.0            # imposed inter-beam frequency difference; drives phi(t) = 2 pi beat t
    phase_rad: float = 0.0          # phi at the trap centre; 0 = linear light at the ion, +-pi/4 = circular
    level_scheme: Literal["jg12_je12", "F1_to_F0"] = "jg12_je12"   # selects which model may be used at all
    def xi(self, mode_freq_hz: float, s_single_beam: float) -> float: ...    # Delta*s/(3*omega), angular
    def limits(self, mode_freq_hz: float, s_single_beam: float) -> tuple[float, float]: ...
        # (fixed-phase <n_0>, phase-averaged <n>); raises for level_scheme != "jg12_je12"
    def moving_gradient_ok(self, cooling_rate_hz: float, mode_freq_hz: float) -> bool: ...  # W < delta < omega

# ---- Run 5: Transport, splitting, merging and junctions (Section 4.6, milestone M12) ----
# ---- transport, splitting and junctions (Section 4.6, milestone M12) -----------------------
@dataclass(frozen=True)
class Zone:                     # one named trapping site or junction leg of a multi-zone array
    name: str                                       # "E", "C", "load", "gate1", ...
    position_m: tuple[float, float, float]          # nominal well centre
    kind: Literal["site", "junction_centre", "junction_apex", "channel"]
    omega_hz: tuple[float, float, float] | None     # nominal secular triple when solved for
    # junction geometry, all None outside a junction:
    barrier_ev: float | None                        # q*phi_ps at the apex, in eV (peak convention)
    axial_anticonfinement: bool                     # True where omega_rf,z is imaginary (apex)
    sum_rule_holds: bool                            # assert omega_x^2+omega_y^2+omega_z^2 = 2 omega_rf^2
    #                                                 only where True; False everywhere in a junction
    micromotion_z1_m: float | None                  # residual axial amplitude
    micromotion_convention: Literal["peak", "rms"]  # declared, never inferred (Section 13)
    shim_split_hz: float | None                     # |omega_x - omega_z| lifted by the shim at a
    #                                                 junction centre; None means left degenerate
    citations: tuple[str, ...]

@dataclass(frozen=True)
class VoltageWaveform:                 # NOTE: distinct from the existing entangling-pulse `VoltageWaveform`;
    #                             M12 renames that one `GateWaveform` and keeps this name for
    #                             voltage waveforms, or this becomes `VoltageWaveform`. Author's call.
    kind: Literal["transport", "split", "merge", "swap", "junction_traverse"]
    shape: Literal["sine", "erf", "blackman", "bezier", "linear", "sin2_distance", "quintic_distance"]
    duration_s: float
    params: dict[str, float]                        # {"t_p_s": ...} for erf; {"b_1": ...} Bezier points
    dac_step_s: float                               # zero-order hold; the J*R_DAC = f_z knob
    filter_chain: tuple["FilterStage", ...]         # FIR then analog low-pass, applied in order
    # commanded schedules, before the filter chain:
    voltages_v: dict[str, np.ndarray] | None        # per electrode name, sampled at dac_step_s
    well_center_m: Callable[[float], float] | np.ndarray | None       # x_well(t), transport only
    omega_axial_hz: Callable[[float], float] | np.ndarray | None      # omega_x(t), transport only
    distance_m: Callable[[float], float] | np.ndarray | None          # d(t), split and merge only
    quartic_v_per_m4: float | None                  # beta, split and merge only
    quadratic_v_per_m2: Callable[[float], float] | None               # alpha(t), split and merge
    tilt_v_per_m: float                             # gamma, including the stray offset gamma_prime
    def delivered(self) -> "VoltageWaveform": ...          # the same object after the filter chain; x_well
    #                                                 and omega are always derived from this, never
    #                                                 from the commanded staircase (Section 4.6)
    def excitation(self, species: Species, mode_hz: float) -> "TransportBudget": ...

@dataclass(frozen=True)
class FilterStage:
    kind: Literal["zoh", "fir", "butterworth", "single_pole"]
    order: int | None; corner_hz: float | None
    taps: np.ndarray | None
    def response(self, f_hz: np.ndarray) -> np.ndarray: ...           # complex, magnitude and phase
    def group_delay_s(self) -> float: ...

@dataclass(frozen=True)
class TransportBudget:          # what a transport, split or merge costs, per mode
    alpha: complex                                  # coherent amplitude in the co-moving frame,
    #                                                 acceleration-form phase (Section 13)
    n_coherent: float                               # |alpha|^2, the Husimi-Kerner gamma
    n_thermal: float                                # integrated Gamma_h(omega(t)) dt, linear in T
    squeeze: complex                                # xi(delta, theta); zero only at constant omega
    reference_omega_hz: float                       # the frequency n_coherent is referenced to;
    #                                                 required, since omega_f and omega_CP differ by 10x
    ermakov_state: tuple[float, float, float]       # (rho, rho_dot, mu) carried across segments
    per_ion: tuple[float, ...] | None               # asymmetric split: which qubit got the kick
    heralded_failure: bool                          # transport failure or reordering (erasure)
    provenance_id: str

@dataclass(frozen=True)
class Transport:                # one scheduled transport-family operation
    waveform: VoltageWaveform
    from_zone: str; to_zone: str
    ions: tuple[int, ...]
    hold_s: float; hold_offset_s: float             # h, quantized to the DAC step
    reverse_of: "Transport | None"                  # exact time reversal; combine is the reverse of split
    overhead_factor: float = 1.10                   # interpolation and electronics (Section 4.6)
    def budget(self, device: "Device") -> dict[int, TransportBudget]: ...   # per mode index
@dataclass(frozen=True)
class Trap:
    ...                                             # unchanged fields
    dc_schedule: dict[str, np.ndarray] | None = None    # M12: V_n(t) per electrode, sampled
    basis_potentials: dict[str, Callable] | None = None # M12: phi_tilde_n(r) per electrode, plus
    #                                                     "rf" for phi_tilde_rf (Section 4.1.6)
    def pseudopotential_v(self, r_m: np.ndarray, species: Species) -> float: ...  # M12: phi_ps in VOLTS
    def split_coefficients(self, t_s: float) -> tuple[float, float, float]: ...   # M12: (alpha, beta, gamma)

@dataclass(frozen=True)
class Device:
    ...                                             # unchanged fields
    zones: tuple[Zone, ...] = ()                    # M12: empty for the single-zone first release
def transport_budget(device: Device, transport: Transport, *, level: str = "analytic") -> dict[int, TransportBudget]: ...
    # level="analytic" runs the Ermakov fast path of M12.2; level="exact" runs the QobjEvo solve of M12.3
def design_waveform(device: Device, kind: str, from_zone: str, to_zone: str, *, duration_s: float,
                    target_quanta: float = 1.0, shape: str = "erf") -> VoltageWaveform: ...
    # inverts the Fourier criterion (transport) or minimizes n_coh(T) + Delta n_th(T) (split)
def split_feasible(device: Device, waveform: VoltageWaveform) -> tuple[bool, str]: ...
    # False with a reason when |gamma| >= gamma_tilde = 1.06 (kappa^3 beta_CP^2)^{1/5}

@dataclass(frozen=True)
class Schedule:
    ...                                             # unchanged fields
    transports: tuple[Transport, ...] = ()          # M12: interleaved with `pulses` by absolute time
```

- `tau_convention` is not decoration. The same physical pulse has τ_field = 2 τ_intensity in the two families the sources use, which is the entire origin of the 1 − x²/24 against 1 − x²/6 quadratic coefficients of the two closed forms; `tau_s` is stored in whichever convention the tag names and converted once, and a measured FWHM is never assigned to it (0.8384014366 τ for a sech intensity envelope, 0.5611 τ for sech²).
- `carrier_order` is a signed tooth-index *difference*, not an absolute optical index of order 10⁶, which is why the carrier-envelope offset drops out and why the noise gain on the beat note is exactly this integer.
- `aom_offset_hz` is signed and is the primary control input: its sign selects the red or the blue sideband. `solve_offset` root-finds |j ν_rep + Δν_M| against a carrier or sideband target over signed integer j rather than hard-coding a tooth index, and it is the same call that builds the feed-forward chain, so substituting ν_rep → ν_rep + δ_r(t) is what the code actually does and the drift cancellation is emergent.
- `pair_order_max` defaults high because the fourth-order Stark sum's weight is centred |j| teeth away from resonance and its partial sums plateau falsely a factor 2.09 low near |k| ~ 10; `stark4_hz` sums the reindexed single sum over the tooth separation l, guards ω_a ≠ l(2πν_rep) and never guards j ≠ 0, and asserts stability under doubling.
- `chain` and `i_sat_w_m2` travel together: Chain I with the plan's πhcΓ/(3λ³), or the source's lumped 0.15 W/cm² with its own g definition. A constructor assertion refuses the cross-product, because combining them counts the D1 line strength twice.
- `guards` returns the named booleans of the validity hierarchy (`teeth_resolved`, `bandwidth_straddles_qubit`, `adiabatic_elimination`, `small_pulse_area`, `sum_rule`, `sideband_resolved`, `lamb_dicke`) and the engine logs them; when the sideband guards fail it routes to the pulse-by-pulse propagator instead of interpolating.
- The kick backend needs one more scalar, which lives on `Tone` rather than on `CombSpec` because it is per pulse: `theta_bessel_rad`, **defined** as the Bessel argument of exp[iΘ_B sin(Δk x̂ + φ)σ_x], with the convention that a perfect spin-dependent kick sits at Σ_k Θ_{B,k} = π and that the finite-duration correction Θ_B → Θ_B sech(ω_q τ/2) is applied once, before assembly.
- `Transition` gains `quadrupole_element_au: float | None` and `quadrupole_convention: Literal["johnson_1_15", "racah_c2"]`, so a reduced quadrupole element can never be fed into a prefactor from the other normalization (the two differ by a factor 5 in the rate); and `lifetime_s` on a metastable `Level` gains a sibling `lifetime_systematics_s: tuple[tuple[str, float], ...]` carrying one-sided signed corrections by name, since the sources never combine them in quadrature with the statistical error.
- `Beam` gains `polarization_amplitudes: tuple[complex, complex, complex] | None` in the (σ⁻, π, σ⁺) spherical basis with an assertion that they are unit-normalized, because the Wineland reductions require it and one source prints no normalization at all; and `modulation: PolarizationModulation | None` for the AOM, PEM and EOM destabilization schemes, whose presence makes `steadystate` illegal and forces the propagate-then-period-average path.
- `NoiseModel.channels` returns the Rayleigh operator as `0.5 * sqrt(Gamma_el) * sigma_z` with a docstring stating that the dissipator prefactor is Γ_el/4 while the observable coherence decays at Γ_el/2, and that both are correct; the recoil-resolved emission channels are returned as one operator per (decay channel, recoil class) pair, never summed.
- The M12 voltage record is named `VoltageWaveform`; the calibrated entangling-pulse record of Sections 7 and 8 keeps the name `Waveform`, so the two never collide.

**2026-09 additions (0.2.0; docs/api_proposal.md and docs/api_implementation_plan.md Phase 1).** The ladder: one executor, five rung modules, three option objects, the builder on `Circuit`, the versioned `Result` record and the IonQ exporters named by the convention they emit. The declarations live in the rung modules named in the comments rather than in `qutip_trap.api`, which is unchanged; `tests/test_api_freeze.py` resolves them there. The rules above hold (frozen records, Hz in the public API, provenance on every derived number), with one exception named below.

```python
# ---- qutip_trap.device.model (rung 4: qutip_trap.physics) ----
@dataclass(frozen=True)
class BeamRoles:                # which beams play which part; None = infer from the wavelengths and the beam count
    gate: "Mapping[int, GateDrive] | None" = None          # per ion, the single-qubit gate drive
    entangling: "Mapping[int, GateDrive] | None" = None    # per ion, the entangling drive; {} declares none
    detection: int | None = None                           # the detection beam's index
    def resolve(self, device: "Device") -> "ResolvedRoles": ...   # the one resolution rule: explicit keywords, then the roles, then the inference

@dataclass(frozen=True)
class ResolvedRoles:
    gate: dict[int, GateDrive]; entangling: dict[int, GateDrive]; detection: int | None
    inferred: tuple[str, ...] = ()                         # the fields the wavelength rules filled in

@dataclass(frozen=True)
class Device:
    ...                                                    # unchanged fields
    roles: BeamRoles = BeamRoles()                         # left out of hash(): the apparatus is the identity, Machine.hash() carries the roles

# ---- qutip_trap.run.levels (rung 0; FidelityLevel, LevelDecision and decide_level are exported by qutip_trap) ----
class FidelityLevel(StrEnum):   # the members equal the strings the code accepted before: "auto", "JOINT_EXACT", "GATE_LOCAL"
    AUTO = "auto"; JOINT_EXACT = "JOINT_EXACT"; GATE_LOCAL = "GATE_LOCAL"

@dataclass(frozen=True)
class LevelDecision:            # why a run integrates at its level (Section 11.5)
    level: FidelityLevel; dimension: int; nnz: int; joint_dimension_max: int; nnz_max: int; estimated: bool

def decide_level(device: Device, circuit: Circuit, options: SolverOptions, *, space: HilbertSpace | None = None) -> LevelDecision: ...

@dataclass(frozen=True)
class Diagnostics:
    ...                                                    # unchanged fields
    level_reason: str = ""                                 # LevelDecision.reason, or the level the caller forced and what auto would choose

# ---- qutip_trap.options (rung 0; the groups also from qutip_trap.dynamics) ----
@dataclass(frozen=True)
class Integration:
    atol: float = 1e-10; rtol: float = 1e-8; nsteps: int = 10**7; integrators: tuple[str, ...] = ("dop853", "vern9")
    rotating_frame: bool = True; propagator_cache: bool = True
@dataclass(frozen=True)
class Truncation:
    joint_dimension_max: int = 4096; nnz_max: int = 2 * 10**7; mode_dimension_max: int = 64
    boundary_population_max: float = 1e-6; freeze_chi_max_rad: float = 0.05; freeze_alpha_max: float = 1e-4
    branch_weight_min: float = 1e-6; margin_check: bool = True; margin_element_tol: float | None = None
    caps: "Mapping[int, int] | None" = None; enr_group: "tuple[Sequence[int], int] | None" = None; space: "HilbertSpace | None" = None
@dataclass(frozen=True)
class Trajectories:
    lindblad_method: Literal["auto", "mesolve", "mcsolve"] = "auto"; mesolve_dimension_max: int = 128; ntraj: int = 64
    improved_sampling: bool = True; trajectory_target_tol: float | None = None; e_ops_for_target_tol: bool = True
@dataclass(frozen=True)
class GateLocal:
    map_accuracy: float = 1e-3; crosstalk_threshold: float = 1e-3; register_dm_max_qubits: int = 12; register_ensemble: int = 64
    tomography_isometry: bool = True; tomography_dropped_weight_max: float | None = None; tomography_tolerance_keyed: bool = True
@dataclass(frozen=True)
class Parallel:
    map: Literal["serial", "parallel", "loky"] = "parallel"; workers: int | None = None; samples: int | None = None
    addressing: bool | None = None                         # single-qubit gates in parallel; None = the device's hardware says
@dataclass(frozen=True)
class Numerics:                 # how the integration is done, nested by concern; SolverOptions is what it builds internally
    integration: Integration = Integration(); truncation: Truncation = Truncation(); trajectories: Trajectories = Trajectories()
    gate_local: GateLocal = GateLocal(); parallel: Parallel = Parallel(); convergence_check: bool = False
    def to_solver_options(self, physics: "Physics | None" = None) -> SolverOptions: ...
@dataclass(frozen=True)
class Physics:                  # which effects are simulated; every default is what run did in 0.1.0
    noise: bool = True; internal_levels: int = 2; scattering: Literal["estimate", "channels"] = "estimate"
    scattering_recoil: Literal["off", "minimal", "vector"] = "minimal"; intensity_noise_channels: bool = True; hardware_chain: bool = True
    stark_compensation: bool = True; crosstalk_suppression: Literal["none", "neighbour", "local"] = "none"; entangler: Literal["ms", "zz"] = "ms"
    extra_channels: tuple[CollapseOp, ...] = (); builder: "BuilderOptions | None" = None; t0_s: float = 0.0; shot_period_s: float | None = None
@dataclass(frozen=True)
class Readout:                  # how the photon record is read
    mode: Literal["fast", "full"] = "fast"; discriminator: "Discriminator | None" = None; povm_samples: int = 20_000

# ---- qutip_trap.machine (rung 0) ----
@dataclass(frozen=True)
class Machine:                  # the executor: device + roles + calibration + policy; variants by dataclasses.replace
    device: Device; table: CalibrationTable | None = None
    physics: Physics = Physics(); numerics: Numerics = Numerics(); readout: Readout = Readout()
    level: FidelityLevel = FidelityLevel.AUTO; name: str = ""
    def run(self, circuit: Circuit, shots: int, *, seed: int = 0, keep_final_state: bool = False, progress: "Callable | None" = None) -> Result: ...
    def compile(self, circuit: Circuit) -> CompileReport: ...
    def schedule(self, circuit: Circuit, *, seed: int = 0) -> Schedule: ...         # compile + calibrate + schedule, nothing integrated
    def calibrated(self, method: Literal["closed_form", "experiments"] = "closed_form", *, seed: int = 0, **scans) -> "Machine": ...
    def estimate(self, circuit: Circuit, *, seed: int = 0) -> "Estimate": ...       # the level, the space and a wall-time guess
    def hash(self) -> str: ...                                                        # device hash + roles + table digest + policy
    def error_model(self): ...                                                        # 0.3.0 (Phase 2.6)
    def specs(self) -> str: ...                                                       # 0.3.0 (Phase 2.5)
    def submit(self, circuit: Circuit, shots: int, *, seed: int = 0): ...             # 0.4.0 (Phase 3.1)

@dataclass(frozen=True)
class Estimate:
    level: FidelityLevel; reason: str; space: HilbertSpace; mode_class: dict[int, Literal["resolved", "frozen", "dropped", "enr"]]
    dimension: int; nnz: int
    n_pulses: int; n_entangling: int; duration_s: float; wall_time_s: float; notes: tuple[str, ...] = ()

# ---- qutip_trap.control.compiler (rung 1: qutip_trap.circuit) ----
@dataclass(frozen=True)
class Circuit:                  # a persistent builder: one method per name of NATIVE_GATES and STANDARD_GATES, each returning a new Circuit
    n_qubits: int; ops: tuple[Operation, ...] = (); measure: tuple[int, ...] = None   # omitted = every qubit
    registers: dict[str, tuple[int, ...]] = None                                     # omitted = {"c": measure}
    def h(self, q: int) -> "Circuit": ...
    def cnot(self, q0: int, q1: int) -> "Circuit": ...
    def gpi2(self, q: int, phase: float) -> "Circuit": ...
    def ms(self, q0: int, q1: int, phi0: float, phi1: float, theta: float) -> "Circuit": ...
    def measured(self, *qubits: int, registers: "Mapping[str, Sequence[int]] | None" = None) -> "Circuit": ...
    def to_openqasm(self, *, declare_native: bool = True) -> str: ...
    def to_ionq(self) -> dict: ...

# ---- qutip_trap.run.pipeline (rung 2: qutip_trap.schedule) ----
@dataclass(frozen=True)
class Prefix:                   # what the compile-calibrate-schedule prefix of run produced
    report: CompileReport; compiled: Circuit; table: CalibrationTable; device: Device; schedule: Schedule
    gate_drives: dict[int, GateDrive]; entangling_drives: dict[int, GateDrive]; options: SolverOptions; notes: tuple[str, ...]

def compile_calibrate_schedule(circuit: Circuit, device: Device, *, table: CalibrationTable | None = None, seed: int = 0,
                               t0_s: float = 0.0, options: SolverOptions | None = None) -> Prefix: ...

# ---- qutip_trap.run.results (rung 0) ----
@dataclass(frozen=True)
class Progress:                 # one step of a run's progress, handed to the progress callback
    stage: str; done: int; total: int; elapsed_s: float

@dataclass(frozen=True)
class Result:
    ...                                                    # unchanged fields; bit_order gains "qubit0_msb" for reversed_bits()
    qubits: tuple[int, ...] | None = None                  # the circuit qubit each column holds
    registers: dict[str, tuple[int, ...]] | None = None    # the circuit's classical registers
    machine_hash: str | None = None; created_at: str = ""; duration_s: float = 0.0
    def to_dict(self, *, per_shot: bool = False) -> dict: ...      # docs/schemas/result.schema.json, schema version 1
    def reversed_bits(self) -> "Result": ...                       # qubit 0 leftmost, for Cirq, Braket and PennyLane comparisons
    def to_ionq_v1_probabilities(self) -> dict: ...                # decimal keys, qubit 0 the 2^0 bit (to_ionq_json is its deprecated alias)
    def to_ionq_v1_histogram(self) -> dict: ...
    def to_ionq_v1_shots(self) -> list: ...
    def to_ionq_v2_probabilities(self) -> dict: ...                # {"probabilities": {"registers": {"output_all": ..., <register>: ...}}}, q[0] leftmost
    def to_ionq_v2_histogram(self) -> dict: ...
    def to_ionq_v2_shots(self) -> dict: ...

# ---- qutip_trap.io.ionq ----
@dataclass(frozen=True)
class IonQJob:                  # a v0.3 or v0.4 job body read back
    circuit: Circuit; backend: str | None = None; shots: int | None = None; name: str | None = None
    metadata: dict[str, str] = field(default_factory=dict); noise: dict[str, Any] | None = None; settings: dict[str, Any] | None = None
    dry_run: bool | None = None; type: str = "ionq.circuit.v1"
def load_job(obj) -> IonQJob: ...
def dump_job(circuit: Circuit, *, backend: str, shots: int = 100, noise=None, settings=None, name=None, metadata=None, dry_run=None) -> dict: ...

# ---- entry point amendments ----
def run(circuit: Circuit, device: Device, shots: int, *, table: CalibrationTable | None = None, t0_s: float = 0.0,
        shot_period_s: float | None = None, samples: int | None = None,
        level: Literal["JOINT_EXACT", "GATE_LOCAL", "auto"] = "auto", seed: int = 0,
        options: SolverOptions | None = None, progress=None) -> Result: ...
        # progress: called with a Progress per pulse and per (sample, branch) run when they run in-process, per sample and per readout
```

- `qutip_trap.io.qasm2.loads(text)` and `dumps(circuit, *, declare_native=True)`, and `qutip_trap.io.ionq.loads(obj)` and `dumps(circuit)`, are the two wire formats in the shape of `json`; `Circuit.from_openqasm`, `from_ionq`, `to_openqasm` and `to_ionq` are the same four on the type.
- `Circuit.measure` and `Circuit.registers` are the declared types once constructed; the omitted argument (None) is the default.
- `Result.bit_order` is `Literal["qubit0_lsb", "qubit0_msb"]`: every run reports `qubit0_lsb`, and `reversed_bits()` is the only source of the other.
- `JointExactEngine`, exported by `qutip_trap.dynamics` as the concrete engine, is the one dataclass of the surface that is not frozen: a service object with a report, a propagator cache and a progress hook, not a record.
- The warnings are classes, not records: `qutip_trap._compat.QutipTrapWarning` is the package's base, `QutipTrapDeprecationWarning` (also a `DeprecationWarning`) marks a deprecated path, and `qutip_trap.hilbert.truncation.TruncationWarning` a cap clamped by `mode_dimension_max` or a boundary population left above `boundary_population_max` after the retries.
- Deprecated in 0.2.0 (docs/deprecations.md): `DevicePreset.run_kwargs()` (returns `{}`), `Result.to_ionq_json`, `to_ionq_histogram`, `to_ionq_shots` (aliases of the v1 names) and `Result.to_ionq_v2` (unchanged behaviour, superseded by the v2 exporters).

**2026-09 additions (0.3.0; docs/api_implementation_plan.md Phase 2).** The option objects become the declared form of ``run``, the laboratory runs on machines, and the inverse direction (the phenomenological error model) is published. As for 0.2.0, the declarations live in the rung modules named in the comments and `tests/test_api_freeze.py` resolves them there; the keyword arguments of 0.1.0 stay accepted for two minor releases, rewritten with a `QutipTrapDeprecationWarning` naming their new home (docs/deprecations.md).

```python
# ---- entry point amendments (qutip_trap.run.job; rung 0) ----
def run(circuit: Circuit, device: Device, shots: int, *, table: CalibrationTable | None = None,
        level: Literal["JOINT_EXACT", "GATE_LOCAL", "auto"] = "auto", seed: int = 0, keep_final_state: bool = False,
        progress=None, physics: "Physics | None" = None, numerics: "Numerics | None" = None,
        readout: "Readout | None" = None, **deprecated) -> Result: ...
        # Machine(device, table=table, physics=physics, numerics=numerics, readout=readout, level=level).run(circuit, shots, seed=seed, ...):
        # the pipeline is run.pipeline.execute on the machine. ``deprecated`` accepts the 0.1.0 keywords (run.job.LEGACY_RUN_KEYWORDS:
        # t0_s, shot_period_s, samples, options, gate_drives, entangling_drives, builder_options, readout as a string, discriminator,
        # povm_samples, space, caps, channels, entangler, parallel, calibrate_kwargs, noise, internal_levels, crosstalk_suppression,
        # stark_compensation, enr_group), each rewritten onto the machine with a warning; removable in v0.5 at the earliest.
        # The pipeline itself is qutip_trap.run.pipeline.execute(machine, circuit, shots, *, seed, keep_final_state, progress),
        # the implementation of Machine.run (not a public name: the method is the API)
```

```python
# ---- the laboratory on machines (qutip_trap.experiments; docs/api_implementation_plan.md 2.2, 2.3) ----
@dataclass(frozen=True)
class ScanParameters:           # a scan's parameters by name, one tuple each; attribute access (scan.requested.durations_s)
    values: "Mapping[str, Any]" = field(default_factory=dict)

@dataclass(frozen=True)
class ExperimentResult:
    ...                                                    # unchanged fields
    requested: "ScanParameters | None" = None              # the scan as asked for (the scanned axis and the settings held)
    realized: "ScanParameters | None" = None               # the scan the hardware chain plays (realized_drive); None where no drive axis
    chi2: float | None = None                              # the principal fit's reduced chi-square; None without a weighted fit
    subject: dict[str, Any] = field(default_factory=dict)  # {"ion": 0, "beam": 2} | {"pair": (0, 1)} | {"mode": 3}, as the table keys it
    created_at: str = field(default_factory=str)           # ISO 8601 UTC; not part of equality
    def plot(self, ax=None): ...                           # needs the plot extra (matplotlib)

class RabiScan(ExperimentResult): ...                      # f_rabi_hz, nbar, contrast, offset
class RamseyFringe(ExperimentResult): ...                  # delta_hz, contrast, phi0_rad, offset; qubit_freq_hz, qubit_offset_hz (ramsey_frequency)
class SidebandSpectrum(ExperimentResult): ...              # carrier_hz, blue_sideband_hz, red_sideband_hz, mode_hz, eta, nbar, omega_bsb_hz
class ThermometryResult(ExperimentResult): ...             # nbar, ratio, duration_s
class HeatingRateFit(ExperimentResult): ...                # ndot_per_s, nbar0
class MSScan(ExperimentResult): ...                        # closure_offset_hz, closure_scale, chi_unit_rad, leakage_min; correction_rad(ion)
class ParityScan(ExperimentResult): ...                    # contrast, phi0_rad, bell_fidelity_bound, P00, P11
class DetectionHistogram(ExperimentResult): ...            # threshold, window_s, eps_B, eps_D, errors
class StarkScan(ExperimentResult): ...                     # stark_shift_hz, coupling_shift_hz; shift_of_beam(beam)
class CrosstalkScan(ExperimentResult): ...                 # rate_hz; eps(j), rate_of(j), phase_rad(j), neighbours
class FieldScan(ExperimentResult): ...                     # B_gauss, qubit_freq_hz, qubit_offset_hz
class MicromotionScan(ExperimentResult): ...               # shims_v, shim_v(name), beta_before
class CrystalImage(ExperimentResult): ...                  # n_ions, n_bright, n_dark, n_lost
        # every typed attribute equals value(key) where the fit produced the key and is None where it did not; quality is
        # "good" | "poor" | "failed" | "exact" (the acceptance test before CalibrationTable.updated_with)

def rabi_scan(machine, ion, durations_s, **kw) -> RabiScan: ...       # machine: Machine | Device (a Device is wrapped in a default Machine)
def ramsey(machine, ion, delays_s, **kw) -> RamseyFringe: ...
def ramsey_frequency(machine, ion, delays_s, **kw) -> RamseyFringe: ...
def micromotion_scan(machine, ion, beam, shim_ranges_v, method="rf_photon_correlation", **kw) -> MicromotionScan: ...
def sideband_spectroscopy(machine, ion, detunings_hz, **kw) -> SidebandSpectrum: ...
def ms_scan(machine, pair, amplitudes, detunings_hz, **kw) -> MSScan: ...
def parity_scan(machine, pair, analysis_phases_rad, **kw) -> ParityScan: ...
def heating_rate(machine, mode, delays_s, **kw) -> HeatingRateFit: ...
def detection_histogram(machine, ion, n_records, **kw) -> DetectionHistogram: ...
        # the machine supplies table, options and builder_options as the defaults of kw (a Device supplies nothing, the 0.1.0
        # behaviour); gate_drive=, gate_drives= and entangling_drives= in kw are deprecated: the device's roles name the drives (v0.5)
def as_machine(machine) -> "Machine": ...                  # qutip_trap: a Device wrapped in a default Machine, else the machine itself

# ---- qutip_trap.calibration ----
def calibrate(machine, *, method: Literal["closed_form", "experiments"] = "closed_form", experiments: tuple[str, ...] = ("all",),
              seed: int = 0, t0_s: float | None = None, surrogate: bool | None = None, **scans) -> "CalibrationReport": ...
        # on a Machine: the CalibrationReport on every call (its .table is what the scheduler reads; t0_s defaults to the machine's
        # Physics.t0_s); on a bare Device: the 0.1.0 result, the CalibrationTable, with surrogate=True/False deprecated in favour of
        # method= (warns from 0.4.0). calibrate_with_report and compile_with_report are deprecated (Machine.compile returns the report).

# ---- qutip_trap.benchmarks ----
def randomized_benchmarking(machine, qubits, lengths, *, n_sequences: int = 4, shots: int = 200, seed: int = 0, budget: bool = True,
                            fix_offset: bool | None = None, variant: str = "clifford", pair: bool | None = None) -> "RBResult": ...
def ghz_fidelity(machine, qubits, *, shots: int = 400, analysis_phases_rad=None, seed: int = 0, budget: bool = True) -> "GHZResult": ...
def quantum_volume(machine, qubits, *, n_circuits: int = 4, shots: int = 200, depth: int | None = None, seed: int = 0,
                   budget: bool = True) -> "QVResult": ...
def gate_channel(machine, kind: str) -> "GateChannel": ...  # cached per Machine.hash() and kind
        # the 0.1.0 **run_kwargs of the benchmarks (table, options, level, noise, ...) are still accepted, each rewritten onto the
        # machine with a warning (v0.5)
```

Three rules bind the surface. The application of Section 14 imports nothing outside this appendix. `control` never imports `calibration`, which sits above it and communicates through `CalibrationTable`. Every derived number reachable through `Device.derived()`, `Result.diagnostics` or an `ExperimentResult` carries a provenance id from the ledger of Section 14.5, so the Section 9.11 coverage test is a set difference.
