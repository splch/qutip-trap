# Check and benchmark scripts behind the `[recomputed here]` numbers in PLAN.md

Run each with `uv run --python 3.13 --with qutip --with numpy --with scipy --with sympy python <script>.py`
(sympy is needed by `check_atomic.py` only). `outputs/` holds the runs of 2026-09-04 on the pinned
Python 3.13.14 with QuTiP 5.3.1, NumPy 2.5.2 and SciPy 1.18.1 (`outputs/versions.txt`); the same scripts
gave identical outputs under Python 3.14 (wall times within 10%).

- `bench_numerics.py`: displacement-operator constructions (expm vs analytic Laguerre elements, ENR),
  right-hand-side evaluation counts in the motion-Schroedinger and interaction pictures (PLAN Sections 5.1.1, 5.2).
- `bench_ms_timing.py`: wall time of a two-ion Moelmer-Soerensen pulse at joint dimensions 48 to 2048 (Section 11.1).
- `check_atomic.py`: Breit-Rabi against numerical diagonalization, clock points of 43Ca+, 9Be+, 25Mg+, the 171Yb+
  quadratic Zeeman coefficient, Wigner-Eckart branching, Gamma to reduced element to I_sat (Sections 4.5, 9.13).
- `check_surface_mixed.py`: gapless-plane electrode potential (Laplace residual, boundary values), five-wire null
  height, mixed-species two-ion modes (Sections 4.1.6, 4.1.7).
- `check_recoil.py`: recoil angular factors by quadrature and 171Yb+ recoil-per-photon numbers (Section 4.2.8).
- `check_recoil2.py`: the recoil follow-up (Run 4R) checks: Gauss-Legendre quadrature identities, the single-kick
  expectation, the alpha -> 0 sideband floor, the Doppler optimum, Cirac's Fig. 8 minima, the Franck-Condon Laguerre
  index, the 40Ca+ Lamb-Dicke parameters, effective two-level ratios and the recoil unit regressions (Sections 4.2.8, 9.3).
- `check_home_table.py`: Home 2013 Table I (Be+/Mg+ pair): the caption's 24Mg+ x, y frequencies are transposed; only
  the corrected order reproduces the printed off-species amplitudes 0.018 (x) and 0.020 (y) (Sections 4.1.7, 9.13).
- `bench_ms_timing.py` (v2, 2026-09-04): closure eta*Omega/eps = 1/4 (S = sum sigma convention) and a realizable
  two-171Yb+ fixture (x-COM 3.000, x-rock 2.828, y-COM 2.900 MHz; eta 0.080, +-0.0824, 0.047). `bench_ms_timing_v1.py`
  and `outputs/bench_ms_timing_v1.out` keep the first version (eta*Omega/eps = 1/2, eta 0.08/0.06/0.05) for comparison.
- `check_anharmonic.py`: phase accumulated by an off-resonant cubic mode coupling g (a+a^dag)^2 (b+b^dag) at
  g/2pi = 1.419 kHz over 100 us versus the first-order g t and the second-order g^2 t/Delta (Sections 4.1.4, 9.12).
- `check_collisions.py`: Langevin collision rate per ion for H2 at 1e-11 torr (Section 6.7).
- `check_atomic.py` now uses the partial 171Yb+ S1/2-P1/2 rate 19.62 MHz (19.72 MHz total x 0.99499); the earlier run
  at 19.6 MHz gave 1.751 e a0 and 50.77 mW/cm^2 against 1.752 and 50.83 now.

Added on 2026-09-04 (evening) by the Run 5 revision:

- `check_ms_closure.py`: integrates the exact bichromatic Moelmer-Soerensen Hamiltonian per tone in the plan's
  (hbar Omega/2) sigma_+ convention and shows the maximally entangling closure is eta*Omega/eps = 1/2 (concurrence
  0.9999), while 1/4 is a chi = pi/16 pulse (0.38); the effective -(hbar eta Omega/2) S_y form reproduces both
  (Sections 4.4.1, 9.4, 13; derivation audit).
- `check_c0_floquet.py`: the exact Floquet function of the Mathieu oscillator under the canonical Wronskian
  normalization Im(u* u') = nu gives the micromotion factor on eta C0 = 1 + 3q^2/16 + O(q^4) (1.001890, 1.007741,
  1.018161 at q = 0.1, 0.2, 0.3), not (1+q/2)^-1, whose u(0) = 1 normalization depends on the rf phase (Sections 4.1.1,
  9.10, 13; derivation audit).
- `bench_ms_timing_v3.py`: the Section 11.1 timing benchmark at the corrected closure eta*Omega/eps = 1/2
  (`outputs/bench_ms_timing_v3.out`); `bench_ms_timing.py` (v2, closure 1/4) and `bench_ms_timing_v1.py` are kept as
  controls.
- `check_sr_lifetime_quad.py`, `check_sr_lifetime_bbr.py`: the 88Sr+ D5/2 reduced quadrupole element from the modern
  lifetime 390.8(1.6) ms (13.81 a0^2 against 14.70 with the superseded 0.345 s) and the blackbody deshelving rate
  (3.7e-14 s^-1 at 300 K) (Sections 4.5.7, 9.14).
- `check_comb.py`: frequency-comb Raman drives: Dirichlet tooth widths, tooth-pair sums and the three closed forms,
  Rosen-Zener ceilings, the C_{n,a} convergence sweep, Stark-nulling roots, lock and PLL arithmetic, the Jacobi-Anger
  branch and the spin-dependent-kick algebra (Sections 4.3.7, 9.15).
- `check_composite.py`: the composite-pulse order ladder with its printed-defect negative controls (SK1 sign, the
  Suzuki recursion factor, the odd-k broadband layer, P2's pulse count), the F_C/F_K/F_avg relations, the epsilon-free
  order certificate, the toggling-frame control matrix and corrected Green Eq. 40, the amplitude filter function with
  its dc polygon and crossovers, the dc frozen-noise floor, the end-to-end dephasing normalization and the CP/UDD
  suppression orders with their finite-pulse collapse (Sections 4.3.5, 6.9, 9.15); mpmath at 60 to 400 digits, since
  double precision returns spurious slopes for the suppression-order tests. Runs about four minutes.
- `check_constants.py`: Berkeland-Boshier dark-state ceilings, optima and the 3j column-ordering identity; Kreuter's
  E2 prefactor and the 40Ca+ D lifetimes end to end; blackbody M1 mixing; collision and reshelving rates; the Uys
  three-operator dissipator in QuTiP; Wineland's detuning optimum and Table 1; the 171Yb+ g_F and Breit-Rabi
  curvature; the recoil second moments and the coherent-sum failure; the polarization-gradient cooling limits and the
  three-axis Lamb-Dicke triple (Sections 8.1, 4.5.5, 4.5.7, 4.2.4, 9.15); needs sympy and qutip.
- `check_transport.py`: the transport, splitting and junction numbers of Section 4.6 (critical point, final-well
  frequency, tilt threshold, transport excitation prefactors, sine- and erf-ramp suppression, heating round trips,
  Bose-Einstein conversions, junction micromotion and the DAC resonance ladder) (Sections 4.6, 9.15).
- `check_critique_v3.py` (2026-09-04, night): the recomputations behind the fold of the four-lens critique (PLAN.md
  Section 9.17, edit batches 19a-19e): sech envelope widths for the tooth convention (1.6768 tau field, 1.1222 tau
  intensity, with the transform pair checked numerically), I_sat with the angular partial rate (50.83 mW/cm^2 against
  8.09 unconverted), the dark-state destabilization optimum (2.33 G), Fock-tail populations of the GHZ fixtures, the
  zigzag pair alpha_crit vs (omega_r/omega_z)_crit, composite-pulse durations, the Monroe 1995 Lamb-Dicke anchor
  geometry, the Morigi EIT fixture in the plan's sign, the Floquet function's sign under the adopted Mathieu convention,
  the 171Yb+ quadratic Zeeman coefficient for four g_J values, the filter-function infrared cutoff, the order_slope
  round-off window, the 355 nm second-order weights, the d_m = 8 row of the Section 5.1.1 table, and QuTiP 5.3.1 probes
  (Dense expm, CSR tensor construction, ENR dims/ptrace, mcsolve target_tol, steadystate on a singular kernel).
  Output `outputs/check_critique_v3.out`.
- `bench_ms_timing_v4.py` (2026-09-04, night): the Section 11.1 benchmark rebuilt after the numerics critique showed
  that v3 timed a Dense operator (joint-generator expm) with an nnz probe that printed -1. v4 builds the drive operator
  as a tensor product of per-mode exponentials (CSR, nnz checked against 2^(N-1) prod d_m^2), keeps the dense build as
  a comparison, runs each with one shared coefficient (QobjEvo merges the four drive terms into one operator) and with
  four distinct coefficients (v3's structure), counts right-hand-side evaluations, and compares final states across
  integrators and constructions. `outputs/bench_ms_timing_v4.out`; `outputs/bench_ms_timing_v3_idle_rerun.out` is the
  unchanged v3 script re-run on the idle machine (reproduces v3's numbers, so CPU contention was not their cause).
- `check_composite.py` gained section 16 (durations Sigma theta_l against the closed forms of Section 4.3.5, asserting
  the concatenations at 8 pi + theta - 4k).

## Re-running everything (milestone M0)

`run_checks.py` re-executes every `check_*.py` under the project interpreter (`uv run python
validation/scripts/run_checks.py`), matches each numeric line of the committed output to the fresh output by
its non-numeric skeleton and compares the numbers to a relative tolerance of 1e-9, except for the residual class
(numbers below 1e-6 in magnitude printed with at most three significant digits: leaked populations, norm losses,
element differences at the round-off level), whose last digits depend on the machine's BLAS and integrator and
which are therefore compared to a factor of 3, their order of magnitude (the first Linux CI run differed from the
macOS oracle by 4.06e-10 against 4.08e-10 on exactly such a number); wall times are never
compared; `bench_*.py` run only with `--bench` and are not compared). `--report` writes
`validation/report/convergence_report.{json,md}` and the fresh outputs, which CI uploads as the
`convergence-report` artifact; it is the first CI job (`.github/workflows/ci.yml`). The committed outputs of
2026-09-04 carry a few capture artifacts (a trailing `done`, an `exit 0`, a QuTiP warning line that went to
stdout) that the comparison ignores because they contain no numbers.
