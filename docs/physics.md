# Rung 4: the physics

`qutip_trap.physics` is the records a device is built from and what the package derives from them: a `Device` is physical
parameters and everything else (mode structure, Lamb-Dicke parameters, Rabi frequencies, Stark shifts, scattering rates,
heating rates, pulse parameters) is derived, each number with the id of its provenance record in
`provenance/ledger.yaml`. `Machine(device)` is the way back up the ladder ([machine.md](machine.md)); the equations are
[physics_notes.md](physics_notes.md).

## The species

`Species` is an ion species with its cited constants: the `Level` records (energies, lifetimes, Landé factors), the
`Transition` records (frequencies, linewidths, dipole and quadrupole elements), the hyperfine and Zeeman structure
(`ZeemanSpectrum` with its `ClockPoint` entries), the `AtomicStructure` the Raman coupling chain reads (the Section 4.5
dipole anchors) and the `MetastableChannels` of a D level (Section 4.5.7: the lifetime, the blackbody and quenching rates).
`species(name)` and `available()` are the shipped tables (171Yb+, 40Ca+, 43Ca+, 88Sr+, 9Be+, 25Mg+), `species_by_name` the
0.1.0 name of the lookup, and `IncompleteSpeciesTable` what a species missing a constant raises rather than guessing it.

## The trap and the crystal

A `Trap` (Section 4.1) is the `RfDrive` (frequency and amplitude), the `DcElectrodes` (the static potential's curvatures)
and, for a surface trap, the `Electrodes` geometry with their basis potentials; `MathieuParameters` are its a and q,
`MicromotionIndex` the modulation index of the micromotion and `AnharmonicTerms` the quartic and cubic terms. The Mathieu
stability functions are `monodromy(a, q)` (the `Monodromy` matrix over one period and its trace) and `is_stable(a, q)`.
`Crystal` and `solve_crystal(trap, species)` are the equilibrium positions and the normal `Mode` structure (frequencies,
eigenvectors, Lamb-Dicke parameters) of a chain, with `ZigzagError` raised where the linear chain is not the ground state;
James's dimensionless forms are `equilibrium_dimensionless(n)` and `axial_modes_dimensionless(u)`. The zero-point length is
`x0_m(mass_kg, omega_rad_s)` and the Lamb-Dicke closed form `lamb_dicke_parameter(k, mass_kg, omega_rad_s)`; the analytic
matrix elements of Section 4.3.1 are `rabi_matrix_element(n_row, n_col, eta)`, `rabi_table(d, eta)` and
`debye_waller_factor(n, eta)`.

## The light, the field, the electronics

A `Beam` is wavelength, direction, polarization (a `PolarizationModulation` where it is modulated, `PolGradientBeams` for a
polarization-gradient pair), waist, power and pointing; `BeamRoles` names which beams play which part on a device and
`ResolvedRoles` is the resolved form with the inferred entries marked. `Field` is the magnetic field and `GradientField`
a field gradient for the gradient drives of Section 4.4.5. The `HardwareChain` is the control electronics of Section 7.10.
The units are the two types of Section 5.6, `Hz` public and `RadPerS` internal, with `rad_s_from_hz` and `hz_from_rad_s` the
one explicit 2π, and `Gauss` and `Tesla` for the field; `ATOMIC_MASS_KG` is the CODATA atomic mass unit.

## The noise

A `NoiseModel` (Section 6) is noise as physical processes: the electric-field spectrum behind heating, the field and laser
spectra behind dephasing, the drifts and the collisions. Since 0.3.0 every default means off (`NoiseModel()` is the quiet
model), `summary(device)` lists the channels that follow from what was set with their units, and `from_experiments(results, device=)` inverts a heating-rate fit into the field spectrum it implies. A `NoiseSpectrum` is always two-sided in angular
frequency (`white_spectrum`, `ou_spectrum`, `gaussian_spectrum`, `power_law_spectrum` build the shapes), a `Drift` a slow
random walk with its rms and correlation time, `Mains` the line pickup, `Collisions` the background-gas model whose
`CollisionEvent` and `collision_rate_per_ion` are what a run heralds. `quiet_noise_model()` is the deprecated 0.1.0 helper
that returned what `NoiseModel()` now is.

## Detection and preparation

A `Detector` (Section 8.4) is the collection efficiency, the quantum efficiency, the dark counts and, for a camera, the
`CameraGeometry`; an `ApparatusPreset` bundles a published apparatus, with `MYERSON_CA40_PMT` (Myerson's 40Ca+ PMT chain)
and `CRAIN_YB171_SNSPD` (Crain's 171Yb+ SNSPD) the two shipped. A `PreparationRecipe` (Section 4.2) is Doppler cooling,
the `SidebandCoolingSpec` (the sideband orders and pulse counts) and optical pumping; `standard_recipe(device)` is the
default recipe a device without one gets. The closed forms behind the recipe are public: `doppler_force_nbar` (the Doppler
limit and the force of Section 4.2.1), `stenholm_coefficients` (the rate coefficients), and the pulsed sideband-cooling
transfer of Section 4.2.2, `thermal_distribution(nbar, d)`, `apply_pulses(...)` (the column-stochastic transfer of a pulse
schedule on the Fock populations) and `mean_occupation(p)`.

## The published models

The validation suite's closed forms are exported for the same use the app makes of them: `HartyParameters` and
`simulate_epg_sets` (Harty's randomized-benchmarking model of Section 9.2), `ms_alpha` and `ms_gamma` (the displacement and
geometric phase of a Mølmer-Sørensen pulse in closed form), `kirchmair_populations` (Kirchmair's thermal populations of
Section 9.4), `thermal_debye_waller_infidelity` with its `ThermalReference` convention and `ballance_thermal_error`
(Ballance's thermal error of a two-qubit gate).

## The device and its derived numbers

`Device` aggregates the crystal, the beams, the field, the noise model, the detector, the electronics, the preparation
recipe and the roles; it is frozen and hashed (`Device.hash()`, the key of the calibration cache and of every cached
tomography). `Device.derived()` returns the `DerivedQuantities` (every computed number with its provenance id),
`Device.specs()` renders them as a report, and `Device.to_dict()` and `Device.from_dict()` are the exact JSON record of
`schemas/device.schema.json`. The example-device helpers are `secular_trap` (a trap from its secular frequencies),
`raman_pair_along_x` (a 355 nm Raman pair), `oblique_detection_beam`, `ideal_hardware`, `crain_snspd_detector` and
`myerson_ca40_pmt_detector`; the whole example devices are on [machine.md](machine.md).
