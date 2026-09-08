"""Angular-momentum algebra shared by all species (PLAN.md Section 4.5; milestone M0a): the facade.

The layer derives every hyperfine-resolved quantity from the cited constants of the species tables and never
stores one. Its pieces, each in its own module and re-exported here:

- :mod:`~qutip_trap.species.wigner`: exact 3j and 6j symbols, Clebsch-Gordan coefficients, spin matrices;
- :mod:`~qutip_trap.species.zeeman`: the hyperfine-Zeeman diagonalization at any field with the Breit-Rabi
  cross-check, adiabatic (F, m_F) labels, field sensitivities in both curvature conventions, the clock-point
  finder (anchors: 43Ca+ 146.0942 G, 9Be+ 119.446 G, 25Mg+ 212.78 G, 171Yb+ 310.87 Hz/G^2; Section 9.13);
- :mod:`~qutip_trap.species.dipole`: Gamma -> reduced element -> I_sat, the Wigner-Eckart elements on the
  uncoupled basis and in Steck's 6j factorization, the directed hyperfine factors (171Yb+ 1/3 : 2/3);
- :mod:`~qutip_trap.species.polarization`: the laboratory-to-atomic-frame decomposition into sigma-, pi, sigma+;
- :mod:`~qutip_trap.species.raman`: Raman couplings, light shifts and Kramers-Heisenberg scattering amplitudes,
  rates, leakage and differential-Rayleigh dephasing from explicit intermediate-state sums;
- :mod:`~qutip_trap.species.quadrupole`: the electric-quadrupole coupling of optical qubits, its per-component
  decay weights and collapse operators, and the second-order ac Stark shift of a driven component by the
  other nine (Section 4.5.7);
- :mod:`~qutip_trap.species.metastable`: the blackbody, collisional and reshelving channels of a metastable
  D level, all defaulting off (Section 4.5.7).

Conventions (Section 13): Steck's normalizations throughout; g_I = -(mu_I/(I mu_N))(m_e/m_p); q_op = m_lower -
m_upper = -q_gamma; wavelengths vacuum; I_sat with the ANGULAR partial rate; C0 is not this layer's business.
"""

from __future__ import annotations

from qutip_trap.species.dipole import (
    absorption_strength,
    coupled_state_vector,
    dipole_operator_uncoupled,
    emission_branching,
    field_amplitude_v_per_m,
    hyperfine_element,
    hyperfine_reduced_factor,
    partial_rate_from_reduced_element,
    rabi_frequency_two_level_rad_s,
    reduced_element_from_partial_rate,
    resonant_cross_section_m2,
    saturation_intensity_random_orientation_w_m2,
    saturation_intensity_w_m2,
    stretched_element_factor,
    wigner_eckart_j,
)
from qutip_trap.species.polarization import (
    atomic_frame,
    linear_polarization,
    operator_index,
    spherical_basis,
    spherical_components,
    to_atomic_frame,
)
from qutip_trap.species.quadrupole import (
    decay_weights,
    e2_stark_shift_rad_s,
    geometric_factor,
    geometric_factor_closed_form,
    geometric_factors,
    lambda_3j,
    quadrupole_collapse_operators,
    rabi_frequency_e2_rad_s,
    reduced_element_a0_squared,
    reduced_element_from_lifetime_m2,
    stretched_closed_form_rad_s,
)
from qutip_trap.species.raman import AtomicStructure, DressedState
from qutip_trap.species.wigner import angular_momentum_matrices, clebsch_gordan, wigner_3j, wigner_6j
from qutip_trap.species.zeeman import (
    MU_B_OVER_H_HZ_PER_G,
    ClockPoint,
    HyperfineZeeman,
    TransitionSensitivity,
    ZeemanSpectrum,
    breit_rabi_hz,
    clock_points,
    g_I_steck,
    hyperfine_zeeman,
    lande_g_f,
    transition_sensitivity,
)

__all__ = [
    "MU_B_OVER_H_HZ_PER_G",
    "AtomicStructure",
    "ClockPoint",
    "DressedState",
    "HyperfineZeeman",
    "TransitionSensitivity",
    "ZeemanSpectrum",
    "absorption_strength",
    "angular_momentum_matrices",
    "atomic_frame",
    "breit_rabi_hz",
    "clebsch_gordan",
    "decay_weights",
    "e2_stark_shift_rad_s",
    "clock_points",
    "coupled_state_vector",
    "dipole_operator_uncoupled",
    "emission_branching",
    "field_amplitude_v_per_m",
    "g_I_steck",
    "geometric_factor",
    "geometric_factor_closed_form",
    "geometric_factors",
    "hyperfine_element",
    "hyperfine_reduced_factor",
    "hyperfine_zeeman",
    "lande_g_f",
    "lambda_3j",
    "linear_polarization",
    "operator_index",
    "partial_rate_from_reduced_element",
    "quadrupole_collapse_operators",
    "rabi_frequency_e2_rad_s",
    "rabi_frequency_two_level_rad_s",
    "reduced_element_a0_squared",
    "reduced_element_from_lifetime_m2",
    "reduced_element_from_partial_rate",
    "resonant_cross_section_m2",
    "saturation_intensity_random_orientation_w_m2",
    "saturation_intensity_w_m2",
    "spherical_basis",
    "spherical_components",
    "stretched_closed_form_rad_s",
    "stretched_element_factor",
    "to_atomic_frame",
    "transition_sensitivity",
    "wigner_3j",
    "wigner_6j",
    "wigner_eckart_j",
]
