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
