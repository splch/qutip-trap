"""Bibliographic sources cited by the species tables (PLAN.md Section 15 and Appendix D).

Keys are what ``Cited.source`` and ``Level.citations`` / ``Transition.citations`` carry. Where PLAN.md
names a source only loosely ("Pinnington et al."), the entry says so rather than inventing a locator.
"""

from __future__ import annotations

from typing import Final

SOURCES: Final[dict[str, str]] = {
    # ---- primary literature ----
    "Olmschenk2007": (
        "S. Olmschenk, K. C. Younge, D. L. Moehring, D. N. Matsukevich, P. Maunz, C. Monroe, Manipulation and "
        "detection of a trapped Yb+ hyperfine qubit, Phys. Rev. A 76, 052314 (2007). Run 4 extraction, PLAN.md 4.5.6."
    ),
    "Han2025": (
        "Han et al., arXiv:2501.09973 (2025): multiconfiguration Dirac-Hartree-Fock and multireference "
        "configuration-interaction g_J of the 171Yb+ ground state, 2.002615(70), with the second-order Zeeman "
        "coefficient 31.0869(22) mHz/uT^2. Adopted by PLAN.md 4.5.1 after the 2026-09-04 critique."
    ),
    "Pinnington_via_Olmschenk2007": (
        "Pinnington et al., the 171Yb+ P1/2 lifetime 8.07(9) ns, as cited by Olmschenk et al. 2007; PLAN.md 4.5.6 "
        "names only 'Pinnington et al.' and infers gamma/2pi = 19.7 MHz from it (the full reference is not given "
        "in the plan and was not fetched here)."
    ),
    "Kreuter2005": (
        "A. Kreuter, C. Becher, G. P. T. Lancaster, A. B. Mundt, C. Russo, H. Haeffner, C. Roos, W. Haensel, "
        "F. Schmidt-Kaler, R. Blatt, M. S. Safronova, Experimental and theoretical study of the 3d 2D-level "
        "lifetimes of 40Ca+, Phys. Rev. A 71, 032504 (2005); arXiv:physics/0409038. Run 5, PLAN.md 4.5.7."
    ),
    "Barton2000": (
        "P. A. Barton, C. J. S. Donald, D. M. Lucas, D. A. Stevens, A. M. Steane, D. N. Stacey, Measurement of "
        "the lifetime of the 3d 2D5/2 state in 40Ca+, Phys. Rev. A 62, 032503 (2000). Via Schindler et al. 2013; "
        "PLAN.md 4.5.7."
    ),
    "Harty2014": (
        "T. P. Harty, D. T. C. Allcock, C. J. Ballance, L. Guidoni, H. A. Janacek, N. M. Linke, D. N. Stacey, "
        "D. M. Lucas, High-fidelity preparation, gates, memory, and readout of a trapped-ion quantum bit, "
        "Phys. Rev. Lett. 113, 220501 (2014). PLAN.md 4.3.3, 4.5.6, 9.13."
    ),
    "Arbes1994_via_Harty2014": (
        "Arbes et al. (1994), the 43Ca+ ground-state zero-field hyperfine splitting 3225.608 MHz, cited by "
        "Harty et al. 2014 and PLAN.md 4.5.6/9.12 (the plan gives A = -806.4020716 MHz; the paper's locator is "
        "not given in the plan)."
    ),
    "Langer2005": (
        "C. Langer et al., Long-lived qubit memory using atomic ions, Phys. Rev. Lett. 95, 060502 (2005). "
        "PLAN.md 4.5.6, 9.13."
    ),
    "Srinivas2021": (
        "R. Srinivas et al., High-fidelity laser-free universal control of trapped ion qubits, Nature 597, 209 "
        "(2021). PLAN.md 4.4.5, 9.13 (25Mg+ clock point 212.8 G, 1.686 GHz)."
    ),
    "Letchumanan2005": (
        "V. Letchumanan, M. A. Wilson, P. Gill, A. G. Sinclair, Lifetime measurement of the metastable 4d 2D5/2 "
        "state in 88Sr+ using a single trapped ion, Phys. Rev. A 72, 012509 (2005); read as Chapter 5 of "
        "V. Letchumanan, PhD thesis, Imperial College London and NPL (2004). PLAN.md 4.5.7, 9.14."
    ),
    "Jiang2009": (
        "D. Jiang, B. Arora, M. S. Safronova, C. W. Clark, Blackbody-radiation shift in a 88Sr+ ion optical "
        "frequency standard, J. Phys. B 42, 154020 (2009); arXiv:0904.2107 (Table 5 quotes 0.3908(16) s)."
    ),
    "Sansonetti2012": (
        "J. E. Sansonetti, Wavelengths, transition probabilities, and energy levels for the spectra of "
        "strontium ions, J. Phys. Chem. Ref. Data 41, 013102 (2012): A(D5/2) = 2.559(10) s^-1 and the 88Sr II "
        "clock frequency 444 779 044 095 484.6 Hz. PLAN.md 4.5.7."
    ),
    "Christensen2020": (
        "J. E. Christensen, D. Hucul, W. C. Campbell, E. R. Hudson, High-fidelity manipulation of a qubit "
        "enabled by a manufactured nucleus, npj Quantum Information 6, 35 (2020). PLAN.md 8.1, 8.4 (133Ba+ "
        "shelving through P3/2, branching 0.74/0.23/0.03, P3/2 hyperfine splitting 623(30) MHz, tau_D ~ 30 s)."
    ),
    "Ozeri2007": (
        "R. Ozeri et al., Errors in trapped-ion quantum gates due to spontaneous photon scattering, Phys. Rev. "
        "A 75, 042329 (2007), Table I. PLAN.md 4.3.2, 4.5.5, 9.13."
    ),
    "Ejtemaee2017": (
        "S. Ejtemaee, P. C. Haljan, 3D Sisyphus cooling of trapped ions, Phys. Rev. Lett. 119, 043001 (2017); "
        "arXiv:1603.01248. PLAN.md 4.2.4, 9.15."
    ),
    "Schindler2013": (
        "P. Schindler et al., A quantum information processor with trapped ions, New J. Phys. 15, 123012 (2013). "
        "PLAN.md 4.5.7 (Run 4Q)."
    ),
    "James1998": (
        "D. F. V. James, Quantum dynamics of cold trapped ions with application to quantum computation, Appl. "
        "Phys. B 66, 181 (1998), Secs. 4-5 and Appendix. PLAN.md 4.5.7 (Run 4Q)."
    ),
    "Roos2000": "C. F. Roos, Controlling the quantum state of trapped ions, PhD thesis, Innsbruck (2000). PLAN.md 4.5.7.",
    "Monroe1995": (
        "C. Monroe, D. M. Meekhof, B. E. King, S. R. Jefferts, W. M. Itano, D. J. Wineland, P. Gould, Resolved-sideband "
        "Raman cooling of a bound atom to the 3D zero-point energy, Phys. Rev. Lett. 75, 4011 (1995). PLAN.md 4.2.1, "
        "4.2.2 (9Be+ Gamma/2pi = 19.4 MHz, best detuning about -30 MHz)."
    ),
    "Steck": (
        "D. A. Steck, Quantum and Atom Optics (revision 0.16.10, 27 June 2026) and the alkali D-line data notes; "
        "the convention source of PLAN.md Section 4.5 and Section 13."
    ),
    # ---- databases, retrieved for this table on 2026-09-04 ----
    "NIST_ASD_5_12": (
        "A. Kramida, Yu. Ralchenko, J. Reader, and NIST ASD Team (2024), NIST Atomic Spectra Database (ver. 5.12), "
        "energy-level tables, https://physics.nist.gov/asd, retrieved 2026-09-04. Level energies in cm^-1 and, "
        "where the database lists them, Lande g factors."
    ),
    "NIST_AWIC": (
        "J. S. Coursey, D. J. Schwab, J. J. Tsai, R. A. Dragoset, Atomic Weights and Isotopic Compositions "
        "(NIST Physical Measurement Laboratory), https://physics.nist.gov/Comp, retrieved 2026-09-04: relative "
        "atomic masses of the isotopes with their uncertainties (the page draws them from the Atomic Mass "
        "Evaluation; the edition was not read from the page)."
    ),
    "CODATA2022_scipy": "CODATA 2022 recommended values via scipy.constants (SciPy 1.18.1); see qutip_trap.units.",
    # ---- the plan itself, where it carries a value without naming the primary source ----
    "PLAN_4_5_1": (
        "PLAN.md Section 4.5.1 (this repository): the value is used there and in Section 13 without a named "
        "primary source; it stands until a primary citation is added."
    ),
    "PLAN_8_1": "PLAN.md Section 8.1 (this repository), which carries the value without naming a primary source.",
    "PLAN_9_13": (
        "PLAN.md Section 9.13 (this repository): the value is a stated test input ('quoted', 'must be cited from "
        "elsewhere') whose primary source the plan does not name."
    ),
    "PLAN_check_atomic": (
        "validation/scripts/check_atomic.py (this repository, Appendix D): the value is a script input whose "
        "primary source is not named in PLAN.md."
    ),
    "PLAN_background": "PLAN.md convention or textbook statement tagged [background] (no source check in the research runs).",
}

__all__ = ["SOURCES"]
