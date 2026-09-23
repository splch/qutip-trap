"""Bibliographic sources cited by the species tables, keyed as ``Cited.source`` and ``Level.citations`` /
``Transition.citations`` carry them."""

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
    "Kreuter2005": (
        "A. Kreuter, C. Becher, G. P. T. Lancaster, A. B. Mundt, C. Russo, H. Haeffner, C. Roos, W. Haensel, "
        "F. Schmidt-Kaler, R. Blatt, M. S. Safronova, Experimental and theoretical study of the 3d 2D-level "
        "lifetimes of 40Ca+, Phys. Rev. A 71, 032504 (2005); arXiv:physics/0409038. Run 5, PLAN.md 4.5.7."
    ),
    "Knoop1995_via_Kreuter2005": (
        "M. Knoop, M. Vedel, F. Vedel, Lifetime, collisional-quenching, and j-mixing measurements of the metastable "
        "3D levels of Ca+, Phys. Rev. A 52, 3763 (1995); the four rate coefficients as quoted by Kreuter et al. 2005 "
        "Sec. III B (quenching 37e-12 (H2) and 170e-12 (N2) cm^3 s^-1, j-mixing 3e-10 (H2) and 13e-10 (N2) cm^3 s^-1), "
        "specific coefficients in cm^3 s^-1 and never rates (Section 13). PLAN.md 4.5.7, 9.16."
    ),
    "AliKim1988_via_Kreuter2005": (
        "M. A. Ali, Y.-K. Kim, Phys. Rev. A 38, 3992 (1988): the theoretical M1 transition rate A12 = 2.45e-6 s^-1 of "
        "3d 2D5/2 -> 3d 2D3/2 at nu = 1.82 THz, as quoted by Kreuter et al. 2005 Eq. 1 for the blackbody D-D mixing "
        "rate W12 = A12 n_bar(nu, T) (a calculation, not a measurement). PLAN.md 4.5.7, 9.16."
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
    "Myerson2008": (
        "A. H. Myerson, D. J. Szwer, S. C. Webster, D. T. C. Allcock, M. J. Curtis, G. Imreh, J. A. Sherman, D. N. Stacey, "
        "A. M. Steane, D. M. Lucas, High-fidelity readout of trapped-ion qubits, Phys. Rev. Lett. 100, 200502 (2008); "
        "arXiv:0802.1684. PLAN.md 8.3, 8.4, 9.5 (R_B = 55800 s^-1, R_D = 442 s^-1, tau = 1168(7) ms, n_c = 5.5, "
        "t_b = 420 us, Eqs. 1-2)."
    ),
    "Burrell2010": (
        "A. H. Burrell, D. J. Szwer, S. C. Webster, D. M. Lucas, Scalable simultaneous multi-qubit readout with 99.99% "
        "single-shot fidelity, Phys. Rev. A 81, 040302(R) (2010); arXiv:0906.3304. PLAN.md 8.3, 8.5, 9.5 (camera "
        "likelihoods, 4.0 %/0.9 % PSF cross-talk at 14 um, 400 us exposures, NA 0.25)."
    ),
    "Acton2006": (
        "M. Acton, K.-A. Brickman, P. C. Haljan, P. J. Lee, L. Deslauriers, C. Monroe, Near-perfect simultaneous "
        "measurement of a qubit register, Quantum Inf. Comput. 6, 465 (2006); arXiv:quant-ph/0511257. PLAN.md 8.1, 8.2, "
        "8.5, 8.8, 9.5 (Eqs. 5-13, 20-22; 111Cd+ gamma/2pi = 60 MHz, omega_HFP/2pi = 800 MHz)."
    ),
    "Noek2013": (
        "R. Noek, G. Vrijsen, D. Gaultney, E. Mount, T. Kim, P. Maunz, J. Kim, High speed, high fidelity detection of an "
        "atomic hyperfine qubit, Opt. Lett. 38, 4735 (2013); arXiv:1304.3511. PLAN.md 8.1-8.3, 9.5 (Eqs. 1-6; eps = 2.2 %, "
        "99.85(1) % at 28.1 us, 99.915(7) % at 99.8 us)."
    ),
    "Crain2019": (
        "S. Crain, C. Cahall, G. Vrijsen, E. E. Wollman, M. D. Shaw, V. B. Verma, S. W. Nam, J. Kim, High-speed "
        "low-crosstalk detection of a 171Yb+ qubit using superconducting nanowire single photon detectors, Commun. Phys. 2, "
        "97 (2019); arXiv:1902.04059. PLAN.md 8.1-8.5, 9.5 (Eqs. 1-8; 472(14) kcps, R_d = 341(13) Hz, R_b = 16.4(5) Hz, "
        "R_bg = 4.2(1) cps, eps_sys = 4.356(6) %, 99.931(6) % at 11 us; alpha = 94 ms at 200 um, 814 ms at 370 um)."
    ),
    "Egan2021": (
        "L. Egan et al., Fault-tolerant control of an error-corrected qubit, Nature 598, 281 (2021); arXiv:2009.11482. "
        "PLAN.md 6.7, 8.4, 9.5: the single-ion SPAM budget 0.71(4) % / 0.22(2) % with its pumping and background line items "
        "(quoted from the plan; the arXiv text carries 0.46(2) % for the Z-basis single-qubit SPAM error)."
    ),
    "Wineland1998": (
        "D. J. Wineland, C. Monroe, W. M. Itano, D. Leibfried, B. E. King, D. M. Meekhof, Experimental issues in coherent "
        "quantum-state manipulation of trapped atomic ions, J. Res. NIST 103, 259 (1998); arXiv:quant-ph/9710025. "
        "PLAN.md 8.2, 8.5, 9.5 (P_N(0) = (1 - eta_d)^N; readout crosstalk as degraded discrimination)."
    ),
    "Christensen2020": (
        "J. E. Christensen, D. Hucul, W. C. Campbell, E. R. Hudson, High-fidelity manipulation of a qubit "
        "enabled by a manufactured nucleus, npj Quantum Information 6, 35 (2020); "
        "doi:10.1038/s41534-020-0265-5; arXiv:1907.13331. PLAN.md 8.1, 8.4 (133Ba+ shelving through P3/2, "
        "branching 0.74/0.23/0.03, P3/2 hyperfine splitting 623(30) MHz, tau_D ~ 30 s). The full text was "
        "read on 2026-09-08 and prints exactly two 133Ba+ hyperfine SPLITTINGS: 'we find Delta_3 = "
        "623(30) MHz' for 6p 2P3/2 and 'we find the 2D5/2 hyperfine splitting Delta_5 = 83(30) MHz'. Both "
        "uncertainties are (30), not the (20) that ba133.py's gap notes carried until 2026-09-08. It prints "
        "NO hyperfine A constant for either level and no g_J anywhere, and it attributes the "
        "0.74/0.23/0.03 branching to Dutta et al., so that citation is secondary."
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
    "Pinnington1997": (
        "E. H. Pinnington, G. Rieger, J. A. Kernahan, Beam-laser measurements of the lifetimes of the 6p levels "
        "in Yb II, Phys. Rev. A 56, 2421 (1997); doi:10.1103/PhysRevA.56.2421. tau(6p 2P1/2) = 8.07(9) ns and "
        "tau(6p 2P3/2) = 6.15(9) ns, both read from the published abstract; this is the full locator behind "
        "PLAN.md 4.5.6's bare 'Pinnington et al.' (audit item E25)."
    ),
    "Pinnington1994": (
        "E. H. Pinnington, R. W. Berends, Q. Ji, Beam-laser lifetime measurements of Yb II energy levels, "
        "Phys. Rev. A 50, 2758 (1994); doi:10.1103/PhysRevA.50.2758. Source of the 37.7(5) ns lifetime of the "
        "33 653.86 cm^-1 3[3/2]1/2 level (the 935.2 nm repump upper state). NOTE: the paper's table was NOT read "
        "(APS returns 403 to automated retrieval); the value reaches this table through secondary quotation and "
        "is corroborated to 1% by the 4.2 MHz natural linewidth of the 935 nm line quoted in arXiv:2111.11504, "
        "so it is tagged 'extracted' and not 'verified'."
    ),
    "Feldker2018": (
        "T. Feldker, H. Fuerst, H. Hirzler, N. V. Ewald, M. Mazzanti, D. Bykov, R. Gerritsma et al., Rydberg "
        "excitation of a single trapped ion / branching ratios of the 6p 2P3/2 state of Yb II, "
        "Phys. Rev. A 97, 032511 (2018); arXiv:1711.04667. Table III: branching fractions of 6p 2P3/2 "
        "0.9875(6) (329 nm), 0.0017(1) (1345 nm), 0.0108(5) (1650 nm), and A(2P3/2) = 875.4(10) MHz for 171Yb+."
    ),
    "Berends1992": (
        "R. W. Berends, L. Maleki, Hyperfine structure and isotope shifts of transitions in neutral and singly "
        "ionized ytterbium, J. Opt. Soc. Am. B 9, 332 (1992); doi:10.1364/JOSAB.9.000332. A(6p 2P3/2) = "
        "877(20) MHz for 171Yb+, the first measurement and the concordant cross-check on Feldker et al. 2018."
    ),
    "Tan2021": (
        "T. R. Tan, C. L. Edmunds, A. R. Milne, M. J. Biercuk, C. Hempel, Precision characterization of the "
        "171Yb+ 2D5/2 state, Phys. Rev. A 104, L010802 (2021); arXiv:2012.14187, Table I: tau(2D5/2) = "
        "7.1(4) ms (F = 2) and 7.4(4) ms (F = 3), hyperfine splitting -190.104(3) MHz, hence A = -63.368(1) MHz "
        "(INVERTED, A < 0)."
    ),
    "Taylor1997": (
        "P. Taylor, M. Roberts, S. V. Gateva-Kostova, R. B. M. Clarke, G. P. Barwood, W. R. C. Rowley, P. Gill, "
        "Investigation of the 2S1/2 - 2D5/2 clock transition in a single ytterbium ion, "
        "Phys. Rev. A 56, 2699 (1997): tau(2D5/2) = 7.2(3) ms in 172Yb+ (isotope-independent to well below the "
        "quoted precision)."
    ),
    "Taylor1999": (
        "P. Taylor, M. Roberts, G. M. Macfarlane, G. P. Barwood, W. R. C. Rowley, P. Gill, Measurement of the "
        "2S1/2 - 2F7/2 transition and hyperfine structure in 171Yb+, Phys. Rev. A 60, 2829 (1999): the 2F7/2 "
        "hyperfine splitting 3620(2) MHz, hence A = 905.0(5) MHz for I = 1/2, J = 7/2."
    ),
    "Lange2021": (
        "R. Lange, N. Huntemann, A. V. Viatkina, C. Tamm, A. Surzhykov, E. Peik et al., Improved limits for "
        "violations of local Lorentz invariance / lifetime of the 171Yb+ 2F7/2 state, "
        "Phys. Rev. Lett. 127, 213001 (2021); arXiv:2107.11229. THE PUBLISHED 1.58(8) yr IS SUPERSEDED: "
        "arXiv:2107.11229v2 (14 May 2026) carries the authors' note that E_0^2 was used where <E^2> = E_0^2/2 "
        "belongs, so the lifetime was underestimated by a factor two; the corrected value is "
        "9.96(50)e7 s = 3.16(16) yr, which is what this table stores."
    ),
    "SansonettiMartin2005": (
        "J. E. Sansonetti, W. C. Martin, Handbook of Basic Atomic Spectroscopic Data, "
        "J. Phys. Chem. Ref. Data 34, 1559 (2005): the transition-probability reference behind NIST ASD's Yb II "
        "A values (ASD reference code T7227), including A(297.143 nm, 3[3/2]1/2 - 2S1/2) = 2.61e7 s^-1. ASD "
        "assigns this set no accuracy rating."
    ),
    "Gerritsma2008": (
        "R. Gerritsma, G. Kirchmair, F. Zaehringer, J. Benhelm, R. Blatt, C. F. Roos, Precision measurement of "
        "the branching fractions of the 4p 2P3/2 decay of Ca II, Eur. Phys. J. D 50, 13 (2008); "
        "doi:10.1140/epjd/e2008-00196-9; arXiv:0807.2905. Branching fractions 0.9347(3) (393 nm, S1/2), "
        "0.0587(2) (854 nm, D5/2) and 0.00661(4) (850 nm, D3/2); the three printed fractions sum to 1.00001, "
        "within their combined uncertainty."
    ),
    "Ramm2013": (
        "M. Ramm, T. Pruttivarasin, M. Kokish, I. Talukdar, H. Haeffner, Precision measurement method for "
        "branching fractions of excited P1/2 states applied to 40Ca+, Phys. Rev. Lett. 111, 023004 (2013); "
        "doi:10.1103/PhysRevLett.111.023004; arXiv:1305.0858: 0.93565(7) into S1/2, 0.06435(7) into D3/2. This "
        "is the dedicated measurement Section 12 records as missing."
    ),
    "Hettrich2015": (
        "M. Hettrich, T. Ruster, H. Kaufmann, C. F. Roos, C. T. Schmiegelow, F. Schmidt-Kaler, "
        "U. G. Poschinger, Measurement of dipole matrix elements with a single trapped ion, "
        "Phys. Rev. Lett. 115, 143003 (2015); doi:10.1103/PhysRevLett.115.143003; arXiv:1505.02574: "
        "<S1/2||d||P1/2> = 2.8928(43) e a0, tau(4p 2P1/2) = 6.904(26) ns, branching 0.93572(25), and the two "
        "PARTIAL rates gamma_PS = 2 pi x 21.57(8) MHz and gamma_PD = 2 pi x 1.482(8) MHz."
    ),
    "JinChurch1993": (
        "J.-Y. Jin, D. A. Church, Precision lifetimes for the Ca+ 4p 2P levels, "
        "Phys. Rev. Lett. 70, 3213 (1993): tau(P1/2) = 7.098(20) ns and tau(P3/2) = 6.924(19) ns. Both are "
        "6-7 sigma above the modern single-ion values (Hettrich 2015, Meir 2020) and Hettrich states the 1993 "
        "P1/2 value disagrees with theory by more than 11 sigma; kept as a contested cross-check only. The "
        "paper itself was not read (APS 403); the digits reach this table through Hettrich 2015 and Meir 2020."
    ),
    "Meir2020": (
        "Z. Meir, M. Sinhal, M. S. Safronova, S. Willitsch, Combining experiments and relativistic theory for "
        "establishing accurate radiative quantities in atoms, Phys. Rev. A 101, 012509 (2020); "
        "arXiv:1909.10516: tau(4p 2P3/2) = 6.639(42) ns for Ca+."
    ),
    "Pinnington1995": (
        "E. H. Pinnington, R. W. Berends, M. Lumsden, Studies of laser-induced fluorescence in fast beams of "
        "Sr+ and Ba+ ions, J. Phys. B 28, 2095 (1995): tau(5p 2P1/2) = 7.39(7) ns and tau(5p 2P3/2) = "
        "6.63(7) ns for Sr II. NOTE: the paper's tables were NOT read (IOP bot wall); both values reach this "
        "table quoted verbatim by Likforman et al. 2016 and Zhang et al. 2016, so they are tagged 'extracted'."
    ),
    "Zhang2016": (
        "H. Zhang, M. Gutierrez, G. H. Low, R. Rines, J. Stuart, T. Wu, I. Chuang, Iterative precision "
        "measurement of branching ratios applied to 5P states in 88Sr+, New J. Phys. 18, 123021 (2016); "
        "doi:10.1088/1367-2630/18/12/123021; arXiv:1605.04210: P1/2 0.94498(8)/0.05502(8) and P3/2 "
        "0.9406(2)/0.0063(3)/0.0531(2) into S1/2/D3/2/D5/2."
    ),
    "Likforman2016": (
        "J.-P. Likforman, V. Tugaye, S. Guibal, L. Guidoni, Precision measurement of the branching fractions of "
        "the 5p 2P1/2 state of 88Sr+, Phys. Rev. A 93, 052507 (2016): p(S1/2) = 0.9449(5) as PUBLISHED (the "
        "arXiv:1511.07686 preprint prints 0.9453(+7/-5); the published value is the one NIST ASD uses)."
    ),
    "Arbes1994": (
        "F. Arbes, M. Benzing, T. Gudjons, F. Kurth, G. Werth, Precise determination of the ground state "
        "hyperfine structure splitting of 43Ca II, Z. Phys. D 31, 27 (1994); doi:10.1007/BF01426573. "
        "E_hfs/h = 3 225 608 286.4(3) Hz, hence A(4s 2S1/2) = E_hfs/(I + 1/2) = -806.40207160(8) MHz with the "
        "sign fixed by mu_I < 0. This is the full locator behind PLAN.md's bare 'Arbes et al.' (audit E25)."
    ),
    "Nortershauser1998": (
        "W. Noertershaeuser, N. Trautmann, K. Wendt, B. A. Bushaw et al., Isotope shifts and hyperfine "
        "structure in the 3d 2D_J - 4p 2P_J transitions in calcium II, Eur. Phys. J. D 2, 33 (1998); "
        "doi:10.1007/s100530050107. 43Ca+ A(4p 2P1/2) = -145.4(1), A(4p 2P3/2) = -31.0(2), "
        "B(4p 2P3/2) = -6.9(1.7), A(3d 2D3/2) = -47.3(2), B(3d 2D3/2) = -3.7(1.9) MHz. NOTE: the paper was "
        "not read directly (Springer blocks automated retrieval); every value is corroborated by two "
        "independent compilations, decisively by Garcia Ruiz et al., Phys. Rev. C 91, 041304(R) (2015) "
        "Table I, which prints the Noertershaeuser and Silverans 1991 rows side by side."
    ),
    "Benhelm2007": (
        "J. Benhelm, G. Kirchmair, U. Rapol, T. Koerber, C. F. Roos, R. Blatt, Measurement of the hyperfine "
        "structure of the S1/2-D5/2 transition in 43Ca+, Phys. Rev. A 75, 032506 (2007), WITH the erratum "
        "Phys. Rev. A 75, 049901(E): A(3d 2D5/2) = -3.8931(2) MHz and B(3d 2D5/2) = -4.241(4) MHz. The "
        "erratum is what fixes the signs (arXiv v1/v2 print them positive)."
    ),
    "Tommaseo2003": (
        "G. Tommaseo, T. Pfeil, G. Revalde, G. Werth, P. Indelicato, J. P. Desclaux, The g_J-factor in the "
        "ground state of Ca+, Eur. Phys. J. D 25, 113 (2003); doi:10.1140/epjd/e2003-00096-6: "
        "g_J(4s 2S1/2) = 2.00225664(9). This is the primary behind the value PLAN.md 9.13 carries uncited "
        "for 43Ca+ (audit E25). MEASURED ON 40Ca+, NOT ON 43Ca+, and not read in the original: Hanley, "
        "Allcock, Harty, Sepiol, Lucas, arXiv:2105.10352 (Phys. Rev. A 104, 052804 (2021)) is where the "
        "digits were read, verbatim -- 'Tommaseo et al. measured g_J = 2.002 256 64(+-0.000 000 09) using "
        "double-resonance spectroscopy of 40Ca+ ions in a Penning trap. One would expect the isotopic "
        "dependence of g_J to be smaller than the experimental measurement uncertainty, based upon similar "
        "measurements using Ba+ isotopes [Marx 1998]' -- and that paper then uses this g_J in its own 43Ca+ "
        "Breit-Rabi fit, which is the isotope transfer this package makes too. Sahoo and Kumar, "
        "Phys. Rev. A 96, 012511 (2017), arXiv:1705.02783 Table 2 compute 2.002267, 1.0e-5 away."
    ),
    "Chwalla2009": (
        "M. Chwalla, J. Benhelm, K. Kim, G. Kirchmair et al., Absolute frequency measurement of the 40Ca+ "
        "4s 2S1/2 - 3d 2D5/2 clock transition, Phys. Rev. Lett. 102, 023002 (2009): "
        "g_J(3d 2D5/2) = 1.2003340(3)."
    ),
    "Hanley2021": (
        "C. J. Hanley, D. T. C. Allcock, T. P. Harty, M. A. Sepiol, D. M. Lucas, arXiv:2105.10352: the 43Ca+ "
        "FREE-ION nuclear moment mu_I = -1.315350(9)(1) mu_N, explicitly UNCORRECTED for shielding by the "
        "ion's bound electrons, which is the moment a trapped-ion Breit-Rabi Hamiltonian takes. Stone's "
        "bare-nucleus -1.31733(6) mu_N is a different quantity (Section 13's mu_I row requires the "
        "convention be declared)."
    ),
    "Stone2019": (
        "N. J. Stone, Table of recommended nuclear magnetic dipole moments, IAEA INDC(NDS)-0794 (2019), and "
        "Table of nuclear electric quadrupole moments, INDC(NDS)-0833 (2021): the diamagnetically CORRECTED "
        "(bare-nucleus) moments. mu_I(9Be) = -1.177430(5) mu_N, mu_I(25Mg) = -0.85533(3) mu_N (corrected), "
        "Q(9Be) = +0.0529(4) b, Q(25Mg) = +0.199(2) b, Q(43Ca) = -0.0408(8) b."
    ),
    "Shiga2011": (
        "N. Shiga, W. M. Itano, J. J. Bollinger, Diamagnetic correction to the 9Be+ ground-state hyperfine "
        "constant, Phys. Rev. A 84, 012510 (2011); arXiv:1106.5760: A_0 = -625.008837044(12) MHz with "
        "A(B) = A_0(1 + k B^2), k = 2.63(18)e-11 T^-2, and the g-factor RATIO "
        "g_I'/g_J = 2.1347798527(10)e-4. Supersedes the 'preliminary' 1983 values. Its ABSTRACT carries "
        "only A_0, k and the ratio, but its Sec. I BODY does print the absolute g_J, read verbatim in the "
        "fetched full text on 2026-09-08: 'The value of g_J for the ground electronic state of 9Be+ has "
        "been determined by measuring the 9Be+ cyclotron frequency and a hyperfine-Zeeman transition "
        "frequency at the same magnetic field [5]. The value is g_J = 2.002 262 39(31), calculated with the "
        "use of the best current value of the proton-electron mass ratio [6].' Reference [5] of that paper "
        "is Wineland, Bollinger, Itano, Phys. Rev. Lett. 50, 628 (1983) -- the MEASUREMENT -- and [6] is "
        "Mohr, Taylor, Newell, Rev. Mod. Phys. 80, 633 (2008), i.e. CODATA 2006, from which Shiga et al. "
        "re-evaluate the digits. So the value is measured (1983) and re-reduced (2011), not calculated; "
        "this entry is where those digits are printed."
    ),
    "WinelandBollingerItano1983": (
        "D. J. Wineland, J. J. Bollinger, W. M. Itano, Laser-fluorescence mass spectroscopy, "
        "Phys. Rev. Lett. 50, 628 (1983), with the erratum Phys. Rev. Lett. 50, 1333(E): the first 9Be+ "
        "g_J = 2.00226206(42) and the 'preliminary' A = -625.008837048(10) MHz."
    ),
    "Nortershauser2009": (
        "W. Noertershaeuser, D. Tiedemann, M. Zakova et al., Nuclear charge radii of 7,9,10Be and the one-"
        "neutron halo nucleus 11Be, Phys. Rev. Lett. 102, 062503 (2009); arXiv:0809.2607, Table II: "
        "A(2p 2P1/2) = -118.00(4) MHz for 9Be+, 90x tighter than Bollinger et al. 1985's -118.6(3.6)."
    ),
    "Bollinger1985": (
        "J. J. Bollinger, J. S. Wells, D. J. Wineland, W. M. Itano, Hyperfine structure of the 2p 2P1/2 "
        "state in 9Be+, Phys. Rev. A 31, 2711 (1985): A(2p 2P1/2) = -118.6(3.6) MHz, and (citing Poulsen "
        "et al.) the EXPERIMENTAL BOUND |A(2p 2P3/2)| < 0.6 MHz."
    ),
    "PuchalskiPachucki2009": (
        "M. Puchalski, K. Pachucki, Fine and hyperfine splitting of the 2P state in Li and Be+, "
        "Phys. Rev. A 79, 032510 (2009): THEORY A(2p 2P3/2) = -1.026(3) MHz and B(2p 2P3/2) = "
        "-2.29940(3) MHz for 9Be+. No measurement of either exists, and the A value exceeds Bollinger et "
        "al. 1985's experimental bound |A| < 0.6 MHz, an unresolved tension (ledger conv.be9_p32_hyperfine)."
    ),
    "Dickopf2024": (
        "J. Dickopf, B. Sikora, A. Kaiser et al., Precision spectroscopy on 9Be overcomes limitations from "
        "nuclear structure, Nature 632, 757 (2024); doi:10.1038/s41586-024-07795-1; arXiv:2409.06306: "
        "g_I = -0.78495442296(42)(11), i.e. mu_I = -1.1774316344 mu_N (diamagnetically corrected), which "
        "reproduces the -1.177432 mu_N this table carries to 3e-7. The same paper carries a "
        "bound-electron g factor of 9Be+, read verbatim in the fetched arXiv full text: 'the "
        "bound-electron g-factor g_s(9Be+) = -2.0022621287(24). For the latter, we use the calculations "
        "performed in ref. [22] and the updated nuclear recoil correction [33, 34].' It is therefore "
        "CALCULATED, not measured, and printed NEGATIVE in their sign convention (Section 13's g-factor row "
        "governs, so the magnitude is what this package stores). Pages 757-761 confirmed through Crossref "
        "on 2026-09-08; 'Nature 632, 673' also circulates and is wrong."
    ),
    "Brewer2019": (
        "S. M. Brewer, J.-S. Chen, K. Beloy et al., Measurements of 25Mg+ ground-state hyperfine constants, "
        "Phys. Rev. A 100, 013409 (2019); arXiv:1903.04661: Delta W/h = 1 788 762 752.85(13) Hz, hence "
        "A(3s 2S1/2) = -596.254250949(45) MHz, and the ratio g_I/g_J = 9.299308313(60)e-5."
    ),
    "ItanoWineland1981": (
        "W. M. Itano, D. J. Wineland, Precision measurement of the ground-state hyperfine constant of 25Mg+, "
        "Phys. Rev. A 24, 1364 (1981): A(3s 2S1/2) = -596.254376(54) MHz and the RATIO g_I/g_J = "
        "9.299484(75)e-5. No absolute g_J is measured there: the field is servoed to the electronic "
        "transition, so only the ratio is determined (audit correction of 2026-09-07)."
    ),
    "Ansbacher1989": (
        "W. Ansbacher, Y. Li, E. H. Pinnington, Precision lifetime measurement for the 3p levels of Mg II "
        "using frequency-doubled laser radiation, Phys. Lett. A 139, 165 (1989): tau(3p 2P1/2) = "
        "3.854(30) ns and tau(3p 2P3/2) = 3.810(40) ns. NOTE: Elsevier blocks automated retrieval; the "
        "digits were read from Kaur et al. 2025 Table VII, not from the original."
    ),
    "KaurSahoo2025": (
        "J. Kaur, B. K. Sahoo et al., arXiv:2504.19515: a relativistic-coupled-cluster compilation of Mg II "
        "properties. Tables VII and X: THEORY A(3p 2P1/2) = -103.4(5) MHz and A(3p 2P3/2) = -19.31(5) MHz "
        "for 25Mg+, with a BLANK experiment column for both -- no measured 3p hyperfine constant exists. "
        "Re-searched on 2026-09-08 for a g_J: the 106 kB extracted full text contains ZERO occurrences of "
        "'g_J', 'g factor' or 'g-factor'. This is the most comprehensive modern Mg+ properties calculation, "
        "by the group that publishes dedicated g_j papers for Ca+, Cd+, Yb+ and Hg+, and it computes no "
        "g_J at all -- the strongest available evidence that no computed 25Mg+ g_J exists either "
        "(mg25.py's gap note records the whole negative result)."
    ),
    "SurSahoo2005": (
        "C. Sur, B. K. Sahoo, R. K. Chaudhuri, B. P. Das, D. Mukherjee, Eur. Phys. J. D 32, 25 (2005), "
        "Table 2: relativistic coupled-cluster hyperfine constants of Mg II, printed as UNSIGNED "
        "magnitudes: |A(3p 2P1/2)| = 101.70, |A(3p 2P3/2)| = 18.89, |B(3p 2P3/2)| = 22.91 MHz."
    ),
    "Knab1987": (
        "H. Knab, K.-D. Schupp, G. Werth, Precision measurement of the ground state hyperfine splitting of "
        "133Ba+, Europhys. Lett. 4, 1361 (1987); doi:10.1209/0295-5075/4/12/004: Delta nu(F = 1 <-> 0) = "
        "9 925 453 554.59(10) Hz. NOT Knab, Knoll, Scheerer, Werth, Z. Phys. D 25, 205 (1993), which is a "
        "g_J paper (audit correction of 2026-09-07); Christensen et al. 2020 cite this EPL paper."
    ),
    "Hucul2017": (
        "D. Hucul, J. E. Christensen, E. R. Hudson, W. C. Campbell, Spectroscopy of a synthetic trapped ion "
        "qubit, Phys. Rev. Lett. 119, 100501 (2017); arXiv:1705.09736, Table I: A(6s 2S1/2) = "
        "-9925.45355459(10) MHz for 133Ba+ (NEGATIVE: mu_I(133Ba) < 0, so F = 0 lies ABOVE F = 1), "
        "A(6p 2P1/2) = -1840(11) MHz and A(5d 2D3/2) = -468.5(1.5) MHz. Table I was READ DIRECTLY in the "
        "arXiv full text on 2026-09-08: the A = 133 row prints '-9925.45355459(10)  -1840(11)  "
        "**-468.5(1.5)stat**', where boldface marks this work, so A(6p 2P1/2) = -1840(11) MHz is a "
        "LITERATURE value the table relays (the caption's reference list includes Hoehle et al., "
        "Phys. Lett. B 62, 390 (1976), whose own text is unreadable behind a ScienceDirect 403) while "
        "A(5d 2D3/2) is Hucul's own, with a +-20 MHz systematic on every measurement. Hucul's own "
        "measurement of the 6p 2P1/2 structure is the SPLITTING Delta_2 = 1840(2)stat MHz (Fig. 2b), equal "
        "to |A| because I = J = 1/2, which corroborates the sign-free magnitude of the relayed constant. "
        "The 137Ba+ row gives A(6p 2P1/2) = 743.7(3), A(5d 2D3/2) = 189.7288(6), B(5d 2D3/2) = "
        "44.5417(16) MHz."
    ),
    "BlattWerth1982": (
        "R. Blatt, G. Werth, Precision determination of the ground-state hyperfine splitting in 137Ba+ using "
        "the ion-storage technique, Phys. Rev. A 25, 1476 (1982): W_hfs/h = 8 037 741 667.69(36) Hz, hence "
        "A(6s 2S1/2) = W_hfs/(I + 1/2) = 4018.87083385(18) MHz."
    ),
    "Villemoes1993": (
        "P. Villemoes, A. Arnesen, F. Heijkenskjoeld, A. Wannstroem, Isotope shifts and hyperfine structure "
        "of 130-138Ba II by fast ion beam laser spectroscopy, J. Phys. B 26, 4289 (1993): 137Ba+ "
        "A(6p 2P1/2) = 743.7(3), A(6p 2P3/2) = 127.2(2), B(6p 2P3/2) = 92.5(2) MHz. NOTE: IOP blocks "
        "automated retrieval; A(6p 2P1/2) is corroborated by two independent secondary sources and the "
        "P3/2 pair by one."
    ),
    "Lewty2013": (
        "N. C. Lewty, B. L. Chuah, R. Cazan, B. K. Sahoo, M. D. Barrett, Spectroscopy on a single trapped "
        "137Ba+ ion for nuclear structure studies, Phys. Rev. A 88, 012518 (2013); arXiv:1305.4453, "
        "Tables II and VI: A(5d 2D3/2) = 189.731494(17), B = 44.537594(34) MHz; A(5d 2D5/2) = "
        "-12.029234(11), B = 59.525520(110) MHz; octupole Omega = 0.05057(54) mu_N b. This paper corrects "
        "the authors' own two earlier Opt. Express papers and supersedes Silverans et al. 1986 by ~1000x."
    ),
    "Arnold2019": (
        "K. J. Arnold, S. R. Chanu, R. Kaewuam, T. R. Tan, L. Yeo, Z. Zhang, M. S. Safronova, "
        "M. D. Barrett, Precision measurements of the 138Ba+ 6s-5d quadrupole transition, "
        "Phys. Rev. A 100, 032503 (2019); arXiv:1905.06523: tau(6p 2P1/2) = 7.855(10) ns and the branching "
        "fractions 0.731823(57) into 6s 2S1/2 and 0.268177(57) into 5d 2D3/2."
    ),
    "ZhangBa2020": (
        "Z. Zhang, K. J. Arnold, S. R. Chanu, R. Kaewuam, M. S. Safronova, M. D. Barrett, Branching "
        "fractions for P3/2 decays in Ba+, Phys. Rev. A 101, 062515 (2020); arXiv:2003.02263: "
        "tau(6p 2P3/2) = 6.2615(72) ns, branching 0.741716(71) (S1/2), 0.028031(23) (D3/2), "
        "0.230253(61) (D5/2), and tau(5d 2D5/2) = 30.14(40) s."
    ),
    "Auchter2014": (
        "C. Auchter, T. W. Noel, M. R. Hoffman, S. R. Williams, B. B. Blinov, Measurement of the branching "
        "fractions and lifetime of the 5D5/2 level in Ba+, Phys. Rev. A 90, 060501(R) (2014): "
        "tau(5d 2D5/2) = 31.2(9) s."
    ),
    "Yu1997": (
        "N. Yu, W. Nagourney, H. Dehmelt, Radiative lifetime measurement of the Ba+ metastable D3/2 state, "
        "Phys. Rev. Lett. 78, 4898 (1997): tau(5d 2D3/2) = 79.8(4.6) s."
    ),
    "Marx1998": (
        "G. Marx, G. Tommaseo, G. Werth, Precise g_J- and g_I-factor measurements of Ba+ isotopes, "
        "Eur. Phys. J. D 4, 279 (1998); doi:10.1007/s100530050210: g_J(6s 2S1/2) = 2.00249192(3), MEASURED "
        "in a Penning trap on 138Ba+ and 135Ba+ (the companion AIP Conf. Proc. 457, 57 (1999), "
        "doi:10.1063/1.57484, states both isotopes and a fractional accuracy of 2e-8; g_I comes from "
        "Delta m_I = 1 transitions in 137Ba+ to 5e-6). NOT READ IN THE ORIGINAL: Springer redirects "
        "link.springer.com/article/10.1007/s100530050210 to an identity provider, OpenAlex and Semantic "
        "Scholar carry no abstract and there is no open-access copy. The digits were read verbatim in the "
        "fetched full text of Arnold et al., Phys. Rev. Lett. 124, 193001 (2020), arXiv:1906.09150v2, which "
        "writes 'g_D = r g_S can be obtained using g_S = 2.002 491 92(3) given in [Marx et al. 1998]', uses "
        "the same digits a second time in its summary, and calls g_S 'known to the 1e-8 level'. Supersedes "
        "Hubrich, Knab, Knoll, Werth, Z. Phys. D 18, 113 (1991), doi:10.1007/bf01418176, whose 2.0024906 "
        "still circulates (quoted with uncertainty (12) in the Lindroth and Ynnerman, Phys. Rev. A 47, 961 "
        "(1993) abstract, and as (11) elsewhere; neither digit was read in Hubrich 1991 itself), and "
        "Knab, Knoll, Scheerer, Werth, Z. Phys. D 25, 205 (1993). Theory brackets it: 2.0024911(30) "
        "(Lindroth and Ynnerman 1993) and 2.0024905(16) (Li, Jiang, Wu, Zhang, Dong, Chin. Phys. Lett. 42, "
        "063101 (2025), doi:10.1088/0256-307X/42/6/063101). ISOTOPE-INDEPENDENT to the quoted precision, "
        "which is a stated assumption and not a measurement on 133Ba+ or 137Ba+: Hanley et al., "
        "arXiv:2105.10352 (Phys. Rev. A 104, 052804 (2021)) write 'One would expect the isotopic dependence "
        "of g_J to be smaller than the experimental measurement uncertainty, based upon similar "
        "measurements using Ba+ isotopes [Marx 1998]'."
    ),
    "Knoll1996": (
        "K. H. Knoell, G. Marx, K. Huebner, F. Schweikert, S. Stahl, C. Weber, G. Werth, Experimental g_J "
        "factor in the metastable 5D3/2 level of Ba+, Phys. Rev. A 54, 1199 (1996); "
        "doi:10.1103/PhysRevA.54.1199: g_J(5d 2D3/2) = 0.7993278(3), MEASURED. NOT READ IN THE ORIGINAL "
        "(APS returns 403 to automated retrieval): the digits were read in the fetched abstract of Li, "
        "Jiang, Wu, Zhang, Dong, Chin. Phys. Lett. 42, 063101 (2025), doi:10.1088/0256-307X/42/6/063101, "
        "which writes 'the present results of 0.7993961(126) and 1.2003942(190) agree with the experimental "
        "results of 0.7993278(3) [Phys. Rev. A 54 1199 (1996)] and 1.20036739(14) [Phys. Rev. Lett. 124 "
        "193001 (2020)]'. Replaces the NIST ASD Lande column's two-digit 0.79 as the stored value."
    ),
    "ArnoldPRL2020": (
        "K. J. Arnold, R. Kaewuam, S. R. Chanu, T. R. Tan, Z. Zhang, M. D. Barrett, Precision measurements "
        "of the 138Ba+ 6s 2S1/2 - 5d 2D5/2 clock transition, Phys. Rev. Lett. 124, 193001 (2020); "
        "arXiv:1906.09150v2, read directly: g_J(5d 2D5/2) = 1.20036739(24), obtained as the measured "
        "Zeeman-splitting ratio r = g_D/g_S = 0.59943681(12) times g_S = 2.00249192(3) from Marx et al. "
        "1998. Replaces the NIST ASD Lande column's 1.12, which is 6.7 % away and is ASD's own error, not a "
        "transcription slip in this repository. Chin. Phys. Lett. 42, 063101 (2025) misquotes the "
        "uncertainty as (14); the PRL's own text says (24), and (24) is what is stored. Hoffman, Noel, "
        "Auchter, Jayakumar, Williams, Blinov, Fortson, Phys. Rev. A 88, 025401 (2013), arXiv:1306.3518 "
        "measured the same quantity earlier as 1.200372(4)stat(7)sys."
    ),
    # ---- databases ----
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
    # ---- values carried without a named primary source ----
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
