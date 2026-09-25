"""The sources ``Cited.source``, the ``citations`` of the species records and the readout presets name."""

from __future__ import annotations

from typing import Final

SOURCES: Final[dict[str, str]] = {
    # ---- 171Yb+ ----
    "Olmschenk2007": (
        "S. Olmschenk, K. C. Younge, D. L. Moehring, D. N. Matsukevich, P. Maunz, C. Monroe, Manipulation and detection of "
        "a trapped Yb+ hyperfine qubit, Phys. Rev. A 76, 052314 (2007)."
    ),
    "Han2025": (
        "Han et al., arXiv:2501.09973 (2025): MCDHF/MRCI g_J of the 171Yb+ ground state, 2.002615(70), and the second-order "
        "Zeeman coefficient 31.0869(22) mHz/uT^2."
    ),
    "Pinnington1997": (
        "E. H. Pinnington, G. Rieger, J. A. Kernahan, Beam-laser measurements of the lifetimes of the 6p levels in Yb II, "
        "Phys. Rev. A 56, 2421 (1997): tau(6p 2P1/2) = 8.07(9) ns, tau(6p 2P3/2) = 6.15(9) ns."
    ),
    "Pinnington1994": (
        "E. H. Pinnington, R. W. Berends, Q. Ji, Beam-laser lifetime measurements of Yb II energy levels, Phys. Rev. A 50, "
        "2758 (1994): the 37.7(5) ns lifetime of the 3[3/2]1/2 level, read by secondary quotation."
    ),
    "Feldker2018": (
        "T. Feldker et al., Phys. Rev. A 97, 032511 (2018); arXiv:1711.04667, Table III: 6p 2P3/2 branching 0.9875(6) / "
        "0.0017(1) / 0.0108(5) and A(2P3/2) = 875.4(10) MHz for 171Yb+."
    ),
    "Berends1992": (
        "R. W. Berends, L. Maleki, Hyperfine structure and isotope shifts of transitions in neutral and singly ionized "
        "ytterbium, J. Opt. Soc. Am. B 9, 332 (1992): A(6p 2P3/2) = 877(20) MHz for 171Yb+."
    ),
    "Tan2021": (
        "T. R. Tan, C. L. Edmunds, A. R. Milne, M. J. Biercuk, C. Hempel, Precision characterization of the 171Yb+ 2D5/2 "
        "state, Phys. Rev. A 104, L010802 (2021); arXiv:2012.14187: the hyperfine splitting -190.104(3) MHz."
    ),
    "Taylor1997": (
        "P. Taylor et al., Investigation of the 2S1/2 - 2D5/2 clock transition in a single ytterbium ion, Phys. Rev. A 56, "
        "2699 (1997): tau(2D5/2) = 7.2(3) ms in 172Yb+."
    ),
    "Taylor1999": (
        "P. Taylor et al., Measurement of the 2S1/2 - 2F7/2 transition and hyperfine structure in 171Yb+, Phys. Rev. A 60, "
        "2829 (1999): the 2F7/2 hyperfine splitting 3620(2) MHz."
    ),
    "Lange2021": (
        "R. Lange et al., Phys. Rev. Lett. 127, 213001 (2021); arXiv:2107.11229: the 171Yb+ 2F7/2 lifetime, corrected in v2 "
        "to 9.96(50)e7 s = 3.16(16) yr."
    ),
    "SansonettiMartin2005": (
        "J. E. Sansonetti, W. C. Martin, Handbook of Basic Atomic Spectroscopic Data, J. Phys. Chem. Ref. Data 34, 1559 "
        "(2005): NIST ASD's Yb II A values (T7227), A(297.143 nm) = 2.61e7 s^-1."
    ),
    # ---- Ca+ ----
    "Kreuter2005": (
        "A. Kreuter et al., Experimental and theoretical study of the 3d 2D-level lifetimes of 40Ca+, Phys. Rev. A 71, "
        "032504 (2005); arXiv:physics/0409038."
    ),
    "Knoop1995_via_Kreuter2005": (
        "M. Knoop, M. Vedel, F. Vedel, Lifetime, collisional-quenching, and j-mixing measurements of the metastable 3D "
        "levels of Ca+, Phys. Rev. A 52, 3763 (1995), as quoted by Kreuter et al. 2005 Sec. III B."
    ),
    "AliKim1988_via_Kreuter2005": (
        "M. A. Ali, Y.-K. Kim, Phys. Rev. A 38, 3992 (1988): the calculated M1 rate A12 = 2.45e-6 s^-1 of 3d 2D5/2 -> 3d "
        "2D3/2, as quoted by Kreuter et al. 2005 Eq. 1."
    ),
    "Barton2000": (
        "P. A. Barton et al., Measurement of the lifetime of the 3d 2D5/2 state in 40Ca+, Phys. Rev. A 62, 032503 (2000)."
    ),
    "Harty2014": (
        "T. P. Harty et al., High-fidelity preparation, gates, memory, and readout of a trapped-ion quantum bit, Phys. Rev. "
        "Lett. 113, 220501 (2014)."
    ),
    "James1998": (
        "D. F. V. James, Quantum dynamics of cold trapped ions with application to quantum computation, Appl. Phys. B 66, "
        "181 (1998)."
    ),
    "Gerritsma2008": (
        "R. Gerritsma et al., Precision measurement of the branching fractions of the 4p 2P3/2 decay of Ca II, Eur. Phys. "
        "J. D 50, 13 (2008); arXiv:0807.2905: 0.9347(3) / 0.0587(2) / 0.00661(4)."
    ),
    "Ramm2013": (
        "M. Ramm, T. Pruttivarasin, M. Kokish, I. Talukdar, H. Haeffner, Precision measurement method for branching "
        "fractions of excited P1/2 states applied to 40Ca+, Phys. Rev. Lett. 111, 023004 (2013); arXiv:1305.0858."
    ),
    "Hettrich2015": (
        "M. Hettrich et al., Measurement of dipole matrix elements with a single trapped ion, Phys. Rev. Lett. 115, 143003 "
        "(2015); arXiv:1505.02574: tau(4p 2P1/2) = 6.904(26) ns and the partial rates 2 pi x 21.57(8) and 1.482(8) MHz."
    ),
    "Meir2020": (
        "Z. Meir, M. Sinhal, M. S. Safronova, S. Willitsch, Phys. Rev. A 101, 012509 (2020); arXiv:1909.10516: "
        "tau(4p 2P3/2) = 6.639(42) ns."
    ),
    "Arbes1994": (
        "F. Arbes, M. Benzing, T. Gudjons, F. Kurth, G. Werth, Precise determination of the ground state hyperfine "
        "structure splitting of 43Ca II, Z. Phys. D 31, 27 (1994): E_hfs/h = 3 225 608 286.4(3) Hz."
    ),
    "Nortershauser1998": (
        "W. Noertershaeuser et al., Isotope shifts and hyperfine structure in the 3d 2D_J - 4p 2P_J transitions in calcium "
        "II, Eur. Phys. J. D 2, 33 (1998), corroborated by Garcia Ruiz et al., Phys. Rev. C 91, 041304(R) (2015) Table I."
    ),
    "Benhelm2007": (
        "J. Benhelm et al., Measurement of the hyperfine structure of the S1/2-D5/2 transition in 43Ca+, Phys. Rev. A 75, "
        "032506 (2007), with the erratum Phys. Rev. A 75, 049901(E), which fixes the signs."
    ),
    "Tommaseo2003": (
        "G. Tommaseo et al., The g_J-factor in the ground state of Ca+, Eur. Phys. J. D 25, 113 (2003): 2.00225664(9), "
        "measured on 40Ca+; read in Hanley et al., arXiv:2105.10352."
    ),
    "Chwalla2009": (
        "M. Chwalla et al., Absolute frequency measurement of the 40Ca+ 4s 2S1/2 - 3d 2D5/2 clock transition, Phys. Rev. "
        "Lett. 102, 023002 (2009): g_J(3d 2D5/2) = 1.2003340(3)."
    ),
    "Hanley2021": (
        "C. J. Hanley, D. T. C. Allcock, T. P. Harty, M. A. Sepiol, D. M. Lucas, Phys. Rev. A 104, 052804 (2021); "
        "arXiv:2105.10352: the 43Ca+ free-ion moment -1.315350(9)(1) mu_N."
    ),
    # ---- Sr+ ----
    "Letchumanan2005": (
        "V. Letchumanan, M. A. Wilson, P. Gill, A. G. Sinclair, Lifetime measurement of the metastable 4d 2D5/2 state in "
        "88Sr+ using a single trapped ion, Phys. Rev. A 72, 012509 (2005)."
    ),
    "Jiang2009": (
        "D. Jiang, B. Arora, M. S. Safronova, C. W. Clark, Blackbody-radiation shift in a 88Sr+ ion optical frequency "
        "standard, J. Phys. B 42, 154020 (2009); arXiv:0904.2107."
    ),
    "Sansonetti2012": (
        "J. E. Sansonetti, Wavelengths, transition probabilities, and energy levels for the spectra of strontium ions, J. "
        "Phys. Chem. Ref. Data 41, 013102 (2012): A(D5/2) = 2.559(10) s^-1 and the clock frequency 444 779 044 095 484.6 Hz."
    ),
    "Pinnington1995": (
        "E. H. Pinnington, R. W. Berends, M. Lumsden, Studies of laser-induced fluorescence in fast beams of Sr+ and Ba+ "
        "ions, J. Phys. B 28, 2095 (1995): tau(5p 2P1/2) = 7.39(7) ns, tau(5p 2P3/2) = 6.63(7) ns."
    ),
    "Zhang2016": (
        "H. Zhang et al., Iterative precision measurement of branching ratios applied to 5P states in 88Sr+, New J. Phys. "
        "18, 123021 (2016); arXiv:1605.04210."
    ),
    # ---- Be+ ----
    "Monroe1995": (
        "C. Monroe et al., Resolved-sideband Raman cooling of a bound atom to the 3D zero-point energy, Phys. Rev. Lett. 75, "
        "4011 (1995): 9Be+ Gamma/2pi = 19.4 MHz."
    ),
    "Shiga2011": (
        "N. Shiga, W. M. Itano, J. J. Bollinger, Diamagnetic correction to the 9Be+ ground-state hyperfine constant, Phys. "
        "Rev. A 84, 012510 (2011); arXiv:1106.5760: A_0 = -625.008837044(12) MHz, and in its Sec. I g_J = 2.002 262 39(31), "
        "the 1983 measurement re-reduced with CODATA 2006."
    ),
    "WinelandBollingerItano1983": (
        "D. J. Wineland, J. J. Bollinger, W. M. Itano, Laser-fluorescence mass spectroscopy, Phys. Rev. Lett. 50, 628 "
        "(1983), erratum 50, 1333(E)."
    ),
    "Nortershauser2009": (
        "W. Noertershaeuser et al., Nuclear charge radii of 7,9,10Be and the one-neutron halo nucleus 11Be, Phys. Rev. Lett. "
        "102, 062503 (2009); arXiv:0809.2607, Table II: A(2p 2P1/2) = -118.00(4) MHz."
    ),
    "Bollinger1985": (
        "J. J. Bollinger, J. S. Wells, D. J. Wineland, W. M. Itano, Hyperfine structure of the 2p 2P1/2 state in 9Be+, "
        "Phys. Rev. A 31, 2711 (1985): -118.6(3.6) MHz and the bound |A(2p 2P3/2)| < 0.6 MHz."
    ),
    "PuchalskiPachucki2009": (
        "M. Puchalski, K. Pachucki, Fine and hyperfine splitting of the 2P state in Li and Be+, Phys. Rev. A 79, 032510 "
        "(2009): theory A and B of 9Be+ 2p 2P3/2."
    ),
    "Dickopf2024": (
        "J. Dickopf et al., Precision spectroscopy on 9Be overcomes limitations from nuclear structure, Nature 632, 757 "
        "(2024); arXiv:2409.06306: g_I = -0.78495442296(42)(11) and the calculated g_s(9Be+) = -2.0022621287(24)."
    ),
    # ---- Mg+ ----
    "Brewer2019": (
        "S. M. Brewer et al., Measurements of 25Mg+ ground-state hyperfine constants, Phys. Rev. A 100, 013409 (2019); "
        "arXiv:1903.04661."
    ),
    "ItanoWineland1981": (
        "W. M. Itano, D. J. Wineland, Precision measurement of the ground-state hyperfine constant of 25Mg+, Phys. Rev. A "
        "24, 1364 (1981): A(3s 2S1/2) = -596.254376(54) MHz and g_I/g_J = 9.299484(75)e-5."
    ),
    "Ansbacher1989": (
        "W. Ansbacher, Y. Li, E. H. Pinnington, Precision lifetime measurement for the 3p levels of Mg II using "
        "frequency-doubled laser radiation, Phys. Lett. A 139, 165 (1989), read in Kaur et al. 2025 Table VII."
    ),
    "KaurSahoo2025": (
        "J. Kaur, B. K. Sahoo et al., arXiv:2504.19515 (2025): relativistic coupled-cluster Mg II properties, theory "
        "A(3p 2P1/2) = -103.4(5) MHz and A(3p 2P3/2) = -19.31(5) MHz."
    ),
    "SurSahoo2005": (
        "C. Sur, B. K. Sahoo, R. K. Chaudhuri, B. P. Das, D. Mukherjee, Eur. Phys. J. D 32, 25 (2005), Table 2: unsigned "
        "coupled-cluster hyperfine constants of Mg II."
    ),
    # ---- Ba+ ----
    "Christensen2020": (
        "J. E. Christensen, D. Hucul, W. C. Campbell, E. R. Hudson, High-fidelity manipulation of a qubit enabled by a "
        "manufactured nucleus, npj Quantum Information 6, 35 (2020); arXiv:1907.13331: the 133Ba+ splittings 623(30) MHz "
        "(6p 2P3/2) and 83(30) MHz (5d 2D5/2)."
    ),
    "Knab1987": (
        "H. Knab, K.-D. Schupp, G. Werth, Precision measurement of the ground state hyperfine splitting of 133Ba+, "
        "Europhys. Lett. 4, 1361 (1987): 9 925 453 554.59(10) Hz."
    ),
    "Hucul2017": (
        "D. Hucul, J. E. Christensen, E. R. Hudson, W. C. Campbell, Spectroscopy of a synthetic trapped ion qubit, Phys. "
        "Rev. Lett. 119, 100501 (2017); arXiv:1705.09736, Table I."
    ),
    "BlattWerth1982": (
        "R. Blatt, G. Werth, Precision determination of the ground-state hyperfine splitting in 137Ba+ using the "
        "ion-storage technique, Phys. Rev. A 25, 1476 (1982): W_hfs/h = 8 037 741 667.69(36) Hz."
    ),
    "Villemoes1993": (
        "P. Villemoes, A. Arnesen, F. Heijkenskjoeld, A. Wannstroem, Isotope shifts and hyperfine structure of 130-138Ba II "
        "by fast ion beam laser spectroscopy, J. Phys. B 26, 4289 (1993)."
    ),
    "Lewty2013": (
        "N. C. Lewty, B. L. Chuah, R. Cazan, B. K. Sahoo, M. D. Barrett, Spectroscopy on a single trapped 137Ba+ ion for "
        "nuclear structure studies, Phys. Rev. A 88, 012518 (2013); arXiv:1305.4453."
    ),
    "Arnold2019": (
        "K. J. Arnold et al., Precision measurements of the 138Ba+ 6s-5d quadrupole transition, Phys. Rev. A 100, 032503 "
        "(2019); arXiv:1905.06523."
    ),
    "ZhangBa2020": (
        "Z. Zhang et al., Branching fractions for P3/2 decays in Ba+, Phys. Rev. A 101, 062515 (2020); arXiv:2003.02263."
    ),
    "Yu1997": (
        "N. Yu, W. Nagourney, H. Dehmelt, Radiative lifetime measurement of the Ba+ metastable D3/2 state, Phys. Rev. Lett. "
        "78, 4898 (1997)."
    ),
    "Marx1998": (
        "G. Marx, G. Tommaseo, G. Werth, Precise g_J- and g_I-factor measurements of Ba+ isotopes, Eur. Phys. J. D 4, 279 "
        "(1998): g_J(6s 2S1/2) = 2.00249192(3) on 138Ba+ and 135Ba+, read in Arnold et al., Phys. Rev. Lett. 124, 193001 "
        "(2020)."
    ),
    "Knoll1996": (
        "K. H. Knoell et al., Experimental g_J factor in the metastable 5D3/2 level of Ba+, Phys. Rev. A 54, 1199 (1996): "
        "0.7993278(3)."
    ),
    "ArnoldPRL2020": (
        "K. J. Arnold et al., Precision measurements of the 138Ba+ 6s 2S1/2 - 5d 2D5/2 clock transition, Phys. Rev. Lett. "
        "124, 193001 (2020); arXiv:1906.09150: g_J(5d 2D5/2) = 1.20036739(24)."
    ),
    # ---- several species ----
    "Ozeri2007": (
        "R. Ozeri et al., Errors in trapped-ion quantum gates due to spontaneous photon scattering, Phys. Rev. A 75, 042329 "
        "(2007), Table I."
    ),
    "Stone2019": (
        "N. J. Stone, Table of recommended nuclear magnetic dipole moments, IAEA INDC(NDS)-0794 (2019), and of nuclear "
        "electric quadrupole moments, INDC(NDS)-0833 (2021)."
    ),
    # ---- the readout presets ----
    "Myerson2008": (
        "A. H. Myerson et al., High-fidelity readout of trapped-ion qubits, Phys. Rev. Lett. 100, 200502 (2008); "
        "arXiv:0802.1684."
    ),
    "Burrell2010": (
        "A. H. Burrell, D. J. Szwer, S. C. Webster, D. M. Lucas, Scalable simultaneous multi-qubit readout with 99.99% "
        "single-shot fidelity, Phys. Rev. A 81, 040302(R) (2010); arXiv:0906.3304."
    ),
    "Acton2006": (
        "M. Acton et al., Near-perfect simultaneous measurement of a qubit register, Quantum Inf. Comput. 6, 465 (2006); "
        "arXiv:quant-ph/0511257."
    ),
    "Noek2013": (
        "R. Noek et al., High speed, high fidelity detection of an atomic hyperfine qubit, Opt. Lett. 38, 4735 (2013); "
        "arXiv:1304.3511."
    ),
    "Crain2019": (
        "S. Crain et al., High-speed low-crosstalk detection of a 171Yb+ qubit using superconducting nanowire single photon "
        "detectors, Commun. Phys. 2, 97 (2019); arXiv:1902.04059."
    ),
    "Egan2021": (
        "L. Egan et al., Fault-tolerant control of an error-corrected qubit, Nature 598, 281 (2021); arXiv:2009.11482."
    ),
    "Wineland1998": (
        "D. J. Wineland et al., Experimental issues in coherent quantum-state manipulation of trapped atomic ions, J. Res. "
        "NIST 103, 259 (1998); arXiv:quant-ph/9710025."
    ),
    # ---- databases, conventions and the plan ----
    "NIST_ASD_5_12": (
        "A. Kramida, Yu. Ralchenko, J. Reader, and NIST ASD Team (2024), NIST Atomic Spectra Database (ver. 5.12), "
        "https://physics.nist.gov/asd: level energies in cm^-1 and, where listed, Lande g factors."
    ),
    "NIST_AWIC": (
        "J. S. Coursey, D. J. Schwab, J. J. Tsai, R. A. Dragoset, Atomic Weights and Isotopic Compositions (NIST), "
        "https://physics.nist.gov/Comp: relative atomic masses of the isotopes."
    ),
    "Steck": "D. A. Steck, Quantum and Atom Optics, and the alkali D-line data notes: the conventions of Sections 4.5 and 13.",
    "PLAN_4_5_1": "PLAN.md Section 4.5.1, which uses the value without naming a primary source.",
    "PLAN_8_1": "PLAN.md Section 8.1, which carries the value without naming a primary source.",
    "PLAN_9_13": "PLAN.md's validation targets (Section 9), which state the value as a test input without naming a primary source.",
    "PLAN_background": "A PLAN.md convention or textbook statement tagged [background] (no source check).",
}
