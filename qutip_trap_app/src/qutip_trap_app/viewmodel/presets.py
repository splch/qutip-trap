"""The published-experiment presets (PLAN.md Section 14.5) as pure data: which Section 9 entries the app runs, the
published numbers each places beside the simulated ones, and the comparison rule.

Every published value carries a ``published.*`` ledger record whose tag is the Section 9 row's tag for the source, and every
simulated value the ``anchor.*`` record of the check that recomputed it. A preset also says whether the simulated number is
expected to agree: Section 9 reports a consistency anchor against the published uncertainty or 20 percent, and the ledger's
corrected forms name the cases (Kirchmair's fidelity, Myerson's optimum) where the first-principles number is not the
measured one. The computations live in :mod:`qutip_trap_app.presets`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from qutip_trap_app.viewmodel.catalogue import Shown

PresetKind = Literal["experiment", "circuit"]

AGREEMENT_FRACTION = 0.2
"""Section 9's default band: the published uncertainty (``SIGMAS`` sigma) or 20 percent, whichever is wider."""

SIGMAS = 2.0


@dataclass(frozen=True)
class PublishedValue:
    """One published number: what the paper says, how uncertain, and which simulated key it is compared with."""

    key: str
    label: str
    value: float
    uncertainty: float | None
    unit: str
    quantity_published: str
    """Catalogue id of the published number (a ``published.*`` ledger record)."""
    quantity_simulated: str
    """Catalogue id of the simulated number (the ``anchor.*`` of the check that recomputed it)."""
    simulated_key: str
    """Which entry of ``PresetResult.simulated`` this value is compared with."""
    expect_agreement: bool = True
    why_not: str = ""
    """When the simulated number is not expected to reproduce the published one, the ledger's reason in one sentence."""
    scale: float = 1.0
    """A display factor (1e6 for an error per gate quoted in units of 1e-6)."""


@dataclass(frozen=True)
class PresetSpec:
    id: str
    title: str
    question: str
    """What the learner will be able to answer after running it."""
    section: str
    row: str
    source: str
    published_ledger_id: str
    simulated_ledger_id: str
    concept_id: str
    """The concept the explain drawer opens beside the result."""
    values: tuple[PublishedValue, ...]
    duration: str
    method: str
    """One sentence on what the simulator computes (shown in Details)."""
    kind: PresetKind = "experiment"
    level: int = 4
    # a circuit preset: what Level 0 loads and runs
    circuit_text: str = ""
    shots: int = 0
    n_ions: int = 0
    preset_kwargs: dict[str, float] = field(default_factory=dict)


def _v(
    key: str,
    label: str,
    value: float,
    uncertainty: float | None,
    unit: str,
    quantity: str,
    simulated_key: str,
    *,
    expect_agreement: bool = True,
    why_not: str = "",
    scale: float = 1.0,
) -> PublishedValue:
    return PublishedValue(
        key,
        label,
        value,
        uncertainty,
        unit,
        f"{quantity}_published",
        f"{quantity}_simulated",
        simulated_key,
        expect_agreement,
        why_not,
        scale,
    )


BELL_QASM_4000 = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'
)
GHZ3_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\n'
    "measure q -> c;\n"
)

# fmt: off
PRESETS: dict[str, PresetSpec] = {p.id: p for p in (
    PresetSpec(
        id="harty_2014", title="Harty 2014: one error in a million",
        question="why a microwave gate's error is set by its timing and detuning, not by the atom",
        section="9.2", row="Microwave RB", source="Harty et al. 2014",
        published_ledger_id="published.harty_2014_epg", simulated_ledger_id="anchor.m2.harty_randomized_benchmarking",
        concept_id="native_gate", level=1, duration="about 5 s",
        method=(
            "randomized benchmarking as a product of 2 x 2 propagators: 12.1 us pi/2 pulses, 14 us dead times, +4.5 Hz "
            "detuning and a 5e-4 pulse-area error over 16 sets of 32 sequences of 2000 gates (Section 4.3.3)"
        ),
        values=(
            _v("epg_measured", "error per gate, measured", 1.0e-6, 0.3e-6, "", "harty_epg", "epg", scale=1e6),
            _v("epg_model", "error per gate, the paper's own model", 0.81e-6, 0.14e-6, "", "harty_epg", "epg", scale=1e6),
            _v("epg_rabi", "pulse-area error alone", 0.3e-6, None, "", "harty_epg", "epg_rabi_only", scale=1e6),
            _v("epg_detuning", "detuning alone", 0.7e-6, None, "", "harty_epg", "epg_detuning_only", scale=1e6),
        ),
    ),
    PresetSpec(
        id="james_1998", title="James 1998: where the ions sit",
        question="how the equilibrium spacing and the axial mode frequencies follow from the Coulomb repulsion alone",
        section="9.1", row="Equilibrium positions; Axial modes", source="James 1998",
        published_ledger_id="published.james_1998_axial_modes", simulated_ledger_id="anchor.trap.james_spectrum",
        concept_id="mode", duration="under a second",
        method=(
            "Newton iteration for the equilibrium of N equal charges in a harmonic well, then the eigenvalues of the "
            "dimensionless axial Hessian (Sections 4.1.2, 4.1.3)"
        ),
        values=(
            _v("u_n2", "outer ion position, two ions (units of l)", (0.5) ** (2.0 / 3.0), None, "", "james_u", "u_n2"),
            _v("u_n3", "outer ion position, three ions (units of l)", (1.25) ** (1.0 / 3.0), None, "", "james_u", "u_n3"),
            _v("mu2", "stretch mode, (nu_2/nu_z)^2, any N", 3.0, None, "", "james_mu", "mu2_n5"),
            _v("mu3_n3", "third axial mode, three ions", 29.0 / 5.0, None, "", "james_mu", "mu3_n3"),
            _v("mu3_n10", "third axial mode, ten ions", 5.841, 0.0005, "", "james_mu", "mu3_n10"),
        ),
    ),
    PresetSpec(
        id="monroe_1995", title="Monroe 1995: how cold Doppler cooling gets",
        question="why Doppler cooling stops at a few quanta and what sets the number",
        section="9.3", row="Doppler anchors", source="Monroe et al. 1995",
        published_ledger_id="published.monroe_1995_doppler", simulated_ledger_id="anchor.m3.monroe_doppler_triple",
        concept_id="cooling_ladder", duration="under a second",
        method=(
            "the semiclassical force model E/(hbar nu) - 1/2 with isotropic emission, and Stenholm's A_+- rate "
            "coefficients for a weak beam along each mode, at Gamma/2pi = 19.4 MHz and Delta = -30 MHz (Sections 4.2.1, 4.2.2)"
        ),
        values=(
            _v("theory_112", "the paper's theoretical occupation, 11.2 MHz mode", 0.484, None, "", "monroe_nbar", "force_112"),
            _v("nbar_112", "measured occupation, 11.2 MHz mode", 0.47, 0.05, "", "monroe_nbar", "force_112"),
            _v("nbar_112_rate", "measured occupation against the rate framework, 11.2 MHz", 0.47, 0.05, "", "monroe_nbar",
               "rate_112", expect_agreement=False,
               why_not=(
                   "at nu/Gamma = 0.58 the semiclassical force model that gives 0.484 is outside its regime; the rate "
                   "framework that is valid there gives 0.53 for a weak beam along the mode, and the paper's three-beam "
                   "geometry is not in the plan (Section 9.3, regime [corrected])"
               )),
            _v("nbar_182", "measured occupation, 18.2 MHz mode", 0.30, None, "", "monroe_nbar", "rate_182",
               expect_agreement=False,
               why_not="the triple is not reproduced by one beam; the x mode comes out twice too hot and the z mode 30 percent too cold"),
            _v("nbar_298", "measured occupation, 29.8 MHz mode", 0.18, None, "", "monroe_nbar", "rate_298",
               expect_agreement=False, why_not="the triple is not reproduced by one beam (Section 9.3)"),
        ),
    ),
    PresetSpec(
        id="roos_2000", title="Roos 2000: how hard a photon kicks a calcium ion",
        question="how the Lamb-Dicke parameter follows from the wavelength, the mass and the trap frequency",
        section="9.3", row="40Ca+ Lamb-Dicke recomputation", source="Roos 2000 (thesis)",
        published_ledger_id="published.roos_2000_lamb_dicke", simulated_ledger_id="anchor.ca40.lamb_dicke_729",
        concept_id="lamb_dicke", duration="under a second",
        method="eta = k x0 with x0 = sqrt(hbar/(2 m omega)) for 40Ca+ at 2 pi x 1 MHz (Section 4.1.7, conv.lamb_dicke)",
        values=(
            _v("eta_729", "Lamb-Dicke parameter at 729 nm", 0.096, 0.0005, "", "roos_eta", "eta_729"),
            _v("eta_393", "Lamb-Dicke parameter at 393 nm", 0.179, 0.0005, "", "roos_eta", "eta_393"),
        ),
    ),
    PresetSpec(
        id="kirchmair_2009", title="Kirchmair 2009: an entangling gate on a hot ion",
        question="why the Molmer-Sorensen gate barely cares how hot the motion is, and what does limit it",
        section="9.4", row="Thermal populations", source="Kirchmair et al. 2009",
        published_ledger_id="published.kirchmair_2009_thermal_ms", simulated_ledger_id="anchor.m4.kirchmair_ca40",
        concept_id="spin_dependent_force", level=3, duration="under a second",
        method=(
            "Kirchmair's exact single-loop propagator: alpha(t) = (eta Omega/2 eps)(e^{i eps t} - 1), the populations "
            "of Eq. 14 on a thermal mode, and the Debye-Waller angle spread |sum_n P_n e^{-i 4 chi eta^2 n}| at "
            "eta = 0.044, nu/2pi = 1.232 MHz, t_g = 50 us (Section 4.4.1)"
        ),
        values=(
            _v("contrast", "parity contrast at nbar = 20", 0.964, None, "", "kirchmair_contrast", "contrast_nbar20"),
            _v("fidelity_50", "Bell-state fidelity at 50 us", 0.993, 0.001, "", "kirchmair_fidelity",
               "fidelity_first_principles", expect_agreement=False,
               why_not=(
                   "the first-principles terms (residual displacement, Debye-Waller) at nbar = 0 are below 1e-5, three "
                   "orders under the measurement; the paper's own budget is technical: laser frequency noise 2e-3, "
                   "heating, a 1.4 percent Rabi error (Section 4.4.7)"
               )),
        ),
    ),
    PresetSpec(
        id="myerson_2008", title="Myerson 2008: reading a calcium qubit in 420 microseconds",
        question="why the readout error has a best window and a best threshold",
        section="9.5", row="40Ca+ optimum", source="Myerson et al. 2008",
        published_ledger_id="published.myerson_2008_readout", simulated_ledger_id="anchor.m5.myerson_optimum_and_recursion",
        concept_id="readout_rates", duration="about 5 s",
        method=(
            "the exact bright and dark count distributions of Myerson's apparatus (55800 detected counts per second, 442 "
            "background, 1.168 s shelf lifetime) at every window, the best threshold at each (Sections 8.2, 8.3)"
        ),
        values=(
            _v("eps_at_point", "error at n_c = 5.5, t_b = 420 us", 1.8e-4, 0.1e-4, "", "myerson_eps", "eps_at_point",
               expect_agreement=False, scale=1e4,
               why_not=(
                   "the exact chain with Poisson statistics gives 1.37e-4 there; the measured optimum sits at a higher "
                   "threshold and a longer window because the PMT dark counts are non-Poissonian (about 20 percent of "
                   "eps_D is cosmic rays), which the model does not carry"
               )),
            _v("window_opt", "the optimum window", 420e-6, None, "s", "myerson_window", "window_opt", expect_agreement=False,
               why_not="the ideal-Poisson chain's optimum is at 320 us and n_c = 3.5 (anchor.m5.myerson_optimum_and_recursion)"),
        ),
    ),
    PresetSpec(
        id="crain_2019", title="Crain 2019: reading ytterbium with a superconducting detector",
        question="how the readout error of this app's own detector compares with the apparatus it was modelled on",
        section="9.5", row="SNSPD operating point", source="Crain et al. 2019",
        published_ledger_id="published.crain_2019_snspd",
        simulated_ledger_id="anchor.m5.crain_corrections_and_operating_point",
        concept_id="spam", duration="about 5 s",
        method=(
            "the exact count distributions from Crain's measured rates (472 kcps detected, R_d = 341 Hz, R_b = 16.4 Hz, "
            "4.2 cps background, eps_sys = 4.356 percent) against the window, best threshold at each (Section 8.3)"
        ),
        values=(
            _v("infidelity", "readout error, first-photon protocol", 6.9e-4, 0.6e-4, "", "crain_eps", "eps_min", scale=1e4),
            _v("window", "average detection time of the protocol", 11e-6, None, "s", "crain_window", "window_opt",
               expect_agreement=False,
               why_not=(
                   "11 us is the average time the stop-on-first-photon protocol runs; a fixed-window threshold reaches its "
                   "minimum at 20 to 25 us (the corrected form of anchor.m5.crain_corrections_and_operating_point)"
               )),
        ),
    ),
    PresetSpec(
        id="bell_check", title="The Bell state, every piece of physics on",
        question="how far a two-ion Bell state lands from the ideal one on this machine, and why",
        section="9.6", row="Two-ion Bell state with all physics on", source="the reference run",
        published_ledger_id="anchor.m6.bell_state", simulated_ledger_id="anchor.m6.bell_state",
        concept_id="target_vs_simulated", kind="circuit", level=0, duration="about 20 s (a full JOINT_EXACT run)",
        method="the two-ion Bell circuit through run() at the full engine, compared with the reference run's numbers",
        circuit_text=BELL_QASM_4000, shots=400, n_ions=2,
        values=(
            _v("p00", "P(00), the reference run's 4000 shots", 0.5015, 0.0079, "", "bell_check_probability", "p00"),
            _v("p11", "P(11), the reference run's 4000 shots", 0.4965, 0.0079, "", "bell_check_probability", "p11"),
            _v("infidelity", "register infidelity 1 - F", 2.336e-3, None, "", "bell_check_infidelity", "infidelity",
               scale=1e3),
        ),
    ),
    PresetSpec(
        id="ghz_three", title="Three ions, one GHZ state",
        question="what changes when a third ion joins: more modes, a frozen zigzag, a smaller register fidelity",
        section="9.6", row="Three- and four-ion GHZ", source="the reference run",
        published_ledger_id="anchor.m10.ghz_fidelity", simulated_ledger_id="anchor.m10.ghz_fidelity",
        concept_id="entanglement_by_ms", kind="circuit", level=0,
        duration="minutes (joint dimension about 1100; runs in the background)",
        method=(
            "the three-ion GHZ circuit on the three-ion example device (2.0 um addressing waist) through run() at the full "
            "engine, compared with the reference run's numbers"
        ),
        circuit_text=GHZ3_QASM, shots=400, n_ions=3, preset_kwargs={"address_waist_m": 2.0e-6},
        values=(
            _v("p000", "P(000), the reference run's 400 shots", 0.545, 0.025, "", "ghz_check_probability", "p000"),
            _v("p111", "P(111), the reference run's 400 shots", 0.4475, 0.025, "", "ghz_check_probability", "p111"),
        ),
    ),
)}
# fmt: on
"""The presets the app ships, in the order the Learn view lists them (Sections 9.1 to 9.6)."""


def experiment_presets() -> tuple[PresetSpec, ...]:
    return tuple(p for p in PRESETS.values() if p.kind == "experiment")


def circuit_presets() -> tuple[PresetSpec, ...]:
    return tuple(p for p in PRESETS.values() if p.kind == "circuit")


def ledger_ids() -> frozenset[str]:
    return frozenset(i for p in PRESETS.values() for i in (p.published_ledger_id, p.simulated_ledger_id))


def catalogue_ids() -> frozenset[str]:
    return frozenset(
        q for p in PRESETS.values() for v in p.values for q in (v.quantity_published, v.quantity_simulated)
    )


# ---- results and comparisons -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesRecord:
    """One curve of a preset's chart (plain arrays, picklable across the worker boundary)."""

    label: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class ChartRecord:
    title: str
    x_title: str
    y_title: str
    series: tuple[SeriesRecord, ...]
    log_x: bool = False
    log_y: bool = False
    markers: tuple[tuple[float, str], ...] = ()
    """Vertical markers (x, label): where the published point sits."""
    bars: bool = False
    """Draw the (single) series as bars rather than a line (Harty's per-set histogram)."""


@dataclass(frozen=True)
class PresetResult:
    """What the worker computed for one experiment preset: the simulated numbers by key, their uncertainties where the
    computation has one, the charts, and the parameters it used."""

    preset_id: str
    simulated: dict[str, float]
    simulated_uncertainty: dict[str, float]
    charts: tuple[ChartRecord, ...]
    parameters: dict[str, float]
    notes: tuple[str, ...]
    wall_time_s: float


@dataclass(frozen=True)
class Comparison:
    """One published number beside its simulated one, each a :class:`Shown` with its own chip, and the verdict."""

    label: str
    published: Shown
    simulated: Shown
    published_uncertainty: float | None
    simulated_uncertainty: float | None
    within: bool
    """Inside the Section 9 band: ``SIGMAS`` combined sigmas or ``AGREEMENT_FRACTION`` of the published value."""
    expect_agreement: bool
    verdict: str
    """One short line: "agrees to 2.9 %" or "differs by 43 % (4.3 sigma), outside the Section 9 band"."""
    why_not: str
    scale: float
    unit: str


def compare(spec: PresetSpec, result: PresetResult) -> tuple[Comparison, ...]:
    """Every published value of the preset beside the simulated number it names."""
    out: list[Comparison] = []
    for v in spec.values:
        if v.simulated_key not in result.simulated:
            continue
        sim = float(result.simulated[v.simulated_key])
        sim_unc = result.simulated_uncertainty.get(v.simulated_key)
        diff = sim - v.value
        combined = math.sqrt((v.uncertainty or 0.0) ** 2 + (sim_unc or 0.0) ** 2)
        within = abs(diff) <= max(SIGMAS * combined, AGREEMENT_FRACTION * abs(v.value))
        rel = abs(diff) / abs(v.value) if v.value != 0.0 else float("inf")
        sig = f" ({abs(diff) / combined:.1f} sigma)" if combined > 0.0 else ""
        if not v.expect_agreement:
            verdict = f"not a first-principles prediction of this number: differs by {rel:.1%}{sig}"
        elif within:
            verdict = f"agrees to {rel:.1%}{sig}"
        else:
            verdict = f"differs by {rel:.0%}{sig}, outside the Section 9 band"
        out.append(
            Comparison(
                label=v.label,
                published=Shown(v.quantity_published, v.value, f"{spec.source}: {v.label}"),
                simulated=Shown(v.quantity_simulated, sim, f"this machine: {v.label}"),
                published_uncertainty=v.uncertainty,
                simulated_uncertainty=sim_unc,
                within=within,
                expect_agreement=v.expect_agreement,
                verdict=verdict,
                why_not=v.why_not,
                scale=v.scale,
                unit=v.unit,
            )
        )
    return tuple(out)


def circuit_comparisons(
    spec: PresetSpec,
    probabilities: dict[str, float],
    error_bars: dict[str, float],
    register_fidelity: float | None,
) -> tuple[Comparison, ...]:
    """A circuit preset's reference numbers beside the run's own histogram and register fidelity (Level 0)."""
    simulated = {f"p{key}": float(p) for key, p in probabilities.items()}
    unc = {f"p{key}": float(error_bars.get(key, 0.0)) for key in probabilities}
    if register_fidelity is not None:
        simulated["infidelity"] = 1.0 - float(register_fidelity)
    return compare(spec, PresetResult(spec.id, simulated, unc, (), {}, (), 0.0))
