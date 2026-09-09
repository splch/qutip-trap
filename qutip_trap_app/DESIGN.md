# qutip-trap-app: design of the learning tool

The application of PLAN.md Section 14 shows one simulated trapped-ion quantum computer at five levels of abstraction, from a cloud customer's histogram down to the matrix elements the solver integrates. This document fixes how that ladder is made into a learning tool that a newcomer can use without help and a physicist can use without irritation. It governs milestones M11.2 to M11.4; milestone M11.1 (the run record, re-simulation, provenance index and view-models, this commit) is the layer everything below reads.

Two rules from the plan are also the two rules of the pedagogy. Every screen is a view over one run record, so nothing on any screen is an illustration: a histogram bar is a pile of recorded shots, a Bloch arrow is the reduced density matrix of the recorded state, a pulse is the scheduler's own tones (Section 14.1, rule 1). And every number carries its provenance chip and every result its convergence badge (rules 3 and 4), so the app teaches epistemic hygiene by construction: the learner sees which statements were verified against a source, which were corrected, and whether the simulation itself converged.

Evidence labels below follow the pedagogy skill: **verified** survived an adversarial three-verifier check against primary sources; **lead** is a quoted but unvoted claim; **UX** is a design-reference rule (NN/g, WCAG 2.2, Material 3, Apple HIG). The learning-science evidence is strongest for declarative material; where the app teaches procedures (reading a schedule, judging convergence) the same mechanics are used but the gains are not promised.

## 1. Who learns here, and what they must be able to do

The pedagogy design starts from four questions.

**Prior knowledge.** Unknown per person, and it flips the sign of instructional support (verified: high-assistance instruction helps low-prior-knowledge learners, d = +0.505, and harms high-prior-knowledge learners, d = -0.428; Tetzlaff et al. 2025). So the app asks once, on first launch, with three plain self-descriptions and a visible way to change the answer at any time:

| Answer | What it changes |
|---|---|
| "New to quantum computing" | explanations open at the one-sentence depth; the worked example runs first; plain labels first, the physics term on hover; numerics panel closed; chips collapsed to their glyph |
| "I know circuits, not the hardware" | Levels 0 and 1 as for a physicist, Levels 2 to 4 as for a newcomer; explanations open at the picture depth |
| "Physicist; I know the hardware" | explanations collapsed, opening at the equation depth; problems before examples; numerics panel and chips expanded; symbols first |

When the learner has not answered, the app assists (verified as the authors' rule: the reversal is asymmetric, so assist when in doubt). The choice is a preference, never a gate: every level and every panel is reachable at every setting.

**Retention target.** Learner-set in the Learn view, default 90 days, phrased as the task: "explain what happens between a circuit and its histogram to a colleague, three months from now". The review schedule is derived from it (Section 4, spacing).

**Kinds of knowledge.** Facts (what a shot is; the native gate set), concepts (superposition as an arrow; entanglement from a spin-dependent force; quanta of vibration; truncation), procedures (read a histogram against its target; read a pulse against the mode spectrum; judge a numerics panel), and discriminations (target against simulated; estimate against calibrated; resolved against frozen against dropped modes; verified against extracted). The mechanics differ by kind: block to build, interleave to discriminate (verified pooled effect g = 0.42 for inductive learning; the per-material direction is only directional).

**The delayed, unaided task.** Given a circuit the learner has not seen and a histogram bar that is off its target, follow the run from the bar to the Hamiltonian term responsible, with the explain panel closed, and name the device parameter that would change it. This is the six-click path of the M11 exit criterion made into the assessment. The app measures nothing else as "learning": in-session correctness and "that makes sense" are shown, labelled as in-session, and never reported as mastery (verified: in-session fluency is a poor proxy).

## 2. The zoom ladder is the learning ladder

Each level introduces exactly the concepts needed to read the next, and every thing on a screen that has an inside can be opened: that is what "dig deeper, literally" means here. The zoom-in targets are the plan's (Section 14.2); the concepts are the learning layer's (`viewmodel/learn.py`).

| Level | The question it answers | Concepts introduced | Open this to go deeper |
|---|---|---|---|
| 0 Machine | What did the machine return, and how does it compare with a perfect one? | shot, histogram, bit order, target against simulated, readout error | a bar opens its shots; a shot opens its photon counts; a gate name opens Level 1; the device card opens the Level 4 pages |
| 1 Circuit | What did each gate do to the qubits? | Bloch arrow, native gates, compilation, virtual Z, entanglement by MS | a gate opens its pulses at Level 2; a channel opens its tomography; the phase register opens the virtual-Z rule |
| 2 Schedule | What light hit which ion, when, and what did it talk to? | pulse, modes, tones and sidebands, loop closure, crosstalk | a pulse opens Level 3; a tone opens the Hamiltonian term it produces; a mode opens the crystal page |
| 3 Dynamics | What happened inside one pulse? | spin-dependent force, Fock states, Debye-Waller, quantum jumps, truncation and convergence | a term opens its matrix elements at Level 4; a jump opens its collapse operator |
| 4 Physics | Where do the numbers come from? | the Hamiltonian, Lamb-Dicke parameter, trap and Mathieu, atomic structure, noise as physics, calibration | any parameter re-derives every level above (Section 14.4) |

The ladder is also the app's answer to Hick's law (UX): the whole physics is never on one screen. Each level has one job, one primary action, and a small set of things to click.

## 3. Pedagogical mechanics

**Predict, then reveal** (verified: retrieval practice beats restudy, g about 0.5). Before a result appears, the app asks for it. Level 0: sketch the histogram by dragging bars, or pick one of three sketches, before Run reveals the target and the simulated bars; the sketch is scored against the record (total variation, and which bars miss by more than two error bars), never against a script. Level 1: point the Bloch arrow before the gate's state is drawn. Level 2: pick which mode the tones sit beside. Level 3: predict whether the loop closes. Every prompt is one tap to skip, and the skip is not recorded as wrong. The feedback names the task ("the sketch differs beyond the uncertainty on 01"), never the person (lead: Kluger and DeNisi 1996, feedback about the self attenuates the effect).

**Worked example first, then faded, then free** (verified: novices profit from full guidance that fades; experts from problems first). The Bell-state job is the worked example: a six-stop tour with every stop annotated (`learn.BELL_TOUR`). The faded version is a three-ion GHZ job with the same six stops and the annotations withheld until asked. The free version is "build your own circuit, predict, run, explain one off-target bar". A physicist starts at the free version and can open the tour from it.

**Three depths of explanation** (UX: progressive disclosure). Every concept explains itself in one plain sentence with no symbols, then as a picture ("what to look at on this screen and what it shows"), then as the equation the ledger states with the plan section and the source. The starting depth follows prior knowledge; the deeper button is always present. The equation depth is the same text the provenance chip opens, so the explanation and the number it explains share one source.

**Self-explanation with a computed reveal** (moderate utility: Dunlosky et al. 2013, thin evidence in 2013 rather than a small effect). At natural pauses the app asks for one sentence ("why are the 01 and 10 bars not exactly zero?") and then shows the simulator's own answer: the intrinsic error budget with its terms and the readout errors, every number a chip. The learner compares their sentence with the physics, and the app does not grade prose.

**Interleaved discrimination drills** (verified pooled effect; block when building, interleave when telling apart). The Learn view mixes short discrimination items in immediate succession: is this number an estimate or calibrated; is this chip verified or extracted; is this mode resolved, frozen or dropped; is this bar's deviation inside its error bar. Concept-building content stays blocked by level.

**Spacing** (verified: 47.3 % recall spaced against 36.7 % massed; the optimal gap is a declining share of the retention target, 20 to 40 % at one week and 5 to 10 % at one year, and overshooting costs little). The Learn view has a review tray. A concept becomes due when its last exposure is older than the long end of the gap computed from the learner's retention target (`learn.review_gap_days`), and the tray re-asks that concept's prompt without the explanation open. The mastery log keeps session accuracy and delayed unaided accuracy apart, and only the second is called evidence. The tray is a place the learner visits, never a notification; the app has no notifications.

**Every explanation ends in a prompt** (the pedagogy skill's rule). `learn.ExplainCard` carries its prompt, and a concept with no prompt fails the build (`tests/test_learn.py`).

**What the app refuses to do.** It does not optimize for in-session accuracy or perceived helpfulness (lead: an unguarded tutor raised practice scores and lowered unaided exam scores, Bastani et al. 2025). It does not draw anything it did not compute. It does not report a number without a tag or a result without a badge. It does not collect engagement metrics; the mastery log is local, exportable by the learner, and never uploaded.

## 4. Usability

**Routes and the zoom bar** (Section 14.6). `/job/{id}` is Level 0 and the home screen; `/job/{id}/circuit/{gate}`, `/job/{id}/schedule/{pulse}`, `/job/{id}/dynamics/{pulse}/{sample}` are Levels 1 to 3; `/device/{page}` are the Level 4 pages; `/learn` is the tour, the drills and the review tray. A breadcrumb zoom bar shows the path (job, gate, pulse, sample) and the level; zooming out is one click on any crumb, the Escape key, or Cmd/Ctrl and minus; zooming in is a click on the thing itself or Cmd/Ctrl and plus. Browser history works in the served mode because the routes are real.

**One primary action per screen** (UX: exactly one primary button per view). Level 0: Run. Levels 1 to 3: Zoom in on the selected item, with Verify deeper as the secondary action. Level 4: Apply (a parameter change), which immediately re-derives the analytic layer and marks the calibrated numbers stale. Learn: Start the tour, or Review, whichever is due.

**Response budgets** (UX: under 400 ms keeps flow; 1 to 10 s a working state; over 10 s progress and cancel), mapped to Section 14.7:

| Action | Budget | What the learner sees |
|---|---|---|
| open any level from the record | under 400 ms | the view, no indicator |
| zoom into a pulse at joint dimension at or below 256 | at or below 1 s | the recorded coarse trace at once, the fine trace replacing it |
| zoom into a pulse at larger dimension | seconds to minutes | recorded trace at once; a progress bar with a cancel button; the view fills in |
| verify deeper, jump ensemble, process matrix | background job | progress, cancel, and the result appearing beside the shallow one when done |
| Level 4 parameter change | immediate for the analytic layer | derived numbers update; calibrated numbers get a stale badge until the user starts the recalibration job (hours, Section 7.5) |

**Layout** (UX numbers). Material window classes: on a compact window the levels are a bottom bar of five labelled icons and the explain panel is a sheet; on medium and expanded windows a navigation rail with the five levels and Learn, the explain drawer on the right, the numerics panel as a collapsible strip at the bottom. Spacing on an 8 pt grid with 4 px half-steps; body text 16 px, reading measure at most 66 characters in the explain drawer; primary controls at least 44 px; every interactive element with hover, focus, active and disabled states; every color pair tested to 4.5:1 for text and 3:1 for outlines in both themes; dark surface #121212 with text at 87 % white and accents desaturated; motion 150 to 250 ms, honoring the platform's reduced-motion setting, never flashing. Charts follow the `dataviz` skill's rules: the categorical palette for series, sequential for heatmaps, the same colors for the same quantity on every level.

**No meaning by colour alone.** Tags are a glyph and a word (`✓ verified`, `✎ corrected`, `❝ extracted`, `○ background`, `⌗ recomputed`, `→ derived`, `? contested`); the badge is the word pass, not checked, or fail, with its reasons; the target bars are hatched and labelled, not merely a second colour.

**Plain language, then the term, then the symbol, then the chip.** Every label in the catalogue (`viewmodel/catalogue.py`) has a plain form ("how fast the ions vibrate together in this mode") and a term form ("mode frequency, omega_m/2pi"); frequencies are shown in Hz with the 2pi conversion one hover away; units are always printed. Errors say what happened and what to do ("the record names no device preset, so re-simulation is unavailable; run the job again to re-simulate").

**Empty, loading and error states are designed.** No job yet: the Bell preset ready to run with one sentence on what will appear. A view whose data the core does not expose says so and names the gap (`Record.core_gaps`) rather than showing a blank. A failed re-simulation shows the recorded trace and the error message beside it.

**Keyboard and screen readers.** A full tab path per screen, visible focus, Escape closes drawers and zooms out, every chart with a table alternative (the same `Shown` values), every chip with its text on focus.

## 5. Screens

**Level 0, the machine.** Left: the device card (species, ion count, native gate set, SPAM per qubit, gate errors labelled estimate or calibrated, the device hash and seed). Centre: the circuit editor with OpenQASM 2 and IonQ JSON import, the shot count, and Run. Right, after Run: the histogram with the target bars beside the simulated ones, the error bars, the total variation to the target, and the discarded-shot count. Clicking a bar lists its shots; clicking a shot shows its bits, the sampled levels, the heralds and, on the full readout path, the photon counts per ion with the threshold and window. The predict-then-reveal sketch sits where the histogram will appear.

**Level 1, the circuit.** The compiled native timeline on ion lanes; each gate shows its name, frame-applied phases, duration, target unitary, calibrated parameters (Rabi frequency or entangling angle with the table entry's status) and its error estimate or tomography summary. Below, the register after the selected gate: populations, Bloch arrows per ion, Pauli expectations, purity, and the fidelity to the target state so far; the phase register per ion with the stark increments. The compile report shows the block residuals.

**Level 2, the schedule.** Pulses on a time axis per ion, idle intervals and the measurement event marked; under it the mode spectrum with the selected pulse's tones drawn against it and each two-indexed detuning labelled carrier, red, blue or far; the waveform's segment table; the closure indicators per mode with chi_m, |alpha_m| and the residual against the drop threshold; crosstalk onto neighbours; the beams' directions.

**Level 3, the dynamics.** For one pulse and one sample: P1 per ion and the coherence |rho_01| against time; the concurrence and Pauli correlators for two qubits; the mean occupation per mode; the phase-space loops <a_m>(t) with how far each failed to close; the Fock distributions at the pulse's start and end (per-time distributions are a recorded core gap); the jumps; the quasi-static values drawn for the sample; the frozen spectators' Debye-Waller contribution. The numerics panel is open here by default for everyone, with the tolerance and cap re-checks one click away.

**Level 4, the physics.** The pages of Section 14.2: species, trap, crystal, light, noise, cooling, readout, and the Hamiltonian builder for the zoomed pulse with its terms and matrix elements. Every drawn element is computed; the one exception, a level diagram's vertical spacing, is labelled not to scale.

**Learn.** The prior-knowledge setting; the retention target; the tour (six stops with their prompts); the faded GHZ exercise; the free exercise; the discrimination drills; the review tray of due prompts; the mastery log with its two accuracies labelled.

**The numerics panel** (every level). Fidelity level, joint dimension and dims, caps per mode, mode classes, boundary population against the policy threshold, margins, integrator and tolerances, samples, trajectories and branches, wall time, the convergence report, and the badge with its reasons and the checks not run.

**The explain drawer** (every level). The Part II subsection governing the screen, the concept cards at the learner's depth with the deeper button, the chips of the quantities on screen, and the card's prompt at the end.

## 6. What M11.1 and M11.2 built

- `record.py`: the run record of Section 14.3 as frozen dataclasses of plain values and arrays, built from a `run()` result and the core's own `RunRecord`; the storage policy in its docstring (always stored, recomputed and cached, not available from the core).
- `codec.py`, `storage.py`: JSON plus `.npy` arrays in one zip with a digest; export and import are bitwise (Section 9.11, record round trip).
- `resim.py`: the engine rebuilt as `run()` built it; the joint state at every gate-step boundary by chaining (bitwise the recorded evolution on the Bell circuit); a zoom keyed by (step, sample, branch, stored points, options digest), so zooming twice recomputes once; the tolerance and cap re-checks of Section 5.5 and 9.9 for the badge.
- `provenance.py`: the index generated from the ledger and PLAN.md into `src/assets/provenance_index.json` (644 records, 82 Part II sections), with a `--check` for CI; chips and sections at run time.
- `viewmodel/`: the catalogue of displayed quantities with ledger ids; Levels 0 to 3; the numerics panel and badge; the learning layer (26 concepts, 3 depths, 28 prompts, the level plans, the spacing rule, the mastery log, the scoring functions, the six-stop tour).
- `core.py`: the one module that imports the core, with the recorded gaps: the concrete engine and two noise-sample keys are not re-exported by `qutip_trap.api` (feature requests filed as `CORE_GAPS`); `Traces` carries no per-time Fock distribution; `run()` reports no wall time and no per-gate register for GATE_LOCAL.
- `replay.py` and `replay_record.py` (M11.2): Level 0's default engine. Each gate kind is extracted once by the core's GATE_LOCAL tomography from the prepared motional state and conjugated to the played phase; the frame covariance of that conjugation is measured per kind (1.5e-10 for GPi2 and 4.9e-10 for MS on the example device) and joins the residual; a hotter motional state widens the residual and the discrepancy together.
- `verify.py` (M11.2): the verify-deeper action; `workers.py`: the worker process with progress events, the live state it keeps per record, and a hard cancel.
- `views/` (M11.2): the shell (rail, breadcrumb zoom bar, Escape and Cmd/Ctrl plus and minus, explain drawer, numerics strip, the first-launch question), Level 0 to 2, Learn, and placeholders that say what M11.3 brings. Every number on screen is a `Shown` rendered with its chip. The pages scroll; the explain cards, the numerics strip and the unitary and Pauli tiles keep their open state across re-renders; a run over a few seconds shows a live elapsed time and a Cancel (Section 4, response budgets); the prediction is asked before every run of a changed circuit and scored beside that run's histogram; the learner (prior knowledge, retention target, mastery log, with skips recorded as exposures) persists on the device through Flet's `SharedPreferences`, which is what lets the review tray of Section 3 come due across launches.

## 7. What counts as success

- The Section 9.11 rows M11.1 owns pass: coarse-graining identity to 1e-12, record round trip bitwise, re-simulation cache, convergence badge, provenance coverage.
- The delayed, unaided six-click task, tried by a newcomer at least one review gap after the tour, with the explain panel closed. This is the only measure the app will report as learning.
- No engagement metrics. Session accuracy is displayed as such and never aggregated.

## 8. Decisions left to the author

- The three prior-knowledge wordings above, and whether a fourth ("teaching this to others") should unlock the tour's annotations as speaker notes.
- The default retention target (90 days here) and whether the review tray may ever be surfaced outside the Learn view.
- Whether the record file format (`qutip-trap-app/record/1`) is published for other tools, which would freeze it.

## 9. Milestones

M11.2 built Levels 0 to 2 and the shell over these view-models, the channel replay engine and the verify-deeper action. M11.3 builds Level 3 with re-simulation and the Level 4 pages with downward propagation and stale badges. M11.4 adds the validation-suite presets with published numbers beside simulated ones, the explain drawer's content for every Part II subsection, the `flet test` navigation and run-flow tests (the six-click path), the rest of the Learn view of Section 5 (the faded GHZ exercise, the free exercise, the discrimination drills, the mastery log's two accuracies), keyboard-focusable chips, desktop bundles and documentation.
