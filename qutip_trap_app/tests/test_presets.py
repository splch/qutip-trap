"""Section 9.11 row "Presets": each Section 9 preset loads, runs and displays the published number beside the simulated one
with the correct tag (PLAN.md Section 14.5; DESIGN.md Section 10). The published value's chip is the Section 9 row's tag for
the source, the simulated value's chip is the anchor of the check that recomputed it, and the verdicts agree with the ledger's
own judgments (the cases it marks as not first-principles predictions come out as such)."""

from __future__ import annotations

import pytest

from qutip_trap_app import presets
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.viewmodel import presets as vm
from qutip_trap_app.viewmodel.catalogue import CATALOGUE


@pytest.fixture(scope="module")
def index() -> ProvenanceIndex:
    return ProvenanceIndex.load()


def test_every_preset_is_well_formed_and_tagged(index: ProvenanceIndex) -> None:
    assert not index.missing(vm.ledger_ids())
    assert vm.catalogue_ids() <= set(CATALOGUE)
    for spec in vm.PRESETS.values():
        assert spec.question and spec.method and spec.duration and spec.row and spec.source
        index.section(spec.section)
        published = index.chip(spec.published_ledger_id)
        simulated = index.chip(spec.simulated_ledger_id)
        if spec.kind == "experiment":
            assert published.id.startswith("published."), spec.id
            assert published.tag == "verified", (
                f"{spec.id}: the published number carries the Section 9 row's tag"
            )
            assert simulated.id.startswith("anchor."), spec.id
            assert simulated.tag in ("recomputed here", "corrected"), spec.id
            assert spec.id in presets.RUNNERS
        else:
            assert spec.circuit_text and spec.shots > 0 and spec.n_ions >= 2
        for v in spec.values:
            pub = CATALOGUE[v.quantity_published].ledger_id
            if spec.kind == "experiment":
                assert pub.startswith("published.") and index.chip(pub).tag == "verified", v.key
            assert v.expect_agreement or v.why_not, f"{spec.id}.{v.key}: a non-prediction says why"
    assert {p.kind for p in vm.PRESETS.values()} == {"experiment", "circuit"}
    assert len(vm.experiment_presets()) == 7 and len(vm.circuit_presets()) == 2


@pytest.mark.parametrize("preset_id", sorted(presets.RUNNERS))
def test_experiment_preset_runs_and_compares(preset_id: str) -> None:
    spec = vm.PRESETS[preset_id]
    stages: list[str] = []
    result = presets.run_preset(preset_id, lambda stage, _f, _m: stages.append(stage))
    assert result.preset_id == preset_id and result.wall_time_s >= 0.0 and stages
    comparisons = vm.compare(spec, result)
    assert len(comparisons) == len(spec.values), "every published value found its simulated number"
    for c in comparisons:
        assert c.published.value is not None and c.simulated.value is not None
        assert c.verdict
        assert CATALOGUE[c.published.quantity].ledger_id.startswith("published.")
        assert CATALOGUE[c.simulated.quantity].ledger_id.startswith("anchor.")
    for ch in result.charts:
        assert ch.series and all(s.x.size == s.y.size for s in ch.series)


def test_the_verdicts_agree_with_the_ledger() -> None:
    """The anchors' own judgments: Harty, James, Roos, Crain and the Kirchmair contrast agree; Myerson's optimum and
    Kirchmair's fidelity are not first-principles predictions and say so."""
    by_key = {}
    for pid in (
        "harty_2014",
        "james_1998",
        "roos_2000",
        "kirchmair_2009",
        "myerson_2008",
        "crain_2019",
        "monroe_1995",
    ):
        result = presets.run_preset(pid)
        for c in vm.compare(vm.PRESETS[pid], result):
            by_key[(pid, c.label)] = c
    assert by_key[("harty_2014", "error per gate, the paper's own model")].within
    assert abs(by_key[("harty_2014", "error per gate, measured")].simulated.value - 0.77e-6) < 0.15e-6
    assert all(c.within for (pid, _), c in by_key.items() if pid == "james_1998")
    assert all(c.within for (pid, _), c in by_key.items() if pid == "roos_2000")
    assert by_key[("crain_2019", "readout error, first-photon protocol")].within
    contrast = by_key[("kirchmair_2009", "parity contrast at nbar = 20")]
    assert contrast.within and abs(contrast.simulated.value - 0.9923) < 1e-3
    fidelity = by_key[("kirchmair_2009", "Bell-state fidelity at 50 us")]
    assert (
        not fidelity.expect_agreement
        and fidelity.why_not
        and "not a first-principles prediction" in fidelity.verdict
    )
    myerson = by_key[("myerson_2008", "error at n_c = 5.5, t_b = 420 us")]
    assert abs(myerson.simulated.value - 1.37e-4) < 0.05e-4 and not myerson.expect_agreement
    monroe = by_key[("monroe_1995", "the paper's theoretical occupation, 11.2 MHz mode")]
    assert monroe.within and abs(monroe.simulated.value - 0.484) < 0.005


def test_circuit_comparisons_read_the_run() -> None:
    spec = vm.PRESETS["bell_check"]
    comps = vm.circuit_comparisons(
        spec, {"00": 0.51, "11": 0.48, "01": 0.005, "10": 0.005}, {"00": 0.025, "11": 0.025}, 0.9975
    )
    labels = {c.label: c for c in comps}
    assert labels["P(00), the check script's 4000 shots"].within
    assert abs(labels["register infidelity 1 - F"].simulated.value - 2.5e-3) < 1e-9
    assert vm.PRESETS["ghz_three"].preset_kwargs == {"address_waist_m": 2.0e-6}


def test_run_preset_refuses_unknown_and_circuit_presets() -> None:
    with pytest.raises(KeyError):
        presets.run_preset("nope")
    with pytest.raises(ValueError, match="circuit preset"):
        presets.run_preset("bell_check")
