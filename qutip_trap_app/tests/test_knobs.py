"""Changing the rf amplitude at Level 4 changes beta, nu, eta and the recalibrated gate
durations consistently with Section 4.1 and the pulse solvers; the device card updates without manual steps. Also the knob
layer's own rules (validation, the explicit-path scaling, the recipe re-derivation) and the device layer's consistency with
the run record it describes."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap_app import core, device_layer, knobs
from qutip_trap_app.record import DeviceRef, LiveRun, Record, calibrate_for, job_for_preset
from qutip_trap_app.viewmodel.physics import (
    cooling_view,
    crystal_view,
    gate_rows,
    knob_rows,
    layer_card_view,
    light_view,
    noise_view,
    readout_view,
    species_view,
    stale_status,
    trap_view,
)

TWO_PI = 2.0 * math.pi


def test_knob_catalogue_and_validation() -> None:
    preset = core.yb171_chain(2)
    table = knobs.knobs_for(preset.device)
    assert len(table) >= 30 and all(k.ledger_id for k in table.values())
    assert {k.page for k in table.values()} <= {
        "species",
        "trap",
        "crystal",
        "light",
        "noise",
        "cooling",
        "readout",
        "hamiltonian",
    }
    with pytest.raises(knobs.KnobError, match="unknown knob"):
        knobs.apply_overrides(preset, {"nope": 1.0})
    with pytest.raises(knobs.KnobError, match="outside the knob's range"):
        knobs.apply_overrides(preset, {"trap.rf_amplitude_scale": -1.0})
    with pytest.raises(knobs.KnobError, match="finite"):
        knobs.validate({"field.b_gauss": float("nan")}, preset.device)
    assert knobs.apply_overrides(preset, {}) is preset
    values = knobs.current_values(preset)
    assert values["field.b_gauss"] == 5.0 and values["trap.rf_amplitude_scale"] == 1.0
    assert "trap.rf_frequency_hz" not in values, "the published preset declares no rf record"


def test_rf_amplitude_scales_q_at_fixed_a_and_beta_follows_the_mathieu_equation() -> None:
    """q proportional to V_rf at fixed a (Section 4.1.1); beta by the monodromy; nu = beta Omega_rf/2; eta ~ 1/sqrt(omega)."""
    preset = core.yb171_chain(2)
    with pytest.raises(knobs.KnobError, match="rf drive frequency"):
        knobs.apply_overrides(preset, {"trap.rf_amplitude_scale": 1.1}, rederive_recipe=False)
    base = knobs.apply_overrides(preset, {"trap.rf_frequency_hz": 40e6}, rederive_recipe=False)
    scaled = knobs.apply_overrides(
        preset, {"trap.rf_frequency_hz": 40e6, "trap.rf_amplitude_scale": 1.1}, rederive_recipe=False
    )
    sp = base.device.crystal.species[0]
    m0, m1 = base.device.trap.mathieu(sp), scaled.device.trap.mathieu(sp)
    assert base.device.trap.omega_hz is not None and scaled.device.trap.omega_hz is not None
    omega_rf = TWO_PI * 40e6
    for k in range(2):  # the radial axes
        assert math.isclose(m1.q_effective[k], 1.1 * m0.q_effective[k], rel_tol=1e-9), (
            "q scales with the rf amplitude"
        )
        assert math.isclose(m1.diagonal_a[k], m0.diagonal_a[k], abs_tol=1e-12), (
            "a is the dc voltages' and stays"
        )
        beta = core.monodromy(m0.diagonal_a[k], 1.1 * m0.q_effective[k]).beta
        assert math.isclose(m1.beta[k], beta, rel_tol=1e-9), (
            "beta from the Mathieu equation, not proportional to V_rf"
        )
        assert math.isclose(scaled.device.trap.omega_hz[k], beta * omega_rf / 2 / TWO_PI, rel_tol=1e-9)
        # a < 0 radially (the axial dc confinement anti-confines the radial axes): beta^2 = a + q^2/2 + ... has a fixed
        # negative offset, so scaling q by 1.1 raises beta by MORE than 1.1, and nu grows faster than V_rf
        assert scaled.device.trap.omega_hz[k] > 1.1 * base.device.trap.omega_hz[k], (
            "a < 0 radially: nu grows faster than V_rf"
        )
        a_k, q_k = m0.diagonal_a[k], m0.q_effective[k]
        assert a_k < 0.0 < q_k
        leading = math.sqrt((a_k + (1.1 * q_k) ** 2 / 2) / (a_k + q_k**2 / 2))
        assert scaled.device.trap.omega_hz[k] / base.device.trap.omega_hz[k] == pytest.approx(
            leading, rel=5e-3
        ), "the lowest-order Mathieu expansion predicts the ratio to half a percent"
    assert m1.beta[2] == pytest.approx(m0.beta[2]), "the axial axis has q_z = 0 and does not move"
    assert scaled.device.trap.omega_hz[2] == base.device.trap.omega_hz[2]
    pair = knobs.entangling_pair(preset)
    assert pair is not None
    dk = base.device.beams[pair[0]].k_vector() - base.device.beams[pair[1]].k_vector()
    eta0 = base.device.crystal.lamb_dicke_matrix(dk, micromotion=None)
    eta1 = scaled.device.crystal.lamb_dicke_matrix(dk, micromotion=None)
    radial = [
        k for k, m in enumerate(base.device.crystal.modes) if m.family != "axial" and abs(eta0[0, k]) > 1e-6
    ]
    assert radial, "the Raman pair couples to the radial modes"
    for k in radial:
        w0, w1 = base.device.crystal.modes[k].omega_hz, scaled.device.crystal.modes[k].omega_hz
        assert w1 > w0
        assert eta1[0, k] / eta0[0, k] == pytest.approx(math.sqrt(w0 / w1), rel=1e-6), (
            "eta falls as 1/sqrt(omega_m)"
        )
    assert scaled.device.hash() != base.device.hash() != preset.device.hash()
    assert scaled.device.preparation is preset.device.preparation, "rederive_recipe=False keeps the recipe"


def test_device_ref_carries_overrides_and_rebuilds_hash_checked() -> None:
    ref = DeviceRef("", "yb171_chain", 2, {}, {"detector.window_s": 3e-5, "field.b_gauss": 6.0})
    preset = ref.build(check=False)
    assert preset.device.detector.window_s == 3e-5 and preset.device.field.B_gauss == 6.0
    ref2 = dataclasses.replace(ref, hash=preset.device.hash())
    assert ref2.build().device.hash() == ref2.hash
    assert ref2.cache_key() != DeviceRef("", "yb171_chain", 2, {}, {}).cache_key()
    circuit = core.Circuit(2, (core.Operation("h", (0,), ()),), (0, 1))
    job, built = job_for_preset("yb171_chain", 2, circuit, 10, overrides={"field.b_gauss": 6.0}, build=False)
    assert built is None and job.device.hash == "" and job.device.overrides == {"field.b_gauss": 6.0}


@pytest.fixture(scope="module")
def layer(bell: tuple[Record, LiveRun]) -> device_layer.DeviceLayer:
    record, live = bell
    preset = record.job.device.build()
    return device_layer.derive_device_layer(preset, table=live.table, sweeps=True)


def test_layer_agrees_with_the_record_it_describes(
    bell: tuple[Record, LiveRun], layer: device_layer.DeviceLayer
) -> None:
    record, live = bell
    assert layer.device_hash == record.device_hash and not layer.stale
    # the crystal page shows the modes the record's device card carries
    assert [m.omega_hz for m in layer.crystal.modes] == pytest.approx(
        [m.omega_hz for m in record.device_card.modes]
    )
    assert np.allclose(layer.crystal.positions_m, live.device.crystal.positions_m)
    # the closed-form solve (Section 4.4.3) is the surrogate's first step at the same beat-note rule and duration; the table
    # stores the waveform after the exact spot check (Section 7.8) rescaled its amplitude, so the played angle per mode is
    # the closed form's times one common factor, and the layer reports that comparison instead of hiding it
    g = layer.gates[0]
    wf = record.table.waveforms["0,1"]
    assert g.duration_s == pytest.approx(wf.duration_s)
    assert abs(g.waveform.chi_total_rad) == pytest.approx(math.pi / 4, rel=1e-9), (
        "the closed form hits pi/4 exactly"
    )
    assert g.table_chi_m == wf.chi_m
    assert abs(sum(wf.chi_m.values())) == pytest.approx(math.pi / 4, abs=1e-3), (
        "the exact check, within its tolerance"
    )
    assert g.table_chi_closed_form_m is not None and g.spot_check_amplitude_ratio is not None
    assert g.surrogate_error is not None
    ratios = [wf.chi_m[m] / g.table_chi_closed_form_m[m] for m in wf.chi_m]
    assert max(ratios) - min(ratios) < 1e-9, (
        "one amplitude factor for every mode: the exact check rescaled, not reshaped"
    )
    assert g.surrogate_error == pytest.approx(ratios[0] - 1.0, abs=1e-9)
    assert 1e-3 < abs(g.surrogate_error) < 0.05, (
        "the Debye-Waller correction of the exact check: percent-level at eta 0.08"
    )
    for m, chi_closed in g.table_chi_closed_form_m.items():
        assert chi_closed == pytest.approx(g.waveform.chi_m[m] * g.spot_check_amplitude_ratio**2, rel=1e-6), (
            "the closed-form angle scales as Omega^2 between the two amplitudes"
        )
    seg = wf.segments[0] if wf.segments else None
    assert seg is not None
    mu_table = next(iter(seg.detuning_hz.values()))
    assert mu_table.value is not None and abs(abs(mu_table.value) - g.mu_hz) < 1.0, (
        "the layer solves at the table's beat note"
    )
    assert g.residual_error is not None and g.residual_error < 1e-12
    # the readout page's exact count distributions agree with the table's sampled detection calibration to statistics
    ion = layer.readout.ions[0]
    eps_b_table = record.table.entries["detection['eps_B']"].value
    n_records = record.job.calibration.detection_records
    sigma = math.sqrt(max(eps_b_table, 1e-4) / n_records) + 1e-4
    assert abs(ion.at_window_eps_b - eps_b_table) < 4 * sigma
    assert ion.window_s == pytest.approx(layer.readout.window_s)
    assert ion.best_eps_b + ion.best_eps_d <= ion.at_window_eps_b + ion.at_window_eps_d + 1e-12
    assert ion.bright_pmf.sum() == pytest.approx(1.0, abs=1e-6) and ion.dark_pmf.sum() == pytest.approx(
        1.0, abs=1e-6
    )
    # the cooling page's final occupations are the run's preparation occupations
    assert layer.cooling is not None
    for m, nb in live.core_record.preparation.nbar.items():
        assert layer.cooling.final_nbar[m] == pytest.approx(nb, rel=1e-9)
    sb = layer.cooling.sidebands[0]
    assert sb.nbar_after_pulse[0] == pytest.approx(sb.nbar_start) and sb.nbar_after_pulse[-1] < 0.05
    assert sb.nbar_final_run > sb.nbar_after_pulse[-1], (
        "the repump recoil the staircase omits raises the run's own value"
    )
    # the light page's Rabi frequency is the table's seed
    rabi = layer.light.drives[0].rabi_hz
    assert rabi == pytest.approx(record.table.entries["rabi[(0, 2)]"].value, rel=1e-9)
    # the stability boundary closes near q = 0.908 on the a = 0 axis (Section 4.1.1)
    assert layer.trap.stability is not None and abs(layer.trap.stability.edge_q_at_a0 - 0.908) < 2e-3
    lower = layer.trap.stability.a_lower
    assert np.all(lower[np.isfinite(lower)] <= 1e-6), "the lower edge a_0(q) is at or below zero"
    # every page view resolves and every knob row belongs to its page
    for view in (
        species_view(layer),
        trap_view(layer),
        crystal_view(layer),
        light_view(layer),
        noise_view(layer),
        cooling_view(layer),
        readout_view(layer, record.table),
    ):
        assert view is not None
    assert gate_rows(layer, record.table)[0].table, "the table's waveform is shown beside the solution"
    for page in ("trap", "readout", "light", "noise", "cooling", "species"):
        assert all(r.knob.page == page for r in knob_rows(layer, page))
    assert "current" in stale_status(layer)


def test_downward_propagation_through_the_calibration_emulation(bell: tuple[Record, LiveRun]) -> None:
    """rf amplitude x1.1 -> beta, nu, eta change as Section 4.1 says; the recalibrated table's waveform closes at
    the new modes with a different amplitude; the device card (the layer's) updates without manual steps."""
    record, live = bell
    ref = dataclasses.replace(
        record.job.device, overrides={"trap.rf_amplitude_scale": 1.1, "trap.rf_frequency_hz": 40e6}
    )
    preset = ref.build(check=False)
    old_layer = device_layer.derive_device_layer(record.job.device.build(), table=live.table, sweeps=False)
    stale_layer = device_layer.derive_device_layer(
        preset, overrides=ref.overrides, table_record=record.table, sweeps=False
    )
    assert stale_layer.stale and "stale" in stale_status(stale_layer)
    assert (
        stale_layer.trap.beta is not None and stale_layer.trap.q is not None and old_layer.trap.beta is None
    )
    # the card's radial modes moved up with beta Omega_rf/2: more than 1.1x, since a < 0 radially (Section 4.1.1)
    w_old, w_new = old_layer.card.modes[3].omega_hz, stale_layer.card.modes[3].omega_hz
    assert 1.1 * w_old < w_new < 1.2 * w_old
    assert stale_layer.card.modes[0].omega_hz == pytest.approx(old_layer.card.modes[0].omega_hz), (
        "the axial COM stays"
    )
    g_old, g_new = old_layer.gates[0], stale_layer.gates[0]
    assert g_new.omega_hz != g_old.omega_hz and g_new.mu_hz > g_old.mu_hz
    # eta follows 1/sqrt(omega) times the micromotion factor C0 of the Mathieu record (Section 4.1.1): the published preset
    # has no rf record (pseudopotential, C0 = 1) and the edited device declares one, so C0 of the coupled modes' axis enters
    assert old_layer.trap.c0 is None and stale_layer.trap.c0 is not None
    for i in g_old.eta:
        for m, a, b, w0, w1 in zip(g_old.modes, g_old.eta[i], g_new.eta[i], g_old.omega_hz, g_new.omega_hz):
            axis = int(np.argmax(np.abs(stale_layer.crystal.modes[m].e_hat)))
            assert b / a == pytest.approx(math.sqrt(w0 / w1) * stale_layer.trap.c0[axis], rel=1e-6)
    assert g_new.residual_error is not None and g_new.residual_error < 1e-12, (
        "the solver closes the loops at the new modes"
    )
    assert g_new.peak_rabi_hz is not None and g_old.peak_rabi_hz is not None
    assert g_new.peak_rabi_hz != pytest.approx(g_old.peak_rabi_hz, rel=1e-3)
    # the recalibration job: the surrogate table for the new device, hash-bound, with the pair's waveform re-solved
    job, _p = job_for_preset(
        "yb171_chain",
        2,
        record.job.circuit.to_core(),
        20,
        seed=record.job.seed,
        options=live.options,
        detection_records=500,
        overrides=ref.overrides,
        preset=preset,
    )
    table = calibrate_for(job, preset.device)
    assert table.is_current_for(preset.device.hash()) and not table.is_current_for(record.device_hash)
    wf = table.waveform_for((0, 1))
    assert wf is not None and abs(abs(wf.chi_total_rad) - math.pi / 4) < 2e-3
    assert all(abs(a) < 1e-6 for a in wf.alpha_m.values()), "the recalibrated waveform closes the loops"
    ratios = [chi / g_new.waveform.chi_m[m] for m, chi in wf.chi_m.items()]
    assert max(ratios) - min(ratios) < 1e-9 and abs(ratios[0] - 1.0) < 1e-3, (
        "the immediate closed-form solution is what the table re-solves, up to the exact spot check's one amplitude factor"
    )
    fresh = device_layer.derive_device_layer(preset, overrides=ref.overrides, table=table, sweeps=False)
    assert not fresh.stale and fresh.table_hash == preset.device.hash()
    card = layer_card_view(fresh, None)
    assert card.overrides and any("V_rf" in r.value.detail for r in card.overrides)
