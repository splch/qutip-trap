"""Section 9.11 row "Channel derivation": an app-side channel-replay run of a two-ion Bell circuit matches JOINT_EXACT
within the channel-derivation residual it reports; a deliberately hotter motional state widens the residual and the
discrepancy together."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from fixtures import BELL, FAST, SEED, SHOTS

from qutip_trap_app.core import rz, standard_recipe, yb171_chain
from qutip_trap_app.record import DeviceRef, LiveRun, Record, calibrate_for, execute, job_for_preset
from qutip_trap_app.replay import ChannelLibrary, conjugate_choi, frame_rotation, readout_polarity, replay
from qutip_trap_app.viewmodel.circuit import bloch_vectors, register_after, timeline
from qutip_trap_app.viewmodel.machine import histogram
from qutip_trap_app.viewmodel.numerics import numerics_panel


def test_frame_rotation_convention() -> None:
    u = rz(0.7)
    assert np.allclose(u @ u.conj().T, np.eye(2))
    r = frame_rotation((0, 1), {0: 0.3, 1: -0.2})
    assert r.shape == (4, 4) and np.allclose(r, np.kron(rz(0.3), rz(-0.2)))
    # the fully depolarizing channel is invariant under any frame rotation
    choi = np.eye(4, dtype=complex) / 4.0
    assert np.allclose(conjugate_choi(choi, u), choi)
    choi2 = np.eye(16, dtype=complex) / 16.0
    assert np.allclose(conjugate_choi(choi2, r), choi2)


def test_replay_matches_joint_exact_within_its_residual(
    bell: tuple[Record, LiveRun], bell_replay: tuple[Record, ChannelLibrary]
) -> None:
    joint, _ = bell
    rep, library = bell_replay
    assert rep.diagnostics.level == "CHANNEL_REPLAY" and rep.replay is not None and not rep.traces
    # the channels the schedule needed: gpi2 on each ion and the fully entangling ms, each with its frame covariance measured
    kinds = {e.kind for e in library.entries.values()}
    assert kinds == {"gpi2", "ms"} and set(library.covariance_by_kind) == {"gpi2", "ms"}
    for e in library.entries.values():
        assert e.covariance_residual is not None and e.covariance_residual < 1e-6, (
            e.key,
            e.covariance_residual,
        )
    assert rep.replay.residual_total > 0.0
    assert joint.results.final_state is not None and rep.results.final_state is not None
    p_joint = np.real(np.diag(joint.results.final_state))
    p_rep = np.real(np.diag(rep.results.final_state))
    discrepancy = float(np.max(np.abs(p_joint - p_rep)))
    assert discrepancy <= rep.replay.residual_total, (discrepancy, rep.replay.residual_terms)
    assert discrepancy > 0.0
    # the histogram agrees with the ideal one within statistics and the Bell physics reads the same at Level 1
    h = histogram(rep)
    assert (
        h.total_variation_to_target < 0.1
        and rep.results.probabilities["00"] + rep.results.probabilities["11"] > 0.97
    )
    gates = timeline(rep)
    ms = next(g for g in gates if g.name.value == "ms")
    after = register_after(rep, ms.index)
    assert "derived" in after.weights_note
    for vec in bloch_vectors(after.rho, 2).values():
        assert np.linalg.norm(vec) < 0.1
    assert ms.channel is not None and ms.channel.average_gate_infidelity < 1e-3
    panel = numerics_panel(rep)
    assert panel.badge.status == "not checked" and panel.derivation_residual is not None
    assert any("verify deeper" in x for x in panel.badge.checks_not_run)


def test_polarity_and_spam_come_from_the_device_and_table(bell_replay: tuple[Record, ChannelLibrary]) -> None:
    rep, _ = bell_replay
    preset = rep.job.device.build()
    assert readout_polarity(preset.device) == (1, 1), "171Yb+ direct fluorescence: |1> (F = 1) is bright"
    eps_b, eps_d = rep.device_card.spam["q0"]
    assert 1e-4 < eps_b < 5e-3 and 1e-4 < eps_d < 5e-3
    assert rep.readout.levels.shape == (SHOTS, 2)


@pytest.mark.slow
def test_hotter_motional_state_widens_residual_and_discrepancy_together(
    bell: tuple[Record, LiveRun], bell_replay: tuple[Record, ChannelLibrary]
) -> None:
    """The second half of the Section 9.11 row: three sideband-cooling pulses per order instead of the standard recipe's
    leave the gate modes hotter; the replay's residual bound grows with (2 nbar + 1) and so does its discrepancy against
    JOINT_EXACT."""
    cold_joint, cold_live = bell
    cold_rep, _ = bell_replay
    base = standard_recipe(yb171_chain(2).device, raman_pair=(0, 1))
    assert base.sideband is not None
    sideband = dataclasses.replace(
        base.sideband,
        pulses_per_order={order: min(k, 3) for order, k in base.sideband.pulses_per_order.items()},
    )
    hot_preset = yb171_chain(2, recipe=dataclasses.replace(base, sideband=sideband))
    hot_job, _ = job_for_preset(
        "yb171_chain",
        2,
        BELL,
        SHOTS,
        seed=SEED,
        options=dataclasses.replace(FAST, branch_weight_min=1e-2),
        detection_records=500,
        preset=hot_preset,
    )
    # the hotter device is passed to execute explicitly: its recipe is not a preset argument the reference could rebuild
    assert hot_job.device == DeviceRef(hot_preset.device.hash(), "yb171_chain", 2, {})
    hot_table = calibrate_for(hot_job, hot_preset.device)
    assert max(hot_table.nbar[m].value for m in (2, 3)) > max(
        cold_live.core_record.preparation.nbar[m] for m in (2, 3)
    ), "the gate modes are hotter"
    hot_joint, _live = execute(hot_job, hot_preset)
    hot_rep = replay(hot_job, hot_preset.device, hot_table, ChannelLibrary())
    assert hot_rep.replay is not None and cold_rep.replay is not None
    assert hot_rep.replay.residual_total > cold_rep.replay.residual_total
    assert hot_joint.results.final_state is not None and hot_rep.results.final_state is not None
    assert cold_joint.results.final_state is not None and cold_rep.results.final_state is not None
    hot_disc = float(
        np.max(
            np.abs(np.diag(hot_joint.results.final_state).real - np.diag(hot_rep.results.final_state).real)
        )
    )
    cold_disc = float(
        np.max(
            np.abs(np.diag(cold_joint.results.final_state).real - np.diag(cold_rep.results.final_state).real)
        )
    )
    assert hot_disc > cold_disc
    assert hot_disc <= hot_rep.replay.residual_total, (hot_disc, hot_rep.replay.residual_terms)
