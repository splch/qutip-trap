"""The verify-deeper action (Section 14.5): the replay against the next engine within its residual; JOINT_EXACT re-checks."""

from __future__ import annotations

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.replay import ChannelLibrary
from qutip_trap_app.verify import deeper_level, verify_deeper


def test_verify_replay_against_the_deeper_engine(bell_replay: tuple[Record, ChannelLibrary]) -> None:
    rep, _ = bell_replay
    assert deeper_level(rep) == "auto"
    report, deep, live = verify_deeper(rep, shots=100)
    assert deep is not None and live is not None and report.deep_level == "JOINT_EXACT"
    assert report.bound is not None and report.discrepancy_populations is not None
    assert report.within_bound, report.notes
    assert report.within_statistics
    assert report.deep_record_key == deep.key() and deep.results.bitstrings.shape[0] == 100


def test_verify_joint_exact_runs_the_rechecks(bell: tuple[Record, LiveRun]) -> None:
    record, live = bell
    assert deeper_level(record) is None
    report, rec, _ = verify_deeper(record, live=live)
    assert report.deep_level is None and report.convergence is not None and report.truncation is not None
    assert report.convergence.converged and report.truncation.converged
    assert rec is not None and rec.zooms, "the re-checks' zooms are cached in the record"
