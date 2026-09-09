"""The verify-deeper action (PLAN.md Section 14.5; milestone M11.2).

Section 14.5: "Verify deeper. At Levels 0 to 2 one action runs the same job at the next-deeper engine, on a subset of shots
when the full run would be slow, and shows the discrepancy against the bound of Section 9.8 or the channel-derivation
residual." The ladder of engines is CHANNEL_REPLAY -> GATE_LOCAL -> JOINT_EXACT (Section 5.4); ``level="auto"`` picks
the deepest engine the Section 11.5 size guards allow. The discrepancy is measured on the REGISTER POPULATIONS, which both
engines return deterministically (``keep_final_state``), so shot noise does not enter it; the histogram distance is
reported beside it with its statistical scale. A JOINT_EXACT record has nothing deeper: the action runs the Section 5.5
convergence re-checks on its entangling steps instead and says so.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import ConvergenceRecord, JobSpec, LiveRun, Record, RecordError, execute
from qutip_trap_app.resim import TruncationCheck, tolerance_recheck, truncation_recheck

Progress = Callable[[str, float | None, str], None]

DEEPER: dict[str, str | None] = {"CHANNEL_REPLAY": "auto", "GATE_LOCAL": "JOINT_EXACT", "JOINT_EXACT": None}


def deeper_level(record: Record) -> str | None:
    """The engine verify-deeper runs for this record, or None when nothing deeper exists."""
    return DEEPER.get(record.diagnostics.level)


@dataclass(frozen=True)
class VerifyReport:
    shallow_level: str
    deep_level: str | None
    shots: int
    discrepancy_populations: float | None
    """max |diag(rho_shallow) - diag(rho_deep)| over the computational states: deterministic, no shot noise."""
    discrepancy_histogram: float
    """(1/2) sum |p_shallow - p_deep| over the two runs' histograms."""
    statistical_scale: float
    """The shot-noise scale of the histogram comparison: sqrt(sum_x p(1-p)/N_shallow + p(1-p)/N_deep)/2."""
    bound: float | None
    """The channel-derivation residual (replay) or Section 9.8's discrepancy bound (GATE_LOCAL) the deep run is held to."""
    within_bound: bool | None
    within_statistics: bool
    deep_record_key: str | None
    wall_time_s: float
    convergence: ConvergenceRecord | None = None
    truncation: TruncationCheck | None = None
    notes: tuple[str, ...] = ()


def _populations(record: Record) -> np.ndarray | None:
    rho = record.results.final_state
    if rho is None:
        return None
    return np.asarray(np.clip(np.real(np.diag(np.asarray(rho))), 0.0, None), dtype=float)


def _histogram_distance(a: Record, b: Record) -> tuple[float, float]:
    keys = set(a.results.probabilities) | set(b.results.probabilities)
    tv = 0.5 * sum(
        abs(a.results.probabilities.get(k, 0.0) - b.results.probabilities.get(k, 0.0)) for k in keys
    )
    na = max(a.results.effective_sample_size, 1.0)
    nb = max(b.results.effective_sample_size, 1.0)
    var = 0.0
    for k in keys:
        pa = a.results.probabilities.get(k, 0.0)
        pb = b.results.probabilities.get(k, 0.0)
        var += pa * (1.0 - pa) / na + pb * (1.0 - pb) / nb
    return tv, 0.5 * math.sqrt(var)


def verify_deeper(
    record: Record,
    *,
    live: LiveRun | None = None,
    shots: int | None = None,
    progress: Progress | None = None,
) -> tuple[VerifyReport, Record | None, LiveRun | None]:
    """Run the record's job at the next-deeper engine and compare; returns the report and the deep record (with its live
    handle) when one was made."""
    t0 = time.perf_counter()
    shallow = record.diagnostics.level
    deep = deeper_level(record)
    job: JobSpec = record.job
    if deep is None:
        if live is None:
            raise RecordError("a JOINT_EXACT record needs its live run to re-check convergence")
        if progress:
            progress(
                "rechecking",
                None,
                "tolerances tightened by ten and caps raised by two on the entangling steps",
            )
        steps = [
            s
            for s in record.schedule.steps
            if s.kind == "gate" and any(g.gate_id in s.target_ids for g in record.schedule.gates)
        ]
        if not steps:
            steps = [s for s in record.schedule.steps if s.kind == "gate"][:1]
        conv: ConvergenceRecord | None = None
        trunc: TruncationCheck | None = None
        rec = record
        for st in steps:
            rec, conv = tolerance_recheck(rec, live, st.index)
            rec, trunc = truncation_recheck(rec, live, st.index)
        ok = bool(conv and conv.converged and trunc and trunc.converged)
        return (
            VerifyReport(
                shallow_level=shallow,
                deep_level=None,
                shots=job.shots,
                discrepancy_populations=None,
                discrepancy_histogram=0.0,
                statistical_scale=0.0,
                bound=None,
                within_bound=None,
                within_statistics=True,
                deep_record_key=None,
                wall_time_s=time.perf_counter() - t0,
                convergence=conv,
                truncation=trunc,
                notes=(
                    "JOINT_EXACT is the deepest engine (Section 5.4): the Section 5.5 re-checks ran on the entangling steps instead",
                    "converged" if ok else "NOT converged: see the numerics panel",
                ),
            ),
            rec,
            live,
        )
    n_shots = int(shots) if shots is not None else int(job.shots)
    deep_job = dataclasses.replace(job, shots=n_shots, level=deep, keep_final_state=True)  # type: ignore[arg-type]
    if progress:
        progress("running deeper", None, f"the same job at {deep} with {n_shots} shots")
    try:
        deep_record, deep_live = execute(deep_job)
    except (
        Exception
    ) as exc:  # the core refuses a level above the size guards with a RunError naming the knobs
        return (
            VerifyReport(
                shallow_level=shallow,
                deep_level=deep,
                shots=n_shots,
                discrepancy_populations=None,
                discrepancy_histogram=0.0,
                statistical_scale=0.0,
                bound=None,
                within_bound=None,
                within_statistics=False,
                deep_record_key=None,
                wall_time_s=time.perf_counter() - t0,
                notes=(f"the deeper run was refused: {exc}",),
            ),
            None,
            None,
        )
    pa = _populations(record)
    pb = _populations(deep_record)
    disc = None if pa is None or pb is None else float(np.max(np.abs(pa - pb)))
    tv, scale = _histogram_distance(record, deep_record)
    if shallow == "CHANNEL_REPLAY" and record.replay is not None:
        bound: float | None = record.replay.residual_total
    elif shallow == "GATE_LOCAL" and record.gate_local is not None:
        bound = record.gate_local.discrepancy_bound + record.diagnostics.dropped_branch_weight
    else:
        bound = None
    within = None if (bound is None or disc is None) else disc <= bound
    notes = [f"deep engine actually run: {deep_record.diagnostics.level}"]
    if disc is not None and bound is not None:
        notes.append(
            f"register populations differ by {disc:.3e} against the bound {bound:.3e}: {'within' if within else 'OUTSIDE'}"
        )
    notes.append(f"histograms differ by {tv:.3g} against a shot-noise scale of {scale:.3g}")
    return (
        VerifyReport(
            shallow_level=shallow,
            deep_level=deep_record.diagnostics.level,
            shots=n_shots,
            discrepancy_populations=disc,
            discrepancy_histogram=tv,
            statistical_scale=scale,
            bound=bound,
            within_bound=within,
            within_statistics=tv <= 3.0 * scale + 1e-12,
            deep_record_key=deep_record.key(),
            wall_time_s=time.perf_counter() - t0,
            notes=tuple(notes),
        ),
        deep_record,
        deep_live,
    )


__all__ = ["DEEPER", "VerifyReport", "deeper_level", "verify_deeper"]
