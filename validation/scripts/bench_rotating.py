"""The exact rotating frame of dynamics/rotating.py on the Section 11.1 rows and the real 100 us gate (performance pass 2026-09-09).

For each Section 11.1 row (the bench_ms_timing_v5.py fixture: two 171Yb+ ions, the bichromatic force on every mode, distinct
coefficients per drive term, dop853 at atol 1e-10, rtol 1e-8, 20 us from |dn dn>|0...0>) the factorized Schroedinger-picture
integration is compared with the same QobjEvo integrated in the rotating frame psi = e^{-i H_0 t} phi (the composite element of
``rotating_frame``): wall time, right-hand-side evaluations (through the coefficient counter), the final states back in the
Schroedinger picture, and the norm of their difference (the solver-tolerance figure that Section 11.1 calls 'identical'). Then
the real fixture of tests/m4_fixtures.py (two 171Yb+ ions, x-COM 3.000 and x-rocking 2.828 MHz, the 100 us single-loop symmetric
pulse at eps/2pi = 10 kHz on the space [2, 2, 9, 15] the cap rule selects) runs through ``JointExactEngine.run_pulses`` with
``SolverOptions.rotating_frame`` off and on. Wall times are never compared by run_checks.py (bench_ scripts are timed only).

Run:  uv run python validation/scripts/bench_rotating.py      (about one minute on the plan's machine)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench_ms_timing_v5 as bench  # noqa: E402  (the Section 11.1 fixture and its coefficient counter)

from qutip_trap.dynamics.rotating import rotating_frame  # noqa: E402

qt.settings.core["auto_tidyup"] = False
OPTIONS = {"method": "dop853", "atol": 1e-10, "rtol": 1e-8, "nsteps": 10**7, "store_final_state": True}


def part1() -> None:
    print("== Part 1: the Section 11.1 rows, factorized kernel, T = 20 us, dop853 at atol 1e-10 rtol 1e-8")
    for nmodes, nmax in [(1, 12), (2, 8), (3, 6), (3, 8)]:
        H, dims, _n_el = bench.build(nmodes, nmax, "factorized")
        psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1), *[qt.basis(nmax, 0)] * nmodes)
        rot = rotating_frame(H, dims, [])
        assert rot is not None
        bench.CALLS[0] = 0
        t0 = time.perf_counter()
        res_s = qt.sesolve(H, psi0, [0, 20e-6], options=OPTIONS)
        dt_s = time.perf_counter() - t0
        ev_s = bench.CALLS[0] // 4
        bench.CALLS[0] = 0
        phi0 = rot.frame.to_frame(psi0, 0.0)
        t0 = time.perf_counter()
        res_r = qt.sesolve(rot.H, phi0, [0, 20e-6], options=OPTIONS)
        dt_r = time.perf_counter() - t0
        ev_r = bench.CALLS[0] // 4
        psi_r = rot.frame.from_frame(res_r.final_state, 20e-6)
        diff = (psi_r - res_s.final_state).norm()
        d = int(np.prod(dims))
        print(
            f"modes={nmodes} nmax={nmax} dim={d:5d}  schrodinger wall={dt_s:6.2f}s evals={ev_s:6d} "
            f"({1e6 * dt_s / ev_s:6.1f} us/eval)  rotating wall={dt_r:6.2f}s evals={ev_r:6d} "
            f"({1e6 * dt_r / ev_r:6.1f} us/eval)  evals x{ev_s / ev_r:4.2f} wall x{dt_s / dt_r:4.2f}  "
            f"||psi_rot - psi_schr|| = {diff:.2e}"
        )


def part2() -> None:
    from qutip_trap.api import SeedSpec, SolverOptions
    from qutip_trap.calibration.entangling import gate_space, ms_schedule
    from qutip_trap.control.table import Waveform
    from qutip_trap.dynamics.engine import JointExactEngine
    from qutip_trap.noise.sampling import quiet_sample
    from tests.m4_fixtures import (
        X_COM_TWO_IONS,
        raman_gate_drives,
        table_with_waveform,
        two_ion_device,
        two_ion_modes,
    )

    print(
        "\n== Part 2: the real 100 us single-loop pulse (tests/m4_fixtures.py) through JointExactEngine.run_pulses"
    )
    dev = two_ion_device()
    modes = two_ion_modes(dev)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=10e3, kernel="rwa")
    table = table_with_waveform((0, 1), wf, device=dev, drives=raman_gate_drives(2), stark_hz={})
    sched = ms_schedule(wf, (0, 1), raman_gate_drives(2), table)
    space = gate_space(modes, 2, waveform=wf)
    finals = {}
    for flag in (False, True):
        eng = JointExactEngine()
        t0 = time.perf_counter()
        tr = eng.run_pulses(
            dev,
            sched,
            space.initial_state([0, 0]),
            space,
            quiet_sample(),
            SeedSpec(0),
            SolverOptions(rotating_frame=flag),
        )
        dt = time.perf_counter() - t0
        rep = eng.last_report
        assert rep is not None
        seg = rep.segments[0]
        finals[flag] = tr
        print(
            f"space dims={space.dims} frame={seg.frame:11s} wall={dt:6.2f}s evals={seg.rhs_evaluations} "
            f"steps/period={seg.steps_per_period:5.1f} kernel={seg.kernel}"
        )
    pops = {k: np.real(np.diag(v.final.internal.full())) for k, v in finals.items()}
    print(f"populations (schrodinger): {np.array2string(pops[False], precision=7)}")
    print(f"populations (rotating):    {np.array2string(pops[True], precision=7)}")
    print(f"||psi_rot - psi_schr|| = {(finals[True].final.joint - finals[False].final.joint).norm():.2e}")
    for m in finals[True].alpha_m:
        print(
            f"max |<a_{m}>_rot - <a_{m}>_schr| over the stored times = "
            f"{np.max(np.abs(finals[True].alpha_m[m] - finals[False].alpha_m[m])):.2e}"
        )


if __name__ == "__main__":
    part1()
    part2()
