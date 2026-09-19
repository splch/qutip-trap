#!/usr/bin/env python
"""Re-run the Appendix D check scripts on the pinned toolchain and compare them with the committed outputs.

PLAN.md milestone M0: "the check scripts of Appendix D committed under validation/scripts/ and re-run on the
pinned toolchain as the first CI job", with "a convergence-report artifact". This runner executes every
``check_*.py`` (and ``bench_*.py`` with ``--bench``) under the current interpreter, compares every numeric
token of the committed ``outputs/<name>.out`` with the fresh output line by line (lines matched by their
non-numeric skeleton, numbers compared to a relative tolerance), and with ``--report`` writes
``validation/report/convergence_report.{json,md}`` plus the fresh outputs, which CI uploads as the
``convergence-report`` artifact. Wall times are recorded but never compared, and neither are lines a script
prefixes with ``MC:`` (Monte Carlo results whose last digits depend on the platform's libm; M5). A script with no
committed output fails the run (a hole in the oracle set is not a pass), fresh numeric lines with no committed
counterpart are counted and reported (a grown script means a stale oracle), and ``--update --only <stem>`` rewrites
one oracle from a fresh run for a deliberate change. The residual class (numbers below 1e-6 printed with at most three
significant digits, compared to a factor RESIDUAL_FACTOR, or to the 1e-6 bound when the pair sits at or across zero) is
recorded in the ledger as ``conv.check_script_residual_class``.

    uv run python validation/scripts/run_checks.py --report
    uv run python validation/scripts/run_checks.py --only check_atomic check_ms_closure
    uv run python validation/scripts/run_checks.py --report --skip check_calibration check_benchmarks check_bloch

CI runs the scripts as a matrix of three jobs (``check_calibration`` alone, ``check_benchmarks`` with ``check_bloch``, and the
rest through ``--skip``), each with its own report artifact: sequentially the 26 scripts took 81 minutes on the runner, 48 of
them ``check_calibration``, and every other job waited on them.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
REPORT_DIR = HERE.parent / "report"

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_SKIP = (
    "MC:",
    "Installed ",
    "Downloaded ",
    "Downloading ",
    "UserWarning",
    "warnings.warn",
    "Using CPython",
    "Creating virtual",
)
# whole lines that are shell-capture artifacts of the committed outputs (a trailing `done`, an `exit 0`),
# never script output
_ARTIFACT_LINES = frozenset({"exit 0", "done"})


def skeleton(line: str) -> str:
    return _NUMBER.sub("#", line).strip()


def numbers(line: str) -> list[float]:
    return [float(tok) for tok in _NUMBER.findall(line)]


def significant_digits(token: str) -> int:
    """Significant digits of a printed number ('4.06e-10' -> 3, '0.00410' -> 3, '1.00' -> 3, '150000.000000000' -> 15)."""
    mantissa = token.lower().split("e")[0].lstrip("+-")
    digits = mantissa.replace(".", "").lstrip("0")
    return len(digits) if digits else 1


RTOL_DEFAULT = 1e-9
RESIDUAL_MAGNITUDE = 1e-6
RESIDUAL_MAX_DIGITS = 3
RESIDUAL_FACTOR = 3.0


def is_residual(token: str) -> bool:
    """A number printed with <= 3 significant digits and below 1e-6 in magnitude: an integrator or round-off residual
    (leaked population, norm loss, element difference) whose last digits depend on the BLAS and integrator of the machine.
    Such numbers are compared to a factor of RESIDUAL_FACTOR (their order of magnitude), everything else to ``rtol``; a
    residual at or across zero has no order of magnitude and is compared to the RESIDUAL_MAGNITUDE bound instead."""
    try:
        value = float(token)
    except ValueError:
        return False
    return abs(value) < RESIDUAL_MAGNITUDE and significant_digits(token) <= RESIDUAL_MAX_DIGITS


def _close(a: float, e: float, rtol: float, atol: float, residual: bool = False) -> bool:
    if abs(a - e) <= max(atol, rtol * abs(e)):
        return True
    if residual:
        if e == 0.0 or a == 0.0 or a * e < 0.0:
            # a residual at or across zero (check_calibration's fitted residual beta: -8.62e-12 here, +1.97e-12 on one
            # Linux runner while a second runner's run passed) has no order of magnitude to compare: the bound is the
            # comparison
            return abs(a - e) <= RESIDUAL_MAGNITUDE
        return 1.0 / RESIDUAL_FACTOR <= a / e <= RESIDUAL_FACTOR
    return False


@dataclass
class Comparison:
    lines_compared: int = 0
    numbers_compared: int = 0
    residual_numbers: int = 0
    max_rel_dev: float = 0.0
    mismatches: list[str] = field(default_factory=list)
    extra_lines: list[str] = field(default_factory=list)
    """Fresh numeric lines with no committed counterpart: a script whose output grew since its oracle was committed.
    Reported (the oracle is then stale and must be regenerated with ``--update``), not a failure, because CI's job is
    to catch a changed number, and a grown script cannot change a committed one."""

    @property
    def ok(self) -> bool:
        return not self.mismatches


def compare(expected: str, actual: str, *, rtol: float, atol: float) -> Comparison:
    """Match every numeric committed line to a fresh line with the same skeleton and compare the numbers."""
    result = Comparison()
    pool: dict[str, list[list[float]]] = {}
    for line in actual.splitlines():
        if any(s in line for s in _SKIP) or line.strip() in _ARTIFACT_LINES:
            continue
        nums = numbers(line)
        if nums:
            pool.setdefault(skeleton(line), []).append(nums)
    for line in expected.splitlines():
        if any(s in line for s in _SKIP) or line.strip() in _ARTIFACT_LINES:
            continue
        tokens = _NUMBER.findall(line)
        exp = [float(t) for t in tokens]
        if not exp:
            continue
        residual_flags = [is_residual(t) for t in tokens]
        key = skeleton(line)
        candidates = pool.get(key, [])
        chosen: list[float] | None = None
        for cand in candidates:
            if len(cand) == len(exp) and all(
                _close(a, e, rtol, atol, residual=r) for a, e, r in zip(cand, exp, residual_flags)
            ):
                chosen = cand
                break
        if chosen is None:
            if candidates:
                result.mismatches.append(f"numbers differ: {line.strip()!r} vs fresh {candidates[0]}")
            else:
                result.mismatches.append(f"line missing from fresh output: {line.strip()!r}")
            continue
        candidates.remove(chosen)
        result.lines_compared += 1
        result.numbers_compared += len(exp)
        for a, e, r in zip(chosen, exp, residual_flags):
            if r:
                result.residual_numbers += 1
            elif e != 0.0:
                result.max_rel_dev = max(result.max_rel_dev, abs(a - e) / abs(e))
    for key, leftovers in pool.items():
        for nums in leftovers:
            result.extra_lines.append(f"{key} {nums}")
    return result


@dataclass
class ScriptReport:
    name: str
    status: str
    wall_s: float
    returncode: int
    lines_compared: int = 0
    numbers_compared: int = 0
    residual_numbers: int = 0
    max_rel_dev: float = 0.0
    mismatches: list[str] = field(default_factory=list)
    note: str = ""
    extra_lines: int = 0


def run_script(path: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(path)], cwd=HERE, capture_output=True, text=True, check=False)
    return proc, time.perf_counter() - t0


def versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "qutip", "sympy", "mpmath"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001  (a missing optional module is reported, not fatal)
            out[mod] = f"unavailable ({exc.__class__.__name__})"
    return out


def write_report(reports: list[ScriptReport], vers: dict[str, str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"versions": vers, "scripts": [asdict(r) for r in reports]}
    (REPORT_DIR / "convergence_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Convergence report: Appendix D check scripts",
        "",
        "Each committed output under `validation/scripts/outputs/` is the oracle for the `[recomputed here]` numbers of",
        "PLAN.md; this run re-executed the scripts on the toolchain below and compared every numeric token: strict class",
        f"to a relative tolerance of {RTOL_DEFAULT:g}; residual class (below {RESIDUAL_MAGNITUDE:g} in magnitude and printed",
        f"with at most {RESIDUAL_MAX_DIGITS} significant digits: integrator and round-off residuals) to a factor {RESIDUAL_FACTOR:g}.",
        "",
        "| component | version |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in vers.items()]
    lines += [
        "",
        "| script | status | wall (s) | lines compared | numbers compared | residual-class numbers | max relative deviation (strict class) | fresh lines without an oracle |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.name} | {r.status} | {r.wall_s:.2f} | {r.lines_compared} | {r.numbers_compared} | {r.residual_numbers} | {r.max_rel_dev:.2e} | {r.extra_lines} |"
        )
    bad = [r for r in reports if r.mismatches]
    if bad:
        lines += ["", "## Mismatches", ""]
        for r in bad:
            lines.append(f"### {r.name}")
            lines += [f"- {m}" for m in r.mismatches]
    (REPORT_DIR / "convergence_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", action="store_true", help="write validation/report/ (the CI artifact)")
    parser.add_argument(
        "--bench", action="store_true", help="also run the bench_*.py timing scripts (minutes)"
    )
    parser.add_argument("--only", nargs="*", default=None, help="script stems to run")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="script stems to leave out (the CI matrix runs the heavy scripts in jobs of their own and the rest here)",
    )
    parser.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite outputs/<stem>.out from the fresh run (only with --only; a deliberate oracle change, never CI)",
    )
    args = parser.parse_args(argv)
    if args.update and args.only is None:
        print(
            "--update needs --only <stems>: an oracle is regenerated one script at a time, on purpose",
            file=sys.stderr,
        )
        return 2

    patterns = ["check_*.py"] + (["bench_*.py"] if args.bench else [])
    scripts = sorted({p for pat in patterns for p in HERE.glob(pat)})
    if args.only is not None:
        wanted = set(args.only)
        scripts = [p for p in scripts if p.stem in wanted]
        missing = wanted - {p.stem for p in scripts}
        if missing:
            print(f"unknown scripts: {sorted(missing)}", file=sys.stderr)
            return 2
    if args.skip:
        unknown = set(args.skip) - {p.stem for p in scripts}
        if unknown:
            print(f"unknown scripts: {sorted(unknown)}", file=sys.stderr)
            return 2
        scripts = [p for p in scripts if p.stem not in set(args.skip)]

    reports: list[ScriptReport] = []
    fresh_dir = REPORT_DIR / "outputs"
    if args.report:
        fresh_dir.mkdir(parents=True, exist_ok=True)
    for path in scripts:
        proc, wall = run_script(path)
        if args.report:
            (fresh_dir / f"{path.stem}.out").write_text(proc.stdout, encoding="utf-8")
            (fresh_dir / f"{path.stem}.err").write_text(proc.stderr, encoding="utf-8")
        committed = OUTPUTS / f"{path.stem}.out"
        if args.update and proc.returncode == 0:
            committed.write_text(proc.stdout, encoding="utf-8")
            print(f"updated {committed.relative_to(HERE)}")
        if proc.returncode != 0:
            rep = ScriptReport(
                path.stem,
                "FAILED",
                wall,
                proc.returncode,
                note=proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
            )
        elif not committed.exists():
            # a check script without its oracle is a hole in the first CI job (PLAN.md M0): a failure, not a note
            rep = ScriptReport(
                path.stem, "NO COMMITTED OUTPUT", wall, 0, note=f"commit outputs/{path.stem}.out"
            )
        elif path.stem.startswith("bench_"):
            rep = ScriptReport(path.stem, "ran (timings not compared)", wall, 0)
        else:
            cmp = compare(committed.read_text(encoding="utf-8"), proc.stdout, rtol=args.rtol, atol=args.atol)
            rep = ScriptReport(
                path.stem,
                "ok" if cmp.ok else "MISMATCH",
                wall,
                0,
                cmp.lines_compared,
                cmp.numbers_compared,
                cmp.residual_numbers,
                cmp.max_rel_dev,
                cmp.mismatches,
                extra_lines=len(cmp.extra_lines),
            )
            if cmp.extra_lines:
                rep.note = f"{len(cmp.extra_lines)} fresh numeric line(s) have no committed counterpart (stale oracle)"
        reports.append(rep)
        print(
            f"{rep.name:32s} {rep.status:28s} {rep.wall_s:7.2f} s  lines={rep.lines_compared} max_rel_dev={rep.max_rel_dev:.1e}"
            + (f"  extra={rep.extra_lines}" if rep.extra_lines else "")
        )
        for m in rep.mismatches:
            print(f"    {m}")

    vers = versions()
    if args.report:
        write_report(reports, vers)
        print(f"report written to {REPORT_DIR}")
    failed = [r for r in reports if r.status in ("FAILED", "MISMATCH", "NO COMMITTED OUTPUT")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
