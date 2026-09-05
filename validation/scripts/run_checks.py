#!/usr/bin/env python
"""Re-run the Appendix D check scripts on the pinned toolchain and compare them with the committed outputs.

PLAN.md milestone M0: "the check scripts of Appendix D committed under validation/scripts/ and re-run on the
pinned toolchain as the first CI job", with "a convergence-report artifact". This runner executes every
``check_*.py`` (and ``bench_*.py`` with ``--bench``) under the current interpreter, compares every numeric
token of the committed ``outputs/<name>.out`` with the fresh output line by line (lines matched by their
non-numeric skeleton, numbers compared to a relative tolerance), and with ``--report`` writes
``validation/report/convergence_report.{json,md}`` plus the fresh outputs, which CI uploads as the
``convergence-report`` artifact. Wall times are recorded but never compared.

    uv run python validation/scripts/run_checks.py --report
    uv run python validation/scripts/run_checks.py --only check_atomic check_ms_closure
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


def _close(a: float, e: float, rtol: float, atol: float) -> bool:
    return abs(a - e) <= max(atol, rtol * abs(e))


@dataclass
class Comparison:
    lines_compared: int = 0
    numbers_compared: int = 0
    max_rel_dev: float = 0.0
    mismatches: list[str] = field(default_factory=list)

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
        exp = numbers(line)
        if not exp:
            continue
        key = skeleton(line)
        candidates = pool.get(key, [])
        chosen: list[float] | None = None
        for cand in candidates:
            if len(cand) == len(exp) and all(_close(a, e, rtol, atol) for a, e in zip(cand, exp)):
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
        for a, e in zip(chosen, exp):
            if e != 0.0:
                result.max_rel_dev = max(result.max_rel_dev, abs(a - e) / abs(e))
    return result


@dataclass
class ScriptReport:
    name: str
    status: str
    wall_s: float
    returncode: int
    lines_compared: int = 0
    numbers_compared: int = 0
    max_rel_dev: float = 0.0
    mismatches: list[str] = field(default_factory=list)
    note: str = ""


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
        "PLAN.md; this run re-executed the scripts on the toolchain below and compared every numeric token.",
        "",
        "| component | version |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in vers.items()]
    lines += [
        "",
        "| script | status | wall (s) | lines compared | numbers compared | max relative deviation |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.name} | {r.status} | {r.wall_s:.2f} | {r.lines_compared} | {r.numbers_compared} | {r.max_rel_dev:.2e} |"
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
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv)

    patterns = ["check_*.py"] + (["bench_*.py"] if args.bench else [])
    scripts = sorted({p for pat in patterns for p in HERE.glob(pat)})
    if args.only is not None:
        wanted = set(args.only)
        scripts = [p for p in scripts if p.stem in wanted]
        missing = wanted - {p.stem for p in scripts}
        if missing:
            print(f"unknown scripts: {sorted(missing)}", file=sys.stderr)
            return 2

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
        if proc.returncode != 0:
            rep = ScriptReport(
                path.stem,
                "FAILED",
                wall,
                proc.returncode,
                note=proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
            )
        elif not committed.exists():
            rep = ScriptReport(path.stem, "no committed output", wall, 0)
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
                cmp.max_rel_dev,
                cmp.mismatches,
            )
        reports.append(rep)
        print(
            f"{rep.name:32s} {rep.status:28s} {rep.wall_s:7.2f} s  lines={rep.lines_compared} max_rel_dev={rep.max_rel_dev:.1e}"
        )
        for m in rep.mismatches:
            print(f"    {m}")

    vers = versions()
    if args.report:
        write_report(reports, vers)
        print(f"report written to {REPORT_DIR}")
    failed = [r for r in reports if r.status in ("FAILED", "MISMATCH")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
