"""The documentation: the pages exist, every ledger id they cite resolves, and the examples run."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from qutip_trap.control import native
from qutip_trap.control.compiler import SWAP_MATRIX
from qutip_trap.control.native import turns_from_rad
from qutip_trap.provenance import load_ledger, repository_root
from qutip_trap.run import results

ROOT = repository_root()
DOCS = ROOT / "docs"
PAGE_NAMES = (
    "machine.md",
    "circuit.md",
    "schedule.md",
    "dynamics.md",
    "physics.md",
    "laboratory.md",
    "experimental.md",
    "physics_notes.md",
    "conventions.md",
    "examples.md",
    "limits.md",
    "deprecations.md",
    "README.md",
)
"""The documentation pages: the ladder (0.4.0; one page per rung, the laboratory, the experimental namespace) and the pages
every release had; the two planning documents under docs/ quote the conventions and are not pages."""
PAGES = tuple(DOCS / name for name in PAGE_NAMES)


@pytest.mark.parametrize("name", PAGE_NAMES)
def test_documentation_pages_exist_and_are_not_empty(name: str) -> None:
    path = DOCS / name
    assert path.exists(), name
    assert len(path.read_text(encoding="utf-8")) > 1500, name


def test_every_ledger_id_cited_in_the_documentation_resolves() -> None:
    """The M10 audit found `conv.crosstalk_ratio` cited in the prose of `docs/physics_notes.md` with no ledger record
    behind it -- one dangling reference among 136, which nothing checked. Every `conv.*` / `anchor.*` id in backticks on
    a documentation page or in the root README must be a record id (a `conv.*`-style wildcard cannot match, the pattern
    stopping at the identifier characters)."""
    records = load_ledger()
    pattern = re.compile(r"`((?:conv|anchor)\.[A-Za-z0-9_.]*[A-Za-z0-9_])`")
    dangling: dict[str, list[str]] = {}
    for path in [*PAGES, ROOT / "README.md"]:
        cited = set(pattern.findall(path.read_text(encoding="utf-8")))
        missing = sorted(c for c in cited if c not in records)
        if missing:
            dangling[path.name] = missing
    assert not dangling, dangling


def _python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_examples_page_has_runnable_blocks_that_name_the_public_api() -> None:
    blocks = _python_blocks(DOCS / "examples.md")
    assert len(blocks) >= 5
    joined = "\n".join(blocks)
    for name in (
        "yb171_chain",
        "run(",
        "randomized_benchmarking",
        "ghz_fidelity",
        "quantum_volume",
        "calibrate",
    ):
        assert name in joined, name
    for block in blocks:
        compile(block, "examples.md", "exec")


@pytest.mark.slow
def test_examples_execute_in_order() -> None:
    """Every ```python block of docs/examples.md runs, in order, in one namespace (a few minutes on the reference machine)."""
    namespace: dict[str, object] = {}
    for k, block in enumerate(_python_blocks(DOCS / "examples.md")):
        exec(compile(block, f"examples.md[{k}]", "exec"), namespace)  # noqa: S102  (the documented examples)
    assert "result" in namespace and "rb" in namespace


# ---- the conventions, stated once (docs/api_implementation_plan.md item 0.3) -------------------------------------------------

BIT_ORDER_SENTENCE = "qubit 0 is the least-significant bit, the rightmost character"
"""The one sentence of the bit order: on the conventions page once, in ``run/results.py`` in the same words."""
TENSOR_ORDER_SENTENCE = "the gate's first qubit is the left Kronecker factor"
"""The one sentence of the tensor order of the native gate matrices, in ``control/native.py``."""
BIT_ORDER_CONTRADICTIONS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"qubit 0 (?:is |as |being )?(?:the )?most[- ]significant",
        r"qubit 0 (?:is |as |being )?(?:the )?leftmost",
        r"qubit 0 (?:is |as |being )?(?:the )?first character",
    )
)


def package_docstrings() -> list[tuple[str, int, str]]:
    """Every docstring of the package (module, class, function and attribute docstrings) with its file and line."""
    out: list[tuple[str, int, str]] = []
    for path in sorted((ROOT / "qutip_trap").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                out.append((str(path.relative_to(ROOT)), node.lineno, node.value.value))
    return out


def test_the_bit_order_sentence_is_stated_once_and_never_contradicted() -> None:
    """After CUDA-Q's two pages that call the same state big-endian and little-endian: the sentence lives on the conventions
    page and nowhere else in the documentation, ``run/results.py`` (where the keys are made) states it in the same words, and
    no docstring or page puts qubit 0 at the other end. The register density matrix's order is a statement about ions and
    tensor factors ("ion 0 the first factor, the most-significant index bit") and is worded so."""
    hits = {page.name: page.read_text(encoding="utf-8").count(BIT_ORDER_SENTENCE) for page in PAGES}
    assert hits == {name: (1 if name == "conventions.md" else 0) for name in PAGE_NAMES}, hits
    assert BIT_ORDER_SENTENCE in (results.__doc__ or ""), (
        "run/results.py must state the bit order in the same words"
    )
    texts = [*package_docstrings(), *((page.name, 0, page.read_text(encoding="utf-8")) for page in PAGES)]
    contradictions = [
        (where, line, pattern.pattern)
        for where, line, text in texts
        for pattern in BIT_ORDER_CONTRADICTIONS
        if pattern.search(text)
    ]
    assert not contradictions, contradictions


def test_the_tensor_order_of_the_native_gate_matrices_is_stated_once_in_native_py() -> None:
    in_package = [
        (where, line) for where, line, text in package_docstrings() if TENSOR_ORDER_SENTENCE in text
    ]
    on_pages = [page.name for page in PAGES if TENSOR_ORDER_SENTENCE in page.read_text(encoding="utf-8")]
    assert in_package == [("qutip_trap/control/native.py", 1)] and on_pages == [], (in_package, on_pages)
    assert TENSOR_ORDER_SENTENCE in (native.__doc__ or "")


def test_ms_is_the_swap_conjugate_of_the_qiskit_ionq_matrix() -> None:
    """The tensor-order statement of ``control/native.py`` against the other order in the field: qiskit-ionq's ``MSGate``
    matrix is written for Qiskit's little-endian factor order (a gate's first qubit the RIGHT factor), so it equals ``ms``
    with the two qubits swapped and not ``ms`` itself. Verified against qiskit-ionq 1.1.1 on 2026-09-11; skipped when
    qiskit-ionq is not installed (it is not a dependency of this package)."""
    qiskit_ionq = pytest.importorskip("qiskit_ionq")
    phi0, phi1, theta = 0.4, -1.3, 0.37  # radians; unequal phases, so that the swap is visible
    theirs = np.asarray(qiskit_ionq.MSGate(turns_from_rad(phi0), turns_from_rad(phi1), turns_from_rad(theta)))
    ours = native.ms(phi0, phi1, theta)
    assert np.allclose(SWAP_MATRIX @ theirs @ SWAP_MATRIX, ours, atol=1e-12)
    assert not np.allclose(theirs, ours, atol=1e-3)
