"""The text budget of DESIGN.md Section 10 (R2, R4, R5): prose lives in the explain drawer, the tooltips and the info
buttons, never in the body of a screen.

A static check over the view modules: every literal string handed to ``ft.Text`` or to the ``status_line`` helper outside the
drawer's own components must be short (a label, a title, a question, a status line under twelve words), and the ``card``
helper takes no subtitle. Tooltips and
``info=`` arguments are where the sentences go, so they are not counted. The check is by the AST, so a rendered Flutter client
is not needed; the same rule the live review applied by eye is applied by machine to every screen on every commit."""

from __future__ import annotations

import ast
from pathlib import Path

VIEWS = Path(__file__).resolve().parents[1] / "src" / "qutip_trap_app" / "views"

MAX_WORDS = 12
"""R5: a status line stays under twelve words; anything longer is a sentence that belongs in the drawer or a tooltip."""

PROSE_FUNCTIONS = {
    "ExplainCardView",
    "ExplainDrawer",
    "SpecificationTile",
    "PredictionCard",
    "ClosurePrediction",
    "ReviewPrompt",
    "DrillView",
    "empty_state",
}
"""Where prose is allowed: the drawer's components, the learning layer's prompts (asked in the learner's words), the empty state."""

ALLOWED_TEXTS = {
    # the one formula shown on the Hamiltonian page (the equation itself, not prose)
    "H/hbar = sum_m omega_m a_m^dag a_m + sum_i (Delta_i/2) sigma_z^i + sum drives (Omega/2) e^{-i(mu t - phi)} sigma_+ prod_m D_m(i eta) + h.c. + Stark",
}


def _literal_words(node: ast.AST) -> int | None:
    """Words in a string literal (or the literal parts of an f-string); None for a non-literal argument."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return len(node.value.split())
    if isinstance(node, ast.JoinedStr):
        return sum(
            len(v.value.split())
            for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
    return None


def _enclosing_functions(tree: ast.Module) -> dict[int, str]:
    """Line -> the innermost def the line belongs to."""
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno is not None:
            spans.append((node.lineno, node.end_lineno, node.name))
    spans.sort(key=lambda s: s[1] - s[0])
    out: dict[int, str] = {}
    for start, end, name in sorted(spans, key=lambda s: -(s[1] - s[0])):
        for line in range(start, end + 1):
            out[line] = name
    return out


def _text_calls(tree: ast.Module) -> list[ast.Call]:
    """Every ``ft.Text(...)`` call and every ``status_line(...)`` call: the two ways a screen puts words on itself."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "Text"
                and isinstance(f.value, ast.Name)
                and f.value.id == "ft"
            ) or (isinstance(f, ast.Name) and f.id == "status_line"):
                calls.append(node)
    return calls


def test_body_text_stays_under_the_budget() -> None:
    offenders: list[str] = []
    counted = 0
    for path in sorted(VIEWS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owners = _enclosing_functions(tree)
        for call in _text_calls(tree):
            if not call.args:
                continue
            words = _literal_words(call.args[0])
            if words is None:
                continue
            counted += 1
            owner = owners.get(call.lineno, "")
            if owner in PROSE_FUNCTIONS:
                continue
            text = call.args[0]
            literal = text.value if isinstance(text, ast.Constant) else None
            if literal in ALLOWED_TEXTS:
                continue
            if words > MAX_WORDS:
                offenders.append(f"{path.name}:{call.lineno} ({owner}): {words} words")
    assert counted > 100, "the walk saw the screens"
    assert not offenders, (
        "prose in a screen body (move it to the drawer, a tooltip or an info button):\n"
        + "\n".join(offenders)
    )


def test_cards_have_no_subtitle_parameter() -> None:
    src = (VIEWS / "common.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    card = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "card")
    names = {a.arg for a in card.args.args + card.args.kwonlyargs}
    assert "subtitle" not in names and {"why", "info"} <= names
    for path in VIEWS.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "card":
                assert not any(k.arg == "subtitle" for k in node.keywords), f"{path.name}:{node.lineno}"


def test_every_level_page_keeps_its_focal_picture() -> None:
    """R1's regression guard: the function that builds each level's screen still calls its focal drawing (layout order is
    the live review's to judge; the AST can only see that the picture is there)."""
    focal = {
        ("level0.py", "ResultsPanel"): "_histogram_chart",
        ("level1.py", "Level1Page"): "_lanes",
        ("level2.py", "Level2Page"): "_spectrum_strip",
        ("level3.py", "Level3Page"): "phase_space",
        ("level4.py", "_trap_page"): "stability_diagram",
    }
    for (name, func), marker in focal.items():
        src = (VIEWS / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
        body = ast.get_source_segment(src, node) or ""
        assert marker in body, (name, func, marker)
