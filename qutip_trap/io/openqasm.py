"""OpenQASM 2 importer, a subset (PLAN.md Sections 1.4, 7.2, 7.6; milestone M6).

Accepted: the ``OPENQASM 2.0;`` header, ``include`` statements (ignored: the qelib1.inc gates are built in), ``qreg`` and
``creg`` declarations (qubit registers are flattened in declaration order; the classical registers a terminal measurement
writes into become ``Circuit.registers``, name -> the measured qubits in bit order, 0.2.0), ``gate`` declarations with parameters (expanded
by inlining, so the client SDKs' OpenQASM 2 export, which declares gpi, gpi2, ms and zz as custom gates built from u, rz,
rxx and rzz, imports through its own definitions, Section 7.6), gate applications with register broadcasting, ``barrier``
(ignored), ``measure`` (a trailing measurement is the circuit's terminal ``measure``; one followed by a later gate on the
same qubit stays a mid-circuit ``measure`` operation the scheduler refuses), ``reset`` (a mid-circuit operation) and
parameter expressions over pi with + - * / ^, unary minus, parentheses and sin, cos, tan, exp, ln, sqrt. Built-in gates:
U/u3/u, u2, u1, CX/cx/cnot, id, x, y, z, h, s, sdg, t, tdg, sx, rx, ry, rz, cz, swap, cp/cu1, rxx, rzz and the native gpi,
gpi2, ms, zz (radians when undeclared, the IR convention). Classical control (``if``) and ``opaque`` are refused.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qutip_trap.control.compiler import Circuit, Operation

_TOKEN = re.compile(
    r"\s+|//[^\n]*|(?P<string>\"[^\"]*\")|(?P<real>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
    r"|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<arrow>->)|(?P<eq>==)|(?P<sym>[;,(){}\[\]+\-*/^])"
)

BUILTIN_ARITY: dict[str, tuple[int, int]] = {
    "U": (1, 3),
    "u3": (1, 3),
    "u": (1, 3),
    "u2": (1, 2),
    "u1": (1, 1),
    "CX": (2, 0),
    "cx": (2, 0),
    "cnot": (2, 0),
    "id": (1, 0),
    "x": (1, 0),
    "y": (1, 0),
    "z": (1, 0),
    "h": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "sx": (1, 0),
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "cz": (2, 0),
    "swap": (2, 0),
    "cp": (2, 1),
    "cu1": (2, 1),
    "rxx": (2, 1),
    "rzz": (2, 1),
    "gpi": (1, 1),
    "gpi2": (1, 1),
    "ms": (2, 3),
    "zz": (2, 1),
}

_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "ln": math.log,
    "sqrt": math.sqrt,
}


class OpenQASMError(ValueError):
    """The text is not in the accepted OpenQASM 2 subset."""


@dataclass
class _Tok:
    kind: str
    text: str
    pos: int


def tokenize(text: str) -> list[_Tok]:
    out: list[_Tok] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise OpenQASMError(f"unexpected character {text[pos]!r} at offset {pos}")
        pos = m.end()
        if m.lastgroup is None:
            continue
        out.append(_Tok(m.lastgroup, m.group(m.lastgroup), m.start()))
    out.append(_Tok("eof", "", len(text)))
    return out


@dataclass(frozen=True)
class GateDef:
    name: str
    params: tuple[str, ...]
    qargs: tuple[str, ...]
    body: tuple[tuple[str, tuple[list[_Tok], ...], tuple[str, ...]], ...]
    """(gate name, parameter expressions as token lists, argument names) per statement."""


class _Parser:
    def __init__(self, text: str) -> None:
        self.toks = tokenize(text)
        self.i = 0
        self.qregs: list[tuple[str, int, int]] = []  # name, size, offset
        self.cregs: dict[str, int] = {}  # declaration order
        self.gates: dict[str, GateDef] = {}
        self.ops: list[Operation] = []
        self.cbits: dict[int, tuple[str, int]] = {}  # position in ops of a measure -> (creg, bit) it writes
        self.n_qubits = 0

    # ---- token helpers
    def peek(self, k: int = 0) -> _Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def take(self) -> _Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, text: str) -> _Tok:
        t = self.take()
        if t.text != text:
            raise OpenQASMError(f"expected {text!r} at offset {t.pos}, found {t.text!r}")
        return t

    def expect_kind(self, kind: str) -> _Tok:
        t = self.take()
        if t.kind != kind:
            raise OpenQASMError(f"expected {kind} at offset {t.pos}, found {t.text!r}")
        return t

    # ---- grammar
    def program(self) -> Circuit:
        if self.peek().text == "OPENQASM":
            self.take()
            version = self.expect_kind("real").text
            if not version.startswith("2"):
                raise OpenQASMError(f"OpenQASM {version} is not the 2.0 subset this importer accepts")
            self.expect(";")
        while self.peek().kind != "eof":
            self.statement()
        if self.n_qubits == 0:
            raise OpenQASMError("no qreg declared")
        # terminal against mid-circuit measurements
        terminal: list[int] = []
        ops = list(self.ops)
        keep: list[Operation] = []
        last_touch: dict[int, int] = {}
        for k, op in enumerate(ops):
            if op.name != "measure":
                for q in op.qubits:
                    last_touch[q] = k
        bits: dict[str, dict[int, int]] = {
            name: {} for name in self.cregs
        }  # creg -> bit -> the qubit that writes it
        for k, op in enumerate(ops):
            if op.name == "measure":
                q = op.qubits[0]
                if last_touch.get(q, -1) > k:
                    keep.append(op)  # a later gate acts on it: mid-circuit
                elif q not in terminal:
                    terminal.append(q)
                if k in self.cbits:
                    name, bit = self.cbits[k]
                    bits[name][bit] = q  # a bit written twice keeps its last writer
            else:
                keep.append(op)
        measure = tuple(sorted(terminal))
        # the registers: every creg a measurement writes into, in declaration order, its written bits in bit order (a
        # declared bit nothing writes is left out; a creg nothing writes into is not a register of the circuit; with no
        # creg written at all the circuit gets the default register over its terminal targets, which is empty when the
        # program measures nothing: run() then measures every ion, the 0.1.0 rule)
        registers = {
            name: tuple(written[bit] for bit in sorted(written)) for name, written in bits.items() if written
        }
        return Circuit(self.n_qubits, tuple(keep), measure, registers if registers else {"c": measure})

    def statement(self) -> None:
        t = self.peek()
        if t.text == "include":
            self.take()
            self.expect_kind("string")
            self.expect(";")
        elif t.text in ("qreg", "creg"):
            self.decl()
        elif t.text == "gate":
            self.gatedecl()
        elif t.text == "barrier":
            self.take()
            while self.peek().text != ";":
                self.take()
            self.expect(";")
        elif t.text == "measure":
            self.take()
            src = self.argument()
            self.expect("->")
            cname, dst = self.cargument()
            self.expect(";")
            if len(src) not in (1, len(dst)) and len(dst) != 1:
                raise OpenQASMError(f"measure register sizes disagree at offset {t.pos}")
            for k, q in enumerate(src):
                self.cbits[len(self.ops)] = (cname, dst[k] if len(dst) > 1 else dst[0])
                self.ops.append(Operation("measure", (q,), ()))
        elif t.text == "reset":
            self.take()
            for q in self.argument():
                self.ops.append(Operation("reset", (q,), ()))
            self.expect(";")
        elif t.text in ("if", "opaque"):
            raise OpenQASMError(f"{t.text!r} statements are outside the accepted subset (offset {t.pos})")
        elif t.kind == "id":
            self.application(env={}, qmap=None)
        else:
            raise OpenQASMError(f"unexpected token {t.text!r} at offset {t.pos}")

    def decl(self) -> None:
        kind = self.take().text
        name = self.expect_kind("id").text
        self.expect("[")
        size = int(self.expect_kind("real").text)
        self.expect("]")
        self.expect(";")
        if size <= 0:
            raise OpenQASMError(f"register {name} has non-positive size")
        if kind == "qreg":
            if any(n == name for n, _s, _o in self.qregs):
                raise OpenQASMError(f"qreg {name} declared twice")
            self.qregs.append((name, size, self.n_qubits))
            self.n_qubits += size
        else:
            self.cregs[name] = size

    def gatedecl(self) -> None:
        self.expect("gate")
        name = self.expect_kind("id").text
        params: list[str] = []
        if self.peek().text == "(":
            self.take()
            while self.peek().text != ")":
                params.append(self.expect_kind("id").text)
                if self.peek().text == ",":
                    self.take()
            self.expect(")")
        qargs: list[str] = []
        while self.peek().text != "{":
            qargs.append(self.expect_kind("id").text)
            if self.peek().text == ",":
                self.take()
        self.expect("{")
        body: list[tuple[str, tuple[list[_Tok], ...], tuple[str, ...]]] = []
        while self.peek().text != "}":
            if self.peek().text == "barrier":
                self.take()
                while self.peek().text != ";":
                    self.take()
                self.expect(";")
                continue
            gname = self.expect_kind("id").text
            exprs: list[list[_Tok]] = []
            if self.peek().text == "(":
                self.take()
                exprs = self.expr_token_lists()
                self.expect(")")
            args: list[str] = []
            while self.peek().text != ";":
                args.append(self.expect_kind("id").text)
                if self.peek().text == ",":
                    self.take()
            self.expect(";")
            for a in args:
                if a not in qargs:
                    raise OpenQASMError(f"gate {name}: body argument {a!r} is not a declared argument")
            body.append((gname, tuple(exprs), tuple(args)))
        self.expect("}")
        self.gates[name] = GateDef(name, tuple(params), tuple(qargs), tuple(body))

    def expr_token_lists(self) -> list[list[_Tok]]:
        """Comma-separated expressions up to the closing parenthesis, as token lists (evaluated at expansion time)."""
        out: list[list[_Tok]] = []
        depth = 0
        current: list[_Tok] = []
        while True:
            t = self.peek()
            if t.kind == "eof":
                raise OpenQASMError("unterminated parameter list")
            if t.text == "(":
                depth += 1
            elif t.text == ")":
                if depth == 0:
                    break
                depth -= 1
            if t.text == "," and depth == 0:
                out.append(current)
                current = []
                self.take()
                continue
            current.append(self.take())
        if current or out:
            out.append(current)
        return out

    def argument(self) -> list[int]:
        """A qubit argument: q[i] or a whole register (broadcast)."""
        name = self.expect_kind("id").text
        reg = next((r for r in self.qregs if r[0] == name), None)
        if reg is None:
            raise OpenQASMError(f"unknown qreg {name!r}")
        _n, size, offset = reg
        if self.peek().text == "[":
            self.take()
            idx = int(self.expect_kind("real").text)
            self.expect("]")
            if idx < 0 or idx >= size:
                raise OpenQASMError(f"{name}[{idx}] out of range")
            return [offset + idx]
        return [offset + k for k in range(size)]

    def cargument(self) -> tuple[str, list[int]]:
        """A classical argument: (creg name, the bit indices) for c[i] or a whole register."""
        name = self.expect_kind("id").text
        if name not in self.cregs:
            raise OpenQASMError(f"unknown creg {name!r}")
        if self.peek().text == "[":
            self.take()
            idx = int(self.expect_kind("real").text)
            self.expect("]")
            if idx < 0 or idx >= self.cregs[name]:
                raise OpenQASMError(f"{name}[{idx}] out of range")
            return name, [idx]
        return name, list(range(self.cregs[name]))

    def application(self, env: dict[str, float], qmap: dict[str, int] | None) -> None:
        """A gate application at the top level (qmap None: register arguments) with broadcasting."""
        name = self.expect_kind("id").text
        values: list[float] = []
        if self.peek().text == "(":
            self.take()
            for toks in self.expr_token_lists():
                values.append(evaluate(toks, env))
            self.expect(")")
        args: list[list[int]] = []
        while self.peek().text != ";":
            args.append(self.argument())
            if self.peek().text == ",":
                self.take()
        self.expect(";")
        if not args:
            raise OpenQASMError(f"gate {name} has no arguments")
        width = max(len(a) for a in args)
        for a in args:
            if len(a) not in (1, width):
                raise OpenQASMError(f"gate {name}: register arguments of different sizes")
        for k in range(width):
            qubits = [a[k] if len(a) > 1 else a[0] for a in args]
            self.emit(name, values, qubits, depth=0)

    def emit(self, name: str, values: Sequence[float], qubits: Sequence[int], *, depth: int) -> None:
        if depth > 64:
            raise OpenQASMError("gate definitions nest too deeply (recursive definition?)")
        if name in self.gates:
            g = self.gates[name]
            if len(values) != len(g.params) or len(qubits) != len(g.qargs):
                raise OpenQASMError(
                    f"gate {name} called with {len(values)} parameters and {len(qubits)} arguments"
                )
            env = dict(zip(g.params, values))
            qmap = dict(zip(g.qargs, qubits))
            for gname, exprs, args in g.body:
                vals = [evaluate(toks, env) for toks in exprs]
                self.emit(gname, vals, [qmap[a] for a in args], depth=depth + 1)
            return
        if name not in BUILTIN_ARITY:
            raise OpenQASMError(
                f"unknown gate {name!r}: declare it with a gate statement or use one of {sorted(BUILTIN_ARITY)}"
            )
        arity, n_params = BUILTIN_ARITY[name]
        if len(qubits) != arity or len(values) != n_params:
            raise OpenQASMError(f"gate {name} takes {arity} argument(s) and {n_params} parameter(s)")
        self.ops.extend(builtin_operations(name, tuple(values), tuple(int(q) for q in qubits)))


def builtin_operations(name: str, values: tuple[float, ...], qubits: tuple[int, ...]) -> list[Operation]:
    """The IR operations of a built-in OpenQASM 2 gate (qelib1.inc definitions; global phases dropped)."""
    if name in ("U", "u3", "u"):
        return [Operation("u3", qubits, values)]
    if name == "u2":
        return [Operation("u3", qubits, (math.pi / 2.0, values[0], values[1]))]
    if name == "u1":
        return [Operation("rz", qubits, (values[0],))]  # u1(lambda) = e^{i lambda/2} RZ(lambda)
    if name in ("CX", "cx"):
        return [Operation("cnot", qubits, ())]
    if name == "cu1":
        return [Operation("cp", qubits, (values[0],))]
    return [Operation(name, qubits, values)]


# ---- parameter expressions --------------------------------------------------------------------------------------------------


def evaluate(tokens: Sequence[_Tok], env: dict[str, float]) -> float:
    """Evaluate an OpenQASM 2 real expression: numbers, pi, identifiers of ``env``, + - * / ^, unary minus, functions."""
    if not tokens:
        raise OpenQASMError("empty expression")
    pos = 0

    def peek() -> str:
        return tokens[pos].text if pos < len(tokens) else ""

    def take() -> _Tok:
        nonlocal pos
        t = tokens[pos]
        pos += 1
        return t

    def additive() -> float:
        v = multiplicative()
        while peek() in ("+", "-"):
            op = take().text
            w = multiplicative()
            v = v + w if op == "+" else v - w
        return v

    def multiplicative() -> float:
        v = unary()
        while peek() in ("*", "/"):
            op = take().text
            w = unary()
            if op == "/":
                if w == 0.0:
                    raise OpenQASMError("division by zero in a parameter expression")
                v = v / w
            else:
                v = v * w
        return v

    def unary() -> float:
        if peek() == "-":
            take()
            return -unary()
        if peek() == "+":
            take()
            return unary()
        return power()

    def power() -> float:
        v = atom()
        if peek() == "^":
            take()
            w = unary()
            return float(v**w)
        return v

    def atom() -> float:
        t = take()
        if t.kind == "real":
            return float(t.text)
        if t.text == "(":
            v = additive()
            if peek() != ")":
                raise OpenQASMError("unbalanced parenthesis in a parameter expression")
            take()
            return v
        if t.kind == "id":
            if t.text == "pi":
                return math.pi
            if t.text in _FUNCTIONS:
                if peek() != "(":
                    raise OpenQASMError(f"{t.text} needs a parenthesized argument")
                take()
                v = additive()
                if peek() != ")":
                    raise OpenQASMError("unbalanced parenthesis in a function call")
                take()
                return float(_FUNCTIONS[t.text](v))
            if t.text in env:
                return float(env[t.text])
            raise OpenQASMError(f"unknown identifier {t.text!r} in a parameter expression")
        raise OpenQASMError(f"unexpected {t.text!r} in a parameter expression")

    value = additive()
    if pos != len(tokens):
        raise OpenQASMError(f"trailing tokens in a parameter expression near {tokens[pos].text!r}")
    return value


def load_openqasm2(text: str) -> Circuit:
    """Import OpenQASM 2 text (the subset in the module docstring) into the IR: angles in radians, the terminal measurements
    as ``Circuit.measure`` and the classical registers they write as ``Circuit.registers`` (0.2.0)."""
    return _Parser(text).program()


__all__ = ["BUILTIN_ARITY", "OpenQASMError", "builtin_operations", "evaluate", "load_openqasm2", "tokenize"]
