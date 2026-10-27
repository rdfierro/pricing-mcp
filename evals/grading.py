"""Graders for agent transcripts. Deterministic, so results are repeatable and explainable.

A transcript is a dict:
    {"question": str,
     "tool_calls": [{"name": str, "input": dict, "output": dict | str, "is_error": bool}],
     "answer": str}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

SKU_RE = re.compile(r"\b[A-Z]{2}-[A-Z]{3}-\d{2}\b")
NUM_RE = re.compile(r"(?<![\w.])[-+−]?\$?\d[\d,]*(?:\.\d+)?%?")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


# --- number grounding ------------------------------------------------------

def _numbers_in(obj: Any) -> set[float]:
    """Every number that appears anywhere in a tool output, including inside detail strings."""
    found: set[float] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.add(abs(float(obj)))
    elif isinstance(obj, str):
        for tok in NUM_RE.findall(SKU_RE.sub(" ", obj)):
            v = _parse(tok)
            if v is not None:
                found.add(v)
                if tok.endswith("%"):
                    # "52%" in a tool's text can be quoted as 0.52, and a "±15%" limit
                    # as the multipliers 1.15 and 0.85.
                    found |= {v / 100, 1 + v / 100, abs(1 - v / 100)}
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found |= _numbers_in(v)
            if "pct" in str(k) and isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) < 1:
                # max_price_change_pct 0.15 can be quoted as 15%, or used as 1.15 / 0.85
                found |= {abs(v) * 100, 1 + abs(v), 1 - abs(v)}
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found |= _numbers_in(v)
    return found


def _parse(tok: str) -> float | None:
    t = tok.replace("$", "").replace(",", "").replace("−", "-").rstrip("%")
    try:
        return abs(float(t))
    except ValueError:
        return None


def _matches(x: float, allowed: set[float], is_pct: bool) -> bool:
    candidates = [x]
    if is_pct:
        candidates.append(x / 100)  # 43.8% may appear as 0.438 in the tool output
    for c in candidates:
        for v in allowed:
            tol = max(0.011, 0.006 * v)  # covers rounding to the displayed precision
            if abs(c - v) <= tol:
                return True
    return False


# A calculation written out in the answer, such as "($9.49 − $9.29 = $0.20)" or
# "0.95 × $23.44 ≈ $22.27". The left side may use numbers, + − × ÷ * / and parentheses.
CALC_RE = re.compile(
    r"(?P<expr>[\(\$\d][\d\$\.,%\s\(\)×x\*÷/\+\-−]*[\d\)%])\s*[=≈]\s*"
    r"(?P<result>[-−]?\$?\d[\d,]*(?:\.\d+)?%?)"
)
# The result written first, with the working in parentheses: "about 1.46 ($12.29 ÷ $8.40)"
# or "36.5% (($9.29 − $5.90) ÷ $9.29)". The parentheses are matched by _balanced below.
RESULT_FIRST_RE = re.compile(r"(?<![\w.$,])(?P<result>\$?\d[\d,]*(?:\.\d+)?%?)\s*\(")
MATH_ONLY_RE = re.compile(r"^[\d\$\.,%\s\(\)×x\*÷/\+\-−]+$")
OPERATOR_RE = re.compile(r"[×x\*÷/\+\-−]")
COUNT_LIMIT = 100  # bare whole numbers up to this are treated as counts ("14 SKUs")


def _calculations(text: str):
    """Yield (expression, result, matched text) for each calculation written in the answer."""
    for m in CALC_RE.finditer(text):
        yield m.group("expr"), m.group("result"), m.group(0)
    for m in RESULT_FIRST_RE.finditer(text):
        inner = _balanced(text, m.end() - 1)
        # A calculation needs two or more numbers; "361 (−13.2%)" is a value and its change.
        if (inner and MATH_ONLY_RE.match(inner) and OPERATOR_RE.search(inner) and "=" not in inner
                and len(NUM_RE.findall(inner)) >= 2):
            yield inner, m.group("result"), f"{m.group('result')} ({inner})"


def _balanced(text: str, open_idx: int) -> str | None:
    """Contents of the parenthesis that opens at open_idx, or None if it never closes."""
    depth = 0
    for i in range(open_idx, min(len(text), open_idx + 200)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
    return None


def _strip_unmatched(expr: str) -> str:
    """Keep only the math after the last unmatched "(".

    In "about $22.65 ($16.99 ÷ 24 × 32 = $22.65)" the regex picks up "$22.65 ($16.99 ÷ 24 × 32";
    the calculation is what follows the open parenthesis.
    """
    open_at: list[int] = []
    for i, ch in enumerate(expr):
        if ch == "(":
            open_at.append(i)
        elif ch == ")" and open_at:
            open_at.pop()
    return expr[open_at[-1] + 1:] if open_at else expr


def _eval_expr(expr: str) -> tuple[float | None, list[tuple[float, bool]]]:
    """Evaluate a written calculation. Returns its value and the operands it used."""
    operands: list[tuple[float, bool]] = []
    expr = _strip_unmatched(expr.strip()).strip()

    def operand(m: re.Match) -> str:
        tok = m.group(0)
        v = float(tok.replace("$", "").replace(",", "").rstrip("%"))
        operands.append((v, tok.endswith("%")))
        return repr(v / 100 if tok.endswith("%") else v)

    py = expr.replace("−", "-").replace("×", "*").replace("÷", "/")
    py = re.sub(r"(?<=[\d\)%])\s*x\s*(?=[\$\d\(])", "*", py)
    py = re.sub(r"\$?\d[\d,]*(?:\.\d+)?%?", operand, py)
    if re.search(r"[^\d\.\s\+\-\*/\(\)e]", py):
        return None, operands
    try:
        return float(eval(py, {"__builtins__": {}}, {})), operands
    except (SyntaxError, ZeroDivisionError, TypeError):
        return None, operands


def _verified_calculations(text: str, allowed: set[float]) -> tuple[set[float], list[str]]:
    """Results of shown calculations whose operands are grounded and whose arithmetic checks out.

    A result can feed a later calculation ("$5.90 ÷ 0.48 = $12.29", then "$12.29 ÷ $8.40 = 1.46"),
    so verification repeats until nothing new is grounded.
    """
    calcs = []
    wrong: list[str] = []
    for expr, result, matched in _calculations(text):
        value, operands = _eval_expr(expr)
        stated = _parse(result)
        if value is None or not operands or stated is None:
            continue
        is_pct = result.endswith("%")
        target = abs(value * 100 if is_pct and abs(value) < 5 else value)
        if abs(target - stated) > max(0.011, 0.006 * stated):
            wrong.append(f"{matched.strip()} (actually {target:.4g})")
            continue
        calcs.append((stated, [op for op in operands if not (op[0] <= COUNT_LIMIT and op[0].is_integer())]))

    verified: set[float] = set()
    changed = True
    while changed:
        changed = False
        for stated, operands in calcs:
            if stated in verified:
                continue
            if all(_matches(v, allowed | verified, pct) for v, pct in operands):
                verified.add(stated)
                changed = True
    return verified, wrong


def check_grounding(transcript: dict[str, Any]) -> CheckResult:
    """Every number the agent states must come from a tool output or the question, or be the
    result of a calculation the agent wrote out, whose inputs are grounded and whose math is right.

    Bare whole numbers up to 100 are treated as counts and not checked.
    """
    allowed = _numbers_in(transcript["question"])
    for call in transcript["tool_calls"]:
        allowed |= _numbers_in(call["input"])
        allowed |= _numbers_in(call["output"])
    text = SKU_RE.sub(" ", transcript["answer"])
    derived, wrong_math = _verified_calculations(re.sub(r"(?<=\d)\s*(?:oz|/oz)\b", "", text), allowed)
    ungrounded = []
    total = 0
    for tok in NUM_RE.findall(text):
        v = _parse(tok)
        if v is None:
            continue
        bare_int = "$" not in tok and "%" not in tok and "." not in tok
        if bare_int and v <= COUNT_LIMIT:
            continue
        total += 1
        is_pct = tok.endswith("%")
        if not (_matches(v, allowed, is_pct) or _matches(v, derived, is_pct)):
            ungrounded.append(tok)
    problems = []
    if ungrounded:
        problems.append(f"Not found in tool output or a shown calculation: {', '.join(ungrounded)}")
    if wrong_math:
        problems.append(f"Wrong arithmetic: {'; '.join(wrong_math)}")
    if problems:
        return CheckResult("grounded_numbers", False, " | ".join(problems))
    return CheckResult("grounded_numbers", True, f"{total} numbers checked, {len(derived)} from shown calculations")


# --- other checks ----------------------------------------------------------

def check_tools_used(transcript: dict[str, Any], required: list[str]) -> CheckResult:
    used = {c["name"] for c in transcript["tool_calls"]}
    missing = [t for t in required if t not in used]
    if missing:
        return CheckResult("tools_used", False, f"Missing: {', '.join(missing)}; used: {', '.join(sorted(used)) or 'none'}")
    return CheckResult("tools_used", True, ", ".join(sorted(used)))


def check_no_tools(transcript: dict[str, Any], forbidden: list[str]) -> CheckResult:
    used = [c["name"] for c in transcript["tool_calls"] if c["name"] in forbidden]
    return CheckResult("no_forbidden_tools", not used, ", ".join(used) if used else "")


def check_mentions(transcript: dict[str, Any], terms: list[str]) -> CheckResult:
    text = transcript["answer"].lower()
    missing = [t for t in terms if t.lower() not in text]
    return CheckResult("mentions", not missing, f"Missing: {', '.join(missing)}" if missing else "")


def check_must_say(transcript: dict[str, Any], patterns: list[str]) -> CheckResult:
    """Each regex must match somewhere in the answer (for ideas with several phrasings)."""
    missing = [p for p in patterns if not re.search(p, transcript["answer"], re.IGNORECASE)]
    return CheckResult("must_say", not missing, f"No match for: {', '.join(missing)}" if missing else "")


def check_not_matches(transcript: dict[str, Any], patterns: list[str]) -> CheckResult:
    hits = [p for p in patterns if re.search(p, transcript["answer"], re.IGNORECASE)]
    return CheckResult("must_not_say", not hits, f"Matched: {', '.join(hits)}" if hits else "")


def check_verdict(transcript: dict[str, Any], tool: str, expected: str,
                  where: dict[str, Any] | None = None) -> CheckResult:
    """The tool returned the expected verdict, and the answer reports it rather than contradicting it.

    where: tool inputs that identify the call to grade (for example the price the user asked about),
    so an agent that goes on to test alternative prices is graded on the question it was asked.
    """
    outs = [c["output"] for c in transcript["tool_calls"]
            if c["name"] == tool and isinstance(c["output"], dict)
            and all(c["input"].get(k) == v for k, v in (where or {}).items())]
    if not outs:
        target = f" with {json.dumps(where)}" if where else ""
        return CheckResult("verdict", False, f"{tool} was never called successfully{target}")
    got = outs[0].get("verdict")
    if got != expected:
        return CheckResult("verdict", False, f"Tool verdict was {got}, expected {expected}")
    words = {"block": r"\b(block|blocked|not allowed|can't|cannot|fails?)\b",
             "warn": r"\b(warn|warning|caution|outside)\b",
             "pass": r"\b(pass|passes|passed|ok|okay|clears?|within)\b"}
    if not re.search(words[expected], transcript["answer"], re.IGNORECASE):
        return CheckResult("verdict", False, f"Answer doesn't communicate '{expected}'")
    return CheckResult("verdict", True, expected)


def check_tool_called_with(transcript: dict[str, Any], spec: dict[str, Any]) -> CheckResult:
    """A specific tool was called with an argument in range and returned a given field value.

    spec: {"tool": ..., "arg": ..., "max": ..., "min": ..., "result_field": ..., "result_value": ...}
    """
    for c in transcript["tool_calls"]:
        if c["name"] != spec["tool"] or not isinstance(c["output"], dict):
            continue
        val = c["input"].get(spec["arg"])
        if val is None:
            continue
        if "max" in spec and not val < spec["max"]:
            continue
        if "min" in spec and not val >= spec["min"]:
            continue
        if "result_field" in spec and c["output"].get(spec["result_field"]) != spec["result_value"]:
            continue
        return CheckResult("tool_called_with", True, f"{spec['tool']}({spec['arg']}={val})")
    return CheckResult("tool_called_with", False, f"No {spec['tool']} call met {json.dumps(spec)}")


def grade(case: dict[str, Any], transcript: dict[str, Any]) -> CaseResult:
    res = CaseResult(case["id"])
    exp = case.get("expect", {})
    if "tools" in exp:
        res.checks.append(check_tools_used(transcript, exp["tools"]))
    if "no_tools" in exp:
        res.checks.append(check_no_tools(transcript, exp["no_tools"]))
    if "verdict" in exp:
        v = exp["verdict"]
        res.checks.append(check_verdict(transcript, v["tool"], v["value"], v.get("where")))
    if "mentions" in exp:
        res.checks.append(check_mentions(transcript, exp["mentions"]))
    if "must_say" in exp:
        res.checks.append(check_must_say(transcript, exp["must_say"]))
    if "must_not_say" in exp:
        res.checks.append(check_not_matches(transcript, exp["must_not_say"]))
    if "tool_called_with" in exp:
        res.checks.append(check_tool_called_with(transcript, exp["tool_called_with"]))
    if exp.get("grounded", True):
        res.checks.append(check_grounding(transcript))
    return res
