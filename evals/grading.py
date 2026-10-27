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
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _numbers_in(v)
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


def check_grounding(transcript: dict[str, Any], ignore_below: float = 1.0) -> CheckResult:
    """Every number the agent states must come from a tool output or the question.

    Small bare integers (counts like "2 violations") are ignored by default.
    """
    allowed = _numbers_in(transcript["question"])
    for call in transcript["tool_calls"]:
        allowed |= _numbers_in(call["input"])
        allowed |= _numbers_in(call["output"])
    text = SKU_RE.sub(" ", transcript["answer"])
    ungrounded = []
    total = 0
    for tok in NUM_RE.findall(text):
        v = _parse(tok)
        if v is None:
            continue
        bare_int = "$" not in tok and "%" not in tok and "." not in tok
        if bare_int and v <= 10:
            continue
        if v < ignore_below and bare_int:
            continue
        total += 1
        if not _matches(v, allowed, tok.endswith("%")):
            ungrounded.append(tok)
    if ungrounded:
        return CheckResult("grounded_numbers", False, f"Not found in tool output: {', '.join(ungrounded)}")
    return CheckResult("grounded_numbers", True, f"{total} numbers checked")


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


def check_not_matches(transcript: dict[str, Any], patterns: list[str]) -> CheckResult:
    hits = [p for p in patterns if re.search(p, transcript["answer"], re.IGNORECASE)]
    return CheckResult("must_not_say", not hits, f"Matched: {', '.join(hits)}" if hits else "")


def check_verdict(transcript: dict[str, Any], tool: str, expected: str) -> CheckResult:
    """The tool returned the expected verdict, and the answer reports it rather than contradicting it."""
    outs = [c["output"] for c in transcript["tool_calls"] if c["name"] == tool and isinstance(c["output"], dict)]
    if not outs:
        return CheckResult("verdict", False, f"{tool} was never called successfully")
    got = outs[-1].get("verdict")
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
        res.checks.append(check_verdict(transcript, exp["verdict"]["tool"], exp["verdict"]["value"]))
    if "mentions" in exp:
        res.checks.append(check_mentions(transcript, exp["mentions"]))
    if "must_not_say" in exp:
        res.checks.append(check_not_matches(transcript, exp["must_not_say"]))
    if "tool_called_with" in exp:
        res.checks.append(check_tool_called_with(transcript, exp["tool_called_with"]))
    if exp.get("grounded", True):
        res.checks.append(check_grounding(transcript))
    return res
