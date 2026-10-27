"""Tests for the graders, using hand-written transcripts (not real model output)."""

from evals.grading import check_grounding, check_tool_called_with, check_verdict, grade

GUARD_OUT = {
    "sku": "GR-PRE-12", "proposed_price": 10.49, "current_price": 9.29, "verdict": "block",
    "checks": [{"rule": "minimum_margin", "severity": "hard", "passed": False,
                "detail": "Margin 43.8% vs premium floor 52% (unit cost $5.90)."}],
}


def tx(answer, calls=None, question="Can we move GR-PRE-12 to $10.49?"):
    return {"question": question, "tool_calls": calls or [], "answer": answer}


def guard_call(price=10.49, out=GUARD_OUT):
    return [{"name": "check_price_guardrails", "input": {"sku": "GR-PRE-12", "proposed_price": price},
             "output": out, "is_error": False}]


def test_grounded_answer_passes():
    t = tx("No. At $10.49 the margin is 43.8%, under the 52% floor for premium.", guard_call())
    assert check_grounding(t).passed


def test_invented_number_fails():
    t = tx("No. Margin would be 41.2%, under the 52% floor.", guard_call())
    r = check_grounding(t)
    assert not r.passed and "41.2%" in r.detail


def test_sku_digits_and_small_counts_are_ignored():
    t = tx("GR-PRE-12 has 2 problems: margin 43.8% is below 52%.", guard_call())
    assert check_grounding(t).passed


def test_percent_matches_fraction_in_output():
    calls = [{"name": "simulate_price_change", "input": {}, "output": {"margin_pct": 0.438}, "is_error": False}]
    assert check_grounding(tx("Margin lands at 43.8%.", calls, question="q")).passed


def test_verdict_must_match_tool_and_answer():
    assert check_verdict(tx("That price is blocked by the margin rule.", guard_call()),
                         "check_price_guardrails", "block").passed
    assert not check_verdict(tx("Looks fine to me.", guard_call()), "check_price_guardrails", "block").passed


def test_tool_called_with_range():
    ok = dict(GUARD_OUT, verdict="pass")
    spec = {"tool": "check_price_guardrails", "arg": "proposed_price", "max": 22.65,
            "result_field": "verdict", "result_value": "pass"}
    assert check_tool_called_with(tx("", guard_call(22.49, ok)), spec).passed
    assert not check_tool_called_with(tx("", guard_call(22.99, ok)), spec).passed


def test_grade_combines_checks():
    case = {"id": "x", "expect": {"tools": ["check_price_guardrails"], "mentions": ["margin"],
                                  "verdict": {"tool": "check_price_guardrails", "value": "block"}}}
    good = grade(case, tx("Blocked: margin 43.8% is under the 52% floor.", guard_call()))
    bad = grade(case, tx("Blocked: margin 40% is too low.", guard_call()))
    assert good.passed
    assert not bad.passed
