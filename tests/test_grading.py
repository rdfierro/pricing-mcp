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


def test_shown_calculation_is_verified_and_grounds_result():
    t = tx("Blocked. A 52% margin needs ($5.90 ÷ (1 − 0.52) = $12.29).", guard_call())
    r = check_grounding(t)
    assert r.passed, r.detail


def test_wrong_arithmetic_fails():
    t = tx("Blocked. A 52% margin needs ($5.90 ÷ (1 − 0.52) = $11.80).", guard_call())
    r = check_grounding(t)
    assert not r.passed and "Wrong arithmetic" in r.detail


def test_calculation_with_invented_input_does_not_ground_result():
    t = tx("Blocked. A 52% margin needs ($6.40 ÷ (1 − 0.52) = $13.33).", guard_call())
    r = check_grounding(t)
    assert not r.passed and "$13.33" in r.detail


def test_unshown_derived_number_fails():
    t = tx("Blocked. You'd need about $12.29 to hit the floor.", guard_call())
    assert not check_grounding(t).passed


def test_counts_are_not_checked():
    t = tx("Blocked. I checked 14 SKUs and 3 rules; margin is 43.8%.", guard_call())
    assert check_grounding(t).passed


def test_verdict_uses_the_price_that_was_asked():
    warn = dict(GUARD_OUT, verdict="warn")
    calls = guard_call() + guard_call(11.20, warn)
    t = tx("$10.49 is blocked; $11.20 only warns.", calls)
    assert check_verdict(t, "check_price_guardrails", "block",
                         {"sku": "GR-PRE-12", "proposed_price": 10.49}).passed


def test_chained_calculations_and_policy_multipliers():
    calls = guard_call() + [{"name": "competitor_price_index", "input": {}, "is_error": False,
                             "output": {"median_competitor": 8.40, "note": "limit is ±15% per move"}}]
    t = tx("Needs ($5.90 ÷ (1 − 0.52) = $12.29), which is an index of ($12.29 ÷ $8.40 = 1.46). "
           "The most one move allows is ($9.29 × 1.15 = $10.68).", calls)
    r = check_grounding(t)
    assert r.passed, r.detail


def test_calculation_formats_seen_in_real_answers():
    calls = guard_call() + [{"name": "list_products", "input": {}, "is_error": False,
                             "output": {"price": 16.99, "median_competitor": 8.40, "current_price": 9.29}}]
    t = tx("The 32 oz would need about $22.65 ($16.99 ÷ 24 × 32 = $22.65). "
           "Margin today is 36.5% ($9.29 − $5.90 = $3.39, and $3.39 ÷ $9.29 = 36.5%). "
           "Index would be about 1.46 ($12.29 ÷ $8.40), with $12.29 from ($5.90 ÷ (1 − 0.52) = $12.29).", calls)
    r = check_grounding(t)
    assert r.passed, r.detail


def test_units_policy_fields_and_percent_forms():
    calls = [{"name": "list_products", "input": {}, "is_error": False,
              "output": {"price_per_oz": 0.708, "current_price": 9.29, "unit_cost": 5.90,
                         "policy": {"max_price_change_pct": 0.15}}}]
    t = tx("Below $0.708 × 32 oz = $22.65. One move tops out near $10.68 ($9.29 × 1.15). "
           "Margin is about 36.5% (($9.29 − $5.90) ÷ $9.29 = 0.365).", calls, question="q")
    r = check_grounding(t)
    assert r.passed, r.detail


def test_prose_derivation_without_math_still_fails():
    calls = [{"name": "list_products", "input": {}, "is_error": False,
              "output": {"current_price": 23.99, "policy": {"max_price_change_pct": 0.15}}}]
    t = tx("A move of ±15% from $23.99 gives about $20.39 at the low end.", calls, question="q")
    assert not check_grounding(t).passed


def test_result_first_with_nested_parentheses():
    t = tx("Margin is 36.5% (($9.29 − $5.90) ÷ $9.29); 52% needs $12.29 ($5.90 ÷ (1 − 0.52)).",
           guard_call())
    r = check_grounding(t)
    assert r.passed, r.detail


def test_value_with_change_in_parentheses_is_not_a_calculation():
    calls = [{"name": "simulate_price_change", "input": {}, "is_error": False,
              "output": {"weekly_units": 361.0, "units_pct": -13.2, "weekly_revenue": 3782.94, "revenue": -78.94}}]
    t = tx("Units fall to 361 (−13.2%) and revenue to $3,782.94 (−$78.94).", calls, question="q")
    r = check_grounding(t)
    assert r.passed, r.detail
