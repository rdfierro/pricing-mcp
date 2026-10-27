"""Deterministic evals for the tools themselves, checked against the data generator's ground truth.

No model or API key needed; this runs in CI on every push.

    python -m evals.tool_evals
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pricing_mcp import analytics
from pricing_mcp.data import load_dataset

HERE = Path(__file__).resolve().parent
ELASTICITY_TOLERANCE = 0.30

# (sku, proposed_price, expected verdict, rules expected to fail)
GUARDRAIL_CASES = [
    ("WB-COR-24", 18.99, "pass", []),
    ("GR-COR-32", 22.49, "pass", []),
    ("GR-PRE-12", 10.49, "block", ["minimum_margin"]),
    ("GR-VAL-12", 4.99, "block", ["max_price_change", "competitor_index"]),
    ("GR-COR-32", 21.99, "warn", ["competitor_index"]),
    ("WB-COR-12", 8.00, "block", ["max_price_change", "minimum_margin", "competitor_index"]),
    ("CB-PRE-32", 11.49, "pass", []),
]


def run() -> list[dict]:
    ds = load_dataset()
    truth = json.loads((HERE / "ground_truth.json").read_text())
    results: list[dict] = []

    for sku, true_e in truth["elasticity"].items():
        est = analytics.price_elasticity(ds, sku)
        lo, hi = est["ci95"]
        err = abs(est["elasticity"] - true_e)
        results.append({
            "suite": "elasticity", "case": sku,
            "passed": err <= ELASTICITY_TOLERANCE and lo <= true_e <= hi,
            "detail": f"true {true_e}, estimate {est['elasticity']} (95% CI {lo} to {hi}), error {err:.2f}",
        })

    found = []
    for cat in sorted({p.category for p in ds.products.values()}):
        for v in analytics.check_price_ladder(ds, cat)["violations"]:
            found.append({"category": cat, "type": v["type"], "skus": v["skus"]})
    expected = truth["ladder_violations"]
    key = lambda v: (v["category"], v["type"], tuple(v["skus"]))
    missed = [v for v in expected if key(v) not in {key(f) for f in found}]
    extra = [v for v in found if key(v) not in {key(e) for e in expected}]
    results.append({
        "suite": "ladder", "case": "all categories", "passed": not missed and not extra,
        "detail": f"expected {len(expected)}, found {len(found)}; missed {missed or 'none'}; extra {extra or 'none'}",
    })

    for sku, pos in truth["competitor_out_of_band"].items():
        cat = ds.products[sku].category
        row = next(r for r in analytics.competitor_price_index(ds, cat)["skus"] if r["sku"] == sku)
        results.append({"suite": "competitor", "case": sku, "passed": row["position"] == pos,
                        "detail": f"expected {pos}, got {row['position']} (index {row['index']})"})
    flagged = sorted(r["sku"] for c in {p.category for p in ds.products.values()}
                     for r in analytics.competitor_price_index(ds, c)["skus"] if r["position"] != "in band")
    results.append({"suite": "competitor", "case": "no false flags",
                    "passed": flagged == sorted(truth["competitor_out_of_band"]),
                    "detail": f"flagged {flagged}"})

    for sku, price, verdict, rules in GUARDRAIL_CASES:
        g = analytics.check_price_guardrails(ds, sku, price)
        failed = sorted(c["rule"] for c in g["checks"] if not c["passed"])
        results.append({"suite": "guardrails", "case": f"{sku} @ ${price}",
                        "passed": g["verdict"] == verdict and failed == sorted(rules),
                        "detail": f"expected {verdict} {sorted(rules)}, got {g['verdict']} {failed}"})

    # Simulation should agree with economics: for elastic demand, a price rise lowers revenue.
    for sku, true_e in truth["elasticity"].items():
        p = ds.products[sku]
        sim = analytics.simulate_price_change(ds, sku, round(p.current_price * 1.05, 2))
        e = sim["elasticity_used"]
        expected_sign = -1 if e < -1 else 1
        got = sim["change"]["revenue"]
        results.append({"suite": "simulation", "case": f"{sku} +5%",
                        "passed": (got < 0) == (expected_sign < 0) and sim["change"]["units_pct"] < 0,
                        "detail": f"elasticity {e}, revenue change {got}, units {sim['change']['units_pct']}%"})
    return results


def main() -> int:
    results = run()
    width = max(len(r["case"]) for r in results)
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{mark}  {r['suite']:<11} {r['case']:<{width}}  {r['detail']}")
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} tool evals passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
