"""Generate the synthetic coffee-category dataset used by the server and the evals.

Everything here is invented. The data is generated from known "true" price
elasticities and seeded with a few deliberate pricing problems, so the eval
suite can check whether the tools (and an agent using them) find the right
answers. Ground truth is written to evals/ground_truth.json, which the server
never reads.

Run:  python scripts/generate_data.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "pricing_mcp" / "data"
EVALS = ROOT / "evals"
SEED = 20261027
WEEKS = 52

# sku, name, category, tier, size_oz, unit_cost, regular_price, true_elasticity, base_weekly_units
PRODUCTS = [
    # Whole bean: value / core / premium across three bag sizes
    ("WB-VAL-12", "Morning Basics Whole Bean 12 oz", "whole_bean", "value", 12, 3.10, 6.49, -2.4, 900),
    ("WB-VAL-24", "Morning Basics Whole Bean 24 oz", "whole_bean", "value", 24, 5.70, 11.49, -2.2, 420),
    ("WB-COR-12", "Ridgeline Roasters Whole Bean 12 oz", "whole_bean", "core", 12, 4.20, 9.99, -1.8, 700),
    ("WB-COR-24", "Ridgeline Roasters Whole Bean 24 oz", "whole_bean", "core", 24, 7.80, 17.99, -1.6, 300),
    ("WB-COR-32", "Ridgeline Roasters Whole Bean 32 oz", "whole_bean", "core", 32, 10.10, 22.99, -1.5, 160),
    ("WB-PRE-12", "Cascade Reserve Whole Bean 12 oz", "whole_bean", "premium", 12, 6.40, 14.99, -1.1, 380),
    ("WB-PRE-24", "Cascade Reserve Whole Bean 24 oz", "whole_bean", "premium", 24, 11.90, 27.99, -1.0, 140),
    # Ground: one deliberate ladder break (32 oz costs more per ounce than 24 oz)
    ("GR-VAL-12", "Morning Basics Ground 12 oz", "ground", "value", 12, 2.90, 5.99, -2.6, 1100),
    ("GR-COR-12", "Ridgeline Roasters Ground 12 oz", "ground", "core", 12, 3.90, 9.49, -1.9, 820),
    ("GR-COR-24", "Ridgeline Roasters Ground 24 oz", "ground", "core", 24, 7.20, 16.99, -1.7, 360),
    ("GR-COR-32", "Ridgeline Roasters Ground 32 oz", "ground", "core", 32, 9.70, 23.99, -1.6, 150),
    # Deliberate tier break: premium priced below core at the same size
    ("GR-PRE-12", "Cascade Reserve Ground 12 oz", "ground", "premium", 12, 5.90, 9.29, -1.2, 410),
    # Cold brew concentrate: small, low elasticity
    ("CB-COR-32", "Ridgeline Cold Brew Concentrate 32 oz", "cold_brew", "core", 32, 3.60, 7.99, -0.9, 520),
    ("CB-PRE-32", "Cascade Reserve Cold Brew 32 oz", "cold_brew", "premium", 32, 4.80, 10.99, -0.8, 260),
]

COMPETITORS = ["Northside Market", "ValuMart", "Harbor Grocers"]

# Deliberate competitive gaps: competitors far below our premium 12 oz,
# and well above our value ground coffee.
COMPETITOR_OVERRIDES = {"WB-PRE-12": 0.76, "GR-VAL-12": 1.14}

POLICY = {
    "currency": "USD",
    "min_margin_pct": {"value": 0.40, "core": 0.50, "premium": 0.52},
    "max_price_change_pct": 0.15,
    "competitor_index_band": {"value": [0.90, 1.05], "core": [0.95, 1.12], "premium": [1.00, 1.25]},
    "min_tier_gap_pct": 0.10,
    "notes": "Margin is (price - unit_cost) / price. Competitor index is our price / median competitor price.",
}


def seasonality(week: int) -> float:
    # Mild winter lift for hot coffee, summer lift handled per category below.
    return 1.0 + 0.12 * math.cos(2 * math.pi * (week - 2) / WEEKS)


def main() -> None:
    rng = random.Random(SEED)
    DATA.mkdir(parents=True, exist_ok=True)
    EVALS.mkdir(parents=True, exist_ok=True)

    with open(DATA / "products.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sku", "name", "category", "tier", "size_oz", "unit_cost", "current_price"])
        for sku, name, cat, tier, size, cost, price, *_ in PRODUCTS:
            w.writerow([sku, name, cat, tier, size, f"{cost:.2f}", f"{price:.2f}"])

    truth = {"elasticity": {}, "ladder_violations": [], "seed": SEED}
    with open(DATA / "weekly_sales.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sku", "week", "price", "units"])
        for sku, _n, cat, _t, _s, _c, reg, elasticity, base in PRODUCTS:
            truth["elasticity"][sku] = elasticity
            for week in range(1, WEEKS + 1):
                # Price varies: regular weeks with small noise, promo weeks 10-25% off,
                # and occasional small list-price tests, so there is enough variation to estimate from.
                r = rng.random()
                if r < 0.22:
                    price = reg * (1 - rng.uniform(0.10, 0.25))
                elif r < 0.32:
                    price = reg * (1 + rng.uniform(0.03, 0.08))
                else:
                    price = reg * (1 + rng.uniform(-0.02, 0.02))
                price = round(price, 2)
                season = seasonality(week)
                if cat == "cold_brew":
                    season = 2 - season  # cold brew peaks in summer
                expected = base * (price / reg) ** elasticity * season
                units = max(0, round(expected * math.exp(rng.gauss(0, 0.08))))
                w.writerow([sku, week, f"{price:.2f}", units])

    with open(DATA / "competitor_prices.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sku", "competitor", "price"])
        for sku, _n, _c, tier, _s, _cost, reg, *_ in PRODUCTS:
            # Competitors sit around our price; premium competitors run a bit lower.
            center = {"value": 1.00, "core": 0.97, "premium": 0.90}[tier]
            center = COMPETITOR_OVERRIDES.get(sku, center)
            for comp in COMPETITORS:
                w.writerow([sku, comp, f"{reg * center * rng.uniform(0.94, 1.06):.2f}"])

    (DATA / "policy.json").write_text(json.dumps(POLICY, indent=2) + "\n")

    truth["ladder_violations"] = [
        {"category": "ground", "type": "size", "skus": ["GR-COR-24", "GR-COR-32"]},
        {"category": "ground", "type": "tier", "skus": ["GR-COR-12", "GR-PRE-12"]},
    ]
    truth["competitor_out_of_band"] = {"WB-PRE-12": "above band", "GR-VAL-12": "below band"}
    (EVALS / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(f"Wrote {len(PRODUCTS)} products and {len(PRODUCTS) * WEEKS} sales rows to {DATA}")


if __name__ == "__main__":
    main()
