"""Pricing analytics. Pure functions over the loaded dataset, no I/O.

Every result carries the inputs and assumptions behind it, so an agent can cite
where a number came from instead of inventing one.
"""

from __future__ import annotations

import itertools
import math
from statistics import median
from typing import Any

import numpy as np

from .data import Dataset, Product

WEEKS_PER_YEAR = 52


class PricingError(ValueError):
    """Raised for bad requests, with a message an agent can show to a user."""


def _product(ds: Dataset, sku: str) -> Product:
    try:
        return ds.products[sku]
    except KeyError:
        known = ", ".join(sorted(ds.products))
        raise PricingError(f"Unknown SKU '{sku}'. Known SKUs: {known}") from None


def _round(x: float, n: int = 2) -> float:
    return float(round(x, n))


# --- Elasticity ------------------------------------------------------------

def price_elasticity(ds: Dataset, sku: str) -> dict[str, Any]:
    """Estimate own-price elasticity with a log-log regression.

    ln(units) = a + e * ln(price) + b1*cos(week) + b2*sin(week)
    The seasonal terms keep a winter or summer peak from being read as a price effect.
    """
    p = _product(ds, sku)
    rows = [r for r in ds.sales if r.sku == sku and r.units > 0]
    if len(rows) < 12:
        raise PricingError(f"Not enough sales history for {sku} ({len(rows)} weeks; need 12).")

    prices = np.array([r.price for r in rows])
    units = np.array([r.units for r in rows], dtype=float)
    weeks = np.array([r.week for r in rows], dtype=float)
    angle = 2 * math.pi * weeks / WEEKS_PER_YEAR
    X = np.column_stack([np.ones(len(rows)), np.log(prices), np.cos(angle), np.sin(angle)])
    y = np.log(units)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(rows) - X.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(cov[1, 1])
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())

    e = float(coef[1])
    price_cv = float(prices.std() / prices.mean())
    if se < 0.15 and price_cv >= 0.05:
        confidence = "high"
    elif se < 0.35:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "sku": sku,
        "name": p.name,
        "elasticity": _round(e),
        "ci95": [_round(e - 1.96 * se), _round(e + 1.96 * se)],
        "r_squared": _round(r2),
        "weeks_used": len(rows),
        "price_range": [_round(float(prices.min())), _round(float(prices.max()))],
        "confidence": confidence,
        "interpretation": (
            f"A 1% price increase changes units by about {e:.2f}%. "
            + ("Demand is elastic: revenue tends to fall when price rises." if e < -1
               else "Demand is inelastic: revenue tends to rise when price rises.")
        ),
        "method": "log-log OLS with annual seasonal terms",
    }


# --- Simulation ------------------------------------------------------------

def simulate_price_change(ds: Dataset, sku: str, new_price: float) -> dict[str, Any]:
    """Project weekly units, revenue and gross margin at a new price."""
    if new_price <= 0:
        raise PricingError("new_price must be greater than zero.")
    p = _product(ds, sku)
    est = price_elasticity(ds, sku)
    e = est["elasticity"]

    # Baseline: average units in weeks sold within 2% of the current price.
    regular = [r.units for r in ds.sales
               if r.sku == sku and abs(r.price / p.current_price - 1) <= 0.02]
    if not regular:
        raise PricingError(f"No weeks near the current price for {sku}; cannot set a baseline.")
    base_units = sum(regular) / len(regular)

    new_units = base_units * (new_price / p.current_price) ** e

    def econ(price: float, units: float) -> dict[str, float]:
        return {
            "price": _round(price),
            "weekly_units": _round(units, 0),
            "weekly_revenue": _round(price * units),
            "weekly_gross_margin": _round((price - p.unit_cost) * units),
            "margin_pct": _round((price - p.unit_cost) / price, 3),
        }

    cur, new = econ(p.current_price, base_units), econ(new_price, new_units)
    return {
        "sku": sku,
        "name": p.name,
        "current": cur,
        "proposed": new,
        "change": {
            "price_pct": _round((new_price / p.current_price - 1) * 100, 1),
            "units_pct": _round((new_units / base_units - 1) * 100, 1),
            "revenue": _round(new["weekly_revenue"] - cur["weekly_revenue"]),
            "gross_margin": _round(new["weekly_gross_margin"] - cur["weekly_gross_margin"]),
        },
        "elasticity_used": e,
        "elasticity_confidence": est["confidence"],
        "assumptions": [
            f"Baseline is the average of {len(regular)} weeks sold within 2% of the current price.",
            "Constant elasticity; competitors do not react; unit cost is unchanged.",
            ("Changes far outside the observed price range "
            f"({est['price_range'][0]}–{est['price_range'][1]}) are less reliable."),
        ],
    }


# --- Guardrails ------------------------------------------------------------

def competitor_median(ds: Dataset, sku: str) -> float | None:
    prices = [c.price for c in ds.competitors if c.sku == sku]
    return median(prices) if prices else None


def check_price_guardrails(ds: Dataset, sku: str, proposed_price: float) -> dict[str, Any]:
    """Check a proposed price against margin, change-size and competitor rules.

    Verdict is "block" if any hard rule fails, "warn" if only soft rules fail, else "pass".
    """
    if proposed_price <= 0:
        raise PricingError("proposed_price must be greater than zero.")
    p = _product(ds, sku)
    pol = ds.policy
    checks: list[dict[str, Any]] = []

    margin = (proposed_price - p.unit_cost) / proposed_price
    floor = pol["min_margin_pct"][p.tier]
    checks.append({
        "rule": "minimum_margin", "severity": "hard", "passed": margin >= floor,
        "detail": f"Margin {margin:.1%} vs {p.tier} floor {floor:.0%} (unit cost ${p.unit_cost:.2f}).",
    })

    change = proposed_price / p.current_price - 1
    max_change = pol["max_price_change_pct"]
    checks.append({
        "rule": "max_price_change", "severity": "hard", "passed": abs(change) <= max_change,
        "detail": f"Change {change:+.1%} from ${p.current_price:.2f}; limit is ±{max_change:.0%} per move.",
    })

    comp = competitor_median(ds, sku)
    if comp:
        index = proposed_price / comp
        lo, hi = pol["competitor_index_band"][p.tier]
        checks.append({
            "rule": "competitor_index", "severity": "soft", "passed": lo <= index <= hi,
            "detail": f"Index {index:.2f} vs median competitor ${comp:.2f}; {p.tier} band is {lo:.2f}–{hi:.2f}.",
        })

    hard_fail = any(not c["passed"] and c["severity"] == "hard" for c in checks)
    soft_fail = any(not c["passed"] and c["severity"] == "soft" for c in checks)
    verdict = "block" if hard_fail else "warn" if soft_fail else "pass"
    return {
        "sku": sku,
        "name": p.name,
        "proposed_price": _round(proposed_price),
        "current_price": p.current_price,
        "verdict": verdict,
        "checks": checks,
    }


# --- Price ladder ----------------------------------------------------------

TIER_ORDER = {"value": 0, "core": 1, "premium": 2}


def check_price_ladder(ds: Dataset, category: str) -> dict[str, Any]:
    """Find ladder breaks in a category.

    Size rule: within a tier, a bigger pack should cost less per ounce.
    Tier rule: at the same size, value < core < premium, with at least the policy's minimum gap.
    """
    items = [p for p in ds.products.values() if p.category == category]
    if not items:
        cats = ", ".join(sorted({p.category for p in ds.products.values()}))
        raise PricingError(f"Unknown category '{category}'. Categories: {cats}")

    gap = ds.policy["min_tier_gap_pct"]
    violations: list[dict[str, Any]] = []

    tiers = sorted({p.tier for p in items}, key=TIER_ORDER.get)
    for tier in tiers:
        row = sorted((p for p in items if p.tier == tier), key=lambda p: p.size_oz)
        for small, big in itertools.pairwise(row):
            if big.price_per_oz >= small.price_per_oz:
                violations.append({
                    "type": "size",
                    "skus": [small.sku, big.sku],
                    "detail": (f"{big.sku} costs ${big.price_per_oz:.3f}/oz, more than "
                               f"{small.sku} at ${small.price_per_oz:.3f}/oz."),
                })

    for size in sorted({p.size_oz for p in items}):
        row = sorted((p for p in items if p.size_oz == size), key=lambda p: TIER_ORDER[p.tier])
        for lower, upper in itertools.pairwise(row):
            needed = lower.current_price * (1 + gap)
            if upper.current_price < needed:
                kind = "below" if upper.current_price <= lower.current_price else "too close to"
                violations.append({
                    "type": "tier",
                    "skus": [lower.sku, upper.sku],
                    "detail": (f"{upper.tier.title()} {upper.sku} at ${upper.current_price:.2f} is {kind} "
                               f"{lower.tier} {lower.sku} at ${lower.current_price:.2f}; "
                               f"policy needs at least {gap:.0%} more (${needed:.2f})."),
                })

    return {
        "category": category,
        "products_checked": len(items),
        "violations": violations,
        "ok": not violations,
        "ladder": [
            {"sku": p.sku, "tier": p.tier, "size_oz": p.size_oz,
             "price": p.current_price, "price_per_oz": _round(p.price_per_oz, 3)}
            for p in sorted(items, key=lambda p: (TIER_ORDER[p.tier], p.size_oz))
        ],
    }


# --- Competitive position -------------------------------------------------

def competitor_price_index(ds: Dataset, category: str) -> dict[str, Any]:
    """Our price versus the median competitor price for each SKU in a category."""
    items = [p for p in ds.products.values() if p.category == category]
    if not items:
        cats = ", ".join(sorted({p.category for p in ds.products.values()}))
        raise PricingError(f"Unknown category '{category}'. Categories: {cats}")
    rows = []
    for p in sorted(items, key=lambda p: (TIER_ORDER[p.tier], p.size_oz)):
        comp = competitor_median(ds, p.sku)
        if comp is None:
            continue
        lo, hi = ds.policy["competitor_index_band"][p.tier]
        idx = p.current_price / comp
        rows.append({
            "sku": p.sku, "tier": p.tier, "our_price": p.current_price,
            "median_competitor": _round(comp), "index": _round(idx),
            "band": [lo, hi],
            "position": "below band" if idx < lo else "above band" if idx > hi else "in band",
        })
    return {"category": category, "skus": rows}


def list_products(ds: Dataset, category: str | None = None) -> dict[str, Any]:
    items = [p for p in ds.products.values() if category in (None, "", p.category)]
    if category and not items:
        cats = ", ".join(sorted({p.category for p in ds.products.values()}))
        raise PricingError(f"Unknown category '{category}'. Categories: {cats}")
    return {
        "products": [
            {"sku": p.sku, "name": p.name, "category": p.category, "tier": p.tier,
             "size_oz": p.size_oz, "current_price": p.current_price, "unit_cost": p.unit_cost}
            for p in sorted(items, key=lambda p: (p.category, TIER_ORDER[p.tier], p.size_oz))
        ],
        "policy": ds.policy,
    }
