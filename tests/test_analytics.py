import asyncio
import json

import pytest

from pricing_mcp import analytics
from pricing_mcp.data import load_dataset
from pricing_mcp.server import mcp


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_elasticity_has_interval_and_label(ds):
    r = analytics.price_elasticity(ds, "WB-COR-12")
    assert r["ci95"][0] < r["elasticity"] < r["ci95"][1]
    assert r["confidence"] in {"high", "medium", "low"}
    assert r["weeks_used"] == 52


def test_unknown_sku_lists_known_skus(ds):
    with pytest.raises(analytics.PricingError, match="Known SKUs"):
        analytics.price_elasticity(ds, "NOPE")


def test_negative_price_rejected(ds):
    with pytest.raises(analytics.PricingError):
        analytics.check_price_guardrails(ds, "WB-COR-12", -1)


def test_simulation_reports_assumptions(ds):
    r = analytics.simulate_price_change(ds, "CB-COR-32", 8.39)
    assert r["assumptions"]
    assert r["change"]["revenue"] > 0  # inelastic: a small rise lifts revenue


def test_clean_category_has_no_ladder_breaks(ds):
    assert analytics.check_price_ladder(ds, "whole_bean")["ok"]


def test_server_exposes_read_only_tools():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {"list_products", "get_price_elasticity", "simulate_price_change",
                     "check_price_guardrails", "check_price_ladder", "competitor_price_index"}
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)


def test_server_call_returns_json():
    res = asyncio.run(mcp.call_tool("check_price_ladder", {"category": "ground"}))
    body = json.loads(res.content[0].text)
    assert len(body["violations"]) == 2


def test_tool_evals_all_pass():
    from evals.tool_evals import run
    failed = [r for r in run() if not r["passed"]]
    assert not failed, failed
