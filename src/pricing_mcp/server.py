"""MCP server exposing read-only pricing analytics tools.

Run over stdio (for Claude Desktop, Claude Code, or any MCP client):
    pricing-mcp
or over streamable HTTP:
    pricing-mcp --http --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import analytics
from .data import default_dataset

log = logging.getLogger("pricing_mcp")

INSTRUCTIONS = """\
Pricing analytics for a packaged-coffee category (synthetic data).
Every tool is read-only. Use list_products first if you don't know the SKUs.
Quote numbers exactly as the tools return them, and say which tool they came from.
A guardrail verdict of "block" means the price must not be used as proposed.
"""

mcp = MCPServer("pricing-mcp", instructions=INSTRUCTIONS)
READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)


def _audited(name: str, fn: Callable[[], dict[str, Any]], **args: Any) -> dict[str, Any]:
    """Run a tool, log one structured audit line, and turn bad input into a clear error."""
    start = time.perf_counter()
    try:
        result = fn()
        status = "ok"
        return result
    except analytics.PricingError as exc:
        status = "rejected"
        raise ToolError(str(exc)) from None
    finally:
        log.info(json.dumps({
            "event": "tool_invocation", "tool": name, "args": args, "status": status,
            "ms": round((time.perf_counter() - start) * 1000, 1),
        }))


@mcp.tool(annotations=READ_ONLY)
def list_products(category: str | None = None) -> dict[str, Any]:
    """List SKUs with tier, pack size, current price and unit cost, plus the pricing policy.

    category: optional filter, one of whole_bean, ground, cold_brew.
    """
    return _audited("list_products", lambda: analytics.list_products(default_dataset(), category),
                    category=category)


@mcp.tool(annotations=READ_ONLY)
def get_price_elasticity(sku: str) -> dict[str, Any]:
    """Estimate how sensitive a SKU's unit sales are to price, from 52 weeks of sales.

    Returns the elasticity, a 95% confidence interval, R², and a confidence label.
    """
    return _audited("get_price_elasticity", lambda: analytics.price_elasticity(default_dataset(), sku),
                    sku=sku)


@mcp.tool(annotations=READ_ONLY)
def simulate_price_change(sku: str, new_price: float) -> dict[str, Any]:
    """Project weekly units, revenue and gross margin if a SKU moves to new_price.

    Uses the SKU's estimated elasticity. Read the returned assumptions before relying on it.
    """
    return _audited("simulate_price_change",
                    lambda: analytics.simulate_price_change(default_dataset(), sku, new_price),
                    sku=sku, new_price=new_price)


@mcp.tool(annotations=READ_ONLY)
def check_price_guardrails(sku: str, proposed_price: float) -> dict[str, Any]:
    """Check a proposed price against the pricing policy.

    Hard rules (minimum margin, maximum change per move) produce "block".
    The soft rule (competitor price index band) produces "warn". Otherwise "pass".
    """
    return _audited("check_price_guardrails",
                    lambda: analytics.check_price_guardrails(default_dataset(), sku, proposed_price),
                    sku=sku, proposed_price=proposed_price)


@mcp.tool(annotations=READ_ONLY)
def check_price_ladder(category: str) -> dict[str, Any]:
    """Find price-ladder breaks in a category.

    Flags bigger packs that cost more per ounce, and higher tiers priced too close to or below lower tiers.
    """
    return _audited("check_price_ladder",
                    lambda: analytics.check_price_ladder(default_dataset(), category),
                    category=category)


@mcp.tool(annotations=READ_ONLY)
def competitor_price_index(category: str) -> dict[str, Any]:
    """Compare each SKU's price with the median competitor price, against its tier's target band."""
    return _audited("competitor_price_index",
                    lambda: analytics.competitor_price_index(default_dataset(), category),
                    category=category)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pricing analytics MCP server")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Logs go to stderr so they never corrupt the stdio protocol stream.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.http:
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
