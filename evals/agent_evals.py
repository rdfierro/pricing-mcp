"""Run Claude against the MCP server's tools and grade the answers.

Needs ANTHROPIC_API_KEY. Each case is a fresh conversation. Tools are called through the
server's own MCP tool layer (same schemas, validation and errors a real client sees).

    python -m evals.agent_evals                  # all cases
    python -m evals.agent_evals --case ladder-ground --repeat 3
    PRICING_EVAL_MODEL=claude-sonnet-5-5 python -m evals.agent_evals

Writes a JSON report with full transcripts to evals/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from mcp.server.mcpserver.exceptions import ToolError

from pricing_mcp.server import INSTRUCTIONS, mcp

from .grading import grade

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = os.environ.get("PRICING_EVAL_MODEL", "claude-sonnet-5-5")
MAX_TURNS = 8
SYSTEM = (
    "You are a pricing analyst's assistant for a coffee category. "
    "Use the tools to answer; never guess numbers. " + INSTRUCTIONS
)


async def tool_specs() -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in await mcp.list_tools()]


async def call_tool(name: str, args: dict[str, Any]) -> tuple[Any, bool]:
    try:
        result = await mcp.call_tool(name, args)
    except ToolError as exc:  # the agent sees the same error text an MCP client would
        return str(exc), True
    text = "".join(getattr(c, "text", "") for c in result.content)
    try:
        return json.loads(text), bool(result.is_error)
    except json.JSONDecodeError:
        return text, bool(result.is_error)


async def run_case(client, model: str, tools: list[dict], question: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    calls: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    answer = ""
    for _ in range(MAX_TURNS):
        resp = await asyncio.to_thread(
            client.messages.create, model=model, max_tokens=1500, system=SYSTEM,
            tools=tools, messages=messages,
        )
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            answer = "".join(b.text for b in resp.content if b.type == "text")
            break
        results = []
        for tu in tool_uses:
            out, is_err = await call_tool(tu.name, tu.input)
            calls.append({"name": tu.name, "input": tu.input, "output": out, "is_error": is_err})
            results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": is_err,
                            "content": out if isinstance(out, str) else json.dumps(out)})
        messages.append({"role": "user", "content": results})
    return {"question": question, "tool_calls": calls, "answer": answer, "usage": usage}


async def main_async(args: argparse.Namespace) -> int:
    try:
        import anthropic
    except ImportError:
        print("Install the eval extras: pip install -e '.[evals]'")
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run agent evals. Tool evals run without it: python -m evals.tool_evals")
        return 2

    cases = yaml.safe_load((HERE / "agent_cases.yaml").read_text())
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    client = anthropic.Anthropic()
    tools = await tool_specs()

    report = {"model": args.model, "started": datetime.now(UTC).isoformat(), "runs": []}
    for case in cases:
        for i in range(args.repeat):
            t0 = time.perf_counter()
            try:
                transcript = await run_case(client, args.model, tools, case["question"])
            except anthropic.APIStatusError as exc:
                # Billing, auth and bad-model errors affect every case, so stop with a clear message.
                msg = exc.body.get("error", {}).get("message") if isinstance(exc.body, dict) else None
                print(f"API error {exc.status_code}: {msg or exc.message}")
                return 2
            except anthropic.APIConnectionError:
                print("Could not reach the Anthropic API. Check your network connection.")
                return 2
            result = grade(case, transcript)
            report["runs"].append({
                "case": case["id"], "attempt": i + 1, "passed": result.passed,
                "seconds": round(time.perf_counter() - t0, 1),
                "checks": [c.__dict__ for c in result.checks], "transcript": transcript,
            })
            mark = "PASS" if result.passed else "FAIL"
            fails = "; ".join(f"{c.name}: {c.detail}" for c in result.checks if not c.passed)
            print(f"{mark}  {case['id']:<28} {fails}")

    runs = report["runs"]
    passed = sum(r["passed"] for r in runs)
    report["summary"] = {
        "passed": passed, "total": len(runs),
        "pass_rate": round(passed / len(runs), 3) if runs else 0,
        "tool_calls": sum(len(r["transcript"]["tool_calls"]) for r in runs),
        "input_tokens": sum(r["transcript"]["usage"]["input_tokens"] for r in runs),
        "output_tokens": sum(r["transcript"]["usage"]["output_tokens"] for r in runs),
    }
    out = HERE / "results" / f"{datetime.now():%Y%m%d-%H%M%S}-{args.model}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    s = report["summary"]
    print(f"\n{s['passed']}/{s['total']} passed ({s['pass_rate']:.0%}) · {s['tool_calls']} tool calls · "
          f"{s['input_tokens']:,} in / {s['output_tokens']:,} out tokens\nReport: {out}")
    return 0 if passed == len(runs) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--case", action="append", help="run only this case id (repeatable)")
    p.add_argument("--repeat", type=int, default=1, help="runs per case, to measure consistency")
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
