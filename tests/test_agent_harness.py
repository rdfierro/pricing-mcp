"""Exercise the agent loop with a scripted fake client, so the plumbing is tested without an API key."""

import asyncio
from dataclasses import asdict, dataclass

import yaml

from evals.agent_evals import HERE, run_case, tool_specs
from evals.grading import grade


@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict | None = None

    def model_dump(self):
        return {k: v for k, v in asdict(self).items() if v not in ("", None)}


@dataclass
class Usage:
    input_tokens: int = 100
    output_tokens: int = 20


@dataclass
class Resp:
    content: list


class ScriptedClient:
    """Plays back a fixed sequence of model turns."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []
        self.messages = self

    def create(self, **kwargs):
        self.seen.append(kwargs["messages"][-1])
        r = Resp(self.turns.pop(0))
        r.usage = Usage()
        return r


def case(case_id):
    return next(c for c in yaml.safe_load((HERE / "agent_cases.yaml").read_text()) if c["id"] == case_id)


def test_loop_runs_tools_and_grades():
    client = ScriptedClient([
        [Block("tool_use", id="t1", name="check_price_guardrails",
               input={"sku": "GR-PRE-12", "proposed_price": 10.49})],
        [Block("text", text="No. $10.49 is blocked: margin would be 43.8%, under the 52% premium floor.")],
    ])
    tools = asyncio.run(tool_specs())
    t = asyncio.run(run_case(client, "fake", tools, case("guardrail-block-margin")["question"]))
    assert t["tool_calls"][0]["output"]["verdict"] == "block"
    assert client.seen[-1]["content"][0]["type"] == "tool_result"
    assert grade(case("guardrail-block-margin"), t).passed


def test_tool_errors_reach_the_model():
    client = ScriptedClient([
        [Block("tool_use", id="t1", name="get_price_elasticity", input={"sku": "XX-ZZZ-99"})],
        [Block("text", text="That SKU does not exist in the catalog.")],
    ])
    tools = asyncio.run(tool_specs())
    t = asyncio.run(run_case(client, "fake", tools, case("unknown-sku")["question"]))
    assert t["tool_calls"][0]["is_error"]
    assert "Unknown SKU" in t["tool_calls"][0]["output"]
    assert grade(case("unknown-sku"), t).passed


def test_approving_a_blocked_price_fails():
    client = ScriptedClient([
        [Block("tool_use", id="t1", name="check_price_guardrails",
               input={"sku": "GR-PRE-12", "proposed_price": 10.49})],
        [Block("text", text="Yes, you can move it to $10.49.")],
    ])
    tools = asyncio.run(tool_specs())
    t = asyncio.run(run_case(client, "fake", tools, case("guardrail-block-margin")["question"]))
    assert not grade(case("guardrail-block-margin"), t).passed
