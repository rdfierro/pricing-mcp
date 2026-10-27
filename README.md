# pricing-mcp

An MCP server that gives AI agents real pricing analytics (elasticity, price simulation, guardrails, price-ladder and competitor checks), plus an eval suite that checks whether an agent using those tools actually gets the right answer.

Built around a packaged-coffee category, because that's where I spent years doing pricing at Starbucks and Nestlé. All data is synthetic.

## Why this exists

Giving an agent pricing tools is easy. Knowing whether it uses them correctly is the hard part. An agent that says "raise the price to $10.49, margin looks fine" when the margin rule says otherwise is worse than no agent. So this project is built backwards from the evals:

- **The data is generated from known answers.** Each SKU's true price elasticity is set in the generator, and a few pricing problems are planted on purpose. The tools are tested against that ground truth, not against themselves.
- **Agents are graded on behavior, not vibes.** Did it call the right tool? Did it report a blocked price as blocked? Does every number in its answer actually appear in a tool result?
- **The tools are built for agents.** Every result carries its inputs, assumptions and confidence. Errors list what's valid. Every tool is read-only, and every call writes a structured audit log line.

## Tools

| Tool | What it answers |
| --- | --- |
| `list_products` | Which SKUs exist, with tier, pack size, price, cost, and the pricing policy |
| `get_price_elasticity` | How sensitive is this SKU to price? Log-log regression over 52 weeks with seasonal controls, a 95% interval, and a confidence label |
| `simulate_price_change` | What happens to weekly units, revenue and margin at a new price, and what does that projection assume |
| `check_price_guardrails` | Is a proposed price allowed? Hard rules (margin floor, ±15% per move) block; the competitor band warns |
| `check_price_ladder` | Do bigger packs cost less per ounce, and are tiers spaced value < core < premium by at least 10%? |
| `competitor_price_index` | Where each SKU sits against the median competitor and its tier's target band |

## Quick start

```bash
pip install -e ".[dev,evals]"
pytest                          # unit tests + tool evals
python -m evals.tool_evals      # tool evals with a readable report
pricing-mcp                     # run the server over stdio
pricing-mcp --http --port 8000  # or over streamable HTTP
```

### Use it from Claude Code

```bash
claude mcp add pricing -- pricing-mcp
```

Then ask: *"Are there any problems with our ground coffee price ladder?"*

### Use it from Claude Desktop or another MCP client

```json
{
  "mcpServers": {
    "pricing": { "command": "pricing-mcp" }
  }
}
```

## Evals

There are two layers.

### 1. Tool evals (deterministic, run in CI)

`python -m evals.tool_evals` checks the analytics against the generator's ground truth:

| Suite | Checks | Result |
| --- | --- | --- |
| Elasticity | Each SKU's estimate is within 0.30 of the true value and the true value falls in the 95% interval | 14 / 14 |
| Price ladder | Finds exactly the two planted breaks, with no false alarms | 1 / 1 |
| Competitor index | Flags exactly the two planted out-of-band SKUs | 3 / 3 |
| Guardrails | Correct verdict and the exact rules that failed, across pass / warn / block cases | 7 / 7 |
| Simulation | Revenue moves in the direction elasticity implies (elastic SKUs lose revenue on a price rise; inelastic ones gain) | 14 / 14 |

**39 / 39 passing.** Largest elasticity error is 0.24 (true −2.6, estimated −2.84).

### 2. Agent evals (needs an API key)

`python -m evals.agent_evals` runs Claude against the server's tools on ten scenarios in [`evals/agent_cases.yaml`](evals/agent_cases.yaml) and grades each transcript:

- **Tools used**: did it call the tool the question needs?
- **Verdict**: when the guardrail tool says `block`, does the answer say so instead of approving the price?
- **Grounding**: every dollar figure, percentage and decimal in the answer must appear in a tool output or the question, or be the result of a calculation the agent wrote out. The grader re-does that arithmetic and checks that its inputs came from tools. Invented numbers, and derived numbers with no work shown, fail the case.
- **Must not say**: no claiming to have changed data through read-only tools, no inventing an elasticity for a SKU that doesn't exist.
- **Multi-step**: asked to fix the ground-coffee ladder, the agent has to find a price for the 32 oz that is cheaper per ounce than the 24 oz but still inside the competitor band, and prove it by getting a `pass` from the guardrail tool.

```bash
export ANTHROPIC_API_KEY=...
python -m evals.agent_evals                        # all cases
python -m evals.agent_evals --repeat 5             # measure consistency
python -m evals.agent_evals --case fix-ladder-multistep
PRICING_EVAL_MODEL=claude-haiku-5-5 python -m evals.agent_evals   # compare models
```

Each run writes a JSON report with full transcripts, pass rate, tool-call count and token usage to `evals/results/`. The held-out run below is saved in [`evals/reports/`](evals/reports) so you can read every transcript. The graders have their own tests in [`tests/test_grading.py`](tests/test_grading.py).

### Results: Claude Sonnet 5.5, October 2026

Held-out run: 10 cases × 3 attempts, graded by a grader frozen before the run.

| | Result |
| --- | --- |
| Cases passed | **27 / 30 (90%)** |
| Behavior checks (right tool, right verdict, no approving a blocked price, no claimed writes, no invented estimates) | **72 / 72** |
| Number grounding | 27 / 30 |
| Tool calls | 100 (about 3.3 per question) |
| Tokens | 187k in, 28k out |

The three misses were all grounding: a "26% below competitors" with no calculation shown, an example price ($5.49) offered unprompted, and a correct calculation written too far from its result for the grader to connect.

### What the first live run taught me

The first run scored **11 / 30**, and reading the transcripts showed most failures were the eval's fault, not the agent's:

- **The grader graded the wrong call.** Asked about $4.99, the agent correctly reported it blocked, then helpfully tested $5.10 too. The grader checked the last guardrail call instead of the one for the price asked about.
- **Correct arithmetic looked like hallucination.** The agent computed things like the price needed to hit the margin floor ($5.90 ÷ (1 − 0.52) = $12.29) and flagged them as its own math. A strict "must appear in tool output" rule failed them. Loosening the rule would have let real inventions through, so instead the agent is asked to show its work, and the grader re-checks it.
- **Phrase checks were too literal.** "Isn't a known SKU" failed a check for the word "not", and "getting $10.49 approved would need a policy exception" tripped a check for "approved".

After fixing those, the same transcripts re-graded at 19 / 30. The remaining failures were genuine: derived numbers with no work shown. A prompt change to show calculations, plus parser fixes for the ways the agent actually writes math, brought a fresh held-out run to 27 / 30. The lesson: when an eval score is low, read the transcripts before you touch the prompt.

## Design notes

- **Deterministic code for the math, the model for judgment.** The agent decides which question to ask and how to explain the answer; it never does the regression or the policy check itself.
- **Hard versus soft rules.** Margin and change-size limits block a price; competitor position only warns. That mirrors how most pricing teams actually run approvals.
- **Results explain themselves.** Simulations list their assumptions and the observed price range they're reliable within. Elasticity comes with an interval and a confidence label, so an agent can say "medium confidence" instead of overstating.
- **Safe by construction.** No tool writes anything. Every tool carries MCP read-only annotations, and the audit log records each call's tool, arguments, status and latency.

## Data

`scripts/generate_data.py` writes 14 SKUs across whole bean, ground and cold brew in value, core and premium tiers, with 52 weeks of sales, three competitors' prices, and a pricing policy. Weekly units follow each SKU's true elasticity with seasonality (hot coffee peaks in winter, cold brew in summer) and noise. Planted problems:

- Ground 32 oz costs more per ounce than ground 24 oz.
- Premium ground 12 oz is priced below core ground 12 oz, and below its own margin floor.
- Premium whole bean 12 oz sits well above competitors; value ground 12 oz sits well below.

Point the server at your own data with `PRICING_MCP_DATA=/path/to/dir` using the same four files.

## Layout

```
src/pricing_mcp/
  analytics.py   pure pricing functions
  server.py      MCP server and tool definitions
  data.py        dataset loader
  data/          synthetic dataset
evals/
  tool_evals.py      deterministic checks against ground truth
  agent_evals.py     model-in-the-loop harness
  agent_cases.yaml   scenarios and expectations
  grading.py         graders
  ground_truth.json  answers from the generator (never read by the server)
scripts/generate_data.py
tests/
```

## License

MIT
