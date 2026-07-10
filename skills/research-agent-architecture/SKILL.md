---
name: research-agent-architecture
description: Maintain the research-harness architecture where each product option is an agent defined as model plus harness. Use when changing orchestrator routing, product agents, loop modes, run-state artifacts, parallel execution, or docs that explain research, optimize, and challenge agents.
---

# Research Agent Architecture

Use this skill when changing the project architecture, especially
`research_harness/orchestrator.py`, `research_harness/loops.py`,
`research_harness/schemas.py`, `research_harness/store.py`, or
`docs/architecture.md`.

## Core Convention

The project follows:

```text
agent = model + harness
```

The model is `LLMClient` or a compatible model client. The harness is the loop,
tools, evaluators, artifact store, run state, budgets, traces, stopping
rules, and orchestration policy.

## Product Agents

Keep these product agents first-class:

- `research`: finds papers/data, extracts claims, critiques, synthesizes.
- `optimize`: improves candidates against evaluators/tests.
- `challenge`: uses the optimization core plus challenge specs, official/proxy
  graders, and solution rendering.

Do not collapse product-agent identity into loop mode. `task_mode` says how the
run executes; `product_agent` says which product option the user invoked.

## Probabilistic Loop Contract

Every product agent run writes `run_state.json` with:

- selected `product_agent`
- selected runtime `task_mode`
- agent-harness definition
- observed actions and their evidence
- stopping rationale
- artifact paths

Do not add a fixed task sequence. The model chooses its next action from the
goal, current state, retrieved evidence, evaluator feedback, failures, and
remaining budget.

## Parallelism Rule

The orchestrator decides parallelism based on dependencies:

- independent role agents may use `asyncio.gather`
- independent variant evaluations may use `asyncio.gather`
- critique and synthesis should run only when the available evidence warrants them

Record the action basis and observed outcome in `LoopTask` records and run state.

## Gotchas

- Role classes such as `LiteratureAgent` and `CriticAgent` are workers inside
  the agent harness; they are not the entire product agent.
- `optimize` and `challenge` share an optimization core, but challenge remains a
  separate product agent because it has external contracts and official graders.
- Updating only docs is not enough when the invariant should be enforced by
  tests or graders.
