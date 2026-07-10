---
name: research-agent-architecture
description: Maintain the single model-directed research-harness execution architecture. Use when changing the orchestrator, ResearchAgent, tool registry, run-state artifacts, or their documentation.
---

# Research Agent Architecture

## Core convention

```text
agent = model + harness
```

Production execution has one `ResearchAgent` loop. The model chooses direct answers, tool use, recovery, user input, and finalization. The harness enforces permissions, budgets, schemas, persistence, validation, and deterministic evaluator/promotion boundaries.

Do not add runtime modes, task routers, fixed phases, mandatory search sequences, or renamed legacy workflows.

## Event contract

Every production run must persist actual events in this order as applicable:

- model turn;
- tool request;
- tool result or budget/safety rejection;
- final validation;
- termination.

Provider-native tool call IDs must have matching tool-result messages. Persist failures in `failed_paths.json`; preserve successful calls in `agent_events.jsonl` and `agent_messages.json`.

## Capability boundaries

- Tools are async and schema-validated.
- Concurrent execution is limited to independent read-only tools.
- External retrieval must return relevant evidence or an explicit failure.
- The model can propose an experiment, but deterministic evaluator and promotion policy own execution and promotion.
- Public decision summaries are allowed; hidden chain-of-thought is not collected or stored.

## Documentation invariant

Keep `README.md`, `docs/architecture.md`, `docs/long_running_tasks.md`, and `TODO.md` aligned with the production CLI. Do not document removed flags such as `--task-mode` or imply that a deterministic utility is available through `autore` until it is exposed as an agent tool.
