---
name: update-architecture
description: Update architecture documentation in docs/architecture.md to reflect changes to the model-directed production path, run-state format, tool boundary, or artifact flow.
---

# Update Architecture Docs

Use this skill when the project architecture changes: run-state format changes, store schema changes, or flow changes to how model requests and tool actions are selected and executed.

## Files to Update

- `docs/architecture.md` — primary architecture reference (mermaid diagrams + prose)
- `skills/research-agent-architecture/SKILL.md` — skill-level invariants

## Production Architecture

Keep this mental model current in all docs:

```text
Session       — `sessions.py` manages context isolation between runs.

Production run — `Orchestrator` starts one `ResearchAgent` trajectory.
                 The harness fixes tools, state, evaluator boundaries, budgets,
                 and safety; the model chooses the next action from evidence.
                 It records model-request/model-response spans, tool actions,
                 and termination in the durable event stream.

The archived evolutionary prototype is intentionally outside this architecture
in `research_harness/future/evolutionary/` and is not an execution path.
```

## Run-State Action Format

Every observed action in `observed_actions` must follow this schema:

```json
{
  "id": "action-001",
  "title": "Action actually taken",
  "status": "passed",
  "decision_basis": "runtime action record; not a predefined sequence"
}
```

Key rules:
- `id` uses `action-NNN` format
- actions are written after they occur, not before
- run_state.json is refreshed after each observed action

## Checklist When Updating Architecture

1. Update the mermaid diagram in `docs/architecture.md` to match the new flow.
2. Update the prose sections (Product Agent Details, Optimization Output Contract) if the behavior changed.
3. Re-run `python docs/gen_overview_diagram.py` to regenerate the SVG.
4. Update `skills/research-agent-architecture/SKILL.md` if any invariants changed.
5. Verify `observed_actions` in a real `run_state.json` records only actions actually taken.
