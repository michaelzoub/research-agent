---
name: update-architecture
description: Update architecture documentation in docs/architecture.md to reflect changes to the three-loop structure, product agents, run-state format, loop modes, or artifact flow. Use whenever the orchestrator, loops, schemas, or store change in a way that affects the documented architecture.
---

# Update Architecture Docs

Use this skill when the project architecture changes: new loop modes, new product agents, run-state format changes, store schema changes, or flow changes to how actions are selected and executed.

## Files to Update

- `docs/architecture.md` — primary architecture reference (mermaid diagrams + prose)
- `docs/gen_overview_diagram.py` — sequence diagram generator (re-run to emit SVG)
- `skills/research-agent-architecture/SKILL.md` — skill-level invariants

## Three-Loop Architecture

Keep this mental model current in all docs:

```text
Outer loop  — Session (sessions.py)
              Manages context isolation and parallel agent runs.
              Resets state between runs so each agent starts clean.

Middle loop — EvolutionaryOuterLoop (loops.py)
              Proposes and evaluates variants across N outer iterations.
              Drives research (query variants → retrieve → score) or
              optimize (code variants → evaluator → score).

Inner loop  — Probabilistic loop (orchestrator._run_loop + agent harness)
              The harness fixes tools, state, evaluator, budgets, and safety.
              The model chooses the next investigation or action from evidence.
              Records observed actions in run_state.json and appends progress.txt.
              Stops on sufficient evidence, safety boundary, or budget exhaustion.
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
