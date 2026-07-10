---
name: research-run-grounding
description: Improve grounded research-agent runs. Use when modifying retrieval, source strategy, claim extraction, hypothesis generation, critic behavior, synthesis reports, citations, local corpus data, or research evals for papers/data finding.
---

# Research Run Grounding

Use this skill when changing research behavior, retrievers, local corpus data,
claim extraction, source handling, critic logic, or synthesis.

## Research Agent Loop

The research product agent should:

```text
goal
  -> propose query variants
  -> retrieve sources
  -> extract traceable claims
  -> score coverage/corroboration/credibility
  -> generate hypotheses or open questions
  -> critique contradictions
  -> synthesize report with citations
```

## Artifact Requirements

Research runs should preserve:

- `sources.json`: URLs, titles, authors, dates, source type, credibility
- `claims.json`: claim text, source IDs, confidence, support level
- `hypotheses.json`: hypothesis, supporting/contradicting claims, next experiment
- `contradictions.json` and `open_questions.json`
- `final_report.md`
- `run_state.json`
- trajectory files such as `progress.txt`, `trace.jsonl`, and `agent_traces.json`

## Quality Rules

- Claims must cite source IDs.
- Reports must not invent sources beyond stored artifacts.
- Low-confidence or contradictory claims should create open questions.
- Local/offline retrieval must remain deterministic for tests.
- Live retrieval should use the same `SearchBackend` interface as local search.
- Advanced or paid services are optional. Use `skills/research-service-directory`
  and `services/` only when live, premium, or structured data would materially
  improve the research result.

## Eval Expectations

Research evals should check:

- run completed
- enough sources and claims
- claims are grounded in source IDs
- report exists and mentions uncertainty/caveats when appropriate
- transcript/progress is complete

## Gotchas

- Retrieval breadth is not enough; claims need traceability.
- Critic and synthesis should run after evidence artifacts exist.
- Do not optimize the report text at the expense of source fidelity.
