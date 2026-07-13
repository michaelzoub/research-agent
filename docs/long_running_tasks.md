# Long-Running Tasks

The production harness runs one model-directed agent trajectory. Long work is a bounded sequence of model turns, tool observations, validation feedback, and durable artifacts—not a selected research or optimization mode.

## Current use

```bash
./autore "Research approaches to the prediction-market challenge, including inventory risk, adverse selection, and evaluation criteria." \
  --retriever arxiv \
  --max-iterations 12
```

The agent may decide which registered search and document tools to use. Inspect the live progress stream and then these artifacts:

```text
outputs/<run>/agent_events.jsonl
outputs/<run>/failed_paths.json
outputs/<run>/sources.json
outputs/<run>/final_report.md
```

## Prediction-market optimization

Enable the official prediction-market challenge grader with `--grader`; `--grader-loops` requests the number of official candidate evaluations:

```bash
./autore "Optimize the prediction-market challenge with literature-guided market making." \
  --grader \
  --grader-loops 6 \
  --max-iterations 20
```

The model controls research and candidate revisions. The registered grader is the only scoring authority, and an unavailable grader produces an error rather than a fabricated score.

## Resuming work

Runs are durable but not yet resumable from an existing event stream. New runs do not automatically ingest previous reports, sources, event logs, or failed paths. Explicit checkpoint/resume support is tracked in [TODO.md](../TODO.md).

## Parallelism

Independent read-only tool calls requested in one model turn run concurrently. The model chooses whether parallel calls are useful; the registry serializes mutating tools and enforces capability and budget boundaries.
