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

## Prediction-market status

The deterministic prediction-market evaluator and challenge utilities remain in the repository, but the single production agent loop does **not yet** expose experiment submission or candidate promotion as tools. Therefore this older command is intentionally unsupported:

The older task-mode selector is intentionally unsupported.

Do not expect `autore` to optimize a strategy or claim a profit target from `--evaluator prediction_market` until the controlled `ExperimentSystem` tools are implemented. The next architecture step is to expose evaluator inspection, experiment proposal, submission, result inspection, and comparison to the same `ResearchAgent`; evaluator execution and promotion remain deterministic and model-independent.

## Resuming work

Runs are durable but not yet resumable from an existing event stream. Use the prior run’s report, sources, event log, and failed paths as context for a new goal. Explicit checkpoint/resume support is tracked in [TODO.md](../TODO.md).

## Parallelism

Independent read-only tool calls requested in one model turn run concurrently. The model chooses whether parallel calls are useful; the registry serializes mutating tools and enforces capability and budget boundaries.
