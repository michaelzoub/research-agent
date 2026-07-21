# research-harness

`autore` runs one model-directed agent loop. The model decides whether to answer directly, use registered tools, recover from failures, request input, inspect evidence, or stop. The harness enforces permissions, budgets, persistence, validation, and deterministic evaluation.

## Quick start

```bash
python3 -m pip install -e .
autore "Explain this concept in plain language" --llm-provider ollama --llm-model ollama/qwen3.5:latest
```

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or run a compatible Ollama model for native tool calling. A local provider is required for tool-using runs; the harness will not fabricate a tool trajectory offline.

## Controls

```bash
autore "Research a current technical question" --retriever web --max-iterations 8
autore "Find source figures for this claim" --max-tool-calls 48 --max-runtime-seconds 300
autore "Analyze this repository's public architecture" --retriever local
autore "Optimize the prediction-market challenge" --grader --grader-loops 8
autore --help
```

All tasks start the same lead `ResearchAgent`; tools are optional capabilities selected by the model. Approved bounded workers are owned internally by `WorkerRegistry` and exposed externally as the ordinary `delegate_task` capability in `ToolRegistry`. The lead remains the sole controller and final synthesizer. Each worker receives an isolated context, artifact store, event stream, permissions, accounting, run identity, and termination state; recursion is initially disabled and aggregate count/parallelism plus per-worker runtime/token/tool-call budgets are enforced. Parallel calls inside one agent loop remain tool concurrency, not workers. With `--grader`, the controller can also run prediction-market optimization batches, exact-replacement parameter sweeps, and evidence-backed learning capture. Workers and sweeps write isolated candidates and use the existing official upstream grader; no score is fabricated and no candidate is promoted without an eligible official measurement. `learnings.md` and `learnings.jsonl` preserve the current run's confirmed findings and dead ends as audit artifacts; new runs do not ingest prior outputs. Registered capabilities include source search, public document fetching, figure inspection, structured document extraction, grounded document analysis, charts, approved-workspace reads, sandboxed analysis, bounded terminal inspection, and optional external services.

The guided CLI does not ask the user to choose an evidence backend: the model selects among registered search capabilities from the objective and observations. `--retriever` remains an advanced capability restriction for offline runs, reproducible tests, or environments where only a specific backend should be exposed.

```mermaid
flowchart TD
    goal["Goal"] --> lead["Lead agent"]
    lead --> decision{"Model action"}
    decision -->|tool call| tool["ToolRegistry\nordinary tool"]
    tool --> lead
    decision -->|delegate_worker / delegate_task| worker["WorkerRegistry\nbounded worker"]
    worker -->|WorkerResult findings| lead
    decision -->|final answer| done["Lead synthesizes result"]
```

Workers complete bounded assignments and return findings to the lead; they never replace the lead controller or final synthesizer. Ordinary tool calls remain tool calls, not workers.

Grader runs separate the full audit transcript from model working context. Every iteration receives a deterministic checkpoint containing the strategy ledger, exact champion/latest code, official edge metrics, fetched literature extracts, and only the newest unresolved tool exchange. Older tool chatter remains in `agent_messages.json` and `agent_events.jsonl` but is not replayed to the model. Fetched documents are keyed by canonical URL and served from the artifact cache on repeat requests.

Document download bytes and model-facing extracted characters are separate limits, so ordinary HTML pages larger than the extract budget are parsed and compacted instead of rejected. Transient provider timeouts are retried once at the agent-loop boundary with the same working state after transport retries are exhausted; malformed or otherwise deterministic provider errors still terminate immediately. Kimi uses a 120-second default request timeout, configurable with `RESEARCH_HARNESS_LLM_TIMEOUT_SECONDS`.

## Artifacts

Each run writes `outputs/<run>/`:

- `final_report.md` — validated answer or explicitly labelled partial result.
- `run_state.json` — actual model turns, tool calls, observations, budgets, and termination state.
- `agent_messages.json` — provider-neutral transcript preserving tool-call IDs and results.
- `agent_events.jsonl` — append-only model-turn, tool-request, and tool-result events, written as they occur.
- `agent_timeline.png` / `agent_timeline.svg` — readable overview and complete event timeline with semantic operation colors and independent status styling.
- `failed_paths.json` — provider, tool, budget, and runtime failures with retryability metadata; successful calls do not appear here.
- `cost.json` — model usage and estimated cost.
- `swarm_results.json`, `sweeps/`, and `learnings.md` — created when the controller uses the grader swarm, sweep, or learning tools.
- `datasets/<id>.json` — complete normalized table/numeric extraction plus source, section/table, and row provenance.
- `document_analyses/<id>.json` — LLM document analysis with source-stated findings separated from model inferences.
- `charts/<id>.svg` and `charts/<id>.json` — deterministic SVG chart and reproducible dataset/config provenance.
- `candidate_graph.json`, `.svg`, and `.png` — immutable optimization candidates and typed lineage edges.
- `champion_history.json` — ordered promotion decisions, separate from lineage. Deprecated `champion_tree.*` compatibility files may also be emitted during migration.

For example, after `fetch_document` returns a verified source ID, the model may call `extract_structured_data` with that ID, then call `generate_svg_chart` using the returned dataset ID and column names. It can call `analyze_research_document` when a paper needs a grounded methodology/results reading. These calls are optional; they do not create a paper-processing pipeline. Chart generation rejects missing datasets, nonnumeric values, and incompatible selected units rather than guessing conversions.

Workspace reads are deny-by-default for sensitive files such as `.env` and `.git`. Document retrieval validates every DNS-resolved host and redirect against private, loopback, link-local, and reserved addresses. HTML documents may be rendered into compact Markdown through curl.md after the target URL passes those checks; direct fetch remains available as the fallback.

Model turns and wall-clock runtime are unbounded when their CLI limits are omitted. Explicit `--max-iterations` and `--max-runtime-seconds` values remain hard limits. The guided CLI defaults to eight official candidate evaluations for grader runs. Fetching an arXiv `/abs/` URL resolves to its PDF and extracts bounded page text with PDF-page locators; the abstract page is not treated as the paper body.

Search tools must return relevant records or an explicit error. A DuckDuckGo bot challenge is surfaced as a tool failure, not bypassed. The agent can recover with registered primary-source APIs or direct public URLs through `fetch_document`, figure inspection, or the bounded terminal tool; arXiv exact IDs are fetched directly and unrelated papers are rejected before persistence. The harness compacts source records before returning them to the model while retaining complete source metadata in the artifact store.

## Test

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/research-harness-pycache python3 -m unittest tests.test_research_agent
env PYTHONPYCACHEPREFIX=/private/tmp/research-harness-pycache python3 -m unittest discover -s tests
autore --help
```

## Version releases

Run the **Bump version** workflow from the GitHub Actions tab on `main`. It increments the patch component by default (`0.1.0` → `0.1.1`); choose `minor` or `major` in the workflow form when needed. The workflow tests the repository, synchronizes `setup.cfg` and `research_harness/__init__.py`, commits the change, and creates and pushes the matching `vX.Y.Z` tag.
