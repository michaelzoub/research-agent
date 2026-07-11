# research-harness

`autore` runs one model-directed agent loop. The model decides whether to answer directly, use registered tools, recover from failures, request input, inspect evidence, or stop. The harness enforces permissions, budgets, persistence, validation, and deterministic evaluation; it does not prescribe a research sequence.

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
autore --help
```

There is no execution-mode flag. All tasks start the same `ResearchAgent`; tools are optional capabilities selected by the model. Registered capabilities include source search, public document fetching, figure inspection (caption, image URL, and aspect-ratio metadata), approved-workspace reads, sandboxed Python analysis, and bounded host-terminal inspection (`curl`, `npm`, `git`, and `rg`). The terminal tool runs direct argv only—never a shell—uses an ephemeral home with no inherited credentials, and is limited to read-only subcommands and public HTTP(S) GET/HEAD requests. A failed or empty discovery call stays in the audit trail but does not consume the successful-evidence allowance, so the agent can recover through another source. Before using a tool, the model records a concise public decision summary. This is an auditable rationale for the action, not hidden chain-of-thought.

## Artifacts

Each run writes `outputs/<run>/`:

- `final_report.md` — validated answer or explicitly labelled partial result.
- `run_state.json` — actual model turns, tool calls, observations, budgets, and termination state.
- `agent_messages.json` — provider-neutral transcript preserving tool-call IDs and results.
- `agent_events.jsonl` — append-only model-turn, tool-request, and tool-result events, written as they occur.
- `failed_paths.json` — provider, tool, budget, and runtime failures with retryability metadata; successful calls do not appear here.
- `cost.json` — model usage and estimated cost.

Workspace reads are deny-by-default for sensitive files such as `.env` and `.git`. Document retrieval validates every DNS-resolved host and redirect against private, loopback, link-local, and reserved addresses. HTML documents may be rendered into compact Markdown through curl.md after the target URL passes those checks; direct fetch remains available as the fallback.

Search tools must return relevant records or an explicit error. A DuckDuckGo bot challenge is surfaced as a tool failure, not bypassed. The agent can recover with registered primary-source APIs or direct public URLs through `fetch_document`, figure inspection, or the bounded terminal tool; arXiv exact IDs are fetched directly and unrelated papers are rejected before persistence. The harness compacts source records before returning them to the model while retaining complete source metadata in the artifact store.

## Test

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/research-harness-pycache python3 -m unittest tests.test_research_agent
env PYTHONPYCACHEPREFIX=/private/tmp/research-harness-pycache python3 -m unittest discover -s tests
autore --help
```
