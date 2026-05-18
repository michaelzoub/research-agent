# research-agent

`research-agent` is a local command-line tool for running research and optimization agent loops. Give it a question or goal, and it can search for evidence, run parallel research variants, critique claims, produce a paper-style report, and optionally optimize code or strategies against an evaluator.

The simple mental model:

```text
your prompt -> setup choices -> research/optimization loop -> report + artifacts
```

You do not need API keys to try it. Without keys, it uses the bundled local corpus and deterministic fallback model. Add OpenAI/Anthropic keys later for live model calls.

## Quick Start

Clone the repo and run the guided setup:

```bash
git clone https://github.com/michaelzoub/research-agent.git
cd research-agent
./autore
```

`./autore` with no arguments opens a selection-based setup. Use Up/Down to move through choices and Enter to select. The setup asks plain questions, then starts the run for you.

The CLI opens with an Autore banner, quick command reminders, and a guided prompt. It also groups the finished run into "Open first", optimization artifacts, and diagnostics so the useful files are easier to spot.

Optional install, so `autore` works from anywhere:

```bash
python3 -m pip install -e .
```

## First Run

Try a small research-only run first:

```bash
./autore
```

When asked for a goal, paste something like:

```text
Research how multi-agent systems improve automated literature review quality.
```

After the run finishes, look in:

```text
outputs/<run_name>/
```

The most useful files are:

| File | What to open first |
| --- | --- |
| `final_report.pdf` | Paper-style report. |
| `final_report_preview.png` | First-page image preview for IDEs. |
| `final_report.md` | Markdown version of the report. |
| `progress.txt` | Step-by-step log of what happened. |
| `run_benchmark.html` | Timeline and decision visuals. |
| `prd.json` | Task plan and pass/fail status. |
| `cost.json` | Model usage and estimated cost. |

## Selection-Based Setup

Run:

```bash
./autore
```

The guided setup asks for these choices:

| Setup step | What it means | Good default |
| --- | --- | --- |
| Goal | The question or task you want the agent to work on. Be specific about what evidence, report, or optimization result you want. | Your prompt |
| Run mode | What kind of work the harness should do. | Auto decide |
| Evaluator | The scoring function for optimization runs. Only matters for optimize or research-then-optimize. | Decide from prompt |
| Evidence source | Where research evidence should come from. | Auto mix |
| Iteration budget | How many outer-loop rounds the agent can spend improving the result. Higher means slower but deeper. | 3-12 |
| Model/lab | Which LLM to use, or whether to rotate through all configured models. | `all-configured` or OpenAI GPT-5.2 |

Run mode options:

| Option | Use when |
| --- | --- |
| Auto decide | You want the harness to infer the best mode from your prompt. |
| Research | You want sources, claims, caveats, and a final report. |
| Optimize | You already have an evaluator and want the agent to improve a candidate solution. |
| Research then optimize | You want the agent to research approaches first, then use that evidence to seed optimization. |

Evaluator options:

| Option | Use when |
| --- | --- |
| Decide from prompt | Best for most users. |
| `length_score` | Tiny demo evaluator for smoke tests. |
| `prediction_market` | Prediction-market challenge evaluator. |
| Custom name | Use when you have added your own evaluator in code. |

Evidence source options:

| Option | What it does |
| --- | --- |
| Auto mix | Chooses a mix of relevant retrievers. Best default. |
| Local corpus | Offline bundled demo corpus. Good for testing. |
| arXiv | Searches arXiv papers. |
| OpenAlex | Searches academic metadata and papers. |
| Semantic Scholar | Searches Semantic Scholar papers. |
| GitHub | Searches code/repository evidence. |
| Web | General web search. |
| Docs/blogs | Documentation and technical blog sources. |
| Twitter/X | Social/web trend evidence if configured. |
| Memory | Prior run artifacts and local world-model memory. |
| Alchemy | Blockchain data, requires `ALCHEMY_API_KEY`. |

You can start with a few flags but still use the menu:

```bash
./autore --interactive --retriever local
```

## Environment

Create `.env.local` for secrets and defaults:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

RESEARCH_HARNESS_CORPUS_PATH=examples/corpus/research_corpus.json
RESEARCH_HARNESS_OUTPUT_DIR=outputs
RESEARCH_HARNESS_RETRIEVER=auto

RESEARCH_HARNESS_LLM_PROVIDER=auto
RESEARCH_HARNESS_LLM_MODEL=all-configured
RESEARCH_HARNESS_LLM_MODELS=openai/gpt-5.5,openai/gpt-5.2,anthropic/claude-opus-4-6,anthropic/claude-sonnet-4-6,anthropic/claude-sonnet-4-5,anthropic/claude-haiku-4-5,local/local-deterministic-fallback
```

`all-configured` uses every available model in `RESEARCH_HARNESS_LLM_MODELS` round-robin. Providers without a valid key are skipped. The local fallback keeps offline runs working.

## Examples

Research report:

```bash
./autore "Research enterprise AI agent adoption patterns" --task-mode research
```

Research first, then optimize:

```bash
./autore "Research approaches for improving a tiny scoring function" \
  --task-mode optimize_query \
  --evaluator length_score \
  --retriever local
```

Prediction-market challenge:

```bash
./autore "Get to \$10 profit in the prediction market challenge. Research strategy ideas first, then optimize." \
  --task-mode optimize_query \
  --evaluator prediction_market
```

Set `PREDICTION_MARKET_USE_UPSTREAM=1` to score through the real upstream CLI instead of the local proxy.

## Outputs

Each run creates `outputs/<NNN>_run_<slug>/`.

| Artifact | Purpose |
| --- | --- |
| `final_report.md` / `final_report.tex` / `final_report.pdf` / `final_report_preview.png` | Final synthesis report in Markdown, LaTeX, PDF, and image-preview form. |
| `progress.txt` | Human-readable progress log. |
| `prd.json` | Organized task map with status and dependencies. |
| `variants.json` / `variant_evaluations.json` | Proposed variants and their scores. |
| `optimizer_seed_context.json` | Research findings used to seed optimization. |
| `optimization_result.json` | Best score, candidate path, and official-result status. |
| `optimal_code.py` | Universal code artifact for the best selected candidate. |
| `solution.py` | Challenge-specific runnable solution when applicable. |
| `run_benchmark.html` | Per-run visual benchmark with Gantt timeline and decision DAG. |
| `run_notebook.ipynb` | Notebook export with trace summaries, artifact counts, cost, and diagnosis data. |
| `harness_diagnosis.json` | Failure taxonomy and debugger localization. |
| `provenance_edges.json` | Claim/source/hypothesis/report provenance graph. |
| `cost.json` / `cost_events.json` | Per-run model usage totals and per-call accounting. |
| `world_model.sqlite` | Cross-run SQLite mirror for dedupe, provenance, and observability. |
| `decision_dag.png` / `agent_timeline.png` | Flow and timing visuals. |

## Evaluation

Run the built-in eval suites:

```bash
python3 -m research_harness.evals --suite core
python3 -m research_harness.evals --suite edge
python3 -m research_harness.evals --suite all
```

The eval outputs go to `eval_outputs/`.

## Development

```bash
python3 -m unittest tests.test_smoke
python3 -m research_harness.evals --suite core --trials 1
./autore-bench
```

For architecture details and long-running task guidance, see [`docs/`](docs/).

## Scriptable Options

Use these when you want repeatable commands instead of the guided setup.

| Flag / option | What it does |
| --- | --- |
| `./autore "your prompt"` | Run directly from a prompt without opening the setup menu. |
| `--interactive` | Open the setup menu even if you already supplied some flags. |
| `--task-mode auto` | Let the harness infer research vs optimization. |
| `--task-mode research` | Force research/report mode. |
| `--task-mode optimize` | Force optimization mode. Usually requires `--evaluator`. |
| `--task-mode optimize_query` | Research first, then optimize from the gathered evidence. |
| `--evaluator length_score` | Use the demo scoring evaluator. |
| `--evaluator prediction_market` | Use the prediction-market challenge evaluator. |
| `--evaluator <custom_name>` | Use a custom evaluator registered in code. |
| `--retriever auto` | Use the default mixed evidence strategy. |
| `--retriever local` | Use only the bundled offline corpus. |
| `--retriever arxiv` / `openalex` / `semantic_scholar` | Use a specific academic retriever. |
| `--retriever github` / `web` / `docs_blogs` / `twitter` / `memory` / `alchemy` | Use a specific non-academic or local-memory retriever. |
| `--llm-provider auto` | Infer provider from the selected model. |
| `--llm-provider openai` / `anthropic` / `local` / `multi` | Force a provider mode. |
| `--llm-model all-configured` | Use every configured available model round-robin. |
| `--no-steering` | Disable live `/article`, `/steer`, and `/note` input during terminal runs. |

During a normal terminal run, you can steer the next proposal round without
stopping the agent:

```text
/article https://example.edu/paper | Paper title | Why this should matter
/steer The last variants all failed on the same observed scorer behavior; explore a different mechanism.
/note Locally found source says the evaluator exposes this state field.
```

Steering input is recorded in `user_steering_inbox.jsonl`, ingested at the next
round boundary, and persisted as user-provided sources/claims before it affects
queries or candidate proposals.
| `--llm-model openai/gpt-5.2` | Use one specific model. |
| `--list-llm-models` | Print the configured model catalog and exit. |
| `--max-iterations N` | Set the outer-loop iteration budget. |
| `--corpus PATH` | Choose the local corpus JSON file. |
| `--output PATH` | Choose where run artifacts are written. |
| `--no-sessions` | Disable session JSONL logging. |
| `--quiet` | Suppress live progress printing. |
