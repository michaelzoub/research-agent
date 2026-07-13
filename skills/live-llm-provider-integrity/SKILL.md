---
name: live-llm-provider-integrity
description: Use when changing LLM provider selection, model catalogs, API-key loading, Kimi/Moonshot support, live model fallbacks, tool-call traces, or costs that depend on whether a real model call succeeded.
---

# Live LLM Provider Integrity

Use this skill whenever a change touches `LLMClient`, model selection, Kimi,
environment loading, tool proposal calls, or traces that claim
which model generated an artifact.

## Required Checks

- `LLMClient(provider="auto", model="kimi/kimi-k2.6")` must resolve to
  `provider="kimi"` and `model="kimi-k2.6"`.
- Direct `LLMClient(...)` construction must load `.env` and `.env.local`
  defaults, not only `cli.main()`.
- A configured live model must report `is_live=True` and `model_label` equal to
  the live model. If it reports `local-deterministic-fallback`, stop and fix
  environment loading or provider routing.
- Never infer “Kimi generated the strategy” from the configured model alone.
  Inspect `agent_traces.json` or `trace.jsonl` for the proposal/controller trace
  status and errors.

## Kimi Rules

- Kimi K2.6 requests must use `temperature: 1`; other temperatures may fail.
- Kimi 429/rate-limit responses should retry with a small bounded backoff.
- Kimi keys may be `MOONSHOT_API_KEY` or `KIMI_API_KEY`.
- Do not print key values. Only report booleans such as `has_kimi_key=True`.

## Challenge Tool Rule

For optimization challenge runs, distinguish these cases:

- **Live LLM-generated code**: a completed `model_turn` contains the native
  tool-call arguments with complete strategy code.
- **Provider failure**: the matching `model_request` / `model_turn` pair ends
  with an error status.
- **No generated candidate**: no evaluation tool call occurred; do not imply a
  deterministic fallback candidate was evaluated.

If the model request failed, do not describe an evaluation as LLM-generated.

## Regression Tests

When changing this area, add or run tests for:

- Kimi model/provider resolution.
- `.env.local` key loading outside the CLI.
- Kimi forced temperature.
- Kimi 429 retry behavior.
- Optimizer traces/evals that fail when fallback/template candidates masquerade
  as agentic LLM-generated strategy improvements.
